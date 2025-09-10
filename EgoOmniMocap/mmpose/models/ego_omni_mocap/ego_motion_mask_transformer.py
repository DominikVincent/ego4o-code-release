import random
from typing import Optional

import numpy as np
import torch
from mmengine.model import BaseModel
import torch.nn as nn
import math

from mmengine.runner import load_checkpoint
from tqdm import tqdm

from mmpose.models.builder import POSE_ESTIMATORS
import torch.nn.functional as F
import clip
from .vqvae import vqvae as vqvae
from .vqvae.parser_util import mtm_args
from ...datasets.datasets.ego_omni_mocap.humanml_utils.motion_process import recover_from_ric, recover_global_limb_rot, \
    recover_global_limb_rot_batch


@POSE_ESTIMATORS.register_module()
class EgoMotionMaskTransformer(BaseModel):
    def __init__(self,
                 input_dim=3,
                 drop_out=0.1,
                 num_emb=128,
                 text_mask_rate=0.5,
                 motion_mask_rate=(0, 0.2, 0.4, 0.5),
                 epoch_gap=(0, 200, 250, 300),
                 recon_loss_weight=0.001,
                 online_prediction=False,
                 with_post_optimization=False,
                 control_joints=(1, 2, 3),
                 pretrained_transformer=None,
                 pretrained_vqvae=None,
                 init_cfg=None,
                 ):
        super().__init__(init_cfg=init_cfg)

        self.current_epoch = 0
        self.input_dim = input_dim
        self.online_prediction = online_prediction
        self.with_post_optimization = with_post_optimization
        # the post optimization is not implemented for the online prediction mode
        assert not (self.online_prediction and self.with_post_optimization)

        self.control_joints = control_joints

        self.transformer_traj_model = TransformerAutoencoder_withCodes_hml_G2_noTraj(input_dim,
                                                                                     drop_out, num_emb)
        if pretrained_transformer is not None:
            loaded_state_dict = torch.load(pretrained_transformer, map_location='cpu')
            try:
                self.transformer_traj_model.load_state_dict(loaded_state_dict, strict=False)
            except RuntimeError as e:
                print(e)
                # solve the shape mismatch problem
                current_model_dict = self.transformer_traj_model.state_dict()
                new_state_dict = {k: v if v.size() == current_model_dict[k].size() else current_model_dict[k] for k, v in
                                  zip(current_model_dict.keys(), loaded_state_dict.values())}
                self.transformer_traj_model.load_state_dict(new_state_dict, strict=False)
                print(f"Loaded Transformer Weights from {pretrained_transformer}")

        args = mtm_args()

        self.vq_net = vqvae.HumanVQVAE(args,  ## use args to define different parameters in different quantizers
                                  args.num_emb,
                                  args.emb_dim,
                                  args.output_emb_width)
        self.vq_net.load_state_dict(torch.load(pretrained_vqvae, map_location='cpu'), strict=True)
        self.vq_net.eval()  # Set the model to evaluation mode

        # set vq net requires_grad to False
        for param in self.vq_net.parameters():
            param.requires_grad = False

        # epoch based mask schedule
        # mask_rates0 = [.0, 0., 0., 0., 0., 0.]
        # mask_rates1 = [.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        # mask_rates2 = [.4, 0.4, 0.4, 0.4, 0.4, 0.4]
        # mask_rates3 = [.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        # self.mask_rates = [mask_rates0, mask_rates1, mask_rates2, mask_rates3]
        self.mask_rates = np.asarray(motion_mask_rate)
        # expand the mask rates to dimension (5, 6)
        self.mask_rates = np.expand_dims(self.mask_rates, axis=1)
        self.mask_rates = np.tile(self.mask_rates, (1, 6))
        self.epoch_gap = epoch_gap

        # loss
        self.bce_loss_fn = torch.nn.CrossEntropyLoss()

        self.recon_loss_weight = recon_loss_weight

    def loss(self, traj_data: torch.Tensor, input_text=None, motion_hml=None, lengths=None,
             data_samples: Optional[list] = None) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        self.vq_net.eval()
        batch_size = traj_data.shape[0]

        x_label_idx = self.vq_net.get_code_idx(motion_hml).detach()  # .permute(0, 2, 1)

        traj_data = traj_data.reshape((-1, 196, 6, self.input_dim))

        # random select mask joint number
        joint_mask = random.choices([0, 1, 2, 3, 4, 5], weights=(1, 2, 2, 4, 4, 2), k=1)[0]
        # joint_mask = 5

        mask_index = next((i for i, v in enumerate(self.epoch_gap) if self.current_epoch < v), len(self.epoch_gap) - 1)
        x_traj_masked = random_mask_seq_update(traj_data, self.mask_rates[mask_index], joint_mask=joint_mask,
                                               no_mask_prob=0, mask_joint_prob=0.9)

        _, pre_codes = self.transformer_traj_model(x_traj_masked, input_text)
        codes_pick_gumbel_softmax = F.gumbel_softmax(pre_codes, tau=1, eps=1e-10, hard=True, dim=-1)

        x_quantized_fromIds = self.vq_net.vqvae.get_x_quantized_from_x_ids(
            codes_pick_gumbel_softmax.permute(0, 2, 3, 1).contiguous())
        sample = self.vq_net.vqvae.forward_decoder_from_quantized_codes(x_quantized_fromIds)

        reshaped_pre_codes = pre_codes.permute(0, 3, 1, 2)

        loss_dict = {}
        latent_loss = 0
        for i in range(batch_size):
            current_len = math.ceil(lengths[i] / 4)

            bce_loss = self.bce_loss_fn(reshaped_pre_codes[i:i + 1, :, :, :current_len],
                                   x_label_idx[i:i + 1, :, :current_len]) / batch_size

            latent_loss += bce_loss  # + traj_loss
        loss_dict['latent_loss'] = latent_loss

        recon_loss = self.recon_loss_weight * F.mse_loss(sample, motion_hml)
        loss_dict['recon_loss'] = recon_loss

        # breakpoint()

        return loss_dict

    def predict(self, traj_data: torch.Tensor, input_text=None, lengths=None,
             data_samples: Optional[list] = None):
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        self.vq_net.eval()
        self.transformer_traj_model.eval()

        # breakpoint()

        traj_data = traj_data.reshape((-1, 196, 6, self.input_dim))
        joint_mask = 5

        mask_index = next((i for i, v in enumerate(self.epoch_gap) if self.current_epoch < v), len(self.epoch_gap) - 1)
        # note: do not need to randomly mask the traj data during the prediction
        x_traj_masked = traj_data.clone()
        # x_traj_masked = random_mask_seq_update(traj_data, self.mask_rates[mask_index], joint_mask=joint_mask)

        # note: set dummy input text if input_text is None
        if input_text is None:
            input_text = [""] * traj_data.shape[0]
        _, pre_codes = self.transformer_traj_model(x_traj_masked, input_text)

        codes_pick_gumbel_softmax = F.gumbel_softmax(pre_codes, tau=1e-3, eps=1e-10, hard=True, dim=-1)

        x_quantized_fromIds = self.vq_net.vqvae.get_x_quantized_from_x_ids(
            codes_pick_gumbel_softmax.permute(0, 2, 3, 1).contiguous())
        # codes_pick = torch.argmax(F.softmax(pre_codes, dim = -1), dim = -1)
        # x_quantized_fromIds = self.vq_net.vqvae.get_x_quantized_from_x_ids(codes_pick.permute(0, 2, 1).contiguous())

        # if we do not need the post optimization, we can return here
        if not self.with_post_optimization:
            sample = self.vq_net.vqvae.forward_decoder_from_quantized_codes(x_quantized_fromIds)
            return sample
        # if we need the post optimization, we can continue to do the post optimization
        else:
            traj_data_inv_norm = hardCode_inv_transform_traj(traj_data)
            goal_dict = {"id": [0, 1, 2, 3, 4, 5],
                         "traj": traj_data_inv_norm}
            x_quantized_init = x_quantized_fromIds
            free_vars = []
            for ele in x_quantized_init:
                ele = ele.detach()
                ele.requires_grad = True
                free_vars.append(ele)

            optimizer = torch.optim.LBFGS(free_vars,
                                          lr=0.1,
                                          max_iter=2500,
                                          tolerance_change=1e-10,  # 1e-10, #1e-30,
                                          max_eval=None,
                                          history_size=300,
                                          line_search_fn='strong_wolfe')
            # Optimize
            gstep = 0
            closure = self.ik_fit(optimizer,
                             smpl=None,
                             source_kpts_model=goal_dict,
                             static_vars=None,
                             vp_model=self.vq_net,
                             on_step=None,
                             gstep=gstep,
                             motionLen=lengths,
                             control_joints=self.control_joints)
            optimizer.step(lambda: closure(free_vars, motion_length=lengths, data_transform=hardCode_inv_transform))
            free_vars = closure.free_vars
            print("optimization done.")
            sample = self.vq_net.vqvae.forward_decoder_from_quantized_codes(free_vars)
            return sample



    def ik_fit(self, optimizer, smpl, source_kpts_model, static_vars, vp_model, extra_params={}, on_step=None, gstep=0,
           motionLen=196, control_joints=None):
        data_loss = extra_params.get('data_loss', torch.nn.SmoothL1Loss(reduction='mean'))

        opt_map = [
            [0, 0],  # root
            [15, 1],  # head
            [20, 2],  # hand1    #left
            [21, 3],  # hand2   #right
            [10, 4],  # foot1   #left
            [11, 5],  # foot2  #right
        ]
        opt_map = [opt_map[joint] for joint in control_joints]
        opt_jointNum = np.array(opt_map)[:, 0].tolist()
        opt_trajNum = np.array(opt_map)[:, 1].tolist()

        def fit(free_vars, motion_length, data_transform):
            fit.gstep += 1
            optimizer.zero_grad()

            pre_Joint = vp_model.vqvae.forward_decoder_from_quantized_codes(free_vars)

            sample = data_transform(pre_Joint.permute(0, 2, 3, 1)).float()
            joint_positions = recover_from_ric(sample, 22)[:, 0, ...]
            if self.input_dim == 3:
                optimization_input = joint_positions
            else:
                joint_orient = recover_global_limb_rot_batch(joint_positions)
                optimization_input = torch.cat([joint_positions, joint_orient], dim=-1)

            opt_objs = {}
            opt_objs['data'] = 0

            # breakpoint()

            for batch_i in range(len(optimization_input)):
                opt_objs['data'] += data_loss(optimization_input[:, :motion_length[batch_i], opt_jointNum, :],
                                             source_kpts_model['traj'][:, :motion_length[batch_i], opt_trajNum,
                                             :].cuda())  # originally remove motion_length

            loss_total = torch.sum(torch.stack(list(opt_objs.values())))
            loss_total.backward(retain_graph=True)
            fit.free_vars = free_vars
            fit.final_loss = loss_total
            return loss_total

        fit.gstep = gstep
        fit.final_loss = None
        fit.free_vars = {}
        return fit

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        print('current epoch', self.current_epoch)

    def forward_feature(self, traj_data: torch.Tensor, input_text=None):
        pass

    def forward(self, traj_data: torch.Tensor, text=None, motion=None, lengths=None,
                data_samples: Optional[list] = None,
                mode: str = 'tensor'):
        if mode == 'tensor':
            return self.forward_feature(traj_data, text)
        elif mode == 'predict':
            if self.online_prediction:
                predictions = self.online_predict(traj_data=traj_data,
                                           input_text=text,
                                           lengths=lengths,
                                           data_samples=data_samples
                                           )
            else:
                predictions = self.predict(traj_data=traj_data,
                                           input_text=text,
                                           lengths=lengths,
                                           data_samples=data_samples
                                           )
            return predictions
        elif mode == 'loss':
            loss = self.loss(traj_data=traj_data,
                             input_text=text,
                             motion_hml=motion,
                             lengths=lengths,
                             data_samples=data_samples)
            return loss

    def online_predict(self, traj_data: torch.Tensor, input_text=None, lengths=None,
             data_samples: Optional[list] = None):
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        self.vq_net.eval()
        self.transformer_traj_model.eval()

        # breakpoint()
        batch_size = traj_data.shape[0]
        traj_data = traj_data.reshape((batch_size, 196, 6, self.input_dim))

        final_sample = torch.empty((batch_size, 263, 1, 196))
        for i in tqdm(range(1, traj_data.shape[1])):
            # split the traj data for each frame
            x_traj_masked = traj_data[:, :i, :, :].clone()
            # pad the input traj data to the same length
            if i < 196:
                x_traj_masked = F.pad(x_traj_masked, (0, 0, 0, 0, 0, 196 - i), "replicate")
            # breakpoint()

            # note: set dummy input text if input_text is None
            if input_text is None:
                input_text = [""] * traj_data.shape[0]
            _, pre_codes = self.transformer_traj_model(x_traj_masked, input_text)
            codes_pick_gumbel_softmax = F.gumbel_softmax(pre_codes, tau=1, eps=1e-10, hard=True, dim=-1)

            x_quantized_fromIds = self.vq_net.vqvae.get_x_quantized_from_x_ids(
                codes_pick_gumbel_softmax.permute(0, 2, 3, 1).contiguous())
            sample = self.vq_net.vqvae.forward_decoder_from_quantized_codes(x_quantized_fromIds)

            # breakpoint()

            final_sample[..., i-1:i] = sample[..., i-1:i]

        return final_sample


# below are directly copied from TLControl

def hardCode_inv_transform_traj(data):
    traj_mean_path = '/CT/EgoMocap/work/EgoOmniMocap/work_dirs/save_tmp/traj_mean.pt'
    traj_std_path = '/CT/EgoMocap/work/EgoOmniMocap/work_dirs/save_tmp/traj_std.pt'
    traj_std = torch.load(traj_std_path).to(data.device)
    traj_mean = torch.load(traj_mean_path).to(data.device)
    return data * traj_std + traj_mean

def hardCode_inv_transform(data):
    motion_mean_path = '/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_mean.pt'
    motion_std_path = '/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_std.pt'
    motion_std = torch.load(motion_std_path)
    motion_mean = torch.load(motion_mean_path)
    motion_std = torch.Tensor(motion_std).to(data.device)
    motion_mean = torch.Tensor(motion_mean).to(data.device)
    return data * motion_std + motion_mean

def random_mask_seq_update(x, mask_rates, max_mask_len=15, joint_mask=5, no_mask_prob=0.1,
                           mask_joint_prob=0.8):
    x_using = x.clone()
    T = x_using.size(1)
    data_dim = x_using.size(-1)

    mask = torch.ones_like(x_using[:, :, :, 0])
    mask_joints = None
    rand_number = random.random()

    if rand_number < no_mask_prob:
        return x_using

    if joint_mask is not None and rand_number < mask_joint_prob:
        mask_joints = random.sample([0, 1, 2, 3, 4, 5], 5)
        mask[:, :, mask_joints] *= .0
    else:
        for i, mask_rate in enumerate(mask_rates):
            total_masked = 0
            need_masked = int(round(mask_rate * T))
            while total_masked < need_masked:
                center = torch.randint(0, T, (1,)).item()
                if total_masked < need_masked - max_mask_len:
                    length = torch.randint(1, max_mask_len + 1, (1,)).item()
                else:
                    length = need_masked - total_masked

                left = max(0, center - length // 2)
                right = min(T, left + length)

                mask[:, left:right, i] *= .0
                total_masked = int(T - torch.sum(mask[0, :, i]).item())
    mask = mask.unsqueeze(-1)
    mask = mask.repeat(1, 1, 1, data_dim)
    return x_using * mask


class TransformerAutoencoder_hml(nn.Module):
    def __init__(self, transfomers_clip=False):
        super(TransformerAutoencoder_hml, self).__init__()

        self.transfomers_clip = transfomers_clip
        self.clip_model = None
        self.clip_text_model = None
        self.clip_tokenizer_transformer = None

        self.load_and_freeze_clip()


    def load_and_freeze_clip(self):
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if self.transfomers_clip:
            from transformers import AutoTokenizer, CLIPTextModel
            clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            clip_text_model.eval()
            for p in clip_text_model.parameters():
                p.requires_grad = False

            self.clip_text_model = clip_text_model
            self.clip_tokenizer_transformer = tokenizer

        clip_model, clip_preprocess = clip.load('ViT-B/32', device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        self.clip_model = clip_model

    def encode_text(self, raw_text, max_text_len=20):
        if self.transfomers_clip:
            inputs = self.clip_tokenizer_transformer(raw_text, padding='max_length',
                                                     truncation=True, return_tensors="pt").to("cuda")
            outputs = self.clip_text_model(**inputs)
            output_last_hidden_states = outputs.last_hidden_state.float()  # shape: [bs, seq_len, dim]
            # convert to shape: [seq_len, bs, dim]
            return output_last_hidden_states.permute(1, 0, 2)
        else:
            # raw_text - list (batch_size length) of strings with input text prompts
            # device = next(self.parameters()).device
            device = "cuda"
            # max_text_len = max_text_len  # Specific hardcoding for humanml dataset
            # assert max_text_len is None  # for this time, should be None
            if max_text_len is not None:
                default_context_length = 77
                context_length = max_text_len + 2  # start_token + 20 + end_token
                assert context_length < default_context_length
                texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(
                    device)  # [bs, context_length] # if n_tokens > context_length -> will truncate
                # print('texts', texts.shape)

                zero_pad = torch.zeros([texts.shape[0], default_context_length - context_length], dtype=texts.dtype,
                                       device=texts.device)
                texts = torch.cat([texts, zero_pad], dim=1)
                # print('texts after pad', texts.shape, texts)
            else:
                texts = clip.tokenize(raw_text, truncate=True).to(
                    device)  # [bs, context_length] # if n_tokens > 77 -> will truncate
            return self.clip_model.encode_text(texts).float()

    def encode_image(self, image):
        # image - torch.Tensor of shape [bs, 3, 224, 224]
        return self.clip_model.encode_image(image).float()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :, :]
        return self.dropout(x)

class TransformerAutoencoder_withCodes_hml_G2_noTraj(TransformerAutoencoder_hml):
    def __init__(self,
                 input_dim=3,
                 dropout=0.1,
                 num_emb=128,
                 max_text_len=20,
                 text_drop_rate=0,
                 image_drop_rate=0,
                 transfomers_clip=False,
                 ):
        super(TransformerAutoencoder_withCodes_hml_G2_noTraj, self).__init__(transfomers_clip=transfomers_clip)
        self.max_text_len = max_text_len
        self.input_dim = input_dim
        self.text_drop_rate = text_drop_rate
        self.image_drop_rate = image_drop_rate
        # if self.input_dim > 3:
        #    self.input_dim *= 28
        self.codesTimeLen = 49
        self.codes_realLength = 196 // self.codesTimeLen  # 4

        self.model_dim1 = 512
        self.model_dim2 = 256

        self.clip_dim = 512  # clip_dim

        # self.clip_version = 'ViT-B/32'
        # self.load_and_freeze_clip(self.clip_version)
        self.embed_text = nn.Linear(self.clip_dim, self.model_dim1)
        self.embed_image = nn.Linear(self.clip_dim, self.model_dim1)

        self.linear_in = nn.Linear(self.codes_realLength * self.input_dim, self.model_dim1)

        # self.pos_encoder = PositionalEncoding(self.model_dim_part, args.dropout)
        self.pos_encoder = PositionalEncoding(self.model_dim1, dropout)

        # self.tokenEmb = TokenTypeEncoding(self.model_dim, self.pos_encoder)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.model_dim1,
            nhead=4,  # args.num_heads,
            dim_feedforward=self.model_dim1 * 4,
            dropout=dropout)
        # Define transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=4)

        self.linear_mid_codeIdx = nn.Linear(self.model_dim1, self.model_dim2)
        # self.linear_mid_traj = nn.Linear(self.model_dim1, self.model_dim2)

        self.num_emb = num_emb

        decoder_layers1 = nn.TransformerEncoderLayer(
            d_model=self.model_dim2,
            nhead=4,  # Number of heads
            dim_feedforward=self.model_dim2 * 4,
            dropout=dropout
        )
        # Define transformer decoder
        self.transformer_codeIdx = nn.TransformerEncoder(
            encoder_layer=decoder_layers1,
            num_layers=3
        )
        self.project_codeIdx = nn.Linear(self.model_dim2, self.num_emb)

    def forward(self, x, x_text, x_image=None):  # Toekn size: T
        # x: bs, T, NumJoint, 3
        # y: bs, t, NumJoint

        assert x_text is not None
        if self.text_drop_rate > 0:
            if random.random() <= self.text_drop_rate:
                x_text = [""] * x.shape[0]
        enc_text = self.encode_text(x_text, max_text_len=self.max_text_len)
        input_feature = self.embed_text(enc_text)
        if not self.transfomers_clip:
            input_feature = input_feature.unsqueeze(0)
        if x_image is not None:
            enc_image = self.encode_image(x_image)
            input_image_feature = self.embed_image(enc_image).unsqueeze(0)
            if self.image_drop_rate > 0:
                if random.random() <= self.image_drop_rate:
                    input_image_feature = input_image_feature * 0.
            input_feature = torch.cat([input_feature, input_image_feature], dim=0)



        # x = x.reshape((-1, 196, 6*3))
        bs, T, NJoints, input_dim = x.shape
        x = x.permute(2, 1, 0, 3)  # x:  NumJoint, T, bs, model_dim
        Tt = T // self.codes_realLength  # the token number in the sequence
        x = x.reshape(NJoints, Tt, self.codes_realLength, bs, input_dim)
        x = x.permute(1, 0, 3, 2, 4)
        x = x.reshape(Tt * NJoints, bs, self.codes_realLength * input_dim)

        x = self.linear_in(x)

        # print('warning! just for testing the evaluation speed!')
        # input_feature  = torch.cat([input_feature.clone(), input_feature.clone(),
        #                             input_feature.clone(), input_feature.clone()], dim=0)

        x = torch.cat([input_feature, x], dim=0)


        x = self.pos_encoder(x)

        result_text = self.transformer_encoder(x)
        # result = result_text[1:, :, :] # Tt*NJoints, bs, model_dim

        pre_codes = self.transformer_codeIdx(self.linear_mid_codeIdx(result_text))
        pre_codes = pre_codes[len(input_feature):, :, :]
        pre_codes = self.project_codeIdx(pre_codes)
        pre_codes = pre_codes.reshape(Tt, NJoints, bs, -1)
        pre_codes = pre_codes.permute(2, 1, 0, 3)
        return None, pre_codes

#