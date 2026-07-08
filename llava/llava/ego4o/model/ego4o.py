from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, \
    IGNORE_INDEX
from llava.ego4o.constants import MOTION_TOKEN_INDEX

from llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                    LlavaLlamaModel)
import os

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
    LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM, unpad_image
from llava.ego4o.motion_limb_vqvae.vqvae import HumanVQVAE
from llava.ego4o.motion_limb_vqvae.parser_util import mtm_args
from llava.ego4o.imu_tokenizer.ego_motion_mask_transformer import TransformerAutoencoder_withCodes_hml_G2_noTraj
from llava.mm_utils import get_anyres_image_grid_shape


class Ego4oMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(Ego4oMetaModel, self).__init__(config)

        self.config = config
        self.config.motion_out_dim = kwargs.get("motion_out_dim", 4096)
        self.load_motion_encoder = kwargs.get("load_motion_encoder", True)
        # self.torch_dtype = kwargs.get("torch_dtype", torch.float16)
        self.initialize_ego4o_modules(self.config)

        self.weight_loaded = False

    # def encode_imu(self, imu_acc, imu_ori):
    #     # here we encode the imu with the imu tokenizer
    #     # imu: [batch_size, seq_len, 3+6]
    #     # return: [batch_size, seq_len, out_dim]
    #     # first convert imu type to the same type as the model
    #     imu_acc = imu_acc.to(dtype=self.imu_tokenizer.encoder.weight.dtype)
    #     imu_ori = imu_ori.to(dtype=self.imu_tokenizer.encoder.weight.dtype)
    #     imu_acc_ori = torch.cat([imu_acc, imu_ori], dim=-1)
    #     batch_size = imu_acc_ori.shape[0]
    #     imu_acc_ori = imu_acc_ori.reshape((-1, self.seq_len, self.sensor_num, self.input_dim))
    #     input_text = [""] * imu_acc_ori.shape[0]
    #
    #     _, pre_codes = self.imu_tokenizer(imu_acc_ori, input_text)
    #     codes_pick_gumbel_softmax = F.gumbel_softmax(pre_codes, tau=1, eps=1e-10, hard=True, dim=-1)
    #     embedding_list = self.vq_net.vqvae.get_x_quantized_from_x_ids(
    #         codes_pick_gumbel_softmax.permute(0, 2, 3, 1).contiguous())
    #     embedding_motion = torch.cat(embedding_list, dim=1).permute(0, 2, 1)  # [batch_size, seq_len, out_dim]
    #     embedding_motion = embedding_motion.to(dtype=self.vq_net_postprocess[0].weight.dtype)
    #     output_motion_features = self.vq_net_postprocess(embedding_motion)
    #     return output_motion_features, codes_pick_gumbel_softmax

    def encode_motion(self, motion, debug=False):
        # here we encode the human motion with vq vae
        # motion: [batch_size, seq_len, embed_dim]
        # return: [batch_size, seq_len, out_dim]
        # first convert motion type to the same type as the model
        # print(f"Loading pretrained vq vae from {self.config.pretrained_vqvae_path}", flush=True)
        # self.vq_net.load_state_dict(torch.load(self.config.pretrained_vqvae_path), strict=True)
        self.vq_net.eval()
        self.vq_net.requires_grad_(False)

        motion = motion.to(dtype=self.vq_net_postprocess[0].weight.dtype)
        # embedding_list = self.vq_net.get_code_embedding(motion)

        code_index = self.vq_net.get_code_idx(motion)
        code_index_input = torch.stack(code_index, dim=-1)
        embedding_list = self.vq_net.vqvae.get_x_quantized_from_x_ids(code_index_input)

        # embedding_list: [batch_size, out_dim, seq_len]
        embedding_motion = torch.cat(embedding_list, dim=1).permute(0, 2, 1)  # [batch_size, seq_len, out_dim]
        embedding_motion = embedding_motion.to(dtype=self.vq_net_postprocess[0].weight.dtype)
        # if debug is true, run the decoder part of the vq vae and save the output
        if debug:
            x_decoder = self.vq_net.vqvae.decoder(embedding_motion.permute(0, 2, 1))
            x_out = self.vq_net.vqvae.postprocess(x_decoder)
            print('input: ', motion)
            print('output: ', x_out)
            # save x out to temp pkl file
            import pickle
            print('saving temp.pkl')
            with open('/CT/EgoMocap/work/LLaVA/temp_out.pkl', 'wb') as f:
                pickle.dump(x_out, f)
            with open('/CT/EgoMocap/work/LLaVA/temp_input.pkl', 'wb') as f:
                pickle.dump(motion.squeeze(-2).permute(0, 2, 1), f)
        output_motion_features = self.vq_net_postprocess(embedding_motion)
        return output_motion_features, code_index_input

    def encode_image_imu(self, imu_acc, imu_ori, img):
        if not hasattr(self, 'imu_tokenizer'):
            raise RuntimeError(
                'encode_image_imu called but the IMU tokenizer was not built '
                '(config.pretrained_imu_tokenizer_path is None — GT-motion variant). '
                'Feed motion_hml instead of IMU tensors.')
        self.imu_tokenizer.eval()
        self.imu_tokenizer.requires_grad_(False)
        self.vq_net.eval()
        self.vq_net.requires_grad_(False)

        imu_acc_ori = torch.cat([imu_acc, imu_ori], dim=-1)
        batch_size = imu_acc_ori.shape[0]
        imu_acc_ori = imu_acc_ori.reshape((-1, 148, 6, 9))
        input_text = [""] * imu_acc_ori.shape[0]

        # imu_acc_ori = imu_acc_ori.to(dtype=self.imu_tokenizer_postprocess[0].weight.dtype)
        # img = img.to(dtype=self.imu_tokenizer_postprocess[0].weight.dtype)

        _, pre_codes = self.imu_tokenizer(imu_acc_ori, input_text, x_image=img)
        code_index = F.gumbel_softmax(pre_codes, tau=1, eps=1e-10, hard=True, dim=-1)
        x_quantized_fromIds = self.vq_net.vqvae.get_x_quantized_from_x_ids(
            code_index.permute(0, 2, 3, 1).contiguous())

        # embedding_list: [batch_size, out_dim, seq_len]
        embedding_motion = torch.cat(x_quantized_fromIds, dim=1).permute(0, 2, 1)  # [batch_size, seq_len, out_dim]
        embedding_motion = embedding_motion.to(dtype=self.imu_tokenizer_postprocess[0].weight.dtype)
        # if debug is true, run the decoder part of the vq vae and save the output

        output_motion_features = self.imu_tokenizer_postprocess(embedding_motion)
        return output_motion_features, code_index

    def initialize_ego4o_modules(self, config):
        # load the modules
        # load the vq vae
        args = mtm_args()
        args.num_emb = 4096
        args.emb_dim = 64
        args.output_emb_width = 64
        self.vq_net = HumanVQVAE(args,  ## use args to define different parameters in different quantizers
                                 args.num_emb,
                                 args.emb_dim,
                                 args.output_emb_width)

        # load the pretrained transformers
        if self.load_motion_encoder:
            print(f"Loading pretrained vq vae from {config.pretrained_vqvae_path}", flush=True)
            pretrain_vae_state_dict = torch.load(config.pretrained_vqvae_path)
            if 'state_dict' in pretrain_vae_state_dict:
                pretrain_vae_state_dict = pretrain_vae_state_dict['state_dict']
            self.vq_net.load_state_dict(pretrain_vae_state_dict, strict=True)

        # freeze the vq vae
        self.vq_net.requires_grad_(False)
        # for i in range(len(self.vq_net.vqvae.limb_encoders)):
        #     self.vq_net.vqvae.limb_encoders[i].requires_grad_(False)
        # for i in range(len(self.vq_net.vqvae.quantizers)):
        #     self.vq_net.vqvae.quantizers[i].requires_grad_(False)

        mlp_depth = 2
        modules = [nn.Linear(args.emb_dim * 6, self.config.motion_out_dim)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(self.config.motion_out_dim, self.config.motion_out_dim))
        self.vq_net_postprocess = nn.Sequential(*modules)

        # init the weights
        for module in self.vq_net_postprocess.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight)
                nn.init.zeros_(module.bias)
        self.vq_net_postprocess.requires_grad_(True)

        # set the mode to train?
        # self.vq_net.eval()

        # init the vq_net decoder input network
        mlp_depth_for_decoder = 2
        modules_for_decoder = [nn.Linear(self.config.motion_out_dim, args.emb_dim * 6)]
        for _ in range(1, mlp_depth_for_decoder):
            modules_for_decoder.append(nn.GELU())
            modules_for_decoder.append(nn.Linear(args.emb_dim * 6, args.emb_dim * 6))
        self.vq_net_decoder_preprocess = nn.Sequential(*modules_for_decoder)

        # init the weights
        for module in self.vq_net_postprocess.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight)
                nn.init.zeros_(module.bias)
        self.vq_net_decoder_preprocess.requires_grad_(True)

        # load the imu tokenizer — only for the original IMU-input variant.
        # GT-motion variant (pretrained_imu_tokenizer_path=None): skip entirely,
        # motion enters via encode_motion and the IMU checkpoint is not needed.
        if getattr(config, 'pretrained_imu_tokenizer_path', None):
            self.imu_tokenizer = TransformerAutoencoder_withCodes_hml_G2_noTraj(
                input_dim=9, dropout=0.1, num_emb=4096,
                max_text_len=None,
                text_drop_rate=0,
                image_drop_rate=0,
                transfomers_clip=False
            )
            print(f"Loading pretrained imu tokenizer from {config.pretrained_imu_tokenizer_path}", flush=True)
            imu_tokenizer_state_dict = torch.load(config.pretrained_imu_tokenizer_path)
            if 'state_dict' in imu_tokenizer_state_dict:
                imu_tokenizer_state_dict = imu_tokenizer_state_dict['state_dict']

            # filter out model start from "transformer_traj_model" and remove the prefix

            imu_tokenizer_state_dict = {k.replace("transformer_traj_model.", ""): v for k, v in imu_tokenizer_state_dict.items() if k.startswith("transformer_traj_model.")}
            self.imu_tokenizer.load_state_dict(imu_tokenizer_state_dict, strict=True)

            # freeze the imu tokenizer
            for param in self.imu_tokenizer.parameters():
                param.requires_grad = False

            self.imu_tokenizer.eval()

            mlp_depth_for_imu_tokenizer = 2
            modules = [nn.Linear(args.emb_dim * 6, self.config.motion_out_dim)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(self.config.motion_out_dim, self.config.motion_out_dim))
            self.imu_tokenizer_postprocess = nn.Sequential(*modules)
            self.imu_tokenizer_postprocess.requires_grad_(True)


class Ego4oConfig(LlamaConfig):
    model_type = "ego4o"
    motion_out_dim = 4096
    motion_recon_weight = 5.0
    # GT-motion reproduction (local): stage-1 script maintains the best_vqvae.pth
    # symlink; override via --pretrained_vqvae_path (absorbed into the config by
    # from_pretrained since it is a declared config attribute).
    pretrained_vqvae_path = '/local/home/dhollidt/repos/ego4o-code-release/EgoOmniMocap/work_dirs/train_nymeria_vqvae_4096_64_hml/best_vqvae.pth'
    # None = GT-motion variant: the IMU multi-modal encoder is never built/loaded
    # (original release: '/CT/EgoMocap/work/EgoOmniMocap/work_dirs/train_nymeria_random_image_text/best_C-MPJPE_epoch_23.pth')
    pretrained_imu_tokenizer_path = None
    input_modality = 'motion'  # motion or imu sensor

class Ego4oModel(Ego4oMetaModel, LlavaLlamaModel):
    config_class = Ego4oConfig

    def __init__(self, config, **kwargs):
        super(Ego4oModel, self).__init__(config, **kwargs)


class Ego4oForCausalLM(LlavaLlamaForCausalLM):
    config_class = Ego4oConfig

    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(Ego4oForCausalLM, self).__init__(config)
        self.model = Ego4oModel(config, **kwargs)

        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.is_pretrain = kwargs.get("is_pretrain", False)
        self.motion_token_loss_weight = kwargs.get("motion_token_loss_weight", 1.0)


        # Initialize weights and apply final processing
        self.post_init()

    # override function: prepare_inputs_labels_for_multimodal
    def prepare_inputs_labels_for_motion(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, motion=None, image_for_imu=None, imu_acc=None, imu_ori=None):

        # first use the parent function to get the inputs and labels
        vision_tower = self.get_vision_tower()
        if images is None and motion is None and imu_acc is None:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx],
                                                                                            self.config.image_grid_pinpoints,
                                                                                            self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(
                                    image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # add the method to encode the motion features

        if image_for_imu is not None and imu_acc is not None and imu_ori is not None:
            motion_features, motion_code_index = self.model.encode_image_imu(imu_acc=imu_acc, imu_ori=imu_ori,
                                                                             img=image_for_imu)
        else:
            motion_features, motion_code_index = self.model.encode_motion(motion, debug=False)
        # motion_features = self.model.encode_motion(motion, debug=True)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in
                     zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        # add motion index and image index
        # only used when the shape of loaded motion / image is not same as the batch size
        # cur_image_idx = 0
        # cur_motion_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_motions = (cur_input_ids == MOTION_TOKEN_INDEX).sum()
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # assert that have motion or image token
            assert num_motions + num_images > 0, f"Error: no motion or image token in the sentence, {cur_input_ids}"

            motion_token_indices = [-1] + torch.where(cur_input_ids == MOTION_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            motion_image_token_indices = sorted(motion_token_indices + image_token_indices)[1:-1]  # need to remove the -1 and the last one
            motion_token_belong = []
            for i in range(1, len(motion_image_token_indices) - 1):
                if motion_image_token_indices[i] in motion_token_indices:
                    motion_token_belong.append('motion')
                else:
                    motion_token_belong.append('image')
            cur_input_ids_only_text = []
            cur_labels = labels[batch_idx]
            cur_labels_only_text = []
            for i in range(len(motion_image_token_indices) - 1):
                cur_input_ids_only_text.append(cur_input_ids[motion_image_token_indices[i] + 1: motion_image_token_indices[i + 1]])
                cur_labels_only_text.append(cur_labels[motion_image_token_indices[i] + 1:motion_image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_only_text]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_only_text))
            cur_input_embeds_only_text = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            assert num_motions + num_images == 1 or num_motions + num_images == 2, f"Error: the number of motion tokens and image tokens is not 1 or 2, {num_motions}, {num_images}"

            for i in range(num_motions + num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_only_text[i])
                cur_new_labels.append(cur_labels_only_text[i])
                if i < num_motions + num_images:
                    # check this belongs to the image or motion
                    if motion_token_belong[i] == 'motion':
                        # note: if the motion and image is always loaded, then use batch index
                        # if motion and image do not exist, then use cur_motion_idx and cur_image_idx
                        # cur_motion_features = motion_features[cur_motion_idx]
                        cur_motion_features = motion_features[batch_idx]
                        # cur_motion_idx += 1
                        cur_new_input_embeds.append(cur_motion_features)
                        if len(cur_motion_features) % 37 != 0:
                            print(f"Error: the length of motion hidden state is not a multiple of 37, ")
                            # pdb.set_trace()
                        cur_new_labels.append(torch.full((cur_motion_features.shape[0],), IGNORE_INDEX,
                                                         device=cur_labels.device, dtype=cur_labels.dtype))
                    else:
                        # cur_image_features = image_features[cur_image_idx]
                        # cur_image_idx += 1
                        cur_image_features = image_features[batch_idx]
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX,
                                                         device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)


            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype,
                                       device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                              device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                             device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, motion_code_index

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,  # not used here
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        motion_hml: Optional[torch.Tensor] = None,
        motion_hml_source: Optional[str] = None,
        img_for_imu: Optional[torch.Tensor] = None,
        init_aligned_imu_acc: Optional[torch.Tensor] = None,
        init_aligned_imu_ori: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            input_ids_clone = input_ids.clone()  # clone for debug

            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                motion_code_index
            ) = self.prepare_inputs_labels_for_motion(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                motion=motion_hml,
                image_for_imu=img_for_imu,
                imu_acc=init_aligned_imu_acc,
                imu_ori=init_aligned_imu_ori,
            )

        # pdb.set_trace()

        result = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        # pdb.set_trace()
        return result

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        motion_hml: Optional[torch.Tensor] = None,
        img_for_imu: Optional[torch.Tensor] = None,
        init_aligned_imu_acc: Optional[torch.Tensor] = None,
        init_aligned_imu_ori: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        # pdb.set_trace()

        if motion_hml is not None or init_aligned_imu_acc is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _,
            ) = self.prepare_inputs_labels_for_motion(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                motion=motion_hml,
                image_for_imu=img_for_imu,
                imu_acc=init_aligned_imu_acc,
                imu_ori=init_aligned_imu_ori,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        motion_hml = kwargs.pop("motion_hml", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        if motion_hml is not None:
            inputs['motion_hml'] = motion_hml
        return inputs


AutoConfig.register("ego4o", Ego4oConfig)
AutoModelForCausalLM.register(Ego4oConfig, Ego4oForCausalLM)
