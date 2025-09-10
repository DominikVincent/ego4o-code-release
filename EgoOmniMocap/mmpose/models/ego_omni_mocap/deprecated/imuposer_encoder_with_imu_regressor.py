import random
from typing import Optional

import numpy as np
import torch
from mmengine.model import BaseModel
import torch.nn as nn
import math

from mmengine.runner import load_checkpoint
from tqdm import tqdm

from mmpose.models.builder import POSE_ESTIMATORS, build_pose_estimator
import torch.nn.functional as F
import clip

from .ego_motion_mask_transformer import TransformerAutoencoder_withCodes_hml_G2_noTraj
from .vqvae import vqvae as vqvae
from .vqvae.parser_util import mtm_args
from ..utils.geometry import rotation_6d_to_matrix
from ...datasets.datasets.ego_omni_mocap.humanml_utils.motion_process import recover_from_ric, recover_global_limb_rot, \
    recover_global_limb_rot_batch


@POSE_ESTIMATORS.register_module()
class IMUPoserEncoderRegressorOptim(BaseModel):
    def __init__(self,
                 input_dim=3 + 6,
                 sensor_num=5,
                 seq_len=196,
                 drop_out=0.1,
                 num_emb=128,
                 text_mask_rate=0.5,
                 recon_loss_weight=0.001,
                 acc_weight=0,
                 ori_weight=1,
                 reg_imu_rot_loss=0.1,
                 with_post_optimization=False,
                 pretrained_transformer=None,
                 pretrained_vqvae=None,
                 init_cfg=None,
                 imu_regressor_cfg=None,
                 imu_regressor_weight=None
                 ):
        super().__init__(init_cfg=init_cfg)

        self.acc_weight = acc_weight
        self.ori_weight = ori_weight
        self.reg_imu_rot_loss=reg_imu_rot_loss
        self.current_epoch = 0
        self.text_mask_rate = text_mask_rate
        self.input_dim = input_dim
        self.sensor_num = sensor_num
        self.seq_len = seq_len
        self.with_post_optimization = with_post_optimization

        self.transformer_traj_model = TransformerAutoencoder_withCodes_hml_G2_noTraj(input_dim, drop_out, num_emb)
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

        # loss
        self.bce_loss_fn = torch.nn.CrossEntropyLoss()

        self.recon_loss_weight = recon_loss_weight

        # ----------------------construct the imu regressor----------------------
        self.imu_regressor_cfg = imu_regressor_cfg
        self.imu_regressor_weight = imu_regressor_weight
        self.imu_regressor = build_pose_estimator(self.imu_regressor_cfg)
        # load the imu regressor weight

        imu_regressor_state_dict = torch.load(self.imu_regressor_weight, map_location='cpu')
        # breakpoint()
        self.imu_regressor.load_state_dict(imu_regressor_state_dict['state_dict'], strict=True)
        self.imu_regressor.eval()
        for param in self.imu_regressor.parameters():
            param.requires_grad = False

    def loss(self, imu_acc, imu_ori, input_text=None, motion_hml=None, lengths=None,
             data_samples: Optional[list] = None) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        self.vq_net.eval()

        imu_acc_ori = torch.cat([imu_acc, imu_ori], dim=-1)
        batch_size = imu_acc_ori.shape[0]
        imu_acc_ori = imu_acc_ori.reshape((-1, self.seq_len, self.sensor_num, self.input_dim))

        x_label_idx = self.vq_net.get_code_idx(motion_hml).detach()  # .permute(0, 2, 1)

        if random.random() <= self.text_mask_rate:
            input_text = [""] * batch_size
        if input_text is None:
            input_text = [""] * imu_acc_ori.shape[0]

        _, pre_codes = self.transformer_traj_model(imu_acc_ori, input_text)
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

    def predict(self, imu_acc, imu_ori, input_text=None, lengths=None, data_samples: Optional[list] = None):
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        self.vq_net.eval()
        self.transformer_traj_model.eval()

        # breakpoint()

        # concat the imu acc and ori in their last dimension
        imu_acc_ori = torch.cat([imu_acc, imu_ori], dim=-1)
        # print(imu_acc_ori.shape)
        imu_acc_ori = imu_acc_ori.reshape((-1, self.seq_len, self.sensor_num, self.input_dim))
        joint_mask = 5


        # note: set dummy input text if input_text is None
        if input_text is None:
            input_text = [""] * imu_acc_ori.shape[0]
        _, pre_codes = self.transformer_traj_model(imu_acc_ori, input_text)

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
            limb_ori_gt = get_limb_orientation_from_imu(imu_ori)
            limb_acc_gt = imu_acc.detach()
            limb_acc_gt.requires_grad = False
            imu_rot_gt = imu_ori.detach().clone()
            input_dict = {
                'limb_ori_gt': limb_ori_gt,
                'limb_acc_gt': limb_acc_gt,
                'imu_rot_gt': imu_rot_gt,
            }

            # firstly transform the imu ori to the orientation of the joint
            x_quantized_init = x_quantized_fromIds
            free_vars = []
            for ele in x_quantized_init:
                ele = ele.detach()
                ele.requires_grad = True
                free_vars.append(ele)

            optimizer = torch.optim.LBFGS(free_vars,
                                          lr=1,
                                          max_iter=200,
                                          tolerance_change=1e-6,  # 1e-10, #1e-30,
                                          max_eval=None,
                                          history_size=20,
                                          line_search_fn='strong_wolfe'
                                          )
            # Optimize
            gstep = 0
            closure = self.ik_fit(optimizer,
                             smpl=None,
                             source_kpts_model=input_dict,
                             static_vars=None,
                             vp_model=self.vq_net,
                             on_step=None,
                             gstep=gstep,
                             motionLen=lengths,
                             control_joints=None)
            optimizer.step(lambda: closure(free_vars, motion_length=lengths, data_transform=hardCode_inv_transform))
            free_vars = closure.free_vars
            print("optimization done.")
            sample = self.vq_net.vqvae.forward_decoder_from_quantized_codes(free_vars)
            return sample



    def ik_fit(self, optimizer, smpl, source_kpts_model, static_vars, vp_model, extra_params={}, on_step=None, gstep=0,
           motionLen=196, control_joints=None):
        data_loss = extra_params.get('data_loss', torch.nn.SmoothL1Loss(reduction='mean'))

        limb_ori_gt = source_kpts_model['limb_ori_gt']
        limb_acc_gt = source_kpts_model['limb_acc_gt']
        imu_rot_gt = source_kpts_model['imu_rot_gt']

        def fit(free_vars, motion_length, data_transform):
            fit.gstep += 1
            optimizer.zero_grad()

            pre_Joint = vp_model.vqvae.forward_decoder_from_quantized_codes(free_vars)

            sample = data_transform(pre_Joint.permute(0, 2, 3, 1)).float()


            joint_positions = recover_from_ric(sample, 22)[:, 0, ...]

            # get joint orientation from joint position
            limb_ori_pred = get_limb_orientation_from_joints(joint_positions)

            limb_acc_pred = syn_acc(joint_positions)
            limb_acc_gt_mid = limb_acc_gt[:, 1:-1]

            opt_objs = {}
            opt_objs['data'] = 0

            if self.reg_imu_rot_loss > 0:
                # calculate the error between reconstructed IMU signal and original IMU signal
                regressed_imu_signal = self.imu_regressor.predict(motion_hml=pre_Joint)
                regressed_imu_signal = regressed_imu_signal.reshape(-1, self.seq_len, 6, 3 + 6)
                regressed_imu_acc = regressed_imu_signal[..., :3]
                regressed_imu_rot = regressed_imu_signal[..., 3:]

                # calculate the loss between the regressed imu ori and gt imu ori
                reg_imu_ori_loss = data_loss(regressed_imu_rot[..., 1:, :], imu_rot_gt[..., 1:, :])
                # print('acc loss:', acc_loss.item())
                opt_objs['data'] += self.reg_imu_rot_loss * reg_imu_ori_loss
                # add logs here...
                current_step = fit.gstep
                if current_step % 10 == 0:
                    # save the motion hml and regressed imu signal
                    # todo: save the signal
                    pass

            acc_loss = 0
            acc_loss += data_loss(limb_acc_pred[:, :, 1] * limb_ori_gt['head'][1][:, 1:-1][..., None],
                                  limb_acc_gt_mid[:, :,1] * limb_ori_gt['head'][1][:, 1:-1][..., None])
            acc_loss += data_loss(limb_acc_pred[:, :, 2] * limb_ori_gt['left_arm'][1][:, 1:-1][..., None],
                                          limb_acc_gt_mid[:, :, 2] * limb_ori_gt['left_arm'][1][:, 1:-1][..., None])
            acc_loss += data_loss(limb_acc_pred[:, :, 3] * limb_ori_gt['right_arm'][1][:, 1:-1][..., None],
                                          limb_acc_gt_mid[:, :, 3] * limb_ori_gt['right_arm'][1][:, 1:-1][..., None])
            acc_loss += data_loss(limb_acc_pred[:, :, 4] * limb_ori_gt['left_leg'][1][:, 1:-1][..., None],
                                          limb_acc_gt_mid[:, :, 4] * limb_ori_gt['left_leg'][1][:, 1:-1][..., None])
            acc_loss += data_loss(limb_acc_pred[:, :, 5] * limb_ori_gt['right_leg'][1][:, 1:-1][..., None],
                                          limb_acc_gt_mid[:, :, 5] * limb_ori_gt['right_leg'][1][:, 1:-1][..., None])

            ori_loss = 0
            ori_loss += data_loss(limb_ori_pred['left_arm'] * limb_ori_gt['left_arm'][1][..., None],
                                          limb_ori_gt['left_arm'][0] * limb_ori_gt['left_arm'][1][..., None])
            ori_loss += data_loss(limb_ori_pred['right_arm'] * limb_ori_gt['right_arm'][1][..., None],
                                          limb_ori_gt['right_arm'][0] * limb_ori_gt['right_arm'][1][..., None])
            ori_loss += data_loss(limb_ori_pred['left_leg'] * limb_ori_gt['left_leg'][1][..., None],
                                          limb_ori_gt['left_leg'][0] * limb_ori_gt['left_leg'][1][..., None])
            ori_loss += data_loss(limb_ori_pred['right_leg'] * limb_ori_gt['right_leg'][1][..., None],
                                          limb_ori_gt['right_leg'][0] * limb_ori_gt['right_leg'][1][..., None])
            opt_objs['data'] += ori_loss * self.ori_weight

            if self.acc_weight > 0:
                opt_objs['data'] += self.acc_weight * acc_loss

            loss_total = torch.sum(torch.stack(list(opt_objs.values())))

            print('loss total:', loss_total.item())

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

    def forward(self, init_aligned_imu_acc, init_aligned_imu_ori,
                text=None, motion_hml=None, lengths=None,
                data_samples: Optional[list] = None,
                mode: str = 'tensor'):
        if mode == 'tensor':
            return self.forward_feature(init_aligned_imu_acc, init_aligned_imu_ori, text)
        elif mode == 'predict':
            predictions = self.predict(imu_acc=init_aligned_imu_acc,
                                       imu_ori=init_aligned_imu_ori,
                                       input_text=text,
                                       lengths=lengths,
                                       data_samples=data_samples
                                       )
            return predictions
        elif mode == 'loss':
            loss = self.loss(imu_acc=init_aligned_imu_acc,
                             imu_ori=init_aligned_imu_ori,
                             input_text=text,
                             motion_hml=motion_hml,
                             lengths=lengths,
                             data_samples=data_samples)
            return loss

def syn_acc(joint_positions):
    r"""
    Synthesize accelerations from joint positions.
    """
    # joint_positions: (B, seq_len, 22, 3)
    velocity_2 = joint_positions[:, 2:] - joint_positions[:, 1:-1]
    velocity_1 = joint_positions[:, 1: -1] - joint_positions[:, :-2]
    acc = (velocity_2 - velocity_1) * 3600 / 30 * (25 / 60) * (25 / 60)
    acc = acc[:, :, [0, 15, 20, 21, 1, 2]]
    return acc

def get_limb_orientation_from_imu(imu_ori):
    # imu ori shape: (B, seq_len, sensor_num, 6)
    # convert from 6d representation to matrix representation
    imu_ori = imu_ori.detach()
    imu_ori.requires_grad = False
    imu_ori_matrix = rotation_6d_to_matrix(imu_ori)
    imu_ori_matrix_det = torch.linalg.det(imu_ori_matrix)
    # if the det is similar to 0, then this sensor is not available
    sensor_available = imu_ori_matrix_det > 0.1

    limb_ori = torch.zeros_like(imu_ori_matrix[:, :, :, :3, 0]).float().to(imu_ori.device)

    # do not convert the head rotation
    # sensor_available[:, :, 1] = False
    head_ori = imu_ori_matrix[:, :, 1, :3, 0]
    head_available = sensor_available[:, :, 1].float()
    left_arm_ori = imu_ori_matrix[:, :, 2, :3, 0]
    left_arm_available = sensor_available[:, :, 2].float()
    right_arm_ori = -1 * imu_ori_matrix[:, :, 3, :3, 0]
    right_arm_available = sensor_available[:, :, 3].float()
    left_leg_ori = -1 * imu_ori_matrix[:, :, 4, :3, 1]
    left_leg_available = sensor_available[:, :, 4].float()
    right_leg_ori = -1 * imu_ori_matrix[:, :, 5, :3, 1]
    right_leg_available = sensor_available[:, :, 5].float()

    result_dict = {
        'head': (head_ori, head_available),
        'left_arm': (left_arm_ori, left_arm_available),
        'right_arm': (right_arm_ori, right_arm_available),
        'left_leg': (left_leg_ori, left_leg_available),
        'right_leg': (right_leg_ori, right_leg_available)
    }
    return result_dict

def get_limb_orientation_from_joints(joint_positions):
    # joint_positions: (B, seq_len, 22, 3)
    # limb_orientation: (B, seq_len, 6, 3)

    B, seq_len = joint_positions.shape[0], joint_positions.shape[1]

    # get the 3d orientation in each limb
    left_arm_ori = joint_positions[:, :, 20] - joint_positions[:, :, 18]
    # normalize the length
    left_arm_ori = left_arm_ori / torch.norm(left_arm_ori, dim=-1, keepdim=True)
    right_arm_ori = joint_positions[:, :, 21] - joint_positions[:, :, 19]
    right_arm_ori = right_arm_ori / torch.norm(right_arm_ori, dim=-1, keepdim=True)
    left_leg_ori = joint_positions[:, :, 4] - joint_positions[:, :, 1]
    left_leg_ori = left_leg_ori / torch.norm(left_leg_ori, dim=-1, keepdim=True)
    right_leg_ori = joint_positions[:, :, 5] - joint_positions[:, :, 2]
    right_leg_ori = right_leg_ori / torch.norm(right_leg_ori, dim=-1, keepdim=True)
    return_dict = {
        'left_arm': left_arm_ori,
        'right_arm': right_arm_ori,
        'left_leg': left_leg_ori,
        'right_leg': right_leg_ori
    }
    return return_dict


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

