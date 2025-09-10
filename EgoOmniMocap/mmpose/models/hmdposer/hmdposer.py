import pdb
from typing import Optional, Union, Dict
import numpy as np
import torch
from mmengine.model import BaseModel
import torch.nn as nn

from mmpose.models.builder import POSE_ESTIMATORS
import torch.nn.functional as F
from mmpose.models.hmdposer.hmdposer_net import HMD_imu_HME_Universe


@POSE_ESTIMATORS.register_module()
class HMDPoserModel(BaseModel):
    def __init__(self,
                 input_dim=3 + 6,
                 sensor_num=6,
                 seq_len=196,
                 ):
        super(HMDPoserModel, self).__init__()

        self.input_dim = input_dim
        self.sensor_num = sensor_num
        self.seq_len = seq_len

        self.hmdposer_net = HMD_imu_HME_Universe(input_dim=135)
        self.current_epoch = 0

    def rotation6d_to_matrix(self, d6):
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        # breakpoint()

        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)

    def matrix_to_rotation6d(self, matrix):
        batch_dim = matrix.size()[:-2]
        return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

    def calculate_rot_velocity(self, rot_6d_seq):
        rot_matrices = self.rotation6d_to_matrix(rot_6d_seq)
        R1 = rot_matrices[:, -1:]
        R2 = rot_matrices[:, 1:]
        relative_rot = torch.matmul(R2, R1.transpose(-2, -1))
        # Pad identity matrices to maintain sequence length
        identity_matrix = torch.eye(3, device=rot_6d_seq.device).unsqueeze(0).unsqueeze(0)
        identity_matrix = identity_matrix.repeat(rot_6d_seq.shape[0], 1, 1, 1)
        relative_rot_pad = torch.cat([relative_rot, identity_matrix], dim=1)
        rot_velocity = self.matrix_to_rotation6d(relative_rot_pad)
        return rot_velocity

    def forward_feature(self, imu_acc, imu_ori):

        torch.cuda.reset_peak_memory_stats()  # Reset peak memory stats
        r"""
        Forward pass of the model to get the features
        """
        batch_size, seq_len, sensor_num, _ = imu_acc.shape
        # root 0, head 1 ,left wrist 2 , right wrist 3 , left hip 4 , right hip 5

        # imu_acc = imu_acc.view(batch_size, seq_len, -1)
        # imu_ori = imu_ori.view(batch_size, seq_len, -1)

        net_input = {}
        net_input['head_rot'] = imu_ori[:, :, 1]
        net_input['head_rot_vel'] = self.calculate_rot_velocity(imu_ori[:, :, 1])
        net_input['head_acc'] = imu_acc[:, :, 1]
        net_input['head_vel'] = torch.zeros_like(imu_acc[:, :, 1])

        net_input['lhand_rot'] = imu_ori[:, :, 2]
        net_input['lhand_rot_vel'] = self.calculate_rot_velocity(imu_ori[:, :, 2])
        net_input['lhand_acc'] = imu_acc[:, :, 2]
        net_input['lhand_vel'] = torch.zeros_like(imu_acc[:, :, 2])

        net_input['rhand_rot'] = imu_ori[:, :, 3]
        net_input['rhand_rot_vel'] = self.calculate_rot_velocity(imu_ori[:, :, 3])
        net_input['rhand_acc'] = imu_acc[:, :, 3]
        net_input['rhand_vel'] = torch.zeros_like(imu_acc[:, :, 3])

        net_input['lfoot_rot'] = imu_ori[:, :, 4]
        net_input['lfoot_rot_vel'] = self.calculate_rot_velocity(imu_ori[:, :, 4])
        net_input['lfoot_acc'] = imu_acc[:, :, 4]

        net_input['rfoot_rot'] = imu_ori[:, :, 5]
        net_input['rfoot_rot_vel'] = self.calculate_rot_velocity(imu_ori[:, :, 5])
        net_input['rfoot_acc'] = imu_acc[:, :, 5]


        x = self.hmdposer_net(net_input)

        # now the output shape is (batch_size, seq_len, 263)
        # convert it to (batch_size, 263, 1, seq_len)
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(2)

        return x

    def predict(self, imu_acc, imu_ori, data_samples):
        r"""
        Forward pass of the model to get the predictions
        """
        x = self.forward_feature(imu_acc, imu_ori)
        return x

    def loss(self, imu_acc, imu_ori, motion_hml, data_samples):
        r"""
        Forward pass of the model to get the loss
        """
        x = self.forward_feature(imu_acc, imu_ori)
        # print(x.shape, motion_hml.shape)
        # motion_hml = motion_hml.squeeze(2)
        # motion_hml = motion_hml.permute(0, 2, 1)
        recon_loss = F.mse_loss(x, motion_hml)
        loss = {'recon_loss': recon_loss}
        return loss

    def forward(self,
                init_aligned_imu_acc, init_aligned_imu_ori,
                motion_hml=None, lengths=None,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        if mode == 'tensor':
            return self.forward_feature(init_aligned_imu_acc, init_aligned_imu_ori)
        elif mode == 'predict':
            predictions = self.predict(imu_acc=init_aligned_imu_acc,
                                       imu_ori=init_aligned_imu_ori,
                                       data_samples=data_samples
                                       )
            return predictions
        elif mode == 'loss':
            loss = self.loss(imu_acc=init_aligned_imu_acc,
                             imu_ori=init_aligned_imu_ori,
                             motion_hml=motion_hml,
                             data_samples=data_samples)
            return loss

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        print('current epoch', self.current_epoch)