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
from mmpose.datasets.datasets.ego_omni_mocap.humanml_utils.motion_process import recover_from_ric

class MLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

@POSE_ESTIMATORS.register_module()
class IMURegressor(BaseModel):
    def __init__(self, seq_len=196,
                 input_dim=263,
                 output_dim=6 * (6 + 3),  # output 6 imu sensor signal, each sensor signal has 6 acc and 3 ori
                 drop_out=0,
                 ):
        super(IMURegressor, self).__init__()
        # idea: use transformer to get imu sensor signal
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.model_dim1 = 512
        self.output_dim = output_dim
        self.linear_in = nn.Linear(self.input_dim, self.model_dim1)
        self.linear_out = nn.Linear(self.model_dim1, self.output_dim)

        self.layers = nn.ModuleList(
            [MLPLayer(self.model_dim1, self.model_dim1) for _ in range(4)]
        )


    def predict(self, motion_hml=None, lengths=None,
                data_samples: Optional[list] = None):
        """

        Args:
            motion_hml: shape: bs, T, 263
            lengths:
            data_samples:
            mode:

        Returns:

        """
        # breakpoint()
        if len(motion_hml.shape) == 4:
            motion_hml = torch.squeeze(motion_hml, dim=-2)
        x = torch.permute(motion_hml, (0, 2, 1))  # bs, T, input_dim = 263
        # recover joint positions from ric
        # human_joints = recover_from_ric(motion_hml, 22)
        # x = torch.reshape(human_joints, (-1, self.seq_len, 66))
        x = torch.permute(x, (1, 0, 2))  # T, bs, input_dim = 66 or 263
        x = self.linear_in(x)

        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.linear_out(x)
        x = torch.permute(x, (1, 0, 2))  # bs, T, output_dim = 6 * (6 + 3)
        return x

    def loss(self, motion_hml=None, lengths=None, init_aligned_imu_acc=None, init_aligned_imu_ori=None,
                data_samples: Optional[list] = None):
        pred_traj_data = self.predict(motion_hml, lengths, data_samples)
        batch_size, seq_len, _ = pred_traj_data.shape
        traj_data = torch.cat([init_aligned_imu_acc, init_aligned_imu_ori], dim=-1)
        gt_traj_data = torch.reshape(traj_data, (batch_size, seq_len, self.output_dim))
        loss = 0
        for i in range(batch_size):
            loss += F.mse_loss(pred_traj_data[i, :lengths[i]], gt_traj_data[i, :lengths[i]])
        # loss = F.mse_loss(pred_traj_data, gt_traj_data)
        loss_dict = {'imu_recon_loss': loss}
        return loss_dict

    def forward(self, motion_hml=None, lengths=None,
                init_aligned_imu_acc=None, init_aligned_imu_ori=None,
                data_samples: Optional[list] = None,
                mode: str = 'tensor'):
        if mode == 'tensor':
            raise Exception('mode tensor is not supported')
        elif mode == 'predict':
            return self.predict(motion_hml, lengths, data_samples)
        elif mode == 'loss':
            return self.loss(motion_hml, lengths, init_aligned_imu_acc, init_aligned_imu_ori, data_samples)
        else:
            raise Exception('mode not supported')
    def set_epoch(self, epoch):
        self.current_epoch = epoch
        print('current epoch', self.current_epoch)

if __name__ == '__main__':
    model = IMURegressor()
    model.set_epoch(1)
    print(model.current_epoch)