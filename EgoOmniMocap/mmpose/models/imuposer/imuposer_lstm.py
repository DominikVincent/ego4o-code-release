import pdb
import random
from typing import Optional, Union, Dict
import numpy as np
import torch
from mmengine.model import BaseModel
import torch.nn as nn

from mmpose.models.builder import POSE_ESTIMATORS
import torch.nn.functional as F

@POSE_ESTIMATORS.register_module()
class IMUPoserLSTMModel(BaseModel):
    def __init__(self,
                 input_dim=3 + 6,
                 sensor_num=6,
                 seq_len=196,
                 ):
        super(IMUPoserLSTMModel, self).__init__()

        self.input_dim = input_dim
        self.sensor_num = sensor_num
        self.seq_len = seq_len

        self.dip_model = RNN(n_input=input_dim * sensor_num, n_output=263, n_hidden=512, bidirectional=True)
        self.current_epoch = 0

    def forward_feature(self, imu_acc, imu_ori):
        r"""
        Forward pass of the model to get the features
        """
        batch_size, seq_len, sensor_num, _ = imu_acc.shape
        imu_acc = imu_acc.view(batch_size, seq_len, -1)
        imu_ori = imu_ori.view(batch_size, seq_len, -1)

        seq_len_input = [seq_len] * batch_size
        seq_len_input = torch.as_tensor(seq_len_input, dtype=torch.int64).cpu()

        x = torch.cat([imu_acc, imu_ori], dim=-1)
        x, x_lens, _ = self.dip_model(x, x_lens=seq_len_input)

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


class RNN(nn.Module):
    r"""
    An RNN Module including a linear input layer, an RNN, and a linear output layer.
    """
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2, bidirectional=True, dropout=0.2):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(n_hidden, n_hidden, n_rnn_layer, bidirectional=bidirectional, batch_first=True)
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.linear2 = nn.Linear(n_hidden * (2 if bidirectional else 1), n_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_lens=None, h=None):
        r"""
        Gets a padded batch
        Step 1: pass through linear layer
        Step 2: pack the padded sequences
        Step 3: pass the packed sequences through the lstm layers
        Step 4: unpack the packed sequences
        Step 5: pass the unpacked sequences (padded) to the linear layers
        Step 6: return the output and output lengths
        """
        # Step 1
        x = F.relu(self.linear1(self.dropout(x)))

        # Step 2
        x = nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)

        # Step 3
        x, h = self.rnn(x, h)

        # Step 4
        x, output_lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # Step 5
        x = self.linear2(x)

        # how to unpad
        # outputs = [output[:output_lens[i]] for i, output in enumerate(outputs)]

        return x, output_lens, h

if __name__ == '__main__':
    model = IMUPoserLSTMModel()
    model.set_epoch(10)
    input_1 = torch.randn(2, 148, 6, 3)
    input_2 = torch.randn(2, 148, 6, 6)
    motion_hml = torch.randn(2, 148, 263)
    output = model(init_aligned_imu_acc=input_1, init_aligned_imu_ori=input_2, motion_hml=motion_hml, mode='loss')
    print(output)