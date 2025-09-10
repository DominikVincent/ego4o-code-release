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
from ...datasets.datasets.ego_omni_mocap.humanml_utils.motion_process import recover_from_ric, recover_global_limb_rot


@POSE_ESTIMATORS.register_module()
class EgoMotionVQVAEOptimizer(BaseModel):
    def __init__(self):
        super().__init__()



    def forward(self, traj_data: torch.Tensor, text=None, motion=None, lengths=None,
                data_samples: Optional[list] = None,
                mode: str = 'tensor'):
        if mode == 'tensor':
            return self.forward_feature(traj_data, text)
        elif mode == 'predict':
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

    def forward_feature(self, traj_data, text):
        raise NotImplementedError
        pass

    def loss(self, traj_data, input_text, motion_hml, lengths, data_samples):
        raise NotImplementedError
        pass

    def predict(self, traj_data, input_text, lengths, data_samples):
        goal_dict = {"id": [0, 1, 2, 3, 4, 5],
                     "traj": traj_data}



        return sample


    def ik_fit_with_rot(self, optimizer, source_kpts_model, vp_model, extra_params={}, gstep=0,
            control_joints=[1, 2, 3]):
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

            sample = data_transform(pre_Joint[0].permute(1, 2, 0)).float()
            joint_positions = recover_from_ric(sample, 22)[0, ...]
            # calculate joint global rotation
            joint_orient = recover_global_limb_rot(joint_positions)
            joint_orient_control = joint_orient[:motion_length, opt_jointNum, :]
            joint_positions_control = joint_positions[:motion_length, opt_jointNum, :]

            joint_pos_orient_control = torch.cat([joint_positions_control, joint_orient_control], dim=-1)

            opt_objs = {}

            opt_objs['data'] = data_loss(joint_pos_orient_control,
                                         source_kpts_model['traj'][:motion_length, opt_trajNum,
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