import pickle
from collections import defaultdict
from os import path as osp
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmpose.registry import METRICS
from ..functional import keypoint_mpjpe
from ...datasets.datasets.ego_omni_mocap.humanml_utils.motion_process import recover_from_ric
from mmpose.utils.keypoints_mapping.joint_converter import JointConverter
from ...utils.humanml_utils.quaternion import qrot_np


@METRICS.register_module()
class SceneEgoMPJPE(BaseMetric):
    ALIGNMENT = {'mpjpe': 'none', 'p-mpjpe': 'procrustes', 'n-mpjpe': 'scale'}

    def __init__(self,
                 mode: str = 'mpjpe',
                 joint_type='mo2cap2',
                 motion_mean_path=None,
                 motion_std_path=None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 save_path: Optional[str] = None,
                 ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        allowed_modes = self.ALIGNMENT.keys()
        if mode not in allowed_modes:
            raise KeyError("`mode` should be 'mpjpe', 'p-mpjpe', or "
                           f"'n-mpjpe', but got '{mode}'.")

        self.mode = mode
        self.joint_type = joint_type
        self.save_path = save_path

        if motion_mean_path is not None and motion_std_path is not None:
            self.motion_mean = torch.load(motion_mean_path)
            self.motion_std = torch.load(motion_std_path)
        else:
            self.motion_mean = None
            self.motion_std = None

        self.smpl_2_mo2cap2 = JointConverter(source_joint_name='smpl', target_joint_name='mo2cap2')
        self.renderpeople_2_mo2cap2 = JointConverter(source_joint_name='renderpeople', target_joint_name='mo2cap2')

    def hardCode_inv_transform(self, data):
        motion_std = torch.Tensor(self.motion_std).to(data.device)
        motion_mean = torch.Tensor(self.motion_mean).to(data.device)
        return data * motion_std + motion_mean

    def process(self, data_batch,
                data_samples) -> None:
        for i, data_sample in enumerate(data_samples):

            # breakpoint()

            # predicted keypoints coordinates, [T, K, D]
            pred_coords = data_sample
            # convert to joint locations
            if self.motion_mean is not None:
                pred_coords = self.hardCode_inv_transform(pred_coords.permute(1, 2, 0)).float()
            pred_coords = recover_from_ric(pred_coords, 22)[0, ...]

            seq_length = data_batch['lengths'][i]
            if pred_coords.ndim == 4:
                pred_coords = np.squeeze(pred_coords, axis=0)

            # ground truth data_info
            gt = data_batch['data_samples']['gt_joints_3d'][i].numpy()
            root_init_xz = data_batch['data_samples']['root_init_xz'][i].numpy()
            root_quat_init = data_batch['data_samples']['root_quat_init'][i].numpy()
            root_quat_init = np.ones(gt.shape[:-1] + (4,)) * root_quat_init
            # breakpoint()

            gt = root_init_xz + gt
            gt = qrot_np(root_quat_init, gt)

            # cut the gt and pred to the same length
            gt = gt[:seq_length]
            pred_coords = pred_coords[:seq_length]

            mask = np.ones((seq_length, 15)).astype(bool)

            # breakpoint()

            # convert smpl and studio to mo2cap2
            pred_coords = self.smpl_2_mo2cap2.convert(pred_coords)
            gt = self.renderpeople_2_mo2cap2.convert(gt)

            # global rotation and translation of gt


            result = {
                'pred_coords': pred_coords,
                'gt_coords': gt,
                'mask': mask,
            }

            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are the corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        if self.save_path is not None:
            logger.info(f'Saving results to {self.save_path}...')
            with open(self.save_path, 'wb') as f:
                pickle.dump(results, f)

        # pred_coords: [N, K, D]
        pred_coords = np.concatenate(
            [result['pred_coords'] for result in results])
        # gt_coords: [N, K, D]
        gt_coords = np.concatenate([result['gt_coords'] for result in results])
        # mask: [N, K]
        mask = np.concatenate([result['mask'] for result in results])

        error_name = self.mode.upper()

        logger.info(f'Evaluating {self.mode.upper()}...')
        metrics = dict()

        metrics[error_name] = keypoint_mpjpe(pred_coords, gt_coords, mask,
                                             self.ALIGNMENT[self.mode])

        return metrics
