import pickle
from collections import defaultdict
from copy import deepcopy
from os import path as osp
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmpose.registry import METRICS
from ..functional import keypoint_mpjpe
from ...datasets.datasets.ego_omni_mocap.humanml_utils.motion_process import recover_from_ric


@METRICS.register_module()
class AgrolMPJPE(BaseMetric):
    ALIGNMENT = {'mpjpe': 'none', 'p-mpjpe': 'procrustes', 'n-mpjpe': 'scale', 'c-mpjpe': 'center'}

    def __init__(self,
                 mode: str = 'mpjpe',
                 motion_mean_path=None,
                 motion_std_path=None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 save_path: Optional[str] = None,
                 save_keys = None,
                 max_save_length=-1,
                 nymeria_mask=False,
                 ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        allowed_modes = self.ALIGNMENT.keys()
        if mode not in allowed_modes:
            raise KeyError("`mode` should be 'mpjpe', 'p-mpjpe', or "
                           f"'n-mpjpe', but got '{mode}'.")

        self.mode = mode
        self.save_path = save_path
        self.max_save_length = max_save_length
        self.nymeria_mask = nymeria_mask
        self.save_keys = save_keys

        if motion_mean_path is not None and motion_std_path is not None:
            self.motion_mean = torch.load(motion_mean_path)
            self.motion_std = torch.load(motion_std_path)
        else:
            self.motion_mean = None
            self.motion_std = None

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
            pred_hml = deepcopy(pred_coords)
            pred_coords = recover_from_ric(pred_coords, 22)[0, ...]


            if pred_coords.ndim == 4:
                pred_coords = np.squeeze(pred_coords, axis=0)

            # ground truth data_info
            gt = data_batch['data_samples']['init_aligned_global_smpl_joints'][i]

            if 'lengths' in data_batch.keys():
                seq_length = data_batch['lengths'][i]
            else:
                seq_length = pred_coords.shape[0]
                assert seq_length == gt.shape[0]

            if 'text' in data_batch.keys():
                text = data_batch['text'][i]
            else:
                text = None

            # cut the gt and pred to the same length
            gt = gt[:seq_length]
            pred_coords = pred_coords[:seq_length]

            if self.nymeria_mask:
                mask = np.ones((seq_length, 22)).astype(bool)
                mask[150:] = False
            else:
                mask = np.ones((seq_length, 22)).astype(bool)

            result = {
                'pred_hml': pred_hml,
                'pred_coords': pred_coords,
                'gt_coords': gt,
                'mask': mask,
                'text': text
            }

            if self.save_keys is not None:
                for key in self.save_keys:
                    result[key] = data_batch['data_samples'][key][i]

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
            if self.max_save_length > 0:
                save_results = results[:self.max_save_length]
            else:
                save_results = results
            with open(self.save_path, 'wb') as f:
                pickle.dump(save_results, f)

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
