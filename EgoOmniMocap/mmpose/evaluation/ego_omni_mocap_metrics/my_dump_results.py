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


@METRICS.register_module()
class MyDumpResults(BaseMetric):

    def __init__(self,
                 save_path,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.save_path = save_path

        assert self.save_path is not None

    def process(self, data_batch,
                data_samples) -> None:
        for i, data_sample in enumerate(data_samples):
            pred_imu_signal = data_sample
            gt = data_batch['data_samples']['init_aligned_global_smpl_joints'][i]
            imu_acc = data_batch['init_aligned_imu_acc'][i]
            imu_ori = data_batch['init_aligned_imu_ori'][i]
            input_motion_hml = data_batch['motion_hml'][i]
            length = data_batch['lengths'][i]
            result = {
                'pred_imu': pred_imu_signal,
                'motion_hml': input_motion_hml,
                'imu_acc': imu_acc,
                'imu_ori': imu_ori,
                'gt_coords': gt,
                'length': length,
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

        logger.info(f'Saving results to {self.save_path}...')
        with open(self.save_path, 'wb') as f:
            pickle.dump(results, f)

        # create dummy metrics
        metrics = dict(
            dummy=0.0,
        )
        return metrics
