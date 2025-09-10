import pickle
from collections import defaultdict
from copy import deepcopy
from functools import partial
from os import path as osp
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmpose.registry import METRICS
from ..functional import keypoint_mpjpe
from ...datasets.datasets.ego_omni_mocap.humanml_utils.motion_process import recover_from_ric


def calculate_mpjpes(result, mode):
    pred_coords = result['pred_coords']
    gt_coords = result['gt_coords']
    mask = result['mask']

    # breakpoint()

    mpjpe_list = keypoint_mpjpe(pred_coords, gt_coords, mask, mode, reduce=False)  # shape: (N_seq, kpts_num)
    return mpjpe_list


def calculate_jitter(result, mode='pred', fps=25):
    if mode == 'pred':
        coords = result['pred_coords']
    elif mode == 'gt':
        coords = result['gt_coords']
    else:
        raise ValueError(f"mode should be 'pred' or 'gt', but got {mode}")

    # calculate the jitter through the sequence
    result = (coords[3:] - 3 * coords[2:-1] + 3 * coords[1:-2] - coords[:-3]) * (
                fps ** 3)  # shape: (N_seq, kpts_num, 3)

    result = np.linalg.norm(result, axis=-1)  # shape: (N_seq, kpts_num, 3) -> (N_seq, kpts_num)

    result = result / 1000  # convert to km
    return result


def calculate_local_rotation_error(result):
    # get the local rotation using ik method
    pass


def calculate_global_rotation_error(result):
    joint_pred, joint_gt = result['pred_coords'], result['gt_coords']
    # t2m_joint_number = 22
    from mmpose.utils.humanml_utils.motion_representation import t2m_kinematic_chain
    from mmpose.utils.geometry_utils.pytorch3d_rotation_convertions import quaternion_to_axis_angle
    from mmpose.datasets.datasets.ego_omni_mocap.humanml_utils.common.quaternion import qbetween
    def get_limb_direction(joint_positions):
        limb_direction = np.zeros_like(joint_positions)
        for chain in t2m_kinematic_chain:
            for i in range(1, len(chain)):
                limb_direction[:, chain[i]] = joint_positions[:, chain[i]] - joint_positions[:, chain[i - 1]]
                # the limb direction at limb_direction[0] is not changed
        # remove the first element
        limb_direction = limb_direction[:, 1:]
        # normalize the limb direction
        limb_direction = limb_direction / np.linalg.norm(limb_direction, axis=-1, keepdims=True)
        return limb_direction

    limb_direction_pred = get_limb_direction(joint_pred)
    limb_direction_gt = get_limb_direction(joint_gt)

    # calculate the relative rotation between each limb and the global direction

    limb_direction_pred = torch.from_numpy(limb_direction_pred).float()
    limb_direction_gt = torch.from_numpy(limb_direction_gt).float()
    limb_quat = qbetween(limb_direction_pred, limb_direction_gt)

    limb_rot_axis_angle = quaternion_to_axis_angle(limb_quat)
    # get the axis angle rotation value
    limb_rot_angle = torch.norm(limb_rot_axis_angle, dim=-1)
    # convert to degree
    limb_rot_angle = limb_rot_angle * 180 / np.pi

    # if torch.isnan(limb_rot_angle).any():
    #     breakpoint()

    return limb_rot_angle.cpu().numpy()


@METRICS.register_module()
class EgoOmniMocapError(BaseMetric):
    ALLOWED_MODES = {'pa-mpjpe': partial(calculate_mpjpes, mode='procrustes'),
                     'c-mpjpe': partial(calculate_mpjpes, mode='center'),
                     'mpjpe': partial(calculate_mpjpes, mode='none'),
                     'n-mpjpe': partial(calculate_mpjpes, mode='scale'),
                     'global-angle': calculate_global_rotation_error,
                     'local-angle': None,
                     'jitter-pred': partial(calculate_jitter, mode='pred', fps=25),
                     'jitter-gt': partial(calculate_jitter, mode='gt', fps=25)
                     }
    ALIGNMENTS = {'mpjpe': 'none', 'p-mpjpe': 'procrustes', 'n-mpjpe': 'scale', 'c-mpjpe': 'center'}

    def __init__(self,
                 mode=('pa-mpjpe', 'c-mpjpe', 'global-angle', 'local-angle', 'jitter-pred', 'jitter-gt'),
                 motion_mean_path=None,
                 motion_std_path=None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 save_path: Optional[str] = None,
                 max_save_length=-1,
                 ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.mode = mode
        for _model in self.mode:
            if _model not in self.ALLOWED_MODES.keys():
                raise KeyError(f"Mode {_model} not in {self.ALLOWED_MODES}")
        self.save_path = save_path
        self.max_save_size = max_save_length

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

    # recover the joints first and then calculate the joint angles with ik method
    def process(self, data_batch,
                data_samples) -> None:
        for i, data_sample in enumerate(data_samples):

            # predicted keypoints coordinates, [T, K, D]
            pred_coords = data_sample
            # convert to joint locations
            if self.motion_mean is not None:
                pred_coords = self.hardCode_inv_transform(pred_coords.permute(1, 2, 0)).float()
            pred_hml = deepcopy(pred_coords)
            pred_coords = recover_from_ric(pred_coords, 22)[0, ...]

            seq_length = data_batch['lengths'][i] if 'lengths' in data_batch.keys() else pred_coords.shape[0]
            if pred_coords.ndim == 4:
                pred_coords = np.squeeze(pred_coords, axis=0)

            # ground truth data_info
            gt = data_batch['data_samples']['init_aligned_global_smpl_joints'][i]
            gt_length = torch.count_nonzero(torch.sum(gt, (-1, -2))).item()
            seq_length = min(seq_length, gt_length)

            if 'text' in data_batch.keys():
                text = data_batch['text'][i]
            else:
                text = None

            # cut the gt and pred to the same length
            gt = gt[:seq_length]

            pred_coords = pred_coords[:seq_length]

            mask = np.ones((seq_length, 22)).astype(bool)

            # if seq_length < 196:
            #     breakpoint()

            # convert the pred coords and gt coords to local and global angles
            # get the local angles with ik method

            result = {
                'pred_hml': pred_hml.cpu().numpy() if torch.is_tensor(pred_hml) else pred_hml,
                'pred_coords': pred_coords.cpu().numpy() if torch.is_tensor(pred_coords) else pred_coords,
                'gt_coords': gt.cpu().numpy() if torch.is_tensor(gt) else gt,
                'mask': mask,
                'text': text,
                # 'smpl_pose_gt': smpl_pose_gt.cpu().numpy() if torch.is_tensor(smpl_pose_gt) else smpl_pose_gt,
            }
            if 'motion_id' in data_batch['data_samples'].keys():
                result['motion_id'] = data_batch['data_samples']['motion_id'][i]
            else:
                result['motion_id'] = None

            if 'combo_name' in data_batch['data_samples'].keys():
                result['combo_name'] = data_batch['data_samples']['combo_name'][i]
            else:
                result['combo_name'] = None

            if 'init_aligned_imu_acc' in data_batch.keys():
                imu_acc = data_batch['init_aligned_imu_acc'][i]
                imu_acc = imu_acc[:seq_length]
                result['imu_acc'] = imu_acc.cpu().numpy() if torch.is_tensor(imu_acc) else imu_acc

            if 'init_aligned_imu_ori' in data_batch.keys():
                imu_ori = data_batch['init_aligned_imu_ori'][i]
                imu_ori = imu_ori[:seq_length]
                result['imu_ori'] = imu_ori.cpu().numpy() if torch.is_tensor(imu_ori) else imu_ori

            if 'smpl_pose' in data_batch['data_samples'].keys():
                smpl_pose_gt = data_batch['data_samples']['smpl_pose'][i]
                smpl_pose_gt = smpl_pose_gt[:seq_length]
                result['smpl_pose_gt'] = smpl_pose_gt.cpu().numpy() if torch.is_tensor(smpl_pose_gt) else smpl_pose_gt

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
            if self.max_save_size > 0:
                # save_results = results[:self.max_save_size]

                # save max save size length, separate uniformly in the sequence
                skip_num = len(results) // self.max_save_size

                save_results = results[::skip_num]
            else:
                save_results = results

            # only save "pred_coords", "gt_coords", "motion_id", "text"
            save_results = [{'pred_coords': res['pred_coords'],
                             'gt_coords': res['gt_coords'],
                             'motion_id': res['motion_id'],
                             'combo_name': res['combo_name'],
                             'text': res['text']} for res in save_results]
            with open(self.save_path, 'wb') as f:
                pickle.dump(save_results, f)

        result_dict = {}

        for eval_mode in self.mode:
            if eval_mode not in result_dict.keys():
                result_dict[eval_mode] = []
            for result in results:
                result_dict[eval_mode].extend(self.ALLOWED_MODES[eval_mode](result))
        # average through the eval modes
        metrics = {}
        for eval_mode in result_dict.keys():
            metrics[eval_mode] = np.mean(result_dict[eval_mode])

        if results[0]['combo_name'] is not None:
            # start evaluating the C-MPJPE and PA_MPJPE for each combo_name
            combo_name_pa_mpjpe_dict = defaultdict(list)
            combo_name_c_mpjpe_dict = defaultdict(list)
            for result in results:
                combo_name_pa_mpjpe_dict[result['combo_name']].extend(self.ALLOWED_MODES['pa-mpjpe'](result))
                combo_name_c_mpjpe_dict[result['combo_name']].extend(self.ALLOWED_MODES['c-mpjpe'](result))
            # calculate the average for each combo_name
            combo_name_pa_mpjpe = {k: np.mean(v) for k, v in combo_name_pa_mpjpe_dict.items()}
            combo_name_c_mpjpe = {k: np.mean(v) for k, v in combo_name_c_mpjpe_dict.items()}

            # add them to metrics, use key like: combo_name_pa-mpjpe
            for k, v in combo_name_pa_mpjpe.items():
                metrics[f"{k}_pa-mpjpe"] = v
            for k, v in combo_name_c_mpjpe.items():
                metrics[f"{k}_c-mpjpe"] = v

        return metrics
