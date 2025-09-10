from copy import deepcopy

import numpy as np
import torch

from typing import List, Union

from mmpose.codecs import *  # noqa: F401, F403
# from mmpose.registry import TRANSFORMS
from mmengine.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
from typing import Dict, Optional, Union, Tuple, List
# from data_loaders.humanml.common.quaternion import qrot_np, qbetween_np, qmul_np, qinv_np, qeuler
from mmpose.utils.humanml_utils.quaternion import qrot_np, qbetween_np, qmul_np, qinv_np, qeuler
from mmpose.datasets.datasets.ego_omni_mocap.humanml_utils.motion_process import recover_from_ric, recover_global_limb_rot

@TRANSFORMS.register_module()
class Rename(BaseTransform):

    def __init__(self, source_name, target_name, copy=False):
        super().__init__()
        self.source_name = source_name
        self.target_name = target_name
        self.copy = copy

    def transform(self, results: dict) -> dict:
        # rename
        if self.copy:
            results[self.target_name] = deepcopy(results[self.source_name])
        else:
            results[self.target_name] = results[self.source_name]
        return results


@TRANSFORMS.register_module()
class Collect(BaseTransform):
    """Collect data from the loader relevant to the specific task.

    This keeps the items in `keys` as it is, and collect items in `meta_keys`
    into a meta item called `meta_name`.This is usually the last stage of the
    data loader pipeline.
    For example, when keys='imgs', meta_keys=('filename', 'label',
    'original_shape'), meta_name='img_metas', the results will be a dict with
    keys 'imgs' and 'img_metas', where 'img_metas' is a DataContainer of
    another dict with keys 'filename', 'label', 'original_shape'.

    Args:
        keys (Sequence[str|tuple]): Required keys to be collected. If a tuple
          (key, key_new) is given as an element, the item retrieved by key will
          be renamed as key_new in collected data.
        meta_name (str): The name of the key that contains meta information.
          This key is always populated. Default: "img_metas".
        meta_keys (Sequence[str|tuple]): Keys that are collected under
          meta_name. The contents of the `meta_name` dictionary depends
          on `meta_keys`.
    """

    def __init__(self, keys, meta_keys, meta_name='data_samples'):
        self.keys = keys
        self.meta_keys = meta_keys
        self.meta_name = meta_name

    def transform(self, results):
        """Performs the Collect formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
              to the next transform in pipeline.
        """
        if 'ann_info' in results:
            results.update(results['ann_info'])

        data = {}
        for key in self.keys:
            if isinstance(key, tuple):
                assert len(key) == 2
                key_src, key_tgt = key[:2]
            else:
                key_src = key_tgt = key
            data[key_tgt] = results[key_src]

        meta = {}
        if len(self.meta_keys) != 0:
            for key in self.meta_keys:
                if isinstance(key, tuple):
                    assert len(key) == 2
                    key_src, key_tgt = key[:2]
                else:
                    key_src = key_tgt = key
                meta[key_tgt] = results[key_src]
        data[self.meta_name] = meta

        return data

    def __repr__(self):
        """Compute the string representation."""
        return (f'{self.__class__.__name__}('
                f'keys={self.keys}, meta_keys={self.meta_keys})')


@TRANSFORMS.register_module()
class SplitMotionSequence(BaseTransform):
    def __init__(self, motion_name, split_length=196, overlap=0.5):
        self.motion_name = motion_name
        self.split_length = split_length
        self.overlap = overlap

    def transform(self, results: dict) -> dict:
        motion = results[self.motion_name]
        if torch.is_tensor(motion):
            motion = motion.cpu().numpy()
        motion_length = motion.shape[0]
        split_length = self.split_length
        overlap = self.overlap
        split_motion_list = []
        for i in range(0, motion_length, int(split_length * (1 - overlap))):
            split_motion_list.append(motion[i:i + split_length])
        results[self.motion_name] = split_motion_list
        return results

@TRANSFORMS.register_module()
class PadMotionSequence(BaseTransform):
    def __init__(self, motion_name, pad_value=0.0, pad_type='constant', pad_length=196):
        self.motion_name = motion_name
        self.pad_value = pad_value
        self.pad_type = pad_type
        self.pad_length = pad_length

    def transform(self, results: dict) -> dict:
        motion = results[self.motion_name]
        if torch.is_tensor(motion):
            motion = motion.cpu().numpy()
        if self.pad_length is None:
            self.pad_length = motion.shape[0]
        if self.pad_type == 'constant':
            pad_value = np.ones_like(motion[0]) * self.pad_value
        else:
            raise NotImplementedError
        padded_motion = np.ones((self.pad_length, *motion.shape[1:])) * pad_value
        padded_motion[:motion.shape[0]] = motion
        results[self.motion_name] = padded_motion
        return results

@TRANSFORMS.register_module()
class ZUp2YUp(BaseTransform):
    def __init__(self, joint_name='global_smpl_motion'):
        self.joint_name = joint_name

        self.rotation_matrix = np.array([[1, 0, 0],
                                            [0, 0, -1],
                                            [0, 1, 0]])
        self.rotation_matrix_torch = torch.from_numpy(self.rotation_matrix).float()

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        global_joints = results[self.joint_name]
        if torch.is_tensor(global_joints):
            rotation_matrix_torch = self.rotation_matrix_torch.to(global_joints.device)
            global_joints = torch.matmul(global_joints, rotation_matrix_torch)
        else:
            global_joints = np.matmul(global_joints, self.rotation_matrix)
        results[self.joint_name] = global_joints

        # from mmpose.utils.visualization.draw import draw_keypoints_3d
        #
        # # visualize aligned pose
        # aligned_pose = global_joints[::20].reshape((-1, 3))
        # mesh = draw_keypoints_3d(aligned_pose)
        # import open3d
        # coor = open3d.geometry.TriangleMesh.create_coordinate_frame()
        # open3d.visualization.draw_geometries([mesh, coor])

        return results

@TRANSFORMS.register_module()
class InitAlignGlobalSMPLJoints(BaseTransform):
    '''
    note: here we use smpl joints rather than smplx joints
    return all information from the alignment process, useful for recovering back the global smplx joints
    '''
    def __init__(self, feet_threshold=0.002, use_default_floor_height=False):
        self.feet_threshold = feet_threshold
        self.default_floor_height = use_default_floor_height
        self.hip_ids = [2, 1]
        self.shoulder_ids = [17, 16]
        self.feet_id_r, self.feet_id_l = [8, 11], [7, 10]

    def align_joint_sequence_origin(self, global_smplx_joints):
        smplx_joints = deepcopy(global_smplx_joints)
        if self.default_floor_height is False:
            floor_height = smplx_joints.min(axis=0).min(axis=0)[1]
            smplx_joints[:, :, 1] -= floor_height
        root_pos_init = smplx_joints[0]
        root_pose_init_xz = -deepcopy(root_pos_init[0] * np.array([1, 0, 1]))
        smplx_joints = smplx_joints + root_pose_init_xz
        r_hip, l_hip = self.hip_ids
        sdr_r, sdr_l = self.shoulder_ids
        across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
        across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
        forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
        target = np.array([[0, 0, 1]])
        root_quat_init = qbetween_np(forward_init, target)
        root_quat_init_ret = deepcopy(root_quat_init)
        root_quat_init = np.ones(smplx_joints.shape[:-1] + (4,)) * root_quat_init
        init_aligned_smplx_joints = qrot_np(root_quat_init, smplx_joints)
        return init_aligned_smplx_joints, root_pose_init_xz, root_quat_init_ret

    def transform(self, results: dict) -> dict:
        smpl_joints = results['global_smpl_motion']
        if torch.is_tensor(smpl_joints):
            smpl_joints = smpl_joints.cpu().numpy()
        # align initial joint
        init_aligned_smplx_joints, root_init_xz, root_quat_init = self.align_joint_sequence_origin(smpl_joints)
        init_aligned_smpl_joints_return = deepcopy(init_aligned_smplx_joints)
        results['init_aligned_global_smpl_joints'] = torch.asarray(init_aligned_smpl_joints_return).float()
        results['root_init_xz'] = torch.from_numpy(root_init_xz).float()
        results['root_quat_init'] = torch.from_numpy(root_quat_init).float()
        return results

@TRANSFORMS.register_module()
class SMPLJoint2Trajectory(BaseTransform):
    def __init__(self, joint_ids=(15, 20, 21),
                 output_joint_ids=(0, 15, 20, 21, 10, 11),
                 joint_name='init_aligned_global_smpl_joints',
                 with_joint_orientation=False):
        self.joint_ids = joint_ids
        self.output_joint_ids = output_joint_ids
        self.with_joint_orientation = with_joint_orientation

        # get the index of joint ids in output joint ids
        self.joint_ids_in_output_joint_ids = []
        for joint_id in self.joint_ids:
            self.joint_ids_in_output_joint_ids.append(self.output_joint_ids.index(joint_id))
        self.joint_name = joint_name

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        init_aligned_joints = results[self.joint_name]
        # get the joint trajectory of specified joint ids
        needed_joint_trajectory = init_aligned_joints[..., self.joint_ids, :]
        joint_trajectory = np.zeros((needed_joint_trajectory.shape[0], len(self.output_joint_ids), 3))
        joint_trajectory[:, self.joint_ids_in_output_joint_ids, :] = needed_joint_trajectory
        if self.with_joint_orientation:
            joint_orient = recover_global_limb_rot(init_aligned_joints)
            needed_joint_orient = joint_orient[..., self.joint_ids, :]
            joint_orient_trajectory = np.zeros((needed_joint_orient.shape[0], len(self.output_joint_ids), 6))
            joint_orient_trajectory[:, self.joint_ids_in_output_joint_ids, :] = needed_joint_orient
            joint_trajectory = np.concatenate([joint_trajectory, joint_orient_trajectory], axis=-1)

        results['joint_trajectory'] = joint_trajectory
        results['trajectory_joint_ids'] = self.joint_ids

        return results
