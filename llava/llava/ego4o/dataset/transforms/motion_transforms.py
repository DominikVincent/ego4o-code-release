import pdb
from copy import deepcopy

import numpy as np
import torch

from llava.ego4o.utils.humanml_utils.motion_process import recover_from_ric

from llava.ego4o.dataset.keypoints_mapping.joint_converter import JointConverter
from typing import Dict, Optional, Union, Tuple, List
from llava.ego4o.utils.humanml_utils.motion_representation import process_file

class ZUp2YUp:
    def __init__(self, joint_name='segment_tXYZ'):
        self.joint_name = joint_name

        self.rotation_matrix = np.array([[1, 0, 0],
                                            [0, 0, -1],
                                            [0, 1, 0]])
        self.rotation_matrix_torch = torch.from_numpy(self.rotation_matrix).float()

    def transform(self, results):
        global_joints = results[self.joint_name]
        if torch.is_tensor(global_joints):
            rotation_matrix_torch = self.rotation_matrix_torch.to(global_joints.device)
            global_joints = torch.matmul(global_joints, rotation_matrix_torch)
        else:
            global_joints = np.matmul(global_joints, self.rotation_matrix)
        results[self.joint_name] = global_joints

        return results


class ConvertNymeriaToHML:
    def __init__(self, joint_name='segment_tXYZ', out_name=None):
        self.joint_name = joint_name
        self.out_name = out_name

        self.joint_converter = JointConverter('nymeria', 'hml')


    def transform(self, result:dict):
        joints = result[self.joint_name]
        if joints.shape[-1] != 3:
            joints_shape = joints.shape
            new_joints_shape = joints_shape[:-1] + (joints_shape[-1] // 3, 3)
            joints = joints.reshape(new_joints_shape)
        # convert the joint to the output format
        out_joints = self.joint_converter.convert(joints)
        if self.out_name is not None:
            result[self.out_name] = out_joints
        else:
            result[self.joint_name] = out_joints
        return result


class RotationMatrixTo6D:
    def __init__(self, rotation_name):
        self.rotation_name = rotation_name

    def transform(self, results: dict) -> dict:
        rotation = results[self.rotation_name]
        if torch.is_tensor(rotation):
            batch_dim = rotation.size()[:-2]
            cont_6d = rotation[..., :2, :].clone().reshape(batch_dim + (6,))
        else:
            batch_dim = rotation.shape[:-2]
            cont_6d = deepcopy(rotation[..., :2, :]).reshape(batch_dim + (6,))
        results[self.rotation_name] = cont_6d
        return results


class InitAlignIMUMotion:
    '''
    deal with IMU signal and
    '''

    def __init__(self, imu_acc_name, imu_ori_name, joint_name, imu_loc_name=None,
                 feet_threshold=0.002, use_default_floor_height=False):
        self.imu_acc_name = imu_acc_name
        self.imu_ori_name = imu_ori_name
        self.joint_name = joint_name
        self.imu_loc_name = imu_loc_name

        self.feet_threshold = feet_threshold
        self.default_floor_height = use_default_floor_height
        self.hip_ids = [2, 1]
        self.shoulder_ids = [17, 16]
        self.feet_id_r, self.feet_id_l = [8, 11], [7, 10]

    def cross_product_matrix(self, cross):
        I = np.eye(3)
        return np.cross(I, cross)

    def rotation_matrix_between(self, vec1, vec2):
        vec1 = vec1 / np.sqrt((vec1 ** 2).sum(axis=-1))[..., np.newaxis]
        vec2 = vec2 / np.sqrt((vec2 ** 2).sum(axis=-1))[..., np.newaxis]
        cross = np.cross(vec1, vec2, axis=-1)
        if np.all(cross == 0):
            return np.eye(3)
        s = np.linalg.norm(cross, axis=-1)
        dot = (vec1 * vec2).sum(axis=-1)
        rot_matrix = np.eye(3) + self.cross_product_matrix(cross) + self.cross_product_matrix(
            cross) @ self.cross_product_matrix(cross) * ((1 - dot) / s ** 2)
        return rot_matrix

    def align_joint_sequence_origin(self, global_smplx_joints, imu_loc_input=None):
        smplx_joints = deepcopy(global_smplx_joints)
        imu_loc = deepcopy(imu_loc_input)
        if self.default_floor_height is False:
            floor_height = smplx_joints.min(axis=0).min(axis=0)[1]
            smplx_joints[:, :, 1] -= floor_height
        root_pos_init = smplx_joints[0]
        root_pose_init_xz = -deepcopy(root_pos_init[0] * np.array([1, 0, 1]))
        smplx_joints = smplx_joints + root_pose_init_xz
        if imu_loc is not None:
            imu_loc = imu_loc + root_pose_init_xz
        r_hip, l_hip = self.hip_ids
        sdr_r, sdr_l = self.shoulder_ids
        across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
        across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
        forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
        target = np.array([[0, 0, 1]])
        root_matrix_init = self.rotation_matrix_between(forward_init, target)
        root_matrix_init_ret = deepcopy(root_matrix_init)
        # init_aligned_smplx_joints = root_matrix_init @ smplx_joints.transpose(0, 2, 1)
        # init_aligned_smplx_joints = init_aligned_smplx_joints.transpose(0, 2, 1)
        init_aligned_smplx_joints = np.einsum('ij, kmj -> kmi', root_matrix_init, smplx_joints)
        if imu_loc is not None:
            init_aligned_imu_loc = np.einsum('ij, kmj -> kmi', root_matrix_init, imu_loc)
            return init_aligned_smplx_joints, root_pose_init_xz, root_matrix_init_ret, init_aligned_imu_loc
        else:
            return init_aligned_smplx_joints, root_pose_init_xz, root_matrix_init_ret

    def rotate_imu_signal(self, imu_acc, imu_ori, root_matrix_init):
        # imu acc shape: (N, 5, 3)
        # imu ori shape: (N, 5, 3, 3)
        N, signal_num = imu_acc.shape[0], imu_acc.shape[1]
        # imu_acc = imu_acc.reshape((N * 5, 3))
        imu_ori = imu_ori.reshape((N * signal_num, 3, 3))
        # imu_acc = root_matrix_init @ imu_acc
        imu_acc_aligned = np.einsum('ij, kmj -> kmi', root_matrix_init, imu_acc)

        # imu_acc_aligned = (root_matrix_init @ imu_acc.T).T

        # convert imu_ori to quanternion
        imu_ori_aligned = root_matrix_init @ imu_ori

        # imu_acc_aligned = imu_acc_aligned.reshape((N, 5, 3))
        imu_ori_aligned = imu_ori_aligned.reshape((N, signal_num, 3, 3))
        return imu_acc_aligned, imu_ori_aligned

    def transform(self, results: dict) -> dict:
        smpl_joints = results[self.joint_name]
        imu_acc = results[self.imu_acc_name]
        imu_ori = results[self.imu_ori_name]


        imu_combo = results['imu_combo']
        if torch.is_tensor(smpl_joints):
            smpl_joints = smpl_joints.cpu().numpy()
        if torch.is_tensor(imu_acc):
            imu_acc = imu_acc.cpu().numpy()
        if torch.is_tensor(imu_ori):
            imu_ori = imu_ori.cpu().numpy()

        # align initial joint
        if self.imu_loc_name is not None:
            imu_loc = results[self.imu_loc_name]
            if torch.is_tensor(imu_loc):
                imu_loc = imu_loc.cpu().numpy()
            init_aligned_smpl_joints, root_init_xz, root_matrix_init, init_aligned_imu_loc \
                = self.align_joint_sequence_origin(smpl_joints, imu_loc)
            results['init_aligned_imu_loc'] = torch.from_numpy(init_aligned_imu_loc).float()
        else:
            init_aligned_smpl_joints, root_init_xz, root_matrix_init = self.align_joint_sequence_origin(smpl_joints)
        if init_aligned_smpl_joints.shape[-2] == 24:
            init_aligned_smpl_joints = init_aligned_smpl_joints[:, :-2, :]
        init_aligned_smpl_joints_return = deepcopy(init_aligned_smpl_joints)
        if init_aligned_smpl_joints.shape[0] > 200:
            pdb.set_trace()
        # 24 smpl joints to 22 smplh body joints
        results['init_aligned_global_smpl_joints'] = torch.asarray(init_aligned_smpl_joints_return).float()
        results['root_init_xz'] = torch.from_numpy(root_init_xz).float()
        results['root_matrix_init'] = torch.from_numpy(root_matrix_init).float()

        # # rotate the imu signal
        aligned_imu_acc, aligned_imu_ori = self.rotate_imu_signal(imu_acc, imu_ori, root_matrix_init)
        results['init_aligned_imu_acc'] = torch.from_numpy(aligned_imu_acc).float()
        results['init_aligned_imu_ori'] = torch.from_numpy(aligned_imu_ori).float()

        return results


class HMLMotionRepresentation:
    def __init__(self, joint_name, drop_last_pose_name_list=None):
        self.joint_name = joint_name
        self.drop_last_pose_name_list = drop_last_pose_name_list
        self.process_smpl_motion = process_file

    def transform(self, results: dict) -> dict:
        init_aligned_joints = deepcopy(results[self.joint_name])
        if torch.is_tensor(init_aligned_joints):
            init_aligned_joints = init_aligned_joints.cpu().numpy()
        if init_aligned_joints.shape[-2] == 24:
            init_aligned_joints = init_aligned_joints[:, :-2, :]  # remove the hand joints

        data, global_positions, positions, l_velocity = self.process_smpl_motion(init_aligned_joints, 0.002,
                                                                                 uniform=False)
        results['motion_hml'] = data

        # print((np.abs(global_positions - init_aligned_joints)).sum())
        # check the recovered joint is correct
        # recovered_joints = recover_from_ric(torch.from_numpy(data), 22)
        # print((np.abs(recovered_joints - init_aligned_joints[:-1])).sum())

        results['hml_global_positions'] = global_positions
        results['hml_joint_positions'] = positions
        results['sent_len'] = torch.tensor(len(data))


        if self.drop_last_pose_name_list is not None:
            for name in self.drop_last_pose_name_list:
                results[name] = results[name][:-1]
                # if len(results[name]) != len(data):
                #     pdb.set_trace()
                assert len(results[name]) == len(data), f'{name}: {len(results[name])} != {len(data)}'

        return results


class PadMotion:
    def __init__(self, seq_len=196, pad_value=0.0, pad_name_list=None, resize_input_sequence=False):
        self.seq_len = seq_len
        self.pad_value = pad_value
        self.pad_name_list = pad_name_list
        self.resize_input_sequence = resize_input_sequence

    def transform(self, results: dict) -> dict:
        final_length = 100000000
        for name in self.pad_name_list:
            data = results[name]
            if self.resize_input_sequence:
                new_data_len = (len(data) // 4) * 4
                data = data[:new_data_len]
                results['sent_len'] = new_data_len
            final_length = min(final_length, len(data))
            if torch.is_tensor(data):
                data = data.cpu().numpy()
            if len(data) < self.seq_len:
                pad_data = np.zeros((self.seq_len - len(data), *data.shape[1:]), dtype=data.dtype)
                pad_data.fill(self.pad_value)
                results[name] = np.concatenate([data, pad_data], axis=0)
            else:
                results[name] = data[:self.seq_len]
        # results['lengths'] = torch.tensor(self.seq_len)
        results['lengths'] = torch.tensor(final_length)
        return results

class ChangeHMLShape:
    def __init__(self, hml_motion_name):
        self.hml_motion_name = hml_motion_name
        pass

    def transform(self, results: dict) -> dict:
        results[self.hml_motion_name] = np.expand_dims(results[self.hml_motion_name].T, axis=-2)
        return results

class AddDummyText:
    def __init__(self, text_name='text', dummy_text=None):
        self.text_name = text_name
        self.dummy_text = dummy_text

    def transform(self, results: dict) -> dict:
        if self.text_name not in results.keys():
            results[self.text_name] = self.dummy_text
        return results

class NormalizeHMLMotion:
    def __init__(self, hml_motion_name, hml_mean_path, hml_std_path):
        self.hml_motion_name = hml_motion_name
        self.hml_mean_path = hml_mean_path
        self.hml_mean = torch.load(self.hml_mean_path)
        self.hml_std_path = hml_std_path
        self.hml_std = torch.load(self.hml_std_path)

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        hml_motion = results[self.hml_motion_name]
        # do the normalize
        hml_motion = (hml_motion - self.hml_mean) / self.hml_std
        results[self.hml_motion_name] = hml_motion
        return results

    def inv_normalize(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        hml_motion = results[self.hml_motion_name]
        # do the unnormalize
        hml_motion = hml_motion * self.hml_std + self.hml_mean
        results[self.hml_motion_name] = hml_motion
        return results


class Collect:
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
