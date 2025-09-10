import json
import os.path
import pdb
import pickle
import random
from copy import deepcopy
from pathlib import Path
import codecs as cs

import numpy as np
import torch
from mmengine.dataset.base_dataset import Compose
from natsort import natsorted
from torch.utils.data import Dataset
from tqdm import tqdm

import mmpose.datasets.datasets.imuposer.math as math
from mmpose.datasets.datasets.imuposer.imuposer_config import Config, amass_combos, tlcontrol_combos
from mmpose.registry import DATASETS
from scipy.spatial.transform import Rotation as R

from mmpose.utils.keypoints_mapping.joint_converter import JointConverter


@DATASETS.register_module()
class NymeriaDataset(Dataset):
    def __init__(self,
                 dataset_dir='/scratch/inf0/user/jianwang/nymeria',
                 seq_len=196,
                 min_seq_len=10,
                 signal_num=5,
                 tlcontrol_joint_sequence=True,
                 random_mask=False,
                 combo_name='global',
                 split="train",
                 pipeline=None,
                 acc_scale=30,
                 with_text=True,
                 test_data_len=None,
                 motion_id_list=None,
                 ):
        super().__init__()

        # record directory names
        self.dataset_dir = dataset_dir

        # load the data
        self.signal_num = signal_num
        self.combo_name = combo_name
        self.tlcontrol_joint_sequence = tlcontrol_joint_sequence
        if self.tlcontrol_joint_sequence is True:
            self.combo_list = tlcontrol_combos
        else:
            self.combo_list = amass_combos
        self.train = split
        self.random_mask = random_mask
        self.seq_len = seq_len
        self.min_seq_len = min_seq_len
        self.joint_converter = JointConverter('nymeria', 'hml')
        self.acc_scale = acc_scale
        self.with_text = with_text
        self.test_data_len = test_data_len
        self.motion_id_list = motion_id_list



        # store the data temporarily
        data_temp_storage_path = '/scratch/inf0/user/jianwang/nymeria/data_temp_storage'
        data_temp_storage_path = os.path.join(data_temp_storage_path, f'{split}_data.pt')

        # if os.path.exists(data_temp_storage_path):
        #     self.data = torch.load(data_temp_storage_path)
        # else:
        #     self.data = self.load_data()
        #     torch.save(self.data, data_temp_storage_path)

        self.data = self.load_data()

        self.pipeline = Compose(pipeline)
        self.metainfo = {'split': split, 'seq_len': seq_len}



    def full_init(self):
        pass

    # convert imu ori and acc to 6 location format
    def convert_imu(self, imu_ori, imu_acc):
        # convert imu from 17 imus to head, lhand, rhand, lhip, rhip
        imu_ori = imu_ori[:, [2, 9, 5, 14, 11]]
        imu_acc = imu_acc[:, [2, 9, 5, 14, 11]]
        return imu_ori, imu_acc

    def convert_nymeria(self, imu_ori_qwxyz, imu_acc, joint_loc):
        # convert from nymeria to hml joint format
        # pdb.set_trace()
        joint_loc = self.joint_converter.convert(joint_loc)
        imu_ori_qwxyz, imu_acc = self.convert_imu(imu_ori_qwxyz, imu_acc)

        def z_up_to_y_up(rotation_matrix):
            # Define the transformation matrix to convert from Z-up to Y-up
            # This is a 90-degree rotation around the X-axis
            transformation_matrix = np.array([
                [1, 0, 0],
                [0, 0, 1],
                [0, -1, 0]
            ])

            # Convert the rotation matrix
            # converted_matrix = transformation_matrix @ rotation_matrix
            converted_matrix = np.einsum('ij, BNjk->BNik', transformation_matrix, rotation_matrix)

            return converted_matrix

        # convert from quanternion to rotation matrix
        len_seq = len(imu_ori_qwxyz)
        # convert from WXYZ to XYZW
        qXYZW_seq = np.concatenate([imu_ori_qwxyz[:, :, 1:], imu_ori_qwxyz[:, :, :1]], axis=-1)

        # pdb.set_trace()
        if np.sum(np.linalg.norm(qXYZW_seq, axis=-1) < 0.05) > 0:
            # print("Warning: some quaternion is not valid")
            # remove the invalid quaternion sequence
            raise ValueError("Invalid quaternion")

        r = R.from_quat(qXYZW_seq.reshape([-1, 4]))
        limb_ori_matrix_seq = r.as_matrix().reshape([len_seq, -1, 3, 3])

        # convert rotation matrix from zup to y up
        limb_ori_matrix_seq = z_up_to_y_up(limb_ori_matrix_seq)
        # convert from zup to y up
        tXYZ_seq = joint_loc[:, :, [0, 2, 1]]
        tXYZ_seq[:, :, 2] *= -1

        # convert aXYZ from zup to yup
        aXYZ_seq = imu_acc[:, :, [0, 2, 1]]
        aXYZ_seq[:, :, 2] *= -1

        return limb_ori_matrix_seq, aXYZ_seq, tXYZ_seq

    def load_data(self):
        if self.train == "train":
            data_file = os.path.join(self.dataset_dir, 'ego4o_image_motion_train.jsonl')
        elif self.train == 'test':
            data_file = os.path.join(self.dataset_dir, 'ego4o_image_motion_test.jsonl')
        else:
            raise ValueError("Invalid split")

        with open(data_file, "r") as f:
            lines = f.readlines()

        if self.train == "test":
            if self.test_data_len is not None:
                skip_num = len(lines) // self.test_data_len
                lines = lines[::skip_num]
                print(f"Warning: only using {self.test_data_len} samples for testing")

        list_data_dict = [json.loads(line) for line in lines]
        motion_dir = os.path.join(self.dataset_dir, 'ego4o_input_motion')
        motion_data_list = {}
        result_data = []

        if self.motion_id_list is not None:
            list_data_dict = [ego_data_item for ego_data_item in list_data_dict if ego_data_item['id'] in self.motion_id_list]

        for ego_data_item in tqdm(list_data_dict):
            image_file = ego_data_item['image']
            if not os.path.exists(image_file):
                image_file = image_file.replace('/scratch/inf0/user/jianwang', '/HPS/EgoSyn/static00')
            motion_file = os.path.join(motion_dir, ego_data_item['motion_file'])
            if motion_file not in motion_data_list:
                with open(motion_file, 'rb') as f:
                    motion_data_seq = pickle.load(f)
                motion_data_list[motion_file] = motion_data_seq
            motion_id = ego_data_item['id']
            motion_data = motion_data_list[motion_file][motion_id]

            imu_ori_qWXYZ = motion_data['sensor_qWXYZ']
            if len(imu_ori_qWXYZ) < self.min_seq_len:
                # print('invalid imu')
                # print(ego_data_item)
                # pdb.set_trace()
                continue
            imu_ori_qWXYZ = np.reshape(imu_ori_qWXYZ, (len(imu_ori_qWXYZ), -1, 4))
            imu_acc_xyz = motion_data['sensor_freeAcceleration']
            imu_acc_xyz = np.reshape(imu_acc_xyz, (len(imu_acc_xyz), -1, 3))
            joint_xyz = motion_data['segment_tXYZ']
            joint_xyz = np.reshape(joint_xyz, (len(joint_xyz), -1, 3))

            if len(joint_xyz) != len(imu_ori_qWXYZ):
                pdb.set_trace()

            motion_description = ego_data_item['conversations'][1]['value']
            assert len(ego_data_item['conversations']) == 2  # make sure only have one conversation, one from human, one from gpt

            # convert nymeria
            try:
                imu_ori, imu_acc, joint_xyz = self.convert_nymeria(imu_ori_qWXYZ, imu_acc_xyz, joint_xyz)
            except ValueError as e:
                # print('invalid quaternion')
                # print(ego_data_item)
                continue

            # add zero root imu
            imu_ori = np.concatenate([np.zeros((len(imu_ori), 1, 3, 3)), imu_ori], axis=1)
            imu_acc = np.concatenate([np.zeros((len(imu_acc), 1, 3)), imu_acc], axis=1)
            imu_ori = torch.as_tensor(imu_ori).float()
            imu_acc = torch.as_tensor(imu_acc).float()

            joints_gt = joint_xyz

            assert self.tlcontrol_joint_sequence is True

            # if self.random_mask is False:
            #     _combo = self.combo_name
            # else:
            #     _combo = random.choice(list(self.combo_list.keys()))
            #
            # for _combo in list(self.combo_list):
            #     # acc N, 6, 3
            #     # ori N, 6, 3, 3
            #     if self.random_mask is False:
            #         if _combo != self.combo_name:
            #             continue
            #
            #     _combo_acc = torch.zeros_like(acc)
            #     _combo_ori = torch.zeros((3, 3)).repeat(ori.shape[0], self.signal_num, 1, 1)
            #
            #     _combo_acc[:, self.combo_list[_combo]] = acc[:, self.combo_list[_combo]]
            #     _combo_ori[:, self.combo_list[_combo]] = ori[:, self.combo_list[_combo]]
            #
            #     imu_acc = _combo_acc[:self.seq_len]
            #     imu_ori = _combo_ori[:self.seq_len]
            #     joints = joints_gt[:self.seq_len]

            assert len(imu_acc) == len(imu_ori) == len(joints_gt)

            data_item = {
                'img_path': image_file,
                'imu_acc_full': imu_acc[:self.seq_len],
                'imu_ori_full': imu_ori[:self.seq_len],
                'joints': joints_gt[:self.seq_len],
                'text_dict':[{'caption': motion_description, 'tokens': ['']}],
                'motion_file': motion_file,
                'motion_id': motion_id,
            }
            result_data.append(data_item)

        return result_data



    def __getitem__(self, idx):

        data_item = deepcopy(self.data[idx])
        text_data = random.choice(data_item['text_dict'])
        if self.with_text:
            caption, tokens = text_data['caption'], text_data['tokens']
            data_item['text'] = caption
            data_item['tokens'] = tokens

        # select the combo name
        if self.random_mask is False:
            _combo = self.combo_name
        else:
            if self.combo_name == 'wo_global':
                choices = deepcopy(list(self.combo_list.keys()))
                choices.remove('global')
                _combo = random.choice(choices)
            else:
                _combo = random.choice(list(self.combo_list.keys()))

        acc = data_item['imu_acc_full']
        ori = data_item['imu_ori_full']
        _combo_acc = torch.zeros_like(acc)
        _combo_ori = torch.zeros((3, 3)).repeat(ori.shape[0], self.signal_num, 1, 1)

        _combo_acc[:, self.combo_list[_combo]] = acc[:, self.combo_list[_combo]]
        _combo_ori[:, self.combo_list[_combo]] = ori[:, self.combo_list[_combo]]

        data_item['imu_acc'] = _combo_acc
        data_item['imu_ori'] = _combo_ori
        data_item['combo_name'] = _combo
        data_item['imu_combo'] = deepcopy(self.combo_list[_combo])

        if len(data_item['imu_acc']) != len(data_item['imu_ori']) or len(data_item['imu_acc']) != data_item['joints'].shape[0]:
            pdb.set_trace()

        data_item = self.pipeline(data_item)

        return data_item

    def __len__(self):
        return len(self.data)
