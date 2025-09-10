import json
import os.path
import pdb
import pickle
import random
from copy import deepcopy, copy

import numpy as np
import torch
import transformers
from fontTools.misc.plistlib import end_data
from mmengine.dataset.base_dataset import Compose
from torch.utils.data import Dataset
from tqdm import tqdm

from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.ego4o.constants import DEFAULT_MOTION_TOKEN
from llava.ego4o.dataset.imuposer.imuposer_config import Config, amass_combos, tlcontrol_combos
from scipy.spatial.transform import Rotation as R

from llava.ego4o.dataset.keypoints_mapping.joint_converter import JointConverter
from PIL import Image

from llava.ego4o.train.train_ego4o_preprocess import preprocess_multimodal, preprocess


class NymeriaDataset(Dataset):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args,
                 dataset_json_path=None,
                 dataset_dir='/scratch/inf0/user/jianwang/nymeria',
                 seq_len=150,
                 min_seq_len=150,
                 signal_num=5,
                 tlcontrol_joint_sequence=True,
                 random_mask=False,
                 combo_name='global',
                 split="train",
                 pipeline=None,
                 with_text=True,
                 data_range=None,
                 always_with_image=False,
                 always_with_motion=False,
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.data_range = data_range
        self.always_with_image = always_with_image
        self.always_with_motion = always_with_motion

        # record directory names
        self.dataset_dir = dataset_dir
        self.dataset_json_path = dataset_json_path

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
        self.with_text = with_text

        # store the data temporarily
        data_temp_storage_path = '/scratch/inf0/user/jianwang/nymeria/data_temp_storage'
        data_temp_storage_path = os.path.join(data_temp_storage_path, f'{split}_data.pt')

        # if os.path.exists(data_temp_storage_path):
        #     self.data = torch.load(data_temp_storage_path)
        # else:
        #     self.data = self.load_data()
        #     torch.save(self.data, data_temp_storage_path)

        self.data = self.load_data()

        if pipeline is None:
            pipeline = [
                dict(type='InitAlignIMUMotion',
                     imu_acc_name='imu_acc',
                     imu_ori_name='imu_ori',
                     joint_name='joints',
                     ),
                dict(type='RotationMatrixTo6D',
                     rotation_name='init_aligned_imu_ori'),
                dict(type='RotationMatrixTo6D',
                     rotation_name='imu_ori'),
                dict(type='HMLMotionRepresentation',
                     joint_name='init_aligned_global_smpl_joints',
                     drop_last_pose_name_list=('init_aligned_imu_acc',
                                               'init_aligned_imu_ori',
                                               'init_aligned_global_smpl_joints')),
                dict(type='NormalizeHMLMotion', hml_motion_name='motion_hml',
                     hml_mean_path='/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_mean.pt',
                     hml_std_path='/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_std.pt'),
                dict(type='PadMotion', seq_len=148,
                     pad_name_list=(
                         'init_aligned_imu_acc', 'init_aligned_imu_ori', 'motion_hml',
                         'init_aligned_global_smpl_joints'),
                     resize_input_sequence=True),
                dict(type='ChangeHMLShape', hml_motion_name='motion_hml'),
                dict(type='ToTensor',
                     keys=['init_aligned_imu_acc', 'init_aligned_imu_ori', 'imu_acc', 'imu_ori',
                           'init_aligned_global_smpl_joints', 'motion_hml',
                           'joints']),
                dict(type='LoadImageFromFile'),
                dict(type='Resize', scale=(224, 224)),
                dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                # dict(
                #     type='Collect',
                #     keys=['motion_hml', 'lengths', 'init_aligned_imu_acc', 'init_aligned_imu_ori', 'text', 'img_path'],
                #     meta_keys=['sent_len', 'init_aligned_global_smpl_joints', 'motion_file', 'motion_id', ],
                #     meta_name='data_samples'
                # )
            ]

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
        if self.dataset_json_path is None:
            if self.train == "train":
                data_file = os.path.join(self.dataset_dir, 'ego4o_image_motion_train.jsonl')
            elif self.train == 'test':
                data_file = os.path.join(self.dataset_dir, 'ego4o_image_motion_test.jsonl')
            else:
                raise ValueError("Invalid split")
        else:
            data_file = self.dataset_json_path

        with open(data_file, "r") as f:
            lines = f.readlines()

        # if self.train == "test":
        if self.data_range is not None:
            if type(self.data_range) is not tuple:
                # sample the data
                sample_margin = len(lines) // self.data_range
                lines = lines[::sample_margin]
            else:
                start_data_idx = self.data_range[0]
                end_data_idx = self.data_range[1]
                if end_data_idx > len(lines):
                    end_data_idx = len(lines)
                lines = lines[start_data_idx: end_data_idx]
                print(f"Warning: only using samples from {start_data_idx} to {end_data_idx} for testing")

        list_data_dict = [json.loads(line) for line in lines]
        motion_dir = os.path.join(self.dataset_dir, 'ego4o_input_motion')
        motion_data_list = {}
        result_data = []

        for ego_data_item in tqdm(list_data_dict):
            image_file = ego_data_item['image']
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

            conversations = ego_data_item['conversations']
            assert len(ego_data_item[
                           'conversations']) == 2  # make sure only have one conversation, one from human, one from gpt

            if self.always_with_image:
                if DEFAULT_IMAGE_TOKEN not in conversations[0]['value']:
                    conversations[0]['value'] = conversations[0]['value'].replace(f"{DEFAULT_MOTION_TOKEN}\n",
                                                                                  f"{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_MOTION_TOKEN}\n")
            if self.always_with_motion:
                if DEFAULT_MOTION_TOKEN not in conversations[0]['value']:
                    conversations[0]['value'] = conversations[0]['value'].replace(f"{DEFAULT_IMAGE_TOKEN}\n",
                                                                                  f"{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_MOTION_TOKEN}\n")
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

            assert len(imu_acc) == len(imu_ori) == len(joints_gt)

            data_item = {
                'img_path': image_file,
                'imu_acc_full': imu_acc[:self.seq_len],
                'imu_ori_full': imu_ori[:self.seq_len],
                'joints': joints_gt[:self.seq_len],
                'conversations': conversations,
                'motion_file': motion_file,
                'motion_id': motion_id,
            }
            result_data.append(data_item)

        return result_data

    @property
    def lengths(self):
        length_list = []
        for sample in self.data:
            img_tokens = 128 if 'image' in sample else 0
            motion_tokens = 37 if 'motion_file' in sample else 0
            length_list.append(
                sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens + motion_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.data:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])

            # only consider if we have motion here
            cur_len = cur_len if 'motion_file' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, idx):

        data_item = deepcopy(self.data[idx])
        assert self.with_text is True

        # select the combo name
        if self.random_mask is False:
            _combo = self.combo_name
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

        if len(data_item['imu_acc']) != len(data_item['imu_ori']) or len(data_item['imu_acc']) != \
            data_item['joints'].shape[0]:
            pdb.set_trace()

        # ----------------------- process the text -----------------------
        if isinstance(idx, int):
            sources = [data_item]
        else:
            sources = data_item
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        if 'img_path' in sources[0] and 'motion_file' in sources[0]:
            image_path = sources[0]['img_path']
            # image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor

            # image_path
            # image_path = os.path.join(image_folder, image_file)
            if not os.path.exists(image_path):
                image_path = image_path.replace('/scratch/inf0/user/jianwang', '/HPS/EgoSyn/static00')
            image = Image.open(image_path).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = deepcopy([e["conversations"] for e in sources])

        # check if require motion output

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('img_path' in data_item or 'motion_file' in data_item)
        )

        # print(data_dict.keys())
        # for key in data_dict:
        #     print(key, data_dict[key])

        if isinstance(idx, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'img_path' in data_item:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])

        data_item.update(data_dict)

        # breakpoint()
        if not os.path.exists(data_item['img_path']):
            data_item['img_path'] = data_item['img_path'].replace('/scratch/inf0/user/jianwang', '/HPS/EgoSyn/static00')

        data_item = self.pipeline(data_item)

        # note: dict: img: the image for the input of imu encoder, image: the image for the LLM encoder
        data_item['img_for_imu'] = data_item['img']

        return data_item

    def __len__(self):
        return len(self.data)
