import os.path
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


@DATASETS.register_module()
class IMUPoserTextDataset(Dataset):
    def __init__(self,
                 dataset_dir='/CT/EgoMocap/work/EgoOmniMocap/scripts/amass_data_dict_25fps',
                 humanml3d_text_dir='/CT/EgoMocap/work/HumanML3D/HumanML3D/texts',
                 seq_len=196,
                 min_seq_len=10,
                 signal_num=5,
                 tlcontrol_joint_sequence=False,
                 random_mask=False,
                 combo_name='global',
                 split="train",
                 pipeline=None,
                 imuposer_root_dir='/CT/EgoMocap/work/IMUPoser',
                 add_wo_text=False,
                 add_w_text=True,
                 mask_imu_prob=0,
                 ):
        super().__init__()

        # record directory names
        self.dataset_dir = dataset_dir
        self.humanml3d_text_dir = humanml3d_text_dir
        self.add_wo_text = add_wo_text
        self.add_w_text = add_w_text
        self.mask_imu_prob = mask_imu_prob

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
        self.config = Config(experiment=f"imuposer_global", model="GlobalModelIMUPoser",
                             project_root_dir=imuposer_root_dir, joints_set=self.combo_list['global'],
                             normalize="no_translation",
                             r6d=True, loss_type="mse", use_joint_loss=True, device="0")

        # change the sequence directory here
        self.config.processed_imu_poser_25fps = Path(dataset_dir)
        self.seq_len = seq_len
        self.min_seq_len = min_seq_len
        self.data = self.load_data()
        self.pipeline = Compose(pipeline)
        self.metainfo = {'split': split, 'seq_len': seq_len}

    def full_init(self):
        pass

    def load_data(self):
        if self.train == "train":
            data_files = [x.name for x in self.config.processed_imu_poser_25fps.iterdir() if "TotalCapture" not in x.name]
        elif self.train == 'val':
            data_files = [x.name for x in self.config.processed_imu_poser_25fps.iterdir() if "TotalCapture" in x.name]
        elif self.train == 'finetune':
            data_files = ["dip_train.pt"]
        elif self.train == 'test':
            data_files = ["dip_test.pt"]
        else:
            raise ValueError("Invalid split")

        result_data = []

        for fname in tqdm(data_files):
            fdata = torch.load(self.config.processed_imu_poser_25fps / fname)

            for i in range(len(fdata)):

                # inputs
                facc = fdata[i]["acc"]
                fori = fdata[i]["ori"]

                # load all the data
                if self.tlcontrol_joint_sequence is False:
                    # use original imuposer joint sequence
                    glb_acc = facc.view(-1, 6, 3)[:, [0, 1, 2, 3, 4]] / self.config.acc_scale
                    glb_ori = fori.view(-1, 6, 3, 3)[:, [0, 1, 2, 3, 4]]
                else:
                    # use tlcontrol joint sequence
                    glb_acc = facc.view(-1, 6, 3)[:, [5, 4, 0, 1, 2, 3]] / self.config.acc_scale
                    glb_ori = fori.view(-1, 6, 3, 3)[:, [5, 4, 0, 1, 2, 3]]

                acc = glb_acc
                ori = glb_ori

                # outputs
                fpose = fdata[i]["pose"]
                fpose = fpose.reshape(fpose.shape[0], -1)

                joints_gt = fdata[i]['joint']
                shape_gt = fdata[i]['shape']
                transl_gt = fdata[i]['tran']
                humanml3d_text_file_name = fdata[i]['humanml3d']['humanml3d_name']
                if humanml3d_text_file_name.endswith('.npy'):
                    humanml3d_text_file_name = humanml3d_text_file_name.replace('.npy', '.txt')
                humanml3d_text_path = os.path.join(self.humanml3d_text_dir, humanml3d_text_file_name)

                for _combo in list(self.combo_list):
                    # acc N, 5, 3
                    # ori N, 5, 3, 3
                    if self.random_mask is False:
                        if _combo != self.combo_name:
                            continue

                    _combo_acc = torch.zeros_like(acc)
                    _combo_ori = torch.zeros((3, 3)).repeat(ori.shape[0], self.signal_num, 1, 1)

                    if random.random() >= self.mask_imu_prob:
                        _combo_acc[:, self.combo_list[_combo]] = acc[:, self.combo_list[_combo]]
                        _combo_ori[:, self.combo_list[_combo]] = ori[:, self.combo_list[_combo]]

                    # maybe we do not need the split here because the maximal frame is 200 frames

                    imu_acc = _combo_acc[:self.seq_len]
                    imu_ori = _combo_ori[:self.seq_len]
                    smpl_pose = fpose[:self.seq_len]
                    joints = joints_gt[:self.seq_len]
                    transl = transl_gt[:self.seq_len]

                    assert len(imu_acc) == len(imu_ori) == len(smpl_pose) == len(joints) == len(transl)

                    data_item = {
                        'imu_acc': imu_acc,
                        'imu_ori': imu_ori,
                        'smpl_pose': smpl_pose,
                        'joints': joints,
                        'shape': deepcopy(shape_gt),
                        'transl': transl,
                        'combo_name': _combo,
                        'imu_combo': deepcopy(self.combo_list[_combo]),
                    }
                    # throw away if the sequence is too short:
                    if data_item['imu_acc'].shape[0] < self.min_seq_len:
                        continue
                    data_item_list = self.process_text_and_imu_signal(data_item, humanml3d_text_path)
                    result_data.extend(data_item_list)

        return result_data

    def process_text_and_imu_signal(self, data_item, text_file):
        text_data = []
        flag = False

        result_list = []

        with cs.open(text_file) as f:
            for line in f.readlines():
                text_dict = {}
                line_split = line.strip().split('#')
                caption = line_split[0]
                tokens = line_split[1].split(' ')
                f_tag = float(line_split[2])
                to_tag = float(line_split[3])
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                text_dict['caption'] = caption
                text_dict['tokens'] = tokens
                if f_tag == 0.0 and to_tag == 0.0:
                    flag = True
                    text_data.append(text_dict)
                else:
                    try:
                        frame_rate = 25
                        new_data_item = {
                            'imu_acc': data_item['imu_acc'][int(f_tag * frame_rate): int(to_tag * frame_rate)],
                            'imu_ori': data_item['imu_ori'][int(f_tag * frame_rate): int(to_tag * frame_rate)],
                            'smpl_pose': data_item['smpl_pose'][int(f_tag * frame_rate): int(to_tag * frame_rate)],
                            'joints': data_item['joints'][int(f_tag * frame_rate): int(to_tag * frame_rate)],
                            'shape': deepcopy(data_item['shape']),
                            'transl': data_item['transl'][int(f_tag * frame_rate): int(to_tag * frame_rate)],
                            'combo_name': deepcopy(data_item['combo_name']),
                            'imu_combo': deepcopy(data_item['imu_combo']),
                            'text_dict': [text_dict],
                        }
                        if len(new_data_item['imu_acc']) < self.min_seq_len or len(new_data_item['imu_acc'] >= 200):
                            continue

                        if self.add_w_text:
                            result_list.append(deepcopy(new_data_item))
                        if self.add_wo_text:
                            data_item_wo_text = deepcopy(new_data_item)
                            data_item_wo_text['text_dict'] = [{'caption': '', 'tokens': ['']}]
                            result_list.append(data_item_wo_text)

                    except:
                        print(line_split)
                        print(line_split[2], line_split[3], f_tag, to_tag)
                        # break

        if flag:
            data_item['text_dict'] = text_data
            if self.add_w_text:
                result_list.append(data_item)
            if self.add_wo_text:
                data_item_wo_text = deepcopy(data_item)
                data_item_wo_text['text_dict'] = [{'caption': '', 'tokens': ['']}]  # dummy text
                result_list.append(data_item_wo_text)
        return result_list

    def __getitem__(self, idx):

        data_item = deepcopy(self.data[idx])
        smpl_pose = data_item['smpl_pose']

        # print(smpl_pose.shape)

        # if self.config.r6d is True:
        #     data_item['smpl_pose'] = math.rotation_matrix_to_r6d(smpl_pose).reshape(-1, 24, 6)[:,
        #                              self.config.pred_joints_set].reshape(-1, 6 * len(self.config.pred_joints_set))
        # else:
        #     data_item['smpl_pose'] = smpl_pose.contiguous()
        text_data = random.choice(data_item['text_dict'])
        caption, tokens = text_data['caption'], text_data['tokens']
        data_item['text'] = caption
        data_item['tokens'] = tokens

        # deal with the text stuff
        data_item = self.pipeline(data_item)

        return data_item

    def __len__(self):
        return len(self.data)
