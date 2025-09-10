import os.path
import random
from copy import deepcopy

import torch
from mmengine.dataset.base_dataset import Compose
from natsort import natsorted
from torch.utils.data import Dataset
from tqdm import tqdm

import mmpose.datasets.datasets.imuposer.math as math
from mmpose.datasets.datasets.imuposer.imuposer_config import Config, amass_combos, tlcontrol_combos
from mmpose.registry import DATASETS


@DATASETS.register_module()
class IMUPoserLocDataset(Dataset):
    def __init__(self, imuposer_root_dir='/home/jianwang/EgoMocap/work/IMUPoser',
                 seq_len=196,
                 min_seq_len=196,
                 signal_num=5,
                 tlcontrol_joint_sequence=False,
                 random_mask=False,
                 combo_name='global',
                 split="train",
                 pipeline=None, ):
        super().__init__()

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

        self.config.processed_imu_poser = self.config.root_dir / "data/processed_imuposer"

        self.seq_len = seq_len
        self.min_seq_len = min_seq_len
        self.data = self.load_data()
        self.pipeline = Compose(pipeline)


    def read_loc_train_data(self, train):
        data_root_path = os.path.join(self.config.processed_imu_poser, 'AMASS_IMU_LOC')
        result = []
        for sub_dataset_dir in os.listdir(data_root_path):
            sub_dataset_path = os.path.join(data_root_path, sub_dataset_dir)
            joint = torch.load(os.path.join(sub_dataset_path, "joint.pt"))
            pose = torch.load(os.path.join(sub_dataset_path, "pose.pt"))
            shape = torch.load(os.path.join(sub_dataset_path, "shape.pt"))
            tran = torch.load(os.path.join(sub_dataset_path, "tran.pt"))
            vacc = torch.load(os.path.join(sub_dataset_path, "vacc.pt"))
            vrot = torch.load(os.path.join(sub_dataset_path, "vrot.pt"))
            imu_loc = torch.load(os.path.join(sub_dataset_path, "imu_loc.pt"))

            data = {
                "joint": joint,
                "pose": pose,
                "shape": shape,
                "tran": tran,
                "acc": vacc,
                "ori": vrot,
                "imu_loc": imu_loc
            }
            result.append(data)
            if train == 'val':
                break
        return result

    def load_data(self):
        if self.train == "train" or self.train == 'val':
            data_files = self.read_loc_train_data(self.train)
        elif self.train == 'finetune':
            data_files = ["dip_train.pt"]
        elif self.train == 'test':
            data_files = ["dip_test.pt"]
        else:
            raise ValueError("Invalid split")

        result_data = []

        for fname in tqdm(data_files):
            if type(fname) == str:
                fdata = torch.load(self.config.processed_imu_poser_25fps / fname)
            else:
                assert type(fname) == dict
                fdata = fname

            for i in range(len(fdata["acc"])):
                # inputs
                facc = fdata["acc"][i]
                fori = fdata["ori"][i]
                imu_loc = fdata["imu_loc"][i]

                # load all the data
                if self.tlcontrol_joint_sequence is False:
                    # use original imuposer joint sequence
                    glb_acc = facc.view(-1, 6, 3)[:, [0, 1, 2, 3, 4]] / self.config.acc_scale
                    glb_ori = fori.view(-1, 6, 3, 3)[:, [0, 1, 2, 3, 4]]
                    glb_imu_loc = imu_loc.view(-1, 6, 3)[:, [0, 1, 2, 3, 4]]
                else:
                    # use tlcontrol joint sequence
                    glb_acc = facc.view(-1, 6, 3)[:, [5, 4, 0, 1, 2, 3]] / self.config.acc_scale
                    glb_ori = fori.view(-1, 6, 3, 3)[:, [5, 4, 0, 1, 2, 3]]
                    glb_imu_loc = imu_loc.view(-1, 6, 3)[:, [5, 4, 0, 1, 2, 3]]

                acc = glb_acc
                ori = glb_ori
                loc = glb_imu_loc

                # outputs
                fpose = fdata["pose"][i]
                fpose = fpose.reshape(fpose.shape[0], -1)

                joints_gt = fdata['joint'][i]
                shape_gt = fdata['shape'][i]
                transl_gt = fdata['tran'][i]

                for _combo in list(self.combo_list):
                    # acc N, 5, 3
                    # ori N, 5, 3, 3
                    if self.random_mask is False:
                        if _combo != self.combo_name:
                            continue

                    _combo_acc = torch.zeros_like(acc)
                    _combo_ori = torch.zeros((3, 3)).repeat(ori.shape[0], self.signal_num, 1, 1)

                    _combo_acc[:, self.combo_list[_combo]] = acc[:, self.combo_list[_combo]]
                    _combo_ori[:, self.combo_list[_combo]] = ori[:, self.combo_list[_combo]]

                    imu_acc_list = torch.split(_combo_acc, self.seq_len)
                    imu_ori_list = torch.split(_combo_ori, self.seq_len)

                    imu_inputs = torch.cat([_combo_acc.flatten(1), _combo_ori.flatten(1)], dim=1)
                    imu_inputs_split = torch.split(imu_inputs, self.seq_len)

                    smpl_pose_list = torch.split(fpose, self.seq_len)
                    joints_list = torch.split(joints_gt, self.seq_len)
                    transl_list = torch.split(transl_gt, self.seq_len)
                    loc_list = torch.split(loc, self.seq_len)

                    assert len(imu_inputs_split) == len(imu_acc_list) == len(imu_ori_list) == len(
                        smpl_pose_list) == len(
                        joints_list) == len(transl_list) == len(loc_list)

                    for seq_id in range(len(imu_inputs_split)):
                        data_item = {
                            'imu': imu_inputs_split[seq_id],
                            'imu_acc': imu_acc_list[seq_id],
                            'imu_ori': imu_ori_list[seq_id],
                            'smpl_pose': smpl_pose_list[seq_id],
                            'joints': joints_list[seq_id],
                            'shape': deepcopy(shape_gt),
                            'transl': transl_list[seq_id],
                            'combo_name': _combo,
                            'imu_combo': deepcopy(self.combo_list[_combo]),
                            'imu_loc': loc_list[seq_id]
                        }
                        # throw away if the sequence is too short:
                        if data_item['imu_acc'].shape[0] < self.min_seq_len:
                            continue
                        result_data.append(data_item)

        return result_data

    def __getitem__(self, idx):

        data_item = deepcopy(self.data[idx])
        smpl_pose = data_item['smpl_pose']
        if self.config.r6d is True:
            smpl_pose = math.axis_angle_to_rotation_matrix(smpl_pose).reshape(-1, 24, 3, 3)
            data_item['smpl_pose'] = math.rotation_matrix_to_r6d(smpl_pose).reshape(-1, 24, 6)[:,
                                     self.config.pred_joints_set].reshape(-1, 6 * len(self.config.pred_joints_set))
        data_item = self.pipeline(data_item)

        return data_item

    def __len__(self):
        return len(self.data)
