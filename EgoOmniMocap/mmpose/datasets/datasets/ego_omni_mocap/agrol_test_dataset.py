# this file read the amass dataset and convert it to the format including head and hand 6d pose trajectory

import glob
import os
import pickle
from copy import deepcopy

import open3d.visualization
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from mmpose.registry import DATASETS  # note: now the dataset is registered under the TLControl project
# from datasets.transforms.shared_transforms import Compose
from mmengine.dataset.base_dataset import Compose
from mmpose.utils.visualization.draw import draw_keypoints_3d


@DATASETS.register_module()
class AgorlTestDataset(Dataset):
    def __init__(self,
                 dataset_root='/CT/EgoMocap/work/AGRoL/dataset/AMASS',
                 seq_len=196,
                 overlap=0,
                 add_last=False,
                 pipeline=None,
                 visualize_data=False,
                 ):
        super(AgorlTestDataset, self).__init__()
        self.dataset_root = dataset_root
        self.split = 'test'
        self.seq_len = seq_len
        self.overlap = overlap
        self.add_last = add_last
        self.visualize_data = visualize_data

        mean_path = 'amass_mean.pt'
        std_path = 'amass_std.pt'
        self.mean = torch.load(os.path.join(self.dataset_root, mean_path))
        self.std = torch.load(os.path.join(self.dataset_root, std_path))

        self.data_list = self.load_data_list()

        self.pipeline = Compose(pipeline)

    def load_data_list(self):
        # the return is list including dicts
        motion_list = self.get_path(self.dataset_root, self.split)

        filename_list = deepcopy(motion_list)
        motion_list = [torch.load(i) for i in tqdm(motion_list)]
        # motion_list = []
        # for i in tqdm(motion_list):
        #     with open(i, 'rb') as f:
        #         motion_i = pickle.load(f)
        #     motion_list.append(motion_i)

        if self.visualize_data:
            global_smpl_motion_vis = motion_list[0]['position_global_full_gt_world'][::20]
            global_smpl_motion_vis = global_smpl_motion_vis.reshape((-1, 3))
            mesh = draw_keypoints_3d(global_smpl_motion_vis)
            coor = open3d.geometry.TriangleMesh.create_coordinate_frame()
            open3d.visualization.draw_geometries([mesh, coor])

        # filter the motion without feet contact
        motion_list = self.filter_motion_wo_feet_contact(motion_list, filename_list=filename_list)

        data_list = []
        for i in range(len(motion_list)):
            if self.seq_len is not None:
                split_motion_list = self.split_motion(motion_list[i]['position_global_full_gt_world'],
                                                      seq_length=self.seq_len, overlap=self.overlap,
                                                      add_last=self.add_last)
                for motion in split_motion_list:
                    result_i = dict(
                        global_smpl_motion=motion,
                        filename=filename_list[i],
                        lengths=len(motion),
                    )
                    data_list.append(result_i)
            else:
                result_i = dict(
                    motion=motion_list[i],
                    global_smpl_motion=motion_list[i]['position_global_full_gt_world'],
                    filename=filename_list[i],
                    lengths=len(motion_list[i]['position_global_full_gt_world']),
                )
                data_list.append(result_i)
        return data_list

    def filter_motion_wo_feet_contact(self, motion_dict, filename_list, feet_id=(8, 11, 7, 10)):
        result_motion_list = []
        for i in range(len(motion_dict)):
            motion = motion_dict[i]['position_global_full_gt_world']
            # get the feet height
            feet_height = motion[:, feet_id, 2]
            # check if the feet in each frame is in contact
            feet_contact = (feet_height < 0.05).int()
            # check if the body is in contact with floor
            feet_contact = feet_contact.sum(axis=1) > 0

            # check if the body is in contact for the whole sequence
            if feet_contact.sum() == len(feet_contact):
                result_motion_list.append(motion_dict[i])
            else:
                print(f"motion {i} is not in contact, the file name is {filename_list[i]}")
        return result_motion_list

    def split_motion(self, motion, seq_length=196, overlap=0, add_last=True):
        # append the remaining motion less than seq len behind

        motion_length = len(motion)
        motion_list = []
        for i in range(0, motion_length - seq_length + 1, seq_length - overlap):
            motion_list.append(motion[i:i + seq_length])
        if add_last:
            if motion_length % (seq_length - overlap) != 0:
                motion_list.append(motion[-seq_length:])
        return motion_list


    def get_path(self, dataset_path, split):
        data_list_path = []
        parent_data_path = glob.glob(dataset_path + "/*")
        for d in parent_data_path:
            if os.path.isdir(d):
                files = glob.glob(d + "/" + split + "/*pt")
                data_list_path.extend(files)
        return data_list_path

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        data = deepcopy(self.data_list[item])
        data = self.pipeline(data)
        return data
