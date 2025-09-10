# this file read the amass dataset and convert it to the format including head and hand 6d pose trajectory

import glob
import os
import pickle
from copy import deepcopy

import numpy as np
import open3d.visualization
import torch
from natsort import natsorted
from torch.utils.data import Dataset
from tqdm import tqdm

from mmpose.registry import DATASETS  # note: now the dataset is registered under the TLControl project
# from datasets.transforms.shared_transforms import Compose
from mmengine.dataset.base_dataset import Compose

from mmpose.utils.keypoints_mapping.joint_converter import dset_to_body_model
from mmpose.utils.visualization.draw import draw_keypoints_3d


@DATASETS.register_module()
class SceneEgoTestDataset(Dataset):
    # load scene ego predicted motion sequence and convert it to standard format

    frame_rate = 25
    def __init__(self,
                 data_pkl_path,
                 seq_len=196,
                 overlap=0,
                 add_last=False,
                 pipeline=None,
                 visualize_data=False,
                 ):
        super(SceneEgoTestDataset, self).__init__()
        self.data_pkl_path = data_pkl_path
        self.split = 'test'
        self.seq_len = seq_len
        self.overlap = overlap
        self.add_last = add_last
        self.visualize_data = visualize_data

        self.mo2cap2_idxs, self.smpl_idxs_mo2cap2 = dset_to_body_model(dset='mo2cap2', model_type='smpl')

        self.data_list = self.load_data_list()

        self.pipeline = Compose(pipeline)


    def load_data_list(self):
        data = self.load_network_pred_pkl(self.data_pkl_path)
        # convert estimated pose from local pose to global pose
        for seq_name in data.keys():
            for i, item in enumerate(data[seq_name]):
                ext_id = item['ext_id']
                ego_camera_pose = item['ego_camera_pose']
                pred_joints_3d = item['pred_joints_3d']
                pred_left_hand_joint_3d = item['pred_left_hand_joint_3d']
                pred_right_hand_joint_3d = item['pred_right_hand_joint_3d']

                pred_joints_3d_confidence = item['pred_joints_3d_confidence']
                pred_left_hand_confidence = item['pred_left_hand_confidence']
                pred_right_hand_confidence = item['pred_right_hand_confidence']

                gt_joints_3d = item['gt_joints_3d']

                # note:
                # 这里出问题了，预测的mano joint可能不是在egocentric space里面的，这个mano的joint的location其实
                # 是受到wrist location的控制的，所以这里需要先提取出来wrist location，然后对mano joint进行处理
                pred_left_wrist = pred_joints_3d[6: 7]
                pred_right_wrist = pred_joints_3d[3: 4]
                ego_pred_left_hand_joint_3d = pred_left_hand_joint_3d + pred_left_wrist - pred_left_hand_joint_3d[0: 1]
                ego_pred_right_hand_joint_3d = pred_right_hand_joint_3d + pred_right_wrist - pred_right_hand_joint_3d[
                                                                                             0: 1]

                data[seq_name][i]['ext_id'] = ext_id
                data[seq_name][i]['seq_name'] = seq_name
                data[seq_name][i]['ego_pred_joints_3d'] = pred_joints_3d
                data[seq_name][i]['ego_pred_left_hand_joint_3d'] = ego_pred_left_hand_joint_3d
                data[seq_name][i]['ego_pred_right_hand_joint_3d'] = ego_pred_right_hand_joint_3d
                ego_smpl_pred_joints_3d = self.convert_mo2cap2_to_smpl_motion(pred_joints_3d)

                transformed_smpl_pred_joints_3d = self.transform_new_body_pose(ego_smpl_pred_joints_3d,
                                                                               ego_camera_pose)
                transformed_pred_joints_3d = self.transform_new_body_pose(pred_joints_3d, ego_camera_pose)
                transformed_pred_left_hand_joint_3d = self.transform_new_body_pose(ego_pred_left_hand_joint_3d,
                                                                                   ego_camera_pose)
                transformed_pred_right_hand_joint_3d = self.transform_new_body_pose(ego_pred_right_hand_joint_3d,
                                                                                    ego_camera_pose)
                transformed_global_gt_joints_3d = self.transform_new_body_pose(gt_joints_3d, ego_camera_pose)
                data[seq_name][i]['global_smpl_motion'] = transformed_smpl_pred_joints_3d
                data[seq_name][i]['global_pred_joints_3d'] = transformed_pred_joints_3d
                data[seq_name][i]['global_pred_left_hand_joint_3d'] = transformed_pred_left_hand_joint_3d
                data[seq_name][i]['global_pred_right_hand_joint_3d'] = transformed_pred_right_hand_joint_3d
                data[seq_name][i]['pred_joints_3d_confidence'] = pred_joints_3d_confidence
                data[seq_name][i]['pred_left_hand_confidence'] = pred_left_hand_confidence
                data[seq_name][i]['pred_right_hand_confidence'] = pred_right_hand_confidence

                data[seq_name][i]['ego_gt_joints_3d'] = gt_joints_3d
                data[seq_name][i]['gt_joints_3d'] = transformed_global_gt_joints_3d
        data_out = []
        if self.seq_len is not None:
            # split sequence
            for seq_name in data.keys():
                data_seq = data[seq_name]
                data_seq = self.split_motion(data_seq, seq_length=self.seq_len,
                                             overlap=self.overlap, add_last=self.add_last)
                data_out.extend(data_seq)
        else:
            for seq_name in data.keys():
                data_seq = data[seq_name]
                data_out.append(data_seq)


        # convert from list of dict to dict containing list
        data_out_list_of_dict = []
        for i in range(len(data_out)):
            # convert from list of dict to dict containing list
            key_names = data_out[i][0].keys()
            data_out_dict = {key: [] for key in key_names}
            for j in range(len(data_out[i])):
                for key in key_names:
                    data_out_dict[key].append(data_out[i][j][key])
            # convert to numpy array
            for key in key_names:
                data_out_dict[key] = np.asarray(data_out_dict[key])
            data_out_dict['lengths'] = 196
            data_out_list_of_dict.append(data_out_dict)

        # breakpoint()
        return data_out_list_of_dict

    def convert_mo2cap2_to_smpl_motion(self, ego_pred_joints_3d):
        # the main problem is to add the head joint in the smpl motion
        # 1. convert mo2cap2 pose to smpl pose

        # 2. add head joint to the smpl pose
        ego_smpl_body_joints = np.zeros((22, 3))
        ego_smpl_body_joints[self.smpl_idxs_mo2cap2] = ego_pred_joints_3d[self.mo2cap2_idxs]

        # set joint center for mo2cap2 -> smplx
        ego_smpl_body_joints[0] = (ego_smpl_body_joints[1] + ego_smpl_body_joints[2]) / 2.0

        # add head joint position for egocentric joint
        ego_head_joint = np.array([0, 0.25, 0.05])
        ego_smpl_body_joints[15] = ego_head_joint
        return ego_smpl_body_joints

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

    def transform_new_body_pose(self, keypoints, fisheye_camera_transformation_matrix):
        keypoints_homo = np.ones((len(keypoints), 4))
        keypoints_homo[:, :3] = keypoints
        global_keypoints_homo = fisheye_camera_transformation_matrix.dot(keypoints_homo.T).T
        transformed_pose = global_keypoints_homo[:, :3].astype(np.float32)
        return transformed_pose

    def load_network_pred_pkl(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        ext_id_list = []
        seq_name_list = []
        pred_joints_3d_list = []
        pred_left_hand_joint_3d_list = []
        pred_right_hand_joint_3d_list = []
        pred_joints_3d_confidence_list = []
        pred_left_hand_confidence_list = []
        pred_right_hand_confidence_list = []
        gt_joints_3d_list = []
        ego_camera_pose_list = []
        for _, pred in enumerate(data):
            pred_joints_3d_item = pred['body_pose_results']['keypoints_pred']
            pred_left_hand_joint_item = pred['left_hands_preds']['joint_cam_transformed']
            pred_right_hand_joint_item = pred['right_hands_preds']['joint_cam_transformed']
            if torch.is_tensor(pred_joints_3d_item):
                pred_joints_3d_item = pred_joints_3d_item.cpu().numpy()
            if torch.is_tensor(pred_left_hand_joint_item):
                pred_left_hand_joint_item = pred_left_hand_joint_item.cpu().numpy()
                pred_right_hand_joint_item = pred_right_hand_joint_item.cpu().numpy()

            # get the uncertainty
            if 'keypoint_confidence' in pred['body_pose_results'].keys():
                confidence_body = pred['body_pose_results']['keypoint_confidence']
                confidence_body = confidence_body[:, :, None].repeat(1, 1, 3)
            else:
                # use default uncertainty
                confidence_body = np.ones_like(pred_joints_3d_item)
                confidence_body[:, 8:] *= 0.5
            if 'keypoint_confidence' in pred['left_hands_preds'].keys():
                confidence_lhand = pred['left_hands_preds']['keypoint_confidence']
            else:
                # use default uncertainty
                confidence_lhand = np.ones_like(pred_left_hand_joint_item) * 0.5

            if 'keypoint_confidence' in pred['right_hands_preds'].keys():
                confidence_rhand = pred['right_hands_preds']['keypoint_confidence']
            else:
                # use default uncertainty
                confidence_rhand = np.ones_like(pred_right_hand_joint_item) * 0.5

            pred_joints_3d_list.extend(pred_joints_3d_item)
            pred_left_hand_joint_3d_list.extend(pred_left_hand_joint_item)
            pred_right_hand_joint_3d_list.extend(pred_right_hand_joint_item)

            pred_joints_3d_confidence_list.extend(confidence_body)
            pred_left_hand_confidence_list.extend(confidence_lhand)
            pred_right_hand_confidence_list.extend(confidence_rhand)
            img_meta_list = pred['img_metas']
            for img_meta_item in img_meta_list:
                ext_id = img_meta_item['ext_id']
                seq_name = img_meta_item['seq_name']
                gt_joints_3d_item = img_meta_item['keypoints_3d']
                ego_camera_pose_item = img_meta_item['ego_camera_pose']
                gt_joints_3d_list.append(gt_joints_3d_item)
                ego_camera_pose_list.append(ego_camera_pose_item)
                ext_id_list.append(ext_id)
                seq_name_list.append(seq_name)

        gt_joints_3d_list = np.array(gt_joints_3d_list)
        ego_camera_pose_list = np.array(ego_camera_pose_list)
        pred_joints_3d_list = np.array(pred_joints_3d_list)
        pred_left_hand_joint_3d_list = np.array(pred_left_hand_joint_3d_list)
        pred_right_hand_joint_3d_list = np.array(pred_right_hand_joint_3d_list)
        pred_joints_3d_confidence_list = np.array(pred_joints_3d_confidence_list)
        pred_left_hand_confidence_list = np.array(pred_left_hand_confidence_list)
        pred_right_hand_confidence_list = np.array(pred_right_hand_confidence_list)

        # split by seq names
        data_by_seq_name = {}
        for i, seq_name in enumerate(seq_name_list):
            if seq_name not in data_by_seq_name.keys():
                data_by_seq_name[seq_name] = []
            ext_id = ext_id_list[i]
            ego_camera_pose = ego_camera_pose_list[i]
            gt_joints_3d = gt_joints_3d_list[i]
            pred_joints_3d = pred_joints_3d_list[i]
            pred_left_hand_joint_3d = pred_left_hand_joint_3d_list[i]
            pred_right_hand_joint_3d = pred_right_hand_joint_3d_list[i]
            pred_joints_3d_confidence = pred_joints_3d_confidence_list[i]
            pred_left_hand_confidence = pred_left_hand_confidence_list[i]
            pred_right_hand_confidence = pred_right_hand_confidence_list[i]
            data_by_seq_name[seq_name].append({'ext_id': ext_id,
                                               'ego_camera_pose': ego_camera_pose,
                                               'gt_joints_3d': gt_joints_3d,
                                               'pred_joints_3d': pred_joints_3d,
                                               'pred_left_hand_joint_3d': pred_left_hand_joint_3d,
                                               'pred_right_hand_joint_3d': pred_right_hand_joint_3d,
                                               'pred_joints_3d_confidence': pred_joints_3d_confidence,
                                               'pred_left_hand_confidence': pred_left_hand_confidence,
                                               'pred_right_hand_confidence': pred_right_hand_confidence,
                                               })
        for seq_name in data_by_seq_name.keys():
            data_by_seq_name[seq_name] = natsorted(data_by_seq_name[seq_name], key=lambda x: x['ext_id'])

        return data_by_seq_name

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        data = deepcopy(self.data_list[item])
        data = self.pipeline(data)
        return data
