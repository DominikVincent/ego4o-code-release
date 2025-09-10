import torch

from mmpose.utils.keypoints_mapping.renderpeople import render_people_joint_names
from mmpose.utils.keypoints_mapping.renderpeople_old import render_people_joint_names as render_people_old_joint_names
from mmpose.utils.keypoints_mapping.mo2cap2 import mo2cap2_joint_names
# from mmpose.data.keypoints_mapping.mo2cap2_with_hands import mo2cap2_with_hands_joint_names
from mmpose.utils.keypoints_mapping.smplx import smplx_joint_names
from mmpose.utils.keypoints_mapping.mano import mano_joint_names, mano_left_hand_joint_names, mano_right_hand_joint_names
from mmpose.utils.keypoints_mapping.smplh import smplh_joint_names
from mmpose.utils.keypoints_mapping.smpl import smpl_joint_names
from mmpose.utils.keypoints_mapping.studio import studio_joint_names
from mmpose.utils.keypoints_mapping.mo2cap2_with_head import mo2cap2_with_head_joint_names
from mmpose.utils.keypoints_mapping.beat import beat_joint_names
from mmpose.utils.keypoints_mapping.nymeria import nymeria_joint_names
from mmpose.utils.keypoints_mapping.hml import hml_joint_names
import numpy as np

def dset_to_body_model_with_kpt_names(model_type='smplx', dset='coco', use_face_contour=False):
    mapping = {}

    if model_type == 'smplx':
        keypoint_names = smplx_joint_names
    elif model_type == 'hml':
        keypoint_names = hml_joint_names
    elif model_type == 'nymeria':
        keypoint_names = nymeria_joint_names
    elif model_type == 'beat':
        keypoint_names = beat_joint_names
    elif model_type == 'mano_left':
        keypoint_names = mano_left_hand_joint_names
    elif model_type == 'mano_right':
        keypoint_names = mano_right_hand_joint_names
    elif model_type == 'studio':
        keypoint_names = studio_joint_names
    elif model_type == 'renderpeople':
        keypoint_names = render_people_joint_names
    elif model_type == 'renderpeople_old':
        keypoint_names = render_people_old_joint_names
    elif model_type == 'mo2cap2':
        keypoint_names = mo2cap2_joint_names
    elif model_type == 'smpl':
        keypoint_names = smpl_joint_names
    elif model_type == 'smplh':
        keypoint_names = smplh_joint_names
    else:
        raise ValueError('Unknown model dataset: {}'.format(model_type))

    if dset == 'mo2cap2':
        dset_keyp_names = mo2cap2_joint_names
    elif dset == 'hml':
        dset_keyp_names = hml_joint_names
    elif dset == 'nymeria':
        dset_keyp_names = nymeria_joint_names
    elif dset == 'beat':
        dset_keyp_names = beat_joint_names
    elif dset == 'mo2cap2_with_head':
        dset_keyp_names = mo2cap2_with_head_joint_names
    elif dset == 'mano_left':
        dset_keyp_names = mano_left_hand_joint_names
    elif dset == 'mano_right':
        dset_keyp_names = mano_right_hand_joint_names
    elif dset == 'studio':
        dset_keyp_names = studio_joint_names
    elif dset == 'renderpeople':
        dset_keyp_names = render_people_joint_names
    elif dset == 'renderpeople_old':
        dset_keyp_names = render_people_old_joint_names
    elif dset == 'smpl':
        dset_keyp_names = smpl_joint_names
    elif dset == 'smplh':
        dset_keyp_names = smplh_joint_names
    elif dset == 'smplx':
        dset_keyp_names = smplx_joint_names
    else:
        raise ValueError('Unknown dset dataset: {}'.format(dset))

    for idx, name in enumerate(keypoint_names):
        if 'contour' in name and not use_face_contour:
            continue
        if name in dset_keyp_names:
            mapping[idx] = dset_keyp_names.index(name)

    model_keyps_idxs = np.array(list(mapping.keys()), dtype=np.int32)
    dset_keyps_idxs = np.array(list(mapping.values()), dtype=np.int32)

    return dset_keyps_idxs, model_keyps_idxs, dset_keyp_names, keypoint_names


def dset_to_body_model(model_type='smplx', dset='coco', use_face_contour=False):

    dset_keyps_idxs, model_keyps_idxs, _, _ = dset_to_body_model_with_kpt_names(model_type=model_type,
                                                                                dset=dset,
                                                                                use_face_contour=use_face_contour)

    return dset_keyps_idxs, model_keyps_idxs

class JointConverter:
    def __init__(self, source_joint_name, target_joint_name):
        self.source_joint_name = source_joint_name
        self.target_joint_name = target_joint_name

        self.source_index, self.target_index, self.source_kpt_names, self.target_kpt_names = dset_to_body_model_with_kpt_names(
            model_type=self.target_joint_name, dset=self.source_joint_name)

    def convert(self, source_pose):
        source_pose_shape = source_pose.shape
        target_pose_shape = source_pose_shape[:-2] + (len(self.target_kpt_names), 3)
        if torch.is_tensor(source_pose):
            target_pose = torch.zeros(target_pose_shape).float().to(source_pose.device)
        else:
            target_pose = np.zeros(target_pose_shape).astype(float)

        target_pose[..., self.target_index, :] = source_pose[..., self.source_index, :]
        return target_pose

if __name__ == '__main__':
    joint_converter = JointConverter(source_joint_name='mo2cap2', target_joint_name='smpl')

    mo2cap2_pose = np.random.rand(3, 4, 5, 15, 3)
    smplh_pose = joint_converter.convert(mo2cap2_pose)
    print(smplh_pose.shape)
    print(smplh_pose[0, 0, 0])

    joint_converter2 = JointConverter(source_joint_name='smpl', target_joint_name='mo2cap2')
    mo2cap2_pose_recover = joint_converter2.convert(smplh_pose)
    print(mo2cap2_pose_recover.shape)
    print(np.sum(mo2cap2_pose_recover - mo2cap2_pose))
