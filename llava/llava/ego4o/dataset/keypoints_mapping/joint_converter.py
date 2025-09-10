import torch
from llava.ego4o.dataset.keypoints_mapping.hml import hml_joint_names
from llava.ego4o.dataset.keypoints_mapping.nymeria import nymeria_joint_names
import numpy as np

def dset_to_body_model_with_kpt_names(model_type='hml', dset='nymeria', use_face_contour=False):
    mapping = {}

    if model_type == 'hml':
        keypoint_names = hml_joint_names
    elif model_type == 'nymeria':
        keypoint_names = nymeria_joint_names

    else:
        raise ValueError('Unknown model dataset: {}'.format(model_type))

    if dset == 'hml':
        dset_keyp_names = hml_joint_names
    elif dset == 'nymeria':
        dset_keyp_names = nymeria_joint_names
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
    joint_converter = JointConverter(source_joint_name='nymeria', target_joint_name='hml')

    nymeria = np.random.rand(23, 3)
    hml_pose = joint_converter.convert(nymeria)
    print(hml_pose)

