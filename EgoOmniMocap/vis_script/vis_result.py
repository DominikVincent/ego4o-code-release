import os
import pickle
from copy import deepcopy

import numpy as np
import open3d
import torch

import torch.nn.functional as F
import os

from mmpose.utils.visualization.pose_visualization_utils import get_cylinder


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def find_optimal_rotation_matrix(list_A, list_B):
    # Ensure the lists have the same length
    assert len(list_A) == len(list_B), "Lists A and B must have the same length"

    # Stack all matrices in A and B
    A_stack = list_A.reshape((-1, 3))
    B_stack = list_B.reshape((-1, 3))

    A_stack_pinv = np.linalg.pinv(A_stack)

    # Compute the rotation matrix R
    R = A_stack_pinv @ B_stack

    # Ensure R is a proper rotation matrix by projecting it onto the space of rotation matrices
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt

    return R


os.environ['PYOPENGL_PLATFORM'] = 'egl'


def visualize_hml3d(raw_result_path, out_dir):
    with open(raw_result_path, 'rb') as f:
        results = pickle.load(f)

    for i, result in enumerate(results):

        if i != 50:
            continue

        # breakpoint()

        print(result['text'])

        if torch.is_tensor(result['pred_coords']):
            result['pred_coords'] = result['pred_coords'].cpu().numpy()
        if torch.is_tensor(result['gt_coords']):
            result['gt_coords'] = result['gt_coords'].cpu().numpy()
        if torch.is_tensor(result['imu_acc']):
            result['imu_acc'] = result['imu_acc'].cpu().numpy()
        imu_ori_matrix = rotation_6d_to_matrix(
            torch.as_tensor(result['imu_ori']) if not torch.is_tensor(result['imu_ori']) else result['imu_ori'])
        result['imu_ori'] = imu_ori_matrix.cpu().numpy()

        frame_id = 10
        body_joints = result['gt_coords']
        imu_acc = result['imu_acc']
        imu_ori = result['imu_ori']

        # visualize the imu and acc
        imu_coord = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1)

        id_dict = {
            'left_wrist': {
                'imu_id': 2,
                'smpl_id': 20,

                'transform_matrix': np.asarray([[-0.31173626, 0.786909, -0.53253615],
                                                [-0.08138918, 0.5362905, 0.8401002],
                                                [-0.94667643, -0.30523238, 0.10313531]])
            },
            'right_wrist': {
                'imu_id': 3,
                'smpl_id': 21,
                'transform_matrix': np.asarray([[0.03774156, 0.9657682, 0.25664654],
                                                [0.9927653, -0.00694198, -0.11986986],
                                                [0.11398488, -0.25931388, 0.9590431]])
            },
            'left_leg': {
                'imu_id': 4,
                'smpl_id': 4,
                'transform_matrix': np.asarray([[-0.18974228, 0.97831357, 0.08306824],
                                                [-0.36027378, 0.0093292, -0.93279994],
                                                [0.9133458, 0.20691891, -0.3506906]])
            },
            'right_leg': {
                'imu_id': 5,
                'smpl_id': 5,
                'transform_matrix': np.asarray([[0.20768249, 0.96378094, 0.16731486],
                                                [-0.30838755, -0.09781241, 0.9462187],
                                                [-0.92831296, 0.24811089, -0.2769041]])
            },

        }

        joint_name = 'right_wrist'

        imu_poser_id = id_dict[joint_name]['imu_id']
        smpl_id = id_dict[joint_name]['smpl_id']
        transform_matrix = id_dict[joint_name].get('transform_matrix', None)
        # visualize body joints
        body_joints_frame_id = deepcopy(body_joints[frame_id])
        from mmpose.utils.visualization.draw import draw_keypoints_3d, get_arrow

        imu_orientation_matrix_i_1 = imu_ori[frame_id][imu_poser_id]

        imu_ori_mat_y_axis = imu_ori[:, imu_poser_id, 1, :]
        # left_wrist_ori = body_joints[:, smpl_id, :] - body_joints[:, smpl_id - 2, :]
        # left_wrist_ori = left_wrist_ori / np.linalg.norm(left_wrist_ori)
        # rot_mat = find_optimal_rotation_matrix(imu_ori_mat_y_axis, left_wrist_ori)

        # right_wrist_ori = body_joints[:, smpl_id, :] - body_joints[:, smpl_id - 2, :]
        # right_wrist_ori = right_wrist_ori / np.linalg.norm(right_wrist_ori)
        # rot_mat = find_optimal_rotation_matrix(imu_ori_mat_y_axis, right_wrist_ori)

        left_leg_ori = body_joints[:, smpl_id, :] - body_joints[:, smpl_id - 3, :]
        left_leg_ori = left_leg_ori / np.linalg.norm(left_leg_ori)
        rot_mat = find_optimal_rotation_matrix(imu_ori_mat_y_axis, left_leg_ori)

        print(rot_mat)

        rot_mat_save = id_dict[joint_name].get('transform_matrix', None)

        imu_accleration_i_rh = imu_acc[frame_id][imu_poser_id]

        imu_orientation_matrix_i_1_res = imu_orientation_matrix_i_1 @ rot_mat_save
        left_leg_ori_frame_norm = left_leg_ori[frame_id] / np.linalg.norm(left_leg_ori[frame_id])

        # imu_coord = imu_coord.rotate(imu_orientation_matrix_i_1 @ rot_mat, center=(0, 0, 0))
        imu_coord = imu_coord.rotate(imu_orientation_matrix_i_1_res, center=(0, 0, 0))

        body_joints_mesh = draw_keypoints_3d(body_joints_frame_id)
        imu_coord_joints = deepcopy(imu_coord)
        imu_coord_joints = imu_coord_joints.translate(body_joints_frame_id[smpl_id])
        # open3d.visualization.draw_geometries([body_joints_mesh, world_coord, imu_coord_joints])

        # visualize imu input
        world_coord = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        start_point = body_joints[frame_id, smpl_id, :]
        end_point = start_point + imu_orientation_matrix_i_1_res[:, 1]
        # accleration_arrow = get_arrow(start_point, end_point)
        accleration_arrow = get_cylinder(start_point, end_point, radius=0.0075)
        open3d.visualization.draw_geometries([body_joints_mesh, world_coord, accleration_arrow])
        open3d.visualization.draw_geometries([body_joints_mesh, world_coord, imu_coord_joints, ])


# calculate the mpjpe for with text and without text

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Visualize evaluation results')
    parser.add_argument('--dir_name', required=False, default='test_nymeria_random_image_text')
    parser.add_argument('--file_name', required=False, default='results_save_global')
    args = parser.parse_args()
    dir_name = args.dir_name
    file_name = args.file_name
    if os.name == 'nt':
        result_pkl_path = fr'Z:/EgoMocap/work/EgoOmniMocap/work_dirs/{dir_name}/{file_name}.pkl'
        out_dir = fr'Z:/EgoMocap/work/EgoOmniMocap/vis_out/visualize_{dir_name}_{file_name}'
    else:
        result_pkl_path = fr'/CT/EgoMocap/work/EgoOmniMocap/work_dirs/{dir_name}/{file_name}.pkl'
        out_dir = fr'/CT/EgoMocap/work/EgoOmniMocap/vis_out/visualize_{dir_name}_{file_name}'
    os.makedirs(out_dir, exist_ok=True)
    visualize_hml3d(result_pkl_path, out_dir)


if __name__ == '__main__':
    main()
