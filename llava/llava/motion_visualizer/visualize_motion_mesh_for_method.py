import argparse
import os
import pickle

import numpy as np
import open3d

from llava.ego4o.dataset.keypoints_mapping.joint_converter import JointConverter
from llava.motion_visualizer.pose_visualization_utils import get_sphere, get_cylinder
from llava.ego4o.dataset.keypoints_mapping.hml import joint_chain_indices

base_path = r'/scratch/inf0/user/jianwang/nymeria/ego4o_input_motion'

def skeleton_to_mesh(pose, joint_color=(0.1, 0.1, 0.7), bone_color=(0.1, 0.9, 0.1)):
    final_mesh = open3d.geometry.TriangleMesh()
    for i in range(len(pose)):
        keypoint_mesh = get_sphere(position=pose[i], radius=0.03, color=joint_color)
        final_mesh = final_mesh + keypoint_mesh

    for chain in joint_chain_indices:
        for i in range(len(chain) - 1):
            start_i = chain[i]
            end_i = chain[i + 1]

            start_point = pose[start_i]
            end_point = pose[end_i]

            line_mesh = get_cylinder(start_point, end_point, radius=0.0075, color=bone_color)
            final_mesh += line_mesh
    return final_mesh

def main(args):
    motion_file_ids = [
        # '37_20230725_s1_julie_taylor_act4_iemk0l',
        '38_20230725_s1_julie_taylor_act4_iemk0l',
        '39_20230725_s1_julie_taylor_act4_iemk0l'
    ]
    joints_seq_nymeria = []
    for motion_file_id in motion_file_ids:
        # split the text after first "_"
        motion_file_name = motion_file_id.split("_", 1)[1]
        motion_file_path= os.path.join(base_path, motion_file_name + '.pkl')

        with open(motion_file_path, 'rb') as f:
            motion_data = pickle.load(f)

        motion_data = motion_data[motion_file_id]

        joints_seq_nymeria.extend(motion_data['segment_tXYZ'])

    joints_seq_nymeria = np.asarray(joints_seq_nymeria)
    if joints_seq_nymeria.shape[-1] != 3:
        new_joints_shape = joints_seq_nymeria.shape[:-1] + (joints_seq_nymeria.shape[-1] // 3, 3)
        joints_seq_nymeria = joints_seq_nymeria.reshape(new_joints_shape)

    # put the feet on the ground
    min_height = joints_seq_nymeria[:, :, 2].min()
    joints_seq_nymeria[:, :, 2] -= min_height

    # put the x-y to the center
    joints_seq_nymeria[:, :, 0] -= joints_seq_nymeria[:, 0:1, 0]
    joints_seq_nymeria[:, :, 1] -= joints_seq_nymeria[:, 0:1, 1]

    # from nymeria to hml
    joint_converter = JointConverter('nymeria', 'hml')
    joints_seq_hml = joint_converter.convert(joints_seq_nymeria)
    # save each joint with open3d
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
    for i, joints in enumerate(joints_seq_hml):
        mesh = skeleton_to_mesh(joints)
        # open3d.visualization.draw_geometries([mesh])
        if args.output_path:

            open3d.io.write_triangle_mesh(os.path.join(args.output_path, f'{i:04d}.ply'), mesh)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Visualize the motion mesh')
    arg_parser.add_argument('--motion_file_id', type=str, default=None, required=False,
                            help='Path to the motion file')
    arg_parser.add_argument('--output_path', type=str, default=None, required=False,
                            help='Path to the output file')
    args = arg_parser.parse_args()
    main(args)
