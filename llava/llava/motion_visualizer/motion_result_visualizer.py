import json
import pdb
import pickle
import sys

import torch

sys.path.append('/CT/EgoMocap/work/LLaVA')
from llava.ego4o.utils.humanml_utils.motion_process import recover_from_ric

import numpy as np
import os

# from mmpose.utils.visualization.draw import draw_keypoints_3d

import matplotlib.pyplot as plt
import matplotlib
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from textwrap import wrap
from typing import Optional

from llava.ego4o.dataset.transforms.motion_transforms import NormalizeHMLMotion, ConvertNymeriaToHML, ZUp2YUp, \
    InitAlignIMUMotion

# skeleton = [[0, 1, 2, 3, 4, 5, 6], [4, 7, 8, 9, 10], [4, 11, 12, 13, 14], [0, 15, 16, 17, 18], [0, 19, 20, 21, 22]]
skeleton = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
                               [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                               [9, 13, 16, 18, 20]]

def plot_3d_motion(save_path: str, joints: np.ndarray, acc=None, title: str='default title',
                   figsize: tuple[int, int] = (10, 10),
                   fps: int = 20, radius: int = 2, kinematic_tree: list = skeleton,
                   limb_ori_matrix=None):
    import warnings
    warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

    title = '\n'.join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        # Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    # data *= 1.3  # scale for visualization
    # if hint is not None:
    #     mask = hint.sum(-1) != 0
    #     hint = hint[mask]
    #     hint *= 1.3

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ["#DD5A37", "#D69E00", "#B75A39", "#DD5A37", "#D69E00",
              "#FF6D00", "#FF6D00", "#FF6D00", "#FF6D00", "#FF6D00",
              "#DDB50E", "#DDB50E", "#DDB50E", "#DDB50E", "#DDB50E", ]

    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    # if hint is not None:
    #     hint[..., 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    if acc is not None:
        # move acc to the location of each joint
        acc = acc.copy().reshape(len(acc), -1, 3)
        acc = acc / 30 # for visualization
        acc = acc + data

    if limb_ori_matrix is not None:
        limb_ori_matrix = limb_ori_matrix.reshape(len(limb_ori_matrix), -1, 3, 3)


    def update(index):
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=90, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])

        nonlocal acc
        if acc is not None:
            # plot acceleration
            for i in range(acc.shape[1]):
                ax.plot3D([data[index, i, 0], acc[index, i, 0]],
                          [data[index, i, 1], acc[index, i, 1]],
                          [data[index, i, 2], acc[index, i, 2]],
                          linewidth=4.0, color='green')

        nonlocal limb_ori_matrix
        if limb_ori_matrix is not None:
            # plot limb orientation
            for i in range(limb_ori_matrix.shape[1]):
                if i != 13:
                    continue
                limb_ori = limb_ori_matrix[index, i]
                orientation_color_list = ['red', 'green', 'blue']
                for j in range(limb_ori.shape[0]):
                    ax.plot3D([data[index, i, 0], data[index, i, 0] + limb_ori[0, j] * 0.1],
                              [data[index, i, 1], data[index, i, 1] + limb_ori[1, j] * 0.1],
                              [data[index, i, 2], data[index, i, 2] + limb_ori[2, j] * 0.1],
                              linewidth=5.0, color=orientation_color_list[j])

        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            linewidth = 3.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)
    ani.save(save_path, fps=fps)
    # ani = FuncAnimation(fig, update, frames=frame_number, repeat=False)
    # FFwriter = FFMpegWriter(fps=fps)
    # ani.save(save_path, writer=FFwriter)
    plt.close()

def visualize_temp(motion_pkl_save_path):
    # motion_pkl_save_path = '/CT/EgoMocap/work/LLaVA/temp.pkl'
    with open(motion_pkl_save_path, 'rb') as f:
        motion_data = pickle.load(f)
        # motion = motion_data[0]
    data_dict = {'motion_hml': motion_data.cpu()}
    normalize = NormalizeHMLMotion(hml_motion_name='motion_hml', hml_mean_path='/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_mean.pt', hml_std_path='/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_std.pt')
    data_dict_out = normalize.inv_normalize(data_dict)
    motion_hml = data_dict_out['motion_hml']

    # convert from hml to joints
    from llava.ego4o.utils.humanml_utils.motion_process import recover_from_ric
    pred_coords = recover_from_ric(motion_hml, 22)[0, ...]

    plot_3d_motion(os.path.splitext(motion_pkl_save_path)[0] + '.gif', pred_coords.numpy(),
                   None, 'motion', (10, 10), 20, 2, skeleton, None)


def main(motion_path):
    with open(motion_path, "rb") as f:
        motion_humanml = pickle.load(f)  # motion data is a dictionary
    # random select one id from motion_list
    motion_humanml = torch.from_numpy(motion_humanml).float()
    # motion_humanml = motion_humanml[:, 40:]

    # inverse normalize
    normalize = NormalizeHMLMotion(hml_motion_name='motion_hml',
                           hml_mean_path='/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_mean.pt',
                           hml_std_path='/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_std.pt')

    input_dict = {
        'motion_hml': motion_humanml
    }
    output_dict = normalize.inv_normalize(input_dict)


    motion = recover_from_ric(output_dict['motion_hml'], 22)[0, ...]
    motion = motion.cpu().numpy()
    motion = motion[:150]

    # breakpoint()

    plot_3d_motion(f'ego4o_output.gif', motion, None, 'motion', (10, 10), 20, 2, skeleton, None)

if __name__ == '__main__':
    # visualize_temp('/CT/EgoMocap/work/LLaVA/temp_out.pkl')
    # visualize_temp('/CT/EgoMocap/work/LLaVA/temp_input.pkl')
    main('/CT/EgoMocap/work/LLaVA/motion_out/output_motion_info.pkl')
