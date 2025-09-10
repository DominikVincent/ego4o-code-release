import os.path

import torch

from imuposer.config import Config
from imuposer import math

# -

config = Config(project_root_dir="../../")

# +
target_fps = 25


# hop = 60 // target_fps

def smooth_avg(acc=None, s=3):
    nan_tensor = (torch.zeros((s // 2, acc.shape[1], acc.shape[2])) * torch.nan)
    acc = torch.cat((nan_tensor, acc, nan_tensor))
    tensors = []
    for i in range(s):
        L = acc.shape[0]
        tensors.append(acc[i:L - (s - i - 1)])

    smoothed = torch.stack(tensors).nanmean(dim=0)
    return smoothed


def _resample(tensor, target_fps):
    r"""
        Resample to the target fps, assumes 60fps input
    """
    indices = torch.arange(0, tensor.shape[0], 60 / target_fps)

    start_indices = torch.floor(indices).long()
    end_indices = torch.ceil(indices).long()
    end_indices[end_indices >= tensor.shape[0]] = tensor.shape[0] - 1  # handling edge cases

    start = tensor[start_indices]
    end = tensor[end_indices]

    floats = indices - start_indices
    for shape_index in range(len(tensor.shape) - 1):
        floats = floats.unsqueeze(1)
    weights = torch.ones_like(start) * floats
    torch_lerped = torch.lerp(start, end, weights)
    return torch_lerped


# -
path_to_save = config.processed_imu_poser_25fps
path_to_save.mkdir(exist_ok=True, parents=True)

# 11 frames at 60 fps = 11*25/60
11 * 25 / 60

# process AMASS first

motion_dir = r'Z:\EgoMocap\work\EgoOmniMocap\scripts\amass_data_dict'
out_motion_dir = r'Z:\EgoMocap\work\EgoOmniMocap\scripts\amass_data_dict_25fps'
os.makedirs(out_motion_dir, exist_ok=True)
motion_names = [f for f in os.listdir(motion_dir) if f.endswith('.pt')]

# first split the motion with the human ml information
for motion_name in motion_names:
    fpath = os.path.join(motion_dir, motion_name)
    print('processing : {}'.format(fpath))
    # resample to 25 fps
    data_all = torch.load(fpath)
    out_data_all = []
    for data_item in data_all:
        humanml3d_info_list = data_item['humanml3d']
        # the target fps here is 20 fps
        # original fps is 60 fps, need to first convert to 60 fps
        for humanml_item in humanml3d_info_list:
            start_frame = humanml_item['start_frame'] * (60 // 20)
            end_frame = humanml_item['end_frame'] * (60 // 20)
            humanml3d_text_name = humanml_item['humanml3d_name']
            if start_frame >= len(data_item['joint']):
                print(f'{fpath} error! start frame is larger than the length of the motion')
                continue
            if end_frame > len(data_item['joint']) + 3:
                print(f'Note: {fpath}! end frame: {end_frame}, len_motion: {len(data_item["joint"])}')
            joint = data_item['joint'][start_frame:end_frame]
            joint_25fps = _resample(joint, target_fps)

            pose = data_item['pose'][start_frame:end_frame]
            pose_25fps = math.axis_angle_to_rotation_matrix(_resample(pose, target_fps).contiguous()).view(-1, 24, 3, 3)

            tran = data_item['tran'][start_frame:end_frame]
            tran_25fps = _resample(tran, target_fps)

            vacc = data_item['vacc'][start_frame:end_frame]
            vrot = data_item['vrot'][start_frame:end_frame]
            vacc_25fps = smooth_avg(_resample(vacc, target_fps), s=5)
            vrot_25fps = _resample(vrot, target_fps)

            length = len(joint_25fps)
            name = data_item['name']

            assert data_item['joint'].shape[0] == data_item['pose'].shape[0]
            shape = data_item['shape']


            # save the data
            out_data = {
                "joint_original": joint,
                'joint': joint_25fps,
                "pose_original": pose,
                'pose': pose_25fps,
                "shape": shape,
                "tran_original": tran,
                'tran': tran_25fps,
                "acc_original": vacc,
                'acc': vacc_25fps,
                "ori_original": vrot,
                'ori': vrot_25fps,
                'length_original': length,
                'name': name,
                'humanml3d': {'start_frame_20fps': humanml_item['start_frame'],
                              'start_frame_60fps': start_frame,
                              'end_frame_20fps': humanml_item['end_frame'],
                              'end_frame_60fps': end_frame,
                              'humanml3d_name': humanml3d_text_name}
            }
            out_data_all.append(out_data)
    # save the data
    torch.save(out_data_all, os.path.join(out_motion_dir, motion_name))


