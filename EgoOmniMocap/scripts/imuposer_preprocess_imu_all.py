r"""
    Preprocess DIP-IMU and TotalCapture test dataset.
    Synthesize AMASS dataset.

"""

# %load_ext autoreload
# %autoreload 2

import torch
import os
import pickle
import numpy as np
from tqdm import tqdm
import glob

from imuposer.config import Config, amass_datasets
from imuposer.smpl.parametricModel import ParametricModel
from imuposer import math

config = Config(project_root_dir="..")

def process_amass(out_dir_name='AMASS_w_name'):
    def _syn_acc(v):
        r"""
        Synthesize accelerations from vertex positions.
        """
        acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
        acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
        return acc

    # left wrist, right wrist, left thigh, right thigh, head, pelvis
    vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
    ji_mask = torch.tensor([18, 19, 1, 2, 15, 0])
    body_model = ParametricModel(config.og_smpl_model_path)

    try:
        processed = [fpath.name for fpath in (config.processed_imu_poser / out_dir_name).iterdir()]
    except:
        processed = []

    for ds_name in amass_datasets:
        if ds_name in processed:
            continue
        data_pose, data_trans, data_beta, length, names = [], [], [], [], []
        print('\rReading', ds_name)
        for npz_fname in tqdm(glob.glob(os.path.join(config.raw_amass_path, ds_name, '*/*_poses.npz'))):
            try:
                cdata = np.load(npz_fname)
            except:
                continue

            framerate = int(cdata['mocap_framerate'])
            if framerate == 120:
                step = 2
            elif framerate == 60 or framerate == 59:
                step = 1
            else:
                continue

            data_pose.extend(cdata['poses'][::step].astype(np.float32))
            data_trans.extend(cdata['trans'][::step].astype(np.float32))
            data_beta.append(cdata['betas'][:10])
            length.append(cdata['poses'][::step].shape[0])
            names.append(npz_fname)

        if len(data_pose) == 0:
            print(f"AMASS dataset, {ds_name} not supported")
            continue

        length = torch.tensor(length, dtype=torch.int)
        shape = torch.tensor(np.asarray(data_beta, np.float32))
        tran = torch.tensor(np.asarray(data_trans, np.float32))
        pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)

        # include the left and right index fingers in the pose
        pose[:, 23] = pose[:, 37]  # right hand
        pose = pose[:, :24].clone()  # only use body + right and left fingers

        # align AMASS global frame with DIP
        amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
        tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
        pose[:, 0] = math.rotation_matrix_to_axis_angle(
            amass_rot.matmul(math.axis_angle_to_rotation_matrix(pose[:, 0])))

        print('Synthesizing IMU accelerations and orientations')
        b = 0
        out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], [], []
        out_name = []
        for i, l in tqdm(list(enumerate(length))):
            if l <= 12:
                b += l
                print('\tdiscard one sequence with length', l)
                continue
            p = math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
            grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)
            out_pose.append(pose[b:b + l].clone())  # N, 24, 3
            out_tran.append(tran[b:b + l].clone())  # N, 3
            out_shape.append(shape[i].clone())  # 10
            out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
            out_vacc.append(_syn_acc(vert[:, vi_mask]))  # N, 6, 3
            out_vrot.append(grot[:, ji_mask])  # N, 6, 3, 3
            out_name.append(names[i])
            b += l

        print('Saving')
        amass_dir = config.processed_imu_poser / out_dir_name
        amass_dir.mkdir(exist_ok=True, parents=True)
        ds_dir = amass_dir / ds_name
        ds_dir.mkdir(exist_ok=True)

        torch.save(out_pose, ds_dir / 'pose.pt')
        torch.save(out_shape, ds_dir / 'shape.pt')
        torch.save(out_tran, ds_dir / 'tran.pt')
        torch.save(out_joint, ds_dir / 'joint.pt')
        torch.save(out_vrot, ds_dir / 'vrot.pt')
        torch.save(out_vacc, ds_dir / 'vacc.pt')
        torch.save(length, ds_dir / 'length.pt')
        torch.save(out_name, ds_dir / 'name.pt')
        print('Synthetic AMASS dataset is saved at', str(ds_dir))

if __name__ == '__main__':
    process_amass()