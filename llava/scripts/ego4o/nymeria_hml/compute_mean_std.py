"""Compute 263-dim Mean/Std over the 30 fps Nymeria features.

Replicates SemanticNymeria/scripts/convert_nymeria_dataset/cal_mean_variance.py
exactly: statistics over ALL items in the feature dir (os.listdir convention),
Std group-averaged per HumanML3D feature block. Uses streaming sums instead of
one giant concatenate (same result, float64 accumulation).

Outputs to the dataset dir:
  Mean.npy / Std.npy                    (263,) float32   (HumanML3D convention)
  info_motion_mean.pt / info_motion_std.pt  torch float32  (what ego4o's
      NormalizeHMLMotion / AgrolMPJPE torch.load)

Run: conda run -n ego4o python compute_mean_std.py
"""
import os
from os.path import join as pjoin

import numpy as np
import torch

OUT_DIR = '/local/home/dhollidt/data/ego4o_nymeria'
DATA_DIR = pjoin(OUT_DIR, 'new_joint_vecs_30fps')
JOINTS_NUM = 22


def mean_variance_streaming(data_dir, joints_num):
    file_list = sorted(os.listdir(data_dir))
    n = 0
    s1 = np.zeros(263, dtype=np.float64)
    s2 = np.zeros(263, dtype=np.float64)
    for i, file in enumerate(file_list):
        data = np.load(pjoin(data_dir, file)).astype(np.float64)
        if np.isnan(data).any():
            print(f'nan in {file}, skipping (mirrors cal_mean_variance)')
            continue
        n += len(data)
        s1 += data.sum(axis=0)
        s2 += (data ** 2).sum(axis=0)
        if (i + 1) % 100 == 0:
            print(f'{i + 1}/{len(file_list)} files, {n} frames')

    Mean = s1 / n
    # population std (ddof=0), same as data.std(axis=0) in the original
    Std = np.sqrt(np.maximum(s2 / n - Mean ** 2, 0.0))

    # --- group averaging, verbatim block structure from cal_mean_variance.py ---
    Std[0:1] = Std[0:1].mean() / 1.0
    Std[1:3] = Std[1:3].mean() / 1.0
    Std[3:4] = Std[3:4].mean() / 1.0
    Std[4: 4 + (joints_num - 1) * 3] = Std[4: 4 + (joints_num - 1) * 3].mean() / 1.0
    Std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = \
        Std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9].mean() / 1.0
    Std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = \
        Std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3].mean() / 1.0
    Std[4 + (joints_num - 1) * 9 + joints_num * 3:] = \
        Std[4 + (joints_num - 1) * 9 + joints_num * 3:].mean() / 1.0

    assert 8 + (joints_num - 1) * 9 + joints_num * 3 == Std.shape[-1]
    return Mean, Std, n, len(file_list)


def main():
    Mean, Std, n_frames, n_files = mean_variance_streaming(DATA_DIR, JOINTS_NUM)
    print(f'{n_files} files, {n_frames} frames')
    print('Mean[:8] ', Mean[:8])
    print('Std[:8]  ', Std[:8])
    assert (Std > 0).all(), 'zero std dims!'

    np.save(pjoin(OUT_DIR, 'Mean.npy'), Mean.astype(np.float32))
    np.save(pjoin(OUT_DIR, 'Std.npy'), Std.astype(np.float32))
    # NOTE: numpy arrays (not torch tensors) inside the .pt — NormalizeHMLMotion
    # computes `numpy_motion - torch.load(...)`, which only broadcasts if the
    # loaded stats are numpy (the TLControl release files are numpy too).
    torch.save(Mean.astype(np.float32), pjoin(OUT_DIR, 'info_motion_mean.pt'))
    torch.save(Std.astype(np.float32), pjoin(OUT_DIR, 'info_motion_std.pt'))
    print(f'saved Mean/Std (.npy + .pt) to {OUT_DIR}')


if __name__ == '__main__':
    main()
