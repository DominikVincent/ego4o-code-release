"""Build path_dict.pkl (the missing glue for the Ego4o AMASS preprocessing).

merge_imuposer_humanml.py expects a pickle mapping
    imuposer_name = last-3 components of the AMASS npz path
      e.g. 'ACCAD/Female1General_c3d/A3 - Swing t2_poses.npz'
  -> list of HumanML3D segments: {'start_frame','end_frame','humanml3d_name'} (20 fps)

That mapping is exactly HumanML3D's index.csv (source_path, start_frame, end_frame, new_name).
We convert the .npy source paths to .npz to match the AMASS filenames stored in name.pt.
"""
import csv
import os
import pickle

INDEX_CSV = '/home/dominik/Documents/repos/HumanML3D/index.csv'
OUT = '/home/dominik/Documents/ego4o_data/path_dict.pkl'

path_dict = {}
n_rows = 0
with open(INDEX_CSV, newline='') as f:
    for row in csv.DictReader(f):
        src = row['source_path']                 # ./pose_data/ACCAD/.../xxx_poses.npy
        parts = src.split('/')
        if len(parts) < 3:
            continue
        imuposer_name = '/'.join(parts[-3:])     # Dataset/Subject/file_poses.npy
        if imuposer_name.endswith('.npy'):
            imuposer_name = imuposer_name[:-4] + '.npz'
        path_dict.setdefault(imuposer_name, []).append({
            'start_frame': int(row['start_frame']),
            'end_frame': int(row['end_frame']),
            'humanml3d_name': os.path.splitext(row['new_name'])[0],
        })
        n_rows += 1

with open(OUT, 'wb') as f:
    pickle.dump(path_dict, f)

print(f'wrote {OUT}: {len(path_dict)} source files, {n_rows} segments')
# sanity peek
k = next(iter(path_dict))
print('example key:', k)
print('example val[0]:', path_dict[k][0])
