import torch
import pickle
import os

# --- local paths (patched from original Windows/UNC paths) ---
IMUPOSER_ROOT = '/home/dominik/Documents/ego4o_data/IMUPoser'
AMASS_W_NAME = os.path.join(IMUPOSER_ROOT, 'data/processed_imuposer/AMASS_w_name')
PATH_DICT_PKL = '/home/dominik/Documents/ego4o_data/path_dict.pkl'
OUT_DIR = '/home/dominik/Documents/ego4o_data/amass_data_dict'
os.makedirs(OUT_DIR, exist_ok=True)

root_dir = AMASS_W_NAME
amass_seq_name_list = sorted(os.listdir(root_dir))
print(amass_seq_name_list)

with open(PATH_DICT_PKL, 'rb') as f:
    path_dict = pickle.load(f)

for amass_seq_name in amass_seq_name_list:
    print('processing {}'.format(amass_seq_name))
    seq_dir = os.path.join(AMASS_W_NAME, amass_seq_name)
    name_data = torch.load(os.path.join(seq_dir, 'name.pt'))
    joint_data = torch.load(os.path.join(seq_dir, 'joint.pt'))
    length_data = torch.load(os.path.join(seq_dir, 'length.pt'))
    pose_data = torch.load(os.path.join(seq_dir, 'pose.pt'))
    shape_data = torch.load(os.path.join(seq_dir, 'shape.pt'))
    tran_data = torch.load(os.path.join(seq_dir, 'tran.pt'))
    vacc_data = torch.load(os.path.join(seq_dir, 'vacc.pt'))
    vrot_data = torch.load(os.path.join(seq_dir, 'vrot.pt'))

    data_dict_list = []

    for i, name in enumerate(name_data):
        name_l = name.split('/')[-3:]
        imuposer_name = '/'.join(name_l)
        if imuposer_name in path_dict.keys():
            assert len(joint_data[i]) == len(pose_data[i])
            data_dict = {
                'name': imuposer_name,
                'joint': joint_data[i],
                'length': len(joint_data[i]),  # discard the length information here
                'pose': pose_data[i],
                'shape': shape_data[i],
                'tran': tran_data[i],
                'vacc': vacc_data[i],
                'vrot': vrot_data[i],
                'humanml3d': path_dict[imuposer_name]
            }
            data_dict_list.append(data_dict)
    save_path = os.path.join(OUT_DIR, '{}.pt'.format(amass_seq_name))
    torch.save(data_dict_list, save_path)
    print(f'  saved {len(data_dict_list)} sequences -> {save_path}')
