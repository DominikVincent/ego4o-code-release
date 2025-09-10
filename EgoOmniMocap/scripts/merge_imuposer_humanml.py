import torch
import pickle
import os

root_dir = r'\\winfs-inf\CT\EgoMocap\work\IMUPoser\data\processed_imuposer\AMASS_w_name'
amass_seq_name_list = sorted(os.listdir(root_dir))
print(amass_seq_name_list)

path_dict_save_path = r'Z:\EgoMocap\work\EgoOmniMocap\scripts\path_dict.pkl'
with open(path_dict_save_path, 'rb') as f:
    path_dict = pickle.load(f)

# load the cmu data for test
for amass_seq_name in amass_seq_name_list:
    print('processing {}'.format(amass_seq_name))
    name_path = r'Z:\EgoMocap\work\IMUPoser\data\processed_imuposer\AMASS_w_name\{}\name.pt'.format(amass_seq_name)
    joint_path = r'Z:\EgoMocap\work\IMUPoser\data\processed_imuposer\AMASS_w_name\{}\joint.pt'.format(amass_seq_name)
    length_path = r'Z:\EgoMocap\work\IMUPoser\data\processed_imuposer\AMASS_w_name\{}\length.pt'.format(amass_seq_name)
    pose_path = r'Z:\EgoMocap\work\IMUPoser\data\processed_imuposer\AMASS_w_name\{}\pose.pt'.format(amass_seq_name)
    shape_path = r'Z:\EgoMocap\work\IMUPoser\data\processed_imuposer\AMASS_w_name\{}\shape.pt'.format(amass_seq_name)
    tran_path = r'Z:\EgoMocap\work\IMUPoser\data\processed_imuposer\AMASS_w_name\{}\tran.pt'.format(amass_seq_name)
    vacc_path = r'Z:\EgoMocap\work\IMUPoser\data\processed_imuposer\AMASS_w_name\{}\vacc.pt'.format(amass_seq_name)
    vrot_path = r'Z:\EgoMocap\work\IMUPoser\data\processed_imuposer\AMASS_w_name\{}\vrot.pt'.format(amass_seq_name)
    name_data = torch.load(name_path)
    joint_data = torch.load(joint_path)
    length_data = torch.load(length_path)
    pose_data = torch.load(pose_path)
    shape_data = torch.load(shape_path)
    tran_data = torch.load(tran_path)
    vacc_data = torch.load(vacc_path)
    vrot_data = torch.load(vrot_path)

    data_dict_list = []

    for i, name in enumerate(name_data):
        name_l = name.split('/')[-3:]
        imuposer_name = '/'.join(name_l)
        if imuposer_name in path_dict.keys():
            print(imuposer_name)
            # assert length_data[i] == len(joint_data[i])
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
    # save
    # print('not saving!!!!')
    save_path = r'Z:\EgoMocap\work\EgoOmniMocap\scripts\amass_data_dict\{}.pt'.format(amass_seq_name)
    torch.save(data_dict_list, save_path)