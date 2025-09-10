from torch.utils.data import Dataset
import codecs as cs
import numpy as np
from tqdm import tqdm
from os.path import join as pjoin
import random
import torch

from mmpose.datasets.datasets.ego_omni_mocap.humanml_utils.motion_process import recover_from_ric, recover_global_limb_rot


class Text2MotionDatasetV2(Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer, traj_with_rot=False,
                 traj_mean_path=None, traj_std_path=None,
                 dim=3):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.traj_with_rot = traj_with_rot
        self.dim = dim
        min_motion_len = 40 if self.opt.dataset_name == 't2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        all_traj = []
        for name in tqdm(id_list):
            motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
            if (len(motion)) < min_motion_len or (len(motion) >= 200):
                continue
            text_data = []
            flag = False

            with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                for line in f.readlines():
                    text_dict = {}
                    line_split = line.strip().split('#')
                    caption = line_split[0]
                    tokens = line_split[1].split(' ')
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])
                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag

                    text_dict['caption'] = caption
                    text_dict['tokens'] = tokens
                    if f_tag == 0.0 and to_tag == 0.0:
                        flag = True
                        text_data.append(text_dict)
                    else:
                        try:
                            n_motion = motion[int(f_tag * 20): int(to_tag * 20)]
                            if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                continue
                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                            while new_name in data_dict:
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                            data_dict[new_name] = {'motion': n_motion,
                                                   'length': len(n_motion),
                                                   'text': [text_dict]}
                            new_name_list.append(new_name)
                            length_list.append(len(n_motion))

                        except:
                            print(line_split)
                            print(line_split[2], line_split[3], f_tag, to_tag, name)
                            # break

            if flag:
                data_dict[name] = {'motion': motion,
                                   'length': len(motion),
                                   'text': text_data}
                new_name_list.append(name)
                length_list.append(len(motion))

        for final_name in new_name_list:
            sample = data_dict[final_name]['motion']  # * std + mean

            if self.traj_with_rot is False:
                # only use the position in the traj
                sample = recover_from_ric(torch.Tensor(sample), 22)
                sample = sample[:, [0, 15, 20, 21, 10, 11], :]
                all_traj.append(sample)
            else:
                # breakpoint()
                # get joint position and joint rotation in the global coordinate system
                joint_locations = recover_from_ric(torch.Tensor(sample), 22)
                joint_global_orientations = recover_global_limb_rot(joint_locations)
                joint_locations = joint_locations[:, [0, 15, 20, 21, 10, 11], :]
                joint_global_orientations = joint_global_orientations[:, [0, 15, 20, 21, 10, 11], :]
                sample = torch.cat((joint_locations, joint_global_orientations), dim=-1)
                all_traj.append(sample)

            data_dict[final_name]['traj'] = sample

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

        assert all_traj[0].shape[-1] == self.dim
        all_traj_tensor = torch.cat(all_traj, dim=0).reshape((-1, self.dim))
        self.traj_std, self.traj_mean = torch.std_mean(all_traj_tensor, dim=0)
        if traj_mean_path is not None and traj_std_path is not None:
            # save the traj mean and traj std
            print('Saving the traj mean and traj std')
            torch.save(self.traj_mean, traj_mean_path)
            torch.save(self.traj_std, traj_std_path)

        self.new_name_list = new_name_list

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data, in_gpu=False):
        # torch.save(self.std, "./evaluate/motion_std.pt")
        # torch.save(self.mean, "./evaluate/motion_mean.pt")
        if in_gpu:
            return data * torch.Tensor(self.std).cuda() + torch.Tensor(self.mean).cuda()
        return data * self.std + self.mean

    def transform_traj(self, data):
        return (data - self.traj_mean) / self.traj_std

    def inv_transform_traj(self, data):
        # torch.save(self.traj_std, "./evaluate/traj_std.pt")
        # torch.save(self.traj_mean, "./evaluate/traj_mean.pt")
        return data * self.traj_std + self.traj_mean


    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)

        motion = motion[idx:idx + m_length]
        if self.traj_with_rot is False:
            cur_traj = recover_from_ric(torch.Tensor(motion), 22)[:, [0, 15, 20, 21, 10, 11], :]
        else:
            joint_locations = recover_from_ric(torch.Tensor(motion), 22)
            joint_global_orientations = recover_global_limb_rot(joint_locations)

            joint_locations = joint_locations[:, [0, 15, 20, 21, 10, 11], :]
            joint_global_orientations = joint_global_orientations[:, [0, 15, 20, 21, 10, 11], :]
            cur_traj = torch.cat((joint_locations, joint_global_orientations), dim=-1)
        traj_data = self.transform_traj(cur_traj)

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)

            traj_data = torch.cat((traj_data, torch.zeros(self.max_motion_length - m_length, 6, self.dim)), dim=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), traj_data