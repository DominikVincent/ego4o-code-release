from copy import deepcopy
from os.path import join as pjoin

import numpy as np
import torch
from mmengine.dataset.base_dataset import Compose
from torch.utils.data import Dataset

from mmpose.datasets.datasets.ego_omni_mocap.humanml_utils.t2m_dataset import Text2MotionDatasetV2
from mmpose.datasets.datasets.ego_omni_mocap.humanml_utils.utils.get_opt import get_opt
from mmpose.datasets.datasets.ego_omni_mocap.humanml_utils.utils.word_vectorizer import WordVectorizer
from mmpose.registry import DATASETS


@DATASETS.register_module()
class HMLTrainDataset(Dataset):
    def __init__(self,
                 mode,
                 data_root='/CT/EgoMocap/work/OmniControl',
                 data_opt_path='./dataset/humanml_opt.txt',
                 split="train",
                 traj_with_rot=False,
                 traj_mean_path=None,
                 traj_std_path=None,
                 pipeline=None,
                 dim=3):
        super(HMLTrainDataset, self).__init__()
        self.mode = mode

        self.dataset_name = 't2m'
        self.dataname = 't2m'

        self.data_root = data_root
        self.data_opt_path = data_opt_path
        self.split = split

        self.load_data_list(traj_with_rot=traj_with_rot,
                            traj_mean_path=traj_mean_path,
                            traj_std_path=traj_std_path, dim=dim)
        self.pipeline = Compose(pipeline)

    def load_data_list(self, traj_with_rot,
                       traj_mean_path,
                       traj_std_path, dim):
        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = self.data_root
        dataset_opt_path = pjoin(abs_base_path, self.data_opt_path)
        device = None  # torch.device('cuda:4') # This param is not in use in this context
        opt = get_opt(dataset_opt_path, device)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = './dataset'

        self.opt = opt
        print('Loading dataset %s ...' % opt.dataset_name)

        self.mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
        self.std = np.load(pjoin(opt.data_root, 'Std.npy'))
        self.split_file = pjoin(opt.data_root, f'{self.split}.txt')
        self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
        self.t2m_dataset = Text2MotionDatasetV2(self.opt, self.mean, self.std, self.split_file,
                                                self.w_vectorizer, traj_with_rot=traj_with_rot,
                                                traj_mean_path=traj_mean_path,
                                                traj_std_path=traj_std_path, dim=dim)
        self.num_actions = 1  # dummy placeholder

        assert len(self.t2m_dataset) > 1, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'

    def process_data(self, data_input):
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, tokens, traj_data = data_input
        # data = {
        #     'word_embeddings': word_embeddings,
        #     'pos_one_hots': pos_one_hots,
        #     'caption': caption,
        #     'sent_len': sent_len,
        #     'motion': motion,
        #     'lengths': m_length,
        #     'text': tokens,
        #     'traj_data': traj_data,
        # }
        data = {
            'motion': torch.tensor(data_input[4].T).float().unsqueeze(1),  # [seqlen, J] -> [J, 1, seqlen]
            'text': data_input[2],  # b[0]['caption']
            'sent_len': data_input[3],
            'tokens': data_input[6],
            'lengths': data_input[5],
            'traj_data': data_input[7],
        }
        data = self.pipeline(data)
        return data

    def __getitem__(self, item):
        data_item = deepcopy(self.t2m_dataset.__getitem__(item))

        return self.process_data(data_item)

    def __len__(self):
        return self.t2m_dataset.__len__()
