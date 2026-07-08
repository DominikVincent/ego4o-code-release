"""Nymeria dataset (mmpose side) consuming precomputed 30 fps HumanML3D features.

Twin of llava/llava/ego4o/dataset/nymeria_hml_dataset.py for the VQ-VAE /
encoder training in EgoOmniMocap. Reads the jsonl + feature arrays built by
llava/scripts/ego4o/nymeria_hml/ and slices one atomic segment per item.

GT joints for the MPJPE evaluators are recovered from the *sliced raw
features* via recover_from_ric, so prediction and GT share the same
slice-canonical anchor (root at origin, identity heading at slice start) —
equivalent to the original pipeline's InitAlignIMUMotion + feature GT joints.
"""
import json
import os

import numpy as np
import torch
from mmengine.dataset.base_dataset import Compose
from torch.utils.data import Dataset

from mmpose.registry import DATASETS
from mmpose.datasets.datasets.ego_omni_mocap.humanml_utils.motion_process import recover_from_ric

DEFAULT_DATASET_DIR = '/local/home/dhollidt/data/ego4o_nymeria'


@DATASETS.register_module()
class NymeriaHMLDataset(Dataset):
    def __init__(self,
                 dataset_dir=DEFAULT_DATASET_DIR,
                 dataset_json_path=None,
                 split="train",
                 seq_len=148,
                 pipeline=None,
                 data_range=None,
                 test_data_len=None,
                 ):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.dataset_json_path = dataset_json_path
        self.train = split
        self.seq_len = seq_len
        self.data_range = data_range

        self.data = self.load_data()
        if test_data_len is not None:
            self.data = self.data[:test_data_len]

        self.pipeline = Compose(pipeline)
        self.metainfo = {'split': split, 'seq_len': seq_len}

    def full_init(self):
        pass

    def load_data(self):
        if self.dataset_json_path is None:
            if self.train in ("train", "val", "test"):
                data_file = os.path.join(self.dataset_dir, f'ego4o_image_motion_{self.train}.jsonl')
            else:
                raise ValueError("Invalid split")
        else:
            data_file = self.dataset_json_path

        with open(data_file, "r") as f:
            lines = f.readlines()

        if self.data_range is not None:
            sample_margin = len(lines) // self.data_range
            lines = lines[::sample_margin]

        result_data = []
        for line in lines:
            item = json.loads(line)
            result_data.append({
                'motion_file': item['motion_file'],
                'motion_id': item['id'],
                'hml_item': item['hml_item'],
                'start_frame': item['start_frame'],
                'end_frame': item['end_frame'],
                'img_path': item.get('image'),
                'text': item['conversations'][1]['value'],
            })
        return result_data

    def __getitem__(self, idx):
        data_item = dict(self.data[idx])

        vec_path = os.path.join(self.dataset_dir, data_item['motion_file'])
        vec = np.load(vec_path, mmap_mode='r')
        motion = np.array(vec[data_item['start_frame']:data_item['end_frame']], dtype=np.float32)
        data_item['motion_hml'] = motion

        # GT joints in the slice-canonical frame (matches recovered predictions)
        joints = recover_from_ric(torch.from_numpy(motion).unsqueeze(0).float(), 22)
        data_item['init_aligned_global_smpl_joints'] = joints.squeeze(0).numpy()

        data_item = self.pipeline(data_item)
        return data_item

    def __len__(self):
        return len(self.data)
