"""Nymeria dataset for Ego4o that consumes precomputed HumanML3D features.

Replaces NymeriaDataset's raw-XSens path for the GT-motion training variant
(text & GT motion & image -> text, no IMU). Data comes from the builder in
`scripts/ego4o/nymeria_hml/` (see build_ego4o_jsonl.py for the jsonl schema):
whole-recording 263-dim feature arrays at 30 fps are sliced per atomic segment
at load time (the HML representation is root-relative per frame, so slices
need no renormalization).

Compared to NymeriaDataset: no IMU tensors, no `img_for_imu`, no combo
masking; the pipeline is only normalize -> pad -> reshape. Text/image handling
matches NymeriaDataset.__getitem__.
"""
import json
import os
from copy import deepcopy

import numpy as np
import torch
import transformers
from mmengine.dataset.base_dataset import Compose
from torch.utils.data import Dataset
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.ego4o.constants import DEFAULT_MOTION_TOKEN
from llava.ego4o.train.train_ego4o_preprocess import preprocess_multimodal, preprocess
import llava.ego4o.dataset.transforms  # noqa: F401 — registers the mmengine transforms

DEFAULT_DATASET_DIR = '/local/home/dhollidt/data/ego4o_nymeria'


class NymeriaHMLDataset(Dataset):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args,
                 dataset_json_path=None,
                 dataset_dir=DEFAULT_DATASET_DIR,
                 seq_len=148,
                 split="train",
                 pipeline=None,
                 data_range=None,
                 always_with_image=False,
                 always_with_motion=False,
                 hml_mean_path=None,
                 hml_std_path=None,
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.data_range = data_range
        self.always_with_image = always_with_image
        self.always_with_motion = always_with_motion

        self.dataset_dir = dataset_dir
        self.dataset_json_path = dataset_json_path
        self.train = split
        self.seq_len = seq_len

        self.data = self.load_data()

        if hml_mean_path is None:
            hml_mean_path = os.path.join(dataset_dir, 'info_motion_mean.pt')
        if hml_std_path is None:
            hml_std_path = os.path.join(dataset_dir, 'info_motion_std.pt')

        if pipeline is None:
            pipeline = [
                dict(type='NormalizeHMLMotion', hml_motion_name='motion_hml',
                     hml_mean_path=hml_mean_path,
                     hml_std_path=hml_std_path),
                dict(type='PadMotion', seq_len=seq_len,
                     pad_name_list=('motion_hml',),
                     resize_input_sequence=True),
                dict(type='ChangeHMLShape', hml_motion_name='motion_hml'),
                dict(type='ToTensor', keys=['motion_hml']),
            ]
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
            if type(self.data_range) is not tuple:
                sample_margin = len(lines) // self.data_range
                lines = lines[::sample_margin]
            else:
                start_data_idx, end_data_idx = self.data_range
                end_data_idx = min(end_data_idx, len(lines))
                lines = lines[start_data_idx: end_data_idx]
                print(f"Warning: only using samples from {start_data_idx} to {end_data_idx} for testing")

        result_data = []
        for line in lines:
            item = json.loads(line)
            conversations = item['conversations']
            assert len(conversations) == 2

            if self.always_with_image:
                if DEFAULT_IMAGE_TOKEN not in conversations[0]['value']:
                    conversations[0]['value'] = conversations[0]['value'].replace(
                        f"{DEFAULT_MOTION_TOKEN}\n",
                        f"{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_MOTION_TOKEN}\n")
            if self.always_with_motion:
                if DEFAULT_MOTION_TOKEN not in conversations[0]['value']:
                    conversations[0]['value'] = conversations[0]['value'].replace(
                        f"{DEFAULT_IMAGE_TOKEN}\n",
                        f"{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_MOTION_TOKEN}\n")

            data_item = {
                'conversations': conversations,
                'motion_file': item['motion_file'],
                'motion_id': item['id'],
                'hml_item': item['hml_item'],
                'start_frame': item['start_frame'],
                'end_frame': item['end_frame'],
            }
            if item.get('image'):
                data_item['img_path'] = item['image']
            result_data.append(data_item)

        return result_data

    def load_motion_slice(self, data_item):
        """Slice the whole-recording 263-dim feature array (unnormalized)."""
        vec_path = os.path.join(self.dataset_dir, data_item['motion_file'])
        vec = np.load(vec_path, mmap_mode='r')
        motion = np.array(vec[data_item['start_frame']:data_item['end_frame']], dtype=np.float32)
        return motion

    @property
    def lengths(self):
        length_list = []
        for sample in self.data:
            img_tokens = 128 if 'img_path' in sample else 0
            motion_tokens = 37 if 'motion_file' in sample else 0
            length_list.append(
                sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens + motion_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.data:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            # only consider if we have motion here
            cur_len = cur_len if 'motion_file' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, idx):
        data_item = deepcopy(self.data[idx])
        data_item['motion_hml'] = self.load_motion_slice(data_item)

        # ----------------------- process the text -----------------------
        # (mirrors NymeriaDataset.__getitem__, minus IMU handling)
        sources = [data_item]

        if 'img_path' in sources[0] and 'motion_file' in sources[0]:
            image_path = sources[0]['img_path']
            processor = self.data_args.image_processor
            image = Image.open(image_path).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('img_path' in data_item or 'motion_file' in data_item)
        )
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                         labels=data_dict["labels"][0])

        # image exists in the data
        if 'img_path' in data_item:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])

        data_item.update(data_dict)

        data_item = self.pipeline(data_item)

        return data_item

    def __len__(self):
        return len(self.data)
