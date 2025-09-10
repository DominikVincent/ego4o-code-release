import argparse
import json
import os
import pdb
import pickle
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Sequence, Optional

import torch
import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.modeling_utils import load_sharded_checkpoint

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER, DEFAULT_IMAGE_PATCH_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.ego4o.constants import DEFAULT_MOTION_TOKEN, MOTION_TOKEN_INDEX
from llava.ego4o.dataset.nymeria_dataset import NymeriaDataset
from llava.ego4o.model.ego4o import Ego4oForCausalLM
from llava.ego4o.train.train_ego4o_preprocess import tokenizer_image_motion_token
# from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.constants import IGNORE_INDEX
from PIL import Image

from llava.ego4o.dataset.transforms import *
from transformers import TextStreamer, AutoTokenizer, BitsAndBytesConfig


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto",
                          device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = Ego4oForCausalLM.from_pretrained(
        model_path,
        **kwargs
    )

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    if device_map != 'auto':
        vision_tower.to(device=device_map, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    motion_folder: Optional[str] = field(default=None)
    model_path: str = field(default=None,
                            metadata={"help": "Path to the model checkpoint."})


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        if 'motion_hml' in instances[0]:
            motion_hml = []
            for instance in instances:
                motion_hml.append(instance['motion_hml'])
            if all(x is not None and x.shape == motion_hml[0].shape for x in motion_hml):
                batch['motion_hml'] = torch.stack(motion_hml)
            else:
                batch['motion_hml'] = motion_hml

        if 'img_for_imu' in instances[0]:
            img_for_imu = []
            for instance in instances:
                img_for_imu.append(instance['img_for_imu'])
            if all(x is not None and x.shape == img_for_imu[0].shape for x in img_for_imu):
                batch['img_for_imu'] = torch.stack(img_for_imu)
            else:
                batch['img_for_imu'] = img_for_imu

            imu_acc = []
            for instance in instances:
                imu_acc.append(instance['init_aligned_imu_acc'])
            batch['init_aligned_imu_acc'] = torch.stack(imu_acc)

            imu_ori = []
            for instance in instances:
                imu_ori.append(instance['init_aligned_imu_ori'])
            batch['init_aligned_imu_ori'] = torch.stack(imu_ori)

        batch['conversations'] = [instance['conversations'] for instance in instances]
        batch['img_path'] = [instance['img_path'] for instance in instances]
        batch['motion_id'] = [instance['motion_id'] for instance in instances]
        batch['motion_file'] = [instance['motion_file'] for instance in instances]
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = NymeriaDataset(tokenizer=tokenizer,
                                   data_args=data_args,
                                   dataset_dir='/scratch/inf0/user/jianwang/nymeria',
                                   dataset_json_path=data_args.data_path,
                                   seq_len=150,
                                   min_seq_len=150,
                                   signal_num=6,
                                   tlcontrol_joint_sequence=True,
                                   random_mask=True,
                                   split="train",
                                   pipeline=None,
                                   with_text=True, )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def main(args):
    # Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = Ego4oForCausalLM.from_pretrained(
        # "/CT/EgoMocap/work/LLaVA/checkpoints/ego4o_imu2text_pretrain",
        args.model_path,
        cache_dir=None,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.float16,
    ).eval().cuda()
    #
    # load_sharded_checkpoint(model, folder=args.model_path, strict=True)

    parser = transformers.HfArgumentParser(DataArguments)
    data_args, = parser.parse_args_into_dataclasses()

    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.float16, device="cuda")

    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True
    data_args.mm_use_im_start_end = False

    test_dataset = NymeriaDataset(tokenizer=tokenizer,
                                  data_args=data_args,
                                  dataset_dir='/scratch/inf0/user/jianwang/nymeria',
                                  dataset_json_path=None,
                                  seq_len=150,
                                  min_seq_len=150,
                                  signal_num=6,
                                  tlcontrol_joint_sequence=True,
                                  random_mask=True,
                                  split="test",
                                  pipeline=None,
                                  with_text=True,
                                  data_range=1000,
                                  always_with_image=True,
                                  always_with_motion=True,
                                  )

    data_idx = 620
    conversation = test_dataset[data_idx]['conversations']

    # query = "<image>\n<motion>\nCan you describe the motion of the person?"
    conv_mode = "v1"
    args.conv_mode = conv_mode
    conv = conv_templates[args.conv_mode].copy()
    # conv.append_message(conv.roles[0], query)
    # conv.append_message(conv.roles[1], None)
    # prompt = conv.get_prompt()
    #
    # input_ids = (
    #     tokenizer_image_motion_token(prompt, tokenizer, return_tensors="pt")
    #     .unsqueeze(0)
    #     .cuda()
    # )

    # breakpoint()
    # input_ids = test_dataset[data_idx]['input_ids'].unsqueeze(0).cuda()
    # input_ids = input_ids[:, :50]
    images_for_llm = test_dataset[data_idx]['image'].unsqueeze(0).cuda().to(torch.float16)
    image_sizes = [x.size for x in images_for_llm]
    img_for_imu = test_dataset[data_idx]['img_for_imu'].unsqueeze(0).cuda().to(torch.float16)
    init_aligned_imu_acc = test_dataset[data_idx]['init_aligned_imu_acc'].unsqueeze(0).cuda().to(torch.float16)
    init_aligned_imu_ori = test_dataset[data_idx]['init_aligned_imu_ori'].unsqueeze(0).cuda().to(torch.float16)
    motion_hml = test_dataset[data_idx]['motion_hml'].unsqueeze(0).cuda().to(torch.float16)

    image_path = test_dataset[data_idx]['img_path']
    gt_conversation = test_dataset[data_idx]['conversations']
    motion_id = test_dataset[data_idx]['motion_id']

    print("Image path: ", image_path)
    print("Motion ID: ", motion_id)
    print("Ground truth conversation: ", gt_conversation)

    # pdb.set_trace()

    # with torch.inference_mode():
    #     output_ids = model.generate(
    #         input_ids,
    #         images=images_for_llm,
    #         image_sizes=image_sizes,
    #         motion_hml=motion_hml,
    #         img_for_imu=img_for_imu,
    #         init_aligned_imu_acc=init_aligned_imu_acc,
    #         init_aligned_imu_ori=init_aligned_imu_ori,
    #         do_sample=False,
    #         temperature=args.temperature,
    #         num_beams=args.num_beams,
    #         max_new_tokens=args.max_new_tokens,
    #         use_cache=True,
    #     )

    # print(output_ids)
    #
    # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    # print(outputs)

    while True:
        try:
            inp = input("USER: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print("ASSISTANT: ", end="")

        # if image is not None:
        #     # first message
        #     if model.config.mm_use_im_start_end:
        #         inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        #     else:
        #         inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        #     image = None

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        input_ids = (
            tokenizer_image_motion_token(prompt, tokenizer, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_for_llm,
                image_sizes=image_sizes,
                motion_hml=motion_hml,
                img_for_imu=img_for_imu,
                init_aligned_imu_acc=init_aligned_imu_acc,
                init_aligned_imu_ori=init_aligned_imu_ori,
                do_sample=False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True
            )

        outputs = tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/motion_to_text_finetune")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--motion_path", type=str, required=False)
    parser.add_argument("--image-file", type=str, required=False,
                        default="https://llava-vl.github.io/static/images/view.jpg")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sep", type=str, default=",")
    args = parser.parse_args()
    main(args)
