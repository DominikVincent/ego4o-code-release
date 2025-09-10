import argparse
import json
import os
import pdb
import pickle
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Sequence, Optional

import numpy as np
import torch
import transformers
from sympy.core.random import shuffle
from tqdm import tqdm
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

import evaluate


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

        assert 'image' in instances[0]
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


def eval_model(args):
    # Model
    # disable_torch_init()

    # tokenizer, model, image_processor, context_len = load_pretrained_model(
    #     args.model_path, None, None,
    # )
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
                                  split="train",
                                  pipeline=None,
                                  with_text=True,
                                  data_range=None,
                                  always_with_image=True,
                                  always_with_motion=True,
                                  )
    save_dir = os.path.join(args.save_path, "nymeria_train" + datetime.now().strftime("%m_%d_%H_%M_%S"))
    os.makedirs(save_dir, exist_ok=True)

    bleu_metric = evaluate.load("bleu")
    bert_metric = evaluate.load("bertscore")
    rougel_metric = evaluate.load("rouge")
    pred_text_list = []
    gt_text_list = []


    data_collector = DataCollatorForSupervisedDataset()
    data_collector.tokenizer = tokenizer
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=False,
        collate_fn=data_collector,
    )

    # save the dictionary of the list
    save_dict = []

    for data_item in tqdm(dataloader):
        query = "<image>\n<motion>\nCan you describe the motion of the person?"
        # query = "<image>\nCan you describe the motion of the person?"
        # query = "<motion>\nCan you describe the motion of the person?"
        conv_mode = "v1"
        args.conv_mode = conv_mode
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_motion_token(prompt, tokenizer, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        # breakpoint()
        # input_ids = test_dataset[data_idx]['input_ids'].unsqueeze(0).cuda()
        # input_ids = input_ids[:, :50]
        images_for_llm = data_item['images'].cuda().to(torch.float16)
        image_sizes = [x.size for x in images_for_llm]
        img_for_imu = data_item['img_for_imu'].cuda().to(torch.float16)
        init_aligned_imu_acc = data_item['init_aligned_imu_acc'].cuda().to(torch.float16)
        init_aligned_imu_ori = data_item['init_aligned_imu_ori'].cuda().to(torch.float16)
        motion_hml = data_item['motion_hml'].cuda().to(torch.float16)

        batch_size = images_for_llm.size(0)
        input_ids = torch.repeat_interleave(input_ids, batch_size, dim=0)

        # pdb.set_trace()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_for_llm,
                image_sizes=image_sizes,
                motion_hml=motion_hml,
                img_for_imu=img_for_imu,
                init_aligned_imu_acc=init_aligned_imu_acc,
                init_aligned_imu_ori=init_aligned_imu_ori,
                # do_sample=True if args.temperature > 0 else False,
                do_sample=False,
                # temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        # print(output_ids)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        # breakpoint()
        # print(outputs[0])

        # print(conversation)

        # evaluate bleu score
        conversation_list = data_item['conversations']
        gt_text = []
        for conversation in conversation_list:
            gt_text.append(conversation[1]['value'])
        pred_text_list.extend(outputs)
        gt_text_list.extend(gt_text)

        for i in range(len(outputs)):
            save_dict.append({
                "pred_text": outputs[i],
                "gt_text": gt_text[i],
                'motion_id': data_item['motion_id'][i],
                'img_path': data_item['img_path'][i],
                'motion_file': data_item['motion_file'][i],
                # 'motion_hml': data_item['motion_hml'][i].cpu().numpy().tolist(),
            })
        # breakpoint()

    # save the list
    with open(os.path.join(save_dir, "pred_text_list.json"), "w") as f:
        json.dump(pred_text_list, f)
    with open(os.path.join(save_dir, "gt_text_list.json"), "w") as f:
        json.dump(gt_text_list, f)

    # save the dictionary of the list
    # with open(os.path.join(save_dir, "result.pkl"), "wb") as f:
    #     pickle.dump(save_dict, f)

    with open(os.path.join(save_dir, "result.json"), "w") as f:
        json.dump(save_dict, f)

    # calculate the bleu score
    bleu_results = bleu_metric.compute(predictions=pred_text_list, references=gt_text_list)
    print("BLEU score: ", bleu_results)
    # calculate the bert score
    bert_results = bert_metric.compute(predictions=pred_text_list, references=gt_text_list,
                                       lang='en', verbose=False, idf=True, rescale_with_baseline=True, )
    print("BERT score: ", np.mean(bert_results['f1']))
    # calculate the rouge score
    rouge_results = rougel_metric.compute(predictions=pred_text_list, references=gt_text_list)
    print("ROUGE score: ", rouge_results)

    # # save the generated text to the save_path
    # with open(os.path.join(save_dir, "generated_text.txt"), "w") as f:
    #     f.write(outputs)
    # # copy the motion to the save_path
    # motion_path = args.motion_path
    # try:
    #     shutil.copy(motion_path, save_dir)
    # except Exception as e:
    #     print(f"Error occurred while copying the file: {e}")
    #
    # ## save the ground truth text to the save_path
    # gt_text_json = '/scratch/inf0/user/jianwang/nymeria/ego4o_input_json_image_motion.jsonl'
    # with open(gt_text_json, "r") as f:
    #     lines = f.readlines()
    # list_data_dict = [json.loads(line) for line in lines]
    #
    # # pdb.set_trace()
    #
    # for list_data_dict_item in list_data_dict:
    #     if list_data_dict_item['motion_id'][0] == motion_id:
    #         gt_text = list_data_dict_item['conversations']
    #         break
    # with open(os.path.join(save_dir, "ground_truth_text.txt"), "w") as f:
    #     json.dump(gt_text, f)
    #
    # # save query to the save_path
    # with open(os.path.join(save_dir, "query.txt"), "w") as f:
    #     f.write(qs)


if __name__ == "__main__":
    # fix the random seed
    torch.manual_seed(3407)
    np.random.seed(3407)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/motion_to_text_finetune/checkpoint-1500")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--motion_index", type=int, required=False, default=10)
    parser.add_argument("--batch_size", type=int, required=False, default=24)
    parser.add_argument("--image-file", type=str, required=False,
                        default="https://llava-vl.github.io/static/images/view.jpg")
    parser.add_argument("--query", type=str, required=False)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--save_path", type=str, default='/home/jianwang/EgoMocap/work/LLaVA/eval_out')
    args = parser.parse_args()

    eval_model(args)
