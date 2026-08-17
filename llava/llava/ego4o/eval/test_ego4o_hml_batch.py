"""Batched motion-understanding eval for the GT-motion Ego4o variant.

Adapted from test_ego4o_image_imu_batch.py:
- NymeriaHMLDataset (precomputed 30 fps HML features) instead of NymeriaDataset
- no IMU tensors anywhere -> model.generate takes the encode_motion path
- supports LoRA checkpoints (--model_base <pretrain dir> --model_path <lora dir>):
  base weights + non_lora_trainables.bin (E_I/E_M) + peft adapter, merged
- local default paths, configurable split/data_range

Two prompting modes:
- default: one shared --query for every sample (the original behaviour; keeps the
  atomic-only model evaluable exactly as before)
- --per_sample_prompt: one fixed question per caption type, looked up in
  llava.ego4o.constants.EVAL_QUESTION_BY_TYPE. Required for the 4-category
  dataset, where hands/arms, legs/feet and body posture are annotated on the SAME
  windows, so the question is the only thing distinguishing the three targets.
  Ragged prompts are left-padded and an explicit attention_mask is passed to
  generate().

Usage (see scripts/ego4o/hml/stage4_eval.sh):
  python -m llava.ego4o.eval.test_ego4o_hml_batch \
      --model_path checkpoints/ego4o_hml_finetune_lora \
      --model_base checkpoints/ego4o_hml_pretrain \
      --split test [--data_range 100] [--per_sample_prompt]
"""
import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Sequence, Optional

import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer

from llava.conversation import conv_templates
from llava.constants import IGNORE_INDEX
from llava.ego4o.constants import EVAL_QUESTION_BY_TYPE
from llava.ego4o.dataset.nymeria_hml_dataset import NymeriaHMLDataset
from llava.ego4o.model.ego4o import Ego4oForCausalLM, Ego4oConfig
from llava.ego4o.train.train_ego4o_preprocess import tokenizer_image_motion_token

import evaluate


def load_model(args):
    """Plain checkpoint, or LoRA checkpoint merged onto --model_base."""
    is_lora = os.path.exists(os.path.join(args.model_path, 'adapter_model.safetensors')) or \
              os.path.exists(os.path.join(args.model_path, 'adapter_model.bin'))
    if is_lora:
        assert args.model_base is not None, '--model_base (the pretrain dir) is required for LoRA checkpoints'
        tokenizer = AutoTokenizer.from_pretrained(args.model_base, use_fast=False)
        # intermediate checkpoint-N dirs carry no config.json (HF writes it only
        # to the run root at the end) -> fall back to the run root, then the base
        for cfg_dir in (args.model_path, os.path.dirname(args.model_path.rstrip('/')), args.model_base):
            if os.path.exists(os.path.join(cfg_dir, 'config.json')):
                break
        print(f'Loading Ego4oConfig from {cfg_dir}', flush=True)
        lora_cfg = Ego4oConfig.from_pretrained(cfg_dir)
        print(f'Loading base model from {args.model_base}', flush=True)
        model = Ego4oForCausalLM.from_pretrained(
            args.model_base,
            config=lora_cfg,
            attn_implementation=args.attn_implementation,
            torch_dtype=torch.float16,
        )
        non_lora_path = os.path.join(args.model_path, 'non_lora_trainables.bin')
        if os.path.exists(non_lora_path):
            print(f'Loading non-LoRA trainables (E_I/E_M) from {non_lora_path}', flush=True)
            non_lora = torch.load(non_lora_path, map_location='cpu')
            non_lora = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora.items()}
            if any(k.startswith('model.model.') for k in non_lora):
                non_lora = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora.items()}
            missing, unexpected = model.load_state_dict(non_lora, strict=False)
            assert not unexpected, f'unexpected non-lora keys: {unexpected[:5]}'
        print(f'Loading + merging LoRA adapter from {args.model_path}', flush=True)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.model_path)
        model = model.merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
        model = Ego4oForCausalLM.from_pretrained(
            args.model_path,
            attn_implementation=args.attn_implementation,
            torch_dtype=torch.float16,
        )
    return tokenizer, model.eval().cuda()


@dataclass
class DataArgsShim:
    is_multimodal: bool = True
    image_aspect_ratio: str = 'pad'
    mm_use_im_start_end: bool = False
    image_processor: object = None


class DataCollatorForSupervisedDataset(object):
    """Collate examples for evaluation (GT-motion: no IMU keys)."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids[:, :self.tokenizer.model_max_length],
            labels=labels[:, :self.tokenizer.model_max_length],
        )

        images = [instance['image'] for instance in instances]
        batch['images'] = torch.stack(images)
        batch['motion_hml'] = torch.stack([instance['motion_hml'] for instance in instances])

        batch['conversations'] = [instance['conversations'] for instance in instances]
        batch['img_path'] = [instance.get('img_path') for instance in instances]
        batch['motion_id'] = [instance['motion_id'] for instance in instances]
        batch['motion_file'] = [instance['motion_file'] for instance in instances]
        # MotionGPT3 join key + caption category (None for pre-4-category jsonl)
        batch['fname'] = [instance.get('fname') for instance in instances]
        batch['caption_type'] = [instance.get('caption_type') for instance in instances]
        batch['gt_texts'] = [instance.get('gt_texts') for instance in instances]
        return batch


def build_prompt(query, tokenizer):
    """Wrap one question in the v1 conversation template and tokenize it.

    Returns a ragged 1-D LongTensor carrying the IMAGE_TOKEN_INDEX (-200) /
    MOTION_TOKEN_INDEX (-300) sentinels; those negative ids are why the batch
    cannot be padded with `tokenizer.pad(...)`.
    """
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    return tokenizer_image_motion_token(conv.get_prompt(), tokenizer, return_tensors="pt")


def left_pad(seqs, pad_id):
    """Left-pad ragged token sequences and return (input_ids, attention_mask).

    Left padding is required for generation: `prepare_inputs_labels_for_motion`
    re-pads the spliced embeddings on `config.tokenizer_padding_side`, and with
    the training default ('right') every sample but the longest would continue
    generating from pad embeddings.
    """
    max_len = max(s.numel() for s in seqs)
    input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    for i, s in enumerate(seqs):
        input_ids[i, max_len - s.numel():] = s
        attention_mask[i, max_len - s.numel():] = True
    return input_ids, attention_mask


def eval_model(args):
    tokenizer, model = load_model(args)

    if args.per_sample_prompt:
        # See left_pad(): generation with a padded batch needs left padding, and
        # the value stored in the checkpoint is the *training* one ('right').
        model.config.tokenizer_padding_side = 'left'

    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.float16, device="cuda")

    data_args = DataArgsShim()
    data_args.image_processor = vision_tower.image_processor

    test_dataset = NymeriaHMLDataset(tokenizer=tokenizer,
                                     data_args=data_args,
                                     dataset_dir=args.dataset_dir,
                                     split=args.split,
                                     data_range=args.data_range,
                                     always_with_image=True,
                                     always_with_motion=True,
                                     )
    save_dir = os.path.join(args.save_path,
                            f"test_nymeria_hml_{args.split}_" + datetime.now().strftime("%m_%d_%H_%M_%S"))
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

    save_dict = []

    used_queries = {}

    for data_item in tqdm(dataloader):
        images_for_llm = data_item['images'].cuda().to(torch.float16)
        image_sizes = [x.size for x in images_for_llm]
        motion_hml = data_item['motion_hml'].cuda().to(torch.float16)
        batch_size = images_for_llm.size(0)

        attention_mask = None
        if args.per_sample_prompt:
            # One fixed prompt per caption category. The three body-part
            # categories are annotated on the SAME windows, so a single shared
            # query would ask the model an ambiguous question.
            queries = []
            for i in range(batch_size):
                # one fixed question per caption type; falls back to --query for
                # a jsonl without caption_type (the pre-4-category dataset)
                q = EVAL_QUESTION_BY_TYPE.get(data_item['caption_type'][i], args.query)
                used_queries[data_item['caption_type'][i]] = q
                queries.append(q)
            seqs = [build_prompt(q, tokenizer) for q in queries]
            input_ids, attention_mask = left_pad(seqs, tokenizer.pad_token_id)
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        else:
            used_queries['*'] = args.query
            input_ids = build_prompt(args.query, tokenizer).unsqueeze(0).cuda()
            input_ids = torch.repeat_interleave(input_ids, batch_size, dim=0)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=images_for_llm,
                image_sizes=image_sizes,
                motion_hml=motion_hml,
                do_sample=False,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        conversation_list = data_item['conversations']
        gt_text = [conversation[1]['value'] for conversation in conversation_list]
        pred_text_list.extend(outputs)
        gt_text_list.extend(gt_text)

        for i in range(len(outputs)):
            save_dict.append({
                "pred_text": outputs[i],
                "gt_text": gt_text[i],
                'gt_texts': data_item['gt_texts'][i],
                'motion_id': data_item['motion_id'][i],
                # MotionGPT3 join key -- the exporter reads it straight from here
                'fname': data_item['fname'][i],
                'caption_type': data_item['caption_type'][i],
                'img_path': data_item['img_path'][i],
                'motion_file': data_item['motion_file'][i],
            })

    with open(os.path.join(save_dir, "pred_text_list.json"), "w") as f:
        json.dump(pred_text_list, f)
    with open(os.path.join(save_dir, "gt_text_list.json"), "w") as f:
        json.dump(gt_text_list, f)
    with open(os.path.join(save_dir, "result.json"), "w") as f:
        json.dump(save_dict, f)
    with open(os.path.join(save_dir, "run_config.json"), "w") as f:
        json.dump({"args": vars(args), "queries_used": used_queries}, f, indent=1)

    def text_metrics(preds, refs):
        out = {}
        out['bleu'] = bleu_metric.compute(predictions=preds, references=refs)
        bert_results = bert_metric.compute(predictions=preds, references=refs, lang='en',
                                           verbose=False, idf=True, rescale_with_baseline=True)
        out['bertscore_f1'] = float(np.mean(bert_results['f1']))
        out['rouge'] = {k: float(v) for k, v in
                        rougel_metric.compute(predictions=preds, references=refs).items()}
        out['count'] = len(preds)
        return out

    metrics = text_metrics(pred_text_list, gt_text_list)
    print("BLEU score: ", metrics['bleu'])
    print("BERT score: ", metrics['bertscore_f1'])
    print("ROUGE score: ", metrics['rouge'])

    # Per-category breakdown: the four categories ask for very different captions
    # (step-by-step actions vs. limb motion vs. posture), so one aggregate number
    # over the mixed set is hard to interpret.
    by_category = {}
    for rec in save_dict:
        by_category.setdefault(rec['caption_type'] or 'NA', []).append(rec)
    if len(by_category) > 1:
        metrics['per_category'] = {}
        for category, recs in sorted(by_category.items()):
            metrics['per_category'][category] = text_metrics(
                [r['pred_text'] for r in recs], [r['gt_text'] for r in recs])
            print(f"[{category}] n={len(recs)} "
                  f"BLEU={metrics['per_category'][category]['bleu']['bleu']:.4f} "
                  f"BERT={metrics['per_category'][category]['bertscore_f1']:.4f}")

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print("results saved to", save_dir)


if __name__ == "__main__":
    torch.manual_seed(3407)
    np.random.seed(3407)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="finetuned checkpoint dir (LoRA dir or full model)")
    parser.add_argument("--model_base", type=str, default=None,
                        help="pretrain dir; required when model_path is a LoRA dir")
    parser.add_argument("--dataset_dir", type=str, default='/local/home/dhollidt/data/ego4o_nymeria')
    parser.add_argument("--split", type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument("--data_range", type=int, default=None,
                        help="subsample: keep every len/N-th item (NymeriaHMLDataset semantics)")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--query", type=str,
                        default="<image>\n<motion>\nCan you describe the motion of the person?",
                        help="single prompt used for every sample; also the fallback for "
                             "samples whose caption_type is unknown under --per_sample_prompt")
    parser.add_argument("--per_sample_prompt", action="store_true",
                        help="prompt each sample with the fixed question for its "
                             "caption_type (llava.ego4o.constants.EVAL_QUESTION_BY_TYPE) "
                             "instead of one shared --query. Required for the "
                             "4-category dataset, where the three body-part categories "
                             "share windows and only the question distinguishes them.")
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--attn_implementation", type=str, default='flash_attention_2')
    parser.add_argument("--save_path", type=str,
                        default='/local/home/dhollidt/repos/ego4o-code-release/llava/eval_out')
    args = parser.parse_args()

    eval_model(args)
