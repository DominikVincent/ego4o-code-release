"""Build the Ego4o training jsonl from processed_nymeria (READ-ONLY sources).

Cuts every recording into its atomic-action segments ("Describe my atomic
actions" lines of `texts/{item}.txt`), maps them to 30 fps feature frames and
to the nearest egocentric frame, and emits ego4o-format jsonl per split:

  {out_dir}/ego4o_image_motion_{train,val,test}.jsonl   finetune (mixed questions)
  {out_dir}/ego4o_motion_text_{train,val,test}.jsonl    pretrain (motion->text questions, no image)
  {out_dir}/build_report.json

jsonl line schema (superset of the original ego4o schema):
  id            "{k}_{seq_name}"   k = atomic index within the recording
  hml_item      "000001"           item id in data_order
  motion_file   "new_joint_vecs_30fps/000001.npy"
  start_frame / end_frame          30 fps feature-frame slice
  image         abs path | null
  fps           30
  motion_id     [id]
  conversations [{from: human, value: question}, {from: gpt, value: caption}]

Question sampling mirrors convert_nymeria_data_smooth_imus.py: one random
question per item at build time, from IMAGE_MOTION+IMAGE+MOTION lists
(restricted to MOTION list when the sample has no image). Seeded, rerunnable.

Run: conda run -n ego4o python build_ego4o_jsonl.py
"""
import argparse
import importlib.util
import json
import os
from os.path import join as pjoin

import numpy as np

PROCESSED_NYMERIA = '/local/home/dhollidt/repos/HumanML3DFork/processed_nymeria_scene_sub_split'
FRAMES_DIR = '/local/home/dhollidt/data/nymeria_frames'
OUT_DIR = '/local/home/dhollidt/data/ego4o_nymeria'
REPO = os.path.abspath(pjoin(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
CONSTANTS_PY = pjoin(REPO, 'llava', 'ego4o', 'constants.py')

FPS = 30
MIN_FRAMES = 150           # mirrors ego4o min_seq_len=150 (5 s @ 30 fps)
ATOMIC_TYPE = 'Describe my atomic actions'
SEED = 20260707


def load_constants():
    """Load the ego4o question lists without importing the llava package.

    ego4o/constants.py does `from llava.constants import DEFAULT_IMAGE_TOKEN`
    ("<image>", llava/constants.py:9); stub that module so we don't need the
    llava package (and its transformers dependency) in this env.
    """
    import sys
    import types
    if 'llava' not in sys.modules:
        llava_pkg = types.ModuleType('llava')
        llava_constants = types.ModuleType('llava.constants')
        llava_constants.DEFAULT_IMAGE_TOKEN = '<image>'
        llava_pkg.constants = llava_constants
        sys.modules['llava'] = llava_pkg
        sys.modules['llava.constants'] = llava_constants
    spec = importlib.util.spec_from_file_location('ego4o_constants', CONSTANTS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_text_file(path):
    """Parse a texts/{item}.txt file -> list of (caption, start_s, end_s, ann_type).

    Format: caption#tokens#start#end#type. Parse from the right in case a
    caption ever contains '#'.
    """
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('#')
            if len(parts) < 5:
                continue
            ann_type = parts[-1]
            start_s, end_s = float(parts[-3]), float(parts[-2])
            caption = '#'.join(parts[:-4])
            entries.append((caption, start_s, end_s, ann_type))
    return entries


def load_frame_times(seq_name):
    """Sorted (time_s, abs_path) of extracted frames for a sequence, or None."""
    seq_dir = pjoin(FRAMES_DIR, seq_name)
    if not os.path.isdir(seq_dir):
        return None
    frames = []
    for fn in os.listdir(seq_dir):
        if not fn.endswith('.jpg'):
            continue
        try:
            t = float(fn[:-len('.jpg')])
        except ValueError:
            continue
        frames.append((t, pjoin(seq_dir, fn)))
    frames.sort()
    return frames if frames else None


def pick_frame(frames, start_s, end_s, tolerance=1.0):
    """Frame nearest the segment midpoint; must fall inside the (padded) segment."""
    if not frames:
        return None
    mid = 0.5 * (start_s + end_s)
    times = np.array([t for t, _ in frames])
    i = int(np.argmin(np.abs(times - mid)))
    t = times[i]
    if start_s - tolerance <= t <= end_s + tolerance:
        return frames[i][1]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default=OUT_DIR)
    args = parser.parse_args()

    consts = load_constants()
    mixed_questions = (consts.IMAGE_MOTION_TO_TEXT_QUESTION_LIST
                       + consts.IMAGE_TO_TEXT_QUESTION_LIST
                       + consts.MOTION_TO_TEXT_QUESTION_LIST)
    motion_questions = consts.MOTION_TO_TEXT_QUESTION_LIST
    rng = np.random.default_rng(SEED)

    with open(pjoin(PROCESSED_NYMERIA, 'data_order.txt')) as f:
        data_order = [line.strip() for line in f if line.strip()]

    splits = {}
    for split in ('train', 'val', 'test'):
        with open(pjoin(PROCESSED_NYMERIA, f'{split}.txt')) as f:
            splits[split] = [line.strip() for line in f if line.strip()]
    all_ids = [i for ids in splits.values() for i in ids]
    assert len(all_ids) == len(set(all_ids)), 'splits overlap!'

    report = {'splits': {}, 'sequences': {}}
    for split, item_ids in splits.items():
        n_seg = n_short = n_noimg = n_notext = n_nofeat = n_badtime = 0
        lines_mixed, lines_motion = [], []
        for item in sorted(item_ids):
            seq_name = data_order[int(item)]
            text_path = pjoin(PROCESSED_NYMERIA, 'texts', f'{item}.txt')
            feat_path = pjoin(args.out_dir, 'new_joint_vecs_30fps', f'{item}.npy')
            if not os.path.exists(text_path):
                n_notext += 1
                continue
            if not os.path.exists(feat_path):
                print(f'WARNING: missing features for {item} ({seq_name}) — run resample first')
                n_nofeat += 1
                continue
            feat_len = np.load(feat_path, mmap_mode='r').shape[0]
            frames = load_frame_times(seq_name)

            atomics = [e for e in parse_text_file(text_path) if e[3] == ATOMIC_TYPE]
            seq_kept = 0
            for k, (caption, start_s, end_s, _) in enumerate(atomics):
                start_f = int(round(start_s * FPS))
                end_f = min(int(round(end_s * FPS)), feat_len)
                if start_f < 0 or start_f >= feat_len:
                    # broken source timestamps (e.g. grace_randolph act0-2 have
                    # entirely negative text times) — never index with these
                    n_badtime += 1
                    continue
                if end_f - start_f < MIN_FRAMES:
                    n_short += 1
                    continue
                image = pick_frame(frames, start_s, end_s)
                if image is None:
                    n_noimg += 1

                item_id = f'{k}_{seq_name}'
                base = {
                    'id': item_id,
                    'hml_item': item,
                    'motion_file': f'new_joint_vecs_30fps/{item}.npy',
                    'start_frame': start_f,
                    'end_frame': end_f,
                    'fps': FPS,
                    'motion_id': [item_id],
                }
                q_list = mixed_questions if image is not None else motion_questions
                q = q_list[int(rng.integers(len(q_list)))]
                lines_mixed.append({**base, 'image': image, 'conversations': [
                    {'from': 'human', 'value': q},
                    {'from': 'gpt', 'value': caption}]})
                q2 = motion_questions[int(rng.integers(len(motion_questions)))]
                lines_motion.append({**base, 'image': None, 'conversations': [
                    {'from': 'human', 'value': q2},
                    {'from': 'gpt', 'value': caption}]})
                n_seg += 1
                seq_kept += 1
            report['sequences'][item] = {'seq': seq_name, 'split': split,
                                         'atomic': len(atomics), 'kept': seq_kept,
                                         'has_frames': frames is not None}

        for name, lines in (('ego4o_image_motion', lines_mixed),
                            ('ego4o_motion_text', lines_motion)):
            out = pjoin(args.out_dir, f'{name}_{split}.jsonl')
            with open(out, 'w') as f:
                for line in lines:
                    f.write(json.dumps(line) + '\n')
            print(f'{out}: {len(lines)} items')
        report['splits'][split] = {'sequences': len(item_ids), 'segments': n_seg,
                                   'dropped_short': n_short, 'dropped_bad_time': n_badtime,
                                   'missing_image': n_noimg,
                                   'missing_text': n_notext, 'missing_features': n_nofeat}

    with open(pjoin(args.out_dir, 'build_report.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report['splits'], indent=1))


if __name__ == '__main__':
    main()
