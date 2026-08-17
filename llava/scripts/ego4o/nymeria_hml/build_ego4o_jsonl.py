"""Build the Ego4o training jsonl from a MotionGPT3 window manifest.

Why a manifest
--------------
Ego4o and MotionGPT3 must be scored on the *identical* population, but this
builder used to select windows itself (atomic-action lines only, >= 5 s, no
point-cloud filter, no maximum length). That drifted from MotionGPT3: on the
atomic test split MotionGPT3 had 25,694 windows, this builder produced 29,242,
and only 23,402 overlapped -- roughly 9 % of MotionGPT3's samples never got an
ego4o prediction and 20 % of ego4o's rows were thrown away by the intersection.

So window selection now lives in exactly one place. MotionGPT3's
`scripts/dump_dataset_manifest.py` instantiates its real dataset classes and
writes one record per window; this script turns those records into ego4o jsonl.
Every filter (caption type, 1-10 s length bounds, stitching on the train split,
the point-cloud drop) has already been applied there, so there is deliberately
no length or type filtering left here.

Inputs (read-only):
  {manifest_dir}/manifest_{split}.jsonl   from MotionGPT3's dump_dataset_manifest
  {out_dir}/new_joint_vecs_30fps/{idx}.npy   30 fps HML features (resample_features_30fps.py)
  {FRAMES_DIR}/{seq_name}/{t}.jpg            0.5 Hz egocentric frames

Outputs:
  {out_dir}/ego4o_image_motion_{train,val,test}.jsonl   finetune (mixed modality questions)
  {out_dir}/ego4o_motion_text_{train,val,test}.jsonl    pretrain (motion-only questions)
  {out_dir}/keys_{split}.txt                            sorted fnames, for find_intersecting_results.py
  {out_dir}/build_report.json

jsonl line schema (superset of the original ego4o schema):
  id            "{fname with '/' -> '-'}"  (+ "#j" for the j-th caption of a window)
  fname         MotionGPT3's join key, verbatim -- may contain '/'
  caption_type  the Nymeria annotation category
  hml_item      "000001"           item id in data_order
  seq_name      the Nymeria recording name
  motion_file   "new_joint_vecs_30fps/000001.npy"
  start_frame / end_frame          30 fps feature-frame slice
  start_s / end_s                  the manifest window, in seconds
  image         abs path | null
  fps           30
  motion_id     [id]
  gt_texts      every caption MotionGPT3 holds for this window
  conversations [{from: human, value: question}, {from: gpt, value: caption}]

Run (env ego4o, CPU only):
  python build_ego4o_jsonl.py --manifest_dir .../results/manifests/nymeria_4cat \
                              --out_dir /local/home/dhollidt/data/ego4o_nymeria_4cat
"""
import argparse
import importlib.util
import json
import os
from os.path import join as pjoin

import numpy as np

FRAMES_DIR = '/local/home/dhollidt/data/nymeria_frames'
OUT_DIR = '/local/home/dhollidt/data/ego4o_nymeria_4cat'
MANIFEST_DIR = '/local/home/dhollidt/repos/MotionGPT3/results/manifests/nymeria_4cat'
REPO = os.path.abspath(pjoin(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
CONSTANTS_PY = pjoin(REPO, 'llava', 'ego4o', 'constants.py')

FPS = 30
MIN_FRAMES_FOR_VQVAE = 4   # PadMotion truncates to (len // 4) * 4 before padding
SEED = 20260817
SPLITS = ('train', 'val', 'test')


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


def load_manifest(path):
    """Read a MotionGPT3 window manifest (one json object per line)."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


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
    parser.add_argument('--manifest_dir', default=MANIFEST_DIR,
                        help='dir holding manifest_{split}.jsonl from MotionGPT3')
    parser.add_argument('--out_dir', default=OUT_DIR)
    parser.add_argument('--categories', nargs='+', default=None,
                        help='caption types to keep (default: all four compared '
                             'categories). Narrowing this to a single type builds an '
                             'atomic-only dataset through the same code path.')
    parser.add_argument('--splits', nargs='+', default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument('--limit', type=int, default=None,
                        help='keep only the first N manifest windows per split (smoke tests)')
    args = parser.parse_args()

    consts = load_constants()
    categories = args.categories or list(consts.NYMERIA_CAPTION_TYPES)
    for category in categories:
        if category not in consts.CATEGORY_QUESTION_LISTS:
            raise SystemExit(f'no question list for caption type {category!r}; '
                             f'known: {sorted(consts.CATEGORY_QUESTION_LISTS)}')
    print(f'categories: {categories}')

    # As in the release, the question used at evaluation is one of the training
    # questions -- body [0] of each category. Nothing to hold out here.
    eval_prompts = {ct: consts.EVAL_QUESTION_BY_TYPE[ct] for ct in categories}

    rng = np.random.default_rng(SEED)
    report = {'categories': categories, 'splits': {}}

    for split in args.splits:
        manifest_path = pjoin(args.manifest_dir, f'manifest_{split}.jsonl')
        if not os.path.exists(manifest_path):
            raise SystemExit(f'missing manifest: {manifest_path}\n'
                             f'Run MotionGPT3 scripts/dump_dataset_manifest.py --split {split} first.')
        windows = load_manifest(manifest_path)
        windows = [w for w in windows if w['category'] in categories]
        if args.limit:
            windows = windows[:args.limit]
        print(f'{split}: {len(windows)} manifest windows')

        feat_len_cache, frames_cache = {}, {}
        lines_mixed, lines_motion = [], []
        n_noimg = n_trunc = n_padded = n_short = n_multi = 0
        per_category = {}

        for w in windows:
            item, seq_name = w['idx'], w['seq_name']
            start_s, end_s, category = w['start'], w['end'], w['category']

            if item not in feat_len_cache:
                feat_path = pjoin(args.out_dir, 'new_joint_vecs_30fps', f'{item}.npy')
                if not os.path.exists(feat_path):
                    raise SystemExit(f'missing features for {item}: {feat_path}\n'
                                     f'Run resample_features_30fps.py first.')
                feat_len_cache[item] = np.load(feat_path, mmap_mode='r').shape[0]
            feat_len = feat_len_cache[item]

            # Seconds -> 30 fps frames. The 30 fps arrays are exact 1.5x resamples
            # of the 20 fps ones MotionGPT3 indexes, so the timelines coincide; the
            # min() mirrors MotionGPT3's own silent clamp at the end of a recording.
            start_f = int(round(start_s * FPS))
            end_f = min(int(round(end_s * FPS)), feat_len)
            n_frames = end_f - start_f
            if n_frames < MIN_FRAMES_FOR_VQVAE:
                # Cannot happen for windows that passed MotionGPT3's >= 1 s filter;
                # never drop one silently, since that would break key-set equality.
                raise SystemExit(
                    f'window {w["fname"]} maps to {n_frames} frames @30fps '
                    f'(start_s={start_s} end_s={end_s} feat_len={feat_len})')
            if n_frames > 148:
                n_trunc += 1
            elif n_frames < 148:
                n_padded += 1
                if n_frames < 60:
                    n_short += 1

            if seq_name not in frames_cache:
                frames_cache[seq_name] = load_frame_times(seq_name)
            image = pick_frame(frames_cache[seq_name], start_s, end_s)
            if image is None:
                n_noimg += 1

            captions = w['captions']
            if len(captions) > 1:
                n_multi += 1
            per_category[category] = per_category.get(category, 0) + 1

            safe_id = w['fname'].replace('/', '-')
            base = {
                'fname': w['fname'],
                'caption_type': category,
                'hml_item': item,
                'seq_name': seq_name,
                'motion_file': f'new_joint_vecs_30fps/{item}.npy',
                'start_frame': start_f,
                'end_frame': end_f,
                'start_s': start_s,
                'end_s': end_s,
                'fps': FPS,
                'gt_texts': captions,
            }

            # Train uses every caption MotionGPT3 holds for the window (its own
            # train dataset does too); val/test stay 1:1 with the manifest so the
            # key sets match exactly.
            train_captions = captions if split == 'train' else captions[:1]
            q_lists = consts.CATEGORY_QUESTION_LISTS[category]
            mixed = q_lists['mixed'] if image is not None else q_lists['motion']
            motion_only = q_lists['motion']

            for j, caption in enumerate(train_captions):
                item_id = safe_id if j == 0 else f'{safe_id}#{j}'
                row = dict(base, id=item_id, motion_id=[item_id])
                q = mixed[int(rng.integers(len(mixed)))]
                lines_mixed.append({**row, 'image': image, 'conversations': [
                    {'from': 'human', 'value': q},
                    {'from': 'gpt', 'value': caption}]})
                q2 = motion_only[int(rng.integers(len(motion_only)))]
                lines_motion.append({**row, 'image': None, 'conversations': [
                    {'from': 'human', 'value': q2},
                    {'from': 'gpt', 'value': caption}]})

        # --- invariants -----------------------------------------------------
        manifest_keys = {w['fname'] for w in windows}
        for name, lines in (('ego4o_image_motion', lines_mixed),
                            ('ego4o_motion_text', lines_motion)):
            keys = {line['fname'] for line in lines}
            assert keys == manifest_keys, (
                f'{name}_{split}: key set differs from the manifest '
                f'(missing {len(manifest_keys - keys)}, extra {len(keys - manifest_keys)})')
        if split != 'train':
            assert len(lines_mixed) == len(manifest_keys), \
                f'{split} must be 1:1 with the manifest, got {len(lines_mixed)} rows'
        written_questions = {line['conversations'][0]['value'] for line in lines_mixed}
        written_questions |= {line['conversations'][0]['value'] for line in lines_motion}
        n_eval_prompt_rows = sum(
            1 for line in lines_mixed
            if line['conversations'][0]['value'] == eval_prompts[line['caption_type']])

        # Shuffle train/val: training-time evaluation reads only the FIRST 2048
        # lines of the val jsonl (train_ego4o.py make_supervised_data_module), so
        # manifest order would make eval_loss -- the early-stopping signal --
        # cover a handful of recordings of one category.
        if split in ('train', 'val'):
            order = rng.permutation(len(lines_mixed))
            lines_mixed = [lines_mixed[i] for i in order]
            lines_motion = [lines_motion[i] for i in order]

        os.makedirs(args.out_dir, exist_ok=True)
        for name, lines in (('ego4o_image_motion', lines_mixed),
                            ('ego4o_motion_text', lines_motion)):
            out = pjoin(args.out_dir, f'{name}_{split}.jsonl')
            with open(out, 'w') as f:
                for line in lines:
                    f.write(json.dumps(line) + '\n')
            print(f'{out}: {len(lines)} rows')

        with open(pjoin(args.out_dir, f'keys_{split}.txt'), 'w') as f:
            for key in sorted(manifest_keys):
                f.write(key + '\n')

        report['splits'][split] = {
            'manifest': manifest_path,
            'windows': len(windows),
            'rows': len(lines_mixed),
            'per_category': dict(sorted(per_category.items())),
            'multi_caption_windows': n_multi,
            'distinct_questions_written': len(written_questions),
            # rows whose sampled question happens to be the category's eval
            # question -- expected, since the eval question is a training one
            'rows_with_the_eval_question': n_eval_prompt_rows,
            'missing_image': n_noimg,
            'truncated_over_148_frames': n_trunc,
            'zero_padded_under_148_frames': n_padded,
            'under_60_frames': n_short,
        }

    with open(pjoin(args.out_dir, 'build_report.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report['splits'], indent=1))


if __name__ == '__main__':
    main()
