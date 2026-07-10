"""Convert ego4o m2t eval output into MotionGPT3-aligned predictions.

MotionGPT3 and this ego4o variant are evaluated on the *same* Nymeria source
(`HumanML3DFork/processed_nymeria_scene_sub_split`: same `data_order.txt`, same
`test.txt`, same `texts/{item}.txt` atomic-action annotations). MotionGPT3 keys every
eval sample by

    fname = "{hml_item}_{start_s:.6f}_{end_s:.6f}_{category_with_underscores}"

(dataset_t2m.py: `'%s_%f_%f_%s' % (name, f_tag, to_tag, caption_type.replace(' ','_'))`,
with `f_tag/to_tag = float(...)` of the raw annotation seconds). ego4o's atomic segments
are the SAME annotations, so we can reconstruct that exact fname per prediction and hand
MotionGPT3's `evaluate_from_prediction.py` a `{fname: pred_text}` map. Both models then
score through MotionGPT3's identical M2TMetrics on the intersection of samples.

This is a pure POST-PROCESSOR: it consumes the `result.json` that
`test_ego4o_hml_batch.py` already writes (fields: pred_text, gt_text, motion_id,
motion_file) plus read-only access to the processed-nymeria `texts/` — no model, no GPU.

Pipeline:
    1) GPUS=.. bash llava/scripts/ego4o/hml/stage4_eval.sh        # -> .../result.json
    2) python -m llava.ego4o.eval.export_predictions_for_mgpt3 \
           --result <save_dir>/result.json --out predictions.json
    3) (in MotionGPT3) python -m evaluate_from_prediction \
           --cfg configs/test_nymeria_env_me2t.yaml --predictions predictions.json

fname reconstruction detail: `motion_id` is "{k}_{seq_name}" (k = atomic index within the
recording, assigned by build_ego4o_jsonl.py's enumerate over atomic-only lines);
`motion_file` is ".../{hml_item}.npy". We re-parse `texts/{hml_item}.txt`, take the k-th
atomic line, and use its raw start_s/end_s -- identical to MotionGPT3's f_tag/to_tag.
"""
import argparse
import json
import os
from os.path import join as pjoin

# Default processed-nymeria root (same dir build_ego4o_jsonl.py used).
PROCESSED_NYMERIA = '/local/home/dhollidt/repos/HumanML3DFork/processed_nymeria_scene_sub_split'
ATOMIC_TYPE = 'Describe my atomic actions'


def parse_atomics(path):
    """texts/{item}.txt -> list of (caption, start_s, end_s) for atomic lines, in file order.

    Mirrors build_ego4o_jsonl.parse_text_file (split from the right; 5-field lines)
    filtered to ATOMIC_TYPE, so the k-th entry here == the k-th ego4o atomic segment.
    """
    atomics = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('#')
            if len(parts) < 5:
                continue
            ann_type = parts[-1]
            if ann_type != ATOMIC_TYPE:
                continue
            start_s, end_s = float(parts[-3]), float(parts[-2])
            caption = '#'.join(parts[:-4])
            atomics.append((caption, start_s, end_s))
    return atomics


def mgpt3_fname(hml_item, start_s, end_s):
    """Reproduce MotionGPT3's dataset_t2m.py fname formatting exactly."""
    return '%s_%f_%f_%s' % (hml_item, start_s, end_s, ATOMIC_TYPE.replace(' ', '_'))


def _norm(t):
    return ' '.join((t or '').strip().lower().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', required=True,
                    help='result.json produced by test_ego4o_hml_batch.py')
    ap.add_argument('--processed_nymeria', default=PROCESSED_NYMERIA,
                    help='processed nymeria root (must match MotionGPT3 NYMERIA_ROOT)')
    ap.add_argument('--out', default=None,
                    help='output predictions.json (default: alongside result.json)')
    args = ap.parse_args()

    texts_dir = pjoin(args.processed_nymeria, 'texts')
    if not os.path.isdir(texts_dir):
        raise FileNotFoundError(f'texts dir not found: {texts_dir}')

    with open(args.result) as f:
        results = json.load(f)

    atomics_cache = {}

    def get_atomics(hml_item):
        if hml_item not in atomics_cache:
            atomics_cache[hml_item] = parse_atomics(pjoin(texts_dir, f'{hml_item}.txt'))
        return atomics_cache[hml_item]

    predictions = {}
    n = 0
    n_ok = 0
    n_caption_mismatch = 0
    n_index_error = 0
    n_collision = 0
    mismatch_examples = []

    for item in results:
        n += 1
        pred_text = item['pred_text']
        gt_text = item.get('gt_text', '')
        motion_file = item['motion_file']
        motion_id = item['motion_id']
        if isinstance(motion_id, (list, tuple)):
            motion_id = motion_id[0]
        motion_id = str(motion_id)

        hml_item = os.path.splitext(os.path.basename(motion_file))[0]  # "000141"
        try:
            k = int(motion_id.split('_')[0])
        except (ValueError, IndexError):
            n_index_error += 1
            continue

        atomics = get_atomics(hml_item)
        if k >= len(atomics):
            n_index_error += 1
            continue

        caption, start_s, end_s = atomics[k]

        # Validate k-indexing against the recorded gt; if it disagrees, try to
        # recover by unique caption match before giving up.
        if _norm(caption) != _norm(gt_text):
            matches = [a for a in atomics if _norm(a[0]) == _norm(gt_text)]
            if len(matches) == 1:
                caption, start_s, end_s = matches[0]
            else:
                n_caption_mismatch += 1
                if len(mismatch_examples) < 10:
                    mismatch_examples.append({
                        'hml_item': hml_item, 'k': k,
                        'atomic_caption': caption, 'gt_text': gt_text,
                    })
                continue

        fname = mgpt3_fname(hml_item, start_s, end_s)
        if fname in predictions:
            n_collision += 1
        predictions[fname] = pred_text
        n_ok += 1

    out_path = args.out or pjoin(os.path.dirname(os.path.abspath(args.result)),
                                 'predictions_mgpt3.json')
    with open(out_path, 'w') as f:
        json.dump(predictions, f, indent=1)

    keys_path = pjoin(os.path.dirname(os.path.abspath(out_path)), 'keys.txt')
    with open(keys_path, 'w') as f:
        f.write('\n'.join(sorted(predictions.keys())) + '\n')

    report = {
        'result_json': os.path.abspath(args.result),
        'processed_nymeria': args.processed_nymeria,
        'total_results': n,
        'exported': n_ok,
        'unique_fnames': len(predictions),
        'fname_collisions': n_collision,
        'caption_mismatch_dropped': n_caption_mismatch,
        'index_errors': n_index_error,
        'mismatch_examples': mismatch_examples,
    }
    report_path = pjoin(os.path.dirname(os.path.abspath(out_path)), 'export_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=1)

    print(json.dumps({k: v for k, v in report.items() if k != 'mismatch_examples'}, indent=1))
    print(f'predictions -> {out_path}')
    print(f'keys        -> {keys_path}')
    print(f'report      -> {report_path}')


if __name__ == '__main__':
    main()
