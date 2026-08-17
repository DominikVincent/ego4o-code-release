"""Convert ego4o m2t eval output into MotionGPT3-aligned predictions.

MotionGPT3 keys every eval sample by

    fname = "{hml_item}_{start_s:.6f}_{end_s:.6f}_{category_with_underscores}"

(dataset_t2m.py: `'%s_%f_%f_%s' % (name, f_tag, to_tag, caption_type.replace(' ','_'))`).
Since the ego4o dataset is built from a MotionGPT3 window manifest
(scripts/ego4o/nymeria_hml/build_ego4o_jsonl.py), that key is carried verbatim
through the jsonl -> dataset -> collator -> result.json, and this script only has
to reshape the records into the shared predictions schema
(motGPT/utils/predictions_io.py): a `{meta, predictions}` object whose
`predictions` is `{fname: {gt_text, pred_text, category, start, end, ...}}`.
MotionGPT3's `evaluate_from_prediction.py` reads it directly; both models then
score through MotionGPT3's identical M2TMetrics on the same population.

This used to reconstruct the fname by re-parsing `texts/{item}.txt`, filtering to
atomic-action lines and taking the k-th one. That was a second, weaker
implementation of the mapping -- it only worked for atomic actions and produced
207 colliding fnames on the last run -- so it is gone. Result files produced
before the fname was threaded through cannot be exported by this script; their
`predictions_mgpt3.json` from the original run is the artifact to keep.

This is a pure POST-PROCESSOR: it consumes the `result.json` that
`test_ego4o_hml_batch.py` writes (fields: pred_text, gt_text, gt_texts, fname,
caption_type, motion_id, motion_file) -- no model, no GPU, no dataset access.

Pipeline:
    1) GPUS=.. bash llava/scripts/ego4o/hml/stage4_eval.sh --per_sample_prompt  # -> result.json
    2) python -m llava.ego4o.eval.export_predictions_for_mgpt3 \
           --result <save_dir>/result.json --out predictions.json
    3) (in MotionGPT3) python -m evaluate_from_prediction \
           --cfg configs/test_nymeria_env_me2t_4cat.yaml --predictions predictions.json --strict
"""
import argparse
import json
import os
from os.path import join as pjoin


def split_fname(fname):
    """Split "{idx}_{start:.6f}_{end:.6f}_{category}" into its parts.

    Mirrors motGPT/utils/predictions_io.py::split_fname, including restoring the
    spaces in the category. Note the category may itself contain a '/'
    ("Describe my legs/feet motion"), which is why the fname is never used as a
    path component.
    """
    parts = fname.split('_')
    if len(parts) < 4:
        return fname, None, None, None
    idx, start, end = parts[0], parts[1], parts[2]
    category = '_'.join(parts[3:])
    try:
        return idx, float(start), float(end), category.replace('_', ' ')
    except ValueError:
        return fname, None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', required=True,
                    help='result.json produced by test_ego4o_hml_batch.py')
    ap.add_argument('--out', default=None,
                    help='output predictions.json (default: alongside result.json)')
    ap.add_argument('--split', default='test',
                    help='dataset split, stored in the shared-format meta block')
    ap.add_argument('--task', default='me2t',
                    help='task name for the meta block (me2t: motion + environment '
                         '-> text, which is what the image-conditioned ego4o run is)')
    args = ap.parse_args()

    with open(args.result) as f:
        results = json.load(f)

    predictions = {}
    n = n_missing_fname = n_collision = 0
    per_category = {}

    for item in results:
        n += 1
        fname = item.get('fname')
        if not fname:
            n_missing_fname += 1
            continue

        idx, start, end, fname_category = split_fname(fname)
        # Prefer the explicit field: the category is more reliably read from the
        # record than re-derived from the fname suffix, which contains a '/'.
        category = item.get('caption_type') or fname_category

        if fname in predictions:
            n_collision += 1
        record = {
            'idx': idx,
            'start': start,
            'end': end,
            'category': category,
            'gt_text': item.get('gt_text', ''),
            'pred_text': item['pred_text'],
            'motion_file': item.get('motion_file'),
        }
        if item.get('gt_texts'):
            record['gt_texts'] = item['gt_texts']
        predictions[fname] = record
        per_category[category] = per_category.get(category, 0) + 1

    if n_missing_fname:
        raise SystemExit(
            f'{n_missing_fname}/{n} records carry no "fname". They were produced before '
            f'the MotionGPT3 join key was threaded through the eval script; re-run the '
            f'eval on a manifest-built dataset, or use the predictions_mgpt3.json that '
            f'the original run already wrote.')
    if n_collision:
        raise SystemExit(
            f'{n_collision} duplicate fnames in {args.result}. The manifest guarantees '
            f'unique windows, so this means the dataset or the eval run is inconsistent.')

    out_path = args.out or pjoin(os.path.dirname(os.path.abspath(args.result)),
                                 'predictions_mgpt3.json')
    meta = {
        'model': 'ego4o',
        'task': args.task,
        'split': args.split,
        'source_result_json': os.path.abspath(args.result),
        'count': len(predictions),
    }
    with open(out_path, 'w') as f:
        json.dump({'meta': meta, 'predictions': predictions}, f,
                  indent=2, ensure_ascii=False)

    keys_path = pjoin(os.path.dirname(os.path.abspath(out_path)), 'keys.txt')
    with open(keys_path, 'w') as f:
        f.write('\n'.join(sorted(predictions.keys())) + '\n')

    report = {
        'result_json': os.path.abspath(args.result),
        'total_results': n,
        'exported': len(predictions),
        'fname_collisions': n_collision,
        'per_category': dict(sorted(per_category.items())),
    }
    report_path = pjoin(os.path.dirname(os.path.abspath(out_path)), 'export_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=1)

    print(json.dumps(report, indent=1))
    print(f'predictions -> {out_path}')
    print(f'keys        -> {keys_path}')
    print(f'report      -> {report_path}')


if __name__ == '__main__':
    main()
