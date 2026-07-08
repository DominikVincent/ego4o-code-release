"""Rebuild the Nymeria HumanML3D features at 30 fps for Ego4o.

Reads the 20 fps `joint_positions.npy` of every recording in
`processed_nymeria_scene_sub_split` (READ-ONLY), linearly interpolates the
joint positions to 30 fps, and reruns the exact feature pipeline that produced
the user's 20 fps `new_joint_vecs` (see hml_feature_lib.py).

Outputs (per item `NNNNNN` from data_order.txt line order):
  {out_dir}/new_joint_vecs_30fps/NNNNNN.npy  (T30-1, 263) float32
  {out_dir}/new_joints_30fps/NNNNNN.npy      (T30-1, 22, 3) float32  (recovered RIC joints)
  {out_dir}/resample_report.json             frame counts + failures

Idempotent: items whose outputs already exist are skipped.

Run (CPU only):
  conda run -n ego4o python resample_features_30fps.py [--workers 16] [--items 000001 ...]
"""
import argparse
import json
import os
import sys
import traceback
from os.path import join as pjoin

# keep BLAS threads low; we parallelize across items
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import multiprocessing as mp

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROCESSED_NYMERIA = '/local/home/dhollidt/repos/HumanML3DFork/processed_nymeria_scene_sub_split'
OUT_DIR = '/local/home/dhollidt/data/ego4o_nymeria'
SRC_FPS = 20
TGT_FPS = 30
JOINTS_NUM = 22
FEET_THRE = 0.002  # same as motion_representation_nymeria.py __main__


def interpolate_positions(positions, src_fps=SRC_FPS, tgt_fps=TGT_FPS):
    """Linear time-interpolation of (T, J, 3) joint positions.

    Frame 0 is preserved exactly; the last 30 fps sample does not extrapolate
    beyond the 20 fps sequence end.
    """
    t_src = np.arange(len(positions)) / src_fps
    duration = t_src[-1]
    n_tgt = int(np.floor(duration * tgt_fps)) + 1
    t_tgt = np.arange(n_tgt) / tgt_fps
    # vectorized linear interpolation (equivalent to scipy interp1d, no dep on grid order edge cases)
    idx = np.searchsorted(t_src, t_tgt, side='right') - 1
    idx = np.clip(idx, 0, len(positions) - 2)
    w = ((t_tgt - t_src[idx]) * src_fps)[:, None, None].astype(np.float64)
    out = positions[idx] * (1.0 - w) + positions[idx + 1] * w
    return out.astype(np.float32)


def process_item(args):
    item, seq_name = args
    import hml_feature_lib as lib  # noqa: PLC0415 (initialized per worker)

    vec_path = pjoin(OUT_DIR, 'new_joint_vecs_30fps', f'{item}.npy')
    joints_path = pjoin(OUT_DIR, 'new_joints_30fps', f'{item}.npy')
    if os.path.exists(vec_path) and os.path.exists(joints_path):
        return item, 'skipped', None

    try:
        src_file = pjoin(PROCESSED_NYMERIA, 'nymeria', seq_name, 'joint_positions.npy')
        source_data = np.load(src_file)[:, :JOINTS_NUM]
        if np.isnan(source_data).any():
            # the 20 fps pipeline raised here too; fill (indices stay aligned) and record
            source_data = lib.fill_nan_with_previous_frame(source_data)
        n_src = len(source_data)

        positions_30 = interpolate_positions(source_data)
        data, ground_positions, positions, l_velocity = lib.process_file(positions_30, FEET_THRE)
        rec_ric_data = lib.recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), JOINTS_NUM)

        np.save(vec_path, data.astype(np.float32))
        np.save(joints_path, rec_ric_data.squeeze().numpy().astype(np.float32))
        return item, 'ok', {'src_frames_20fps': int(n_src),
                            'frames_30fps': int(len(positions_30)),
                            'feature_frames': int(len(data))}
    except Exception as e:  # noqa: BLE001 — keep going, report at the end
        return item, 'failed', f'{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}'


def init_worker(tgt_source_file):
    import hml_feature_lib as lib  # noqa: PLC0415
    example = np.load(tgt_source_file)[:, :JOINTS_NUM]
    lib.init_target_skeleton(example)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--items', nargs='*', default=None,
                        help='optional subset of item ids (e.g. 000001)')
    args = parser.parse_args()

    os.makedirs(pjoin(OUT_DIR, 'new_joint_vecs_30fps'), exist_ok=True)
    os.makedirs(pjoin(OUT_DIR, 'new_joints_30fps'), exist_ok=True)

    with open(pjoin(PROCESSED_NYMERIA, 'data_order.txt')) as f:
        data_order = [line.strip() for line in f if line.strip()]

    # tgt skeleton = frame 0 of item 000000 (first line of data_order == first
    # element of the sorted glob the original script used)
    tgt_source_file = pjoin(PROCESSED_NYMERIA, 'nymeria', data_order[0], 'joint_positions.npy')

    todo = [(str(i).zfill(6), seq) for i, seq in enumerate(data_order)]
    if args.items:
        wanted = set(args.items)
        todo = [t for t in todo if t[0] in wanted]

    print(f'{len(todo)} items, {args.workers} workers -> {OUT_DIR}')
    report = {}
    with mp.Pool(args.workers, initializer=init_worker, initargs=(tgt_source_file,)) as pool:
        for n, (item, status, info) in enumerate(pool.imap_unordered(process_item, todo), 1):
            report[item] = {'status': status, 'info': info}
            if status == 'failed':
                print(f'[{n}/{len(todo)}] {item} FAILED: {info}')
            elif n % 25 == 0 or n == len(todo):
                print(f'[{n}/{len(todo)}] {item} {status}', flush=True)

    ok = sum(1 for r in report.values() if r['status'] == 'ok')
    skipped = sum(1 for r in report.values() if r['status'] == 'skipped')
    failed = {k: v for k, v in report.items() if v['status'] == 'failed'}
    summary = {'ok': ok, 'skipped': skipped, 'failed': len(failed),
               'total': len(report), 'items': report}
    report_path = pjoin(OUT_DIR, 'resample_report.json')
    # merge with a previous report if resuming
    if os.path.exists(report_path):
        with open(report_path) as f:
            old = json.load(f).get('items', {})
        for k, v in old.items():
            if k not in summary['items'] or summary['items'][k]['status'] == 'skipped':
                summary['items'].setdefault(k, v)
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=1)
    print(f'done: {ok} ok, {skipped} skipped, {len(failed)} failed -> {report_path}')
    if failed:
        print('FAILED items:', ', '.join(sorted(failed)))


if __name__ == '__main__':
    main()
