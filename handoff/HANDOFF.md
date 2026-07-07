# Ego4o reproduction — handoff to the server

Ego4o = "Egocentric Human Motion Capture and Understanding from Multi-Modal Input"
(arXiv 2504.08449). Repo has two parts: `EgoOmniMocap/` (mmpose fork — VQ-VAE +
IMU/multimodal encoder) and `llava/` (LLaVA fork — the LLM, stage 3).

The paper has **3 training stages**:
1. **Part-aware VQ-VAE** (motion→codes→motion). Training code is NOT in the release;
   it lives in the external **TLControl** repo, which publishes the exact weights.
2. **Multi-modal encoder** (IMU + optional image/text → motion codes). Trained in
   `EgoOmniMocap` via mmpose `tools/train.py`.
3. **LLM** (LLaVA-Vicuna-7B) for motion understanding, in `llava/`.

## Part A — What has already been done (locally, on a 4090)

Stages **1 and 2 on AMASS only** (no Nymeria, no LLM) are done and validated:

- Built a conda env (`ego4o`) for the mmpose side — see `ego4o_pip_freeze.txt`.
- Downloaded TLControl's released weights and confirmed they load into the repo's
  model classes: **VQ-VAE is 128 codes / 128 dim**, loads `strict=True`.
- Ran the full AMASS→synthetic-IMU→HumanML3D-segment preprocessing (18/20 AMASS
  subsets; KIT+EKUT are 100 fps and are **intentionally dropped** — the original
  IMUPoser/Ego4o script keeps only 60/120 fps).
- Trained the stage-2 IMU→code encoder (30 epochs, ~3 h, 9 GB). **Best held-out
  (ACCAD) val: 117.9 mm C-MPJPE / 79.9 mm P-MPJPE.**
- Verified the VQ-VAE reconstruction ceiling: **48.6 mm C-MPJPE / 28.6 mm PA-MPJPE**,
  matching the paper's 44.93 / 32.72 — i.e. the decoder is faithful; the remaining
  error is the sparse-IMU code-prediction gap, which Nymeria training + image/text
  conditioning closes in the paper (→ 69 mm PA-MPJPE, 84–96 mm MPJPE).

Result: the stage-1/2 pipeline is proven end-to-end. What remains is the **Nymeria +
LLM** work, which needs the big compute and the Nymeria dataset — that's your job.

## Part B — Files in this bundle and where they go in the server repo

Assume the server has a fresh clone of the same repo at `<REPO>/` (with `EgoOmniMocap/`
and `llava/`). **Every path in these files is hardcoded to the local machine — edit the
paths at the top of each before running.** Search-and-replace `/home/dominik/Documents`.

| Bundle file | Copy to (server) | Notes |
|---|---|---|
| `CLAUDE.md` | `<REPO>/CLAUDE.md` | repo overview; read first |
| `reusable/make_path_dict.py` | anywhere (e.g. `<REPO>/tools/`) | builds `path_dict.pkl` from HumanML3D `index.csv` — only needed if re-running AMASS stage 2 on the server |
| `reusable/transpose_25fps.py` | same | fixes 25fps output format (list-of-dicts→dict-of-lists) for AMASS stage 2 |
| `reusable/verify_tlcontrol_ckpts.py` | same | sanity: TLControl ckpts load into repo model classes |
| `reusable/vqvae_recon_eval.py` | same | VQ-VAE reconstruction MPJPE check |
| `repo_patches/EgoOmniMocap/mmpose/datasets/datasets/imuposer/imuposer_dataset.py` | overwrite same path | adds a **portable** `hold_out_val` kwarg (leakage-free AMASS val). Safe to overwrite. |
| `repo_patches/EgoOmniMocap/mmpose/utils/humanml_utils/motion_representation.py` | apply edit only | **edit the one line** `data_dir = '.../HumanML3D/joints/'` to the server's HumanML3D path (needed by every stage that builds HML motion) |
| `repo_patches/EgoOmniMocap/scripts/*.py` | reference | AMASS preprocessing, path-specific; only if reproducing AMASS/DIP stage 2 |
| `repo_patches/EgoOmniMocap/configs/imuposer/train_imuposer_dataset_mask_local.py` | `<REPO>/EgoOmniMocap/configs/imuposer/` | template for a local-path config; mirror this pattern for the Nymeria configs |

If the server is going straight to **Nymeria + LLM** and does NOT need to reproduce the
AMASS/DIP-IMU numbers, the only strictly-needed items are `CLAUDE.md`, the
`motion_representation.py` path edit, and the env recipe. The rest is for AMASS stage 2.

## Part C — What the server agent should do next (stage 3)

### C0. Environments
Two separate conda envs (they conflict — mmcv vs transformers/deepspeed):
- **mmpose side** (VQ-VAE + encoder training): reproduce from `ego4o_pip_freeze.txt`.
  Key pins: python 3.10, torch 2.1.2+cu121, **numpy==1.23.5** (chumpy needs np.bool),
  mmcv==2.1.0, mmengine, mmdet, mmpose 1.3.0 (`pip install -e EgoOmniMocap`),
  OpenAI `clip` (`pip install git+https://github.com/openai/CLIP.git`), open3d.
- **llava side** (stage 3): per `llava/pyproject.toml` — torch 2.1.2, transformers
  4.37.2, deepspeed 0.12.6, peft/LoRA, plus `clip`, `mmengine`, `mmcv`, `mmpose`,
  `scipy`, `fontTools`, `natsort` (the ego4o package imports these).

### C1. Shared artifacts (both stages need these)
- Clone **TLControl** (github.com/HiWilliamWWL/TLControl) and `gdown` its weights:
  `save_weights_vq/best_model_epoch_hml_emaReset.pth` (VQ-VAE 128/128),
  `save_weights/update_design/withEmaReset_stage3.pth` (transformer),
  and `demo/info_motion_mean.pt` / `info_motion_std.pt` (263-dim HML stats).
  These are referenced by nearly every config/dataset as
  `.../projects/TLControl/...` — repoint those paths.

### C2. Nymeria data (the big blocker)
The repo consumes **pre-extracted** Nymeria: a dir of `atomic/` (misspelled
`automic`) per-recording pickles (`segment_tXYZ`, `sensor_qWXYZ`,
`sensor_freeAcceleration` + `text['Describe my atomic actions']`), a `summary/` dir,
extracted egocentric `images/`, and the split files
`ego4o_image_motion_train.jsonl` / `_test.jsonl`.
**The code that extracts these from raw Nymeria is NOT in the release**, and neither is
the scene/identity split script. The server agent must either (a) reimplement extraction
from raw Nymeria using the Nymeria SDK (atomic-action narrations → 5 s segments @ 30 fps,
XSens body motion + IMU + ego frames), or (b) request the preprocessed data + split from
the authors (Jian Wang, MPI-INF). Then run
`llava/scripts/ego4o/convert_nymeria_data_*.py` (repoint `/scratch/inf0/user/jianwang/nymeria`).
See `CLAUDE.md` "Data pipeline" for the exact fields and the train/test split logic.

### C3. Nymeria VQ-VAE finetune (stage 1, Nymeria)
`EgoOmniMocap/configs/nymeria_vqvae/train_nymeria_vqvae_4096_64.py` — this is the
**4096 codes / 64 dim** VQ-VAE the LLM later expects (note: different geometry from the
TLControl 128/128; it `init_cfg`s from the TLControl VQ-VAE and trains on Nymeria motion).
Produces `work_dirs/train_nymeria_vqvae_4096_64/best_*.pth`. Repoint all cluster paths
and the `info_motion_mean/std.pt` paths. Run with `tools/train.py`.

### C4. Nymeria multimodal encoder (stage 2, Nymeria)
`EgoOmniMocap/configs/nymeria/train_nymeria_random_image_text.py` — the image/text +
IMU encoder (`IMUPoserEncoder` with CLIP image/text, random masking). Loads the Nymeria
VQ-VAE from C3 + the TLControl transformer init. Produces
`work_dirs/train_nymeria_random_image_text/best_*.pth`. This is the encoder checkpoint
the LLM loads.

### C5. LLM (stage 3)
In `llava/`. `llava/llava/ego4o/model/ego4o.py` **hardcodes**:
- `pretrained_vqvae_path = '.../work_dirs/train_nymeria_vqvae_4096_64/best...pth'` (from C3)
- `pretrained_imu_tokenizer_path = '.../work_dirs/train_nymeria_random_image_text/best...pth'` (from C4)
Repoint both. Then run, in order:
1. `llava/scripts/ego4o/pretrain_ego4o.sh` (motion↔text alignment; base
   `liuhaotian/llava-v1.6-vicuna-7b`, vision tower `openai/clip-vit-large-patch14-336`).
2. `llava/scripts/ego4o/finetune_ego4o_multi_modal.sh` (LoRA rank 128/alpha 256,
   image+motion+text). Multi-GPU via DeepSpeed ZeRO-2 — this is the compute-heavy step
   (~7B model, ~170k samples, paper uses 4 epochs).
Then eval with `llava/llava/ego4o/eval/test_*` and `llava/scripts/ego4o/test_nymeria*.sh`.

### Recurring gotchas (seen locally, will recur on the server)
- Hardcoded absolute paths everywhere: `/CT/EgoMocap/...`, `/scratch/inf0/user/jianwang/nymeria`,
  `/HPS/EgoSyn/static00/...`, and Windows `Z:\`/`\\winfs-inf` in the AMASS scripts.
- `automic` (misspelled) is the on-disk Nymeria atomic dir name.
- `info_motion_mean/std.pt` (263-dim, from TLControl) are needed for HML normalization
  by the Nymeria dataset, the encoder, and the LLM eval.
- Reference-skeleton load in `mmpose/utils/humanml_utils/motion_representation.py`
  hardcodes `.../HumanML3D/joints/000021.npy` — repoint to the server's HumanML3D.
- Two CLIPs: ViT-B/32 (encoder side, OpenAI `clip` pkg) and ViT-L/14-336 (LLM vision tower).
- The released `imuposer_process_25fps.py` writes list-of-dicts but `IMUPoserDataset`
  wants dict-of-lists (see `transpose_25fps.py`) — only relevant if redoing AMASS stage 2.
