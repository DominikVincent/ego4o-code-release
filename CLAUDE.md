# CLAUDE.md

Guidance for working in this repository.

## What this is

Code release for **Ego4o: Egocentric Human Motion Capture and Understanding from
Multi-Modal Input** (Wang et al., CVPR 2025, [arXiv:2504.08449](https://arxiv.org/abs/2504.08449)).
"o" = *omni* (multi-modal). The paper PDF is in the repo root (`ego4o.pdf`).

Ego4o does two coupled tasks from consumer wearables, and works with whatever
modalities are present at inference time:

1. **Motion capture** — 3D body pose from **1–3 sparse IMUs** (head / wrists / hips,
   i.e. VR headset + smartwatches + phone), optionally aided by an egocentric image
   and/or a text description of the motion.
2. **Motion understanding** — generating a natural-language description of the motion.

Key idea: a **part-aware motion VQ-VAE** (per-limb codebooks) + a transformer
**multi-modal encoder** project IMU/image/text into a discrete motion-code space; the
VQ-VAE decoder reconstructs pose (with optional **test-time optimization** against the
IMU signal). For understanding, motion codes + image feed a **fine-tuned LLaVA
(Vicuna-7B)** LLM. Generated descriptions are looped back to improve mocap accuracy.

## Repository layout

Two independent components (both `README.md` files are empty):

- **`llava/`** — the main Ego4o codebase, forked from LLaVA. All novel work is under
  **`llava/llava/ego4o/`**:
  - `model/ego4o.py` — the Ego4o multi-modal LLM model.
  - `motion_limb_vqvae/` — part-aware motion VQ-VAE (built on TLControl). Body split
    into 6 groups (head, L/R arm, root, L/R leg), each with its own codebook.
  - `imu_tokenizer/` — transformer multi-modal encoder projecting IMU/image/text → motion codes.
  - `dataset/` — Nymeria dataset + transforms (see Data pipeline below).
  - `train/` — training entrypoints (`train_ego4o.py`, `train_mem_ego4o.py`).
  - `eval/` — evaluation scripts (batched test runners, select-best, ablations).
  - `serve/` — CLI demo.
  - `constants.py` — instruction/question templates + `<motion>` token definitions.
  - Note: `bak/` subdirs everywhere hold dead/older code — **ignore them**.
- **`EgoOmniMocap/`** — an MMPose fork used as the pose backbone and for IMU
  preprocessing utilities (`scripts/`, `vis_script/`).

## Data pipeline (Nymeria)

Builders live in `llava/scripts/ego4o/convert_nymeria_data_*.py`; the loader is
`llava/llava/ego4o/dataset/nymeria_dataset.py` (`NymeriaDataset`).

**Input the repo assumes already exists** (NOT produced by this repo — see gotchas):
- `atomic/` (note: dir is misspelled `automic` in code) — one pickle **per Nymeria
  recording** (`seq_name`), holding a *list* of atomic-action segments. Each segment has
  `motion` (XSens `segment_tXYZ`, `sensor_qWXYZ`, `sensor_freeAcceleration` @240 fps) and
  `text['Describe my atomic actions']` (the Nymeria **atomic-action narration**).
- `summary/` — higher-level Nymeria narrations (referenced but **not used** in the
  released image+motion pipeline).
- `images/` — egocentric frames per segment.

**Builder → intermediate format** (`convert_nymeria_data_smooth_imus.py` and siblings):
- Iterate recordings; for each **atomic segment** emit **one training item**.
- Downsample motion **240 → 30 fps**; save to `ego4o_input_motion/{seq_name}.pkl` keyed
  by `{i}_{seq_name}`; write a `.jsonl` mapping each item to an image + a conversation.
- The `gpt` answer is the atomic-action narration; the `human` question is randomly
  sampled from `constants.py` (task types: motion→text, image+motion→text, image→text,
  and (motion/text)→motion).

**Loader conversions** (`NymeriaDataset.convert_nymeria`):
- Nymeria 23-joint skeleton → HumanML3D (`keypoints_mapping/`).
- 17 IMUs subsampled to 5 placements: **head, L/R wrist, L/R hip** (`convert_imu`);
  a zero "root" IMU is prepended → `signal_num=6`.
- Quaternions → rotation matrices → 6D; everything rotated **Z-up → Y-up**.
- Motion → 263-dim HumanML3D representation, normalized (precomputed mean/std),
  padded to 148 frames. `seq_len=150` = 5 s @ 30 fps.
- Random modality masking (`random_mask=True`): drop image/text, randomly select 1–3
  active IMU combos (`imuposer_config.tlcontrol_combos`).

### The "170k sequences" — what they actually are

Nymeria has only ~900 raw **recordings**. The paper's "~170k sequences, each 5 s" are
**not** recordings — they are the **atomic-action annotation segments**: each recording
is cut at its atomic-action narration boundaries, and the builder emits one sample per
narration (`for i, atomic_data in enumerate(atomic_data_list)`). So 170k ≈ total
atomic-action narrations across all recordings (~170k × 5 s ≈ 236 h, matching Nymeria's
scale). Each sample keeps its origin in `id = {i}_{seq_name}` and `motion_file = {seq_name}.pkl`.

### Train/test split

Paper (§4.1): ~170k → **~119k train / ~51k test**, split by **different scenes and
motion-capture identities** (subject- and scene-disjoint, at the recording level, so
segments from one recording never cross the split). The loader reads pre-split files
`ego4o_image_motion_train.jsonl` / `ego4o_image_motion_test.jsonl` (`load_data`, keyed on
the `split` arg). It is **NOT** a random per-segment split and **NOT** split by
annotation type. Caveat: the exact partition script is not in the release (see gotchas).

## Running

Scripts in `llava/scripts/ego4o/`. Base model: `liuhaotian/llava-v1.6-vicuna-7b`,
vision tower `openai/clip-vit-large-patch14-336`. Multi-GPU via DeepSpeed.

- **Pretrain** (motion↔text alignment, freeze most, tune motion MLP adapter):
  `bash scripts/ego4o/pretrain_ego4o.sh`
- **Multi-modal finetune** (LoRA, image + motion + text): `bash scripts/ego4o/finetune_ego4o_multi_modal.sh`
- **Test on Nymeria**: `bash scripts/ego4o/test_nymeria.sh` (and `test_nymeria_*.sh`
  variants: `wo_image`, `wo_motion`, `select_best`, `global`).

## Gotchas

- **Hardcoded absolute paths everywhere**: `/scratch/inf0/user/jianwang/nymeria/...`,
  `/CT/EgoMocap/...`, `/HPS/EgoSyn/static00/...` (cluster paths from the authors). These
  must be repointed before anything runs. The loader even hardcodes a path-rewrite
  fallback (`/scratch/inf0/user/jianwang` → `/HPS/EgoSyn/static00`).
- **Not in the release**: (1) the code that extracts `atomic/summary` pickles from raw
  Nymeria, and (2) the script that produces the train/test `.jsonl` split. The repo
  consumes these as pre-existing inputs.
- Dir misspelled `automic` (not `atomic`) in code — matches the on-disk name.
- `bak/` directories and commented-out blocks are stale; don't treat them as current.
- Hardcoded normalization stat paths in `nymeria_dataset.py` point to the
  `EgoOmniMocap/projects/TLControl/demo/` mean/std files.
