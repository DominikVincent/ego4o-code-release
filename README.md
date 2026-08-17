# Ego4o reproduction — GT-motion variant (Nymeria)

Reproduction of **Ego4o: Egocentric Human Motion Capture and Understanding from
Multi-Modal Input** (Wang et al., CVPR 2025, [arXiv:2504.08449](https://arxiv.org/abs/2504.08449))
on Nymeria, with **one deliberate change**: the model consumes **ground-truth motion
instead of IMU signals** (text & GT motion & image → text). The IMU multi-modal
encoder (paper §3.2 / stage 2) is skipped entirely; GT motion enters the LLM through
the release's own `encode_motion` path (motion → frozen part-aware VQ-VAE codes →
motion MLP adapter E_M → LLM tokens).

This README documents **everything that was changed or added** relative to the
original release, the dataset build, and how to train. See `CLAUDE.md` for a map of
the original release and `handoff/HANDOFF.md` for prior AMASS-side work.

---

## 1. Environments

Two conda envs (they conflict: mmcv vs transformers/deepspeed):

| env | side | rebuild with |
|---|---|---|
| `ego4o` | mmpose (VQ-VAE training, data builders) | `handoff/setup_ego4o_env.sh` |
| `ego4o_llava` | LLM (stages 2/3, eval) | `handoff/setup_ego4o_llava_env.sh` + flash-attn 2.5.8 wheel¹ |

¹ `pip install flash-attn` fails twice on this machine: recent versions need torch≥2.2 and
the 2.5.8 source build hits a cross-device rename bug. Install the prebuilt wheel:
`pip install flash_attn-2.5.8+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`
(from github.com/Dao-AILab/flash-attention releases; a copy sits in
`/local/home/dhollidt/data/ego4o_weights/`). Metric deps: `pip install evaluate bert_score rouge_score nltk`.
**peft must be 0.4.0** — llava's pyproject leaves it unpinned; newer peft needs accelerate>0.21
and breaks `from transformers import Trainer` (`clear_device_cache` import error).

## 2. Dataset build

The release consumes pre-extracted pickles + jsonl that were never published. We build
an equivalent dataset from `HumanML3DFork/processed_nymeria_scene_sub_split`
(20 fps HumanML3D-format Nymeria, all annotation types, sequence-level split) and the
0.5 Hz egocentric frames in `/local/home/dhollidt/data/nymeria_frames/`. Sources are
read-only; everything lands in the new dataset dir.

**Full documentation: [`llava/scripts/ego4o/nymeria_hml/README.md`](llava/scripts/ego4o/nymeria_hml/README.md)** —
window provenance, the jsonl schema, prompt design, and the legacy atomic-only dataset.

Two datasets exist:

| dir | categories | windows (train/val/test) | used by |
|---|---|---|---|
| `/local/home/dhollidt/data/ego4o_nymeria` | atomic actions only | 110,441 / 15,217 / 29,449 | the original atomic model (frozen artifact; **not** reproducible from HEAD) |
| `/local/home/dhollidt/data/ego4o_nymeria_4cat` | atomic + hands/arms + legs/feet + body posture | 172,399 / 23,023 / 54,930 | the 4-category model compared against MotionGPT3 |

The second one's **windows come from a MotionGPT3 manifest**, not from this repo:
`scripts/dump_dataset_manifest.py` (MotionGPT3) instantiates MotionGPT3's real dataset
classes and dumps one record per window, and `build_ego4o_jsonl.py` turns those into
jsonl. This exists because the previous, independent implementation drifted — on the
atomic test split MotionGPT3 had 25,694 windows, this repo produced 29,242, and only
23,402 overlapped, so ~9 % of MotionGPT3's samples never got an ego4o prediction.

Builders (run in this order, env `ego4o`, CPU only) — `llava/scripts/ego4o/nymeria_hml/`:

1. **`resample_features_30fps.py`** — linearly interpolates each recording's 20 fps
   `joint_positions.npy` to **30 fps** (paper trains at 30 fps: 5 s → 150 frames → 148)
   and reruns the *verbatim* HumanML3D feature pipeline copied from
   `HumanML3DFork/motion_representation_nymeria.py` (`hml_feature_lib.py`; target
   skeleton = frame 0 of data_order item 000000, exactly as the 20 fps run).
   Output: `new_joint_vecs_30fps/` (T,263) + `new_joints_30fps/` (T,22,3), 886 items, 31 GB.
   Validated: positional dims identical to the 20 fps features (ratio 1.000), velocity
   dims × 20/30, recovered joints agree to µm at time-aligned frames.
2. **`compute_mean_std.py`** — recomputes 263-dim stats @30 fps (kept only as
   `Mean/Std_recomputed_30fps.npy` for reference). **The active normalization stats are
   the user's HumanML3D-aligned `Mean.npy`/`Std.npy`** (copied from processed_nymeria,
   saved additionally as `info_motion_mean/std.pt` — numpy arrays inside the .pt, which
   is what `NormalizeHMLMotion`/`AgrolMPJPE` expect). Rationale: keep normalization
   compatible with HumanML3D so the TLControl VQ-VAE init transfers. Note: velocity
   dims are ~1.5× mis-scaled at 30 fps under these stats, and they don't match the
   current features exactly (they predate a data revision) — accepted deliberately for
   cross-paper consistency.
3. **`build_ego4o_jsonl.py`** — consumes a MotionGPT3 window manifest: maps each window's
   `start/end` seconds → 30 fps feature frames, picks the frame nearest the window
   midpoint from `nymeria_frames/<seq>/<t>.jpg` (±1 s), samples one question per row from
   the per-category training lists in `constants.py` (seeded), answer = the narration.
   It applies **no** window filtering of its own — the manifest is the population.
   `--categories` (default: all four) narrows the build; `--limit` truncates for smoke
   tests. Rerunnable.

Output jsonl (schema is a superset of the release's: adds `hml_item`, `start_frame`,
`end_frame`, and — new — `fname` (MotionGPT3's join key), `caption_type` and `gt_texts`;
motion is sliced from the whole-recording feature arrays at load time — the 263-dim HML
representation is root-relative per frame, so slices need no renormalization):

| split | windows | rows | atomic | hands/arms | legs/feet | posture | with image |
|---|---|---|---|---|---|---|---|
| `ego4o_image_motion_train.jsonl` | 172,399 | 177,741 | 106,195 | 23,059 | 21,437 | 21,708 | 99.4 % |
| `ego4o_image_motion_val.jsonl` | 23,023 | 23,023 | 16,145 | 2,293 | 2,292 | 2,293 | 100 % |
| `ego4o_image_motion_test.jsonl` | 54,930 | 54,930 | 25,694 | 9,746 | 9,746 | 9,744 | 100 % |
| `ego4o_motion_text_{split}.jsonl` | same windows | | | | | | motion→text questions only (stage-2 pretrain) |

Train emits one row per (window, caption) so multi-caption windows are not wasted;
val/test stay 1:1 with the manifest so their key sets match it exactly.

Notes on the source data (see the builder README for the full list):
- The three `20230928_s0_grace_randolph_act{0,1,2}` recordings have entirely **negative
  text timestamps**; MotionGPT3's length filter drops those windows, so they never reach
  the manifest.
- Time sync text↔motion↔image was verified by construction (all three extractions anchor at
  `NymeriaDataProvider.timespan_ns[0]`) **and** empirically (camera-vs-head-joint speed
  correlation peaks at exactly 0.00 s, r≈0.96–0.98, on 4 sequences).
- ego4o's window is fixed at 148 frames @30 fps: 44 % of test windows are shorter and get
  zero-padded, 56 % are longer and get truncated (mostly by 2 frames — the median atomic
  window is exactly 5.00 s = 150 frames). Mean temporal coverage 97.7 %.

## 3. Loaders (new, originals untouched)

- `llava/llava/ego4o/dataset/nymeria_hml_dataset.py` — `NymeriaHMLDataset` (LLM side).
  Slices `new_joint_vecs_30fps` per segment (mmap), pipeline = normalize → pad(148) →
  reshape to (263,1,148); text/image handling identical to `NymeriaDataset`; **no IMU keys**
  (their absence routes `prepare_inputs_labels_for_motion` into `encode_motion`).
- `EgoOmniMocap/mmpose/datasets/datasets/nymeria/nymeria_hml_dataset.py` — registered
  mmpose twin for VQ-VAE training; GT joints for the MPJPE evaluators are recovered via
  `recover_from_ric` from the sliced raw features (same slice-canonical anchor as predictions).

## 4. Weights

`/local/home/dhollidt/data/ego4o_weights/TLControl/`:
- `save_weights_vq/best_model_epoch_hml_emaReset.pth` — TLControl VQ-VAE (128 codes/128 dim),
  the **init** for stage 1 (gdown folder `1DX1CxGDLYzVblMJaAV81TNXOeRr7m9LE`, from the
  TLControl README). Only the 512-wide conv backbone transfers; the 6 codebooks
  (4096×64 here vs 128×128) and the encoder-out/decoder-in projections are re-initialized —
  expected, mmengine logs "size mismatch" warnings for exactly those keys.
- `demo/info_motion_mean/std.pt` — TLControl's HumanML3D stats (reference only; committed in
  the TLControl repo). Our dataset ships its own stats (§2.2).
- Base LLM `liuhaotian/llava-v1.6-vicuna-7b` + `openai/clip-vit-large-patch14-336` are in the
  HF cache (auto-downloaded).

## 5. Training — run these scripts in order

Scripts: `llava/scripts/ego4o/hml/`. Each aborts if the requested GPUs are in use
(shared server!); select GPUs via `GPUS=…`. wandb uses your login (`WANDB_MODE=offline` to disable).
**Operational details — checkpoint chaining, resume, adapting hyperparameters, picking the
best early-stopped checkpoint — are in [`training_info.md`](training_info.md).** A sequential
runner with skip logic exists too: `run_all_stages.sh`.

```bash
# Stage 1 — part-aware VQ-VAE finetune (paper stage C3). SINGLE GPU (code hardcodes .cuda()).
#   env ego4o, ~26k iters / 30 epochs, a few hours, <10 GB VRAM.
GPUS=2 bash llava/scripts/ego4o/hml/stage1_train_vqvae.sh
#   -> EgoOmniMocap/work_dirs/train_nymeria_vqvae_4096_64_hml/best_vqvae.pth (symlink)
#   sanity: val C-MPJPE should land in the tens of mm (paper recon ceiling ≈45 mm)

# Stage 2 — LLM motion<->text alignment pretrain (paper stage C5a). 2 GPUs.
#   Trains ONLY E_M (vq_net_postprocess). 1 epoch over 110k motion->text samples.
GPUS=2,3 bash llava/scripts/ego4o/hml/stage2_pretrain_llm.sh
#   -> llava/checkpoints/ego4o_hml_pretrain

# Stage 3 — multi-modal finetune (paper stage C5b). 2 GPUs.
#   LoRA r=128 α=256 on the LLM + full training of E_I (mm_projector) and E_M.
#   4 epochs, eval on val every 500 steps, early stop patience 3 on eval_loss.
GPUS=2,3 bash llava/scripts/ego4o/hml/stage3_finetune_llm.sh
#   -> llava/checkpoints/ego4o_hml_finetune_lora (adapter + non_lora_trainables.bin)

# Stage 4 — eval on the test split (BLEU / BERTScore / ROUGE, paper Tab. 5). 1 GPU.
GPUS=2 bash llava/scripts/ego4o/hml/stage4_eval.sh                  # full 29,449
GPUS=2 bash llava/scripts/ego4o/hml/stage4_eval.sh --data_range 500 # quick subsample
#   -> llava/eval_out/test_nymeria_hml_test_<ts>/{metrics.json,result.json,...}
```

The original IMU encoder stage (paper stage 2 / handoff C4,
`EgoOmniMocap/configs/nymeria/train_nymeria_random_image_text.py`) is **intentionally not run**.

## 6. Code changes vs. the release (complete list)

**Modified** (all changes are additive/gated; the IMU-input path still works if you set
`pretrained_imu_tokenizer_path`):

- `llava/llava/ego4o/model/ego4o.py`
  - `Ego4oConfig.pretrained_vqvae_path` → local stage-1 symlink (was authors' cluster path);
    `pretrained_imu_tokenizer_path` → `None` (was cluster path).
  - `initialize_ego4o_modules`: the IMU tokenizer block (instantiation + strict checkpoint
    load + `imu_tokenizer_postprocess`) is now **gated** on `pretrained_imu_tokenizer_path`
    — previously it loaded unconditionally and crashed without the (unreleased) checkpoint.
  - `encode_image_imu`: clear error if called without the IMU modules.
- `llava/llava/ego4o/train/train_ego4o.py`
  - `make_supervised_data_module`: `NymeriaHMLDataset` (train + val-eval) instead of
    `NymeriaDataset` with hardcoded `/scratch/inf0/...`; also fixes the release bug where a
    nonexistent `max_data_num` kwarg would crash the eval dataset.
  - Collator: IMU branch removed (batch = input_ids/labels/attention_mask/images/motion_hml).
  - New args: `--pretrained_vqvae_path`, `--pretrained_imu_tokenizer_path` (ModelArguments;
    absorbed into the config by `from_pretrained`), `--dataset_dir` (DataArguments),
    `--early_stopping_patience` (TrainingArguments → `EarlyStoppingCallback`).
  - LoRA: after `get_peft_model`, `vq_net_postprocess` (E_M) is re-enabled for training —
    paper §3.3.2 trains E_I + E_M + LoRA; peft had frozen E_M (E_I is re-enabled by existing
    code in `llava_arch.py`). Both land in `non_lora_trainables.bin`.
  - The redundant second IMU-tokenizer reload and the IMU freezes are gated (`hasattr`/path).
  - Early stopping uses a small custom callback: transformers 4.37's `EarlyStoppingCallback`
    requires `load_best_model_at_end`, whose in-training reload is **broken for
    LoRA + DeepSpeed** (expects a full engine state, gets adapter-only weights). The custom
    callback only stops; the best checkpoint (lowest `eval_loss`, recorded in
    `trainer_state.json`) stays on disk and is directly evaluable.
- `llava/llava/train/llava_trainer.py`
  - `_save_checkpoint`: under LoRA, every intermediate `checkpoint-N/` now also gets
    `non_lora_trainables.bin` (E_I/E_M), so the early-stopping "best" checkpoint can be fed
    straight to the eval script (`MODEL_PATH=.../checkpoint-N`).
- `llava/llava/ego4o/utils/humanml_utils/motion_representation.py`: the import-time
  `np.load` of the authors' HumanML3D `joints/000021.npy` is wrapped in try/except
  (`tgt_offsets=None`) — it is only needed for `process_file(uniform=True)`, which the
  precomputed-feature path never calls.

**Modified for the 4-category comparison** (see `llava/scripts/ego4o/nymeria_hml/README.md`):

- `llava/scripts/ego4o/nymeria_hml/build_ego4o_jsonl.py` — **rewritten** to consume a
  MotionGPT3 window manifest (`--manifest_dir`, `--categories`) instead of selecting
  windows itself. The atomic-only filter, the `MIN_FRAMES = 150` rule and the
  `texts/` parsing are gone; `load_constants` / `load_frame_times` / `pick_frame` are
  unchanged. Consequence: the legacy `ego4o_nymeria` dataset dir is no longer
  reproducible from HEAD (its rules are documented in the builder README instead).
- `llava/llava/ego4o/constants.py` — appended `CATEGORY_QUESTION_BODIES` (9 per
  category, composed over the release's three modality prefixes) and
  `EVAL_QUESTION_BY_TYPE` (body [0] of each category with the `<image>\n<motion>\n`
  prefix). As in the release, the evaluation question is one of the training questions.
  Existing lists untouched.
- `llava/llava/ego4o/dataset/nymeria_hml_dataset.py` — `load_data` carries `fname`,
  `caption_type` and `gt_texts` through (with `.get()` defaults, so the legacy jsonl
  still loads).
- `llava/llava/ego4o/eval/test_ego4o_hml_batch.py` — `--per_sample_prompt` (default off,
  so the atomic model still evaluates exactly as before) prompts each sample with the
  fixed question for its caption type. Needs left padding, an explicit
  `attention_mask` and `model.config.tokenizer_padding_side = 'left'` (the checkpoints
  store the *training* value, `'right'`, under which every sample but the longest would
  generate from pad embeddings). `result.json` now carries `fname`/`caption_type`, and
  `metrics.json` gains a per-category breakdown.
- `llava/llava/ego4o/eval/export_predictions_for_mgpt3.py` — reads `fname` straight from
  `result.json`; the k-th-atomic-line reconstruction (which produced 207 colliding fnames)
  is **deleted**. Old result files can no longer be exported — their original
  `predictions_mgpt3.json` is the artifact to keep.
- `llava/scripts/ego4o/hml/stage{2,3}_*.sh` — `OUTPUT_DIR` env var (defaults unchanged),
  so a second model can be trained without overwriting the first.

On the **MotionGPT3** side: `motGPT/models/base.py` no longer does
`fname.split('/')[-1]` when writing predictions. Two Nymeria caption types contain a
slash, so that collapsed all 9,746 legs and all 9,746 arms predictions onto the single
keys `feet_motion` / `arms_motion`. The full fname is now the record key and a
`'/'→'-'` copy is used only for the `.npy` sidecar filenames. Added there:
`scripts/dump_dataset_manifest.py`, `configs/test_nymeria_env_me2t_4cat.yaml`,
`configs/dump_train_manifest_4cat.yaml`.

**Added**:
- `llava/scripts/zero2.json`, `llava/scripts/zero2_offload.json` — the release's training
  scripts reference these DeepSpeed configs but they were missing (standard LLaVA ZeRO-2).
  Root cause found: `llava/.gitignore` ignores `*.json`, so the authors' configs were never
  committed; an exception is added so ours are.
- `llava/llava/ego4o/eval/test_ego4o_hml_batch.py` — GT-motion eval (adapted from
  `test_ego4o_image_imu_batch.py`): NymeriaHMLDataset, no IMU kwargs in `generate`,
  **LoRA loading** (base + `non_lora_trainables.bin` + peft merge), metrics saved to json.
- `llava/scripts/ego4o/hml/stage{1..4}*.sh` + `gpu_guard.sh` — the runbook above.
- `EgoOmniMocap/configs/nymeria_vqvae/train_nymeria_vqvae_4096_64_hml.py` — stage-1 config
  (original hyperparameters; dataset/stats/checkpoint paths localized; val on the val split).
- `EgoOmniMocap/mmpose/datasets/datasets/nymeria/nymeria_hml_dataset.py` (+ `__init__` export),
  `llava/llava/ego4o/dataset/nymeria_hml_dataset.py` — loaders (§3).
- `llava/scripts/ego4o/nymeria_hml/*` — dataset builders (§2).
- `handoff/setup_ego4o_env.sh`, `handoff/setup_ego4o_llava_env.sh`, `handoff/ego4o_bulk_reqs.txt` — env recipes.

## 7. Deviations from the paper / release (and why)

| deviation | why |
|---|---|
| GT motion instead of IMU; stage-2 encoder skipped | the point of this reproduction (user's comparison) |
| 30 fps features obtained by interpolating the 20 fps HML data | avoids re-extracting all of Nymeria from raw; validated equivalent (§2.1) |
| Normalization stats = HumanML3D-aligned (not recomputed) | deliberate, for VQ-VAE transfer + consistency with the user's own paper |
| Finetune uses **LoRA r=128 α=256, lr 2e-4** (+E_I/E_M fully trained) | paper text §3.3.2; NOTE the released script did **full FT @ 2e-5, 2 epochs** — we follow the paper (user decision) |
| 4 epochs + early stopping on val loss (patience 3 @ 500-step evals) | paper says 4 epochs; release script said 2; early stop added as a guard |
| LLM per-device batch 64 (release: 16/24) → effective 128, 2×H200 | hardware headroom; lr kept at 1e-3; release GPU count unknown |
| Question sampling: `<image>`-questions only for samples that have a frame | release assumed an image per segment; 0.3 % of train lacks frames |
| Train/val/test = user's sequence-level split (596/85/172 recordings) | required for comparison with the user's paper; paper's exact split was never released |
| VQ-VAE eval on **val** split each epoch (release: test) | keeps test untouched for final numbers |
| Windows come from a MotionGPT3 manifest, not from this repo's own rules | the two implementations drifted (23,402 of 25,694 atomic test windows overlapped); a fair comparison needs one definition of the population |
| The 4-category model shares the **frozen stage-1 VQ-VAE** with the atomic model (stage 1 is not re-run) | keeps the motion tokenizer identical across the two ego4o models. Caveat: that VQ-VAE was trained on the ≥ 150-frame legacy dataset, so short (zero-padded) windows are slightly out of its training distribution |
| Short windows are zero-padded to 148 frames, long ones truncated | ego4o has a fixed window and no length mask; MotionGPT3 feeds true variable-length motion. Inherent to the baseline, not a choice made here |
| Atomic keeps the release's evaluation prompt; the other three mirror MotionGPT3's `EVAL_TEMPLATES_ME2T` | atomic stays comparable to the earlier atomic-only run, while the new categories ask for the same kind of caption as MotionGPT3 |
| ego4o is not trained on `Describe my activity`; the MotionGPT3 checkpoint was | activity windows are ~30 s and would be truncated to 4.93 s. Excluded from both evaluations |

## 8. Smoke-test status (2026-07-08)

Every stage was smoke-tested end-to-end on GPUs 2/3 before handover:
- **Stage 1**: 2 truncated epochs — losses ↓, C/P-MPJPE compute, TLControl partial init logs the
  expected size-mismatch warnings, checkpoint strict-loads into the llava-side `HumanVQVAE`.
- **Stage 2**: 20 steps @ bs 32×2 — loss ≈2.9→, ~94 samples/s, full checkpoint (14 GB) saved with
  `pretrained_imu_tokenizer_path: null` and E_M + VQ-VAE weights included; no IMU modules built.
- **Stage 3**: 25 steps LoRA — eval_loss 1.44→1.36, adapter + `non_lora_trainables.bin`
  (4 E_I + 4 E_M keys) saved per checkpoint and at the end, best checkpoint tracked.
- **Stage 4**: merged-LoRA generation on 9 test samples → fluent scene-appropriate text,
  BLEU/BERTScore/ROUGE computed and saved (`llava/eval_out/`).
Smoke artifacts: `EgoOmniMocap/work_dirs/smoke_vqvae/`, `llava/checkpoints/smoke_pretrain/`
(14 GB — delete when the real stage-2 run exists), `llava/checkpoints/smoke_finetune_lora/`.

## 9. Data-flow summary (GT-motion variant)

```
new_joint_vecs_30fps/{item}.npy ──slice──> motion_hml (263,1,148, normalized)
    └── frozen VQ-VAE (stage 1, 6 limb codebooks 4096×64) → codes → embeddings (384)
        └── E_M vq_net_postprocess (384→4096) ──┐
frame.jpg → CLIP ViT-L/14-336 → E_I mm_projector ──┤ spliced at <motion>/<image> tokens
question/answer text → tokenizer ──────────────────┴─> Vicuna-7B (LoRA) → description
```
