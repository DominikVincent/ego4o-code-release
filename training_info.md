# Ego4o GT-motion training — runbook

How to run, adapt, and chain the four training stages. Background and the full
change log are in `README.md`; this file is the operational reference.

All scripts live in **`llava/scripts/ego4o/hml/`** and were smoke-tested end-to-end
(README §8). Every script starts with a GPU guard that **aborts if a requested GPU
has >2 GiB allocated** (shared server) — pick GPUs with `GPUS=…`.

```bash
# one-shot (skips already-completed stages, safe to re-run):
GPUS_SINGLE=2 GPUS_DUAL=2,3 bash llava/scripts/ego4o/hml/run_all_stages.sh

# or stage by stage:
GPUS=2   bash llava/scripts/ego4o/hml/stage1_train_vqvae.sh
GPUS=2,3 bash llava/scripts/ego4o/hml/stage2_pretrain_llm.sh
GPUS=2,3 bash llava/scripts/ego4o/hml/stage3_finetune_llm.sh
GPUS=2   bash llava/scripts/ego4o/hml/stage4_eval.sh
```

## Stage overview & checkpoint chain

| stage | script | env | GPUs | input checkpoint(s) | output |
|---|---|---|---|---|---|
| 1. VQ-VAE finetune | `stage1_train_vqvae.sh` | `ego4o` | **exactly 1** (code hardcodes `.cuda()`) | TLControl VQ-VAE init: `/local/home/dhollidt/data/ego4o_weights/TLControl/save_weights_vq/best_model_epoch_hml_emaReset.pth` (path inside `EgoOmniMocap/configs/nymeria_vqvae/train_nymeria_vqvae_4096_64_hml.py`) | `EgoOmniMocap/work_dirs/train_nymeria_vqvae_4096_64_hml/best_C-MPJPE_epoch_*.pth` + **`best_vqvae.pth` symlink** (created by the script) |
| 2. LLM pretrain (E_M alignment) | `stage2_pretrain_llm.sh` | `ego4o_llava` | 2 | base LLM `liuhaotian/llava-v1.6-vicuna-7b` (HF cache) + stage-1 VQ-VAE via `$VQVAE_CKPT` (default: the `best_vqvae.pth` symlink) | `llava/checkpoints/ego4o_hml_pretrain/` (full 14 GB model incl. E_M + VQ-VAE weights + config) |
| 3. LoRA finetune | `stage3_finetune_llm.sh` | `ego4o_llava` | 2 | stage-2 output via `$PRETRAIN_DIR` (default `llava/checkpoints/ego4o_hml_pretrain`); VQ-VAE weights ride along inside it | `llava/checkpoints/ego4o_hml_finetune_lora/` (adapter + `non_lora_trainables.bin`; every `checkpoint-N/` too) |
| 4. Eval | `stage4_eval.sh` | `ego4o_llava` | 1 | `$MODEL_PATH` (default: stage-3 dir) merged onto `$MODEL_BASE` (default: stage-2 dir) | `llava/eval_out/test_nymeria_hml_test_<ts>/{metrics,result,...}.json` |

The chain is wired through **environment variables with sensible defaults** — to repoint
a stage at a different checkpoint, set the variable instead of editing the script:

```bash
# examples
VQVAE_CKPT=/path/to/other_vqvae.pth       GPUS=2,3 bash stage2_pretrain_llm.sh
PRETRAIN_DIR=$PWD/llava/checkpoints/my_pretrain GPUS=2,3 bash stage3_finetune_llm.sh
MODEL_PATH=.../ego4o_hml_finetune_lora/checkpoint-3500 MODEL_BASE=.../ego4o_hml_pretrain \
    GPUS=2 bash stage4_eval.sh
DATASET_DIR=/path/to/other_dataset ...    # any LLM stage; default /local/home/dhollidt/data/ego4o_nymeria
MASTER_PORT=29600 ...                     # if the default deepspeed port is taken
WANDB_MODE=offline ...                    # disable wandb syncing (login is used by default)
```

## Training the 4-category model (second dataset)

The 4-category model (atomic actions + hands/arms + legs/feet + body posture, for the
head-to-head against MotionGPT3) reuses **stage 1 unchanged** — both ego4o models share
the frozen `best_vqvae.pth`, which keeps the motion tokenizer identical across them — and
re-runs stages 2-4 on `/local/home/dhollidt/data/ego4o_nymeria_4cat`. Build that dataset
first: see [`llava/scripts/ego4o/nymeria_hml/README.md`](llava/scripts/ego4o/nymeria_hml/README.md).

```bash
export LLAVA=$PWD/llava
export DATASET_DIR=/local/home/dhollidt/data/ego4o_nymeria_4cat

# Stage 2 — motion<->text alignment pretrain (E_M only, 1 epoch)
DATASET_DIR=$DATASET_DIR OUTPUT_DIR=./checkpoints/ego4o_4cat_pretrain \
GPUS=2,3 bash llava/scripts/ego4o/hml/stage2_pretrain_llm.sh

# Stage 3 — LoRA finetune. PRETRAIN_DIR is mandatory: without it stage 3 would
# silently finetune the ATOMIC pretrain checkpoint on the new data.
DATASET_DIR=$DATASET_DIR PRETRAIN_DIR=$LLAVA/checkpoints/ego4o_4cat_pretrain \
OUTPUT_DIR=./checkpoints/ego4o_4cat_finetune_lora \
GPUS=2,3 bash llava/scripts/ego4o/hml/stage3_finetune_llm.sh

# Stage 4 — eval with the per-category prompts
MODEL_PATH=$LLAVA/checkpoints/ego4o_4cat_finetune_lora \
MODEL_BASE=$LLAVA/checkpoints/ego4o_4cat_pretrain GPUS=2 \
bash llava/scripts/ego4o/hml/stage4_eval.sh --dataset_dir $DATASET_DIR --per_sample_prompt
```

Then score both models through MotionGPT3 on the identical population, **per category**:

```bash
python -m llava.ego4o.eval.export_predictions_for_mgpt3 --result <save_dir>/result.json
# in MotionGPT3, env mgpt3:
python -m test --cfg configs/test_nymeria_env_me2t_4cat.yaml            # mgpt3 predictions
python find_intersecting_results.py <ego4o predictions_mgpt3.json> <mgpt3 predictions.json> \
       --out results/compare_ego4o/isect_4cat.txt
python -m evaluate_from_prediction --cfg configs/test_nymeria_env_me2t_4cat.yaml \
       --predictions <each> --restrict_keys results/compare_ego4o/isect_4cat.txt \
       --per_category --strict
```

`--per_category` adds a `per_category` block to the metrics json with one entry per
caption type, so the result reads "this good on arms, this good on legs, this good on
atomic actions". Caveat: `M2TMetrics` computes R-precision over retrieval pools of 32
drawn from the scored set, so per-category R-precision is comparable *between models*
but not against the aggregate; Bleu/Rouge/Cider/BertScore are per-sample and unaffected.
ego4o's own `metrics.json` already carries the same breakdown.

**On the prompts.** Each category is evaluated with one fixed question
(`EVAL_QUESTION_BY_TYPE` in `llava/llava/ego4o/constants.py`) that is also one of the
questions the model trained on — the release's convention, where the eval query is
literally `IMAGE_MOTION_TO_TEXT_QUESTION_LIST[0]`. `--per_sample_prompt` is required
here because the three body-part categories are annotated on the same windows, so with
one shared query the model would get identical inputs with three contradictory targets.
Worth one sentence in the writeup: MotionGPT3 trains on 307-335 paraphrases per
category resampled every epoch (that diversity is part of its method) versus ego4o's
27 fixed per row, though neither model is being tested on an unseen instruction.

Notes specific to this run:

- **`OUTPUT_DIR`/`PRETRAIN_DIR` are env vars, not appended flags.** Appending a second
  `--output_dir` via `"$@"` relies on the arg parser's last-wins behaviour; losing that
  race would overwrite the existing atomic checkpoints.
- `--per_sample_prompt` is **required** here. The three body-part categories are annotated
  on the same windows, so with one shared query the model gets identical inputs with three
  contradictory targets. It is off by default so the atomic model still evaluates exactly
  as before.
- Stage 2 discards the question entirely (`--version plain` → `preprocess_plain` replaces
  it with a bare `<motion>` token), so the per-category prompts only matter from stage 3 on.
  It is still re-run, because the caption mix changed.
- **Runtime**: train rows go from 110,441 to 177,741 and the test split from 29,449 to
  54,930, so budget roughly 1.6× the atomic run for stage 3 and 1.9× for stage 4.
- Early stopping still reads only the first 2,048 lines of
  `ego4o_image_motion_val.jsonl` (hardcoded in `make_supervised_data_module`); the builder
  shuffles val with a fixed seed so that slice stays category-representative.

## Picking the "best" stage-3 checkpoint (early stopping)

Stage 3 trains up to 4 epochs, evaluates `eval_loss` on 2,048 val samples every 250 steps,
and stops after 3 evals without improvement. Because `load_best_model_at_end` is broken for
LoRA+DeepSpeed in this transformers version (see README §6), **the final save is the
last step, not necessarily the best**. The best one is recorded in
`llava/checkpoints/ego4o_hml_finetune_lora/checkpoint-<last>/trainer_state.json`
(`best_model_checkpoint` / `best_metric`), and every `checkpoint-N/` contains both the
adapter and `non_lora_trainables.bin`, so it is directly evaluable:

```bash
MODEL_PATH=llava/checkpoints/ego4o_hml_finetune_lora/checkpoint-<BEST> GPUS=2 bash stage4_eval.sh
```

With early stopping (patience 3 × 250 steps) the last and best checkpoints are usually
within noise of each other; evaluate both if in doubt (`--data_range 500` for a quick pass).

## Resume / interruption

- **Stage 1** (mmengine): append `--resume` → `GPUS=2 bash stage1_train_vqvae.sh --resume`
  (auto-picks the latest epoch checkpoint in the work dir).
- **Stages 2/3** (HF trainer): automatic — if `checkpoint-*` dirs exist in the output dir,
  training resumes from the latest one. To restart from scratch, delete the output dir.
  Note stage 2 saves only every 24k steps (≈ once per epoch), so an interrupted pretrain
  usually restarts from 0.
- **`run_all_stages.sh`** skips stages whose final artifact already exists
  (stage 1: `best_vqvae.pth`; stage 2: `model.safetensors.index.json`; stage 3:
  `non_lora_trainables.bin`) and re-enters unfinished ones.

## Adapting hyperparameters

Stay-close-to-paper defaults are baked in; the knobs you might touch:

- **GPU count**: stages 2/3 use `deepspeed --include localhost:$GPUS` — any comma list works.
  Effective batch = `per_device_train_batch_size × #GPUs` (grad-accum is 1). If you change
  the number of GPUs, adjust `--per_device_train_batch_size` (or add
  `--gradient_accumulation_steps`) to keep the effective batch: stages 2 and 3 both target
  **128** (the LLaVA-recipe scale for lr 1e-3 / 2e-4). Stage 1 must stay single-GPU.
- **Batch size / OOM**: defaults are 64/GPU for both LLM stages — measured peak 83 GiB/GPU
  of 143 GiB for the LoRA finetune (the heavier stage), so there is still headroom but no
  need to go higher: stage 1's batch (128) is the authors' value and is update-count-bound,
  not memory-bound (uses ~2.7 GiB). If you hit OOM (e.g. on smaller GPUs), first lower the per-device batch
  + raise grad-accum; as a last resort switch stage 3 to `--deepspeed ./scripts/zero2_offload.json`
  (CPU optimizer offload, ~2× slower — this is what the original release used).
- **Learning rates**: stage 2 `1e-3` (adapter only), stage 3 `2e-4` LoRA + `--mm_projector_lr 2e-5`
  for E_I. Keep these when changing batch sizes moderately (the release did not scale lr either).
- **Epochs / early stopping**: stage 3 `--num_train_epochs 4`, `--early_stopping_patience 3`
  (set `0` to disable), `--eval_steps/--save_steps 250` (keep them equal).
- **LoRA vs full finetune**: to reproduce the *released* script instead of the paper text,
  drop the three `--lora_*` flags + `--mm_projector_lr`, set `--learning_rate 2e-5`, and use
  `--deepspeed ./scripts/zero2_offload.json`. Stage 4 then needs no `--model_base`
  (`MODEL_BASE=""` won't be read; pass `MODEL_PATH` pointing at the full checkpoint).
- **Extra flags**: anything appended to a stage script is forwarded to the underlying
  command (`"$@"`), e.g. `bash stage2_pretrain_llm.sh --num_train_epochs 2`.

## Monitoring & sanity checks

- wandb: stage 1 logs to project `nymeria_vq_vae`; stages 2/3 to the default llava project
  (`--report_to wandb`). Logs also land in `EgoOmniMocap/work_dirs/.../<ts>.log` and the
  deepspeed console output.
- **Stage 1 acceptance**: val `C-MPJPE` should reach the tens-of-mm range (paper's VQ-VAE
  reconstruction ceiling ≈45 mm C-MPJPE). If it plateaus far above that, stop and investigate
  before spending LLM compute. ~860 iters/epoch × 30 epochs, a few hours, <10 GB VRAM.
- **Stage 2**: train loss should fall smoothly from ≈3; ≈860 steps total
  (110k samples / effective batch 128, 1 epoch, well under an hour).
- **Stage 3**: watch `eval_loss` (smoke started ≈1.4 and fell immediately). ≈860 steps/epoch
  at effective batch 128; evals every 250 steps (~1 min each).
- **Stage 4**: full test split (29,449 samples) takes a few hours on one GPU at bs 24;
  use `--data_range 500` for a quick preview. Compare BLEU / BERTScore / ROUGE-L with the
  paper's motion-understanding table.

## Prerequisites recap (already satisfied on this machine)

- Dataset at `/local/home/dhollidt/data/ego4o_nymeria` (see README §2; rerun
  `build_ego4o_jsonl.py` if more frames/texts arrive — e.g. the two frame-poor sequences or
  regenerated grace_randolph texts).
- TLControl VQ-VAE weights at `/local/home/dhollidt/data/ego4o_weights/TLControl/…`.
- Envs `ego4o` / `ego4o_llava` (recipes in `handoff/`; note peft==0.4.0 + the flash-attn
  2.5.8 prebuilt wheel).
- Base LLM + CLIP tower in the HF cache (auto-downloaded on first use otherwise).
