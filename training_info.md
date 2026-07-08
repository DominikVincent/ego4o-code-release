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

## Picking the "best" stage-3 checkpoint (early stopping)

Stage 3 trains up to 4 epochs, evaluates `eval_loss` on 2,048 val samples every 500 steps,
and stops after 3 evals without improvement. Because `load_best_model_at_end` is broken for
LoRA+DeepSpeed in this transformers version (see README §6), **the final save is the
last step, not necessarily the best**. The best one is recorded in
`llava/checkpoints/ego4o_hml_finetune_lora/checkpoint-<last>/trainer_state.json`
(`best_model_checkpoint` / `best_metric`), and every `checkpoint-N/` contains both the
adapter and `non_lora_trainables.bin`, so it is directly evaluable:

```bash
MODEL_PATH=llava/checkpoints/ego4o_hml_finetune_lora/checkpoint-<BEST> GPUS=2 bash stage4_eval.sh
```

With early stopping (patience 3 × 500 steps) the last and best checkpoints are usually
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
  `--gradient_accumulation_steps`) to keep the effective batch: stage 2 targets 64,
  stage 3 targets 48. Stage 1 must stay single-GPU.
- **Batch size / OOM**: H200s have plenty of headroom at the defaults (pretrain 32/GPU,
  finetune 24/GPU). If you hit OOM (e.g. on smaller GPUs), first lower the per-device batch
  + raise grad-accum; as a last resort switch stage 3 to `--deepspeed ./scripts/zero2_offload.json`
  (CPU optimizer offload, ~2× slower — this is what the original release used).
- **Learning rates**: stage 2 `1e-3` (adapter only), stage 3 `2e-4` LoRA + `--mm_projector_lr 2e-5`
  for E_I. Keep these when changing batch sizes moderately (the release did not scale lr either).
- **Epochs / early stopping**: stage 3 `--num_train_epochs 4`, `--early_stopping_patience 3`
  (set `0` to disable), `--eval_steps/--save_steps 500` (keep them equal).
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
- **Stage 2**: train loss should fall smoothly from ≈3 (smoke: 94 samples/s at bs 32×2 ⇒
  ≈20 min/epoch over 110k samples; 1 epoch total).
- **Stage 3**: watch `eval_loss` (smoke started ≈1.4 and fell immediately). ≈2,300 steps/epoch
  at effective batch 48; evals every 500 steps.
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
