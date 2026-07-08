#!/usr/bin/env bash
# Runs all Ego4o GT-motion training stages sequentially:
#   stage1 VQ-VAE -> stage2 LLM pretrain -> stage3 LoRA finetune -> stage4 eval
# Completed stages are skipped (detected via their output artifacts), so the
# script is safe to re-run after an interruption (stages 2/3 additionally
# auto-resume from their own checkpoint-* dirs via the HF trainer).
#
#   GPUS_SINGLE=2 GPUS_DUAL=2,3 bash run_all_stages.sh
#
# See training_info.md (repo root) for how to adapt stages / repoint checkpoints.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAVA="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO="$(cd "$LLAVA/.." && pwd)"

GPUS_SINGLE="${GPUS_SINGLE:-2}"     # stage 1 (VQ-VAE) + stage 4 (eval): one GPU
GPUS_DUAL="${GPUS_DUAL:-2,3}"       # stages 2/3 (LLM): up to two GPUs

VQVAE_BEST="$REPO/EgoOmniMocap/work_dirs/train_nymeria_vqvae_4096_64_hml/best_vqvae.pth"
PRETRAIN_OUT="$LLAVA/checkpoints/ego4o_hml_pretrain"
FINETUNE_OUT="$LLAVA/checkpoints/ego4o_hml_finetune_lora"

echo "=== Stage 1/4: VQ-VAE finetune (GPU $GPUS_SINGLE) ==="
if [ -e "$VQVAE_BEST" ]; then
    echo "skip: $VQVAE_BEST exists"
else
    GPUS="$GPUS_SINGLE" bash "$SCRIPT_DIR/stage1_train_vqvae.sh"
fi

echo "=== Stage 2/4: LLM motion<->text pretrain (GPUs $GPUS_DUAL) ==="
if [ -f "$PRETRAIN_OUT/model.safetensors.index.json" ]; then
    echo "skip: $PRETRAIN_OUT is complete"
else
    GPUS="$GPUS_DUAL" bash "$SCRIPT_DIR/stage2_pretrain_llm.sh"
fi

echo "=== Stage 3/4: LoRA multi-modal finetune (GPUs $GPUS_DUAL) ==="
if [ -f "$FINETUNE_OUT/non_lora_trainables.bin" ]; then
    echo "skip: $FINETUNE_OUT is complete"
else
    GPUS="$GPUS_DUAL" bash "$SCRIPT_DIR/stage3_finetune_llm.sh"
fi

echo "=== Stage 4/4: test-split eval (GPU $GPUS_SINGLE) ==="
GPUS="$GPUS_SINGLE" bash "$SCRIPT_DIR/stage4_eval.sh"

echo "=== All stages done. Metrics: see the newest dir under $LLAVA/eval_out/ ==="
