#!/usr/bin/env bash
# Stage 3 (paper stage C5b): multi-modal finetune (image + GT motion + text).
# LoRA r=128 alpha=256 on the LLM (paper §3.3.2); E_I (mm_projector) and E_M
# (vq_net_postprocess) train fully; CLIP tower + VQ-VAE frozen. Env: ego4o_llava.
# 4 epochs with early stopping on val loss (patience 3 evals @ every 500 steps).
#   GPUS=2,3 bash stage3_finetune_llm.sh
# Deviations from release documented in README (release script did full FT, 2 epochs).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAVA="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/gpu_guard.sh"

GPUS="${GPUS:-2,3}"
check_gpus_free "$GPUS"
# wandb: uses your login by default; set WANDB_MODE=offline to disable syncing

PRETRAIN_DIR="${PRETRAIN_DIR:-$LLAVA/checkpoints/ego4o_hml_pretrain}"
[ -d "$PRETRAIN_DIR" ] || { echo "ABORT: pretrain checkpoint not found: $PRETRAIN_DIR — run stage2 first." >&2; exit 1; }
DATASET_DIR="${DATASET_DIR:-/local/home/dhollidt/data/ego4o_nymeria}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ego4o_llava

cd "$LLAVA"
deepspeed --include "localhost:$GPUS" --master_port "${MASTER_PORT:-29512}" \
    llava/ego4o/train/train_mem_ego4o.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --motion_token_loss_weight 1 \
    --load_motion_encoder False \
    --freeze_motion_encoder True \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "$PRETRAIN_DIR" \
    --version v1 \
    --dataset_dir "$DATASET_DIR" \
    --data_path "$DATASET_DIR/ego4o_image_motion_train.jsonl" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --tf32 True \
    --output_dir ./checkpoints/ego4o_hml_finetune_lora \
    --num_train_epochs 4 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --early_stopping_patience 3 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    "$@"
