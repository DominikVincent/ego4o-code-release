#!/usr/bin/env bash
# Stage 2 (paper stage C5a): motion<->text alignment pretrain. Trains ONLY the
# motion MLP adapter (E_M / vq_net_postprocess); LLM+vision+VQ-VAE frozen.
# Data: motion->text questions only (version plain). Env: ego4o_llava.
#   GPUS=2,3 bash stage2_pretrain_llm.sh
# Deviation from release: per-device batch 32 (was 16) for the H200s; lr kept.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAVA="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO="$(cd "$LLAVA/.." && pwd)"
source "$SCRIPT_DIR/gpu_guard.sh"

GPUS="${GPUS:-2,3}"
check_gpus_free "$GPUS"
# wandb: uses your login by default; set WANDB_MODE=offline to disable syncing

VQVAE_CKPT="${VQVAE_CKPT:-$REPO/EgoOmniMocap/work_dirs/train_nymeria_vqvae_4096_64_hml/best_vqvae.pth}"
[ -e "$VQVAE_CKPT" ] || { echo "ABORT: VQ-VAE checkpoint not found: $VQVAE_CKPT — run stage1 first." >&2; exit 1; }
DATASET_DIR="${DATASET_DIR:-/local/home/dhollidt/data/ego4o_nymeria}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ego4o_llava

cd "$LLAVA"
deepspeed --include "localhost:$GPUS" --master_port "${MASTER_PORT:-29511}" \
    llava/ego4o/train/train_mem_ego4o.py \
    --freeze_motion_encoder True \
    --is_pretrain True \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path liuhaotian/llava-v1.6-vicuna-7b \
    --version plain \
    --dataset_dir "$DATASET_DIR" \
    --data_path "$DATASET_DIR/ego4o_motion_text_train.jsonl" \
    --pretrained_vqvae_path "$VQVAE_CKPT" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter False \
    --tune_motion_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --tf32 True \
    --output_dir ./checkpoints/ego4o_hml_pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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
