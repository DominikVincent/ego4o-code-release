#!/usr/bin/env bash
# Stage 4: motion-understanding eval on the test split (BLEU / BERTScore / ROUGE).
# Loads the LoRA checkpoint merged onto the pretrain base. Env: ego4o_llava, 1 GPU.
#   GPUS=2 bash stage4_eval.sh                     # full test split (29,449)
#   GPUS=2 bash stage4_eval.sh --data_range 100    # quick subsampled run
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAVA="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/gpu_guard.sh"

GPUS="${GPUS:-2}"
check_gpus_free "$GPUS"
export CUDA_VISIBLE_DEVICES="$GPUS"

MODEL_PATH="${MODEL_PATH:-$LLAVA/checkpoints/ego4o_hml_finetune_lora}"
MODEL_BASE="${MODEL_BASE:-$LLAVA/checkpoints/ego4o_hml_pretrain}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ego4o_llava

cd "$LLAVA"
python -m llava.ego4o.eval.test_ego4o_hml_batch \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --split test \
    "$@"
