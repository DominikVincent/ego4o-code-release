#!/usr/bin/env bash
# Stage 1 (paper stage C3): finetune the part-aware VQ-VAE (4096 codes / 64 dim)
# on the 30 fps Nymeria HML features. SINGLE GPU ONLY (the VQVAE code hardcodes
# .cuda(); select the physical GPU via GPUS). ~26k iters / 30 epochs, a few hours.
# Env: ego4o (mmpose side).
#   GPUS=2 bash stage1_train_vqvae.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
source "$SCRIPT_DIR/gpu_guard.sh"

GPUS="${GPUS:-2}"
check_gpus_free "$GPUS"
export CUDA_VISIBLE_DEVICES="$GPUS"
# wandb: uses your login by default; set WANDB_MODE=offline to disable syncing

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ego4o

cd "$REPO/EgoOmniMocap"
WORK_DIR="work_dirs/train_nymeria_vqvae_4096_64_hml"
python tools/train.py configs/nymeria_vqvae/train_nymeria_vqvae_4096_64_hml.py \
    --work-dir "$WORK_DIR" "$@"

# stable name for the best checkpoint — the LLM stages point at this symlink
BEST=$(ls -t "$WORK_DIR"/best_C-MPJPE_epoch_*.pth | head -1)
ln -sf "$(basename "$BEST")" "$WORK_DIR/best_vqvae.pth"
echo "Best VQ-VAE: $BEST  (symlinked as $WORK_DIR/best_vqvae.pth)"
