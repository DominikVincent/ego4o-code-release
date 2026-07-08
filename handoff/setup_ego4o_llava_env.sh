#!/usr/bin/env bash
# Builds the `ego4o_llava` conda env (LLM / stage-3 side of the Ego4o repro).
# Per llava/pyproject.toml + handoff C0: torch 2.1.2+cu121, transformers 4.37.2,
# deepspeed 0.12.6, plus the ego4o extras (mmengine/mmcv/mmpose editable, clip,
# scipy, fontTools, natsort). Separate from the `ego4o` env (they conflict).
# NOTE: flash-attn (needed by train_mem_ego4o.py) is NOT installed here — it is
# a long native build; install right before training:  pip install flash-attn --no-build-isolation
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -n ego4o_llava python=3.10 -y
conda activate ego4o_llava

# numpy<2 first: torch 2.1.2 wheels are built against numpy 1.x
pip install "numpy==1.26.4"

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# llava package + pinned deps (transformers 4.37.2, tokenizers 0.15.1, ...)
pip install -e "$REPO/llava"
pip install -e "$REPO/llava[train]"   # deepspeed 0.12.6, ninja, wandb

# mmcv prebuilt wheel (cu121/torch2.1) + mm ecosystem for the ego4o transforms
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
pip install mmengine==0.10.7

# ego4o extras
pip install --no-deps "git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
pip install --no-deps -e "$REPO/EgoOmniMocap"   # mmpose 1.3.0 (model/dataset registry)
pip install scipy fontTools natsort ftfy regex

# mmcv's dep chain may have bumped numpy to 2.x — force it back
pip install "numpy==1.26.4"
# setuptools<81 so pkg_resources exists (mmengine needs it)
pip install "setuptools<81" wheel

python - <<'PY'
import torch, transformers, numpy
print('torch', torch.__version__, '| transformers', transformers.__version__, '| numpy', numpy.__version__)
import llava, mmengine, mmcv, clip  # noqa
print('llava/mmengine/mmcv/clip import OK')
PY
echo "ego4o_llava env ready."
