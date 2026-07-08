#!/usr/bin/env bash
# Reproduces the `ego4o` conda env (mmpose side: VQ-VAE + multimodal encoder).
# Rebuilt from handoff/ego4o_pip_freeze.txt. CPU-only install; safe to run without a GPU.
# NOTE: this is the mmpose-side env only. The llava/LLM stage 3 needs a SEPARATE env
#       (torch2.1.2 + transformers==4.37.2 + deepspeed==0.12.6, see llava/pyproject.toml);
#       the two conflict (mmcv vs transformers/deepspeed) and must not share one env.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"

# 1) Fresh env
conda create -n ego4o python=3.10 -y
conda activate ego4o

# 2) Pin build-time deps first (chumpy/xtcocotools need numpy<2 + Cython at build)
pip install numpy==1.23.5 Cython==3.2.8

# 3) Torch stack from the cu121 wheel index (triton 2.1.0 comes as a torch dep)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 4) mmcv 2.1.0 — openmmlab prebuilt wheel matching torch2.1/cu121 (avoids ~30min source build)
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

# 5) Bulk of the frozen deps, exact pins, --no-deps (a freeze is complete, so no resolution needed).
#    This also pins numpy back to 1.23.5 after mmcv's deps bumped it to 2.x.
#    ego4o_bulk_reqs.txt = ego4o_pip_freeze.txt minus torch/torchvision/triton/mmcv,
#    the git/editable lines (clip, imuposer, mmpose), and the local pypangolin wheel.
pip install --no-deps -r "$REPO/handoff/ego4o_bulk_reqs.txt"

# 6) Editable / git packages (all --no-deps)
pip install --no-deps "git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
pip install --no-deps "git+https://github.com/FIGLAB/IMUPoser@a0e9d33b9c57d6737d711185b62084e17684b0e2#subdirectory=src"
pip install --no-deps -e "$REPO/EgoOmniMocap"

# 7) setuptools<81 so pkg_resources still exists (mmengine.get_installed_path needs it).
#    setuptools 81+ removed the bundled pkg_resources.
pip install "setuptools<81" wheel

# Skipped vs the original freeze:
#   pypangolin @ file://... local wheel (open3d/pangolin viz only; not importable path on this server)
echo "ego4o env ready. Verify: conda run -n ego4o python -c 'from mmpose.utils import register_all_modules; register_all_modules(); print(\"ok\")'"
