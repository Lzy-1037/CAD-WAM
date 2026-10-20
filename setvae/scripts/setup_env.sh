#!/usr/bin/env bash
#
#
# Everything lives inside the conda env: no sudo, no change to the NVIDIA driver, system CUDA,
# /usr/local/cuda, system Python or any other env.  The CUDA compiler comes from conda (cuda-nvcc 12.6,
# matching torch's cu126 wheels) and the extension is compiled for the GPUs' own compute capability.
# Mirrors: if conda / pip downloads are slow, point them at your usual mirror or proxy first.
set -euo pipefail
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
PROJ="${PROJ:-$REMOTE/code/SetVAE}"
ENV_NAME="${ENV_NAME:-setvae}"
CHECK_GPU="${CHECK_GPU:-4}"                   # a free card for the build-arch probe and smoke test

source "${CONDA_SH}"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo ">> env ${ENV_NAME} already exists: reusing it (not recreating)"
else
    echo ">> creating env ${ENV_NAME}"
    conda create -y -n "${ENV_NAME}" -c nvidia -c conda-forge \
        python=3.10 cuda-nvcc=12.6 cuda-cudart-dev=12.6 cuda-version=12.6 \
        gcc_linux-64=12.4 gxx_linux-64=12.4 ninja
fi
set +u; conda activate "${ENV_NAME}"; set -u      # conda activation hooks reference unset variables

echo ">> python packages (versions as in the local setvae env)"
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
pip install numpy==1.26.4 scipy==1.15.3 matplotlib==3.10.9 tensorboard==2.21.0 einops==0.8.2 \
    pandas==2.3.3 tqdm opencv-python-headless==4.11.0.86 deepspeed==0.19.5 \
    scikit-learn==1.7.2 imageio==2.37.4 six==1.17.0

echo ">> GPU check"
CUDA_VISIBLE_DEVICES="${CHECK_GPU}" python - <<'PY'
import torch
assert torch.cuda.is_available(), 'torch cannot see a GPU'
print('torch', torch.__version__, '| cuda', torch.version.cuda, '|', torch.cuda.get_device_name(0),
      '| capability', torch.cuda.get_device_capability(0), '| torch archs', torch.cuda.get_arch_list())
PY
ARCH=$(CUDA_VISIBLE_DEVICES="${CHECK_GPU}" python -c "import torch; print('%d.%d' % torch.cuda.get_device_capability(0))")

echo ">> building StructuralLosses for compute capability ${ARCH}"
cd "${PROJ}"
TORCH_CUDA_ARCH_LIST="${ARCH}" bash metrics/pytorch_structural_losses/build_ext.sh

echo ">> smoke test (model + compiled Chamfer + EMD, forward and backward)"
CUDA_VISIBLE_DEVICES="${CHECK_GPU}" python smoke_test.py
echo ">> environment ready: conda activate ${ENV_NAME}"
