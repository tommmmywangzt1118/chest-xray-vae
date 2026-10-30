#!/bin/bash
# Create a private conda env (by path, not in the shared base) and download the dataset.
# Usage: ENV=$HOME/envs/vae bash setup_env.sh
set -e
ENV=${ENV:-$HOME/envs/vae}

conda create -y -p "$ENV" python=3.11
# cu128 wheels are needed for Blackwell GPUs; use another index URL for older CUDA drivers.
"$ENV/bin/pip" install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128
"$ENV/bin/pip" install --no-cache-dir -r requirements.txt
"$ENV/bin/python" -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Kaggle chest X-ray dataset (~2.3 GB). kagglehub prints the local path of the extracted files.
"$ENV/bin/python" -c "import kagglehub; print(kagglehub.dataset_download('paultimothymooney/chest-xray-pneumonia'))"
