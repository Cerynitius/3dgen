#!/bin/bash
# Image generator (Qwen-Image-2.1-Turbo FP8) on one 24 GB card.
# IMGGEN_ENV: the conda env with diffusers / torchao; ASSETGEN_GPU: the card (index or UUID).
E=${IMGGEN_ENV:-$HOME/miniconda3/envs/imggen21}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${ASSETGEN_GPU:-GPU-fa585f6b-5370-e4df-92d4-49ba460da3ad}
export CUDA_HOME=$E/lib/python3.12/site-packages/nvidia/cu13 PATH=$E/lib/python3.12/site-packages/nvidia/cu13/bin:$E/bin:$PATH
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
cd "$(dirname "$0")"
exec $E/bin/python assetgen.py "$@"
