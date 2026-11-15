#!/bin/bash
# Start the walk server: walk + FastFix3D on GPU 0 (PRO 6000), tile baking on GPUs 1, 2.
source ~/fam3dgs/env-dev.sh
export PYTHONPATH=$HOME/fam3dgs/stubs PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2
cd "$(dirname "$0")"
exec ~/fam3dgs/venv-difix/bin/python server.py --port "${PORT:-8095}" "$@"
