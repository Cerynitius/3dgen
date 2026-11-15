#!/bin/bash
# Walk server: walk + FastFix3D on --walk-gpu (default 0), tiles built on --build-gpus.
# FAM3DGS_ENV: script that sets up the CUDA / gsplat environment; WALK_PY: the Python to run with.
source "${FAM3DGS_ENV:-$HOME/fam3dgs/env-dev.sh}"
export PYTHONPATH=${WALK_PYTHONPATH:-$HOME/fam3dgs/stubs} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${WALK_CUDA_DEVICES:-0,1,2}
cd "$(dirname "$0")"
exec "${WALK_PY:-$HOME/fam3dgs/venv-difix/bin/python}" server.py --port "${PORT:-8095}" "$@"
