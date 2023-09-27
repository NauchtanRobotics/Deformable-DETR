#!/usr/bin/env bash

set -x

EXP_DIR=/home/qninhdt/code/Deformable-DETR-lmao/experiments
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --resume ${EXP_DIR}/checkpoint.pth \
    ${PY_ARGS}