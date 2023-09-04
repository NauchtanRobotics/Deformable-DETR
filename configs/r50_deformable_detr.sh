#!/usr/bin/env bash

set -x

EXP_DIR=/content/drive/MyDrive/ml-data/r50_deformable_detr
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --resume ${EXP_DIR}/checkpoint.pth \
    ${PY_ARGS}
