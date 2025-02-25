#!/usr/bin/env bash
cd ..
python ./train_dqn.py \
    --device cuda:$1
    --batch-size 256 \
    --buffer-size 10000 \
    --total-timesteps 300000 \
    ${@:2}