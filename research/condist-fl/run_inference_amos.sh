#!/bin/bash

# set default CUDA device
DEFAULT_CUDA_DEVICE="0"   # 0, 1, 2, 3

# allow overriding the default CUDA device with a command-line argument
CUDA_DEVICE=${2:-$DEFAULT_CUDA_DEVICE}

export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}

# set default model name to evaluate
DEFAULT_MODEL_NAME="global"   # kidney, liver, pancreas, spleen, global

# allow overriding the default model name with a command-line argument
MODEL_NAME=${1:-$DEFAULT_MODEL_NAME}


python run_infer.py \
  --data_root /data2/open_dataset/AMOS22 \
  --data_list /data2/open_dataset/AMOS22/amos22_ct.json \
  --data_list_key training \
  --model best_${MODEL_NAME}_model_GA.pt \
  --output infer_GA_amos \
  --model_type MedNeXt   # nnUNet or MedNeXt


  python run_infer.py \
  --data_root /data2/open_dataset/AMOS22 \
  --data_list /data2/open_dataset/AMOS22/amos22_ct.json \
  --data_list_key validation \
  --model best_${MODEL_NAME}_model_GA.pt \
  --output infer_GA_amos \
  --model_type MedNeXt   # nnUNet or MedNeXt