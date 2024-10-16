#!/bin/bash

# set default CUDA device
DEFAULT_CUDA_DEVICE="1"   # 0, 1, 2, 3

# allow overriding the default CUDA device with a command-line argument
CUDA_DEVICE=${2:-$DEFAULT_CUDA_DEVICE}

export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}

# set default directory for data
DEFAULT_DATA_ROOT="Pancreas"  # Pancreas, Liver, KiTS19, Spleen

# allow overriding the default data directory with a command-line argument
DATA_ROOT=${3:-$DEFAULT_DATA_ROOT}

# set default model name to evaluate
DEFAULT_MODEL_NAME="pancreas"   # pancreas, liver, kidney, spleen

# allow overriding the default model name with a command-line argument
MODEL_NAME=${1:-$DEFAULT_MODEL_NAME}


python run_infer.py \
  --data_root /neodata/open_dataset/ConDistFL/data/${DATA_ROOT} \
  --data_list /neodata/open_dataset/ConDistFL/data/${DATA_ROOT}/datalist.json \
  --data_list_key testing \
  --model best_${MODEL_NAME}_model.pt \
  --output infer \
  --model_type nnUNet   # nnUNet or MedNeXt

python run_infer.py \
  --data_root /neodata/open_dataset/ConDistFL/data/${DATA_ROOT} \
  --data_list /neodata/open_dataset/ConDistFL/data/${DATA_ROOT}/datalist.json \
  --data_list_key testing \
  --model best_${MODEL_NAME}_model_GA.pt \
  --output infer_GA \
  --model_type nnUNet   # nnUNet or MedNeXt