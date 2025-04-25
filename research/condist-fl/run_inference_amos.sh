#!/bin/bash

# Default CUDA device and model name
DEFAULT_CUDA_DEVICE="0"           # Available options: 0, 1, 2, 3
DEFAULT_MODEL_NAME="spleen"       # Available options: kidney, liver, pancreas, spleen, global

# Parse arguments: model name and CUDA device
MODEL_NAME=${1:-$DEFAULT_MODEL_NAME}   # First argument: model name
CUDA_DEVICE=${2:-$DEFAULT_CUDA_DEVICE} # Second argument: CUDA device

# Export CUDA environment variable
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}

# Display configuration
echo "========================================"
echo "Model Name:        ${MODEL_NAME}"
echo "CUDA Device:       ${CUDA_DEVICE}"
echo "========================================"

# Processing training data
echo "Processing training data..."
python run_infer.py \
  --data_root /data2/open_dataset/AMOS22 \
  --data_list /data2/open_dataset/AMOS22/amos22_ct.json \
  --data_list_key training \
  --model best_${MODEL_NAME}_model.pt \
  --output infer_amos \
  --model_type MedNeXt   # Options: nnUNet or MedNeXt

# Processing validation data
echo "Processing validation data..."
python run_infer.py \
  --data_root /data2/open_dataset/AMOS22 \
  --data_list /data2/open_dataset/AMOS22/amos22_ct.json \
  --data_list_key validation \
  --model best_${MODEL_NAME}_model.pt \
  --output infer_amos \
  --model_type MedNeXt   # Options: nnUNet or MedNeXt

echo "Inference completed!"
