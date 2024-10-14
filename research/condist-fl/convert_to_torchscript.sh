#!/bin/bash

# Set default CUDA device to use
DEFAULT_CUDA_DEVICE="0"

# Allow overriding the default CUDA device with a command-line argument
CUDA_DEVICE=${1:-$DEFAULT_CUDA_DEVICE}

# Set the CUDA device to use
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE


# python convert_to_torchscript.py \
#   --config workspace/simulate_job/app_pancreas/config/config_task.json \
#   --weights workspace/simulate_job/app_pancreas/models/best_model.pt \
#   --app pancreas \
#   --output best_pancreas_model.pt


python convert_to_torchscript.py \
  --config workspace_GA_mednext/simulate_job/app_pancreas/config/config_task.json \
  --weights workspace_GA_mednext/simulate_job/app_pancreas/models/best_model.pt \
  --app pancreas \
  --output best_pancreas_model.pt  $@