#!/bin/bash

# Set default CUDA device to use
DEFAULT_CUDA_DEVICE="0"

# Allow overriding the default CUDA device with a command-line argument
CUDA_DEVICE=${2:-$DEFAULT_CUDA_DEVICE}

# Set the CUDA device to use
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# Set default site to evaluate
DEFAULT_SITE="kidney"     # server, kidney, liver, pancreas, spleen

# Allow overriding the default site with a command-line argument
SITE=${1:-$DEFAULT_SITE}

# Shift positional parameters so that $@ does not include $1 and $2
shift 2

if [ "$SITE" = "server" ]; then
  echo "Processing server checkpoint"

  # For server checkpoint
  python convert_to_torchscript.py \
    --config workspace/simulate_job/app_${SITE}/config/config_fed_server.json \
    --weights workspace/simulate_job/app_${SITE}/best_FL_global_model.pt \
    --app "$SITE" \
    --output best_global_model.pt "$@"  

  python convert_to_torchscript.py \
  --config workspace_GA/simulate_job/app_${SITE}/config/config_fed_server.json \
  --weights workspace_GA/simulate_job/app_${SITE}/best_FL_global_model.pt \
  --app "$SITE" \
  --output best_global_model_GA.pt "$@"

else
  echo "Processing client checkpoint"

  # For client checkpoint
  python convert_to_torchscript.py \
    --config workspace/simulate_job/app_${SITE}/config/config_task.json \
    --weights workspace/simulate_job/app_${SITE}/models/best_model.pt \
    --app "$SITE" \
    --output best_${SITE}_model.pt "$@"

  python convert_to_torchscript.py \
    --config workspace_GA/simulate_job/app_${SITE}/config/config_task.json \
    --weights workspace_GA/simulate_job/app_${SITE}/models/best_model.pt \
    --app "$SITE" \
    --output best_${SITE}_model_GA.pt "$@"
fi
