#!/bin/bash

# Set default CUDA device to use
DEFAULT_CUDA_DEVICE="1"

# Allow overriding the default CUDA device with a command-line argument
CUDA_DEVICE=${1:-$DEFAULT_CUDA_DEVICE}

# Set the CUDA device to use
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

python run_validate.py \
     -w workspace_mednext_225 \
     -o cross_site_validate_mednext.json $@