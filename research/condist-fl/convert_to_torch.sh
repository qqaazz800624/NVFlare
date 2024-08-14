#!/bin/bash

# For server checkpoint
python convert_to_torchscript.py \
  --config workspace/simulate_job/app_pancreas/config/config_task.json \
  --weights workspace/simulate_job/app_pancreas/models/best_model.pt \
  --app pancreas \
  --output best_pancreas_model.pt