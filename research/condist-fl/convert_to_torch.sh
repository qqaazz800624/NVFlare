#!/bin/bash

# For server checkpoint
python convert_to_torchscript.py \
  --config workspace/simulate_job/app_server/config/config_fed_server.json \
  --weights workspace/simulate_job/app_server/best_FL_global_model.pt \
  --app server \
  --output best_FL_global_model.pt