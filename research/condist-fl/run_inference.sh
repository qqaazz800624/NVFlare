#!/bin/bash

python run_infer.py \
  --data_root /neodata/open_dataset/ConDistFL/data/Liver \
  --data_list /neodata/open_dataset/ConDistFL/data/Liver/datalist.json \
  --data_list_key testing \
  --model best_pancreas_model.pt \
  --output infer_GA_mednext