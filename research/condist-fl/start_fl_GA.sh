#!/bin/bash

export PYTHONPATH=$PWD/src:$PYTHONPATH

nvflare simulator \
  -w $PWD/workspace_GA_test \
  -c liver,spleen,pancreas,kidney \
  -gpu 4,5,6,7 \
  jobs/condist_GA
