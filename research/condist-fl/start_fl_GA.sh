#!/bin/bash

export PYTHONPATH=$PWD/src:$PYTHONPATH

nvflare simulator \
  -w $PWD/workspace_GA_test \
  -c liver,spleen,pancreas,kidney \
  -gpu 1,0,2,3 \
  jobs/condist_GA
