#!/bin/bash

export PYTHONPATH=$PWD/src:$PYTHONPATH

nvflare simulator \
  -w $PWD/workspace_GA_new_mednext \
  -c liver,spleen,pancreas,kidney \
  -gpu 0,1,2,3 \
  jobs/condist_GA_new
