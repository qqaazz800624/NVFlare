#!/bin/bash

export PYTHONPATH=$PWD/src:$PYTHONPATH

nvflare simulator \
  -w $PWD/workspace_test \
  -c liver,spleen,pancreas,kidney \
  -gpu 0,2,1,3 \
  jobs/condist_GA_evidential
