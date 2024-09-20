#!/bin/bash

export PYTHONPATH=$PWD/src:$PYTHONPATH

nvflare simulator \
  -w $PWD/workspace_GA_condist \
  -c liver,spleen,pancreas,kidney \
  -gpu 2,1,0,3 \
  jobs/condist_GA
