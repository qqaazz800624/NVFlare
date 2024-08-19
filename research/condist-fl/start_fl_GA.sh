#!/bin/bash

export PYTHONPATH=$PWD/src:$PYTHONPATH

nvflare simulator \
  -w $PWD/workspace_GA \
  -c liver,spleen,pancreas,kidney \
  -gpu 1,2,3,4 \
  jobs/condist_GA
