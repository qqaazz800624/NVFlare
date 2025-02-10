#!/bin/bash

export PYTHONPATH=$PWD/src:$PYTHONPATH

nvflare simulator \
  -w $PWD/workspace_non_ConDist_mednext \
  -c liver,spleen,pancreas,kidney \
  -gpu 0,1,2,3 \
  jobs/non_condist_mednext
