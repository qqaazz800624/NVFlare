#!/bin/bash

export PYTHONPATH=$PWD/src:$PYTHONPATH

nvflare simulator \
  -w $PWD/workspace_non_ConDist_non_GA \
  -c liver,spleen,pancreas,kidney \
  -gpu 0,2,4,6 \
  jobs/condist_mednext
