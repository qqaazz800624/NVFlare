#!/bin/bash

export PYTHONPATH=$PWD/src:$PYTHONPATH

nvflare simulator \
  -w $PWD/workspace_mednext_225 \
  -c liver,spleen,pancreas,kidney \
  -gpu 0,1,2,3 \
  jobs/condist_mednext