#!/bin/bash

export PYTHONPATH=$PWD/src:$PYTHONPATH

nvflare simulator \
  -w $PWD/workspace_GA_evidential \
  -c liver,spleen,pancreas,kidney \
  -gpu 3,0,1,2 \
  jobs/condist_GA_evidential
