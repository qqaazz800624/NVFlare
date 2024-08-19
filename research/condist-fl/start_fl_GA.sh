#!/bin/bash

export PYTHONPATH=$PWD/src:$PYTHONPATH

nvflare simulator \
  -w $PWD/workspace \
  -c liver,spleen,pancreas,kidney \
  -gpu 1,2,3,4 \
  jobs/condist_GA
