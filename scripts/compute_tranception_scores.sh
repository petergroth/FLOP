#!/bin/bash

# Assuming Tranception has been cloned to current directory
export PYTHONPATH="Tranception:$PYTHONPATH"
for DATASET in "gh114" "ppat" "cm"
do
  python src/representations/compute_tranception_scores.py $DATASET
done
