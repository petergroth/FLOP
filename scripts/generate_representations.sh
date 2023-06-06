#!/bin/bash

# SCRIPT TO GENERATE ALL EMBEDDINGS (EXCLUDING EVE)
for DATASET in "gh114" "ppat" "cm"
do
  python src/representations/generate_ct_representations.py $DATASET
  python src/representations/compute_esm_if1_lls.py $DATASET
  python src/representations/generate_esm_representations.py $DATASET esm_1b
  python src/representations/generate_esm_representations.py $DATASET esm_2
  python src/representations/generate_esm_if1_representations.py $DATASET
  python src/representations/generate_onehot_representations.py $DATASET
  python src/representations/generate_mifst_representations.py $DATASET
done
