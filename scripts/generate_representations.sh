#!/bin/bash

# SCRIPT TO GENERATE ALL EMBEDDINGS (EXCLUDING EVE)
for DATASET in "gh114" "ppat" "cm"
do
  python src/representations/compute_ct_embeddings.py $DATASET
  python src/representations/compute_esm_if_lls.py $DATASET
  python src/representations/compute_tranception_score.py $DATASET
  python src/representations/generate_esm_embeddings.py $DATASET esm1b
  python src/representations/generate_esm_embeddings.py $DATASET esm2
  python src/representations/generate_esm_if1_embeddings.py $DATASET
  python src/representations/generate_onehot_encodings.py $DATASET
done
