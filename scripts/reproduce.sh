#!/bin/bash
THREADS=20
# SCRIPT TO REPRODUCE RESULTS IN PAPER.
# ASSUMES ALL EMBEDDINGS ARE CREATED AND THAT DATASET HAS BEEN SPLIT

# Main results
python src/training/fit_model.py --dataset gh114 --low_n --threads $THREADS
python src/training/fit_model.py --dataset cm --threads $THREADS
python src/training/fit_model.py --dataset ppat --threads $THREADS

# Ablation results
python src/training/fit_model.py --dataset gh114 --low_n --threads $THREADS --ablation
python src/training/fit_model.py --dataset cm --threads $THREADS --ablation
python src/training/fit_model.py --dataset ppat --threads $THREADS --ablation

# Classification for CM (appendix)
python src/training/fit_model.py --dataset cm --threads $THREADS --task classification
