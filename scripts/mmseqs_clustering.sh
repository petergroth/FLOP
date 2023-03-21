#!/bin/bash

# SCRIPT TO CREATE PHYLOGENETIC TREE BASED ON FAMILY-WIDE MSA

DATASET=$1
THREADS=20

if [ "${DATASET}" = "gh114" ]; then
  MIN_SEQ_ID=0.40
elif [ "${DATASET}" = "cm" ]; then
  MIN_SEQ_ID=0.1
elif [ "${DATASET}" = "ppat" ]; then
  MIN_SEQ_ID=0.5
fi

mmseqs easy-cluster "data/raw/${DATASET}/${DATASET}.fasta" "data/interim/${DATASET}/${DATASET}_clust" \
  tmp --min-seq-id $MIN_SEQ_ID -c 0.8 --threads $THREADS
rm -rf tmp
