#!/bin/bash

# SCRIPT TO CREATE PHYLOGENETIC TREE BASED ON FAMILY-WIDE MSA

DATASET="PPAT"
THREADS=20

MIN_SEQ_ID=0.5
mmseqs easy-cluster "data/raw/${DATASET}/${DATASET}.fasta" "data/interim/${DATASET}/${DATASET}_clust" \
  tmp --min-seq-id $MIN_SEQ_ID -c 0.8 --threads $THREADS
rm -rf tmp
