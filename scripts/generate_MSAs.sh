#!/bin/bash

# ENSURE FAMSA ENVIRONMENT VARIABLE IS SET
# NUMBER OF THREADS
THREADS=20

for DATASET in "gh114" "ppat" "cm"
do
    IN_FILE="data/raw/${DATASET}/${DATASET}_family.fasta"
    OUT_FILE="data/processed/${DATASET}/${DATASET}_family.aln.fasta"
    $FAMSA -t $THREADS $IN_FILE $OUT_FILE
done
