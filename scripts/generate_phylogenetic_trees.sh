#!/bin/bash

# SCRIPT TO CREATE PHYLOGENETIC TREE BASED ON FAMILY-WIDE MSA.
# REQUIRES csvtk (https://bioinf.shenwei.me/csvtk/), FastTree (www.microbesonline.org/fasttree/),
# and SeqKit (https://bioinf.shenwei.me/seqkit/).
# For the visualization, requires ETE Toolkit (http://etetoolkit.org/)

DATASET=$1

# EXTRACT SEQUENCE IDS
cat "data/interim/${DATASET}/${DATASET}.csv" | csvtk cut -f name | csvtk del-header > \
  "data/interim/${DATASET}/${DATASET}_ids.txt"

# EXTRACT SUBSET OF MSA
seqkit grep --pattern-file  "data/interim/${DATASET}/${DATASET}_ids.txt" \
  "data/raw/${DATASET}/${DATASET}_family.aln.fasta" > "data/interim/${DATASET}/${DATASET}_local.aln.fasta"

# GENERATE TREE
FastTree <"data/interim/${DATASET}/${DATASET}_local.aln.fasta"> "data/interim/${DATASET}/${DATASET}.tree"

python src/visualization/visualize_phylo_tree.py --dataset "$1"



