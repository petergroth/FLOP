#!/bin/bash


# Compute sequence identity matrix for all datasets
$FAMSA -dist_export -pid -square_matrix data/raw/ppat/ppat.fasta data/interim/ppat/all_vs_all.csv
$FAMSA -dist_export -pid -square_matrix data/raw/gh114/gh114.fasta data/interim/gh114/all_vs_all.csv
$FAMSA -dist_export -pid -square_matrix data/raw/cm/cm.fasta data/interim/cm/all_vs_all.csv

python src/data/data_utils.py

