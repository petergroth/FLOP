#!/bin/bash
THREADS=20
python src/data/compile_cm.py --log --force --visualize --initial_threshold 0.3 --active --force_graphpart \
  --threads $THREADS

python src/data/compile_gh114.py --target target_4 --log --force --visualize --initial_threshold 0.3 --force_graphpart \
  --threads $THREADS

python src/data/compile_ppat.py --log --force --visualize --initial_threshold 0.3 --force_graphpart \
  --threads $THREADS