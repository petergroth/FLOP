#!/bin/bash

METRIC="spearman"
IMAGE_FORMAT="pdf"
REGRESSOR="Ridge"

# GENERATE PLOTS WITH ALL THREE REGRESSORS FOR EACH DATASET
for DATASET in "gh114" "ppat" "cm"
do
    python src/process_results.py \
      --dataset $DATASET \
      --metric $METRIC \
      --image_format $IMAGE_FORMAT \
      --save_csv \
      --bar_plot

    for ABLATION in "holdout" "random"
    do
      python src/visualization/generate_additional_ablation_figures.py \
      --dataset $DATASET \
      --metric $METRIC \
      --image_format $IMAGE_FORMAT \
      --ablation_method $ABLATION
    done
done

# GENERATE COMBINED PLOT (USED IN MANUSCRIPT)
python src/visualize_results.py --regressor $REGRESSOR --image_format $IMAGE_FORMAT --metric $METRIC \
  --unsupervised
python src/visualize_results.py --regressor $REGRESSOR --image_format $IMAGE_FORMAT --metric $METRIC \
  --unsupervised --ablation

# GENERATE HISTOGRAMS FOR EACH DATASET
python src/visualization/generate_histograms.py --image_format $IMAGE_FORMAT
