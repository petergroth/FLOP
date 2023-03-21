#!/bin/bash

# SET STATIC PARAMETERS
MSA_list="data/interim/eve_mappings.csv"
model_parameters_location="EVE/EVE/default_model_params.json"

for DATASET in "gh114" "cm" "ppat"
do
  # SET DATASET SPECIFIC PARAMETERS
  if [ "${DATASET}" = "gh114" ]; then
    protein_index=0
  elif [ "${DATASET}" = "cm" ]; then
    protein_index=1
  elif [ "${DATASET}" = "ppat" ]; then
    protein_index=2
  fi
  MSA_data_folder="data/raw/$DATASET"
  MSA_weights_location="data/interim/$DATASET"
  VAE_checkpoint_location="models/EVE/$DATASET"
  training_logs_location="logs/$DATASET/EVE"
  dataset_pickle="data/interim/$DATASET/${DATASET}_EVE_preprocessed.pkl"

  # GENERATE EVE QUERY
  python src/representations/generate_eve_query.py $DATASET

  # TRAIN MODEL WITH 3 DIFFERENT SEEDS
  for SEED in 0 1 2
  do
    model_name_suffix="${SEED}"
    echo "Running train_EVE.py on $DATASET dataset. Seed = ${SEED}"
    python src/representations/train_EVE.py \
          --MSA_data_folder ${MSA_data_folder} \
          --MSA_list ${MSA_list} \
          --protein_index ${protein_index} \
          --MSA_weights_location ${MSA_weights_location} \
          --VAE_checkpoint_location ${VAE_checkpoint_location} \
          --model_name_suffix ${model_name_suffix} \
          --model_parameters_location ${model_parameters_location} \
          --training_logs_location ${training_logs_location} \
          --dataset_pickle ${dataset_pickle} \
          --seed ${SEED}

    # GENERATE LATENT EMBEDDINGS
    echo "Computing EVE scores on $DATASET dataset. Seed = ${SEED}"
    python src/representations/compute_eve_elbos.py "$DATASET" ${model_name_suffix}

    # COMPUTER ELBOS FOR UNSUPERVISED PREDICTIONS
    echo "Generating EVE embeddings for $DATASET dataset. Seed = ${SEED}"
    python src/representations/generate_eve_representations.py "$DATASET" ${model_name_suffix}

  done
done