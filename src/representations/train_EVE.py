import argparse
import json
import os
import pickle

import pandas as pd

from EVE.EVE import VAE_model
from EVE.utils import data_utils

if __name__ == "__main__":
    """Script to train EVE.
    Adapted from https://github.com/OATML-Markslab/EVE/blob/master/train_VAE.py

    """
    # fmt: off
    parser = argparse.ArgumentParser(description="VAE")
    parser.add_argument('--MSA_data_folder', type=str, help='Folder where MSAs are stored')
    parser.add_argument('--MSA_list', type=str, help='List of proteins and corresponding MSA file name')
    parser.add_argument('--protein_index', type=int, help='Row index of protein in input mapping file')
    parser.add_argument('--MSA_weights_location', type=str, help='Location where weights for each sequence in the MSA will be stored')
    parser.add_argument('--theta_reweighting', type=float, help='Parameters for MSA sequence re-weighting')
    parser.add_argument('--VAE_checkpoint_location', type=str, help='Location where VAE model checkpoints will be stored')
    parser.add_argument('--model_name_suffix', default='Jan1', type=str, help='model checkpoint name will be the protein name followed by this suffix')
    parser.add_argument('--model_parameters_location', type=str, help='Location of VAE model parameters')
    parser.add_argument('--training_logs_location', type=str, help='Location of VAE model parameters')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset_pickle', type=str, help="Path for pickled dataset. Will be generated if not found.")
    args = parser.parse_args()

    mapping_file = pd.read_csv(args.MSA_list)
    protein_family = mapping_file['protein_name'][args.protein_index]
    msa_location = args.MSA_data_folder + os.sep + mapping_file['msa_location'][args.protein_index]
    print("Protein family: " + str(protein_family))
    print("MSA file: " + str(msa_location))

    if args.theta_reweighting is not None:
        theta = args.theta_reweighting
    else:
        try:
            theta = float(mapping_file['theta'][args.protein_index])
        except:
            theta = 0.2
    print("Theta MSA re-weighting: " + str(theta))

    # If pickled dataset not found, generate and save
    if not os.path.exists(args.dataset_pickle):
        print("Pickled dataset not found. Pre-processing...")
        threshold_sequence_frac_gaps = 0.5
        threshold_focus_cols_frac_gaps = 0.3

        data = data_utils.MSA_processing(
            MSA_location=msa_location,
            theta=theta,
            use_weights=True,
            protein_family=protein_family,
            weights_location=f"{args.MSA_weights_location}{os.sep}"
            f"{protein_family}_theta_{str(theta)}.npy",
            threshold_sequence_frac_gaps=threshold_sequence_frac_gaps,
            threshold_focus_cols_frac_gaps=threshold_focus_cols_frac_gaps,
        )
        with open(args.dataset_pickle, "wb") as handle:
            pickle.dump(data, handle)
            print("Dataset saved with pickle.")

    # Load data
    with open(args.dataset_pickle, "rb") as handle:
        data = pickle.load(handle)
        print("Dataset loaded with pickle.")

    model_name = f"{protein_family}_{args.model_name_suffix}"
    print(f"Model name: {model_name}")

    model_params = json.load(open(args.model_parameters_location))

    model = VAE_model.VAE_model(
        model_name=model_name,
        data=data,
        encoder_parameters=model_params["encoder_parameters"],
        decoder_parameters=model_params["decoder_parameters"],
        random_seed=args.seed,
    )
    model = model.to(model.device)

    model_params["training_parameters"]['training_logs_location'] = args.training_logs_location
    model_params["training_parameters"]['model_checkpoint_location'] = args.VAE_checkpoint_location

    print("Starting to train model: " + model_name)
    model.train_model(data=data, training_parameters=model_params["training_parameters"])

    print("Saving model: " + model_name)
    model.save(model_checkpoint=model_params["training_parameters"]['model_checkpoint_location'] + os.sep + model_name + "_final",
               encoder_parameters=model_params["encoder_parameters"],
               decoder_parameters=model_params["decoder_parameters"],
               training_parameters=model_params["training_parameters"]
               )
    # fmt: off
