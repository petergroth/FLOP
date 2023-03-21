"""Script to compute (unsupervised) ELBOs"""
import argparse
import json
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from EVE.EVE import VAE_model


def main(dataset: str, suffix: str):
    """Function to compute (unsupervised) ELBOs.
    Adapted from https://github.com/OATML-Markslab/EVE/blob/master/compute_evol_indices.py

    Args:
        dataset: Name of dataset
        suffix: Model suffix

    """
    num_samples = 20000
    model_parameters_location = f"EVE/EVE/default_model_params.json"
    VAE_checkpoint = f"models/EVE/{dataset}/{dataset}_{suffix}"
    output_dir = f"representations/{dataset}"
    dataset_pickle = f"data/interim/{dataset}/{dataset}_EVE_preprocessed.pkl"
    batch_size = 2048
    # Load dataframe
    df = pd.read_csv(f"data/interim/{dataset}/{dataset}.csv")
    # Load data
    with open(dataset_pickle, "rb") as handle:
        data = pickle.load(handle)
        print("Loaded dataset with pickle.")
    embedding_dim = data.seq_len
    model_name = f"{dataset}_{suffix}"
    model_params = json.load(open(model_parameters_location))

    # Load model
    model = VAE_model.VAE_model(
        model_name=model_name,
        data=data,
        encoder_parameters=model_params["encoder_parameters"],
        decoder_parameters=model_params["decoder_parameters"],
        random_seed=42,
    )

    try:
        checkpoint_name = f"{VAE_checkpoint}_best"
        checkpoint = torch.load(checkpoint_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Initialized VAE with checkpoint '{checkpoint_name}'.")
    except:
        print("Unable to locate VAE model checkpoint")
        sys.exit(0)

    # Send to GPU
    if torch.cuda.is_available():
        model = model.cuda()
        print("Moved model to GPU.")

    model.eval()

    with torch.no_grad():
        # Iterate through all relevant sequences
        one_hot = np.zeros((len(df), embedding_dim, 20))
        for i, name in tqdm(enumerate(df["name"].tolist()), desc="One-hot encoding", total=len(df)):
            sequence = data.seq_name_to_sequence[f">{name}"]
            # One-hot encoding
            for j, letter in enumerate(sequence):
                if letter in data.aa_dict:
                    k = data.aa_dict[letter]
                    one_hot[i, j, k] = 1.0

        dataloader = torch.utils.data.DataLoader(
            torch.tensor(one_hot), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

        prediction_matrix = torch.zeros(len(df), num_samples)

        for i, batch in enumerate(tqdm(dataloader, f"Computing ELBOs")):
            x = batch.type(model.dtype).to(model.device)
            for j in tqdm(range(num_samples)):
                seq_predictions, _, _ = model.all_likelihood_components(x)
                prediction_matrix[(i * batch_size) : (i * batch_size + len(x)), j] = seq_predictions
            tqdm.write("\n")

        # Average over predictions
        mean_predictions = prediction_matrix.mean(dim=1, keepdim=False)
        evol_indices = mean_predictions.detach().cpu().numpy()

    df["ELBO"] = evol_indices
    df = df[["name", "ELBO"]]
    df.to_csv(f"{output_dir}/{dataset}_EVE_ELBO_{suffix}.csv", index_label="index")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("suffix", type=str)
    args = parser.parse_args()

    main(dataset=args.dataset, suffix=args.suffix)
