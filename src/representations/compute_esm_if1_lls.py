# Adapted from https://github.com/facebookresearch/esm/blob/main/examples/inverse_folding/score_log_likelihoods.py

import argparse

import esm
import esm.inverse_folding
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def score_dataset_backbones(dataset: str):
    """Function to compute LL for sequence conditioned on its structure using ESM-IF1 model

    Args:
        dataset: Name of dataset

    """
    # Define paths
    pretrained_path = "models/esm_if1_gvp4_t16_142M_UR50.pt"
    output_path = f"representations/{dataset}/esm_if1_likelihoods.csv"
    pdb_dir = f"data/raw/{dataset}/pdb"

    # Extract data
    df = pd.read_csv(f"data/interim/{dataset}/{dataset}.csv", index_col=0)
    if dataset == "cm":
        df = pd.read_csv(f"data/raw/{dataset}/{dataset}.csv", index_col=0)
        df = df[
            df["comment"].isin(
                ["natural sequences", "bmDCA designed sequences, T=0.33", "bmDCA designed sequences, T=0.66"]
            )
        ]
    df = df[["name"]]
    lls = np.zeros(len(df))

    # Load model
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(pretrained_path)
    print("Successfully loaded model.")
    model.eval()

    with torch.no_grad():
        for i, name in enumerate(tqdm(df["name"])):
            coords, native_seq = esm.inverse_folding.util.load_coords(f"{pdb_dir}/{name}.pdb", "A")

            ll, _ = esm.inverse_folding.util.score_sequence(model, alphabet, coords, native_seq)
            lls[i] = ll

    df["ll"] = lls

    df.to_csv(output_path)


def main():
    parser = argparse.ArgumentParser(description="Score sequences based on a given structure.")
    parser.add_argument("dataset", type=str, help="Name of dataset.")
    args = parser.parse_args()

    score_dataset_backbones(args.dataset)


if __name__ == "__main__":
    main()
