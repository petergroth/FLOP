import argparse

import numpy as np
import pandas as pd
from PyBioMed.PyProtein import CTD


def compute_ct(dataset: str):
    """Function to compute and save compositional and transitional descriptors for dataset

    Args:
        dataset: Name of dataset

    """
    df = pd.read_csv(f"data/interim/{dataset}/{dataset}.csv", index_col=0)
    if dataset == "cm":
        # Get representations for ablation study sequences as well
        df = pd.read_csv(f"data/interim/{dataset}/{dataset}_all.csv", index_col=0)
    outdir = f"representations/{dataset}/ct"

    for i, row in df.iterrows():
        name, seq = row["name"], row["sequence"]
        protein_description_C = CTD.CalculateC(seq)
        protein_description_T = CTD.CalculateT(seq)
        arr_C = np.array(list(protein_description_C.values()))
        arr_T = np.array(list(protein_description_T.values()))

        np.save(f"{outdir}/{name}.npy", np.concatenate((arr_C, arr_T)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()
    compute_ct(args.dataset)
