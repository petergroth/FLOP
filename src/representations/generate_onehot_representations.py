import argparse
import os

import numpy as np
import pandas as pd
from Bio import SeqIO


def generate_onehot_msa_encoding(dataset: str):
    """Create one-hot encoded representation given MSA

    Args:
        dataset: Name of dataset

    """
    # Dataset specific paths
    output_dir = f"representations/{dataset}/onehot"
    msa_path = f"data/raw/{dataset}/{dataset}_family.aln.fasta"
    os.makedirs(output_dir, exist_ok=True)

    # AA dictionary
    aa_dict = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

    # Determine maximum sequence length
    df = pd.read_csv(f"data/processed/{dataset}/{dataset}.csv", index_col=0)
    dummy = next(iter(SeqIO.parse(open(msa_path), "fasta")))
    seq_len = len(dummy.seq)

    # Create zero-padded one-hot encoding and save to disk
    for fasta in SeqIO.parse(open(msa_path), "fasta"):
        one_hot = np.zeros((seq_len, 20))
        if fasta.id in df["name"].tolist():
            for j, letter in enumerate(str(fasta.seq)):
                if letter in aa_dict:
                    k = aa_dict[letter]
                    one_hot[j, k] = 1.0
            np.save(file=f"{output_dir}/{fasta.id}.npy", arr=one_hot)


def generate_padded_onehot_encoding(dataset: str):
    """Create one-hot encoded representation using post-padding

    Args:
        dataset: Name of dataset

    """
    # Dataset specific paths
    output_dir = f"representations/{dataset}/onehot_pad"
    os.makedirs(output_dir, exist_ok=True)

    # AA dictionary
    aa_dict = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

    # Determine maximum sequence length
    df = pd.read_csv(f"data/interim/{dataset}/{dataset}.csv", index_col=0)
    if dataset == "cm":
        df = pd.read_csv(f"data/interim/{dataset}/{dataset}_all.csv", index_col=0)

    seq_len = df["sequence"].str.len().max()

    # Create zero-padded one-hot encoding and save to disk
    for row in df.itertuples(index=False):
        one_hot = np.zeros((seq_len, 20))
        for j, letter in enumerate(row[1]):
            if letter in aa_dict:
                k = aa_dict[letter]
                one_hot[j, k] = 1.0
        np.save(file=f"{output_dir}/{row[0]}.npy", arr=one_hot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Name of dataset.")
    parser.add_argument("--mode", type=str, default="MSA", help="Either MSA or pad.")
    args = parser.parse_args()
    if args.mode == "MSA":
        generate_onehot_msa_encoding(args.dataset)
    elif args.mode == "pad":
        generate_padded_onehot_encoding(args.dataset)
