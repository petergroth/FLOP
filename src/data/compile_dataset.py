"""Script/function to process and split any dataset. See README for required input format."""
import argparse
import logging
import os.path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from src.data.data_utils import generate_CV_partitions
from src.visualization.visualization_funcs import show_CV_split_distribution, show_target_histogram


def parse_dataset(
    dataset: str,
    name_key: str,
    target: str,
    dataset_path: str,
    log: bool = True,
    force: bool = False,
    visualize: bool = True,
    initial_threshold: float = 0.3,
    threads: int = 20,
    fasta_path: str = None,
    sequence_key: str = "sequence",
    force_graphpart: bool = False,
):
    """Function to obtain fair splits for supervised machine learning. Generated splits are guaranteed to be "
        (1) separated at sequence identity threshold,
        (2) stratified on the target attribute, and
        (3) balanced for similar-sized partitions.
    Cross-validation applied to the generated splits allows for learning across sub-families without introducing
    excessive data-leakage.

    Args:
        dataset: Name of dataset.
        name_key: Column name for protein identifier.
        target: Column name for target values
        dataset_path: Path for dataset-file. Either tsv/csv.
        log: Whether to save compilation information in logging file.
        force: Whether to rerun splitting procedure.
        visualize: Will create histogram of continuous + binarized targets for full dataset and for generated
            partitions.
        initial_threshold: Initial sequence identity threshold for GraphsPart.
        threads: Number of threads to use for dataset splitting [default=20]
        fasta_path: If sequence-column not present in dataset-file, will load from FASTA file. Assumes same keys in
            FASTA file as in name_key column in dataset-file.
        sequence_key: Column name for protein sequences. Will be ignored if fasta_path is supplied-
        force_graphpart: Set to True to force recomputation of pairwise distances. CRUCIAL if raw dataset file is
            changed.

    Returns: Compiled dataset as Pandas DataFrame with columns "name", "sequence", "target_reg", "target_class",
    "part_0", "part_1", "part_2".
    """

    ####################
    # Definitions      #
    ####################

    # Define input paths
    raw_seq_path = f"{dataset_path}"
    assert dataset_path[-3:] in ["csv", "tsv"]
    sep = "," if dataset_path[-3] == "c" else "\t"

    # Define output path
    out_csv_path = f"data/processed/{dataset}/{dataset}.csv"
    out_interim_csv_path = f"data/interim/{dataset}/{dataset}.csv"

    # Create directories (if missing)
    os.makedirs(f"data/processed/{dataset}", exist_ok=True)
    os.makedirs(f"data/interim/{dataset}", exist_ok=True)
    if visualize:
        os.makedirs(f"figures/{dataset}/splits", exist_ok=True)

    # Setup logging
    if log:
        logging.root.handlers = []
        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                logging.FileHandler(filename=f"data/interim/{dataset}/{dataset}_compilation.log", mode="w"),
                logging.StreamHandler(),
            ],
        )

    else:
        logging.basicConfig(level=logging.ERROR)

    # Define GraphPart parameters
    alignment_mode = "needle"
    threshold_inc = 0.05
    n_partitions = 3
    min_pp_split = 1 / (n_partitions + 1)

    ####################
    # Build DataFrame  #
    ####################

    if not os.path.exists(out_interim_csv_path) or force:
        logging.info(f"Generating CSV.")
        # Read and filter input file
        df_seq = pd.read_csv(raw_seq_path, sep=sep, index_col=0)

        # Ensure no duplicates
        assert df_seq[name_key].is_unique, "Identifiers must be unique. Consider preprocessing to remove duplicates."

        if fasta_path is None:
            df = df_seq.rename(columns={f"{name_key}": "name", target: "target_reg", f"{sequence_key}": "sequence"})[
                ["name", "target_reg", "sequence"]
            ]
        else:
            # Extract sequences from FASTA instead
            from Bio import SeqIO

            df = df_seq.rename(columns={f"{name_key}": "name", target: "target_reg"})[["name", "target_reg"]]
            # Extract sequences
            df["sequence"] = ""
            names = df["name"].to_list()

            for seq in SeqIO.parse(fasta_path, "fasta"):
                if seq.id in names:
                    df.loc[df["name"] == seq.id, "sequence"] = str(seq.seq)

        # Fit GMM to target values for binarization
        y = df["target_reg"].values.reshape(-1, 1)
        gm = GaussianMixture(n_components=2).fit(y)

        # Binary label active/inactive
        label = gm.predict(y)
        df["target_class"] = label.astype(int)

        # Save as csv file
        df.to_csv(out_interim_csv_path, index_label="index")
        logging.info(f"Created {out_interim_csv_path}.")

    else:
        df = pd.read_csv(out_interim_csv_path, index_col=0)
        logging.info("Loaded CSV.")

    #######################
    # Generate splits     #
    #######################

    threshold = np.nan
    if "part_0" not in df:
        ckpt_path = f"data/interim/{dataset}/{dataset}_cv_graphpart_edges.csv"
        ids, threshold = generate_CV_partitions(
            df,
            initial_threshold,
            dataset,
            alignment_mode,
            n_partitions,
            threads,
            min_pp_split,
            threshold_inc,
            ckpt_path,
            "target_class",
            force_graphpart,
        )
        # Partition headers
        headers = [f"part_{i}" for i in range(n_partitions)]

        for header, idx in zip(headers, ids):
            df[header] = 0
            df.loc[idx, header] = 1

        # Remove high energy and profile model generations
        df.to_csv(out_interim_csv_path, index_label="index")

    # Discard sequences not in partitions
    df = df[df[["part_0", "part_1", "part_2"]].sum(axis=1).astype(bool)]

    if visualize:
        show_CV_split_distribution(df, threshold, dataset, n_partitions)
        show_target_histogram(df=df, target="target_reg", dataset=dataset)
        show_target_histogram(df=df, target="target_class", dataset=dataset)

    df = df[["name", "sequence", "target_reg", "target_class", "part_0", "part_1", "part_2"]]
    df = df.reset_index(drop=True)
    df.to_csv(out_csv_path, index_label="index")
    logging.info(f"Final dataset size: {len(df)}.")
    logging.info(f"File saved in {out_csv_path}")


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Function to obtain fair splits for supervised machine learning."
        "Generated splits are guaranteed to be "
        "(1) seperated at sequence identity threshold, "
        "(2) stratified on the target attribute, and "
        "(3) balanced for similar-sized partitions. Cross-validation "
        "applied to the generated splits allows for learning across sub-"
        "families without introducing excessive data-leakage."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Name of dataset.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Full/relative path for TSV/CSV file.")
    parser.add_argument("--target", type=str, required=True, help="Column name for target value.")
    parser.add_argument("--name_key", type=str, required=True, help="Column name for protein ids.")
    parser.add_argument(
        "--log",
        action="store_true",
        default=True,
        help="Include to log dataset compilation. Will overwrite existing logs.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Include to force creation of dataset. Will overwrite existing splits."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Include to visualize and save histogram over target values for the generated splits. "
        "Will overwrite existing figures",
    )
    parser.add_argument(
        "--force_graphpart",
        action="store_true",
        help="Include to recompute pairwise distance. CRUCIAL if dataset file changes.",
    )
    parser.add_argument(
        "--threads", type=int, default=20, help="Number of threads to use for dataset splitting [default=20]"
    )
    parser.add_argument(
        "--fast_path",
        type=str,
        default=None,
        help="If no sequence column in dataset-file, supply via FASTA file. Assume correspondence between "
        "FASTA-identifier and `name_key` values from `dataset_path` file.",
    )
    parser.add_argument(
        "--initial_threshold", type=float, default=0.1, help="Initial sequence identity split thresholds [default=0.1]"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()
    parse_dataset(**vars(args))
