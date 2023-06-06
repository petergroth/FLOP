"""Script/function to process and split GH114 dataset"""
import argparse
import logging
import os.path
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from src.data.data_utils import generate_CV_partitions
from src.visualization.visualization_funcs import show_CV_split_distribution, show_target_histogram


def parse_gh114(
    log: bool = True,
    force: bool = False,
    visualize: bool = True,
    force_graphpart: bool = False,
    threads: int = 20,
    initial_threshold: float = 0.1,
):
    """Function to obtain fair splits for supervised machine learning for GH114 dataset. Generated splits are
    guaranteed to be
        (1) separated at sequence identity threshold,
        (2) stratified on the target attribute, and
        (3) balanced for similar-sized partitions.
    Cross-validation applied to the generated splits allows for learning across sub-families without introducing
    excessive data-leakage.

    Args:
        log: Whether to save compilation information in logging file.
        force: Whether to rerun splitting procedure.
        visualize: Will create histogram of continuous + binarized targets for full dataset and for generated
            partitions.
        initial_threshold: Initial sequence identity threshold for GraphsPart.
        threads: Number of threads to use for dataset splitting [default=20]
        force_graphpart: Set to True to force recomputation of pairwise distances. CRUCIAL if raw dataset file is
            changed.

    Returns: Compiled dataset as Pandas DataFrame with columns "name", "sequence", "target_reg", "target_class",
    "part_0", "part_1", "part_2".
    """

    ####################
    # Definitions      #
    ####################

    # Setup paths
    dataset = "gh114"
    interim_dir = Path("data", "interim", dataset)
    processed_dir = Path("data", "processed", dataset)
    out_csv_path = processed_dir / f"{dataset}.csv"
    out_interim_csv_path = interim_dir / f"{dataset}.csv"
    log_path = interim_dir / f"compile_{dataset}.log"
    ckpt_path = interim_dir / f"{dataset}_cv_graphpart_edges.csv"
    raw_seq_path = Path("data", "raw", dataset, f"{dataset}.csv")
    # Ensure directories exist
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    if log:
        logging.root.handlers = []
        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                logging.FileHandler(filename=log_path, mode="w"),
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
        df_seq = pd.read_csv(raw_seq_path, index_col=0)
        df = df_seq.rename(columns={"target": "target_reg"})[["name", "target_reg", "sequence"]]

        # Fit GMM to target values for binarization
        y = df["target_reg"].values.reshape(-1, 1)
        label = GaussianMixture(n_components=2).fit_predict(y)
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

        df.to_csv(out_interim_csv_path, index_label="index")

    # Clean and save dataframe for modelling
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

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", action="store_true", help="Include to log dataset compilation. Will overwrite existing logs."
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
        help="Force recomputation of pairwise distances. Important if raw data file is changed in any way.",
    )
    parser.add_argument(
        "--threads", type=int, default=20, help="Number of threads to use for dataset splitting [default=20]"
    )
    parser.add_argument(
        "--initial_threshold", type=float, default=0.1, help="Initial sequence identity split thresholds [default=0.1]"
    )
    args = parser.parse_args()
    df = parse_gh114(**vars(args))
