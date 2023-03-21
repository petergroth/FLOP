"""This file contains various helper functions that are used in other scripts."""
import logging
import os
from typing import Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
import torch.linalg

from graph_part import stratified_k_fold


def precompute_edges(
    df: pd.DataFrame,
    checkpoint_path: str,
    alignment_mode: str = "needle",
    threads: int = 10,
    force_graphpart: bool = False,
) -> None:
    """Computes all pairwise distances between protein sequences with GraphPart. Partitioning deliberately fails
    but saves file with all edges for further use.

    Args:
        df: Input dataframe. Requires column "sequence"
        checkpoint_path: Path to save edge-file.
        alignment_mode: Needle-algorithm for pairwise differences
        threads: Number of used threads for multiprocessing
        force_graphpart: Forces the computation of pair-wise distances instead of using saved edge-file
    """

    # Prepare values for GraphPart
    sequences = df["sequence"].tolist()

    # Precompute edges if generating test split
    if not os.path.exists(checkpoint_path) or force_graphpart:
        logging.info("Pre-computing distances.")
        try:
            stratified_k_fold(
                sequences=sequences,
                alignment_mode=alignment_mode,
                partitions=2,
                threads=threads,
                threshold=0.01,
                save_checkpoint_path=checkpoint_path,
            )
        except RuntimeError:
            # GraphPart will (likely) fail but still save the computed edges for further use.
            pass
    else:
        logging.info("Using pre-computed distances.")


def generate_CV_partitions(
    df: pd.DataFrame,
    initial_threshold: float,
    dataset: str,
    alignment_mode: str = "needle",
    n_partitions: int = 3,
    threads: int = 10,
    min_pp_split: float = 0.25,
    threshold_inc: float = 0.05,
    checkpoint_path: Union[str, None] = None,
    label: str = "target_class",
    force_graphpart: bool = False,
) -> Tuple[List[Iterable], float]:
    """Function to generate homology-based, stratified partitions for CV of similar sizes.

    Args:
        df: Input dataframe. Must have columns "sequence" and "target_class"
        initial_threshold: Starting point for clustering algorithm.
        dataset: Name of dataset.
        alignment_mode: Algorithm to compute distances.
        n_partitions: Number of CV partitions.
        threads: Number of threads for multiprocessing.
        min_pp_split: Minimum fraction of sequences per partition.
        threshold_inc: Threshold incrementer if partitioning not possible.
        checkpoint_path: Precomputed-edge destination.
        label: Column containing categoricals to use for stratification.
        force_graphpart: Force re-computation of edges.

    Returns:
        ids, threshold: List of integers corresponding to indices in input DataFrame. Threshold is the value at which
        the partitioning was possible.
    """

    # Prepare values for GraphPart
    threshold_limit = 1.0
    threshold = initial_threshold

    # Resort to lists as GraphPart bug prevents using pandas directly
    sequences = df["sequence"].tolist()
    labels = df[label].tolist()

    if checkpoint_path is None:
        checkpoint_path = f"data/interim/{dataset}/{dataset}_graphpart_edges.csv"

    # Precompute edges
    precompute_edges(
        df=df,
        checkpoint_path=checkpoint_path,
        alignment_mode=alignment_mode,
        threads=threads,
        force_graphpart=force_graphpart,
    )
    # Use (now) precomputed weights
    alignment_mode = "precomputed"

    # Proceed to main generation script
    logging.info(f"Generating {n_partitions} partitions.")
    while threshold <= threshold_limit:
        try:
            # Run GraphPart
            ids = stratified_k_fold(
                sequences=sequences,
                labels=labels,
                alignment_mode=alignment_mode,
                threads=threads,
                edge_file=checkpoint_path,
                threshold=threshold,
                metric_column=2,
                partitions=n_partitions,
            )  # metric_column warning due to bug in GraphPart

            # Inspect splits
            labels_arr = np.array(labels)
            n_eff = len([x for xs in ids for x in xs])
            p_obs = np.zeros(n_partitions)
            p_class = np.zeros(n_partitions)
            n_obs = np.zeros(n_partitions, dtype=int)

            for i in range(n_partitions):
                n_obs[i] = len(ids[i])
                p_obs[i] = n_obs[i] / n_eff
                # Class probability
                p_class[i] = labels_arr[ids[i]].sum() / n_obs[i]

            # Verify that each split has at least min_pp_split % of sequences. If not, increase threshold.
            if (p_obs < min_pp_split).any():
                logging.info(
                    f"Partition not possible at threshold {round(threshold, 3)}. Less than {min_pp_split * 100:.0f} "
                    f"% of sequences found in a split:"
                )
                [
                    logging.info(f"- {p_obs[i] * 100:.2f} % ({n_obs[i]}/{n_eff}) in split {i + 1}.")
                    for i in range(n_partitions)
                ]
                logging.info(
                    f"Increasing threshold from {round(threshold, 3)} to "
                    f"{round(threshold + threshold_inc, 3)} to achieve balance."
                )
                threshold += threshold_inc
                continue

            # Print split details
            logging.info(f"Dataset successfully split at threshold {round(threshold, 3)}.")
            logging.info(f"Summary:")
            [
                logging.info(
                    f"- {p_obs[i] * 100:.2f} % ({n_obs[i]}/{n_eff}) in split {i + 1}, with p(class=1) = {p_class[i] * 100:.2f} %."
                )
                for i in range(n_partitions)
            ]

            return ids, round(threshold, 3)

        except RuntimeError:
            # If GraphPart fails, splits not possible. Increase threshold.
            logging.info(
                f"Partition not possible. Increasing threshold from {round(threshold, 3)} to "
                f"{round(threshold + threshold_inc, 3)}."
            )
            threshold += threshold_inc


def extract_all_embeddings(
    df: pd.DataFrame, dataset: str, embedding_type: str, target: str = "target", suffix: str = None
) -> Tuple[np.array, np.array, List[str]]:
    """Loads embedding/representation for given dataset.

    Args:
        df: Dataframe with `target` column containing regression values and `names` column with identifiers for all
            sequences.
        dataset: Dataset
        embedding_type: Name of representation/embedding
        target: Column in dataset csv corresponding to target value
        suffix: In case of stochastic model (e.g., EVE), specifies subfolder.

    Returns:
        Function returns array over embeddings, array over target values, and protein names

    """

    y = df[target].values
    n_obs = len(df)
    names = df["name"].tolist()
    embedding_dir = f"representations/{dataset}/{embedding_type}"
    dim_dict = {
        "esm_1b": 1280,
        "esm_2": 2560,
        "esm_if1": 512,
        "onehot": 0,
        "onehot_pad": 0,
        "eve": 50,
        "af2": 384,
        "ct": 42,
    }
    dim = dim_dict[embedding_type]
    embeddings = np.zeros((n_obs, dim))

    if embedding_type == "esm_1b":
        # Extract embeddings
        for i, name in enumerate(names):
            embedding = torch.load(f"{embedding_dir}/{name}.pt")
            embeddings[i] = embedding["mean_representations"][33].numpy()

    elif embedding_type == "esm_2":
        # Extract embeddings
        for i, name in enumerate(names):
            embedding = torch.load(f"{embedding_dir}/{name}.pt")
            embeddings[i] = embedding["mean_representations"][36].numpy()

    elif embedding_type == "esm_if1":
        # Extract embeddings
        for i, name in enumerate(names):
            embedding = torch.load(f"{embedding_dir}/{name}.pt").numpy()
            embeddings[i] = np.mean(embedding, axis=0)

    elif embedding_type == "onehot":
        # Find sequence length
        dummy = np.load(f"{embedding_dir}/{df.iloc[0]['name']}.npy").flatten()
        dim = len(dummy)

        # Extract embeddings
        embeddings = np.zeros((n_obs, dim))
        for i, name in enumerate(names):
            embedding = np.load(f"{embedding_dir}/{name}.npy")
            embeddings[i] = embedding.flatten()

    elif embedding_type == "onehot_pad":  # DO NOT USE
        # Find sequence length
        dummy = np.load(f"{embedding_dir}/{df.iloc[0]['name']}.npy").flatten()
        dim = len(dummy)

        # Extract embeddings
        embeddings = np.zeros((n_obs, dim))
        for i, name in enumerate(names):
            embedding = np.load(f"{embedding_dir}/{name}.npy")
            embeddings[i] = embedding.flatten()

    elif embedding_type == "eve":
        embedding_dir = f"{embedding_dir}/{suffix}"
        # Extract embeddings
        for i, name in enumerate(names):
            embedding = torch.load(f"{embedding_dir}/{name}.pt")
            embeddings[i] = embedding.numpy()

    elif embedding_type == "af2":
        # Extract embeddings
        for i, name in enumerate(names):
            embedding = np.load(f"{embedding_dir}/{name}.npy")
            embeddings[i] = np.mean(embedding, axis=0)

    elif embedding_type == "ct":
        # Extract embeddings
        for i, name in enumerate(names):
            embeddings[i] = np.load(f"{embedding_dir}/{name}.npy")

    else:
        raise NotImplementedError
    return embeddings, y, names


def compute_median_seq_id(dataset: str):
    """Function to compute median sequence identity of a dataset, given an all vs. all percentage matrix.
    This can be created with `famsa  -dist_export -pid -square_matrix  INPUT.fasta all_vs_all.csv`

    """

    df = pd.read_csv(f"data/interim/{dataset}/all_vs_all.csv", index_col=0)
    vals = df.values[np.tril_indices(n=len(df), k=-1)]
    mean = np.mean(vals)
    median = np.median(vals)
    std = np.std(vals)
    print(f"{dataset} mean sequence identity: {mean:.3f}")
    print(f"{dataset} median sequence identity: {median:.3f}")
    print(f"{dataset} standard deviation, sequence identity: {std:.3f}")


def repr_dict() -> dict:
    return {
        "onehot": "MSA (1-HOT)",
        "eve": "EVE",
        "af2": "Evoformer (AF2)",
        "ct": "CT",
        "esm_1b": "ESM-1B",
        "esm_2": "ESM-2",
        "esm_if1": "ESM-IF1",
        "onehot_pad": "UNALIGNED (1-HOT)",
    }
