import argparse
from typing import Tuple

from src.data.data_utils import repr_dict
from src.training.training_utils import fit_model_CV, fit_model_holdout, fit_model_random


def run_ablation(dataset: str, embedding_types: Tuple[str, ...], low_n: bool = False) -> None:
    """Function to run CV learning setup for a given dataset

    Args:
        dataset: One of gh114, cm, ppat
        embedding_types: List of embedding/representation types

    """
    eve_suffixes = ("0", "1", "2")
    threads = 20
    n_partitions = 3
    out_dir = f"results/ablation"
    task = "regression"
    embedding_dict = repr_dict()

    if dataset == "cm":
        print(f"Running CV on {dataset} dataset using all sequences.")
        # Iterate through all embedding types sequentially
        for embedding_type in embedding_types:
            print(f"Fitting regressors to {embedding_dict[embedding_type]}.")
            df = fit_model_CV(
                dataset, embedding_type, n_partitions, eve_suffixes, threads, ablation=True, task=task, low_n=low_n
            )
            out_path = f"{out_dir}/{dataset}_results_{embedding_type}_all.csv"
            df.to_csv(out_path, index_label="index")

    print(f"Running CV on {dataset} dataset with random partitions.")
    # Iterate through all embedding types sequentially
    for embedding_type in embedding_types:
        print(f"Fitting regressors to {embedding_dict[embedding_type]}.")
        seeds = [0, 1, 2]
        df = fit_model_random(dataset, embedding_type, eve_suffixes, seeds, threads, task, low_n)
        out_path = f"{out_dir}/{dataset}_results_{embedding_type}_random.csv"
        df.to_csv(out_path, index_label="index")

    print(f"Running holdout-validation on {dataset} dataset.")
    # Iterate through all embedding types sequentially
    for embedding_type in embedding_types:
        print(f"Fitting regressors to {embedding_dict[embedding_type]}.")
        split_key = "holdout"
        df = fit_model_holdout(dataset, embedding_type, split_key, eve_suffixes, threads, task, low_n)
        out_path = f"{out_dir}/{dataset}_results_{embedding_type}_holdout.csv"
        df.to_csv(out_path, index_label="index")


def run_CV(
    dataset: str,
    embedding_types: Tuple[str, ...],
    task: str = "regression",
    low_n: bool = False,
    save_predictions: bool = False,
    threads: int = 20,
) -> None:
    """Function to run CV learning setup for a given dataset

    Args:
        dataset: One of gh114, cm, ppat.
        embedding_types: List of embedding/representation types
        task: Either `regression` or `classification`.
        low_n: Whether to limit max number of neighbours in KNN to 5
        save_predictions: Whether to save all predictions on test sets. Will group each embedding/regressor.
        threads: Number of threads to use for sklearn parallelization

    """
    print(f"Running CV on {dataset} dataset.")
    eve_suffixes = ("0", "1", "2")
    n_partitions = 3
    out_dir = f"results/{dataset}"
    ablation = False
    embedding_dict = repr_dict()

    # Iterate through all embedding types sequentially
    for embedding_type in embedding_types:
        print(f"Fitting predictors to {embedding_dict[embedding_type]} (task={task}).")
        df = fit_model_CV(
            dataset, embedding_type, n_partitions, eve_suffixes, threads, ablation, task, low_n, save_predictions
        )
        if task == "regression":
            out_path = f"{out_dir}/{dataset}_results_{embedding_type}.csv"
        else:
            out_path = f"{out_dir}/{dataset}_results_{embedding_type}_classification.csv"
        df.to_csv(out_path, index_label="index")

    print("Finished.")


def main(
    dataset: str,
    ablation: bool,
    task: str,
    low_n: bool,
    save_predictions: bool,
    threads: int,
    embedding_types: Tuple["str", ...],
):
    if ablation:
        run_ablation(dataset, embedding_types, low_n)
    else:
        run_CV(dataset, embedding_types, task, low_n, save_predictions, threads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str, help="Name of dataset.")
    parser.add_argument("--ablation", action="store_true", help="Use alternative ablation splits")
    parser.add_argument("--low_n", action="store_true", help="Include to limit K=5 for KNN regressor/classifier.")
    parser.add_argument("--save_predictions", action="store_true", help="Whether to save predictions.")
    parser.add_argument("--task", default="regression", type=str, help="Regression/classification.")
    parser.add_argument("--threads", default=20, type=int, help="Number of threads.")
    parser.add_argument(
        "--embedding_types", type=str, nargs="+", default=("ct", "af2", "esm_1b", "esm_2", "esm_if1", "eve", "onehot")
    )
    args = parser.parse_args()
    main(**vars(args))
