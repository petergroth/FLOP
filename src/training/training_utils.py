from itertools import permutations
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from math import factorial
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import matthews_corrcoef, mean_squared_error, roc_auc_score, f1_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import Pipeline

from src.data.data_utils import extract_all_embeddings


def fit_model_random(
    dataset: str,
    embedding_type: str,
    eve_suffixes: Tuple[str, ...],
    seeds: [int, ...],
    threads: int = 20,
    task: str = "regression",
    low_n: bool = False,
) -> pd.DataFrame:
    """Function to evaluate regressors using randomly sampled partitions. For ablation study.

    Args:
        dataset: Name of dataset (`cm`, `gh114`, `ppat`)
        embedding_type:  Input representation/embedding name
        eve_suffixes: Indicators for trained EVE models
        seeds: List of random seeds
        threads: Number of threads for multiprocessing
        task: Either `regression` or `classification`

    Returns:
        DataFrame containing metrics and evaluation details.

    """
    # Load data
    df = pd.read_csv(f"data/processed/{dataset}/{dataset}.csv", index_col=0)

    # Define regression parameters
    config = setup_CV_grids(task, low_n)
    if task == "regression":
        metric_1_str, metric_2_str, metric_3_str = "spearman", "rmse", "mae"
        target = "target_reg"
    else:
        metric_1_str, metric_2_str, metric_3_str = "mcc", "auroc", "f1"
        target = "target_class"

    if embedding_type == "eve":
        n_experiments = len(seeds) * len(config["representations"]) * len(eve_suffixes)
        n_suffix = len(eve_suffixes)
    else:
        n_experiments = len(seeds) * len(config["representations"])
        n_suffix = 1
        eve_suffixes = [None]

    # Allocate result arrays
    metric_1_arr = np.zeros((n_experiments, 3))
    metric_2_arr = np.zeros((n_experiments, 3))
    metric_3_arr = np.zeros((n_experiments, 3))
    seed_arr = np.zeros(n_experiments, dtype=int)
    model_lst, suffix_lst = [], []
    # Counter to track experiments
    global_counter = 0

    for i in range(n_suffix):
        suffix = eve_suffixes[i]
        # Extract embeddings
        embeddings, y, names = extract_all_embeddings(
            df=df, dataset=dataset, embedding_type=embedding_type, suffix=suffix, target=target
        )
        # Determine fixed split sizes
        n_obs = len(y)
        n_train = int(n_obs * 0.5)
        n_val = int(n_obs * 0.25)

        # Split index for CV in sklearn
        split_index = np.repeat([-1, 0], [n_train, n_val])
        cv = PredefinedSplit(test_fold=split_index)

        # Iterate through seeds
        for seed in seeds:
            np.random.seed(seed)
            # Create permutation
            perm = np.random.permutation(n_obs)
            train_idx = perm[:n_train]
            val_idx = perm[n_train : (n_train + n_val)]
            test_idx = perm[(n_train + n_val) :]
            # Extract inputs/targets
            embedding_train = embeddings[train_idx]
            embedding_val = embeddings[val_idx]
            embedding_test = embeddings[test_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]
            y_test = y[test_idx]

            for model_str, param_grid in zip(config["representations"], config["param_grids"]):
                # Define grid
                grid = GridSearchCV(
                    estimator=config["pipe"],
                    param_grid=param_grid,
                    scoring=config["scoring"],
                    verbose=0,
                    n_jobs=threads,
                    cv=cv,
                )

                # Fit all representations in grid
                grid.fit(np.concatenate((embedding_train, embedding_val)), np.concatenate((y_train, y_val)))
                # Extract and fit best model to training set
                model = grid.best_estimator_["model"].fit(embedding_train, y_train)

                # Get training, validation, and test predictions
                # Get training, validation, and test predictions
                preds_train = model.predict(embedding_train)
                preds_val = model.predict(embedding_val)
                preds_test = model.predict(embedding_test)

                # Compute and save Spearman correlations
                metric_1, metric_2, metric_3 = compute_metrics(
                    task, preds_train, preds_val, preds_test, y_train, y_val, y_test
                )

                # Fill in results
                metric_1_arr[global_counter] = metric_1
                metric_2_arr[global_counter] = metric_2
                metric_3_arr[global_counter] = metric_3
                model_lst.append(model_str)
                seed_arr[global_counter] = seed
                suffix_lst.append(suffix)
                global_counter += 1

    # Construct and return DataFrame
    df_results = pd.DataFrame(
        {
            "model": model_lst,
            "seed": seed_arr,
            "suffix": suffix_lst,
            f"train_{metric_1_str}": metric_1_arr[:, 0],
            f"val_{metric_1_str}": metric_1_arr[:, 1],
            f"test_{metric_1_str}": metric_1_arr[:, 2],
            f"train_{metric_2_str}": metric_2_arr[:, 0],
            f"val_{metric_2_str}": metric_2_arr[:, 1],
            f"test_{metric_2_str}": metric_2_arr[:, 2],
            f"train_{metric_3_str}": metric_3_arr[:, 0],
            f"val_{metric_3_str}": metric_3_arr[:, 1],
            f"test_{metric_3_str}": metric_3_arr[:, 2],
        }
    )
    df_results["embedding"] = embedding_type
    df_results["strategy"] = "random"

    return df_results


def fit_model_holdout(
    dataset: str,
    embedding_type: str,
    split_key: str,
    eve_suffixes: Tuple[str, ...],
    threads: int = 20,
    task: str = "regression",
    low_n: bool = False,
) -> pd.DataFrame:
    """Function to evaluate regressors using holdout validation. Strongly recommended to use CV strategy instead.

    Args:
        dataset: Name of dataset (`cm`, `gh114`, `ppat`)
        embedding_type: Input representation/embedding name
        split_key: Column name in dataframe containing `train`, `val`, `test` indicators.
        eve_suffixes: Indicators for trained EVE models
        threads: Number of threads for multiprocessing
        task: Either `regression` or `classification`
        low_n: Restricts the number of neighbours in KNN models to 5 (for small datasets)

    Returns:
        DataFrame containing metrics and evaluation details.
    """

    # Load data
    df = pd.read_csv(f"data/processed/{dataset}/{dataset}.csv", index_col=0)

    # Define regression parameters
    config = setup_CV_grids(task, low_n)
    if task == "regression":
        metric_1_str, metric_2_str, metric_3_str = "spearman", "rmse", "mae"
        target = "target_reg"
    else:
        metric_1_str, metric_2_str, metric_3_str = "mcc", "auroc", "f1"
        target = "target_class"

    if embedding_type == "eve":
        n_experiments = len(config["representations"]) * len(eve_suffixes)
        n_suffix = len(eve_suffixes)
    else:
        n_experiments = len(config["representations"])
        n_suffix = 1
        eve_suffixes = [None]

    # Allocate result arrays
    metric_1_arr = np.zeros((n_experiments, 3))
    metric_2_arr = np.zeros((n_experiments, 3))
    metric_3_arr = np.zeros((n_experiments, 3))
    model_lst, suffix_lst = [], []
    # Counter to track experiments
    global_counter = 0

    for i in range(n_suffix):
        suffix = eve_suffixes[i]
        # Extract embeddings
        embeddings, y, names = extract_all_embeddings(
            df=df, dataset=dataset, embedding_type=embedding_type, suffix=suffix, target=target
        )
        # Chose part_0, part_1, part_2 as train, validation, test.
        # Extract inputs/targets
        train_idx = df["part_0"].values.astype(bool)
        val_idx = df["part_1"].values.astype(bool)
        test_idx = df["part_2"].values.astype(bool)

        embedding_train = embeddings[train_idx]
        embedding_val = embeddings[val_idx]
        embedding_test = embeddings[test_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]

        n_train, n_val, n_test = len(y_train), len(y_val), len(y_test)

        # Split index for CV
        split_index = np.repeat([-1, 0], [n_train, n_val])
        cv = PredefinedSplit(test_fold=split_index)
        for model_str, param_grid in zip(config["representations"], config["param_grids"]):
            # Define grid
            grid = GridSearchCV(
                estimator=config["pipe"],
                param_grid=param_grid,
                scoring=config["scoring"],
                verbose=0,
                n_jobs=threads,
                cv=cv,
            )

            # Fit all representations in grid
            grid.fit(np.concatenate((embedding_train, embedding_val)), np.concatenate((y_train, y_val)))
            # Extract and fit best model to training set
            model = grid.best_estimator_["model"].fit(embedding_train, y_train)

            # Get training, validation, and test predictions
            preds_train = model.predict(embedding_train)
            preds_val = model.predict(embedding_val)
            preds_test = model.predict(embedding_test)

            # Compute and save metrics
            metric_1, metric_2, metric_3 = compute_metrics(
                task, preds_train, preds_val, preds_test, y_train, y_val, y_test
            )
            metric_1_arr[global_counter] = metric_1
            metric_2_arr[global_counter] = metric_2
            metric_3_arr[global_counter] = metric_3
            # Fill in results
            model_lst.append(model_str)
            suffix_lst.append(suffix)
            global_counter += 1

    # Construct and return DataFrame
    df_results = pd.DataFrame(
        {
            "model": model_lst,
            "suffix": suffix_lst,
            f"train_{metric_1_str}": metric_1_arr[:, 0],
            f"val_{metric_1_str}": metric_1_arr[:, 1],
            f"test_{metric_1_str}": metric_1_arr[:, 2],
            f"train_{metric_2_str}": metric_2_arr[:, 0],
            f"val_{metric_2_str}": metric_2_arr[:, 1],
            f"test_{metric_2_str}": metric_2_arr[:, 2],
            f"train_{metric_3_str}": metric_3_arr[:, 0],
            f"val_{metric_3_str}": metric_3_arr[:, 1],
            f"test_{metric_3_str}": metric_3_arr[:, 2],
        }
    )

    df_results["embedding"] = embedding_type
    df_results["strategy"] = split_key

    return df_results


def fit_model_CV(
    dataset: str,
    embedding_type: str,
    n_partitions: int,
    eve_suffixes: Tuple[str, ...],
    threads: int = 20,
    ablation: bool = False,
    task: str = "regression",
    low_n: bool = False,
    save_predictions: bool = False,
) -> pd.DataFrame:
    """Function to evaluate regression performance using CV on dataset for a specific input representation.

    Args:
        dataset: Name of dataset (`cm`, `gh114`, `ppat`)
        embedding_type: Input representation/embedding name
        n_partitions: Number of CV partitions. Default is 3.
        eve_suffixes: Indicators for trained EVE models
        threads: Number of threads for multiprocessing.
        ablation: Boolean indicator for whether ablation study is being done. Affects the loading of the cm dataset.
        task: Either `regression` or `classification`
        low_n: Whether to limit number of neighbours in KNN
        save_predictions: Save predictions.

    Returns:
        DataFrame containing metrics and evaluation details.
    """
    # Load dataset
    df = pd.read_csv(f"data/processed/{dataset}/{dataset}.csv", index_col=0)

    # Define regression parameters
    target = "target_reg" if task == "regression" else "target_class"
    config = setup_CV_grids(task, low_n)
    if task == "regression":
        metric_1_str, metric_2_str, metric_3_str = "spearman", "rmse", "mae"
    else:
        metric_1_str, metric_2_str, metric_3_str = "mcc", "auroc", "f1"

    # If running ablation study, use inactive + active sequences for cm dataset.
    if (ablation and dataset == "cm" and task == "regression") or (dataset == "cm" and task == "classification"):
        df = pd.read_csv(f"data/processed/{dataset}/{dataset}_all.csv", index_col=0)

    partition_headers = [f"part_{i}" for i in range(n_partitions)]
    if embedding_type == "eve":
        n_experiments = int(
            len(config["representations"])
            * len(eve_suffixes)
            * int(factorial(n_partitions) / (factorial(n_partitions - 3)))
        )
        n_suffix = len(eve_suffixes)
    else:
        n_experiments = len(config["representations"]) * int(factorial(n_partitions) / (factorial(n_partitions - 3)))
        n_suffix = 1
        eve_suffixes = [None]

    if save_predictions:
        df_predictions = pd.DataFrame(
            columns=["target", "predictions", "regressor", "embedding", "name", "combination"]
        )

    # Allocate result arrays
    metric_1_arr = np.zeros((n_experiments, 3))
    metric_2_arr = np.zeros((n_experiments, 3))
    metric_3_arr = np.zeros((n_experiments, 3))
    model_lst, suffix_lst, train_id, val_id, test_id = [], [], [], [], []

    # Counter to track experiments
    global_counter = 0

    for i in range(n_suffix):
        suffix = eve_suffixes[i]

        # Extract embeddings
        embeddings, y, names = extract_all_embeddings(
            df=df, dataset=dataset, embedding_type=embedding_type, suffix=suffix, target=target
        )

        # Iterate through all CV partition permutations
        for perm_idx, perm in enumerate(permutations(partition_headers, 3)):
            # Extract inputs/targets
            train_idx = df[perm[0]].values.astype(bool)
            val_idx = df[perm[1]].values.astype(bool)
            test_idx = df[perm[2]].values.astype(bool)

            embedding_train = embeddings[train_idx]
            embedding_val = embeddings[val_idx]
            embedding_test = embeddings[test_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]
            y_test = y[test_idx]

            # Split index for CV in sklearn
            n_train = len(embedding_train)
            n_val = len(embedding_val)
            split_index = np.repeat([-1, 0], [n_train, n_val])
            cv = PredefinedSplit(test_fold=split_index)

            for model_str, param_grid in zip(config["representations"], config["param_grids"]):
                # Define grid
                grid = GridSearchCV(
                    estimator=config["pipe"],
                    param_grid=param_grid,
                    scoring=config["scoring"],
                    verbose=0,
                    n_jobs=threads,
                    cv=cv,
                )

                # Fit all representations in grid
                grid.fit(np.concatenate((embedding_train, embedding_val)), np.concatenate((y_train, y_val)))
                # Extract and fit best model to training set
                model = grid.best_estimator_["model"].fit(embedding_train, y_train)

                # Get training, validation, and test predictions
                preds_train = model.predict(embedding_train)
                preds_val = model.predict(embedding_val)
                preds_test = model.predict(embedding_test)

                # Compute and save metrics
                metric_1, metric_2, metric_3 = compute_metrics(
                    task, preds_train, preds_val, preds_test, y_train, y_val, y_test
                )
                metric_1_arr[global_counter] = metric_1
                metric_2_arr[global_counter] = metric_2
                metric_3_arr[global_counter] = metric_3
                # Fill in results
                model_lst.append(model_str)
                suffix_lst.append(suffix)
                train_id.append(perm[0])
                val_id.append(perm[1])
                test_id.append(perm[2])
                global_counter += 1

                if save_predictions:
                    df_p = pd.DataFrame({"target": y_test, "predictions": preds_test})
                    df_p["regressor"] = model_str
                    df_p["embedding"] = embedding_type
                    df_p["name"] = np.array(names)[test_idx]
                    df_p["combination"] = perm_idx
                    df_predictions = pd.concat((df_predictions, df_p))

    if save_predictions:
        df_predictions = df_predictions.reset_index(drop=True)
        df_predictions.to_csv(f"predictions/{dataset}/{dataset}_{embedding_type}.csv")

    # Construct and return DataFrame
    df_results = pd.DataFrame(
        {
            "model": model_lst,
            "suffix": suffix_lst,
            f"train_{metric_1_str}": metric_1_arr[:, 0],
            f"val_{metric_1_str}": metric_1_arr[:, 1],
            f"test_{metric_1_str}": metric_1_arr[:, 2],
            f"train_{metric_2_str}": metric_2_arr[:, 0],
            f"val_{metric_2_str}": metric_2_arr[:, 1],
            f"test_{metric_2_str}": metric_2_arr[:, 2],
            f"train_{metric_3_str}": metric_3_arr[:, 0],
            f"val_{metric_3_str}": metric_3_arr[:, 1],
            f"test_{metric_3_str}": metric_3_arr[:, 2],
            "train_id": train_id,
            "val_id": val_id,
            "test_id": test_id,
        }
    )

    df_results["embedding"] = embedding_type
    df_results["strategy"] = "CV"

    if embedding_type != "eve":
        df_results["suffix"] = df_results["suffix"].replace({None: np.nan})
    return df_results


def setup_CV_grids(task: str, low_n: bool = False) -> Dict:
    """Function to create and return parameters for sklearn grid search

    Args:
        task: Either regression or classification
        low_n: Will limit max number in KNN to 5. For small datasets.

    Returns:
        Returns dictionary over grid parameters.
    """
    neighbours = [1, 2, 5] if low_n else [1, 2, 5, 10, 25]

    if task == "regression":
        model_strs = ["KNN", "Ridge", "RandomForest", "MLP"]
        scoring = "neg_mean_squared_error"
        knn_grid = [{"model": [KNeighborsRegressor()], "model__n_neighbors": neighbours}]

        lm_grid = [
            {
                "model": [Ridge(random_state=0)],
                "model__alpha": [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 10, 25, 50, 100],
            }
        ]

        rf_grid = [
            {
                "model": [RandomForestRegressor(random_state=0)],
                "model__max_features": ["sqrt", "log2"],
                "model__min_samples_split": [2, 5],
                "model__n_estimators": [100, 200],
            }
        ]

        mlp_grid = [
            {
                "model": [MLPRegressor(random_state=0, max_iter=2000)],
                "model__hidden_layer_sizes": [(100,), (10,)],
                "model__alpha": [0.0001, 0.01, 0],
                "model__solver": ["lbfgs", "adam"],
            }
        ]
    elif task == "classification":
        scoring = "neg_log_loss"
        model_strs = ["KNN", "LogReg", "RandomForest", "MLP"]
        knn_grid = [{"model": [KNeighborsClassifier()], "model__n_neighbors": neighbours}]

        lm_grid = [
            {
                "model": [LogisticRegression(random_state=0, penalty="l2", max_iter=1000)],
                "model__C": [0.1, 0.15, 0.20, 1, 5, 10, 15, 20, 25, 30, 40, 50],
            }
        ]

        rf_grid = [
            {
                "model": [RandomForestClassifier(random_state=0)],
                "model__max_features": ["sqrt", "log2"],
                "model__min_samples_split": [2, 5],
                "model__n_estimators": [100, 200],
            }
        ]
        mlp_grid = [
            {
                "model": [MLPClassifier(random_state=0, max_iter=2000)],
                "model__hidden_layer_sizes": [(100,), (10,)],
                "model__alpha": [0.0001, 0.01, 0],
                "model__solver": ["lbfgs", "adam"],
            }
        ]
    else:
        raise NotImplementedError

    param_grid_list = [knn_grid, lm_grid, rf_grid, mlp_grid]
    pipe = Pipeline([("model", "passthrough")])

    return {"param_grids": param_grid_list, "pipe": pipe, "representations": model_strs, "scoring": scoring}


def compute_metrics(
    task: str,
    preds_train: np.ndarray,
    preds_val: np.ndarray,
    preds_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    constant: bool = False,
) -> Tuple[np.array, np.array, np.array]:
    """Function to compute classification/regression metrics for train/val/test predictions.

    Args:
        task: One of classification/regression
        preds_train: Array over predictions on training partition
        preds_val: Array over predictions on validation partition
        preds_test: Array over predictions on test partition
        y_train: Array of target values of training partition
        y_val: Array of target values of validation partition
        y_test: Array of target values of test partition
        constant: In case of constant input, do not compute Spearman correlation (as it will produce errors).

    Returns:
        Returns two array of length 3 containing MCC/AUROC/F1 for classification and Spearman correlation/RMSE/MAE for
        regression.
    """
    if task == "classification":
        corr_1 = matthews_corrcoef(y_train, preds_train)
        corr_2 = matthews_corrcoef(y_val, preds_val)
        corr_3 = matthews_corrcoef(y_test, preds_test)
        auroc_1 = roc_auc_score(y_train, preds_train)
        auroc_2 = roc_auc_score(y_val, preds_val)
        auroc_3 = roc_auc_score(y_test, preds_test)
        f1_1 = f1_score(y_train, preds_train, average="micro")
        f1_2 = f1_score(y_val, preds_val, average="micro")
        f1_3 = f1_score(y_test, preds_test, average="micro")
        corr = np.array((corr_1, corr_2, corr_3))
        auroc = np.array((auroc_1, auroc_2, auroc_3))
        f1 = np.array((f1_1, f1_2, f1_3))
        return corr, auroc, f1
    elif task == "regression":
        if not constant:
            corr_1, _ = spearmanr(y_train, preds_train)
            corr_2, _ = spearmanr(y_val, preds_val)
            corr_3, _ = spearmanr(y_test, preds_test)
            corr = np.array((corr_1, corr_2, corr_3))
        else:
            corr = None
        rmse_1 = mean_squared_error(y_train, preds_train, squared=False)
        rmse_2 = mean_squared_error(y_val, preds_val, squared=False)
        rmse_3 = mean_squared_error(y_test, preds_test, squared=False)
        mae_1 = mean_absolute_error(y_train, preds_train)
        mae_2 = mean_absolute_error(y_val, preds_val)
        mae_3 = mean_absolute_error(y_test, preds_test)
        rmse = np.array((rmse_1, rmse_2, rmse_3))
        mae = np.array((mae_1, mae_2, mae_3))
        return corr, rmse, mae
    else:
        raise ValueError


def summarise_results(
    dataset: str,
    metric: str,
    ablation: bool,
    embedding_types: Tuple[str, ...],
    all: bool = False,
    unsupervised: bool = False,
    ablation_method: str = None,
):
    """Function to aggregate results for easy analysis/visualization.

    Args:
        dataset: One of `cm`, `gh114`, or`ppat`
        metric: One of (`spearman`,`rmse`,`mae`) for regression task and (`mcc`,`auroc`,`f1`) for classification.
        ablation: If ablation study, load hardcoded files.
        embedding_types: Tuple of string names for embeddings to include in results.
        all: Add to include training and validation metrics.
        unsupervised: Include to include zero-shot correlations for EVE and ESM-IF1
        ablation_method: Include to manually choose which ablation study for the dataset (thus ignoring predefined
        choices)

    Returns: Grouped DataFrame with mean values and standard errors over metrics, grouped by regression model and
    input representations.

    """
    df = pd.DataFrame()
    for embedding_type in embedding_types:
        if not ablation:
            if metric in ["mcc", "auroc", "f1"]:
                df_embed = pd.read_csv(
                    f"results/{dataset}/{dataset}_results_{embedding_type}_classification.csv", index_col=0
                )
            else:
                df_embed = pd.read_csv(f"results/{dataset}/{dataset}_results_{embedding_type}.csv", index_col=0)
        else:
            if ablation_method is None:
                ablation_dict = {"cm": "all", "gh114": "holdout", "ppat": "random"}
                df_embed = pd.read_csv(
                    f"results/ablation/{dataset}_results_{embedding_type}_{ablation_dict[dataset]}.csv", index_col=0
                )
            else:
                df_embed = pd.read_csv(
                    f"results/ablation/{dataset}_results_{embedding_type}_{ablation_method}.csv", index_col=0
                )
        df = pd.concat((df, df_embed))
    if all:
        df_grouped = df.groupby(["model", "embedding"])[[f"train_{metric}", f"val_{metric}", f"test_{metric}"]].agg(
            ["mean", "sem"]
        )
    else:
        df_grouped = df.groupby(["model", "embedding"])[[f"test_{metric}"]].agg(["mean", "sem"])

    if unsupervised:
        # Load unsupervised scores
        df_unsup = pd.read_csv(f"results/unsupervised_correlations_cv.csv", index_col=0)
        df_unsup = df_unsup[df_unsup["dataset"] == dataset]
        df_grouped.loc[("Ridge", "EVE*"), ("test_spearman", "mean")] = df_unsup["spearman_elbo"].item()
        df_grouped.loc[("Ridge", "EVE*"), ("test_spearman", "sem")] = df_unsup["spearman_elbo_sem"].item()
        df_grouped.loc[("Ridge", "ESM-IF1*"), ("test_spearman", "mean")] = df_unsup["spearman_esm_if1"].item()
        df_grouped.loc[("Ridge", "ESM-IF1*"), ("test_spearman", "sem")] = df_unsup["spearman_esm_if1_sem"].item()
        df_grouped.loc[("Ridge", "Tranception*"), ("test_spearman", "mean")] = df_unsup["spearman_tranception"].item()
        df_grouped.loc[("Ridge", "Tranception*"), ("test_spearman", "sem")] = df_unsup[
            "spearman_tranception_sem"
        ].item()

    return df_grouped
