"""Script to summarize, save, and plot results for a dataset"""
import argparse
from typing import Tuple

import pandas as pd

from src.data.data_utils import repr_dict
from src.visualization.visualization_funcs import results_single_dataset
from src.training.training_utils import summarise_results
from src import REPRESENTATIONS, DATASETS, REPRESENTATION_ORDER


def print_results_table(
    datasets: Tuple[str, ...], metric: str, embedding_types: Tuple[str, ...]
):
    """Function to print latex-formatted table for manuscript.

    Args:
        datasets: Datasets to include
        metric: Chosen metric
        ablation: Whether ablation results or not
        embedding_types: Input representations to include

    """
    df_latex = pd.DataFrame(index=embedding_types)
    predictor = "RandomForest" if metric in ["spearman", "rmse", "mae"] else "LogReg"
    for dataset in datasets:
        df_raw = summarise_results(dataset, metric, False, embedding_types)
        # Clean dataframe
        df = df_raw.loc[(f"{predictor}",)]
        df = df.astype(float)
        df = df.round(decimals=2)
        df_latex[dataset] = (
            "$"
            + df[(f"test_{metric}", "mean")].astype(str)
            + " \pm "
            + df[(f"test_{metric}", "sem")].astype(str)
            + "$"
        )

    embedding_names = repr_dict()
    df_latex = df_latex.rename(index=embedding_names)
    df_latex = df_latex.reindex(REPRESENTATION_ORDER)

    print(
        df_latex.style.to_latex(
            hrules=True,
            column_format="l|ccc",
            caption=f"{metric}, predictor: {predictor}",
            position_float="centering",
        )
    )

    df_unsupervised = pd.read_csv(
        f"results/unsupervised_correlations.csv", index_col=0
    ).set_index("dataset")
    df_unsupervised = df_unsupervised.astype(float)
    df_unsupervised = df_unsupervised.round(decimals=2)
    df_latex_2 = pd.DataFrame(
        index=["ESM-IF1$^\dagger$", "EVE$^\dagger$", "Tranception$^\dagger$"],
        columns=datasets,
    )
    for dataset in datasets:
        df_latex_2.loc["EVE$^\dagger$", dataset] = (
            "\multicolumn{1}{l}{$"
            + df_unsupervised.loc[dataset, "spearman_elbo"].astype(str)
            + "$}"
        )
        df_latex_2.loc["ESM-IF1$^\dagger$", dataset] = (
            "\multicolumn{1}{l}{$"
            + df_unsupervised.loc[dataset, "spearman_esm_if1"].astype(str)
            + "$}"
        )
        df_latex_2.loc["Tranception$^\dagger$", dataset] = (
            "\multicolumn{1}{l}{$"
            + df_unsupervised.loc[dataset, "spearman_tranception"].astype(str)
            + "$}"
        )

    df_latex_2 = df_latex_2.rename(index=embedding_names)
    df_latex_2 = df_latex_2.sort_index()

    print(
        df_latex_2.style.to_latex(
            hrules=True,
            column_format="l|rrr",
            caption=f"{metric}, predictor: {predictor}",
            position_float="centering",
        )
    )


def print_results_table_ablation(
    datasets: Tuple[str, ...], metric: str, embedding_types: Tuple[str, ...]
):
    """Function to print latex-formatted table for manuscript.

    Args:
        datasets: Datasets to include
        metric: Chosen metric
        ablation: Whether ablation results or not
        embedding_types: Input representations to include

    """
    embedding_names = repr_dict()
    predictor = "RandomForest" if metric in ["spearman", "rmse", "mae"] else "LogReg"

    # Load ablation results in numeric and latex formats
    df_latex = pd.DataFrame(index=embedding_types)
    df_ablation = pd.DataFrame(index=embedding_types)
    for dataset in datasets:
        df_raw = summarise_results(dataset, metric, True, embedding_types)
        # Clean dataframe
        df = df_raw.loc[(f"{predictor}",)]
        df = df.astype(float)
        df_ablation[dataset] = df[(f"test_{metric}", "mean")]
        df = df.round(decimals=2)
        df_latex[dataset] = (
            "$"
            + df[(f"test_{metric}", "mean")].astype(str)
            + " \pm "
            + df[(f"test_{metric}", "sem")].astype(str)
            + "$"
        )

    df_ablation = df_ablation.rename(index=embedding_names) 
    df_ablation = df_ablation.reindex(REPRESENTATION_ORDER)
    df_latex = df_latex.rename(index=embedding_names)
    df_latex = df_latex.reindex(REPRESENTATION_ORDER)

    # Load main results to compute delta
    df_main = pd.DataFrame(index=embedding_types)
    for dataset in datasets:
        df_raw = summarise_results(dataset, metric, False, embedding_types)
        # Clean dataframe
        df = df_raw.loc[(f"{predictor}",)]
        df = df.astype(float)
        df_main[dataset] = df[(f"test_{metric}", "mean")]
    df_main = df_main.rename(index=embedding_names)
    df_main = df_main.reindex(REPRESENTATION_ORDER)

    # Compute delta
    df_delta = df_ablation - df_main
    df_delta = df_delta.round(decimals=2)
    df_delta = df_delta.rename(
        columns={dataset: f"$\delta$ {dataset}" for dataset in datasets}
    )
    for col in df_delta.columns:
        # Make cells green and red for positive and negative deltas, respectively
        df_delta[col] = df_delta[col].apply(
            lambda x: f"\\cellcolor{{green!25}}$+{x}$"
            if x > 0
            else f"\\cellcolor{{red!25}}${x}$"
        )

    # Add deltas to latex table
    df_latex = pd.concat([df_latex, df_delta], axis=1)
    # Reorder columns
    order = []
    for dataset in datasets:
        order.append(dataset)
        order.append(f"$\delta$ {dataset}")

    df_latex = df_latex[order]

    print(
        df_latex.style.to_latex(
            hrules=True,
            column_format="l|llllll",
            caption=f"{metric}, predictor: {predictor}",
            position_float="centering",
        )
    )


def main(
    dataset: str,
    embedding_types: tuple,
    to_latex: bool,
    save_csv: bool,
    metric: str,
    ablation: bool = False,
    image_format: str = "pdf",
    bar_plot: bool = False,
    path: str = None,
):
    if to_latex:
        if ablation:
            print_results_table_ablation(
                datasets=DATASETS, metric=metric, embedding_types=embedding_types
            )
        else:
            print_results_table(
                datasets=DATASETS, metric=metric, embedding_types=embedding_types
            )

    if save_csv:
        df = summarise_results(dataset, metric, ablation, embedding_types)
        df.to_csv(f"results/{dataset}/{dataset}_summary.csv")

    if bar_plot:
        results_single_dataset(
            dataset=dataset,
            embedding_types=embedding_types,
            metric=metric,
            save_fig=True,
            path=path,
            image_format=image_format,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Name of dataset.")
    parser.add_argument(
        "--metric", type=str, default="spearman", help="Which metric to use."
    )
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--image_format", type=str, default="pdf")
    parser.add_argument(
        "--ablation", action="store_true", help="If results from ablation study."
    )
    parser.add_argument(
        "--to_latex", action="store_true", help="Generates latex-formatted tables."
    )
    parser.add_argument(
        "--save_csv", action="store_true", help="Saves summary of results."
    )
    parser.add_argument(
        "--bar_plot", action="store_true", help="Draw and save bar-plot over results."
    )
    parser.add_argument(
        "--embedding_types", type=str, nargs="+", default=REPRESENTATIONS
    )

    args = parser.parse_args()
    main(**vars(args))
