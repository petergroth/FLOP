"""Script to summarize, save, and plot results for a dataset"""
import argparse
from typing import Tuple

import pandas as pd

from src.data.data_utils import repr_dict
from src.visualization.visualization_funcs import test_results_barplot
from src.training.training_utils import summarise_results


def print_latex_tables(datasets: Tuple[str, ...], metric: str, ablation: bool, embedding_types: Tuple[str, ...]):
    """Function to print latex-formatted table for manuscript.

    Args:
        datasets: Datasets to include
        metric: Chosen metric
        ablation: Whether ablation results or not
        embedding_types: Input representations to include

    """
    df_latex = pd.DataFrame(index=embedding_types)
    predictor = "Ridge" if metric in ["spearman", "rmse", "mae"] else "LogReg"
    for dataset in datasets:
        df_raw = summarise_results(dataset, metric, ablation, embedding_types)
        # Clean dataframe
        df = df_raw.loc[(f"{predictor}",)]
        df = df.astype(float)
        df = df.round(decimals=2)
        df_latex[dataset] = (
            "$" + df[(f"test_{metric}", "mean")].astype(str) + " \pm " + df[(f"test_{metric}", "sem")].astype(str) + "$"
        )

    embedding_names = repr_dict()
    df_latex = df_latex.rename(index=embedding_names)
    df_latex = df_latex.sort_index()

    print(
        df_latex.style.to_latex(
            hrules=True, column_format="l|ccc", caption=f"{metric}, ablation: {ablation}", position_float="centering"
        )
    )

    # Print unsupervised scores
    if not ablation:
        df_unsupervised = pd.read_csv(f"results/unsupervised_correlations_cv.csv", index_col=0).set_index("dataset")
    else:
        df_unsupervised = pd.read_csv(f"results/unsupervised_ablation_correlations_cv.csv", index_col=0).set_index(
            "dataset"
        )
    df_unsupervised = df_unsupervised.astype(float)
    df_unsupervised = df_unsupervised.round(decimals=2)
    df_latex_2 = pd.DataFrame(index=["ESM-IF1*", "EVE*", "Tranception*"], columns=datasets)
    for dataset in datasets:
        df_latex_2.loc["EVE*", dataset] = (
            "$"
            + df_unsupervised.loc[dataset, "spearman_elbo"].astype(str)
            + " \pm "
            + df_unsupervised.loc[dataset, "spearman_elbo_sem"].astype(str)
            + "$"
        )
        df_latex_2.loc["ESM-IF1*", dataset] = (
            "$"
            + df_unsupervised.loc[dataset, "spearman_esm_if1"].astype(str)
            + " \pm "
            + df_unsupervised.loc[dataset, "spearman_esm_if1_sem"].astype(str)
            + "$"
        )
        df_latex_2.loc["Tranception*", dataset] = (
            "$"
            + df_unsupervised.loc[dataset, "spearman_tranception"].astype(str)
            + " \pm "
            + df_unsupervised.loc[dataset, "spearman_tranception_sem"].astype(str)
            + "$"
        )

    df_latex_2 = df_latex_2.rename(index=embedding_names)
    df_latex_2 = df_latex_2.sort_index()

    print(
        df_latex_2.style.to_latex(
            hrules=True, column_format="l|ccc", caption=f"{metric}, ablation: {ablation}", position_float="centering"
        )
    )

    df_unsupervised = pd.read_csv(f"results/unsupervised_correlations.csv", index_col=0).set_index("dataset")
    df_unsupervised = df_unsupervised.astype(float)
    df_unsupervised = df_unsupervised.round(decimals=2)
    df_latex_2 = pd.DataFrame(index=["ESM-IF1**", "EVE**", "Tranception**"], columns=datasets)
    for dataset in datasets:
        df_latex_2.loc["EVE**", dataset] = (
            "\multicolumn{1}{l}{" + "$" + df_unsupervised.loc[dataset, "spearman_elbo"].astype(str) + "$" + "}"
        )
        df_latex_2.loc["ESM-IF1**", dataset] = (
            "\multicolumn{1}{l}{" + "$" + df_unsupervised.loc[dataset, "spearman_esm_if1"].astype(str) + "$" + "}"
        )
        df_latex_2.loc["Tranception**", dataset] = (
            "\multicolumn{1}{l}{" + "$" + df_unsupervised.loc[dataset, "spearman_tranception"].astype(str) + "$" + "}"
        )

    df_latex_2 = df_latex_2.rename(index=embedding_names)
    df_latex_2 = df_latex_2.sort_index()

    print(
        df_latex_2.style.to_latex(
            hrules=True, column_format="l|rrr", caption=f"{metric}, ablation: {ablation}", position_float="centering"
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
    bar_plot: bool = True,
    path: str = None,
):
    if to_latex:
        print_latex_tables(
            datasets=("gh114", "cm", "ppat"), metric=metric, ablation=ablation, embedding_types=embedding_types
        )

    if save_csv:
        df = summarise_results(dataset, metric, ablation, embedding_types)
        df.to_csv(f"results/{dataset}/{dataset}_summary.csv")

    if bar_plot:
        test_results_barplot(
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
    parser.add_argument("--metric", type=str, default="spearman", help="Which metric to use.")
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--image_format", type=str, default="pdf")
    parser.add_argument("--ablation", action="store_true", help="If results from ablation study.")
    parser.add_argument("--to_latex", action="store_true", help="Generates latex-formatted tables.")
    parser.add_argument("--save_csv", action="store_true", help="Saves summary of results.")
    parser.add_argument("--bar_plot", action="store_true", help="Draw and save bar-plot over results.")
    parser.add_argument(
        "--embedding_types", type=str, nargs="+", default=("af2", "esm_1b", "esm_2", "esm_if1", "eve", "onehot", "ct")
    )

    args = parser.parse_args()
    main(**vars(args))
