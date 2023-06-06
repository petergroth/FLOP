from pathlib import Path
from typing import Tuple, Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import umap
from sklearn.decomposition import PCA

from src import REPRESENTATION_ORDER
from src.data.data_utils import repr_dict
from src.training.training_utils import extract_all_embeddings, summarise_results

# Prevent printing loading messages on pdfs with plotly
pio.kaleido.scope.mathjax = None


def show_target_histogram(
    df: pd.DataFrame, dataset: str, target: Union[str, List[str]] = "target"
):
    """Function to visualize target distribution of one dataset during its compilation

    Args:
        df: Dataset
        dataset: Dataset string name
        target: Target column

    """
    sns.set_style("dark")
    task = "regression" if target == "target_reg" else "classification"
    fig_dir = Path("figures", dataset)
    fig_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(target, str):
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.histplot(
            data=df[df[["part_0", "part_1", "part_2"]].sum(axis=1) == 1],
            x=target,
            color="#F58A00",
            bins=2 if target == "target_class" else 10,
        )
        ax.set_title(
            f"Histogram over {task} target values for {dataset.upper()} dataset."
        )
        plt.savefig(fig_dir / f"targets_{task}_histogram.png")
        print(f"Saved plot in {fig_dir / f'targets_{task}_histogram.png'}")
        plt.show()
    elif isinstance(target, list):
        n = len(target)
        fig, ax = plt.subplots(n // 2, 2, figsize=(10, 5 * (n // 2)))
        for i, tar in enumerate(target):
            axi = ax.flatten()[i]
            sns.histplot(
                data=df,
                x=tar,
                color="#F58A00",
                ax=axi,
                bins=2 if target == "target_class" else 10,
            )
            axi.set_title(f"Target = {tar}")
        plt.suptitle(f"Histogram over {task} targets for {dataset.upper()} dataset.")
        plt.savefig(fig_dir / f"targets_{task}_histogram.png")
        print(f"Saved plot in {fig_dir / f'targets_{task}_histogram.png'}")
        plt.show()


def show_CV_split_distribution(
    df: pd.DataFrame, threshold: Union[float, None], dataset: str, n_partitions: int
):
    # Setup plotting
    sns.set_style("dark")
    color_palette = sns.color_palette("colorblind")
    y_limits = (df["target_reg"].min(), df["target_reg"].max())
    n_eff = int(df[[f"part_{i}" for i in range(n_partitions)]].sum().sum())

    fig, ax = plt.subplots(
        n_partitions, 2, figsize=(10, 5 * n_partitions), sharex="col"
    )

    # Histogram over target values
    for i in range(n_partitions):
        axi = ax[i, 0]
        sns.histplot(
            data=df[df[f"part_{i}"] == 1],
            x="target_reg",
            ax=axi,
            bins=12,
            binrange=y_limits,
            color=color_palette[i],
            alpha=0.9,
        )
        axi.set_title(f"Target values (partition {i + 1})")

        axi = ax[i, 1]
        sns.countplot(
            data=df[df[f"part_{i}"] == 1],
            x="target_class",
            ax=axi,
            color=color_palette[i],
            alpha=0.9,
        )
        axi.set_title(f"Binarized target (partition {i + 1})")

    plt.suptitle(
        f"{dataset.upper()} dataset overview.\n"
        f"Cross-validation partitions (K={n_partitions}) at threshold {threshold}.\n"
        f"N = {len(df)}, Neff = {n_eff}.\n"
        f"N_i = {df[[f'part_{i}' for i in range(n_partitions)]].sum().tolist()}/{n_eff}.",
        fontsize="x-large",
    )

    [axi.set_ylabel("Count") for axi in ax.flatten()]
    [axi.set_xlabel("") for axi in ax.flatten()]

    fig_dir = Path("figures", dataset, "splits")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f"{dataset}_CV_K={n_partitions}_distribution.pdf")
    plt.savefig(fig_dir / f"{dataset}_CV_K={n_partitions}_distribution.png")
    plt.show()


def visualize_embedding(
    dataset: str, embedding_type: str, suffix: Union[str, None] = None
):
    """Function to visualize embedding in 2D using UMAP and PCA

    Args:
        dataset: Name of dataset
        embedding_type: Which data representation to visualize
        suffix: Specify which EVE model (if embedding_type == "EVE (z)")

    """
    np.random.seed(42)
    if embedding_type == "eve":
        assert suffix is not None

    df = pd.read_csv(f"data/processed/{dataset}/{dataset}.csv", index_col=0)
    df = df[df[["part_0", "part_1", "part_2"]].sum(axis=1) == 1.0]
    df["split"] = ""
    df.loc[df["part_0"] == 1, "split"] = "part_0"
    df.loc[df["part_1"] == 1, "split"] = "part_1"
    df.loc[df["part_2"] == 1, "split"] = "part_2"
    split = df["split"].values

    # Extract embeddings
    embeddings, y, names = extract_all_embeddings(
        df=df,
        dataset=dataset,
        embedding_type=embedding_type,
        target="target",
        suffix=suffix,
    )

    # PCA
    x_pca = PCA(n_components=2).fit_transform(embeddings)

    # UMAP
    x_umap = umap.UMAP(random_state=42).fit_transform(embeddings)

    # Setup plotting
    sns.set_style("dark")
    s = 80

    df_plot = pd.DataFrame(
        {
            "x_pca": x_pca[df.index.values, 0],
            "y_pca": x_pca[df.index.values, 1],
            "x_umap": x_umap[df.index.values, 0],
            "y_umap": x_umap[df.index.values, 1],
            "target": y[df.index.values],
            "split": split[df.index.values],
        }
    )
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    sns.scatterplot(
        data=df_plot,
        x="x_pca",
        y="y_pca",
        hue="split",
        ax=ax[0, 0],
        s=s,
        style="split",
        alpha=0.5,
    )
    sns.scatterplot(
        data=df_plot, x="x_pca", y="y_pca", hue="target", ax=ax[1, 0], s=s, alpha=0.5
    )
    sns.scatterplot(
        data=df_plot,
        x="x_umap",
        y="y_umap",
        hue="split",
        ax=ax[0, 1],
        s=s,
        style="split",
        alpha=0.5,
    )
    sns.scatterplot(
        data=df_plot, x="x_umap", y="y_umap", hue="target", ax=ax[1, 1], s=s, alpha=0.5
    )

    ax[0, 0].set_title(f"PCA")
    ax[1, 0].set_title(f"PCA")
    ax[0, 1].set_title(f"UMAP")
    ax[1, 1].set_title(f"UMAP")

    plt.suptitle(f"{embedding_type} embeddings on {dataset.upper()} dataset")
    plt.subplots_adjust(hspace=0.3)

    path = f"figures/{dataset}/representations/{dataset}_{embedding_type}_dimensionality_reduction"
    plt.savefig(f"{path}.pdf")
    plt.savefig(f"{path}.png")
    print(f"Saved figure in {path}.<pdf,png>")
    plt.show()


def results_single_dataset(
    dataset: str,
    embedding_types: Tuple[str, ...],
    metric: str,
    save_fig: bool = True,
    image_format: str = "png",
    ablation: bool = False,
    ablation_method: str = None,
    **kwargs,
):
    """Visualize results for KNN, linear model, and random forest regressors/classifiers for one dataset.

    Args:
        dataset: Name of dataset
        embedding_types: Which representations to include
        metric: Which metric to include
        save_fig: Include to save.
        image_format: Output file format. PDF for report.
        ablation: If results are from ablation study.
        ablation_method: Specify ablation method.

    """

    df = summarise_results(
        dataset=dataset,
        metric=metric,
        embedding_types=embedding_types,
        all=False,
        ablation=ablation,
        ablation_method=ablation_method,
    )
    # Alter multiindex in rows/cols
    df = df.reset_index()
    df.columns = list(map("_".join, df.columns.values))
    df = df.rename(columns={"model_": "model", "embedding_": "embedding"})

    # Rename embedding names
    df["embedding"] = df["embedding"].replace(repr_dict())

    fig = px.bar(
        data_frame=df,
        x="embedding",
        color="embedding",
        y=f"test_{metric}_mean",
        error_y=f"test_{metric}_sem",
        facet_col="model",
        width=900,
        height=350,
        color_discrete_sequence=px.colors.qualitative.Plotly,
        category_orders={
            "model": ["KNN", "Ridge", "RandomForest", "MLP"]
            if metric in ["rmse", "spearman", "mae"]
            else ["KNN", "LogReg", "RandomForest", "MLP"],
            "embedding": REPRESENTATION_ORDER,
        },
        title=f"Dataset: {dataset.upper()}. Metric: {metric.capitalize()}",
    )
    fig.update_layout(
        font={"family": "Times New Roman"},
        yaxis={
            "visible": True,
            "showticklabels": True,
            "title": {"text": metric.capitalize(), "font": {"size": 20}},
            "tickfont": {"size": 15},
        },
        legend={
            "title": {"text": "Embedding", "font": {"size": 18}},
            "font": {"size": 14},
        },
        xaxis={"title": {"text": " "}},
        xaxis1={"title": {"text": " "}},
        xaxis2={"title": {"text": " "}},
        xaxis3={"title": {"text": " "}},
        title={"x": 0.5},
        margin={"b": 0, "l": 60}
    )
    fig.update_xaxes(showticklabels=False)
    # Fix axes
    if metric in ["auroc", "spearman", "mcc", "f1"]:
        fig.update_yaxes(range=[0, 1])
    else:
        fig.update_yaxes(matches=None, showticklabels=True)

    # Updated column headers
    for i, reg in enumerate(["KNN", "Ridge", "RandomForest", "MLP"]):
        fig["layout"]["annotations"][i]["text"] = reg
    # Updated y-label
    fig["layout"]["yaxis"]["title"]["text"] = (
        metric.capitalize() if metric == "spearman" else metric.upper()
    )
    # Center title

    if save_fig:
        if ablation:
            if ablation_method is None:
                path = f"figures/{dataset}/results/test_ablation_results_{metric}"
            else:
                path = f"figures/{dataset}/results/test_ablation_{ablation_method}_results_{metric}"
        else:
            path = f"figures/{dataset}/results/test_results_{metric}"
        pio.write_image(fig, f"{path}.{image_format}", format=image_format)
        print(f"Saved figure to {path}.{image_format}.")
    else:
        fig.show()


def all_test_results_barplot(
    datasets: Tuple[str, ...],
    regressor: str,
    embedding_types: Tuple[str, ...],
    metric: str,
    save_fig: bool = True,
    image_format: str = "png",
    unsupervised: bool = True,
    ablation: bool = False,
):
    """Visualize results for report.

    Args:
        datasets: Which three datasets to show.
        regressor: Regressor name (Ridge)
        embedding_types: Input representations to include.
        metric: Metric of interest.
        save_fig: Include to save locally.
        image_format: Output file-format.
        unsupervised: Include to add unsupervised correlations as dotted/dashed lines.
        ablation: Include to get ablation study results.


    """
    df = pd.DataFrame()
    for dataset in datasets:
        df_i = summarise_results(
            dataset=dataset,
            metric=metric,
            embedding_types=embedding_types,
            all=False,
            ablation=ablation,
            unsupervised=False,
        )
        df_i = df_i.loc[(regressor)]
        df_i["dataset"] = dataset
        df = pd.concat((df, df_i))

    # Alter multiindex in rows/cols
    df = df.reset_index()
    df.columns = list(map("_".join, df.columns.values))
    df = df.rename(
        columns={"model_": "model", "embedding_": "embedding", "dataset_": "dataset"}
    )

    # Rename embedding names
    df["embedding"] = df["embedding"].replace(repr_dict())

    fig = px.bar(
        data_frame=df,
        x="embedding",
        color="embedding",
        y=f"test_{metric}_mean",
        error_y=f"test_{metric}_sem",
        facet_col="dataset",
        width=900,
        height=300,
        color_discrete_sequence=px.colors.qualitative.Plotly,
        category_orders={
            "model": datasets,
            "embedding": REPRESENTATION_ORDER,
        },
        # title=f"Metric: {metric}",
    )

    fig.update_layout(
        font={"family": "Times New Roman"},
        yaxis={
            "visible": True,
            "showticklabels": True,
            "title": {"text": metric.capitalize(), "font": {"size": 20}},
            "tickfont": {"size": 15},
        },
        legend={
            "title": {"text": "Embedding", "font": {"size": 18}},
            "font": {"size": 14},
        },
        xaxis={"title": {"text": " "}},
        xaxis1={"title": {"text": " "}},
        xaxis2={"title": {"text": " "}},
        xaxis3={"title": {"text": " "}},
        title={"x": 0.5},
        margin={"b": 0, "l": 60}
    )

    # Update annotations
    for i, dat in enumerate(datasets):
        fig["layout"]["annotations"][i]["text"] = dat.upper()
        fig["layout"]["annotations"][i]["font"]["size"] = 20

    fig.update_xaxes(showticklabels=False)
    if metric in ["auroc", "spearman", "mcc", "f1"]:
        fig.update_yaxes(range=[0, 1])
    else:
        fig.update_yaxes(matches=None, showticklabels=True)

    if metric == "spearman" and unsupervised and not ablation:
        df_unsupervised = pd.read_csv(
            "results/unsupervised_correlations.csv", index_col=0
        )
        for i in range(3):
            elbo = df_unsupervised.loc[
                df_unsupervised["dataset"] == datasets[i], "spearman_elbo"
            ]
            ll = df_unsupervised.loc[
                df_unsupervised["dataset"] == datasets[i], "spearman_esm_if1"
            ]
            tranception = df_unsupervised.loc[
                df_unsupervised["dataset"] == datasets[i], "spearman_tranception"
            ]
            fig.add_hline(
                y=elbo.item(), col=i + 1, line_dash="dash", line_width=1, opacity=1
            )
            fig.add_hline(
                y=ll.item(), col=i + 1, line_dash="dot", line_width=1, opacity=1
            )
            fig.add_hline(
                y=tranception.item(),
                col=i + 1,
                line_dash="solid",
                line_width=1,
                opacity=1,
            )

    if save_fig:
        if not ablation:
            path = f"figures/test_results_{metric}_{regressor}"
        else:
            path = f"figures/ablation_test_results_{metric}_{regressor}"
        pio.write_image(fig, f"{path}.{image_format}", format=image_format)
        print(f"Saved figure in {path}.{image_format}.")
    else:
        fig.show()


def all_metrics_test_results_barplot(
    regressor: str,
    dataset: str,
    regression: bool,
    embedding_types: Tuple[str, ...],
    path: Union[str, None] = None,
    image_format: str = "pdf",
):
    """Function to show all metrics for specific dataset/predictor combination.

    Args:
        regressor: Name of regressor/classifier
        dataset: Dataset name
        regression: True for regression, False for classification
        embedding_types: Which input representations to include
        path: Overwrite default output path
        image_format: Output file format


    """

    if regression:
        metrics = ["spearman", "rmse", "mae"]
    else:
        metrics = ["mcc", "auroc", "f1"]

    df = pd.DataFrame()
    for metric in metrics:
        df_i = summarise_results(
            dataset=dataset,
            metric=metric,
            embedding_types=embedding_types,
            all=False,
            ablation=False,
        )
        df_i = df_i.loc[(regressor)]
        df_i["metric"] = metric
        df_i = df_i.rename(columns={(f"test_{metric}"): "test_metric"})
        df = pd.concat((df, df_i))

    # Alter multiindex in rows/cols
    df = df.reset_index()
    df.columns = list(map("_".join, df.columns.values))
    df = df.rename(
        columns={"model_": "model", "embedding_": "embedding", "metric_": "metric"}
    )

    # Rename embedding names
    df["embedding"] = df["embedding"].replace(repr_dict())

    fig = px.bar(
        data_frame=df,
        x="embedding",
        color="embedding",
        y=f"test_metric_mean",
        error_y=f"test_metric_sem",
        facet_col="metric",
        width=900,
        height=400,
        color_discrete_sequence=px.colors.qualitative.Plotly,
        category_orders={"metric": metrics},
        title=f"Dataset: {dataset}. Regressor: {regressor}",
        facet_col_spacing=0.05 if regression else 0.03,
    )

    fig["layout"]["xaxis"]["title"]["text"] = ""
    fig["layout"]["xaxis1"]["title"]["text"] = ""
    fig["layout"]["xaxis2"]["title"]["text"] = ""
    fig["layout"]["xaxis3"]["title"]["text"] = ""
    fig.update_xaxes(showticklabels=False)
    if not regression:
        fig.update_yaxes(range=[0, 1])
    else:
        fig.update_yaxes(row=1, col=1, range=[0, 1])
        fig.update_yaxes(matches=None, showticklabels=True)

    fig.update_layout(title={"x": 0.5})
    if path is None:
        path = f"figures/{dataset}/test_results_all_metrics_{regressor}"
    else:
        path = f"figures/{dataset}/{path}"
    pio.write_image(fig, f"{path}.{image_format}", format=image_format)


def generate_target_histograms(image_format: str = "pdf"):
    """Function to visualize target distributions.
    Args:
        image_format: PDF for report, PNG for quick inspection.

    """
    # Combine datasets
    datasets = ("gh114", "cm", "ppat")
    for dataset in datasets:
        # Load dataframe and perform manual binning
        df_d = pd.read_csv(f"data/processed/{dataset}/{dataset}.csv", index_col=0)
        df_d = df_d[["target_reg"]]
        df_d["dataset"] = dataset
        fig = px.histogram(
            data_frame=df_d, x="target_reg", width=500, height=400, nbins=20
        )
        fig.update_xaxes(matches=None, showticklabels=True)
        fig.update_yaxes(matches=None, showticklabels=True)
        path = f"figures/histograms/target_histogram_{dataset}"
        pio.write_image(fig, f"{path}.{image_format}", format=image_format)
        print(f"Saved figure in {path}.{image_format}.")

    # Special case for CM
    dataset = "cm"
    df_d = pd.read_csv(f"data/interim/{dataset}/{dataset}_all.csv", index_col=0)
    df_d = df_d[["target_reg"]]
    df_d["dataset"] = "cm_all"
    fig = px.histogram(data_frame=df_d, x="target_reg", width=500, height=400, nbins=20)

    fig.update_xaxes(matches=None, showticklabels=True)
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.add_vline(x=0.42, line_dash="dash")
    path = f"figures/histograms/target_histogram_{dataset}_all"
    pio.write_image(fig, f"{path}.{image_format}", format=image_format)
    print(f"Saved figure in {path}.{image_format}.")


def generate_partition_histograms(image_format: str = "pdf"):
    """Function to visualize target distributions per CV partition.
    Args:
        image_format: PDF for report, PNG for quick inspection.

    """
    # Combine datasets
    datasets = ("gh114", "cm", "ppat")
    for dataset in datasets:
        # Load dataframe and perform manual binning
        df_d = pd.read_csv(f"data/processed/{dataset}/{dataset}.csv", index_col=0)
        df_d["dataset"] = dataset

        df_d["partition"] = ""
        df_d.loc[df_d["part_0"] == 1, "partition"] = "part_0"
        df_d.loc[df_d["part_1"] == 1, "partition"] = "part_1"
        df_d.loc[df_d["part_2"] == 1, "partition"] = "part_2"

        fig = px.histogram(
            data_frame=df_d,
            x="target_reg",
            color="partition",
            width=500,
            height=400,
            nbins=20,
        )

        fig.update_xaxes(matches=None, showticklabels=True)
        fig.update_yaxes(matches=None, showticklabels=True)
        path = f"figures/histograms/partition_histogram_{dataset}"
        pio.write_image(fig, f"{path}.{image_format}", format=image_format)
        print(f"Saved figure in {path}.{image_format}.")


def show_results_all_predictors(
    datasets: Tuple[str, ...],
    embedding_types: Tuple[str, ...],
    metric: str,
    ablation: bool,
    ablation_method: str,
    save_fig: bool = True,
    image_format: str = "png",
    **kwargs,
):
    """Visualize all ablation results (either random or holdout) for all datasets using four regressors.

    Args:
        datasets: Name of datasets
        embedding_types: Which representations to include
        metric: Which metric to include
        save_fig: Include to save.
        ablation: True/False
        ablation_method: Specify ablation method.
        image_format: Output file format. PDF for report.

    """
    df = pd.DataFrame()
    for dataset in datasets:
        df_i = summarise_results(
            dataset=dataset,
            metric=metric,
            embedding_types=embedding_types,
            all=False,
            ablation=ablation,
            unsupervised=False,
            ablation_method=ablation_method,
        )
        df_i["dataset"] = dataset
        df = pd.concat((df, df_i))

    # Alter multiindex in rows/cols
    df = df.reset_index()
    df.columns = list(map("_".join, df.columns.values))
    df = df.rename(
        columns={"model_": "model", "embedding_": "embedding", "dataset_": "dataset"}
    )

    # Rename embedding names
    df["embedding"] = df["embedding"].replace(repr_dict())
    if ablation_method == "holdout":
        title = f"Results with hold-out validation (ablation). Metric: {metric.capitalize()}."
    elif ablation_method == "random":
        title = f"Results with repeated random splitting (ablation). Metric: {metric.capitalize()}."
    else:
        title = f"Results for using all predictors. Metric: {metric.capitalize()}."

    fig = px.bar(
        data_frame=df,
        x="embedding",
        color="embedding",
        y=f"test_{metric}_mean",
        error_y=f"test_{metric}_sem",
        facet_col="model",
        facet_row="dataset",
        width=900,
        height=700,
        color_discrete_sequence=px.colors.qualitative.Plotly,
        category_orders={
            "model": ["KNN", "Ridge", "RandomForest", "MLP"]
            if metric in ["rmse", "spearman", "mae"]
            else ["KNN", "LogReg", "RandomForest", "MLP"],
            "embedding": REPRESENTATION_ORDER,
        },
        title=title,
    )

    fig.update_layout(
        font={"family": "Times New Roman"},
        legend={
            "title": {"text": "Embedding", "font": {"size": 18}},
            "font": {"size": 14},
        },
        title={"x": 0.5},
        margin={"b": 0, "l": 60}
    )

    # Remove x-labels (non-informative)
    fig["layout"]["xaxis"]["title"]["text"] = ""
    fig["layout"]["xaxis2"]["title"]["text"] = ""
    fig["layout"]["xaxis3"]["title"]["text"] = ""
    fig["layout"]["xaxis4"]["title"]["text"] = ""
    # Dataset as y-labels
    fig["layout"]["yaxis"]["title"]["text"] = datasets[2].upper()
    fig["layout"]["yaxis5"]["title"]["text"] = datasets[1].upper()
    fig["layout"]["yaxis9"]["title"]["text"] = datasets[0].upper()

    # Update annotations
    for i, reg in enumerate(["KNN", "Ridge", "RandomForest", "MLP"]):
        fig["layout"]["annotations"][i]["text"] = reg
    for i, dat in enumerate(datasets, 1):
        fig["layout"]["annotations"][-i]["text"] = ""

    # Capitalize legend header
    fig["layout"]["legend"]["title"]["text"] = "Embedding"

    fig.update_xaxes(showticklabels=False)
    if metric in ["auroc", "spearman", "mcc", "f1"]:
        fig.update_yaxes(range=[0, 1])
    else:
        fig.update_yaxes(matches=None, showticklabels=True)

    if save_fig:
        if ablation:
            path = f"figures/all_{ablation_method}_{metric}_results"
        else:
            path = f"figures/all_results_{metric}"
        pio.write_image(fig, f"{path}.{image_format}", format=image_format)
        print(f"Saved figure to {path}.{image_format}.")
    else:
        fig.show()


if __name__ == "__main__":
    datasets: Tuple[str, ...] = ("gh114", "cm", "ppat")
    embedding_types: Tuple[str, ...] = (
        "ct",
        "af2",
        "esm_1b",
        "esm_2",
        "esm_if1",
        "eve",
        "onehot",
        "mif",
        "mifst",
    )
    metric: str = "spearman"
    ablation: bool = False
    if ablation:
        # ablation_method: str = "holdout"
        ablation_method: str = "random"
    else:
        ablation_method = None
    save_fig: bool = True
    image_format: str = "pdf"
    show_results_all_predictors(
        datasets,
        embedding_types,
        metric,
        ablation,
        ablation_method,
        save_fig,
        image_format,
    )
