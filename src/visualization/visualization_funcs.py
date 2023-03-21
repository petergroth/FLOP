from typing import Tuple, Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import umap
from sklearn.decomposition import PCA

from src.data.data_utils import repr_dict
from src.training.training_utils import extract_all_embeddings, summarise_results


def show_target_histogram(df: pd.DataFrame, dataset: str, target: Union[str, List[str]] = "target"):
    """Function to visualize target distribution of one dataset during its compilation

    Args:
        df: Dataset
        dataset: Dataset string name
        target: Target column

    """
    sns.set_style("dark")
    task = "regression" if target == "target_reg" else "classification"
    if isinstance(target, str):
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.histplot(
            data=df[df[["part_0", "part_1", "part_2"]].sum(axis=1) == 1],
            x=target,
            color="#F58A00",
            bins=2 if target == "target_class" else 10,
        )
        ax.set_title(f"Histogram over {task} target values for {dataset.upper()} dataset.")
        path = f"figures/{dataset}/target_{task}_histogram.png"
        plt.savefig(f"{path}")
        print(f"Saved plot in {path}")
        plt.show()
    elif isinstance(target, list):
        n = len(target)
        fig, ax = plt.subplots(n // 2, 2, figsize=(10, 5 * (n // 2)))
        for i, tar in enumerate(target):
            axi = ax.flatten()[i]
            sns.histplot(data=df, x=tar, color="#F58A00", ax=axi, bins=2 if target == "target_class" else 10)
            axi.set_title(f"Target = {tar}")
        plt.suptitle(f"Histogram over {task} targets for {dataset.upper()} dataset.")
        path = f"figures/{dataset}/targets_{task}_histogram.png"
        plt.savefig(f"{path}")
        print(f"Saved plot in {path}")
        plt.show()


def show_CV_split_distribution(df: pd.DataFrame, threshold: Union[float, None], dataset: str, n_partitions: int):
    # Setup plotting
    sns.set_style("dark")
    color_palette = sns.color_palette("colorblind")
    y_limits = (df["target_reg"].min(), df["target_reg"].max())
    n_eff = int(df[[f"part_{i}" for i in range(n_partitions)]].sum().sum())

    fig, ax = plt.subplots(n_partitions, 2, figsize=(10, 5 * n_partitions), sharex="col")

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
        sns.countplot(data=df[df[f"part_{i}"] == 1], x="target_class", ax=axi, color=color_palette[i], alpha=0.9)
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

    plt.savefig(f"figures/{dataset}/splits/{dataset}_CV_K={n_partitions}_distribution.pdf")
    plt.savefig(f"figures/{dataset}/splits/{dataset}_CV_K={n_partitions}_distribution.png")
    plt.show()


def visualize_embedding(dataset: str, embedding_type: str, suffix: Union[str, None] = None):
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
        df=df, dataset=dataset, embedding_type=embedding_type, target="target", suffix=suffix
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

    sns.scatterplot(data=df_plot, x="x_pca", y="y_pca", hue="split", ax=ax[0, 0], s=s, style="split", alpha=0.5)
    sns.scatterplot(data=df_plot, x="x_pca", y="y_pca", hue="target", ax=ax[1, 0], s=s, alpha=0.5)
    sns.scatterplot(data=df_plot, x="x_umap", y="y_umap", hue="split", ax=ax[0, 1], s=s, style="split", alpha=0.5)
    sns.scatterplot(data=df_plot, x="x_umap", y="y_umap", hue="target", ax=ax[1, 1], s=s, alpha=0.5)

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


def test_results_barplot(
    dataset: str,
    embedding_types: Tuple[str, ...],
    metric: str,
    save_fig: bool = True,
    path: Union[str, None] = None,
    image_format: str = "png",
    ablation: bool = False,
    ablation_method: str = None,
):
    """Visualize results for KNN, linear model, and random forest regressors/classifiers for one dataset.

    Args:
        dataset: Name of dataset
        embedding_types: Which representations to include
        metric: Which metric to include
        save_fig: Include to save.
        path: Include to overwrite default output path.
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
        height=400,
        color_discrete_sequence=px.colors.qualitative.G10,
        category_orders={
            "model": ["KNN", "Ridge", "RandomForest", "MLP"]
            if metric in ["rmse", "spearman", "mae"]
            else ["KNN", "LogReg", "RandomForest", "MLP"],
            "embedding": ["CT", "ESM-1B", "ESM-2", "ESM-IF1", "EVE", "Evoformer (AF2)", "MSA (1-HOT)"],
        },
        title=f"Dataset: {dataset.upper()}. Metric: {metric}",
    )
    fig.update_layout({"yaxis": {"title": f"{metric.capitalize()}", "visible": True, "showticklabels": True}})
    fig["layout"]["xaxis"]["title"]["text"] = ""
    fig["layout"]["xaxis2"]["title"]["text"] = ""
    fig["layout"]["xaxis3"]["title"]["text"] = ""
    fig.update_xaxes(showticklabels=False)
    if metric in ["auroc", "spearman", "mcc", "f1"]:
        fig.update_yaxes(range=[0, 1])

    fig.update_layout(title={"x": 0.5})

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
    df = df.rename(columns={"model_": "model", "embedding_": "embedding", "dataset_": "dataset"})

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
        height=400,
        color_discrete_sequence=px.colors.qualitative.G10,
        category_orders={
            "model": datasets,
            "embedding": ["CT", "ESM-1B", "ESM-2", "ESM-IF1", "EVE", "Evoformer (AF2)", "MSA (1-HOT)"],
        },
        title=f"Metric: {metric}",
    )
    fig.update_layout({"yaxis": {"title": f"{metric.capitalize()}", "visible": True, "showticklabels": True}})
    fig["layout"]["xaxis"]["title"]["text"] = ""
    fig["layout"]["xaxis2"]["title"]["text"] = ""
    fig["layout"]["xaxis3"]["title"]["text"] = ""
    fig.update_xaxes(showticklabels=False)
    if metric in ["auroc", "spearman", "mcc", "f1"]:
        fig.update_yaxes(range=[0, 1])

    fig.update_layout(title={"x": 0.5})

    if metric == "spearman" and unsupervised:
        if ablation:
            df_unsupervised = pd.read_csv("results/unsupervised_ablation_correlations_cv.csv", index_col=0)
        else:
            df_unsupervised = pd.read_csv("results/unsupervised_correlations_cv.csv", index_col=0)
        for i in range(3):
            elbo = df_unsupervised.loc[df_unsupervised["dataset"] == datasets[i], "spearman_elbo"]
            ll = df_unsupervised.loc[df_unsupervised["dataset"] == datasets[i], "spearman_esm_if1"]
            tranception = df_unsupervised.loc[df_unsupervised["dataset"] == datasets[i], "spearman_tranception"]
            fig.add_hline(
                y=elbo.item(), col=i + 1, line_dash="dash", line_width=1, opacity=1
            )  # line=dict(color=px.colors.qualitative.G10[5]))
            fig.add_hline(
                y=ll.item(), col=i + 1, line_dash="dot", line_width=1, opacity=1
            )  # line=dict(color=px.colors.qualitative.G10[4]))
            fig.add_hline(
                y=tranception.item(), col=i + 1, line_dash="solid", line_width=1, opacity=1
            )  # line=dict(color=px.colors.qualitative.G10[4]))

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
            dataset=dataset, metric=metric, embedding_types=embedding_types, all=False, ablation=False
        )
        df_i = df_i.loc[(regressor)]
        df_i["metric"] = metric
        df_i = df_i.rename(columns={(f"test_{metric}"): "test_metric"})
        df = pd.concat((df, df_i))

    # Alter multiindex in rows/cols
    df = df.reset_index()
    df.columns = list(map("_".join, df.columns.values))
    df = df.rename(columns={"model_": "model", "embedding_": "embedding", "metric_": "metric"})

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
        color_discrete_sequence=px.colors.qualitative.G10,
        category_orders={"metric": metrics},
        title=f"Dataset: {dataset}. Regressor: {regressor}",
        facet_col_spacing=0.05 if regression else 0.03,
    )

    fig["layout"]["xaxis"]["title"]["text"] = ""
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
        fig = px.histogram(data_frame=df_d, x="target_reg", width=500, height=400, nbins=20)
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

        fig = px.histogram(data_frame=df_d, x="target_reg", color="partition", width=500, height=400, nbins=20)

        fig.update_xaxes(matches=None, showticklabels=True)
        fig.update_yaxes(matches=None, showticklabels=True)
        path = f"figures/histograms/partition_histogram_{dataset}"
        pio.write_image(fig, f"{path}.{image_format}", format=image_format)
        print(f"Saved figure in {path}.{image_format}.")
