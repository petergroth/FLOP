"""Script to compute correlations between unsupervised fitness proxies and target values."""
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, sem


def compute_unsupervised_correlation(dataset: str) -> Tuple:
    """Function to compute correlations between unsupervised fitness proxies and target values.

    Args:
        dataset:  One of `gh114`, `cm`, and `ppat`

    Returns: Tuple of mean value of EVE correlations and the LL correlation from ESM-IF1.

    """

    # ELBOs
    df = pd.read_csv(f"data/processed/{dataset}/{dataset}.csv", index_col=0)
    target = "target_reg"
    corrs = np.zeros(3)
    vals = [0, 1, 2]
    # Load dataset
    for i in vals:
        df_elbo = pd.read_csv(f"representations/{dataset}/{dataset}_EVE_ELBO_{i}.csv", index_col=0)
        df_elbo = df_elbo.rename(columns={"ELBO": f"ELBO_{i}"})
        df = pd.merge(left=df, right=df_elbo, on="name", how="left")
        df = df.dropna()
        r, _ = spearmanr(df[target].values, df[f"ELBO_{i}"].values)
        corrs[i] = r

    mu = np.mean(corrs)
    df_ll = pd.read_csv(f"representations/{dataset}/esm_if1_likelihoods.csv", index_col=0)
    df = pd.merge(left=df, right=df_ll, on="name", how="left")
    ll, _ = spearmanr(df[target].values, df["ll"].values)

    df_t = pd.read_csv(f"representations/{dataset}/{dataset}_tranception_scores.csv", index_col=0)
    df = pd.merge(left=df, right=df_t[["name", "avg_score"]], on="name", how="left")
    df = df.dropna()
    ll_tranception, _ = spearmanr(df[target].values, df["avg_score"].values)

    return mu, ll, ll_tranception


def compute_unsupervised_correlation_cv(dataset: str) -> Tuple:
    """Function to compute correlations between unsupervised fitness proxies and target values for each
    partition.

    Args:
        dataset:  One of `gh114`, `cm`, and `ppat`

    Returns: Tuple of mean value of EVE correlations and the LL correlation from ESM-IF1.

    """
    df = pd.read_csv(f"data/processed/{dataset}/{dataset}.csv", index_col=0)
    target = "target_reg"
    corrs = np.zeros((3, 3))
    vals = [0, 1, 2]

    # Compute EVE correlations
    for i in vals:
        df_elbo = pd.read_csv(f"representations/{dataset}/{dataset}_EVE_ELBO_{i}.csv", index_col=0)
        df_elbo = df_elbo.rename(columns={"ELBO": f"ELBO_{i}"})
        df = pd.merge(left=df, right=df_elbo, on="name", how="left")
        df = df.dropna()
        # For eqch partition
        for j in range(3):
            df_j = df[df[f"part_{j}"].astype(bool)]
            r, _ = spearmanr(df_j[target].values, df_j[f"ELBO_{i}"].values)
            corrs[i, j] = r

    mu = np.mean(corrs, axis=0)

    # Compute likelihoods
    df_ll = pd.read_csv(f"representations/{dataset}/esm_if1_likelihoods.csv", index_col=0)
    df = pd.merge(left=df, right=df_ll, on="name", how="left")
    lls = np.zeros(3)
    for j in range(3):
        df_j = df[df[f"part_{j}"].astype(bool)]
        r, _ = spearmanr(df_j[target].values, df_j["ll"].values)
        lls[j] = r

    # Compute likelihoods
    df_t = pd.read_csv(f"representations/{dataset}/{dataset}_tranception_scores.csv", index_col=0)
    df = pd.merge(left=df, right=df_t[["name", "avg_score"]], on="name", how="left")
    df = df.dropna()
    lls_t = np.zeros(3)
    for j in range(3):
        df_j = df[df[f"part_{j}"].astype(bool)]
        r, _ = spearmanr(df_j[target].values, df_j["avg_score"].values)
        lls_t[j] = r

    return mu, lls, lls_t


def compute_unsupervised_ablation_correlation_cv(dataset: str) -> Tuple:
    """Function to compute correlations between unsupervised fitness proxies and target values for ablation study.

    Args:
        dataset:  One of `gh114`, `cm`, and `ppat`

    Returns: Tuple of mean value of EVE correlations and the LL correlation from ESM-IF1.

    """
    if dataset == "cm":
        df = pd.read_csv(f"representations/{dataset}/{dataset}_all.csv", index_col=0)
        target = "target_reg"
        corrs = np.zeros((3, 3))
        vals = [0, 1, 2]

        # Compute EVE correlations
        for i in vals:
            df_elbo = pd.read_csv(f"representations/{dataset}/{dataset}_EVE_ELBO_{i}.csv", index_col=0)
            df_elbo = df_elbo.rename(columns={"ELBO": f"ELBO_{i}"})
            df = pd.merge(left=df, right=df_elbo, on="name", how="left")
            df = df.dropna()
            # For eqch partition
            for j in range(3):
                df_j = df[df[f"part_{j}"].astype(bool)]
                r, _ = spearmanr(df_j[target].values, df_j[f"ELBO_{i}"].values)
                corrs[i, j] = r

        mu = np.mean(corrs, axis=0)

        # Compute likelihoods
        df_ll = pd.read_csv(f"representations/{dataset}/esm_if1_likelihoods.csv", index_col=0)
        df = pd.merge(left=df, right=df_ll, on="name", how="left")
        lls = np.zeros(3)
        for j in range(3):
            df_j = df[df[f"part_{j}"].astype(bool)]
            r, _ = spearmanr(df_j[target].values, df_j["ll"].values)
            lls[j] = r

        # Compute likelihoods
        df_t = pd.read_csv(f"representations/{dataset}/{dataset}_tranception_scores.csv", index_col=0)
        df = pd.merge(left=df, right=df_t[["name", "avg_score"]], on="name", how="left")
        df = df.dropna()
        lls_t = np.zeros(3)
        for j in range(3):
            df_j = df[df[f"part_{j}"].astype(bool)]
            r, _ = spearmanr(df_j[target].values, df_j["avg_score"].values)
            lls_t[j] = r

    elif dataset == "gh114":
        df = pd.read_csv(f"data/processed/{dataset}/{dataset}.csv", index_col=0)
        target = "target_reg"
        corrs = np.zeros((3, 1))
        vals = [0, 1, 2]
        j = 0
        # Compute EVE correlations
        for i in vals:
            df_elbo = pd.read_csv(f"representations/{dataset}/{dataset}_EVE_ELBO_{i}.csv", index_col=0)
            df_elbo = df_elbo.rename(columns={"ELBO": f"ELBO_{i}"})
            df = pd.merge(left=df, right=df_elbo, on="name", how="left")
            df = df.dropna()

            df_j = df[df[f"part_{j}"].astype(bool)]
            r, _ = spearmanr(df_j[target].values, df_j[f"ELBO_{i}"].values)
            corrs[i, j] = r

        mu = np.mean(corrs, axis=0)

        # Compute likelihoods
        df_ll = pd.read_csv(f"representations/{dataset}/esm_if1_likelihoods.csv", index_col=0)
        df_t = pd.read_csv(f"representations/{dataset}/{dataset}_tranception_scores.csv", index_col=0)
        df = pd.merge(left=df, right=df_ll, on="name", how="left")
        df = pd.merge(left=df, right=df_t[["name", "avg_score"]], on="name", how="left")
        df = df.dropna()
        lls = np.zeros(1)
        lls_t = np.zeros(1)

        df_j = df[df[f"part_{j}"].astype(bool)]
        r, _ = spearmanr(df_j[target].values, df_j["ll"].values)
        lls[j] = r
        r, _ = spearmanr(df_j[target].values, df_j["avg_score"].values)
        lls_t[j] = r

    elif dataset == "ppat":
        df = pd.read_csv(f"data/processed/{dataset}/{dataset}.csv", index_col=0)
        target = "target_reg"
        corrs = np.zeros((3, 3, 3))
        lls = np.zeros((3, 3))
        lls_t = np.zeros((3, 3))
        vals = [0, 1, 2]
        seeds = [0, 1, 2]

        # Determine fixed split sizes
        n_obs = len(df)
        n_train = int(n_obs * 0.5)
        n_val = int(n_obs * 0.25)

        # Iterate through seeds
        for seed in seeds:
            np.random.seed(seed)
            # Create permutation
            perm = np.random.permutation(n_obs)
            train_idx = perm[:n_train]
            val_idx = perm[n_train : (n_train + n_val)]
            test_idx = perm[(n_train + n_val) :]

            # Shuffle partitions randomly
            df = pd.read_csv(f"data/processed/{dataset}/{dataset}.csv", index_col=0)
            df[["part_0", "part_1", "part_2"]] = 0
            df.loc[train_idx, "part_0"] = 1
            df.loc[val_idx, "part_1"] = 1
            df.loc[test_idx, "part_2"] = 1

            # Compute EVE correlations
            for i in vals:
                df_elbo = pd.read_csv(f"representations/{dataset}/{dataset}_EVE_ELBO_{i}.csv", index_col=0)
                df_elbo = df_elbo.rename(columns={"ELBO": f"ELBO_{i}"})
                df = pd.merge(left=df, right=df_elbo, on="name", how="left")
                df = df.dropna()
                # For each partition
                for j in range(3):
                    df_j = df[df[f"part_{j}"].astype(bool)]
                    r, _ = spearmanr(df_j[target].values, df_j[f"ELBO_{i}"].values)
                    corrs[seed, i, j] = r

            # Compute likelihoods
            df_ll = pd.read_csv(f"representations/{dataset}/esm_if1_likelihoods.csv", index_col=0)
            df = pd.merge(left=df, right=df_ll, on="name", how="left")

            for j in range(3):
                df_j = df[df[f"part_{j}"].astype(bool)]
                r, _ = spearmanr(df_j[target].values, df_j["ll"].values)
                lls[seed, j] = r

            # Compute likelihoods
            df_t = pd.read_csv(f"representations/{dataset}/{dataset}_tranception_scores.csv", index_col=0)
            df = pd.merge(left=df, right=df_t[["name", "avg_score"]], on="name", how="left")
            df = df.dropna()
            for j in range(3):
                df_j = df[df[f"part_{j}"].astype(bool)]
                r, _ = spearmanr(df_j[target].values, df_j["avg_score"].values)
                lls_t[seed, j] = r

        mu = np.mean(corrs, axis=0)
        lls = np.mean(lls, axis=0)
        lls_t = np.mean(lls_t, axis=0)

    return mu, lls, lls_t


def main(args):
    """Main function."""
    if args.all_to_csv:
        datasets = []
        elbo_r = []
        ll_r = []
        tranception_r = []
        if not args.ablation:
            for dataset in ["cm", "gh114", "ppat"]:
                mu, r, trn = compute_unsupervised_correlation(dataset=dataset)
                datasets.append(dataset)
                elbo_r.append(mu)
                ll_r.append(r)
                tranception_r.append(trn)

            df = pd.DataFrame(
                {
                    "dataset": datasets,
                    "spearman_elbo": elbo_r,
                    "spearman_esm_if1": ll_r,
                    "spearman_tranception": tranception_r,
                }
            )
            df.to_csv("results/unsupervised_correlations.csv")

        # Compute unsupervised correlations on CV partitions
        datasets, elbo_mu, elbo_sem, ll_mu, ll_sem, trn_mu, trn_sem = [], [], [], [], [], [], []
        for dataset in ["cm", "gh114", "ppat"]:
            if not args.ablation:
                mu, lls, lls_t = compute_unsupervised_correlation_cv(dataset=dataset)
            else:
                mu, lls, lls_t = compute_unsupervised_ablation_correlation_cv(dataset=dataset)
            datasets.append(dataset)
            elbo_mu.append(np.mean(mu))
            elbo_sem.append(np.mean(sem(mu)))
            ll_mu.append(np.mean(lls))
            ll_sem.append(sem(lls))
            trn_mu.append(np.mean(lls_t))
            trn_sem.append(sem(lls_t))

        df = pd.DataFrame(
            {
                "dataset": datasets,
                "spearman_elbo": elbo_mu,
                "spearman_elbo_sem": elbo_sem,
                "spearman_esm_if1": ll_mu,
                "spearman_esm_if1_sem": ll_sem,
                "spearman_tranception": trn_mu,
                "spearman_tranception_sem": trn_sem,
            }
        )
        if args.ablation:
            df.to_csv("results/unsupervised_ablation_correlations_cv.csv")
        else:
            df.to_csv("results/unsupervised_correlations_cv.csv")

    else:
        mu, r = compute_unsupervised_correlation_cv(args.dataset)
        print(f"Dataset: {args.dataset}\n" f"ELBOs: {mu}\n" f"LLs: {r}")

        mu_mean = np.mean(mu)
        mu_sig = np.std(mu)
        r_mean = np.mean(r)
        r_sig = np.std(r)

        print(
            f"Dataset: {args.dataset}\n"
            f"ELBOs: mean = {mu_mean}, std = {mu_sig}\n"
            f"LLs: mean = {r_mean}, std = {r_sig}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_to_csv", action="store_true", default=False)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--ablation", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
