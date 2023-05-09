"""Visualize phylogenetic tree for dataset"""
import argparse
import os
from collections import defaultdict
import random

import pandas as pd
from ete3 import Tree, TreeStyle, NodeStyle, RectFace

os.environ["QT_QPA_PLATFORM"] = "offscreen"


def draw_phylogenetic_tree(dataset: str, how: str):
    """Draw phylogenetic tree for dataset

    Args:
        dataset: Name of dataset
        how: How to color the tree. Either "flop", "mmseqs" or "random"
    """
    # Specify dataset and path to tree-file
    tree_path = f"data/interim/{dataset}/{dataset}.tree"

    if how == "flop":
        pdf_file = f"figures/phylo_tree/{dataset}_local_tree.pdf"
    elif how == "mmseqs":
        pdf_file = f"figures/phylo_tree/{dataset}_local_tree_mmseqs.pdf"
    elif how == "random":
        pdf_file = f"figures/phylo_tree/{dataset}_local_tree_random.pdf"
    else:
        raise ValueError

    # Define colors
    c1, c2, c3 = "#9B5DE5", "#FCA17D", "#FEE440"

    # Dummy function
    def def_value():
        return "white"

    # Assign colors
    colors = defaultdict(def_value)
    target = defaultdict(def_value)
    df = pd.read_csv(f"data/processed/{dataset}/{dataset}.csv", index_col=0)

    if how == "flop":
        for part, c in zip(["part_0", "part_1", "part_2"], [c1, c2, c3]):
            for name in df[df[part] == 1]["name"]:
                colors[name] = c
                color = "black" if df.loc[df["name"] == name, "target_class"].item() == 1 else "white"
                target[name] = color

    elif how == "mmseqs":
        pdf_file = f"figures/{dataset}_local_tree_mmseqs.pdf"
        df_mmseqs = pd.read_csv(f"data/interim/{dataset}/{dataset}_clust_cluster.tsv", sep="\t", header=None)

        largest_two_clusters = df_mmseqs.groupby(0).count().sort_values(1, ascending=False).nlargest(2, columns=1)
        seq1 = largest_two_clusters.index[0]
        seq2 = largest_two_clusters.index[1]

        cluster1 = df_mmseqs[df_mmseqs[0] == seq1]
        cluster2 = df_mmseqs[df_mmseqs[0] == seq2]
        cluster3 = df_mmseqs[(df_mmseqs[0] != seq1) & (df_mmseqs[0] != seq2)]

        for ID in df_mmseqs[1].tolist():
            if ID in cluster1[1].tolist():
                colors[ID] = c1
            elif ID in cluster2[1].tolist():
                colors[ID] = c2
            elif ID in cluster3[1].tolist():
                colors[ID] = c3
            if ID in df["name"].tolist():
                color = "black" if df.loc[df["name"] == ID, "target_class"].item() == 1 else "white"
                target[ID] = color

    elif how == "random":
        for name in df["name"].tolist():
            colors[name] = random.choice([c1, c2, c3])
            color = "black" if df.loc[df["name"] == name, "target_class"].item() == 1 else "white"
            target[name] = color

    # Generate tree viz
    t = Tree(tree_path)
    t.ladderize()
    for node in t.traverse():
        nstyle = NodeStyle()
        if node.name in colors:
            # For sparse, comment out the following
            nstyle["bgcolor"] = colors[node.name]
            nstyle["fgcolor"] = colors[node.name]
            node.set_style(nstyle)
        nstyle["size"] = 0
        nstyle["vt_line_width"] = 2
        nstyle["hz_line_width"] = 2
        node.set_style(nstyle)
        N = RectFace(bgcolor=colors[node.name], fgcolor=colors[node.name], width=60, height=24)
        node.add_face(N, 0, position="aligned")
        N = RectFace(bgcolor=target[node.name], fgcolor=target[node.name], width=20, height=24)
        node.add_face(N, 1, position="aligned")
        node.name = " "

    ts = TreeStyle()
    ts.mode = "c"
    ts.root_opening_factor = 0.45
    ts.show_branch_support = False
    ts.show_scale = False
    t.convert_to_ultrametric()
    t.render(pdf_file, tree_style=ts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str)
    args = parser.parse_args()
    for how in ["flop", "mmseqs", "random"]:
        draw_phylogenetic_tree(how=how, **vars(args))
