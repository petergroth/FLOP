"""Adapted from https://github.com/microsoft/protein-sequence-models/blob/main/scripts/extract_mif.py"""

import argparse
from pathlib import Path

import pandas as pd
import torch.cuda
from sequence_models.constants import PROTEIN_ALPHABET
from sequence_models.collaters import StructureCollater, SimpleCollater
from sequence_models.pdb_utils import parse_PDB, process_coords
from sequence_models.pretrained import load_carp, load_gnn, MIF
from tqdm import tqdm


def load_model_and_alphabet_from_local(model_name: str, carp_name: str):
    model_data = torch.load(model_name, map_location="cpu")
    collater = SimpleCollater(PROTEIN_ALPHABET, pad=True)
    gnn = load_gnn(model_data)
    cnn = None
    if model_data["model"] == "mif-st":
        cnn_data = torch.load(carp_name, map_location="cpu")
        cnn = load_carp(cnn_data)
    collater = StructureCollater(collater, n_connections=30)
    model = MIF(gnn, cnn=cnn)
    return model, collater


def generate_mifst_embeddings(dataset: str, model_name: str):
    """Function to compute MIFST embeddings for dataset

    Args:
        dataset: Name of dataset
        model_name: either mifst or mif

    """
    # Define paths
    output_dir = Path("representations", dataset, model_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir = Path("data", "raw", dataset, "pdb")
    data_path = Path("data", "interim", dataset, f"{dataset}.csv")
    if dataset == "cm":
        data_path = Path("data", "interim", dataset, f"{dataset}_all.csv")
    model_path = Path("models", f"{model_name}.pt")
    carp_path = Path("models", "carp_640M.pt")

    # Load model
    model, collater = load_model_and_alphabet_from_local(str(model_path), str(carp_path))
    print("Successfully loaded model.")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model to GPU")

    # Load data
    df = pd.read_csv(data_path, index_col=0)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        seq = row['sequence']
        name = row['name']
        pdb_path = pdb_dir / f"{name}.pdb"
        coords, wt, _ = parse_PDB(str(pdb_path))
        coords = {
            'N': coords[:, 0],
            'CA': coords[:, 1],
            'C': coords[:, 2]
        }
        dist, omega, theta, phi = process_coords(coords)
        batch = [[seq, torch.tensor(dist, dtype=torch.float),
                  torch.tensor(omega, dtype=torch.float),
                  torch.tensor(theta, dtype=torch.float), torch.tensor(phi, dtype=torch.float)]]
        src, nodes, edges, connections, edge_mask = collater(batch)
        if torch.cuda.is_available():
            src = src.to(device="cuda")
            nodes = nodes.to(device="cuda")
            edges = edges.to(device="cuda")
            connections = connections.to(device="cuda")
            edge_mask = edge_mask.to(device="cuda")
        with torch.no_grad():
            rep = model(src, nodes, edges, connections, edge_mask, result="repr")[0]
            # Save representation
            output_path = output_dir / f"{name}.pt"
            torch.save(rep.mean(dim=0).detach().cpu(), output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='Name of dataset.')
    parser.add_argument('--model_name', type=str, help='Name of pretrained model.', default="mifst")
    args = parser.parse_args()
    generate_mifst_embeddings(**vars(args))
