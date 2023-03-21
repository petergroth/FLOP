import argparse
import os

import esm
import esm.inverse_folding
import torch
from tqdm import tqdm


def generate_esm_if1_embedding(dataset: str, trim_length: int = 4):
    """Function to compute ESM-IF1 embeddings given dataset and structures.

    Args:
        dataset: Name of dataset
        trim_length: Removes file-type from paths

    Returns:

    """
    # Define paths
    pretrained_path = "models/esm_if1_gvp4_t16_142M_UR50.pt"
    output_dir = f"representations/{dataset}/esm_if1"
    pdb_dir = f"data/raw/{dataset}/pdb"

    # Extract sequence names
    pdb_paths = os.listdir(pdb_dir)

    # Load model
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(pretrained_path)
    print("Successfully loaded model.")
    model.eval()

    with torch.no_grad():
        for path in tqdm(pdb_paths):
            # Load structure, extract coordinates
            structure = esm.inverse_folding.util.load_structure(f"{pdb_dir}/{path}", chain="A")
            coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)

            # Embed and save
            rep = esm.inverse_folding.util.get_encoder_output(model, alphabet, coords)
            torch.save(rep.cpu().detach(), f"{output_dir}/{path[:(len(path)-trim_length)]}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--trim_length", type=int, default=4, help="A value of 4 removes <.pdb>")
    args = parser.parse_args()
    generate_esm_if1_embedding(**vars(args))
