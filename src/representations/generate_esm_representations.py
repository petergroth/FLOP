"""Adaption of https://github.com/facebookresearch/esm/blob/main/scripts/extract.py"""
import argparse

import esm
import torch
from esm import FastaBatchedDataset
from tqdm import tqdm


def generate_esm_embeddings(dataset: str, model_name: str, include):
    """Function to compute ESM-1B or ESM-2 embeddings for dataset

    Args:
        dataset: Name of dataset
        model_name: either esm_1b or esm_2
        include: Which representation to include. Either: per_tok, mean, or bos.

    """
    # Define paths

    output_dir = f"representations/{dataset}/{model_name}"
    if model_name == "esm_1b":
        pretrained_path = "models/esm1b_t33_650M_UR50S.pt"
    elif model_name == "esm_2":
        pretrained_path = "models/esm2_t36_3B_UR50D.pt"
    elif model_name == "esm_if1":
        print("ERROR. Use generate_esm_if1_representations.py instead")
        raise ValueError
    else:
        raise ValueError

    # Load model
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(pretrained_path)
    print("Successfully loaded model.")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model to GPU")

    # Final representation is of interest
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in [-1]]

    # Dataset specific paths
    fasta_path = f"data/raw/{dataset}/{dataset}.fasta"

    # Load dataset
    dataset = FastaBatchedDataset.from_file(fasta_path)
    batches = dataset.get_batch_indices(1024, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches)
    print(f"Read {fasta_path} with {len(dataset)} sequences")

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in tqdm(enumerate(data_loader)):
            # Move to GPU
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)
            # Generate and extract representations over batch
            out = model(toks, repr_layers=repr_layers, return_contacts=False)
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}
            # Iterate through sequence and save results
            for i, label in enumerate(labels):
                output_path = f"{output_dir}/{label}.pt"
                # Create dictionary of representations
                result = {"label": label}
                if "per_tok" in include:
                    result["representations"] = {
                        layer: t[i, 1 : len(strs[i]) + 1].clone() for layer, t in representations.items()
                    }
                if "mean" in include:
                    result["mean_representations"] = {
                        layer: t[i, 1 : len(strs[i]) + 1].mean(0).clone() for layer, t in representations.items()
                    }
                if "bos" in include and model_name == "esm2":
                    result["bos_representations"] = {layer: t[i, 0].clone() for layer, t in representations.items()}
                # Save file
                torch.save(result, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Name of dataset.")
    parser.add_argument("model_name", type=str, help="Which ESM model to use. Either esm_1b or esm_2.")
    parser.add_argument(
        "--include", type=str, help="Either 'mean', 'per_tok', or 'bos'.", nargs="+", default=["mean"]
    )
    args = parser.parse_args()
    generate_esm_embeddings(**vars(args))
