import argparse

import pandas as pd
from Bio import SeqIO


def generate_query_sequence(dataset: str, verbose: bool = True):
    """Function to parse MSA sequences from CSV file and generate a query sequence for use in EVE
    The resulting string will have the character A where any of the sequences have a non-gap in their MSA. The
    remaining positions will be gap-filled.

    Args:
        dataset: One of `cm`, `ppat` or `gh114`.
        verbose: Log info or not.

    """

    msa_path = f"data/raw/{dataset}/{dataset}_family.aln.fasta"
    csv_path = f"data/processed/{dataset}/{dataset}.csv"
    name_key = "name"

    # Load first query to extract sequence length
    dummy = next(iter(SeqIO.parse(open(msa_path), "fasta")))
    seq_len = len(str(dummy.seq))

    # Load csv to generate queries
    df = pd.read_csv(csv_path)
    names = df[name_key].tolist()

    # Initialise query as all-gaps
    query_sequence = ["-"] * seq_len
    if verbose:
        print("Generating query sequence.")

    # Iterating through MSAs to generate query
    fasta_sequences = SeqIO.parse(open(msa_path), "fasta")
    for seq in fasta_sequences:
        name, sequence = seq.id, str(seq.seq)
        if name in names:
            for i, char in enumerate(sequence):
                if char != "-":
                    query_sequence[i] = "G"

    query_sequence = "".join(query_sequence)

    if verbose:
        print(f"Length of MSA: {len(query_sequence)}.")
        print(f'Non-gaps in query: {len(query_sequence.replace("-", ""))}')
        print(f'First non-gap at position {query_sequence.find("G")}.')
        print(f'Last non-gap at position {query_sequence[::-1].find("G")}.')
        print(
            f'Non-gap interval length: {query_sequence[::-1].find("G") - query_sequence.find("A")}'
        )

    with open(f"data/interim/{dataset}/{dataset}_EVE_query.txt", "w") as f:
        f.write(query_sequence)

    if verbose:
        print(
            f"Query sequence written to data/interim/{dataset}/{dataset}_EVE_query.txt"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()
    generate_query_sequence(args.dataset)
