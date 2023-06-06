"""Script to compute Tranception scores. Adapted from
https://github.com/OATML-Markslab/Tranception/blob/main/score_tranception_proteingym.py"""
import os
import argparse
import json
import pandas as pd

import torch

from transformers import PreTrainedTokenizerFast
import tranception
from tranception import config, model_pytorch


def main(dataset: str):
    if dataset == "cm":
        df = pd.read_csv(f"data/processed/{dataset}/{dataset}_all.csv", index_col=0)[["name", "sequence"]]
    else:
        df = pd.read_csv(f"data/processed/{dataset}/{dataset}.csv", index_col=0)[["name", "sequence"]]

    # Unpacked args
    checkpoint = "models/Tranception_Large/pytorch_model.bin"
    MSA_filename = f"data/raw/{dataset}/{dataset}_family.aln.fasta"
    MSA_weight_file_name = None
    MSA_start = 1
    MSA_end = 12000
    DMS_id = df["name"].tolist()
    num_workers = 1
    indel_mode = False

    df["mutated_sequence"] = df["sequence"]

    model_name = checkpoint.split("/")[-1]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="Tranception/tranception/utils/tokenizers/Basic_tokenizer",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )

    target_seq = None
    config = json.load(open("models/Tranception_Large/config.json"))
    config = tranception.config.TranceptionConfig(**config)
    config.attention_mode = "tranception"
    config.position_embedding = "grouped_alibi"
    config.tokenizer = tokenizer
    config.scoring_window = "optimal"

    config.retrieval_aggregation_mode = "aggregate_indel" if indel_mode else "aggregate_substitution"
    config.MSA_filename = MSA_filename
    config.full_protein_length = MSA_end
    config.MSA_weight_file_name = MSA_weight_file_name
    config.MSA_start = MSA_start
    config.MSA_end = MSA_end

    model = tranception.model_pytorch.TranceptionLMHeadModel.from_pretrained(
        pretrained_model_name_or_path=checkpoint, config=config
    )

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    print(torch.cuda.device_count())

    DMS_data = df.copy()
    all_scores = model.score_mutants(
        DMS_data=DMS_data,
        target_seq=target_seq,
        scoring_mirror=True,
        batch_size_inference=1,
        num_workers=num_workers,
        indel_mode=indel_mode,
    )

    all_scores = all_scores.drop_duplicates()
    df_c = pd.merge(df, all_scores[["mutated_sequence", "avg_score"]], on="mutated_sequence", how="left")

    df_c.to_csv(f"representations/{dataset}/{dataset}_tranception_scores.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()
    main(**vars(args))
