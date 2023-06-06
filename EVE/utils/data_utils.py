import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import tqdm


class MSA_processing:
    def __init__(
        self,
        MSA_location="",
        theta=0.2,
        use_weights=True,
        weights_location="./data/weights",
        preprocess_MSA=True,
        threshold_sequence_frac_gaps=0.5,
        threshold_focus_cols_frac_gaps=0.3,
        remove_sequences_with_indeterminate_AA_in_focus_cols=True,
        protein_family="",  # Input added to extract query sequence.
    ):
        """
        Parameters:
        - msa_location: (path_in) Location of the MSA data. Constraints on input MSA format:
            - focus_sequence is the first one in the MSA data
            - first line is structured as follows: ">focus_seq_name/start_pos-end_pos" (e.g., >SPIKE_SARS2/310-550)
            - corresponding sequence data located on following line(s)
            - then all other sequences follow with ">name" on first line, corresponding data on subsequent lines
        - theta: (float) Sequence weighting hyperparameter. Generally: Prokaryotic and eukaryotic families =  0.2; Viruses = 0.01
        - use_weights: (bool) If False, sets all sequence weights to 1. If True, checks weights_location -- if non empty uses that;
            otherwise compute weights from scratch and store them at weights_location
        - weights_location: (path_in) Location to load from/save to the sequence weights
        - preprocess_MSA: (bool) performs pre-processing of MSA to remove short fragments and positions that are not well covered.
        - threshold_sequence_frac_gaps: (float, between 0 and 1) Threshold value to define fragments
            - sequences with a fraction of gap characters above threshold_sequence_frac_gaps are removed
            - default is set to 0.5 (i.e., fragments with 50% or more gaps are removed)
        - threshold_focus_cols_frac_gaps: (float, between 0 and 1) Threshold value to define focus columns
            - positions with a fraction of gap characters above threshold_focus_cols_pct_gaps will be set to lower case (and not included in the focus_cols)
            - default is set to 0.3 (i.e., focus positions are the ones with 30% of gaps or less, i.e., 70% or more residue occupancy)
        - remove_sequences_with_indeterminate_AA_in_focus_cols: (bool) Remove all sequences that have indeterminate AA (e.g., B, J, X, Z) at focus positions of the wild type
        - protein_family: str() Used to load query sequence.
        """
        # fmt: off
        np.random.seed(2021)
        self.MSA_location = MSA_location
        self.weights_location = weights_location
        self.theta = theta
        self.alphabet = "ACDEFGHIKLMNPQRSTVWY"
        self.use_weights = use_weights
        self.preprocess_MSA = preprocess_MSA
        self.threshold_sequence_frac_gaps = threshold_sequence_frac_gaps
        self.threshold_focus_cols_frac_gaps = threshold_focus_cols_frac_gaps
        self.remove_sequences_with_indeterminate_AA_in_focus_cols = (
            remove_sequences_with_indeterminate_AA_in_focus_cols
        )
        self.protein_family = protein_family  # Addition
        self.gen_alignment()
        # fmt: on

    def gen_alignment(self):
        """Read training alignment and store basics in class instance"""
        # fmt: off
        self.aa_dict = {}
        for i, aa in enumerate(self.alphabet):
            self.aa_dict[aa] = i

        self.seq_name_to_sequence = defaultdict(str)

        #######################
        # Load query sequence #
        #######################
        with open(f"data/interim/{self.protein_family}/{self.protein_family}_EVE_query.txt", "r") as f:
            query_sequence = f.readlines()[0]
        self.focus_seq_name = "QUERY"
        self.seq_name_to_sequence[self.focus_seq_name] = query_sequence
        #######################

        # Read MSA and add to dict
        name = ""
        with open(self.MSA_location, "r") as msa_data:
            for i, line in enumerate(msa_data):
                line = line.rstrip()
                if line.startswith(">"):
                    name = line.split("\t")[0]  # Will be fasta dependent.
                else:
                    self.seq_name_to_sequence[name] += line

        # MSA pre-processing to remove inadequate columns and sequences
        if self.preprocess_MSA:
            msa_df = pd.DataFrame.from_dict(self.seq_name_to_sequence, orient='index', columns=['sequence'])
            # Data clean up
            msa_df.sequence = msa_df.sequence.apply(lambda x: x.replace(".", "-")).apply(
                lambda x: ''.join([aa.upper() for aa in x]))
            # Remove columns that would be gaps in the wild type
            non_gap_wt_cols = [aa != '-' for aa in msa_df.sequence[self.focus_seq_name]]
            msa_df['sequence'] = msa_df['sequence'].apply(
                lambda x: ''.join([aa for aa, non_gap_ind in zip(x, non_gap_wt_cols) if non_gap_ind]))
            assert 0.0 <= self.threshold_sequence_frac_gaps <= 1.0, "Invalid fragment filtering parameter"
            assert 0.0 <= self.threshold_focus_cols_frac_gaps <= 1.0, "Invalid focus position filtering parameter"
            msa_array = np.array([list(seq) for seq in msa_df.sequence])
            gaps_array = np.array(list(map(lambda seq: [aa == '-' for aa in seq], msa_array)))
            # Identify fragments with too many gaps
            seq_gaps_frac = gaps_array.mean(axis=1)
            seq_below_threshold = seq_gaps_frac <= self.threshold_sequence_frac_gaps

            ###########################
            # Ensure target sequences #
            # are kept                #
            ###########################
            df_targets = pd.read_csv(
                f"data/interim/{self.protein_family}/{self.protein_family}.csv"
            )
            target_names = [">" + name for name in df_targets["name"].tolist()]
            target_idx = np.zeros(len(msa_df), dtype=bool)
            for i, name in enumerate(msa_df.index):
                if name.split("\t")[0] in target_names:
                    target_idx[i] = True
            seq_below_threshold[target_idx] = True
            ###########################

            print("Proportion of sequences dropped due to fraction of gaps: " + str(
                round(float(1 - seq_below_threshold.sum() / seq_below_threshold.shape) * 100, 2)) + "%")
            # Identify focus columns
            columns_gaps_frac = gaps_array[seq_below_threshold].mean(axis=0)
            index_cols_below_threshold = columns_gaps_frac <= self.threshold_focus_cols_frac_gaps
            print("Proportion of non-focus columns removed: " + str(
                round(float(1 - index_cols_below_threshold.sum() / index_cols_below_threshold.shape) * 100, 2)) + "%")
            # Lower case non focus cols and filter fragment sequences
            msa_df['sequence'] = msa_df['sequence'].apply(lambda x: ''.join(
                [aa.upper() if upper_case_ind else aa.lower() for aa, upper_case_ind in
                 zip(x, index_cols_below_threshold)]))
            msa_df = msa_df[seq_below_threshold]
            # Overwrite seq_name_to_sequence with clean version
            self.seq_name_to_sequence = defaultdict(str)
            for seq_idx in range(len(msa_df['sequence'])):
                self.seq_name_to_sequence[msa_df.index[seq_idx]] = msa_df.sequence[seq_idx]

        self.focus_seq = self.seq_name_to_sequence[self.focus_seq_name]
        self.focus_cols = [ix for ix, s in enumerate(self.focus_seq) if s == s.upper() and s != '-']
        self.focus_seq_trimmed = [self.focus_seq[ix] for ix in self.focus_cols]
        self.seq_len = len(self.focus_cols)
        self.alphabet_size = len(self.alphabet)

        # Move all letters to CAPS; keeps focus columns only
        for seq_name, sequence in self.seq_name_to_sequence.items():
            sequence = sequence.replace(".", "-")
            self.seq_name_to_sequence[seq_name] = [sequence[ix].upper() for ix in self.focus_cols]

        # Remove sequences that have indeterminate AA (e.g., B, J, X, Z) in the focus columns
        if self.remove_sequences_with_indeterminate_AA_in_focus_cols:
            alphabet_set = set(list(self.alphabet))
            seq_names_to_remove = []
            for seq_name, sequence in self.seq_name_to_sequence.items():
                for letter in sequence:
                    if letter not in alphabet_set and letter != "-":
                        seq_names_to_remove.append(seq_name)
                        continue
            seq_names_to_remove = list(set(seq_names_to_remove))
            for seq_name in seq_names_to_remove:
                ##############################
                # Ensure that sequences with #
                # target values are kept     #
                ##############################
                if self.preprocess_MSA:
                    if seq_name in target_names:
                        print("REMOVING TARGET SEQUENCE")
                        sys.exit(0)
                ##############################
                del self.seq_name_to_sequence[seq_name]

        #########################
        # Remove query sequence #
        #########################
        self.seq_name_to_sequence.pop("QUERY")
        #########################

        # Encode the sequences
        print("Encoding sequences")
        self.one_hot_encoding = np.zeros(
            (len(self.seq_name_to_sequence.keys()), len(self.focus_cols), len(self.alphabet)))
        for i, seq_name in enumerate(self.seq_name_to_sequence.keys()):
            sequence = self.seq_name_to_sequence[seq_name]
            for j, letter in enumerate(sequence):
                if letter in self.aa_dict:
                    k = self.aa_dict[letter]
                    self.one_hot_encoding[i, j, k] = 1.0

        if self.use_weights:
            try:
                self.weights = np.load(file=self.weights_location)
                print("Loaded sequence weights from disk")
            except:
                print("Computing sequence weights")
                list_seq = self.one_hot_encoding
                list_seq = list_seq.reshape((list_seq.shape[0], list_seq.shape[1] * list_seq.shape[2]))

                def compute_weight(seq):
                    number_non_empty_positions = np.dot(seq, seq)
                    if number_non_empty_positions > 0:
                        denom = np.dot(list_seq, seq) / np.dot(seq, seq)
                        denom = np.sum(denom > 1 - self.theta)
                        return 1 / denom
                    else:
                        return 0.0  # return 0 weight if sequence is fully empty

                self.weights = np.array(list(map(compute_weight, list_seq)))
                np.save(file=self.weights_location, arr=self.weights)
        else:
            # If not using weights, use an isotropic weight matrix
            print("Not weighting sequence data")
            self.weights = np.ones(self.one_hot_encoding.shape[0])

        self.Neff = np.sum(self.weights)
        self.num_sequences = self.one_hot_encoding.shape[0]

        print("Neff =", str(self.Neff))
        print("Data Shape =", self.one_hot_encoding.shape)

        # fmt: on
