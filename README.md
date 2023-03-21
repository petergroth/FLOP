# FLOP: Tasks for Fitness Landscapes Of Protein wildtypes

This is the official code repository for the paper "FLOP: Tasks for Fitness Landscapes Of Protein wildtypes" (LINK TO PAPER).

## Installation
After cloning this repository, an environment can be built directly via

```bash
conda env create -f flop.yml
```

### Install Graph-Part
The splitting procedure uses Graph-Part (https://github.com/graph-part/graph-part). The repository can be installed (as per the README) via

```bash
git clone https://github.com/fteufel/graph-part.git
cd graph-part
pip install .
```

### Optional dependencies
The generation of protein representations relies on various codebases.

#### ESM
To generate *Evolutionary Scale Modeling* representations from https://github.com/facebookresearch/esm, install 

```bash
pip install fair-esm
```
#### EVE 
To generate EVE-based representations from https://github.com/OATML-Markslab/EVE, clone and install the repository locally.
Note, the function `EVE/utils/data_utils.py` function has been altered to work with the wildtype query sequence. This file is therefore included in this repository.
 
#### Tranception
For unsupervised Tranception scoring from https://github.com/OATML-Markslab/Tranception, clone and install the repository locally.

#### PyBioMed
For compositional/transitional representation generation from https://github.com/gadsbyfly/PyBioMed, clone and install the repository locally.

## Reproduce results

```bash

# Process and split datasets [optional]
bash scripts/compile_dataset.sh

# Re-train EVE [optional]
bash scripts/train_EVE_models.sh

# Generate representations. Requires optional dependencies. [optional]
bash scripts/generate_representations.sh

# Run cross-validation and ablation studies
bash scripts/reproduce.sh

# Process results and generate figures
bash scripts/process_results.sh

```

## Adding new representations
To run the setup on a novel representation, a directory should be created for each dataset: `./embeddings/$dataset/new_representation`, in which all the representation of each sequence is saved as individual files (following the current convention).

A new option should then be added to the `extract_all_embeddings`-function in `./src/data/data_utils.py`, which follows the convention present of sequentially loading all representations and saving them in a numpy array.

This representation can then be benchmarked for a dataset by running 

```bash
python src/training/fit_model.py --dataset dataset --embedding_types "new_representation"
```

The results can be processed, saved, and visualized with

```bash
python src/process_results.py --dataset datset --save_csv --bar_plot --embedding_types "new_representation"
```

NOTE: This might overwrite existing figures/result files.


## Recreate phylogenetic trees
To recreate the included phylogenetic trees, run the following 
```bash
bash scripts/generate_phylogenetic_trees.sh ppat
bash scripts/mmseqs_clustering.sh ppat
python src/visualization/visualize_phylo_tree.py --dataset ppat
```
This procedure generates a tree using FastTree (www.microbesonline.org/fasttree/). 
The tree is visualized with three different coloring-schemes; one according to the CV partitions prescribed in the manuscript, one according to an MMseqs-clustering, and one random coloring corresponding to random splitting.
The following dependencies are introduced: 
- csvtk for handling csv/tsv files (https://bioinf.shenwei.me/csvtk/)
- FastTree for tree generation (www.microbesonline.org/fasttree/)
- SeqKit for handling MSAs (https://bioinf.shenwei.me/seqkit/)
- ETE Toolkit for tree visualization (http://etetoolkit.org/)
- MMseqs2 for clustering (https://github.com/soedinglab/MMseqs2)


## Project organization
------------

    ├── data/
    │   ├── interim/                # Files used in splitting, representation creating etc.
    │   │   ├── cm/                        
    │   │   ├── gh114/                        
    │   │   ├── ppat/                        
    │   │   └── eve_mappings.csv    # File for training EVE                      
    │   │                    
    │   ├── processed/              # Cleaned CSV-files with ids, sequences, targets, splits.
    │   └── raw/                    # Raw data files. 
    │
    ├── EVE/utils/data_utils.py     # Altered file from EVE to work well with wildtype query sequence
    │
    ├── figures/                    # Various figures used in the manuscript
    │
    ├── models/                     # Location for pretrained/trained models (ESM, Tranception, EVE)
    │   ├── EVE/                
    │   │   ├── cm/                        
    │   │   ├── gh114/                                                
    │   │   └── ppat/                    
    │   │                    
    │   ├── ...
    │   └── ...
    │
    ├── representations/
    │   ├── cm/    
    │   │   ├── af2/                        
    │   │   ├── ct/                         
    │   │   ├── esm_1b/                     
    │   │   ├── esm_2/                      
    │   │   ├── esm_if1/                    
    │   │   ├── eve/                         
    │   │   │   ├── 0/                      # Latents from EVE trained with seed 0 (1 file per sequence)
    │   │   │   ├── 1/                      # Latents from EVE trained with seed 1 (1 file per sequence)
    │   │   │   └── 2/                      # Latents from EVE trained with seed 2 (1 file per sequence)
    │   │   │
    │   │   ├── onehot/
    │   │   ├── cm_EVE_ELBO_0.csv           # ELBOs for EVE model trained with seed 0 
    │   │   ├── cm_EVE_ELBO_1.csv           # ELBOs for EVE model trained with seed 1 
    │   │   ├── cm_EVE_ELBO_2.csv           # ELBOs for EVE model trained with seed 2 
    │   │   ├── cm_tranception_scores.csv   # Tranception scores
    │   │   └── esm_if1_likelihoods.csv     # ESM-IF1 structured-conditioned likelihoods
    │   │
    │   ├── gh114/
    │   │   └── ...
    │   │
    │   └── ppat/   
    │       └── ...
    │
    ├── results/
    │
    ├── scripts/     # Shell scripts to run experiments and generate figures
    │   ├── compile_datasets.sh             # Compile datasets with 
    │   ├── generate_MSAs.sh                # Procedure to generate MSAs (given family-wide FASTAs)
    │   ├── generate_phylogenetic_trees.sh  # Comp. (and visualize) phylogenetic tree based on MSA
    │   ├── generate_representations.sh     # Gen. all representations (excepting EVE-based latents and ELBOs)
    │   ├── mmseqs_clustering.sh            # Runs a simple MMseqs2 easy-clust clustering for a dataset
    │   ├── process_results.sh              # Process results and draw figures
    │   ├── reproduce.sh                    # Run all experiments
    │   └── train_EVE_models.sh             # Train EVE, compute ELBOs and extract latents for all datasets
    │
    ├── src/        
    │   ├── __init__.py
    │   ├── process_results.py   # Script to process results and print $\latex$-formated tables
    │   ├── visualize_results.py # Script to generate visualization of CV results + ablation
    │   │
    │   ├── data/       
    │   │   ├── __init__.py
    │   │   ├── compile_cm.py       # Compilation of cm dataset         
    │   │   ├── compile_dataset.py  # Compilation of general datasets [for new data]
    │   │   ├── compile_gh114.py    # Compilation of gh114 dataset
    │   │   ├── compile_ppat.py     # Compilation of ppat dataset
    │   │   └── data_utils.py       # Helper functions for dataset creation
    │   │
    │   ├── representations/ 
    │   │   ├── __init__.py
    │   │   ├── compute_esm_if1_lls.py              # Comp. ESM-IF1 sequence likehoods (given structure)
    │   │   ├── compute_eve_elbos.py                # Comp. EVE ELBOs (given trained EVE)
    │   │   ├── compute_tranception_scores.py       # Comp. Tranception scores 
    │   │   ├── generate_ct_representations.py      # Gen. CT (compositional-transitional biological descriptors)
    │   │   ├── generate_esm_if1_representations.py # Gen. ESM-IF1 embeddings
    │   │   ├── generate_esm_representations.py     # Gen. ESM-1B/2 embeddings
    │   │   ├── generate_eve_query.py               # Gen. query sequence for training EVE
    │   │   ├── generate_eve_representations.py     # Gen. latent EVE representation (given trained EVE)
    │   │   ├── generate_onehot_representations.py  # Gen. (flattened) onehot-representation  (given MSA)
    │   │   └── train_EVE.py                        # Modified script to train EVE models. Adapted from 
    │   │                                           # https://github.com/OATML-Markslab/EVE/blob/master/train_VAE.py
    │   │
    │   ├── training/
    │   │   ├── __init__.py
    │   │   ├── compute_unsupervised_correlations.py    # Comp. (unsupervised) Spearman correlations for EVE ELBOs, 
    │   │   │                                           # ESM-IF1 likelihoods and Tranception scores
    │   │   ├── fit_model.py                            # Run CV for a dataset (+ ablation studies)
    │   │   └── training_utils.py                       # Various functions used in fit_model.py
    │   │
    │   └── visualization/
    │       ├── __init__.py
    │       ├── generate_additional_ablation_figures.py # Gen. figures from ablation studies
    │       ├── generate_histograms.py                  # Gen. various histograms 
    │       ├── visualization_funcs.py                  # Various functions for visualization of data + results
    │       └── visualize_phylo_tree.py                 # Gen. phylogenetic tree viz. based on precomputed tree
    │
    ├── LICENSE
    ├── README.md     
    └── flop.yml   # Project dependencies. Create environment with conda. To recreate representations, additional 
                   # repos are required.
    

--------