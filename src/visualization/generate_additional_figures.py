import argparse

from src.visualization.visualization_funcs import results_single_dataset, show_results_all_predictors
from src import REPRESENTATIONS, DATASETS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Name of dataset.")
    parser.add_argument("--metric", type=str, default="spearman", help="Which metric to use.")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--image_format", type=str, default="pdf")
    parser.add_argument("--ablation_method", type=str)
    parser.add_argument("--all_ablation", action="store_true")
    parser.add_argument(
        "--embedding_types", type=str, nargs="+", default=REPRESENTATIONS
    )

    args = parser.parse_args()

    if args.ablation:
        del args.ablation
        if args.all_ablation:
            args.datasets = DATASETS
            show_results_all_predictors(ablation=True, **vars(args))
        else:
            results_single_dataset(ablation=True, **vars(args))
    else:
        del args.ablation
        args.datasets = DATASETS
        show_results_all_predictors(ablation=False, **vars(args))

