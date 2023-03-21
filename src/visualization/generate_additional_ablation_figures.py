import argparse

from src.visualization.visualization_funcs import results_single_dataset, show_ablations_all_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Name of dataset.")
    parser.add_argument("--metric", type=str, default="spearman", help="Which metric to use.")
    parser.add_argument("--image_format", type=str, default="pdf")
    parser.add_argument("--ablation_method", type=str, required=True)
    parser.add_argument("--all_ablation", action="store_true")
    parser.add_argument(
        "--embedding_types", type=str, nargs="+", default=("af2", "esm_1b", "esm_2", "esm_if1", "eve", "onehot", "ct")
    )

    args = parser.parse_args()
    if args.all_ablation:
        args.datasets = ["gh114", "cm", "ppat"]
        show_ablations_all_data(**vars(args))
    else:
        results_single_dataset(ablation=True, **vars(args))
