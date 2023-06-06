import argparse
from src.visualization.visualization_funcs import all_test_results_barplot
from src import REPRESENTATIONS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regressor", type=str, default="Ridge", help="Name of regressor to use for plot")
    parser.add_argument("--image_format", type=str, default="pdf")
    parser.add_argument("--metric", type=str, default="spearman")
    parser.add_argument(
        "--unsupervised",
        action="store_true",
        default=False,
        help="Whether to include horizontal lines for unsupervised predictors.",
    )
    parser.add_argument("--ablation", action="store_true", default=False)
    args = parser.parse_args()

    embedding_types = REPRESENTATIONS
    datasets = ("gh114", "cm", "ppat")
    all_test_results_barplot(
        datasets=datasets,
        embedding_types=embedding_types,
        regressor=args.regressor,
        metric=args.metric,
        save_fig=True,
        image_format=args.image_format,
        unsupervised=args.unsupervised,
        ablation=args.ablation,
    )
