import argparse

from src.visualization.visualization_funcs import generate_target_histograms, generate_partition_histograms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_format", type=str, default="pdf")

    args = parser.parse_args()
    generate_target_histograms(args.image_format)
    generate_partition_histograms(args.image_format)
