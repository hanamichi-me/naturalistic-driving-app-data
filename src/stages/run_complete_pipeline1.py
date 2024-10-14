import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.stages.visualization import complete_pipeline
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import math


def main(config_path, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    axes[0, 0] = complete_pipeline(config_path=config_path, feature_set_index=0, ax=axes[0, 0])
    axes[0, 1] = complete_pipeline(config_path=config_path, feature_set_index=1, ax=axes[0, 1])
    axes[1, 0] = complete_pipeline(config_path=config_path, feature_set_index=2, ax=axes[1, 0])
    axes[1, 1] = complete_pipeline(config_path=config_path, feature_set_index=3, ax=axes[1, 1], save_path=f"{save_dir}/output_plot_target_04.png")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete pipeline for visualization")
    parser.add_argument('--config_path', type=str, default="params.yaml", help="Path to configuration file")
    parser.add_argument('--save_dir', type=str, default="output", help="Directory to save output plots")
    
    args = parser.parse_args()
    main(args.config_path, args.save_dir)


