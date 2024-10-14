import sys
import os
# 添加项目的根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.stages.visualization import complete_pipeline
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import math


def main(config_path, save_dir):
    # 创建2x2的子图
    fig, axes = plt.subplots(3, 3, figsize=(24, 18))

    # 调用 complete_pipeline 生成不同的图像
    axes[0, 0] = complete_pipeline(config_path=config_path, feature_set_index=4, ax=axes[0, 0])
    axes[0, 1] = complete_pipeline(config_path=config_path, feature_set_index=5, ax=axes[0, 1])
    axes[0, 2] = complete_pipeline(config_path=config_path, feature_set_index=6, ax=axes[0, 2])
    axes[1, 0] = complete_pipeline(config_path=config_path, feature_set_index=7, ax=axes[1, 0])
    axes[1, 1] = complete_pipeline(config_path=config_path, feature_set_index=8, ax=axes[1, 1])
    axes[1, 2] = complete_pipeline(config_path=config_path, feature_set_index=9, ax=axes[1, 2])
    axes[2, 0] = complete_pipeline(config_path=config_path, feature_set_index=10, ax=axes[2, 0])
    axes[2, 1] = complete_pipeline(config_path=config_path, feature_set_index=11, ax=axes[2, 1])
    axes[2, 2] = complete_pipeline(config_path=config_path, feature_set_index=12, ax=axes[2, 2],save_path=f"{save_dir}/output_plot_target_13.png")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete pipeline for visualization")
    parser.add_argument('--config_path', type=str, default="params.yaml", help="Path to configuration file")
    parser.add_argument('--save_dir', type=str, default="output", help="Directory to save output plots")
    
    args = parser.parse_args()
    main(args.config_path, args.save_dir)

# def main(config_path, save_dir):
#     # 从 features_and_targets.py 中加载特征和目标列表
#     from features_and_targets1 import features_and_targets
    
#     n_features = len(features_and_targets)
#     cols = 3  # 固定列数为3
#     rows = math.ceil(n_features / cols)  # 根据特征集的数量确定行数

#     # 创建动态数量的子图
#     fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))

#     # 将 axes 转换为 1D 数组，便于通过索引访问
#     axes = axes.flatten()

#     # 遍历每个特征集，调用 complete_pipeline 生成对应的图像
#     for i, feature_set in enumerate(features_and_targets):
#         save_path = None
#         if i == len(features_and_targets) - 1:  # 如果是最后一个图，则保存
#             save_path = f"{save_dir}/output_plot_target_{i+1}.png"
        
#         print(f"Processing feature set {i+1}: {feature_set['target']}")
        
#         # 调用 complete_pipeline 处理每个特征集
#         axes[i] = complete_pipeline(
#             config_path=config_path,
#             feature_set_index=i,
#             ax=axes[i],
#             save_path=save_path
#         )

#     # 调整布局并显示
#     plt.tight_layout()
#     plt.show()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run complete pipeline for visualization")
#     parser.add_argument('--config_path', type=str, default="params.yaml", help="Path to configuration file")
#     parser.add_argument('--save_dir', type=str, default="output", help="Directory to save output plots")
    
#     args = parser.parse_args()

#     # 确保输出目录存在
#     os.makedirs(args.save_dir, exist_ok=True)

#     # 调用 main 函数
#     main(args.config_path, args.save_dir)
