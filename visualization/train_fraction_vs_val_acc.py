#!/usr/bin/env python
"""
绘制单张「训练数据占比 vs 最佳验证准确率」图（限定10w steps以内最大值）
优化：固定Y轴上限为102，清晰展示接近100%的区域（无局部放大）
"""
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import argparse
from typing import List, Tuple

# ===================== 辅助函数：加载实验数据（无修改） =====================
def load_single_exp_data(exp_dir: str) -> Tuple[float, float]:
    """加载单个实验的“训练数据占比”和“10w steps以内最佳验证准确率”"""
    try:
        # 读取hparams.yaml中的train_data_pct（转为小数）
        hparams_path = os.path.join(exp_dir, "hparams.yaml")
        with open(hparams_path, "r", encoding="utf-8") as f:
            hparams = yaml.safe_load(f)
        train_data_fraction = hparams["train_data_pct"] / 100.0

        # 读取metrics文件中的最佳val_accuracy（限定10w steps以内）
        metrics_dir = os.path.join(exp_dir, "metrics")
        metrics_files = [f for f in os.listdir(metrics_dir) if f.endswith(".csv")]
        if not metrics_files:
            raise FileNotFoundError("未找到metrics CSV文件")
        
        metrics_path = os.path.join(metrics_dir, metrics_files[0])
        df = pd.read_csv(metrics_path)

        # 关键修改：过滤出10w steps及以内的数据（默认列名为"step"，若你的列名不同可自行修改）
        max_step_limit = 100000  # 10w steps
        if "global_step" not in df.columns:
            raise KeyError(f"metrics文件 {metrics_path} 中无'global_step'列，请确认列名是否正确（如'epoch'）")

        # 过滤数据：仅保留global_step <= 100000的记录
        df_filtered = df[df["global_step"] <= max_step_limit]
        if df_filtered.empty:
            raise ValueError(f"过滤后（global_step <= {max_step_limit}）无有效数据")

        # 取过滤后的数据最大值
        best_val_acc = df_filtered["val_accuracy"].max()

        return (train_data_fraction, best_val_acc)
    except Exception as e:
        print(f"⚠️ 加载实验{exp_dir}失败：{e}")
        return (None, None)

def load_single_config_data(config_dir: str) -> List[Tuple[float, float]]:
    """加载单个配置的所有实验数据（用于绘制单张图）"""
    config_data = []
    # 遍历配置下的所有实验目录
    for exp_dir in os.listdir(config_dir):
        full_exp_dir = os.path.join(config_dir, exp_dir)
        if not os.path.isdir(full_exp_dir):
            continue
        
        fraction, acc = load_single_exp_data(full_exp_dir)
        if fraction is not None and acc is not None:
            config_data.append((fraction, acc))
    
    config_data.sort(key=lambda x: x[0])  # 按训练数据占比从小到大排序
    return config_data

# ===================== 核心函数：绘制单张独立图（固定Y轴上限为102，无局部放大） =====================
def plot_single_chart(
    data: List[Tuple[float, float]],
    chart_title: str,
    save_path: str,
    figsize: Tuple[int, int] = (5, 4)
) -> None:
    """
    绘制单张“训练数据占比 vs 最佳验证准确率”独立图
    固定Y轴上限为102，预留顶部空间，清晰展示接近100%的区域
    无局部放大，保持简洁
    """
    # 创建单张图画布
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # 绘制数据（有数据画散点+折线，无数据显示提示文字）
    if data:
        fractions = [d[0] for d in data]
        accuracies = [d[1] for d in data]
        # 绘制散点和折线（样式不变）
        ax.scatter(fractions, accuracies, color="#cccccc", s=15, alpha=0.8)
        ax.plot(fractions, accuracies, color="#1e88e5", linewidth=2)
    else:
        ax.text(0.5, 0.5, "无有效数据", ha="center", va="center", transform=ax.transAxes)

    # 单张图样式配置（核心修改：固定Y轴范围 0~102）
    ax.set_title(chart_title, fontsize=10, fontweight="bold")
    ax.set_xlim(0.2, 0.8)  # 固定x轴范围（20%~80%训练数据占比）
    ax.set_ylim(0, 102)    # 核心优化：固定Y轴上限为102，预留2%空间，避免100%贴顶
    # x轴显示为百分比格式（0.5 → 50%）
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.grid(True, which="both", alpha=0.3, linestyle="--")  # 网格线增强可读性
    ax.set_xlabel("Training data fraction", fontsize=9)
    ax.set_ylabel("Best validation accuracy (≤10w steps)", fontsize=9)  # 标注10w steps限制

    # 保存图片并关闭画布（释放内存，避免内存泄漏）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 自动创建保存目录
    plt.tight_layout()  # 自动调整布局，避免标题/标签重叠
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ 单张图已成功保存至：{save_path}")

# ===================== Parser配置：适配统一入口脚本（无修改） =====================
def add_parser(subparsers):
    """
    为当前工具添加命令行参数（适配统一可视化入口脚本）
    :param subparsers: 统一入口脚本的子解析器
    """
    parser = subparsers.add_parser(
        "train_fraction_vs_val_acc",
        help="绘制单张「训练数据占比 vs 10w steps以内最佳验证准确率」趋势图",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 必选参数
    parser.add_argument(
        "--config-dir", "-i",
        required=True,
        type=str,
        help="单个配置的实验根目录（如：my_experiments/all_configs/full_batch_adam）"
    )
    parser.add_argument(
        "--chart-title", "-t",
        required=True,
        type=str,
        help="单张图的标题（通常为配置名称，如：Full Batch Adam）"
    )
    parser.add_argument(
        "--save-path", "-o",
        required=True,
        type=str,
        help="单张图的保存路径（如：my_visualization/single_charts/full_batch_adam.png）"
    )
    # 可选参数
    parser.add_argument(
        "--figsize-w",
        type=int,
        default=5,
        help="单张图的宽度（默认：5）"
    )
    parser.add_argument(
        "--figsize-h",
        type=int,
        default=4,
        help="单张图的高度（默认：4）"
    )

# ===================== 主函数：执行绘制逻辑（无修改） =====================
def main(args):
    """
    工具主执行函数（适配统一可视化入口脚本）
    :param args: 命令行解析后的参数对象
    """
    # 1. 加载配置数据
    config_exp_data = load_single_config_data(args.config_dir)
    # 2. 组装图尺寸
    fig_size = (args.figsize_w, args.figsize_h)
    # 3. 绘制并保存单张图
    plot_single_chart(
        data=config_exp_data,
        chart_title=args.chart_title,
        save_path=args.save_path,
        figsize=fig_size
    )

# ===================== 单独运行入口（可选） =====================
if __name__ == "__main__":
    # 手动构造参数（单独运行时使用）
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers(dest="cmd"))
    args = parser.parse_args()
    main(args)