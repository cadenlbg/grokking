#!/usr/bin/env python
"""
无空白填充，灵活拼接多张单图为网格大图
配套统一可视化入口脚本，实现add_parser和main方法
"""
import os
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
from typing import List, Optional, Tuple

# ===================== 核心函数：无空白填充，灵活拼接网格图 =====================
def merge_charts_to_grid(
    single_chart_paths: List[str],
    merge_save_path: str,
    figsize: Optional[Tuple[int, int]] = None,
    global_title: str = "Training Data Fraction vs Best Validation Accuracy",
    n_cols: Optional[int] = 3,
    n_rows: Optional[int] = None,
    show_empty_subplot: bool = False
) -> None:
    """
    灵活拼接多张单图为网格大图（无空白填充，仅显示现有图片）
    :param single_chart_paths: 单张图路径列表（支持png/jpg/jpeg格式）
    :param merge_save_path: 拼接后大图保存路径
    :param figsize: 大图尺寸，默认根据行列数自动计算（适配单图(5,4)）
    :param global_title: 大图全局标题
    :param n_cols: 自定义列数（如3、2、4），None则自动计算最优列数
    :param n_rows: 自定义行数（可选，优先级低于n_cols，指定后列数自动适配）
    :param show_empty_subplot: 是否显示多余空白子图（False=隐藏，更整洁）
    """
    # 步骤1：参数校验 & 动态计算行列数（仅基于现有图片数量，无填充）
    chart_count = len(single_chart_paths)
    if chart_count == 0:
        print("❌ 错误：单张图路径列表为空，无法拼接")
        return

    # 优先按 n_cols 计算行数
    if n_cols is not None and n_rows is None:
        n_cols = max(1, n_cols)  # 列数至少为1
        n_rows = math.ceil(chart_count / n_cols)  # 向上取整得到行数
    # 其次按 n_rows 计算列数
    elif n_rows is not None and n_cols is None:
        n_rows = max(1, n_rows)  # 行数至少为1
        n_cols = math.ceil(chart_count / n_rows)  # 向上取整得到列数
    # 若都未指定，自动计算最优行列数（接近正方形，减少空白）
    else:
        n_cols = math.ceil(math.sqrt(chart_count))
        n_rows = math.ceil(chart_count / n_cols)

    total_subplot_count = n_rows * n_cols  # 网格总子图数（仅用于布局计算）

    # 步骤2：自动计算大图尺寸（若未指定）
    if figsize is None:
        single_fig_w, single_fig_h = (5, 4)  # 与单图尺寸匹配
        fig_w = single_fig_w * n_cols
        fig_h = single_fig_h * n_rows
        figsize = (fig_w, fig_h)

    # 步骤3：创建动态网格画布并拼接图片
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    # 处理单行列/单图的边界情况（避免axes格式异常）
    if n_rows == 1 and n_cols == 1:
        axes_flat = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = axes.flatten()

    # 遍历现有图片，依次放入对应子图
    for idx in range(chart_count):
        ax = axes_flat[idx]
        chart_path = single_chart_paths[idx]
        # 读取并显示单张图（确保文件存在）
        if os.path.exists(chart_path):
            img = mpimg.imread(chart_path)
            ax.imshow(img)
        ax.axis("off")  # 关闭子图坐标轴，保持整洁

    # 处理多余空白子图（隐藏/保留可选，无填充逻辑）
    if not show_empty_subplot and chart_count < total_subplot_count:
        # 隐藏多余空白子图，视觉更整洁
        for idx in range(chart_count, total_subplot_count):
            axes_flat[idx].set_visible(False)
    else:
        # 若保留空白子图，也关闭其坐标轴
        for idx in range(chart_count, total_subplot_count):
            axes_flat[idx].axis("off")

    # 步骤4：大图全局样式配置
    fig.suptitle(global_title, fontsize=16, y=0.98)
    fig.supxlabel("Training data fraction", fontsize=14, y=0.05)
    fig.supylabel("Best validation accuracy", fontsize=14, x=0.05)

    # 步骤5：保存大图（无临时文件，无需清理操作）
    os.makedirs(os.path.dirname(merge_save_path), exist_ok=True)
    plt.tight_layout(rect=[0.06, 0.06, 1, 0.95])  # 预留标题/标签空间
    plt.savefig(merge_save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ 网格大图已保存：{merge_save_path}（布局：{n_rows}行 × {n_cols}列，有效图片：{chart_count}张）")

# ===================== Parser配置：适配统一入口脚本 =====================
def add_parser(subparsers):
    """
    为当前工具添加命令行参数（适配统一可视化入口脚本）
    :param subparsers: 统一入口脚本的子解析器
    """
    parser = subparsers.add_parser(
        "merge_charts_to_grid",
        help="无空白填充，灵活拼接多张单图为网格大图",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 必选参数
    parser.add_argument(
        "--single-charts-dir", "-d",
        required=True,
        type=str,
        help="单张图的存放目录（如：my_visualization/single_charts）"
    )
    parser.add_argument(
        "--merge-save-path", "-o",
        required=True,
        type=str,
        help="拼接后大图的保存路径（如：my_visualization/merge_grid_no_blank.png）"
    )
    # 可选参数
    parser.add_argument(
        "--global-title", "-g",
        type=str,
        default="Training Data Fraction vs Best Validation Accuracy",
        help="网格大图的全局标题"
    )
    parser.add_argument(
        "--n-cols",
        type=int,
        default=3,
        help="自定义网格列数（默认：3）"
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=None,
        help="自定义网格行数（优先级低于列数，默认：自动计算）"
    )
    parser.add_argument(
        "--show-empty-subplot",
        action="store_true",
        help="是否显示多余空白子图（默认：不显示，添加此参数则显示）"
    )
    parser.add_argument(
        "--figsize-w",
        type=int,
        default=None,
        help="大图宽度（默认：根据行列数自动计算）"
    )
    parser.add_argument(
        "--figsize-h",
        type=int,
        default=None,
        help="大图高度（默认：根据行列数自动计算）"
    )

# ===================== 主函数：执行拼接逻辑 =====================
def main(args):
    """
    工具主执行函数（适配统一可视化入口脚本）
    :param args: 命令行解析后的参数对象
    """
    # 1. 读取单张图目录下所有有效图片
    single_chart_paths = []
    # 按文件名排序，保证拼接顺序稳定
    for filename in sorted(os.listdir(args.single_charts_dir)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            chart_full_path = os.path.join(args.single_charts_dir, filename)
            single_chart_paths.append(chart_full_path)
    
    # 2. 组装大图尺寸
    fig_size = None
    if args.figsize_w is not None and args.figsize_h is not None:
        fig_size = (args.figsize_w, args.figsize_h)
    
    # 3. 执行拼接
    merge_charts_to_grid(
        single_chart_paths=single_chart_paths,
        merge_save_path=args.merge_save_path,
        figsize=fig_size,
        global_title=args.global_title,
        n_cols=args.n_cols,
        n_rows=args.n_rows,
        show_empty_subplot=args.show_empty_subplot
    )

# ===================== 单独运行入口（可选） =====================
if __name__ == "__main__":
    # 手动构造参数（单独运行时使用）
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers(dest="cmd"))
    args = parser.parse_args()
    main(args)