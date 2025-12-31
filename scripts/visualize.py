#!/usr/bin/env python
"""
统一可视化工具入口脚本
支持的可视化功能：
1. train_vs_val：训练/验证损失/准确率趋势图（原有功能）
2. train_fraction_vs_val_acc：训练数据占比 vs 最佳验证准确率 单张图（新增功能）
3. merge_charts_to_grid：无空白填充，灵活拼接多张单图为网格大图（新增功能）
"""
import os
import sys
import argparse

# ========== 关键：修正PROJECT_ROOT（避免路径错误，适配脚本在项目根目录的场景） ==========
# 若脚本在项目根目录，直接取当前目录；若在子目录，可调整..的层级
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ========== 导入可视化工具模块（保留train_vs_val，新增另外两个工具） ==========
try:
    from visualization import train_vs_val
    from visualization import train_fraction_vs_val_acc
    from visualization import merge_charts_to_grid
except ImportError as e:
    print(f"错误：导入可视化模块失败！{e}")
    print(f"请确保 {PROJECT_ROOT}/visualization 目录下存在对应工具脚本")
    sys.exit(1)

def main():
    # 1. 创建主解析器
    main_parser = argparse.ArgumentParser(
        description="Grok项目统一可视化工具入口",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 2. 创建子解析器（用于区分不同可视化工具）
    subparsers = main_parser.add_subparsers(
        dest="visualization_tool",  # 存储用户选择的工具名称
        required=True,  # 强制用户指定可视化工具
        title="可选可视化工具",
        description="请选择要使用的可视化功能，每个功能有独立的参数说明"
    )

    # 3. 为每个可视化工具添加子命令（调用对应工具的add_parser方法，保留原有train_vs_val）
    # 3.1 保留：添加 train_vs_val 工具的子命令
    train_vs_val.add_parser(subparsers)

    # 3.2 新增：添加 train_fraction_vs_val_acc（单张图）工具的子命令
    train_fraction_vs_val_acc.add_parser(subparsers)

    # 3.3 新增：添加 merge_charts_to_grid（无空白拼接大图）工具的子命令
    merge_charts_to_grid.add_parser(subparsers)

    # 4. 解析命令行参数
    args = main_parser.parse_args()

    # 5. 根据用户选择的工具，调用对应主函数（保留train_vs_val，新增另外两个分支）
    if args.visualization_tool == "train_vs_val":
        # 保留：调用训练/验证趋势图工具
        train_vs_val.main(args)
    elif args.visualization_tool == "train_fraction_vs_val_acc":
        # 新增：调用训练数据占比 vs 最佳验证准确率 单张图工具
        train_fraction_vs_val_acc.main(args)
    elif args.visualization_tool == "merge_charts_to_grid":
        # 新增：调用无空白填充的网格大图拼接工具
        merge_charts_to_grid.main(args)
    else:
        # 兜底提示（理论上不会触发，因为subparsers.required=True）
        print(f"错误：不支持的可视化工具 {args.visualization_tool}")
        main_parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()