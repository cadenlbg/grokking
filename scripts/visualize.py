#!/usr/bin/env python
import os
import sys
import argparse

# 关键：将项目根目录加入Python路径，确保能导入visualization模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 导入可视化工具（已封装add_parser）
from visualization import train_vs_val

def main():
    # 主入口Parser
    main_parser = argparse.ArgumentParser(description="统一调用可视化工具（存放于visualization目录）")
    subparsers = main_parser.add_subparsers(
        dest="tool",
        required=True,
        help="可选可视化工具"
    )

    # ========== 直接调用封装好的add_parser，无需手动写参数 ==========
    train_vs_val.add_parser(subparsers)

    # 解析参数
    args = main_parser.parse_args()

    # 根据选择的工具调用对应函数
    if args.tool == "train_vs_val":
        train_vs_val.main(args)

if __name__ == "__main__":
    main()