#!/usr/bin/env python
import os
import torch
import numpy as np
from argparse import ArgumentParser, Namespace
from pathlib import Path
import yaml

# 导入 MLP 和 Transformer 训练类
from grok.mlp_trainer import TrainableMLP
from grok.transformer_trainer import TrainableTransformer

# 导入自定义模块获取常量
from grok.data import VALID_OPERATORS, DEFAULT_DATA_DIR

# ========== 关键修改：从 optimizer.py 导入 CustomAdamW ==========
from grok.optimizer import CustomAdamW

# ==============================================================================
# 统一参数解析（重构：全局参数 + 子命令（mlp/transformer））
# ==============================================================================
def add_args() -> ArgumentParser:
    parser = ArgumentParser(description="统一训练入口（MLP/Transformer + One-Hot/Embedding）")
    
    # -------------------------- 全局公共参数（所有模型共享） --------------------------
    # （原有参数保持不变）
    parser.add_argument("--random_seed", type=int, default=-1, help="随机种子（-1 不固定）")
    parser.add_argument("--gpu", type=int, default=0, help="GPU 卡号（-1 用 CPU）")
    parser.add_argument("--max_steps", type=int, default=100000, help="最大训练步数")
    
    # 数据相关公共参数
    parser.add_argument("--math_operator", type=str, default="+", choices=VALID_OPERATORS, help="算术运算符")
    parser.add_argument("--operand_length", type=int, help="操作数长度（适用于多位数运算）")
    parser.add_argument("--use_mask", action="store_true", default=False, help="按标签拆分数据集")
    parser.add_argument("--train_data_pct", type=float, default=5, help="训练集占比（%）")
    parser.add_argument("--datadir", type=str, default=DEFAULT_DATA_DIR, help="数据集目录")
    
    # 训练相关公共参数
    parser.add_argument("--batchsize", type=float, default=0, help="批次大小配置（-1=全量，0=自动计算，0<N<1=比例，N>1=固定值）")
    parser.add_argument("--max_context_len", type=int, default=50, help="最大序列长度")
    parser.add_argument("--dropout", type=float, default=0.0, help=" dropout 概率")
    parser.add_argument("--warmup_steps", type=int, default=10, help="学习率预热步数")
    parser.add_argument("--anneal_lr_steps", type=int, default=100000, help="学习率退火步数")
    parser.add_argument("--anneal_lr", action="store_true", default=False, help="是否启用学习率退火")
    parser.add_argument("--max_lr", type=float, default=1e-3, help="最大学习率")
    parser.add_argument("--weight_decay", type=float, default=0, help="权重衰减系数")
    parser.add_argument("--weight_decay_kind", type=str, default="to_zero", help="权重衰减类型")
    parser.add_argument("--noise_factor", type=float, default=0, help="梯度噪声系数")
    parser.add_argument("--logdir", type=str, default="logs", help="日志和检查点目录")
    
    # ========== 优化器类型选择（默认adamw，支持自定义优化器） ==========
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "custom_sgd", "custom_rmsprop", "custom_momentum"],
        help="选择训练使用的优化器（默认：adamw，支持：custom_sgd、custom_rmsprop、custom_momentum）"
    )

    # ========== 各自定义优化器的可调参数 ==========
    # CustomSGD 专属参数
    parser.add_argument(
        "--sgd_momentum",
        type=float,
        default=0.9,
        help="CustomSGD/CustomMomentum的动量系数（默认：0.9，仅对应优化器生效）"
    )
    parser.add_argument(
        "--sgd_nesterov",
        action="store_true",
        default=False,
        help="CustomSGD是否使用Nesterov动量（默认：False，仅custom_sgd生效）"
    )

    # CustomRMSprop 专属参数
    parser.add_argument(
        "--rmsprop_alpha",
        type=float,
        default=0.99,
        help="CustomRMSprop的移动平均衰减系数（默认：0.99，仅custom_rmsprop生效）"
    )
    parser.add_argument(
        "--rmsprop_eps",
        type=float,
        default=1e-8,
        help="CustomRMSprop的数值稳定项（默认：1e-8，仅custom_rmsprop生效）"
    )

    # CustomMomentum 专属参数
    parser.add_argument(
        "--momentum_dampening",
        type=float,
        default=0.0,
        help="CustomMomentum的阻尼系数（默认：0.0，仅custom_momentum生效）"
    )

    # -------------------------- 模型子命令（原有逻辑不变） --------------------------
    subparsers = parser.add_subparsers(
        dest="model_type",
        required=True,
        help="模型类型选择（mlp / transformer）"
    )
    mlp_parser = subparsers.add_parser("mlp", help="MLP 模型训练（支持 onehot/embedding 编码）")
    mlp_parser = TrainableMLP.add_model_specific_args(mlp_parser)
    transformer_parser = subparsers.add_parser("transformer", help="Transformer 模型训练（固定 embedding 编码）")
    transformer_parser = TrainableTransformer.add_model_specific_args(transformer_parser)
    
    return parser

# ==============================================================================
# 其余逻辑（save_hparams、train、主入口）保持不变，无需修改
# ==============================================================================
def save_hparams(hparams: Namespace, save_path: str) -> None:
    """保存超参数"""
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(vars(hparams), f, sort_keys=False)
    print(f"超参数保存至：{save_path}")

def train(hparams: Namespace):
    """训练主函数"""
    # 初始化目录
    hparams.logdir = os.path.abspath(hparams.logdir)
    print(f"实验目录：{hparams.logdir}")
    print(f"模型类型：{hparams.model_type}")
    if hparams.model_type == "mlp":
        print(f"编码方式：{hparams.encoding}")
    else:
        print(f"编码方式：embedding（Transformer 固定）")
    
    # 保存超参数
    save_hparams(hparams, os.path.join(hparams.logdir, "hparams.yaml"))
    
    # 设置随机种子
    if hparams.random_seed != -1:
        torch.manual_seed(hparams.random_seed)
        torch.cuda.manual_seed(hparams.random_seed)
        np.random.seed(hparams.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 初始化模型（根据子命令选择）
    if hparams.model_type == "mlp":
        model = TrainableMLP(hparams)
    elif hparams.model_type == "transformer":
        model = TrainableTransformer(hparams)
    else:
        raise ValueError(f"不支持的模型类型：{hparams.model_type}")
    
    # 训练和测试
    model.fit()
    test_logs = model.test()
    print(f"\n测试完成！")
    print(f"测试损失：{test_logs['test_loss']:.4f}")
    print(f"测试准确率：{test_logs['test_accuracy']:.2f}%")

if __name__ == "__main__":
    parser = add_args()
    args = parser.parse_args()
    
    # 参数校验（忽略模型不相关的专属参数）
    if args.model_type == "transformer":
        ignore_args = ["encoding", "embedding_dim", "mlp_hidden_dims"]
        for arg in ignore_args:
            if hasattr(args, arg) and getattr(args, arg) is not None:
                print(f"警告：Transformer 不使用 --{arg} 参数，已忽略")
    elif args.model_type == "mlp":
        ignore_args = ["n_layers", "n_heads", "d_model", "weight_noise", "non_linearity", "save_activations", "save_outputs"]
        for arg in ignore_args:
            if hasattr(args, arg) and getattr(args, arg) is not None:
                print(f"警告：MLP 不使用 --{arg} 参数，已忽略")
    
    # 启动训练
    train(args)