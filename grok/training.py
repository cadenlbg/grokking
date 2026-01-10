#!/usr/bin/env python
import os
import torch
import numpy as np
from argparse import ArgumentParser, Namespace
from pathlib import Path
import yaml

# 核心训练器导入 - 统一接口 三模型无缝切换
from grok.mlp_trainer import TrainableMLP
from grok.transformer_trainer import TrainableTransformer
from grok.lstm_trainer import TrainableLSTM

# 数据模块常量 - 算子校验 & 默认数据集路径
from grok.data import VALID_OPERATORS, DEFAULT_DATA_DIR

# 自定义优化器库 - 全模型统一优化策略
from grok.optimizer import CustomAdamW

# ============================================= 核心参数解析器 =============================================
def add_args() -> ArgumentParser:
    parser = ArgumentParser(description="加法任务专属训练入口 | MLP/Transformer/LSTM三模型支持 | 二元/K元加法适配 | OneHot/Embedding双编码")

    # ---------------------- 全局公共参数 - 所有模型通用 ----------------------
    parser.add_argument("--random_seed", type=int, default=-1, help="随机种子(-1不固定,≥0固定保证复现)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU卡号(-1强制使用CPU,≥0指定GPU编号)")
    parser.add_argument("--max_steps", type=int, default=100000, help="训练最大步数，优先级高于epoch")

    parser.add_argument("--math_operator", type=str, default="+", choices=VALID_OPERATORS, help="加法算子(+二元加法|+k_mod_97 K元加法)")
    parser.add_argument("--operand_length", type=int, help="操作数长度，加法任务预留兼容参数")
    parser.add_argument("--use_mask", action="store_true", default=False, help="掩码拆分数据集，提升泛化能力")
    parser.add_argument("--train_data_pct", type=float, default=5, help="训练集占总数据集的百分比")
    parser.add_argument("--datadir", type=str, default=DEFAULT_DATA_DIR, help="数据集根目录")
    parser.add_argument("--k", type=int, default=2, help="相加项数(k≥2)，k=2二元加法，k≥3自动切K元加法")
    parser.add_argument('--modulus', type=int, default=97, help="加法运算模数，推荐质数97/101/61/127")

    # ---------------------- 训练公共超参数 - 所有模型共用 ----------------------
    parser.add_argument("--batchsize", type=float, default=0, help="批次配置(-1全量|0自动|0<N<1比例|N>1固定值)")
    parser.add_argument("--max_context_len", type=int, default=50, help="最大序列长度，K元加法建议≥60")
    parser.add_argument("--dropout", type=float, default=0.0, help="随机失活概率，正则化防过拟合")
    parser.add_argument("--warmup_steps", type=int, default=10, help="学习率预热步数，稳定训练初期")
    parser.add_argument("--anneal_lr_steps", type=int, default=100000, help="学习率退火步数")
    parser.add_argument("--anneal_lr", action="store_true", default=False, help="启用余弦退火学习率")
    parser.add_argument("--max_lr", type=float, default=1e-3, help="学习率峰值")
    parser.add_argument("--weight_decay", type=float, default=0, help="权重衰减系数")
    parser.add_argument("--weight_decay_kind", type=str, default="to_zero", help="权重衰减策略类型")
    parser.add_argument("--noise_factor", type=float, default=0, help="梯度噪声注入系数，提升泛化")
    parser.add_argument("--logdir", type=str, default="logs", help="日志/检查点/超参文件保存目录")

    # ---------------------- 优化器选择及专属参数 ----------------------
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "custom_sgd", "custom_rmsprop", "custom_momentum"], help="训练优化器选择")
    parser.add_argument("--sgd_momentum", type=float, default=0.0, help="SGD/动量优化器的动量系数，仅对应优化器生效")
    parser.add_argument("--sgd_nesterov", action="store_true", default=False, help="SGD启用Nesterov动量，仅custom_sgd生效")
    parser.add_argument("--rmsprop_alpha", type=float, default=0.99, help="RMSprop衰减系数，仅custom_rmsprop生效")
    parser.add_argument("--rmsprop_eps", type=float, default=1e-8, help="RMSprop数值稳定项，仅custom_rmsprop生效")
    parser.add_argument("--momentum_dampening", type=float, default=0.0, help="动量阻尼系数，仅custom_momentum生效")

    # ---------------------- 模型子命令 - 独立专属参数，互不干扰 ----------------------
    subparsers = parser.add_subparsers(dest="model_type", required=True, help="模型类型三选一: mlp / transformer / lstm")
    
    mlp_parser = subparsers.add_parser("mlp", help="MLP模型 | 支持OneHot/Embedding双编码")
    mlp_parser = TrainableMLP.add_model_specific_args(mlp_parser)

    transformer_parser = subparsers.add_parser("transformer", help="Transformer模型 | 固定Embedding编码")
    transformer_parser = TrainableTransformer.add_model_specific_args(transformer_parser)

    lstm_parser = subparsers.add_parser("lstm", help="LSTM模型 | 支持OneHot/Embedding双编码")
    lstm_parser = TrainableLSTM.add_model_specific_args(lstm_parser)

    return parser

# ============================================= 工具函数 =============================================
def save_hparams(hparams: Namespace, save_path: str) -> None:
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(vars(hparams), f, sort_keys=False)
    print(f"超参数已保存: {save_path}")

# ============================================= 训练主逻辑 =============================================
def train(hparams: Namespace):
    # K元加法自动切换算子
    if hparams.k >=3 and hparams.math_operator == "+":
        hparams.math_operator = "+k_mod_97"
        print(f"\n检测到k={hparams.k}≥3，自动切换为K元加法算子：+k_mod_97")

    # 初始化实验目录
    hparams.logdir = os.path.abspath(hparams.logdir)
    print(f"\n===== 实验配置 =====")
    print(f"实验目录: {hparams.logdir} | 模型类型: {hparams.model_type}")
    print(f"运算模数: {hparams.modulus} | 任务类型: {hparams.math_operator} k={hparams.k}")
    print(f"编码方式: {hparams.encoding if hparams.model_type in ['mlp','lstm'] else 'embedding(Transformer固定)'}")
    print(f"====================")

    # 保存超参
    save_hparams(hparams, os.path.join(hparams.logdir, "hparams.yaml"))

    # 固定随机种子
    if hparams.random_seed != -1:
        torch.manual_seed(hparams.random_seed)
        torch.cuda.manual_seed(hparams.random_seed)
        np.random.seed(hparams.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 模型实例化
    if hparams.model_type == "mlp":
        model = TrainableMLP(hparams)
    elif hparams.model_type == "transformer":
        model = TrainableTransformer(hparams)
    elif hparams.model_type == "lstm":
        model = TrainableLSTM(hparams)
    else:
        raise ValueError(f"不支持的模型类型: {hparams.model_type}，仅支持mlp/transformer/lstm")

    # 训练+测试全流程
    model.fit()
    test_logs = model.test()

    # 打印测试结果
    print(f"\n===== 训练完成 · 测试指标 =====")
    print(f"测试损失: {test_logs['test_loss']:.4f}")
    print(f"测试准确率: {test_logs['test_accuracy']:.2f}%")
    print(f"指标日志路径: {hparams.logdir}/metrics/metric.csv")

# ============================================= 程序主入口 =============================================
if __name__ == "__main__":
    parser = add_args()
    args = parser.parse_args()

    # 无关参数校验 - 仅警告 不删除 避免核心参数丢失
    if args.model_type == "transformer":
        ignore_args = ["encoding", "embedding_dim", "mlp_hidden_dims"]
        for arg in ignore_args:
            if hasattr(args, arg) and getattr(args, arg) is not None:
                print(f"警告:Transformer不使用--{arg} 参数，已忽略")
    elif args.model_type == "mlp":
        ignore_args = ["n_layers", "n_heads", "d_model", "weight_noise", "non_linearity", "save_activations", "save_outputs"]
        for arg in ignore_args:
            if hasattr(args, arg) and getattr(args, arg) is not None:
                print(f"警告:MLP不使用--{arg} 参数，已忽略")
    elif args.model_type == "lstm":
        ignore_args = ["n_layers", "n_heads", "d_model", "weight_noise", "non_linearity", "save_activations", "save_outputs"]
        for arg in ignore_args:
            if hasattr(args, arg) and getattr(args, arg) is not None:
                print(f"警告:LSTM不使用--{arg} 参数，已忽略")

    # 启动训练
    train(args)