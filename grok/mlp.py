#!/usr/bin/env python
from argparse import ArgumentParser, Namespace
from typing import List, Union, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# 自定义线性层 继承原生nn.Linear | 核心特性：强制float32精度 + 训练阶段可添加权重/偏置噪声
class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        self.weight_noise = kwargs.pop("weight_noise", 0.0)
        super().__init__(*args, **kwargs)
        # 强制权重为float32 统一精度避免训练异常
        self.weight = nn.Parameter(self.weight.float())
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias.float())

    def forward(self, input: Tensor) -> Tensor:
        # 输入张量强制转float32 与权重精度对齐
        input = input.float()
        # 训练阶段且噪声系数>0 才给权重/偏置添加高斯噪声 提升泛化性
        if self.weight_noise > 0 and self.training:
            bias = self.bias + torch.randn_like(self.bias) * self.weight_noise if self.bias is not None else None
            weight = self.weight + torch.randn_like(self.weight) * self.weight_noise
        else:
            bias = self.bias
            weight = self.weight
        # 原生线性层计算逻辑
        return F.linear(input, weight, bias)

# 自定义层归一化 继承原生nn.LayerNorm | 核心特性：强制float32精度 + 训练阶段可添加权重/偏置噪声
# 噪声逻辑与自定义Linear层完全一致 保证网络噪声策略统一
class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        self.weight_noise = kwargs.pop("weight_noise", 0.0)
        super().__init__(*args, **kwargs)
        # 强制参数为float32 统一精度标准
        if self.weight is not None:
            self.weight = nn.Parameter(self.weight.float())
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias.float())

    def forward(self, input: Tensor) -> Tensor:
        # 输入张量强制转float32 与参数精度对齐
        input = input.float()
        # 训练态+噪声系数>0 执行噪声注入逻辑
        if self.weight_noise > 0 and self.training:
            bias = self.bias + torch.randn_like(self.bias) * self.weight_noise if self.bias is not None else None
            weight = self.weight + torch.randn_like(self.weight) * self.weight_noise
        else:
            bias = self.bias
            weight = self.weight
        # 原生层归一化计算逻辑
        return F.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

# 核心网络：可高度自定义配置的多层感知机(MLP)
# 完全复用上述自定义Linear/LayerNorm层 与Transformer模块的噪声/精度策略完全兼容
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        weight_noise: float = 0.0,
        dropout: float = 0.1,
        non_linearity: str = "relu",
        use_layer_norm: bool = False,
        bias: bool = True
    ) -> None:
        super().__init__()
        # 赋值基础网络配置参数
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims) + 1
        self.weight_noise = weight_noise
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm

        # 激活函数映射表 与原生Transformer网络激活函数选型保持一致
        non_linearities = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU
        }
        if non_linearity not in non_linearities:
            raise ValueError(f"激活函数仅支持: {list(non_linearities.keys())}")
        self.activation = non_linearities[non_linearity]()

        # 有序字典构建网络层 保证层的执行顺序严格可控
        layers = OrderedDict()
        prev_dim = input_dim

        # 循环构建：输入层 -> 所有隐藏层 每一层由「线性层+可选LN+激活+可选Dropout」组成
        for i, hidden_dim in enumerate(hidden_dims):
            layers[f"linear_{i+1}"] = Linear(prev_dim, hidden_dim, bias=bias, weight_noise=weight_noise)
            if use_layer_norm:
                layers[f"layernorm_{i+1}"] = LayerNorm(hidden_dim, weight_noise=weight_noise)
            layers[f"activation_{i+1}"] = self.activation
            if dropout > 0:
                layers[f"dropout_{i+1}"] = nn.Dropout(p=dropout)
            prev_dim = hidden_dim

        # 构建输出层：最后一层隐藏层 -> 输出层 无激活/归一化/Dropout
        layers[f"linear_output"] = Linear(prev_dim, output_dim, bias=bias, weight_noise=weight_noise)

        # 封装有序层为Sequential网络
        self.model = nn.Sequential(layers)

        # 执行权重初始化
        self._init_weights()
    
    def _init_weights(self) -> None:
        # 网络权重初始化策略：线性层使用Xavier均匀初始化 偏置项置零 保证训练收敛性
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        # 前向传播核心逻辑 | 输入shape: (*, input_dim) 输出shape: (*, output_dim)
        x = x.float()
        return self.model(x)
    
    def get_model_config(self) -> dict:
        # 返回模型完整配置信息字典 包含维度/超参/参数量 便于日志打印与配置溯源
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dims": self.hidden_dims,
            "num_layers": self.num_layers,
            "weight_noise": self.weight_noise,
            "dropout": self.dropout,
            "non_linearity": self.activation.__class__.__name__.lower(),
            "use_layer_norm": self.use_layer_norm,
            "num_parameters": sum(p.numel() for p in self.parameters())
        }

def parse_args() -> Namespace:
    # 命令行参数解析函数 | 分模块配置参数 层级清晰 包含完整的参数校验与默认值
    parser = ArgumentParser(description="灵活配置的多层感知机(MLP) - 适配算术任务训练")
    
    # 核心必选配置 - 网络维度相关
    parser.add_argument("--input-dim", type=int, required=True, help="模型输入特征维度")
    parser.add_argument("--output-dim", type=int, required=True, help="模型输出特征维度")
    parser.add_argument("--hidden-dims", type=int, nargs="+", required=True,
                      help="隐藏层维度列表，空格分隔，如 512 256 128 代表3层隐藏层")
    
    # 可选网络配置 - 训练策略与结构相关
    parser.add_argument("--non-linearity", type=str, default="relu",
                      choices=["relu", "gelu", "tanh", "sigmoid", "leaky_relu"],
                      help="网络激活函数类型")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout正则化概率，0则关闭")
    parser.add_argument("--weight-noise", type=float, default=0.0, help="训练时权重噪声强度，0则关闭")
    parser.add_argument("--use-layer-norm", action="store_true", help="是否在每层添加层归一化")
    parser.add_argument("--no-bias", action="store_false", dest="bias", help="是否禁用所有线性层的偏置项")
    
    # 硬件配置
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="模型运行设备，自动优先使用cuda")
    
    return parser.parse_args()