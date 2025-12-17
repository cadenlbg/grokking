#!/usr/bin/env python
from argparse import ArgumentParser, Namespace
from typing import List, Union, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# 复用原代码中的自定义Linear层（支持weight_noise和float32）
class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        self.weight_noise = kwargs.pop("weight_noise", 0.0)
        super().__init__(*args, **kwargs)
        # 确保权重是float32类型
        self.weight = nn.Parameter(self.weight.float())
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias.float())

    def forward(self, input: Tensor) -> Tensor:
        # 确保输入是float32类型
        input = input.float()
        if self.weight_noise > 0 and self.training:
            bias = self.bias if self.bias is None else self.bias + torch.randn_like(self.bias) * self.weight_noise
            weight = self.weight + torch.randn_like(self.weight) * self.weight_noise
        else:
            bias = self.bias
            weight = self.weight
            
        return F.linear(
            input,
            weight,
            bias,
        )

# 复用原代码中的LayerNorm层（支持weight_noise和float32）
class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        self.weight_noise = kwargs.pop("weight_noise", 0.0)
        super().__init__(*args, **kwargs)
        # 确保权重是float32类型
        if self.weight is not None:
            self.weight = nn.Parameter(self.weight.float())
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias.float())

    def forward(self, input: Tensor) -> Tensor:
        # 确保输入是float32类型
        input = input.float()
        if self.weight_noise > 0 and self.training:
            bias = self.bias if self.bias is None else self.bias + torch.randn_like(self.bias) * self.weight_noise
            weight = self.weight + torch.randn_like(self.weight) * self.weight_noise
        else:
            bias = self.bias
            weight = self.weight
        return F.layer_norm(
            input,
            self.normalized_shape,
            weight,
            bias,
            self.eps,
        )

class MLP(nn.Module):
    """
    灵活配置的多层MLP
    支持：
    - 自定义输入维度、输出维度
    - 自定义各隐藏层维度（通过列表指定）
    - 可选的层归一化
    - weight_noise（与原Transformer代码一致）
    - 多种激活函数
    - Dropout
    """
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
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims) + 1  # 隐藏层数量 + 输出层
        self.weight_noise = weight_noise
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        
        # 激活函数映射（与原Transformer代码一致）
        non_linearities = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU
        }
        if non_linearity not in non_linearities:
            raise ValueError(f"不支持的激活函数: {non_linearity}，可选: {list(non_linearities.keys())}")
        self.activation = non_linearities[non_linearity]()
        
        # 构建网络层
        layers = OrderedDict()
        
        # 输入层 -> 第一层隐藏层
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            # 线性层
            layers[f"linear_{i+1}"] = Linear(
                prev_dim, hidden_dim, bias=bias, weight_noise=weight_noise
            )
            
            # 可选的层归一化
            if use_layer_norm:
                layers[f"layernorm_{i+1}"] = LayerNorm(hidden_dim, weight_noise=weight_noise)
            
            # 激活函数
            layers[f"activation_{i+1}"] = self.activation
            
            # Dropout
            if dropout > 0:
                layers[f"dropout_{i+1}"] = nn.Dropout(p=dropout)
            
            prev_dim = hidden_dim
        
        # 最后一层：隐藏层 -> 输出层
        layers[f"linear_output"] = Linear(
            prev_dim, output_dim, bias=bias, weight_noise=weight_noise
        )
        
        # 构建Sequential网络
        self.model = nn.Sequential(layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self) -> None:
        """初始化网络权重（保持与原代码风格一致）"""
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        Args:
            x: 输入张量，shape = (*, input_dim)
        Returns:
            输出张量，shape = (*, output_dim)
        """
        # 确保输入是float32类型
        x = x.float()
        
        # 前向传播
        return self.model(x)
    
    def get_model_config(self) -> dict:
        """获取模型配置信息"""
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
    """解析命令行参数"""
    parser = ArgumentParser(description="灵活配置的多层MLP（支持自定义层数和各层维数）")
    
    # 核心配置
    parser.add_argument("--input-dim", type=int, required=True, help="输入维度")
    parser.add_argument("--output-dim", type=int, required=True, help="输出维度")
    parser.add_argument("--hidden-dims", type=int, nargs="+", required=True,
                      help="隐藏层维度列表（如：--hidden-dims 512 256 128 表示3层隐藏层，维数分别为512、256、128）")
    
    # 网络配置
    parser.add_argument("--non-linearity", type=str, default="relu",
                      choices=["relu", "gelu", "tanh", "sigmoid", "leaky_relu"],
                      help="激活函数类型")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout概率")
    parser.add_argument("--weight-noise", type=float, default=0.0, help="权重噪声强度（训练时添加）")
    parser.add_argument("--use-layer-norm", action="store_true", help="是否使用层归一化")
    parser.add_argument("--no-bias", action="store_false", dest="bias", help="是否不使用偏置项")
    
    # 设备配置
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="运行设备（cuda/cpu）")
    
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()
    
    # 创建MLP模型
    mlp = FlexibleMLP(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        hidden_dims=args.hidden_dims,
        weight_noise=args.weight_noise,
        dropout=args.dropout,
        non_linearity=args.non_linearity,
        use_layer_norm=args.use_layer_norm,
        bias=args.bias
    ).to(args.device)
    
    # 打印模型配置
    print("=" * 60)
    print("MLP 模型配置")
    print("=" * 60)
    config = mlp.get_model_config()
    for key, value in config.items():
        print(f"{key:<20}: {value}")
    print("=" * 60)
    
    # 打印模型结构
    print("\n模型结构:")
    print(mlp)
    print("=" * 60)
    
    # 测试前向传播
    batch_size = 32
    test_input = torch.randn(batch_size, args.input_dim).to(args.device)
    test_output = mlp(test_input)
    
    print(f"\n前向传播测试:")
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {test_output.shape}")
    print("测试成功！")

if __name__ == "__main__":
    main()