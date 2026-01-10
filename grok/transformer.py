#!/usr/bin/env python
from argparse import ArgumentParser, Namespace
from typing import Tuple, List, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import cos, sin, sqrt
from torch import tensor, Tensor
from torch.optim.lr_scheduler import LambdaLR

# 自定义线性层 继承PyTorch原生nn.Linear
# 核心扩展特性：训练阶段可注入权重/偏置高斯噪声 + 全流程强制float32精度 防止精度不匹配
class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        self.weight_noise = kwargs.pop("weight_noise", 0.0)
        super().__init__(*args, **kwargs)
        # 强制权重参数为float32 统一网络精度标准
        self.weight = nn.Parameter(self.weight.float())
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias.float())

    def forward(self, input: Tensor) -> Tensor:
        # 输入张量强制转float32 与权重精度对齐
        input = input.float()
        # 仅训练模式+噪声系数>0 才注入高斯噪声 提升模型泛化性 推理阶段无噪声
        if self.weight_noise > 0 and self.training:
            bias = self.bias + torch.randn_like(self.bias) * self.weight_noise if self.bias is not None else None
            weight = self.weight + torch.randn_like(self.weight) * self.weight_noise
        else:
            bias = self.bias
            weight = self.weight
            
        return F.linear(input, weight, bias)

# 自定义层归一化 继承PyTorch原生nn.LayerNorm
# 核心扩展特性：与自定义Linear层完全一致的噪声注入策略 + 强制float32精度 保证噪声策略全局统一
class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        self.weight_noise = kwargs.pop("weight_noise", 0.0)
        super().__init__(*args, **kwargs)
        # 强制层归一化参数为float32 统一精度
        if self.weight is not None:
            self.weight = nn.Parameter(self.weight.float())
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias.float())

    def forward(self, input: Tensor) -> Tensor:
        input = input.float()
        # 训练态噪声注入逻辑 与Linear层完全一致
        if self.weight_noise > 0 and self.training:
            bias = self.bias + torch.randn_like(self.bias) * self.weight_noise if self.bias is not None else None
            weight = self.weight + torch.randn_like(self.weight) * self.weight_noise
        else:
            bias = self.bias
            weight = self.weight
        return F.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

# 自定义词嵌入层 继承PyTorch原生nn.Embedding
# 核心扩展特性：训练阶段词嵌入权重高斯噪声注入 + 强制float32精度 适配整体网络精度标准
class Embedding(nn.Embedding):
    def __init__(self, *args, **kwargs):
        self.weight_noise = kwargs.pop("weight_noise", 0.0)
        super().__init__(*args, **kwargs)
        # 强制词嵌入权重为float32
        self.weight = nn.Parameter(self.weight.float())

    def forward(self, input: Tensor) -> Tensor:
        # 训练态噪声注入 仅对词嵌入权重生效
        if self.weight_noise > 0 and self.training:
            weight = self.weight + torch.randn_like(self.weight) * self.weight_noise
        else:
            weight = self.weight
        return F.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse
        )

# Transformer核心组件：单头自注意力层
# 实现标准缩放点积注意力 负责将输入特征投影为Q/K/V 计算注意力权重并加权求和
class AttentionHead(nn.Module):
    def __init__(self, d_model: int, d_key: int, weight_noise: float = 0.0) -> None:
        super().__init__()
        self.d_key = d_key
        # Q/K/V 三个特征投影矩阵 无偏置项 统一注入权重噪声
        self.Wq = Linear(d_model, d_key, bias=False, weight_noise=weight_noise)
        self.Wk = Linear(d_model, d_key, bias=False, weight_noise=weight_noise)
        self.Wv = Linear(d_model, d_key, bias=False, weight_noise=weight_noise)
        # 注意力权重归一化用Softmax
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Union[Tensor, None] = None,
        save_activations: bool = False,
    ) -> Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:
        # 统一强制float32精度
        queries = queries.float()
        keys = keys.float()
        values = values.float()
        
        # 特征投影：将输入维度映射到指定的key维度
        queries = self.Wq(queries)
        keys = self.Wk(keys)
        values = self.Wv(values)

        # 缩放点积注意力核心计算：Q @ K^T / sqrt(d_k) 缓解维度膨胀导致的梯度问题
        attn = torch.matmul(queries, torch.transpose(keys, -2, -1))
        attn = attn / sqrt(self.d_key)

        # 掩码处理：对未来位置的注意力权重置负无穷 Softmax后权重为0 实现因果掩码
        if mask is not None:
            mask = mask.float()
            attn.masked_fill_(mask == 0, float("-inf"))

        # 注意力权重归一化 得到合法的概率分布
        attn = self.softmax(attn)

        # 加权求和：用注意力权重对V做加权 得到最终注意力输出
        result: Tensor = torch.matmul(attn, values)
        
        # 可选保存注意力权重和V值 用于后续可视化/分析 不参与梯度计算
        if save_activations:
            leaf_attn = attn.clone().detach()
            leaf_values = values.clone().detach()
        else:
            leaf_attn = None
            leaf_values = None

        return result, leaf_attn, leaf_values

# Transformer核心组件：多头自注意力层
# 拼接多个单头注意力 特征维度拼接后做一次线性投影 实现特征的多维度并行交互
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, weight_noise: float = 0.0) -> None:
        super().__init__()
        self.d_key = int(d_model / heads)
        # 创建指定数量的单头注意力层 按头均分特征维度
        attn_heads = [AttentionHead(d_model, self.d_key, weight_noise=weight_noise) for _ in range(heads)]
        self.attn_heads = nn.ModuleList(attn_heads)
        # 多头输出特征拼接后的投影矩阵 还原原特征维度
        self.Wo = Linear(d_model, d_model, bias=False, weight_noise=weight_noise)

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Tensor = None,
        save_activations=False,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        # 统一强制float32精度
        queries = queries.float()
        keys = keys.float()
        values = values.float()
        
        # 并行执行所有单头注意力计算
        head_outputs = [
            h(queries=queries, keys=keys, values=values, mask=mask, save_activations=save_activations)
            for h in self.attn_heads
        ]
        # 提取所有单头的输出特征
        head_results = [output[0] for output in head_outputs]

        # 可选收集所有头的注意力权重和V值
        if save_activations:
            layer_attns = list([output[1] for output in head_outputs])
            layer_values = list([output[2] for output in head_outputs])
        else:
            layer_attns = []
            layer_values = []

        # 拼接所有单头特征 + 投影还原维度 完成多头注意力计算
        multihead_result = torch.cat(head_results, dim=-1)
        multihead_result = self.Wo(multihead_result)
        return multihead_result, layer_attns, layer_values

# Transformer核心组件：前馈神经网络FFN
# 标准两层线性结构 + 激活函数 实现特征的非线性变换 维度默认扩张4倍再还原
class FFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        multiplier: int = 4,
        non_linearity: str = "relu",
        weight_noise: float = 0.0,
    ) -> None:
        super().__init__()
        d_ff = int(multiplier * d_model)
        # 激活函数映射 支持ReLU/GELU两种主流选择 适配不同训练需求
        non_linearities = {"relu": nn.ReLU, "gelu": nn.GELU}
        # 构建FFN序列结构：升维 -> 激活 -> 降维
        self.ffn = nn.Sequential(
            Linear(d_model, d_ff, bias=False, weight_noise=weight_noise),
            non_linearities[non_linearity](),
            Linear(d_ff, d_model, bias=False, weight_noise=weight_noise),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        return self.ffn(x)

# Transformer核心组件：单层解码器块
# 标准Decoder结构 = 掩码自注意力 + 残差连接+层归一化 + FFN + 残差连接+层归一化
# 采用预归一化架构 归一化在残差相加前 提升训练稳定性与收敛速度
class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        dropout: float,
        non_linearity: str = "relu",
        weight_noise: float = 0.0,
    ) -> None:
        super().__init__()
        # 掩码自注意力层 防止前瞻信息泄露
        self.self_attn = MultiHeadAttention(d_model, heads, weight_noise=weight_noise)
        self.self_attn_norm = LayerNorm(d_model, weight_noise=weight_noise)

        # 前馈网络 + Dropout正则化 + 层归一化
        self.ffn = FFN(d_model, non_linearity=non_linearity, weight_noise=weight_noise)
        self.ffn_drop = nn.Dropout(p=dropout)
        self.ffn_norm = LayerNorm(d_model, weight_noise=weight_noise)

    def forward(
        self,
        x: Tensor,
        self_attn_mask: Tensor = None,
        save_activations: bool = False,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        x = x.float()
        # 掩码自注意力 + 残差连接 + 层归一化
        a1, layer_attns, layer_values = self.self_attn(x, x, x, self_attn_mask, save_activations)
        a1 = self.self_attn_norm(x + a1)

        # FFN + Dropout + 残差连接 + 层归一化
        a2 = self.ffn(a1)
        a2 = self.ffn_drop(a2)
        a2 = self.ffn_norm(a1 + a2)

        return a2, layer_attns, layer_values

# Transformer解码器：堆叠多个解码器块
# 按顺序执行多层特征变换 逐层提取高阶语义信息 收集各层注意力权重用于分析
class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        num_blocks: int,
        dropout: float,
        non_linearity: str = "relu",
        weight_noise: float = 0.0,
    ) -> None:
        super().__init__()
        # 堆叠指定数量的解码器块 形成深度Transformer结构
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(d_model, heads, dropout, non_linearity, weight_noise=weight_noise)
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        x: Tensor,
        self_attn_mask: Tensor = None,
        save_activations=False,
    ) -> Tuple[Tensor, List[List[Tensor]], List[List[Tensor]]]:
        x = x.float()
        a = x
        attentions = []
        values = []
        # 逐层执行解码器块计算
        for block in self.blocks:
            a, layer_attentions, layer_values = block(a, self_attn_mask, save_activations=save_activations)
            if save_activations:
                attentions.append(layer_attentions)
                values.append(layer_values)
        return a, attentions, values

# 完整的因果Transformer解码器模型（核心主类）
# 端到端封装：词嵌入+位置编码+因果掩码+多层解码器+输出投影层
# 适配序列建模任务 纯解码器架构 无编码器 完美适配算术等式生成/预测任务
class Transformer(nn.Module):
    def __init__(
        self,
        n_layers: int = 4,
        n_heads: int = 4,
        d_model: int = 256,
        dropout: float = 0.1,
        max_context_len: int = 1024,
        vocab_len: int = 2000,
        non_linearity: str = "relu",
        weight_noise: float = 0.0,
    ) -> None:
        super().__init__()
        # 模型核心超参赋值
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = dropout
        self.max_context_len = max_context_len
        self.non_linearity = non_linearity
        self.vocab_len = vocab_len

        # 词嵌入层：将token索引映射为稠密特征向量
        self.embedding = Embedding(vocab_len, d_model, weight_noise=weight_noise)
        
        # 注册非训练参数：位置编码+因果掩码 不参与梯度更新 节省显存
        self.register_buffer("position_encoding", self._position_encoding(max_context_len, d_model).float())
        self.register_buffer("self_attn_mask", self.make_mask(max_context_len).float())

        # 解码器主体
        self.decoder = Decoder(
            d_model, n_heads, n_layers, dropout, self.non_linearity, weight_noise=weight_noise
        )

        # 输出投影层：将模型特征维度映射到词汇表大小 做token预测
        self.linear = Linear(d_model, vocab_len, bias=False, weight_noise=weight_noise)

    @staticmethod
    def make_mask(context_len: int) -> Tensor:
        # 生成下三角因果掩码 对角线及左下为1 右上为0 防止模型看到未来token
        return torch.ones([context_len, context_len]).tril()

    @classmethod
    def _position_encoding(cls, context_len: int, d_model: int) -> Tensor:
        # 标准正弦余弦位置编码 为序列注入位置信息 无训练参数 固定计算逻辑
        rows = [
            torch.tensor(
                [sin(pos / (10000 ** (i / d_model))) if i % 2 == 0 else cos(pos / (10000 ** ((i - 1) / d_model)))
                 for i in range(d_model)], dtype=torch.float32
            )
            for pos in range(context_len)
        ]
        stack = torch.stack(rows, dim=1)
        return stack.T

    def embed(self, indices: Tensor) -> Tensor:
        # 词嵌入+位置编码融合：token特征 + 位置特征 得到最终输入特征
        indices = indices.long()
        context_len = indices.shape[-1]
        pe = self.position_encoding[:context_len, :]
        embedded = self.embedding(indices)
        return pe + embedded

    def forward(
        self,
        x: Tensor,
        pos: int = None,
        save_activations: bool = False,
    ) -> Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:
        # 核心前向传播逻辑：输入token索引 -> 嵌入编码 -> 解码特征 -> 输出预测
        # pos参数可选：指定返回某一位置的预测值 适配等式结果单位置预测需求
        
        # 确保输入张量与模型权重在同一设备
        x = x.to(self.embedding.weight.device)

        # 生成适配当前序列长度的因果掩码
        this_max_context_len = x.shape[-1]
        self_attn_mask = self.self_attn_mask[:this_max_context_len, :this_max_context_len]

        # 嵌入编码 + 解码器特征提取
        x = self.embed(x)
        decoded, attentions, values = self.decoder(x, self_attn_mask, save_activations=save_activations)

        # 可选截取指定位置的特征 只预测目标位置token 减少计算量
        if pos is not None:
            decoded = decoded[:, pos, :]

        # 输出投影到词汇表维度 得到token预测概率
        y_hat = self.linear(decoded)
        return y_hat, attentions, values