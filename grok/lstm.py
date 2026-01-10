#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List

# 完全相同的Linear类
class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        self.weight_noise = kwargs.pop("weight_noise", 0.0)
        super().__init__(*args, **kwargs)
        self.weight = nn.Parameter(self.weight.float())
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias.float())

    def forward(self, input: Tensor) -> Tensor:
        input = input.float()
        if self.weight_noise > 0 and self.training:
            bias = self.bias if self.bias is None else self.bias + torch.randn_like(self.bias) * self.weight_noise
            weight = self.weight + torch.randn_like(self.weight) * self.weight_noise
        else:
            bias = self.bias
            weight = self.weight
            
        return F.linear(input, weight, bias)

# 完全相同的LayerNorm类
class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        self.weight_noise = kwargs.pop("weight_noise", 0.0)
        super().__init__(*args, **kwargs)
        if self.weight is not None:
            self.weight = nn.Parameter(self.weight.float())
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias.float())

    def forward(self, input: Tensor) -> Tensor:
        input = input.float()
        if self.weight_noise > 0 and self.training:
            bias = self.bias if self.bias is None else self.bias + torch.randn_like(self.bias) * self.weight_noise
            weight = self.weight + torch.randn_like(self.weight) * self.weight_noise
        else:
            bias = self.bias
            weight = self.weight
        return F.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

class LSTM(nn.Module):
    """
    LSTM网络
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: Optional[int] = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        weight_noise: float = 0.0,
        dropout: float = 0.1,
        bidirectional: bool = False,
        use_layer_norm: bool = False,
        bias: bool = True
    ) -> None:
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.weight_noise = weight_noise
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_layer_norm = use_layer_norm
        
        # Embedding层
        if embedding_dim is not None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embedding = None
        
        # LSTM层
        lstm_input_dim = embedding_dim if embedding_dim is not None else vocab_size
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            bias=bias
        )
        
        # 输出层
        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.linear_output = Linear(
            output_dim, vocab_size, bias=bias, weight_noise=weight_noise
        )
        
        # 层归一化
        if use_layer_norm:
            self.layer_norm = LayerNorm(output_dim, weight_noise=weight_noise)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Embedding, Linear)):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len = x.shape
        
        if self.embedding is not None:
            x_embedded = self.embedding(x)
        else:
            x_embedded = F.one_hot(x, num_classes=self.vocab_size).float()
        
        lstm_out, _ = self.lstm(x_embedded)
        
        if self.use_layer_norm:
            lstm_out = self.layer_norm(lstm_out)
        
        output = self.linear_output(lstm_out)
        
        return output
    
    def get_model_config(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "weight_noise": self.weight_noise,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "use_layer_norm": self.use_layer_norm,
            "num_parameters": sum(p.numel() for p in self.parameters())
        }