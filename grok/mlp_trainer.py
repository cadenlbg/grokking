#!/usr/bin/env python
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from argparse import Namespace, ArgumentParser
from tqdm import tqdm
import yaml
from pathlib import Path
import json

# 导入自定义模块
from grok.data import ArithmeticDataset, ArithmeticIterator

DEFAULT_MLP_HIDDEN_DIMS = [512, 256, 128]
DEFAULT_EMBEDDING_DIM = 128

def save_hparams(hparams: Namespace, save_path: str) -> None:
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(vars(hparams), f, sort_keys=False)

class TrainableMLP:
    def __init__(self, hparams: Namespace) -> None:
        self.hparams = hparams
        self.device = torch.device(
            f"cuda:{hparams.gpu}" if torch.cuda.is_available() and hparams.gpu >= 0 else "cpu"
        )
        
        # 数据准备
        self.prepare_data()
        self.vocab_size = len(self.train_dataset.tokenizer)
        # 获取 pad token 的索引（关键：用于序列补齐）
        self.pad_token_idx = self.train_dataset.tokenizer.stoi.get("<PAD>", 0)
        
        # 初始化编码相关
        self._init_encoding()
        
        # 初始化 MLP 模型（现在 input_dim 是 max_context_len * vocab_size，与 pad 后的维度一致）
        self.mlp = MLP(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.hparams.mlp_hidden_dims,
            dropout=self.hparams.dropout,
        ).to(self.device)
        
        # Embedding 层（仅 Embedding 模式）
        if self.hparams.encoding == "embedding":
            self.embedding = nn.Embedding(
                num_embeddings=self.vocab_size,
                embedding_dim=self.hparams.embedding_dim,
                padding_idx=self.pad_token_idx  # Embedding 层也要指定 pad_token
            ).to(self.device)
            # 位置编码
            self.positional_encoding = self._get_positional_encoding(
                max_len=self.hparams.max_context_len,
                d_model=self.hparams.embedding_dim
            ).to(self.device)
        
        # 类型转换
        self.mlp = self.mlp.float()
        for buffer in self.mlp.buffers():
            buffer.data = buffer.data.float()
        
        # 训练相关变量
        self.train_batchsize = 0
        self.batches_per_epoch = 0
        self.current_epoch = 0
        self.global_step = 0
        
        # 日志和检查点目录
        self.logdir = hparams.logdir
        self.checkpoint_path = os.path.join(self.logdir, "checkpoints")
        os.makedirs(self.checkpoint_path, exist_ok=True)
        
        # 日志文件初始化
        self.log_file = os.path.join(self.logdir, "metrics", "metrics.csv")
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self._init_log_file()

    def _init_encoding(self):
        """初始化编码方式和输入/输出维度"""
        if self.hparams.encoding == "onehot":
            self.input_dim = self.hparams.max_context_len * self.vocab_size
            self.output_dim = self.input_dim
        elif self.hparams.encoding == "embedding":
            self.input_dim = self.hparams.max_context_len * self.hparams.embedding_dim
            self.output_dim = self.hparams.max_context_len * self.vocab_size
        else:
            raise ValueError(f"不支持的编码方式：{self.hparams.encoding}")

    def _get_positional_encoding(self, max_len: int, d_model: int) -> Tensor:
        """生成正弦余弦位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser) -> ArgumentParser:
        """添加 MLP 专属参数（只保留独有的，公共参数在 train.py 中）"""
        # MLP 网络结构参数
        parser.add_argument(
            "--mlp_hidden_dims",
            type=int,
            nargs="+",
            default=DEFAULT_MLP_HIDDEN_DIMS,
            help="MLP 隐藏层维度列表（默认：512 256 128）"
        )
        
        # MLP 编码方式参数（独有）
        parser.add_argument(
            "--encoding",
            type=str,
            default="onehot",
            choices=["onehot", "embedding"],
            help="MLP 编码方式（onehot/embedding）"
        )
        parser.add_argument(
            "--embedding_dim",
            type=int,
            default=DEFAULT_EMBEDDING_DIM,
            help="Embedding 编码维度（仅 encoding=embedding 时生效）"
        )
        
        return parser

    def prepare_data(self) -> None:
        """加载数据集"""
        (self.train_dataset, self.val_dataset,) = ArithmeticDataset.splits(
            train_pct=self.hparams.train_data_pct,
            operator=self.hparams.math_operator,
            operand_length=self.hparams.operand_length,
            data_dir=self.hparams.datadir,
            use_mask=self.hparams.use_mask,
        )

    def _pad_sequence(self, x: Tensor) -> Tensor:
        """将序列 pad 到 max_context_len（关键修复）"""
        batch_size, seq_len = x.shape
        max_len = self.hparams.max_context_len
        
        if seq_len < max_len:
            # 需要 pad：在序列末尾添加 pad_token_idx，形状变为 (batch_size, max_len)
            pad_length = max_len - seq_len
            pad_tensor = torch.full((batch_size, pad_length), self.pad_token_idx, device=self.device)
            x_padded = torch.cat([x, pad_tensor], dim=1)
        elif seq_len > max_len:
            # 序列过长：截断到 max_context_len
            x_padded = x[:, :max_len]
        else:
            # 长度正好，无需处理
            x_padded = x
        
        return x_padded

    def _encode_input(self, x: Tensor) -> Tensor:
        """编码输入（One-Hot/Embedding）+ 序列 pad（关键修复）"""
        # 第一步：将序列 pad 到 max_context_len（确保维度一致）
        x_padded = self._pad_sequence(x)
        batch_size, seq_len = x_padded.shape  # 现在 seq_len = max_context_len
        
        if self.hparams.encoding == "onehot":
            # One-Hot 编码：(batch_size, seq_len) → (batch_size, seq_len, vocab_size) → (batch_size, seq_len*vocab_size)
            onehot = F.one_hot(x_padded, num_classes=self.vocab_size).float()
            return onehot.reshape(batch_size, -1)
        elif self.hparams.encoding == "embedding":
            # Embedding 编码：(batch_size, seq_len) → (batch_size, seq_len, embedding_dim) → (batch_size, seq_len*embedding_dim)
            embed = self.embedding(x_padded)
            embed = embed + self.positional_encoding[:, :seq_len, :]
            return embed.reshape(batch_size, -1)

    def train_dataloader(self) -> ArithmeticIterator:
        iterator = ArithmeticIterator(
            self.train_dataset, self.device, batchsize_hint=self.hparams.batchsize
        )
        self.train_batchsize = iterator.batchsize
        self.batches_per_epoch = len(iterator)
        return iterator

    def val_dataloader(self) -> ArithmeticIterator:
        return ArithmeticIterator(self.val_dataset, self.device, batchsize_hint=-1)

    def _scheduler_lr(self, step: int) -> float:
        """学习率调度"""
        max_lr = self.hparams.max_lr
        min_lr = max_lr / 10
        warmup_steps = self.hparams.warmup_steps
        
        if not self.hparams.anneal_lr:
            return (step / max(warmup_steps, 1)) * max_lr if step <= warmup_steps else max_lr
        else:
            if step <= warmup_steps:
                return (step / max(warmup_steps, 1)) * max_lr
            elif step <= warmup_steps + self.hparams.anneal_lr_steps:
                t = (step - warmup_steps) / self.hparams.anneal_lr_steps
                return min_lr + (max_lr - min_lr) * (1 + np.cos(np.pi * t)) / 2
            else:
                return min_lr

    def configure_optimizers(self):
        """配置优化器"""
        params = list(self.mlp.parameters())
        if self.hparams.encoding == "embedding":
            params.extend(list(self.embedding.parameters()))
        
        from grok.training import CustomAdamW  # 导入共用优化器
        optimizer = CustomAdamW(
            params,
            betas=(0.9, 0.98),
            eps=1e-8,
            lr=1,
            weight_decay=self.hparams.weight_decay,
            noise_factor=self.hparams.noise_factor,
            weight_decay_form=self.hparams.weight_decay_kind,
        )
        scheduler = LambdaLR(optimizer, lr_lambda=self._scheduler_lr)
        return optimizer, scheduler

    def _accuracy(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """计算准确率（同时对 y 进行 pad 匹配）"""
        # 对目标序列也进行 pad，确保和预测序列长度一致
        y_padded = self._pad_sequence(y)
        batch_size, seq_len = y_padded.shape
        
        y_hat_seq = y_hat.reshape(batch_size, seq_len, self.vocab_size)
        y_hat_pred = y_hat_seq.argmax(dim=-1)
        
        # 等号右侧准确率
        eq_token_index = self.train_dataset.tokenizer.stoi.get("=", -1)
        if eq_token_index == -1:
            raise ValueError("Tokenizer 中未找到 '=' 符号")
        
        # 找到等号位置（取第一个样本的等号位置，假设批次内所有样本等号位置一致）
        eq_position = torch.nonzero(y_padded[0, :] == eq_token_index, as_tuple=True)[0]
        if len(eq_position) == 0:
            raise ValueError("样本中未找到 '=' 符号")
        eq_position = eq_position[0].item()
        
        y_rhs = y_padded[..., eq_position + 1:]
        y_hat_rhs = y_hat_pred[..., eq_position + 1:]
        
        row_acc = torch.min((y_hat_rhs == y_rhs), dim=-1).values.float() * 100
        return row_acc.mean()

    def _step(self, batch, batch_idx, train=True):
        """单批次前向传播"""
        x = batch["text"]
        y = batch["target"]
        
        self.mlp.train(train)
        if self.hparams.encoding == "embedding":
            self.embedding.train(train)
        
        with torch.set_grad_enabled(train):
            x_encoded = self._encode_input(x)  # 编码前已 pad
            y_hat_flat = self.mlp(x_encoded)
            
            # 处理输出：确保目标序列也 pad 到 max_context_len
            y_padded = self._pad_sequence(y)
            batch_size, seq_len = y_padded.shape
            y_hat_seq = y_hat_flat.reshape(batch_size, seq_len, self.vocab_size).transpose(-2, -1)
            
            # 等号右侧损失
            eq_token_index = self.train_dataset.tokenizer.stoi.get("=", -1)
            if eq_token_index == -1:
                raise ValueError("Tokenizer 中未找到 '=' 符号")
            eq_position = torch.nonzero(y_padded[0, :] == eq_token_index, as_tuple=True)[0]
            if len(eq_position) == 0:
                raise ValueError("样本中未找到 '=' 符号")
            eq_position = eq_position[0].item()
            
            y_rhs = y_padded[..., eq_position + 1:]
            y_hat_rhs = y_hat_seq[..., eq_position + 1:]
            
            loss = F.cross_entropy(y_hat_rhs, y_rhs)
            
            # 准确率
            with torch.no_grad():
                acc = self._accuracy(y_hat_flat, y)
            
            coeff = float(y.shape[0]) / len(self.train_dataset) if train else float(y.shape[0]) / len(self.val_dataset)
        
        return loss, acc, coeff

    def _init_log_file(self):
        """初始化日志文件"""
        if not os.path.exists(self.log_file):
            headers = [
                "epoch", "global_step", "train_loss", "train_accuracy", 
                "train_perplexity", "learning_rate", "val_loss", "val_accuracy",
                "val_perplexity", "model_type", "encoding"
            ]
            with open(self.log_file, "w") as f:
                f.write(",".join(headers) + "\n")

    def _log_metrics(self, metrics_dict):
        """写入日志"""
        with open(self.log_file, "r") as f:
            headers = f.readline().strip().split(",")
        
        values = []
        for h in headers:
            val = metrics_dict.get(h, "NaN")
            if isinstance(val, Tensor):
                val = val.item()
            values.append(str(val))
        
        with open(self.log_file, "a") as f:
            f.write(",".join(values) + "\n")

    def training_epoch(self, train_loader, optimizer, scheduler, global_pbar):
        """训练一个epoch"""
        start_time = time.time()
        total_loss = 0.0
        total_acc = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            loss, acc, coeff = self._step(batch, batch_idx, train=True)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            self.global_step += 1
            
            # 更新进度条
            global_pbar.set_postfix({
                "epoch": self.current_epoch,
                "loss": f"{loss.item():.4f}",
                "encoding": self.hparams.encoding
            })
            global_pbar.update(1)
            
            total_loss += (coeff * loss).item()
            total_acc += (coeff * acc).item()
        
        return {
            "train_loss": torch.tensor(total_loss),
            "train_accuracy": torch.tensor(total_acc),
            "train_perplexity": torch.exp(torch.tensor(total_loss)),
            "learning_rate": scheduler.get_last_lr()[0],
            "time_per_epoch": time.time() - start_time
        }

    def validation_epoch(self, val_loader):
        """验证一个epoch"""
        self.mlp.eval()
        if self.hparams.encoding == "embedding":
            self.embedding.eval()
        
        total_loss = 0.0
        total_acc = 0.0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                loss, acc, coeff = self._step(batch, batch_idx, train=False)
                total_loss += (coeff * loss).item()
                total_acc += (coeff * acc).item()
        
        # 保存检查点
        if self.current_epoch > 0 and (self.current_epoch & (self.current_epoch - 1)) == 0:  # 2的幂次epoch
            checkpoint = {
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "mlp_state_dict": self.mlp.state_dict(),
                "embedding_state_dict": self.embedding.state_dict() if self.hparams.encoding == "embedding" else None,
                "hparams": vars(self.hparams)
            }
            torch.save(checkpoint, os.path.join(self.checkpoint_path, f"epoch_{self.current_epoch}.ckpt"))
        
        return {
            "val_loss": torch.tensor(total_loss),
            "val_accuracy": torch.tensor(total_acc),
            "val_perplexity": torch.exp(torch.tensor(total_loss))
        }

    def fit(self):
        """训练主循环"""
        train_loader = self.train_dataloader()
        val_loader = self.val_dataloader()
        optimizer, scheduler = self.configure_optimizers()
        
        # 初始检查点
        init_ckpt = {
            "epoch": 0,
            "global_step": 0,
            "mlp_state_dict": self.mlp.state_dict(),
            "embedding_state_dict": self.embedding.state_dict() if self.hparams.encoding == "embedding" else None,
            "hparams": vars(self.hparams)
        }
        torch.save(init_ckpt, os.path.join(self.checkpoint_path, "init.pt"))
        
        # 进度条
        global_pbar = tqdm(total=self.hparams.max_steps, desc=f"MLP ({self.hparams.encoding})")
        
        try:
            while self.global_step < self.hparams.max_steps:
                train_logs = self.training_epoch(train_loader, optimizer, scheduler, global_pbar)
                val_logs = self.validation_epoch(val_loader)
                
                # 合并日志
                all_logs = {
                    "epoch": self.current_epoch,
                    "global_step": self.global_step,
                    "model_type": "mlp",
                    "encoding": self.hparams.encoding,
                    **train_logs,
                    **val_logs
                }
                self._log_metrics(all_logs)
                self.current_epoch += 1
        finally:
            global_pbar.close()

    def test(self):
        """测试"""
        test_loader = self.val_dataloader()
        self.mlp.eval()
        if self.hparams.encoding == "embedding":
            self.embedding.eval()
        
        all_losses = []
        all_accs = []
        
        with torch.no_grad():
            for batch in test_loader:
                loss, acc, _ = self._step(batch, 0, train=False)
                all_losses.append(loss)
                all_accs.append(acc)
        
        loss = torch.cat(all_losses).mean()
        acc = torch.cat(all_accs).mean()
        perplexity = torch.exp(loss)
        
        # 保存测试日志
        test_log = {
            "model_type": "mlp",
            "encoding": self.hparams.encoding,
            "test_loss": loss.item(),
            "test_accuracy": acc.item(),
            "test_perplexity": perplexity.item()
        }
        with open(os.path.join(self.logdir, "test_metrics.json"), "w") as f:
            json.dump(test_log, f, indent=2)
        
        return test_log

# 导入 MLP 模型（避免循环导入）
from grok.mlp import MLP