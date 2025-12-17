#!/usr/bin/env python
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from argparse import Namespace, ArgumentParser
from tqdm import tqdm
import yaml
import pickle
from pathlib import Path
import json

# 导入自定义模块
from grok.data import ArithmeticDataset, ArithmeticIterator

class TrainableTransformer:
    def __init__(self, hparams: Namespace) -> None:
        self.hparams = hparams
        self.device = torch.device(
            f"cuda:{hparams.gpu}" if torch.cuda.is_available() and hparams.gpu >= 0 else "cpu"
        )
        
        # 数据准备
        self.prepare_data()
        
        # 初始化 Transformer
        self.transformer = Transformer(
            hparams.n_layers,
            hparams.n_heads,
            hparams.d_model,
            hparams.dropout,
            hparams.max_context_len,
            len(self.train_dataset.tokenizer),
            hparams.non_linearity,
            weight_noise=self.hparams.weight_noise,
        ).to(self.device)
        
        self.transformer = self.transformer.float()
        for buffer in self.transformer.buffers():
            buffer.data = buffer.data.float()
        
        # 训练相关变量
        self.train_batchsize = 0
        self.batches_per_epoch = 0
        self.current_epoch = 0
        self.global_step = 0
        self.next_epoch_to_eval = -1
        self.next_train_epoch_to_log = 0
        
        # 日志和检查点目录
        self.logdir = hparams.logdir
        self.checkpoint_path = os.path.join(self.logdir, "checkpoints")
        os.makedirs(self.checkpoint_path, exist_ok=True)
        
        # 日志文件初始化
        self.log_file = os.path.join(self.logdir, "metrics", "metrics.csv")
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self._init_log_file()

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser) -> ArgumentParser:
        """添加 Transformer 专属参数（只保留独有的，公共参数在 train.py 中）"""
        # Transformer 网络结构参数（独有）
        parser.add_argument("--n_layers", type=int, default=2, help="Transformer 层数")
        parser.add_argument("--n_heads", type=int, default=4, help="多头注意力头数")
        parser.add_argument("--d_model", type=int, default=128, help="Transformer 模型维度")
        parser.add_argument("--weight_noise", type=float, default=0.0, help="权重噪声")
        parser.add_argument("--non_linearity", type=str, default="relu", help="激活函数类型")
        
        # Transformer 独有功能参数
        parser.add_argument("--save_activations", action="store_true", default=False, help="是否保存激活值")
        parser.add_argument("--save_outputs", action="store_true", default=False, help="是否保存输出")
        
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
        from grok.training import CustomAdamW  # 导入共用优化器
        optimizer = CustomAdamW(
            self.transformer.parameters(),
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
        """计算准确率"""
        y_hat = torch.max(y_hat, dim=-2).indices
        row_acc = torch.min((y_hat == y), dim=-1).values.float() * 100
        return row_acc.mean()

    def _step(self, batch, batch_idx, train=True):
        """单批次前向传播"""
        x = batch["text"]
        y = batch["target"]
        
        self.transformer.train(train)
        with torch.set_grad_enabled(train):
            y_hat, attentions, values = self.transformer(x=x, save_activations=self.hparams.save_activations)
            y_hat = y_hat.transpose(-2, -1)
            
            # 等号右侧处理
            eq_token_index = self.train_dataset.tokenizer.stoi["="]
            eq_position = int(torch.nonzero(y[0, :] == eq_token_index).squeeze())
            y_rhs = y[..., eq_position + 1:]
            y_hat_rhs = y_hat[..., eq_position + 1:]
            
            coeff = float(y.shape[0]) / len(self.train_dataset) if train else float(y.shape[0]) / len(self.val_dataset)
            loss = F.cross_entropy(y_hat_rhs, y_rhs)
            
            with torch.no_grad():
                acc = self._accuracy(y_hat_rhs, y_rhs)
        
        return loss, acc, coeff, attentions, values

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
        
        epoch_to_log = self.current_epoch == self.next_train_epoch_to_log
        
        for batch_idx, batch in enumerate(train_loader):
            loss, acc, coeff, _, _ = self._step(batch, batch_idx, train=True)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            self.global_step += 1
            
            # 进度条
            global_pbar.set_postfix({
                "epoch": self.current_epoch,
                "loss": f"{loss.item():.4f}",
                "model_type": "transformer"
            })
            global_pbar.update(1)
            
            if epoch_to_log:
                total_loss += (coeff * loss).item()
                total_acc += (coeff * acc).item()
        
        if epoch_to_log:
            self.next_train_epoch_to_log = max(int(1.01 * self.next_train_epoch_to_log), self.next_train_epoch_to_log + 1)
            return {
                "train_loss": torch.tensor(total_loss),
                "train_accuracy": torch.tensor(total_acc),
                "train_perplexity": torch.exp(torch.tensor(total_loss)),
                "learning_rate": scheduler.get_last_lr()[0],
                "time_per_epoch": time.time() - start_time
            }
        return {}

    def validation_epoch(self, val_loader):
        """验证一个epoch"""
        if self.current_epoch <= self.next_epoch_to_eval:
            return {}
        
        self.transformer.eval()
        total_loss = 0.0
        total_acc = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                loss, acc, coeff, _, _ = self._step(batch, 0, train=False)
                total_loss += (coeff * loss).item()
                total_acc += (coeff * acc).item()
        
        self.next_epoch_to_eval = max(int(1.02 * self.next_epoch_to_eval), self.next_epoch_to_eval + 1)
        
        # 保存检查点
        if self.current_epoch > 0 and (self.current_epoch & (self.current_epoch - 1)) == 0:
            checkpoint = {
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "transformer_state_dict": self.transformer.state_dict(),
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
            "transformer_state_dict": self.transformer.state_dict(),
            "hparams": vars(self.hparams)
        }
        torch.save(init_ckpt, os.path.join(self.checkpoint_path, "init.pt"))
        
        # 进度条
        global_pbar = tqdm(total=self.hparams.max_steps, desc="Transformer Training")
        
        try:
            while self.global_step < self.hparams.max_steps:
                train_logs = self.training_epoch(train_loader, optimizer, scheduler, global_pbar)
                val_logs = self.validation_epoch(val_loader)
                
                all_logs = {
                    "epoch": self.current_epoch,
                    "global_step": self.global_step,
                    "model_type": "transformer",
                    "encoding": "embedding",  # Transformer 固定用 Embedding
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
        self.transformer.eval()
        
        all_losses = []
        all_accs = []
        
        with torch.no_grad():
            for batch in test_loader:
                loss, acc, _, _, _ = self._step(batch, 0, train=False)
                all_losses.append(loss)
                all_accs.append(acc)
        
        loss = torch.cat(all_losses).mean()
        acc = torch.cat(all_accs).mean()
        perplexity = torch.exp(loss)
        
        # 保存测试日志
        test_log = {
            "model_type": "transformer",
            "encoding": "embedding",
            "test_loss": loss.item(),
            "test_accuracy": acc.item(),
            "test_perplexity": perplexity.item()
        }
        with open(os.path.join(self.logdir, "test_metrics.json"), "w") as f:
            json.dump(test_log, f, indent=2)
        
        return test_log

# 导入 Transformer 模型（避免循环导入）
from grok.transformer import Transformer