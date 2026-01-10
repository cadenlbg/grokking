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

from grok.data import ArithmeticDataset, ArithmeticIterator

DEFAULT_LSTM_HIDDEN_DIM = 256
DEFAULT_LSTM_LAYERS = 2
DEFAULT_EMBEDDING_DIM = 256

class TrainableLSTM:
    def __init__(self, hparams: Namespace) -> None:
        self.hparams = hparams
        self.device = torch.device(
            f"cuda:{hparams.gpu}" if torch.cuda.is_available() and hparams.gpu >= 0 else "cpu"
        )
        
        self.prepare_data()
        self.get_length()
        self.vocab_size = len(self.train_dataset.tokenizer)
        
        self.lstm = LSTM(
            vocab_size=self.vocab_size,
            embedding_dim=self.hparams.embedding_dim if self.hparams.encoding == "embedding" else None,
            hidden_dim=self.hparams.lstm_hidden_dim,
            num_layers=self.hparams.lstm_layers,
            weight_noise=self.hparams.weight_noise,
            dropout=self.hparams.dropout,
            bidirectional=self.hparams.bidirectional,
            use_layer_norm=self.hparams.use_layer_norm,
        ).to(self.device)
        
        self.lstm = self.lstm.float()
        for buffer in self.lstm.buffers():
            buffer.data = buffer.data.float()
        
        self.train_batchsize = 0
        self.batches_per_epoch = 0
        self.current_epoch = 0
        self.global_step = 0
        self.next_epoch_to_eval = -1
        self.next_train_epoch_to_log = 0
        
        self.logdir = hparams.logdir
        self.checkpoint_path = os.path.join(self.logdir, "checkpoints")
        os.makedirs(self.checkpoint_path, exist_ok=True)
        
        self.log_file = os.path.join(self.logdir, "metrics", "metrics.csv")
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self._init_log_file()
    
    def get_length(self):
        sample = self.train_dataset.data[0, :]
        print(f"样本形状: {sample.shape}")
        print(f"样本内容: {sample}")
        
        eq_token_index = self.train_dataset.tokenizer.stoi.get("=", -1)
        eos_token_index = self.train_dataset.tokenizer.stoi.get("<|eos|>", -1)
        
        print(f"等号token索引: {eq_token_index}")
        print(f"EOS token索引: {eos_token_index}")
        
        if eq_token_index == -1:
            print("错误: Tokenizer中没有找到'='符号")
        if eos_token_index == -1:
            print("错误: Tokenizer中没有找到'<|eos|>'符号")
        
        eq_positions = (sample == eq_token_index).nonzero(as_tuple=True)[0]
        eos_positions = (sample == eos_token_index).nonzero(as_tuple=True)[0]
        
        print(f"等号位置: {eq_positions}")
        print(f"EOS位置: {eos_positions}")
        
        if len(eq_positions) > 0 and len(eos_positions) > 0:
            eq_pos = eq_positions[0].item()
            eos_pos = eos_positions[0].item()
            eos_pos_right = eos_positions[1].item() if len(eos_positions) > 1 else eos_pos
            
            print(f"等号位置索引: {eq_pos}")
            print(f"EOS位置索引: {eos_pos},{eos_pos_right}")
            
            self.lhs_len = eq_pos + 1
            self.rhs_len = eos_pos_right - eq_pos - 1
        
        print(f"最终长度: LHS={self.lhs_len}, RHS={self.rhs_len}")

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--lstm_hidden_dim",
            type=int,
            default=DEFAULT_LSTM_HIDDEN_DIM,
            help="LSTM 隐藏层维度（默认：256）"
        )
        parser.add_argument(
            "--lstm_layers",
            type=int,
            default=DEFAULT_LSTM_LAYERS,
            help="LSTM 层数（默认：2）"
        )
        parser.add_argument(
            "--bidirectional",
            action="store_true",
            default=False,
            help="是否使用双向LSTM"
        )
        
        parser.add_argument(
            "--encoding",
            type=str,
            default="embedding",
            choices=["onehot", "embedding"],
            help="LSTM 编码方式（onehot/embedding）"
        )
        parser.add_argument(
            "--embedding_dim",
            type=int,
            default=DEFAULT_EMBEDDING_DIM,
            help="Embedding 编码维度（仅 encoding=embedding 时生效）"
        )
        
        parser.add_argument(
            "--use_layer_norm",
            action="store_true",
            default=False,
            help="是否在LSTM输出后使用层归一化"
        )
        parser.add_argument(
            "--weight_noise",
            type=float,
            default=0.0,
            help="权重噪声强度（训练时添加）"
        )
        
        parser.add_argument("--k", type=int, default=2, help="k个数相加的个数（k≥2）")
        
        return parser

    def prepare_data(self) -> None:
        user_specified_op = self.hparams.math_operator
        default_binary_op = "+"
        
        if user_specified_op == default_binary_op or user_specified_op is None:
            if self.hparams.k >= 3:
                math_operator = f"+{self.hparams.k}_mod_97"
            else:
                math_operator = default_binary_op
        else:
            math_operator = user_specified_op
        
        (self.train_dataset, self.val_dataset,) = ArithmeticDataset.splits(
            train_pct=self.hparams.train_data_pct,
            operator=math_operator,
            operand_length=self.hparams.operand_length,
            data_dir=self.hparams.datadir,
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
        params = list(self.lstm.parameters())
        
        from grok.optimizer import CustomAdamW
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
        """完全仿照MLP的准确率计算"""
        y_hat = torch.max(y_hat, dim=-2).indices
        row_acc = torch.min((y_hat == y), dim=-1).values.float() * 100
        return row_acc.mean()
    
    def _tokens_to_string(self, tokens: Tensor) -> str:
        tokenizer = self.train_dataset.tokenizer
        result = []
        for idx in tokens:
            if hasattr(tokenizer, 'itos') and idx.item() < len(tokenizer.itos):
                result.append(tokenizer.itos[idx.item()])
            else:
                for char, token_idx in tokenizer.stoi.items():
                    if token_idx == idx.item():
                        result.append(char)
                        break
                else:
                    result.append(f"[{idx.item()}]")
        return "".join(result)
    
    def _step(self, batch, batch_idx, train=True):
        """单批次前向传播，只预测等号右侧"""
        x = batch["text"]  # 输入：等号左侧（包括等号）
        y = batch["target"]  # 目标：整个序列右移一位

        self.lstm.train(train)

        with torch.set_grad_enabled(train):
            # LSTM直接处理输入，不需要额外的编码
            y_hat_seq = self.lstm(x)  # 形状: (batch_size, seq_len, vocab_size)

            batch_size = x.shape[0]

            # 提取等号右侧的真实目标
            eq_token_index = self.train_dataset.tokenizer.stoi.get("=", -1)
            if eq_token_index == -1:
                raise ValueError("Tokenizer 中未找到 '=' 符号")

            # 找到等号位置（在目标序列y中）
            eq_position = torch.nonzero(y[0, :] == eq_token_index, as_tuple=True)[0]
            if len(eq_position) == 0:
                raise ValueError("样本中未找到 '=' 符号")
            eq_position = eq_position[0].item()

            # 提取等号右侧部分
            y_rhs = y[:, eq_position + 1:eq_position + 1 + self.rhs_len]  # (batch, rhs_len)

            # 提取预测的等号右侧部分
            y_hat_rhs = y_hat_seq[:, eq_position + 1:eq_position + 1 + self.rhs_len, :]  # (batch, rhs_len, vocab_size)
            y_hat_rhs = y_hat_rhs.transpose(1, 2)  # (batch, vocab_size, rhs_len)

            # 计算损失（只对等号右侧）
            loss = F.cross_entropy(y_hat_rhs, y_rhs)

            # 准确率
            with torch.no_grad():
                acc = self._accuracy(y_hat_rhs, y_rhs)

            coeff = float(y.shape[0]) / len(self.train_dataset) if train else float(y.shape[0]) / len(self.val_dataset)

        return loss, acc, coeff

    def _init_log_file(self):
        if not os.path.exists(self.log_file):
            headers = [
                "epoch", "global_step", "train_loss", "train_accuracy", 
                "train_perplexity", "learning_rate", "val_loss", "val_accuracy",
                "val_perplexity", "model_type", "encoding", "k_value",
                "lstm_hidden_dim", "lstm_layers", "bidirectional", "use_layer_norm"
            ]
            with open(self.log_file, "w") as f:
                f.write(",".join(headers) + "\n")

    def _log_metrics(self, metrics_dict):
        def is_nan_or_inf(val):
            if val is None or val == "NaN":
                return True
            if isinstance(val, Tensor):
                val = val.item()
            if isinstance(val, (float, int)):
                return np.isnan(val) or np.isinf(val)
            return False

        train_acc = metrics_dict.get("train_accuracy", "NaN")
        val_acc = metrics_dict.get("val_accuracy", "NaN")
        if is_nan_or_inf(train_acc) and is_nan_or_inf(val_acc):
            return

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
        start_time = time.time()
        total_loss = 0.0
        total_acc = 0.0
        
        epoch_to_log = self.current_epoch == self.next_train_epoch_to_log

        for batch_idx, batch in enumerate(train_loader):
            loss, acc, coeff = self._step(batch, batch_idx, train=True)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            self.global_step += 1
            
            global_pbar.set_postfix({
                "epoch": self.current_epoch,
                "loss": f"{loss.item():.4f}",
                "encoding": self.hparams.encoding,
                "lstm_layers": self.hparams.lstm_layers
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
        if self.current_epoch <= self.next_epoch_to_eval:
            return {}
        
        self.lstm.eval()
        
        total_loss = 0.0
        total_acc = 0.0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                loss, acc, coeff = self._step(batch, batch_idx, train=False)
                total_loss += (coeff * loss).item()
                total_acc += (coeff * acc).item()

        self.next_epoch_to_eval = max(int(1.02 * self.next_epoch_to_eval), self.next_epoch_to_eval + 1)
        
        if self.current_epoch > 0 and (self.current_epoch & (self.current_epoch - 1)) == 0:
            checkpoint = {
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "lstm_state_dict": self.lstm.state_dict(),
                "hparams": vars(self.hparams)
            }
            torch.save(checkpoint, os.path.join(self.checkpoint_path, f"epoch_{self.current_epoch}.ckpt"))
        
        return {
            "val_loss": torch.tensor(total_loss),
            "val_accuracy": torch.tensor(total_acc),
            "val_perplexity": torch.exp(torch.tensor(total_loss))
        }

    def fit(self):
        train_loader = self.train_dataloader()
        val_loader = self.val_dataloader()
        optimizer, scheduler = self.configure_optimizers()
        
        init_ckpt = {
            "epoch": 0,
            "global_step": 0,
            "lstm_state_dict": self.lstm.state_dict(),
            "hparams": vars(self.hparams)
        }
        torch.save(init_ckpt, os.path.join(self.checkpoint_path, "init.pt"))
        
        global_pbar = tqdm(total=self.hparams.max_steps, desc=f"LSTM ({self.hparams.encoding}, layers={self.hparams.lstm_layers})")
        
        try:
            while self.global_step < self.hparams.max_steps:
                train_logs = self.training_epoch(train_loader, optimizer, scheduler, global_pbar)
                val_logs = self.validation_epoch(val_loader)
                
                all_logs = {
                    "epoch": self.current_epoch,
                    "global_step": self.global_step,
                    "model_type": "lstm",
                    "encoding": self.hparams.encoding,
                    "k_value": self.hparams.k,
                    "lstm_hidden_dim": self.hparams.lstm_hidden_dim,
                    "lstm_layers": self.hparams.lstm_layers,
                    "bidirectional": self.hparams.bidirectional,
                    "use_layer_norm": self.hparams.use_layer_norm,
                    **train_logs,
                    **val_logs
                }
                self._log_metrics(all_logs)
                self.current_epoch += 1
        finally:
            global_pbar.close()

    def test(self):
        test_loader = self.val_dataloader()
        self.lstm.eval()
        
        all_losses = []
        all_accs = []
        
        with torch.no_grad():
            for batch in test_loader:
                loss, acc, _ = self._step(batch, 0, train=False)
                all_losses.append(loss.unsqueeze(0))
                all_accs.append(acc.unsqueeze(0))
        
        loss = torch.cat(all_losses).mean()
        acc = torch.cat(all_accs).mean()
        perplexity = torch.exp(loss)
        
        test_log = {
            "model_type": "lstm",
            "encoding": self.hparams.encoding,
            "lstm_hidden_dim": self.hparams.lstm_hidden_dim,
            "lstm_layers": self.hparams.lstm_layers,
            "bidirectional": self.hparams.bidirectional,
            "use_layer_norm": self.hparams.use_layer_norm,
            "k_value": self.hparams.k,
            "test_loss": loss.item(),
            "test_accuracy": acc.item(),
            "test_perplexity": perplexity.item()
        }
        with open(os.path.join(self.logdir, "test_metrics.json"), "w") as f:
            json.dump(test_log, f, indent=2)
        
        return test_log

from grok.lstm import LSTM