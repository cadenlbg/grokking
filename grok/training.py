#!/usr/bin/env python

import argparse
import copy
import json
import logging
import math
import os
import sys
import pickle
import yaml
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from functools import reduce
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

import grok.metrics as metrics
from grok.data import (
    DEFAULT_DATA_DIR,
    EOS_TOKEN,
    VALID_OPERATORS,
    ArithmeticDataset,
    ArithmeticIterator,
)
from grok.transformer import Transformer
from grok.measure import get_sharpness

DEFAULT_LOG_DIR = "logs"

def save_hparams(hparams: Namespace, save_path: str) -> None:
    """将超参数保存为YAML文件"""
    # 创建保存目录（如果不存在）
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    # 将Namespace转换为字典并保存
    with open(save_path, "w") as f:
        yaml.dump(vars(hparams), f, sort_keys=False)
    print(f"超参数已保存至: {save_path}")

class TrainableTransformer:
    """
    Adds training methods to train a generic transformer on arithmetic equations
    (纯PyTorch实现，无PyTorch Lightning依赖)
    """

    def __init__(self, hparams: Namespace) -> None:
        """
        :param hparams: An argparse.Namespace with parameters defined in
                        self.add_model_specific_args().
        """
        self.hparams = hparams
        self.device = torch.device(f"cuda:{hparams.gpu}" if torch.cuda.is_available() and hparams.gpu >= 0 else "cpu")
        
        self.prepare_data()

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

        self.transformer = self.transformer.float()  # 转换可学习参数为 float32
        for buffer in self.transformer.buffers():  # 转换所有 buffer 为 float32
            buffer.data = buffer.data.float()

        self.margin = torch.Tensor([0]).to(self.device)
        self.next_epoch_to_eval = -1
        self.next_train_epoch_to_log = 0
        
        # 初始化训练相关变量
        self.train_batchsize = 0
        self.batches_per_epoch = 0
        self.current_epoch = 0
        self.global_step = 0
        
        # 日志和检查点相关
        self.logdir = hparams.logdir
        self.checkpoint_path = os.path.join(self.logdir, "checkpoints")
        os.makedirs(self.checkpoint_path, exist_ok=True)

        # 日志文件
        pct = f"{hparams.train_data_pct}".replace(".", "p")
        max_step = f"{hparams.max_steps}"
        operator = hparams.math_operator.replace("+", "plus").replace("-", "minus").replace("*", "mul")
        filename = f"metrics_pct{pct}_maxstep{max_step}_operator{operator}.csv"
        
        log_dir = os.path.join(self.logdir, "metrics")
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, filename)
        self._init_log_file()


    @staticmethod
    def add_model_specific_args(parser: ArgumentParser) -> ArgumentParser:
        """
        Defines the hyperparameter arguments needed by instances of this
        class. This is intended to be called when parsing command line
        arguments.

        :param parser: an argparse.ArgumentParser created by the caller
        :returns: the argument parser with the command line arguments added
                  for this class.
        """
        parser.add_argument(
            "--batchsize",
            type=float,
            default=0,
            help="-1 -> entire dataset, 0 -> auto-calculate, 0<N<1 -> fraction of dataset, N>1 -> N",
        )

        parser.add_argument("--n_layers", type=int, default=2)
        parser.add_argument("--n_heads", type=int, default=4)
        parser.add_argument("--d_model", type=int, default=128)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--weight_noise", type=float, default=0.0)
        parser.add_argument("--non_linearity", type=str, default="relu")
        parser.add_argument("--max_context_len", type=int, default=50)

        parser.add_argument("--math_operator", type=str, default="+")
        parser.add_argument(
            "--operand_length",
            type=int,
            help="for list operations, the length of the lists",
        )

        parser.add_argument("--train_data_pct", type=float, default=5)
        parser.add_argument("--warmup_steps", type=int, default=10)
        parser.add_argument("--anneal_lr_steps", type=int, default=100000)
        parser.add_argument("--anneal_lr", dest="anneal_lr", action="store_true")
        parser.set_defaults(anneal_lr=False)

        parser.add_argument("--max_lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=0)
        parser.add_argument("--weight_decay_kind", type=str, default="to_zero")
        parser.add_argument("--noise_factor", type=float, default=0)

        parser.add_argument(
            "--save_activations", dest="save_activations", action="store_true"
        )
        parser.set_defaults(save_activations=False)
        parser.add_argument("--save_outputs", dest="save_outputs", action="store_true")
        parser.set_defaults(save_outputs=False)

        parser.add_argument(
            "--logdir",
            type=str,
            default=DEFAULT_LOG_DIR,
        )
        parser.add_argument(
            "--datadir",
            type=str,
            default=DEFAULT_DATA_DIR,
        )

        return parser

    def prepare_data(self) -> None:
        """
        Loads training data to self.train_dataset
        Loads validation data to self.val_dataset
        """
        (self.train_dataset, self.val_dataset,) = ArithmeticDataset.splits(
            train_pct=self.hparams.train_data_pct,
            operator=self.hparams.math_operator,
            operand_length=self.hparams.operand_length,
            data_dir=self.hparams.datadir,
        )

    def train_dataloader(self) -> ArithmeticIterator:
        """
        :returns: an iterator for self.train_dataset
        """
        iterator = ArithmeticIterator(
            self.train_dataset,
            self.device,
            batchsize_hint=self.hparams.batchsize,
        )
        self.train_batchsize = iterator.batchsize
        self.batches_per_epoch = len(iterator)

        return iterator

    def val_dataloader(self) -> ArithmeticIterator:
        """
        :returns: an iterator for self.val_dataset
        """
        iterator = ArithmeticIterator(
            self.val_dataset,
            self.device,
            batchsize_hint=-1,  # no need to batch validation data
        )
        return iterator

    def test_dataloader(self) -> ArithmeticIterator:
        """
        :returns: an iterator for self.val_dataset
        """
        iterator = ArithmeticIterator(
            self.val_dataset, self.device, batchsize_hint=-1
        )
        return iterator

    def _scheduler_lr(self, step: int) -> float:
        """
        :returns: the learning_rate for this training step
        """
        max_lr = self.hparams.max_lr
        min_lr = self.hparams.max_lr / 10
        warmup_steps = self.hparams.warmup_steps
        
        if not self.hparams.anneal_lr:
            if step <= warmup_steps:
                lr = (float(step) / max(warmup_steps, 1)) * max_lr
            else:
                lr = max_lr
        else:
            if step <= warmup_steps:
                lr = (float(step) / max(warmup_steps, 1)) * max_lr
            elif step <= self.hparams.anneal_lr_steps + warmup_steps:
                effective_step = step - warmup_steps
                t = effective_step / self.hparams.anneal_lr_steps
                cos = (1 + np.cos(np.pi * t)) / 2
                lr = min_lr + (max_lr - min_lr) * cos
            else:
                lr = min_lr
        return lr

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, LambdaLR]:
        """
        :returns: optimizer and scheduler
        """
        optimizer = CustomAdamW(
            self.transformer.parameters(),
            betas=(0.9, 0.98),
            eps=1e-8,
            lr=1,  # 由scheduler控制实际学习率
            weight_decay=self.hparams.weight_decay,
            noise_factor=self.hparams.noise_factor,
            weight_decay_form=self.hparams.weight_decay_kind,
        )
        
        scheduler = LambdaLR(optimizer, lr_lambda=self._scheduler_lr)
        return optimizer, scheduler

    def _accuracy(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """
        Takes the most likely solution predicted for each equation and
        calculates the frac of equations in the batch for which these
        answers were correct

        :param y_hat: The softmax tensor output of the transformer
        :param y: A tensor of the token ids for the correct answers to each
                  equation in the batch
        :returns: the fraction of equations correctly answered
        """
        # find max prediction from output
        y_hat = torch.max(y_hat, dim=-2).indices  # batchsize x num_rhs_tokens
        row_accuracy = torch.min((y_hat == y), dim=-1).values  # shape: batchsize
        accuracy = row_accuracy.float() * 100  # shape: batchsize
        return accuracy

    def _step(
        self,
        batch: Dict,
        batch_idx: int,
        train: bool = True,
        reduction: str = "mean",
        grads: bool = False,
    ) -> Tuple[Tensor, ...]:
        """
        Performs one forward pass on a training or validation batch

        :param batch: The batch of equations to process
        :param batch_idx: which batch this is in the epoch.
        :param train: True is this is a training batch, false otherwise
        :returns: The loss from the predicted solutions to the equation,
                  The accuracy of the predicted solutions
                  The fraction of this dataset contained in this batch
                  The portion of the input equations left of the equal sign
                  The softmax probilities for the solutions to the equations
                  A list lists of attention matrices by layer and head
                  A list lists of value matrices by layer and head
        """
        x = batch["text"]  # shape = batchsize * context_len
        y = batch["target"]  # shape = batchsize * context_len
        
        self.transformer.train(train)
        with torch.set_grad_enabled(train or grads):
            y_hat, attentions, values = self.transformer(
                x=x, save_activations=self.hparams.save_activations
            )  # shape = batchsize * context_len * vocab_size
            y_hat = y_hat.transpose(-2, -1)  # shape = batchsize * vocab_size * context_len

            # Note: each sample must have exactly one '=' and all of them must
            # have it in the same position.
            eq_token_index = self.train_dataset.tokenizer.stoi["="]
            eq_position_t = torch.nonzero(y[0, :] == eq_token_index, as_tuple=False)
            eq_position = int(eq_position_t.squeeze())

            # only calculate loss/accuracy on right hand side of the equation
            y_rhs = y[..., eq_position + 1 :]
            y_hat_rhs = y_hat[..., eq_position + 1 :]
            x_lhs = x[..., : eq_position + 1]

            if train:
                coeff = float(batch["target"].shape[0]) / len(self.train_dataset)
            else:
                coeff = float(batch["target"].shape[0]) / len(self.val_dataset)
            loss = F.cross_entropy(y_hat_rhs, y_rhs, reduction=reduction)

            with torch.no_grad():
                acc = self._accuracy(y_hat_rhs, y_rhs)
                if reduction == "mean":
                    acc = acc.mean()

            grad_vec = None
            if grads:
                loss.backward()
                for p in self.transformer.parameters():
                    if p.grad is not None:
                        p.grad.data.div_(batch["text"].shape[0])
                        if grad_vec is None:
                            grad_vec = p.grad.data.view(-1)
                        else:
                            grad_vec = torch.cat((grad_vec, p.grad.data.view(-1)))
                return loss, grad_vec

        return loss, acc, coeff, x_lhs, y_hat_rhs, attentions, values

    def _init_log_file(self):
        """初始化日志CSV文件"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                # 写入CSV表头
                headers = [
                    "epoch", "global_step", "train_loss", "train_accuracy", 
                    "train_perplexity", "learning_rate", "len_train_ds", 
                    "len_val_ds", "batches_per_epoch", "time_per_epoch",
                    "fwd_time_in_epoch", "val_loss", "val_accuracy", 
                    "val_perplexity", "full_train_loss", "full_train_acc"
                ]
                # 添加参数范数的表头
                for name, _ in self.transformer.named_parameters():
                    headers.append(f"paramnorm_{name}")
                f.write(",".join(headers) + "\n")

    def _log_metrics(self, metrics_dict: Dict[str, Any]):
        
    # 第一步：提取并判断 train_accuracy 和 val_accuracy 是否有效（不为 NaN）
    
        train_acc = metrics_dict.get("train_accuracy")
        val_acc = metrics_dict.get("val_accuracy")
    
    # 处理 Tensor/np.ndarray 转数值，同时判断是否为 NaN
        def is_valid_acc(acc):
            if acc is None:
                return False
        # 转换为数值（处理 Tensor/np.ndarray）
            if isinstance(acc, (Tensor, np.ndarray)):
                acc_val = acc.item()
            else:
                acc_val = acc
        # 判断是否为有效数值（非 NaN、非无穷）
            return not (np.isnan(acc_val) or np.isinf(acc_val))
    
    # 过滤条件：train_acc 有效 OR val_acc 有效（满足其一则保留）
        if not (is_valid_acc(train_acc) or is_valid_acc(val_acc)):
            return  # 两者都为 NaN，不写入文件
    
    # 第二步：原有逻辑（只处理满足条件的行）
    # 确保所有指标都有值，没有的用NaN填充
        all_headers = []
        with open(self.log_file, "r") as f:
            all_headers = f.readline().strip().split(",")
    
        values = []
        for header in all_headers:
            if header in metrics_dict:
                val = metrics_dict[header]
                if isinstance(val, Tensor):
                    val = val.item()
                elif isinstance(val, np.ndarray):
                    val = val.item()
                values.append(str(val))
            else:
                values.append("NaN")
    
        with open(self.log_file, "a") as f:
            f.write(",".join(values) + "\n")


    def _save_inputs(self, x_lhs: Tensor, ds: str) -> None:
        """
        Saves the input equations to disk for analysis later

        :param x_lhs: 输入的左侧表达式张量
        :param ds: a string ('train' or 'val') naming which dataset
        """
        logdir = os.path.join(self.logdir, "inputs", ds)
        os.makedirs(logdir, exist_ok=True)
        pickle_file = os.path.join(logdir, f"{ds}.pt")

        with open(pickle_file, "wb") as fh:
            torch.save(x_lhs, fh)

    def _merge_batch_activations(
        self, partial_activations: List[List[Tensor]]
    ) -> List[List[Tensor]]:
        """
        Merges the head_attentions / head_values from all batches in
        this epoch.

        :param partial_activations: A list of
                                   (lists of lists of activations by layer and head)
        :returns: A lists of lists of activations by layer and head
        """
        num_layers = len(partial_activations[0])
        num_heads = len(partial_activations[0][0])
        activations: List = []
        for _ in range(num_layers):
            activations.append([])
            for _ in range(num_heads):
                activations[-1].append([])

        for minibatch_activations in partial_activations:
            for l, layer_activations in enumerate(minibatch_activations):
                for h, head_attn in enumerate(layer_activations):
                    activations[l][h].append(head_attn)

        for l in range(num_layers):
            for h in range(num_heads):
                activations[l][h] = torch.cat(activations[l][h])

        return activations

    def _save_activations(self, outputs: List[Dict[str, Any]], ds: str) -> None:
        """
        Saves activations out to disk for analysis later

        :param outputs: a list of dicts from training/validation step
        :param ds: 'train' or 'val'
        """
        output: Dict[str, Any] = {}
        if self.hparams.save_outputs:
            y_hat_rhs = torch.cat([x["y_hat_rhs"] for x in outputs])
            output["y_hat_rhs"] = y_hat_rhs
        if self.hparams.save_activations:
            partial_attentions = [o["partial_attentions"] for o in outputs]
            attentions = self._merge_batch_activations(partial_attentions)
            partial_values = [o["partial_values"] for o in outputs]
            values = self._merge_batch_activations(partial_values)
            output["attentions"] = attentions
            output["values"] = values
        
        if self.hparams.save_outputs or self.hparams.save_activations:
            logdir = os.path.join(self.logdir, "outputs", ds)
            os.makedirs(logdir, exist_ok=True)
            pickle_file = os.path.join(logdir, f"epoch_{self.current_epoch:010}.pt")
            with open(pickle_file, "wb") as fh:
                torch.save(output, fh)

    def training_epoch(self, train_loader: ArithmeticIterator, optimizer: torch.optim.Optimizer, scheduler: LambdaLR,global_pbar: tqdm) -> Dict[str, Any]:
        """
        执行一个训练epoch
        """
        self.transformer.train()
        training_epoch_start_time = time.time()
        fwd_time_in_epoch = 0
        
        outputs = []
        total_train_loss = 0.0
        total_train_acc = 0.0
        
        epoch_is_to_be_logged = self.current_epoch == self.next_train_epoch_to_log

        for batch_idx, batch in enumerate(train_loader):
            start = time.time()
            
            # 前向传播
            loss, accuracy, coeff, x_lhs, y_hat_rhs, attentions, values = self._step(
                batch=batch, batch_idx=batch_idx, train=True
            )
            fwd_time_in_epoch += time.time() - start
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            self.global_step += 1
            
            global_pbar.set_postfix({
                "epoch": self.current_epoch,
                "step": self.global_step,
                "loss": f"{loss.item():.4f}"  # 实时显示当前batch的loss
            })
            global_pbar.update(1)  # 每处理1个batch，进度+1

            # 累积损失和准确率
            if epoch_is_to_be_logged:
                total_train_loss += (coeff * loss).item()
                total_train_acc += (coeff * accuracy).item()
                
                output_dict = {
                    "y_hat_rhs": y_hat_rhs,
                    "partial_attentions": attentions,
                    "partial_values": values,
                }
                if self.current_epoch == 0:
                    output_dict["x_lhs"] = x_lhs
                outputs.append(output_dict)

        # 日志记录
        logs = {}
        if epoch_is_to_be_logged:
            self.next_train_epoch_to_log = max(
                int(1.01 * self.next_train_epoch_to_log),
                self.next_train_epoch_to_log + 1,
            )
            
            train_loss = torch.tensor(total_train_loss)
            train_accuracy = torch.tensor(total_train_acc)
            perplexity = torch.exp(train_loss)
            lr = scheduler.get_last_lr()[0]
            
            # 保存输入和激活
            if self.hparams.save_activations or self.hparams.save_outputs:
                if self.current_epoch == 0:
                    all_x_lhs = torch.cat([x["x_lhs"] for x in outputs])
                    self._save_inputs(all_x_lhs, ds="train")
                self._save_activations(outputs, ds="train")
            
            logs.update({
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "train_perplexity": perplexity,
                "learning_rate": lr,
                "len_train_ds": len(self.train_dataset),
                "len_val_ds": len(self.val_dataset),
                "batches_per_epoch": self.batches_per_epoch,
                "time_per_epoch": time.time() - training_epoch_start_time,
                "fwd_time_in_epoch": fwd_time_in_epoch,
            })

        return logs

    def validation_epoch(self, val_loader: ArithmeticIterator) -> Dict[str, Any]:
        """
        执行一个验证epoch
        """
        self.transformer.eval()
        outputs = []
        total_val_loss = 0.0
        total_val_acc = 0.0
        
        if self.next_epoch_to_eval < self.current_epoch:
            self.next_epoch_to_eval = self.current_epoch
        
        validation_is_real = self.current_epoch == self.next_epoch_to_eval
        
        if not validation_is_real:
            return {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                loss, accuracy, coeff, x_lhs, y_hat_rhs, attentions, values = self._step(
                    batch=batch, batch_idx=batch_idx, train=False
                )
                
                total_val_loss += (coeff * loss).item()
                total_val_acc += (coeff * accuracy).item()
                
                output_dict = {
                    "y_hat_rhs": y_hat_rhs,
                    "partial_attentions": attentions,
                    "partial_values": values,
                }
                if self.current_epoch == 0:
                    output_dict["x_lhs"] = x_lhs
                outputs.append(output_dict)
        
        # 更新下一次验证的epoch
        self.next_epoch_to_eval = max(
            int(1.02 * self.next_epoch_to_eval), self.next_epoch_to_eval + 1
        )
        
        # 计算验证指标
        val_loss = torch.tensor(total_val_loss)
        val_accuracy = torch.tensor(total_val_acc)
        val_perplexity = torch.exp(val_loss)
        
        # 保存输入和激活
        if self.hparams.save_activations or self.hparams.save_outputs:
            if self.current_epoch == 0:
                all_x_lhs = torch.cat([x["x_lhs"] for x in outputs])
                self._save_inputs(all_x_lhs, ds="val")
            self._save_activations(outputs, ds="val")
        
        # 计算训练集全量指标
        train_data = self.train_dataset.data.to(self.device)
        training_data = {"text": train_data[:, :-1], "target": train_data[:, 1:]}
        with torch.no_grad():
            tr_loss, tr_acc, *_ = self._step(training_data, 0, train=False)
        
        # 计算参数范数
        param_norms = {}
        for name, param in self.transformer.named_parameters():
            n_params = param.numel()
            param_norms[f"paramnorm_{name}"] = torch.norm(param, 2).detach().cpu().numpy() / np.sqrt(n_params)
        
        logs = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_perplexity": val_perplexity,
            "full_train_loss": tr_loss,
            "full_train_acc": tr_acc,
            **param_norms
        }
        
        # 保存检查点（2的幂次epoch）
        if (self.current_epoch > 0 and 
            int(2 ** (int(np.log(self.current_epoch) / np.log(2)))) == self.current_epoch):
            checkpoint = {
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "model_state_dict": self.transformer.state_dict(),
                "hyper_parameters": vars(self.hparams),
            }
            checkpoint_file = os.path.join(self.checkpoint_path, f"epoch_{self.current_epoch}.ckpt")
            torch.save(checkpoint, checkpoint_file)
        
        return logs

    def fit(self) -> None:
        """
        训练主循环
        """
        # 初始化数据加载器
        train_loader = self.train_dataloader()
        val_loader = self.val_dataloader()
        
        # 初始化优化器和调度器
        optimizer, scheduler = self.configure_optimizers()
        
        # 保存初始模型
        init_checkpoint = {
            "epoch": 0,
            "global_step": 0,
            "model_state_dict": self.transformer.state_dict(),
            "hyper_parameters": vars(self.hparams),
        }
        torch.save(init_checkpoint, os.path.join(self.checkpoint_path, "init.pt"))
        
        total_steps = self.hparams.max_steps
        pbar = tqdm(total=total_steps, desc="Training", unit="step")

        try:
            # 训练循环
            while self.global_step < self.hparams.max_steps:
                # 训练epoch
                train_logs = self.training_epoch(train_loader, optimizer, scheduler,pbar)
                
                # 验证epoch
                val_logs = self.validation_epoch(val_loader)
                
                # 合并日志并保存
                all_logs = {
                    "epoch": self.current_epoch,
                    "global_step": self.global_step,
                    **train_logs,
                    **val_logs
                }
                self._log_metrics(all_logs)
                self.current_epoch += 1
        finally:
            pbar.close()

    def test(self) -> Dict[str, Any]:
        """
        测试过程
        """
        test_loader = self.test_dataloader()
        self.transformer.eval()
        
        all_losses = []
        all_accuracies = []
        outputs = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                loss, accuracy, coeff, x_lhs, y_hat_rhs, attentions, values = self._step(
                    batch=batch, batch_idx=batch_idx, train=False, reduction="none"
                )
                
                all_losses.append(loss)
                all_accuracies.append(accuracy)
                
                output_dict = {
                    "y_hat_rhs": y_hat_rhs,
                    "partial_attentions": attentions,
                    "partial_values": values,
                }
                if self.current_epoch == 0:
                    output_dict["x_lhs"] = x_lhs
                outputs.append(output_dict)
        
        # 合并结果
        loss = torch.cat(all_losses, dim=0)
        accuracy = torch.cat(all_accuracies, dim=0)
        perplexity = torch.exp(loss)
        
        # 保存测试输出
        if self.hparams.save_activations or self.hparams.save_outputs:
            logdir = os.path.join(self.logdir, "outputs", "test")
            os.makedirs(logdir, exist_ok=True)
            pickle_file = os.path.join(logdir, "test_outputs.pt")
            
            output_data = {
                "test_loss": loss,
                "test_accuracy": accuracy,
                "test_perplexity": perplexity,
            }
            if self.hparams.save_outputs:
                output_data["y_hat_rhs"] = torch.cat([x["y_hat_rhs"] for x in outputs])
            if self.hparams.save_activations:
                partial_attentions = [o["partial_attentions"] for o in outputs]
                attentions = self._merge_batch_activations(partial_attentions)
                partial_values = [o["partial_values"] for o in outputs]
                values = self._merge_batch_activations(partial_values)
                output_data["attentions"] = attentions
                output_data["values"] = values
            
            with open(pickle_file, "wb") as fh:
                torch.save(output_data, fh)
        
        logs = {
            "test_loss": loss,
            "test_accuracy": accuracy,
            "test_perplexity": perplexity,
        }
        
        # 保存测试日志
        test_log_file = os.path.join(self.logdir, "test_metrics.json")
        with open(test_log_file, "w") as f:
            json.dump({
                "mean_test_loss": loss.mean().item(),
                "mean_test_accuracy": accuracy.mean().item(),
                "mean_test_perplexity": perplexity.mean().item()
            }, f, indent=2)
        
        return logs


def train(hparams: Namespace) -> None:
    """
    This is the main trainer_method. This sets up and runs experiment with
    the defined hyperparameters

    :param hparams: An argparse.Namespace with all of the relevant hyperparameters
    """

    # Process the args
    if hparams.logdir is None:
        hparams.logdir = os.environ.get("LOGDIR", ".")
    hparams.logdir = os.path.abspath(hparams.logdir)

    # Make sure d_model, heads, and d_key are compatible
    assert (
        hparams.d_model % hparams.n_heads == 0
    ), "n_heads=%s does not evenly divide d_model=%s" % (
        hparams.n_heads,
        hparams.d_model,
    )
    hparams.d_key = hparams.d_model / hparams.n_heads

    hparams_save_path = os.path.join(hparams.logdir, "hparams.yaml")
    save_hparams(hparams, hparams_save_path)

    # Set up the RNGs for repeatability
    if hparams.random_seed != -1:
        torch.manual_seed(hparams.random_seed)
        torch.cuda.manual_seed(hparams.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create the model
    model = TrainableTransformer(hparams)

    # 开始训练
    model.fit()

    # 训练结束后进行测试
    test_logs = model.test()
    print(f"Test completed. Mean Test Accuracy: {test_logs['test_accuracy'].mean().item():.2f}%")

    return hparams.logdir


def compute_sharpness(hparams: Namespace, ckpts: List[str]) -> None:
    """
    This is the compute_sharpness method. This loads a series of checkpoints in
    the defined hyperparameters

    :param hparams: An argparse.Namespace with all of the relevant hyperparameters
    :param ckpts: 检查点文件路径列表
    """

    # Process the args
    if hparams.logdir is None:
        hparams.logdir = os.environ.get("LOGDIR", ".")
    hparams.logdir = os.path.abspath(hparams.logdir)

    # Make sure d_model, heads, and d_key are compatible
    assert (
        hparams.d_model % hparams.n_heads == 0
    ), "n_heads=%s does not evenly divide d_model=%s" % (
        hparams.n_heads,
        hparams.d_model,
    )
    hparams.d_key = hparams.d_model / hparams.n_heads

    # Set up the RNGs for repeatability
    if hparams.random_seed != -1:
        torch.manual_seed(hparams.random_seed)
        torch.cuda.manual_seed(hparams.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 创建结果目录
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    for i, ckpt in enumerate(ckpts):
        print(f"Loading checkpoint {ckpt}")
        checkpoint = torch.load(ckpt, map_location=f"cuda:{hparams.gpu}" if torch.cuda.is_available() and hparams.gpu >=0 else "cpu")
        
        # 重建模型
        hps = Namespace(**checkpoint["hyper_parameters"])
        # 覆盖必要的hparams
        hps.gpu = hparams.gpu
        hps.random_seed = hparams.random_seed
        
        model = TrainableTransformer(hps)
        model.transformer.load_state_dict(checkpoint["model_state_dict"])
        model.transformer.to(model.device)
        
        # 计算sharpness
        phi = get_sharpness(model.train_dataloader(), model.transformer)
        results = {ckpt: phi}
        
        # 保存结果
        pickle.dump(results, open(os.path.join(results_dir, f"results_SD-{i}.pkl"), "wb"))
        print(f"Saved sharpness results for checkpoint {ckpt}")


def add_args(parser=None) -> ArgumentParser:
    """
    Parses the command line arguments

    :returns: an argparse.ArgumentParser with all of the needed arguments
    """
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=-1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser = TrainableTransformer.add_model_specific_args(parser)
    return parser


class CustomAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        noise_factor=0.0,
        weight_decay_form="to_zero",
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not weight_decay_form in ["to_zero", "to_init", "jiggle", "honest"]:
            raise ValueError(
                f"Invalid weight decay form: {weight_decay_form}, should be one of ['to_zero', 'to_init', 'jiggle']"
            )
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            noise_factor=noise_factor,
            weight_decay_form=weight_decay_form,
        )
        super(CustomAdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CustomAdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad

                if group["weight_decay"] > 0:
                    if group["weight_decay_form"] == "honest":
                        grad = grad + group["weight_decay"] * p.detach()

                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group["amsgrad"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if group["weight_decay_form"] == "to_init":
                        state["init"] = p.detach().clone()
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                if group["weight_decay"] > 0:
                    if group["weight_decay_form"] == "to_zero":
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    elif group["weight_decay_form"] == "to_init":
                        p.add_(
                            (state["init"] - p) * (group["lr"] * group["weight_decay"])
                        )
                    elif group["weight_decay_form"] == "jiggle":
                        p.mul_(
                            torch.exp(
                                torch.randn(1).to(p.device)
                                * (group["lr"] * group["weight_decay"])
                            )
                        )
                    elif group["weight_decay_form"] == "honest":
                        pass
                    else:
                        raise ValueError(
                            f"Invalid weight decay form: {group['weight_decay_form']}"
                        )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                        group["eps"]
                    )
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                        group["eps"]
                    )

                step_size = group["lr"] / bias_correction1

                upd = exp_avg / denom
                # add uniform gaussian noise to the update
                if group["noise_factor"] > 0:
                    upd += torch.randn_like(upd) * group["noise_factor"]
                p.add_(-step_size * upd)

        return loss


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(
            closure
        )  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        grad_norms = [
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        norm = torch.norm(
            torch.stack(grad_norms),
            p=2,
        )
        return norm


def main():
    """主函数"""
    parser = add_args()
    # 添加模式选择参数
    parser.add_argument("--mode", type=str, default="train", choices=["train", "compute_sharpness"],
                      help="运行模式：train（训练）或 compute_sharpness（计算sharpness）")
    parser.add_argument("--ckpts", nargs="+", help="compute_sharpness模式下需要的检查点文件路径列表")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train(args)
    elif args.mode == "compute_sharpness":
        if not args.ckpts:
            raise ValueError("compute_sharpness模式需要指定--ckpts参数")
        compute_sharpness(args, args.ckpts)


if __name__ == "__main__":
    main()