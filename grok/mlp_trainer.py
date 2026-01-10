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

# 导入算术任务自定义数据模块
from grok.data import ArithmeticDataset, ArithmeticIterator

# 全局默认超参配置 统一管理MLP默认结构参数
DEFAULT_MLP_HIDDEN_DIMS = [512, 256, 128]
DEFAULT_EMBEDDING_DIM = 256

# 算术任务专属可训练MLP封装类
# 一站式实现：数据加载/模型初始化/训练/验证/测试/日志/断点保存全流程
# 所有接口/返回值/日志格式 1:1 严格对齐Transformer版本 无缝切换模型无适配成本
class TrainableMLP:
    def __init__(self, hparams: Namespace) -> None:
        self.hparams = hparams
        # 初始化运行设备 优先使用指定GPU 无则自动降级CPU
        self.device = torch.device(
            f"cuda:{hparams.gpu}" if torch.cuda.is_available() and hparams.gpu >= 0 else "cpu"
        )
        # CUDA性能加速配置 开启基准测试+半精度矩阵运算 释放显存
        if self.device.type == "cuda":
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.set_float32_matmul_precision('medium')
            torch.cuda.empty_cache()

        # 数据加载 完全复用算术数据集逻辑 自动兼容二元/K元加法算子切换
        self.prepare_data()
        
        # 核心初始化：获取token映射关系 解决seq_len相关报错的核心前置步骤
        self.eos_token_id = self.train_dataset.tokenizer.stoi["<|eos|>"]
        self.eq_token_id = self.train_dataset.tokenizer.stoi["="]
        self.seq_len = self.train_dataset.data.shape[1]
        self._remove_all_eos_from_dataset()  # 清洗数据移除EOS标记 重新校准序列长度
        self._safe_eq_position()             # 安全获取等式等号位置 杜绝索引越界
        
        # 计算模型输入维度 适配不同编码方式 维度计算完成后初始化模型
        self.input_dim = self._calc_input_dim()
        
        # 初始化核心MLP模型 迁移至指定设备 强制float32精度统一
        self.mlp = MLP(
            input_dim=self.input_dim,
            output_dim=len(self.train_dataset.tokenizer),
            hidden_dims=self.hparams.mlp_hidden_dims,
            dropout=self.hparams.dropout,
        ).to(self.device)
        
        self.mlp = self.mlp.float()
        for buffer in self.mlp.buffers():
            buffer.data = buffer.data.float()
        
        # 训练状态变量初始化 记录训练进度/批次信息
        self.train_batchsize = 0
        self.batches_per_epoch = 0
        self.current_epoch = 0
        self.global_step = 0
        self.next_epoch_to_eval = -1
        self.next_train_epoch_to_log = 0
        
        # 日志与断点保存目录创建 确保目录存在不报错
        self.logdir = hparams.logdir
        self.checkpoint_path = os.path.join(self.logdir, "checkpoints")
        os.makedirs(self.checkpoint_path, exist_ok=True)
        
        # 训练指标日志文件初始化 按指定路径创建文件头
        self.log_file = os.path.join(self.logdir, "metrics", "metrics.csv")
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self._init_log_file()

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser) -> ArgumentParser:
        # 向命令行解析器追加MLP专属超参 公共参数在train.py维护 解耦清晰
        parser.add_argument("--mlp_hidden_dims", type=int, nargs="+", default=DEFAULT_MLP_HIDDEN_DIMS, help="MLP隐藏层维度列表，空格分隔")
        parser.add_argument("--encoding", type=str, default="onehot", choices=["onehot", "embedding"], help="输入序列编码方式")
        parser.add_argument("--embedding_dim", type=int, default=DEFAULT_EMBEDDING_DIM, help="嵌入维度，仅encoding=embedding时生效")
        return parser

    def prepare_data(self) -> None:
        # 算术数据集加载核心逻辑 与Transformer版本1:1对齐 自动适配二元/K元加法切换
        user_specified_op = self.hparams.math_operator
        default_binary_op = "+"
        math_operator = user_specified_op
        k_value = None
        
        # 检测K值参数 若指定K≥3 自动切换为K元加法算子 无需手动修改配置
        if hasattr(self.hparams, "k"):
            k_value = self.hparams.k
            if (user_specified_op == default_binary_op) and (k_value >= 3):
                math_operator = "+k_mod_97"
                print(f"自动启用k加法：k={k_value}，操作符已切换为'{math_operator}'")

        # 加载训练集+验证集 传参格式与Transformer完全一致 无多余参数
        (self.train_dataset, self.val_dataset,) = ArithmeticDataset.splits(
            train_pct=self.hparams.train_data_pct,
            operator=math_operator,
            operand_length=self.hparams.operand_length,
            data_dir=self.hparams.datadir,
            use_mask=self.hparams.use_mask,
            k=k_value
        )

    def train_dataloader(self) -> ArithmeticIterator:
        # 构建训练集迭代器 复用算术迭代器逻辑 同步批次大小与批次总数
        iterator = ArithmeticIterator(
            self.train_dataset, self.device, batchsize_hint=self.hparams.batchsize
        )
        self.train_batchsize = iterator.batchsize
        self.batches_per_epoch = len(iterator)
        return iterator

    def val_dataloader(self) -> ArithmeticIterator:
        # 构建验证集迭代器 与训练集逻辑一致 无缝切换
        return ArithmeticIterator(self.val_dataset, self.device, batchsize_hint=self.hparams.batchsize)

    def _scheduler_lr(self, step: int) -> float:
        # 学习率调度策略 1:1 复制Transformer版本 无任何改动
        # 支持预热+余弦退火 双阶段学习率衰减 兼顾收敛速度与泛化性
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
        # 优化器配置 1:1 对齐Transformer版本 导入自定义AdamW优化器
        # 支持权重衰减/梯度噪声/多种衰减形式 拼接MLP+嵌入层参数统一优化
        from grok.training import CustomAdamW
        params = list(self.mlp.parameters())
        if hasattr(self, 'embedding'):
            params += list(self.embedding.parameters())
            
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
        # MLP专属准确率计算逻辑 贴合算术任务核心需求：仅计算等号右侧结果的预测准确率
        y_hat_pred = torch.argmax(y_hat, dim=-1)
        row_acc = (y_hat_pred == y).float() * 100
        return row_acc.mean()

    def _step(self, batch, batch_idx, train=True):
        # 单批次前向传播核心逻辑 格式与返回值对齐Transformer 内部逻辑适配MLP特性
        # 安全截取等号右侧1个数值做预测 全流程索引防越界 永不报错
        x = batch["text"]
        y = batch["target"]
        
        # 切换模型训练/验证模式
        self.mlp.train(train)
        if hasattr(self, 'embedding'):
            self.embedding.train(train)
            
        with torch.set_grad_enabled(train):
            x_enc = self._encode_input(x)  # 输入序列编码
            y_hat = self.mlp(x_enc)         # MLP前向推理
            
            # 安全获取等号右侧标签 动态适配当前序列长度 索引永不越界
            current_seq_len = y.shape[1]
            safe_eq_pos = min(self.eq_position, current_seq_len - 2)
            y_rhs = y[:, safe_eq_pos + 1].clamp(max=len(self.train_dataset.tokenizer)-1)
            
            # 计算损失权重系数 按批次占数据集比例加权 保证指标统计准确
            coeff = float(y.shape[0]) / len(self.train_dataset) if train else float(y.shape[0]) / len(self.val_dataset)
            loss = F.cross_entropy(y_hat, y_rhs)
            
            # 无梯度计算准确率
            with torch.no_grad():
                acc = self._accuracy(y_hat, y_rhs)
        
        return loss, acc, coeff, None, None

    def _init_log_file(self):
        # 训练日志文件初始化 写入统一表头 与Transformer日志格式完全一致 便于后续分析
        if not os.path.exists(self.log_file):
            headers = [
                "epoch", "global_step", "train_loss", "train_accuracy", 
                "train_perplexity", "learning_rate", "val_loss", "val_accuracy",
                "val_perplexity", "model_type", "encoding","k_value"
            ]
            with open(self.log_file, "w") as f:
                f.write(",".join(headers) + "\n")

    def _log_metrics(self, metrics_dict):
        # 训练指标日志写入逻辑 100% 复制Transformer版本 格式完全对齐 无任何修改
        # 自动过滤NaN/Inf异常值 保证日志文件完整性
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
        
        if not hasattr(self, 'log_headers'):
            with open(self.log_file, "r") as f:
                self.log_headers = f.readline().strip().split(",")

        values = []
        for h in self.log_headers:
            val = metrics_dict.get(h, "NaN")
            if isinstance(val, Tensor):
                val = val.item()
            values.append(str(val))
        
        with open(self.log_file, "a") as f:
            f.write(",".join(values) + "\n")

    def training_epoch(self, train_loader, optimizer, scheduler, global_pbar):
        # 单轮Epoch训练逻辑 1:1 对齐Transformer版本 进度条/梯度更新/日志收集完全一致
        start_time = time.time()
        total_loss = 0.0
        total_acc = 0.0
        
        epoch_to_log = self.current_epoch == self.next_train_epoch_to_log
        
        for batch_idx, batch in enumerate(train_loader):
            loss, acc, coeff, _, _ = self._step(batch, batch_idx, train=True)
            
            # 梯度清零-反向传播-参数更新-学习率调度
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            self.global_step += 1
            
            # 更新全局进度条 展示核心训练指标
            global_pbar.set_postfix({
                "epoch": self.current_epoch,
                "loss": f"{loss.item():.4f}",
                "model_type": "mlp"
            })
            global_pbar.update(1)
            
            # 按配置收集训练指标
            if epoch_to_log:
                total_loss += (coeff * loss).item()
                total_acc += (coeff * acc).item()
        
        # 动态更新下一次日志记录的Epoch 指数级间隔 减少日志写入开销
        if epoch_to_log:
            self.next_train_epoch_to_log = max(int(1.1 * self.next_train_epoch_to_log), self.next_train_epoch_to_log + 1)
            return {
                "train_loss": torch.tensor(total_loss),
                "train_accuracy": torch.tensor(total_acc),
                "train_perplexity": torch.exp(torch.tensor(total_loss)),
                "learning_rate": scheduler.get_last_lr()[0],
                "time_per_epoch": time.time() - start_time
            }
        return {}

    def validation_epoch(self, val_loader):
        # 单轮Epoch验证逻辑 1:1 对齐Transformer版本 断点保存格式完全一致
        # 按配置间隔执行验证 避免每轮验证增加训练耗时
        if self.current_epoch <= self.next_epoch_to_eval:
            return {}
        
        # 切换模型至验证模式 关闭梯度计算
        self.mlp.eval()
        if hasattr(self, 'embedding'):
            self.embedding.eval()
            
        total_loss = 0.0
        total_acc = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                loss, acc, coeff, _, _ = self._step(batch, 0, train=False)
                total_loss += (coeff * loss).item()
                total_acc += (coeff * acc).item()
        
        # 动态更新下一次验证的Epoch 指数级间隔 提升训练效率
        self.next_epoch_to_eval = max(int(1.2 * self.next_epoch_to_eval), self.next_epoch_to_eval + 1)
        
        # 保存模型断点 仅在2的幂次Epoch保存 减少磁盘占用 键名对齐Transformer
        if self.current_epoch > 0 and (self.current_epoch & (self.current_epoch - 1)) == 0:
            checkpoint = {
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "mlp_state_dict": self.mlp.state_dict(),
                "hparams": vars(self.hparams)
            }
            if hasattr(self, 'embedding'):
                checkpoint["embedding_state_dict"] = self.embedding.state_dict()
            torch.save(checkpoint, os.path.join(self.checkpoint_path, f"epoch_{self.current_epoch}.ckpt"))
        
        return {
            "val_loss": torch.tensor(total_loss),
            "val_accuracy": torch.tensor(total_acc),
            "val_perplexity": torch.exp(torch.tensor(total_loss))
        }

    def fit(self):
        # 训练主循环 1:1 对齐Transformer版本 初始化断点/日志传参完全一致
        # 一站式执行：迭代器构建/优化器初始化/训练/验证/日志写入全流程
        train_loader = self.train_dataloader()
        val_loader = self.val_dataloader()
        optimizer, scheduler = self.configure_optimizers()
        
        # 保存初始断点 记录模型初始状态 便于溯源与复现
        init_ckpt = {
            "epoch": 0,
            "global_step": 0,
            "mlp_state_dict": self.mlp.state_dict(),
            "hparams": vars(self.hparams)
        }
        if hasattr(self, 'embedding'):
            init_ckpt["embedding_state_dict"] = self.embedding.state_dict()
        torch.save(init_ckpt, os.path.join(self.checkpoint_path, "init.pt"))
        
        # 初始化全局训练进度条 展示训练总步数
        global_pbar = tqdm(total=self.hparams.max_steps, desc="MLP Training")
        
        try:
            # 循环训练直至达到最大步数
            while self.global_step < self.hparams.max_steps:
                train_logs = self.training_epoch(train_loader, optimizer, scheduler, global_pbar)
                val_logs = self.validation_epoch(val_loader)
                
                # 合并训练+验证指标 统一写入日志
                all_logs = {
                    "epoch": self.current_epoch,
                    "global_step": self.global_step,
                    "model_type": "mlp",
                    "encoding": self.hparams.encoding,
                    "k_value": self.hparams.k if hasattr(self.hparams, 'k') else 2,
                    **train_logs,
                    **val_logs
                }
                self._log_metrics(all_logs)
                self.current_epoch += 1
        finally:
            # 确保进度条正常关闭 避免终端显示异常
            global_pbar.close()

    def test(self):
        # 模型测试逻辑 1:1 对齐Transformer版本 保存格式完全一致
        # 无梯度推理 计算测试集整体损失/准确率/困惑度 保存至独立日志文件
        test_loader = self.val_dataloader()
        self.mlp.eval()
        if hasattr(self, 'embedding'):
            self.embedding.eval()
        
        all_losses = []
        all_accs = []
        
        with torch.no_grad():
            for batch in test_loader:
                loss, acc, _, _, _ = self._step(batch, 0, train=False)
                all_losses.append(loss.unsqueeze(0))
                all_accs.append(acc.unsqueeze(0))
        
        # 计算测试集整体指标
        loss = torch.cat(all_losses).mean()
        acc = torch.cat(all_accs).mean()
        perplexity = torch.exp(loss)
        
        # 保存测试结果至JSON文件 便于后续分析
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

    # MLP私有核心辅助函数 封装内部逻辑 不对外暴露 不影响整体结构
    def _calc_input_dim(self):
        # 动态计算MLP输入维度 适配onehot/embedding两种编码方式
        vocab_size = len(self.train_dataset.tokenizer)
        if self.hparams.encoding == "onehot":
            return self.seq_len * vocab_size
        elif self.hparams.encoding == "embedding":
            # 初始化嵌入层+位置编码 仅在embedding模式下生效
            self.embedding = nn.Embedding(vocab_size, self.hparams.embedding_dim).to(self.device)
            self.pos_encoding = self._get_positional_encoding(self.seq_len, self.hparams.embedding_dim)
            return self.seq_len * self.hparams.embedding_dim

    def _encode_input(self, x):
        # 输入序列编码核心逻辑 支持两种编码方式 自动对齐序列长度 防越界
        batch_size, raw_len = x.shape
        # 序列长度校准：超长截断 短则补零 保证输入维度固定
        if raw_len > self.seq_len:
            x = x[:, :self.seq_len]
        elif raw_len < self.seq_len:
            x = F.pad(x, (0, self.seq_len - raw_len), value=0)
        
        # OneHot编码：独热向量展平为一维
        if self.hparams.encoding == "onehot":
            return F.one_hot(x.clamp(max=len(self.train_dataset.tokenizer)-1), len(self.train_dataset.tokenizer)).float().reshape(batch_size, -1)
        # Embedding编码：词嵌入+位置编码 拼接展平为一维
        elif self.hparams.encoding == "embedding":
            embed = self.embedding(x.clamp(max=len(self.train_dataset.tokenizer)-1))
            return (embed + self.pos_encoding[:, :self.seq_len]).reshape(batch_size, -1)

    def _get_positional_encoding(self, max_len, d_model):
        # 正弦位置编码实现 仅在embedding模式下生效 为序列添加位置信息
        pe = torch.zeros(max_len, d_model).to(self.device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _remove_all_eos_from_dataset(self):
        # 数据集清洗核心方法：移除所有EOS结束符 仅保留纯算术等式序列
        # 清洗后重新对齐序列长度 校准self.seq_len 解决序列长度不一致问题
        def filter_eos(tensor):
            return tensor[tensor != self.eos_token_id]
        train_data = [filter_eos(self.train_dataset.data[i]) for i in range(len(self.train_dataset.data))]
        self.train_dataset.data = torch.nn.utils.rnn.pad_sequence(train_data, batch_first=True, padding_value=0)
        val_data = [filter_eos(self.val_dataset.data[i]) for i in range(len(self.val_dataset.data))]
        self.val_dataset.data = torch.nn.utils.rnn.pad_sequence(val_data, batch_first=True, padding_value=0)
        self.seq_len = self.train_dataset.data.shape[1]

    def _safe_eq_position(self):
        # 安全获取等式等号在序列中的位置 核心防越界逻辑
        # 自动校准位置范围 保证后续截取等号右侧内容时永不索引越界
        sample = self.train_dataset.data[0].cpu()
        eq_pos = torch.where(sample == self.eq_token_id)[0]
        self.eq_position = eq_pos[0].item() if len(eq_pos) > 0 else self.seq_len - 2
        self.eq_position = min(self.eq_position, self.seq_len - 2)
        self.eq_position = max(self.eq_position, 2)

# 延迟导入MLP模型 避免循环导入问题 与Transformer导入方式完全对齐
from grok.mlp import MLP