import itertools
import math
import os
import sys
import random

import torch
from torch import Tensor, LongTensor
import numpy as np
from typing import Tuple, List, Dict, Any, Union, Optional
from tqdm import tqdm

from mod import Mod
import blobfile as bf

# 全局核心配置-仅保留加法相关算子映射 算子符号:算子描述
VALID_OPERATORS = {
    "+": "addition",
    "+k_mod_97": "k-addition_mod_97",
}

# 全局固定常量定义
EOS_TOKEN = "<|eos|>"
EQ_TOKEN = "="
MODULUS = 97
NUMS = list(range(MODULUS))
DEFAULT_DATA_DIR = "data"

def render(operand, join_str=""):
    # 统一渲染各类操作数为字符串格式 适配列表/元组/数组/Mod对象/基础数字
    if isinstance(operand, list) or isinstance(operand, tuple) or isinstance(operand, np.ndarray):
        return join_str.join(map(render, operand))
    elif isinstance(operand, Mod):
        return str(operand._value)
    else:
        return str(operand)

def create_data_files(data_dir: str = DEFAULT_DATA_DIR):
    # 数据文件生成入口方法 生成词汇表+数据集文件
    ArithmeticTokenizer.create_token_file(data_dir)
    ArithmeticDataset.create_dataset_files(data_dir)

class ArithmeticTokenizer:
    # 算术任务专属分词器：维护token与id的双向映射 完成文本<->张量的转换
    token_file = "tokens.txt"

    def __init__(self, data_dir=DEFAULT_DATA_DIR) -> None:
        self.token_file = bf.join(data_dir, self.token_file)
        self.itos = self.get_tokens()
        self.stoi: Dict[str, int] = dict([(s, i) for i, s in enumerate(self.itos)])

    def _encode(self, s: str) -> Tensor:
        # 单条文本转id张量 内部调用方法
        return LongTensor([self.stoi[t] for t in s.split(" ")])

    def encode(self, obj: Union[str, List]) -> Tensor:
        # 对外暴露的编码方法 支持单文本/文本列表批量编码
        if isinstance(obj, str):
            return self._encode(obj)
        elif isinstance(obj, list):
            return torch.stack([self._encode(s) for s in obj], dim=0)
        else:
            raise NotImplementedError

    def decode(self, tensor: Tensor, with_brackets: bool = False) -> str:
        # 张量转文本 支持是否给token添加前后括号标识
        indices = tensor.long()
        l, r = ("<", ">") if with_brackets else ("", "")
        tokens = [l + self.itos[i] + r for i in indices]
        return " ".join(tokens)

    def __len__(self) -> int:
        # 返回词汇表大小
        return len(self.itos)

    @classmethod
    def get_tokens(cls):
        # 构建加法任务专属极简词汇表 无冗余token
        tokens = [EOS_TOKEN, EQ_TOKEN] + list(sorted(list(VALID_OPERATORS.keys()))) + list(map(render, NUMS))
        return tokens

class ArithmeticDataset:
    # 加法算术等式数据集核心类 负责等式生成、数据划分、过滤、格式化等全流程

    @staticmethod
    def _extract_label(eq_str: str, tokenizer: ArithmeticTokenizer) -> Optional[int]:
        # 从等式字符串中提取计算结果标签 兼容任意元加法等式格式
        eq_clean = eq_str.replace(EOS_TOKEN, "").strip()
        tokens = eq_clean.split(" ")
        if EQ_TOKEN not in tokens:
            return None
        
        eq_idx = tokens.index(EQ_TOKEN)
        label_tokens = tokens[eq_idx+1:]
        if not label_tokens:
            return None
        
        label_str = "".join(label_tokens).strip()
        try:
            return int(label_str)
        except ValueError:
            if label_str.startswith("Mod(") and label_str.endswith(")"):
                return int(label_str.split(",")[0].split("(")[1])
            return None

    @staticmethod
    def _filter_by_mask(eqs: List[str], tokenizer: ArithmeticTokenizer) -> Tuple[List[str], List[str]]:
        # 按结果标签值过滤划分数据集 train:结果>20 val:结果<20
        train_eqs = []
        val_eqs = []
        for eq in eqs:
            label = ArithmeticDataset._extract_label(eq, tokenizer)
            if label is None:
                continue
            if label > 20:
                train_eqs.append(eq)
            elif label < 20:
                val_eqs.append(eq)
        print(f"Mask过滤后：训练集{len(train_eqs)}条（label>20），验证集{len(val_eqs)}条（label<20）")
        return train_eqs, val_eqs
    
    @classmethod
    def splits(
        cls,
        train_pct: float,
        operator: str,
        operand_length: Optional[int] = None,
        data_dir: str = DEFAULT_DATA_DIR,
        use_mask: bool = False,
        k=None,
        modulus: int = MODULUS
    ):
        # 数据集划分主方法 支持百分比划分/标签值过滤划分 双模式
        assert (0 < train_pct) and (train_pct < 100)
        ds_name = cls.get_dsname(operator, operand_length)
        eqs = cls.make_data(operator, operand_length, k=k, modulus=modulus)
        tokenizer = ArithmeticTokenizer(data_dir)

        if use_mask:
            train_eqs, val_eqs = cls._filter_by_mask(eqs, tokenizer)
            if not train_eqs:
                print("警告：mask过滤后训练集为空！")
            if not val_eqs:
                print("警告：mask过滤后验证集为空！")
        else:
            train_rows, _ = cls.calc_split_len(train_pct, len(eqs))
            train_eqs = eqs[:train_rows]
            val_eqs = eqs[train_rows:]
            
        train_ds = cls(ds_name, train_eqs, train=True, data_dir=data_dir)
        val_ds = cls(ds_name, val_eqs, train=False, data_dir=data_dir)
        return train_ds, val_ds

    @classmethod
    def calc_split_len(cls, train_pct, ds_len):
        # 计算训练集/验证集的数量划分
        train_rows = round(ds_len * (train_pct / 100.0))
        val_rows = ds_len - train_rows
        return train_rows, val_rows

    def __init__(self, name, data: Union[Tensor, List[str]], train, data_dir) -> None:
        # 数据集初始化 完成文本数据向张量的编码
        self.tokenizer = ArithmeticTokenizer(data_dir)
        self.name = name
        self.train = train
        if isinstance(data, list):
            self.data = self.tokenizer.encode(data)
        else:
            self.data = data

    def __len__(self) -> int:
        # 返回数据集样本数量
        return self.data.shape[0]

    @classmethod
    def _make_binary_operation_data(cls, operator: str, operands=None, modulus: int = MODULUS) -> List[str]:
        # 生成二元加法等式数据 (a + b) % mod = c 固定二元格式
        operands = operands or list(range(modulus))
        tuples = itertools.product(operands, repeat=2)
        eqs = []
        for a, b in tuples:
            c = (a + b) % modulus
            eq = " ".join(map(render, [a, operator, b, "=", c]))
            eqs.append(eq)
        return eqs
    
    @staticmethod
    def _make_kary_addition_mod_data(
        k: int, 
        modulus: int = MODULUS, 
        operands: Optional[List[int]] = None
    ) -> List[str]:
        # 生成K元加法取模等式数据 核心兼容k≥2 统一等式格式 解决token解析问题
        operands = operands or list(range(modulus))
        k_tuples = itertools.product(operands, repeat=k)
        eqs = []

        for k_operands in k_tuples:
            sum_result = sum(k_operands) % modulus
            sum_result_mod = Mod(sum_result, modulus)
            operand_strs = [str(op) for op in k_operands]
            lhs_str = " + ".join(operand_strs)
            eq_str = f"{lhs_str} {EQ_TOKEN} {render(sum_result_mod)}"
            eqs.append(eq_str)
        return eqs

    @classmethod
    def get_dsname(cls, operator, operand_length) -> str:
        # 生成规范的数据集名称
        operator, noise_level = cls._get_operator_and_noise_level(operator)
        ds_name = VALID_OPERATORS[operator]
        if operand_length is not None:
            ds_name += f"_length-{operand_length}"
        if noise_level > 0:
            ds_name += f"_noise-{noise_level}"
        return ds_name

    @classmethod
    def get_file_path(cls, operator, operand_length=None, data_dir=DEFAULT_DATA_DIR):
        # 获取数据集文件的完整路径+名称
        ds_name = cls.get_dsname(operator, operand_length)
        ds_file = bf.join(data_dir, f"{ds_name}_data.txt")
        return ds_file, ds_name

    @classmethod
    def _get_operator_and_noise_level(cls, operator):
        # 解析算子名+噪声等级 兼容带噪声的算子格式
        if "_noisy" in operator:
            operator, noise_level = operator.split("_noisy_")
            return operator, int(noise_level)
        else:
            return operator, 0

    @classmethod
    def make_data(cls, operator, operand_length=None, operands=None, shuffle=True, seed=0, k: int = 2, modulus: int = MODULUS) -> List[str]:
        # 数据生成总入口 分发二元加法/K元加法分支 统一后处理逻辑
        operator, noise_level = cls._get_operator_and_noise_level(operator)
        assert operator in VALID_OPERATORS, f"仅支持加法算子：{VALID_OPERATORS.keys()}"
        
        data = []
        if operator == "+k_mod_97":
            data = cls._make_kary_addition_mod_data(k=k, modulus=modulus, operands=operands)
        elif operator == "+":
            data = cls._make_binary_operation_data(operator, operands=operands, modulus=modulus)
        
        # 数据随机洗牌
        rng = np.random.RandomState(seed=seed)
        if shuffle and len(data) > 0:
            rng.shuffle(data)
        
        # 注入噪声数据（按需启用）
        if noise_level > 0 and len(data) > 0:
            random_answer_eqns = rng.choice(data, size=min(noise_level, len(data)))
            random_answers = [random_eq.split(f" {EQ_TOKEN} ")[1] for random_eq in random_answer_eqns]
            for i in range(min(noise_level, len(data))):
                data[i] = data[i].split(f" {EQ_TOKEN} ")[0] + f" {EQ_TOKEN} " + random_answers[i]
        
        # 统一添加EOS结束符 标准化等式格式
        data = [f"{EOS_TOKEN} {eq} {EOS_TOKEN}" for eq in data]
        return data

class ArithmeticIterator(torch.utils.data.IterableDataset):
    # PyTorch标准数据集迭代器 支持批次加载、数据洗牌、设备部署 适配训练流程
    def __init__(
        self,
        dataset: ArithmeticDataset,
        device: torch.device,
        batchsize_hint: float = 0,
        shuffle: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batchsize = self.calculate_batchsize(len(dataset), batchsize_hint=batchsize_hint)
        self.device = device
        self.reset_iteration(shuffle=shuffle)

    @staticmethod
    def calculate_batchsize(ds_size: int, batchsize_hint: int = 0) -> int:
        # 智能计算批次大小 适配多种入参规则
        if batchsize_hint == -1:
            return int(ds_size)
        elif batchsize_hint == 0:
            return int(min(512, math.ceil(ds_size / 2.0)))
        elif (batchsize_hint > 0) and (batchsize_hint < 1):
            return int(math.ceil(ds_size * batchsize_hint))
        elif batchsize_hint > 1:
            return int(min(batchsize_hint, ds_size))
        else:
            raise ValueError("batchsize_hint must be >= -1")

    def reset_iteration(self, shuffle=True):
        # 重置迭代器索引 训练集随机洗牌 验证集顺序遍历
        self.index = 0
        if shuffle and self.dataset.train:
            self.permutation = torch.randperm(len(self.dataset))
        else:
            self.permutation = torch.arange(len(self.dataset))

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, Tensor]:
        # 生成单批次数据 text:输入序列 target:预测序列(位移一位)
        batch_begin = self.index * self.batchsize
        if batch_begin > len(self.dataset) - 1:
            self.reset_iteration()
            raise StopIteration
        batchsize_int = int(self.batchsize)
        indices = self.permutation[batch_begin : batch_begin + self.batchsize]
        text = self.dataset.data[indices, :-1]
        target = self.dataset.data[indices, 1:]
        batch = {"text": text.to(self.device), "target": target.to(self.device)}
        self.index += 1
        return batch

    def __len__(self) -> int:
        # 返回批次总数
        return math.ceil(len(self.dataset) / self.batchsize)