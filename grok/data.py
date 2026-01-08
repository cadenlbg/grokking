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

# ===================== 仅保留【加法相关】算子 =====================
VALID_OPERATORS = {
    # 二元加法（基础）
    "+": "addition",
    # K元加法取模（核心保留，兼容k≥2）
    "+k_mod_97": "k-addition_mod_97",
}

# 固定常量
EOS_TOKEN = "<|eos|>"
EQ_TOKEN = "="
MODULUS = 97  # 兜底默认值，实际使用 train.py传入的--modulus参数
NUMS = list(range(MODULUS)) # 兜底数字列表，实际动态生成
DEFAULT_DATA_DIR = "data"

# 把不同操作数或对象渲染成字符串
def render(operand, join_str=""):
    if (
        isinstance(operand, list)
        or isinstance(operand, tuple)
        or isinstance(operand, np.ndarray)
    ):
        return join_str.join(map(render, operand))
    elif isinstance(operand, Mod):
        return str(operand._value)
    else:
        return str(operand)

#调用以下函数生成数据文件
def create_data_files(data_dir: str = DEFAULT_DATA_DIR):
    ArithmeticTokenizer.create_token_file(data_dir)
    ArithmeticDataset.create_dataset_files(data_dir)

# 词法分析器类,用于存储token文本到token id的映射并进行转换
class ArithmeticTokenizer:
    """Stores the list of token text to token id mappings and converts between them"""
    token_file = "tokens.txt"

    def __init__(self, data_dir=DEFAULT_DATA_DIR) -> None:
        self.token_file = bf.join(data_dir, self.token_file)
        self.itos = self.get_tokens()
        self.stoi: Dict[str, int] = dict([(s, i) for i, s in enumerate(self.itos)])
    
    # 正向翻译: 文本→id
    def _encode(self, s: str) -> Tensor:
        return LongTensor([self.stoi[t] for t in s.split(" ")])

    def encode(self, obj: Union[str, List]) -> Tensor:
        if isinstance(obj, str):
            return self._encode(obj)
        elif isinstance(obj, list):
            return torch.stack([self._encode(s) for s in obj], dim=0)
        else:
            raise NotImplementedError
    
    # 反向翻译: id→文本
    def decode(self, tensor: Tensor, with_brackets: bool = False) -> str:
        indices = tensor.long()
        if with_brackets:
            l = "<"
            r = ">"
        else:
            l = ""
            r = ""
        tokens = [l + self.itos[i] + r for i in indices]
        return " ".join(tokens)

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    # 构建词汇表 - 仅加法相关token，无冗余
    def get_tokens(cls):
        tokens = (
            [EOS_TOKEN, EQ_TOKEN]
            + list(sorted(list(VALID_OPERATORS.keys())))
            + list(map(render, NUMS))
        )
        return tokens


class ArithmeticDataset:
    """A Dataset of arithmetic equations - 仅加法数据集"""

    @staticmethod
    def _extract_label(eq_str: str, tokenizer: ArithmeticTokenizer) -> Optional[int]:
        """【修复点1：兼容k元加法等式】从等式字符串中提取标签（加法结果），适配任意个+号的等式"""
        eq_clean = eq_str.replace(EOS_TOKEN, "").strip()
        tokens = eq_clean.split(" ")
        if EQ_TOKEN not in tokens:
            return None
        
        # 核心兼容：无论等式左边有多少个+号，只取=后面的所有内容作为标签
        eq_idx = tokens.index(EQ_TOKEN)
        label_tokens = tokens[eq_idx+1:]
        if not label_tokens:
            return None
        
        label_str = "".join(label_tokens).strip() # 修复：用join无空格，适配k加法的纯数字标签
        try:
            return int(label_str)
        except ValueError:
            if label_str.startswith("Mod(") and label_str.endswith(")"):
                return int(label_str.split(",")[0].split("(")[1])
            return None

    # 按mask规则过滤数据：train(label>20) / val(label<20)
    @staticmethod
    def _filter_by_mask(eqs: List[str], tokenizer: ArithmeticTokenizer) -> Tuple[List[str], List[str]]:
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
    # 创建训练集和验证集 - 保留动态模数+K参数
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
    # 计算训练集和验证集的划分长度
    def calc_split_len(cls, train_pct, ds_len):
        train_rows = round(ds_len * (train_pct / 100.0))
        val_rows = ds_len - train_rows
        return train_rows, val_rows

    def __init__(self, name, data: Union[Tensor, List[str]], train, data_dir) -> None:
        self.tokenizer = ArithmeticTokenizer(data_dir)
        self.name = name
        self.train = train
        if isinstance(data, list):
            self.data = self.tokenizer.encode(data)
        else:
            self.data = data

    def __len__(self) -> int:
        return self.data.shape[0]

    @classmethod
    # ===================== 仅保留【二元加法】核心逻辑 =====================
    def _make_binary_operation_data(cls, operator: str, operands=None, modulus: int = MODULUS) -> List[str]:
        # 仅处理加法 + 
        operands = operands or list(range(modulus))
        tuples = itertools.product(operands, repeat=2)
        eqs = []
        for a, b in tuples:
            c = (a + b) % modulus  # 二元加法取模
            eq = " ".join(map(render, [a, operator, b, "=", c]))
            eqs.append(eq)
        return eqs
    
    @staticmethod
    # ===================== 核心保留【K元加法】逻辑 + 【修复点2：关键修复】 =====================
    def _make_kary_addition_mod_data(
        k: int, 
        modulus: int = MODULUS, 
        operands: Optional[List[int]] = None
    ) -> List[str]:
        """生成k个数相加后取模的等式数据集（k≥2，兼容二元/多元）【修复：统一等式格式+Mod对象渲染】"""
        operands = operands or list(range(modulus))
        k_tuples = itertools.product(operands, repeat=k)
        eqs = []

        for k_operands in k_tuples:
            sum_result = sum(k_operands) % modulus
            sum_result_mod = Mod(sum_result, modulus) # 修复：统一用Mod对象，和二元加法格式一致
            operand_strs = [str(op) for op in k_operands]
            lhs_str = " + ".join(operand_strs)
            # 修复：等式格式统一为「数字 + 数字 + ... = 结果」，和二元加法完全一致的token分割规则
            eq_str = f"{lhs_str} {EQ_TOKEN} {render(sum_result_mod)}"
            eqs.append(eq_str)
        return eqs

    @classmethod
    def get_dsname(cls, operator, operand_length) -> str:
        operator, noise_level = cls._get_operator_and_noise_level(operator)
        ds_name = VALID_OPERATORS[operator]
        if operand_length is not None:
            ds_name += f"_length-{operand_length}"
        if noise_level > 0:
            ds_name += f"_noise-{noise_level}"
        return ds_name

    @classmethod
    def get_file_path(cls, operator, operand_length=None, data_dir=DEFAULT_DATA_DIR):
        ds_name = cls.get_dsname(operator, operand_length)
        ds_file = bf.join(data_dir, f"{ds_name}_data.txt")
        return ds_file, ds_name

    @classmethod
    def _get_operator_and_noise_level(cls, operator):
        if "_noisy" in operator:
            operator, noise_level = operator.split("_noisy_")
            return operator, int(noise_level)
        else:
            return operator, 0

    @classmethod
    # ===================== 主数据生成函数 - 仅加法分支 + 【修复点3：兼容k加法的算子名】 =====================
    def make_data(cls, operator, operand_length=None, operands=None, shuffle=True, seed=0, k: int = 2, modulus: int = MODULUS) -> List[str]:
        operator, noise_level = cls._get_operator_and_noise_level(operator)
        assert operator in VALID_OPERATORS, f"仅支持加法算子：{VALID_OPERATORS.keys()}"
        
        data = []
        # 分支1：K元加法 (优先级更高)
        if operator == "+k_mod_97":
            data = cls._make_kary_addition_mod_data(k=k, modulus=modulus, operands=operands)
        # 分支2：二元加法
        elif operator == "+":
            data = cls._make_binary_operation_data(operator, operands=operands, modulus=modulus)
        
        # 数据洗牌
        rng = np.random.RandomState(seed=seed)
        if shuffle and len(data) > 0:
            rng.shuffle(data)
        
        # 噪声注入（保留，按需启用）
        if noise_level > 0 and len(data) > 0:
            random_answer_eqns = rng.choice(data, size=min(noise_level, len(data)))
            random_answers = [random_eq.split(f" {EQ_TOKEN} ")[1] for random_eq in random_answer_eqns]
            for i in range(min(noise_level, len(data))):
                data[i] = data[i].split(f" {EQ_TOKEN} ")[0] + f" {EQ_TOKEN} " + random_answers[i]
        
        # 添加EOS标记 【修复：统一空格分割规则，彻底解决token解析问题】
        data = [f"{EOS_TOKEN} {eq} {EOS_TOKEN}" for eq in data]
        return data


class ArithmeticIterator(torch.utils.data.IterableDataset):
    """数据集迭代器 - 完整保留，兼容批次配置"""
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
        self.index = 0
        if shuffle and self.dataset.train:
            self.permutation = torch.randperm(len(self.dataset))
        else:
            self.permutation = torch.arange(len(self.dataset))

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, Tensor]:
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
        return math.ceil(len(self.dataset) / self.batchsize)