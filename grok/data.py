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
from concurrent.futures import ProcessPoolExecutor

from sympy.combinatorics.permutations import Permutation
from mod import Mod

import blobfile as bf

#先给出所有支持的操作符及其名称
VALID_OPERATORS = {
    # 加减乘除
    "+": "addition",
    "-": "subtraction",
    "*": "muliplication",
    "/": "division",
    # 幂运算和多项式
    "**2+": "squarepoly",
    "**3+": "cubepoly",
    "x**2+y**2_mod_97": "quad1",
    "x**2+y**2+x*y_mod_97": "quad2",
    "x**2+y**2+x*y+x_mod_97": "quad3",
    "x**3+x*y_mod_97": "cube1",
    "x**3+x*y**2+y_mod_97": "cube2",
    # 混合运算
    "(x._value//y)if(y._value%2==1)else(x-y)_mod_97": "mix1",
    # 对称S5群的置换运算
    "s5": "s5",
    "s5conj": "s5conj",
    "s5aba": "s5aba",
    # 奇偶分支运算
    "+*": "even-addition_odd-multiplication",
    "+-": "even-addition_odd-subtraction",
    # 列表运算
    "sort": "sort",
    "reverse": "reverse",
    "copy": "copy",
}
#序列开始结束标记,=,模数97,数字列表
EOS_TOKEN = "<|eos|>"
EQ_TOKEN = "="
MODULUS = 97
NUMS = list(range(MODULUS))

#默认数据目录
DEFAULT_DATA_DIR = "data"

# 把不同操作数或对象渲染成字符串
def render(operand, join_str=""):
    if (
        isinstance(operand, list)
        or isinstance(operand, tuple)
        or isinstance(operand, np.ndarray)
    ):
        return join_str.join(map(render, operand))
    elif isinstance(operand, Permutation):
        return "".join(map(str, operand.array_form))
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
    ''' 初始化流程：
        1. 确定词汇表文件路径
        → 2. 调用get_tokens()生成词汇表（itos）
         → 3. 基于itos构建反向映射stoi。'''
    
    token_file = "tokens.txt"

    def __init__(self, data_dir=DEFAULT_DATA_DIR) -> None:
        self.token_file = bf.join(data_dir, self.token_file)
        # 列表: token id到token文本的映射
        self.itos = self.get_tokens()
        # 字典: token文本到token id的映射
        self.stoi: Dict[str, int] = dict([(s, i) for i, s in enumerate(self.itos)])
    
    # 正向翻译: 文本→id
    def _encode(self, s: str) -> Tensor:
        return LongTensor([self.stoi[t] for t in s.split(" ")])

    def encode(self, obj: Union[str, List]) -> Tensor:
        """
        Convert a string of text into a rank-1 tensor of token ids
        or convert a list of strings of text into a rank-2 tensor of token ids

        :param obj: the string or list of strings to convert
        :returns: a tensor of the token ids
        """
        if isinstance(obj, str):
            return self._encode(obj)
        elif isinstance(obj, list):
            return torch.stack([self._encode(s) for s in obj], dim=0)
        else:
            raise NotImplementedError
    
    # 反向翻译: id→文本
    def decode(self, tensor: Tensor, with_brackets: bool = False) -> str:
        """
        Convert a tensor of token ids into a string of text

        :param tensor: a tensor of the token ids
        :param with_brackets: if true, the returned string will include <> brackets
                              around the text corresponding to each token.
        :returns: string of these tokens.
        """
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
        """
        :returns: the number of tokens in this vocabulary
        """
        return len(self.itos)

    @classmethod
    # 构建“全量覆盖”的词汇表
    def get_tokens(cls):
        tokens = (
            [EOS_TOKEN, EQ_TOKEN]
            + list(sorted(list(VALID_OPERATORS.keys())))
            + list(map(render, NUMS))
            + list(map(render, itertools.permutations(range(5))))  # s5
        )
        return tokens


class ArithmeticDataset:
    """A Dataset of arithmetic equations"""

    @staticmethod
    def _extract_label(eq_str: str, tokenizer: ArithmeticTokenizer) -> Optional[int]:
        """
        从等式字符串中提取标签（结果），转换为数值（仅处理数值类结果）
        :param eq_str: 等式字符串（含EOS_TOKEN）
        :param tokenizer: 分词器（用于解析tokens）
        :return: 标签数值（非数值结果返回None）
        """
        # 去除EOS_TOKEN，分割为tokens
        eq_clean = eq_str.replace(EOS_TOKEN, "").strip()
        tokens = eq_clean.split(" ")
        if EQ_TOKEN not in tokens:
            return None  # 无效等式，跳过
        
        # 找到"="后面的部分（标签tokens）
        eq_idx = tokens.index(EQ_TOKEN)
        label_tokens = tokens[eq_idx+1:]
        if not label_tokens:
            return None
        
        # 合并标签tokens，处理不同格式（数字、Mod对象、字符串）
        label_str = " ".join(label_tokens).strip()
        try:
            # 处理纯数字（如"8"）
            return int(label_str)
        except ValueError:
            # 处理Mod对象（如"Mod(8,97)"）
            if label_str.startswith("Mod(") and label_str.endswith(")"):
                return int(label_str.split(",")[0].split("(")[1])
            # 非数值结果（如S5排列"01234"）返回None，不纳入mask过滤
            return None

    # 新增：按mask规则过滤数据
    @staticmethod
    def _filter_by_mask(eqs: List[str], tokenizer: ArithmeticTokenizer) -> Tuple[List[str], List[str]]:
        """
        按mask规则过滤数据：
        - train_eqs: 标签 > 20 的等式
        - val_eqs: 标签 < 20 的等式
        :return: (train_eqs, val_eqs)
        """
        train_eqs = []
        val_eqs = []
        for eq in eqs:
            label = ArithmeticDataset._extract_label(eq, tokenizer)
            if label is None:
                continue  # 非数值结果，不纳入任何集
            if label > 20:
                train_eqs.append(eq)
            elif label < 20:
                val_eqs.append(eq)
        # 打印过滤统计信息
        print(f"Mask过滤后：训练集{len(train_eqs)}条（label>20），验证集{len(val_eqs)}条（label<20）")
        return train_eqs, val_eqs
    
    @classmethod
    # 创建训练集和验证集
    def splits(
        cls,
        train_pct: float,
        operator: str,
        operand_length: Optional[int] = None,
        data_dir: str = DEFAULT_DATA_DIR,
        use_mask: bool = False,  # 新增：接收mask开关
    ):
        """
        Creates training and validation datasets

        :param train_pct: percentage of total equations used for training data
        :param operator: The arithmetic operator for this dataset e.g. '+', '-', '*', '/', 'sort'
        :param operand_length: for list based datasets the length of the lists
        :returns: (train_dataset, validation_dataset)
        """
        assert (0 < train_pct) and (train_pct < 100)
        # 生成名称
        ds_name = cls.get_dsname(operator, operand_length)
        # 生成所有等式
        eqs = cls.make_data(operator, operand_length)
        # 创建临时tokenizer用于解析标签（仅mask启用时需要）
        tokenizer = ArithmeticTokenizer(data_dir)

        if use_mask:
            # 按label过滤得到训练集和验证集
            train_eqs, val_eqs = cls._filter_by_mask(eqs, tokenizer)
            # 若过滤后训练集/验证集为空，抛出警告
            if not train_eqs:
                print("警告：mask过滤后训练集为空！请检查数据或调整mask规则")
            if not val_eqs:
                print("警告：mask过滤后验证集为空！请检查数据或调整mask规则")
        else:
            train_rows, _ = cls.calc_split_len(train_pct, len(eqs))
            train_eqs = eqs[:train_rows]
            val_eqs = eqs[train_rows:]
        #初始化train_set和val_set实例
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
        """
        :param data: A list of equations strings. Each equation must have an '=' in it.
        """
        """
        :param name: 数据集名称（如"addition"）
        :param data: 输入数据（等式文本列表 或 Token ID张量）
        :param train: 是否为训练集（影响后续迭代器的洗牌逻辑）
        :param data_dir: 分词器词汇表存储路径
    """

        self.tokenizer = ArithmeticTokenizer(data_dir)
        self.name = name
        self.train = train
        if isinstance(data, list):
            self.data = self.tokenizer.encode(data)
        else:
            self.data = data

    def __len__(self) -> int:
        """
        :returns: total number of equations in this dataset
        """
        return self.data.shape[0]

    # @classmethod
    # def _render(cls, operand):
    #    return render(operand, join_str=" ")
    #
    # @classmethod
    # def _render_eq(parts):
    #    return " ".join(map(render, parts))

    @classmethod
    def _make_binary_operation_data(cls, operator: str, operands=None) -> List[str]:
        #先生成所有运算数
        if operator == "s5":
            operands = operands or list(range(5))
            elems = map(np.array, itertools.permutations(operands))
            tuples = itertools.product(elems, repeat=2)
        elif operator in ["s5conj", "s5aba"]:
            operands = operands or list(range(5))
            elems = map(Permutation, itertools.permutations(operands))
            tuples = itertools.product(elems, repeat=2)
        elif "_mod_" in operator:
            modulo = int(operator.split("_mod_")[-1])
            elems = [Mod(i, modulo) for i in range(modulo)]
            tuples = itertools.product(elems, repeat=2)
        else:
            operands = operands or NUMS
            tuples = itertools.product(operands, repeat=2)

        # if operator == "s5":
        #     print("elems", list(elems))
        #     print("tuples", list(tuples))
        eqs = []
        for a, b in tuples:
            if operator == "/":
                if b == 0:
                    continue
                else:
                    c = a
                    a = (b * c) % MODULUS
            elif operator == "s5":
                c = b[a]
            elif operator == "s5conj":
                c = a * b * (a.__invert__())
            elif operator == "s5aba":
                c = a * b * a
            elif operator == "+*":
                if a % 2 == 0:
                    c = (a + b) % MODULUS
                else:
                    c = (a * b) % MODULUS
            elif operator == "+-":
                if a % 2 == 0:
                    c = (a + b) % MODULUS
                else:
                    c = (a - b) % MODULUS
            elif "_mod_" in operator:
                expression = operator.split("_mod_")[0]
                function = eval(f"lambda x, y: ({expression})")
                c = function(a, b)
            else:
                c = eval(f"({a} {operator} {b}) % {MODULUS}")
            eq = " ".join(map(render, [a, operator, b, "=", c]))
            eqs.append(eq)

        # if operator == "s5":
        #     print("eqs", eqs)
        return eqs

    # @staticmethod
    # def _render_unop_example(operator, lhs, rhs):
    #    return " ".join([operator, render(lhs), "=", render(rhs)])

    @staticmethod
    def _make_unary_operation_data(operator: str, operands: Tensor) -> List[str]:
        """
        :param operator: The unary operator to apply to each operand e.g. '+'
        :param operands: A tensor of operands
        :returns: list of equations"""
        num_examples = len(operands)

        if operator == "sort":
            rhs = torch.sort(operands, dim=1)[0]
        elif operator == "reverse":
            rhs = torch.flip(operands, dims=(1,))
        elif operator == "copy":
            rhs = operands
        else:
            raise Exception("unsupported operator")

        def func(L, R):
            L = map(str, L)
            R = map(str, R)
            return f"{operator} {' '.join(L)} = {' '.join(R)}"

        if num_examples < 1000000000:
            eqs = [
                func(L, R)
                for L, R in tqdm(
                    zip(operands.tolist(), rhs.tolist()), total=num_examples
                )
            ]
        else:
            with ProcessPoolExecutor() as executor:
                eqs = executor.map(func, tqdm(zip(operands, rhs), total=num_examples))

        return eqs

    # @staticmethod
    # def _make_s5_data(abstract=False) -> List[str]:
    #    elems = itertools.permutations([0, 1, 2, 3, 4])
    #    pairs = itertools.product(elems, repeat=2)
    #    eqs = []
    #    for a, b in pairs:
    #        a = np.array(a)
    #        b = np.array(b)
    #        c = b[a]
    #        eq = " ".join(map(render, (a, "s5", b, "=", c)))
    #        eq = cls._render_eq([a, , b, "=", c])
    #        eqs.append(eq)
    #
    #    return eqs

    @classmethod
    #生成标准化数据集名称（如“addition_length-2_noise-50”），用于数据集文件命名和管理。
    def get_dsname(cls, operator, operand_length) -> str:
        operator, noise_level = cls._get_operator_and_noise_level(operator)
        ds_name = VALID_OPERATORS[operator]
        if operand_length is not None:
            ds_name += f"_length-{operand_length}"
        if noise_level > 0:
            ds_name += f"_noise-{noise_level}"
        return ds_name

    @classmethod
    # 生成数据集文件路径和名称
    def get_file_path(cls, operator, operand_length=None, data_dir=DEFAULT_DATA_DIR):
        ds_name = cls.get_dsname(operator, operand_length)
        ds_file = bf.join(data_dir, f"{ds_name}_data.txt")
        return ds_file, ds_name

    @classmethod
    # 解析运算类型字符串，提取噪声注入级别
    def _get_operator_and_noise_level(cls, operator):
        if "_noisy" in operator:
            operator, noise_level = operator.split("_noisy_")
            return operator, int(noise_level)
        else:
            return operator, 0

    @classmethod
    #功能：接收运算类型，自动判断调用“二元运算生成”或“一元运算生成”方法，同时支持数据洗牌、噪声注入，返回带EOS标记的等式列表。
    def make_data(cls, operator, operands=None, shuffle=True, seed=0) -> List[str]:
        operator, noise_level = cls._get_operator_and_noise_level(operator)
        assert operator in VALID_OPERATORS
        
        # 根据运算类型调用相应的数据生成方法
        if operator not in ["sort", "reverse", "copy"]:
            data = cls._make_binary_operation_data(operator)
        else:
            data = cls._make_unary_operation_data(operator, operands)
        
        # 数据洗牌
        rng = np.random.RandomState(seed=seed)
        if shuffle:
            rng.shuffle(data)
        
        # 噪声注入
        if noise_level > 0:
            random_answer_eqns = rng.choice(data, size=noise_level)
            random_answers = [
                random_eq.split(" = ")[1] for random_eq in random_answer_eqns
            ]
            for i in range(noise_level):
                data[i] = data[i].split(" = ")[0] + " = " + random_answers[i]
        
        # 添加EOS标记
        data = [EOS_TOKEN + " " + eq + " " + EOS_TOKEN for eq in data]

        return data

    # @classmethod
    # def create_data_file(
    #    cls, operator, operand_length=None, shuffle=True, data_dir=DEFAULT_DATA_DIR
    # ):
    #    if VALID_OPERATORS[operator]["binary_eval"]:
    #        cls.write_dataset(
    #            cls.make_binary_operation_data(operator), paths["ds_file"]
    #        )
    #
    #    pass

    # @classmethod
    # def write_dataset(eqs: List[str], ds_file: str):
    #    print(f"-> writing {ds_file}", flush=True)
    #    with open(ds_file, "w") as fh:
    #        fh.writelines([EOS_TOKEN + " " + eq + " " + EOS_TOKEN + "\n" for eq in eqs])

    @classmethod
    # 构建所有可能的数字列表组合生成列表操作数（如长度为2的数字排列列表），为一元运算提供输入数据。
    def _make_lists(cls, sizes=[2, 3], nums=NUMS):
        lists: dict = {}
        for size in sizes:
            lists[size] = torch.tensor(
                list(itertools.permutations(nums, r=size)),
                dtype=torch.int,
            )
        return lists


class ArithmeticIterator(torch.utils.data.IterableDataset):
    """
    An iterator over batches of data in an ArithmeticDataset
    """

    def __init__(
        self,
        dataset: ArithmeticDataset,
        device: torch.device,
        batchsize_hint: float = 0,
        shuffle: bool = True,
    ) -> None:
        """
        :param dataset: the dataset to iterate over
        :param device: the torch device to send batches to
        :param batchsize_hint: * 0 means we use a default batchsize
                               * -1 means the entire dataset
                               * float between 0 and 1 means each batch is
                                 that fraction of the DS
                               * int > 1 means that specific batch size
        :param shuffle: whether or not to randomly shuffle the dataset
        """
        self.dataset = dataset
        self.batchsize = self.calculate_batchsize(
            len(dataset), batchsize_hint=batchsize_hint
        )
        self.device = device
        self.reset_iteration(shuffle=shuffle)

    @staticmethod
    def calculate_batchsize(ds_size: int, batchsize_hint: int = 0) -> int:
        """
        Calculates which batch size to use

        :param ds_size: the number of equations in the dataset
        :param batchsize_hint: * 0 means we use a default batchsize
                               * -1 means the entire dataset
                               * float between 0 and 1 means each batch is
                                 that fraction of the DS
                               * int > 1 means that specific batch size
        :returns: the actual batchsize to use
        """

        if batchsize_hint == -1:
            return ds_size
        elif batchsize_hint == 0:
            return min(512, math.ceil(ds_size / 2.0))
        elif (batchsize_hint > 0) and (batchsize_hint < 1):
            return math.ceil(ds_size * batchsize_hint)
        elif batchsize_hint > 1:
            return min(batchsize_hint, ds_size)
        else:
            raise ValueError("batchsize_hint must be >= -1")

    def reset_iteration(self, shuffle=True):
        self.index = 0
        if shuffle and self.dataset.train:
            self.permutation = torch.randperm(len(self.dataset))
        else:
            self.permutation = torch.arange(len(self.dataset))

    def __iter__(self):
        """
        :returns: this iterator
        """
        return self

    def __next__(self) -> Dict[str, Tensor]:
        """
        Returns one batch of data.

        :raises: StopIteration when we're out of data
        :returns: batch tensor of shape (self.batchsize, tokens_per_eq)
        """

        batch_begin = self.index * self.batchsize
        if batch_begin > len(self.dataset) - 1:
            self.reset_iteration()
            raise StopIteration
        indices = self.permutation[batch_begin : batch_begin + self.batchsize]
        text = self.dataset.data[indices, :-1]
        target = self.dataset.data[indices, 1:]
        batch = {"text": text.to(self.device), "target": target.to(self.device)}
        self.index += 1
        return batch

    def __len__(self) -> int:
        """
        :returns: the total number of batches
        """
        return math.ceil(len(self.dataset) / self.batchsize)
