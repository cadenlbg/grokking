#!/usr/bin/env python
import grok
import os
from pathlib import Path

# 运算符名称映射字典
OPERATOR_MAPPING = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
    "**2+": "square_poly",
    "**3+": "cube_poly",
    "x**2+y**2_mod_97": "quad1",
    "x**2+y**2+x*y_mod_97": "quad2",
    "x**2+y**2+x*y+x_mod_97": "quad3",
    "x**3+x*y_mod_97": "cube1",
    "x**3+x*y**2+y_mod_97": "cube2",
    "(x._value//y)if(y._value%2==1)else(x-y)_mod_97": "mix1",
    "s5": "s5",
    "s5conj": "s5_conj",
    "s5aba": "s5_aba",
    "+*": "even_add_odd_mul",
    "+-": "even_add_odd_sub",
    "sort": "sort",
    "reverse": "reverse",
    "copy": "copy"
}

# 1. 解析参数
parser = grok.training.add_args()
hparams = parser.parse_args()

# 2. 提取关键参数构造实验子目录名
pct = hparams.train_data_pct
operator_name = OPERATOR_MAPPING.get(hparams.math_operator, hparams.math_operator)
max_steps = hparams.max_steps
random_seed = hparams.random_seed
exp_name = f"mlp_exp_pct{pct}_op{operator_name}_steps{max_steps}_random_seed{random_seed}"

# 3. 以用户指定logdir为根目录，创建唯一子目录（解决路径覆盖）
user_logdir = Path(hparams.logdir)
exp_dir = user_logdir / exp_name
exp_dir.mkdir(parents=True, exist_ok=True)

# 4. 更新logdir为最终实验目录
hparams.logdir = str(exp_dir.resolve())
hparams.datadir = os.path.abspath(hparams.datadir)

# 5. 仅打印必要信息
print(f"实验目录: {hparams.logdir}")
print(hparams)

# 6. 启动训练
print(grok.training.train(hparams))