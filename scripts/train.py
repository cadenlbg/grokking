#!/usr/bin/env python
import grok
import os
from pathlib import Path  # 用于路径处理更方便

OPERATOR_MAPPING = {
    "+": "add",                  # addition → 缩写add
    "-": "sub",                  # subtraction → 缩写sub
    "*": "mul",                  # muliplication → 缩写mul（修正原字典拼写）
    "/": "div",                  # division → 缩写div
    "**2+": "square_poly",       # squarepoly → 保留核心
    "**3+": "cube_poly",         # cubepoly → 保留核心
    "x**2+y**2_mod_97": "quad1", # 保持原名称
    "x**2+y**2+x*y_mod_97": "quad2",
    "x**2+y**2+x*y+x_mod_97": "quad3",
    "x**3+x*y_mod_97": "cube1",
    "x**3+x*y**2+y_mod_97": "cube2",
    "(x._value//y)if(y._value%2==1)else(x-y)_mod_97": "mix1",
    "s5": "s5",
    "s5conj": "s5_conj",         # 下划线分隔，更清晰
    "s5aba": "s5_aba",
    "+*": "even_add_odd_mul",    # 替换连字符为下划线
    "+-": "even_add_odd_sub",
    "sort": "sort",
    "reverse": "reverse",
    "copy": "copy"
}

parser = grok.training.add_args()
# 解析参数（先解析，才能获取到train_data_pct等参数值）
hparams = parser.parse_args()

# 1. 提取用于命名的关键参数（处理特殊字符和格式）
pct = hparams.train_data_pct
operator_name = OPERATOR_MAPPING.get(hparams.math_operator, hparams.math_operator)
max_steps = hparams.max_steps
random_seed=hparams.random_seed

# 2. 构造新文件夹名称（包含关键参数信息）
exp_name = f"mlp_exp_pct{pct}_op{operator_name}_steps{max_steps}_random_seed{random_seed}"

# 3. 定义根目录my_experiments，并创建完整路径
root_dir = Path("my_experiments/mlp")
exp_dir = root_dir / exp_name  # 拼接路径：my_experiments/exp_pct30.0_opadd_steps100000...

# 4. 创建文件夹（如果不存在）
exp_dir.mkdir(parents=True, exist_ok=True)  # parents=True确保父目录也会被创建

# 5. 更新hparams的logdir和datadir到新路径（确保文件存到新文件夹）
hparams.logdir = str(exp_dir.resolve())  # 转为绝对路径
hparams.datadir = os.path.abspath(hparams.datadir)  # 保持原datadir处理逻辑

print(f"实验目录: {hparams.logdir}")
print(hparams)
print(grok.training.train(hparams))