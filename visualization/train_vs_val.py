import argparse
import os
import csv
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.signal import savgol_filter  # 核心：Savitzky-Golay滤波器

def load_hparams(hparams_path):
    """读取hparams.yaml中的运算类型和训练数据占比"""
    try:
        with open(hparams_path, "r", encoding="utf-8") as f:
            hparams = yaml.safe_load(f)
        # 运算类型映射（修正拼写错误）
        task_map = {
            "+": "addition", "-": "subtraction", "*": "multiplication", "/": "division",
            "**2+": "squarepoly", "**3+": "cubepoly",
            "x**2+y**2_mod_97": "quad1", "x**2+y**2+x*y_mod_97": "quad2",
            "x**2+y**2+x*y+x_mod_97": "quad3", "x**3+x*y_mod_97": "cube1",
            "x**3+x*y**2+y_mod_97": "cube2", "(x._value//y)if(y._value%2==1)else(x-y)_mod_97": "mix1",
            "s5": "s5", "s5conj": "s5conj", "s5aba": "s5aba",
            "+*": "even-addition_odd-multiplication", "+-": "even-addition_odd-subtraction",
            "sort": "sort", "reverse": "reverse", "copy": "copy",
        }
        math_op = hparams.get("math_operator", "+").strip()
        task_name = task_map.get(math_op, "Arithmetic Task")
        train_pct = hparams.get("train_data_pct", 50)
        return task_name, train_pct
    except Exception as e:
        print(f"⚠️ 读取hparams.yaml失败：{e}")
        return "Arithmetic Task", 50  # 异常时默认值

def load_csv_data(csv_path):
    """适配目录结构：加载CSV数据+对应hparams信息"""
    metrics_dir = os.path.dirname(csv_path)  # CSV所在的metrics文件夹
    exp_dir = os.path.dirname(metrics_dir)   # 实验根目录（含hparams.yaml）
    hparams_path = os.path.join(exp_dir, "hparams.yaml")
    task_name, train_pct = load_hparams(hparams_path)
    
    # 读取并过滤step和准确率数据
    train_steps, train_acc = [], []
    val_steps, val_acc = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                step = int(row["global_step"])
            except (ValueError, KeyError):
                continue
            # 过滤空值和无效数值
            if row.get("train_accuracy") and row["train_accuracy"].replace('.', '').isdigit():
                train_steps.append(step)
                train_acc.append(float(row["train_accuracy"]))
            if row.get("val_accuracy") and row["val_accuracy"].replace('.', '').isdigit():
                val_steps.append(step)
                val_acc.append(float(row["val_accuracy"]))
    
    return {
        "task": task_name,
        "train_pct": train_pct,
        "train": (train_steps, train_acc),
        "val": (val_steps, val_acc)
    }

def sg_smooth_data(steps, acc, window_length=15, polyorder=2):
    """
    Savitzky-Golay滤波器平滑数据（核心平滑函数）
    :param steps: 原始步骤序列（列表）
    :param acc: 原始准确率序列（列表）
    :param window_length: 滤波窗口长度（必须为奇数，越大越平滑）
    :param polyorder: 多项式拟合阶数（通常2-3）
    :return: 平滑后的steps（numpy数组）和acc（numpy数组）
    """
    # 数据量不足时，不进行滤波，直接返回原始数据（转为numpy数组）
    if len(acc) < window_length or window_length % 2 == 0:
        # 自动调整窗口长度为有效奇数（不超过数据长度）
        window_length = min(len(acc), window_length)
        window_length = window_length if window_length % 2 == 1 else window_length - 1
        if window_length < 3:
            return np.array(steps), np.array(acc)
    
    # 先排序（避免step乱序影响滤波效果）
    df = pd.DataFrame({"step": steps, "acc": acc}).sort_values("step").reset_index(drop=True)
    smooth_acc = savgol_filter(df["acc"].values, window_length=window_length, polyorder=polyorder)
    # 限制准确率不超过100%（避免滤波后异常值）
    smooth_acc = np.clip(smooth_acc, 0, 100.0)
    
    return df["step"].values, smooth_acc

def get_val_convergence_step(val_steps, val_acc, threshold=99.5):
    """
    专门获取Val_acc首次达到阈值的步骤（修复numpy数组判断歧义）
    :param val_steps: 平滑后的验证步骤序列（numpy数组）
    :param val_acc: 平滑后的验证准确率序列（numpy数组）
    :param threshold: 收敛阈值（默认99.5%）
    :return: 收敛步骤（未达标返回None）
    """
    # 修复：用size判断numpy数组是否为空，而非直接not val_steps
    if val_steps.size == 0 or val_acc.size == 0:
        return None
    for step, accuracy in zip(val_steps, val_acc):
        if accuracy >= threshold:
            return step
    return None  # Val_acc未达到99.5%，返回None

def plot_train_val_acc(data, save_path, sg_window=15, sg_poly=2, val_conv_threshold=99.5):
    """
    绘制SG滤波平滑后的对比图（无实线虚线区分）
    x轴自适应逻辑：Val_acc达标则用收敛步骤*1.2，未达标则用max_steps
    修复所有numpy数组布尔判断歧义问题
    """
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # 提取原始数据并进行Savitzky-Golay平滑
    train_steps, train_acc = data["train"]
    val_steps, val_acc = data["val"]
    train_steps_sm, train_acc_sm = sg_smooth_data(train_steps, train_acc, window_length=sg_window, polyorder=sg_poly)
    val_steps_sm, val_acc_sm = sg_smooth_data(val_steps, val_acc, window_length=sg_window, polyorder=sg_poly)

    # 绘制曲线（均为实线，仅用颜色区分train/val）
    ax.plot(train_steps_sm, train_acc_sm, color="#e53935", label="train", linewidth=1.5)
    ax.plot(val_steps_sm, val_acc_sm, color="#1e88e5", label="val", linewidth=1.5)

    # ========== 核心：x轴自适应（修复numpy数组判断问题） ==========
    # 1. 获取Val_acc收敛步骤（阈值99.5%）
    val_conv_step = get_val_convergence_step(val_steps_sm, val_acc_sm, threshold=val_conv_threshold)
    # 2. 确定所有步骤的最大值（max_steps）
    all_steps = []
    # 修复：用size判断numpy数组是否非空
    if train_steps_sm.size > 0:
        all_steps.append(np.max(train_steps_sm))
    if val_steps_sm.size > 0:
        all_steps.append(np.max(val_steps_sm))
    # 兜底：避免all_steps为空
    max_steps = np.max(all_steps) if len(all_steps) > 0 else 1000
    # 3. 确定x轴上下限
    min_x = 1  # 对数坐标下避免0值
    if train_steps_sm.size > 0:
        min_x = min(min_x, np.min(train_steps_sm))
    if val_steps_sm.size > 0:
        min_x = min(min_x, np.min(val_steps_sm))
    
    # 关键判断：Val_acc达标则用收敛步骤*1.2，未达标则用max_steps
    if val_conv_step is not None:
        max_x = val_conv_step * 1.2  # 收敛后，预留20%空间
    else:
        max_x = max_steps  # Val_acc未达标，显示到最大步骤

    # 图表样式配置
    ax.set_xscale("log")
    ax.set_xlim(left=min_x, right=max_x)  # 自适应x轴范围
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())  # y轴百分比格式
    ax.set_ylim(0, 105)  # y轴范围固定（0-105%，预留少量空间）
    ax.grid(True, which="both", alpha=0.3, linestyle="--")  # 网格线
    # 标题显示train_data_pct
    ax.set_title(
        f"{data['task']} (Training on {data['train_pct']}% of Data)",
        fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Optimization Steps", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.legend(loc="lower right", fontsize=12, framealpha=0.9)

    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"图表已保存：{save_path}")

# ========== 封装 add_parser 函数（核心新增） ==========
def add_parser(subparsers):
    """
    封装train_vs_val工具的参数定义，供入口脚本调用
    :param subparsers: 入口脚本的子命令解析器
    """
    train_val_parser = subparsers.add_parser(
        "train_vs_val",
        help="绘制SG滤波平滑后的训练集/验证集准确率对比图（自动遍历metrics目录）"
    )
    # 所有参数定义集中在这里
    train_val_parser.add_argument("-i", "--input_dir", default="temp_result", help="实验数据根目录（默认：temp_result）")
    train_val_parser.add_argument("-o", "--output_dir", default="my_visualization_results", help="图片保存根目录")
    train_val_parser.add_argument("--sg_window", type=int, default=15, help="SG滤波窗口长度（奇数，越大越平滑，默认15）")
    train_val_parser.add_argument("--sg_poly", type=int, default=2, help="SG滤波多项式阶数（默认2）")
    train_val_parser.add_argument("--val_conv_threshold", type=float, default=99.5, help="Val_acc收敛阈值（默认99.5）")
    return train_val_parser

def main(args=None):
    # 若外部未传递args，则自己解析（兼容独立运行）
    if args is None:
        parser = argparse.ArgumentParser()
        # 调用自身的add_parser逻辑（复用参数定义，避免冗余）
        add_parser(parser.add_subparsers(dest="tool", required=False))
        args = parser.parse_args()

    # 遍历所有metrics子文件夹下的CSV文件
    for root, dirs, files in os.walk(args.input_dir):
        if os.path.basename(root) == "metrics":  # 仅处理metrics文件夹
            for file in files:
                if file.endswith(".csv"):  # 仅处理CSV文件
                    csv_path = os.path.join(root, file)
                    try:
                        # 加载数据
                        data = load_csv_data(csv_path)
                        # 构建保存目录（复刻原始目录结构）
                        relative_dir = os.path.relpath(root, args.input_dir)
                        save_dir = os.path.join(args.output_dir, relative_dir)
                        os.makedirs(save_dir, exist_ok=True)
                        # 构建保存文件名
                        save_filename = f"{os.path.splitext(file)[0]}_sg_smooth_accuracy.png"
                        save_path = os.path.join(save_dir, save_filename)
                        # 绘图保存
                        plot_train_val_acc(
                            data, save_path,
                            sg_window=args.sg_window,
                            sg_poly=args.sg_poly,
                            val_conv_threshold=args.val_conv_threshold
                        )
                    except Exception as e:
                        print(f"处理文件{csv_path}失败：{e}")

    print(f"所有图表已保存至根目录：{args.output_dir}")

if __name__ == "__main__":
    main()