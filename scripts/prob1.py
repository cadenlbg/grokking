import argparse
import os
import csv
import yaml
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def load_hparams(hparams_path):
    """读取hparams.yaml中的运算类型和训练数据占比"""
    try:
        with open(hparams_path, "r", encoding="utf-8") as f:
            hparams = yaml.safe_load(f)
        # 运算类型映射（匹配参考图）
        task_map = {
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
        math_op = hparams.get("math_operator", "+").strip()
        task_name = task_map.get(math_op, "Arithmetic Task")
        train_pct = hparams.get("train_data_pct", 50)
        return task_name, train_pct
    except Exception as e:
        print(f"⚠️ 读取hparams.yaml失败：{e}")
        return "Arithmetic Task", 50  # 异常时默认值

def load_csv_data(csv_path):
    """适配目录结构：CSV在metrics子文件夹，hparams在实验根目录"""
    # 1. 获取实验根目录（metrics文件夹的上层目录）
    metrics_dir = os.path.dirname(csv_path)  # 当前CSV所在的metrics文件夹
    exp_dir = os.path.dirname(metrics_dir)   # 实验根目录（含hparams.yaml）
    # 2. 读取实验根目录下的hparams.yaml
    hparams_path = os.path.join(exp_dir, "hparams.yaml")
    task_name, train_pct = load_hparams(hparams_path)
    
    # 3. 读取CSV中的step和准确率数据
    train_steps, train_acc = [], []
    val_steps, val_acc = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 提取global_step
            try:
                step = int(row["global_step"])
            except (ValueError, KeyError):
                continue
            # 收集train_accuracy（过滤空值）
            if row.get("train_accuracy") and row["train_accuracy"].replace('.', '').isdigit():
                train_steps.append(step)
                train_acc.append(float(row["train_accuracy"]))
            # 收集val_accuracy（过滤空值）
            if row.get("val_accuracy") and row["val_accuracy"].replace('.', '').isdigit():
                val_steps.append(step)
                val_acc.append(float(row["val_accuracy"]))
    
    return {
        "task": task_name,
        "train_pct": train_pct,
        "train": (train_steps, train_acc),
        "val": (val_steps, val_acc)
    }

def plot_train_val_acc(data, save_path):
    """保持原绘图样式（标题、刻度、颜色等）"""
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # 绘制train/val曲线
    train_steps, train_acc = data["train"]
    val_steps, val_acc = data["val"]
    ax.plot(train_steps, train_acc, color="#e53935", label="train", linewidth=1.2)
    ax.plot(val_steps, val_acc, color="#1e88e5", label="val", linewidth=1.2)

    # 样式配置（完全匹配参考图）
    ax.set_xscale("log")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(0, 105)
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    ax.set_title(
        f"{data['task']} (training on {data['train_pct']}% of data)",
        fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Optimization Steps", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.legend(loc="lower right", fontsize=12)

    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ 已保存：{save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", default="temp_result", help="实验根目录（默认：temp_result）")
    parser.add_argument("-o", "--output_dir", default="my_visualization_results", help="图片保存目录")
    args = parser.parse_args()

    # 仅处理每个实验文件夹下metrics子文件夹中的CSV
    for root, dirs, files in os.walk(args.input_dir):
        # 筛选：当前目录是"metrics"子文件夹
        if os.path.basename(root) == "metrics":
            for file in files:
                if file.endswith(".csv"):
                    csv_path = os.path.join(root, file)
                    # 加载数据（含hparams信息）
                    data = load_csv_data(csv_path)
                    # 构建保存路径（复刻原实验目录结构）
                    # 示例：temp_result/exp_pct30.../metrics → my_visualization_results/exp_pct30.../metrics
                    relative_dir = os.path.relpath(root, args.input_dir)
                    save_dir = os.path.join(args.output_dir, relative_dir)
                    os.makedirs(save_dir, exist_ok=True)
                    # 图片命名：原CSV名+_accuracy.png
                    save_filename = f"{os.path.splitext(file)[0]}_accuracy.png"
                    save_path = os.path.join(save_dir, save_filename)
                    # 绘图保存
                    plot_train_val_acc(data, save_path)

    print(f"\n🎉 所有图表已保存至：{args.output_dir}")

if __name__ == "__main__":
    main()