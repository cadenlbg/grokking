import argparse
import os
import csv
import yaml
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def load_experiment_data(expt_dir):
    """加载单个实验的train/val准确率和任务信息"""
    # 读取hparams获取任务名、训练数据占比
    hparams_path = os.path.join(expt_dir, "default", "version_0", "hparams.yaml")
    with open(hparams_path, "r", encoding="utf-8") as f:
        hparams = yaml.safe_load(f)
    # 映射任务名（匹配示例图风格）
    task_map = {
        "+": "Addition",
        "%": "Modular Division",
        "-": "Subtraction",
        "*": "Multiplication"
    }
    task_name = task_map.get(hparams.get("math_operator", "+"), "Arithmetic Task")
    train_pct = hparams.get("train_data_pct", 50)

    # 读取metrics.csv中的step、train_acc、val_acc
    metrics_path = os.path.join(expt_dir, "default", "version_0", "metrics.csv")
    train_steps, train_acc = [], []
    val_steps, val_acc = [], []
    with open(metrics_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            # 收集训练准确率（非空行）
            if row["train_accuracy"]:
                train_steps.append(step)
                train_acc.append(float(row["train_accuracy"]))
            # 收集验证准确率（非空行）
            if row["val_accuracy"]:
                val_steps.append(step)
                val_acc.append(float(row["val_accuracy"]))

    return {
        "task": task_name,
        "train_pct": train_pct,
        "train": (train_steps, train_acc),
        "val": (val_steps, val_acc)
    }


def plot_train_val_acc(data, save_dir):
    """绘制与示例图一致的train/val准确率对比图"""
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # 绘制曲线（匹配示例风格，调整颜色）
    train_steps, train_acc = data["train"]
    val_steps, val_acc = data["val"]
    ax.plot(train_steps, train_acc, color="#e53935", label="train", linewidth=1.2)  # 训练：红色
    ax.plot(val_steps, val_acc, color="#1e88e5", label="val", linewidth=1.2)    # 验证：蓝色

    # 样式配置（完全匹配示例）
    ax.set_xscale("log")  # x轴对数刻度
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())  # y轴百分比显示
    ax.set_ylim(0, 105)  # y轴范围
    ax.grid(True, which="both", alpha=0.3, linestyle="--")  # 网格线
    ax.set_title(
        f"{data['task']} (training on {data['train_pct']}% of data)",
        fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Optimization Steps", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.legend(loc="lower right", fontsize=12)

    # 保存图片
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "train_val_accuracy.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ 对比图已保存至：{save_path}")


def main():
    # 仅保留-i/-o参数（无额外功能）
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True, help="实验数据目录（如./temp_result）")
    parser.add_argument("-o", "--output_dir", required=True, help="图片保存目录（如./my_visualization_results）")
    args = parser.parse_args()

    # 加载实验数据（默认取input_dir下第一个实验目录）
    expt_dirs = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    if not expt_dirs:
        raise ValueError(f"❌ {args.input_dir}中未找到实验目录")
    expt_data = load_experiment_data(os.path.join(args.input_dir, expt_dirs[0]))

    # 绘图并保存
    plot_train_val_acc(expt_data, args.output_dir)


if __name__ == "__main__":
    main()