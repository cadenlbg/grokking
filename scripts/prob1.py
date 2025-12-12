import argparse
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def load_csv_data(csv_path):
    """加载CSV中的global_step、train_accuracy、val_accuracy（适配你的数据格式）"""
    train_steps, train_acc = [], []
    val_steps, val_acc = [], []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 提取step（你的CSV列名为global_step）
            try:
                step = int(row["global_step"])
            except (ValueError, KeyError):
                continue  # 跳过step异常行
            
            # 提取train_accuracy（过滤空值和非数值）
            if row.get("train_accuracy") and row["train_accuracy"].replace('.', '').replace('-', '').isdigit():
                train_steps.append(step)
                train_acc.append(float(row["train_accuracy"]))
            
            # 提取val_accuracy（过滤空值和非数值）
            if row.get("val_accuracy") and row["val_accuracy"].replace('.', '').replace('-', '').isdigit():
                val_steps.append(step)
                val_acc.append(float(row["val_accuracy"]))
    
    # 生成默认标题（无hparams时用CSV文件名）
    file_name = os.path.splitext(os.path.basename(csv_path))[0]
    task_name = file_name.replace("metrics_", "").replace("_", " ").title()
    
    return {
        "task": task_name,
        "train_pct": 50,  # 默认训练数据占比（可按需修改）
        "train": (train_steps, train_acc),
        "val": (val_steps, val_acc)
    }

def plot_train_val_acc(data, save_path):
    """完全复刻参考代码的图表样式"""
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # 绘制曲线（保持参考代码的颜色和线条样式）
    train_steps, train_acc = data["train"]
    val_steps, val_acc = data["val"]
    ax.plot(train_steps, train_acc, color="#e53935", label="train", linewidth=1.2)  # 红色train线
    ax.plot(val_steps, val_acc, color="#1e88e5", label="val", linewidth=1.2)    # 蓝色val线

    # 样式配置（完全匹配参考代码）
    ax.set_xscale("log")  # x轴对数刻度
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())  # y轴百分比显示
    ax.set_ylim(0, 105)  # y轴范围（留5%边距）
    ax.grid(True, which="both", alpha=0.3, linestyle="--")  # 虚实网格线
    ax.set_title(
        f"{data['task']} (training on {data['train_pct']}% of data)",
        fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Optimization Steps", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.legend(loc="lower right", fontsize=12)

    # 保存图片（高清晰度）
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ 已保存：{save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", default="temp_result", help="CSV文件根目录（默认：temp_result）")
    parser.add_argument("-o", "--output_dir", default="my_visualization_results", help="图片保存目录（默认：my_visualization_results）")
    args = parser.parse_args()

    # 遍历temp_result下所有CSV文件（保持目录结构）
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".csv"):
                # 加载当前CSV数据
                csv_path = os.path.join(root, file)
                data = load_csv_data(csv_path)
                
                # 构建保存路径（复刻原目录结构）
                relative_dir = os.path.relpath(root, args.input_dir)
                save_dir = os.path.join(args.output_dir, relative_dir)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{os.path.splitext(file)[0]}_acc.png")
                
                # 绘图并保存
                plot_train_val_acc(data, save_path)

    print(f"\n🎉 所有图表已保存至：{args.output_dir}")

if __name__ == "__main__":
    main()