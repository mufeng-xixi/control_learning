"""Plot learning curves from one or more runs' metrics.csv files.

Example:
    python -m locolab.plots runs/InvertedPendulum-v5_seed0 --out results/curve.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 使用 Agg 后端，确保在无图形界面的服务器上运行
import matplotlib.pyplot as plt


def load_metrics(run_dir: str | Path) -> tuple[list[int], list[float]]:
    """📥 从 metrics.csv 加载步数和返回。

    📝 参数详细说明:

    Args:
        run_dir (str | Path): 📁 运行目录路径。
                              - 包含 metrics.csv 文件的目录
                              - 通常是 train() 函数返回的 run_dir

    Returns:
        tuple[list[int], list[float]]: (步数列表, 返回列表)
            - steps (list[int]): 训练步数列表
              - 每个元素对应 metrics.csv 中的一行
              - 对应列: "steps" (整数)
            - rets (list[float]): 平均 episode 返回列表
              - 每个元素对应 metrics.csv 中的一行
              - 过滤规则: 仅当 "mean_ep_return" != "nan" 时包括
              - 对应列: "mean_ep_return" (浮点数)

    🔍 过滤规则:
        - 仅包含 "mean_ep_return" != "nan" 的行
        - 这确保了只有在成功收集了 episode 数据时才包含
        - NaN 可能出现在还没有收集足够 episode 的 early epochs

    📊 返回数据结构:
        - steps: 训练过程中的离散步数
        - rets: 对应的平均 episode 返回
        - 两个列表长度应相等，可用于绘制学习曲线

    💡 使用示例:
        ```python
        steps, rets = load_metrics("runs/InvertedPendulum-v5_seed0")
        # steps: [0, 4000, 8000, ...]
        # rets: [-, 100.5, 200.3, ...] (可能有 NaN 被过滤)
        ```
    """
    # 📂 初始化空列表
    steps, rets = [], []

    # 📄 打开 metrics.csv 文件
    # Path(run_dir) / "metrics.csv": 构建文件路径
    # .open(): 以文本模式打开
    with (Path(run_dir) / "metrics.csv").open() as f:
        # csv.DictReader: 将 CSV 逐行解析为字典
        # 每一行都成为一个键值对字典
        for row in csv.DictReader(f):
            # 🔍 过滤条件: 仅当平均返回不为 NaN 时包含
            # "mean_ep_return" 是 csv.DictReader 解析出的键
            if row["mean_ep_return"] != "nan":
                # ✅ 包含该行数据
                # steps.append(int(row["steps"])):
                #   - row["steps"]: 字符串形式的步数
                #   - int(): 转换为整数
                #   - 存入 steps 列表
                steps.append(int(row["steps"]))

                # rets.append(float(row["mean_ep_return"])):
                #   - row["mean_ep_return"]: 字符串形式的平均返回
                #   - float(): 转换为浮点数
                #   - 存入 rets 列表
                rets.append(float(row["mean_ep_return"]))

    # 📊 返回元组
    # 两个列表的元组，可用于绘图
    return steps, rets


def plot(run_dirs: list[str], out: str) -> None:
    """📊 绘制学习曲线对比图。

    📝 参数详细说明:

    Args:
        run_dirs (list[str]): 📁 要对比的运行目录列表。
                              - 字符串列表，每个元素为目录路径
                              - 每个目录应包含 metrics.csv 文件
                              - 通常是 train() 返回的多个 run_dir
                              - 例如: ["runs/seed0", "runs/seed1"]

        out (str): 🖼️ 输出图片路径。
                   - 字符串，指定图片保存的位置
                   - 默认值: "results/curve.png"
                   - 图片将以此路径保存
                   - 目录将在保存前自动创建 (如果不存在)

    🔧 绘图配置:
        - figure size: (7, 4.2) - 合理的论文图尺寸
        - linewidth: 1.8 - 实线的粗细
        - x-label: "environment steps" - X轴标签
        - y-label: "mean episode return (training)" - Y轴标签
        - 调整 spine: 隐藏 top 和 right 的边框
        - 使用 tight_layout: 自动调整布局以防止重叠

    📋 绘图流程:
        1. 创建 figure 和 axes: fig, ax = plt.subplots(figsize=(7, 4.2))
        2. 遍历每个 run_dir:
           - 加载数据: load_metrics(rd)
           - 绘制曲线: ax.plot(steps, rets, ...)
           - 为每条曲线添加标签: label=Path(rd).name
        3. 配置坐标轴和图例
        4. 保存图片: fig.savefig(out, dpi=160)
        5. 显式关闭 figure: plt.close()

    📝 使用示例:
        ```python
        # 单个运行
        plot("runs/InvertedPendulum-v5_seed0", "results/curve.png")

        # 多个运行对比
        plot(
            ["runs/InvertedPendulum-v5_seed0", "runs/InvertedPendulum-v5_seed1"],
            "results/compare.png"
        )
        ```
    """
    # 📊 创建 figure 和子图
    # figsize=(7, 4.2): 标准的学习曲线图尺寸
    fig, ax = plt.subplots(figsize=(7, 4.2))

    # 🔄 遍历每个运行目录
    for rd in run_dirs:
        # 📊 加载数据
        # load_metrics(rd): 返回 (steps, rets)
        steps, rets = load_metrics(rd)

        # 📊 绘制曲线
        # ax.plot: 绘制折线图
        # Path(rd).name: 获取目录的名称 (不含路径部分)
        # 用作图例标签
        ax.plot(steps, rets, label=Path(rd).name, linewidth=1.8)

    # 🎚️ 配置 X 轴标签
    ax.set_xlabel("environment steps")

    # 🎚️ 配置 Y 轴标签
    ax.set_ylabel("mean episode return (training)")

    # 🎛️ 调整 spine (边框)
    # ax.spine["top"].set_visible(False): 隐藏上边框
    # ax.spine["right"].set_visible(False): 隐藏右边框
    # 这样可以得到更干净的图形外观，符合论文规范
    ax.spine[["top", "right"]].set_visible(False)

    # 🔧 调整布局
    # fig.tight_layout(): 自动调整子图参数
    # 防止标签和标题重叠
    fig.tight_layout()

    # 📁 确保输出目录存在
    # Path(out).parent.mkdir(parents=True, exist_ok=True):
    # - Path(out): 输出路径对象
    # - .parent: 获取父目录
    # - mkdir(parents=True): 创建父目录及其所有父级目录
    # - exist_ok=True: 如果目录已存在则不报错
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    # 💾 保存图片
    # fig.savefig: 保存 figure
    # dpi=160: 设置分辨率 (160 dots per inch)
    # 这在学术论文和报告中提供了足够的清晰度
    fig.savefig(out, dpi=160)

    # 🧹 关闭 figure
    # plt.close(): 释放内存
    # 在批量生成多张图时非常重要，防止内存泄漏
    plt.close()


def main() -> None:
    """🖥️ 命令行入口点。

    使用 argparse 解析命令行参数并调用 plot 函数。

    示例用法:
        python -m locolab.plots runs/InvertedPendulum-v5_seed0 --out results/curve.png
        python -m locolab.plots runs/seed0 runs/seed1 --out results/compare.png
    """
    # 📋 创建参数解析器
    p = argparse.ArgumentParser(description=__doc__)

    # 📁 添加 run_dirs 参数 (可选，多个)
    # nargs="+": 接受一个或多个参数
    # help: 在帮助信息中显示的描述
    p.add_argument("run_dirs", nargs="+")

    # --out 参数 (可选)
    # default: "results/curve.png" (默认输出路径)
    # help: 在帮助信息中显示的描述
    p.add_argument("--out", default="results/curve.png")

    # 解析命令行参数
    a = p.parse_args()

    # 🚀 调用 plot 函数
    # 传入解析后的 run_dirs 和 out 参数
    plot(a.run_dirs, a.out)


# 🔧 当该模块作为主程序运行时
if __name__ == "__main__":
    main()