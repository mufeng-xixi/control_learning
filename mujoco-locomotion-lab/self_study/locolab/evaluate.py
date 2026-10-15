"""Evaluate a trained policy with the deterministic (mean) action over fresh seeds.

The evaluation seeds never overlap the training seed, so the reported numbers
come from episodes the agent has not been tuned on.

Example:
    python -m locolab.evaluate runs/InvertedPendulum-v5_seed0 --episodes 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from .ppo import ActorCritic

# 📝 固定的种子基底，确保评估种子与训练种子不重叠
# 训练种子通常从 0 开始, 所以评估种子从 10_000 开始
# 这确保了评估是在全新的随机序列上进行的，提高评估结果的可信度
EVAL_SEED_BASE = 10_000  # far away from training seeds (0..99)


def evaluate(run_dir: str | Path, episodes: int = 100) -> dict:
    """📈 评估训练好的策略与返回性能指标。

    📝 参数详细说明:

    Args:
        run_dir (str | Path): 📁 训练输出目录路径。
                              - 字符串或 pathlib.Path 对象
                              - 通常是 train() 返回的目录路径
                              - 必须包含以下文件:
                                - config.json (训练配置)
                                - policy.pt (模型权重)

        episodes (int): 🔢 评估的 episode 数量。
                        - 默认值: 100
                        - 决定了评估的统计样本量
                        - 较大的 episodes 会给出更稳定的统计估计
                        - 类型: int，范围: >= 1

    Returns:
        dict: 📊 包含评估结果的字典。
              包含的关键字段:
              - env_id (str): 环境标识符
              - episodes (int): 评估的 episode 数量
              - mean_return (float): 平均返回
              - std_return (float): 返回的标准差
              - min_return (float): 最小返回
              - max_return (float): 最大返回
              - mean_length (float): 平均 episode 长度 (步数)

    🔄 评估流程:

    1. 📂 加载配置:
        - 从 run_dir/config.json 读取配置信息
        - 解析 JSON 格式
        - 提取 env_id 以创建环境

    2. 🎮 创建环境:
        - gymnasium.make(cfg["env_id"])
        - 根据配置中的环境 ID 创建环境

    3. 🧠 初始化网络:
        - ActorCritic(env.observation_space.shape[0], env.action_space.shape[0])
        - 根据观测和动作的维度初始化网络
        - 观测维度: env.observation_space.shape[0]
        - 动作维度: env.action_space.shape[0]

    4. 💾 加载权重:
        - torch.load(run_dir / "policy.pt", weights_only=True)
        - 从 checkpoint 文件加载模型权重
        - weights_only=True: 安全加载选项，防止任意代码执行

    5. ⚙️ 设置评估模式:
        - ac.eval(): 将网络设置为评估模式
        - 这会禁用 dropout 和 batch normalization 的训练行为
        - 确保确定性输出 (使用均值而非采样)

    4. 🔄 运行 episode 循环:
        for ep in range(episodes):
            a. 重置环境:
               - obs, _ = env.reset(seed=EVAL_SEED_BASE + ep)
               - 使用固定的种子基底加上 episode 索引
               - 确保每个 episode 都有唯一的种子

            b. 初始化统计变量:
               - done, ep_ret, ep_len = False, 0.0, 0
               - done: episode 是否结束的布尔标志
               - ep_ret: 当前 episode 的累计返回 (浮点数)
               - ep_len: 当前 episode 的步数计数器 (整数)

            c. 步进循环:
               - while not done:
                  - ac.act_deterministic(obs): 使用确定性动作
                     - 返回策略网络的均值输出
                     - 不进行采样，保证可重复性
                  - obs, rew, terminated, truncated, _ = env.step(act)
                  - 累计奖励: ep_ret += float(rew)
                  - 累计步数: ep_len += 1
                  - 更新结束标志: done = terminated or truncated

            c. 记录结果:
               -  episode 的返回和长度添加到列表
               - returns.append(ep_ret)
               - lengths.append(ep_len)

    5. 📊 计算统计指标:
        - mean_return: float(np.mean(returns))
        - std_return: float(np.std(returns))
        - min_return: float(np.min(returns))
        - max_return: float(np.max(returns))
        - mean_length: float(np.mean(lengths))

    6. 💾 保存评估结果:
        - (run_dir / "eval.json").write_text(json.dumps(result, indent=2))
        - 将结果保存为 JSON 文件，便于查看和分析

    7. 📤 返回结果:
        - 返回包含所有指标的字典

    💡 评估最佳实践:
        - 使用足够多的 episodes (>= 50) 以获得可靠的统计
        - 确保评估种子与训练种子不重叠
        - 使用确定性动作 (act_deterministic) 以获得稳定的结果
        - 保存评估结果以便与其他运行进行对比

    📝 备注:
        - 评估不涉及网络训练，仅用于性能评估
        - 返回的指标有助于判断代理是否良好地泛化化
        - 相比训练，评估计算成本较低
    """
    # 📂 将 run_dir 转换为 Path 对象 (如果尚非该类型)
    run_dir = Path(run_dir)

    # 📖 从配置文件加载信息
    # json.loads: 解析 JSON 字符串
    # .read_text(): 从文件读取文本内容
    cfg = json.loads((run_dir / "config.json").read_text())

    # 🎮 创建环境
    # 根据配置中的 env_id 创建 gymnasium 环境
    env = gym.make(cfg["env_id"])

    # 🧠 初始化 ActorCritic 网络
    # 根据环境的观测和动作空间的维度
    ac = ActorCritic(
        env.observation_space.shape[0],  # 观测维度
        env.action_space.shape[0]        # 动作维度
    )

    # 💾 加载训练好的模型权重
    # torch.load: 加载检查点文件
    # weights_only=True: 安全选项，仅加载权重而非整个对象
    # run_dir / "policy.pt": 模型权重的路径
    ac.load_state_dict(torch.load(run_dir / "policy.pt", weights_only=True))

    # ⚙️ 设置为评估模式
    # ac.eval(): 关键步骤
    # - 禁用 dropout
    # - 固定 batch normalization 的统计量
    # - 确保确定性输出 (均值而非采样)
    ac.eval()

    # 📊 初始化结果收集列表
    returns, lengths = [], []

    # 🔄 循环评估 episode
    for ep in range(episodes):
        # 🔄 重置环境
        # seed 使用 EVAL_SEED_BASE + ep 确保每个 episode 唯一
        obs, _ = env.reset(seed=EVAL_SEED_BASE + ep)

        # 📊 初始化episode统计变量
        done, ep_ret, ep_len = False, 0.0, 0
        # done: episode结束的标志 (布尔值)
        # ep_ret: 当前episode的累计返回 (浮点数)
        # ep_len: 当前episode的步数 (整数)

        # 🔄 步进循环: 直到episode结束
        while not done:
            # 🎯 获取确定性动作
            # ac.act_deterministic(obs): 返回均值而非采样
            # 这确保了评估的可重复性和确定性
            act = ac.act_deterministic(obs)

            # 🔄 与环境交互
            obs, rew, terminated, truncated, _ = env.step(act)

            # 📊 累计奖励和步数
            ep_ret += float(rew)  # 累加奖励
            ep_len += 1           # 步数 +1

            # ⚠️ 判断 episode 是否结束
            # terminated: 任务是否自然结束 (如倒立摆倒下)
            # truncated: 是否因步数限制而结束 (如超过最大步数)
            done = terminated or truncated

        # 📊 记录 episode 结果
        returns.append(ep_ret)  # 添加返回到列表
        lengths.append(ep_len)  # 添加长度到列表

    # 🌟 关闭环境
    # 释放资源，特别是 MuJoCo 的渲染上下文
    env.close()

    # 📊 计算统计指标
    # numpy 的统计函数
    result = dict(
        env_id=cfg["env_id"],              # 环境标识
        episodes=episodes,                 # 评估的episode数
        mean_return=float(np.mean(returns)),       # 平均返回
        std_return=float(np.std(returns)),       # 返回的标准差
        min_return=float(np.min(returns)),       # 最小返回
        max_return=float(np.max(returns)),       # 最大返回
        mean_length=float(np.mean(lengths)),     # 平均长度 (步数)
    )

    # 💾 保存评估结果到 JSON 文件
    # .write_text(): 写入文本内容
    # json.dumps(result, indent=2): 用缩进美化的 JSON 格式
    # run_dir / "eval.json": 保存路径
    (run_dir / "eval.json").write_text(json.dumps(result, indent=2))

    # 📤 返回结果字典
    # 供主程序使用或进一步分析
    return result


def main() -> None:
    """🖥️ 命令行入口点。

    使用 argparse 解析命令行参数并调用 evaluate 函数。

    示例用法:
        python -m locolab.evaluate runs/InvertedPendulum-v5_seed0 --episodes 100
        python -m locolab.evaluate runs/InvertedPendulum-v5_seed0
    """
    # 📋 创建参数解析器
    p = argparse.ArgumentParser(description=__doc__)

    # 📁 添加 run_dir 参数 (必选)
    # positional: 位置参数，必须提供
    p.add_argument("run_dir")

    # --episodes 参数 (可选)
    # type: int (确保是整数)
    # default: 100 (默认评估次数)
    # help: 在帮助信息中显示的描述
    p.add_argument("--episodes", type=int, default=100)

    # 解析命令行参数
    a = p.parse_args()

    # 🚀 调用 evaluate 函数
    # 传入 run_dir 和 episodes 参数
    result = evaluate(a.run_dir, a.episodes)

    # 📊 打印结果 (美化的 JSON 格式)
    print(json.dumps(result, indent=2))


# 🔧 当该模块作为主程序运行时
if __name__ == "__main__":
    main()