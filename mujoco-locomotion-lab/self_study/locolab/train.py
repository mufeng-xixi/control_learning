"""Train PPO on a MuJoCo gymnasium env.

Example:
    python -m locolab.train --env InvertedPendulum-v5 --epochs 50
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from .buffer import RolloutBuffer
from .ppo import PPO


def set_seed(seed: int) -> None:
    """🔢 设置随机种子，确保实验可复现。

    📝 参数说明:

    Args:
        seed (int): 🔢 种子整数。
                    - 用于初始化随机数生成器
                    - 确保实验结果的可重现性
                    - 类型: int，范围: >= 0

    🔢 设置的随源:
        - random.seed(seed): Python 的 random 模块
        - np.random.seed(seed): NumPy 的随机模块
        - torch.manual_seed(seed): PyTorch 的张量随机模块

    💡 重要性:
        - 确保实验的可重复性，便于调试和对比
        - 特别是在发表论文或报告结果时必不可少
        - 设置后，相同的种子将产生相同的随机序列

    📝 使用示例:
        >>> set_seed(42)
        >>> # 此后的随机操作将可预测
    """
    # 🌟 设置 Python 的 random 模块种子
    random.seed(seed)

    # 📊 设置 NumPy 的随机种子
    np.random.seed(seed)

    # ⚡ 设置 PyTorch 的手动种子
    # 这影响张量的随机初始化和抽样
    torch.manual_seed(seed)


def train(
    env_id: str = "InvertedPendulum-v5",
    epochs: int = 50,
    steps_per_epoch: int = 4000,
    seed: int = 0,
    out_dir: str = "runs",
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Path:
    """🚀 运行训练并返回包含 checkpoint + metrics 的 run 目录。

    📝 参数详细说明:

    Args:
        env_id (str): 🏷️ 环境标识。
                      - 字符串，对应 gymnasium 环境 ID
                      - 默认: "InvertedPendulum-v5"
                      - 可选其他值: "Hopper-v5", "HalfCheetah-v5" 等
                      - 对应环境的观测空间和动作空间的维度

        epochs (int): 🔄 训练的总轮数 (Epochs)。
                     - 每个 epoch 包含 steps_per_epoch 步的环境交互
                     - 默认值: 50
                     - 决定了训练的总时长
                     - 典型值: 20-100，取决于任务复杂度

        steps_per_epoch (int): 📐 每个 epoch 的步数。
               - 决定了每个 epoch 收集的经验量
               - 默认值: 4000
               - 典型值: 1000-5000，取决于环境的效率
               - 注意: 该值应 >= 缓冲区容量，避免溢出

        seed (int): 🎲 随机种子。
                    - 用于 initialize set_seed()
                    - 默认值: 0
                    - 确保实验的可重复性
                    - 整数，建议使用 0-1000 之间的值

        out_dir (str): 📁 输出目录路径。
                       - 训练结果的保存位置
                       - 默认值: "runs"
                       - 结果将保存在 out_dir/{env_id}_seed{seed}/ 下
                       - 目录结构:
                         runs/
                         └── InvertedPendulum-v5_seed0/
                             ├── policy.pt (模型权重)
                             ├── config.json (配置信息)
                             └── metrics.csv (训练指标)

        gamma (float): ⚖️ 折现因子 γ。
                       - 控制对未来奖励的重视程度
                       - 范围: (0, 1]
                       - 默认值: 0.99 (常用设置)
                       - γ = 1: 对所有未来奖励完全重视
                       - γ < 1: 未来奖励的重要性指数衰减

        lam (float): 🧮 GAE (广义优势估计) 参数 λ。
                     - 控制优势估计的方差-偏差权衡
                     - 范围: [0, 1]
                     - 默认值: 0.95 (常用设置)
                     - λ = 1: 退化为 Monte Carlo 返回
                     - λ = 0: 退化为单步 TD 误差

    📝 返回值:

    Path: 📂 返回的路径对象指向训练运行的目录。
        - 类型: pathlib.Path
        - 内容: 包含训练结果的目录路径
        - 结构:
            run_dir/
            ├── policy.pt (模型权重)
            ├── config.json (训练配置)
            └── metrics.csv (每轮的训练指标)

        - 用法示例:
            ```python
            run_dir = train(env_id="InvertedPendulum-v5", epochs=50)
            # run_dir 是一个 Path 对象
            # 可以访问: run_dir / "policy.pt", run_dir / "metrics.csv" 等
            ```

    📝 训练流程概览:

    1. 🌟 初始化阶段:
        - 调用 set_seed(seed) 设置随机种子
        - 创建 gymnasium 环境: env = gym.make(env_id)
        - 获取观测空间和动作空间的维度:
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
        - 初始化 PPO 代理: agent = PPO(obs_dim, act_dim)
        - 初始化 RolloutBuffer: buf = RolloutBuffer(steps_per_epoch, obs_dim, act_dim, γ, λ)
        - 创建运行目录: run_dir = Path(out_dir) / f"{env_id}_seed{seed}"
        - 创建 metrics.csv 文件并写入表头

    2. 🔄 主训练循环 (epochs 次):
        对于每个 epoch:
            a. 初始化 epoch 的统计变量:
               - ep_returns: 存储每个 episode 的返回
               - ep_lens: 存储每个 episode 的长度
               - ep_ret: 当前 episode 的累计返回
               - ep_len: 当前 episode 的步数

            b. 收集阶段 (steps_per_epoch 步):
                - 对于每一步:
                   - 调用 agent.ac.step(obs) 获取动作、价值、对数概率
                   - 与环境交互: next_obs, rew, terminated, truncated, _ = env.step(act)
                   - 将经验存入缓冲区: buf.add(obs, act, rew, val, logp)
                   - 更新当前观测: obs = next_obs
                   - 累计奖励: ep_ret += float(rew)
                   - 累计步数: ep_len += 1
                   - 判断 episode 是否结束:
                      - 检查 terminated 或 truncated 或达到 steps_per_epoch
                      - 如果 episode 结束:
                          - 根据 terminated/截断的类型设置 last_val
                          - 调用 buf.finish_path(last_val)
                          - 记录 episode 的返回和长度
                          - 重置 ep_ret, ep_len
                          - 重置观测: obs, _ = env.reset()

            c. 更新阶段:
                - 调用 agent.update(buf.get()) 进行 PPO 更新
                - 计算 epoch 的平均指标:
                   - mean_ret: 平均 episode 返回
                   - mean_len: 平均 episode 长度
                   - losses: 包含 pi_loss, vf_loss, kl
                - 将指标写入 metrics.csv
                - 打印训练进度信息

    3. 💾 训练结束后:
        - 保存模型权重: torch.save(agent.ac.state_dict(), run_dir / "policy.pt")
        - 保存配置信息:
            config = dict(env_id=env_id, epochs=epochs, steps_per_epoch=steps_per_epoch, seed=seed, γ, λ)
            (run_dir / "config.json").write_text(json.dumps(config, indent=2))
        - 关闭环境: env.close()
        - 返回 run_dir 路径

    💡 训练技巧:
        - 如果训练不收敛，尝试调整 clip_ratio (0.1-0.3)
        - 如果 KL 散度发散，减小 target_kl (0.005-0.01)
        - 确保 steps_per_epoch 足够大，以覆盖足够的状态空间
        - 监控 metrics.csv 中的 kl 一列，如果过大请及时调整

    📝 依赖:
        - from .buffer import RolloutBuffer: 用于存储经验
        - from .ppo import PPO: 用于策略优化
        - gymnasium: 环境接口
        - torch: 深度学习框架
        - numpy: 数值计算
    """
    # 🌟 设置随机种子，确保可重复性
    set_seed(seed)

    # 🏗️ 创建环境
    # gymnasium.make 返回 (observation, info) 元组
    # 我们只关心 observation，用 _ 忽略 info
    env = gym.make(env_id)

    # 👁️ 获取观测空间的维度
    # observation_space.shape 返回一个元组，如 (4,) 对应 InvertedPendulum-v5
    obs_dim = env.observation_space.shape[0]

    # 🎯 获取动作空间的维度
    # action_space.shape 返回一个元组，如 (1,) 对应 InvertedPendulum-v5
    act_dim = env.action_space.shape[0]

    # ⚙️ 初始化 PPO 代理
    # 传入观测和动作的维度
    agent = PPO(obs_dim, act_dim)

    # 📊 初始化 RolloutBuffer
    # size: steps_per_epoch (收集的总步数)
    # obs_dim, act_dim: 空间维度
    # gamma, lam: GAE 参数
    buf = RolloutBuffer(steps_per_epoch, obs_dim, act_dim, gamma=gamma, lam=lam)

    # 📁 创建运行目录
    # f-string 格式化: "{env_id}_seed{seed}"
    # 例如: "InvertedPendulum-v5_seed0"
    run_dir = Path(out_dir) / f"{env_id}_seed{seed}"

    # 📂 确保目录存在 (如果不存在则创建)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 📊 创建 metrics.csv 文件并写入表头
    # 使用 csv.writer 创建写入器
    # writerow 方法写入第一行的表头
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        # csv.writer 对象，使用 newline="" 防止 Windows 额外换行
        csv.writer(f).writerow(
            ["epoch", "steps", "mean_ep_return", "mean_ep_len", "pi_loss", "vf_loss", "kl"]
        )

    # 👁️ 重置环境
    # env.reset() 返回 (obs, info)
    # 只保留 obs，用 _ 忽略 info
    obs, _ = env.reset(seed=seed)

    # 🔢 初始化 epoch 循环的统计变量
    # ep_returns: 存储每个 episode 的总返回
    # ep_lens: 存储每个 episode 的步数
    ep_returns, ep_lens = [], []

    # ep_ret: 当前 episode 的累计返回 (浮点数)
    # ep_len: 当前 episode 的步数计数器 (整数)
    ep_ret, ep_len = 0.0, 0

    # 🔄 主循环: 遍历 epochs (从 1 到 epochs)
    for epoch in range(1, epochs + 1):
        # --- 收集阶段 ---
        # 遍历 steps_per_epoch 步
        for t in range(steps_per_epoch):
            # 🎯 获取动作、价值估计和对数概率
            # agent.ac.step(obs) 返回 (action, val, logp)
            act, val, logp = agent.ac.step(obs)

            # 🔄 与环境交互
            # env.step(act) 返回 (next_obs, rew, terminated, truncated, _)
            next_obs, rew, terminated, truncated, _ = env.step(act)

            # 📥 将经验存入缓冲区
            # buf.add() 将当前步的数据存入缓冲区
            # 参数: obs, act, rew, val, logp
            buf.add(obs, act, rew, val, logp)

            # 🔄 更新当前观测
            # 将 next_obs 设置为下一步的 obs
            obs = next_obs

            # 📊 累计奖励和步数
            # ep_ret: 当前 episode 的总返回 (浮点数)
            ep_ret += float(rew)
            # ep_len: 当前 episode 的步数 (整数)
            ep_len += 1

            # ⚠️ episode 结束判断
            # 检查是否达到步数上限或 episode 自然结束
            epoch_done = t == steps_per_epoch - 1

            if terminated or truncated or epoch_done:
                # 🔢 确定最后一个状态的价值 (bootstrapping value)
                if terminated:
                    # 如果是 terminated (任务失败/成功完成)
                    # 价值设为 0，表示Episode结束
                    last_val = 0.0
                else:
                    # 如果是 truncated (时间步耗尽，但未终止)
                    # 使用价值网络估计剩余价值
                    last_val = agent.ac.value(
                        torch.as_tensor(obs, dtype=torch.float32)
                    ).item()

                # 📋 调用 finish_path 计算 GAE 和 returns
                # 这会更新 buf.adv 和 buf.ret
                buf.finish_path(last_val)

                # 📊 记录 episode 统计数据
                # 只有在 episode 真正结束 (terminated 或 truncated) 时记录
                if terminated or truncated:
                    # 将 episode 的返回添加到列表
                    ep_returns.append(ep_ret)
                    # 将 episode 的长度添加到列表
                    ep_lens.append(ep_len)

                # 🔄 重置 episode 统计变量
                # ep_ret: 重置为 0.0
                ep_ret, ep_len = 0.0, 0

                # 🔄 重置观测
                # env.reset() 返回 (obs, info)
                # 开始新的 episode
                obs, _ = env.reset()

        # 🔄 更新阶段
        # 调用 agent.update() 使用缓冲区中的数据更新策略和价值网络
        # buf.get() 返回数据字典并重置缓冲区指针
        losses = agent.update(buf.get())

        # 📊 计算 epoch 的统计指标
        # 计算平均 episode 返回
        # np.mean(ep_returns) 如果列表为空返回 nan
        # 使用三元运算符处理空列表的情况
        mean_ret = float(np.mean(ep_returns)) if ep_returns else float("nan")

        # 计算平均 episode 长度
        mean_len = float(np.mean(ep_lens)) if ep_lens else float("nan")

        # 📝 将统计数据写入 metrics.csv
        # 使用 csv.writer 的 writerow 方法
        # f-string 格式化输出，保留两位小数
        with metrics_path.open("a", newline="") as f:
            # csv.writer 的 writerow 方法
            csv.writer(f).writerow(
                [
                    epoch,  # 当前 epoch 数 (从 1 开始)
                    epoch * steps_per_epoch,  # 总步数 (epoch * steps_per_epoch)
                    f"{mean_ret:.2f}",  # 平均返回，保留2位小数
                    f"{mean_len:.1f}",  # 平均长度，保留1位小数
                    f"{losses['pi_loss']:.4f}",  # 策略损失，保留4位小数
                    f"{losses['vf_loss']:.4f}",  # 价值损失，保留4位小数
                    f"{losses['kl']:.5f}",  # KL散度，保留5位小数
                ]
            )

        # 📢 打印训练进度
        # f-string 格式化输出
        # {:3d}: 右对齐，宽度为3的整数
        # {:9.2f}: 右对齐，宽度为9，小数点后2位
        # {:7.1f}: 右对齐，宽度为7，小数点后1位
        # {:.4f}: 保留4位小数
        print(
            f"epoch {epoch:3d}  return {mean_ret:9.2f}  "
            f"len {mean_len:7.1f}  kl {losses['kl']:.4f}"
        )

    # 💾 训练结束后的收尾工作

    # 📜 保存模型权重
    # agent.ac.state_dict(): 获取 ActorCritic 网络的状态字典
    # run_dir / "policy.pt": 将权重保存到指定路径
    # .pt 扩展名: PyTorch 模型权重的常用扩展名
    torch.save(agent.ac.state_dict(), run_dir / "policy.pt")

    # 📝 保存训练配置信息
    # 构建包含训练参数的字典
    config = dict(
        env_id=env_id,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        seed=seed,
        gamma=gamma,
        lam=lam,
    )
    # .write_text(): 将字符串写入文件
    # json.dumps(indent=2): 美化的 JSON 格式，缩进 2  spaces
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    # 🔚 关闭环境
    # 释放资源，特别是 MuJoCo 的渲染资源
    env.close()

    # 📂 返回 run_dir 路径
    # 这允许调用者访问保存的模型和指标
    return run_dir


def main() -> None:
    """🖥️ 命令行入口点。

    使用 argparse 解析命令行参数并调用 train 函数。

    示例用法:
        python -m locolab.train --env InvertedPendulum-v5 --epochs 50
        python -m locolab.train --env Hopper-v5 --epochs 100 --seed 42
    """
    # 📋 创建参数解析器
    # __doc__ 的使用: 将文档字符串作为描述显示
    p = argparse.ArgumentParser(description=__doc__)

    # 🏷️ 添加 --env 参数
    # default: "InvertedPendulum-v5" (环境ID)
    # help: 在帮助信息中显示的描述
    p.add_argument("--env", default="InvertedPendulum-v5")

    # 🔢 添加 --epochs 参数
    # type: int (确保输入是整数)
    # default: 50 (默认训练轮数)
    p.add_argument("--epochs", type=int, default=50)

    # 🔢 添加 --steps-per-epoch 参数
    # type: int
    # default: 4000 (每epoch的步数)
    p.add_argument("--steps-per-epoch", type=int, default=4000)

    # 🔢 添加 --seed 参数
    # type: int
    # default: 0 (随机种子)
    p.add_argument("--seed", type=int, default=0)

    # --out-dir 参数
    # default: "runs" (输出目录)
    p.add_argument("--out-dir", default="runs")

    # 解析命令行参数
    a = p.parse_args()

    # 🚀 调用 train 函数
    # 传入解析后的参数
    train(a.env, a.epochs, a.steps_per_epoch, a.seed, a.out_dir)


# 🔧 当该模块作为主程序运行时
if __name__ == "__main__":
    main()