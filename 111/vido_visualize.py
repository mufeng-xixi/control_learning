#!/usr/bin/env python3
"""
Direct MuJoCo real-time visualization of trained InvertedPendulum-v5 policy.
Renders episodes with the trained policy in real-time without video recording.
"""

import os
import json

# Ensure MuJoCo GL is set before importing mujoco/gymnasium
os.environ["MUJOCO_GL"] = "egl"

import torch
import gymnasium as gym

# ---- 从项目源码里导入模型类----
from locolab.ppo import PPO

print("="*60)
print("InvertedPendulum-v5 Trained Policy Real-Time Visualization")
print("="*60)

# ==== 1. 加载 checkpoint ====
ckpt_dir = "results/checkpoints/InvertedPendulum-v5_seed0"

# 读配置
with open(os.path.join(ckpt_dir, "config.json")) as f:
    cfg = json.load(f)

# 构建模型并获取观测/动作维度
ppo_keys = {"pi_lr", "vf_lr", "clip_ratio", "train_iters", "target_kl", "ent_coef"}
cfg_filtered = {k: v for k, v in cfg.items() if k in ppo_keys or k in ("obs_dim", "act_dim")}

# 创建临时环境获取维度
temp_env = gym.make("InvertedPendulum-v5")
obs_dim = temp_env.observation_space.shape[0]
act_dim = temp_env.action_space.shape[0]
temp_env.close()

print(f"Observation dimension: {obs_dim}")
print(f"Action dimension: {act_dim}")
print(f"Checkpoint directory: {ckpt_dir}")

model = PPO(obs_dim, act_dim, **cfg_filtered)

# 加载权重（文件名为 policy.pt）
ckpt_path = os.path.join(ckpt_dir, "policy.pt")
state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
model.ac.load_state_dict(state)
model.ac.eval()

print("Model weights loaded successfully!")
print("-" * 60)

# ==== 2. 环境设置 (不录制视频) ====
print("Setting up environment for real-time rendering...")
env = gym.make("InvertedPendulum-v5", render_mode="human")  # "human" for real-time display

# ==== 3. 跑若几 episode ====
n_episodes = 3
print(f"\nRunning {n_episodes} episodes with trained policy (press close window to exit)...\n")

for ep in range(n_episodes):
    obs, info = env.reset(seed=ep)
    done = False
    total_reward = 0.0
    step_count = 0
    
    while not done:
        with torch.no_grad():
            # 评估时用确定性动作（取均值），不要采样
            action = model.ac.act_deterministic(obs)
        action = action.numpy() if isinstance(action, torch.Tensor) else action
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        done = terminated or truncated
        # 小延时让渲染更可见
        import time
        time.sleep(0.01)
    
    print(f"Episode {ep}: steps={step_count}, return={total_reward:.2f}")

env.close()
print("-" * 60)
print("Visualization complete!")

# 可以在这里添加额外的分析代码
# 例如：实时记录轨迹、绘制状态图等