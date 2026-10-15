import os
os.environ["MUJOCO_GL"] = "egl"  # 必须在导入 mujoco / gymnasium 之前

import json
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# ---- 从项目源码里导入模型类（名字以实际为准）----
from locolab.ppo import PPO

# ==== 1. 加载 checkpoint ====
ckpt_dir = "results/checkpoints/Hopper-v5_seed0"

# 读配置（确认文件是否存在，名称可能不同）
with open(os.path.join(ckpt_dir, "config.json")) as f:
    cfg = json.load(f)

# 构建模型并获取观测/动作维度
# 从 config 中移除非模型参数（如 env_id），避免传入 PPO 构造函数
# PPO 支持的参数：pi_lr, vf_lr, clip_ratio, train_iters, target_kl, ent_coef
ppo_keys = {"pi_lr", "vf_lr", "clip_ratio", "train_iters", "target_kl", "ent_coef"}
cfg_filtered = {k: v for k, v in cfg.items() if k in ppo_keys or k in ("obs_dim", "act_dim")}

# 创建临时环境获取维度
temp_env = gym.make("Hopper-v5")
obs_dim = temp_env.observation_space.shape[0]
act_dim = temp_env.action_space.shape[0]
temp_env.close()

model = PPO(obs_dim, act_dim, **cfg_filtered)

# 加载权重（文件名为 policy.pt，注意 PyTorch 版本兼容性）
ckpt_path = os.path.join(ckpt_dir, "policy.pt")
state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
model.ac.load_state_dict(state)
model.ac.eval()

# ==== 2. 环境 + 录像 ====
env = gym.make("Hopper-v5", render_mode="rgb_array")
env = RecordVideo(env, video_folder="videos",
                  episode_trigger=lambda ep: True)  # 每个 episode 都录

# ==== 3. 跑若干 episode ====
n_episodes = 3
for ep in range(n_episodes):
    obs, info = env.reset(seed=ep)
    done = False
    total_reward = 0.0
    while not done:
        with torch.no_grad():
            # 评估时用确定性动作（取均值），不要采样
            action = model.ac.act_deterministic(obs)
        action = action.numpy() if isinstance(action, torch.Tensor) else action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"Episode {ep}: return = {total_reward:.2f}")

env.close()
print("Videos saved to ./videos")