# 📦 MuJoCo Locomotion Lab: `locolab` 文件夹文档

## 📁 目录结构

```text
locolab/
├── __init__.py           # ✅ Package 初始化
├── buffer.py             # 📊 RolloutBuffer + GAE 计算
├── ppo.py                # 🧠 ActorCritic 网络 + PPO 训练逻辑
├── train.py              🚂 # 完整训练管道
├── evaluate.py           # 📈 策略评估
└── plots.py              # 📊 学习曲线绘制
```

---

## 📁 1. `buffer.py` - Rollout Buffer with GAE

### 🎯 作用
实现 PPO 算法所需的**滚动缓冲区**，用于存储序列数据并计算**广义优势估计（GAE）**。

### 🏗️ RolloutBuffer 类

| 方法 | 作用 |
|------|------|
| `__init__(size, obs_dim, act_dim, γ=0.99, λ=0.95)` | 🔧 初始化预分配的 numpy 数组，存储观测、动作、奖励、价值、对数概率。`γ` (gamma) 为折现因子，`λ` (lambda) 为 GAE 参数。 |
| `add(obs, act, rew, val, logp)` | 📥 🔍 **每步调用**，将当前经验存入缓冲区，包含检查溢出边界。 |
| `finish_path(last_val=0.0)` | ⚡ **每个 episode 结束** 时调用。计算从 `path_start` 到 `ptr` 的 GAE 和 returns。`last_val` 用于 bootstrapping：<br>• `0` → 真正的 terminal state<br>• `V(s_T)` → 截断的 episode |
| `get()` | ✅ **缓冲区满** 调用。返回归一化的 `torch.Tensor` 字典，并重置指针 `ptr=0, path_start=0`，准备下一个 rollout。 |

### 📊 数据结构一览

| 变量 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `obs` | `(size, obs_dim)` | `float32` | 观测状态 |
| `act` | `(size, act_dim)` | `float32` | 动作 |
| `rew` | `(size,)` | `float32` | 奖励 |
| `val` | `(size,)` | `float32` | 价值估计 |
| `logp` | `(size,)` | `float32` | 对数概率 |
| `adv` | `(size,)` | `float32` | **广义优势** |
| `ret` | `(size,)` | `float32` | **返回** (Advantage + Value) |

### 🔢 GAE 计算公式

<div align=center>

$$
\begin{aligned}
\delta_t &= r_t + \gamma V(s_{t+1}) - V(s_t) \\[4pt]
A_t &= \delta_t + \gamma \lambda A_{t+1} \quad \text{(倒序累计)}
\end{aligned}
$$

</div>

<details><summary>💡 公式解析</summary>

- **$\delta_t$ (Delta)**: 单步优势残差，即即时奖励加折现下一状态价值减当前状态价值
- **$A_t$ (Advantage)**: 广义优势，通过在 $\delta_t$ 上叠加 $\gamma\lambda$ 因子的倒序累计得到
- **参数意义**: <br>• **$γ (gamma)$**: 折现因子，通常设为 0.99，决定了对未来奖励的重视程度<br>• **$λ (lambda)$**: GAE 参数，通常设为 0.95，控制偏差与方差的权衡<br>• **当 $λ=1$**: 退化为普通蒙特卡洛 returns<br>• **当 $λ=0$**: 退化为单步 TD 误差

</details>

---

## 🧠 2. `ppo.py` - PPO Agent with Actor-Critic

### 🎯 作用
实现 PPO 算法的**核心网络架构**和**训练逻辑**，包括策略网络、价值网络和 clipped objective 更新。

### 🧩 mlp 函数

```python
def mlp(sizes: list[int], activation=nn.Tanh) -> nn.Sequential
```

| 参数 | 说明 |
|------|------|
| `sizes` | 📐 各层维度列表，例如 `[obs_dim, 64, 64, act_dim]` |
| `activation` | 🎚️ 激活函数，默认 `nn.Tanh` |

| 返回 | 说明 |
|------|------|
| `nn.Sequential` | 🧩 包含交替排列的线性层和激活层 |

### 🧬 ActorCritic 类

<div align=center>

| 方法 | 作用 |
|------|------|
| `__init__(obs_dim, act_dim, hidden=(64, 64))` | 🏗️ 初始化策略网络 `pi_net` 和价值网络 `v_net`，以及可学习的 `log_std` |
| `dist(obs)` | 📊 返回 `Normal` 分布，用于采样动作 |
| `value(obs)` | 💎 返回状态的价值估计 |
| `step(obs)` | 🎲 **采样动作**，返回 `(action, value, log-prob)`，专为 rollout 收集 |
| `act_deterministic(obs)` | 🎯 **确定性动作**（均值），专为评估使用，不采样 |

</div>

#### 🌐 网络架构图

```text
┌─────────────────────────────┐
│           Input: obs          │
│       (obs_dim dimensions)    │
└─────────────────────┬───────┘
                      │
              ┌───────▼───────┐
              │   pi_net:     │
              │ obs→hidden→…→ │
              │  act_dim units│
              └───────▲───────┘
                      │
              ┌───────▼───────┐
              │   v_net:      │
              │ obs→hidden→…→ │
              │     1 unit    │
              └───────▲───────┘
                      │
              ┌───────▼───────┐
              │  log_std: -0.5 │
              └───────────────┘
```

#### 🧪 关键属性

| 属性 | 说明 |
|------|------|
| `pi_net` | 🎯 策略网络 (π), 输出动作分布参数 |
| `v_net` | 💰 价值网络 (V), 输出状态价值 |
| `log_std` | 📏 可学习的对数标准差，初始化 -0.5 |

### 🚀 PPO 类

| 方法 | 作用 |
|------|------|
| `__init__(obs_dim, act_dim, **kwargs)` | ⚙️ 初始化 ActorCritic 网络、Adam 优化器 (pi_opt, vf_opt) 和 PPO 超参数 |
| `update(data)` | 🔄 **训练核心**，接收 `dict(obs, act, ret, adv, logp)` 并执行：<br>① 策略梯度更新 (循环 `train_iters` 次)<br> - 计算比率 `ratio = exp(logp - logp_old)`<br> - Clipped objective: `min(ratio*adv, clipped)*adv`<br> - 加入熵正则: `- ent_coef * entropy`<br> - KL 早停: `if kl > 1.5 * target_kl: break`<br>② 价值函数更新 (循环 `train_iters` 次)<br> - MSE Loss: `(value(obs) - ret)^2`<br>③ 返回字典: `{pi_loss, vf_loss, kl}` |

### ⚖️ PPO 关键机制

| 概念 | 公式/说明 |
|------|-----------|
| **Clipped Ratio** | $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ <br>目标: $\hat{A}_t \cdot \min(r_t(\theta), \clip(r_t(\theta), 1-\epsilon, 1+\epsilon))$ |
| **KL Divergence** | $D_{KL}(\pi_{\theta_{old}} || \pi_\theta)$ <br>若 $> 1.5 \cdot \target_{kl}$ 则提前终止训练 |
| **Entropy Regularization** | $H(\pi) = -\sum \pi(a|s) \log \pi(a|s)$ <br>鼓励探索，通过 `ent_coef` 控制权重 |

---

## 🚂 3. `train.py` - 完整训练管道

### 🎯 作用
从头开始运行 PPO 训练，包含环境交互、缓冲区管理、模型保存和指标记录。

### ⚙️ set_seed 函数

```python
def set_seed(seed: int) -> None
```

| 作用 | 说明 |
|------|------|
| 设置随机种子 | 确保实验可复现 |
| 影响 | `random.seed`, `np.random.seed`, `torch.manual_seed` |

### 🚂 train 函数

```python
def train(env_id, epochs, steps_per_epoch, seed, out_dir, γ, λ) → Path
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `env_id` | 🏷️ 环境标识 | `"InvertedPendulum-v5"` |
| `epochs` | 🔄 训练轮数 | `50` |
| `steps_per_epoch` | 📐 每轮步数 | `4000` |
| `seed` | 🎲 随机种子 | `0` |
| `out_dir` | 📁 输出目录 | `"runs"` |
| `γ` (gamma) | ⚖️ 折现因子 | `0.99` |
| `λ` (lambda) | 🧮 GAE 参数 | `0.95` |

#### 🔄 训练流程图

```text
┌─────────────────────────────┐
│        🚀 开始训练           │
│   set_seed → 创建环境 → 初始化 │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│    🔄 主循环: epoch 1 ~ N   │
│   │                         │
│   ├─> 收集 steps_per_epoch 步 │
│   │   ├─ act ← agent.ac.step(obs)│
│   │   ├─ obs, rew, done ← env.step(act)│
│   │   └─ buf.add(obs, act, rew, val, logp)│
│   │                         │
│   ├─✅ episode 结束          │
│   │   ├─ buf.finish_path(last_val)│
│   │   ├─ 记录 ep_ret, ep_len │
│   │   └─ obs ← env.reset()   │
│   │                         │
│   ├─🔄 PPO update: agent.update(buf.get())│
│   │                         │
│   ├─📊 记录 metrics → CSV  │
│   │   └─ print progress      │
│   └───────────────────────┘
└─────────────────────────────┘
```

#### 💾 训练结束后

| 动作 | 说明 |
|------|------|
| `torch.save(agent.ac.state_dict())` | 💾 保存模型权重至 `policy.pt` |
| `(run_dir / "config.json")` | 📝 保存配置: env_id, epochs, steps_per_epoch, seed, γ, λ |
| `env.close()` | 🛑 关闭环境 |

---

## 📈 4. `evaluate.py` - 策略评估

### 🎯 作用
使用**确定性动作**（均值）评估已训练策略，返回性能指标。评估 seed 与训练 seed **无重叠**，确保报告数值来自于 agent 未被调优过的 episode。

### ⚙️ evaluate 函数

```python
def evaluate(run_dir, episodes=100) → dict
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `run_dir` | 📁 训练输出目录路径 | — |
| `episodes` | 🔢 评估 episode 数量 | `100` |

| 返回字段 | 说明 |
|----------|------|
| `env_id` | 🏷️ 环境标识 |
| `episodes` | 🔢 评估的 episode 数量 |
| `mean_return` | 📊 平均返回 |
| `std_return` | 📈 返回的标准差 |
| `min_return` | 📉 最小返回 |
| `max_return` | 📈 最大返回 |
| `mean_length` | 📏 平均步长 |

#### 🔄 评估流程

```text
┌─────────────────────────────┐
│  📂 从 run_dir 加载 config    │
│  🎮 创建环境                  │
│  🧠 初始化 ActorCritic 网络   │
│  💾 加载 policy.pt 权重       │
│  ⚙️ 设置 ac.eval()            │
│  🔄 循环 episodes 次:         │
│     ├─ obs ← env.reset()     │
│     ├─ act ← ac.act_deterministic(obs)│
│     ├─ obs, rew, done ← env.step(act)│
│     └─ 记录 ep_ret, ep_len    │
│  📊 计算统计指标              │
│  💾 保存 eval.json            │
│  📤 返回结果字典              │
└─────────────────────────────┘
```

---

## 📊 5. `plots.py` - 学习曲线绘制

### 🎯 作用
从一个或多个运行目录的 `metrics.csv` 中加载数据，绘制学习曲线对比图。

### ⚙️ 函数概览

| 函数 | 作用 | 关键参数 |
|------|------|----------|
| `load_metrics(run_dir)` | 📥 从 CSV 加载步数和返回 | — |
| `plot(run_dirs, out)` | 📊 绘制对比曲线 | `run_dirs`: 目录列表, `out`: 输出路径 |

### 📥 load_metrics

```python
def load_metrics(run_dir) → tuple[list, list]
```

| 返回 | 说明 |
|------|------|
| `steps` | 📐 步数列表 (过滤了 NaN 的行) |
| `rets` | 📊 返回列表 (mean_ep_return) |

### 📊 plot

```python
def plot(run_dirs: list[str], out: str = "results/curve.png") → None
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `run_dirs` | 📁 要对比的运行目录列表 | — |
| `out` | 🖼️ 输出图片路径 | `"results/curve.png"` |

#### 🎨 绘图效果示例

<div align=center>
<img src="https://i.imgur.com/6XQkq7l.png" width="500" alt="Learning Curve Example">
</div>

| 特性 | 说明 |
|------|------|
| 图片尺寸 | `figsize=(7, 4.2)` |
| 调整 | 隐藏 top/right spine |
| 图例 | 每个 run_dir 独立标记 |
| 分辨率 | `dpi=160` |
| 保存路径 | 自动创建父目录 |

---

## 📦 使用示例

### 🚀 1. 启动训练

```bash
python -m locolab.train --env InvertedPendulum-v5 --epochs 50
```

### 📊 2. 评估模型性能

```bash
python -m locolab.evaluate runs/InvertedPendulum-v5_seed0 --episodes 100
```

### 📈 3. 绘制学习曲线

```bash
# 单个运行
python -m locolab.plots runs/InvertedPendulum-v5_seed0 --out results/curve.png

# 多个运行对比
python -m locolab.plots runs/InvertedPendulum-v5_seed0 runs/InvertedPendulum-v5_seed1 --out results/compare.png
```

### 💻 自定义脚本导入

```python
from locolab.buffer import RolloutBuffer
from locolab.ppo import PPO, ActorCritic
from locolab.train import train, set_seed
from locolab.evaluate import evaluate
from locolab.plots import plot, load_metrics
```

---

## 🔗 模块间依赖关系

```mermaid
graph TD
    %% 样式
    classDef repo fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef module fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;

    buffer([buffer.py]):::module
    ppo([ppo.py]):::module
    train([train.py]):::module
    evaluate([evaluate.py]):::module
    plots([plots.py]):::module

    %% 依赖关系
    train --> buffer
    train --> ppo
    evaluate --> ppo
    plots --> train (via metrics.csv)

    %% 子模块
    subgraph locolab 内部
        ppo_actor([ActorCritic])
        ppo_ppo([PPO])
        buffer_Buffer([RolloutBuffer])
    end

    ppo --> ppo_actor
    ppo --> ppo_ppo
    buffer --> buffer_Buffer
```

---

## 🛠️ 常见使用注意事项

| 场景 | 建议 |
|------|------|
| 🔧 设置种子 | 训练前始终调用 `set_seed()` 确保可复现 |
| 📦 缓冲区容量 | `RolloutBuffer.size` 应 ≥ `steps_per_epoch` |
| 📊 GAE 参数 | `γ` (0.99) 控制折现，`λ` (0.95) 控制偏差-方差权衡 |
| 🚫 KL 发散 | 若 `kl > 1.5 × target_kl`，PPO 会自动早停 |
| 📈 评估可靠性 | evaluate 中的 `EVAL_SEED_BASE = 10_000` 确保与训练隔离 |
| 📐 绘图对比 | 同一环境下对比不同 seed 或超参数的曲线 |

---

## 📜 版本和维护

| 项目 | 说明 |
|------|------|
| **库名称** | `locolab` |
| **功能模块** | 5 个核心文件 |
| **主要算法** | PPO (Proximal Policy Optimization) |
| **环境依赖** | MuJoCo, Gymnasium, PyTorch |
| **Python 版本** | 3.13+ |
| **作者** | MuJoCo Locomotion Lab 项目 |

---
---
*📄 文档生成时间: 2026*
*📁 文件位置: `/home/jun/jun/2_RL_learning/mujoco-locomotion-lab/learning/locolab_documentation.md`*