# MuJoCo  locomotion Lab: 测试文档 (中文)

## 项目概述
本文档详细说明 `tests/` 目录下的测试文件，这些测试验证 PPO（Proximal Policy Optimization）实现、滚动缓冲区以及训练管道的正确性。

### 测试文件结构
`tests/` 目录包含 3 个测试文件：
1. **`test_buffer.py`** - RolloutBuffer 测试
2. **`test_ppo.py`** - PPO 代理测试  
3. **`test_train_smoke.py`** - 训练管道烟雾测试

所有测试可以使用 pytest 运行：
```bash
cd /home/jun/jun/2_RL_learning/mujoco-locomotion-lab
python -m pytest tests/
```

---

## 1. `test_buffer.py` - RolloutBuffer 测试 (4 个测试)

### 作用
测试 `locolab.buffer` 中的 `RolloutBuffer` 类，专门验证广义优势估计（GAE）计算、优势归一化和缓冲区容量管理。

### 关键函数及其作用

#### `fill(buf, n, rew=1.0)` - 辅助函数
填充缓冲区以供测试使用。
- **参数:**
  - `buf`: RolloutBuffer 实例
  - `n`: 要添加的步数
  - `rew`: 奖励值（默认: 1.0）
- **操作:** 添加 `n` 条带有零观测/动作、指定奖励和零值/终止标志的记录

#### `test_gae_matches_hand_computation()` - GAE 计算测试
验证 GAE 计算是否与手动计算相匹配。
- **测试设置:** 创建一个 3 步的缓冲区，gamma=0.5，lam=0.5
- **执行:** 用 reward=1 填充 3 步，然后用 last_val=0.0 完成路径
- **期望优势:** `[1 + 0.25 + 0.0625, 1 + 0.25, 1.0]` = `[1.3125, 1.25, 1.0]`
- **验证:** 检查 `buf.adv[:3]` 和 `buf.ret[:3]` 是否匹配期望值
- **目的:** 确保 GAE 实现对优势估计是正确的

#### `test_bootstrap_on_truncation()` - 截断时的 Bootstrapping 测试
测试在 episode 被截断（而非自然结束）时的价值函数 bootstrapping。
- **测试设置:** 创建 1 步的缓冲区，gamma=0.9，lam=1.0
- **执行:** 添加 1 步 reward=2.0，然后用 last_val=10.0 完成路径（bootstrap value）
- **期望优势:** `delta = reward + gamma*last_val - 0 = 2 + 0.9*10 = 11.0`
- **验证:** `buf.adv[0] == 11.0`
- **目的:** 验证截断 episode 在优势计算中的正确处理

#### `test_get_normalizes_advantages_and_resets()` - 优势归一化和重置测试
测试优势是否正确归一化，且缓冲区在 `get()` 后能正确重置。
- **测试设置:** 创建 4 步的缓冲区，填充 2 步 reward=1，完成，填充 2 步 reward=5，完成
- **数据提取:** 调用 `buf.get()` 检索所有数据
- **断言:**
  - `data["obs"]` 是 torch.Tensor
  - 优势均值接近 0（归一化）
  - 优势标准差（unbiased=False）接近 1.0（归一化）
  - 缓冲区指针重置为 0（准备重用）
- **目的:** 验证优势归一化和缓冲区状态管理

#### `test_overflow_raises()` - 溢出检测测试
测试缓冲区溢出是否被正确检测并抛出错误。
- **测试设置:** 创建一个容量为 1 步的小缓冲区（1 step, capacity 3, 1 dimension）
- **执行:** 填充 1 步，然后尝试添加另一步
- **期望:** 抛出 `AssertionError`
- **目的:** 确保缓冲区容量限制被强制执行

### RolloutBuffer 类概述
`RolloutBuffer` 存储 PPO 训练的轨迹：
- 存储观测、动作、奖励、价值、对数概率
- 计算广义优势估计（GAE）
- 提供 `get()` 方法检索归一化数据
- 有容量限制防止溢出

---

## 2. `test_ppo.py` - PPO 代理测试 (3 个测试)

### 作用
测试 `locolab.ppo` 中的 PPO 代理实现，特别是 `ActorCritic` 网络和 PPO 训练循环。

### 关键函数及其作用

#### `test_shapes_and_determinism()` - 网络形状和确定性测试
验证 ActorCritic 网络产生正确的输出形状和确定性动作。
- **测试设置:** 将种子设为 0，创建 obs_dim=5, act_dim=2 的 ActorCritic
- **执行:**
  - 使用随机观测运行 `ac.step(obs)`
  - 用相同的观测运行 `ac.act_deterministic(obs)` 两次
- **断言:**
  - `a.shape == (2,)` - 动作形状匹配动作维度
  - `v` 和 `logp` 是浮点数
  - 确定性动作相等: `np.array_equal(d1, d2)`
- **目的:** 确保网络输出正确性和评估时的确定性行为

#### `test_update_improves_objective_on_synthetic_batch()` - 策略改进测试
测试 PPO 训练是否实际上在合成数据上改善策略目标。
- **测试设置:** 创建 obs_dim=4, act_dim=1, train_iters=20 的 PPO 代理
- **合成批次:** 生成 256 个观测，采样动作和对数概率
- **优势构造:** 创建 favoring positive actions（ favoring positive actions）的优势（归一化）
- **训练:** 调用 `agent.update()` 合成数据
- **断言:**
  - `mean_after > mean_before` - 策略网络均值改善
  - `np.isfinite(losses["vf_loss"])` - 价值函数损失有限
- **目的:** 验证 PPO 训练实际上从数据中学习

#### `test_kl_early_stop_bounds_divergence()` - KL 散度早停测试
测试 KL 散度监控是否用于防止策略崩溃。
- **测试设置:** 创建 obs_dim=4, act_dim=1, train_iters=200, target_kl=0.015 的 PPO 代理
- **合成批次:** 与之前类似的数据生成
- **训练:** 调用 `agent.update()` 随机优势
- **断言:** `losses["kl"] < 10 * agent.target_kl`
- **目的:** 验证 KL 散度保持在可接受范围内，如果发散则早停

### ActorCritic 和 PPO 类
- **ActorCritic:** 具有策略网络 (`pi_net) 和价值网络 (`v_net) 的神经网络
- **PPO:** 具有克制目标、价值函数训练和基于 KL 散度早停的 Proximal Policy Optimization 训练器

---

## 3. `test_train_smoke.py` - 训练管道烟雾测试 (1 个测试)

### 作用
使用 pytest 的 `tmp_path`  fixtures 进行端到端训练管道的烟雾测试。

### 关键函数及其作用

#### `test_train_eval_roundtrip(tmp_path)` - 端到端训练-评估测试
使用 `tmp_path` fixture 测试完整的训练和评估管道。
- **训练阶段:**
  - 调用 `train()` 配置:
    - env_id: "InvertedPendulum-v5"
    - epochs: 2
    - steps_per_epoch: 200
    - seed: 0
    - out_dir: tmp_path（临时目录）
  - 断言:
    - `policy.pt` 文件存在于 run_dir
    - `metrics.csv` 文件存在于 run_dir
    - config.json 有 `env_id == "InvertedPendulum-v5"`
  
- **评估阶段:**
  - 用 run_dir 和 episodes=3 调用 `evaluate()`
  - 断言:
    - `result["episodes"] == 3`
    - `result["mean_return"] > 0`（期望返回为正）
    - `eval.json` 文件存在于 run_dir

### train 和 evaluate 函数
- **`train()`:** 训练 PPO 代理指定的 epochs/steps，保存 checkpoints 和 metrics
- **`evaluate()`:** 使用训练好的策略运行评估 episode，返回包括 mean_return 在内的指标

### 烟雾测试的目的
烟雾测试是轻量级测试，验证主要组件是否工作而无需运行完整的训练迭代。此测试确保：
1. 训练管道可以保存 checkpoints 和 config
2. 评估管道可以加载训练好的策略并进行评估
3. 必要的文件（policy.pt, config.json, metrics.csv, eval.json）正确生成
4. 完整的 train-eval 回环正常工作

---

## 总结

### 测试覆盖情况

| 测试文件 | 关注点 | 关键概念 |
|----------|--------|----------|
| `test_buffer.py` | 数据存储 & GAE | Rollout 存储，优势估计，归一化 |
| `test_ppo.py` | 策略优化 | ActorCritic 网络，PPO 更新，KL 监控 |
| `test_train_smoke.py` | 管道完整性 | 训练，评估，checkpoint 保存 |

**总测试函数数:** 8 个
- buffer: 4 个测试
- PPO: 3 个测试
- train smoke: 1 个测试

所有测试共同确保 MuJoCo Locomotion Lab 的核心 RL 组件在部署到真实环境之前正常工作。

---

## 如何运行测试

### 前提条件
- 安装 pytest: `pip install pytest`
- 项目依赖: `pip install -r requirements.txt`

### 运行所有测试
```bash
cd /home/jun/jun/2_RL_learning/mujoco-locomotion-lab
python -m pytest tests/
```

### 运行特定测试
```bash
# 仅运行 buffer 测试
python -m pytest tests/test_buffer.py

# 仅运行 PPO 测试
python -m pytest tests/test_ppo.py

# 仅运行训练烟雾测试
python -m pytest tests/test_train_smoke.py
```

### 预期输出
测试应无失败通过，验证以下核心组件的正确性：
- RolloutBuffer GAE 计算
- PPO 策略改进
- KL 散度监控
- 训练管道端到端功能

---

## 常见问题排查

### 常见测试问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 测试超时 | MuJoCo 初始化缓慢 | 确保 MuJoCo 许可证/环境已配置 |
| `import locolab` 失败 | 包未安装 | `pip install -e .` 或添加到 PYTHONPATH |
| GAE 计算错误 | gamma/lambda 参数不正确 | 检查测试期望是否匹配实现 |
| KL 散度过高 | 学习率过大 / 过多的 epochs | 调整 PPO 超参数 |
| 评估返回 0 | 策略未训练足够 | 增加 epochs/steps 进行训练 |

### 添加新测试
添加新测试时：
1. 遵循现有命名约定：`test_*` 函数
2. 使用适当的 fixtures（如 `tmp_path`）
3. 包含清晰的 docstring 或注释
4. 每个测试函数测试一个特定的功能
5. 添加后运行所有测试以确保无回归

---

## 测试结果概览

测试验证了 MuJoCo Locomotion Lab 核心强化学习组件的正确性，确保：
- RolloutBuffer 正确存储轨迹并计算 GAE
- PPO 代理使用克制目标改善策略
- KL 散度监控防止策略发散
- 训练-评估管道完整保存和加载所有必要文件

所有测试共同为强化学习项目的稳定性提供了信心基础。