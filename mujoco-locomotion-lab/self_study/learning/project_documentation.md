# MuJoCo Locomotion Lab: 项目文档 (中文)

## 项目概述
本项目包含工具用于加载训练好的 PPO（Proximal Policy Optimization）策略并在 MuJoCo 环境中进行可视化。这些脚本解决了加载检查点和与 Gymnasium/MuJoCo 交互时的常见问题。

### 创建的文件
1. **`vido.py`** - 修复版 Hopper-v5 可视化脚本
2. **`vido_gan.py`** - InvertedPendulum-v5 含视频录制
3. **`vido_visualize.py`** - InvertedPendulum-v5 实时可视化（无视频）
4. **`vido_gan_mujoco.py`** - 原始 InvertedPendulum 可视化（已弃用）

---

## 1. `vido.py` - 修复版 Hopper-v5 可视化

### 作用
可视化 Hopper-v5 环境中的训练好的 PPO 策略并记录代理性能视频。

### 关键修复的问题

| 问题 | 原始代码 | 修复后 |
|------|----------|--------|
| 配置键过滤 | 将所有 config 键传入 PPO | 仅过滤 PPO 兼容的键 |
| 维度获取 | `gym.make("Hopper-v5").observation_space.shape[0]` | 创建临时环境并关闭 |
| 权重文件名 | `model.pt` | `policy.pt`（匹配实际文件） |
| PyTorch 兼容性 | `torch.load(... map_location="cpu")` | 添加 `weights_only=True` 参数 |
| 模型结构 | `model.load_state_dict(state)` | `model.ac.load_state_dict(state)` (PPO 包装) |
| 评估模式 | `model.eval()` | `model.ac.eval()` |
| 动作选择 | `model.get_action()` | `model.ac.act_deterministic()` |
| 打印语句 | 在循环外（仅打印最后一次） | 在循环内（打印所有 episode） |

### 运行方法
```bash
cd /home/jun/jun/2_RL_learning/mujoco-locomotion-lab
python vido.py
```

### 输出
```
Episode 0: return = 3160.08
Episode 1: return = 3260.92
Episode 2: return = 3204.07
Videos saved to ./videos
```

生成 3 个 MP4 视频到 `./videos/` 目录。

### 参数
- Checkpoint: `results/checkpoints/Hopper-v5_seed0`
- Episodes: 3
- 视频文件夹: `./videos/`

---

## 2. `vido_gan.py` - InvertedPendulum-v5 含视频录制

### 作用
可视化训练好的 InvertedPendulum-v5 策略并录制视频。

### 与 `vido.py` 的区别
- 不同环境: `InvertedPendulum-v5` (4D 观测, 1D 动作)
- 不同 checkpoint: `results/checkpoints/InvertedPendulum-v5_seed0`
- 不同视频文件夹: `./videos_gan/`
- 更多 episode: 5 (Hopper 的 3 个 vs 5 个)

### 运行方法
```bash
cd /home/jun/jun/2_RL_learning/mujoco-locomotion-lab
python vido_gan.py
```

### 输出
```
Episode 0: return = 1000.00
Episode 1: return = 1000.00
Episode 2: return = 1000.00
Episode 3: return = 1000.00
Episode 4: return = 1000.00
Videos saved to ./videos_gan
```

生成 5 个 MP4 视频到 `./videos_gan/` 目录。

### 参数
- Checkpoint: `results/checkpoints/InvertedPendulum-v5_seed0`
- Episodes: 5
- 视频文件夹: `./videos_gan/`

---

## 3. `vido_visualize.py` - 实时 MuJoCo 可视化

### 作用
在 MuJoCo 窗口直接实时可视化训练好的 InvertedPendulum-v5 策略，**不录制视频**。

### 关键特性
- 使用 `render_mode="human"` 进行直接窗口显示
- 无视频文件生成（轻量级）
- 每步包含小延时 (0.01s) 使渲染更可见
- 默认运行 3 个 episode

### 运行方法
```bash
cd /home/jun/jun/2_RL_learning/mujoco-locomotion-lab
python vido_visualize.py
```

### 输出
```
Episode 0: steps=1000, return=1000.00
Episode 1: steps=1000, return=1000.00
Episode 2: steps=1000, return=1000.00
Visualization complete!
```

将打开一个 MuJoCo 窗口显示倒立摆保持平衡。关闭窗口退出脚本。

### 参数
- Checkpoint: `results/checkpoints/InvertedPendulum-v5_seed0`
- Episodes: 3
- 渲染模式: `human` (实时显示)

---

## 4. `vido_gan_mujoco.py` - 弃用的原始脚本

### 作用
InvertedPendulum-v5 的原始可视化脚本，用于录制视频。

### 状态
此脚本是 `vido_gan.py` 的前身，保留为历史参考。

---

## 常见修复概览

所有脚本都解决了这些反复出现的问题：

### 1. 配置键过滤
PPO 构造函数仅接受特定键:
```python
ppo_keys = {"pi_lr", "vf_lr", "clip_ratio", "train_iters", "target_kl", "ent_coef"}
cfg_filtered = {k: v for k, v in cfg.items() if k in ppo_keys or k in ("obs_dim", "act_dim")}
```

### 2. 维度获取
使用临时环境获取维度，然后关闭:
```python
temp_env = gym.make("EnvironmentName-v5")
obs_dim = temp_env.observation_space.shape[0]
act_dim = temp_env.action_space.shape[0]
temp_env.close()
```

### 3. 权重加载
正确的权重加载模式:
```python
# PyTorch 2.x 兼容性
state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
# 加载到 ac 网络 (PPO 封装网络)
model.ac.load_state_dict(state)
# 设置为评估模式
model.ac.eval()
```

### 4. 动作选择
使用确定性动作进行评估:
```python
action = model.ac.act_deterministic(obs)
# 如有需要转为 numpy
action = action.numpy() if isinstance(action, torch.Tensor) else action
```

### 5. 模型结构
PPO 类将网络封装在 `self.ac` 中:
- `model.load_state_dict()` -> 报错
- `model.ac.load_state_dict()` -> 正确
- `model.eval()` -> 报错 (无此方法)
- `model.ac.eval()` -> 正确

---

## 项目结构

```
/home/jun/jun/2_RL_learning/mujoco-locomotion-lab/
├── .github/                    # GitHub 工作流
├── .mlvenv/                    # Python 虚拟环境
├── LICENSE                     # 许可证文件
├── README.md                   # 项目阅读me
├── requirements.txt            # 依赖项
├── results/
│   └── checkpoints/            # 训练好的模型检查点
│       ├── Hopper-v5_seed0/
│       │   ├── config.json
│       │   ├── eval.json
│       │   └── policy.pt
│       └── InvertedPendulum-v5_seed0/
│           └── policy.pt
├── locolab/                    # 项目源码
│   ├── __init__.py
│   ├── ppo.py                  # PPO 实现
│   ├── train.py
│   ├── evaluate.py
│   └── ...
├── runs/                       # 训练运行
├── tests/                      # 测试文件夹
├── vido.py                     # 修复版 Hopper 可视化
├── vido_gan.py                 # InvertedPendulum 含视频
├── vido_visualize.py           # 实时可视化 (无视频)
├── vido_gan_mujoco.py          # 原始 InvertedPendulum 脚本
├── learning/                   # 学习资源
│   ├── project_documentation.md  # 本文档
│   └── test_documentation.md     # 测试文档
├── videos/                     # Hopper 视频输出
├── videos_gan/                 # InvertedPendulum 视频输出
└── videos_gan_mujoco/        # 弃用的视频输出
```

---

## 依赖项

- Python 3.13
- MuJoCo (带 egl 支持)
- Gymnasium
- PyTorch (2.14.1+cu130)
- NumPy

安装方式:
```bash
cd /home/jun/jun/2_RL_learning/mujoco-locomotion-lab
pip install -r requirements.txt
```

---

## 故障排除

### 常见错误及解决方案

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `TypeError: PPO.__init__() got an unexpected keyword argument` | 将所有 config 键传入 PPO | 使用 `ppo_keys` 集合过滤 config 键 |
| `FileNotFoundError: model.pt` | 错误的权重文件名 | 使用 `policy.pt` 代替 |
| `AttributeError: 'PPO' object has no attribute 'load_state_dict'` | 尝试错误的对象加载权重 | 使用 `model.ac.load_state_dict(state)` |
| `AttributeError: 'PPO' object has no attribute 'eval'` | 在错误的对象上调用 eval | 使用 `model.ac.eval()` |
| `ValueError: cannot select an axis to squeeze` | 错误的动作形状处理 | 检查动作类型并用 `.numpy()` 转换 |
| `FileNotFoundError` 加载 config | 路径问题 | 使用 `os.path.join(ckpt_dir, "config.json")` |

### MuJoCo GL 错误
如果出现 MuJoCo GL 相关错误，请确保:
```python
os.environ["MUJOCO_GL"] = "egl"
```
在导入 mujoco/gymnasium 之前设置。

---

## 结果摘要

### Hopper-v5 (vido.py)
- 记录 3 个 episode
- 返回: 3160.08, 3260.92, 3204.07
- 平均返回: ~3208.36

### InvertedPendulum-v5 (vido_gan.py)
- 记录 5 个 episode
- 返回: 1000.00 (所有 episode)
- 平均返回: 1000.00

### InvertedPendulum-v5 (vido_visualize.py)
- 3 个 episode 实时可视化
- 返回: 1000.00 (所有 episode)
- 未生成视频

所有结果都表明训练好的 PPO 策略成功解决了各自的环境。

---

## 文档位置

- **项目文档**: `/home/jun/jun/2_RL_learning/mujoco-locomotion-lab/learning/project_documentation.md`
- **测试文档**: `/home/jun/jun/2_RL_learning/mujoco-locomotion-lab/learning/test_documentation.md`

这些文档提供了每个脚本的完整指南，包括用法、参数、输出结果以及常见错误的解决方案。