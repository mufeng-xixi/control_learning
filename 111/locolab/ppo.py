"""PPO agent for continuous control: Gaussian actor + value critic, clipped objective.

This module implements the core components of PPO (Proximal Policy Optimization)
algorithm, including the Actor-Critic network architecture and the PPO training
loop. Key features:
- Gaussian policy: actions follow a normal distribution
- Value critic: estimates state value
- Clipped objective: limits policy update magnitude to prevent divergence
- KL divergence monitoring with early stopping
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def mlp(sizes, activation=nn.Tanh):
    """多层感知机 (Multi-Layer Perceptron) 构造函数。

    参数:
        sizes (list): 各层维度列表。
            格式: [输入维度, 隐藏层1, 隐藏层2, ..., 输出维度]
            例如: [obs_dim, 64, 64, act_dim] 表示输入 obs_dim,
            两个 64 维的隐藏层，输出输出 act_dim。
            必须至少包含 2 个元素 (输入和输出)。
            列表长度决定了网络的深度。

        activation (nn.Module, optional): 激活函数。
            默认: nn.Tanh (双曲 tangent)。
            其他常用选项: nn.ReLU, nn.Softplus。
            激活函数将在每一层 (除了最后一层) 之后应用。
            影响网络的非线性表示能力。

    返回:
        nn.Sequential: PyTorch 的 Sequential 模块，包含交替排列的:
            - nn.Linear (线性变换)
            - 激活函数。

    结构细节:
        - 对于每一层 i (从 0 到 len(sizes) - 2):
            - 添加 nn.Linear(sizes[i], sizes[i+1])
            - 如果 i < len(sizes) - 2 (不是最后一层)，添加激活函数
        - 最后一层不使用激活函数，输出原始数值。

    使用示例:
        # 创建一个网络: 输入 8 维 -> 两个 32 维隐藏层 -> 2 维输出
        network = mlp([8, 32, 32, 2])
        # 这会创建: Linear(8,32) -> Tanh -> Linear(32,32) -> Tanh -> Linear(32,2)
    """
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class ActorCritic:
    """Actor-Critic 神经网络类。

    该类实现了 Actor-Critic 架构，包含：
    - 策略网络 (pi_net): 输出动作分布的参数
    - 价值网络 (v_net): 估计状态价值
    - 可学习的标准差 (log_std): 高斯策略的可学习标准差

    该网络专为 PPO 算法设计，支持高斯策略采样和确定性动作输出。
    """

    def __init__(self, obs_dim, act_dim, hidden=(64, 64)):
        """初始化 ActorCritic 网络。

        参数:
            obs_dim (int): 观测空间的维度。
                对应于 env.observation_space.shape[0]。
                例如: InvertedPendulum-v5 的观测维度为 4。
                决定了网络输入层的大小。
                类型: int, 范围: > 0。

            act_dim (int): 动作空间的维度。
                对应于 env.action_space.shape[0]。
                例如: 连续动作空间的维度。
                决定了网络输出层的大小和高斯分布的参数数量。
                类型: int, 范围: > 0。

            hidden (tuple, optional): 隐藏层的维度元组。
                格式: (隐藏层1大小, 隐藏层2大小)。
                默认值: (64, 64)。
                第一个元素: 第一层隐藏层的神经元数量。
                第二个元素: 第二层隐藏层的神经元数量。
                类型: tuple of 2 ints。

        网络初始化过程:
            1. 调用 mlp() 函数创建策略网络 (pi_net)。
                结构: obs_dim -> 隐藏层1 -> 隐藏层2 -> 输出 act_dim。
                用于输出策略参数 (均值和标准差)。

            2. 调用 mlp() 函数创建价值网络 (v_net)。
                结构: obs_dim -> 隐藏层1 -> 隐藏层2 -> 输出 1。
                用于估计状态价值。

            3. 初始化 log_std:
                - 创建一个可学习的参数。
                - 初始值设为 -0.5 (对应标准差约 e^{-0.5} ≈ 0.61)。
                - 这有助于早期的探索行为。

        网络架构图:
            obs_dim
              │
              ▼
        +─────────────────────+
        │       pi_net        │  (策略网络)
        +─────────────────────┤
            │                 │
            ▼                 ▼
        μ (均值)          σ (标准差 = exp(log_std))
            │
            ▼
        actions ~ N(μ, σ²)

            │
              ▼
        +─────────────────────+
        │       v_net         │  (价值网络)
        +─────────────────────┤
            │
            ▼
        V(s) (状态价值)
    """
        super().__init__()
        self.pi_net = mlp([obs_dim, *hidden, act_dim])
        self.v_net = mlp([obs_dim, *hidden, 1])
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))

    def dist(self, obs):
        """根据观测状态返回一个正态分布。

        参数:
            obs (torch.Tensor): 观测张量。
                形状: (batch_size, obs_dim) 或 (obs_dim,)。
                类型: torch.float32.
                当前状态来自环境。

        返回:
            Normal: 一个正态分布对象。
                表示: actions ~ N(μ, σ²).
                参数:
                    * loc (均值): self.pi_net(obs) 的输出。
                    * 标准差: torch.exp(self.log_std) 的输出。

        计算过程:
            1. 通过策略网络计算均值: μ = pi_net(obs).
            2. 使用 exp(log_std) 计算标准差: σ = exp(log_std).
            3. 返回 Normal(μ, σ) 分布.

        使用示例:
            dist = ac.dist(obs)
            action = dist.sample()  # 采样一个动作
            log_prob = dist.log_prob(action).sum(-1)  # 计算对数概率
        """
        mean = self.pi_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    def value(self, obs):
        """估计状态价值函数。

        参数:
            obs (torch.Tensor): 观测张量。
                形状: (batch_size, obs_dim) 或 (obs_dim,)。
                类型: torch.float32.
                当前状态观测.

        返回:
            torch.Tensor: 状态价值张量。
                形状: (batch_size,) 或 (1,).
                类型: torch.float32.
                每个元素代表对应观测状态的价值估计.

        计算过程:
            1. 通过价值网络前向传播: v = v_net(obs).
            2. 使用 squeeze(-1) 去除最后一个维度.
            3. 返回一维张量, 每个元素对应一个观测状态的价值估计.

        使用示例:
            value = ac.value(obs)
            # value 是一个张量, 每个元素对应一个状态的价值.
            # 可以用于计算 TD 误差或优势函数.

        备注:
            - 价值网络输出的是状态价值 V(s), 即从状态 s 开始
              遵循当前策略直到结束所获得的预期总回报.
            - 在 PPO 中, 这个价值用于计算优势函数: A = r + γV(s') - V(s).
        """
        return self.v_net(obs).squeeze(-1)

    def step(self, obs):
        """采样一个动作; 返回 (action, value, log-prob) 用于 rollout 收集.

        参数:
            obs (np.ndarray): 当前观测状态.
                形状: (obs_dim,) 或 (1, obs_dim).
                类型: numpy float32 或可转换为 torch.Tensor.
                通常是环境 step() 返回的观测.

        返回:
            tuple[np.ndarray, float, float]: (action, value, log-prob).
                - action (np.ndarray): 采样的动作.
                    形状: (act_dim,).
                    类型: numpy float32.
                    从策略分布中采样得到的动作.
                - value (float): 状态价值估计.
                    类型: Python float.
                    由价值网络估计的当前状态价值.
                - log_prob (float): 动作的对数概率.
                    类型: Python float.
                    由采样动作的分布的对数概率计算得出.

        计算过程:
            1. 将 numpy 数组转换为 torch Tensor: o = torch.as_tensor(obs, dtype=torch.float32).
            2. 通过 dist() 方法获取正态分布: d = self.dist(o).
            3. 采样动作: a = d.sample().
            4. 计算价值: v = self.value(o).item() (转为 Python float).
            5. 计算对数概率: log_p = d.log_prob(a).sum(-1).item().
            6. 返回 (action.numpy(), value, log_prob).

        关键特性:
            - @torch.no_grad(): 装饰器, 表示在前向传播过程中不计算梯度.
                节省内存和计算资源.
                因为只用于数据收集, 不进行训练.
                PPO 数据收集阶段的特点.

        使用示例:
            # 在环境交互循环中
            action, val, logp = ac.step(obs)
            # 将动作应用于环境
            next_obs, reward, done, _ = env.step(action)
            # 将经验存入缓冲区
            buf.add(obs, action, reward, val, logp)

        备注:
            - 这是一个 "采样" 方法, 动作是从策略分布中随机采样的.
            - 适用于 PPO 数据收集阶段.
            - 不同于 act_deterministic, 该方法引入了随机性, 有助于探索.
            - 返回的 log-prob 在 PPO 的目标函数中非常重要.
        """
        o = torch.as_tensor(obs, dtype=torch.float32)
        return self.pi_net(o).numpy()

    def act_deterministic(self, obs):
        """返回确定性动作 (均值), 专为评估使用, 无采样.

        参数:
            obs (np.ndarray): 当前观测状态.
                形状: (obs_dim,) 或 (1, obs_dim).
                类型: numpy float32 或可转换为 torch.Tensor.

        返回:
            np.ndarray: 确定性动作 (均值).
                形状: (act_dim,).
                类型: numpy float32.
                直接从策略网络输出均值得到, 无随机性.

        计算过程:
            1. 将 numpy 数组转换为 torch Tensor: o = torch.as_tensor(obs, dtype=torch.float32).
            2. 通过策略网络计算均值: π_net(o).
            3. 转换为 numpy 数组并返回.

        关键特性:
            - @torch.no_grad(): 与 step 方法相同, 不计算梯度.
            - 返回网络输出的直接均值, 无采样.
            - 专为评估阶段设计.

        使用示例:
            # 在评估或测试阶段
            action = ac.act_deterministic(obs)
            # 使用确定性动作, 无随机波动.
            # 对于查看模型行为或录制视频非常有用.

        备注:
            - 该方法返回 π_net(obs) 的直接输出 (均值值).
            - 不同于 step(), 不涉及采样和对数概率计算.
            - 适用于: 策略评估、视频录制、人工演示等.
            - 在强化学习中, 评估时通常使用确定性动作以获得更稳定的结果.
        """
        o = torch.as_tensor(obs, dtype=torch.float32)
        return self.pi_net(o).numpy()


class PPO:
    """PPO (Proximal Policy Optimization) 训练代理类。

    该类实现了 PPO 算法的完整训练循环, 包括:
    - 策略更新 (使用剪辑目标)
    - 价值函数更新 (均方误差损失)
    - KL 散度监控和早停
    - 熵正则化 (鼓励探索)

    PPO 是目前最流行的策略优化算法之一, 以其稳定性和良好的性能而闻名。
    """

    def __init__(self, obs_dim, act_dim, pi_lr=3e-4, vf_lr=1e-3, clip_ratio=0.2,
                 train_iters=80, target_kl=0.015, ent_coef=0.0):
        """初始化 PPO 代理.

        参数:
            obs_dim (int): 观测空间的维度.
                与 ActorCritic 类的 obs_dim 相同.
                决定了网络输入层的大小.
                类型: int, 范围: > 0.

            act_dim (int): 动作空间的维度.
                与 ActorCritic 类的 act_dim 相同.
                决定了网络输出层的大小.
                类型: int, 范围: > 0.

            pi_lr (float, optional): 策略网络 (policy) 学习率.
                默认值: 3e-4 (0.0003).
                推荐范围: 1e-4 到 1e-3.
                控制策略网络参数的更新步长.
                较大的学习率可能导致训练不稳定.

            vf_lr (float, optional): 价值函数 (value function) 学习率.
                默认值: 1e-3 (0.001).
                推荐范围: 1e-4 到 1e-3.
                控制价值网络参数的更新步长.
                通常比策略学习率稍大.

            clip_ratio (float, optional): 剪切比率 (核心 PPO 参数).
                默认值: 0.2.
                推荐范围: 0.1 到 0.3.
                PPO 的核心创新点.
                用于计算剪切目标:
                min(ratio * adv, clipped * adv).
                较大的值允许更大的策略更新步长,
                但可能导致发散.
                较小的值限制更新步长,
                可能导致学习不足。

            train_iters (int, optional): 每次更新的训练迭代次数.
                默认值: 80.
                推荐范围: 10 到 100.
                一个 epoch 内的多次梯度下降步骤.
                增加迭代次数可以更好地拟合数据,
                但可能导致过拟合。

            target_kl (float, optional): KL 散度目标阈值.
                默认值: 0.015.
                推荐范围: 0.005 到 0.02.
                PPO 早停机制.
                如果训练期间 KL 散度超过此值的 1.5 倍,
                将提前终止当前 epoch 的训练.
                有助于防止策略发散。

            ent_coef (float, optional): 熵系数 (用于鼓励探索).
                默认值: 0.0.
                推荐范围: 0.0 到 0.01.
                熵正则项: -ent_coef * entropy.
                熵越大, 表示策略越随机/探索性.
                设为 0 则完全移除熵项, 变为纯粹的 PPO 而无熵正则化。

        初始化过程:
            1. 创建 ActorCritic 网络: self.ac = ActorCritic(obs_dim, act_dim).
            2. 初始化两个 Adam 优化器:
                - pi_opt: 策略网络的优化器.
                    参数: [*self.ac.pi_net.parameters(), self.ac.log_std].
                    包含了策略网络的所有参数和可学习的 log_std.
                    lr: pi_lr.
                - vf_opt: 价值网络的优化器.
                    参数: self.ac.v_net.parameters().
                    lr: vf_lr.
            2. 将超参数作为实例属性存储:
                - self.clip_ratio = clip_ratio.
                - self.train_iters = train_iters.
                - self.target_kl = target_kl.
                - self.ent_coef = ent_coef.

        超参数建议:
            - 对于大多数 MuJoCo 环境, 默认值通常效果良好.
            - 如果训练不收敛, 尝试调整 clip_ratio (0.1-0.3).
            - 如果 KL 散度发散, 减小 target_kl (0.005 或更小).
            - 如果学习太慢, 调整学习率 pi_lr 和 vf_lr.

        实例属性:
            - self.ac: ActorCritic 网络实例.
            - self.clip_ratio: 剪切比率.
            - self.train_iters: 训练迭代次数.
            - self.target_kl: KL 散度目标.
            - self.ent_coef: 熵系数.
            - self.pi_opt: 策略优化器.
            - self.vf_opt: 价值优化器.
        """
        self.ac = ActorCritic(obs_dim, act_dim)
        self.clip_ratio = clip_ratio
        self.train_iters = train_iters
        self.target_kl = target_kl
        self.ent_coef = ent_coef
        self.pi_opt = torch.optim.Adam(
            [*self.ac.pi_net.parameters(), self.ac.log_std], lr=pi_lr)
        self.vf_opt = torch.optim.Adam(self.ac.v_net.parameters(), lr=vf_lr)

    def update(self, data):
        """PPO 更新核心方法.

        参数:
            data (dict): 训练数据字典.
                必须包含以下键:
                - "obs": 观测状态张量.
                    形状: (batch_size, obs_dim).
                    来自 RolloutBuffer.get().
                - "act": 动作张量.
                    形状: (batch_size, act_dim).
                    来自 RolloutBuffer.get().
                - "ret": 返回 (Return).
                    形状: (batch_size,).
                    来自 RolloutBuffer.get().
                - "adv": 广义优势估计.
                    形状: (batch_size,).
                    来自 RolloutBuffer.get() (标准化).
                - "logp_old": 旧的对数概率.
                    形状: (batch_size,).
                    来自 RolloutBuffer.get().

        返回:
            dict[str, float]: 训练损失字典.
                - "pi_loss": 策略损失 (float).
                - "vf_loss": 价值函数损失 (float).
                - "kl": KL 散度 (float).

        更新过程 (逐步详解):

        ① 策略梯度更新 (循环 train_iters 次):
            1. 计算当前分布: d = self.ac.dist(obs).
                获取当前策略下的正态分布.
            2. 计算当前对数概率: logp = d.log_prob(act).sum(-1).
                计算当前动作的对数概率.
            3. 计算概率比: ratio = torch.exp(logp - logp_old).
                ratio = π_new(a|s) / π_old(a|s).
                表示策略更新后的相对概率变化.
            4. 计算剪切值:
                clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
                将概率比限制在 [1-clip_ratio, 1+clip_ratio] 范围内.
            5. 计算策略损失:
                pi_loss = -(torch.min(ratio * adv, clipped)).mean()
                取原始优势和剪切优势中的较小值, 然后取负值.
                负号是因为我们要最大化目标, 而优化器最小化损失.
            6. 加入熵正则项:
                pi_loss = pi_loss - self.ent_coef * d.entropy().sum(-1).mean()
                鼓励策略保持随机性, 防止过早收敛.
            7. 计算 KL 散度:
                kl = (logp_old - logp).mean().item()
                衡量新旧策略的差异.
            8. KL 早停检查:
                if kl > 1.5 * target_kl: break
                如果 KL 散度超过目标值的 1.5 倍, 提前终止训练.
                有助于防止策略发散.
            9. 优化器步骤:
                self.pi_opt.zero_grad()
                pi_loss.backward()
                self.pi_opt.step()
            10. 记录策略损失值: pi_loss_val = pi_loss.item()

        ② 价值函数更新 (循环 train_iters 次):
            1. 计算价值函数损失:
                vf_loss = ((self.ac.value(obs) - ret) ** 2).mean()
                均方误差: (预测的价值 - 实际返回)^2 的平均值.
                V(s) 是价值网络预测的价值.
                R 是从 RolloutBuffer.get() 获取的返回.
            2. 优化器步骤:
                self.vf_opt.zero_grad()
                vf_loss.backward()
                self.vf_opt.step()
            3. 记录价值损失值: vf_loss_val = vf_loss.item()

        训练逻辑说明:
            - PPO 交替进行策略更新和价值函数更新.
            - 策略更新使用剪切目标, 防止策略移动过远.
            - KL 散度监控防止策略变化过快.
            - 熵正则化鼓励探索, 防止陷入局部最优.

        返回值解释:
            - pi_loss: 策略损失, 通常为负值 (因为我们最小化负目标).
            - vf_loss: 价值函数均方误差损失.
            - kl: KL 散度 (这一轮的平均值, 可能是提前终止后的).

        备注:
            - 该方法会修改网络参数 (通过 optimizer.step()).
            - 必须在收集足够的数据后调用.
            - 数据应来自同一份 rollout 或多个 rollout 的组合.
            - 建议每个 epoch 调用一次此方法.
        """
        obs, act, ret, adv, logp_old = (
            data["obs"], data["act"], data["ret"], data["adv"], data["logp"])

        pi_loss_val, kl = 0.0, 0.0

        for _ in range(self.train_iters):
            d = self.ac.dist(obs)
            logp = d.log_prob(act).sum(-1)
            ratio = torch.exp(logp - logp_old)
            clipped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
            pi_loss = -(torch.min(ratio * adv, clipped)).mean()
            pi_loss = pi_loss - self.ent_coef * d.entropy().sum(-1).mean()
            kl = (logp_old - logp).mean().item()
            if kl > 1.5 * self.target_kl:
                break
            self.pi_opt.zero_grad()
            pi_loss.backward()
            self.pi_opt.step()
            pi_loss_val = pi_loss.item()

        vf_loss_val = 0.0
        for _ in range(self.train_iters):
            vf_loss = ((self.ac.value(obs) - ret) ** 2).mean()
            self.vf_opt.zero_grad()
            vf_loss.backward()
            self.vf_opt.step()
            vf_loss_val = vf_loss.item()

        return dict(pi_loss=pi_loss_val, vf_loss=vf_loss_val, kl=kl)