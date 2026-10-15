"""带有广义优势估计 (Generalized Advantage Estimation, GAE) 的 Rollout Buffer。

该模块实现了 PPO (Proximal Policy Optimization) 算法所需的滚动缓冲区，
用于存储 Agent 与环境交互的轨迹数据，并在每个 Episode 结束时计算 GAE
用于策略梯度更新。

缓冲区设计原则：
- 所有数组预先分配，避免运行时动态扩容的开销
- On-policy 存储，意味着数据必须按顺序收集且在使用后清空
- 支持 truncated (截断) 和 terminal (终止) 两种 Episode 结束情况
- GAE 参数 γ (gamma) 和 λ (lambda) 可以调节优势估计的偏差-方差权衡
"""

from __future__ import annotations  # 用于延迟类型注解，兼容旧版 Python

import numpy as np  # 用于高效的数组运算，预分配内存
import torch  # 用于将 numpy 数组转换为 PyTorch Tensor，供 PPO 更新时使用


class RolloutBuffer:
    """存储一个 on-policy rollout (一次策略执行的完整经验序列)
    并计算广义优势估计 (GAE) 的缓冲区类。

    该缓冲区专为 PPO 算法设计，具有以下特点：
    - 数组预分配：提前申请内存，避免频繁的内存重分配
    - 指针机制：通过 ptr 和 path_start 跟踪当前位置和 Episode 边界
    - GAE 计算：在 finish_path() 中计算广义优势，用于 PPO 的目标函数
    - 数据归一化：get() 方法返回时会对优势进行标准化处理
    """

    def __init__(self, size: int, obs_dim: int, act_dim: int, gamma: float = 0.99, lam: float = 0.95):
        """初始化 RolloutBuffer 实例。

        📝 详细参数说明：

        Args:
            size (int): 🔢 缓冲区的最大容量，即最多可以存储的步数。
                        这通常设置为 steps_per_epoch 的值，
                        确保在一个epoch内不会覆盖未完成的数据。
                        类型: int，范围: > 0

            obs_dim (int): 👁️ 观测空间的维度。
                           对应于环境 observation_space.shape[0]。
                           例如 InvertedPendulum-v5 的观测维度为 4。
                           类型: int，范围: > 0

            act_dim (int): 🎯 动作空间的维度。
                           对应于环境 action_space.shape[0]。
                           例如离散动作可能是 1（单个动作连续），
                           而箱子推取可能是多维。
                           类型: int，范围: > 0

            gamma (float, optional): 🎚️ 折现因子 γ，取值范围 (0, 1]。
                                     决定了对未来奖励的重视程度。
                                     γ = 1 表示对所有未来奖励完全重视，
                                     γ = 0 表示只关注当前奖励。
                                     默认值: 0.99，这在强化学习中是常见选择，
                                     表明未来奖励的重要性随时间指数衰减。

            lam (float, optional): 🎚️ GAE (广义优势估计) 参数 λ，取值范围 [0, 1]。
                                   控制优势估计的方差与偏差的权衡。
                                   - λ = 1: 退化为 Monte Carlo 返回，方差大但无偏
                                   - λ = 0: 退化为单步 TD（时差）误差，方差小但有偏
                                   - 常用取值: 0.95，在偏差和方差之间取得平衡
                                   默认值: 0.95

        📝 初始化时会完成以下操作：
        - 预分配 numpy 数组，用于存储观测、动作、奖励、价值、对数概率
        - 初始化 GAE 相关参数 γ 和 λ
        - 重置指针 ptr (当前写入位置) 和 path_start (当前 Episode 的起始位置)
        - 设置最大容量 max_size

        💡 备注：
        - 所有 numpy 数组使用 float32 类型，兼容 PyTorch 并节省内存
        - 缓冲区为 on-policy，意味着必须在完成一个 Episode 或
          填满缓冲区前不能重用（否则会丢失数据）
        """
        # 🔢 缓冲区容量，即最多可以存储的步数
        self.max_size = size

        # 👁️ 观测状态的维度，用于预分配观测数组
        # 形状将为 (size, obs_dim)
        self.obs_dim = obs_dim

        # 🎯 动作空间的维度，用于预分配动作数组
        # 形状将为 (size, act_dim)
        self.act_dim = act_dim

        # ⏱️ 折现因子 γ，用于 GAE 计算和未来奖励的衰减
        self.gamma = gamma

        # 🧮 GAE 参数 λ，用于控制优势估计的方差-偏差权衡
        self.lam = lam

        # 📍 当前写入指针，表示下一个将要被写入的位置
        # 初始化为 0，表示缓冲区起始位置
        self.ptr = 0

        # 📍 Episode 开始指针，标记当前 Episode 的起始位置
        # 在 finish_path() 中使用，计算从该位置到 ptr 的 GAE
        self.path_start = 0

        # 📦 预分配的 numpy 数组，存储观测状态
        # 形状: (size, obs_dim)，类型: float32
        # 使用 float32 是为了兼容 PyTorch 并节省内存
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)

        # 🎯 预分配的 numpy 数组，存储动作
        # 形状: (size, act_dim)，类型: float32
        self.act = np.zeros((size, act_dim), dtype=np.float32)

        # 💰 预分配的 numpy 数组，存储每一步得到的价值估计
        # 形状: (size,)，类型: float32
        # 由网络的价值网络 (v_net) 输出
        self.val = np.zeros(size, dtype=np.float32)

        # 📝 预分配的 numpy 数组，存储每一步的对数概率
        # 形状: (size,)，类型: float32
        # 由采样动作时的分布 log-prob 计算得出
        self.logp = np.zeros(size, dtype=np.float32)

        # 📊 预分配的 numpy 数组，存储计算出的广义优势估计
        # 形状: (size,)，类型: float32
        # 在 finish_path() 计算后填充
        self.adv = np.zeros(size, dtype=np.float32)

        # 📈 预分配的 numpy 数组，存储计算出的返回 (return = advantage + value)
        # 形状: (size,)，类型: float32
        # 在 finish_path() 中用 adv + val 填充
        self.ret = np.zeros(size, dtype=np.float32)

    def add(self, obs, act, rew, val, logp) -> None:
        """📥 每一步调用，将当前的经验数据存入缓冲区。

        📝 详细参数说明：

        Args:
            obs (np.ndarray or sequence): 👁️ 当前的观测状态。
                                          - 形状应为 (obs_dim,) 或 (1, obs_dim)
                                          - 将被存入 self.obs[self.ptr]
                                          - 通常是环境 step() 返回的观测

            act (np.ndarray or sequence): 🎯 当前执行的动作。
                                          - 形状应为 (act_dim,) 或 (1, act_dim)
                                          - 将被存入 self.act[self.ptr]
                                          - 通常是 agent 根据 policy 采样或确定性选择的动作

            rew (float or np.ndarray): 💰 当前步得到的奖励。
                                       - 标量 float 值
                                       - 将被存入 self.rew[self.ptr]
                                       - 通常是环境 step() 返回的 reward

            val (float or np.ndarray): 💰 当前步的价值估计。
                                        - 标量 float 值
                                        - 将被存入 self.val[self.ptr]
                                        - 通常是 critic 网络对当前状态的价值预测

            logp (float or np.ndarray): 📝 当前动作的对数概率。
                                        - 标量 float 值
                                        - 将被存入 self.logp[self.ptr]
                                        - 通常是采样动作时分布的 log-probability

        🔍 运行时检查:
            - 会断言当前指针 ptr 小于最大容量 max_size
            - 如果缓冲区已满 (ptr >= max_size) 会抛出 AssertionError：
              "buffer overflow"
            - 这防止了数据溢出和覆盖未处理的旧数据

        📝 存储细节:
            - 每调用一次，ptr 自动加 1
            - 数据将被存入预分配好的 numpy 数组中
            - 确保数据按顺序存储，便于后续的 GAE 计算
            - 适用于 on-policy 算法，数据使用后应重置缓冲区

        💡 备注:
            - 此方法通常在环境交互循环中每一步调用
            - 数据在调用 get() 之前必须保留，get() 会重置指针
            - 适用于 PPO 算法的数据收集阶段
        """
        # 🔍 检查缓冲区是否溢出
        # 如果 ptr 已经达到 max_size，说明缓冲区已满
        assert self.ptr < self.max_size, "buffer overflow"

        # 📦 将观测状态存入预分配的数组
        # self.ptr 当前指向的位置将被赋值
        self.obs[self.ptr] = obs

        # 🎯 将动作存入预分配的数组
        self.act[self.ptr] = act

        # 💰 将奖励存入预分配的数组
        self.rew[self.ptr] = rew

        # 💰 将价值估计存入预分配的数组
        self.val[self.ptr] = val

        # 📝 将对数概率存入预分配的数组
        self.logp[self.ptr] = logp

        # 📈 移动指针，指向下一个位置
        self.ptr += 1

    def finish_path(self, last_val: float = 0.0) -> None:
        """⚡ 在每个 Episode 结束或 rollout 结束时调用。

        📝 详细参数说明：

        Args:
            last_val (float, optional): 💰 bootstrapping (外推) 的值。
                                        - 表示 Episode 结束后状态的价值
                                        - 0: 表示真正的 terminal state (Episode 自然结束)
                                          , 下一个状态不再有价值
                                        - V(s_T): 表示截断的 episode，
                                          下一个状态仍有价值 (例如时间步耗尽)
                                        - 默认值: 0.0，当不确定时使用 0
                                        - 这个值会被用来计算最后一个步骤的优势

        🔢 计算流程:
            1. 确定数据 slice: sl = slice(path_start, ptr)
               - path_start: 当前 Episode 的起始位置
               - ptr: 当前指针，即 Episode 结束的位置
               - 这 slice 了出当前 Episode 的所有数据

            2. 拼接 last_val:
               - rews = np.append(self.rew[sl], last_val)
               - vals = np.append(self.val[sl], last_val)
               - 这样做是为了在计算最后一个步骤的 delta 时有值可用

            3. 计算一步步的 TD 误差 (Temporal Difference Error):
               - deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
               - 公式: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
               - 其中 r_t 是奖励，γ 是折现因子，V 是价值函数

            4. 计算广义优势估计 (GAE):
               - 初始化 running = 0.0
               - 倒序遍历 deltas (从最后一步到第一步)
               - running = δ_t + γ * λ * running
               - adv[t] = running
               - 这实现了递归公式: A_t = δ_t + γλ A_{t+1}

            5. 存储结果:
               - self.adv[sl] = adv  (将 GAE 存入缓冲区)
               - self.ret[sl] = adv + self.val[sl]  (返回 = 优势 + 价值)

            6. 更新 path_start:
               - self.path_start = ptr  (标记下一个 Episode 的起始位置)

        💡 算法逻辑:
            - 该方法必须在每个 Episode 结束时调用
            - 必须在调用 get() 之前调用
            - 正确处理 terminal 和 truncated 的区别
            - last_val 的选择影响了优势估计的准确性

        📈 返回:
            - 无直接返回值 (修改内部状态)
            - self.adv 和 self.ret 数组会被填充
            - self.path_start 更新为 ptr，准备下一个 Episode
        """
        # 🔢 获取当前 Episode 的数据 slice
        # path_start 是上一个 Episode 的起始位置
        # ptr 是当前的位置 (Episode 结束的位置)
        sl = slice(self.path_start, self.ptr)

        # 📋 拼接奖励序列，最后使用 last_val 代替
        # 这样可以计算最后一步的 TD 误差
        rews = np.append(self.rew[sl], last_val)

        # 📊 拼接价值序列，最后使用 last_val 代替
        vals = np.append(self.val[sl], last_val)

        # ⚖️ 计算一步步的 TD 误差 (Temporal Difference Error)
        # 公式: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        # rews[:-1] 是当前步的奖励 (去掉最后的 last_val)
        # vals[1:] 是下一步的价值 (去掉第一个值)
        # vals[:-1] 是当前步的价值 (保留所有值但去掉最后的 last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]

        # 🧮 计算广义优势估计 (Generalized Advantage Estimation)
        # 初始化运行优势
        adv = np.zeros_like(deltas)
        running = 0.0

        # 🔄 倒序遍历计算 GAE
        # 从最后一步开始，向前回溯
        for t in reversed(range(len(deltas))):
            # 递归公式: A_t = δ_t + γλ A_{t+1}
            # running 维护了 A_{t+1}，即下一步的优势
            running = deltas[t] + self.gamma * self.lam * running
            adv[t] = running

        # 📦 将计算出的优势存入缓冲区
        self.adv[sl] = adv

        # 📈 计算返回 (Return = Advantage + Value)
        # 这是 PPO 目标函数中使用的值
        self.ret[sl] = adv + self.val[sl]

        # 🔄 更新 path_start 为当前 ptr
        # 这标记了下一个 Episode 的起始位置
        self.path_start = self.ptr

    def get(self) -> dict[str, torch.Tensor]:
        """✅ 当缓冲区满后调用，获取所有数据并重置缓冲区指针。

        📝 详细返回说明:

        Returns:
            dict[str, torch.Tensor]: 包含以下键的字典:
                - "obs" (torch.Tensor): 观测状态
                  - 形状: (max_size, obs_dim)
                  - 类型: float32 torch Tensor
                  - 来源: 从 self.obs 转换而来

                - "act" (torch.Tensor): 动作
                  - 形状: (max_size, act_dim)
                  - 类型: float32 torch Tensor
                  - 来源: 从 self.act 转换而来

                - "ret" (torch.Tensor): 返回 (Return)
                  - 形状: (max_size,)
                  - 类型: float32 torch Tensor
                  - 来源: 从 self.ret 转换而来

                - "adv" (torch.Tensor): 广义优势估计
                  - 形状: (max_size,)
                  - 类型: float32 torch Tensor
                  - 来源: 从 self.adv 转换而来 (已标准化)

                - "logp" (torch.Tensor): 对数概率
                  - 形状: (max_size,)
                  - 类型: float32 torch Tensor
                  - 来源: 从 self.logp 转换而来

        🔧 处理流程:
            1. 🔍 检查缓冲区是否已满:
                - assert self.ptr == self.max_size
                - 确保在使用前缓冲区已填满
                - 如果未满，会抛出 AssertionError: "buffer not full"

            2. 🔄 重置指针:
                - self.ptr, self.path_start = 0, 0
                - 将指针重置回起始位置，准备下一个 rollout

            3. 📊 复制并标准化优势:
                - adv = self.adv.copy()  (复制一份，避免修改原数据)
                - adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                - 标准化优势：减去均值，除以标准差 + 小常数防止除零
                - 这是 PPO 中常见的做法，有助于训练稳定性

            4. 📦 构建数据字典:
                - 整合 obs, act, ret, adv, logp
                - 所有数据都是 numpy 数组

            5. 🔧 转换为 PyTorch Tensor:
                - {k: torch.as_tensor(v) for k, v in data.items()}
                - 转换所有数据为 torch Tensor
                - 这样可以直接在 PPO 的 update() 方法中使用

        💡 使用时机:
            - 仅在缓冲区填满 (ptr == max_size) 时调用
            - 通常在一个 epoch 的数据收集结束后调用
            - 调用后会重置缓冲区，可以开始新的 epoch

        📝 注意事项:
            - 调用此方法会**永久性**重置缓冲区指针
            - 在调用前确保所有数据都已通过 finish_path() 处理
            - 返回的 Tensor 共享内存（或复制自）numpy 数组
            - 建议在使用前将 Tensor 移至正确的设备 (CPU/GPU)
        """
        # 🔍 检查缓冲区是否已满
        # 只有当 ptr == max_size 时，缓冲区才填满
        assert self.ptr == self.max_size, "buffer not full"

        # 🔄 重置指针，准备下一个 rollout
        # ptr 重置为 0，表示从头开始写入
        # path_start 重置为 0，表示下一个 Episode 的起始位置
        self.ptr, self.path_start = 0, 0

        # 📊 复制优势，防止后续操作修改原数据
        adv = self.adv.copy()

        # 📊 标准化优势 (Standardization)
        # 减去均值，除以标准差可以加速神经网络的收敛
        # + 1e-8 是为了防止标准差为0时的除零错误
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # 📦 构建要返回的数据字典
        # 包含所有需要的训练数据
        data = dict(
            obs=self.obs,      # 观测
            act=self.act,      # 动作
            ret=self.ret,      # 返回
            adv=adv,           # 标准化后的优势
            logp=self.logp     # 对数概率
        )

        # 🔧 将所有数据转换为 PyTorch Tensor
        # 这样可以直接在 PPO 的 update 方法中使用
        return {k: torch.as_tensor(v) for k, v in data.items()}