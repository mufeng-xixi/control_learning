"""Rollout buffer with generalized advantage estimation (GAE)."""

from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    """Stores one on-policy rollout and computes GAE advantages.

    All arrays are preallocated for `size` steps. Call `add` once per step,
    `finish_path` at every episode boundary (or rollout end), then `get`
    to obtain tensors for the PPO update.
    """

    def __init__(self, size: int, obs_dim: int, act_dim: int, gamma: float = 0.99, lam: float = 0.95):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros(size, dtype=np.float32)
        self.val = np.zeros(size, dtype=np.float32)
        self.logp = np.zeros(size, dtype=np.float32)
        self.adv = np.zeros(size, dtype=np.float32)
        self.ret = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start, self.max_size = 0, 0, size

    def add(self, obs, act, rew, val, logp) -> None:
        assert self.ptr < self.max_size, "buffer overflow"
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.val[self.ptr] = val
        self.logp[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val: float = 0.0) -> None:
        """Compute GAE and returns for the slice since the last episode start.

        `last_val` bootstraps the value of the state after the final step:
        0 for a true terminal state, V(s_T) for a truncated episode.
        """
        sl = slice(self.path_start, self.ptr)
        rews = np.append(self.rew[sl], last_val)
        vals = np.append(self.val[sl], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        adv = np.zeros_like(deltas)
        running = 0.0
        for t in reversed(range(len(deltas))):
            running = deltas[t] + self.gamma * self.lam * running
            adv[t] = running
        self.adv[sl] = adv
        self.ret[sl] = adv + self.val[sl]
        self.path_start = self.ptr

    def get(self) -> dict[str, torch.Tensor]:
        assert self.ptr == self.max_size, "buffer not full"
        self.ptr, self.path_start = 0, 0
        adv = self.adv.copy()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        data = dict(obs=self.obs, act=self.act, ret=self.ret, adv=adv, logp=self.logp)
        return {k: torch.as_tensor(v) for k, v in data.items()}
