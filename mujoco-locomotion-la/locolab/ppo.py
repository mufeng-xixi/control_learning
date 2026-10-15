"""PPO agent for continuous control: Gaussian actor + value critic, clipped objective."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def mlp(sizes: list[int], activation=nn.Tanh) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, int] = (64, 64)):
        super().__init__()
        self.pi_net = mlp([obs_dim, *hidden, act_dim])
        self.v_net = mlp([obs_dim, *hidden, 1])
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))

    def dist(self, obs: torch.Tensor) -> Normal:
        return Normal(self.pi_net(obs), torch.exp(self.log_std))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v_net(obs).squeeze(-1)

    @torch.no_grad()
    def step(self, obs: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Sample an action; return (action, value, log-prob) for rollout collection."""
        o = torch.as_tensor(obs, dtype=torch.float32)
        d = self.dist(o)
        a = d.sample()
        return a.numpy(), self.value(o).item(), d.log_prob(a).sum(-1).item()

    @torch.no_grad()
    def act_deterministic(self, obs: np.ndarray) -> np.ndarray:
        """Mean action, used for evaluation only."""
        o = torch.as_tensor(obs, dtype=torch.float32)
        return self.pi_net(o).numpy()


class PPO:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        pi_lr: float = 3e-4,
        vf_lr: float = 1e-3,
        clip_ratio: float = 0.2,
        train_iters: int = 80,
        target_kl: float = 0.015,
        ent_coef: float = 0.0,
    ):
        self.ac = ActorCritic(obs_dim, act_dim)
        self.clip_ratio = clip_ratio
        self.train_iters = train_iters
        self.target_kl = target_kl
        self.ent_coef = ent_coef
        self.pi_opt = torch.optim.Adam(
            [*self.ac.pi_net.parameters(), self.ac.log_std], lr=pi_lr
        )
        self.vf_opt = torch.optim.Adam(self.ac.v_net.parameters(), lr=vf_lr)

    def update(self, data: dict[str, torch.Tensor]) -> dict[str, float]:
        obs, act, ret, adv, logp_old = (
            data["obs"], data["act"], data["ret"], data["adv"], data["logp"]
        )
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
