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
    random.seed(seed)
    np.random.seed(seed)
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
    """Run training and return the run directory with checkpoint + metrics."""
    set_seed(seed)
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = PPO(obs_dim, act_dim)
    buf = RolloutBuffer(steps_per_epoch, obs_dim, act_dim, gamma=gamma, lam=lam)

    run_dir = Path(out_dir) / f"{env_id}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "steps", "mean_ep_return", "mean_ep_len", "pi_loss", "vf_loss", "kl"])

    obs, _ = env.reset(seed=seed)
    for epoch in range(1, epochs + 1):
        ep_returns, ep_lens = [], []
        ep_ret, ep_len = 0.0, 0
        for t in range(steps_per_epoch):
            act, val, logp = agent.ac.step(obs)
            next_obs, rew, terminated, truncated, _ = env.step(act)
            buf.add(obs, act, rew, val, logp)
            obs = next_obs
            ep_ret += float(rew)
            ep_len += 1
            epoch_done = t == steps_per_epoch - 1
            if terminated or truncated or epoch_done:
                if terminated:
                    last_val = 0.0
                else:
                    last_val = agent.ac.value(torch.as_tensor(obs, dtype=torch.float32)).item()
                buf.finish_path(last_val)
                if terminated or truncated:
                    ep_returns.append(ep_ret)
                    ep_lens.append(ep_len)
                    obs, _ = env.reset()
                ep_ret, ep_len = 0.0, 0
        losses = agent.update(buf.get())
        mean_ret = float(np.mean(ep_returns)) if ep_returns else float("nan")
        mean_len = float(np.mean(ep_lens)) if ep_lens else float("nan")
        with metrics_path.open("a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, epoch * steps_per_epoch, f"{mean_ret:.2f}", f"{mean_len:.1f}",
                 f"{losses['pi_loss']:.4f}", f"{losses['vf_loss']:.4f}", f"{losses['kl']:.5f}"]
            )
        print(f"epoch {epoch:3d}  return {mean_ret:9.2f}  len {mean_len:7.1f}  kl {losses['kl']:.4f}")

    torch.save(agent.ac.state_dict(), run_dir / "policy.pt")
    (run_dir / "config.json").write_text(json.dumps(
        dict(env_id=env_id, epochs=epochs, steps_per_epoch=steps_per_epoch, seed=seed,
             gamma=gamma, lam=lam), indent=2))
    env.close()
    return run_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="InvertedPendulum-v5")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--steps-per-epoch", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs")
    a = p.parse_args()
    train(a.env, a.epochs, a.steps_per_epoch, a.seed, a.out_dir)


if __name__ == "__main__":
    main()
