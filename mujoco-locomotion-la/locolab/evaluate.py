"""Evaluate a trained policy with the deterministic (mean) action over fresh seeds.

The evaluation seeds never overlap the training seed, so the reported numbers
come from episodes the agent has not been tuned on.

Example:
    python -m locolab.evaluate runs/InvertedPendulum-v5_seed0 --episodes 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from .ppo import ActorCritic

EVAL_SEED_BASE = 10_000  # far away from training seeds (0..99)


def evaluate(run_dir: str | Path, episodes: int = 100) -> dict:
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    env = gym.make(cfg["env_id"])
    ac = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0])
    ac.load_state_dict(torch.load(run_dir / "policy.pt", weights_only=True))
    ac.eval()

    returns, lengths = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=EVAL_SEED_BASE + ep)
        done, ep_ret, ep_len = False, 0.0, 0
        while not done:
            act = ac.act_deterministic(obs)
            obs, rew, terminated, truncated, _ = env.step(act)
            ep_ret += float(rew)
            ep_len += 1
            done = terminated or truncated
        returns.append(ep_ret)
        lengths.append(ep_len)
    env.close()

    result = dict(
        env_id=cfg["env_id"],
        episodes=episodes,
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        min_return=float(np.min(returns)),
        max_return=float(np.max(returns)),
        mean_length=float(np.mean(lengths)),
    )
    (run_dir / "eval.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("--episodes", type=int, default=100)
    a = p.parse_args()
    result = evaluate(a.run_dir, a.episodes)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
