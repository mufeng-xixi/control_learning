import numpy as np
import pytest
import torch

from locolab.buffer import RolloutBuffer


def fill(buf, n, rew=1.0):
    for _ in range(n):
        buf.add(np.zeros(3), np.zeros(1), rew, 0.0, 0.0)


def test_gae_matches_hand_computation():
    # 3 steps, gamma=0.5, lam=0.5, zero values, reward 1 each step, terminal end.
    buf = RolloutBuffer(3, 3, 1, gamma=0.5, lam=0.5)
    fill(buf, 3)
    buf.finish_path(last_val=0.0)
    # deltas = [1,1,1]; adv_t = sum_k (gamma*lam)^k delta_{t+k} with gl=0.25
    expected = [1 + 0.25 + 0.0625, 1 + 0.25, 1.0]
    assert np.allclose(buf.adv[:3], expected)
    assert np.allclose(buf.ret[:3], expected)  # values are zero so ret == adv


def test_bootstrap_on_truncation():
    buf = RolloutBuffer(1, 3, 1, gamma=0.9, lam=1.0)
    fill(buf, 1, rew=2.0)
    buf.finish_path(last_val=10.0)
    # delta = 2 + 0.9*10 - 0 = 11
    assert np.isclose(buf.adv[0], 11.0)


def test_get_normalizes_advantages_and_resets():
    buf = RolloutBuffer(4, 3, 1)
    fill(buf, 2, rew=1.0)
    buf.finish_path(0.0)
    fill(buf, 2, rew=5.0)
    buf.finish_path(0.0)
    data = buf.get()
    assert isinstance(data["obs"], torch.Tensor)
    assert abs(data["adv"].mean().item()) < 1e-6
    assert abs(data["adv"].std(unbiased=False).item() - 1.0) < 1e-2
    assert buf.ptr == 0  # ready for reuse


def test_overflow_raises():
    buf = RolloutBuffer(1, 3, 1)
    fill(buf, 1)
    with pytest.raises(AssertionError):
        fill(buf, 1)
