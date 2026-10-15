import numpy as np
import torch

from locolab.ppo import PPO, ActorCritic


def test_shapes_and_determinism():
    torch.manual_seed(0)
    ac = ActorCritic(5, 2)
    obs = np.zeros(5, dtype=np.float32)
    a, v, logp = ac.step(obs)
    assert a.shape == (2,)
    assert isinstance(v, float) and isinstance(logp, float)
    d1 = ac.act_deterministic(obs)
    d2 = ac.act_deterministic(obs)
    assert np.array_equal(d1, d2)


def test_update_improves_objective_on_synthetic_batch():
    torch.manual_seed(0)
    agent = PPO(4, 1, train_iters=20)
    n = 256
    obs = torch.randn(n, 4)
    with torch.no_grad():
        d = agent.ac.dist(obs)
        act = d.sample()
        logp = d.log_prob(act).sum(-1)
    # advantage favors positive actions: the policy mean should move up
    adv = act.squeeze(-1).clone()
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ret = torch.ones(n)
    mean_before = agent.ac.pi_net(obs).mean().item()
    losses = agent.update(dict(obs=obs, act=act, ret=ret, adv=adv, logp=logp))
    mean_after = agent.ac.pi_net(obs).mean().item()
    assert mean_after > mean_before
    assert np.isfinite(losses["vf_loss"])


def test_kl_early_stop_bounds_divergence():
    torch.manual_seed(0)
    agent = PPO(4, 1, train_iters=200, target_kl=0.015)
    n = 128
    obs = torch.randn(n, 4)
    with torch.no_grad():
        d = agent.ac.dist(obs)
        act = d.sample()
        logp = d.log_prob(act).sum(-1)
    adv = torch.randn(n)
    losses = agent.update(dict(obs=obs, act=act, ret=torch.zeros(n), adv=adv, logp=logp))
    assert losses["kl"] < 10 * agent.target_kl
