"""Sanity tests: run with  `python -m pytest -q`  (about 20 s on a laptop)."""
import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sagin.baselines import GreedyTrackPolicy, RandomPolicy, run_policy  # noqa: E402
from sagin.config import AlgoConfig, EnvConfig  # noqa: E402
from sagin.env import SAGINDisasterEnv, fifo_serve  # noqa: E402
from marl.mappo import ActorPolicy, MAPPOLagrangian  # noqa: E402
from marl.networks import Actor  # noqa: E402


def test_shapes_and_finiteness():
    env = SAGINDisasterEnv(EnvConfig(), n_envs=3, seed=0)
    obs = env.reset()
    assert obs["obs"].shape == (3, env.U, env.obs_dim)
    assert obs["state"].shape == (3, env.state_dim)
    assert obs["mask"].shape == (3, env.U, env.M) and obs["mask"][..., 0].all()
    pol = RandomPolicy(0)
    for _ in range(20):
        obs, r, cost, done, info = env.step(pol.act(env, obs))
        assert np.isfinite(obs["obs"]).all() and np.isfinite(obs["state"]).all()
        assert r.shape == (3,) and cost.shape == (3, 3)


def test_packet_conservation():
    """arrived = delivered + dropped + still queued (users + UAV buffers), at every slot."""
    env = SAGINDisasterEnv(EnvConfig(), n_envs=4, seed=1)
    obs = env.reset()
    pol = GreedyTrackPolicy()
    pol.reset(env)
    for _ in range(60):
        obs, *_ = env.step(pol.act(env, obs))
        lhs = env.ep["arrived"]
        rhs = env.ep["delivered"] + env.ep["dropped"] + env.Q.sum((1, 2)) + env.B.sum((1, 2, 3))
        assert np.allclose(lhs, rhs, atol=1e-6)


def test_fifo_serves_oldest_first():
    x = np.array([[1.0, 2.0, 3.0]])  # ages 0,1,2
    s = fifo_serve(x, np.array([4.0]))
    assert np.allclose(s, [[0.0, 1.0, 3.0]])


def test_cost_identities():
    """Episode-average costs are <= 0 exactly when the pooled reliability targets hold."""
    c = EnvConfig()
    env = SAGINDisasterEnv(c, n_envs=8, seed=2)
    m = run_policy(env, GreedyTrackPolicy(), seed=3)
    assert np.all((m["cost_outage"] <= 1e-9) == (m["outage"] <= c.eps_out + 1e-9))
    assert np.all((m["cost_pdr"] <= 1e-9) == (m["pdr"] >= c.pdr_min - 1e-9))
    assert np.all((m["cost_aoi"] <= 1e-9) == (m["aoi"] <= c.aoi_max + 1e-9))


def test_actor_logprob_consistency_and_pl_normalisation():
    torch.manual_seed(0)
    env = SAGINDisasterEnv(EnvConfig(), n_envs=2, seed=0)
    obs = env.reset()
    actor = Actor(env.layout)
    o = torch.from_numpy(obs["obs"]).reshape(-1, env.obs_dim)
    mk = torch.from_numpy(obs["mask"]).reshape(-1, env.M)
    act, lp = actor.act(o, mk)
    lp2, _ = actor.evaluate(o, mk, act)
    assert torch.allclose(lp, lp2, atol=1e-4)
    # Plackett-Luce probabilities of all ordered draws sum to one
    actor.NC = 2
    sc = torch.randn(1, 5)
    tot = sum(float(actor._pl(sc, torch.tensor([p]))[0].exp()) for p in itertools.permutations(range(5), 2))
    assert abs(tot - 1.0) < 1e-5


def test_one_training_iteration_runs():
    for algo in ("mappo_l", "ippo_l"):
        tr = MAPPOLagrangian(EnvConfig(T=20), AlgoConfig(algo=algo, n_envs=2, epochs=1, n_minibatch=2, val_every=0))
        buf, vals, met = tr.collect()
        st = tr.update(buf, vals)
        assert np.isfinite(st["pi_loss"]) and np.isfinite(st["v_loss"])
        assert (tr.lam >= 0).all()
        m = run_policy(SAGINDisasterEnv(EnvConfig(T=20), 2, seed=0), ActorPolicy(tr.actor), seed=1)
        assert 0.0 <= m["pdr"].mean() <= 1.0
