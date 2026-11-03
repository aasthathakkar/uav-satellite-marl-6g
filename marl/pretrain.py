"""Optional behaviour-cloning (BC) warm start for the shared actor.

The Greedy-Track heuristic (centralised, uses privileged information) is rolled out and the
decentralised actor is fitted to its actions from the agents' *local* observations, by maximum
likelihood of the hybrid action distribution with the Gaussian standard deviations frozen.
All learned methods can start from the same BC actor, so that differences between them come only
from the RL objective (constrained vs. unconstrained, central vs. local critic).

python -m marl.pretrain --episodes 12 --steps 2000 --out checkpoints/bc_actor.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sagin.baselines import GreedyTrackPolicy  # noqa: E402
from sagin.config import AlgoConfig, EnvConfig  # noqa: E402
from sagin.env import SAGINDisasterEnv  # noqa: E402
from marl.networks import Actor  # noqa: E402


def collect_demos(ecfg, n_batches=12, n_envs=16, seed=777):
    O, Mk, A = [], [], {k: [] for k in ("move", "sched", "power", "bh", "bhp")}
    for b in range(n_batches):
        env = SAGINDisasterEnv(ecfg, n_envs, seed=seed + b)
        pol = GreedyTrackPolicy()
        obs = env.reset(seed=seed + 1000 + b)
        pol.reset(env)
        done = False
        while not done:
            act = pol.act(env, obs)
            O.append(obs["obs"].reshape(-1, env.obs_dim))
            Mk.append(obs["mask"].reshape(-1, env.M))
            for k in A:
                v = np.asarray(act[k])
                A[k].append(v.reshape(env.E * env.U, *v.shape[2:]))
            obs, r, cost, done, info = env.step(act)
        m = env.episode_metrics()
        print(f"demo batch {b}: pdr={m['pdr'].mean():.3f} outage={m['outage'].mean():.3f} aoi={m['aoi'].mean():.2f}",
              flush=True)
    data = dict(obs=np.concatenate(O).astype(np.float32), mask=np.concatenate(Mk))
    for k in A:
        data[k] = np.concatenate(A[k])
    for k in ("move", "power", "bhp"):
        data[k] = np.clip(data[k], -0.97, 0.97).astype(np.float32)
    return data


def behaviour_clone(actor: Actor, data, steps=1500, batch=4096, lr=1e-3, seed=0, verbose=True):
    g = torch.Generator().manual_seed(seed)
    n = len(data["obs"])
    T = {k: torch.from_numpy(v) for k, v in data.items()}
    stds = [actor.move_logstd, actor.pow_logstd, actor.bhp_logstd]
    for p in stds:
        p.requires_grad_(False)
    opt = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=lr)
    t0 = time.time()
    for it in range(steps):
        idx = torch.randint(0, n, (batch,), generator=g)
        act = {k: T[k][idx] for k in ("move", "sched", "power", "bh", "bhp")}
        logp, _ = actor.evaluate(T["obs"][idx], T["mask"][idx], act)
        loss = -logp.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        opt.step()
        if verbose and (it % 250 == 0 or it == steps - 1):
            print(f"BC step {it:5d}  nll={loss.item():.3f}  ({time.time() - t0:.0f}s)", flush=True)
    for p in stds:
        p.requires_grad_(True)
    return actor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12, help="batches of 16 demonstration episodes")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--out", default="checkpoints/bc_actor.pt")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    ecfg, acfg = EnvConfig(), AlgoConfig()
    data = collect_demos(ecfg, a.episodes)
    actor = Actor(SAGINDisasterEnv(ecfg, 1).layout, acfg.hidden, acfg.init_logstd)
    behaviour_clone(actor, data, a.steps, batch=a.batch)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(dict(actor=actor.state_dict(), env_cfg=ecfg.to_dict(), algo_cfg=acfg.to_dict()), a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
