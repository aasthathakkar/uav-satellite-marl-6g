"""Reliability-aware MAPPO-Lagrangian with centralised training and decentralised execution (CTDE).

Constrained Markov game:  max_pi J_R(pi)  s.t.  J_Ci(pi) <= 0,  i in {outage, PDR, AoI}
(the per-slot costs returned by the environment are already shifted by their thresholds).

Lagrangian relaxation:    min_{lambda >= 0} max_pi  J_R(pi) - sum_i lambda_i J_Ci(pi)

* primal step : PPO-clip on the combined advantage  A = (A_R - sum_i lambda_i A_Ci) / (1 + sum_i lambda_i),
                one shared actor for all UAVs (parameter sharing + agent id);
                one centralised critic with 4 heads (reward + 3 costs) trained on the global state;
* dual step   : PI-controlled projected ascent  lambda_i <- [ I_i + K_P * ema(Jhat_Ci) ]_+ ,
                I_i <- [ I_i + K_I * Jhat_Ci ]_+    (K_P = 0 gives plain dual ascent).

Variants (same code path):
  mappo_l   : proposed (central critic, learned multipliers)
  mappo     : unconstrained (lambda = 0)
  mappo_pen : fixed penalty (lambda = penalty_coef, not learned)
  ippo_l    : decentralised critics on local observations, learned multipliers
"""
from __future__ import annotations

import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from sagin.config import AlgoConfig, EnvConfig
from sagin.env import SAGINDisasterEnv
from .networks import Actor, CentralCritic, LocalCritic

ACT_KEYS = ("move", "sched", "power", "bh", "bhp")


class ValueNorm:
    """Debiased running mean/std of value targets (one per head), as recommended for MAPPO."""

    def __init__(self, n, beta=0.99, eps=1e-5):
        self.beta, self.eps = beta, eps
        self.m = np.zeros(n)
        self.sq = np.zeros(n)
        self.debias = 0.0

    def update(self, x):
        x = x.reshape(-1, x.shape[-1])
        b = self.beta
        self.m = b * self.m + (1 - b) * x.mean(0)
        self.sq = b * self.sq + (1 - b) * (x ** 2).mean(0)
        self.debias = b * self.debias + (1 - b)

    def stats(self):
        d = max(self.debias, self.eps)
        mean = self.m / d
        var = np.maximum(self.sq / d - mean ** 2, 1e-4)
        return mean, np.sqrt(var)

    def normalize(self, x):
        mean, std = self.stats()
        return (x - mean) / std

    def denormalize(self, x):
        mean, std = self.stats()
        return x * std + mean

    def state_dict(self):
        return dict(m=self.m.tolist(), sq=self.sq.tolist(), debias=self.debias)

    def load_state_dict(self, d):
        self.m, self.sq, self.debias = np.array(d["m"]), np.array(d["sq"]), d["debias"]


def to_env_action(act, E, U):
    out = {}
    for k in ACT_KEYS:
        v = act[k].cpu().numpy()
        out[k] = v.reshape(E, U, *v.shape[1:])
    return out


class MAPPOLagrangian:
    def __init__(self, env_cfg: EnvConfig, cfg: AlgoConfig):
        self.ecfg, self.cfg = env_cfg, cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.set_num_threads(cfg.torch_threads)
        self.env = SAGINDisasterEnv(env_cfg, cfg.n_envs, seed=cfg.seed)
        self.E, self.U, self.T = cfg.n_envs, env_cfg.n_uav, env_cfg.T
        self.n_cost = SAGINDisasterEnv.N_COST
        self.actor = Actor(self.env.layout, cfg.hidden, cfg.init_logstd)
        self.central = cfg.algo != "ippo_l"
        # central critic: one reward head per agent (hybrid rewards) + one head per team cost
        self.critic = (CentralCritic(self.env.state_layout, cfg.critic_hidden, self.U + self.n_cost) if self.central
                       else LocalCritic(self.env.layout, cfg.critic_hidden, 1 + self.n_cost))
        if cfg.init_actor:
            ck = torch.load(cfg.init_actor, map_location="cpu", weights_only=False)
            self.actor.load_state_dict(ck["actor"])
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor, eps=1e-5)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic, eps=1e-5)
        self.vnorm = ValueNorm(1 + self.n_cost)
        self.learn_lambda = cfg.algo in ("mappo_l", "ippo_l")
        if cfg.algo == "mappo":
            lam0 = 0.0
        elif cfg.algo == "mappo_pen":
            lam0 = cfg.penalty_coef
        else:
            lam0 = cfg.lambda_init
        self.lam = np.full(self.n_cost, lam0, dtype=np.float64)
        self.lam_I = self.lam.copy()
        self.lam_ema = np.zeros(self.n_cost)
        self.it = 0
        self.env_steps = 0
        if cfg.resume:
            ck = torch.load(cfg.resume, map_location="cpu", weights_only=False)
            self.actor.load_state_dict(ck["actor"])
            self.critic.load_state_dict(ck["critic"])
            self.vnorm.load_state_dict(ck["vnorm"])
            self.lam = np.array(ck["lam"], dtype=np.float64)
            self.lam_I = np.array(ck["lam_I"], dtype=np.float64)
            # lambda = I + K_P * ema  ->  recover the EMA term when it was not stored
            self.lam_ema = (np.array(ck["lam_ema"]) if "lam_ema" in ck
                            else (self.lam - self.lam_I) / max(cfg.lambda_kp, 1e-9))
            self.it, self.env_steps = int(ck["it"]), int(ck["env_steps"])

    # ------------------------------------------------------------------ rollout
    def collect(self):
        env, E, U, T = self.env, self.E, self.U, self.T
        obs = env.reset()
        F_o, F_s, M = env.obs_dim, env.state_dim, env.M
        buf = dict(
            obs=np.zeros((T, E, U, F_o), np.float32), mask=np.zeros((T, E, U, M), bool),
            state=np.zeros((T, E, F_s), np.float32), logp=np.zeros((T, E, U), np.float32),
            rew=np.zeros((T, E, U), np.float32), cost=np.zeros((T, E, self.n_cost), np.float32),
            rew_team=np.zeros((T, E), np.float32),
            move=np.zeros((T, E, U, 3), np.float32), sched=np.zeros((T, E, U, env.NC), np.int64),
            power=np.zeros((T, E, U, env.NC), np.float32), bh=np.zeros((T, E, U), np.int64),
            bhp=np.zeros((T, E, U), np.float32),
        )
        vals = np.zeros((T, E, U, 1 + self.n_cost), np.float32)
        for t in range(T):
            o = torch.from_numpy(obs["obs"]).reshape(E * U, -1)
            m = torch.from_numpy(obs["mask"]).reshape(E * U, -1)
            act, logp = self.actor.act(o, m)
            with torch.no_grad():
                if self.central:
                    v = self._central_to_agents(self.critic(torch.from_numpy(obs["state"])).numpy())
                else:
                    v = self.critic(o).numpy().reshape(E, U, -1)
            vals[t] = v
            buf["obs"][t], buf["mask"][t], buf["state"][t] = obs["obs"], obs["mask"], obs["state"]
            buf["logp"][t] = logp.numpy().reshape(E, U)
            ea = to_env_action(act, E, U)
            for k in ACT_KEYS:
                buf[k][t] = ea[k]
            obs, r, cost, done, info = env.step(ea)
            a_mix = self.cfg.reward_mix
            buf["rew"][t] = a_mix * info["reward_agents"] + (1.0 - a_mix) * r[:, None]
            buf["rew_team"][t], buf["cost"][t] = r, cost
        self.env_steps += T * E
        return buf, vals, env.episode_metrics()

    def _central_to_agents(self, v):
        """(..., U + n_cost) central-critic output -> (..., U, 1 + n_cost) per-agent value vectors."""
        U = self.U
        r = v[..., :U, None]
        cst = np.broadcast_to(v[..., None, U:], v.shape[:-1] + (U, self.n_cost))
        return np.concatenate([r, cst], -1)

    # ------------------------------------------------------------------ advantages
    def gae(self, buf, vals_norm):
        c = self.cfg
        V = self.vnorm.denormalize(vals_norm)                          # (T,E,U,4)
        x = np.concatenate([buf["rew"][..., None],
                            np.broadcast_to(buf["cost"][:, :, None, :], buf["rew"].shape + (self.n_cost,))], -1)
        T = V.shape[0]
        adv = np.zeros_like(V)
        last = np.zeros_like(V[0])
        for t in reversed(range(T)):
            nv = V[t + 1] if t + 1 < T else 0.0
            delta = x[t] + c.gamma * nv - V[t]
            last = delta + c.gamma * c.gae_lambda * last
            adv[t] = last
        return adv, adv + V

    # ------------------------------------------------------------------ PPO + dual update
    def update(self, buf, vals):
        c, E, U, T = self.cfg, self.E, self.U, self.T
        adv, ret = self.gae(buf, vals)
        self.vnorm.update(ret)
        ret_n = self.vnorm.normalize(ret).astype(np.float32)
        lam = self.lam
        a = (adv[..., 0] - (adv[..., 1:] * lam).sum(-1)) / (1.0 + lam.sum())
        a = ((a - a.mean()) / (a.std() + 1e-8)).astype(np.float32)   # (T,E,U)

        n = T * E
        flat = lambda x: torch.from_numpy(np.ascontiguousarray(x.reshape(n, *x.shape[2:])))  # noqa: E731
        obs, mask, state = flat(buf["obs"]), flat(buf["mask"]), flat(buf["state"])
        acts = {k: flat(buf[k]) for k in ACT_KEYS}
        logp_old, A, R = flat(buf["logp"]), flat(a), flat(ret_n)
        stats = dict(pi_loss=0.0, v_loss=0.0, entropy=0.0, kl=0.0, clipfrac=0.0)
        n_upd = 0
        mb_size = n // c.n_minibatch
        for _ in range(c.epochs):
            perm = torch.randperm(n)
            for i in range(c.n_minibatch):
                idx = perm[i * mb_size:(i + 1) * mb_size]
                B = len(idx)
                o = obs[idx].reshape(B * U, -1)
                m = mask[idx].reshape(B * U, -1)
                act = {k: acts[k][idx].reshape(B * U, *acts[k].shape[2:]) for k in ACT_KEYS}
                logp, ent = self.actor.evaluate(o, m, act)
                lo = logp_old[idx].reshape(-1)
                adv_b = A[idx].reshape(-1)
                ratio = torch.exp(logp - lo)
                s1 = ratio * adv_b
                s2 = torch.clamp(ratio, 1 - c.clip, 1 + c.clip) * adv_b
                pi_loss = -torch.min(s1, s2).mean() - c.ent_coef * ent.mean()
                if self.it >= c.critic_warmup:
                    self.opt_a.zero_grad()
                    pi_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), c.max_grad_norm)
                    self.opt_a.step()

                if self.central:
                    v = self.critic(state[idx])
                    Rb = R[idx]                                   # (B,U,4)
                    tgt = torch.cat([Rb[..., 0], Rb[:, 0, 1:]], -1)  # (B, U + n_cost)
                else:
                    v = self.critic(o)
                    tgt = R[idx].reshape(B * U, -1)
                v_loss = F.huber_loss(v, tgt, delta=10.0)
                self.opt_c.zero_grad()
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.opt_c.step()
                with torch.no_grad():
                    stats["pi_loss"] += pi_loss.item()
                    stats["v_loss"] += v_loss.item()
                    stats["entropy"] += ent.mean().item()
                    stats["kl"] += (lo - logp).mean().item()
                    stats["clipfrac"] += ((ratio - 1).abs() > c.clip).float().mean().item()
                n_upd += 1
        for k in stats:
            stats[k] /= max(n_upd, 1)

        # dual (Lagrange multiplier) update with the rollout estimate of every constraint
        jc = buf["cost"].mean((0, 1)).astype(np.float64)
        if self.learn_lambda:
            sig = np.clip(jc, -c.lambda_signal_clip, c.lambda_signal_clip)
            self.lam_I = np.clip(self.lam_I + c.lambda_lr * sig, 0.0, c.lambda_max)
            self.lam_ema = 0.9 * self.lam_ema + 0.1 * sig if (self.it > 0 or self.cfg.resume) else sig.copy()
            self.lam = np.clip(self.lam_I + c.lambda_kp * self.lam_ema, 0.0, c.lambda_max)
        stats["jc"] = jc
        return stats

    # ------------------------------------------------------------------ validation / model selection
    def validate(self):
        """Deterministic roll-out on fixed held-out scenarios (same seeds every call)."""
        from sagin.baselines import run_policy
        if not hasattr(self, "val_env"):
            self.val_env = SAGINDisasterEnv(self.ecfg, self.cfg.n_envs, seed=self.cfg.val_seed)
        m = run_policy(self.val_env, ActorPolicy(self.actor, deterministic=True), severity=None,
                       seed=self.cfg.val_seed + 1)
        return {k: float(np.mean(v)) for k, v in m.items()}

    def violation(self, m):
        e = self.ecfg
        return (max(0.0, m["outage"] / e.eps_out - 1.0) + max(0.0, (e.pdr_min - m["pdr"]) / (1.0 - e.pdr_min))
                + max(0.0, m["aoi"] / e.aoi_max - 1.0))

    # ------------------------------------------------------------------ main loop
    def train(self, n_iters=None, log_path=None, ckpt_dir=None, verbose=True):
        c = self.cfg
        n_iters = n_iters or c.n_iters
        fields = ["iter", "env_steps", "time", "reward", "goodput_mbps", "pdr", "outage", "outage_acc",
                  "outage_bh", "aoi", "aoi_viol", "e_uav_kj", "e_gu_j", "jain", "frac_gw", "frac_hold",
                  "handovers", "conflicts", "collisions", "jc_outage", "jc_pdr", "jc_aoi",
                  "lam_outage", "lam_pdr", "lam_aoi", "pi_loss", "v_loss", "entropy", "kl", "clipfrac"]
        f = None
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            f = open(log_path, "w", newline="")
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
        t0 = time.time()
        vf, vw = None, None
        best = (-1, -np.inf)  # (feasible flag, score)
        vfields = ["iter", "env_steps", "reward", "goodput_mbps", "pdr", "outage", "outage_acc", "outage_bh",
                   "aoi", "aoi_viol", "e_uav_kj", "e_gu_j", "jain", "violation"]
        if log_path:
            vf = open(os.path.join(os.path.dirname(log_path), "val.csv"), "w", newline="")
            vw = csv.DictWriter(vf, fieldnames=vfields)
            vw.writeheader()

        def do_val():
            nonlocal best
            vm = self.validate()
            vm["violation"] = self.violation(vm)
            vrow = {k: vm.get(k) for k in vfields}
            vrow["iter"], vrow["env_steps"] = self.it, self.env_steps
            if vw:
                vw.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in vrow.items()})
                vf.flush()
            feas = 1 if vm["violation"] <= 0.0 else 0
            score = vm["reward"] if feas else -vm["violation"]
            if ckpt_dir and (feas, score) > best:
                best = (feas, score)
                self.save(os.path.join(ckpt_dir, "best.pt"))
            if verbose:
                print(f"   [val] it {self.it} R={vm['reward']:.1f} pdr={vm['pdr']:.3f} out={vm['outage']:.3f} "
                      f"aoi={vm['aoi']:.2f} viol={vm['violation']:.3f} {'feasible' if feas else ''}", flush=True)

        do_val()
        for _ in range(n_iters):
            buf, vals, met = self.collect()
            st = self.update(buf, vals)
            self.it += 1
            row = dict(iter=self.it, env_steps=self.env_steps, time=round(time.time() - t0, 1))
            for k in ["reward", "goodput_mbps", "pdr", "outage", "outage_acc", "outage_bh", "aoi", "aoi_viol",
                      "e_uav_kj", "e_gu_j", "jain", "frac_gw", "frac_hold", "handovers", "conflicts", "collisions"]:
                row[k] = float(np.mean(met[k]))
            for i, nm in enumerate(SAGINDisasterEnv.COST_NAMES):
                row[f"jc_{nm}"] = float(st["jc"][i])
                row[f"lam_{nm}"] = float(self.lam[i])
            for k in ["pi_loss", "v_loss", "entropy", "kl", "clipfrac"]:
                row[k] = st[k]
            if f:
                w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
                f.flush()
            if verbose and (self.it % c.log_every == 0):
                print(f"[{c.algo} s{c.seed}] it {self.it:4d} t={row['time']:7.1f}s R={row['reward']:7.2f} "
                      f"gp={row['goodput_mbps']:5.2f} pdr={row['pdr']:.3f} out={row['outage']:.3f} "
                      f"aoi={row['aoi']:5.2f} lam=[{self.lam[0]:.2f},{self.lam[1]:.2f},{self.lam[2]:.2f}] "
                      f"ent={row['entropy']:.2f} kl={row['kl']:.4f} cf={row['clipfrac']:.3f}", flush=True)
            if c.val_every and self.it % c.val_every == 0:
                do_val()
            if ckpt_dir and (self.it % c.save_every == 0 or self.it == n_iters):
                self.save(os.path.join(ckpt_dir, f"ckpt_{self.it:04d}.pt"))
        if ckpt_dir:
            self.save(os.path.join(ckpt_dir, "final.pt"))
        if f:
            f.close()
        if vf:
            vf.close()

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                        lam=self.lam.tolist(), lam_I=self.lam_I.tolist(), lam_ema=self.lam_ema.tolist(),
                        vnorm=self.vnorm.state_dict(),
                        it=self.it, env_steps=self.env_steps,
                        env_cfg=self.ecfg.to_dict(), algo_cfg=self.cfg.to_dict()), path)


def load_actor(path, env_layout=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ecfg = EnvConfig(**ck["env_cfg"])
    acfg = AlgoConfig(**ck["algo_cfg"])
    layout = env_layout or SAGINDisasterEnv(ecfg, 1).layout
    actor = Actor(layout, acfg.hidden, acfg.init_logstd)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    return actor, ecfg, acfg, ck


class ActorPolicy:
    """Wraps a trained actor so that it can be rolled out like the heuristic baselines."""

    def __init__(self, actor, deterministic=True, name="MAPPO-L"):
        self.actor, self.det, self.name = actor, deterministic, name

    def reset(self, env):
        pass

    def act(self, env, obs):
        E, U = env.E, env.U
        o = torch.from_numpy(obs["obs"]).reshape(E * U, -1)
        m = torch.from_numpy(obs["mask"]).reshape(E * U, -1)
        act, _ = self.actor.act(o, m, deterministic=self.det)
        return to_env_action(act, E, U)
