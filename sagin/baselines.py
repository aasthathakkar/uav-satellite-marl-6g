"""Non-learning benchmark policies.

* RandomPolicy        : uniformly random feasible actions.
* StaticKMeansPolicy  : UAVs hover over k-means centroids of the users (fixed altitude), max-SNR
                        association with queue-weighted max-rate scheduling, full power, and
                        "best visible link" backhaul (gateway if reachable, else highest satellite).
* GreedyTrackPolicy   : centralised greedy heuristic (tuned by grid search: 200 m altitude, 8 dB fade
                        margin, 5 dB admission margin): UAVs track demand-weighted Voronoi centroids
                        (Lloyd step every slot), AoI-aware max-weight scheduling on outdated CSI that
                        leaves an RB idle rather than schedule a user likely to be in outage,
                        channel-inversion power control, and load-aware greedy backhaul assignment.

All policies return the action dict expected by ``SAGINDisasterEnv.step``.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .env import SAGINDisasterEnv, db2lin, lin2db


def _move_towards(env, target_xy, target_h, speed=None):
    c = env.cfg
    d = target_xy - env.uav[..., :2]
    dist = np.linalg.norm(d, axis=-1, keepdims=True)
    v = env.cfg.v_max if speed is None else speed
    step = np.minimum(dist / c.tau, v)
    vxy = d / np.maximum(dist, 1e-6) * step / c.v_max
    vz = np.clip((target_h - env.uav[..., 2]) / (c.vz_max * c.tau), -1.0, 1.0)
    return np.concatenate([vxy, vz[..., None]], -1)


def _schedule(env, priority):
    """Pick the n_rb highest-priority candidates of each UAV (priority: (E,U,N))."""
    return np.argsort(-priority, axis=-1)[..., :env.NC]


class RandomPolicy:
    name = "Random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self, env):
        pass

    def act(self, env: SAGINDisasterEnv, obs):
        E, U, NC, N = env.E, env.U, env.NC, env.N
        rng = self.rng
        mask = obs["mask"].copy()
        mask[..., 0] = ~mask[..., 1:].any(-1)  # hold only if nothing else is feasible
        logits = np.where(mask, rng.random(mask.shape), -1.0)
        return dict(
            move=rng.uniform(-1, 1, (E, U, 3)),
            sched=np.argsort(rng.random((E, U, N)), axis=-1)[..., :NC],
            power=rng.uniform(-1, 1, (E, U, NC)),
            bh=np.argmax(logits, -1),
            bhp=rng.uniform(-1, 1, (E, U)),
        )


class StaticKMeansPolicy:
    name = "Static-KMeans"

    def __init__(self, height=150.0, iters=20, seed=0):
        self.h = height
        self.iters = iters
        self.rng = np.random.default_rng(seed)

    def reset(self, env):
        E, U = env.E, env.U
        self.targets = np.zeros((E, U, 2))
        for e in range(E):
            X = env.gu[e]
            cent = X[self.rng.choice(len(X), U, replace=False)]
            for _ in range(self.iters):
                lab = np.argmin(((X[:, None] - cent[None]) ** 2).sum(-1), 1)
                for j in range(U):
                    if (lab == j).any():
                        cent[j] = X[lab == j].mean(0)
            cost = np.linalg.norm(env.uav[e, :, None, :2] - cent[None], axis=-1)
            ri, ci = linear_sum_assignment(cost)
            self.targets[e, ri] = cent[ci]

    def _sched(self, env):
        N, K = env.N, env.K
        cand = env.cand_idx
        snr_own = 10 * np.log10(np.maximum(env.p_gu_max * env._gbar_obs / env.noise_rb, 1e-12))
        best_u = np.argmax(env._gbar_obs, axis=1)  # (E,K)
        own = np.take_along_axis(best_u[:, None, :].repeat(env.U, 1), cand, -1) == np.arange(env.U)[None, :, None]
        q = np.take_along_axis(np.broadcast_to(env.Q.sum(-1)[:, None], (env.E, env.U, K)), cand, -1)
        rate = np.log2(1 + db2lin(np.take_along_axis(snr_own, cand, -1)))
        prio = (q > 1e-6) * q * rate + 1e3 * own * (q > 1e-6)
        return _schedule(env, prio)

    def _backhaul(self, env, obs):
        S = env.S
        elev, _, vis, _ = env.sat_geometry(env.t * env.cfg.tau)
        score = np.where(vis, np.sin(elev), -1.0)  # (E,S)
        best = np.argmax(score, -1) + 1
        bh = np.where(vis.any(-1), best, 0)[:, None].repeat(env.U, 1)
        return np.where(obs["mask"][..., S + 1], S + 1, bh)

    def act(self, env, obs):
        E, U, NC = env.E, env.U, env.NC
        return dict(
            move=_move_towards(env, self.targets, self.h),
            sched=self._sched(env),
            power=np.ones((E, U, NC)),
            bh=self._backhaul(env, obs),
            bhp=np.ones((E, U)),
        )


class GreedyTrackPolicy:
    name = "Greedy-Track"

    def __init__(self, height=200.0, margin_db=8.0, aoi_w=1.0, min_snr_db=5.0, seed=0):
        self.h = height
        self.margin = margin_db
        self.aoi_w = aoi_w
        self.min_snr_db = min_snr_db  # if set: leave RBs idle rather than schedule weak users

    def reset(self, env):
        pass

    def act(self, env: SAGINDisasterEnv, obs):
        c, E, U, K, S, NC = env.cfg, env.E, env.U, env.K, env.S, env.NC
        cand = env.cand_idx
        gbar = env._gbar_obs                                     # (E,U,K)
        q = env.Q.sum(-1)
        urg = q + self.aoi_w * env.aoi / c.aoi_max               # demand weight
        # Lloyd step on the demand-weighted Voronoi partition (association = strongest UAV)
        assoc = np.argmax(gbar, axis=1)                          # (E,K)
        oh = (assoc[:, None, :] == np.arange(U)[None, :, None]).astype(float)
        w = oh * (urg[:, None, :] + 1e-3)
        tgt = np.einsum("euk,ekd->eud", w, env.gu) / np.maximum(w.sum(-1, keepdims=True), 1e-9)
        empty = oh.sum(-1) < 0.5
        tgt = np.where(empty[..., None], env.uav[..., :2], tgt)
        move = _move_towards(env, tgt, self.h)

        # AoI-aware max-weight scheduling restricted to own (best-served) users
        own = np.take_along_axis(np.broadcast_to(assoc[:, None], (E, U, K)), cand, -1) == np.arange(U)[None, :, None]
        qc = np.take_along_axis(np.broadcast_to(q[:, None], (E, U, K)), cand, -1)
        ac = np.take_along_axis(np.broadcast_to(env.aoi[:, None], (E, U, K)), cand, -1)
        gc = np.take_along_axis(env.g_last, cand, -1)            # outdated CSI of the last slot
        rate = np.log2(1 + env.p_gu_max * gc / env.noise_rb)
        prio = (qc > 1e-6) * (ac / c.aoi_max + qc) * rate + 1e3 * own * (qc > 1e-6)
        if self.min_snr_db is not None:
            snr_last = lin2db(env.p_gu_max * gc / env.noise_rb) - env.iot_last[..., None]
            ok = (snr_last >= c.gamma_th_db + self.min_snr_db) & (qc > 1e-6)
            prio = np.where(ok, prio, -1.0)
            prio = np.concatenate([prio, np.zeros((E, U, NC))], -1)  # idle options score 0
        sched = _schedule(env, prio)

        # channel-inversion power control with fade margin
        g_s = np.take_along_axis(gc, np.minimum(sched, env.N - 1), -1)
        p_req_dbm = c.gamma_th_db + self.margin + lin2db(env.noise_rb / np.maximum(g_s, 1e-30)) + 30.0
        p_dbm = np.clip(p_req_dbm, c.p_gu_min_dbm, c.p_gu_max_dbm)
        power = 2.0 * (p_dbm - c.p_gu_min_dbm) / (c.p_gu_max_dbm - c.p_gu_min_dbm) - 1.0

        # load-aware greedy backhaul assignment (largest buffer first)
        t_sec = env.t * c.tau
        elev, slant, vis, _ = env.sat_geometry(t_sec)
        snr_sat = db2lin(env.sat_snr_db(elev, slant, c.p_bh_max_dbm))     # (E,U,S)
        rg, d3 = env.gw_geometry()
        snr_gw = db2lin(env.gw_snr_db(d3, c.p_bh_max_dbm))                # (E,U)
        mask = obs["mask"]
        buf = env.B.sum((2, 3))
        order = np.argsort(-buf, axis=-1)
        bh = np.zeros((E, U), dtype=np.int64)
        load = np.zeros((E, env.M))
        rel = np.concatenate([np.ones((E, U, 1)), np.where(env.ge_good, 1.0, c.ge_pbg), np.ones((E, U, 1))], -1)
        for j in range(U):
            u = order[:, j]
            ar = np.arange(E)
            n1 = load + 1.0
            r_sat = (c.b_sat / n1[:, 1:S + 1]) * np.log2(1 + snr_sat[ar, u] * n1[:, 1:S + 1])
            r_gw = np.minimum((c.b_gw / n1[:, S + 1]) * np.log2(1 + snr_gw[ar, u] * n1[:, S + 1]), c.gw_cap / n1[:, S + 1])
            est = np.concatenate([np.zeros((E, 1)), r_sat, r_gw[:, None]], -1) * rel[ar, u]
            est = np.where(mask[ar, u], est, -1.0)
            est[:, 0] = 0.0
            ch = np.argmax(est, -1)
            bh[ar, u] = ch
            load[ar, ch] += (ch > 0)
        return dict(move=move, sched=sched, power=power, bh=bh, bhp=np.ones((E, U)))


BASELINES = {
    "random": RandomPolicy,
    "static_kmeans": StaticKMeansPolicy,
    "greedy_track": GreedyTrackPolicy,
}


def run_policy(env: SAGINDisasterEnv, policy, severity=None, seed=None, record=False):
    """Roll out one synchronous batch of episodes and return per-episode metrics (dict of arrays)."""
    obs = env.reset(severity=severity, seed=seed)
    policy.reset(env)
    traj = []
    done = False
    while not done:
        act = policy.act(env, obs)
        if record:
            traj.append(dict(uav=env.uav.copy(), bh=np.asarray(act["bh"]).copy()))
        obs, r, cost, done, info = env.step(act)
        if record:
            traj[-1].update(bh_exec=info["bh"].copy(), out_bh=info["out_bh"].copy())
    m = env.episode_metrics()
    return (m, traj) if record else m
