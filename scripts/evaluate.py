"""Evaluate trained MARL policies and heuristic baselines on held-out disaster scenarios.

Produces (in --out, default results/):
  eval_main.csv       per-episode metrics, severity ~ U[0,1]                (main comparison table)
  sweep_severity.csv  per-episode metrics for fixed severities             (robustness to link failures)
  sweep_load.csv      per-episode metrics for scaled traffic load          (robustness to traffic)
  summary_main.csv    mean / std per method (main setting)
  traj_<method>.npz   one recorded episode (trajectories, backhaul choices) for plotting

Example
-------
python scripts/evaluate.py --method MAPPO-L=checkpoints/mappo_l_s0/final.pt,checkpoints/mappo_l_s1/final.pt \
                           --method MAPPO=checkpoints/mappo_s0/final.pt --episodes 64
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from sagin.baselines import GreedyTrackPolicy, RandomPolicy, StaticKMeansPolicy, run_policy  # noqa: E402
from sagin.config import EnvConfig  # noqa: E402
from sagin.env import SAGINDisasterEnv  # noqa: E402
from marl.mappo import ActorPolicy, load_actor  # noqa: E402

METRICS = ["reward", "goodput_mbps", "pdr", "outage", "outage_acc", "outage_bh", "aoi", "aoi_viol",
           "e_uav_kj", "e_gu_j", "jain", "frac_gw", "frac_hold", "handovers", "conflicts", "collisions",
           "cost_outage", "cost_pdr", "cost_aoi", "alt_m", "speed_ms", "idle_rb"]

BASELINE_FACTORIES = {
    "Random": lambda: RandomPolicy(seed=0),
    "Static-KMeans": lambda: StaticKMeansPolicy(),
    "Greedy-Track": lambda: GreedyTrackPolicy(),
}


def rollout_rows(make_policy, ecfg, severity, n_episodes, seed0, n_envs=16, extra=None):
    rows = []
    n_batches = int(np.ceil(n_episodes / n_envs))
    for b in range(n_batches):
        env = SAGINDisasterEnv(ecfg, n_envs, seed=seed0 + b)
        torch.manual_seed(seed0 + 31 * b)  # reproducible action sampling of the learned policies
        m = run_policy(env, make_policy(), severity=severity, seed=seed0 + 7919 * (b + 1))
        for e in range(n_envs):
            r = {k: float(m[k][e]) for k in METRICS}
            r["severity"] = float(m["severity"][e])
            r["episode"] = b * n_envs + e
            r.update(extra or {})
            rows.append(r)
    return rows[:n_episodes] if len(rows) > n_episodes else rows


def build_methods(args):
    methods = {}  # label -> list of (seed_label, factory)
    for name, fac in BASELINE_FACTORIES.items():
        methods[name] = [("-", fac)]
    for spec in args.method or []:
        label, paths = spec.split("=", 1)
        lst = []
        for p in paths.split(","):
            actor, _, acfg, _ = load_actor(p)
            lst.append((f"s{acfg.seed}", (lambda a=actor: ActorPolicy(a, deterministic=args.deterministic))))
        methods[label] = lst
    return methods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", action="append", help="LABEL=ckpt1[,ckpt2,...] (one entry per seed)")
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--sweep-episodes", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="results")
    ap.add_argument("--deterministic", action="store_true",
                    help="use the mode of every action head instead of sampling (default: sample, i.e. the policy as trained)")
    ap.add_argument("--skip-sweeps", action="store_true")
    ap.add_argument("--skip-main", action="store_true", help="only run the sweeps")
    ap.add_argument("--sweeps", nargs="*", default=["severity", "load"], help="which sweeps to run")
    ap.add_argument("--only", nargs="*", help="evaluate only these method labels")
    ap.add_argument("--sweep-only", nargs="*", help="restrict the sweeps to these method labels")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ecfg = EnvConfig()
    methods = build_methods(a)
    if a.only:
        methods = {k: v for k, v in methods.items() if k in a.only}

    if not a.skip_main:
        main_eval(a, ecfg, methods)
    if a.skip_sweeps:
        return
    run_sweeps(a, methods)


def main_eval(a, ecfg, methods):
    # ---------------- main comparison (severity ~ U[0,1])
    rows = []
    for label, lst in methods.items():
        for seed_label, fac in lst:
            rows += rollout_rows(fac, ecfg, None, a.episodes, a.seed, extra=dict(method=label, seed=seed_label))
        df = pd.DataFrame([r for r in rows if r["method"] == label])
        print(f"{label:16s} " + " ".join(f"{k}={df[k].mean():.3f}" for k in
                                        ["reward", "goodput_mbps", "pdr", "outage", "aoi", "e_uav_kj", "e_gu_j"]), flush=True)
    main_df = pd.DataFrame(rows)
    main_df.to_csv(os.path.join(a.out, "eval_main.csv"), index=False)
    g = main_df.groupby("method", sort=False)[METRICS]
    summ = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std"))
    # seed-level spread of the learned methods (std of per-seed means)
    seed_means = main_df.groupby(["method", "seed"], sort=False)[METRICS].mean()
    summ = summ.join(seed_means.groupby("method", sort=False).std().add_suffix("_seedstd"))
    summ.to_csv(os.path.join(a.out, "summary_main.csv"))

    # ---------------- recorded episode for trajectory / backhaul plots
    for label, lst in methods.items():
        if label == "Random":
            continue
        env = SAGINDisasterEnv(ecfg, 1, seed=a.seed)
        torch.manual_seed(a.seed)
        m, traj = run_policy(env, lst[0][1](), severity=0.7, seed=a.seed + 1, record=True)
        elev = np.stack([env.sat_geometry(t * ecfg.tau)[0][0] for t in range(ecfg.T)])
        np.savez(os.path.join(a.out, f"traj_{label}.npz"),
                 uav=np.stack([x["uav"][0] for x in traj]), bh=np.stack([x["bh_exec"][0] for x in traj]),
                 out_bh=np.stack([x["out_bh"][0] for x in traj]), gu=env.gu[0], gw=env.gw[0],
                 gw_range=ecfg.gw_range, elev=elev, area=ecfg.area,
                 metrics=np.array([float(m[k][0]) for k in METRICS]), metric_names=np.array(METRICS))



def run_sweeps(a, methods):
    ecfg = EnvConfig()
    if a.sweep_only:
        methods = {k: v for k, v in methods.items() if k in a.sweep_only}
    # ---------------- severity sweep
    rows = []
    for sev in ([0.0, 0.25, 0.5, 0.75, 1.0] if "severity" in a.sweeps else []):
        for label, lst in methods.items():
            for seed_label, fac in lst:
                rows += rollout_rows(fac, ecfg, sev, a.sweep_episodes, a.seed + 101,
                                     extra=dict(method=label, seed=seed_label, sweep=sev))
        print(f"severity {sev} done", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "sweep_severity.csv"), index=False)

    # ---------------- traffic load sweep
    rows = []
    for ls in ([0.6, 0.8, 1.0, 1.2, 1.4] if "load" in a.sweeps else []):
        cfg = EnvConfig(load_scale=ls)
        for label, lst in methods.items():
            for seed_label, fac in lst:
                rows += rollout_rows(fac, cfg, None, a.sweep_episodes, a.seed + 202,
                                     extra=dict(method=label, seed=seed_label, sweep=ls))
        print(f"load {ls} done", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "sweep_load.csv"), index=False)


if __name__ == "__main__":
    main()
