"""Train MAPPO-Lagrangian (or a baseline variant) on the disaster SAGIN environment.

Examples
--------
python scripts/train.py --algo mappo_l --seed 0 --iters 400
python scripts/train.py --algo mappo   --seed 0 --iters 400          # unconstrained
python scripts/train.py --algo mappo_pen --penalty 1.0 --seed 0      # fixed penalty
python scripts/train.py --algo ippo_l  --seed 0                      # decentralised critics
python scripts/train.py --algo mappo_l --env eps_out=0.05 aoi_max=8  # override any EnvConfig field
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sagin.config import AlgoConfig, EnvConfig  # noqa: E402
from marl.mappo import MAPPOLagrangian  # noqa: E402


def parse_overrides(pairs, cfg):
    for p in pairs or []:
        k, v = p.split("=", 1)
        cur = getattr(cfg, k)
        setattr(cfg, k, type(cur)(v) if not isinstance(cur, bool) else v.lower() in ("1", "true", "yes"))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="mappo_l", choices=["mappo_l", "mappo", "mappo_pen", "ippo_l"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--penalty", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--env", nargs="*", help="EnvConfig overrides, e.g. eps_out=0.05")
    ap.add_argument("--algo-cfg", nargs="*", help="AlgoConfig overrides, e.g. lr_actor=1e-4")
    a = ap.parse_args()

    ecfg = parse_overrides(a.env, EnvConfig())
    name = f"{a.algo}{('_' + a.tag) if a.tag else ''}_s{a.seed}"
    acfg = AlgoConfig(algo=a.algo, seed=a.seed, n_iters=a.iters, n_envs=a.envs, penalty_coef=a.penalty,
                      torch_threads=a.threads, out_dir=os.path.join(a.out, name))
    acfg = parse_overrides(a.algo_cfg, acfg)
    os.makedirs(acfg.out_dir, exist_ok=True)
    with open(os.path.join(acfg.out_dir, "config.json"), "w") as f:
        json.dump(dict(env=ecfg.to_dict(), algo=acfg.to_dict()), f, indent=2)
    trainer = MAPPOLagrangian(ecfg, acfg)
    trainer.train(a.iters, log_path=os.path.join(acfg.out_dir, "progress.csv"),
                  ckpt_dir=os.path.join(a.ckpt, name))


if __name__ == "__main__":
    main()
