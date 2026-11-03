"""Turn the evaluation CSVs into Markdown tables (used verbatim in README.md and docs/REPORT.md).

python scripts/summarize.py --results results
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sagin.config import EnvConfig  # noqa: E402

ORDER = ["MAPPO-L (proposed)", "MAPPO-L, 600 iterations", "MAPPO", "MAPPO-Pen", "IPPO-L", "MAPPO-L, no safety margin",
         "MAPPO-L, no warm start", "Greedy-Track", "Static-KMeans", "Random"]


def fmt(mu, sd, nd):
    return f"{mu:.{nd}f}" if (sd is None or not np.isfinite(sd) or sd == 0) else f"{mu:.{nd}f} ± {sd:.{nd}f}"


def main_table(df, c):
    methods = [m for m in ORDER if m in df.method.unique()]
    seed_mean = df.groupby(["method", "seed"])[["reward", "goodput_mbps", "pdr", "outage", "aoi", "e_uav_kj",
                                                 "e_gu_j", "jain"]].mean()
    rows = ["| Method | Reward | Goodput [Mbit/s] | PDR (≥ %.2f) | Outage (≤ %.2f) | Mean AoI (≤ %.0f) | UAV energy [kJ] | "
            "User energy [J] | Jain | Constraints met |" % (c.pdr_min, c.eps_out, c.aoi_max),
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m in methods:
        sm = seed_mean.loc[m]
        mu = sm.mean()
        sd = sm.std() if len(sm) > 1 else pd.Series(0.0, index=sm.columns)
        ok = [mu.pdr >= c.pdr_min, mu.outage <= c.eps_out, mu.aoi <= c.aoi_max]
        mark = " ".join(("✓" if o else "✗") + n for o, n in zip(ok, ["PDR", "out", "AoI"]))
        name = f"**{m}**" if m.startswith("MAPPO-L (") else m
        rows.append(f"| {name} | {fmt(mu.reward, sd.reward, 1)} | {fmt(mu.goodput_mbps, sd.goodput_mbps, 2)} | "
                    f"{fmt(mu.pdr, sd.pdr, 3)} | {fmt(mu.outage, sd.outage, 3)} | {fmt(mu.aoi, sd.aoi, 2)} | "
                    f"{fmt(mu.e_uav_kj, sd.e_uav_kj, 1)} | {fmt(mu.e_gu_j, sd.e_gu_j, 2)} | {mu.jain:.3f} | {mark} |")
    return "\n".join(rows)


def behaviour_table(df):
    methods = [m for m in ORDER if m in df.method.unique()]
    cols = [("alt_m", "Mean altitude [m]", 0), ("speed_ms", "Mean speed [m/s]", 1), ("idle_rb", "Idle RBs", 3),
            ("frac_gw", "Backhaul via gateway", 3), ("frac_hold", "Backhaul hold", 3),
            ("handovers", "Backhaul handovers / episode", 1), ("conflicts", "Association conflicts / slot", 2)]
    cols = [c for c in cols if c[0] in df.columns]
    rows = ["| Method | " + " | ".join(c[1] for c in cols) + " |", "| --- |" + " --- |" * len(cols)]
    for m in methods:
        d = df[df.method == m]
        rows.append(f"| {m} | " + " | ".join(f"{d[c].mean():.{nd}f}" for c, _, nd in cols) + " |")
    return "\n".join(rows)


def sweep_table(df, metric, nd, label):
    methods = [m for m in ORDER if m in df.method.unique() and m != "Random"]
    piv = df.groupby(["method", "sweep"])[metric].mean().unstack("sweep")
    cols = list(piv.columns)
    rows = [f"| {label} | " + " | ".join(f"{x:g}" for x in cols) + " |", "| --- |" + " --- |" * len(cols)]
    for m in methods:
        rows.append(f"| {m} | " + " | ".join(f"{piv.loc[m, x]:.{nd}f}" for x in cols) + " |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    c = EnvConfig()
    out = []
    df = pd.read_csv(os.path.join(a.results, "eval_main.csv"))
    n_ep = int(df.groupby(["method", "seed"]).size().max())
    out.append(f"### Main comparison ({n_ep} held-out episodes per policy, severity ~ U[0,1]; "
               "learned methods: mean ± std over seeds)\n")
    out.append(main_table(df, c))
    out.append("\n### Learned behaviour (same episodes)\n")
    out.append(behaviour_table(df))
    for fn, lab in [("sweep_severity.csv", "severity σ"), ("sweep_load.csv", "load ×")]:
        p = os.path.join(a.results, fn)
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        for met, nd, name in [("pdr", 3, "PDR"), ("outage", 3, "Outage"), ("aoi", 2, "Mean AoI"), ("e_uav_kj", 1, "UAV energy [kJ]")]:
            out.append(f"\n### {name} vs {lab}\n")
            out.append(sweep_table(d, met, nd, f"{name} \\ {lab}"))
    text = "\n".join(out) + "\n"
    with open(os.path.join(a.results, "tables.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
