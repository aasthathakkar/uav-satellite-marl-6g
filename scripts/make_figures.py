"""Generate all figures of the report from training logs (runs/) and evaluation results (results/).

python scripts/make_figures.py --runs runs --results results --out results/figures
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch  # noqa: E402

# ---------------------------------------------------------------- style (validated reference palette)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
METHOD_STYLE = {  # colour follows the method (entity), plus a secondary encoding (line style / marker)
    "MAPPO-L (proposed)": dict(color=CAT[0], ls="-", marker="o"),
    "MAPPO": dict(color=CAT[1], ls="--", marker="s"),
    "MAPPO-Pen": dict(color=CAT[2], ls="-.", marker="^"),
    "IPPO-L": dict(color=CAT[3], ls=":", marker="D"),
    "Greedy-Track": dict(color=CAT[4], ls="-", marker="v"),
    "Static-KMeans": dict(color=CAT[5], ls="--", marker="P"),
    "Random": dict(color=CAT[6], ls=":", marker="X"),
    "MAPPO-L, no warm start": dict(color=CAT[7], ls="--", marker="o"),
    "MAPPO-L, no safety margin": dict(color="#86b6ef", ls="-", marker="o"),
    "MAPPO-L, 600 iterations": dict(color="#184f95", ls="-", marker="*"),
}
ALGO_LABEL = {"mappo_l": "MAPPO-L (proposed)", "mappo": "MAPPO", "mappo_pen": "MAPPO-Pen", "ippo_l": "IPPO-L",
              "mappo_l_scratch": "MAPPO-L, no warm start", "mappo_l_nomargin": "MAPPO-L, no safety margin"}
MAIN_ALGOS = ["MAPPO-L (proposed)", "MAPPO", "MAPPO-Pen", "IPPO-L"]
from sagin.config import EnvConfig  # noqa: E402

_C = EnvConfig()
THRESH = dict(outage=_C.eps_out, pdr=_C.pdr_min, aoi=_C.aoi_max)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.8,
    "axes.axisbelow": True,
})


def style(m):
    return METHOD_STYLE.get(m, dict(color=INK2, ls="-", marker="o"))


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def threshold(ax, y, label):
    ax.axhline(y, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.annotate(label, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(-4, 3),
                textcoords="offset points", ha="right", va="bottom", fontsize=7.5, color=INK2)


# ---------------------------------------------------------------- training curves
def load_runs(runs_dir):
    out = {}
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "progress.csv"))):
        name = os.path.basename(os.path.dirname(path))
        algo = name.rsplit("_s", 1)[0]
        if algo not in ALGO_LABEL:  # e.g. mappo_l_long (continuations) are plotted separately
            continue
        df = pd.read_csv(path)
        if len(df) < 5:
            continue
        out.setdefault(ALGO_LABEL[algo], []).append(df)
    return out


def smooth(y, w=10):
    y = np.asarray(y, float)
    if len(y) < w:
        return y
    k = np.ones(w) / w
    pad = np.concatenate([np.full(w - 1, y[0]), y])
    return np.convolve(pad, k, mode="valid")


def band(ax, runs, col, label, st):
    n = min(len(r) for r in runs)
    x = runs[0]["env_steps"].values[:n] / 1e6
    Y = np.stack([smooth(r[col].values[:n]) for r in runs])
    mu = Y.mean(0)
    ax.plot(x, mu, color=st["color"], ls=st["ls"], label=label + (f" ({len(runs)} seeds)" if len(runs) > 1 else ""))
    if len(runs) > 1:
        ax.fill_between(x, Y.min(0), Y.max(0), color=st["color"], alpha=0.15, lw=0)


def fig_training(runs, heur, out):
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0))
    panels = [("reward", "Episode reward (energy-aware goodput)", None),
              ("pdr", "Packet delivery ratio", ("pdr", f"target {THRESH['pdr']:.2f}")),
              ("outage", "Outage probability", ("outage", f"limit {THRESH['outage']:.2f}")),
              ("aoi", "Mean AoI [slots]", ("aoi", f"limit {THRESH['aoi']:.0f}"))]
    for ax, (col, title, th) in zip(axes.flat, panels):
        for lab in MAIN_ALGOS:
            if lab in runs:
                band(ax, runs[lab], col, lab, style(lab))
        if heur is not None and "Greedy-Track" in heur.index:
            ax.axhline(heur.loc["Greedy-Track", col], color=style("Greedy-Track")["color"], lw=1.0, ls=(0, (1, 2)))
        if th:
            threshold(ax, THRESH[th[0]], th[1])
        ax.set_title(title, loc="left")
        ax.set_xlabel("environment steps [millions]")
    h, l = axes.flat[0].get_legend_handles_labels()
    if heur is not None:
        h.append(Line2D([], [], color=style("Greedy-Track")["color"], lw=1.0, ls=(0, (1, 2))))
        l.append("Greedy-Track (heuristic, eval.)")
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out, "fig_training_curves")


def fig_lambda(runs, out):
    lab = "MAPPO-L (proposed)"
    if lab not in runs:
        return
    rr = runs[lab]
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 5.2), sharex=True)
    cons = [("outage", "outage", "Outage probability", f"limit {THRESH['outage']:.2f}"),
            ("pdr", "pdr", "Packet delivery ratio", f"target {THRESH['pdr']:.2f}"),
            ("aoi", "aoi", "Mean AoI [slots]", f"limit {THRESH['aoi']:.0f}")]
    st = style(lab)
    for j, (lam, met, title, tl) in enumerate(cons):
        band(axes[0, j], rr, f"lam_{lam}", lab, st)
        axes[0, j].set_title(f"Lagrange multiplier  $\\lambda_{{\\mathrm{{{lam}}}}}$", loc="left")
        band(axes[1, j], rr, met, lab, st)
        threshold(axes[1, j], THRESH[met], tl)
        axes[1, j].set_title(title, loc="left")
        axes[1, j].set_xlabel("environment steps [millions]")
    fig.tight_layout()
    save(fig, out, "fig_lagrange_multipliers")


def fig_extended(runs_dir, out):
    """MAPPO-L seeds trained for 300 iterations and then continued to 600 (runs mappo_l_sN + mappo_l_long_sN)."""
    seeds = []
    for base in sorted(glob.glob(os.path.join(runs_dir, "mappo_l_s*", "progress.csv"))):
        sd = base.split("mappo_l_s")[-1].split(os.sep)[0]
        parts = [pd.read_csv(base)]
        for tag in ("long", "long2"):  # continuation runs (resumed from checkpoints)
            lp = os.path.join(runs_dir, f"mappo_l_{tag}_s{sd}", "progress.csv")
            if os.path.exists(lp):
                d = pd.read_csv(lp)
                parts.append(d[d.iter > parts[-1].iter.max()])
        if len(parts) > 1:
            seeds.append(pd.concat(parts, ignore_index=True))
    if not seeds:
        return
    lab, st = "MAPPO-L (proposed)", style("MAPPO-L (proposed)")
    m = _C
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.6), sharex=True)
    panels = [("pdr", "Packet delivery ratio", m.pdr_min, m.pdr_min + 0.1 * (1 - m.pdr_min)),
              ("outage", "Outage probability", m.eps_out, m.eps_out * 0.9),
              ("aoi", "Mean AoI [slots]", m.aoi_max, m.aoi_max * 0.9),
              ("e_uav_kj", "UAV propulsion energy [kJ / UAV]", None, None),
              ("reward", "Episode reward", None, None)]
    for ax, (col, title, nominal, train_t) in zip(axes.flat, panels):
        band(ax, seeds, col, lab, st)
        if nominal is not None:
            threshold(ax, nominal, "evaluation target")
            ax.axhline(train_t, color=INK2, lw=0.8, ls=(0, (1, 2)))
            ax.annotate("training target (10 % margin)", xy=(0.0, train_t), xycoords=("axes fraction", "data"),
                        xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=INK2)
        ax.axvline(0.96, color=GRID, lw=1.2)  # end of the equal-budget phase (300 iterations)
        ax.set_title(title, loc="left")
    ax = axes.flat[5]
    for j, (nm, c) in enumerate([("outage", CAT[1]), ("pdr", CAT[2]), ("aoi", CAT[6])]):
        n = min(len(r) for r in seeds)
        x = seeds[0]["env_steps"].values[:n] / 1e6
        Y = np.stack([r[f"lam_{nm}"].values[:n] for r in seeds])
        ax.plot(x, Y.mean(0), color=c, label=f"λ {nm}")
        ax.fill_between(x, Y.min(0), Y.max(0), color=c, alpha=0.15, lw=0)
    ax.set_title("Lagrange multipliers", loc="left")
    ax.legend(loc="upper left")
    for ax in axes[1]:
        ax.set_xlabel("environment steps [millions]")
    fig.tight_layout()
    save(fig, out, "fig_extended_training")


def fig_ablation(runs, heur, out):
    labs = [l for l in ("MAPPO-L (proposed)", "MAPPO-L, no warm start") if l in runs]
    if len(labs) < 2:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
    for ax, (col, title, th) in zip(axes, [("reward", "Episode reward", None),
                                           ("pdr", "Packet delivery ratio", "pdr"),
                                           ("aoi", "Mean AoI [slots]", "aoi")]):
        for lab in labs:
            band(ax, runs[lab], col, lab, style(lab))
        if heur is not None and "Greedy-Track" in heur.index:
            ax.axhline(heur.loc["Greedy-Track", col], color=style("Greedy-Track")["color"], lw=1.0, ls=(0, (1, 2)))
        if th:
            threshold(ax, THRESH[th], "target" if th == "pdr" else "limit")
        ax.set_title(title, loc="left")
        ax.set_xlabel("environment steps [millions]")
    h, l = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], color=style("Greedy-Track")["color"], lw=1.0, ls=(0, (1, 2))))
    l.append("Greedy-Track (heuristic)")
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, out, "fig_ablation_warm_start")


# ---------------------------------------------------------------- evaluation bars
ORDER = ["MAPPO-L (proposed)", "MAPPO-L, 600 iterations", "MAPPO", "MAPPO-Pen", "IPPO-L", "MAPPO-L, no safety margin",
         "MAPPO-L, no warm start", "Greedy-Track", "Static-KMeans", "Random"]


def fig_comparison(main, out):
    methods = [m for m in ORDER if m in main["method"].unique()]
    seed_means = main.groupby(["method", "seed"])[["goodput_mbps", "pdr", "outage", "aoi", "e_uav_kj", "reward"]].mean()
    panels = [("goodput_mbps", "Goodput [Mbit/s]", None), ("pdr", "Packet delivery ratio", ("pdr", "target")),
              ("outage", "Outage probability", ("outage", "limit")), ("aoi", "Mean AoI [slots]", ("aoi", "limit")),
              ("e_uav_kj", "UAV propulsion energy [kJ / UAV]", None)]
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.4), sharey=True)
    y = np.arange(len(methods))[::-1]
    for ax, (col, title, th) in zip(axes, panels):
        mu = [main[main.method == m][col].mean() for m in methods]
        err = [seed_means.loc[m][col].std() if m in seed_means.index.get_level_values(0) and
               len(seed_means.loc[m]) > 1 else 0.0 for m in methods]
        cols = [style(m)["color"] for m in methods]
        ax.barh(y, mu, color=cols, height=0.62)
        for yi, v, e in zip(y, mu, err):
            if e and e > 0:
                ax.errorbar(v, yi, xerr=e, color=INK2, lw=0.9, capsize=2.5)
            txt = f"{v:.1f}" if col in ("aoi", "goodput_mbps", "e_uav_kj") else f"{v:.3f}"
            ax.annotate(txt, xy=(v + (e or 0), yi), xytext=(4, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=7.5, color=INK)
        if th:
            ax.axvline(THRESH[th[0]], color=INK2, lw=1.0, ls=(0, (4, 3)))
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.25)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(methods)
    fig.tight_layout()
    save(fig, out, "fig_comparison")


def fig_tradeoff(main, out):
    """Reliability versus UAV propulsion energy: one point per method (mean over episodes and seeds)."""
    methods = [m for m in ORDER if m in main["method"].unique() and m != "Random"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    for ax, (col, title, th) in zip(axes, [("pdr", "Packet delivery ratio", "pdr"),
                                           ("outage", "Outage probability", "outage"),
                                           ("aoi", "Mean AoI [slots]", "aoi")]):
        for m in methods:
            d = main[main.method == m]
            sm = d.groupby("seed")[["e_uav_kj", col]].mean()
            st = style(m)
            ax.errorbar(sm["e_uav_kj"].mean(), sm[col].mean(),
                        xerr=sm["e_uav_kj"].std() if len(sm) > 1 else None,
                        yerr=sm[col].std() if len(sm) > 1 else None,
                        fmt=st["marker"], color=st["color"], ms=7, mec=SURFACE, mew=1.0, elinewidth=0.9,
                        capsize=2, label=m)
        threshold(ax, THRESH[th], "target" if th == "pdr" else "limit")
        ax.set_xlabel("UAV propulsion energy [kJ per UAV per episode]")
        ax.set_title(title, loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, out, "fig_energy_reliability_tradeoff")


def fig_sweep(df, xcol_label, name, out):
    methods = [m for m in ORDER if m in df["method"].unique() and m != "Random"]
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.3))
    panels = [("pdr", "Packet delivery ratio", "pdr"), ("outage", "Outage probability", "outage"),
              ("aoi", "Mean AoI [slots]", "aoi"), ("goodput_mbps", "Goodput [Mbit/s]", None)]
    for ax, (col, title, th) in zip(axes, panels):
        for m in methods:
            d = df[df.method == m].groupby("sweep")[col].mean()
            st = style(m)
            ax.plot(d.index, d.values, color=st["color"], ls=st["ls"], marker=st["marker"], ms=5,
                    mec=SURFACE, mew=1.0, label=m)
        if th:
            threshold(ax, THRESH[th], "limit" if th != "pdr" else "target")
        ax.set_title(title, loc="left")
        ax.set_xlabel(xcol_label)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(methods), bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, out, name)


# ---------------------------------------------------------------- trajectory + backhaul timeline
BH_COLORS = ["#c3c2b7", CAT[0], CAT[1], CAT[2], CAT[5]]  # hold, sat1, sat2, sat3, gateway


def fig_trajectory(npz_path, out, name):
    d = np.load(npz_path, allow_pickle=True)
    uav, gu, gw, area = d["uav"], d["gu"], d["gw"], float(d["area"])
    U = uav.shape[1]
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.scatter(gu[:, 0], gu[:, 1], s=14, color=INK2, alpha=0.75, label="ground users", zorder=2)
    ax.add_patch(Circle(gw, float(d["gw_range"]), fill=False, ls=(0, (3, 3)), ec=CAT[5], lw=1.0))
    ax.scatter([gw[0]], [gw[1]], marker="^", s=90, color=CAT[5], ec=SURFACE, zorder=4, label="surviving BS (gateway)")
    ucols = [CAT[0], CAT[1], CAT[2], CAT[6]]
    for u in range(U):
        ax.plot(uav[:, u, 0], uav[:, u, 1], color=ucols[u], lw=1.6, zorder=3, label=f"UAV {u + 1}")
        ax.scatter(uav[0, u, 0], uav[0, u, 1], marker="o", s=30, facecolor=SURFACE, ec=ucols[u], lw=1.5, zorder=5)
        ax.scatter(uav[-1, u, 0], uav[-1, u, 1], marker="o", s=34, color=ucols[u], ec=SURFACE, lw=1.0, zorder=5)
    ax.set_xlim(0, area)
    ax.set_ylim(0, area)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("UAV trajectories (hollow = start, filled = end)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    save(fig, out, name)


def fig_backhaul(npz_path, out, name):
    d = np.load(npz_path, allow_pickle=True)
    elev, bh, ob = np.degrees(d["elev"]), d["bh"], d["out_bh"]
    T, U = bh.shape
    t = np.arange(T)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.0, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[1.1, 1]))
    for s in range(elev.shape[1]):
        a1.plot(t, np.where(elev[:, s] > 0, elev[:, s], np.nan), color=BH_COLORS[s + 1], label=f"LEO-{s + 1}")
    a1.axhline(25.0, color=INK2, lw=1.0, ls=(0, (4, 3)), label="min. elevation (25°)")
    a1.set_ylim(0, 92)
    a1.set_ylabel("elevation [deg]")
    a1.set_title("Satellite passes and learned backhaul selection", loc="left")
    a1.legend(loc="upper left", ncol=4)
    for u in range(U):
        for k in range(5):
            sel = bh[:, u] == k
            a2.scatter(t[sel], np.full(sel.sum(), u), marker="|", s=90, color=BH_COLORS[k], lw=2)
        fail = ob[:, u]
        a2.scatter(t[fail], np.full(fail.sum(), u + 0.32), marker="x", s=14, color=INK, lw=0.9)
    a2.set_yticks(range(U))
    a2.set_yticklabels([f"UAV {u + 1}" for u in range(U)])
    a2.set_ylim(-0.6, U - 0.2)
    a2.set_xlabel("time slot")
    a2.grid(axis="y", visible=False)
    hl = [Line2D([], [], color=BH_COLORS[k], lw=3) for k in range(5)] + \
        [Line2D([], [], color=INK, marker="x", ls="none", ms=5)]
    a2.legend(hl, ["hold", "LEO-1", "LEO-2", "LEO-3", "gateway", "link failure"], loc="upper center",
              bbox_to_anchor=(0.5, -0.32), ncol=6)
    fig.tight_layout()
    save(fig, out, name)


# ---------------------------------------------------------------- system model schematic
def fig_system(out):
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.axhspan(4.6, 6.0, color="#eef3fb", zorder=0)
    ax.axhspan(2.2, 4.6, color="#f6f6f3", zorder=0)
    ax.axhspan(0.0, 2.2, color="#f3f1ea", zorder=0)
    for y, head, sub in [(5.8, "SPACE", "LEO passes, Ka-band\nbackhaul, rain fade,\nON/OFF link failures"),
                         (4.4, "AIR", "UAV base stations\n= MAPPO-L agents"),
                         (2.0, "GROUND", "users in hotspots,\nbursty uplink traffic,\ndamaged BS")]:
        ax.text(0.12, y, head, fontsize=9, fontweight="bold", color=INK, va="top")
        ax.text(0.12, y - 0.3, sub, fontsize=7.8, color=INK2, va="top", linespacing=1.25)
    sats = [(3.4, 5.0), (5.9, 5.15), (8.6, 4.95)]
    for i, (x, y) in enumerate(sats):
        ax.scatter([x], [y], marker="s", s=140, color=BH_COLORS[i + 1], ec=SURFACE, zorder=4)
        ax.text(x + 0.18, y + 0.03, f"LEO-{i + 1}", fontsize=8, va="center")
    uavs = [(3.0, 3.3), (4.9, 2.95), (6.9, 3.45), (9.0, 3.05)]
    for i, (x, y) in enumerate(uavs):
        ax.scatter([x], [y], marker="X", s=140, color=CAT[[0, 1, 2, 6][i]], ec=SURFACE, zorder=4)
        ax.text(x + 0.15, y + 0.2, f"UAV {i + 1}", fontsize=8)
    rng = np.random.default_rng(3)
    centres = [(2.9, 0.85), (5.0, 0.75), (6.8, 0.95), (9.1, 0.85)]
    for cx, cy in centres:
        pts = np.column_stack([cx + 0.3 * rng.standard_normal(7), cy + 0.16 * rng.standard_normal(7)])
        ax.scatter(pts[:, 0], pts[:, 1], s=10, color=INK2, zorder=3)
    bs = (7.95, 0.5)
    ax.scatter([bs[0]], [bs[1]], marker="^", s=150, color=CAT[5], ec=SURFACE, zorder=4)
    ax.text(bs[0], bs[1] - 0.3, "damaged BS", fontsize=7.5, va="center", ha="center")

    def arrow(p, q, col, ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, color=col, lw=1.1, ls=ls, zorder=2))
    for (ux, uy), (cx, cy) in zip(uavs, centres):
        arrow((cx, cy + 0.3), (ux, uy - 0.13), INK2)
    arrow((uavs[0][0], uavs[0][1] + 0.15), (sats[0][0], sats[0][1] - 0.12), BH_COLORS[1])
    arrow((uavs[1][0], uavs[1][1] + 0.15), (sats[1][0], sats[1][1] - 0.12), BH_COLORS[2])
    arrow((uavs[2][0], uavs[2][1] + 0.15), (sats[1][0] + 0.1, sats[1][1] - 0.12), BH_COLORS[2], ls="--")
    arrow((uavs[3][0] - 0.05, uavs[3][1] - 0.15), (bs[0] + 0.08, bs[1] + 0.2), CAT[5])
    ax.text(6.55, 4.3, "shared beam", fontsize=7.2, color=INK2)
    ax.text(9.25, 2.0, "UAV-BS\nbackhaul", fontsize=7.2, color=INK2)
    save(fig, out, "fig_system_model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_system(a.out)
    runs = load_runs(a.runs)
    heur = None
    mp = os.path.join(a.results, "eval_main.csv")
    if os.path.exists(mp):
        main_df = pd.read_csv(mp)
        heur = main_df.groupby("method")[["reward", "pdr", "outage", "aoi", "goodput_mbps"]].mean()
        fig_comparison(main_df, a.out)
        fig_tradeoff(main_df, a.out)
    if runs:
        fig_training(runs, heur, a.out)
        fig_lambda(runs, a.out)
        fig_ablation(runs, heur, a.out)
        fig_extended(a.runs, a.out)
    for fn, lab, nm in [("sweep_severity.csv", "disaster severity σ (link-failure / blockage intensity)", "fig_sweep_severity"),
                        ("sweep_load.csv", "traffic load scale", "fig_sweep_load")]:
        p = os.path.join(a.results, fn)
        if os.path.exists(p):
            fig_sweep(pd.read_csv(p), lab, nm, a.out)
    tp = os.path.join(a.results, "traj_MAPPO-L (proposed).npz")
    if os.path.exists(tp):
        fig_trajectory(tp, a.out, "fig_trajectory_mappo_l")
        fig_backhaul(tp, a.out, "fig_backhaul_timeline")
    tp = os.path.join(a.results, "traj_Greedy-Track.npz")
    if os.path.exists(tp):
        fig_trajectory(tp, a.out, "fig_trajectory_greedy")


if __name__ == "__main__":
    main()
