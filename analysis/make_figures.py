"""
Generate the figures backing analysis/findings.md.

Charts (matplotlib, colorblind-safe validated palette, clean light style):
  1. indicator_effects.png   - univariate signal (|AUC-0.5|) per college metric
  2. hitrate_by_quartile.png - hit rate by quartile for the headline metrics
  3. ppa_distribution.png     - career PPA: hits vs non-hits
  4. model_auc.png            - forward AUC: PFF/no-PFF vs draft-capital baselines
  5. calibration.png          - post-draft model calibration
  6. projections_board.png    - top projected QBs, 2024/2025/2026

Run: python3 -m analysis.make_figures
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model.data import add_derived, join_labels_profiles
from model.train import (FIXED_C, PICK_ONLY_FEATS, POST_DRAFT_FEATS,
                         PRE_DRAFT_FEATS, PRE_DRAFT_NO_PFF_FEATS, forward_oof)
from model import predict as pred

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGDIR, exist_ok=True)

# validated palette (light surface)
BLUE, AQUA, YELLOW, GREEN = "#2a78d6", "#1baf7a", "#eda100", "#008300"
VIOLET, RED, MAGENTA, ORANGE = "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"
GOOD, CRIT = "#0ca30c", "#d03b3b"
INK, INK2, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e6e5e1"
GROUP_COLOR = {"ppa": BLUE, "combine": AQUA, "pff": VIOLET, "context": YELLOW, "capital": ORANGE}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 11, "font.family": "DejaVu Sans",
    "axes.edgecolor": GRID, "axes.linewidth": 1.0,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _load():
    joined, _ = join_labels_profiles()
    joined = add_derived(joined)
    ind = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "indicators.csv"))
    return joined, ind


def fig_indicator_effects(ind):
    d = ind.sort_values("abs_auc", ascending=True).tail(15).copy()
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    y = np.arange(len(d))
    colors = [GROUP_COLOR.get(g, MUTED) for g in d["group"]]
    ax.barh(y, d["abs_auc"] - 0.5, left=0.5, color=colors, height=0.68,
            edgecolor=SURF, linewidth=1.2, zorder=3)
    ax.axvline(0.5, color=INK2, lw=1.4, zorder=2)
    for yi, (_, r) in zip(y, d.iterrows()):
        arrow = "up" if "higher" in r["direction"] else "dn"
        ax.text(r["abs_auc"] + 0.006, yi, f"{r['abs_auc']:.2f}", va="center", ha="left",
                fontsize=9.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m}  (n={n})" for m, n in zip(d["metric"], d["n"])], fontsize=9.5)
    ax.set_xlim(0.5, 0.92)
    ax.set_xlabel("Univariate AUC  (0.50 = no signal;  distance from 0.50 = separating power)")
    # group legend (only groups actually shown)
    present = [g for g in ["ppa", "combine", "pff", "context", "capital"] if g in set(d["group"])]
    handles = [plt.Line2D([0], [0], marker="s", ls="", ms=10, color=GROUP_COLOR[g],
                          label={"ppa": "CFBD PPA", "combine": "Combine", "pff": "PFF",
                                 "context": "Context", "capital": "Draft capital"}[g])
               for g in present]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9, ncol=1)
    ax.grid(axis="y", visible=False)
    fig.suptitle("Which college metrics separate NFL hits from non-hits?", fontweight="bold",
                 fontsize=14, y=0.99)
    fig.text(0.5, 0.945, "Final-label draft classes 2015-2022 (n=88, 17 hits). Bars past the line = more hits at higher values.",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(FIGDIR, "indicator_effects.png"), dpi=150)
    plt.close(fig)


def fig_hitrate_quartile(joined):
    fin = joined[joined["label_status"] == "final"].copy()
    metrics = [("career_average_ppa_all", "Career PPA per play"),
               ("career_total_ppa_all", "Career total production (PPA)"),
               ("combine_broad", "Broad jump (explosiveness)"),
               ("neg_log_pick", "Draft capital (-log pick)")]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.2), sharey=True)
    ramp = ["#b7d3f6", "#6da7ec", "#2a78d6", "#184f95"]
    for ax, (col, name) in zip(axes, metrics):
        sub = fin[[col, "hit"]].dropna()
        q = pd.qcut(sub[col], 4, labels=["Q1\nlow", "Q2", "Q3", "Q4\nhigh"], duplicates="drop")
        rate = sub.groupby(q, observed=True)["hit"].agg(["mean", "size"])
        x = np.arange(len(rate))
        ax.bar(x, rate["mean"], color=ramp[:len(rate)], width=0.72, edgecolor=SURF,
               linewidth=1.2, zorder=3)
        for xi, (m, s) in zip(x, rate[["mean", "size"]].values):
            ax.text(xi, m + 0.015, f"{m*100:.0f}%", ha="center", va="bottom", fontsize=9.5, color=INK)
        ax.axhline(fin["hit"].mean(), color=RED, lw=1.4, ls=(0, (4, 3)), zorder=2)
        ax.set_xticks(x); ax.set_xticklabels(rate.index, fontsize=9)
        ax.set_title(name, fontsize=10.5, pad=6)
        ax.set_ylim(0, 0.85); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Hit rate")
    axes[-1].text(3.4, fin["hit"].mean() + 0.02, "overall\n20%", color=RED, fontsize=8.5, ha="right")
    fig.suptitle("Hit rate rises with college production and draft capital (by quartile)",
                 fontweight="bold", x=0.5, y=1.0, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(FIGDIR, "hitrate_by_quartile.png"), dpi=150)
    plt.close(fig)


def fig_ppa_distribution(joined):
    fin = joined[joined["label_status"] == "final"].copy()
    col = "career_average_ppa_all"
    hits = fin.loc[fin["hit"] == 1, col].dropna()
    non = fin.loc[fin["hit"] == 0, col].dropna()
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    bins = np.linspace(-0.1, 0.9, 22)
    ax.hist(non, bins=bins, color=MUTED, alpha=0.55, label=f"Non-hits (n={len(non)})", zorder=2)
    ax.hist(hits, bins=bins, color=BLUE, alpha=0.85, label=f"Hits (n={len(hits)})", zorder=3)
    ax.axvline(non.median(), color=INK2, lw=1.6, ls=(0, (4, 3)), zorder=4)
    ax.axvline(hits.median(), color=BLUE, lw=1.8, zorder=4)
    ax.text(hits.median() + 0.01, ax.get_ylim()[1]*0.92, f"hits median {hits.median():.2f}",
            color=BLUE, fontsize=9)
    ax.text(non.median() - 0.01, ax.get_ylim()[1]*0.78, f"non-hits {non.median():.2f}",
            color=INK2, fontsize=9, ha="right")
    ax.set_xlabel("Career PPA per play (CFBD expected points added)")
    ax.set_ylabel("Number of QBs")
    ax.set_title("The single most useful all-classes college stat: career PPA per play",
                 fontweight="bold", pad=10)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "ppa_distribution.png"), dpi=150)
    plt.close(fig)


def fig_model_auc(joined):
    train = joined[joined["label_status"].isin(["final", "provisional"])].copy()
    train["y"] = train["hit"].astype(int)
    train["w"] = np.where(train["label_status"] == "final", 1.0, 0.5)
    train = train.reset_index(drop=True)
    specs = [("Round-1 = hit\n(rule of thumb)", None, MUTED),
             ("Draft slot only\n(the market's bet)", PICK_ONLY_FEATS, ORANGE),
             ("College + combine\n(no PFF)", PRE_DRAFT_NO_PFF_FEATS, BLUE),
             ("College + combine + PFF\n(pre-draft)", PRE_DRAFT_FEATS, VIOLET),
             ("College + combine + slot\n(post-draft)", POST_DRAFT_FEATS, GREEN)]
    aucs = []
    from sklearn.metrics import roc_auc_score
    y_pick, _, yrs = forward_oof(train, PICK_ONLY_FEATS, FIXED_C["pick_only"])
    fin = train[(train["label_status"] == "final") & (train["draft_year"].isin(yrs))]
    aucs.append(roc_auc_score(fin["y"], (fin["round"] == 1).astype(int)))
    fixed_names = ["pick_only", "pre_draft_no_pff", "pre_draft", "post_draft"]
    for (_, feats, _c), name in zip(specs[1:], fixed_names):
        y, p, _ = forward_oof(train, feats, FIXED_C[name])
        aucs.append(roc_auc_score(y, p))
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(specs))
    colors = [s[2] for s in specs]
    ax.bar(x, aucs, color=colors, width=0.62, edgecolor=SURF, linewidth=1.4, zorder=3)
    for xi, a in zip(x, aucs):
        ax.text(xi, a + 0.008, f"{a:.3f}", ha="center", va="bottom", fontsize=11, color=INK, fontweight="bold")
    ax.axhline(0.5, color=INK2, lw=1.2)
    ax.text(-0.42, 0.505, "coin flip", color=INK2, fontsize=8.5, ha="left")
    ax.set_xticks(x); ax.set_xticklabels([s[0] for s in specs], fontsize=9.5)
    ax.set_ylim(0.5, 0.92); ax.set_ylabel("Forward-by-draft-year AUC")
    ax.set_title("Full PFF helps pre-draft a little; draft capital still dominates",
                 fontweight="bold", pad=10)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "model_auc.png"), dpi=150)
    plt.close(fig)


def fig_calibration(joined):
    train = joined[joined["label_status"].isin(["final", "provisional"])].copy()
    train["y"] = train["hit"].astype(int)
    train["w"] = np.where(train["label_status"] == "final", 1.0, 0.5)
    train = train.reset_index(drop=True)
    y, p, _ = forward_oof(train, POST_DRAFT_FEATS, FIXED_C["post_draft"])
    bins = [0, 0.1, 0.2, 0.35, 0.6, 1.01]
    df = pd.DataFrame({"y": y, "p": p})
    df["b"] = pd.cut(df["p"], bins, right=False)
    g = df.groupby("b", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")).reset_index()
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    ax.plot([0, 0.85], [0, 0.85], color=MUTED, lw=1.5, ls=(0, (4, 3)), zorder=2, label="perfect calibration")
    ax.plot(g["pred"], g["obs"], "-o", color=GREEN, lw=2, ms=9, zorder=4, label="post-draft model")
    for _, r in g.iterrows():
        ax.text(r["pred"], r["obs"] + 0.03, f"n={int(r['n'])}", ha="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, 0.85); ax.set_ylim(0, 0.85)
    ax.set_xlabel("Predicted hit probability"); ax.set_ylabel("Observed hit rate")
    ax.set_title("Post-draft model is reasonably calibrated", fontweight="bold", pad=10)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "calibration.png"), dpi=150)
    plt.close(fig)


def fig_projections_board():
    p = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "projections.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2))
    class_color = {2024: BLUE, 2025: AQUA, 2026: VIOLET}
    for ax, yr in zip(axes, [2024, 2025, 2026]):
        s = p[p["draft_season"] == yr].sort_values("draft_adjusted_hit_prob").tail(8)
        y = np.arange(len(s))
        ax.barh(y, s["draft_adjusted_hit_prob"], color=class_color[yr], height=0.68,
                edgecolor=SURF, linewidth=1.2, zorder=3)
        for yi, v in zip(y, s["draft_adjusted_hit_prob"]):
            ax.text(v + 0.008, yi, f"{v*100:.0f}%", va="center", fontsize=9, color=INK2)
        ax.set_yticks(y); ax.set_yticklabels(s["canonical_name"], fontsize=9.5)
        ax.set_xlim(0, max(0.8, s["draft_adjusted_hit_prob"].max() + 0.12))
        ax.set_title(f"{yr} class", fontweight="bold", fontsize=12)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Draft-adjusted hit probability")
    fig.suptitle("Projection board: top QBs by draft-adjusted hit probability",
                 fontweight="bold", x=0.5, y=0.99, fontsize=13.5)
    fig.text(0.5, 0.93, "Drafted QBs use max(post-draft model, pick-only market baseline); PFF pre-draft deltas remain in model/projections.csv.",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(FIGDIR, "projections_board.png"), dpi=150)
    plt.close(fig)


def main():
    joined, ind = _load()
    fig_indicator_effects(ind)
    fig_hitrate_quartile(joined)
    fig_ppa_distribution(joined)
    fig_model_auc(joined)
    fig_calibration(joined)
    fig_projections_board()
    print("wrote figures to", FIGDIR)
    print("\n".join(sorted(os.listdir(FIGDIR))))


if __name__ == "__main__":
    main()
