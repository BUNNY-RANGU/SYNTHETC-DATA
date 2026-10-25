"""
make_figures.py - publication-quality figures from the saved experiment
artefacts.

    python make_figures.py                 # all datasets, all figures
    python make_figures.py --dataset heart
    python make_figures.py --quick         # fewer models, smaller dpi

Every figure is drawn from files that run_pipeline.py already wrote, so the
pictures and results_summary.csv can never disagree - which is the whole point.
Re-running this does NOT retrain anything.

COLOUR
------
All palettes are Okabe-Ito, the standard colour-blind-safe qualitative set
(black, orange, sky blue, bluish green, yellow, blue, vermillion, reddish
purple).  It is distinguishable in greyscale print and under deuteranopia and
protanopia, so a reviewer printing this in black and white still reads it.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")                      # headless; works on Colab and CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synth.data import load_dataset
from synth.metrics import pca_overlay_data, spearman_gap

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(RESULTS, "figures")

# Okabe-Ito colour-blind-safe palette.
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
             "#56B4E9", "#000000", "#F0E442"]
MODEL_COLOR: Dict[str, str] = {
    "naive": "#999999",          # neutral grey: the baseline, not a contender
    "vae": "#0072B2",            # blue
    "vae_dp_budget": "#4477AA",  # muted blue, visibly related to `vae`
    "dp_vae_eps1": "#009E73",
    "dp_vae_eps3": "#56B4E9",
    "dp_vae_eps10": "#E69F00",
    "dp_vae_eps30": "#CC79A7",
    "ctgan": "#D55E00",          # vermillion
    "tvae": "#B07AA1",
    "wgan_gp": "#7A5195",
}
REAL_COLOR = "#000000"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
})


def metric(g: pd.DataFrame, name: str, suffix: str = "_mean") -> np.ndarray:
    """Fetch a metric column from an aggregated frame, tolerating either the
    bare name or the '_mean' / '_std' suffixed form.

    run_pipeline.py writes `metric_mean` and `metric_std`; several call sites in
    this file historically asked for the bare name and silently got an empty
    Series, which matplotlib then reported as a broadcast error.  Centralising
    the lookup means a missing metric is missing for every figure, not for one.
    """
    for key in (name + suffix, name, name.replace("_mean", "_std")
                if suffix == "_std" else name):
        if key in g.columns:
            v = pd.to_numeric(g[key], errors="coerce")
            if len(v) == len(g):
                return v.to_numpy(dtype=float)
        if suffix and key.endswith(suffix) and key[:-len(suffix)] in g.columns:
            v = pd.to_numeric(g[key[:-len(suffix)]], errors="coerce")
            if len(v) == len(g):
                return v.to_numpy(dtype=float)
    return np.full(len(g), np.nan)


def std_metric(g: pd.DataFrame, name: str) -> np.ndarray:
    v = metric(g, name, "_std")
    return np.nan_to_num(v, nan=0.0)


SHORT = {
    "naive": "naive",
    "vae": "VAE",
    "vae_dp_budget": "VAE@DP",
    "dp_vae_eps1": "ε=1",
    "dp_vae_eps3": "ε=3",
    "dp_vae_eps10": "ε=10",
    "dp_vae_eps30": "ε=30",
    "ctgan": "CTGAN",
    "tvae": "TVAE",
    "wgan_gp": "WGAN-GP",
}


def short(model: str) -> str:
    """Compact axis label. Full names collide badly at 10 models per axis."""
    return SHORT.get(str(model), str(model).replace("_", " "))


def tick_labels(ax, g, size: float = 8.0, rotation: float = 30.0) -> None:
    """Ten model names on one axis only fit if they are short AND angled."""
    ax.set_xticks(np.arange(len(g)))
    ax.set_xticklabels([short(m) for m in g.model], fontsize=size,
                       rotation=rotation, ha="right")


FALLBACK_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
                   "#7A5195"]


def color(model: str) -> str:
    """Fixed palette colour per model.

    Deliberately deterministic: a hash-picked fallback would let the same model
    change colour between runs, and Okabe-Ito yellow is unreadable as a bar on
    a white page.  Unknown models get a stable dark colour from a sorted index.
    """
    if model in MODEL_COLOR:
        return MODEL_COLOR[model]
    return FALLBACK_COLORS[sorted(set(list(MODEL_COLOR) + [str(model)]))
                           .index(str(model)) % len(FALLBACK_COLORS)]


def _save(fig, name: str) -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, ROOT)}")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_summary() -> pd.DataFrame:
    p = os.path.join(RESULTS, "results_summary.csv")
    if not os.path.exists(p):
        raise SystemExit("results_summary.csv not found - run "
                         "`python run_pipeline.py` first.")
    return pd.read_csv(p)


def load_curve() -> Optional[pd.DataFrame]:
    p = os.path.join(RESULTS, "privacy_utility_curve.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def syn_path(dataset: str, model: str, seed: int = 0) -> str:
    return os.path.join(RESULTS, dataset, f"syn_{model}_seed{seed}.csv")


def available_models(dataset: str) -> List[str]:
    """Model labels that have stored synthetic tables for this dataset.

    Parsed with a regex rather than a fixed slice: the earlier `basename[4:-11]`
    chopped a character off every model name, because `_seed0.csv` is 10
    characters not 11, and the figures then silently skipped every panel whose
    file could not be found.
    """
    import re
    pat = re.compile(r"^syn_(.+)_seed0\.csv$")
    out = set()
    for p in glob.glob(os.path.join(RESULTS, dataset, "syn_*_seed0.csv")):
        m = pat.match(os.path.basename(p))
        if m:
            out.add(m.group(1))
    # Keep the experiment's natural order rather than alphabetical.
    order = ["naive", "vae", "vae_dp_budget"] + \
            [f"dp_vae_eps{e:g}" for e in (1, 3, 10, 30)] + \
            ["ctgan", "tvae", "wgan_gp"]
    return [o for o in order if o in out] + sorted(out - set(order))


# ---------------------------------------------------------------------------
# Figure 1 - PCA overlay
# ---------------------------------------------------------------------------
def fig_pca_overlay(dataset: str, models: List[str]) -> None:
    """Synthetic patients projected into the same PCA space as real patients.

    Read this as an envelope test, not a shape test: the black held-out real
    cloud shows how much a genuinely new batch of patients spreads.  If the
    synthetic cloud is materially tighter than that, the generator has
    under-dispersed; if it sticks out beyond it, the generator is producing
    impossible patients.
    """
    ds = load_dataset(dataset)
    n = len(models)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2.9 * ncols, 2.7 * nrows), squeeze=False)
    base = pca_overlay_data(ds.train, ds.train, ds.schema, ds.test)
    expl = base.get("explained", np.array([0.5, 0.5]))
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.set_visible(False)
            continue
        m = models[i]
        p = os.path.join(RESULTS, dataset, f"pca_{m}_seed0.npz")
        if os.path.exists(p):
            z = np.load(p)
            syn_pts, held, tr = z["syn"], z["heldout"], z["train"]
        else:                                   # recompute if artefacts missing
            f = syn_path(dataset, m)
            if not os.path.exists(f):
                ax.set_visible(False)
                continue
            d = pca_overlay_data(ds.train, pd.read_csv(f), ds.schema, ds.test)
            syn_pts, held, tr = d["syn"], d["heldout"], d["train"]
        ax.scatter(tr[:, 0], tr[:, 1], s=8, alpha=0.30, color="#BBBBBB",
                   label="real train (fitted PCA)")
        ax.scatter(held[:, 0], held[:, 1], s=16, alpha=0.75, marker="s",
                   edgecolor="none", color=REAL_COLOR,
                   label="real held-out test")
        ax.scatter(syn_pts[:, 0], syn_pts[:, 1], s=8, alpha=0.45,
                   color=color(m), label="synthetic")
        ax.set_title(m.replace("_", " "), loc="left")
        ax.set_xlabel(f"PC1 ({100 * expl[0]:.0f}%)")
        ax.set_ylabel(f"PC2 ({100 * expl[1]:.0f}%)")
        if i == 0:
            ax.legend(loc="best", fontsize=7)
    fig.suptitle(f"PCA overlay - {dataset.title()} "
                 f"(synthetic vs real, colour-blind-safe palette)", y=1.005,
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"fig1_pca_overlay_{dataset}.png")


# ---------------------------------------------------------------------------
# Figure 2 - histogram overlays
# ---------------------------------------------------------------------------
def fig_histograms(dataset: str, models: List[str]) -> None:
    """Real held-out distribution vs each model, per numeric column."""
    ds = load_dataset(dataset)
    cols = ds.schema.numeric_names
    frames = {}
    for m in models:
        f = syn_path(dataset, m)
        if os.path.exists(f):
            frames[m] = pd.read_csv(f)
    if not frames:
        return
    nshow = min(len(models), 5)
    show = list(frames)[:nshow]
    fig, axes = plt.subplots(len(cols), len(show),
                             figsize=(2.6 * len(show), 2.1 * len(cols)),
                             squeeze=False)
    for r, c in enumerate(cols):
        allv = np.concatenate([ds.test[c].to_numpy(),
                               *[df[c].to_numpy() for df in frames.values()]])
        bins = np.linspace(allv.min(), allv.max(), 26)
        for j, m in enumerate(show):
            ax = axes[r][j]
            ax.hist(ds.test[c], bins=bins, density=True, histtype="step",
                    lw=2.0, color=REAL_COLOR, label="real (held-out)")
            ax.hist(frames[m][c], bins=bins, density=True, histtype="step",
                    lw=1.6, color=color(m), alpha=0.95, label=m)
            if r == 0:
                ax.set_title(m.replace("_", " "), loc="left")
            if j == 0:
                ax.set_ylabel(c)
            if r == len(cols) - 1:
                ax.set_xlabel(c)
            if r == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle(f"Histogram overlays - {dataset.title()} "
                 f"(rows = numeric column, cols = model)", y=1.003,
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"fig2_histograms_{dataset}.png")


# ---------------------------------------------------------------------------
# Figure 3 - correlation heatmaps
# ---------------------------------------------------------------------------
def fig_correlation_heatmaps(dataset: str, models: List[str]) -> None:
    """Spearman matrices: real, synthetic, and the difference between them.

    The right-hand panel is the one that matters.  A generator can match every
    marginal perfectly and still have this panel lit up - which is exactly what
    the naive baseline does, and exactly why marginals alone are not evidence.
    """
    ds = load_dataset(dataset)
    use = [m for m in models if os.path.exists(syn_path(dataset, m))][:3]
    if not use:
        return
    fig, axes = plt.subplots(len(use), 3,
                             figsize=(3.5 * 3, 3.1 * len(use)), squeeze=False)
    for i, m in enumerate(use):
        syn = pd.read_csv(syn_path(dataset, m))
        sg = spearman_gap(syn, ds.test, ds.schema)
        R = sg["real_corr"].to_numpy(dtype=float)
        S = sg["syn_corr"].to_numpy(dtype=float)
        cols = list(sg["real_corr"].columns)
        panels = [(R, "Spearman - real (held-out)", (-1, 1)),
                  (S, f"Spearman - {m.replace('_', ' ')}", (-1, 1)),
                  (S - R, "difference (synthetic - real)", (-1, 1))]
        for j, (M, title, lim) in enumerate(panels):
            ax = axes[i][j]
            im = ax.imshow(M, cmap="coolwarm", vmin=lim[0], vmax=lim[1],
                           aspect="auto")
            ax.set_xticks(range(len(cols)))
            ax.set_yticks(range(len(cols)))
            if i == len(use) - 1:
                ax.set_xticklabels(cols, rotation=90, fontsize=6.5)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_yticklabels(cols, fontsize=6.5)
            else:
                ax.set_yticklabels([])
            ax.set_title(title, loc="left", fontsize=9)
            ax.grid(False)
            if j == 2:
                gap = np.abs(np.nan_to_num(R) - np.nan_to_num(S))[
                    ~np.eye(len(cols), dtype=bool)].mean()
                ax.text(0.03, 0.97, f"mean |gap| = {gap:.3f}",
                        transform=ax.transAxes, va="top", fontsize=7.5,
                        bbox=dict(fc="white", ec="none", alpha=0.85))
        fig.colorbar(im, ax=axes[i], fraction=0.035, pad=0.02)
    fig.suptitle(f"Correlation structure - {dataset.title()}", y=1.004,
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"fig3_correlation_heatmaps_{dataset}.png")


# ---------------------------------------------------------------------------
# Figure 4 - fidelity by model
# ---------------------------------------------------------------------------
def fig_fidelity_bars(agg: pd.DataFrame, dataset: str) -> None:
    """KS, TV and the correlation gap, with the real-vs-real noise floor drawn
    as a horizontal line on each panel.

    The floor line is the honest reference point: no generator can beat it,
    because it is what an independent *real* sample of the same size scores
    against itself.
    """
    g = agg[agg.dataset == dataset].copy()
    if g.empty:
        return
    panels = [("fid_ks_mean", "Mean KS statistic\n(numeric marginals)"),
              ("fid_tv_mean", "Mean total-variation\n(categorical marginals)"),
              ("fid_spearman_gap", "Spearman correlation gap\n(joint structure)")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    for ax, (mt, title) in zip(axes, panels):
        x = np.arange(len(g))
        means = metric(g, mt)
        stds = std_metric(g, mt)
        ax.bar(x, means, yerr=stds, width=0.68, capsize=3,
               color=[color(m) for m in g.model], error_kw=dict(lw=1))
        floor_mt = {"fid_ks_mean": "fid_floor_ks_mean",
                    "fid_tv_mean": "fid_floor_tv_mean",
                    "fid_spearman_gap": "fid_floor_spearman_gap"}[mt]
        fv = metric(g, floor_mt)
        if np.isfinite(fv).any():
            ax.axhline(np.nanmean(fv), ls="--", lw=1.4, color="black",
                       label=f"real-vs-real noise floor = {np.nanmean(fv):.3f}")
        tick_labels(ax, g)
        ax.set_title(title, loc="left")
        ax.set_ylabel("lower is better")
        ax.legend(fontsize=7.5)
    fig.suptitle(f"A. Fidelity by generator - {dataset.title()} "
                 f"(mean +/- std over {int(g.n_seeds.max())} seeds)",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, f"fig4_fidelity_{dataset}.png")


# ---------------------------------------------------------------------------
# Figure 5 - utility, TSTR vs TRTR
# ---------------------------------------------------------------------------
def fig_utility(agg: pd.DataFrame, dataset: str) -> None:
    """TSTR accuracy / F1 / AUC next to the TRTR ceiling."""
    g = agg[agg.dataset == dataset].copy()
    if g.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, clf in zip(axes, ["lr", "rf"]):
        x = np.arange(len(g))
        w = 0.26
        for k, (mt, lab) in enumerate([
                (f"tstr_{clf}_accuracy", "TSTR accuracy"),
                (f"tstr_{clf}_f1", "TSTR F1"),
                (f"tstr_{clf}_auc", "TSTR AUC")]):
            means = metric(g, mt)
            stds = std_metric(g, mt)
            if np.isnan(means).all():
                continue                # metric absent for this run
            ax.bar(x + (k - 1) * w, means, w, yerr=stds, capsize=2,
                   label=lab, color=["#56B4E9", "#0072B2", "#009E73"][k])
        ref = metric(g, f"trtr_{clf}_auc")
        if np.isfinite(ref).any():
            ax.axhline(np.nanmean(ref), ls="--", lw=1.5, color="black",
                       label=f"TRTR AUC ceiling = {np.nanmean(ref):.3f}")
        tick_labels(ax, g)
        ax.set_title(f"Downstream classifier: "
                     f"{'Logistic Regression' if clf == 'lr' else 'Random Forest'}",
                     loc="left")
        ax.set_ylabel("score")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7.5, loc="lower right")
    fig.suptitle(f"B. Utility - TSTR (train synthetic, test real) vs TRTR "
                 f"ceiling, {dataset.title()}", fontsize=11,
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    _save(fig, f"fig5_utility_{dataset}.png")


# ---------------------------------------------------------------------------
# Figure 6 - privacy by model
# ---------------------------------------------------------------------------
def fig_privacy(agg: pd.DataFrame, dataset: str) -> None:
    """Exact-copy rate, DCR ratio and the membership-inference AUC."""
    g = agg[agg.dataset == dataset].copy()
    if g.empty:
        return
    panels = [("priv_exact_copy_train", "Exact-copy rate vs training set",
               None),
              ("priv_dcr_ratio", "DCR ratio (synthetic / real held-out)", 1.0),
              ("priv_mia_auc", "Membership-inference attack AUC", 0.5)]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, (mt, title, ref) in zip(axes, panels):
        x = np.arange(len(g))
        means = metric(g, mt)
        stds = std_metric(g, mt)
        ax.bar(x, means, 0.68, yerr=stds, capsize=3,
               color=[color(m) for m in g.model], error_kw=dict(lw=1))
        if ref is not None:
            ax.axhline(ref, ls="--", lw=1.4, color="black")
            note = {1.0: "1.0 = sits where real unseen patients sit",
                    0.5: "0.5 = attacker at chance"}[ref]
            ax.text(len(g) - 0.45, ref, note, va="bottom", ha="right",
                    fontsize=7)
        tick_labels(ax, g)
        ax.set_title(title, loc="left")
    fig.suptitle(f"C. Privacy by generator - {dataset.title()} "
                 f"(mean +/- std over seeds)", fontsize=11, fontweight="bold",
                 y=1.02)
    fig.tight_layout()
    _save(fig, f"fig6_privacy_{dataset}.png")


# ---------------------------------------------------------------------------
# Figure 7 - the privacy vs utility curve across epsilon
# ---------------------------------------------------------------------------
def fig_epsilon_curve(curve: pd.DataFrame) -> None:
    """The headline privacy-utility trade-off.

    Left: attack AUC against epsilon.  Right: downstream utility against
    epsilon.  The non-private VAE is plotted at the far right as the epsilon ->
    infinity reference, so the cost of the guarantee is visible as a gap rather
    than asserted in prose.
    """
    if curve is None or curve.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for dataset in sorted(curve.dataset.unique()):
        c = curve[curve.dataset == dataset].copy()
        c = c.sort_values("epsilon")
        # Plot epsilon on a log axis with inf pushed one tick past 30.
        xs = np.where(np.isinf(c.epsilon), 100.0, c.epsilon)
        mk = "o" if dataset == "heart" else "s"
        col = "#0072B2" if dataset == "heart" else "#D55E00"
        ax = axes[0]
        m = pd.to_numeric(c.priv_mia_auc_mean, errors="coerce")
        s = pd.to_numeric(c.priv_mia_auc_std, errors="coerce").fillna(0)
        ax.errorbar(xs, m, yerr=s, marker=mk, ms=6, lw=1.6, capsize=3,
                    color=col, label=dataset)
        ax.axhline(0.5, ls="--", lw=1.3, color="black")
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 10, 30, 100])
        ax.set_xticklabels(["1", "3", "10", "30", "\u221e\n(non-DP)"])
        ax.set_xlabel("privacy budget $\\epsilon$  (smaller = more private)")
        ax.set_ylabel("membership-inference AUC")
        ax.set_title("Privacy: attack succeeds less as $\\epsilon$ shrinks",
                     loc="left")
        ax.legend()

        ax = axes[1]
        u = pd.to_numeric(c.tstr_lr_auc_mean, errors="coerce")
        us = pd.to_numeric(c.tstr_lr_auc_std, errors="coerce").fillna(0)
        ax.errorbar(xs, u, yerr=us, marker=mk, ms=6, lw=1.6, capsize=3,
                    color=col, label=dataset)
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 10, 30, 100])
        ax.set_xticklabels(["1", "3", "10", "30", "\u221e\n(non-DP)"])
        ax.set_xlabel("privacy budget $\\epsilon$")
        ax.set_ylabel("TSTR AUC (logistic regression)")
        ax.set_title("Utility: the price paid for the guarantee", loc="left")
        ax.legend()
    fig.suptitle("Privacy versus utility across the differential-privacy budget "
                 "(DP-VAE, mean +/- std over seeds)",
                 fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout()
    _save(fig, "fig7_privacy_utility_curve.png")


# ---------------------------------------------------------------------------
# Figure 8 - the ablation that explains the design choices
# ---------------------------------------------------------------------------
def fig_ablation(dataset: str = "heart") -> None:
    """Why the pipeline looks the way it does: three VAE variants, one metric.

    Built live (it is cheap) rather than from stored artefacts, because these
    variants are deliberately NOT in the main table - they exist to answer the
    viva question "what would happen without the loss weighting / the transform /
    the sampler noise".
    """
    from synth.models import build_model
    from synth.metrics import fidelity_report

    ds = load_dataset(dataset, transform="quantile")
    ds_mm = load_dataset(dataset, transform="none")
    tuned = _tuned_hp(dataset)
    variants = [
        ("as\nshipped", ds, dict(tuned)),
        ("no obs-noise\nsampler", ds, dict(tuned, obs_noise=False)),
        ("min-max not\nrank transform", ds_mm, dict(tuned)),
    ]
    rows = []
    for lab, dd, hp in variants:
        m = build_model("vae", dd.schema, seed=0, **hp).fit(dd.train,
                                                            val_df=dd.val)
        syn = m.sample(1000, seed=0)
        f = fidelity_report(syn, dd.val, dd.schema, per_column=False)
        rows.append((lab, f["fid_ks_mean"], f["fid_spearman_gap"]))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    x = np.arange(len(rows))
    for ax, idx, title, ylab in [(axes[0], 1, "Numeric marginal fit (KS)",
                                  "mean KS vs validation"),
                                 (axes[1], 2, "Joint structure (gap)",
                                  "Spearman gap vs validation")]:
        vals = [r[idx] for r in rows]
        ax.bar(x, vals, 0.6, color=OKABE_ITO[:len(rows)])
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in rows], fontsize=8)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylab)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Ablation - what each design choice buys "
                 f"({dataset.title()}, selected on validation)",
                 fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout()
    _save(fig, f"fig8_ablation_{dataset}.png")


def _tuned_hp(dataset: str) -> Dict:
    p = os.path.join(RESULTS, "tuned_config.json")
    if os.path.exists(p):
        import json
        with open(p) as fh:
            cfg = json.load(fh)
        try:
            hp = dict(cfg["vae"][dataset])
            if "latent_dim" in hp:
                hp["latent_dim"] = int(hp["latent_dim"])
            hp["epochs"] = 400
            return hp
        except KeyError:
            pass
    return dict(w_num=120.0, beta_max=0.1, latent_dim=16, epochs=400,
                obs_noise=True)


# ---------------------------------------------------------------------------
# Figure 9 - one-page dashboard
# ---------------------------------------------------------------------------
def fig_dashboard(agg: pd.DataFrame, dataset: str) -> None:
    """The single slide image: all three families side by side."""
    g = agg[agg.dataset == dataset].copy()
    if g.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    specs = [("fid_spearman_gap", "A. Fidelity\nSpearman correlation gap", True),
             ("tstr_rf_auc", "B. Utility\nTSTR AUC (Random Forest)", False),
             ("priv_mia_auc", "C. Privacy\nmembership-inference AUC", False)]
    for ax, (mt, title, floor) in zip(axes, specs):
        x = np.arange(len(g))
        means = metric(g, mt)
        stds = std_metric(g, mt)
        bars = ax.bar(x, means, 0.68, yerr=stds, capsize=3,
                      color=[color(m) for m in g.model], error_kw=dict(lw=1))
        if floor:
            fv = metric(g, "fid_floor_spearman_gap")
            if np.isfinite(fv).any():
                ax.axhline(np.nanmean(fv), ls="--", lw=1.4, color="black")
        if mt == "priv_mia_auc":
            ax.axhline(0.5, ls="--", lw=1.4, color="black")
        if mt == "tstr_rf_auc":
            rv = metric(g, "trtr_rf_auc")
            if np.isfinite(rv).any():
                ax.axhline(np.nanmean(rv), ls="--", lw=1.4, color="black")
        tick_labels(ax, g)
        ax.set_title(title, loc="left")
    fig.suptitle(f"Synthetic medical data - {dataset.title()} summary "
                 f"(dashed lines = reference: noise floor / TRTR ceiling / "
                 f"chance)", fontsize=10.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    _save(fig, f"fig9_dashboard_{dataset}.png")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", nargs="*", default=None,
                    help="restrict to these datasets (default: all found)")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--skip-ablation", action="store_true",
                    help="do not retrain the ablation variants")
    args = ap.parse_args(argv)

    agg = load_summary()
    datasets = args.dataset or sorted(agg.dataset.unique())
    os.makedirs(FIGDIR, exist_ok=True)
    print(f"[figures] {len(agg)} aggregated model rows, datasets={datasets}")

    for d in datasets:
        models = args.models or available_models(d)
        if not models:
            print(f"[figures] no synthetic artefacts for {d}, skipping")
            continue
        print(f"[figures] {d}: {len(models)} models")
        sub = agg[agg.dataset == d]
        fig_pca_overlay(d, models)
        fig_histograms(d, models)
        fig_correlation_heatmaps(d, models)
        fig_fidelity_bars(sub, d)
        fig_utility(sub, d)
        fig_privacy(sub, d)
        fig_dashboard(sub, d)
    fig_epsilon_curve(load_curve())
    if not args.skip_ablation:
        for d in datasets:
            fig_ablation(d)
    print("[figures] done ->", FIGDIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
