"""
The experiment driver: model configs, the 3-seed loop, and mean +/- std
aggregation.

SPLIT POLICY, stated precisely because "we did not tune on test" is the claim
most likely to be challenged in a viva:

  * train (60%) - the ONLY data any generator ever sees.
  * val   (20%) - the ONLY data used to choose hyper-parameters
                  (synth.tune), and the ONLY data used to decide which model
                  family is "best".
  * test  (20%) - read exactly once per model per seed, inside the final
                  evaluation stage.  It is never used for fitting, never used
                  for model selection, and never used to pick a threshold.

Note the honest nuance: "touched once" means one *evaluation pass* per model,
not one number overall - reporting mean +/- std over 3 seeds necessarily reads
the test split 3 times per model.  What matters, and what we guarantee, is that
no decision taken during tuning or model selection was informed by it.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data import DatasetSplit, GLOBAL_SEED, load_dataset
from .metrics import EvalResult, evaluate_model, spearman_gap, pca_overlay_data
from .models import build_model
from .tune import CONFIG_PATH, load_tuned

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

SEEDS = [0, 1, 2]
EPSILONS = [1.0, 3.0, 10.0, 30.0]

# DP-SGD spends privacy per *optimiser step*, so the number of steps is itself a
# privacy decision.  We fix the step budget at roughly the same value for every
# dataset (≈3400 noisy steps at batch 32) rather than fixing the epoch count,
# because 400 epochs means 2260 steps on Cleveland and 5750 on Pima - the same
# "epsilon = 10" would then mean very different things in the two rows.
# For Cleveland this is exactly the budget selected on the validation split
# (epochs 600 / batch 32 / clip 1.0 beat 150, 400 and 1000 on validation TSTR).
DP_STEP_BUDGET = 3400
DP_BATCH = 32
DP_CLIP_NORM = 1.0


def dp_epochs(n_train: int) -> int:
    """Epoch count that hits the fixed DP step budget for this dataset size."""
    return int(round(DP_STEP_BUDGET * DP_BATCH / max(n_train, 1)))

# Metrics that get a mean +/- std column in the summary table.
HEADLINE = [
    "fid_ks_mean", "fid_tv_mean", "fid_spearman_gap", "fid_spearman_gap_vs_floor",
    "tstr_lr_auc", "tstr_lr_f1", "tstr_lr_accuracy",
    "tstr_rf_auc", "tstr_rf_f1", "tstr_rf_accuracy",
    "retention_lr", "retention_rf",
    "priv_exact_copy_train", "priv_dcr_ratio", "priv_mia_auc",
    "priv_mia_auc_likelihood",
    "fid_valid_rate", "syn_pos_rate",
]


def model_configs(dataset: str, quick: bool = False,
                  n_train: Optional[int] = None) -> List[Tuple[str, str, Dict]]:
    """The list of (row_label, model_name, hyperparameters) we will run.

    Returns labels, not just names, because the DP-VAE appears four times - once
    per epsilon - and each needs its own row in the results table.

    Two extra rows exist purely to keep the epsilon curve honest:
      * `vae` - the headline model at the full epoch budget.
      * `vae_dp_budget` - the SAME non-private VAE trained at the DP step budget.
        Without it the right-hand end of the privacy/utility curve would compare
        a 400-epoch model against 600-epoch DP models, and any reader could
        reasonably ask whether they were looking at a privacy effect or an
        optimisation-budget effect.  With it, the epsilon sweep varies exactly
        one thing: the noise multiplier.
    """
    tuned = load_tuned("vae", dataset, fallback={
        "w_num": 20.0, "beta_max": 0.3, "latent_dim": 8, "obs_noise": True})
    if n_train is None:
        n_train = len(load_dataset(dataset).train)
    ep = 80 if quick else 400
    dpe = (40 if quick else dp_epochs(n_train))
    ctp = 30 if quick else 300
    wgp = 300 if quick else 2000
    cfgs: List[Tuple[str, str, Dict]] = [
        ("naive", "naive", {}),
        ("vae", "vae", dict(tuned, epochs=ep)),
        ("vae_dp_budget", "vae", dict(tuned, epochs=dpe)),
    ]
    for eps in EPSILONS:
        cfgs.append((f"dp_vae_eps{eps:g}", "dp_vae",
                     dict(tuned, epochs=dpe, batch_size=DP_BATCH, epsilon=eps,
                          max_grad_norm=DP_CLIP_NORM)))
    cfgs += [
        ("ctgan", "ctgan", dict(epochs=ctp, batch_size=100, embedding_dim=32,
                                dim=(128, 128))),
        ("tvae", "tvae", dict(epochs=ctp, batch_size=100, embedding_dim=32,
                              dim=(128, 128))),
        ("wgan_gp", "wgan_gp", dict(epochs=wgp, batch_size=32, z_dim=16,
                                    hidden=(128, 64))),
    ]
    return cfgs


def _n_params(m) -> int:
    """Parameter count, for the model-comparison table.  Works for the torch
    models and for the ctgan package objects; returns 0 if neither exposes one."""
    net = getattr(m, "net", None) or getattr(m, "model", None)
    params = getattr(net, "parameters", None)
    if params is None:
        return 0
    try:
        return int(sum(p.numel() for p in params() if p.requires_grad))
    except Exception:
        return 0


def run_one(label: str, model_name: str, hp: Dict, ds: DatasetSplit,
            seeds: Sequence[int], n_syn: int, save_dir: Optional[str] = None,
            verbose: bool = True) -> List[Dict]:
    """Fit + sample + evaluate one model across the seed list."""
    rows: List[Dict] = []
    for s in seeds:
        t0 = time.time()
        m = build_model(model_name, ds.schema, seed=s, **hp)
        m.fit(ds.train, val_df=ds.val)
        syn = m.sample(n_syn, seed=s)
        res: EvalResult = evaluate_model(label, syn, ds.train, ds.val, ds.test,
                                         ds.schema, seed=s,
                                         extra_notes=list(m.notes), model=m)
        row = dict(res.metrics)
        row.update({"dataset": ds.name, "model": label, "model_kind": model_name,
                    "seed": s, "wall_seconds": time.time() - t0,
                    "privacy": getattr(m, "privacy_note", "non-private"),
                    "n_params": _n_params(m),
                    "n_train": len(ds.train), "n_val": len(ds.val),
                    "n_test": len(ds.test)})
        if hasattr(m, "val_loss"):
            row["val_loss"] = m.val_loss()
        if hasattr(m, "noise_multiplier"):
            row["dp_sigma"] = m.noise_multiplier
            row["dp_achieved_epsilon"] = m.achieved_epsilon
            row["dp_steps"] = m.total_steps
            row["dp_delta"] = m.spent_delta
        if save_dir:
            _save_artifacts(save_dir, ds, label, s, syn, res)
        rows.append(row)
        if verbose:
            print(f"  {label:>16s} seed={s} "
                  f"gap={row.get('fid_spearman_gap', float('nan')):.4f} "
                  f"ks={row.get('fid_ks_mean', float('nan')):.4f} "
                  f"tstrAUC(lr)={row.get('tstr_lr_auc', float('nan')):.4f} "
                  f"copy={row.get('priv_exact_copy_train', float('nan')):.4f} "
                  f"miaAUC={row.get('priv_mia_auc', float('nan')):.4f} "
                  f"({time.time() - t0:.1f}s)")
    return rows


def _save_artifacts(save_dir: str, ds: DatasetSplit, label: str, seed: int,
                   syn: pd.DataFrame, res: EvalResult) -> None:
    """Persist everything make_figures.py needs, one file per (model, seed).

    Figures are generated from these artefacts rather than by retraining, so
    `python make_figures.py` is fast and always shows exactly the numbers in
    results_summary.csv.
    """
    d = os.path.join(save_dir, ds.name)
    os.makedirs(d, exist_ok=True)
    syn.to_csv(os.path.join(d, f"syn_{label}_seed{seed}.csv"), index=False)
    pca = pca_overlay_data(ds.train, syn, ds.schema, ds.test)
    if pca:
        np.savez(os.path.join(d, f"pca_{label}_seed{seed}.npz"), **pca)
    sg = spearman_gap(syn, ds.test, ds.schema)
    sg["syn_corr"].to_csv(os.path.join(d, f"correlation_{label}_seed{seed}.csv"))
    sg["real_corr"].to_csv(os.path.join(d, f"correlation_real_seed{seed}.csv"))


def aggregate(rows: List[Dict], metrics: Sequence[str] = HEADLINE) -> pd.DataFrame:
    """Collapse the seed dimension into mean +/- std, NaN-aware.

    A NaN metric (for example TSTR AUC when a seed produced single-class
    synthetic data) is dropped from that model's mean, and `n_seeds_ok` records
    how many seeds actually contributed, so a 1-seed average can never be
    silently presented as a 3-seed one.
    """
    df = pd.DataFrame(rows)
    out_rows = []
    for label, g in df.groupby("model", sort=False):
        rec = {"dataset": g["dataset"].iloc[0], "model": label,
               "n_seeds": len(g), "n_seeds_ok": 0}
        applicable = []
        for mt in metrics:
            if mt not in g.columns:
                rec[f"{mt}_mean"] = float("nan")
                rec[f"{mt}_std"] = float("nan")
                continue
            v = pd.to_numeric(g[mt], errors="coerce").to_numpy(dtype=float)
            good = v[np.isfinite(v)]
            rec[f"{mt}_mean"] = float(good.mean()) if len(good) else float("nan")
            # std with ddof=1 needs >= 2 points; with one seed the spread is
            # genuinely unknown, so report NaN rather than a fake 0.
            rec[f"{mt}_std"] = (float(good.std(ddof=1)) if len(good) > 1
                                else float("nan"))
            # A metric that is NaN for EVERY seed is not "missing data", it is
            # not applicable to this model - dp_sigma on a non-DP VAE, or the
            # likelihood-gap attack on a generator with no likelihood. Counting
            # those as failures made n_seeds_ok read 0 for almost every model.
            if len(good):
                applicable.append(len(good))
        rec["n_seeds_ok"] = int(np.min(applicable)) if applicable else 0
        rec["wall_seconds_mean"] = float(pd.to_numeric(
            g["wall_seconds"], errors="coerce").mean())
        rec["privacy"] = g["privacy"].iloc[0] if "privacy" in g else ""
        out_rows.append(rec)
    return pd.DataFrame(out_rows)


def fmt(v: float, sd: float, digits: int = 4) -> str:
    """Render 'mean +/- std', and 'n/a (single seed)' style fallbacks."""
    if v is None or not np.isfinite(v):
        return "n/a"
    if sd is None or not np.isfinite(sd):
        return f"{v:.{digits}f}"
    return f"{v:.{digits}f}+/-{sd:.{digits}f}"


def results_markdown(agg: pd.DataFrame, ds: DatasetSplit,
                     metrics: Sequence[str] = HEADLINE) -> str:
    """A one-page, paste-straight-into-the-report results table."""
    cols = ["model"] + [m for m in metrics]
    lines = [f"# Results - {ds.name} dataset",
             "",
             f"n = {len(pd.concat([ds.train, ds.val, ds.test]))} patients "
             f"(train {len(ds.train)} / val {len(ds.val)} / "
             f"test {len(ds.test)}), "
             f"prevalence {ds.train[ds.schema.label].mean():.3f}.  "
             f"Values are mean +/- std over {agg['n_seeds'].max()} seeds.  "
             f"Fidelity and utility are measured against the held-out test "
             f"split.",
             "",
             "| Model | KS | TV | Spearman gap (x noise floor) | TSTR AUC (LR) "
             "| TSTR AUC (RF) | Retention LR | Retention RF | Exact copy "
             "| DCR ratio | MIA AUC |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    key = ["fid_ks_mean", "fid_tv_mean", "fid_spearman_gap_vs_floor",
           "tstr_lr_auc", "tstr_rf_auc", "retention_lr", "retention_rf",
           "priv_exact_copy_train", "priv_dcr_ratio", "priv_mia_auc"]
    for _, r in agg.iterrows():
        cells = [str(r["model"])]
        for mt in key:
            cells.append(fmt(r.get(f"{mt}_mean", float("nan")),
                             r.get(f"{mt}_std", float("nan")), 3))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["",
              f"TRTR (real/real) ceiling AUC: LR "
              f"{fmt(_first(agg,'trtr_lr_auc_mean'), _first(agg,'trtr_lr_auc_std'))}"
              f", RF "
              f"{fmt(_first(agg,'trtr_rf_auc_mean'), _first(agg,'trtr_rf_auc_std'))}.",
              ""]
    return "\n".join(lines)


def _first(agg: pd.DataFrame, col: str) -> float:
    """TRTR is a property of the data, not the generator, so every row shares
    the same value; take the first finite one."""
    if col not in agg.columns:
        return float("nan")
    v = pd.to_numeric(agg[col], errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(v[0]) if len(v) else float("nan")


def run_dataset(dataset: str, seeds: Sequence[int] = SEEDS, n_syn: int = 1000,
                quick: bool = False, save_dir: str = RESULTS,
                verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame,
                                              List[Dict]]:
    """Full experiment for one dataset.  Returns (raw_rows_df, aggregate_df)."""
    ds = load_dataset(dataset)
    os.makedirs(os.path.join(save_dir, dataset), exist_ok=True)
    if verbose:
        print(f"[{dataset}] train={len(ds.train)} val={len(ds.val)} "
              f"test={len(ds.test)}  "
              f"missing={ds.missing_counts}")
        print(f"[{dataset}] tuned VAE config (selected on validation): "
              f"{load_tuned('vae', dataset)}")
    all_rows: List[Dict] = []
    for label, name, hp in model_configs(dataset, quick=quick):
        all_rows += run_one(label, name, hp, ds, seeds, n_syn,
                            save_dir=save_dir, verbose=verbose)
    raw = pd.DataFrame(all_rows)
    metrics = HEADLINE + ["trtr_lr_auc", "trtr_rf_auc", "trtr_lr_f1",
                          "trtr_rf_f1", "trtr_lr_accuracy", "trtr_rf_accuracy",
                          "val_loss", "fid_floor_ks_mean", "fid_floor_tv_mean",
                          "fid_floor_spearman_gap", "fid_train_spearman_gap"]
    agg = aggregate(raw, metrics)
    return raw, agg, all_rows
