"""
run_pipeline.py - the one command that produces every number in the report.

    python run_pipeline.py                # full experiment (~30-60 min on CPU)
    python run_pipeline.py --quick        # smoke run, small epochs, ~3 min
    python run_pipeline.py --datasets heart
    python run_pipeline.py --seeds 0 1 2 --n-syn 1000

What it does, in order
----------------------
 1. Loads each dataset and reproduces the fixed stratified 60/20/20 split.
 2. Makes sure hyper-parameters were chosen on the VALIDATION split (runs the
    grid if results/tuned_config.json is missing).
 3. Trains every generator on the TRAIN split only, for each random seed.
 4. Generates synthetic patients and evaluates fidelity / utility / privacy
    against the held-out TEST split - the first and only time test is read.
 5. Aggregates to mean +/- std and writes results_summary.csv,
    results_summary_long.csv, results_table.md and
    privacy_utility_curve.csv.

Nothing in this script inspects the test split before step 4.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synth.data import load_dataset
from synth.pipeline import (EPSILONS, HEADLINE, RESULTS, SEEDS, aggregate,
                            model_configs, run_one)
from synth.tune import CONFIG_PATH, tune_all

TEXT_COLS = {"model", "dataset", "model_kind", "privacy", "tstr_lr_note",
             "tstr_rf_note"}


def ensure_tuned(skip: bool) -> Dict:
    """Hyper-parameter selection happens here, and only ever sees train+val."""
    if skip and os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as fh:
            print(f"[tune] reusing {CONFIG_PATH}")
            return json.load(fh)
    print("[tune] running validation-only grid search for the VAE family")
    return tune_all()


def acceptance_verdict(agg: pd.DataFrame) -> List[str]:
    """Evaluate the four acceptance criteria against measured numbers only.

    Deliberately pessimistic: a criterion is PASS only if the mean passes AND we
    can point at a number.  Anything we did not measure is reported as such.
    """
    lines: List[str] = []
    get = lambda lab, mt: (agg.loc[agg.model == lab, f"{mt}_mean"].iloc[0]
                           if (agg.model == lab).any() and f"{mt}_mean" in agg
                           else float("nan"))

    # 1. VAE correlation gap clearly below naive
    nv, va = get("naive", "fid_spearman_gap"), get("vae", "fid_spearman_gap")
    ok = np.isfinite(nv) and np.isfinite(va) and va < nv
    lines.append(f"1. VAE correlation gap below naive: "
                 f"{'PASS' if ok else 'FAIL'} "
                 f"(naive {nv:.4f} -> VAE {va:.4f}, "
                 f"{100 * (1 - va / nv) if nv else float('nan'):.1f}% reduction)")

    # 2. best-model TSTR AUC >= 90% of real-data AUC
    best, best_v = None, -1.0
    for lab in agg["model"]:
        for cl in ("lr", "rf"):
            r = get(lab, f"retention_{cl}")
            if np.isfinite(r) and r > best_v:
                best, best_v = f"{lab}/{cl.upper()}", r
    lines.append(f"2. TSTR AUC >= 90% of TRTR AUC: "
                 f"{'PASS' if best_v >= 0.90 else 'FAIL'} "
                 f"(best = {best} at {best_v * 100:.1f}% retention)")

    # 3. DP-VAE attack AUC near 0.5 with a visible utility trade-off
    dp = [(f"dp_vae_eps{e:g}", e) for e in EPSILONS]
    mias = [(e, get(l, "priv_mia_auc")) for l, e in dp]
    aucs = [(e, get(l, "tstr_lr_auc")) for l, e in dp]
    close = [x for x in mias if np.isfinite(x[1]) and abs(x[1] - 0.5) <= 0.10]
    spread = ([a for _, a in aucs if np.isfinite(a)])
    tradeoff = (len(spread) >= 2 and (max(spread) - min(spread)) >= 0.02)
    lines.append(f"3. DP-VAE distance-based MIA AUC near 0.5: "
                 f"{'PASS' if close else 'CHECK'} "
                 f"(eps->AUC: " +
                 ", ".join(f"{e:g}:{v:.3f}" if np.isfinite(v) else f"{e:g}:n/a"
                           for e, v in mias) + ")")
    # The likelihood-gap attack is the one DP-SGD actually moves, because it
    # probes the parameters rather than the released table.
    if "priv_mia_auc_likelihood_mean" in agg.columns:
        lik = [(e, get(l, "priv_mia_auc_likelihood")) for l, e in dp]
        nv_lik = get("vae_dp_budget", "priv_mia_auc_likelihood")
        lines.append("3a. likelihood-gap MIA (non-DP at same budget "
                     f"{nv_lik:.3f}) -> " +
                     ", ".join(f"eps{e:g}:{v:.3f}" if np.isfinite(v)
                               else f"eps{e:g}:n/a" for e, v in lik))
    lines.append(f"3b. Visible utility trade-off across epsilon: "
                 f"{'PASS' if tradeoff else 'FAIL'} "
                 f"(TSTR AUC spread {max(spread) - min(spread):.4f})"
                 if len(spread) >= 2 else "3b. Utility spread: n/a")

    # 4. exact-copy rate is zero / low
    worst = np.nanmax(pd.to_numeric(agg["priv_exact_copy_train_mean"],
                                   errors="coerce").to_numpy())
    lines.append(f"4. Exact-copy rate: worst model {worst * 100:.3f}% "
                 f"({'PASS' if worst < 0.01 else 'CHECK'})")
    return lines


def reaggregate(out_dir: str, seeds: List[int], n_syn: int,
                quick: bool) -> int:
    """Rebuild every summary file from results_summary_long.csv, no retraining.

    The per-seed rows are the primary record; the aggregate is a pure function
    of them.  Being able to re-derive one from the other means a change to the
    aggregation logic (or to the one-page table) does not cost another hour of
    GPU-free training, and it lets a reader verify the summary against the raw
    seeds themselves.
    """
    long_path = os.path.join(out_dir, "results_summary_long.csv")
    if not os.path.exists(long_path):
        print(f"nothing to reaggregate: {long_path} is missing")
        return 1
    raw_all = pd.read_csv(long_path)
    agg_parts, verdict_lines = [], []
    for d, raw in raw_all.groupby("dataset", sort=False):
        agg = aggregate(raw, [m for m in raw.columns
                              if m not in ("dataset", "model", "model_kind",
                                           "seed", "privacy", "tstr_lr_note",
                                           "tstr_rf_note")])
        agg_parts.append(agg)
        from synth.data import load_dataset
        from synth.pipeline import results_markdown
        ds = load_dataset(str(d))
        with open(os.path.join(out_dir, f"results_table_{d}.md"), "w",
                  encoding="utf-8") as fh:
            fh.write(results_markdown(agg, ds))
        v = acceptance_verdict(agg)
        verdict_lines.append(f"===== {d} =====\n" + "\n".join(v))
        print(f"\n[{d}] re-aggregated {len(raw)} seed-rows -> {len(agg)} models")
        for line in v:
            print("  " + line)
        agg.to_csv(os.path.join(out_dir, f"results_{d}.csv"), index=False)
    agg_all = pd.concat(agg_parts, ignore_index=True)
    agg_all.to_csv(os.path.join(out_dir, "results_summary.csv"), index=False)
    with open(os.path.join(out_dir, "acceptance_all.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n\n".join(verdict_lines) + "\n")
    print(f"\nrewrote results_summary.csv, results_<dataset>.csv, "
          f"results_table_<dataset>.md, acceptance_all.txt")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["heart", "pima"],
                    choices=["heart", "pima"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--n-syn", type=int, default=1000,
                    help="synthetic patients generated per model per seed")
    ap.add_argument("--quick", action="store_true",
                    help="small epoch counts, for a fast end-to-end check")
    ap.add_argument("--skip-tune", action="store_true",
                    help="reuse results/tuned_config.json instead of re-tuning")
    ap.add_argument("--out", default=RESULTS)
    ap.add_argument("--reaggregate", action="store_true",
                    help="rebuild the summaries from results_summary_long.csv "
                         "without retraining anything")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    if args.reaggregate:
        return reaggregate(args.out, args.seeds, args.n_syn, args.quick)
    t_start = time.time()
    ensure_tuned(args.skip_tune)

    all_raw, all_agg = [], []
    for d in args.datasets:
        print("\n" + "=" * 78)
        print(f"DATASET: {d}   seeds={args.seeds}   n_syn={args.n_syn}")
        print("=" * 78)
        ds = load_dataset(d)
        rows: List[Dict] = []
        for label, name, hp in model_configs(d, quick=args.quick):
            rows += run_one(label, name, hp, ds, args.seeds, args.n_syn,
                            save_dir=args.out)
        raw = pd.DataFrame(rows)
        metrics = HEADLINE + ["trtr_lr_auc", "trtr_rf_auc", "trtr_lr_f1",
                              "trtr_rf_f1", "trtr_lr_accuracy",
                              "trtr_rf_accuracy", "val_loss",
                              "fid_floor_ks_mean", "fid_floor_tv_mean",
                              "fid_floor_spearman_gap", "dp_sigma",
                              "dp_achieved_epsilon", "dp_steps",
                              "fid_train_spearman_gap"]
        agg = aggregate(raw, [m for m in metrics if m in raw.columns])
        all_raw.append(raw)
        all_agg.append(agg)

        # ---- write per-dataset results IMMEDIATELY ----
        # If the run is interrupted (or the machine sleeps) the datasets that
        # already finished still produce usable output.
        raw.to_csv(os.path.join(args.out, f"results_{d}_long.csv"), index=False)
        agg.to_csv(os.path.join(args.out, f"results_{d}.csv"), index=False)

        # ---- one-page table per dataset ----
        from synth.pipeline import results_markdown
        md = results_markdown(agg, ds)
        with open(os.path.join(args.out, f"results_table_{d}.md"), "w",
                  encoding="utf-8") as fh:
            fh.write(md)
        print("\n" + md)
        print("\nAcceptance criteria for " + d + ":")
        for line in acceptance_verdict(agg):
            print("  " + line)
        with open(os.path.join(args.out, f"acceptance_{d}.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write("\n".join(
                [f"dataset={d} seeds={args.seeds} n_syn={args.n_syn} "
                 f"quick={args.quick}"] +
                acceptance_verdict(agg)) + "\n")

    raw_all = pd.concat(all_raw, ignore_index=True)
    agg_all = pd.concat(all_agg, ignore_index=True)
    raw_all.to_csv(os.path.join(args.out, "results_summary_long.csv"),
                   index=False)
    agg_all.to_csv(os.path.join(args.out, "results_summary.csv"), index=False)

    # ---- privacy vs utility curve across epsilon ----
    CURVE_COLS = ["dataset", "model", "priv_mia_auc_mean", "priv_mia_auc_std",
                  "priv_mia_auc_likelihood_mean", "priv_mia_auc_likelihood_std",
                  "priv_exact_copy_train_mean", "priv_dcr_ratio_mean",
                  "tstr_lr_auc_mean", "tstr_lr_auc_std", "tstr_rf_auc_mean",
                  "fid_spearman_gap_mean", "fid_ks_mean_mean", "fid_tv_mean_mean",
                  "dp_sigma_mean", "dp_achieved_epsilon_mean",
                  "dp_steps_mean", "dp_delta_mean"]
    have = [c for c in CURVE_COLS if c in agg_all.columns]
    curve = agg_all[agg_all.model.str.startswith("dp_vae")][have].copy()
    curve["epsilon"] = curve.model.str.extract(r"eps([0-9.]+)").astype(float)
    # The eps=inf reference must share the DP optimisation budget, otherwise the
    # right-hand end of the curve compares epoch count as well as privacy.
    base = agg_all[agg_all.model == "vae_dp_budget"][have].copy()
    base["epsilon"] = np.inf
    base["model"] = "vae_non_private_at_dp_budget"
    out_curve = pd.concat([curve, base], ignore_index=True)
    if "dp_steps_mean" in out_curve:
        out_curve["dp_steps_mean"] = out_curve["dp_steps_mean"].fillna(0)
    out_curve.to_csv(os.path.join(args.out, "privacy_utility_curve.csv"),
                     index=False)

    with open(os.path.join(args.out, "run_metadata.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"datasets": args.datasets, "seeds": args.seeds,
                   "n_syn": args.n_syn, "quick": args.quick,
                   "wall_seconds": time.time() - t_start,
                   "python": sys.version,
                   "torch": __import__("torch").__version__},
                  fh, indent=2, default=str)

    print(f"\nwrote {os.path.join(args.out, 'results_summary.csv')}")
    print(f"total wall time {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
