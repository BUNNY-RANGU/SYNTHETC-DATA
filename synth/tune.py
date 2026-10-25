"""
Hyper-parameter selection on the VALIDATION split only.

WHAT WAS TUNED ON WHAT - this is the sentence the report has to be able to say
out loud, so it is enforced by code rather than by discipline:

  * `tune()` sees `train` and `val` and nothing else.  It never receives the
    test frame; there is no code path from the test split into the search.
  * the selection score is a fidelity composite computed between the synthetic
    sample and the VALIDATION rows, so no test information can enter it.
  * `run_pipeline.py` loads the winning configuration from
    `results/tuned_config.json` and only then touches the test split, once.
"""

from __future__ import annotations

import itertools
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .data import DatasetSplit, load_dataset
from .metrics import fidelity_report, utility_report
from .models import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
CONFIG_PATH = os.path.join(RESULTS, "tuned_config.json")


def val_score(syn: pd.DataFrame, val: pd.DataFrame, schema) -> Dict[str, float]:
    """Composite validation score, all terms 'lower is better'.

    We deliberately do NOT select on the validation reconstruction loss, because
    that loss contains the very w_num / beta we are searching over - comparing
    candidate models by a score whose definition changed between them is
    meaningless.  Instead we score candidates on what the project is actually
    about: distributional fidelity to held-out-but-not-test rows.
    """
    f = fidelity_report(syn, val, schema, per_column=False)
    out = {
        "gap": f["fid_spearman_gap"],
        "ks": f["fid_ks_mean"],
        "tv": f["fid_tv_mean"],
    }
    terms = [v for v in out.values() if np.isfinite(v)]
    out["score"] = float(np.mean(terms)) if terms else float("nan")
    return out


GRID = {
    # Search space.  Kept deliberately small: with 181 training rows a bigger
    # grid would overfit the validation split, which is the failure mode we are
    # trying not to have.
    "vae": dict(
        w_num=[20.0, 120.0],
        beta_max=[0.03, 0.1, 0.3],
        latent_dim=[8, 16, 32],
        obs_noise=[True],
    ),
    # The DP variant inherits the winning non-DP settings; only epsilon is
    # swept separately, because epsilon is a privacy requirement, not a knob
    # you are allowed to tune for better utility.
    "dp_vae": dict(),
}


def tune(dataset: str = "heart", epochs: int = 400, n_sample: int = 1000,
         seed: int = 0, verbose: bool = True, save: bool = True) -> Dict:
    """Grid-search one model family on train -> val and return the best hp."""
    ds = load_dataset(dataset)
    grid = GRID["vae"]
    keys = list(grid)
    rows: List[Dict] = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        hp = dict(zip(keys, combo))
        hp.update(epochs=epochs)
        m = build_model("vae", ds.schema, seed=seed, **hp)
        m.fit(ds.train, val_df=ds.val)
        syn = m.sample(n_sample, seed=seed)
        sc = val_score(syn, ds.val, ds.schema)
        rec = dict(hp)
        rec.update(sc)
        rows.append(rec)
        if verbose:
            print(f"  w_num={hp['w_num']:5g} beta={hp['beta_max']:5g} "
                  f"lat={hp['latent_dim']:3d} -> gap={sc['gap']:.4f} "
                  f"ks={sc['ks']:.4f} tv={sc['tv']:.4f} score={sc['score']:.4f}")
    df = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    if save:
        os.makedirs(RESULTS, exist_ok=True)
        df.to_csv(os.path.join(RESULTS, f"tuning_grid_{dataset}.csv"),
                  index=False)
    best = df.iloc[0].to_dict()
    best_hp = {k: best[k] for k in keys}
    # int-valued latent dim comes back as float from the DataFrame
    if "latent_dim" in best_hp:
        best_hp["latent_dim"] = int(best_hp["latent_dim"])
    return {"best_hp": best_hp, "table": df, "dataset": dataset}


def tune_all(datasets=("heart", "pima"), save: bool = True) -> Dict:
    """Tune every model family that has a grid, for every dataset."""
    out: Dict[str, Dict] = {}
    for d in datasets:
        print(f"[tune] {d}: VAE grid on validation only")
        res = tune(d)
        print(f"[tune] {d} best -> {res['best_hp']}")
        out[d] = res["best_hp"]
    cfg = {"vae": out, "dp_vae": {d: dict(v, epsilon_note="epsilon swept in "
                                                 "pipeline")
                                  for d, v in out.items()}}
    if save:
        os.makedirs(RESULTS, exist_ok=True)
        with open(CONFIG_PATH, "w") as fh:
            json.dump(cfg, fh, indent=2, sort_keys=True)
    return cfg


def load_tuned(model_name: str, dataset: str,
               fallback: Optional[Dict] = None) -> Dict:
    """Read the winning hyper-parameters for (model, dataset) if we have them."""
    if not os.path.exists(CONFIG_PATH):
        return dict(fallback or {})
    with open(CONFIG_PATH) as fh:
        cfg = json.load(fh)
    try:
        return dict(cfg[model_name][dataset])
    except KeyError:
        return dict(fallback or {})


if __name__ == "__main__":
    tune_all()
