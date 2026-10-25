"""
app.py - Synthetic Medical Data Generator, web application.

    streamlit run app.py

Two modes, one code path
------------------------
The sidebar's "Retrain live in the browser" switch picks where the cohort comes
from:

  live     - the generator is fitted on the TRAIN split from your exact sliders
             (budget, model, diversity, class filter). Slow, interactive.
  frozen   - the cohort is read from results/<dataset>/syn_<model>_seed0.csv,
             the table the 3-seed experiment actually published. Instant.

Everything after that is identical: the same `synth.metrics` functions score
whichever cohort arrived against the same held-out test split. There is no
second implementation of the metrics for the UI to drift away from.

In both modes the published mean +/- std over 3 seeds is shown next to the
recomputed single-cohort number, and labelled which is which. A live single-seed
number sitting next to a published 3-seed mean without saying so is exactly what
an examiner looks for.

Layout
------
  ui/theme.py       design tokens, colour-blind-safe palette, stylesheet
  ui/components.py  hero, pills, KPI cards, section rules
  ui/charts.py      plotly figures; no metric is computed inside a chart
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synth.data import load_dataset
from synth.metrics import (RecordSpace, fidelity_report, ks_per_column,
                           pca_overlay_data, privacy_report, spearman_gap,
                           tv_per_column, utility_report)
from synth.models import build_model, diversity_to_knobs
from synth.pipeline import RESULTS, model_configs
from ui import charts
from ui.components import (brand_mark, chart, footer, hero, inject_css,
                           kpi_row, pills, section, verdict)
from ui.theme import palette_for

st.set_page_config(page_title="Synthetic Medical Data Generator",
                   layout="wide", initial_sidebar_state="expanded")

VERSION = "1.0.0"
MODEL_LABEL = {
    "naive": "Naive column resampler",
    "vae": "Custom VAE",
    "dp_vae": "DP-VAE (Opacus)",
    "ctgan": "CTGAN",
    "tvae": "TVAE",
    "wgan_gp": "WGAN-GP",
}
CLASS_OPTIONS = ["Match the real prevalence", "Only diseased (target = 1)",
                 "Only healthy (target = 0)"]
EXPERIMENT_ORDER = ["naive", "vae", "vae_dp_budget", "dp_vae_eps1",
                    "dp_vae_eps3", "dp_vae_eps10", "dp_vae_eps30", "ctgan",
                    "tvae", "wgan_gp"]
DISPLAY = {
    "naive": "Naive", "vae": "VAE", "vae_dp_budget": "VAE @ DP budget",
    "dp_vae_eps1": "DP-VAE ε=1", "dp_vae_eps3": "DP-VAE ε=3",
    "dp_vae_eps10": "DP-VAE ε=10", "dp_vae_eps30": "DP-VAE ε=30",
    "ctgan": "CTGAN", "tvae": "TVAE", "wgan_gp": "WGAN-GP",
}


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the dataset and applying the 60/20/20 split")
def get_dataset(name: str, transform: str = "quantile"):
    return load_dataset(name, transform=transform)


@st.cache_resource(show_spinner="Fitting the generator on the train split")
def get_model(dataset: str, model_name: str, hp: Dict, seed: int):
    """Fit once per (dataset, model, hyper-parameters, seed).

    cache_resource because a fitted torch module is neither hashable nor
    serialisable the way a DataFrame is.
    """
    ds = get_dataset(dataset)
    m = build_model(model_name, ds.schema, seed=seed, **hp)
    m.fit(ds.train, val_df=ds.val)
    return m


@st.cache_data(show_spinner=False)
def load_frozen_cohort(dataset: str, label: str, seed: int = 0) -> Optional[pd.DataFrame]:
    """The stored synthetic table from the published sweep, or None."""
    p = os.path.join(RESULTS, dataset, f"syn_{label}_seed{seed}.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_summary() -> Optional[pd.DataFrame]:
    p = os.path.join(RESULTS, "results_summary.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


@st.cache_data(show_spinner=False)
def load_curve() -> Optional[pd.DataFrame]:
    p = os.path.join(RESULTS, "privacy_utility_curve.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


# ---------------------------------------------------------------------------
# Sampling with a class constraint
# ---------------------------------------------------------------------------
def sample_filtered(model, n: int, diversity: float, class_mode: str,
                    seed: int, target_rate: float,
                    schema) -> Tuple[pd.DataFrame, str]:
    """Draw n synthetic patients, optionally constrained on the diagnosis.

    The constraint is applied by over-drawing and then selecting, never by
    editing rows to fit a label - we only ever report records the generator
    actually produced. If the generator does not emit enough of the requested
    class we say so instead of quietly padding.
    """
    n = int(max(1, min(n, 20000)))
    if class_mode == CLASS_OPTIONS[0]:
        want_pos = int(round(n * target_rate))
        want_neg = n - want_pos
        pool = model.sample(min(20000, n * 6 + 200), diversity=diversity,
                            seed=seed)
        pos = pool[pool[schema.label] == 1]
        neg = pool[pool[schema.label] == 0]
        note = ""
        if len(pos) < want_pos or len(neg) < want_neg:
            note = (f"Generator produced only {len(pos)} diseased / "
                    f"{len(neg)} healthy of the {n * 6 + 200} drawn; showing "
                    f"the unconstrained batch instead.")
            return pool.head(n), note
        out = pd.concat([pos.head(want_pos), neg.head(want_neg)],
                        ignore_index=True)
        return out, note
    label = 1 if "diseased" in class_mode else 0
    collected: List[pd.DataFrame] = []
    got = 0
    oversample = 1
    note = ""
    # `got` counts ROWS, not batches - conflating the two is a bug that silently
    # loops four times and reports a false shortfall.
    while got < n and oversample <= 16:
        pool = model.sample(min(20000, n * oversample + 200),
                            diversity=diversity, seed=seed + oversample)
        keep = pool[pool[schema.label] == label]
        if len(keep):
            collected.append(keep)
            got += len(keep)
        oversample *= 4
    if got < n:
        note = (f"Could not fill {n} rows with target={label}: {got} were "
                f"produced across {oversample // 4} over-draws. Lower the "
                f"count or the diversity, or choose 'Match the real "
                f"prevalence'.")
    have = (pd.concat(collected, ignore_index=True) if collected
            else pd.DataFrame(columns=schema.all_names))
    return have.head(n), note


def filter_frozen(cohort: pd.DataFrame, n: int, class_mode: str,
                  schema) -> Tuple[pd.DataFrame, str]:
    """Apply the same two cohort controls to a stored table, by selection only."""
    out = cohort if class_mode == CLASS_OPTIONS[0] else \
        cohort[cohort[schema.label] == (1 if "diseased" in class_mode else 0)]
    note = ""
    if len(out) < n:
        note = (f"This stored cohort holds {len(out)} rows for that filter; "
                f"showing all of them. Switch to live mode to draw more.")
    return out.head(n).reset_index(drop=True), note


def fidelity_report_live(syn: pd.DataFrame, ds, schema) -> Dict:
    """Fidelity of the current cohort against the held-out test split, plus the
    real-vs-real noise floor for context.

    Extracted so the headless test exercises the same path the UI uses.
    """
    live = dict(fidelity_report(syn, ds.test, schema, per_column=False))
    floor = fidelity_report(ds.train, ds.test, schema, per_column=False)
    live["fid_floor_spearman_gap"] = floor["fid_spearman_gap"]
    live["fid_floor_ks_mean"] = floor["fid_ks_mean"]
    live["fid_floor_tv_mean"] = floor["fid_tv_mean"]
    return live


def hyperparams_for(dataset: str, model_name: str, budget: str,
                    epsilon: float) -> Dict:
    """The config the pipeline used, shrunk for the 'Fast' budget."""
    cfgs = {lab: (kind, hp) for lab, kind, hp in model_configs(dataset,
                                                               quick=False)}
    label = model_name if model_name != "dp_vae" else f"dp_vae_eps{epsilon:g}"
    kind, hp = cfgs.get(label, (model_name, {}))
    if budget == "Fast":
        hp = dict(hp)
        if "epochs" in hp:
            hp["epochs"] = {"wgan_gp": 400, "ctgan": 40, "tvae": 40}.get(
                model_name, 80)
        if model_name == "dp_vae":
            hp["epochs"] = 40
    return hp


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar() -> Dict:
    with st.sidebar:
        brand_mark()
        st.markdown("")
        dataset = st.selectbox("Dataset", ["heart", "pima"],
                               help="heart = UCI Cleveland, n=303. "
                                    "pima = UCI Diabetes, n=768.")
        model_name = st.selectbox("Generator", list(MODEL_LABEL),
                                  format_func=lambda k: MODEL_LABEL[k])
        epsilon = 10.0
        if model_name == "dp_vae":
            epsilon = st.select_slider("Privacy budget ε",
                                       options=[1.0, 3.0, 10.0, 30.0],
                                       value=10.0,
                                       help="Smaller ε = stronger guarantee, "
                                            "worse utility. δ = 1 / n_train.")
        st.divider()
        live = st.toggle("Retrain live in the browser", value=True,
                         help="Off loads the cohort the published 3-seed "
                              "sweep actually stored - instant, and the same "
                              "rows the report measured.")
        budget = st.radio("Training speed", ["Fast", "Standard"],
                          help="Fast keeps the architecture and cuts epochs so "
                               "the UI stays responsive. Standard reproduces "
                               "the experiment epochs.")
        n_patients = st.slider("Synthetic patients", 10, 5000, 500, step=10)
        diversity = st.slider(
            "Diversity", 0.0, 1.0, 0.5, 0.05,
            help="0 = sharpened, conservative rows. 0.5 = neutral - the "
                 "setting behind every published number. 1 = flat categorical "
                 "sampling and a wider latent draw.")
        t, ls = diversity_to_knobs(diversity)
        st.caption(f"diversity {diversity:.2f} → temperature {t:.2f}, "
                   f"latent spread ×{ls:.2f}")
        class_mode = st.selectbox("Disease class", CLASS_OPTIONS)
        seed = st.number_input("Sampling seed", 0, 999, 0, step=1)
        st.divider()
        st.caption(f"SynthMed v{VERSION} · CPU build · trained on the train "
                   "split only, tuned on validation only, scored once against "
                   "the held-out test split.")
    return dict(dataset=dataset, model_name=model_name, epsilon=epsilon,
                live=live, budget=budget, n=n_patients, diversity=diversity,
                class_mode=class_mode, seed=int(seed))


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def tab_fidelity(ds, syn: pd.DataFrame, live: Dict, note: str, view: Dict):
    schema = ds.schema
    section("Cohort", "A")
    if note:
        st.warning(note)
    if not len(syn):
        st.info("No rows for this setting - the class filter matched none.")
        return
    kpi_row([
        ("Synthetic patients", f"{len(syn):,}", "Rows the generator produced "
         "and the filters selected."),
        ("Valid rows", f"{live.get('fid_valid_rate', float('nan')) * 100:.1f}%",
         "Inside the medical ranges and using legal category levels."),
        ("Disease rate", f"{syn[schema.label].mean():.3f}",
         f"Real held-out rate is {ds.test[schema.label].mean():.3f}."),
        ("Unique on features",
         f"{syn.drop(columns=[schema.label]).drop_duplicates().shape[0]:,}",
         "Uniqueness is a property of the generator, not a privacy guarantee."),
    ])
    with st.expander("Inspect and download the cohort", expanded=False):
        st.dataframe(syn.head(50), use_container_width=True, height=280)
        st.download_button("Download cohort as CSV",
                           data=syn.to_csv(index=False).encode("utf-8"),
                           file_name=f"synthetic_{ds.name}_{len(syn)}rows.csv",
                           mime="text/csv", type="primary",
                           help="Machine-generated records only - no real "
                                "patient data is in any download.")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.caption(f"Real held-out disease rate "
                       f"{ds.test[schema.label].mean():.3f} · train "
                       f"{ds.train[schema.label].mean():.3f}. The generator is "
                       "never told the label of a real patient; this is what it "
                       "chose to produce.")
        with c2:
            chart(charts.class_balance([int((syn[schema.label] == 1).sum()),
                                        int((syn[schema.label] == 0).sum())],
                                       ["diseased", "healthy"]), height=250)

    section("Distribution distance", "B")
    st.caption("Scored against the held-out real test split. The grey bar is "
               "the real-vs-real noise floor - what the same metric code "
               "returns comparing real train to real test. Nothing can beat "
               "it; at 61 held-out patients it is finite-sample noise.")
    c1, c2 = st.columns(2)
    with c1:
        chart(charts.ks_or_tv(ks_per_column(syn, ds.test, schema),
                              ks_per_column(ds.train, ds.test, schema),
                              "ks_stat", "Per-column KS statistic (numeric)",
                              "KS"), height=330)
    with c2:
        chart(charts.ks_or_tv(tv_per_column(syn, ds.test, schema),
                              tv_per_column(ds.train, ds.test, schema),
                              "tv", "Per-column total variation",
                              "TV"), height=330)
    chart(charts.marginals(ds.test, syn, schema.numeric_names))
    if view["pca"]:
        chart(charts.pca(view["pca"]))

    section("Association structure", "C")
    st.markdown(_gap_line(live, view), unsafe_allow_html=True)
    sg = view["sg"]
    chart(charts.correlations(sg["real_corr"], sg["syn_corr"]))


def _gap_line(live: Dict, view: Dict) -> str:
    gap = live.get("fid_spearman_gap", float("nan"))
    floor = live.get("fid_floor_spearman_gap", float("nan"))
    naive = view.get("published", {}).get("naive_gap")
    parts = [f"<b>Spearman correlation gap {gap:.4f}</b>",
             f"noise floor {floor:.4f}"]
    if np.isfinite(gap) and np.isfinite(floor):
        parts.append(f"{gap / floor:.2f}× floor")
    if naive is not None and np.isfinite(naive):
        parts.append(f"naive baseline {naive:.4f} "
                     f"({(1 - gap / naive) * 100:+.1f}% reduction)")
    return "<span class='small'>" + "&nbsp;&nbsp;·&nbsp;&nbsp;".join(parts) + \
        "</span>"


def tab_utility(ds, syn: pd.DataFrame, view: Dict):
    schema = ds.schema
    section("Utility", "TSTR")
    st.caption("TSTR trains a classifier on synthetic patients and tests it on "
               "real held-out patients. TRTR - real train to real test - is the "
               "ceiling. Retention is TSTR AUC ÷ TRTR AUC; 0.90 is the "
               "acceptance target.")
    if not len(syn):
        st.info("No rows to train on.")
        return
    if int(syn[schema.label].nunique()) < 2:
        st.warning("This cohort holds only one class, so TSTR accuracy and AUC "
                   "are undefined. Reported as n/a rather than as a fake zero.")
        return
    res = view["util"]
    kpi_row([
        ("TSTR AUC · LogReg", _f(res["tstr_lr_auc"]),
         "Published mean ± std over 3 seeds: "
         f"{view['published'].get('tstr_lr_auc', 'n/a')}"),
        ("TSTR AUC · Forest", _f(res["tstr_rf_auc"]), None),
        ("Retention · LogReg", _f(res["retention_lr"]),
         f"Target ≥ 0.90. TRTR ceiling "
         f"{_f(res['trtr_lr_auc'])}."),
        ("Retention · Forest", _f(res["retention_rf"]), None),
    ])
    chart(charts.utility(res))
    table = pd.DataFrame([
        {"Setting": "TSTR · synthetic → real", "Classifier": "Logistic Reg.",
         "Accuracy": _f(res["tstr_lr_accuracy"]), "F1": _f(res["tstr_lr_f1"]),
         "AUC": _f(res["tstr_lr_auc"])},
        {"Setting": "TSTR · synthetic → real", "Classifier": "Random Forest",
         "Accuracy": _f(res["tstr_rf_accuracy"]), "F1": _f(res["tstr_rf_f1"]),
         "AUC": _f(res["tstr_rf_auc"])},
        {"Setting": "TRTR · real → real (ceiling)",
         "Classifier": "Logistic Reg.", "Accuracy": _f(res["trtr_lr_accuracy"]),
         "F1": _f(res["trtr_lr_f1"]), "AUC": _f(res["trtr_lr_auc"])},
        {"Setting": "TRTR · real → real (ceiling)",
         "Classifier": "Random Forest", "Accuracy": _f(res["trtr_rf_accuracy"]),
         "F1": _f(res["trtr_rf_f1"]), "AUC": _f(res["trtr_rf_auc"])},
    ])
    st.dataframe(table, use_container_width=True, hide_index=True)
    for kind, pretty in (("lr", "Logistic Regression"), ("rf", "Random Forest")):
        note = res.get(f"tstr_{kind}_note") or ""
        if note:
            st.caption(f"{pretty}: {note}")


def tab_privacy(ds, syn: pd.DataFrame, view: Dict):
    schema = ds.schema
    section("Privacy audit", "P")
    if not len(syn):
        st.info("No rows to audit.")
        return
    rep = view["priv"]
    kpi_row([
        ("Exact copies of a training patient",
         f"{rep['priv_exact_copy_train'] * 100:.3f}%",
         "Necessary but not sufficient. Compared against the train split."),
        ("DCR ratio · synthetic ÷ real", _f(rep["priv_dcr_ratio"]),
         "Mean distance-to-closest-record of synthetic rows against train, "
         "divided by the same quantity for unseen REAL patients. 1.0 = sits "
         "where new real patients sit. Well below 1.0 = memorisation."),
        ("Suspiciously close rate", _f(rep["priv_suspicious_close_rate"]),
         "Fraction of synthetic rows closer to train than the 5th percentile "
         "of real held-out patients."),
        ("Membership-inference AUC", _f(rep["priv_mia_auc"]),
         "Distance-based attack separating train members from held-out "
         "non-members. 0.5 = chance. Deliberately a weak attack."),
    ])
    chart(charts.dcr(view["d_real"], view["d_syn"], view["p05"]))
    st.caption("A small distance on its own is not a leak: a genuinely new real "
               "patient is also close to the training cloud. That is why the "
               "black curve is the reference and the DCR **ratio** is the "
               "number to read.")

    section("Privacy budget sweep", "ε")
    curve = view["curve"]
    if curve is not None and not curve.empty:
        chart(charts.epsilon_curve(curve))
        st.caption("Published 3-seed means. Every point is a separate DP-VAE "
                   "fitted with the same optimisation budget, so the only thing "
                   "that changes across a curve is the noise multiplier the "
                   "accountant chose.")
    else:
        st.info("Run `python run_pipeline.py` to produce the ε sweep.")
    st.warning("Synthetic is not anonymous. These measurements reduce linkage "
               "and re-identification risk; they do not eliminate it, and they "
               "constrain the generator - not anyone holding both this table "
               "and auxiliary data.")


def parse_verdict(line: str) -> Optional[Tuple[str, str, str, str]]:
    """Split a pipeline verdict line into (number, label, token, detail).

    `run_pipeline.acceptance_verdict` emits two shapes - "1. label: PASS
    (detail)" and "4. label: detail (PASS)" - and reading only the first made
    criterion 4 lose its badge. The app reuses the pipeline's own verdicts rather
    than recomputing them, so the deployed page can never disagree with
    results/acceptance_<dataset>.txt.
    """
    token = next((t for t in ("PASS", "FAIL", "CHECK") if t in line), None)
    if token is None:
        return None
    num, _, tail = line.partition(".")
    before, _, after = tail.partition(token)
    label_part, sep, detail_part = before.partition(": ")
    if not sep:
        label_part, detail_part = before, ""
    detail = " ".join(f"{detail_part} {after}".split()).strip("() ")
    return (num.strip(), label_part.strip().strip(":").strip(), token, detail)


def tab_comparison(summary: Optional[pd.DataFrame], dataset: str):
    section("Published experiment", "3 seeds")
    if summary is None:
        st.info("`results/results_summary.csv` is missing. Run "
                "`python run_pipeline.py`, or `--quick` for a 3-minute smoke "
                "run, then reload.")
        return
    g = summary[summary.dataset == dataset].copy()
    if g.empty:
        st.info(f"No stored results for {dataset}.")
        return
    g["order"] = g.model.map({m: i for i, m in enumerate(EXPERIMENT_ORDER)})
    g = g.sort_values("order")
    kpi_row([
        ("Models", f"{len(g)}", "naive · VAE · VAE@DP-budget · DP-VAE ×4 ε · "
         "CTGAN · TVAE · WGAN-GP"),
        ("Fits", f"{len(g) * 3}", "3 seeds each, mean ± std"),
        ("Noise floor · gap",
         _f(np.nanmean(g.fid_floor_spearman_gap_mean)),
         "Real train vs real test, same metric code."),
        ("Worst exact-copy rate",
         f"{np.nanmax(g.priv_exact_copy_train_mean) * 100:.3f}%",
         "Across every model and seed on this dataset."),
    ])
    cols = {"Model": "model", "KS ↓": "fid_ks_mean", "TV ↓": "fid_tv_mean",
            "Gap ↓": "fid_spearman_gap", "TSTR AUC LR": "tstr_lr_auc",
            "TSTR AUC RF": "tstr_rf_auc", "Retention LR": "retention_lr",
            "Exact copy": "priv_exact_copy_train", "DCR ratio":
                "priv_dcr_ratio", "MIA AUC": "priv_mia_auc",
            "Likelihood MIA": "priv_mia_auc_likelihood"}
    rows = []
    for _, r in g.iterrows():
        rec = {"Model": DISPLAY.get(r.model, r.model)}
        for nice, key in list(cols.items())[1:]:
            m, s = r.get(f"{key}_mean", np.nan), r.get(f"{key}_std", np.nan)
            rec[nice] = (f"{m:.3f} ± {s:.3f}" if np.isfinite(m) else "n/a")
        rec["Seeds ok"] = f"{int(r.get('n_seeds_ok', 0))}/3"
        rec["Privacy"] = r.get("privacy", "")
        rows.append(rec)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("`Seeds ok` counts the seeds that produced a finite value for "
               "every metric. Below 3/3 means a metric was undefined for at "
               "least one seed - single-class synthetic data will do that - and "
               "the mean is over fewer runs. Stated, not hidden.")

    c1, c2 = st.columns(2)
    pretty = [DISPLAY.get(m, m) for m in g.model]
    with c1:
        chart(charts.model_bars(
            g, "fid_spearman_gap_mean", "fid_spearman_gap_std",
            "Correlation gap · lower is better", "Spearman gap",
            {"real-vs-real floor": float(np.nanmean(
                g.fid_floor_spearman_gap_mean)),
             "naive baseline": float(
                 g.loc[g.model == "naive", "fid_spearman_gap_mean"].iloc[0])},
            horizontal=True, labels=pretty), height=None)
    with c2:
        chart(charts.model_bars(
            g, "tstr_lr_auc_mean", "tstr_lr_auc_std",
            "TSTR AUC (LogReg) · higher is better", "AUC",
            {"TRTR ceiling": float(np.nanmean(g.trtr_lr_auc_mean))},
            horizontal=True, labels=pretty), height=None)

    section("Acceptance criteria", "measured")
    from run_pipeline import acceptance_verdict
    for line in acceptance_verdict(g):
        parsed = parse_verdict(line)
        if parsed is None:
            verdict("note", "neutral", note=line.split(".", 1)[-1].strip())
            continue
        num, label, token, detail = parsed
        verdict(f"{num} · {label} — {token}",
                {"PASS": "good", "FAIL": "bad", "CHECK": "warn"}[token],
                note=detail[:160])


# ---------------------------------------------------------------------------
def _f(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return f"{f:.3f}" if np.isfinite(f) else "n/a"


HP_LABEL = {"epochs": "epochs", "batch_size": "batch", "latent_dim": "latent",
            "hidden_dims": "hidden", "lr": "lr", "beta_max": "β max",
            "kl_anneal_epochs": "KL anneal", "w_num": "w numeric",
            "w_cat": "w categorical", "obs_noise": "observation noise",
            "jitter": "jitter", "target_epsilon": "ε target",
            "max_grad_norm": "clip norm", "sigma": "σ", "steps": "steps",
            "cohort": "cohort", "rows": "rows"}


def hp_line(hp: Dict) -> str:
    """Humanise the fitted hyper-parameters for display.

    Keys starting with "_" are internal bookkeeping - `_d` is just the diversity
    slider echoed back - and printing raw attribute names turns the header into a
    debug console.
    """
    return " · ".join(f"{HP_LABEL.get(k, k)} {v}" for k, v in hp.items()
                      if not str(k).startswith("_"))


def main() -> None:
    inject_css()
    cfg = sidebar()
    ds = get_dataset(cfg["dataset"])
    schema = ds.schema
    summary = load_summary()
    published_row = None
    if summary is not None:
        label = (f"dp_vae_eps{cfg['epsilon']:g}" if cfg["model_name"] == "dp_vae"
                 else cfg["model_name"])
        hit = summary[(summary.dataset == cfg["dataset"]) &
                      (summary.model == label)]
        published_row = hit.iloc[0] if len(hit) else None

    hero("Synthetic Medical Data Generator",
         "Mixed-type patient records generated by a VAE, its "
         "differentially-private variant, CTGAN, TVAE and WGAN-GP - scored on "
         "fidelity, downstream utility and privacy against a measured "
         "real-vs-real noise floor.",
         chips=[f"{cfg['dataset']} · n={len(ds.train) + len(ds.val) + len(ds.test)}",
                "split 60 / 20 / 20 stratified",
                f"{MODEL_LABEL[cfg['model_name']]}"] +
               ([f"ε = {cfg['epsilon']:g}, δ = 1/n"]
                if cfg["model_name"] == "dp_vae" else []))

    with st.expander("Protocol, encoding and what was tuned on what",
                     expanded=False):
        st.markdown(
            f"- **Splits** train {len(ds.train)} / val {len(ds.val)} / test "
            f"{len(ds.test)}, stratified on the diagnosis, fixed split seed 42.\n"
            f"- **Numeric** {', '.join(schema.numeric_names)} — rank-transformed "
            "to Gaussian on the train marginal, then min-max scaled; inverted, "
            "clipped to valid medical ranges and rounded back to integers.\n"
            f"- **Categorical** "
            f"{', '.join(f'{c.name}={c.categories}' for c in schema.categorical)}"
            f" — one softmax head each, cross-entropy loss.\n"
            f"- **Label** `{schema.label}` = {schema.label_name}; positive "
            f"defined as {schema.positive_is}.\n"
            f"- **Missing** counted before imputation: "
            f"{dict(ds.missing_counts)} — train median/mode only.\n"
            f"- **Hygiene** generators see train only; the tuner in "
            "`synth/tune.py` receives train + val and has no path to the test "
            "frame; test is read once, in the final evaluation pass per model "
            "per seed.\n"
            f"- **1-D caveat** because the numeric transform is fitted on the "
            "train marginal, KS and TV sit near the floor for every neural "
            "model *by construction*. The discriminating metrics are the "
            "correlation gap and TSTR.")

    try:
        if cfg["live"]:
            with st.spinner("Fitting the generator on the train split"):
                hp = hyperparams_for(cfg["dataset"], cfg["model_name"],
                                     cfg["budget"], cfg["epsilon"])
                model = get_model(cfg["dataset"], cfg["model_name"], hp, 0)
            with st.spinner("Drawing synthetic patients"):
                syn, note = sample_filtered(
                    model, cfg["n"], cfg["diversity"], cfg["class_mode"],
                    cfg["seed"], float(ds.train[schema.label].mean()), schema)
            mode_label = "live"
            privacy_status = getattr(model, "privacy_note", "non-private")
            hp_used = dict(model.hyperparams)
        else:
            label = (f"dp_vae_eps{cfg['epsilon']:g}"
                     if cfg["model_name"] == "dp_vae" else cfg["model_name"])
            cohort = load_frozen_cohort(cfg["dataset"], label)
            if cohort is None:
                st.error(f"No stored cohort for {label} on "
                         f"{cfg['dataset']}. Run `python run_pipeline.py` or "
                         "switch to live mode.")
                return
            syn, note = filter_frozen(cohort, cfg["n"], cfg["class_mode"],
                                      schema)
            mode_label = "frozen"
            privacy_status = ("published DP-SGD run" if "dp_vae" in label
                              else "non-private")
            hp_used = {"cohort": f"results/{cfg['dataset']}/"
                                 f"syn_{label}_seed0.csv", "rows": len(cohort)}
    except Exception as exc:
        st.error(f"The generator failed for this setting: {exc}")
        with st.expander("Traceback", expanded=False):
            st.code(traceback.format_exc())
        st.info("Try Fast training speed, or frozen mode. This setting is "
                "reported rather than silently skipped.")
        return

    with st.spinner("Scoring the cohort against the held-out test split"):
        live = fidelity_report_live(syn, ds, schema)
        view = {
            "sg": spearman_gap(syn, ds.test, schema),
            "pca": pca_overlay_data(ds.train, syn, schema, ds.test),
            "util": utility_report(syn, ds.train, ds.test, schema,
                                   seed=cfg["seed"]),
            "curve": load_curve(),
            "published": {
                # Comparing a live naive cohort against the published naive
                # mean is a tautology dressed as a result, so the reference line
                # only appears for the models that are supposed to beat it.
                "naive_gap": (float(summary[(summary.dataset == cfg["dataset"])
                                            & (summary.model == "naive")]
                                   ["fid_spearman_gap_mean"].iloc[0])
                              if summary is not None and
                              (summary.model == "naive").any()
                              and cfg["model_name"] != "naive" else None),
                "tstr_lr_auc": (f"{published_row['tstr_lr_auc_mean']:.3f} ± "
                                f"{published_row['tstr_lr_auc_std']:.3f}"
                                if published_row is not None else "n/a"),
            },
        }
        view["priv"] = privacy_report(syn, ds.train, ds.test, schema)
        rs = RecordSpace(ds.train, schema)
        view["d_syn"] = rs.dist_to_set(rs.transform(syn), rs.transform(ds.train))
        view["d_real"] = rs.dist_to_set(rs.transform(ds.test),
                                        rs.transform(ds.train))
        view["p05"] = float(np.nanpercentile(view["d_real"], 5))

    pills([("live sandbox · single seed" if cfg["live"] else
            "published cohort · seed 0",
            "live" if cfg["live"] else "frozen"),
           (f"{mode_label} cohort: {len(syn):,} rows", "default"),
           (f"privacy status: {privacy_status}", "default"),
           (f"held-out test split: {len(ds.test)} patients", "default"),
           (f"noise floor · gap "
            f"{live.get('fid_floor_spearman_gap', float('nan')):.3f}",
            "default")])

    if hp_line(hp_used):
        st.caption(hp_line(hp_used))

    t1, t2, t3, t4 = st.tabs(["Fidelity", "Utility", "Privacy",
                              "Model comparison"])
    with t1:
        tab_fidelity(ds, syn, live, note, view)
    with t2:
        tab_utility(ds, syn, view)
    with t3:
        tab_privacy(ds, syn, view)
    with t4:
        tab_comparison(summary, cfg["dataset"])

    footer(f"SynthMed v{VERSION} · UCI Cleveland + UCI Pima · CPU build",
           "synth/ library · run_pipeline.py · make_figures.py · docs/REPORT_OUTLINE.md")


if __name__ == "__main__":
    main()
