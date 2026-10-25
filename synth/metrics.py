"""
Evaluation: the three families that actually decide whether synthetic medical
data is usable.

    A. Fidelity  - does it look like the real thing, column by column and
                   joint by joint?
    B. Utility   - does a model trained on it still work on real patients?
                   (TSTR vs TRTR)
    C. Privacy   - can we find a real patient inside it, and can an attacker
                   tell who was in the training set?

PROTOCOL, stated once because it governs every number in this file
-----------------------------------------------------------------
    train (60%)  -> generators are fit here ONLY
    val   (20%)  -> hyper-parameters are chosen here ONLY
    test  (20%)  -> touched exactly once, at the end, for the headline numbers

Fidelity and utility are therefore computed against the *held-out test split*,
never against the data the generator was trained on.  Comparing synthetic to
train-real would flatter every model, because a generator that has simply
memorised its training set scores perfectly on it - which is a privacy failure,
not a fidelity success.  Where we do report a train-real comparison (the
distance-to-closest-record memorisation check) it is labelled `_train`.

Every public function is NaN-safe: a degenerate case returns NaN and records a
note rather than raising, because one broken model must not abort a 3-seed x
6-model x 2-dataset sweep.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr

from .data import Schema
from .encoding import TabularEncoder

warnings.filterwarnings("ignore", category=RuntimeWarning)

NAN = float("nan")


# ---------------------------------------------------------------------------
# Small guards
# ---------------------------------------------------------------------------
def _nan_if(fn, *args, **kw):
    """Run a metric, returning NaN instead of raising on a degenerate input."""
    try:
        v = fn(*args, **kw)
    except Exception:
        return NAN
    return float(v) if v is not None else NAN


def _empty(df: pd.DataFrame) -> bool:
    return df is None or len(df) == 0


# ===========================================================================
# A. FIDELITY
# ===========================================================================
def ks_per_column(syn: pd.DataFrame, real: pd.DataFrame,
                  schema: Schema) -> pd.DataFrame:
    """Two-sample Kolmogorov-Smirnov test for every numeric column.

    KS statistic = the largest vertical gap between the two empirical CDFs, so
    it is 0 when the marginals match and 1 when they are disjoint.  We report
    both the statistic (effect size) and the p-value.  With a 61-row held-out
    split the p-value has little power, so the statistic is the honest headline.
    """
    rows = []
    for c in schema.numeric:
        if c.name not in syn.columns or c.name not in real.columns:
            rows.append({"column": c.name, "ks_stat": NAN, "ks_pvalue": NAN})
            continue
        a = syn[c.name].to_numpy(dtype=float)
        b = real[c.name].to_numpy(dtype=float)
        if len(a) == 0 or len(b) == 0 or np.allclose(a, a[0]) and \
                np.allclose(b, b[0]):
            rows.append({"column": c.name, "ks_stat": NAN, "ks_pvalue": NAN})
            continue
        try:
            res = ks_2samp(a, b, method="asymp")
            rows.append({"column": c.name, "ks_stat": float(res.statistic),
                         "ks_pvalue": float(res.pvalue)})
        except Exception:
            rows.append({"column": c.name, "ks_stat": NAN, "ks_pvalue": NAN})
    return pd.DataFrame(rows)


def tv_per_column(syn: pd.DataFrame, real: pd.DataFrame,
                  schema: Schema) -> pd.DataFrame:
    """Total-variation distance for every categorical column, plus the label.

    TV(p, q) = 0.5 * sum_k |p_k - q_k| over the category vocabulary.  It is the
    natural analogue of the KS statistic for discrete data: 0 = identical
    marginals, 1 = completely disjoint supports.  Computed over the union of
    categories so an illegal invented level would show up rather than vanish.
    """
    groups = [(c.name, list(c.categories)) for c in schema.categorical]
    groups.append((schema.label, [0, 1]))
    rows = []
    for name, cats in groups:
        if name not in syn.columns or name not in real.columns:
            rows.append({"column": name, "tv": NAN})
            continue
        cats = sorted(set(cats) | set(np.unique(syn[name])) |
                      set(np.unique(real[name])))
        p = np.array([(syn[name].to_numpy() == k).mean() for k in cats],
                     dtype=float)
        q = np.array([(real[name].to_numpy() == k).mean() for k in cats],
                     dtype=float)
        rows.append({"column": name, "tv": float(0.5 * np.abs(p - q).sum())})
    return pd.DataFrame(rows)


def spearman_gap(syn: pd.DataFrame, real: pd.DataFrame,
                 schema: Schema) -> Dict:
    """How much does the rank-correlation structure drift?

    Spearman rather than Pearson because clinical measures are skewed
    (cholesterol, insulin) and outliers would dominate a Pearson matrix.  The
    'gap' is the mean absolute difference between the two correlation matrices
    over all off-diagonal pairs; `gap_max` is the single worst pair, which is
    what you actually want to know when a specific clinical association (say
    max heart rate vs. exercise-induced angina) has been destroyed.

    Categorical codes are treated as ordered levels; for `thal` {3,6,7} that is
    a mild abuse of a rank statistic and is listed in the limitations.
    """
    cols = schema.all_names
    if _empty(syn) or _empty(real):
        return {"spearman_gap": NAN, "spearman_gap_max": NAN,
                "real_corr": pd.DataFrame(), "syn_corr": pd.DataFrame()}
    try:
        R = real[cols].corr(method="spearman").to_numpy(dtype=float)
        S = syn[cols].corr(method="spearman").to_numpy(dtype=float)
    except Exception:
        return {"spearman_gap": NAN, "spearman_gap_max": NAN,
                "real_corr": pd.DataFrame(), "syn_corr": pd.DataFrame()}
    k = len(cols)
    mask = ~np.eye(k, dtype=bool)
    diff = np.abs(np.nan_to_num(R) - np.nan_to_num(S))
    return {
        "spearman_gap": float(diff[mask].mean()) if k > 1 else NAN,
        "spearman_gap_max": float(diff[mask].max()) if k > 1 else NAN,
        "real_corr": pd.DataFrame(R, index=cols, columns=cols),
        "syn_corr": pd.DataFrame(S, index=cols, columns=cols),
    }


def pca_overlay_data(train: pd.DataFrame, syn: pd.DataFrame,
                     schema: Schema,
                     heldout: Optional[pd.DataFrame] = None) -> Dict:
    """2-D PCA coordinates for the overlay figure.

    PCA is fitted on the *encoded* representation (min-max numerics + one-hot
    categoricals) because plain PCA on mixed types is meaningless.  The
    projection basis comes from the real training split, then real held-out and
    synthetic rows are projected onto it, so all three clouds are directly
    comparable in one coordinate system.

    The held-out real cloud is the control: it shows how far a *genuine* new
    batch of 61 patients drifts from the training cloud.  A synthetic cloud
    inside that envelope is doing its job.
    """
    from sklearn.decomposition import PCA

    enc = TabularEncoder(schema)
    if _empty(syn) or _empty(train):
        return {}
    Xtr = enc.encode(train)
    pca = PCA(n_components=min(2, Xtr.shape[1]), random_state=0)
    pca.fit(Xtr)
    out = {
        "train": pca.transform(Xtr),
        "syn": pca.transform(enc.encode(syn)),
        "explained": pca.explained_variance_ratio_,
    }
    if heldout is not None and len(heldout):
        out["heldout"] = pca.transform(enc.encode(heldout))
    return out


def fidelity_report(syn: pd.DataFrame, real: pd.DataFrame, schema: Schema,
                    prefix: str = "fid", per_column: bool = True) -> Dict:
    """Flatten every fidelity metric into scalar columns for the results CSV."""
    if _empty(syn) or _empty(real):
        return {f"{prefix}_ks_mean": NAN, f"{prefix}_tv_mean": NAN,
                f"{prefix}_spearman_gap": NAN,
                f"{prefix}_spearman_gap_max": NAN,
                f"{prefix}_valid_rate": 0.0}
    ks = ks_per_column(syn, real, schema)
    tv = tv_per_column(syn, real, schema)
    sg = spearman_gap(syn, real, schema)
    out = {
        f"{prefix}_ks_mean": float(np.nanmean(ks["ks_stat"])),
        f"{prefix}_ks_max": float(np.nanmax(ks["ks_stat"])),
        f"{prefix}_tv_mean": float(np.nanmean(tv["tv"])),
        f"{prefix}_tv_max": float(np.nanmax(tv["tv"])),
        f"{prefix}_spearman_gap": sg["spearman_gap"],
        f"{prefix}_spearman_gap_max": sg["spearman_gap_max"],
        f"{prefix}_valid_rate": float(validity_rate(syn, schema)),
    }
    if not per_column:
        return out
    # Per-column detail is kept so a reviewer can see *which* association broke.
    for _, r in ks.iterrows():
        out[f"{prefix}_ks_{r['column']}"] = float(r["ks_stat"])
    for _, r in tv.iterrows():
        out[f"{prefix}_tv_{r['column']}"] = float(r["tv"])
    return out


def validity_rate(syn: pd.DataFrame, schema: Schema) -> float:
    """Fraction of generated rows inside valid medical ranges / legal levels.

    Should be exactly 1.0 for every model, because post-processing enforces it;
    we still measure it rather than assume it, and it would drop below 1 if the
    sanitiser were ever bypassed.
    """
    ok = np.ones(len(syn), dtype=bool)
    for c in schema.numeric:
        lo, hi = schema.clip_bounds(c)
        ok &= (syn[c.name].to_numpy(dtype=float) >= lo) & \
              (syn[c.name].to_numpy(dtype=float) <= hi)
    for c in schema.categorical:
        ok &= np.isin(syn[c.name].to_numpy(), c.categories)
    ok &= np.isin(syn[schema.label].to_numpy(), [0, 1])
    return float(ok.mean()) if len(syn) else NAN


# ===========================================================================
# B. UTILITY  (TSTR / TRTR)
# ===========================================================================
def _make_clf(kind: str, seed: int):
    """The two downstream classifiers.  Deliberately boring and standard:
    Logistic Regression and Random Forest, exactly as specified."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    if kind == "lr":
        return LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=seed,
                                      n_jobs=1)
    raise ValueError(kind)


def _scale(x_fit: np.ndarray, x_use: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Standardise with statistics taken from the *training side* only."""
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(x_fit)
    return sc.transform(x_fit), sc.transform(x_use)


def run_downstream(train_df: pd.DataFrame, test_df: pd.DataFrame,
                   schema: Schema, kind: str, seed: int) -> Dict[str, float]:
    """Train a classifier on `train_df`, test it on the held-out real `test_df`.

    Handles the three ways this legitimately breaks:
      * synthetic data contains only one class  -> cannot train -> NaN + note
      * the real test split has only one class  -> no meaningful AUC -> NaN
      * a constant column                       -> StandardScaler leaves it
    """
    res = {f"{kind}_accuracy": NAN, f"{kind}_f1": NAN, f"{kind}_auc": NAN,
           f"{kind}_pos_rate": NAN, f"{kind}_note": ""}
    if _empty(train_df) or _empty(test_df):
        res[f"{kind}_note"] = "empty train or test"
        return res
    feats = schema.feature_names
    X_fit = train_df[feats].to_numpy(dtype=float)
    X_use = test_df[feats].to_numpy(dtype=float)
    y_fit = train_df[schema.label].to_numpy(dtype=int)
    y_use = test_df[schema.label].to_numpy(dtype=int)
    res[f"{kind}_pos_rate"] = float(y_fit.mean()) if len(y_fit) else NAN

    if len(np.unique(y_fit)) < 2:
        # Single-class synthetic data: the generator collapsed onto one
        # diagnosis class.  AUC is undefined, not zero - report NaN.
        res[f"{kind}_note"] = "single_class_train"
        return res
    if len(np.unique(y_use)) < 2:
        res[f"{kind}_note"] = "single_class_test"
        return res

    Xf, Xu = _scale(X_fit, X_use)
    try:
        clf = _make_clf(kind, seed).fit(Xf, y_fit)
        pred = clf.predict(Xu)
        proba = clf.predict_proba(Xu)[:, 1]
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        res[f"{kind}_accuracy"] = float(accuracy_score(y_use, pred))
        res[f"{kind}_f1"] = float(f1_score(y_use, pred, zero_division=0))
        res[f"{kind}_auc"] = float(roc_auc_score(y_use, proba))
    except Exception as exc:                        # pragma: no cover
        res[f"{kind}_note"] = f"error:{type(exc).__name__}"
    return res


def utility_report(syn: pd.DataFrame, train: pd.DataFrame,
                   test: pd.DataFrame, schema: Schema,
                   seed: int = 0) -> Dict:
    """TSTR (train on synthetic, test on real) next to TRTR (real on real).

    TRTR is the ceiling: the best a downstream analyst could do with the real
    cohort.  `retention_<clf> = tstr_auc / trtr_auc` is the headline utility
    number, and the acceptance criterion is retention >= 0.90 for the best model.

    Note the scaler asymmetry, which is deliberate: in TSTR the downstream user
    only possesses synthetic data, so the standardisation statistics must come
    from the synthetic training set.  Using real-data statistics there would
    quietly smuggle in the very data we are pretending they cannot access.
    """
    out: Dict = {}
    for kind in ("lr", "rf"):
        tstr = run_downstream(syn, test, schema, kind, seed)
        trtr = run_downstream(train, test, schema, kind, seed)
        out[f"tstr_{kind}_accuracy"] = tstr[f"{kind}_accuracy"]
        out[f"tstr_{kind}_f1"] = tstr[f"{kind}_f1"]
        out[f"tstr_{kind}_auc"] = tstr[f"{kind}_auc"]
        out[f"tstr_{kind}_pos_rate"] = tstr[f"{kind}_pos_rate"]
        out[f"tstr_{kind}_note"] = tstr[f"{kind}_note"]
        out[f"trtr_{kind}_accuracy"] = trtr[f"{kind}_accuracy"]
        out[f"trtr_{kind}_f1"] = trtr[f"{kind}_f1"]
        out[f"trtr_{kind}_auc"] = trtr[f"{kind}_auc"]
        ref, got = trtr[f"{kind}_auc"], tstr[f"{kind}_auc"]
        out[f"retention_{kind}"] = (got / ref) if (np.isfinite(ref)
                                                   and np.isfinite(got)
                                                   and ref > 0) else NAN
    return out


# ===========================================================================
# C. PRIVACY
# ===========================================================================
class RecordSpace:
    """Euclidean space in which patient distances are meaningful.

    Mixed-type rows cannot be compared on raw values (a 1-unit change in age is
    trivial, a 1-unit change in `ca` is a whole category).  So we take the
    mode-encoded vector and z-score each dimension using TRAIN statistics.  The
    distance is then divided by sqrt(dim) so the number is comparable between
    datasets of different width.

    Note that standardising on train statistics is exactly what an attacker
    *without* access to the real data cannot do - so this choice makes our
    privacy numbers conservative (it hands the attacker a better-conditioned
    metric than they would have in reality).
    """

    def __init__(self, train: pd.DataFrame, schema: Schema):
        self.enc = TabularEncoder(schema)
        Xtr = self.enc.encode(train)
        self.mean = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        # A constant encoded dimension (a category present in only one level)
        # would divide by zero; those dimensions carry no discriminative
        # information, so we set their scale to 1 and they contribute a
        # constant offset that cancels out of every comparison.
        sd[sd < 1e-8] = 1.0
        self.scale = sd
        self.dim = Xtr.shape[1]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if _empty(df):
            return np.zeros((0, self.dim))
        return (self.enc.encode(df) - self.mean) / self.scale

    def dist_to_set(self, queries: np.ndarray, refs: np.ndarray) -> np.ndarray:
        """Per-row nearest-neighbour distance, scaled by sqrt(dim)."""
        if len(queries) == 0:
            return np.zeros(0)
        if len(refs) == 0:
            return np.full(len(queries), NAN)
        return _chunked_min_dist(queries, refs, self.dim)


def exact_copy_rate(syn: pd.DataFrame, real: pd.DataFrame,
                    schema: Schema) -> float:
    """Fraction of synthetic rows identical to some real row (all columns).

    This is the blunt-force privacy test: a generator that emits real patients
    verbatim has failed regardless of what any distance metric says.
    """
    if _empty(syn) or _empty(real):
        return NAN
    key_syn = _row_keys(syn, schema)
    key_real = set(_row_keys(real, schema))
    return float(np.mean([k in key_real for k in key_syn]))


def _row_keys(df: pd.DataFrame, schema: Schema) -> List[tuple]:
    """Hashable tuple per row, with floats rounded so 'equal' means equal."""
    cols = schema.all_names
    vals = df[cols].to_numpy(dtype=float)
    return [tuple(np.round(row, 6).tolist()) for row in vals]


def _chunked_min_dist(A: np.ndarray, B: np.ndarray, dim: int) -> np.ndarray:
    """Nearest-neighbour distance from every row of A to the set B.

    Chunked so a user asking the demo for 20 000 synthetic patients does not
    allocate a 20 000 x 460 x 30 float tensor and die.
    """
    if len(A) == 0 or len(B) == 0:
        return np.full(len(A), NAN)
    out = np.empty(len(A))
    for i in range(0, len(A), 1024):
        a = A[i:i + 1024]
        d = ((a[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
        out[i:i + len(a)] = np.sqrt(d.min(axis=1))
    return out / np.sqrt(max(int(dim), 1))


def dcr_summary(dist: np.ndarray) -> Dict[str, float]:
    dist = dist[np.isfinite(dist)]
    if len(dist) == 0:
        return {"mean": NAN, "median": NAN, "p05": NAN, "p95": NAN}
    return {"mean": float(dist.mean()), "median": float(np.median(dist)),
            "p05": float(np.percentile(dist, 5)),
            "p95": float(np.percentile(dist, 95))}


def privacy_report(syn: pd.DataFrame, train: pd.DataFrame,
                   test: pd.DataFrame, schema: Schema,
                   prefix: str = "priv") -> Dict[str, float]:
    """All privacy metrics for one model, as a flat dict.

    Distance-to-closest-record (DCR) is read against three references, and the
    comparison against the *held-out real* baseline is the part people usually
    get wrong:

      syn -> train      small  = the generator has memorised its own input
      syn -> held-out   small  = synthetic rows crowd real unseen patients
      real -> train     the yardstick.  A *legitimate* new real patient is also
                        close to the training cloud; you cannot call a model
                        leaky just because DCR is small in absolute terms.

    So the headline privacy number is the RATIO meanDCR(syn->train) /
    meanDCR(real held-out -> train).  Near 1 means the synthetic rows sit where
    new real patients would sit.  Much below 1 means memorisation.
    """
    if _empty(syn):
        return {f"{prefix}_{k}": NAN for k in (
            "exact_copy_train", "exact_copy_test", "dcr_syn_train_mean",
            "dcr_syn_heldout_mean", "dcr_real_heldout_mean",
            "dcr_ratio", "suspicious_close_rate", "mia_auc")}
    rs = RecordSpace(train, schema)
    S = rs.transform(syn)
    TR = rs.transform(train)
    TE = rs.transform(test)

    d_syn_train = rs.dist_to_set(S, TR)
    d_syn_test = rs.dist_to_set(S, TE)
    # Reference distribution: held-out real patients to the training cloud.
    d_real_train = rs.dist_to_set(TE, TR)

    a = dcr_summary(d_syn_train)
    b = dcr_summary(d_syn_test)
    c = dcr_summary(d_real_train)

    # "Suspiciously close": closer to a training patient than 5% of real
    # unseen patients ever get to the training cloud.
    thresh = c["p05"]
    sus = NAN
    if np.isfinite(thresh):
        sus = float(np.mean(d_syn_train[np.isfinite(d_syn_train)] < thresh)) \
            if len(d_syn_train) else NAN

    # Distance-based membership inference attack.
    mia = membership_inference_auc(TR, TE, S)

    out = {
        f"{prefix}_exact_copy_train": exact_copy_rate(syn, train, schema),
        f"{prefix}_exact_copy_test": exact_copy_rate(syn, test, schema),
        f"{prefix}_dcr_syn_train_mean": a["mean"],
        f"{prefix}_dcr_syn_train_median": a["median"],
        f"{prefix}_dcr_syn_heldout_mean": b["mean"],
        f"{prefix}_dcr_real_heldout_mean": c["mean"],
        f"{prefix}_dcr_real_heldout_p05": c["p05"],
        f"{prefix}_dcr_ratio": (a["mean"] / c["mean"])
            if (np.isfinite(a["mean"]) and np.isfinite(c["mean"])
                and c["mean"] > 0) else NAN,
        f"{prefix}_suspicious_close_rate": sus,
        f"{prefix}_mia_auc": mia,
    }
    return out


def membership_inference_auc(train: np.ndarray, heldout: np.ndarray,
                             syn: np.ndarray) -> float:
    """AUC of a nearest-neighbour membership-inference attack.

    The attacker knows the generator family and holds the synthetic table.  For
    a candidate patient they compute the distance to the nearest synthetic
    record.  If the generator memorised the training set, *members* sit closer
    to the synthetic cloud than non-members do, so -min_distance is a usable
    membership signal and the AUC rises above 0.5.

      AUC ~ 0.5  the attacker cannot tell  -> good privacy
      AUC ~ 1.0  membership is readable off the synthetic data -> leak

    Limitations we state rather than hide: this is a *weak* attack.  It uses no
    model gradients, no shadow models and a single distance threshold, and both
    groups are drawn from the same real distribution, so a perfectly generalising
    model can still score slightly off 0.5 purely from sampling noise.  It is a
    relative comparison across epsilon, not a certificate of privacy.
    """
    from sklearn.metrics import roc_auc_score

    if len(syn) == 0 or len(train) == 0 or len(heldout) == 0:
        return NAN
    dim = syn.shape[1]
    d_member = _chunked_min_dist(train, syn, dim)
    d_non = _chunked_min_dist(heldout, syn, dim)
    if not np.isfinite(d_member).all() or not np.isfinite(d_non).all():
        return NAN
    labels = np.r_[np.ones(len(d_member)), np.zeros(len(d_non))]
    scores = -np.r_[d_member, d_non]
    if len(np.unique(labels)) < 2:
        return NAN
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return NAN


def _per_sample_loss(model, et) -> np.ndarray:
    """Per-sample ELBO loss (reconstruction + beta * KL) for a VAE-family model."""
    import torch
    from .models import kl_divergence_loss, vae_reconstruction_loss
    beta = getattr(model, "beta_max", 1.0)
    X = torch.tensor(et.X)
    N = torch.tensor(et.num)
    C = torch.tensor(et.cat)
    out = []
    model.net.eval()
    with torch.no_grad():
        for i in range(0, max(len(X), 1), 256):
            x, nn_, cc = X[i:i + 256], N[i:i + 256], C[i:i + 256]
            num_hat, cat_logits, mu, logvar = model.net(x)
            rec = vae_reconstruction_loss(num_hat, cat_logits, nn_, cc,
                                          et.widths, w_num=model.w_num,
                                          w_cat=model.w_cat)
            kl = kl_divergence_loss(mu, logvar)
            out.append((rec + beta * kl).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def likelihood_gap_mia(model, train: pd.DataFrame, heldout: pd.DataFrame,
                       schema: Schema) -> float:
    """A second, stronger membership-inference attack: the likelihood gap.

    The distance-based attack only ever sees the released table.  This one uses
    the model.  A generator that has overfitted assigns *higher* likelihood -
    here, lower per-sample ELBO loss - to the records it trained on than to
    unseen records from the same population, so -loss separates members from
    non-members and the AUC of that separation is a direct measure of
    memorisation.

    Why it matters for the DP experiment: DP-SGD exists precisely to stop the
    parameters depending on any individual row, so this gap is the quantity it
    should shrink.  The distance-based attack is largely blind to that
    difference.  This is the classic generalisation-gap attack, and it needs no
    shadow models.

    Returns NaN for models with no per-sample likelihood - the naive resampler
    and the ctgan package objects genuinely do not have one, and we report that
    rather than inventing a proxy.
    """
    from sklearn.metrics import roc_auc_score
    from .models import EncodedTable

    if not hasattr(model, "net") or getattr(model, "enc_table", None) is None:
        return NAN
    if _empty(train) or _empty(heldout):
        return NAN
    try:
        l_member = _per_sample_loss(model, EncodedTable(train, schema))
        l_non = _per_sample_loss(model, EncodedTable(heldout, schema))
    except Exception:
        return NAN
    if len(l_member) == 0 or len(l_non) == 0:
        return NAN
    if not (np.isfinite(l_member).all() and np.isfinite(l_non).all()):
        return NAN
    labels = np.r_[np.ones(len(l_member)), np.zeros(len(l_non))]
    scores = -np.r_[l_member, l_non]
    if len(np.unique(labels)) < 2:
        return NAN
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return NAN


# ===========================================================================
# Aggregate one-model report
# ===========================================================================
@dataclass
class EvalResult:
    """Everything measured for one (model, seed) pair."""

    model: str
    seed: int
    metrics: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    syn: Optional[pd.DataFrame] = None
    extra: Dict = field(default_factory=dict)


def evaluate_model(model_name: str, syn: pd.DataFrame, train: pd.DataFrame,
                   val: pd.DataFrame, test: pd.DataFrame, schema: Schema,
                   seed: int, extra_notes: Sequence[str] = (),
                   model=None) -> EvalResult:
    """Run all three metric families once.  The single entry point used by
    run_pipeline.py, the notebook and the Streamlit app.

    THE MOST IMPORTANT LINE IN THIS FILE is the `fid_noise_floor` block.  Even a
    second copy of the *real* data, drawn independently from the same
    population, would show a non-zero KS statistic and a non-zero Spearman gap,
    simply because the held-out split has only 61 rows.  So we measure that
    floor by comparing real-train to real-test with the identical metric code,
    and then report each model's gap *relative to the floor*.  Without it, "the
    VAE has a correlation gap of 0.08" is uninterpretable; with it, "1.1x the
    real-vs-real noise floor" is a claim an examiner can actually judge.
    """
    metrics: Dict = {}
    notes: List[str] = list(extra_notes)
    # ---- A. fidelity: headline is synthetic vs held-out real ----------------
    metrics.update(fidelity_report(syn, test, schema, prefix="fid"))
    # ---- the real-vs-real noise floor, same metric code ---------------------
    floor = fidelity_report(train, test, schema, prefix="fid_floor",
                            per_column=False)
    metrics.update(floor)
    # ---- diagnostic only: synthetic vs the data it was trained on -----------
    metrics.update(fidelity_report(syn, train, schema, prefix="fid_train",
                                   per_column=False))
    for key in ("spearman_gap", "ks_mean", "tv_mean"):
        f = floor.get(f"fid_floor_{key}", NAN)
        v = metrics.get(f"fid_{key}", NAN)
        metrics[f"fid_{key}_vs_floor"] = (v / f) if (np.isfinite(f) and
                                                     np.isfinite(v) and
                                                     f > 0) else NAN
    # ---- B. utility ---------------------------------------------------------
    metrics.update(utility_report(syn, train, test, schema, seed=seed))
    # ---- C. privacy ---------------------------------------------------------
    metrics.update(privacy_report(syn, train, test, schema))
    # The likelihood-gap attack needs the fitted model, not just its output, so
    # it is optional: callers that pass `model=` get a second, stronger privacy
    # signal that DP-SGD genuinely moves.
    if model is not None:
        metrics["priv_mia_auc_likelihood"] = likelihood_gap_mia(
            model, train, test, schema)
    else:
        metrics["priv_mia_auc_likelihood"] = NAN
    metrics["n_synthetic"] = float(len(syn))
    metrics["syn_pos_rate"] = (float(syn[schema.label].mean())
                               if len(syn) else NAN)
    for k in ("tstr_lr_note", "tstr_rf_note"):
        if isinstance(metrics.get(k), str) and metrics[k]:
            notes.append(f"{k}: {metrics[k]}")
    return EvalResult(model=model_name, seed=seed, metrics=metrics,
                      notes=notes, syn=syn)
