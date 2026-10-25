"""
Mixed-type encoding / decoding for tabular generators.

WHY MODE-BASED (one-hot) ENCODING AND NOT LABEL ENCODING
--------------------------------------------------------
Neural generators work on continuous vectors.  The two standard choices are

  * integer / label encoding  - cheap, but it invents an ordering that does not
    exist.  `thal` takes values {3, 6, 7} meaning normal / fixed defect /
    reversible defect; 6 is not "twice as normal" as 3.
  * mode-based (one-hot)      - expands a k-level column into k binary slots.
    No false ordering.  This is the encoding CTGAN uses and is the accepted
    default for mixed-type tabular generation.

We therefore use one-hot for categoricals and min-max for numerics, and we
attach a *purpose-built output head* to each group so that the decoder can
optimise cross-entropy for the categorical slots and MSE for the numeric ones -
exactly the split the project specification asks for.

The label column (`target`) is treated as a 2-level categorical group, so the
generator produces the diagnosis along with the symptoms.  That is what lets
the Streamlit app offer class-conditional sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .data import CatCol, NumCol, Schema

EPS = 1e-8


class RankNumericTransform:
    """Per-column invertible map  raw values <-> [0, 1], fitted on TRAIN only.

    WHY THIS EXISTS (and why plain min-max is not good enough)
    ----------------------------------------------------------
    Clinical numerics are not uniformly spread.  Two concrete pathologies in
    the Cleveland table:

      * `chol` has a long right tail (126-564 with most values near 240).  Under
        min-max scaling everything sits in [0, 0.3] and the decoder spends 70%
        of its output range on empty space.  Measured effect: KS = 0.28 and the
        synthetic tail was truncated at 479 instead of 564.
      * `oldpeak` has 28.7% of rows at *exactly* 0.0 ("no ST depression").  That
        is a point mass.  A decoder trained on MSE outputs a conditional mean,
        which is a smooth density with no atoms, and it can never reproduce a
        spike.  Measured effect: KS = 0.43 and the synthetic zero-fraction
        collapsed from 0.287 to 0.001.

    A rank-to-Gaussian (quantile) transform fixes both, because the map is
    monotone: it spreads each column over the full [0, 1] range AND it has flat
    inverse regions, so a whole band of decoder outputs snaps back to exactly
    0.0 and re-creates the atom.  This is the same family of preprocessing that
    CTGAN's log-normal transformer applies.

    THE HONESTY CAVEAT, which belongs in the report not just here
    -------------------------------------------------------------
    Because this transform is fit on the training marginal, 1-D marginal
    fidelity becomes partly guaranteed by construction: every neural model ends
    up with a KS statistic near the real-vs-real noise floor.  So KS/TV are NOT
    the metric that distinguishes these models.  The metrics that do are the
    joint ones - Spearman correlation gap and TSTR utility.  (The naive
    baseline also matches the marginals exactly by construction, so the
    comparison with it remains apples-to-apples.)

    Set mode='minmax' to switch it off; `make_figures.py` plots both as an
    ablation so the effect is visible rather than asserted.
    """

    def __init__(self, schema: Schema, df: pd.DataFrame, mode: str = "quantile",
                 seed: int = 0):
        self.schema = schema
        self.mode = mode
        self.cols = schema.numeric_names
        self._seed = seed
        if mode == "minmax" or not self.cols:
            self.qt = None
            return
        from sklearn.preprocessing import QuantileTransformer

        X = df[self.cols].to_numpy(dtype=float)
        n = len(X)
        # n_quantiles cannot exceed the number of rows.
        nq = int(max(10, min(1000, n)))
        self.qt = QuantileTransformer(
            n_quantiles=nq, output_distribution="normal",
            subsample=max(1000, n), random_state=seed)
        self.qt.fit(X)
        Z = self.qt.transform(X)
        # After the normal map, rescale column-wise into [0, 1] using the
        # observed transformed envelope, clipped to +/-3 sigma so a handful of
        # extreme ranks do not stretch the whole range.
        Zc = np.clip(Z, -3.0, 3.0)
        self.lo = Zc.min(axis=0)
        self.hi = Zc.max(axis=0)
        bad = (self.hi - self.lo) < 1e-6          # constant column edge case
        self.lo[bad] = -1.0
        self.hi[bad] = 1.0

    def forward(self, X: np.ndarray) -> np.ndarray:
        """raw -> [0, 1]"""
        if self.qt is None:
            return X
        Z = np.clip(self.qt.transform(X), -3.0, 3.0)
        return np.clip((Z - self.lo) / (self.hi - self.lo), 0.0, 1.0)

    def inverse(self, U: np.ndarray) -> np.ndarray:
        """[0, 1] -> raw"""
        if self.qt is None:
            return U
        Z = np.asarray(U, dtype=float) * (self.hi - self.lo) + self.lo
        return self.qt.inverse_transform(Z)


@dataclass
class Slot:
    """One contiguous block of the encoded vector belonging to one column."""

    name: str
    kind: str           # 'numeric' | 'categorical'
    start: int          # first index in the vector
    width: int          # 1 for numeric, k for categorical
    categories: Optional[List[int]] = None

    @property
    def stop(self) -> int:
        return self.start + self.width


def coerce_dtypes(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Give every column the dtype a real clinical table would have.

    The generators work in float space end to end, so an integer column arrives
    as float64 holding 167.0.  The value is correct, but the *presentation* is
    not: the Streamlit data grid renders it as `166.99999999999997` (a
    float64->JS round-trip artifact), and the downloaded CSV says `167.0` for a
    heart rate.  Both make a correct record look broken to a reviewer, so the
    cast happens here, after clipping and rounding, and never changes a value
    that is not already integral.
    """
    out = df.copy()
    for c in schema.numeric:
        if c.integer:
            out[c.name] = out[c.name].to_numpy(dtype=float).round().astype("int64")
        elif c.decimals is not None:
            out[c.name] = out[c.name].to_numpy(dtype=float).round(c.decimals)
    for c in schema.categorical:
        out[c.name] = out[c.name].to_numpy(dtype=float).round().astype("int64")
    out[schema.label] = out[schema.label].to_numpy(dtype=float).round().astype("int64")
    return out[schema.all_names]


class TabularEncoder:
    """Bidirectional DataFrame <-> float vector mapping for one dataset.

    Fit-on-train: the min-max bounds are copied from the Schema, which the
    loader populated from the TRAIN split only.  No dataset statistics are
    re-estimated here.
    """

    def __init__(self, schema: Schema):
        self.schema = schema
        self.slots: List[Slot] = []
        cursor = 0
        # Order matters and must match Schema.all_names exactly.
        for c in schema.numeric:
            self.slots.append(Slot(c.name, "numeric", cursor, 1))
            cursor += 1
        for c in schema.categorical:
            self.slots.append(
                Slot(c.name, "categorical", cursor, max(c.n, 1), c.categories))
            cursor += max(c.n, 1)
        # The diagnosis label is the final categorical group.
        self.label_slot = Slot(schema.label, "categorical", cursor, 2, [0, 1])
        self.slots.append(self.label_slot)
        cursor += 2
        self.dim = cursor

    def unscale_numeric(self, U: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`scale_numeric` - [0,1] back to raw clinical units."""
        names = self.schema.numeric_names
        U = np.clip(np.asarray(U, dtype=float), 0.0, 1.0)
        tf = self.schema.num_tf
        if tf is not None:
            return tf.inverse(U)
        lo = np.array([self._bounds(c)[0] for c in names])
        hi = np.array([self._bounds(c)[1] for c in names])
        return U * (hi - lo) + lo

    # ------------------------------------------------------------------ encode
    def scale_numeric(self, raw: np.ndarray) -> np.ndarray:
        """Raw clinical values -> [0, 1], using the *train-fitted* transform.

        Exposed as its own method so any model that works in raw units (the
        naive resampler) can hand us numbers on exactly the same scale as the
        neural models.  Getting this wrong silently double-transforms a column
        and wrecks its marginal - which is precisely the bug this method
        exists to prevent.
        """
        names = self.schema.numeric_names
        raw = np.asarray(raw, dtype=float)
        if raw.shape[1] != len(names):
            raise ValueError(f"expected {len(names)} numeric columns, got "
                             f"{raw.shape[1]}")
        tf = self.schema.num_tf
        if tf is not None:
            return tf.forward(raw)
        lo = np.array([self._bounds(c)[0] for c in names])
        hi = np.array([self._bounds(c)[1] for c in names])
        return np.clip((raw - lo) / (hi - lo + EPS), 0.0, 1.0)

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        """DataFrame -> (n, dim) float32 array with every entry in [0, 1]."""
        n = len(df)
        out = np.zeros((n, self.dim), dtype=np.float32)
        num_names = self.schema.numeric_names
        if num_names:
            U = self.scale_numeric(df[num_names].to_numpy(dtype=float))
            for i, slot in enumerate(
                    [s for s in self.slots if s.kind == "numeric"]):
                out[:, slot.start] = U[:, i]
        for slot in [s for s in self.slots if s.kind == "categorical"]:
            col = df[slot.name].to_numpy(dtype=float)
            for i, cval in enumerate(slot.categories):
                onehot = (np.round(col).astype(int) == int(cval))
                out[:, slot.start + i] = onehot.astype(np.float32)
        return out

    def decode_logits(self, numeric: np.ndarray,
                      cat_probs: List[np.ndarray],
                      temperature: float = 1.0,
                      rng: Optional[np.random.Generator] = None,
                      ) -> pd.DataFrame:
        """Build a real-valued DataFrame from raw network outputs.

        Parameters
        ----------
        numeric   : (n, n_numeric) values, typically already squashed to [0,1].
        cat_probs : one (n, k) array per categorical slot, in slot order,
                    including the label group last.  Probabilities need not be
                    normalised - we softmax them here so the same routine serves
                    a VAE (softmax outputs) and a GAN (arbitrary logits).
        temperature: >1 flattens the categorical distribution (more diversity,
                    more noise); <1 sharpens it toward the model's top choice.
                    This single knob is what drives the Streamlit 'diversity'
                    slider, so the demo and the experiments use one mechanism.

        Sampling vs argmax
        ------------------
        We *sample* the categorical slots instead of taking argmax.  Argmax
        collapses a predicted distribution such as (0.55 male, 0.45 female) into
        'male' for every single synthetic patient, which would silently destroy
        the marginal distribution we are trying to preserve.
        """
        rng = rng or np.random.default_rng(0)
        cat_slots = [s for s in self.slots if s.kind == "categorical"]
        if len(cat_probs) != len(cat_slots):
            raise ValueError(
                f"expected {len(cat_slots)} categorical blocks, got "
                f"{len(cat_probs)}")

        data: dict = {}
        num_names = self.schema.numeric_names
        if num_names:
            # Back to raw clinical units.  The rank transform inverts itself
            # through its flat regions, which is what re-creates point masses
            # like oldpeak == 0.0; plain min-max just rescales.
            raw = self.unscale_numeric(np.asarray(numeric, dtype=float))
            for i, name in enumerate(num_names):
                data[name] = self._postprocess_numeric(raw[:, i], name)

        for slot, probs in zip(cat_slots, cat_probs):
            p = self._softmax(np.asarray(probs, dtype=float), temperature)
            idx = self._sample_rows(p, rng)
            cats = slot.categories
            # Defensive clamp: a caller that hands us a wider probability block
            # than the fixed vocabulary (e.g. phantom bins) must not be able to
            # index past the end and invent a category that does not exist.
            idx = np.clip(idx, 0, len(cats) - 1)
            data[slot.name] = np.array([cats[j] for j in idx], dtype=float)

        df = pd.DataFrame(data)
        return coerce_dtypes(df[self.schema.all_names], self.schema)

    # ----------------------------------------------------------------- helpers
    def _bounds(self, name: str) -> Tuple[float, float]:
        """Encoding window for a numeric column (train-observed bounds)."""
        for c in self.schema.numeric:
            if c.name == name:
                return c.observed_lo, c.observed_hi
        raise KeyError(name)

    def _postprocess_numeric(self, raw: np.ndarray, name: str) -> np.ndarray:
        """Clip to a valid medical range and round to the right type.

        This is the "clip and round generated values to valid medical ranges
        and integer types" requirement, in one place.
        """
        col = self.schema.get(name)
        lo_clip, hi_clip = self.schema.clip_bounds(col)
        v = np.clip(np.asarray(raw, dtype=float), lo_clip, hi_clip)
        if isinstance(col, NumCol) and col.integer:
            v = np.round(v).clip(np.ceil(lo_clip), np.floor(hi_clip))
        elif isinstance(col, NumCol) and col.decimals is not None:
            v = np.round(v, col.decimals)
        return v

    @staticmethod
    def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Numerically stable softmax with an optional temperature."""
        t = max(float(temperature), 1e-3)          # never divide by ~0
        z = np.asarray(logits, dtype=float) / t
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(np.clip(z, -60, 60))            # clip guards under/overflow
        s = e.sum(axis=1, keepdims=True)
        # A row summing to 0 (all -inf inputs) would give NaN; fall back to
        # the uniform distribution rather than poisoning the whole batch.
        bad = (s[:, 0] <= 0) | ~np.isfinite(s[:, 0])
        if bad.any():
            e[bad] = 1.0 / e.shape[1]
            s = e.sum(axis=1, keepdims=True)
        return e / s

    @staticmethod
    def _sample_rows(probs: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
        """Draw one category index per row from a (n, k) probability matrix."""
        n, k = probs.shape
        cum = np.cumsum(probs, axis=1)
        cum[:, -1] = 1.0                            # kill float drift
        u = rng.random((n, 1))
        return (u > cum).sum(axis=1).clip(0, k - 1)


def make_target_vector(df: pd.DataFrame, label: str) -> np.ndarray:
    """Extract the 0/1 diagnosis column as an int array (for TSTR)."""
    return df[label].to_numpy(dtype=int)
