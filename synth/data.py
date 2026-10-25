"""
Dataset schemas, loading, missing-value handling and stratified splitting.

WHY THIS FILE EXISTS
--------------------
Medical tables are *mixed type*: some columns are continuous numbers (age,
cholesterol), some are discrete integers (number of vessels), some are
categorical codes (chest-pain type 1-4).  Every downstream decision - which
loss function to use, how to clip a generated value, which metric is legal -
depends on knowing that structure up front.  So we declare it once, explicitly,
in a `Schema` object instead of letting it be guessed by pandas.

SECURITY / PRIVACY NOTE
-----------------------
All statistics used to clean the data (medians, modes, min/max) are computed on
the TRAINING split only.  Validation and test rows are imputed with those
training statistics.  This is deliberate: if we imputed with the full dataset
the held-out rows would leak information into the generator and the
privacy numbers would be optimistic and wrong.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Root of the project; raw files live in <root>/data
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

# Fixed seed used everywhere unless overridden.
GLOBAL_SEED = 42


# ----------------------------------------------------------------------------
# Column descriptors
# ----------------------------------------------------------------------------
@dataclass
class NumCol:
    """A continuous (or discrete-integer) column.

    lo / hi       define the *plausible medical range* used to clip generated
                  values.  They are intentionally wider than the observed range
                  so that clipping almost never fires on realistic output, but
                  always fires on nonsense output (e.g. age -3 or 400).
    observed_lo / observed_hi are filled in at load time from the TRAIN split
                  and are what min-max encoding actually uses.
    integer       if True the decoded value is rounded to a whole number.
    decimals      if not integer, number of decimal places to round to.
    """

    name: str
    lo: float
    hi: float
    integer: bool = False
    decimals: Optional[int] = None
    observed_lo: float = 0.0
    observed_hi: float = 1.0


@dataclass
class CatCol:
    """A categorical column.

    categories is the *fixed, ordered* vocabulary learned from the real data.
    Keeping it explicit means a generator can never invent a category that does
    not exist in the real cohort (e.g. thal == 5), which would be an obvious
    tell in a viva demo.
    """

    name: str
    categories: List[int] = field(default_factory=list)
    observed_lo: float = 0.0
    observed_hi: float = 0.0

    @property
    def n(self) -> int:
        return len(self.categories)


@dataclass
class Schema:
    """Full description of one tabular dataset."""

    name: str
    numeric: List[NumCol]
    categorical: List[CatCol]
    label: str
    label_name: str
    positive_is: str
    # Invertible numeric pre-transform fitted on the TRAIN split only.  Set by
    # load_dataset(); see synth.encoding.RankNumericTransform for why it exists
    # and what it makes the fidelity metrics mean.  None => plain min-max.
    num_tf: Optional[Any] = None

    # ------------------------------------------------------------------ helpers
    @property
    def numeric_names(self) -> List[str]:
        return [c.name for c in self.numeric]

    @property
    def categorical_names(self) -> List[str]:
        return [c.name for c in self.categorical]

    @property
    def feature_names(self) -> List[str]:
        """Every feature column, in a stable order: numeric then categorical."""
        return self.numeric_names + self.categorical_names

    @property
    def all_names(self) -> List[str]:
        return self.feature_names + [self.label]

    def get(self, name: str):
        """Return the column descriptor (NumCol or CatCol) for a column name."""
        for c in self.numeric:
            if c.name == name:
                return c
        for c in self.categorical:
            if c.name == name:
                return c
        if name == self.label:
            return CatCol(name=self.label, categories=[0, 1])
        raise KeyError(name)

    def is_numeric(self, name: str) -> bool:
        return name in self.numeric_names

    def clip_bounds(self, col: NumCol):
        """Effective clipping window used when decoding generated numbers.

        The declared window is the *plausible medical range*.  Real clinical
        data occasionally contains values outside it (the Pima file has a
        diastolic pressure of 24 mmHg).  Clipping generated data to a window
        that is narrower than the real data would create a fake fidelity
        artifact - a hard spike at the boundary that the real distribution does
        not have - so we widen the window to cover the observed envelope.
        Net effect: clipping always kills nonsense (age -3) but never distorts
        values that genuinely occur in the cohort.
        """
        lo = min(col.lo, col.observed_lo)
        hi = max(col.hi, col.observed_hi)
        return float(lo), float(hi)

    def refresh_observed_bounds(self, df: pd.DataFrame) -> "Schema":
        """Record the observed min/max of each column from *df*.

        Call this on the TRAIN split only.  Encoding needs these bounds.
        """
        for c in self.numeric:
            lo, hi = df[c.name].min(), df[c.name].max()
            # Guard against a degenerate constant column: keep the wider
            # medical range so the denominator below is never zero.
            if not (np.isfinite(lo) and np.isfinite(hi)) or hi - lo < 1e-8:
                lo, hi = c.lo, c.hi
            c.observed_lo, c.observed_hi = float(lo), float(hi)
        for c in self.categorical:
            c.observed_lo = 0
            c.observed_hi = c.n - 1
        return self


# ----------------------------------------------------------------------------
# Schemas for the two datasets
# ----------------------------------------------------------------------------
def heart_schema() -> Schema:
    """UCI Heart Disease - Cleveland, 303 rows x 14 columns.

    The `num` column is 0-4 (number of vessels with >50% stenosis).  Following
    the UCI data-description's own advice we collapse it to a binary
    diagnosis: 0 = no disease, 1 = disease present.
    """
    return Schema(
        name="heart",
        numeric=[
            NumCol("age", lo=18, hi=100, integer=True),
            NumCol("trestbps", lo=70, hi=250, integer=True),   # mmHg
            NumCol("chol", lo=100, hi=700, integer=True),      # mg/dl
            NumCol("thalach", lo=60, hi=230, integer=True),    # bpm
            NumCol("oldpeak", lo=0.0, hi=10.0, decimals=1),    # ST depression
        ],
        categorical=[
            CatCol("sex", [0, 1]),
            CatCol("cp", [1, 2, 3, 4]),            # chest pain type
            CatCol("fbs", [0, 1]),                 # fasting sugar > 120 mg/dl
            CatCol("restecg", [0, 1, 2]),          # ECG result
            CatCol("exang", [0, 1]),               # exercise induced angina
            CatCol("slope", [1, 2, 3]),            # ST segment slope
            CatCol("ca", [0, 1, 2, 3]),            # major vessels coloured
            CatCol("thal", [3, 6, 7]),             # thalassemia
        ],
        label="target",
        label_name="coronary artery disease",
        positive_is="num > 0 (any stenosis detected)",
    )


def pima_schema() -> Schema:
    """UCI Pima Indians Diabetes, 768 rows x 9 columns.

    The raw file is all-numeric, which would mean the categorical machinery is
    never exercised on the secondary dataset.  We therefore add ONE clinically
    motivated categorical column, `bmi_class`, using the standard WHO BMI cut
    points.  bmi itself stays numeric, so no information is lost; this is stated
    as a deliberate design choice in the report and in the limitations.
    """
    return Schema(
        name="pima",
        numeric=[
            NumCol("pregnancies", lo=0, hi=20, integer=True),
            NumCol("glucose", lo=40, hi=300, integer=True),        # OGTT mg/dl
            NumCol("blood_pressure", lo=40, hi=200, integer=True),  # mmHg
            NumCol("skin_thickness", lo=5, hi=99, integer=True),   # mm
            NumCol("insulin", lo=10, hi=1000, integer=True),        # mu U/ml
            NumCol("bmi", lo=12, hi=75, decimals=1),
            NumCol("diabetes_pedigree", lo=0.05, hi=2.5, decimals=3),
            NumCol("age", lo=21, hi=100, integer=True),
        ],
        categorical=[
            # 0=underweight(<18.5) 1=normal(18.5-24.9) 2=overweight(25-29.9)
            # 3=obese(>=30)
            CatCol("bmi_class", [0, 1, 2, 3]),
        ],
        label="target",
        label_name="diabetes onset within 5 years",
        positive_is="outcome == 1",
    )


SCHEMAS = {"heart": heart_schema, "pima": pima_schema}


# ----------------------------------------------------------------------------
# Raw loading
# ----------------------------------------------------------------------------
CLEVELAND_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

PIMA_COLUMNS = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin",
    "bmi", "diabetes_pedigree", "age", "outcome",
]


def _cleveland_path() -> str:
    """Locate (or download) the processed Cleveland file."""
    path = os.path.join(DATA_DIR, "processed.cleveland.data")
    if os.path.exists(path):
        return path
    os.makedirs(DATA_DIR, exist_ok=True)
    import urllib.request

    url = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
    zip_path = os.path.join(DATA_DIR, "heart.zip")
    urllib.request.urlretrieve(url, zip_path)
    import zipfile

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)
    return path


def _pima_path() -> str:
    """Locate (or download) the Pima file.

    The canonical UCI copy is 768 rows x 9 columns with physically impossible
    zeros marking missing values.  The Jason Brownlee mirror is byte-identical
    to that canonical file and is used as the fallback because the UCI static
    endpoint for this dataset 404s.
    """
    for name in ("pima.csv", "pima3.csv", "pima-indians-diabetes.data"):
        p = os.path.join(DATA_DIR, name)
        if os.path.exists(p):
            return p
    os.makedirs(DATA_DIR, exist_ok=True)
    import urllib.request

    url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
           "pima-indians-diabetes.data.csv")
    out = os.path.join(DATA_DIR, "pima.csv")
    urllib.request.urlretrieve(url, out)
    return out


def load_heart_raw() -> pd.DataFrame:
    """Read Cleveland and return the cleaned mixed-type DataFrame.

    Steps
    -----
    1. Read the comma separated file with no header (14 columns).
    2. Replace the UCI missing marker '?' with NaN.
    3. Coerce every column to numeric.
    4. Binarise `num` -> `target` (0/1) as the diagnosis label.
    """
    df = pd.read_csv(_cleveland_path(), header=None,
                     names=CLEVELAND_COLUMNS)
    df = df.replace("?", np.nan)
    for c in CLEVELAND_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    return df


def load_pima_raw() -> pd.DataFrame:
    """Read Pima and return the cleaned mixed-type DataFrame.

    Steps
    -----
    1. Read the 9 column file.
    2. Convert the impossible zeros (glucose/bp/skin/insulin/bmi == 0) to NaN.
       A real human never has BMI 0 or 0 mm of skin fold, so 0 is a missing
       marker in this dataset, not a value.  `pregnancies == 0` IS valid
       (nulliparous women), so it is left alone.
    3. Rename `outcome` -> `target`.
    4. Derive the WHO `bmi_class` categorical from the (still possibly NaN) bmi.
    """
    df = pd.read_csv(_pima_path(), header=None, names=PIMA_COLUMNS)
    impossible_zero = ["glucose", "blood_pressure", "skin_thickness",
                       "insulin", "bmi"]
    for c in impossible_zero:
        df[c] = df[c].replace(0, np.nan)
    df = df.rename(columns={"outcome": "target"})

    # WHO BMI cut points.  NaN bmi -> NaN class, imputed later on train stats.
    bins = [-np.inf, 18.5, 25.0, 30.0, np.inf]
    df["bmi_class"] = pd.cut(df["bmi"], bins=bins, labels=[0, 1, 2, 3, ],
                             right=False).astype("float")
    return df


RAW_LOADERS = {"heart": load_heart_raw, "pima": load_pima_raw}


# ----------------------------------------------------------------------------
# Missing values + splitting
# ----------------------------------------------------------------------------
def impute_from_train(df: pd.DataFrame, train: pd.DataFrame,
                      schema: Schema) -> pd.DataFrame:
    """Fill NaN in `df` using medians/modes taken from `train` only.

    Numeric columns -> train median.
    Categorical     -> train mode (most frequent level).

    The median is used rather than the mean because clinical measures such as
    insulin and skin thickness are heavily right skewed.
    """
    out = df.copy()
    for c in schema.numeric:
        med = float(train[c.name].median())
        out[c.name] = out[c.name].fillna(med)
    for c in schema.categorical:
        vc = train[c.name].value_counts()
        mode = int(vc.index[0]) if len(vc) else (
            c.categories[0] if c.categories else 0)
        # Snap any unexpected level back into the fixed vocabulary so the
        # one-hot encoder never sees an unseen category.
        vals = pd.to_numeric(out[c.name], errors="coerce").fillna(mode)
        arr = np.round(vals).astype(int).to_numpy()
        allowed = set(c.categories)
        arr = np.array([v if v in allowed else mode for v in arr])
        out[c.name] = arr
    out[schema.label] = out[schema.label].fillna(0).astype(int)
    return out


def stratified_split(df: pd.DataFrame, seed: int = GLOBAL_SEED,
                     train_frac: float = 0.60, val_frac: float = 0.20):
    """Split into 60/20/20 train/val/test, stratified on the label.

    Implemented as two `train_test_split` calls because sklearn has no native
    three-way split.  Stratification keeps the disease prevalence identical in
    all three parts, which matters enormously at n=303.
    """
    from sklearn.model_selection import train_test_split

    test_frac = 1.0 - train_frac - val_frac
    # First carve out the test portion from the whole set.
    rest, test = train_test_split(
        df, test_size=test_frac, random_state=seed, stratify=df["target"])
    # Then split what remains into train / val.  val_frac of the original is
    # val_frac/(train_frac+val_frac) of `rest`.
    rel_val = val_frac / (train_frac + val_frac)
    train, val = train_test_split(
        rest, test_size=rel_val, random_state=seed, stratify=rest["target"])
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


@dataclass
class DatasetSplit:
    """Bundles a schema with its three disjoint splits."""

    name: str
    schema: Schema
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    missing_counts: Dict[str, int] = field(default_factory=dict)
    transform: str = "quantile"

    @property
    def sizes(self) -> Dict[str, int]:
        return {"train": len(self.train), "val": len(self.val),
                "test": len(self.test)}


def _reorder(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Put columns in the canonical order numeric -> categorical -> label.

    Every model, metric and figure assumes this order, so normalising it once
    here removes a whole class of silent bugs.
    """
    return df[schema.all_names].copy()


def load_dataset(name: str, seed: int = GLOBAL_SEED,
                 deduplicate: bool = True,
                 transform: str = "quantile") -> DatasetSplit:
    """Load, clean and split one of the bundled datasets.

    Parameters
    ----------
    name : 'heart' | 'pima'
    seed : controls the stratified split; fixed by default so every run of the
           pipeline produces exactly the same splits (reproducibility).
    deduplicate : drop *exact duplicate whole rows* from the raw table.
           This matters for privacy metrics: if two real patients are literally
           the same row, an 'exact copy' is not a model leak.  Cleveland has no
           exact duplicates; Pima does not either in the canonical file, so this
           normally removes nothing - but we check rather than assume.

    Returns
    -------
    DatasetSplit with NaN-free, canonically ordered train/val/test frames.
    """
    name = name.lower()
    if name not in SCHEMAS:
        raise ValueError(f"unknown dataset {name!r}; choose {list(SCHEMAS)}")

    schema = SCHEMAS[name]()
    df = RAW_LOADERS[name]()

    n0 = len(df)
    if deduplicate:
        df = df.drop_duplicates(keep="first").reset_index(drop=True)
    n_dedup = n0 - len(df)

    # Record missingness BEFORE imputation so the report can state it honestly.
    missing = {c: int(df[c].isna().sum()) for c in df.columns
               if df[c].isna().any()}

    train_raw, val_raw, test_raw = stratified_split(df, seed=seed)

    # --- Step 1: clean the TRAIN split with its own statistics.
    # (Fitting the imputer on train and then applying it to train is a no-op in
    #  terms of leakage - it is just how we make `impute_from_train` the single
    #  code path, so val and test provably use the same numbers.)
    train = impute_from_train(train_raw, train_raw, schema)
    # --- Step 2: impute val/test with TRAIN statistics only (no leakage).
    val = impute_from_train(val_raw, train, schema)
    test = impute_from_train(test_raw, train, schema)

    # --- Step 3: observed scaling bounds come from the train split only.
    schema.refresh_observed_bounds(train)
    # --- Step 4: fit the invertible numeric transform on the TRAIN split only.
    # Local import breaks the data <-> encoding circular dependency.
    from .encoding import RankNumericTransform
    if transform and transform != "none":
        schema.num_tf = RankNumericTransform(schema, train, mode=transform,
                                             seed=seed)
    schema.transform_name = transform or "minmax"

    train, val, test = (_reorder(t, schema) for t in (train, val, test))

    return DatasetSplit(name=name, schema=schema, train=train, val=val,
                        test=test,
                        missing_counts=dict(missing,
                                            dropped_exact_duplicates=n_dedup),
                        transform=transform or "minmax")


def dataset_frame(ds: DatasetSplit, which: str) -> pd.DataFrame:
    """Small helper so callers can say ds.train/ds.val/ds.test by name."""
    return {"train": ds.train, "val": ds.val, "test": ds.test}[which]
