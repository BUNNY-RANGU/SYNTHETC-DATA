import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from synth.data import load_dataset, SCHEMAS

pd.set_option("display.width", 200)
for name in SCHEMAS:
    ds = load_dataset(name)
    print("=" * 78)
    print(name, ds.sizes, "missing:", ds.missing_counts)
    print("cols:", list(ds.train.columns))
    print("label rate train/val/test: %.3f %.3f %.3f" % (
        ds.train.target.mean(), ds.val.target.mean(), ds.test.target.mean()))
    print("numeric observed bounds:",
          [(c.name, c.observed_lo, c.observed_hi) for c in ds.schema.numeric])
    print("NaN left?", {c: int(ds.train[c].isna().sum()) for c in ds.schema.all_names if ds.train[c].isna().any()})
    print("cat levels ok?",
          all(set(ds.train[c.name].unique()) <= set(c.categories) for c in ds.schema.categorical))
    print(ds.train.describe().loc[["min", "max"]].to_string())
