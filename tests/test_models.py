import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from synth.data import load_dataset
from synth.models import build_model, REGISTRY

ds = load_dataset("heart")

def check_valid(syn, tag):
    s = ds.schema
    assert list(syn.columns) == s.all_names, f"{tag}: col order {list(syn.columns)}"
    assert len(syn) > 0, f"{tag}: empty"
    assert not syn.isna().any().any(), f"{tag}: NaN present"
    for c in s.numeric:
        lo, hi = s.clip_bounds(c)
        v = syn[c.name]
        assert v.between(lo, hi).all(), f"{tag}: {c.name} out of [{lo},{hi}] min={v.min()} max={v.max()}"
        if c.integer:
            assert np.allclose(v % 1, 0), f"{tag}: {c.name} not integer"
    for c in s.categorical:
        assert set(syn[c.name].unique()) <= set(c.categories), f"{tag}: {c.name} illegal level {set(syn[c.name].unique())}"
    assert set(syn["target"].unique()) <= {0, 1}, f"{tag}: bad target"
    return float(syn.target.mean())

for name in ["naive", "vae", "wgan_gp", "ctgan", "tvae"]:
    t = time.time()
    hp = {}
    if name in ("vae", "wgan_gp"): hp = dict(epochs=60 if name == "vae" else 120)
    if name in ("ctgan", "tvae"): hp = dict(epochs=20)
    m = build_model(name, ds.schema, seed=0, **hp)
    m.fit(ds.train, val_df=ds.val)
    syn = m.sample(200, seed=0)
    r = check_valid(syn, name)
    print(f"{name:9s} OK fit={m.fit_seconds:6.1f}s rate={r:.3f} cols={syn.shape[1]}")
    print(syn.head(3).to_string())
print("\n=== diversity knob sanity (vae) ===")
m = build_model("vae", ds.schema, seed=0, epochs=60).fit(ds.train, val_df=ds.val)
for d in [0.0, 0.5, 1.0]:
    s1 = m.sample(300, diversity=d, seed=1)
    s2 = m.sample(300, diversity=d, seed=1)
    uniq = s1.drop_duplicates().shape[0]
    print(f"div={d:.1f} unique_rows={uniq}/300 deterministic={s1.equals(s2)} cp_rate={check_valid(s1,'v'):.3f}")
    print("   ", {c: list(np.round(s1[c].value_counts(normalize=True).head(3).values,3)) for c in ["cp","thal"]})
