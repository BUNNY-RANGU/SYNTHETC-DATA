import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from synth.data import load_dataset
from synth.models import build_model
from synth.metrics import (evaluate_model, ks_per_column, tv_per_column,
                           spearman_gap, pca_overlay_data, privacy_report)

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 60)
ds = load_dataset("heart")

# --- sanity: the noise floor itself (real train vs real test) ---
floor = spearman_gap(ds.train, ds.test, ds.schema)
print("real-train vs real-test spearman gap (noise floor): %.4f" % floor["spearman_gap"])
fr = evaluate_model("REALREF", ds.train, ds.train, ds.val, ds.test, ds.schema, 0)
print("floor ks_mean=%.4f tv_mean=%.4f" % (fr.metrics["fid_floor_ks_mean"], fr.metrics["fid_floor_tv_mean"]))

rows = []
for name, hp in [("naive", {}), ("vae", dict(epochs=400)),
                 ("ctgan", dict(epochs=200)), ("tvae", dict(epochs=200)),
                 ("wgan_gp", dict(epochs=1500))]:
    t = time.time()
    m = build_model(name, ds.schema, seed=0, **hp).fit(ds.train, val_df=ds.val)
    syn = m.sample(1000, seed=0)
    r = evaluate_model(name, syn, ds.train, ds.val, ds.test, ds.schema, 0)
    k = r.metrics
    rows.append(dict(model=name,
                     sgap=k["fid_spearman_gap"], sgap_vs_floor=k["fid_spearman_gap_vs_floor"],
                     ks=k["fid_ks_mean"], tv=k["fid_tv_mean"],
                     tstr_lr_auc=k["tstr_lr_auc"], tstr_rf_auc=k["tstr_rf_auc"],
                     ret_lr=k["retention_lr"], ret_rf=k["retention_rf"],
                     copy=k["priv_exact_copy_train"], dcr_ratio=k["priv_dcr_ratio"],
                     mia=k["priv_mia_auc"], fit=round(time.time()-t,1)))
print("\nTRTR auc lr/rf: %.4f / %.4f" % (fr.metrics["trtr_lr_auc"] if "trtr_lr_auc" in fr.metrics else 0, 0))
print(pd.DataFrame(rows).to_string(index=False))
