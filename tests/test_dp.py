import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from synth.data import load_dataset
from synth.models import build_model

ds = load_dataset("heart")
for eps in [1.0, 3.0, 10.0, 30.0]:
    t = time.time()
    m = build_model("dp_vae", ds.schema, seed=0, epsilon=eps, epochs=60,
                    batch_size=32, max_grad_norm=1.0)
    m.fit(ds.train, val_df=ds.val)
    syn = m.sample(300, seed=0)
    print(f"eps={eps:5g} active={m._dp_active} sigma={m.noise_multiplier:7.3f} "
          f"steps={m.total_steps:4d} achieved_eps={m.achieved_epsilon:8.3f} "
          f"val={m.val_loss():.4f} fit={time.time()-t:5.1f}s rate={syn.target.mean():.3f}")
    if m.notes:
        print("   NOTES:", m.notes)
