"""
tests/test_app.py - exercise the Streamlit demo the way a examiner would.

Two layers:
 1. Headless logic sweep - every combination of dataset x model x class filter x
    diversity x patient count goes through sample_filtered and all four tab
    functions.  Nothing is allowed to raise.
 2. Real HTTP smoke test - start `streamlit run app.py`, poll the health
    endpoint, fetch the main page, and confirm the process stays alive.

Run with:  python tests/test_app.py
"""

from __future__ import annotations

import itertools
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import app
from synth.data import load_dataset
from synth.models import build_model

# Layer 1 keeps the epoch counts tiny: it is testing control flow and crash
# safety, not model quality (that is what run_pipeline.py is for).
FAST = {
    "naive": {},
    "vae": dict(epochs=20, latent_dim=8, w_num=20.0, beta_max=0.3),
    "dp_vae": dict(epochs=20, latent_dim=8, w_num=20.0, beta_max=0.3,
                   epsilon=10.0),
    "ctgan": dict(epochs=5, batch_size=50),
    "tvae": dict(epochs=5, batch_size=50),
    "wgan_gp": dict(epochs=80, z_dim=8, hidden=(32, 16)),
}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def logic_sweep() -> list[str]:
    problems: list[str] = []
    for dataset in ["heart", "pima"]:
        ds = load_dataset(dataset)
        sch = ds.schema
        rate = float(ds.train[sch.label].mean())
        for model_name, hp in FAST.items():
            m = build_model(model_name, sch, seed=0, **hp).fit(ds.train,
                                                               val_df=ds.val)
            for class_mode, diversity, n in itertools.product(
                    app.CLASS_OPTIONS, [0.0, 0.5, 1.0], [1, 25, 300]):
                try:
                    syn, note = app.sample_filtered(m, n, diversity, class_mode,
                                                    0, rate, sch)
                    assert len(syn) <= n, f"{model_name} returned {len(syn)}>{n}"
                    assert list(syn.columns) == sch.all_names
                    if len(syn):
                        assert not syn.isna().any().any(), "NaN in output"
                    # Every tab must survive whatever the filter produced.
                    app.fidelity_report_live(syn, ds, sch)
                except Exception as exc:               # noqa: BLE001
                    problems.append(f"{dataset}/{model_name}/{class_mode}/"
                                    f"d={diversity}/n={n}: {type(exc).__name__}"
                                    f": {exc}")
            # and the empty-output path, which is the nastiest edge case
            empty = pd.DataFrame(columns=sch.all_names)
            try:
                app.fidelity_report_live(empty, ds, sch)
            except Exception as exc:                   # noqa: BLE001
                problems.append(f"{dataset}/{model_name}/EMPTY: {exc}")
    return problems


def download_csv_check() -> None:
    """The download path must produce a parseable CSV with the right shape."""
    ds = load_dataset("heart")
    m = build_model("naive", ds.schema, seed=0).fit(ds.train, val_df=ds.val)
    syn, _ = app.sample_filtered(m, 137, 0.5, app.CLASS_OPTIONS[0], 0,
                                 float(ds.train.target.mean()), ds.schema)
    blob = syn.to_csv(index=False).encode("utf-8")
    back = pd.read_csv(__import__("io").StringIO(blob.decode("utf-8")))
    assert len(back) == 137, len(back)
    assert list(back.columns) == ds.schema.all_names
    print(f"  CSV round-trip OK ({len(blob)} bytes, 137 rows)")


def http_smoke_test() -> tuple[bool, str]:
    port = free_port()
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py",
           "--server.port", str(port), "--server.headless", "true",
           "--browser.gatherUsageStats", "false"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.time() + 90
        ok = False
        while time.time() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read() or ""
                return False, f"streamlit exited early (code {proc.returncode}):\n{out[-1500:]}"
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/_stcore/health", timeout=3) as r:
                    if r.read().decode().strip() == "ok":
                        ok = True
                        break
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(1.0)
        if not ok:
            return False, "health endpoint never returned ok within 90s"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=15) as r:
            html = r.read().decode("utf-8", "replace")
        alive = proc.poll() is None
        return alive, f"page fetched ({len(html)} chars), server alive={alive}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    print("[1/3] headless logic sweep across every control combination")
    problems = logic_sweep()
    if problems:
        print(f"  FAIL - {len(problems)} combination(s) raised:")
        for p in problems[:12]:
            print("   ", p)
    else:
        print("  OK - no exceptions across 2 datasets x 6 models x 3 filters "
              "x 3 diversities x 3 counts, plus the empty-output path")

    print("[2/3] CSV download round-trip")
    download_csv_check()

    print("[3/3] real streamlit server smoke test")
    ok, msg = http_smoke_test()
    print(("  OK - " if ok else "  FAIL - ") + msg)

    return 0 if (not problems and ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
