"""
make_dist.py - build the zip that the Colab notebook unpacks.

    python make_dist.py        -> synthetic_medical_data.zip in the project root

The zip contains the `synth/` library, the two raw datasets and the four entry
point scripts, so a Colab session can reproduce everything without a git repo.
It deliberately excludes results/ and figures/ (those are outputs, not inputs)
and excludes anything under a dot-directory.
"""

from __future__ import annotations

import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "synthetic_medical_data.zip")

INCLUDE = [
    "synth",
    "ui",
    "data",
    "run_pipeline.py",
    "make_figures.py",
    "make_notebook.py",
    "app.py",
    ".streamlit",
    "requirements.txt",
    "README.md",
    "docs",
    "synthetic_medical_data_generator.ipynb",
]
SKIP_DIR_NAMES = {"__pycache__", ".ipynb_checkpoints", ".git"}
SKIP_SUFFIXES = (".pyc", ".zip", ".pt", ".pth")
# The UCI bundle has several unused regional .data files; only the two we use
# are worth shipping.
DATA_ALLOW = {"processed.cleveland.data", "pima.csv", "pima3.csv"}


def walk():
    for name in INCLUDE:
        p = os.path.join(HERE, name)
        if not os.path.exists(p):
            print(f"  [skip] missing {name}")
            continue
        if os.path.isfile(p):
            yield name, p
            continue
        for root, dirs, files in os.walk(p):
            dirs[:] = [d for d in dirs if d not in SKIP_DIR_NAMES]
            for f in sorted(files):
                if f.endswith(SKIP_SUFFIXES):
                    continue
                full = os.path.join(root, f)
                rel = os.path.relpath(full, HERE).replace(os.sep, "/")
                if rel.startswith("data/") and os.path.basename(rel) \
                        not in DATA_ALLOW and not rel.endswith(".names"):
                    continue
                yield rel, full


def main() -> int:
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        n = 0
        for rel, full in walk():
            z.write(full, rel)
            n += 1
    size = os.path.getsize(OUT)
    print(f"wrote {OUT}  ({n} files, {size / 1024:.0f} KB)")
    print("upload this file in Colab (files panel) before running cell 3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
