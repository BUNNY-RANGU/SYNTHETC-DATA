"""
synth - Synthetic Medical Data Generator
========================================

A teaching-oriented library for generating synthetic electronic-health-record
style tables and for measuring *fidelity*, *utility* and *privacy*.

Modules
-------
synth.data       Dataset schemas, missing-value handling, stratified splits.
synth.encoding   Mixed-type (numeric + categorical) encode/decode with clipping.
synth.models     Every generator: Naive, VAE, DP-VAE, CTGAN, TVAE, WGAN-GP.
synth.metrics    Fidelity / Utility / Privacy metric families.
synth.pipeline   Seed loop + results table assembly used by run_pipeline.py.

Quick start
-----------
    from synth.data import load_dataset
    from synth.models import VAEModel
    from synth.metrics import fidelity_report, utility_report, privacy_report

    ds = load_dataset("heart")
    model = VAEModel().fit(ds.train)
    synthetic = model.sample(1000, seed=0)
"""

__version__ = "1.0.0"

__all__ = ["data", "encoding", "models", "metrics", "pipeline", "tune"]
