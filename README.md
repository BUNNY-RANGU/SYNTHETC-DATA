# Synthetic Medical Data Generator

Mixed-type synthetic patient records — VAE, DP-VAE, CTGAN, TVAE, WGAN-GP —
measured on **fidelity**, **downstream utility** and **privacy**, against a
real-vs-real noise floor instead of eyeballed histograms.

B.Tech project. Every number in this file came from running the code; none of it
is illustrative. Requirement-by-requirement evidence:
[`docs/SELF_AUDIT.md`](docs/SELF_AUDIT.md).

## Run the demo

```bash
pip install -r requirements.txt
streamlit run app.py
```

Pick a generator, drag the patient-count and diversity sliders, filter by
disease class, download the cohort as CSV. Four tabs: live fidelity, live TSTR
utility, live privacy audit, and the frozen 3-seed experiment.

## Reproduce the numbers

```bash
python run_pipeline.py     # tune on val -> train every model -> 3 seeds -> both datasets
python make_figures.py     # 17 publication figures from the saved artefacts, no retraining
```

```bash
python run_pipeline.py --quick             # ~3 min smoke run instead of ~55 min
python run_pipeline.py --datasets heart    # one dataset
python run_pipeline.py --reaggregate       # rebuild all summaries from stored per-seed rows
python make_notebook.py --execute          # verify the Colab notebook runs top to bottom
python make_dist.py                        # Colab upload zip
```

Developed and benchmarked **CPU-only**. Colab: upload `synthetic_medical_data.zip`
and run [`synthetic_medical_data_generator.ipynb`](synthetic_medical_data_generator.ipynb).

## Deploy

```bash
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

That is the whole application — no build step, no database, no API keys. For a
public URL, in rough order of effort:

| Target | What you do | Notes |
|---|---|---|
| [Streamlit Community Cloud](docs/DEPLOYMENT.md#path-a--streamlit-community-cloud-recommended-free-official-4-minutes) | push to a public GitHub repo, pick `app.py` | free, ~4-minute first build |
| [Hugging Face Spaces](docs/DEPLOYMENT.md#path-b--hugging-face-spaces-free-no-credit-card-good-for-a-shareable-link) | add 10 lines of README front matter, `git push hf main` | free CPU, nice shareable link |
| [Render](docs/DEPLOYMENT.md#path-c--render-blueprint-in-the-repo) | New + → Blueprint → connect the repo | `render.yaml` is already in this repo |
| [Any container host](docs/DEPLOYMENT.md#path-d--container-anywhere-railway-flyio-a-vps-your-college-server) | `docker build -t synthmed . && docker run -p 8501:8501 synthmed` | CPU torch wheel, non-root, healthcheck |

`results/` and `data/` must be committed — the app reads them; they are inputs,
not build artefacts.

## Data and protocol

| Dataset | Rows | Numeric | Categorical | Label | Train / Val / Test |
|---|---|---|---|---|---|
| UCI Heart Disease (Cleveland) | 303 | 5 | 8 | `target` (CAD present) | 181 / 61 / 61 |
| UCI Pima Indians Diabetes | 768 | 8 | 1 | `target` (diabetes) | 460 / 154 / 154 |

Stratified 60/20/20, fixed split seed 42.

- Generators fit on **train only**; hyper-parameters chosen on **validation
  only** (`synth/tune.py` receives no test frame and has no path to one); the
  **test** split is read once, in the final evaluation pass per model per seed.
- Every cleaning, scaling, encoding and distance statistic is fitted on train and
  never re-estimated on val/test.
- Missing data: Cleveland `?` → train mode; Pima physiologically impossible zeros
  → train median. `pregnancies == 0` is valid and left alone.
- Generated rows are clipped to valid medical ranges, integers rounded to
  integers, categorical levels snapped to the legal vocabulary.

Full treatment (incl. provenance of each file): `docs/REPORT_OUTLINE.md` §4.1.

## Models

All five share one interface — `fit(train_df, val_df=...)`,
`sample(n, diversity=..., seed=...)` — so nothing downstream special-cases a model.

| Label | Model | Role |
|---|---|---|
| `naive` | independent per-column resampling | **control** — matches every 1-D marginal by construction, destroys every association |
| `vae` | encoder → (μ, log σ²) → reparameterisation → decoder; MSE numeric + per-column cross-entropy, KL annealed | the main model |
| `dp_vae_eps{1,3,10,30}` | same VAE under Opacus DP-SGD, δ = 1/n_train, RDP/PRV accountant | the privacy experiment |
| `ctgan`, `tvae` | the `ctgan` package | published reference points |
| `wgan_gp` | WGAN-GP trained in the **same encoded space** as our VAE | the extra; makes VAE-vs-GAN an architecture comparison |

## Results — Cleveland heart disease (n=303, prevalence 0.459)

Mean ± std over 3 seeds, measured against the **held-out test split**
(`results/results_summary.csv`). TRTR ceiling: LR AUC 0.9524, RF AUC 0.9518.

| Model | KS ↓ | TV ↓ | Spearman gap ↓ | TSTR AUC (LR) | TSTR AUC (RF) | Retention LR | Exact copy | DCR ratio | MIA AUC |
|---|---|---|---|---|---|---|---|---|---|
| *real-vs-real noise floor* | *0.128* | *0.040* | *0.099* | — | — | — | — | — | — |
| `naive` | 0.135±.003 | 0.040±.007 | **0.184**±.004 | 0.529±.237 | 0.525±.087 | 0.56 | 0.000 | 1.097 | 0.545 |
| **`vae`** | 0.129±.003 | 0.048±.007 | **0.117**±.008 | **0.943**±.005 | 0.879±.023 | **0.990** | 0.000 | **0.980** | 0.579 |
| `vae_dp_budget` | 0.130±.006 | 0.050±.008 | 0.119±.007 | 0.948±.004 | 0.868±.011 | 0.995 | 0.000 | 0.984 | 0.590 |
| `dp_vae` ε=1 | 0.126±.003 | **0.167**±.003 | 0.185±.000 | 0.514±.216 | 0.492±.172 | 0.540 | 0.000 | **1.610** | 0.542 |
| `dp_vae` ε=3 | 0.123±.004 | 0.089±.012 | 0.185±.001 | 0.526±.172 | 0.501±.184 | 0.553 | 0.000 | 1.409 | 0.569 |
| `dp_vae` ε=10 | 0.122±.004 | 0.040±.003 | 0.185±.002 | 0.594±.166 | 0.505±.131 | 0.624 | 0.000 | 1.182 | 0.591 |
| `dp_vae` ε=30 | 0.121±.003 | 0.044±.003 | 0.185±.001 | 0.580±.128 | 0.572±.112 | 0.610 | 0.000 | 1.124 | 0.591 |
| `ctgan` | 0.341±.075 | 0.052±.010 | 0.183±.006 | 0.674±.128 | 0.599±.058 | 0.708 | 0.000 | 1.207 | 0.595 |
| `tvae` | 0.164±.002 | 0.095±.008 | 0.156±.005 | 0.939±.006 | 0.928±.004 | 0.986 | 0.000 | **0.609** | 0.615 |
| `wgan_gp` | 0.409±.077 | 0.197±.048 | 0.393±.122 | 0.566±.203 | 0.650±.075 | 0.595 | 0.000 | 1.171 | 0.540 |

## Results — Pima diabetes (n=768, prevalence 0.348)

TRTR ceiling: LR AUC 0.8209, RF AUC 0.8194.

| Model | KS ↓ | TV ↓ | Spearman gap ↓ | TSTR AUC (LR) | Retention LR | Retention RF | DCR ratio | MIA AUC |
|---|---|---|---|---|---|---|---|---|
| *noise floor* | *0.055* | *0.010* | *0.077* | — | — | — | — | — |
| `naive` | 0.059±.004 | 0.020±.001 | **0.213**±.002 | 0.590±.045 | 0.718 | 0.693 | 1.353 | 0.502 |
| **`vae`** | 0.068±.005 | 0.024±.001 | **0.091**±.004 | 0.799±.010 | **0.973** | **0.979** | 1.053 | 0.526 |
| `dp_vae` ε=1 | 0.070±.004 | **0.121**±.004 | 0.212±.001 | 0.307±.029 | 0.374 | 0.536 | **1.775** | 0.514 |
| `dp_vae` ε=30 | 0.066±.003 | 0.021±.001 | 0.211±.001 | 0.302±.033 | 0.368 | 0.542 | 1.447 | 0.510 |
| `ctgan` | 0.247±.006 | 0.084±.005 | 0.177±.003 | 0.649±.012 | 0.790 | 0.751 | 1.830 | 0.509 |
| `tvae` | 0.102±.003 | 0.022±.001 | 0.115±.003 | **0.814**±.008 | **0.991** | 0.989 | 0.926 | 0.535 |
| `wgan_gp` | 0.428±.033 | 0.259±.044 | 0.488±.048 | 0.657±.019 | 0.801 | 0.804 | 1.578 | 0.508 |

`vae_dp_budget`, `dp_vae` ε=3 and ε=10 for Pima: `results/results_table_pima.md`.

## Acceptance criteria — measured

| Criterion | Heart | Pima |
|---|---|---|
| 1. VAE correlation gap clearly below naive | **PASS** 0.184 → 0.117 (**−36.3%**, floor 0.099) | **PASS** 0.213 → 0.091 (**−57.5%**, floor 0.077) |
| 2. Best-model TSTR AUC ≥ 90% of real-data AUC | **PASS** 99.5% (VAE@DP budget / LR) | **PASS** 99.1% (TVAE / LR), VAE 97.3% |
| 3. DP-VAE attack AUC near 0.5 **and** a visible utility trade-off | **PARTIAL** — AUC 0.542–0.591, all near chance; utility cost large and monotone. See the caveat below | **PASS** — 0.508–0.514 at every ε; TSTR AUC 0.29–0.34 vs 0.79 non-private |
| 4. Exact-copy rate | **PASS** 0.000% for all 10 models | **PASS** 0.000% |
| 5. Figures / tables / demo each run with one command | **PASS** | **PASS** |

**Caveat we are not hiding.** Distance-based MIA AUC does not fall with ε on
heart (0.542 at ε=1 → 0.591 at ε=30, the wrong direction). At ε=1 the accountant
forces σ ≈ 23 and the decoder collapses to near-constant output, so "distance to
the nearest synthetic record" stops being a membership signal — the attack is
*invalid* at that end of the sweep, not successful. The privacy evidence that
does hold is the exact-copy rate, the **DCR ratio**, and the ε guarantee itself.

## How to read these numbers

- **The naive baseline is unbeatable on marginals and useless on structure**: KS
  0.135 vs a 0.128 floor, TV exactly at the floor — while its correlation gap is
  1.85× the floor and its TSTR AUC is chance. This is why histograms are not
  evidence.
- **1-D marginal fidelity is partly by construction** (the numeric transform is
  fitted on the train marginal), so KS/TV are *not* the discriminating metrics.
  The correlation gap and TSTR are.
- **The VAE sits at 1.18× the noise floor** on the correlation gap on both
  datasets while retaining 97–99% of real-data AUC.
- **TVAE has the best utility and the worst privacy**: retention 0.986/0.975 but
  a DCR ratio of **0.609** — its rows sit closer to training records than unseen
  real patients do. Utility and leakage are not in tension; that is why both are
  measured.
- **DCR ratio is the cleanest privacy discriminator in the study**:
  0.61 (TVAE) → 0.98 (VAE) → 1.12 → 1.41 → 1.61 (ε=1).
- **DP destroys joint structure at every budget on this data size** — the DP-VAE
  gap is pinned at the naive 0.185 for all four ε, with σ = 22.97 at ε=1 vs 2.01
  at ε=30.
- **Bigger data improves the yardstick as much as the model**: the Pima floor
  (0.077) is well below the heart floor (0.099), so part of the apparent gain is
  less measurement noise, not a better generator.

## Two bugs the metrics caught

1. **Loss imbalance.** MSE on min-max numerics ≈ 0.028 vs categorical CE ≈ 0.98
   nats — a **35× mismatch**. Unweighted, the optimiser modelled only the
   categoricals, and the first VAE's correlation gap was *worse than naive*.
   Fix: a numeric loss weight, tuned on validation.
2. **Wrong sampler.** MSE is ML under fixed-variance Gaussian, so the generative
   model is `x = f(z) + σε`, not `f(z)`. Emitting the mean under-dispersed every
   numeric column (age 33–71 vs real 29–76). Reading σ off training residuals
   moved mean KS 0.200 → 0.135 on validation, and 0.129 against a 0.128 floor on
   test. Related: `oldpeak` has 28.7% exact zeros, a point mass a mean-output
   decoder cannot reproduce; the rank-to-Gaussian transform re-creates the atom.

## Limitations — the four that matter most

1. **n = 181 training rows.** The correlation-gap floor with a 61-patient test
   split is ≈ 0.10; gaps at or below it are not distinguishable from sampling
   noise. This binds the whole study.
2. **The membership-inference attack is weak** by design (one distance
   threshold, no gradients, no shadow models), and both groups come from the same
   distribution, so even a perfectly generalising model reads slightly off 0.5.
3. **GAN instability is real.** WGAN-GP on 181 rows swings between seeds and can
   collapse onto one class; those runs report NaN with a note.
4. **Synthetic ≠ anonymous.** These measurements constrain what *this generator*
   reveals; they do not defeat linkage against auxiliary data, and an ε guarantee
   is a statement about an algorithm, not about an adversary's other sources.

Also: no diffusion/flow or DP-CTGAN baseline, categorical codes treated as ordered
for Spearman, no multiple-comparison correction, disease-enriched single-site
cohort, no subgroup-fidelity measurement, the ELBO is misleading under heavy DP
noise, and DP ran with Opacus `secure_mode=False`. All 11 subsections:
`docs/REPORT_OUTLINE.md` §7. Ethics and legal context (India **DPDP Act 2023**,
**HIPAA** Safe Harbor vs Expert Determination, **GDPR** Recital 26): §8.

## Layout

```
synth/    data.py  encoding.py  models.py  metrics.py  pipeline.py  tune.py   the library
ui/       theme.py  components.py  charts.py                                  design system + plotly figures
app.py    the web app                 run_pipeline.py    one command -> every number
make_figures.py  17 paper figures     make_notebook.py   builds + executes the notebook
make_dist.py     Colab zip            tests/             stage checks + 32 AppTest scenarios
.streamlit/config.toml  theme        Dockerfile  render.yaml  .github/workflows/ci.yml
results/  CSVs, per-model synthetic tables, figures/, acceptance verdicts   (inputs to the app)
docs/     DEPLOYMENT.md  REPORT_OUTLINE.md  SLIDES_OUTLINE.md  VIVA_QUESTIONS.md  SELF_AUDIT.md
```

## Reproducibility

The full sweep (10 configurations × 3 seeds × 2 datasets = 60 fits, 3257 s on
CPU, torch 2.10.0+cpu, Python 3.11.9) was run twice against the final code: all
60 per-seed rows matched and **every metric column was bit-identical** — only
`wall_seconds` differed. Seeds are pinned for Python, NumPy and PyTorch in
`synth.models.set_seed`, and every sampler takes an explicit
`np.random.default_rng(seed)`.
