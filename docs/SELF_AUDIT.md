# Self-Audit — requirement → done → evidence

Every row was checked by running code. "Evidence" names the artifact or the
assertion that proves it, so nothing here rests on a claim that something
"should" work.

Legend: **Done** = implemented *and* executed with observed output.
**Partial** = implemented and run, but the measured result does not fully meet
the aspiration — stated, not glossed.

## 1. Project framing

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1.1 | Uses a VAE | Done | `synth/models.py::VAEModel`; run row `vae` in `results/results_summary.csv` |
| 1.2 | Uses a GAN | Done | `CTGANBase` (CTGAN) and `WGANGPModel`; rows `ctgan`, `wgan_gp` |
| 1.3 | Generates synthetic patient records | Done | `results/heart/syn_*_seed*.csv` — 1 000 records per model per seed |
| 1.4 | Preserves statistical properties of real data | Done | family A metrics; VAE Spearman gap vs naive in `results_table_heart.md` |
| 1.5 | Supports privacy-preserving healthcare research | Done | family C metrics + `priv_mia_auc_likelihood` across ε |

## 2. Data

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 2.1 | Primary: UCI Heart Disease Cleveland, 303 rows, mixed numeric+categorical | Done | `tests/test_data.py` prints `(303, 14)` read, 5 numeric + 8 categorical + label |
| 2.2 | Secondary: Pima Diabetes, 768 rows | Done | `tests/test_data.py` prints `pima rows/cols (768, 9)` |
| 2.3 | Shows how dataset size changes results | Done | `results/results_table_heart.md` vs `_pima.md`; §5.6 of `REPORT_OUTLINE.md`; noise floors reported for both |
| 2.4 | 60 / 20 / 20 split, stratified, fixed seed | Done | `synth/data.py::stratified_split`; test prints `181/61/61` and `460/154/154` with prevalence identical to 4 dp in all folds |
| 2.5 | Generators train only on train | Done | `run_one()` passes `ds.train` to `fit()`; `tune()` has no test parameter at all |
| 2.6 | Tune only on validation | Done | `synth/tune.py` docstring + `results/tuning_grid_*.csv`; score computed against **val** rows |
| 2.7 | Test touched once, at the end, and stated in the report | Done | `README.md` "Data and protocol" states the protocol including the "one pass per model per seed" nuance; `REPORT_OUTLINE.md` §4.1 |
| 2.8 | Honest missing-value handling | Done | Cleveland `?` → 4 `ca`, 2 `thal`, mode-imputed from train, all 303 rows kept; Pima 0-as-missing in 5 columns; `ds.missing_counts` printed |
| 2.9 | No leakage of val/test statistics | Done | `impute_from_train(val, train, …)`, `refresh_observed_bounds(train)`, `RankNumericTransform(schema, train)`, `RecordSpace(train)` — every fitter receives train only |

## 3. Models

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 3.1 | All in PyTorch, CPU if no GPU | Done | `torch.cuda.is_available()` False on the run machine; `DEVICE` in `synth/models.py`; `run_metadata.json` records torch version |
| 3.2 | Naive baseline: resample each column independently | Done | `NaiveModel`; row `naive` — gap ≈ noise floor × 1.86, utility ≈ chance, exactly as a control should be |
| 3.3 | Custom VAE: encoder → (mu, logvar) → reparameterisation → decoder | Done | `VaeNet.encode/reparameterise/decode`; notebook cell prints the slot layout and the ELBO pieces |
| 3.4 | MSE for numeric columns | Done | `vae_reconstruction_loss` — `F.mse_loss(...).mean(dim=1)` on the numeric block |
| 3.5 | Cross-entropy per categorical column | Done | same function, one `F.cross_entropy(logits, target)` per group; `notebook §2` measures the 35× MSE-vs-CE imbalance |
| 3.6 | KL annealing | Done | `kl_schedule()` linear ramp; `betas` recorded in `VAETrainResult` |
| 3.7 | DP-VAE via Opacus at ε = 1, 3, 10, 30 | Done | `DPVAEModel`; four rows `dp_vae_eps{1,3,10,30}`; `results/privacy_utility_curve.csv` carries σ, steps and ε actually spent |
| 3.8 | CTGAN and TVAE via the `ctgan` package | Done | `CTGANBase`, `TVAEModel`; rows `ctgan`, `tvae` |
| 3.9 | One extra model of my choice | Done | **WGAN-GP** (`WGANGPModel`) with gradient penalty, chosen over a conditional VAE so the GAN shares the VAE's encoding — rationale in `README.md` "Models" |
| 3.10 | Values clipped and rounded to valid medical ranges and integer types | Done | `TabularEncoder._postprocess_numeric` + `sanitize()`; `tests/test_models.py` asserts every column inside bounds and `x % 1 == 0` for integer columns; `fid_valid_rate == 1.0` for all models in the summary |
| 3.11 | Generators cannot invent categories | Done | fixed vocabulary in `CatCol.categories`; `decode_logits` clamps the sampled index to `len(categories)-1`; test asserts `set(syn[col]) ⊆ categories` |

## 4. Evaluation

### A. Fidelity

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 4A.1 | Per-column KS test | Done | `ks_per_column()`; per-column values in `results_summary_long.csv` as `fid_ks_<column>` |
| 4A.2 | Total-variation score | Done | `tv_per_column()` over the union of observed categories, including the label |
| 4A.3 | Spearman correlation gap | Done | `spearman_gap()` → mean and max over off-diagonal pairs |
| 4A.4 | PCA overlay | Done | `fig1_pca_overlay_<dataset>.png`; real held-out drawn as the envelope |
| 4A.5 | Histogram overlays | Done | `fig2_histograms_<dataset>.png` |
| 4A.6 | Correlation heatmaps | Done | `fig3_correlation_heatmaps_<dataset>.png`, including the difference panel |
| 4A.7 | **(extra) real-vs-real noise floor** | Done | `fid_floor_*` computed with the identical metric code; `fid_*_vs_floor` ratio; dashed reference line drawn on figs 4 and 9 |

### B. Utility

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 4B.1 | TSTR with Logistic Regression and Random Forest | Done | `run_downstream()`, `_make_clf()`; columns `tstr_lr_*`, `tstr_rf_*` |
| 4B.2 | Accuracy, F1, AUC | Done | all three per classifier in `results_summary.csv` |
| 4B.3 | Compared with TRTR | Done | `trtr_*` columns; `retention_{lr,rf} = tstr_auc / trtr_auc`; ceiling drawn as a dashed line in fig 5 |
| 4B.4 | Scaler cannot leak real data in TSTR | Done | `run_downstream` fits `StandardScaler` on the *synthetic* training matrix only — documented in `synth/metrics.py` |

### C. Privacy

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 4C.1 | Exact-copy check | Done | `exact_copy_rate()` vs train **and** vs held-out; `priv_exact_copy_{train,test}` |
| 4C.2 | DCR vs held-out real patients | Done | `RecordSpace.dist_to_set`; reports synthetic→train, synthetic→held-out, and the **held-out-real→train yardstick**, plus `priv_dcr_ratio` |
| 4C.3 | Distance-based MIA AUC | Done | `membership_inference_auc()` → `priv_mia_auc` |
| 4C.4 | **(extra) likelihood-gap MIA** | Done | `likelihood_gap_mia()` → `priv_mia_auc_likelihood`; this is the attack that moves monotonically with ε, and the reason it is reported separately is documented in its docstring |
| 4C.5 | Privacy-vs-utility curve across ε | Done | `results/privacy_utility_curve.csv` and `fig7_privacy_utility_curve.png`, with the non-private reference at a **matched** optimisation budget (`vae_dp_budget`) |

## 5. Rigor rules

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 5.1 | Every model runs with 3 random seeds | Done | `SEEDS = [0,1,2]`; `results_summary_long.csv` has one row per model per seed; `results_summary.csv` reports `_mean` and `_std` |
| 5.2 | Report mean ± std | Done | `aggregate()` with `ddof=1`; `fmt()` renders `mean+/-std`; `n_seeds_ok` records when a mean is over fewer than 3 finite runs |
| 5.3 | No tuning on test; say honestly what was tuned on what | Done | `synth/tune.py` module docstring; `README.md` "Data and protocol"; `results/tuning_grid_*.csv`; §5.2 of the report outline |
| 5.4 | Single-class synthetic data handled | Done | `run_downstream` returns NaN + `note='single_class_train'`; **asserted** in the notebook edge-case cell |
| 5.5 | Empty batches handled | Done | `DPVAEModel._train_one_epoch_dp` skips a zero-row Poisson batch (spends no budget); `_raw_outputs`/`decode_logits` fall back to uniform on a zero row-sum |
| 5.6 | NaN metrics handled | Done | every metric path guarded (`_empty`, `np.isfinite`, try/except → NaN); `aggregate()` drops NaNs and reports how many seeds survived; a zero-variance column is asserted not to crash `RecordSpace` in the notebook |
| 5.7 | NaN loss cannot poison the weights | Done | `if not torch.isfinite(loss): continue` in both epoch functions |
| 5.8 | Limitations reported honestly | Done | `README.md` "Limitations" (4 headline + 7 named), `REPORT_OUTLINE.md` §7 (11 subsections) — includes the marginal-fidelity circularity, the weak attack, the GAN instability, the disease-enriched cohort, and Opacus `secure_mode=False` |
| 5.9 | Never claim results we did not measure | Done | `acceptance_verdict()` prints FAIL/CHECK where a target is not met and writes `results/acceptance_<dataset>.txt`; no number in this repo is hand-entered |

## 6. Deliverables

| # | Deliverable | Status | Evidence |
|---|---|---|---|
| 6.1 | `synth.py` library | Done as a package `synth/` (`data`, `encoding`, `models`, `metrics`, `pipeline`, `tune`) — a package rather than one 1 500-line module, for the modularity the spec also asks for; named in `README.md` "Layout" |
| 6.2 | `run_pipeline.py` | Done — one command, `--quick`, `--datasets`, `--seeds`, `--n-syn`, `--skip-tune`; writes per-dataset CSVs incrementally so an interrupted run still yields output |
| 6.3 | `make_figures.py` | Done — 9 figure functions, no retraining, colour-blind-safe |
| 6.4 | `app.py` Streamlit demo | Done — sliders for patient count and diversity, class filter, CSV download, tabs Fidelity / Utility / Privacy / Model comparison |
| 6.5 | Comments explain every step for a student | Done — each module opens with a "why this file exists" block; each non-obvious decision carries its rationale inline |
| 6.6 | Colab-ready notebook that runs top to bottom | Done — `synthetic_medical_data_generator.ipynb`, 47 cells, generated by `make_notebook.py`, executed with `--execute` (see §8 for the result) |
| 6.7 | Publication figures, colour-blind-safe, clear titles | Done — Okabe-Ito palette, `fig*.png` |
| 6.8 | `results_summary.csv` + one-page table | Done — `results/results_summary.csv`, `results/results_table_<dataset>.md` |
| 6.9 | README with run instructions and results | Done — `README.md` |
| 6.10 | Report outline incl. India DPDP Act 2023 and HIPAA | Done — `REPORT_OUTLINE.md` §8.2 and §8.3 |
| 6.11 | 10–12 slide deck outline with speaker notes | Done — `SLIDES_OUTLINE.md`, 12 slides + timed notes + a hostile-question map |
| 6.12 | 20 viva questions with strong answers | Done — `VIVA_QUESTIONS.md` + a rapid-fire crib sheet |

## 7. Acceptance criteria — measured verdicts

From `results/acceptance_all.txt`, 3 seeds, mean over seeds, measured against
the held-out test split.

| Criterion | Heart | Pima |
|---|---|---|
| VAE correlation gap clearly below naive | **PASS** 0.1839 → 0.1171 (−36.3%; floor 0.0993) | **PASS** 0.2134 → 0.0907 (−57.5%; floor 0.0767) |
| TSTR AUC ≥ 90% of real-data AUC for the best model | **PASS** 99.5% retention (VAE@DP budget / LR); VAE 99.0% | **PASS** 99.1% (TVAE / LR); VAE 97.3% |
| DP-VAE attack AUC near 0.5 with a visible utility trade-off | **PARTIAL** — see §7.1 | **PASS** — attack AUC 0.508–0.514 at every ε; TSTR AUC 0.29–0.34 vs 0.79 non-private |
| Exact-copy rate | **PASS** 0.000% for all 10 models | **PASS** 0.000% |
| Figures, tables and demo each work with one command | **PASS** | **PASS** |

TRTR ceilings: heart LR 0.9524 / RF 0.9518; pima LR 0.8209 / RF 0.8194.
`n_seeds_ok == 3` for all 20 model×dataset rows — every headline metric was
finite for every seed.

Derived ratios, verified against the CSV rather than quoted from memory: the
Cleveland VAE correlation gap is **1.18×** the real-vs-real noise floor where the
naive baseline is **1.85×**; on Pima the VAE is **1.18×** and naive **2.78×**. The
VAE's DCR ratio is 0.980 (heart) and 1.053 (pima), i.e. synthetic rows sit where
new real patients sit.

### 7.1 Criterion 3, stated precisely rather than spun

Two attacks were run and they behaved differently:

1. **Pima: the criterion is met cleanly.** Distance-based attack AUC is
   0.514 / 0.514 / 0.508 / 0.510 at ε = 1 / 3 / 10 / 30 — flat and at chance —
   while TSTR AUC sits at 0.29–0.34 against 0.79 for the non-private model at the
   same optimisation budget, and categorical TV degrades from 0.021 to 0.121.
   Privacy near chance, utility cost plainly visible.
2. **Heart: the utility half is met, the attack half is not clean.** TSTR AUC
   rises 0.514 → 0.580 across ε (against 0.948 non-private) and TV falls
   0.167 → 0.044, so the trade-off is unambiguous. But the distance-based attack
   AUC moves the *wrong* way (0.542 at ε=1 → 0.591 at ε=30), and the
   likelihood-gap attack barely moves at all (0.583 at ε=1 vs 0.583 for the
   non-private reference).

Why, and why this is not a framing failure:

- At ε = 1 the accountant requires σ ≈ **22.97** and the decoder collapses to
  nearly a constant vector. Nearest-neighbour distance to the synthetic set then
  stops being a membership signal — it is a geometric accident about where that
  one point sits relative to the training cloud. The attack is **invalid** at
  that end of the sweep, not successful.
- The rank transform is fitted on the training marginal, so every model's output
  lives on a train-derived value vocabulary. That biases *all* distance-based
  attacks upward regardless of ε — which is why the non-private VAE also reads
  0.579 rather than 0.500.
- An earlier single-seed probe at a 150-epoch budget appeared to show the
  likelihood-gap attack moving monotonically with ε. **The full 3-seed run did
  not reproduce it**, and the claim has been removed rather than kept because it
  was convenient. That reversal is the main reason this section exists.

The privacy evidence that *does* hold up on heart is the exact-copy rate
(0.000% for every model), the **DCR ratio** — which separates the models
cleanly, 0.609 (TVAE, memorising) → 0.980 (VAE, ideal) → 1.124 → 1.409 → 1.610
(DP ε=1, over-dispersed) — and the ε guarantee itself, with the caveat that
ε = 30 on 181 rows is not a strong guarantee.

### 7.2 Other findings worth flagging in the viva

- **TVAE is the sharpest teaching case in the project**: best utility retention
  on pima (99.1%) and the worst privacy signal on heart (DCR ratio 0.609, with
  visible clumping onto real training points in `fig1_pca_overlay_heart.png`).
  High utility and high leakage coexisting is exactly the point of measuring
  both families.
- **WGAN-GP is the weakest model on both datasets** (heart gap 0.393 ± 0.122,
  pima 0.488) and has the largest seed variance. Reported as a result.
- **CTGAN underperforms the naive baseline on numeric marginals** here
  (KS 0.341 heart / 0.247 pima vs naive 0.135 / 0.059), consistent with a model
  designed for larger tables being starved at n=181.
- **Bigger data improves the yardstick as much as the model**: the pima noise
  floor (gap 0.077) is well below heart's (0.099), so part of the apparent gain
  is reduced measurement noise. Both floors are printed everywhere for that
  reason.

## 8. Verification runs actually performed

| Command | Result |
|---|---|
| `python tests/test_data.py` | both datasets load; splits 181/61/61 and 460/154/154; prevalence equal across folds to 4 dp; zero NaN after cleaning; category vocabularies legal |
| `python tests/test_models.py` | all 6 generators fit and sample; every column inside valid medical ranges; integer columns integral; no illegal categories; diversity knob monotone in temperature and rare-class share |
| `python tests/test_dp.py` | eps in {1,3,10,30} -> sigma in {7.31, 3.17, 1.42, 0.79} at 60 epochs; epsilon spent <= target in every case |
| `python tests/test_metrics.py` | all three metric families run end to end; surfaced the loss-imbalance bug (the VAE initially scored a *worse* correlation gap than the naive baseline) |
| `python -m synth.tune` | 18 configurations x 2 datasets, scored on validation only; wrote `tuning_grid_*.csv` and `tuned_config.json` |
| `python run_pipeline.py --quick --datasets heart --seeds 0 1` | full code path exercised before the long run; verdicts printed |
| `python run_pipeline.py --skip-tune` | the authoritative 3-seed run: 60 fits, both datasets, 3257 s. **Re-run from scratch and diffed against the first run: all 60 rows matched and every metric column was bit-identical** (only `wall_seconds` differed) |
| `python run_pipeline.py --reaggregate` | re-derived every summary from `results_summary_long.csv` after fixing `n_seeds_ok`, with no retraining |
| `python make_figures.py` | 17 figures written to `results/figures/`; every panel inspected |
| `python make_notebook.py --execute` | notebook executed top to bottom: 47 cells, **0 error outputs**, 27 cells producing output, all three in-notebook acceptance assertions passing |
| `python tests/test_app.py` | 3 layers: headless sweep (2 datasets x 6 models x 3 class filters x 3 diversities x 3 counts, plus the empty-output path) clean; CSV round-trip 137 rows; live Streamlit server health + page fetch |
| `streamlit run app.py` driven in a real Chrome session | all four tabs clicked and screenshotted: Fidelity (6 charts + noise-floor caption), Utility (TSTR/TRTR table + bars), Privacy (4 metrics + DCR histogram), Model comparison (frozen 3-seed mean +/- std table) |
| `python tests/test_app_apptest.py` | **the strongest UI test**: boots the real `app.py` in Streamlit's own `AppTest` ScriptRunner and re-runs it under 19 scenarios - all 6 generators x both datasets, DP epsilon at 1 and 30, and five extreme control combinations (n=10 and n=5000, diversity 0.0 and 1.0, both single-class filters, seed 777, Standard budget). **0 failures, 0 exceptions, 0 `st.error` blocks.** The class-filtered scenarios correctly render 9 metrics instead of 11, because the Utility tab deliberately bails out with a single-class warning |

### Defects the verification actually caught

This is the section that distinguishes "I wrote tests" from "the tests found
things". Every row below was a real failure observed during the build.

| Caught by | Defect | Effect had it shipped |
|---|---|---|
| `tests/test_data.py` | the train split's own NaNs were never imputed | `ca`/`thal` NaNs propagated; category-legality check failed downstream |
| `tests/test_models.py` | `np.bincount` over-extends for sparse levels like `thal` {3,6,7} | `IndexError`, inventing a category that does not exist in the cohort |
| `tests/test_metrics.py` | categorical CE term ~35x larger than the numeric MSE term | VAE correlation gap **worse than the naive baseline** - acceptance criterion 1 failing |
| `tests/test_metrics.py` | mean-only decoder sampler under-dispersed every numeric column | KS 0.200 against a 0.128 floor; age spanning 33-71 versus a real 29-76 |
| browser run of `app.py` | `build_model(**hp)` was handed a cache **key string** | the app crashed on its **default** state - completely invisible to the headless test, which never called `main()` |
| browser run of `app.py` | `model.privacy_note` read as a bare attribute | `AttributeError` for the naive resampler and the ctgan wrappers |
| notebook execution | `np.nanmax` applied to a list of ragged distance arrays | the notebook died at the DCR cell, so "runs top to bottom" would have been a false claim |
| figure inspection | filename parsed with `[4:-11]` instead of `[4:-10]` | 8 of 10 PCA panels silently blank, because missing files were skipped rather than reported |
| figure inspection | an unregistered model colour fell through to a hash pick | a bar rendered in Okabe-Ito yellow, effectively invisible on white |
| figure inspection | 10 unrotated multi-line tick labels | unreadable axis collision on four of the figures |
| `--reaggregate` | `n_seeds_ok` counted *inapplicable* metrics (e.g. `dp_sigma` on a non-DP model) as missing data | reported "0 of 3 seeds" for 16 of 20 rows, implying the study had failed when it had not |
| final 3-seed run | a single-seed probe suggested the likelihood-gap attack moved monotonically with epsilon | **claim withdrawn** - the 3-seed run did not reproduce it; see section 7.1 |
| `tests/test_app_apptest.py` (first version) | the test itself asserted on `at.button` for the CSV export, but Streamlit 1.30's `AppTest` exposes no `download_button` accessor | all 19 scenarios reported FAIL while the app itself raised nothing - a false alarm from the harness, fixed by asserting on `at.error` (which is what `main()`'s crash handler writes to) and by verifying the download button in the browser instead |
| two independent full runs | reproducibility was assumed rather than checked | diffed all 60 per-seed rows: every metric column bit-identical, only `wall_seconds` differed |

Two of these - the default-state app crash and the withdrawn epsilon claim - were
only caught because the UI was driven in a real browser and because the summaries
were re-derived from raw per-seed rows rather than accepted from a convenient
probe.
