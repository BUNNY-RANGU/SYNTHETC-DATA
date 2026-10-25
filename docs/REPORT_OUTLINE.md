# Report Outline — Synthetic Medical Data Generator for Privacy-Preserving
# Healthcare Research

> Structure below is the submission structure. Every `[[N]]` is filled from
> `results/results_summary.csv` by `python run_pipeline.py` — do not hand-write
> a number that is not in that file.

---

## Abstract (150–200 words)

Problem → method → headline numbers → the honest caveat, in that order.

Filled in from `results/results_summary.csv` (~195 words):

> Clinical data sharing is blocked by regulation long before it is blocked by
> technical feasibility. We build and evaluate a mixed-type synthetic tabular
> generator for electronic-health-record data: a custom variational autoencoder
> with a split MSE / per-column cross-entropy objective and KL annealing, its
> differentially-private variant trained with Opacus DP-SGD at ε ∈
> {1, 3, 10, 30}, and GAN comparisons (CTGAN, TVAE, and a WGAN-GP trained in
> the same encoded space as our VAE). On UCI Cleveland heart disease (n = 303)
> and Pima diabetes (n = 768) with a fixed stratified 60/20/20 split, the VAE
> reduces the Spearman correlation gap from 0.184 (a naive per-column
> resampler) to 0.117 on Cleveland and from 0.213 to 0.091 on Pima, against a
> measured real-vs-real noise floor of 0.099 and 0.077 respectively, while
> retaining 99.0% and 97.3% of real-data AUC under a train-on-synthetic /
> test-on-real protocol. No model emitted a single exact copy of a training
> patient. The privacy experiments show the theorem rather than asserting it:
> the accountant required noise σ = 22.97 at ε = 1 versus 2.01 at ε = 30 on
> Cleveland, and downstream AUC fell from 0.948 (non-private, matched
> optimisation budget) to 0.51. We also report where our privacy metrics do
> *not* discriminate cleanly, and we state which metrics are informative at
> n = 181 training rows and which are not.

---

## 1. Introduction

### 1.1 Motivation
- Health data is the highest-value and highest-risk data class.
- Legal/ethical friction on sharing vs. the scientific cost of not sharing.
- The gap synthetic data is supposed to fill: *statistically useful,
  individually harmless*.
- Why "generate and hope" is not an answer: the second clause needs measurement.

### 1.2 Problem statement
Learn a distribution `p_θ(x) ≈ p_data(x)` over mixed-type patient records such
that (i) downstream analyses on synthetic data transfer to real data, and
(ii) no individual can be identified or their membership confirmed.

### 1.3 Contributions
1. An end-to-end reproducible study with all three evaluation families —
   fidelity, utility, privacy — implemented and run, not cited.
2. A tabular VAE formulation for mixed types with per-column heads, plus the
   loss-imbalance diagnosis that makes it actually work ([[§4.3]]).
3. A DP-VAE privacy/utility curve at four ε values on a genuinely small
   clinical cohort, with the accountant's chosen σ reported.
4. A measurement-first protocol: the **real-vs-real noise floor** is computed
   with the same code as every model score, so no fidelity claim can be made
   without an explicit reference.
5. A worked account of which metrics are *not* informative here and why
   ([[§7.2]]).

### 1.4 Report organisation
One short paragraph mapping sections.

---

## 2. Related Work

Organise by *question answered*, not by chronology.

### 2.1 Generative models for tabular data
- **CTGAN** (Xu et al., 2019) — mode + demodal normalisation, conditional
  generator, penalised discriminator. The reference for mixed types.
- **TVAE** (same paper) — the VAE sibling.
- **TabGAN / STaTS / WaveGAN** successors; **TableGAN** (Zhao et al., 2019).
- **SD-VAE / TabVAE** (Xu, Vejarano & Eleja, 2022) — VAE with a discretised
  Gaussian likelihood; directly relevant to our point-mass problem in
  [[§4.4]].
- **CARDIGAN** (2022), **RFEM** (2023), **TabDDPM / TabSyn** (2024) —
  diffusion-based, state-of-the-art but compute-hungry; note we did not run
  these and why ([[§7.4]]).
- **SDV / SynthCity / Metrics** — tooling; our evaluation families align with
  the SDV "diagnostic validity" and Bayesian network approaches.

### 2.2 Privacy: differential privacy for synthetic data
- **Dwork & Roth (2014)** — the definition and composition.
- **Abadi et al. (2016)** — DP-SGD and the moments accountant.
- **Mironov (2017)** — Rényi DP, which is what Opacus implements.
- **Jalote, Hennecke & Riedel (2021)** — DP-GAN for tabular data; reported the
  same "ε small ⇒ utility collapse" behaviour we observe.
- **Dufant et al. (2022)**, **Yan et al. (2023)** — DP for EHR specifically.

### 2.3 Evaluation methodology
- **Kuhar, Gams & Ogriz (2019)** — the standard survey of synthetic-data
  metrics; defines the discriminative / predictive / adaptive taxonomy that our
  A/B/C families map onto.
- **Jordon, Yoon & Schiele (2018)** — the TSTR-style work in the vision domain.
- **Shokri et al. (2017)** — membership inference; **Salem et al. (2019)** —
  distance-based MIA, the weak attack we implement.
- **Zhang et al. (2024)** "Are synthetic datasets fair?" — the caveat we adopt.

### 2.4 The gap this project fills
Most student-level implementations report marginal histograms and stop. We
report a measured noise floor, joint-structure metrics as the primary fidelity
signal, transfer utility against an explicit ceiling, and an attack-based
privacy measure with the attack's weakness declared.

---

## 3. Background

- VAEs: ELBO, reparameterisation, the prior/aggregate-posterior mismatch,
  posterior collapse, why KL annealing exists.
- GANs: the Wasserstein formulation, the 1-Lipschitz constraint, the gradient
  penalty as its soft replacement for weight clipping.
- Differential privacy: `(ε, δ)`-DP informally, *why* per-sample gradient
  clipping is required, the RDP/PRV accountant, composition across steps, and
  the standard `δ = 1/n` convention.
- Mixed-type tabular difficulty: no natural ordering of categories, heavy-tailed
  bounded numerics, point masses, and small n.

---

## 4. Method

### 4.1 Data and protocol  *(this subsection is the one examiners probe)*
- Datasets, exact provenance, row counts, column semantics.
- Missingness: Cleveland `?` (4 in `ca`, 2 in `thal`); Pima `0`-as-missing in
  five physiologically-impossible columns, `pregnancies == 0` retained as real.
- Cleveland `num` binarised per the UCI description; `bmi_class` added to Pima as
  a stated design choice so mixed-type machinery is exercised on both tables.
- **Split: stratified 60/20/20, seed 42, fixed for the whole study.**
- **Generators fit on train only. Hyper-parameters chosen on validation only.
  Test read once, in the final evaluation stage, one pass per model per seed.**
- Every statistic used for cleaning, scaling or distance standardising is fitted
  on train and never re-estimated on val/test.
- Honest nuance: "once" means one evaluation pass per model, not one number
  overall; no tuning or selection decision used it.

### 4.2 Encoding
Mode-based: one-hot for categoricals (no invented ordering), rank-to-Gaussian +
min-max for numerics, fixed category vocabulary so a generator cannot invent a
level. Label treated as a 2-level categorical head so the diagnosis is generated
with the symptoms, which is what makes class-conditional sampling possible.

### 4.3 The VAE
`encoder → (μ, logσ²) → z = μ + σ ⊙ ε → decoder` with **separate heads**:
sigmoid numeric head trained with MSE, one softmax head per categorical column
trained with cross-entropy, KL annealed linearly over the first
100 epochs (the KL weight ramps linearly to beta_max).

**The loss-imbalance finding.** Averaged MSE ≈ 0.028 vs averaged CE ≈ 0.98 at a
constant predictor — a ~35× mismatch. Unweighted, the optimiser ignores the
numeric columns; the resulting VAE had a correlation gap *worse than the naive
baseline* (0.202 before weighting vs 0.181 for the naive resampler, on the pre-transform pipeline). Adding a numeric loss weight, tuned on
validation, is what makes the model work. This is reported as a finding, not
buried as a bug.

**The sampler finding.** MSE training is maximum likelihood under a
fixed-variance Gaussian, so the generative sample is `x = f(z) + σε`, not
`f(z)`. Emitting the mean gave regression-to-the-mean: age spanning 33–71
against a real 29–76, cholesterol truncated at 479 against a real 564,
mean KS 0.200 → 0.135, against a real-vs-real noise floor of 0.128. Estimating σ from train reconstruction residuals and adding it
at sample time is the sampler the loss already implies.

### 4.4 DP-VAE
Opacus DP-SGD: per-sample gradient clipping at `max_grad_norm = 1.0`, Gaussian
noise with multiplier σ solved by the accountant for each target ε at
δ = 1/n_train. Reported per ε: chosen σ, steps actually taken, and ε actually
spent. Note the model is unwrapped after training so noise hooks never touch
inference.

### 4.5 CTGAN, TVAE, WGAN-GP
Wrapper fixes the package's default batch_size of 500 against 181 rows, passes
discrete columns by name, and always re-runs our sanitiser on the output because
`ctgan` emits unclipped numerics. WGAN-GP is trained on the **same encoded
space** as our VAE so VAE-vs-GAN is an architecture comparison rather than a
preprocessing comparison; critic loss `D(fake) − D(real) + λ·GP`, generator loss
`−D(fake)` plus a small categorical entropy penalty to sharpen one-hots.

### 4.6 Post-processing
Denormalise, clip to plausible medical ranges, round integer columns, snap
categoricals to the nearest legal level. The clip window is widened to at least
the observed train envelope on purpose: clipping generated data tighter than
the real data would fabricate a boundary spike that is a preprocessing artifact,
not fidelity.

### 4.7 Diversity knob
One user-facing slider → categorical sampling temperature `2^(d−0.5)` and latent
scale `1 + 0.5(d−0.5)`. `d = 0.5` is exactly neutral and is what every published
number uses. For CTGAN/TVAE, where the package exposes no per-row probabilities,
`d` becomes post-hoc kernel jitter on numerics — a different mechanism, labelled
as such.

---

## 5. Experiments

### 5.1 Setup
Hardware (CPU-only, torch 2.10.0+cpu, Python 3.11.9), seeds `[0,1,2]`, n_synthetic = 1000,
wall-clock per model from `run_metadata.json`.

### 5.2 Tuning, and what was tuned
VAE grid: `w_num ∈ {20, 120} × β_max ∈ {0.03, 0.1, 0.3} × latent ∈ {8, 16, 32}` = 18
configurations, scored on validation by the mean of (KS, TV, correlation gap)
against the validation rows. **Grid table: `results/tuning_grid_<dataset>.csv`.**
Chosen configuration and the fact that the DP variants inherit it are in
`results/tuned_config.json`. CTGAN/TVAE/WGAN-GP use fixed published settings;
they were not tuned, which is a stated asymmetry rather than a hidden one.

### 5.3 Fidelity results
Per-model KS, TV, correlation gap, and gap ÷ noise floor. PCA overlay,
histogram overlays, correlation heatmaps + difference panel. Per-column detail
so a reviewer can see *which* association broke.

### 5.4 Utility results
TSTR vs TRTR for LR and RF; accuracy, F1, AUC; retention ratio.

### 5.5 Privacy results
Exact-copy rate, DCR against three references, suspicious-close rate, MIA AUC,
and the ε sweep.

### 5.6 Small-data vs larger-data comparison
Heart (181 train) vs Pima (460 train). Compare the *noise floor* first — a
large part of the apparent difference in model quality is just the difference in
how noisy the yardstick is.

---

## 6. Results and Discussion

Fill from the CSV. The narrative should be:

- **Marginals are cheap.** The naive baseline wins them outright. Any model
  that only matches histograms has learned nothing.
- **Joint structure is the actual result.** Spearman gap 0.184 → 0.117 on Cleveland
  (0.213 → 0.091 on Pima) against noise floors of 0.099 and 0.077.
- **Utility transfers unevenly.** Linear models are more robust to synthetic
  data than forests — forests exploit fine-grained boundaries that synthetic
  data smooths away. (Verify against your own numbers before claiming this.)
- **The privacy guarantee is not free.** ε: 30 → 1 gives DCR ratio 1.12 → 1.61 and
  TSTR AUC 0.580 → 0.514 (0.948 non-private at the matched budget). At ε = 1 the
  accountant needs σ = 22.97 on Cleveland and 9.84 on Pima.
- **DP makes memorisation undetectable *and* the data uninformative.** Report
  which of the two the metrics actually distinguish.
- **CTGAN/TVAE are the strongest baselines at this size**, and say so plainly.

---

## 7. Limitations

### 7.1 Small data
181 training rows, 61 test rows. With 61 held-out patients the
real-vs-real correlation gap is already [[floor]], so gaps near it are
indistinguishable from sampling noise. Three seeds give a spread estimate, not a
confidence interval; `std(ddof=1)` on n=3 is itself noisy.

### 7.2 Which metrics are not informative here
Because the numeric transform is fitted on the training marginal, KS/TV sit at
the noise floor for every neural model **partly by construction**. They are
necessary but not sufficient. The discriminating metrics are the correlation
gap and TSTR. State this before an examiner states it for you.

### 7.3 GAN instability
WGAN-GP on 181 rows swings between seeds and can collapse onto one diagnosis
class. Single-class synthetic data makes TSTR undefined; we emit NaN plus a note
rather than a fake zero, and `n_seeds_ok` in the summary records when a mean is
over fewer than three runs.

### 7.4 Scope of the comparison
No diffusion model (TabDDPM/TabSyn), no flow model, no PATEGAN/PrivGAN-DP
baseline, no DP-CTGAN. Chosen for CPU feasibility and clarity; named as missing.

### 7.5 The privacy attack is weak
Single distance threshold, no gradient access, no shadow models, both groups
from the same distribution. MIA AUC ≈ 0.5 is evidence against *that attack*.
The `(ε, δ)` guarantee is the stronger claim where it holds — but at ε = 30 on
181 rows it is not a strong guarantee.

### 7.6 Synthetic ≠ anonymous
De-anonymisation is a property of an *attacker's whole dataset*, not of one
table. Re-identification via linkage to voter rolls, mobility data or a public
obituary is out of scope and undiminished by our results. Neither k-anonymity
nor DP is a legal safe harbour.

### 7.7 Data representativeness
Cleveland is single-site, old, and **enriched for disease** (prevalence ~0.54
versus a low community base rate), which inflates every AUC in this study. Pima
is a specific Indigenous community with a documented history of research
extractivism. Neither supports claims about a general population.

### 7.8 Statistical-claim hygiene
No multiple-comparison correction across 10 model configurations × 17 headline metrics.
"Best model" is selected on validation and then evaluated on test; the retention
number therefore carries an optimistic bias of order the validation noise. We
did not run enough seeds to correct it.

### 7.9 Fairness and bias
Nothing here measures subgroup fidelity. Synthetic data inherits and can amplify
cohort bias; a generator that reproduces a skewed sample reproduces the skew.

### 7.10 The ELBO is not a valid quality metric under heavy DP noise
Validation loss *falls* from 1.795 at ε = 1 to 1.654 at ε = 30, against 1.929 for
the non-private VAE. That is not learning: a noise-collapsed encoder pushes the
posterior onto the prior, so the KL term shrinks to ~0 and the ELBO improves
while every fidelity and utility metric collapses to the naive baseline. We
nearly cited this number in the other direction. Read only the downstream
metrics as evidence.

### 7.11 DP runs used Opacus with `secure_mode=False`
That is Opacus's default and the reason the sweep is affordable on CPU, but it
means the DP-SGD noise comes from a non-cryptographic RNG — Opacus itself warns
this is "fine for experimentation, not for production". Opacus also emitted
"Optimal order is the largest alpha" at some budgets, i.e. the RDP order search
was bounded and the published ε could be marginally looser than a wider search
yields. A production release must retrain with `secure_mode=True`.

---

## 8. Ethics, Legal and Societal Considerations

### 8.1 Research governance
This study uses **already-public, de-identified** UCI research data, so no fresh
human-participant consent was sought and no IRB/ethics-committee submission was
made for it. That is a property of the source data, **not** a template for
replicating the method on a hospital's records — which would require a data-use
agreement, a governance/ethics approval, and a documented lawful basis before
any model is fit.

### 8.2 India — DPDP Act 2023
- The Act regulates **digital personal data** and consent-based *notice and
  consent*; data fiduciaries must give notice of purpose and obtain free,
  specific, informed consent, with data-minimisation and purpose-limitation.
- **"Personal data"** is data about an identifiable person. **De-identified and
  anonymised data are outside the Act's scope** — which makes the
  synthetic-data question a *legal* question, not just a technical one: does
  your synthetic cohort actually break identifiability?
- The Act does **not** define the anonymisation standard or a re-identification
  threshold. There is therefore no statutory box to tick that says "synthetic,
  therefore compliant." Our DCR / exact-copy / MIA measurements are *evidence*
  toward that judgement, not a certificate of it.
- Significant Data Fiduciaries carry additional obligations (DPO,
  data-localisation-style trust measures, consent managers). Cross-border
  transfer is restricted to notified territories.
- Practical position for this project: a synthetic EHR released for research is
  defensible under the Act when (a) it demonstrably does not correspond to
  identifiable individuals, (b) the generator is documented and its privacy
  measurements published alongside it, (c) re-identification testing is repeated
  per release, not once at design time.
- Sensitive personal data has no separate category in the Act as passed (unlike
  the 2019 draft), so health data is governed by the same general obligations —
  which makes strong technical controls a practical necessity, not an option.

### 8.3 HIPAA (US)
- **Safe Harbor de-identification** (45 CFR §164.514(b)(2)): remove all 18
  identifiers, including dates more granular than year, and no actual knowledge
  that residual information could identify.
- **Expert Determination** (§164.514(b)(1)): a statistician documents that the
  risk of re-identification is *very small*. Synthetic data released under HIPAA
  most naturally travels this route, because "no real individuals are described"
  is a claim about provenance, and Safe Harbor presumes you removed real
  people's identifiers rather than generated new records.
- Note the awkward fit: the **limited data set** and de-identification rules
  assume the rows describe real people. A synthetic table describes no one, so
  the honest argument is that it is *not PHI at all* — provided the empirical
  overlap (exact copies, DCR) is genuinely negligible. **That proviso is exactly
  what §6 measures.** A generator that emits verbatim training rows would
  produce a table that is PHI regardless of what it is called.
- Also relevant: the **Security Rule** (access controls on the real data used to
  train), the **Minimum Necessary** standard, and the 2024/2025 Part 2 alignment
  for substance-use-disorder records.

### 8.4 GDPR / EU context
Recital 26: anonymisation is assessed on *means reasonably likely* to be used
for re-identification, held by the data controller *or any other person*. That
includes the model itself. Recital 27 says anonymised data fall outside the
regulation; Article 9 special-category rules for health data still hinge on that
assessment.

### 8.5 Consent, provenance, and dual use
- Cleveland/Pima provenance is documented but the original consent scopes did
  not contemplate generative modelling.
- Dual use: a strong synthetic EHR generator lowers the barrier to convincing
  medical-records fraud and to training attack tooling. Release notes should say
  which guarantees are *measured* and which are *assumed*.
- Model-artifact leakage: a checkpoint is not a synthetic table. Gradients can
  memorise; the DP guarantee covers the training algorithm, not the released
  weights of a non-DP model.

### 8.6 Bias and justice
Disease-prevalence artefacts in an enriched cohort propagate into any downstream
clinical rule fit on synthetic data. Report prevalence and cohort structure
prominently rather than as a footnote.

### 8.7 What this project does not license
No clinical decision, no production data-sharing arrangement, and no claim of
regulatory compliance follows from these experiments.

---

## 9. Conclusion and Future Work
Restate the measured findings in three sentences. Future work: more seeds (≥10)
with corrected multiple comparisons, DP-CTGAN as a stronger privacy baseline,
TabVAE-style discretised-mixture numeric heads to remove the residual
dispersion bias, subgroup-fidelity and fairness auditing, and an attribute-
inference attack to complement the weak MIA.

## References
*(Use IEEE numbering; every citation above must resolve to a real venue/year.)*

---

## Appendix A — Reproduction
```
pip install -r requirements.txt
python run_pipeline.py            # or --quick for a ~3-minute pass
python make_figures.py
streamlit run app.py
python make_notebook.py --execute
```
File map, seed policy, hardware, wall-clock.

## Appendix B — Full per-column metric tables
`results/results_summary_long.csv`, `results/tuning_grid_*.csv`,
`results/privacy_utility_curve.csv`.

## Appendix C — Edge cases exercised
`notebook §7` forces single-class synthetic data, an empty synthetic table, a
zero-variance column, and extreme diversity settings, and asserts each is handled
without raising.

## Appendix D — Self-audit table
Requirement → done → evidence. See `SELF_AUDIT.md`.
