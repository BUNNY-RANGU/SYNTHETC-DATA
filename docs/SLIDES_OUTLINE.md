# Deck Outline — Synthetic Medical Data Generator (12 slides + speaker notes)

Target: 12 minutes presenting, 8 minutes questions. Colour-blind-safe palette
(Okabe-Ito) throughout; every number on a slide is traceable to
`results/results_summary.csv`.

Design rule for the whole deck: **one claim per slide, one piece of evidence per
slide, and the reference line drawn on the chart.** If a slide needs a paragraph
of explanation to be true, it is a report slide, not a deck slide.

---

## Slide 1 — Title
**Synthetic Medical Data Generator**
*VAE and GAN methods for privacy-preserving healthcare research*
Name · B.Tech project submission · date
Small footer: "UCI Cleveland heart disease (n=303) · Pima diabetes (n=768) ·
PyTorch · Opacus · CTGAN · all results reproducible with one command"

> **Speaker notes (~20 s).** Introduce yourself in one line. Then state the
> tension the whole project sits on: clinical data is the most valuable and the
> most restricted data there is, and the restriction blocks good research long
> before it blocks bad actors. Say plainly that the contribution of this project
> is not "I generated data" — it is "I measured whether generating it is safe,
> and I reported where the measurement says no." Do not apologise for the size of
> the dataset in the opening; §11 handles that honestly and early is worse.

---

## Slide 2 — The problem, in one picture
Three stacked panels:
1. Real patient row → highlighted: this row is a person.
2. "Anonymised" row with quasi-identifiers → still linkable.
3. Synthetic row → matches no person, still lets you fit a model.
Caption: *the question is not "does it look real" but "does it work, and does it
leak".*

> **Speaker notes (~60 s).** Walk left to right. Make the point that a table can
> look perfectly realistic and still be a privacy failure — the classic failure
> mode of synthetic-data work is celebrating realistic-looking histograms.
> Mention the two-sided requirement up front because it structures everything
> after: fidelity and utility are the *usefulness* axis, privacy is the *safety*
> axis, and they trade against each other. Name one concrete use case — a
> researcher outside a hospital who needs to prototype a risk model without
> touching PHI — so the audience has somewhere to put the abstraction.

---

## Slide 3 — Data and the split protocol
Table of the two datasets: rows, numeric/categorical counts, prevalence.
A horizontal 60/20/20 bar labelled **TRAIN / VAL / TEST** with three rules
underneath:
- generators fit on train **only**
- hyper-parameters chosen on validation **only**
- test read **once**, at the end, for headline numbers
Callout: *every* cleaning statistic (median, mode, scaling bounds, distance
standardisation) is fitted on train and never re-estimated.

> **Speaker notes (~70 s).** This is the slide examiners attack, so deliver it
> precisely and without hedging. Say the split is stratified on the diagnosis at
> a fixed seed 42, so all folds carry identical prevalence and every run in the
> deck uses the same folds. Mention the two cleaning decisions that are genuinely
> debatable and that you thought about them: Cleveland's `?` markers were
> mode-imputed to keep all 303 rows, and Pima's zeros in glucose / blood pressure
> / skin thickness / insulin / BMI were treated as missing rather than as
> measurements, because a living person has no such value of zero — while
> `pregnancies == 0` is real and was left alone. Then volunteer the nuance before
> anyone asks: "touched once" means one evaluation pass per model per seed, not
> literally one number, because reporting mean ± std across three seeds
> necessarily reads the test split three times. What is guaranteed is that no
> tuning or model-selection decision used it.

---

## Slide 4 — Why tabular is harder than images
Four small visuals, one line each:
- `thal ∈ {3, 6, 7}` — no ordering, so one-hot not integer
- `chol` heavy right tail — min-max wastes 70% of the range
- `oldpeak` has 28.7% exact zeros — a *point mass*
- n = 181 — the estimator is the bottleneck, not the architecture

> **Speaker notes (~60 s).** Make the point-mass example concrete because it is
> the single most instructive thing in the project: a decoder trained with MSE
> outputs a conditional *mean*, and a mean is a smooth density that cannot
> contain an atom. So the model is structurally incapable of producing the thing
> the data actually does 29% of the time. Say that the fix is a train-fitted
> rank-to-Gaussian transform whose inverse has flat regions, so a whole band of
> decoder outputs snaps back to exactly 0.0. Also say the honest cost of that
> fix, because it comes back in §7: it makes 1-D marginals partly guaranteed by
> construction, which is precisely why marginals are not the interesting metric
> here.

---

## Slide 5 — The models, as a comparison table
| Label | What it is | Why it is in the study |
|---|---|---|
| naive | resample each column independently | the control |
| **VAE** | custom, (μ, log σ²) → reparameterise → split heads, KL annealing | the model |
| **DP-VAE** | same VAE under Opacus DP-SGD, ε ∈ {1, 3, 10, 30} | the privacy experiment |
| CTGAN / TVAE | published baselines | reference points |
| WGAN-GP | same encoded space as the VAE | the extra |
One architecture diagram of the VAE with the two output heads visible.

> **Speaker notes (~60 s).** Spend the time on *why each model is present*, not
> on what each model is. The naive baseline's job is to be unbeatable on
> marginals, and it succeeds — that is the point, it makes marginal plots
> meaningless as evidence. Emphasise that WGAN-GP was chosen as the extra
> specifically so it shares the VAE's encoding, which turns VAE-versus-GAN into
> an architecture comparison instead of an accident of preprocessing. Say that
> the two output heads exist so the loss can be MSE on numerics and
> cross-entropy per categorical column, which is what the specification asked
> for and is also what caused the bug in the next slide.

---

## Slide 6 — Two findings from building the VAE
**Finding 1 — loss imbalance.** Numeric MSE ≈ 0.028 vs categorical CE ≈ 0.98 at
a constant predictor: a **35×** mismatch. Unweighted, the numeric columns are
never modelled, and the VAE's correlation gap was *worse than the naive
baseline's*. Fixed with a numeric loss weight tuned on validation.
**Finding 2 — the wrong sampler.** MSE is maximum likelihood under a
fixed-variance Gaussian, so the sample is `x = f(z) + σ ε`, not `f(z)`. Emitting
the mean under-dispersed everything (age 33–71 against a real 29–76, cholesterol
capped at 479 against a real 564). σ is read off the training reconstruction
residuals.
Small before/after bar chart for both.

> **Speaker notes (~90 s).** This is the strongest slide in the deck and it is a
> failure slide, which is why it works. Deliver finding 1 as a debugging story:
> the metric was telling the truth about a broken model, and the truth was
> uncomfortable enough that you went and found the cause instead of changing the
> metric. For finding 2, say the sentence that makes it a principle rather than a
> hack: "we were not adding noise to the output to make it look better; we were
> sampling the distribution the loss function already assumed." If asked whether
> the loss weight is just a tuned magic number, answer directly: it is one entry
> in a validation-only grid of 18 configurations, the grid is in
> `results/tuning_grid_heart.csv`, and the winning value changed when the
> transform and sampler changed — which is evidence that the search was doing
> real work rather than confirming a guess.

---

## Slide 7 — Fidelity: the noise floor, then the models
Chart: Spearman correlation gap per model, with a dashed horizontal line =
**real-train versus real-test**, and a dotted line = **naive**.
Numbers: naive **0.184** → VAE **0.117**, floor **0.099** (Pima: 0.213 → 0.091, floor 0.077).
Second panel: PCA overlay (naive vs VAE vs CTGAN), and the correlation
*difference* heatmap for the naive model.

> **Speaker notes (~90 s).** Lead with the floor, never with the model. Explain
> that comparing synthetic data to 61 held-out patients guarantees a non-zero gap
> even if the generator were perfect, because the *real* data scores 0.099 against
> itself — so a gap of 0.13 on this dataset is not a small number, it is roughly
> the smallest measurable number. Then show naive at 0.184 versus VAE at 0.117
> and state criterion 1 is met. Use the naive difference heatmap to land the
> central idea of the project: naive matches every marginal and still fails
> completely, which is exactly why "our synthetic data looks like the real data"
> is not evidence of anything. Expect the question "why Spearman and not
> Pearson" — answer: skewed clinical measures make Pearson dominated by outliers,
> and Spearman is rank-based so it survives the monotone transform; concede that
> treating `{3, 6, 7}` as ordered is a mild abuse.

---

## Slide 8 — Utility: can you actually learn from it?
Grouped bar chart: TSTR accuracy / F1 / AUC per model for LR and RF, with the
**TRTR ceiling** drawn as a dashed line.
Headline: best retention **99.5%** of real-data AUC (VAE at the DP budget / LR);
the headline VAE itself retains 99.0% (LR) and 92.4% (RF).
Note: scaler in TSTR is fitted on the synthetic training set.

> **Speaker notes (~70 s).** Frame TSTR as the only question a data-sharing
> use case cares about: can someone with *only* the synthetic table build a
> working model? Explain the deliberate scaler asymmetry before anyone asks — a
> downstream user who has synthetic data does not also have the real data, so
> standardisation must come from synthetic, and using real statistics there would
> quietly smuggle back the information the experiment pretends they cannot access.
> Note that linear models retain better than random forests, and give the reason:
> forests exploit fine decision boundaries that a smoothed generator does not
> reproduce. Mention the confounder you are aware of: Cleveland is enriched for
> disease at ~54% prevalence, which inflates every AUC in this deck compared to
> a real community cohort.

---

## Slide 9 — Privacy: three attacks, one honest caveat
Three small charts: exact-copy rate, DCR ratio (with the 1.0 line labelled "where
real unseen patients sit"), MIA AUC (with the 0.5 line labelled "attacker at
chance").
Inset histogram: DCR distributions — synthetic versus real-held-out, same axes.

> **Speaker notes (~90 s).** The DCR inset is the slide's substance, so narrate
> it: a genuinely *new real* patient is also close to the training cloud, so a
> small absolute DCR is not automatically a leak — you must compare against the
> real-held-out curve, and the ratio is what you read. Point out that the
> membership-inference attack is deliberately weak — one distance threshold, no
> gradient access, no shadow models — and say out loud that a 0.5 AUC is evidence
> against *that* attack, not a proof of privacy. Close the slide with the sentence
> you want remembered: **"synthetic is not anonymous."** A generator constrains
> what this model reveals; it says nothing about someone who joins the synthetic
> table to voter rolls, mobility data, or a public obituary.

---

## Slide 10 — The privacy–utility trade-off, measured
The ε curve: attack AUC versus ε on the left, TSTR AUC versus ε on the right,
log x-axis, non-private VAE as the ε → ∞ reference point.
Table row per ε: chosen σ, steps, ε actually spent, TSTR AUC, MIA AUC.

> **Speaker notes (~80 s).** This is the scientific centre of the project, so
> slow down. Explain what the accountant is doing: given ε, δ = 1/n, and the step
> count, it solves for the noise multiplier — at ε = 1 it needs σ ≈ 22.97, and
> that much Gaussian noise on 181 rows is why the table stops resembling data.
> State the key line: **this is not our implementation failing, it is the
> theorem.** Then be scrupulous about the direction of the claim — a tight ε is a
> statement about the algorithm, and it is the *stronger* privacy result than the
> MIA number; the MIA AUC near 0.5 is the weaker one. If asked why ε = 30 is
> almost indistinguishable from non-private, answer honestly: with 181 rows the
> composition bound is loose, so a large ε constrains very little.

---

## Slide 11 — Small data versus larger data, and what is not proven
Left: heart (181 train). Right: pima (460 train). Same metric, same protocol.
Show **both noise floors**, not just both model scores.
A short "not proven" list: no diffusion baseline, no DP-CTGAN, 3 seeds, no
multiple-comparison correction, no subgroup fairness audit, model-selection
optimism not quantified.

> **Speaker notes (~80 s).** The purpose of this slide is to show you can tell
> signal from yardstick noise: a large part of why pima looks better than heart is
> that its measurement is less noisy, and you separated those two effects rather
> than reporting the improvement as a property of the model. Then read the "not
> proven" list out loud, briefly and without apology. Volunteering the limits
> before being asked is the single highest-return thing you can do in a viva, and
> it also protects you: if you have already named a limitation, a question about
> it becomes a discussion rather than a gotcha. Say specifically that "best model
> chosen on validation then scored on test" leaves a residual optimistic bias you
> did not quantify, because three seeds is not enough to estimate it.

---

## Slide 12 — Conclusion and what follows
Three sentences of measured findings.
One sentence of what synthetic EHR data is actually good for today.
Three next steps: ≥10 seeds with corrected comparisons, a discretised-mixture
numeric head to kill the residual dispersion bias, and an attribute-inference
attack to complement the weak MIA.
Footer: `python run_pipeline.py && python make_figures.py && streamlit run app.py`

> **Speaker notes (~50 s).** Land three claims, each tied to a number you showed:
> marginals are cheap and joint structure is the real result; our VAE retains
> [[X]]% of real-data utility while beating the naive baseline's correlation gap
> by 36%; and tightening ε drives the attack toward chance and utility down with it,
> which is the price rather than a defect. End on the governance line, because it
> is the part a non-technical examiner remembers: this method's value is that it
> lets research happen without moving real patients' records, and the reason that
> claim is defensible is that it was measured — exact copies, closest-record
> distances, and an attack — rather than assumed. Then stop talking; the one-line
> reproduction command in the footer is your invitation to the questions.

---

## Anticipated hostile questions and where in the deck they are answered
| Question | Slide |
|---|---|
| "Why is your KS so good — isn't that circular?" | 4 and 7 (transform + floor) |
| "You tuned on validation, how do I know test is clean?" | 3 (protocol + the volunteered nuance) |
| "Your dataset has 181 rows. Is anything here real?" | 7 and 11 (noise floor) |
| "Isn't a 0.5 attack AUC meaningless with 61 negatives?" | 9 (weak-attack caveat) |
| "Why not just publish anonymised real data?" | 2 and 9 (linkage, synthetic ≠ anonymous) |
| "What does ε = 1 actually protect you from?" | 10 (algorithm-level statement, composition) |
| "Your WGAN collapsed at one seed, didn't it?" | 11 (NaN + note, not a zero) |

## Slide-build notes
- Every chart must carry its reference line *on the chart*, not in the caption.
- No more than six model names on any axis; group the four DP-ε bars together and
  label the group, not each bar.
- Use `results/figures/fig9_dashboard_<dataset>.png` as the single "everything on
  one slide" backup image if a live chart has to be replaced.
- Fonts ≥ 16 pt for axis text; the deck must survive being printed in greyscale,
  which the Okabe-Ito palette does.
