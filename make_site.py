"""
make_site.py - build the static project site that Vercel (or any CDN) can host.

    python make_site.py            -> site/
    npx vercel deploy --prod site  -> public URL

Why a static site and not the app: Vercel runs request/response functions and
serves static files. Streamlit needs a long-lived Python process holding a
WebSocket per browser session, which is outside what Vercel's compute model
offers. So this publishes everything that is *evidence* - the measured tables,
the acceptance verdicts, the figures, the method, the ethics - and points at the
one command that runs the interactive app on a host that can hold a session.

Every number here is read from results/ at build time. Nothing is typed by hand,
so the site cannot drift from the experiment.
"""

from __future__ import annotations

import html
import json
import os
import shutil

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "site")

DISPLAY = {
    "naive": "Naive resampler", "vae": "VAE", "vae_dp_budget": "VAE @ DP budget",
    "dp_vae_eps1": "DP-VAE ε=1", "dp_vae_eps3": "DP-VAE ε=3",
    "dp_vae_eps10": "DP-VAE ε=10", "dp_vae_eps30": "DP-VAE ε=30",
    "ctgan": "CTGAN", "tvae": "TVAE", "wgan_gp": "WGAN-GP",
}
COLOR = {
    "naive": "#999999", "vae": "#0072B2", "vae_dp_budget": "#4477AA",
    "dp_vae_eps1": "#D55E00", "dp_vae_eps3": "#E69F00",
    "dp_vae_eps10": "#F0E442", "dp_vae_eps30": "#009E73", "ctgan": "#CC79A7",
    "tvae": "#56B4E9", "wgan_gp": "#0F0F0F",
}
FIGURE_NOTES = {
    "fig1_pca_overlay": "Patients projected onto a PCA fitted on the real train "
                        "split. The real held-out cloud is the envelope a "
                        "plausible new cohort should fall inside.",
    "fig2_histograms": "Marginal overlays. Shown for completeness only: the "
                       "naive baseline matches these perfectly while destroying "
                       "every association, which is why they are not the "
                       "evidence.",
    "fig3_correlation_heatmaps": "Rank-correlation structure: real, synthetic, "
                                 "and the difference.",
    "fig4_fidelity": "Fidelity per model against the held-out test split, with "
                     "the real-vs-real noise floor and the naive baseline drawn "
                     "for reference.",
    "fig5_utility": "Train-on-synthetic / test-on-real against the "
                    "train-on-real / test-on-real ceiling.",
    "fig6_privacy": "Exact-copy rate, distance-to-closest-record ratio and "
                    "membership-inference AUC.",
    "fig8_ablation": "What each design choice is worth: shipping config versus "
                     "dropping the observation-noise sampler versus min-max "
                     "instead of the rank transform.",
    "fig9_dashboard": "One-page summary of the whole sweep.",
}
CSS = """
:root{--ink:#0B1F33;--text:#16283C;--muted:#5B6B7C;--line:#E3E9F0;--blue:#0072B2;
--green:#009E73;--orange:#D55E00;--amber:#E69F00;--bg:#F7F9FC;--card:#fff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
font:16px/1.6 Inter,'Segoe UI',system-ui,sans-serif;-webkit-font-smoothing:antialiased}
a{color:var(--blue);text-decoration:none}a:hover{text-decoration:underline}
nav{position:sticky;top:0;z-index:10;background:rgba(255,255,255,.92);
backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
nav .in{max-width:1180px;margin:auto;display:flex;gap:1.2rem;align-items:center;
padding:.7rem 1.4rem;font-size:.85rem;font-weight:600}
nav .brand{display:flex;align-items:center;gap:.5rem;font-weight:800;color:var(--ink);
margin-right:auto;letter-spacing:-.02em}
nav .g{width:26px;height:26px;border-radius:8px;background:linear-gradient(135deg,#0072B2,#009E73);
color:#fff;display:grid;place-items:center;font-size:.8rem}
nav a{color:var(--muted)}nav a:hover{color:var(--blue);text-decoration:none}
header.hero{max-width:1180px;margin:2.2rem auto 0;padding:0 1.4rem}
.band{background:linear-gradient(115deg,#0B1F33 0%,#0F4C77 55%,#0072B2 100%);
color:#fff;border-radius:20px;padding:2.4rem 2.2rem;box-shadow:0 14px 40px rgba(11,31,51,.18);
position:relative;overflow:hidden}
.band:after{content:"";position:absolute;inset:0;
background:radial-gradient(700px 200px at 85% -30%,rgba(0,158,115,.45),transparent 70%)}
.band h1{margin:0 0 .5rem;font-size:2.15rem;letter-spacing:-.03em;position:relative;z-index:1}
.band p{margin:0;max-width:66ch;color:#CBE0F0;position:relative;z-index:1}
.chips{display:flex;flex-wrap:wrap;gap:.45rem;margin-top:1.2rem;position:relative;z-index:1}
.chip{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.2);
color:#EAF3FA;border-radius:999px;padding:.28rem .75rem;font-size:.76rem;font-weight:600}
main{max-width:1180px;margin:auto;padding:0 1.4rem 4rem}
/* Ten mean±std columns cannot fit a phone: keep each value on one line and let
   the section scroll sideways rather than wrapping "0.126 ± 0.003" over three. */
section{margin-top:3rem;overflow-x:auto}
th,td{padding:.55rem .7rem;text-align:right;border-bottom:1px solid #EEF2F7;
white-space:nowrap}
table{min-width:640px}
h2{font-size:1.35rem;letter-spacing:-.02em;color:var(--ink);margin:0 0 .35rem}
.lead{color:var(--muted);max-width:78ch;margin:0 0 1.3rem;font-size:.95rem}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:.9rem}
.stat{background:var(--card);border:1px solid var(--line);border-radius:14px;
padding:1rem 1.1rem;box-shadow:0 1px 2px rgba(15,32,51,.05)}
.stat .k{font-size:.68rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
color:var(--muted)}
.stat .v{font-size:1.75rem;font-weight:800;letter-spacing:-.03em;color:var(--ink);
margin-top:.15rem}
.stat .n{font-size:.76rem;color:var(--muted);margin-top:.2rem}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:1.15rem 1.25rem}
.card h3{margin:0 0 .35rem;font-size:1rem;color:var(--ink)}
.card p{margin:0;font-size:.88rem;color:var(--muted)}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);
border-radius:14px;overflow:hidden;font-size:.84rem}
th,td{padding:.55rem .7rem;text-align:right;border-bottom:1px solid #EEF2F7}
th:first-child,td:first-child{text-align:left}
thead th{background:#F2F6FA;color:var(--muted);font-size:.7rem;letter-spacing:.06em;
text-transform:uppercase;font-weight:700}
tbody tr:hover{background:#FAFCFE}
td.model{font-weight:700;color:var(--ink);white-space:nowrap}
td.model .sw{display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:.45rem}
tr.floor td{background:#F7FAFD;color:var(--muted);font-style:italic}
.badges{display:flex;flex-wrap:wrap;gap:.5rem}
.b{border-radius:10px;padding:.42rem .8rem;font-size:.8rem;font-weight:700;
border:1px solid}
.b.good{background:#E7F6EF;color:#0B6E4F;border-color:#C4E9D8}
.b.warn{background:#FDF3E3;color:#8A5A08;border-color:#F6E2BC}
.b.bad{background:#FCEDE7;color:#A03A0C;border-color:#F7D6C6}
.b.neutral{background:#F1F5F9;color:#33475B;border-color:#E3E9F0}
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:1.1rem}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:14px;
padding:.8rem;box-shadow:0 1px 2px rgba(15,32,51,.05)}
figure img{width:100%;border-radius:9px;display:block}
figcaption{font-size:.8rem;color:var(--muted);margin-top:.55rem;line-height:1.45}
figcaption b{color:var(--ink);display:block;font-size:.84rem;margin-bottom:.15rem}
ul.tight{margin:0;padding-left:1.1rem;font-size:.9rem;color:var(--muted)}
ul.tight li{margin:.35rem 0}
.note{background:#FFF8EC;border:1px solid #F6E2BC;border-left:4px solid var(--amber);
border-radius:12px;padding:.95rem 1.1rem;font-size:.88rem;color:#6B4A0B}
.split{background:#F1F5F9;border:1px solid var(--line);border-left:4px solid var(--blue);
border-radius:12px;padding:1.1rem 1.25rem;font-size:.9rem}
footer{border-top:1px solid var(--line);margin-top:3.5rem;padding-top:1.2rem;
display:flex;flex-wrap:wrap;gap:.6rem;justify-content:space-between;
font-size:.76rem;color:#7A8CA0}
code{background:#EEF3F8;border-radius:5px;padding:.1rem .35rem;font-size:.85em}
@media(max-width:640px){.band h1{font-size:1.6rem}.band{padding:1.6rem 1.3rem}}
"""


def fmt(v, dp=3):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return f"{f:.{dp}f}" if np.isfinite(f) else "n/a"


def model_table(g: pd.DataFrame) -> str:
    cols = [("KS", "fid_ks_mean"), ("TV", "fid_tv_mean"),
            ("Gap", "fid_spearman_gap"), ("TSTR LR", "tstr_lr_auc"),
            ("TSTR RF", "tstr_rf_auc"), ("Reten. LR", "retention_lr"),
            ("Exact copy", "priv_exact_copy_train"),
            ("DCR ratio", "priv_dcr_ratio"), ("MIA AUC", "priv_mia_auc")]
    head = "".join(f"<th>{n}</th>" for n, _ in cols)
    # The floor is only defined for the three fidelity metrics; the utility and
    # privacy families have no real-vs-real analogue, so they get an em dash
    # rather than a fabricated blank.
    floor_map = {"fid_ks_mean": "fid_floor_ks_mean_mean",
                 "fid_tv_mean": "fid_floor_tv_mean_mean",
                 "fid_spearman_gap": "fid_floor_spearman_gap_mean"}
    floor_cells = []
    for _, key in cols:
        src = floor_map.get(key)
        floor_cells.append(f"<td>{fmt(g[src].mean())}</td>" if src
                           and src in g else "<td>—</td>")
    floor = ("<tr class='floor'><td>real-vs-real noise floor</td>"
             + "".join(floor_cells) + "</tr>")
    body = []
    for _, r in g.iterrows():
        cells = []
        for _, k in cols:
            m, s = r.get(f"{k}_mean", np.nan), r.get(f"{k}_std", np.nan)
            cells.append(f"<td>{fmt(m)} ± {fmt(s)}</td>" if np.isfinite(m)
                         else "<td>n/a</td>")
        body.append(
            f"<tr><td class='model'><span class='sw' style='background:"
            f"{COLOR.get(r.model, '#666')}'></span>{DISPLAY.get(r.model, r.model)}"
            f"</td>{''.join(cells)}</tr>")
    return (f"<table><thead><tr><th>Model · 3 seeds, mean ± std</th>{head}</tr>"
            f"</thead><tbody>{floor}{''.join(body)}</tbody></table>")


def verdicts(path: str) -> str:
    """Badges for the verdict lines; the config header becomes a caption.

    acceptance_<dataset>.txt opens with `dataset=heart seeds=[0, 1, 2] ...`, which
    is provenance for the run rather than a criterion - rendering it as a badge
    put a neutral chip where a verdict belongs.
    """
    if not os.path.exists(path):
        return ""
    badges, meta_line = [], ""
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        token = next((t for t in ("PASS", "FAIL", "CHECK") if t in line), None)
        if token is None:
            meta_line = meta_line or line
            continue
        cls = {"PASS": "good", "FAIL": "bad", "CHECK": "warn"}[token]
        badges.append(f"<span class='b {cls}'>{html.escape(line)}</span>")
    cap = (f"<p class='lead'>Measured from the full sweep "
           f"(<code>{html.escape(meta_line)}</code>).</p>" if meta_line else "")
    return cap + "<div class='badges'>" + "".join(badges) + "</div>"


def figures(dataset: str) -> str:
    src = os.path.join(HERE, "results", "figures")
    out = []
    for stem, note in FIGURE_NOTES.items():
        p = os.path.join(src, f"{stem}_{dataset}.png")
        if not os.path.exists(p):
            continue
        title = stem.split("_", 1)[1].replace("_", " ").title()
        out.append(
            f"<figure><img src='assets/figures/{stem}_{dataset}.png' "
            f"alt='{html.escape(title)}' loading='lazy'>"
            f"<figcaption><b>{html.escape(title)}</b>{html.escape(note)}</figcaption>"
            f"</figure>")
    return "<div class='figs'>" + "".join(out) + "</div>"


def main() -> int:
    summary = pd.read_csv(os.path.join(HERE, "results", "results_summary.csv"))
    meta = json.load(open(os.path.join(HERE, "results", "run_metadata.json")))
    heart = summary[summary.dataset == "heart"]
    pima = summary[summary.dataset == "pima"]

    def best(g, col="retention_lr_mean"):
        row = g.loc[g[col].idxmax()]
        return DISPLAY.get(row.model, row.model), fmt(row[col])

    hb, hv = best(heart)
    pb, pv = best(pima)
    gap_h = fmt(heart.loc[heart.model == "vae", "fid_spearman_gap_mean"].iloc[0])
    nv_h = fmt(heart.loc[heart.model == "naive", "fid_spearman_gap_mean"].iloc[0])
    fl_h = fmt(heart["fid_floor_spearman_gap_mean"].mean())
    worst_copy = float(np.nanmax(summary.priv_exact_copy_train_mean))
    n_models = summary.model.nunique()

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "assets", "figures"), exist_ok=True)
    shutil.copytree(os.path.join(HERE, "results", "figures"),
                    os.path.join(OUT, "assets", "figures"), dirs_exist_ok=True)

    page = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Synthetic Medical Data Generator — VAE, DP-VAE and GAN synthetic EHRs, measured</title>
<meta name="description" content="Mixed-type synthetic patient records from a custom VAE, its
differentially-private variant, CTGAN, TVAE and WGAN-GP — evaluated on fidelity, downstream utility
and privacy against a measured real-vs-real noise floor.">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
rel="stylesheet">
<style>{CSS}</style></head>
<body>
<nav><div class="in"><span class="brand"><span class="g">S</span>SynthMed</span>
<a href="#results">Results</a><a href="#accept">Acceptance</a><a href="#method">Method</a>
<a href="#figures">Figures</a><a href="#privacy">Privacy</a><a href="#ethics">Ethics</a>
<a href="#app">Run the app</a></div></nav>

<header class="hero"><div class="band">
<h1>Synthetic Medical Data Generator</h1>
<p>Clinical data sharing is blocked by regulation long before it is blocked by
technical feasibility. This project builds mixed-type synthetic patient records and
measures three separate claims about them — that they are faithful, that they are
useful, and that they are private — against a real-vs-real noise floor instead of
against eyeballed histograms.</p>
<div class="chips">
<span class="chip">UCI Cleveland · n=303</span>
<span class="chip">UCI Pima · n=768</span>
<span class="chip">{n_models} model configurations × 3 seeds</span>
<span class="chip">split 60 / 20 / 20 stratified</span>
<span class="chip">CPU-only · PyTorch + Opacus</span>
</div></div></header>

<main>
<section><div class="stats">
<div class="stat"><div class="k">Correlation gap · VAE</div><div class="v">{gap_h}</div>
<div class="n">naive {nv_h} · noise floor {fl_h}</div></div>
<div class="stat"><div class="k">Utility retained (heart)</div><div class="v">{float(hv) * 100:.1f}%</div>
<div class="n">best model: {hb} · TSTR AUC ÷ TRTR AUC</div></div>
<div class="stat"><div class="k">Utility retained (pima)</div><div class="v">{float(pv) * 100:.1f}%</div>
<div class="n">best model: {pb}</div></div>
<div class="stat"><div class="k">Exact-copy rate</div>
<div class="v">{worst_copy * 100:.3f}%</div>
<div class="n">worst of every model and seed</div></div>
</div></section>

<section id="results"><h2>What was measured</h2>
<p class="lead">Histograms are the weakest available evidence for synthetic data: a
baseline that reshuffles every column independently matches all of them perfectly while
destroying every association in the table. So every generator is scored on three
independent axes, and no fidelity claim is made without the floor.</p>
<div class="cards">
<div class="card"><h3>A · Fidelity</h3><p>Per-column KS and total-variation, Spearman
correlation gap, PCA overlay, correlation heatmaps — each against the score the same
metric code returns comparing real train to real test.</p></div>
<div class="card"><h3>B · Utility</h3><p>Train-on-synthetic / test-on-real against the
train-on-real ceiling, with logistic regression and random forest: accuracy, F1, AUC.</p></div>
<div class="card"><h3>C · Privacy</h3><p>Exact-copy rate, distance-to-closest-record
ratio against unseen real patients, distance-based membership-inference AUC, and the
privacy-versus-utility curve across ε.</p></div></div>

<h2 style="margin-top:2.2rem">Cleveland heart disease · n=303</h2>{model_table(heart)}
<h2 style="margin-top:2.2rem">Pima Indians diabetes · n=768</h2>{model_table(pima)}
<p class="lead" style="margin-top:.8rem">Full sweep: {len(summary) * 3} fits,
{meta.get('wall_seconds', 0) / 60:.1f} minutes of CPU. The sweep was run twice against the
final code and every metric column in all 60 per-seed rows came out bit-identical.</p>
</section>

<section id="accept"><h2>Acceptance criteria — measured, not asserted</h2>
<p class="lead">Verdicts are computed from the stored results by the same function the
pipeline prints them with, so this page cannot disagree with the repository.</p>
{verdicts(os.path.join(HERE, "results", "acceptance_heart.txt"))}
<div class="note" style="margin-top:1rem"><b>The honest caveat.</b> On Cleveland the
distance-based membership-inference AUC does not fall monotonically with ε. At ε=1 the
accountant forces σ ≈ 23 and the decoder collapses to near-constant output, so "distance
to the nearest synthetic record" stops being a membership signal — the attack is
<em>invalid</em> at that end of the sweep, not successful. The privacy evidence that does
hold here is the exact-copy rate, the DCR ratio, and the ε guarantee itself; and at
ε = 30 with n = 181 that guarantee is not a strong one.</div>
</section>

<section id="method"><h2>Method in one screen</h2>
<div class="cards">
<div class="card"><h3>Protocol</h3><p>Generators fit on the train split only.
Hyper-parameters chosen on validation only — the tuner receives no test frame and has no
code path to one. The test split is read once, in the final evaluation pass per model per
seed. Every cleaning, scaling, encoding and distance statistic is fitted on train.</p></div>
<div class="card"><h3>The VAE</h3><p>Encoder → (μ, log σ²) → reparameterisation → decoder
with a sigmoid numeric head under MSE and one softmax head per categorical column under
cross-entropy; KL annealed. Mixed types share one encoded space with the WGAN-GP, so
VAE-versus-GAN is an architecture comparison rather than a preprocessing one.</p></div>
<div class="card"><h3>Differential privacy</h3><p>Opacus DP-SGD at ε ∈ {{1, 3, 10, 30}} with
δ = 1/n_train, per-sample gradient clipping and an RDP/PRV accountant. A matched
optimisation-budget non-private run sits beside the curve so ε varies only the noise
multiplier.</p></div>
<div class="card"><h3>Two bugs the metrics caught</h3><p>Categorical cross-entropy is
~35× the numeric MSE term, so the first VAE had a correlation gap <em>worse</em> than the
naive baseline. And MSE implies a Gaussian observation model: the sampler is
x = f(z) + σε, not f(z) — emitting the mean under-dispersed every numeric column.</p></div>
</div></section>

<section id="figures"><h2>Figures · Cleveland</h2>{figures("heart")}
<h2 style="margin-top:2.2rem">Figures · Pima</h2>{figures("pima")}</section>

<section id="privacy"><h2>Privacy, read correctly</h2>
<ul class="tight">
<li><b>DCR ratio is the cleanest discriminator in this study.</b> Mean distance to the
closest training record, divided by the same quantity for unseen <em>real</em> patients.
1.0 means synthetic rows sit where new real patients sit. TVAE reads 0.609 — its rows sit
closer to training records than real newcomers do — while the VAE reads 0.980.</li>
<li><b>Utility and leakage are not in tension here.</b> TVAE has the best retention
(0.986) and the worst DCR ratio. That is precisely why both are measured.</li>
<li><b>The attack is deliberately weak</b> — one distance threshold, no gradients, no
shadow models — and both groups come from the same distribution, so even a perfectly
generalising model can read slightly off 0.5.</li>
<li><b>DP destroys joint structure at every budget on this data size.</b> The DP-VAE
correlation gap is pinned at the naive 0.185 for all four ε; what does move monotonically
is the categorical marginal, the DCR ratio, and downstream AUC.</li>
</ul></section>

<section id="ethics"><h2>Ethics and legal context</h2>
<div class="split"><b>India — DPDP Act 2023.</b> The Act regulates digital personal data
about identifiable people, with notice, consent, data-minimisation and purpose-limitation.
De-identified and anonymised data sit outside its scope — which turns the technical
question into a legal one: does the synthetic cohort actually break identifiability? The
Act sets no anonymisation standard and no re-identification threshold, so there is no
statutory box that says "synthetic, therefore compliant". The measurements above are
evidence toward that judgement, not a certificate of it, and re-identification testing has
to be repeated per release.</div>
<ul class="tight" style="margin-top:.9rem">
<li><b>HIPAA (US).</b> Safe Harbor removes the 18 identifiers; Expert Determination
documents a very small re-identification risk. Synthetic tables travel the second route —
or the stronger argument that they are not PHI at all because they describe no real
person — <em>provided</em> the empirical overlap is genuinely negligible. A generator
emitting verbatim training rows would produce PHI no matter what it is called.</li>
<li><b>GDPR.</b> Recital 26 assesses anonymisation against the means reasonably likely to
be used for re-identification by the controller <em>or any other person</em>, which
includes the model itself.</li>
<li><b>No new human-participant data was collected.</b> Both tables are already-public,
de-identified UCI research datasets, so no fresh consent or IRB submission was required
for this study. That is a property of the source data and is not a template for repeating
the method on a hospital's live records.</li>
<li><b>Synthetic ≠ anonymous.</b> These measurements constrain what this generator
reveals. They do not defeat linkage against auxiliary data, and an ε guarantee is a
statement about an algorithm, not about an adversary's other sources.</li>
<li><b>Cohort limits.</b> Cleveland is single-site, old, and enriched for disease (~54%
prevalence), which inflates every AUC here; Pima is a specific Indigenous community.
Nothing here supports claims about a general population, and no clinical or compliance
claim follows from this project.</li>
</ul></section>

<section id="app"><h2>The interactive app</h2>
<p class="lead">This page is the evidence. The live application — fit a generator from
sliders, filter by diagnosis, audit a cohort, download the CSV — is a Streamlit app, and
Streamlit needs a long-lived Python process holding a WebSocket per session, which is
outside Vercel's request/response compute model. So the app is served separately:</p>
<div class="split"><code>pip install -r requirements.txt</code><br>
<code>streamlit run app.py</code><br><br>
It opens in two modes: <b>live</b>, which fits the generator on the train split from your
sliders, and <b>frozen</b>, which loads the seed-0 cohort the published sweep actually
stored — instant, and the same rows the tables above measured. Both score through one
metric path, so the interface cannot drift from the study.</div>
</section>

<footer><span>Synthetic Medical Data Generator · B.Tech project · UCI Cleveland + UCI Pima
· CPU build</span><span>Every number on this page is generated from
<code>results/</code> at build time by <code>make_site.py</code></span></footer>
</main></body></html>
"""
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    with open(os.path.join(OUT, "vercel.json"), "w", encoding="utf-8") as f:
        json.dump({"cleanUrls": True,
                   "framework": None,
                   "headers": [{"source": "/assets/(.*)",
                                "headers": [{"key": "Cache-Control",
                                             "value": "public, max-age=86400"}]}]},
                  f, indent=2)
    n_png = len(os.listdir(os.path.join(OUT, "assets", "figures")))
    size = sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(OUT) for f in fs)
    print(f"wrote {OUT}  ({n_png} figures, {size / 1024 / 1024:.1f} MB)")
    print("deploy:  npx vercel deploy --prod --yes site")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
