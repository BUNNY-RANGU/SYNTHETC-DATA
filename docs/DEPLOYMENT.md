# Deployment

The web app is `app.py`. Everything else in this repository is either the
library it calls (`synth/`, `ui/`) or the artefacts it reads (`results/`,
`data/`).

## What the deployed app needs to contain

| Path | Why it must be in the deploy bundle |
|---|---|
| `app.py`, `synth/`, `ui/`, `.streamlit/config.toml` | the application |
| `data/` | the two raw UCI tables. The loaders verify shape on read and do **not** re-download, so a host without outbound fetches still works. |
| `results/` | the frozen 3-seed artefacts. "Published cohort" mode and the whole Model-comparison tab read these CSVs. Without them the app still runs, in live mode only. |
| `requirements.txt` | pinned-enough dependency list |

`docs/`, `tests/` and the notebook are not needed at runtime, but they cost
nothing to ship and they are what makes the deployed app defensible in a viva.

---

## Path A - Streamlit Community Cloud (recommended: free, official, ~4 minutes)

1. Push this folder to a **public** GitHub repository.
2. On <https://share.streamlit.io>, sign in with the same GitHub account →
   **Deploy a new app** → pick that repository.
3. Set **Main file path** to `app.py` and **Python version** to 3.11.
4. Deploy. First build takes ~3–5 minutes (it installs torch).

To choose the URL, edit the project name in the app dashboard before or after
the first deploy; the address is `<account>-<app-name>-<hash>.streamlit.app`.

Updating: `git push` to the branch you connected. Streamlit rebuilds and
redeploys automatically.

Things that go wrong, in the order people hit them:

- **"Import error: no module named synth"** — the main file path is not `app.py`
  at the repository root. `app.py` must be the entry point, not a copy inside a
  subfolder, because it does `sys.path.insert(0, dirname(abspath(__file__)))`.
- **White screen for a minute, then content** — normal cold start: torch,
  opacus, ctgan and scikit-learn are all imported before the first render.
- **OOM / restart while training a model** — free instances are ~800 MB to 1 GB.
  Keep **Training speed = Fast** (the default) and cohorts under ~2 000 rows, or
  use **frozen mode**, which trains nothing.
- **The comparison tab is empty** — `results/` was not committed.

---

## Path B - Hugging Face Spaces (free, no credit card, good for a shareable link)

Same repository, but the Space reads its app configuration from the **front
matter of the root `README.md`**. Add this block above the existing README title
(for a Space only — it is harmless in GitHub, it just renders as a YAML block):

```yaml
---
title: Synthetic Medical Data Generator
emoji: "&#129656;"
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.30.0"
app_file: app.py
pinned: false
license: mit
short_description: VAE / DP-VAE / CTGAN synthetic EHRs, measured
---
```

Then, with the `hf` CLI installed and `hf auth login` done once:

```bash
hf spaces create synthmed-demo --sdk streamlit --repo-type model 2>/dev/null || true
git remote add hf https://huggingface.co/spaces/<your-user>/synthmed-demo
git push hf main
```

CPU-only is the free default and this project is CPU-only, so nothing extra is
needed. Do **not** request ZeroGPU: it changes the hardware but not the RAM
ceiling you care about, and the published numbers would then come from a
different device than the ones in `README.md`.

---

## Path C - Render (blueprint in the repo)

`render.yaml` at the repository root is a Render blueprint.
**New + → Blueprint →** connect the repo and it creates the service with the
health check, the start command and the region already set.

`plan: starter` is deliberate: the free 512 MB tier can be OOM-killed while the
DP-VAE loads. If you want free-only, set `plan: free`, switch the app to frozen
mode, and expect restarts on live training.

---

## Path D - Container anywhere (Railway, Fly.io, a VPS, your college server)

```bash
docker build -t synthmed .
docker run -p 8501:8501 synthmed
```

The image installs torch from the CPU wheel index, runs as a non-root user, and
declares a `HEALTHCHECK` against Streamlit's own `/_stcore/health`.

For Railway: **New → Deploy from GitHub repo**, detector picks the Dockerfile.
For Fly.io: `fly launch --from-dockerfile synthmed && fly deploy`.

---

## Path E - Serve it yourself (no third party, works for a live demo)

```bash
python -m streamlit run app.py \
    --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

Same machine: <http://localhost:8501>. Other devices on the same network:
`http://<your-LAN-IP>:8501`. This is the most reliable option for a lab
demonstration, because it does not depend on anyone else's uptime or a cold
start in front of an examiner.

To make it survive a reboot on Linux:

```ini
# /etc/systemd/system/synthmed.service
[Unit]
Description=Synthetic Medical Data Generator
After=network.target

[Service]
WorkingDirectory=/opt/GEN_AI
ExecStart=/opt/GEN_AI/.venv/bin/python -m streamlit run app.py \
  --server.address 0.0.0.0 --server.port 8501 --server.headless true
Restart=always
User=appuser

[Install]
WantedBy=multi-user.target
```

Exposing it beyond your network needs TLS and access control — put it behind
Caddy or nginx with a basic-auth or a VPN, not bare on a public port. The app
itself has no authentication, because it holds no real patient data: everything
it renders is generated or already-public UCI research data.

---

## Post-deploy checklist

Run these five checks on the live URL before you quote it to anyone. The first
two are scripted in `tests/test_app.py`, which starts a local server, polls the
health endpoint and confirms the page skeleton.

| # | Check | Expected |
|---|---|---|
| 1 | `<url>/_stcore/health` | `ok` |
| 2 | Open Fidelity tab | hero band, 4 KPI cards, KS + TV bars, marginal overlays, PCA overlay, three correlation heatmaps |
| 3 | Switch dataset to `pima` | reloads, all panels redraw, comparison table changes to 10 models |
| 4 | Switch generator through all six, frozen mode | no traceback panel appears; every cohort scores |
| 5 | Download CSV, reopen it | row count matches the slider, integer columns are integers, categories inside the legal levels |

If check 4 ever shows a traceback, the panel prints the real exception — that is
the intended failure mode, not a silent blank page.

## Keeping the deployed numbers honest

The app labels which numbers are which: a **live sandbox** pill means a single
seed fitted from your sliders, a **published cohort** pill means the stored seed-0
table from the 3-seed sweep, and the Model-comparison tab is always the 3-seed
mean ± std. Do not deploy a fork that drops those pills: a live single-seed
number next to a published mean, unlabelled, is the one thing an examiner will
pick on.
