"""
ui.theme - design tokens and the stylesheet.

Why tokens are in one place: the paper figures (make_figures.py) and this app
must show the same model in the same colour, or a reviewer comparing fig4 with
the Fidelity tab is being shown two different things. The Okabe-Ito palette is
colour-blind-safe and is the one the figures already use.

Streamlit's own theming (config.toml) only reaches a handful of surfaces, so the
rest is CSS. Every selector here is keyed on a `data-testid`, not on a class
name, because class names change between Streamlit versions and testids do not.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
# Okabe-Ito: the eight-hue set that stays distinguishable under deuteranopia,
# protanopia and tritanopia.
OKABE_ITO = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "bluish_green": "#009E73",
    "purple": "#CC79A7",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#0F0F0F",
}

# Brand chrome. Blue is deliberately the same hue as OKABE_ITO["blue"] so the
# interface and the data share one accent.
BRAND = {
    "primary": OKABE_ITO["blue"],
    "primary_dark": "#005788",
    "accent": OKABE_ITO["vermillion"],
    "good": OKABE_ITO["bluish_green"],
    "warn": OKABE_ITO["orange"],
    "bad": OKABE_ITO["vermillion"],
    "ink": "#0B1F33",
    "text": "#16283C",
    "muted": "#5B6B7C",
    "line": "#E3E9F0",
    "surface": "#FFFFFF",
    "surface_alt": "#F5F8FB",
    "sidebar": "#0B1F33",
}

# Model -> colour, shared with make_figures.py.
MODEL_COLOR = {
    "naive": "#999999",
    "vae": OKABE_ITO["blue"],
    "vae_dp_budget": "#4477AA",
    "dp_vae_eps1": OKABE_ITO["vermillion"],
    "dp_vae_eps3": OKABE_ITO["orange"],
    "dp_vae_eps10": OKABE_ITO["yellow"],
    "dp_vae_eps30": OKABE_ITO["bluish_green"],
    "ctgan": OKABE_ITO["purple"],
    "tvae": OKABE_ITO["sky_blue"],
    "wgan_gp": OKABE_ITO["black"],
}
FALLBACK = [OKABE_ITO[k] for k in
            ("blue", "vermillion", "bluish_green", "purple", "orange",
             "sky_blue", "black")]


def palette_for(name: str) -> str:
    """Stable colour for a model label, with a deterministic fallback."""
    if name in MODEL_COLOR:
        return MODEL_COLOR[name]
    return FALLBACK[sum(ord(c) for c in name) % len(FALLBACK)]


# ---------------------------------------------------------------------------
# Plotly layout defaults
# ---------------------------------------------------------------------------
PLOT = {
    "font_family": "Inter, 'Segoe UI', system-ui, sans-serif",
    "font_size": 12.5,
    "colorway": [OKABE_ITO[k] for k in
                 ("blue", "vermillion", "bluish_green", "purple", "orange",
                  "sky_blue", "black")],
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "grid": "#E9EEF4",
    "axis": "#5B6B7C",
    "margin": dict(l=8, r=8, t=44, b=8, autoexpand=True),
}

# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], button, input, select {
  font-family: Inter, 'Segoe UI', system-ui, -apple-system, sans-serif;
}
#MainMenu, header [data-testid="stDeployButton"], footer { visibility: hidden; }
/* Streamlit's default 2 px rainbow strip along the top edge. */
#stDecoration { display: none; }
[data-testid="stHeader"] { background: transparent; height: 3.1rem; }
[data-testid="stAppViewContainer"] > .main { padding: 1.6rem 2.4rem 4rem; }
.block-container { padding-top: 0; max-width: 1400px; }

/* ---------- sidebar: dark control deck ---------- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0B1F33 0%, #10283F 100%);
  border-right: 1px solid rgba(255,255,255,.06);
}
[data-testid="stSidebar"] * { color: #E8EEF5; }
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 { color: #FFFFFF; letter-spacing: -.02em; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] small { color: #9DB2C6 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] [data-baseweb="slider"] {
  background: rgba(255,255,255,.05) !important;
  border-color: rgba(255,255,255,.16) !important;
  border-radius: 10px !important;
}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
  font-size: .8rem; font-weight: 600; letter-spacing: .01em;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.12); }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
  padding: .35rem .7rem; border-radius: 8px; background: rgba(255,255,255,.05);
  margin-right: .4rem;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) {
  background: #0072B2; color: #fff;
}
div[data-testid="stSidebarContent"] { padding-top: 1.4rem; }

/* brand mark at the top of the sidebar */
.brand { display: flex; align-items: center; gap: .65rem; margin-bottom: .2rem; }
.brand .glyph {
  width: 34px; height: 34px; border-radius: 10px; flex: 0 0 auto;
  background: linear-gradient(135deg, #0072B2, #009E73);
  display: grid; place-items: center; color: #fff; font-weight: 800;
  box-shadow: 0 2px 10px rgba(0,114,178,.35);
}
.brand .name { font-size: 1.02rem; font-weight: 700; color: #fff;
  letter-spacing: -.02em; line-height: 1.1; }
.brand .sub { font-size: .68rem; color: #8FA6BC; font-weight: 600;
  text-transform: uppercase; letter-spacing: .12em; }

/* ---------- hero ---------- */
.hero {
  position: relative; overflow: hidden; border-radius: 18px;
  padding: 1.5rem 1.7rem; margin-bottom: 1.2rem;
  background: linear-gradient(115deg, #0B1F33 0%, #0F4C77 55%, #0072B2 100%);
  color: #fff; box-shadow: 0 10px 30px rgba(11,31,51,.18);
}
.hero:after {
  content: ""; position: absolute; inset: 0; pointer-events: none;
  background: radial-gradient(700px 190px at 88% -30%, rgba(0,158,115,.42),
              transparent 70%);
}
.hero h1 { font-size: 1.62rem; font-weight: 800; letter-spacing: -.03em;
  margin: 0 0 .35rem; color: #fff; position: relative; z-index: 1; }
.hero p  { font-size: .93rem; color: #C9DCEC; margin: 0; max-width: 62ch;
  line-height: 1.55; position: relative; z-index: 1; }
.hero .meta { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .95rem;
  position: relative; z-index: 1; }

/* ---------- chips / pills ----------
   Two surfaces, deliberately separate rules. A pill under the hero sits on
   white, so it needs a light fill; the same class inside the hero band sits on
   navy, so it needs the translucent inverse. Writing only the dark variant made
   the status row invisible against the page. */
.pill {
  display: inline-flex; align-items: center; gap: .38rem; padding: .26rem .7rem;
  border-radius: 999px; font-size: .735rem; font-weight: 600;
  background: #F1F5F9; color: #33475B; border: 1px solid #E3E9F0;
  white-space: nowrap;
}
.pill .dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.pill.live  { background: #E7F6EF; border-color: #C4E9D8; color: #0B6E4F; }
.pill.frozen{ background: #EAF2FB; border-color: #CFE4F5; color: #0B5D91; }
.hero .pill { background: rgba(255,255,255,.12); border-color: rgba(255,255,255,.18);
  color: #EAF3FA; }
.hero .pill.solid { background: #fff; color: #0B1F33; border-color: #fff; }
.chip {
  display: inline-flex; align-items: center; gap: .35rem; padding: .2rem .6rem;
  border-radius: 8px; font-size: .74rem; font-weight: 600;
  background: #F1F5F9; color: #33475B; border: 1px solid #E3E9F0;
}
.chip.good { background: #E7F6EF; color: #0B6E4F; border-color: #C7EBDC; }
.chip.warn { background: #FDF3E3; color: #8A5A08; border-color: #F6E2BC; }
.chip.bad  { background: #FCEDE7; color: #A03A0C; border-color: #F7D6C6; }

/* ---------- KPI cards (built on st.metric so it stays accessible) ---------- */
[data-testid="stMetric"] {
  background: #fff; border: 1px solid #E6EBF2; border-radius: 14px;
  padding: .85rem 1rem .9rem; box-shadow: 0 1px 2px rgba(15,32,51,.05);
  transition: box-shadow .18s ease, transform .18s ease;
}
[data-testid="stMetric"]:hover { box-shadow: 0 6px 18px rgba(15,32,51,.09);
  transform: translateY(-1px); }
[data-testid="stMetricLabel"] p { font-size: .7rem !important; font-weight: 700;
  letter-spacing: .07em; text-transform: uppercase; color: #5B6B7C; }
[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 800;
  letter-spacing: -.03em; color: #0B1F33; }
[data-testid="stMetricValue"] > div { font-size: 1.5rem; font-weight: 800; }
[data-testid="stMetricDelta"] { font-weight: 700; }

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: .3rem; border-bottom: 1px solid #E3E9F0;
  padding-bottom: .25rem; }
.stTabs [data-baseweb="tab"] {
  height: auto; padding: .5rem 1.05rem; border-radius: 10px;
  background: #F5F8FB; border: 1px solid transparent; font-weight: 600;
  font-size: .88rem; color: #4C6076;
}
.stTabs [data-baseweb="tab"]:hover { background: #EAF1F8; }
.stTabs [aria-selected="true"] { background: #0072B2 !important; color: #fff !important;
  box-shadow: 0 4px 14px rgba(0,114,178,.28); }
.stTabs [data-baseweb="tab"] p { font-size: .88rem !important; font-weight: 700 !important;
  color: inherit !important; margin: 0 !important; }

/* ---------- content ---------- */
h2, h3 { letter-spacing: -.02em; color: #0B1F33; }
[data-testid="stSubheader"] { font-weight: 700 !important; }
.sec { display: flex; align-items: center; gap: .6rem; margin: 1.35rem 0 .55rem; }
.sec .t { font-size: .98rem; font-weight: 800; color: #0B1F33;
  letter-spacing: -.02em; }
.sec .n { font-size: .66rem; font-weight: 800; color: #fff; background: #0072B2;
  border-radius: 6px; padding: .1rem .38rem; letter-spacing: .06em; }
.sec .rule { flex: 1; height: 1px; background: #E3E9F0; }
.small { font-size: .8rem; color: #5B6B7C; line-height: 1.5; }
[data-testid="stCaptionContainer"] { font-size: .79rem; color: #5B6B7C; }
hr { border-color: #E9EEF4; margin: 1.1rem 0; }

/* ---------- data ---------- */
[data-testid="stDataFrame"] { border: 1px solid #E6EBF2; border-radius: 12px;
  overflow: hidden; padding: .2rem; }
[data-testid="stExpander"] { border: 1px solid #E6EBF2; border-radius: 12px;
  background: #FBFCFE; margin-bottom: .5rem; }
[data-testid="stExpander"] summary p { font-weight: 600; color: #33475B; }
[data-testid="stAlert"] { border-radius: 12px; border-left-width: 4px;
  font-size: .86rem; }
.stProgress > div > div > div > div { background: #0072B2; }

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button {
  border-radius: 10px; font-weight: 700; border: 1px solid #D8E1EA;
  background: #fff; color: #0B1F33; padding: .5rem 1.05rem;
  transition: all .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: #0072B2; color: #0072B2; box-shadow: 0 2px 10px rgba(0,114,178,.12);
}
.stDownloadButton > button[kind="primary"], .stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #0072B2, #0B6E9E); color: #fff;
  border-color: transparent; box-shadow: 0 4px 14px rgba(0,114,178,.3);
}
.stDownloadButton > button[kind="primary"]:hover,
.stButton > button[kind="primary"]:hover { color: #fff; filter: brightness(1.06); }

/* ---------- footer ---------- */
.foot { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center;
  justify-content: space-between; margin-top: 2.2rem; padding-top: 1rem;
  border-top: 1px solid #E9EEF4; font-size: .74rem; color: #7A8CA0; }
"""

INJECT = f"<style>{CSS}</style>"
