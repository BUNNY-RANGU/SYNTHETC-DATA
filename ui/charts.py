"""
ui.charts - every figure the app draws, as plotly objects.

Separate from `make_figures.py` on purpose: that module writes static print-
quality PNGs, this one draws interactive screen figures with hover values. They
share `ui.theme.palette_for`, which is the part that must not drift.

Nothing here computes a metric - the numbers arrive already measured from
`synth.metrics`. A charting module that also recomputes statistics is where a
demo and the paper quietly disagree.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui.theme import OKABE_ITO, palette_for

GRID = "#E9EEF4"
INK = "#0B1F33"
MUTED = "#5B6B7C"


def _bar(x, y, name, color, **kw) -> go.Bar:
    return go.Bar(x=x, y=y, name=name, marker_color=color,
                  marker_line_width=0, hovertemplate="%{x}<br>%{y:.4f}"
                  "<br><extra></extra>", **kw)


def ks_or_tv(ks: pd.DataFrame, floor: pd.DataFrame, col: str,
             title: str, ylab: str) -> go.Figure:
    """Per-column distribution distance, synthetic against the noise floor."""
    fig = go.Figure()
    fig.add_trace(_bar(ks.column, ks[col], "synthetic", OKABE_ITO["blue"]))
    fig.add_trace(_bar(floor.column, floor[col], "real vs real (floor)",
                       "#B9C4CF"))
    fig.update_layout(barmode="group", title=title,
                      yaxis_title=ylab, legend=dict(orientation="h", y=1.12))
    return fig


def class_balance(counts: Sequence[int], labels: Sequence[str]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(labels), y=list(counts),
                         marker_color=[OKABE_ITO["vermillion"],
                                       OKABE_ITO["blue"]],
                         text=[f"{c:,}" for c in counts],
                         textposition="outside", textfont=dict(color=INK),
                         hovertemplate="%{x}<br>%{y:,} rows<extra></extra>",
                         showlegend=False))
    fig.update_layout(title="Class balance of the generated cohort",
                      yaxis_title="rows", yaxis_range=[0, max(counts) * 1.18])
    return fig


def marginals(test: pd.DataFrame, syn: pd.DataFrame, cols: List[str]
              ) -> go.Figure:
    """Histogram overlays. One row of small multiples, shared-free axes."""
    fig = make_subplots(rows=1, cols=len(cols), subplot_titles=cols)
    bins = 24
    for i, c in enumerate(cols, start=1):
        lo = float(np.nanmin([np.nanmin(test[c]), np.nanmin(syn[c])]))
        hi = float(np.nanmax([np.nanmax(test[c]), np.nanmax(syn[c])]))
        edges = np.linspace(lo, hi, bins + 1)
        for frame, name, color, width in (
                (test, "real held-out", INK, 2.4),
                (syn, "synthetic", OKABE_ITO["vermillion"], 2.0)):
            hist, _ = np.histogram(frame[c].to_numpy(float), bins=edges,
                                   density=True)
            centres = 0.5 * (edges[1:] + edges[:-1])
            fig.add_trace(go.Scatter(
                x=centres, y=hist, name=name, legendgroup=name,
                showlegend=(i == 1), line=dict(color=color, width=width),
                hovertemplate=f"{c}<br>density %{{y:.3f}}<extra></extra>"),
                row=1, col=i)
        fig.update_xaxes(range=[lo, hi], showgrid=True, gridcolor=GRID)
        fig.update_yaxes(showgrid=True, gridcolor=GRID,
                         title_text="density" if i == 1 else None)
    fig.update_layout(height=300, margin=dict(l=8, r=8, t=44, b=52),
                      legend=dict(orientation="h", yanchor="top", y=-0.16,
                                  xanchor="center", x=0.5,
                                  bgcolor="rgba(0,0,0,0)"),
                      title="Marginal overlays against the held-out real data")
    return fig


def pca(pca_data: Dict) -> go.Figure:
    fig = go.Figure()
    e = pca_data["explained"]
    traces = [("train", "real train (the basis)", "#C3CDD7", 6, "circle"),
              ("heldout", "real held-out", INK, 11, "square"),
              ("syn", "synthetic", OKABE_ITO["vermillion"], 8, "circle")]
    for key, name, color, size, symbol in traces:
        pts = pca_data[key]
        fig.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1], mode="markers", name=name,
            marker=dict(color=color, size=size, symbol=symbol,
                        opacity=0.62 if key != "heldout" else 0.9,
                        line=dict(width=0)),
            hovertemplate=f"{name}<br>PC1 %{{x:.2f}}<br>PC2 %{{y:.2f}}"
                          "<extra></extra>"))
    fig.update_layout(
        title="Patients projected onto a PCA fitted on real train",
        xaxis_title=f"PC1 ({100 * e[0]:.0f}% variance)",
        yaxis_title=f"PC2 ({100 * e[1]:.0f}% variance)",
        legend=dict(orientation="h", y=1.14), height=430)
    return fig


def correlations(real: pd.DataFrame, syn: pd.DataFrame) -> go.Figure:
    """Real, synthetic and the difference - three heatmaps, one colour scale."""
    names = list(real.columns)
    diff = syn.to_numpy(float) - real.to_numpy(float)
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["real (held-out)", "synthetic",
                                        "difference"],
                        shared_yaxes=True)
    mats = [(real.to_numpy(float), "RdBu", False),
            (syn.to_numpy(float), "RdBu", False),
            (diff, "PuOr", False)]
    for i, (M, cs, _) in enumerate(mats, start=1):
        v = None if i == 3 else [-1, 1]
        if i == 3:
            v = [-np.nanmax(np.abs(diff)) or -1, np.nanmax(np.abs(diff)) or 1]
        fig.add_trace(go.Heatmap(z=M, x=names, y=names, colorscale=cs,
                                 zmin=v[0], zmax=v[1], showscale=(i == 3),
                                 showlegend=False,
                                 colorbar=dict(thickness=10, len=0.9,
                                               title="Δ" if i == 3 else None),
                                 hovertemplate="%{x} vs %{y}<br>"
                                               "%{z:.3f}<extra></extra>"),
                      row=1, col=i)
    for i in range(1, 4):
        fig.update_xaxes(tickangle=90, tickfont=dict(size=8), row=1, col=i)
        fig.update_yaxes(tickfont=dict(size=8), row=1, col=i)
    fig.update_layout(height=420, margin=dict(l=8, r=8, t=64, b=8),
                      title="Rank-correlation structure")
    return fig


def utility(res: Dict, ceiling_note: str = "TRTR AUC ceiling") -> go.Figure:
    clf_names = ["Logistic Regression", "Random Forest"]
    fig = go.Figure()
    for name, key, color in (("TSTR accuracy", "accuracy", OKABE_ITO["sky_blue"]),
                             ("TSTR F1", "f1", OKABE_ITO["blue"]),
                             ("TSTR AUC", "auc", OKABE_ITO["bluish_green"])):
        fig.add_trace(_bar(clf_names,
                           [res[f"tstr_lr_{key}"], res[f"tstr_rf_{key}"]],
                           name, color))
    # The ceiling markers must be placed with the same *category* labels as the
    # bars. Passing numeric x here makes plotly invent a second, linear x-axis
    # and the ceiling line silently detaches from the groups it describes.
    fig.add_trace(go.Scatter(
        x=clf_names,
        y=[res["trtr_lr_auc"], res["trtr_rf_auc"]],
        mode="markers+lines", name=ceiling_note,
        line=dict(color=INK, width=2, dash="dash"),
        marker=dict(symbol="line-ew", size=26, color=INK,
                    line=dict(width=2.4)),
        hovertemplate="ceiling %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group", title="Train-on-synthetic, test-on-real",
                      yaxis_title="score", yaxis_range=[0, 1.06],
                      legend=dict(orientation="h", y=1.16))
    return fig


def dcr(d_real: np.ndarray, d_syn: np.ndarray, threshold: float) -> go.Figure:
    fig = go.Figure()
    bins = np.linspace(0, float(np.nanmax([np.nanmax(d_syn),
                                           np.nanmax(d_real)])) or 1.0, 40)
    for arr, name, color in ((d_real[np.isfinite(d_real)],
                              "real held-out → train", INK),
                             (d_syn[np.isfinite(d_syn)],
                              "synthetic → train", OKABE_ITO["vermillion"])):
        fig.add_trace(go.Histogram(x=arr, xbins=dict(start=bins[0],
                               end=bins[-1], size=(bins[-1] - bins[0]) / 39),
                               name=name, histnorm="density",
                               marker_color=color, opacity=0.92,
                               hovertemplate=f"{name}<br>%{{x}}: %{{y}}"
                                             "<extra></extra>"))
    fig.add_vline(x=threshold, line=dict(color=OKABE_ITO["purple"], width=2,
                                         dash="dash"),
                  annotation_text="memorisation threshold (5th pct of real)",
                  annotation_position="top right",
                  annotation_font_color=OKABE_ITO["purple"])
    fig.update_layout(title="Distance to the closest training record",
                      xaxis_title="scaled Euclidean distance",
                      yaxis_title="density", height=330,
                      legend=dict(orientation="h", y=1.14))
    return fig


def epsilon_curve(curve: pd.DataFrame) -> go.Figure:
    """Attack AUC and utility against the privacy budget, both datasets."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Membership-inference attack AUC",
                                        "Downstream utility (TSTR, LR)"],
                        shared_xaxes=True)
    colors = {"heart": OKABE_ITO["blue"], "pima": OKABE_ITO["vermillion"]}
    for d in sorted(curve.dataset.unique()):
        c = curve[curve.dataset == d].sort_values("epsilon")
        xs = np.where(np.isinf(c.epsilon.to_numpy(float)), 100.0, c.epsilon)
        color = colors.get(d, palette_for(d))
        for col, key in ((1, "priv_mia_auc"), (2, "tstr_lr_auc")):
            y = c[f"{key}_mean"].to_numpy(float)
            err = np.nan_to_num(c[f"{key}_std"].to_numpy(float))
            fig.add_trace(go.Scatter(
                x=xs, y=y, name=d, legendgroup=d, showlegend=(col == 1),
                mode="lines+markers", line=dict(color=color, width=2.2),
                marker=dict(size=9, color=color, line=dict(width=0)),
                error_y=dict(type="data", array=err, color=color, width=2),
                hovertemplate=f"{d} · ε=%{{x}}<br>{key}=%{{y:.3f}}"
                              "<extra></extra>"), row=1, col=col)
    fig.add_hline(y=0.5, line=dict(color=INK, width=1.4, dash="dash"), row=1,
                  col=1, annotation_text="chance", annotation_font_size=10)
    for col in (1, 2):
        fig.update_xaxes(type="log", tickvals=[1, 3, 10, 30, 100],
                         ticktext=["1", "3", "10", "30", "∞"],
                         title_text="ε (smaller = more private)",
                         showgrid=True, gridcolor=GRID, row=1, col=col)
        fig.update_yaxes(showgrid=True, gridcolor=GRID, row=1, col=col)
    fig.update_layout(height=360, margin=dict(l=8, r=8, t=64, b=56),
                      legend=dict(orientation="h", yanchor="top", y=-0.14,
                                  xanchor="center", x=0.5,
                                  bgcolor="rgba(0,0,0,0)"),
                      title="The privacy trade-off, measured")
    return fig


def model_bars(g: pd.DataFrame, mean_col: str, std_col: Optional[str],
               title: str, ylab: str, ref_lines: Dict[str, float],
               horizontal: bool = False,
               labels: Optional[Sequence[str]] = None) -> go.Figure:
    """One bar per model, coloured by the shared model palette.

    `labels` is the display text; the colour still comes from the raw model key,
    so a prettier label cannot silently break the palette link to the figures.
    """
    names = list(labels) if labels is not None else list(g.model)
    colors = [palette_for(m) for m in g.model]
    fig = go.Figure()
    err = (np.nan_to_num(g[std_col].to_numpy(float)) if std_col
           and std_col in g else None)
    # Reference lines are annotated on alternating corners. Two `add_vline`
    # annotations on the same side overlap into unreadable text, which is what
    # happens when the floor and the naive baseline sit close together.
    anchors = ["top right", "bottom right", "top left", "bottom left"]
    if horizontal:
        fig.add_trace(go.Bar(
            y=names, x=g[mean_col].to_numpy(float), orientation="h",
            marker_color=colors,
            error_x=dict(type="data", array=err, color="#7A8CA0", width=2),
            hovertemplate="%{y}<br>%{x:.4f}<extra></extra>", showlegend=False))
        for i, (name, value) in enumerate(ref_lines.items()):
            fig.add_vline(x=value, line=dict(color="#7A8CA0", width=1.6,
                                             dash="dot"),
                          annotation_text=name, annotation_font_size=10,
                          annotation_position=anchors[i % len(anchors)])
        fig.update_xaxes(title_text=ylab, showgrid=True, gridcolor=GRID)
        fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
        fig.update_layout(height=60 + 40 * len(g))
    else:
        fig.add_trace(go.Bar(
            x=names, y=g[mean_col].to_numpy(float),
            marker_color=colors,
            error_y=dict(type="data", array=err, color="#7A8CA0", width=2),
            hovertemplate="%{x}<br>%{y:.4f}<extra></extra>", showlegend=False))
        for i, (name, value) in enumerate(ref_lines.items()):
            fig.add_hline(y=value, line=dict(color="#7A8CA0", width=1.6,
                                             dash="dot"),
                          annotation_text=name, annotation_font_size=10,
                          annotation_position=anchors[i % len(anchors)])
        fig.update_yaxes(title_text=ylab, showgrid=True, gridcolor=GRID)
        fig.update_xaxes(tickfont=dict(size=9))
    fig.update_layout(title=title, margin=dict(l=8, r=8, t=56, b=8),
                      bargap=0.3, showlegend=False)
    return fig
