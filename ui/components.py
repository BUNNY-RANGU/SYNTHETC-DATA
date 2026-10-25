"""
ui.components - the small vocabulary of the interface.

Everything the app renders goes through one of these six calls, so the look is
defined once. `kpi_row` deliberately wraps `st.metric` rather than emitting its
own HTML: st.metric is what the automated tests assert on, it is keyboard and
screen-reader reachable, and it gets the card styling from CSS instead of from
markup that would have to be kept in sync by hand.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import plotly.graph_objects as go
import streamlit as st

from ui.theme import INJECT, PLOT


def inject_css() -> None:
    st.markdown(INJECT, unsafe_allow_html=True)


def brand_mark() -> None:
    st.markdown(
        """
        <div class="brand">
          <div class="glyph">S</div>
          <div>
            <div class="name">SynthMed</div>
            <div class="sub">synthetic EHR lab</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, chips: Sequence[str] = ()) -> None:
    meta = "".join(f'<span class="pill">{c}</span>' for c in chips)
    st.markdown(
        f"""
        <div class="hero">
          <h1>{title}</h1>
          <p>{subtitle}</p>
          <div class="meta">{meta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pills(items: Iterable[Tuple[str, str]]) -> None:
    """Status line under the hero. tone in {default, live, frozen}."""
    out = []
    for text, tone in items:
        cls = f"pill {tone}" if tone != "default" else "pill"
        out.append(f'<span class="{cls}"><span class="dot"></span>{text}</span>')
    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;gap:.4rem;margin:-.35rem 0 1rem">'
        + "".join(out) + "</div>",
        unsafe_allow_html=True,
    )


def section(title: str, number: Optional[str] = None) -> None:
    n = f'<span class="n">{number}</span>' if number else ""
    st.markdown(
        f'<div class="sec">{n}<span class="t">{title}</span>'
        f'<span class="rule"></span></div>',
        unsafe_allow_html=True,
    )


def verdict(label: str, tone: str = "neutral", note: str = "") -> None:
    """PASS / PARTIAL / FAIL badge. tone: good | warn | bad | neutral."""
    cls = {"good": "chip good", "warn": "chip warn", "bad": "chip bad",
           "neutral": "chip"}[tone]
    extra = f'<span class="small">&nbsp;&nbsp;{note}</span>' if note else ""
    st.markdown(
        f'<span class="{cls}">{label}</span>{extra}', unsafe_allow_html=True)


def kpi_row(items: Sequence[Sequence], per_row: int = 4) -> None:
    """items = (label, value, help, delta); the last two are optional."""
    items = list(items)
    for start in range(0, len(items), per_row):
        cols = st.columns(per_row)
        for col, it in zip(cols, items[start:start + per_row]):
            label, value = it[0], it[1]
            help_text = it[2] if len(it) > 2 else None
            delta = it[3] if len(it) > 3 else None
            col.metric(label, value, delta=delta, help=help_text)


def note(text: str, kind: str = "info") -> None:
    getattr(st, kind)(text)


def chart(fig: go.Figure, height: Optional[int] = None) -> None:
    """Apply the house layout and render, without the plotly toolbar."""
    fig.update_layout(
        font=dict(family=PLOT["font_family"], size=PLOT["font_size"],
                  color="#16283C"),
        colorway=PLOT["colorway"],
        paper_bgcolor=PLOT["paper_bgcolor"],
        plot_bgcolor=PLOT["plot_bgcolor"],
        margin=PLOT["margin"],
        hoverlabel=dict(bgcolor="rgba(255,255,255,.96)",
                        font=dict(family=PLOT["font_family"],
                                  size=PLOT["font_size"], color="#16283C")),
        title=dict(x=0.0, xanchor="left", font=dict(size=14.5,
                 color="#0B1F33")),
        legend=dict(orientation="h", yanchor="bottom", y=1.0,
                    xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        bargap=0.22,
    )
    if height:
        fig.update_layout(height=height)
    for axis in ("xaxis", "yaxis"):
        fig.update_layout(**{axis: dict(showgrid=True, gridcolor=PLOT["grid"],
                                         zeroline=False,
                                         linecolor=PLOT["grid"],
                                         tickfont_color=PLOT["axis"])})
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False, "responsive": True})


def footer(left: str, right: str) -> None:
    st.markdown(
        f'<div class="foot"><span>{left}</span><span>{right}</span></div>',
        unsafe_allow_html=True,
    )
