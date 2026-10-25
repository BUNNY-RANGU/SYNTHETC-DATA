"""
ui - the presentation layer of the web app.

Kept separate from `synth/` on purpose: `synth/` knows how to generate and
measure data and must stay usable from the CLI, the notebook and the tests
without importing anything that renders. `ui/` is allowed to know about
Streamlit and Plotly; `synth/` is not.

    ui.theme     design tokens + the stylesheet
    ui.charts    every figure the app draws, returned as plotly objects
    ui.components hero, KPI cards, status pills, section headers
"""

from ui.theme import CSS, INJECT, PLOT, palette_for  # noqa: F401
from ui.components import (chart, hero, kpi_row, note, pills, section,  # noqa: F401
                           verdict)

__all__ = ["CSS", "INJECT", "PLOT", "palette_for",
           "hero", "kpi_row", "pills", "section", "note", "verdict", "chart"]
