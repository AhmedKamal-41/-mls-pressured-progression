"""Shared Streamlit styling + UI primitives for the Pressured Progression app.

Palette: violet #7C3AED primary, cyan #06B6D4 secondary, on a dark three-layer
background (base → panel → card). Inter for UI text, JetBrains Mono for all
numerals. Fonts fall back gracefully if the user's machine doesn't have them.
"""

from __future__ import annotations

import streamlit as st

# ---- colors ----
VIOLET = "#7C3AED"
CYAN = "#06B6D4"
INK = "#E5E7EB"
MUTED = "#9CA3AF"
BG_BASE = "#0B0F1A"
BG_PANEL = "#12182B"
BG_CARD = "#1B2237"
BORDER = "#2A3148"
AMBER = "#F59E0B"


DEFAULT_CAVEAT = (
    "Associational, not causal. Observational data. Patterns described here "
    "reflect team, roster, schedule, and league variance — not the effect of "
    "any individual coach or player."
)


_CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

  html, body, [data-testid="stAppViewContainer"] {{
    background: {BG_BASE};
    color: {INK};
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }}
  [data-testid="stHeader"] {{
    background: {BG_BASE};
  }}
  [data-testid="stSidebar"] {{
    background: {BG_PANEL};
    border-right: 1px solid {BORDER};
  }}
  /* Numerals: anything in code/mono class uses JetBrains Mono */
  code, .pp-mono, .pp-value, .pp-delta {{
    font-family: 'JetBrains Mono', 'DejaVu Sans Mono', Consolas, monospace;
    font-feature-settings: "tnum" 1, "lnum" 1;
  }}

  /* Headings */
  h1, h2, h3, h4 {{
    color: {INK};
    letter-spacing: -0.01em;
  }}
  h1 {{ font-weight: 700; }}
  h2 {{ font-weight: 600; }}

  /* Metric card */
  .pp-card {{
    background: {BG_CARD};
    border-radius: 10px;
    padding: 18px 22px;
    border-top: 3px solid {VIOLET};
    border-left: 1px solid {BORDER};
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    margin: 6px 0;
  }}
  .pp-value {{
    font-size: 32px;
    font-weight: 700;
    line-height: 1.05;
    color: {VIOLET};
  }}
  .pp-label {{
    font-size: 12px;
    color: {MUTED};
    margin-top: 6px;
    letter-spacing: 0.03em;
    text-transform: uppercase;
  }}
  .pp-delta {{
    font-size: 13px;
    color: {CYAN};
    margin-top: 8px;
  }}

  /* Caveat */
  .pp-caveat {{
    background: {BG_PANEL};
    border-left: 3px solid {AMBER};
    border-radius: 6px;
    padding: 14px 18px;
    margin-top: 24px;
    color: {MUTED};
    font-size: 13px;
    line-height: 1.55;
  }}
  .pp-caveat strong {{ color: {INK}; }}

  /* Section panels */
  .pp-panel {{
    background: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 16px 20px;
    margin: 10px 0;
  }}

  /* Narrative cameo blocks (prose-only context) */
  .pp-narrative {{
    min-height: 140px;
  }}
  .pp-narrative-label {{
    font-size: 11px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: 10px;
  }}
  .pp-narrative-body {{
    font-size: 13px;
    line-height: 1.55;
    color: {INK};
    margin: 0;
  }}

  /* Streamlit tables + dataframes on dark bg */
  [data-testid="stDataFrame"] {{
    background: {BG_CARD};
    border-radius: 8px;
  }}

  /* Hide streamlit's default chrome that fights the dark layout */
  footer {{ visibility: hidden; }}
  [data-testid="stStatusWidget"] {{ display: none; }}
</style>
"""


def inject_css() -> None:
    """Call once per page after st.set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)


def metric_card(value: str, label: str, delta: str | None = None, *, tone: str = "violet") -> None:
    """Dense KPI card. `tone='violet'` for primary, `tone='cyan'` for secondary marks."""
    color = VIOLET if tone == "violet" else CYAN
    delta_html = f"<div class='pp-delta'>{delta}</div>" if delta else ""
    st.markdown(
        f"""
        <div class="pp-card" style="border-top-color:{color}">
          <div class="pp-value" style="color:{color}">{value}</div>
          <div class="pp-label">{label}</div>
          {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def caveat_box(
    text: str | None = None,
    *,
    source: str | None = None,
    scope: str | None = None,
) -> None:
    """Associational caveat. Used at the end of every page."""
    body = text or DEFAULT_CAVEAT
    parts: list[str] = []
    if source:
        parts.append(f"<strong>Source:</strong> {source}")
    if scope:
        parts.append(f"<strong>Scope:</strong> {scope}")
    parts.append(body)
    st.markdown(
        f"""<div class="pp-caveat">{"<br/>".join(parts)}</div>""",
        unsafe_allow_html=True,
    )


def pitch_plot(
    *,
    figsize: tuple[float, float] = (10, 6.5),
    pitch_color: str = BG_PANEL,
    line_color: str = MUTED,
):
    """Thin wrapper over mplsoccer.Pitch that returns (fig, ax, pitch).

    Using StatsBomb pitch dimensions (120x80) to align with event data in
    data/raw/events/ and data/raw/statsbomb/.
    """
    from mplsoccer import Pitch

    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color=pitch_color,
        line_color=line_color,
        linewidth=1,
    )
    fig, ax = pitch.draw(figsize=figsize)
    fig.patch.set_facecolor(BG_BASE)
    return fig, ax, pitch


def dense_cols(n: int):
    """Slightly tighter column layout; returns list of st.columns."""
    return st.columns(n, gap="small")
