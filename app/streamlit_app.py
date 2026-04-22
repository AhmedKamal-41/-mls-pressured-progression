"""Pressured Progression — 3-page Streamlit app entry point.

Per spec §8, uses `st.Page` + `st.navigation` rather than the auto-pages
directory convention. Each page is a standalone script under `app/pages/`.
Dark three-layer UI theme injected from `components.style`.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from components.style import inject_css

st.set_page_config(
    page_title="Pressured Progression",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

APP_DIR = Path(__file__).resolve().parent

pages = [
    st.Page(
        str(APP_DIR / "pages" / "1_inter_miami.py"),
        title="Inter Miami Diagnostic",
        default=True,
        icon="🎯",
    ),
    st.Page(
        str(APP_DIR / "pages" / "2_leverkusen.py"),
        title="Leverkusen Pre/Post",
        icon="⚗️",
    ),
    st.Page(
        str(APP_DIR / "pages" / "3_league_context.py"),
        title="MLS League Context",
        icon="📡",
    ),
]

nav = st.navigation(pages)
nav.run()
