"""Page 2 — Leverkusen Alonso window (marts only; no network).

Required: `data/marts/leverkusen_overlay.csv`.

Optional (when present, enable richer pre/post UI):
- `leverkusen_prepost_delta.csv` — columns: metric, delta, delta_ci_lo, delta_ci_hi (23/24 − 22/23).
- `leverkusen_prepost_levels.csv` — columns: metric, pre_mean, pre_ci_lo, pre_ci_hi,
  post_mean, post_ci_lo, post_ci_hi (for build-up + patience panels).
- `leverkusen_season_snapshot.csv` — columns: season_key (2223 or 2324), n_matches,
  n_labeled_possessions.
- `leverkusen_2324_buildup_labels.parquet` — fills 23/24 counts when snapshot omits them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from components.leverkusen_plots import (
    fig_forest_from_overlay,
    fig_forest_sorted,
    fig_miami_overlay_bars,
    fig_single_metric_compare,
)
from components.style import caveat_box, dense_cols, metric_card

ROOT = Path(__file__).resolve().parents[2]
MARTS = ROOT / "data" / "marts"

OVERLAY_PATH = MARTS / "leverkusen_overlay.csv"
PREPOST_DELTA_PATH = MARTS / "leverkusen_prepost_delta.csv"
SNAPSHOT_PATH = MARTS / "leverkusen_season_snapshot.csv"
PREPOST_LEVELS_PATH = MARTS / "leverkusen_prepost_levels.csv"
LABELS_2324 = MARTS / "leverkusen_2324_buildup_labels.parquet"

st.markdown("# Bayer Leverkusen 2022/23 → 2023/24 — The Alonso Window")
st.caption(
    "Event data and CIs read from `data/marts/`; Alonso’s role is a calendar label, "
    "not a treatment effect here."
)

overlay = pd.read_csv(OVERLAY_PATH)
prepost_delta = pd.read_csv(PREPOST_DELTA_PATH) if PREPOST_DELTA_PATH.exists() else None
prepost_levels = pd.read_csv(PREPOST_LEVELS_PATH) if PREPOST_LEVELS_PATH.exists() else None
snapshot = pd.read_csv(SNAPSHOT_PATH) if SNAPSHOT_PATH.exists() else None


# ---- n_matches / n_labeled from optional snapshot or 23/24 labels parquet ----
def _season_val(skey: str, col: str) -> str:
    if snapshot is None or col not in snapshot.columns:
        return "—"
    sub = snapshot.loc[snapshot["season_key"] == skey, col]
    if sub.empty or pd.isna(sub.iloc[0]):
        return "—"
    v = sub.iloc[0]
    s = str(v)
    if s.replace(".", "", 1).isdigit() and float(v) == int(float(v)):
        return str(int(v))
    return s


n_pre_m, n_pre_c = _season_val("2223", "n_matches"), _season_val("2223", "n_labeled_possessions")
n_post_m, n_post_c = _season_val("2324", "n_matches"), _season_val("2324", "n_labeled_possessions")
if LABELS_2324.exists():
    try:
        lab = pd.read_parquet(LABELS_2324)
        if n_post_m == "—":
            n_post_m = str(lab["match_id"].nunique())
        if n_post_c == "—":
            n_post_c = str(len(lab))
    except (OSError, KeyError, ValueError):
        pass
if n_post_m == "—":
    n_post_m = "34"

# ---- largest delta: internal pre/post if available, else cross-case overlay ----
if prepost_delta is not None and "delta" in prepost_delta.columns:
    imax = prepost_delta["delta"].abs().idxmax()
    top = prepost_delta.loc[imax]
    top_metric = str(top["metric"]) if "metric" in top.index else ""
    top_d = float(top["delta"])
    top_dlo = float(top.get("delta_ci_lo", top_d))
    top_dhi = float(top.get("delta_ci_hi", top_d))
    largest_val = f"{top_d:+.3f}"
    largest_delta = f"{top_metric}  95% CI [{top_dlo:+.3f}, {top_dhi:+.3f}]"
    direction = "higher" if top_d > 0 else "lower"
    direction_lbl = f"22/23 → 23/24: {top_metric} {direction}"
else:
    o2 = overlay.copy()
    o2["_a"] = o2["diff_leverkusen_minus_miami"].abs()
    top = o2.sort_values("_a", ascending=False).iloc[0]
    top_metric = str(top["metric"])
    top_d = float(top["diff_leverkusen_minus_miami"])
    largest_val = f"{top_d:+.3f}"
    largest_delta = f"{top_metric}  Δ(Lv−IM)"
    direction_lbl = "Cross-case overlay (not intra-Leverkusen pre/post)"


c1, c2, c3, c4 = dense_cols(4)
with c1:
    metric_card(f"{n_pre_m} / {n_post_m}", "Matches (pre / post)", tone="violet")
with c2:
    metric_card(f"{n_pre_c} / {n_post_c}", "Labeled possessions (pre / post)", tone="cyan")
with c3:
    metric_card(largest_val, "Largest |Δ|", largest_delta, tone="violet")
with c4:
    dtxt = direction_lbl[:40] + ("…" if len(direction_lbl) > 40 else "")
    metric_card(dtxt, "Δ direction", tone="cyan")

if prepost_delta is not None:
    st.caption(
        "Metric deltas: Leverkusen 23/24 − Leverkusen 22/23 (persisted mart; "
        "`leverkusen_prepost_delta.csv`)"
    )
    fig_f = fig_forest_sorted(
        prepost_delta,
        delta_col="delta",
        ci_lo_col="delta_ci_lo",
        ci_hi_col="delta_ci_hi",
        xlabel="Δ 23/24 − 22/23 (95% CI)",
    )
else:
    st.caption(
        "`leverkusen_prepost_delta.csv` absent — Bundesliga 22/23 unavailable in StatsBomb "
        "Open Data; forest shows cross-case deltas (Leverkusen 23/24 − Inter Miami 2023)."
    )
    fig_f = fig_forest_from_overlay(overlay)

st.pyplot(fig_f, clear_figure=True)
plt.close(fig_f)

col_l, col_r = st.columns(2, gap="medium")


def _level_row(metric: str) -> pd.Series | None:
    if prepost_levels is None:
        return None
    row = prepost_levels.loc[prepost_levels["metric"] == metric]
    return None if row.empty else row.iloc[0]


def _overlay_row(metric: str) -> pd.Series | None:
    row = overlay.loc[overlay["metric"] == metric]
    return None if row.empty else row.iloc[0]


with col_l:
    st.caption("Raw build-up failure rate")
    lr = _level_row("raw_buildup_failure_rate")
    orow = _overlay_row("raw_buildup_failure_rate")
    if lr is not None:
        fb = fig_single_metric_compare(
            "",
            lr.get("pre_mean"),
            lr.get("pre_ci_lo"),
            lr.get("pre_ci_hi"),
            lr.get("post_mean"),
            lr.get("post_ci_lo"),
            lr.get("post_ci_hi"),
            xlabel="rate",
        )
    elif orow is not None:
        fb = fig_single_metric_compare(
            "",
            None,
            None,
            None,
            float(orow["leverkusen_mean"]),
            float(orow["leverkusen_ci_lo"]),
            float(orow["leverkusen_ci_hi"]),
            xlabel="rate",
        )
    else:
        fb = None
    if fb:
        st.pyplot(fb, clear_figure=True)
        plt.close(fb)
    else:
        st.info("No build-up snapshot in marts.")

with col_r:
    st.caption("Patience composite")
    lr2 = _level_row("patience_composite")
    orow2 = _overlay_row("patience_composite")
    if lr2 is not None:
        fp = fig_single_metric_compare(
            "",
            lr2.get("pre_mean"),
            lr2.get("pre_ci_lo"),
            lr2.get("pre_ci_hi"),
            lr2.get("post_mean"),
            lr2.get("post_ci_lo"),
            lr2.get("post_ci_hi"),
            xlabel="composite scale",
        )
    elif orow2 is not None:
        fp = fig_single_metric_compare(
            "",
            None,
            None,
            None,
            float(orow2["leverkusen_mean"]),
            float(orow2["leverkusen_ci_lo"]),
            float(orow2["leverkusen_ci_hi"]),
            xlabel="composite scale",
        )
    else:
        fp = None
    if fp:
        st.pyplot(fp, clear_figure=True)
        plt.close(fp)
    else:
        st.info("No patience composite snapshot in marts.")

st.caption("Inter Miami 2023 vs Leverkusen 23/24 — paired metrics from overlay mart")
fig_o = fig_miami_overlay_bars(overlay)
st.pyplot(fig_o, clear_figure=True)
plt.close(fig_o)

caveat_box(
    source="StatsBomb Open Data (`leverkusen_overlay.csv`; optional intra-Leverkusen CSVs).",
    scope=(
        "Associational framing only. Confounds include roster churn, opponent schedule, league "
        "effects, and sample asymmetry (e.g. 34 vs 6 matches). "
        "Isolating Xabi Alonso’s contribution "
        "from these factors is not possible here."
    ),
)
