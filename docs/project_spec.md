# Pressured Progression — Project Spec

> **Status:** Stub generated from initial project context on 2026-04-21. Treat as the working contract until expanded. Any new top-level metric MUST be proposed here before implementation (hard don't #3).

## 1. Thesis

Two failure modes in MLS:
1. **Build-up collapse** when pressed in own half
2. **Post-regain waste** — winning the ball but rushing the attack

---

**Scope note (post-Phase 3 data audit).** StatsBomb Open Data covers only 6 Inter Miami 2023 matches in MLS. Full-league event-level analysis is not possible on free data. The project is scoped to one event-level MLS case study (Inter Miami 2023, the Messi season) paired with one event-level European analog (Bayer Leverkusen 2023/24 under Xabi Alonso), with ASA season summaries providing league-wide context. This is a narrower but data-honest claim: "here is how one high-profile MLS team handled pressure, measured against a European team that transformed a similar profile in the same window."

---

## 2. Scope and framing

### Temporal framing (strict)
- **Base data year:** MLS 2023 (StatsBomb Open Data coverage constraint)
- **Supplementary:** ASA 2020–present for trend context
- **Narrative-only:** Philadelphia Union 2026 (Inquirer), San Diego FC 2025 (Guardian). Prose only — never attach statistical claims without explicit "as of 2023" labeling.

### Case study teams
- **Philadelphia Union 2023** — primary failure case
- **Columbus Crew 2023** — internal positive control (MLS Cup winners under Wilfried Nancy; full StatsBomb 2023 coverage)
- **San Diego FC** — narrative cameo only; no event-level claims

### European analog candidates (to validate coverage before use)
Brighton 2021/22 & 2023/24, Bologna 2022/23 & 2023/24, Girona 2022/23 & 2023/24, Leverkusen 2022/23 & 2023/24.

## 3. Data sources

| Source | Access | Role |
|---|---|---|
| StatsBomb Open Data | `statsbombpy` | Primary event data (MLS 2023 + European analogs) |
| American Soccer Analysis | v1 REST API (`https://app.americansocceranalysis.com/api/v1/`) | Trend context 2020–present |
| FBref | Rate-limited HTTP + BS4 (max 3/min) | Style-vector features MLS ↔ Europe |

## 4. Key abstractions

- **`possession_chain`** — consecutive events with constant `team_id`
- **`regain`** — defensive action transitioning possession opp → us
- **`buildup_failure_label`** — binary target. Defensive-third origin + opponent pressure within 3 actions + failure termination (turnover; forced long ball; opponent shot ≤10s; backward reset + turnover ≤5 actions).

### Module C — Leverkusen 23/24 × Inter Miami 2023 Overlay (direct)

---

No similarity-matching engine. No 8-dimension style vector. Compute Module A (raw build-up failure rate) and Module B (post-regain metrics) on Leverkusen 2023/24 and overlay side-by-side against Inter Miami 2023 on the same six Module B metrics plus the Module A rate. Cross-league, cross-team comparison — associational only; differences are NOT causal and NOT attributable to Alonso.

**Pre/post-vs-2022/23 comparison dropped** after the Phase 5 data-reality audit (Bundesliga 2022/23 is not in StatsBomb Open Data; see `docs/data_reality.md` §1 + Phase 5 appendix). Any "what changed under Alonso" language moves to Layer 2 / article prose, not a code module.

Layer 2 (coaching literature) remains: vocabulary for describing what changed, sourced to Spielverlagerung, The Athletic, Coaches' Voice. Layer 2 is article-writing material, not a code module.

---

## 5. Case study structure

---

### Case A — Inter Miami 2023 (primary, event-level)
6 matches of StatsBomb Open Data + 360. Full Module A + B. Messi's first half-season is the narrative hook.

### Case B — Bayer Leverkusen 2023/24 (European analog, event-level)
34 matches of StatsBomb Open Data + 360. Full Module A + B metrics computed and compared side-by-side against Inter Miami 2023. The pre/post-vs-2022/23 comparison was dropped after the Phase 5 data-reality audit — Bundesliga 22/23 is absent from StatsBomb Open Data (see `docs/data_reality.md` §1 + Phase 5 appendix). Not a similarity-search result — a directly chosen analog based on data availability and the narrative clarity of a star-led transformation year.

### Narrative cameos (prose only, no statistical claims)
- Philadelphia Union 2026 season-opening slump (Inquirer hook)
- Columbus Crew 2023 MLS Cup under Wilfried Nancy (positive-control vibe — cited, not measured)
- San Diego FC 2025 build-up extreme (Guardian, future-looking)

---

## 6. Top-level metrics (contract)

> Adding or changing a metric requires editing this section first.

### Build-up phase
- Defensive-third pass completion under pressure
- Buildup failure rate (per opportunity)
- Time-to-first-progressive-action under pressure

### Post-regain phase
- Post-regain retention rate (ball held ≥N actions after regain)
- Post-regain entry rate (into final third within X seconds)
- Post-regain shot rate

### Style vector (8-dim, for analog matching)
Populated from FBref columns that align MLS ↔ European top-5. Exact columns finalized after the FBref audit in `docs/data_reality.md`.

## 7. Deliverables

- Article with pre/post metric movement around tactical pivots
- Tactical Precedent Library (associational, not prescriptive — hard don't #4)
- Streamlit app for interactive exploration

## 8. Dashboard

Streamlit multi-page app (`st.Page` / `st.navigation`), **3 pages**:

- **Page 1:** Inter Miami Diagnostic
- **Page 2:** Leverkusen Pre/Post
- **Page 3:** MLS League Context (ASA, live-refreshing)

## 9. Framing rules (non-negotiable)

### Hard don'ts

- Don't add new top-level metrics without editing **§6** first.
- Leakage check threshold scales with sample size: at **n < 200**, use **CV_mean − 0.10** instead of **− 0.05** (small-sample chance variance).
- **`opp_press_height`** coordinate convention (opponent attacking frame, **x flipped**) documented here, not only in module docstrings.
- **`support_density_ff`** dropped from the feature set unless and until **360 JSON 404s** are resolved.

### Output framing

- **Associational language only** in docstrings and outputs ("coincided with", "associated with"). Never "caused" / "because of".
- Tactical Precedent Library is framed as "teams with similar profiles that improved did X; their metrics moved in Y direction" — never "do X".

## 10. Pipeline

```
data/raw → data/core (typed, validated) → data/marts (analysis-ready)
```

Core marts:
- `possession_chains`
- `pressure_events`
- `regain_sequences`
- `team_season_features`

## 11. Stack

- Python 3.11
- Pandas, NumPy, DuckDB (event-level analytics ride on DuckDB, not pandas)
- scikit-learn, XGBoost, SHAP
- matplotlib, mplsoccer, Plotly, Streamlit
- Pydantic for schemas, Ruff for linting, pytest for tests

## 12. Directory structure

```
src/pressured_progression/
  ingest/        # statsbomb.py, asa.py, fbref.py
  core/          # typed, validated core tables
  sequences/     # possession_chains, regain_sequences
  features/      # team_season_features, style vectors
  labeling/      # buildup_failure_label construction
  models/        # classifiers, similarity
  matching/      # European analog matcher
  viz/           # plots
  analysis/      # notebooks-as-scripts
notebooks/       # Jupyter exploration
tests/
data/
  raw/           # immutable, gitignored
  core/          # gitignored
  marts/         # can be committed if small
docs/            # project_spec.md, data_reality.md
app/             # Streamlit app
```

## 13. References

- Data audit: `docs/data_reality.md` (authoritative coverage + base-year justification)
