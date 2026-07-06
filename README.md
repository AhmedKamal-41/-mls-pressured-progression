# Pressured Progression — Full Project Documentation (A→Z)

> A complete, self-contained walkthrough of the **Pressured Progression** soccer-analytics project: what it does, the science behind it, the data pipeline, every module, every mart, the models, the tests, the app, the metrics, and every honest limitation. If you read only one file to understand this repository, read this one.

---

## Table of contents

1. [Executive overview](#1-executive-overview)
2. [The two failure modes (the thesis)](#2-the-two-failure-modes-the-thesis)
3. [Headline findings & every metric](#3-headline-findings--every-metric)
4. [The hard data-reality story (why the scope is what it is)](#4-the-hard-data-reality-story-why-the-scope-is-what-it-is)
5. [Data sources](#5-data-sources)
6. [Architecture & directory map](#6-architecture--directory-map)
7. [The data pipeline, stage by stage](#7-the-data-pipeline-stage-by-stage)
8. [Core abstractions & schemas](#8-core-abstractions--schemas)
9. [Module A — build-up failure (labeling + ML)](#9-module-a--build-up-failure-labeling--ml)
10. [Module B — post-regain waste](#10-module-b--post-regain-waste)
11. [Module C — the Leverkusen overlay](#11-module-c--the-leverkusen-overlay)
12. [The machine-learning model in detail](#12-the-machine-learning-model-in-detail)
13. [Data marts — every output file](#13-data-marts--every-output-file)
14. [Figures & the executive summary](#14-figures--the-executive-summary)
15. [The Streamlit dashboard](#15-the-streamlit-dashboard)
16. [Notebooks](#16-notebooks)
17. [Testing](#17-testing)
18. [CI, linting & pre-commit](#18-ci-linting--pre-commit)
19. [How to reproduce everything](#19-how-to-reproduce-everything)
20. [Tech stack](#20-tech-stack)
21. [Constants & thresholds cheat-sheet](#21-constants--thresholds-cheat-sheet)
22. [Known limitations, tech debt & caveats](#22-known-limitations-tech-debt--caveats)
23. [Glossary](#23-glossary)
24. [Author & license](#24-author--license)

---

## 1. Executive overview

**Pressured Progression** measures two distinct ways a soccer team can fail under pressure, using event-level tracking data:

1. **Build-up collapse** — when a team is pressed deep in its own half and its attempt to play out fails.
2. **Post-regain waste** — when a team wins the ball back but the recovery never turns into a real attacking threat.

The project puts hard numbers (with uncertainty bands) on both failure modes for one MLS team — **Inter Miami 2023** (Messi's half-season, 6 matches) — and benchmarks them against a European comparator — **Bayer Leverkusen 2023/24** (Xabi Alonso's unbeaten title season, 34 matches). It is built entirely on **open data** (StatsBomb Open Data + the American Soccer Analysis API), so anyone can reproduce it.

It is an **end-to-end technical portfolio project**: raw data ingestion → typed/validated core tables → analysis-ready marts → an interpretable ML model (XGBoost + SHAP) → bootstrap-CI visualizations → a one-page executive PDF → an interactive multi-page Streamlit app → a long-form written article. Every number in every chart traces back to a file in `data/marts/` and a line of code.

**The single most important framing rule:** every reported result is **associational, not causal**. The project never claims a coach, player, or tactic *caused* anything. It describes what co-occurred in the data, with explicit uncertainty and explicit scope limits.

- **Package name:** `pressured-progression` (version `0.1.0`)
- **Python:** 3.11+
- **Author:** Ahmed Kali (`ahmedkali841@gmail.com`)
- **License:** MIT

---

## 2. The two failure modes (the thesis)

### Build-up collapse under pressure
The classic "chasing shadows in your own half" scenario. In code, a build-up sequence is flagged when:
- possession **starts in the defensive third** (`start_x < 40` on a 0–120 pitch), **and**
- an **opponent pressing action appears within the first 3 on-ball touches**,

and it terminates via one of four failure endings (turnover, forced long ball, opponent shot within 10s, or a backward reset that leads to a loss). Plainly: *you're trying to climb out while someone leans on you, and you don't manage it.*

### Post-regain waste
Shifts the clock forward one beat. Instead of the guarded exit from deep, this asks what the team does with a **newly won** ball. A **regain** is a defensive action (interception, tackle, ball recovery, duel, block) that flips possession to the focal team. The subsequent possession chain is measured for: did it reach the final third? How long until a shot? Was the shot rushed and low-quality? Did possession collapse again within 4 actions? *If the ball returns and the next few seconds behave like hurry rather than setup, the window is scored as wasted.*

Both failure modes coexist in real football; **neither implies the other**. Collapse is about buildup under harassment; waste is about the impulse right after flipping the ball.

---

## 3. Headline findings & every metric

### The headline
- **Inter Miami 2023 build-up failure rate:** **34.2%** (95% bootstrap CI **[25.3%, 45.3%]**) across **73** qualifying chains in 6 matches.
- **Post-regain waste:** **90.4%** of Inter Miami's **353** regains produced **no shot within 15 seconds** (319 of 353). Median time-to-shot when a shot *did* happen: **5.5 s**.
- **Leverkusen 2023/24 failure rate:** **~21%** (95% CI **[16%, 25%]**) across **394** chains in 34 matches — a cross-team difference of roughly **−14 percentage points** (CI excludes zero).
- **Leverkusen final-third entry after regain:** **81%** vs Inter Miami **67%** — a **+14 pp** difference (CI excludes zero).
- **Model utility is modest at small n.** XGBoost CV ROC-AUC **0.54 ± 0.08** on 73 chains — a **logistic-regression baseline (0.586) actually beats it**. Reported honestly, with no overstatement.

### Module B — Inter Miami's six post-regain metrics (with 95% bootstrap CIs)
Source: `data/marts/team_post_regain.csv`, all over **n = 353** regains.

| Metric | Point estimate | 95% CI |
|---|---:|---|
| xG per regain | 0.00827 | [0.0044, 0.0135] |
| Time-to-shot (median seconds) | 5.5 | [4.0, 9.0] |
| Rushed-shot rate | 0.353 | [0.136, 0.485] |
| Final-third entry rate | 0.669 | [0.568, 0.772] |
| Loss-within-4-actions rate | 0.0113 | [0.0027, 0.024] |
| Patience composite | −10.878 | [−11.32, −10.28] |

### Module C — Leverkusen 23/24 vs Inter Miami 2023 overlay
Source: `data/marts/leverkusen_overlay.csv`. Difference = **Leverkusen − Miami**.

| Metric | Leverkusen | Miami | Difference | 95% diff CI | Separable? |
|---|---:|---:|---:|---|:---:|
| Raw build-up failure rate | 0.206 | 0.342 | −0.137 | [−0.262, −0.035] | ✅ |
| xG per regain | 0.0178 | 0.00827 | +0.00958 | [+0.0032, +0.0159] | ✅ |
| Time-to-shot (median s) | 5.0 | 5.5 | −0.50 | [−5.01, +1.5] | ❌ |
| Rushed-shot rate | 0.278 | 0.353 | −0.0748 | [−0.211, +0.123] | ❌ |
| Final-third entry rate | 0.808 | 0.669 | +0.140 | [+0.020, +0.250] | ✅ |
| Loss-within-4-actions rate | 0.0131 | 0.0113 | +0.0018 | [−0.0116, +0.0128] | ❌ |
| Patience composite | −9.79 | −10.88 | +1.09 | [+0.365, +1.80] | ✅ |

**"Separable"** means the 95% CI for the difference excludes zero (column `ci_excludes_zero`). Leverkusen is materially better on **4 of 7** metrics at 95% confidence; the other 3 are not distinguishable from noise at this sample size.

### The Regain Sankey accounting (Inter Miami, 353 regains, 15-second window)
- **34** regains → a modeled shot within 15 s
- **201** → reached the final third without an immediate shot
- **4** → "lost within 4 actions"
- **114** → "other" (still in the accounting, no moral judgment implied)

---

## 4. The hard data-reality story (why the scope is what it is)

This is the most important honesty section, documented fully in `docs/data_reality.md`. The project's *original* design does not survive contact with open data, and the repo is transparent about that.

**What broke:**
1. **The two named MLS case-study teams don't exist in the data.** The spec named **Philadelphia Union 2023** (primary failure case) and **Columbus Crew 2023** (positive control). **Neither team has a single event-level match in StatsBomb Open Data.** The entire MLS 2023 Open Data corpus is **6 Inter Miami fixtures** (the Messi release, Aug–Oct 2023). There is no "MLS-wide distribution" to rank teams within.
2. **3 of 4 European analog candidates are absent.** Brighton (men's) never appears (only the women's FAWSL team is in Open Data); Bologna only has Serie A 2015/16; Girona only La Liga 2017/18–2018/19. **Only Bayer Leverkusen 2023/24** has full 34-match coverage with 360 data.
3. **The planned Leverkusen pre/post comparison died.** Bundesliga **2022/23 is entirely absent** from Open Data (only 2015/16 and 2023/24 exist). So "what changed under Alonso" cannot be a code module — it becomes article prose only. Module C was reframed from a pre/post design into a **side-by-side overlay** (Leverkusen 23/24 vs Inter Miami 2023).
4. **360 freeze-frame data 404s.** All 6 MLS 2023 matches advertise `match_status_360 = "available"` in the catalog, but the underlying 360 JSON files return **HTTP 404** on the StatsBomb GitHub repo. Consequence: the planned `support_density_ff` feature (which needs freeze-frames) is **all-null** and was **dropped** from the model.
5. **FBref is blocked.** Both target pages returned **HTTP 403** behind a Cloudflare "Just a moment…" challenge — on the *first* request, before any rate-limit signal. This killed the planned 8-dimension "style vector" for analog matching. The similarity-matching engine was never built (`src/pressured_progression/matching/` is an empty package).

**How the project responded:** it narrowed to a **data-honest** claim — *"here is how one high-profile MLS team handled pressure, measured against one European comparator, with league context from season aggregates"* — rather than overreaching on data that doesn't exist. This reframing is itself a headline result of the project.

---

## 5. Data sources

| Source | Role | Access | Status |
|---|---|---|---|
| **StatsBomb Open Data** | Event-level data: MLS 2023 (Inter Miami), Bundesliga 2023/24 (Leverkusen) | `statsbombpy` | ✅ Used |
| **American Soccer Analysis (ASA) v1 API** | MLS 2020–present team-season aggregates (league context) | REST (`https://app.americansocceranalysis.com/api/v1/`) | ✅ Used |
| **FBref** | Intended style-vector features | HTTP + BeautifulSoup | ❌ Cloudflare-blocked (403) |

**ASA endpoints used** (all returned HTTP 200):
- `teams/xgoals` — 31 teams; xG for/against, shots, goals, points, xpoints.
- `teams/goals-added` — 31 teams; goals-added by action type (Dribbling, Fouling, Interrupting, Passing, Receiving, Shooting).
- `teams/xpass` — 31 teams; pass completion vs expected, average vertical distance.

**StatsBomb competition/season IDs used in code:**
- MLS 2023: `competition_id = 44`, `season_id = 107`
- Bundesliga 2023/24: `competition_id = 9`, `season_id = 281`

---

## 6. Architecture & directory map

```
pressured-progression/
├── src/pressured_progression/     # the installable Python package
│   ├── core/          schemas.py            # Pydantic column contracts + validate_columns()
│   ├── ingest/        asa.py                # ASA API audit → data/raw/asa/
│   │                  asa_league_baseline.py# MLS 2023 xG/shot baseline → marts
│   │                  events_adapter.py     # statsbombpy wide frame → EventRow shape
│   │                  fbref.py              # FBref scraper (blocked)
│   │                  leverkusen_ingest.py  # Leverkusen 23/24 events → parquet
│   │                  statsbomb.py          # Open Data catalog audit
│   ├── sequences/     possession_chain.py   # events → possession chains
│   │                  regain.py             # events → regain events
│   ├── features/      buildup_features.py   # per-chain feature vectors (Module A X)
│   │                  post_regain.py        # Module B metrics + figures
│   ├── labeling/      buildup_failure.py    # binary failure label + failure_type
│   ├── models/        buildup_failure_xgb.py# XGBoost + LogReg baseline + calibration
│   ├── matching/      (empty — analog matcher never built; see §4)
│   ├── viz/           (empty package)
│   └── analysis/      smoke_buildup_failure.py     # Module A smoke run on MLS 2023
│                      run_buildup_pipeline.py      # full Module A driver
│                      leverkusen_overlay.py        # Module C
│                      build_case_study_figures.py  # 7 case-study PNGs
│                      build_executive_summary.py   # one-page PDF
├── app/               streamlit_app.py      # 3-page dashboard entry point
│   ├── components/    style.py, asa_refresh.py, inter_miami_plots.py, leverkusen_plots.py
│   └── pages/         1_inter_miami.py, 2_leverkusen.py, 3_league_context.py
├── data/
│   ├── raw/           # immutable source data (gitignored)
│   ├── core/          # typed/validated intermediate tables (gitignored)
│   └── marts/         # analysis-ready outputs (committed if small) + models/ + cache/
├── docs/              project_spec.md, data_reality.md, article_draft.md,
│                      executive_summary.pdf, figures/*.png, outreach/*.md
├── notebooks/         03–07 *.ipynb (read-only render notebooks)
├── tests/             8 test files + fixtures
├── .github/workflows/ ci.yml
├── pyproject.toml, README.md, project_info.md, LICENSE
├── .pre-commit-config.yaml, .gitignore
```

**Pipeline data flow (spec §10):**
```
data/raw  →  data/core (typed, validated)  →  data/marts (analysis-ready)
```

Every module resolves the repo root with `ROOT = Path(__file__).resolve().parents[N]` (N=3 for package modules, N=2 for app files) so paths work regardless of the working directory.

---

## 7. The data pipeline, stage by stage

The full pipeline is a sequence of Python module invocations (each is runnable with `python -m ...`). In order:

1. **`ingest.statsbomb`** — audits the Open Data catalog, counts matches per competition-season, writes `data/raw/statsbomb_catalog.csv`. (This is the audit that discovered the coverage gaps.)
2. **`ingest.asa`** — probes the 3 ASA endpoints, dumps raw JSON + a schema index to `data/raw/asa/`.
3. **`ingest.asa_league_baseline`** — pulls MLS 2023 team xGoals, computes the league `xg_per_shot` mean/std → `data/marts/asa_mls_2023_baseline.csv` (the reference line for Module B's patience composite).
4. **`ingest.leverkusen_ingest`** — fetches Leverkusen 23/24 matches + events, writes per-match parquet + a manifest under `data/raw/statsbomb/leverkusen_2324/`.
5. **`analysis.smoke_buildup_failure`** — runs the Module A labeler on MLS 2023 (Inter Miami as the substrate; Philly/Columbus documented as zero-coverage), writes per-match label parquets to `data/core/buildup_labels/`.
6. **`analysis.run_buildup_pipeline`** — the full Module A driver: builds features → trains the model → computes SHAP → persists everything (features parquet, model joblibs, importance/SHAP/CV/OOF marts, calibration figure).
7. **`features.post_regain`** — Module B: detects regains, enriches them, aggregates the six metrics with bootstrap CIs → `regain_events.parquet`, `team_post_regain.csv`, figures.
8. **`analysis.leverkusen_overlay`** — Module C: computes Leverkusen's metrics and overlays them against Inter Miami's → Leverkusen parquets, `leverkusen_overlay.csv`, overlay figures.
9. **`analysis.build_case_study_figures`** — renders the 7 `case_study_*.png` figures from the marts.
10. **`analysis.build_executive_summary`** — assembles the one-page `docs/executive_summary.pdf`.

Then: `streamlit run app/streamlit_app.py` for the dashboard, and `pytest` for the tests.

**Key engineering choices:**
- **DuckDB `read_parquet`** unions per-match parquet files for full-season aggregation, keeping large event scans off pandas memory (spec §11: "event-level analytics ride on DuckDB, not pandas").
- **Idempotent ingestion** — event fetchers skip matches whose parquet already exists.
- **Bootstrap seed `20260421`** and **match-level resampling** are used consistently across every CI computation for reproducibility.

---

## 8. Core abstractions & schemas

Defined in `src/pressured_progression/core/schemas.py` as Pydantic `BaseModel` **column contracts** (they describe DataFrame shapes, not row-by-row validation). The helper `validate_columns(df, schema)` raises `ValueError` listing any missing columns; extra columns are allowed. It is called at every stage boundary.

- **`EventRow`** — the canonical event shape: `match_id, period, minute, second, team_id, player_id, event_type, location_x, location_y, outcome, under_pressure, pass_end_x, pass_end_y, possession_id`. `extra="allow"` so downstream extras (`pass_recipient_id`, `shot_statsbomb_xg`, `pass_length`, `duration`, `possession_team_id`) pass through.
- **`PossessionChain`** — one row per `(match_id, possession_id)` owned by one team: `start_x/y, end_x/y, end_type, action_count, under_pressure_count, terminal_outcome, first_receiver_id`.
- **`RegainEvent`** — `time_seconds, regain_x/y, regain_zone, regaining_team_id, subsequent_possession_id, next_shot_seconds, next_shot_xg`.
- **`FailureType`** (StrEnum) — `turnover_own_half`, `forced_long_ball`, `opp_shot_within_10s`, `backward_reset_turnover`, `none`.
- **`BuildUpFailureLabel`** — `match_id, possession_id, is_failure, failure_type`.

### Possession chains (`sequences/possession_chain.py`)
`build_possession_chains(events)` sorts events by `(match_id, period, minute, second)`, derives the "possession team" as the team of the **first event** in each possession block (to drop interleaved opponent defensive actions), keeps only owner rows, and emits one chain row per block. `end_type` is classified by the terminal event: `goal`/`shot`/`loss`/`clearance`/`foul_won`/`half_end`/etc. Absolute time is `(period-1)*45*60 + minute*60 + second`.

### Regains (`sequences/regain.py`)
`detect_regains(events, focal_team_id)`. A regain fires when an event is a **defensive action** (`{Interception, Ball Recovery, Tackle, Duel, Block}`), is by the focal team, and the focal team owns the resulting possession. Pitch zones by x: **`<40` defensive**, **`40–80` middle**, **`≥80` attacking**. The next focal shot is searched within **`SHOT_WINDOW_S = 15.0` seconds**; its xG comes from `shot_statsbomb_xg`.

---

## 9. Module A — build-up failure (labeling + ML)

### Labeling (`labeling/buildup_failure.py`)
A chain **qualifies** if it starts in the defensive third (`start_x < 40`) **and** any of its first 3 events is `under_pressure`. Qualifying chains are assigned the **first matching** failure type, in this priority order (`we_lost` = the next possession block is owned by a different team):

1. **`turnover_own_half`** — `we_lost` and terminal event is a turnover and terminal `location_x < 60`.
2. **`forced_long_ball`** — `we_lost` and some pass with length **> 40 m** with ≤2 actions remaining after it.
3. **`opp_shot_within_10s`** — `we_lost` and the opponent shoots within **10 s** of the chain ending.
4. **`backward_reset_turnover`** — `we_lost` and a pass moving backward by **> 5** x-units (`pass_end_x < location_x − 5`) with ≤5 actions remaining.
5. Else **`none`** (not a failure).

Thresholds: `LONG_BALL_METERS = 40.0`, `BACKWARD_X_DELTA = 5.0`, `TURNOVER_X_MAX = 60.0`, `OPP_SHOT_WINDOW_S = 10.0`.

### Features (`features/buildup_features.py`)
One feature row per qualifying chain. The model's `X` (`FEATURE_COLUMNS`, 17 columns total):

**Continuous / engineered (6):**
- `defthird_pressure_density` — opponent `Pressure` events in `[t0, t0+30s]` divided by own chain-event count in the window.
- `first_receiver_pressure_binary` — was the first receiver pressed within 2 s of receiving? (`FR_PRESSURE_BINARY_S = 2.0`)
- `first_receiver_pressure_seconds` — seconds to that pressure (searched up to 5 s ahead).
- `recent_pass_reach` — mean Euclidean length of the focal team's **last 3 passes** before the chain (0.0 if fewer than 3).
- `opp_press_height` — mean `location_x` of opponent defensive events in `[t0−30s, t0)`, reported in the **opponent's own attacking frame** (higher = pressing closer to our goal). *This coordinate convention is a documented "hard rule."*
- `first_receiver_betweenness` — weighted betweenness centrality of the first receiver in a per-match passing-network `DiGraph` (built with `networkx`).

**One-hot game-state (11):**
- `sd_*` (5): score-differential buckets — `minus2_or_worse, minus1, zero, plus1, plus2_or_better`. Score crediting counts both `Shot`+`outcome=="Goal"` **and** `Own Goal For`.
- `min_*` (6): minute buckets — `0_15, 16_30, 31_45plus, 46_60, 61_75, 76_90plus`.

**Dropped:** `support_density_ff` — required 360 freeze-frame data that 404s (see §4). It is explicitly excluded from `FEATURE_COLUMNS`.

Output: `data/marts/buildup_features.parquet` (shape **73 × 22**: keys + 17 features + `is_failure` + `team_name`).

### Driver (`analysis/run_buildup_pipeline.py`)
Reads events + labels via DuckDB, rebuilds chains, joins the `team_id`, filters to qualifying chains, extracts features, joins the label, trains the model, computes SHAP, and persists everything. Teams with no coverage (Philly, Columbus) get a NaN row noted `"no StatsBomb Open Data coverage"`. **Halts (exit 3) if fewer than 3 unique matches exist.**

---

## 10. Module B — post-regain waste

`features/post_regain.py`. After detecting regains, each is enriched by inspecting the subsequent possession chain:
- `reached_final_third` — any event with `location_x ≥ 80` (`FINAL_THIRD_X = 80.0`).
- `lost_within_4_actions` — chain length ≤ 4 **and** terminal is a turnover (`LOST_WITHIN_ACTIONS = 4`).
- `chain_end_type` — `{shot, loss, foul_won, halt, other}`.

**The six metrics (`_apply_all`):**
1. `xg_per_regain` = Σ next-shot xG (NaN→0) / n_regains.
2. `time_to_shot_median` = median of non-null next-shot seconds.
3. `rushed_shot_rate` = among regains that produced a shot, fraction with xG **< 0.05** **and** time **< 8 s** (`RUSHED_XG_MAX = 0.05`, `RUSHED_SECONDS_MAX = 8.0`).
4. `regain_to_final_third_rate` = mean of `reached_final_third`.
5. `regain_to_loss_rate` = mean of `lost_within_4_actions`.
6. `patience_composite` = `z(xg_per_regain) − z(rushed_shot_rate)`, using the ASA baseline means/stds; **degrades to just `z(xg_per_regain)`** when the rushed-shot baseline is NaN (which it is, because ASA season aggregates lack time-since-possession data).

**Uncertainty:** `_bootstrap_metric` resamples **match IDs with replacement** (`n_boot = 1000`, `ci = 0.95`, `seed = 20260421`), computing percentile CIs. `aggregate_team_season` returns one row per metric with `point_estimate, ci_lo, ci_hi, n_regains`.

**Figures produced:** metrics bar, time-to-shot histogram, regain hexbin heatmap (pitch lines at x=40/80), and a Plotly Sankey (regain → {shot ≤15s, final-third no shot, lost ≤4, other}). A `log_sanity` step warns if `n_regains < 300` and reports the no-shot percentage.

**Outputs:** `data/marts/regain_events.parquet` (353 × 12), `data/marts/team_post_regain.csv` (6 metrics).

---

## 11. Module C — the Leverkusen overlay

`analysis/leverkusen_overlay.py`. Computes Leverkusen 23/24's raw build-up failure rate + the six Module B metrics, then overlays them against Inter Miami's persisted aggregates. (The pre/post-vs-2022/23 design was dropped — 22/23 is absent from Open Data.)

Flow:
- `build_leverkusen_marts()` — loads events via DuckDB, determines the focal team as the one appearing in **all** matches, builds chains → filters to focal → labels failures → computes regain events. Writes `leverkusen_2324_chains.parquet` (2863 × 12), `leverkusen_2324_regains.parquet` (2131 × 12), `leverkusen_2324_buildup_labels.parquet` (394 × 4).
- `_diff_bootstrap` — **independently** resamples each team's matches (`n_boot = 1000`, seed `20260421`), computing `leverkusen_metric − miami_metric` per iteration, and returns 2.5/97.5 percentile difference CIs.
- `_assemble_overlay` — one row per metric with both teams' mean+CI, the difference, difference CI, `ci_excludes_zero` flag, and the associational caveat string.

**Outputs:** `data/marts/leverkusen_overlay.csv` (7 metrics) plus figures `leverkusen_overlay_bar.png`, `leverkusen_diff_forest.png`, `leverkusen_vs_miami_regain_heatmaps.png`.

---

## 12. The machine-learning model in detail

`models/buildup_failure_xgb.py` — an **interpretable ML** pipeline built to stress-test whether collapse-shaped possessions carry a measurable geometric signature beyond the rule book.

**Cross-validation:** `GroupKFold(5)` grouped by `match_id` so entire matches stay together across folds (prevents leakage from correlated score trajectories within one game).

**Baseline:** `Pipeline(StandardScaler(with_mean=False) → LogisticRegression(class_weight="balanced", max_iter=1000, solver="liblinear"))`.

**XGBoost:** `RandomizedSearchCV` (`scoring="roc_auc"`, `n_iter=30`) over:
- `max_depth ∈ {3,4,5,6,7}`
- `learning_rate ~ loguniform(0.03, 0.2)`
- `n_estimators ∈ {200,400,600,800}`
- `min_child_weight ∈ {1,3,5,10}`
- `subsample ~ uniform(0.7, 1.0)`
- `colsample_bytree ~ uniform(0.7, 1.0)`

Base config: `objective="binary:logistic"`, `tree_method="hist"`, `eval_metric="logloss"`, `scale_pos_weight = n_neg/n_pos`.

**Calibration:** `CalibratedClassifierCV(method="isotonic")`; a raw (uncalibrated) XGB is also refit on the full data for SHAP.

**Leakage check (small-n aware):** picks one holdout match, trains on the rest, and requires `holdout_AUC > CV_mean − threshold`. Threshold **scales with sample size**: `0.10` when `n < 200`, else `0.05` (`LEAKAGE_SMALL_N_CUTOFF = 200`). This is a deliberate small-sample relaxation — at n=73, any held-out fold can swing ±0.08 by chance.

**Interpretation:** `shap.TreeExplainer` on the **raw** XGB (per spec §4.4). Rankings written to `shap_feature_ranking.csv` (by mean |SHAP|) and `team_shap_profile.csv` (signed mean per team). A calibration curve is saved with quantile binning (`bins = min(10, max(2, sqrt(n)))`).

### What the model actually showed (honest reporting)
- **CV ROC-AUC:** 0.538 ± 0.081 (XGB) vs **0.585 ± 0.067 (LogReg baseline)** — the tuned model **underperforms** the baseline on discrimination by ~0.047 AUC. XGB wins only on Brier (0.280 vs 0.313, a calibration-weighted score).
- **Calibration is visibly off** — non-monotonic, zigzagging across the diagonal (small-n + near-chance discrimination).
- **Gain vs SHAP disagree on which features matter.** XGBoost *gain* importance is dominated by game-state buckets (`sd_zero`, `min_31_45plus`, …), while mean |SHAP| is dominated by pressure geometry (`first_receiver_betweenness` #1, `defthird_pressure_density` #2, `opp_press_height` #3). This gain-vs-SHAP divergence is expected: game-state features split often on small effects; geometry features split rarely with larger effects.
- **The honest takeaway:** at n=73, the "sophistication" isn't earning its keep yet. The model is presented as an *illustrative, small-n* exercise, not a deployable scout.

**Top features by mean |SHAP|** (from `shap_feature_ranking.csv`): `first_receiver_betweenness` (0.515), `defthird_pressure_density` (0.465), `opp_press_height` (0.323), `sd_zero` (0.303), `first_receiver_pressure_seconds` (0.227), `recent_pass_reach` (0.180). The remaining 9 one-hot dummies have zero SHAP (never used in any split).

**Persisted models:** `data/marts/models/buildup_failure_xgb_calibrated.joblib` (~1.57 MB, for prediction) and `buildup_failure_xgb_raw.joblib` (~328 KB, for SHAP).

---

## 13. Data marts — every output file

### CSVs (committed)
| File | Rows | What it is |
|---|---:|---|
| `asa_mls_2023_baseline.csv` | 2 | MLS 2023 league reference: `xg_per_shot` (mean 0.1042, std 0.0088, n=29 teams); `rushed_shot_rate` (all NaN — not derivable from ASA). |
| `asa_team_lookup.csv` | 0 | `team_id → team_name` lookup — header only, unpopulated (so the app never fabricates IDs). |
| `buildup_failure_importance.csv` | 17 | Native XGBoost `feature_importances_`. Top: `sd_zero` 0.149, `first_receiver_betweenness` 0.139, `defthird_pressure_density` 0.131. 9 dummies at 0.0. |
| `cv_metrics.csv` | 10 | Per-fold CV metrics (5 folds × 2 models `logreg_baseline`/`xgb_tuned`): `roc_auc, pr_auc, brier, n_train, n_test, n_pos_*`. |
| `leverkusen_overlay.csv` | 7 | Module C overlay (both teams + diff + `ci_excludes_zero` + caveat). |
| `shap_feature_ranking.csv` | 17 | Global SHAP ranking: `feature, mean_abs_shap, mean_signed_shap, rank`. |
| `team_buildup_failure.csv` | 3 | Team failure rates. Inter Miami: 6 matches, 73 chains, 25 failures, 0.342 [0.253, 0.453]. Philly & Columbus: all-NaN, zero coverage. |
| `team_post_regain.csv` | 6 | Inter Miami's six Module B metrics with CIs, n_regains=353. |
| `team_shap_profile.csv` | 1 | Inter Miami's mean signed SHAP per feature (17 columns). |

### Parquets
| File | Shape | What it is |
|---|---|---|
| `buildup_features.parquet` | 73 × 22 | Model training matrix (features + `is_failure` + `team_name`). |
| `regain_events.parquet` | 353 × 12 | Inter Miami regain events (source of `team_post_regain.csv`). |
| `oof_predictions.parquet` | 73 × 5 | Out-of-fold CV predictions (`y_true, y_pred_proba, fold`). |
| `leverkusen_2324_chains.parquet` | 2863 × 12 | Leverkusen possession chains. |
| `leverkusen_2324_regains.parquet` | 2131 × 12 | Leverkusen regain events. |
| `leverkusen_2324_buildup_labels.parquet` | 394 × 4 | Leverkusen failure labels (`is_failure, failure_type`). |

### Models & cache
- `data/marts/models/` — the two `.joblib` model files.
- `data/marts/cache/` — `asa_league_bundle.json` disk snapshot for the live dashboard (gitignored; regenerated locally).

---

## 14. Figures & the executive summary

**19 PNGs in `docs/figures/`:**
- **Case study series (for notebook 06):** `case_study_01_hook.png` (KPI tiles), `_02_pressure_exposure.png` (regain hexbin), `_03_buildup_failure_rate.png` (failure bar + CI), `_04_top_shap.png` (top-5 SHAP), `_05_post_regain_metrics.png` (6 metrics), `_06_sankey.png` (regain outcomes), `_07_summary.png` (headline tiles).
- **Model diagnostics:** `calibration_curve.png`, `shap_summary.png` (beeswarm), `shap_dep_1/2/3.png` (dependence plots).
- **Module B:** `post_regain_metrics_bar.png`, `post_regain_sankey.png`, `time_to_shot_hist.png`, `regain_zones_heatmap.png`.
- **Module C:** `leverkusen_diff_forest.png` (Leverkusen−Miami forest — cyan when CI excludes zero, gray otherwise), `leverkusen_overlay_bar.png`, `leverkusen_vs_miami_regain_heatmaps.png`.

> Note: Module B emits both legacy filenames (`post_regain_metrics_bar.png`) and notebook-facing `case_study_*` variants of the same underlying tables — alternate styling entry points, same data.

**Executive summary** (`build_executive_summary.py` → `docs/executive_summary.pdf`): a one-page A4-landscape, matplotlib-native, recruiter-facing PDF (TrueType-embedded fonts). It renders Inter Miami's Module B metrics, the Leverkusen−Miami forest, a CI-normalized dumbbell profile overlay, a findings block, a methods block, and a footer — loading only from `data/marts/`.

---

## 15. The Streamlit dashboard

`app/streamlit_app.py` — a **3-page app** using the programmatic navigation API (`st.Page` + `st.navigation`), dark theme (base→panel→card layering, violet `#7C3AED` / cyan `#06B6D4` accents, Inter for text + JetBrains Mono for all numerals). `st.set_page_config(layout="wide")`, `inject_css()` once. Every page ends with an amber **"Associational, not causal"** caveat box.

Shared components:
- **`style.py`** — palette, injected CSS, `metric_card()`, `caveat_box()`, `pitch_plot()` (mplsoccer StatsBomb 120×80 pitch), `dense_cols()`.
- **`asa_refresh.py`** — live ASA fetch with a **24-hour `st.cache_data` TTL**, atomic disk-snapshot fallback (`data/marts/cache/asa_league_bundle.json`), and DataFrame transforms (`merge_league_frames`, `summarize`, `goals_added_rank_table`, etc.). `resolve_league_bundle()` tries network → falls back to disk → raises only if both fail.
- **`inter_miami_plots.py` / `leverkusen_plots.py`** — the chart builders (matplotlib + plotly).

**Page 1 — Inter Miami Diagnostic** (offline; reads `team_buildup_failure.csv`, `team_post_regain.csv`, `asa_mls_2023_baseline.csv`, `regain_events.parquet`): KPI strip (matches, labeled possessions, regains, failure rate + CI), regain hexbin pitch map, six-metric post-regain bar chart with the ASA league xG/shot reference line, a regain-outcome Sankey, and a time-to-shot histogram.

**Page 2 — Leverkusen Pre/Post** (offline; reads `leverkusen_overlay.csv` + optional prepost marts): KPI strip (pre/post counts, largest |Δ|, Δ direction), a forest plot (intra-Leverkusen deltas if available, else cross-case vs Miami), single-metric compare bars for raw failure rate and patience composite, and paired Leverkusen-vs-Miami overlay bars. Gracefully degrades because the true pre/post marts don't exist (22/23 absent).

**Page 3 — MLS League Context** (live; hits ASA): a selectable-season league snapshot with 24h caching + offline fallback. KPI strip (teams, league xG/shot ± σ, mean points, team-games, optional xPass means), an xGoals-for-vs-against scatter with league-mean crosshairs (Inter Miami highlighting disabled because the lookup is empty), a goals-added ranking table, and three **prose-only narrative cameos** (Philadelphia Union, Columbus Crew, San Diego FC) explicitly labeled as *no statistical claims*.

---

## 16. Notebooks

All five are **read-only render notebooks** — they load persisted artifacts from `data/marts/` and `docs/figures/` and never recompute events. Every figure repeats the associational caveat.

- **`03_buildup_failure_model.ipynb`** — Module A model presentation; SHAP summary from the raw XGB (per spec §4.4); documents the Philly/Columbus zero-coverage NaN rows.
- **`04_post_regain_metrics.ipynb`** — Module B; 6 Inter Miami matches, 353 regains, six metrics with 1000-iteration match-resampled bootstrap CIs, ASA xG/shot reference line.
- **`05_leverkusen_overlay.ipynb`** — Module C side-by-side overlay (Leverkusen cyan, Miami violet); explains why pre/post was dropped.
- **`06_inter_miami_case_study.ipynb`** — the Inter Miami narrative case study (Messi hook; the 6 matches are the entire public MLS 2023 corpus).
- **`07_leverkusen_analog.ipynb`** — argues why Leverkusen 23/24 is the deliberately chosen analog (not a similarity-search output) and what is/isn't recoverable under Open Data constraints.

---

## 17. Testing

A standard pytest suite (`tests/`), all **pure/deterministic unit tests over synthetic DataFrames** — no network, no data files. `tests/fixtures.py` provides `ev(**kw)` (one StatsBomb-style event with sensible defaults; team IDs `FOCAL=100`, `OPP=200`) and `frame(rows)`. Bootstrap tests are made deterministic with `seed=42`.

| Test file | Covers | Sample assertions |
|---|---|---|
| `test_smoke.py` | Package import | `pressured_progression.__version__` is truthy. |
| `test_possession_chain.py` | `build_possession_chains` | Single clean chain; break on new `possession_id`; break on team turnover. |
| `test_regain.py` | `detect_regains` | Tackle→shot within 15s window; interception→loss (no shot); recovery with shot past window → None. Verifies zones. |
| `test_post_regain.py` | `compute_regain_events`, `aggregate_team_season` | Enriched fields (final-third, loss-within-4); bootstrap aggregation (`xg_per_regain ≈ 0.05`, `time_to_shot_median == 7.0`, seed 42). |
| `test_buildup_failure.py` | `label_buildup_failures` | One test per failure type, plus happy path + two non-qualifying cases (origin past halfway; no pressure). Coordinates are hand-tuned to isolate each failure type's precedence. |
| `test_buildup_features.py` | `extract_buildup_features`, `FEATURE_COLUMNS` | All features computed; `recent_pass_reach == 10.0`; own-goal score crediting; `support_density_ff` absent; empty-chains → empty. |
| `test_asa_refresh.py` | App-layer ASA transforms | `goals_added_totals_df` sums across actions (13.75 / 7.5); `goals_added_rank_table` orders descending. (Manipulates `sys.path` to import `app/components`.) |

`tests/__init__.py` is empty (just makes `tests` importable for `from tests.fixtures import ...`).

The suite effectively encodes the project's domain thresholds as executable specs: own-half x<40, long-ball >40m, backward delta <−5, 10s/15s windows, rushed xG<0.05 within 8s, and the minute/score-diff buckets.

---

## 18. CI, linting & pre-commit

**CI** (`.github/workflows/ci.yml`) — a single `lint-and-test` job on `ubuntu-latest`, triggered on every `push` and `pull_request`:
1. `actions/checkout@v4`
2. `actions/setup-python@v5` — **Python 3.11 only** (no matrix), pip-cached.
3. `pip install -e ".[dev]"`
4. `ruff check .` (lint)
5. `ruff format --check .` (formatting, non-mutating)
6. `pytest`

**Pre-commit** (`.pre-commit-config.yaml`) — `astral-sh/ruff-pre-commit` v0.6.9: `ruff --fix` + `ruff-format` (both auto-mutate locally). Note the asymmetry: local hooks fix, CI only checks — so a commit that bypasses hooks will fail CI.

**Ruff config** (in `pyproject.toml`): line-length 100, target py311, lint rules `E, F, W, I, UP, B, SIM`, double-quote format.

**`.gitignore`** excludes the usual Python/venv/cache/IDE/OS artifacts, secrets (`.env*`, keeps `.env.example`), and — importantly — the reproducible/large data: `data/raw/**` and `data/core/**` (each keeps a `.gitkeep`), `*.duckdb*`, and `data/marts/cache/*.json`. `data/marts/` itself is committed.

---

## 19. How to reproduce everything

Requires **Python 3.11+**.

```bash
# 1. clone & enter
git clone https://github.com/pressured-progression/pressured-progression.git
cd pressured-progression

# 2. virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate      # macOS/Linux

# 3. install (editable, with dev extras) + hooks
pip install -e ".[dev]"
pre-commit install

# 4. run the ingest + analysis pipeline (in order)
python -m pressured_progression.ingest.statsbomb
python -m pressured_progression.ingest.asa
python -m pressured_progression.ingest.asa_league_baseline
python -m pressured_progression.ingest.leverkusen_ingest
python -m pressured_progression.analysis.smoke_buildup_failure
python -m pressured_progression.analysis.run_buildup_pipeline
python -m pressured_progression.features.post_regain
python -m pressured_progression.analysis.leverkusen_overlay
python -m pressured_progression.analysis.build_case_study_figures
python -m pressured_progression.analysis.build_executive_summary

# 5. launch the dashboard
streamlit run app/streamlit_app.py

# 6. run the tests
pytest
```

The committed `data/marts/` means you can run the notebooks, dashboard, and figure/PDF builders **without** re-fetching raw data. The ingest steps are only needed to rebuild `data/raw` and `data/core` from scratch.

---

## 20. Tech stack

- **Language:** Python 3.11+
- **Data:** pandas, NumPy, **DuckDB** (event-level scans/unions), pyarrow (parquet)
- **Ingestion:** `statsbombpy`, `requests`, BeautifulSoup4 + lxml (FBref), Pydantic (schemas)
- **ML:** scikit-learn, XGBoost, SHAP, scipy, networkx (passing-network betweenness)
- **Viz:** matplotlib, mplsoccer, Plotly (+ kaleido for Sankey export), joblib (model persistence)
- **App:** Streamlit (multi-page `st.navigation`)
- **Tooling:** Ruff (lint + format), pytest, pre-commit, ipykernel
- **CI:** GitHub Actions

---

## 21. Constants & thresholds cheat-sheet

| Constant | Value | Where | Meaning |
|---|---|---|---|
| Defensive-third origin | `start_x < 40` | labeling / regain | Own-third boundary (0–120 pitch) |
| Final-third boundary | `x ≥ 80` | post_regain | `FINAL_THIRD_X` |
| Pressure window (qualify) | first 3 actions | labeling | Opponent pressure must appear early |
| Long ball | `> 40 m` | labeling | `LONG_BALL_METERS` |
| Backward reset | `pass_end_x < location_x − 5` | labeling | `BACKWARD_X_DELTA` |
| Turnover x-cap | `< 60` | labeling | `TURNOVER_X_MAX` |
| Opp-shot window | `10 s` | labeling | `OPP_SHOT_WINDOW_S` |
| Regain shot window | `15 s` | regain | `SHOT_WINDOW_S` |
| Rushed shot | xG `< 0.05` **and** `< 8 s` | post_regain | `RUSHED_XG_MAX`, `RUSHED_SECONDS_MAX` |
| Loss-within | `≤ 4 actions` + turnover | post_regain | `LOST_WITHIN_ACTIONS` |
| Pressure density window | `30 s` | features | `PRESS_DENSITY_WINDOW_S` |
| First-receiver pressure (binary) | `≤ 2 s` | features | `FR_PRESSURE_BINARY_S` |
| Bootstrap iterations | `1000` (2000 in smoke) | multiple | Match-resampled |
| Bootstrap seed | `20260421` | multiple | Reproducibility |
| Leakage threshold | `0.10` if n<200 else `0.05` | model | Small-n relaxation |
| CV | `GroupKFold(5)` by `match_id` | model | No match leakage |
| Zones (x) | `<40` def, `40–80` mid, `≥80` att | regain | `_zone_for_x` |

---

## 22. Known limitations, tech debt & caveats

**Hard scope limits (data reality):**
- **Not a full-MLS study.** Only Inter Miami has MLS 2023 event data (6 matches). No league-wide distribution exists.
- **The two spec-named case teams (Philly, Columbus) are unmeasurable** at event level — zero Open Data coverage.
- **Small n.** 73 qualifying chains; several cross-team differences are swallowed by bootstrap bands.
- **Cross-league confound.** Bundesliga vs MLS differences may reflect competition environment as much as team habit.
- **No causal claims.** Everything is associational. Nothing attributes results to Alonso, Messi, or any tactic.
- **360 data 404s** → `support_density_ff` dropped.
- **FBref blocked** → the 8-dim style vector and the analog-matching engine (`matching/`) were never built.
- **Sample asymmetry** — 34 Leverkusen matches vs 6 Miami matches.

**Model humility:**
- Tuned XGBoost (0.538 AUC) underperforms a LogReg baseline (0.585); calibration zigzags. Scores are **not** deployable as scouting verdicts.
- The leakage check *fails* at the strict 0.05 threshold even with no real leakage — which is exactly why the threshold scales to 0.10 at small n.

**Tech debt noted in `project_info.md`:**
- Score-diff computation originally handled only `Shot=="Goal"` (missed own goals) — the features module now credits `Own Goal For` too.
- `support_density_ff` is dead weight until 360 JSONs land — kept out of the feature set.
- Duplicate figure paths (`post_regain_*` vs `case_study_*`) render the same tables via different styling entry points.

---

## 23. Glossary

- **Possession chain** — consecutive on-ball events owned by one team within a StatsBomb possession block.
- **Regain** — a defensive action (interception/tackle/recovery/duel/block) that flips possession to the focal team.
- **Build-up failure** — a qualifying defensive-third-origin, pressured chain that ends in one of four failure modes.
- **xG** — expected goals; the modeled scoring probability of a shot (StatsBomb `shot_statsbomb_xg`).
- **Bootstrap CI** — an uncertainty band built by resampling **matches** with replacement (not individual events), so the interval reflects match-to-match variability.
- **SHAP** — additive per-feature contributions explaining a model prediction; here summarized as global mean |SHAP| rankings.
- **Patience composite** — `z(xg_per_regain) − z(rushed_shot_rate)`; higher = more patient, higher-quality post-regain attacking.
- **Associational** — describes co-occurrence in the data; explicitly **not** a causal claim.
- **360 / freeze-frame** — StatsBomb's tracking snapshot at each event (positions of all visible players). Advertised-but-404 for MLS 2023 here.

---

## 24. Author & license

- **Author:** Ahmed Kali — `ahmedkali841@gmail.com`
- **Repository:** `github.com/pressured-progression/pressured-progression` *(placeholder until pushed)*
- **License:** MIT (see `LICENSE`)
- **Related docs:** `README.md`, `docs/project_spec.md` (the working contract), `docs/data_reality.md` (coverage audit), `docs/article_draft.md` (long-form writeup), `project_info.md` (Phase-3 calibration snapshot), `docs/executive_summary.pdf` (one-pager).

> **One sentence worth the scroll:** the value here is *disciplined description under known data limits* — a named label set, reproducible pipelines, uncertainty on every bar, and a Leverkusen overlay anchored in what open data actually stores — not a playbook imported from anyone.
