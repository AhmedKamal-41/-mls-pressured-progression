# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project: Pressured Progression

Measures two MLS failure modes and finds European analogs via statistical similarity:

1. **Build-up collapse** when pressed in own half
2. **Post-regain waste** — winning the ball but rushing the attack

Output includes pre/post metric movement around documented tactical pivots.

## Temporal framing (strict)

- **Base data year: MLS 2023** — StatsBomb Open Data coverage constraint. All event-level statistical claims anchor here.
- **Supplementary:** ASA 2020–present for trend context.
- **Narrative-only:** Philadelphia Union 2026 (Inquirer), San Diego FC 2025 (Guardian). These appear in article prose only — never attach statistical claims to them without an explicit "as of 2023" label.

## Case study teams

- **Philadelphia Union 2023** — primary failure case
- **Columbus Crew 2023** — internal positive control (MLS Cup winners, full StatsBomb 2023 coverage under Wilfried Nancy)
- **San Diego FC** — narrative cameo only, no event-level claims

## Data sources

- **StatsBomb Open Data** via `statsbombpy`
- **American Soccer Analysis API** — `https://app.americansocceranalysis.com/api/v1/`
- **FBref** via rate-limited requests (**max 3/min**)

## Pipeline

```
data/raw  →  data/core (typed, validated)  →  data/marts (analysis-ready)
```

Core marts:
- `possession_chains`
- `pressure_events`
- `regain_sequences`
- `team_season_features`

## Key abstractions

- **`possession_chain`** — consecutive events with constant `team_id`
- **`regain`** — defensive action transitioning possession opp → us
- **`buildup_failure_label`** — binary target. Defensive-third origin + opponent pressure within 3 actions + failure termination (turnover, forced long ball, opponent shot ≤10s, or backward reset + turnover ≤5 actions).

## Coding conventions

- **Python 3.11**; type hints on public functions
- **Ruff** for linting, **pytest** for tests, **pydantic** for schemas
- **DuckDB** for local analytics on events — do NOT load a full season into pandas
- Every core transformation has a pytest test
- **Associational language only** in docstrings and outputs: "coincided with", "associated with". Never "caused" / "because of".

## Hard don'ts

- **Don't auto-commit.** Stage only — the user reviews and pushes.
- **Don't attach StatsBomb 2023 numbers to a 2026 team claim without labeling.**
- **Don't add new top-level metrics without editing `docs/project_spec.md` first.**
- **Don't frame the Tactical Precedent Library as prescriptive** ("do X"). It's associational precedent only: "teams with similar profiles that improved did X; their metrics moved in Y direction."

## Reference

- Full spec: `docs/project_spec.md`
- Data audit: `docs/data_reality.md` (write on day 1, update as you go)
