# project_info.md

State-of-project calibration document. Written 2026-04-21 after Phase 3 (build-up failure modeling on MLS 2023). Numbers pulled from `data/marts/*`, `docs/figures/*`, and the Phase 3 pipeline stdout. "Don't know yet" marks anything not in the repo.

> Note on filename: your prompt opened with `project_phase.md` and closed with `put that in project_info.md`. I went with `project_info.md` per the last line. Let me know if you want it renamed.

## **A — Model performance (XGBoost on Inter Miami 2023, n=73 chains over 6 matches)**

1. **CV ROC-AUC:** 0.538 ± 0.081 (GroupKFold×5 by match_id) — *source: Phase 3 stdout; not persisted to disk. Tech debt.*
2. **CV PR-AUC:** 0.485 ± 0.118
3. **CV Brier:** 0.280 ± 0.016
4. **Held-out match sanity check:** held-out match=3877170, held-out AUC = **0.458**, CV mean = 0.538. Spec threshold: held-out > CV mean − 0.05 = 0.488. **FAIL** (0.458 < 0.488). Not true label leakage — small-n structural correlation: score-diff one-hots had η² vs match_id of 0.31–0.65 (each match has one score trajectory).
5. **Baseline LogReg ROC-AUC:** 0.585 ± 0.067. XGBoost **underperformed** baseline by **−0.047 AUC**. XGB did win on Brier (0.280 vs 0.313), which is calibration-weighted loss, not discrimination.
6. **Calibration curve:** **visibly off** — 8 bins, non-monotonic, zigzags across the diagonal (0.5 observed at 0.12 predicted; dips to 0.11 at 0.29; rises to 0.56 at 0.50; dips to 0.22 at 0.68). Small-n + near-chance discrimination.

## **B — Class balance and sample sizes**

7. **Total labeled possessions:** **73** (all Inter Miami; Philly and Columbus contribute 0 each — no Open Data coverage).
8. **Class balance:** 25/73 = **34.2%** labeled failure.
9. **Matches with 0 qualifying possessions:** **0 of 6** — every Inter Miami match had ≥ 6 qualifying chains. Min 6, max 23.
10. **Median qualifying possessions per match:** **11.5** (per-match counts: 6, 8, 11, 12, 13, 23).

## **C — Feature importance**

11. **Top 5 by mean |SHAP|** (raw XGB, *source: Phase 3 stdout; not persisted as a CSV — tech debt*):
   1. `first_receiver_betweenness` — 0.606
   2. `defthird_pressure_density` — 0.521
   3. `sd_zero` — 0.371
   4. `opp_press_height` — 0.296
   5. `recent_pass_reach` — 0.225
12. **Game-state feature ranks by mean |SHAP|:**
   - `sd_zero` — **#3** (only game-state bucket in top 5)
   - All other `sd_*` and `min_*` buckets rank below #5. Exact mean-|SHAP| for them wasn't captured to disk. Signed mean SHAP (from `team_shap_profile.csv`) for the rest has |value| < 0.03.
   - In contrast, **XGBoost gain importance** (`buildup_failure_importance.csv`) is dominated by game-state buckets — top 5 by gain: `sd_plus1` (0.122), `sd_plus2_or_better` (0.087), `min_31_45plus` (0.086), `min_61_75` (0.084), `min_0_15` (0.084). Gain vs |SHAP| divergence is typical: game-state features split often on small effects; betweenness/pressure density split rarely with larger effects.
13. **`support_density_ff` vs `recent_pass_reach`:**
   - `support_density_ff` has **0/73 non-null values** (360 JSONs return HTTP 404 for all 6 matches — see `docs/data_reality.md` §1 addendum). Mean SHAP = 0.0; gain = 0.0. Dead weight in this run.
   - `recent_pass_reach` carries the support-adjacent signal (mean |SHAP| = 0.225, rank 5). Not equivalent to on-pitch support density, but the only non-null proxy present.
14. **Near-zero SHAP / gain (drop candidates for v2):**
   - `support_density_ff` (definitional — all null)
   - `sd_minus1`, `sd_minus2_or_worse`, `min_16_30`, `min_76_90plus` — all at 0.0 gain importance (never used in any tree split)
   - Candidate to drop: all five above, conditional on future data confirming they remain inactive once n grows.

## **D — The core Philly vs. Columbus result**

15. **Philadelphia Union 2023 failure rate + CI:** **undefined** — 0 matches in StatsBomb Open Data. `team_buildup_failure.csv` shows NaN.
16. **Philly rank:** undefined.
17. **Columbus Crew 2023 failure rate + CI:** **undefined** — 0 matches in Open Data. NaN in mart.
18. **Columbus rank:** undefined.
19. **Absolute gap:** undefined.
20. **CI overlap / statistical meaningfulness:** undefined — no point estimates, no CIs.

## **E — MLS distribution shape**

21. **MLS median failure rate:** 0.342 — a single team (Inter Miami), so the "median" is the point estimate.
22. **MLS range:** min = max = Inter Miami 0.342 (95% bootstrap CI [0.253, 0.453]). No distribution across teams.
23. **Surprising teams:** N/A — only one team in the MLS 2023 event-level corpus. Distribution shape cannot be assessed.

## **F — Data coverage reality**

24. **MLS 2023 matches with usable event data after filtering:** **6** (all Inter Miami; the Messi data release).
25. **Teams with < 15 matches:** **all of them**. Only one MLS team is represented, with 6 matches. Flag for Phase 4: **catastrophic** for any MLS-wide team-season modeling ambition.
26. **Total regains across MLS 2023 (focal = Inter Miami):** **353** across 6 matches (per-match: 48, 54, 56, 59, 66, 70). Mean 58.8, median 57.5. Sufficient sample size for Module B *on Inter Miami specifically*; not sufficient to compare across MLS teams.

## **G — European analog readiness**

27. **StatsBomb Open Data coverage of analog team-seasons:**

   | Team-season | Coverage |
   |---|---|
   | Brighton 21/22 (men) | **No** — women's team only in Open Data (FAWSL) |
   | Brighton 23/24 (men) | **No** — same |
   | Bologna 22/23 | **No** — only Serie A 15/16 |
   | Bologna 23/24 | **No** |
   | Girona 22/23 | **No** — only La Liga 17/18 & 18/19 |
   | Girona 23/24 | **No** |
   | Leverkusen 22/23 | **No** — only Bundesliga 15/16 + 23/24 |
   | Leverkusen 23/24 | **Yes** — 34 matches, all with 360 available |

28. **Module A + B directly computable for analog teams:**
   - **Leverkusen 23/24:** yes, directly.
   - **All others:** no — will need Phase 5 proxies (or paid data).

## **H — Anything surprising**

29. **Findings that complicate the project thesis:**
   - **Philly and Columbus have zero StatsBomb Open Data matches in 2023.** The two named case-study teams cannot be measured at event level on the chosen data stack. This is the headline finding and it invalidates the Phase 0 thesis design as-stated.
   - **MLS Open Data 2023 = 6 Inter Miami matches only** (the Messi release). There is no "MLS-wide distribution" to rank teams within.
   - **Tuned XGBoost underperforms a LogReg baseline** on AUC (0.538 vs 0.585). Not necessarily a methodology problem — at n=73 with tree ensembles, this inversion is within noise. But it does mean the "sophistication" isn't earning its keep yet.
   - **Gain vs mean-|SHAP| disagree** on which features the model uses. Gain picks score/minute buckets; SHAP picks betweenness/pressure. Expected phenomenon, but worth naming in any write-up.
   - **Catalog ≠ repo for 360 data.** All 6 MLS 2023 matches advertise `match_status_360 = "available"` but the 360 JSON files 404. Logged in `docs/data_reality.md` §1.
   - **Leakage check fails at n=73** even when there's no actual leakage. The spec's 0.05 threshold is too tight for this sample size; score-diff one-hots naturally get 30–65% η² vs match_id when each match has one score trajectory.

30. **Tech debt to address before Phase 4:**
   - CV metrics (ROC-AUC / PR-AUC / Brier per fold) are not persisted. `TrainingResult` lives only in process stdout. Should write `cv_metrics.csv` and `oof_predictions.parquet` to `data/marts/`.
   - Mean-|SHAP| ranking isn't saved either — `team_shap_profile.csv` stores signed means per team, not absolute means per feature. Add `shap_feature_ranking.csv`.
   - `support_density_ff` is all null until 360 JSONs land. Either drop the feature until then, or keep and accept the zero contribution.
   - `_available_teams()` in `run_buildup_pipeline.py` is defined but unused — dead code.
   - Leakage-check threshold (0.05) should be relaxed or scaled with n. At n=73, any held-out fold can land ±0.08 by chance alone.
   - `opp_press_height` coordinate convention (opponent's attacking frame) is documented in the module docstring but not in `docs/project_spec.md` §5. Fold in — hard don't #3 says metric definitions live there.
   - Score-diff computation assumes `shot_outcome = "Goal"` only — misses own goals (separate StatsBomb event type). Low frequency but should be handled before cross-league work.

## **I — Gut read**

31. **Philly-as-failure-case / Columbus-as-positive-control framing:** **no, it should change.** Neither team is measurable event-level in our data. The options are:
   - **(a)** Accept Inter Miami 2023 as the MLS event-level case (the Messi team under pressure — has a natural build-up-distress narrative) and reframe around that single team with ASA season aggregates for MLS context.
   - **(b)** Procure paid StatsBomb / Opta / Wyscout data for Philly + Columbus 2023. Moves the project out of "open data" scope.
   - **(c)** Drop MLS event-level entirely; build the thesis on ASA season aggregates + FBref (once FBref blocking is resolved) across the full MLS 2023 table. Loses possession-chain resolution.
   - My lean: **(a)** pairs naturally with Leverkusen 23/24 as the European analog (both have full 360, both are narratively "a single star team's year"). Symmetric data, symmetric claim granularity. Columbus 23 as "Nancy positive control" is a bigger deviation from the original pitch but more data-honest.

32. **Thesis holding up in the data:**
   - **Can't be evaluated yet.** The two failure modes (build-up collapse, post-regain waste) are computable. Whether MLS teams *systematically* fail at them, and whether Philly / Columbus sit at the extremes, is untested and untestable on Open Data.
   - Before writing anything in Week 6, the case-study identity question has to be settled (item 31). Writing the article with "case study = Inter Miami / Leverkusen" vs "case study = Philly / Columbus (ASA aggregates only)" produces very different pieces.
   - Recommend one decision meeting to pick (a) / (b) / (c), then `docs/project_spec.md` §2 + §5 edits (per hard don't #3), then Phase 4 starts from the reframed scope.
