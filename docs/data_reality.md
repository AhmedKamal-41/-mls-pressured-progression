# Data Reality

**Written:** 2026-04-21 (day 1). Update as coverage questions are resolved.

<!-- REVIEW(spec §3): project_spec.md §3 assigns StatsBomb the role “Primary event data (MLS 2023 + European analogs)” and FBref “Rate-limited HTTP + BS4.” This doc’s audit implies those role lines are overstated for Open Data / current FBref access — resolve in spec or add “as audited [date]” qualifiers in §3. -->

This document records what the three data sources *actually* give us, versus what the project spec assumed. The headline: **StatsBomb Open Data does not have the MLS 2023 teams the spec names as case studies, and three of the four European analog candidates are also absent.** The spec's base-year anchor and its analog set both need revisiting.

---

## 1. StatsBomb Open Data — coverage audit

Source: `statsbombpy.sb.competitions()` + `sb.matches(...)` for each target competition-season. Raw catalog saved to `data/raw/statsbomb_catalog.csv`.

<!-- REVIEW(dating): Tie every coverage snapshot to a frozen artifact or date (e.g. catalog file mtime, git hash, or explicit “queried YYYY-MM-DD”) so “6 MLS 2023 matches” doesn’t read as timeless truth if Open Data changes. -->

### MLS coverage

| Season | Matches | 360 data | Notes |
|---|---|---|---|
| 2023 | **6** | **6 / 6** | **All 6 are Inter Miami fixtures** (the Messi data release). No Philadelphia Union, no Columbus Crew, no other teams. |

No other MLS seasons are present in Open Data.

**Implication:** The project spec names Philadelphia Union 2023 as the primary failure case and Columbus Crew 2023 as the internal positive control. Neither team has *any* StatsBomb event-level coverage. The buildup-failure label, possession chains, and regain sequences — as defined in `docs/project_spec.md` §4 — cannot be computed for these teams from Open Data.

<!-- REVIEW(spec crosswalk): Contradicts project_spec.md §2 (“Columbus Crew 2023 — … full StatsBomb 2023 coverage”). Not §3 per se, but readers will notice; either footnote spec pending update or cite the audit date here explicitly. -->

### Coverage for the four European analog candidates

Scanned every Open Data season for any appearance of the candidate teams:

| Candidate | Spec-requested seasons | What Open Data actually has | Verdict |
|---|---|---|---|
| **Brighton & Hove Albion** | 2021/22, 2023/24 (men's, PL) | Only women's team (FAWSL 2018/19, 2019/20, 2020/21). **No men's PL data.** | **NO-GO** |
| **Bologna** | 2022/23, 2023/24 | Serie A 2015/16 only | **NO-GO** |
| **Girona** | 2022/23, 2023/24 | La Liga 2017/18, 2018/19 (pre–big leap) | **NO-GO** |
| **Bayer Leverkusen** | 2022/23, 2023/24 | Bundesliga 2015/16 (full season, **no 360**); Bundesliga **2023/24 (34 matches, all with 360)** | **GO** for 23/24; no-go for 22/23 |

<!-- REVIEW(spec §3): Spec’s European list includes seasons 2021/22–2023/24; this table’s “what Open Data has” mixes seasons (15/16, 17/18, etc.). Good for go/no-go, but add one line clarifying whether “NO-GO” means “not in Open Data at requested seasons” vs “no men’s team in Open Data at all” (e.g. Brighton) so §3 readers don’t over-interpret. -->

### 360 availability

Open Data 360 coverage is sparse. The only target-competition seasons with any 360 data:

| Competition | Season | 360 matches |
|---|---|---|
| MLS | 2023 | 6 / 6 (all Inter Miami) |
| La Liga | 2020/21 | 35 / 35 (Messi's final Barça season) |
| Bundesliga | 2023/24 | 34 / 34 (Leverkusen only) |

The named big-5 leagues otherwise have 360 counts of zero in Open Data.

<!-- REVIEW(dating): “Zero in Open Data” is a snapshot claim — same as above, pin to catalog date/version. -->

#### Catalog vs. repo: 360 JSON files for MLS 2023 return 404

Addendum (2026-04-21, Phase 4): the match catalog reports `match_status_360 == "available"` for all 6 MLS 2023 matches, but the underlying 360 JSON files on the StatsBomb Open Data GitHub repo return **HTTP 404** for every one of those 6 match IDs (verified via `statsbombpy.sb.frames`). This is distinct from the catalog-level claim — the catalog says "available," the data is missing from the repo.

Downstream implication: the `support_density_ff` feature in `features/buildup_features.py` — which requires freeze-frame data — is **all null for all chains** in the MLS 2023 run. The extractor handles this per spec ("Null where freeze_frame absent") but any feature-importance or SHAP signal involving `support_density_ff` will be degenerate in this dataset.

To recover: retry after StatsBomb publishes the 360 JSONs, or acquire 360 via their paid API.

#### Phase 5 appendix — Leverkusen 22/23 absent from Open Data (re-confirmation)

Addendum (2026-04-21, Phase 5): before starting Module C, re-scanned every Open-Data competition-season for any Leverkusen appearance. Confirmed:

| Competition | Season | Leverkusen matches | with 360 |
|---|---|---|---|
| 1. Bundesliga | 2015/2016 | 306 (all Bundesliga 15/16) | 0 |
| 1. Bundesliga | 2023/2024 | 34 | 34 |

There is no Bundesliga 2022/23 season in the Open Data catalog at all (catalog lists only 2015/16 and 2023/24 for Bundesliga). The pre/post-vs-22/23 design in the original Module C spec cannot be executed on Open Data; Module C was reframed to a Leverkusen 23/24 × Inter Miami 2023 overlay. Spec §4 Module C and §5 Case B were patched accordingly.

---

## 2. American Soccer Analysis (ASA) v1 API — endpoint audit

Source: `https://app.americansocceranalysis.com/api/v1/mls/...`. All three endpoints returned **HTTP 200**. Raw JSON in `data/raw/asa/`.

<!-- REVIEW(dating): HTTP 200 + schema keys are time-stamped only by doc “Written” date and raw JSON paths; say explicitly when endpoints were probed if API versioning or fields could drift. -->

<!-- REVIEW(metric availability): Keys listed describe response shape, not which seasons/years ASA exposes — align wording with project_spec.md §3 “Trend context 2020–present” (still true?) and note any endpoint-level season coverage limits. -->

### `teams/xgoals` — 31 teams

Keys: `team_id`, `count_games`, `shots_for`, `shots_against`, `goals_for`, `goals_against`, `goal_difference`, `xgoals_for`, `xgoals_against`, `xgoal_difference`, `goal_difference_minus_xgoal_difference`, `points`, `xpoints`.

### `teams/goals-added` — 31 teams

Keys: `team_id`, `minutes`, `data` (nested list — action-type goals-added breakdown: Dribbling, Fouling, Interrupting, Passing, Receiving, Shooting per side).

### `teams/xpass` — 31 teams

Keys: `team_id`, `count_games`, plus for/against/difference triples for `attempted_passes`, `pass_completion_percentage`, `xpass_completion_percentage`, `passes_completed_over_expected`, `passes_completed_over_expected_p100`, `avg_vertical_distance`.

### ASA fit for this project

- ASA is **team-season aggregate**, not event-level. It cannot build possession chains, pressure events, or regain sequences.
- `avg_vertical_distance` and `passes_completed_over_expected` are the closest things ASA gives us to build-up style proxies at season level.

<!-- REVIEW(proxy): Explicitly tag these as proxies in downstream outputs/tables (not just “closest things”) per your proxy rule — this bullet says “proxies” once; ensure project_spec §5 metrics aren’t implied to be derivable from ASA alone. -->

- All endpoints default to current season (2026) — parameters (`season`, `start_date`, `end_date`) will need to be passed to pin to 2023 for the spec's base year.

<!-- REVIEW(dating): “Default 2026” is a dated claim as of audit; if ASA changes defaults, this sentence rots — consider “as of [probe date]”. -->

---

## 3. FBref — blocked

Both target pages (Philadelphia Union 2023, Brighton 23/24) returned **HTTP 403** with a Cloudflare "Just a moment..." challenge page. This held across both a research-identifying UA and a standard Chrome UA. Rate-limited requests are not the blocker — first-request 403 before any rate signal exists.

<!-- REVIEW(spec §3 inconsistency): Spec lists FBref access as “Rate-limited HTTP + BS4”; this section says access failed at 403 / Cloudflare before rate limits matter. Resolve §3 wording (blocked vs rate-limited) vs reality. -->

<!-- REVIEW(dating): 403 probe undated beyond document day 1 — add probe timestamp or artifact (screenshot/hash) if this claim must survive dispute. -->

**Options (to decide before Day 2):**

1. **Use FBref's per-page CSV export** manually — FBref's own "Share & Export > Get table as CSV" produces the same data without gated scraping. Viable for ~10 team-seasons; painful at scale.
2. **Sports Reference stats bulk files / FBref partner access** — requires outreach.
3. **Headless browser** (Playwright) — solves the JS challenge but adds infra.
4. **Drop FBref in favor of ASA-only style features for MLS**, and `statsbombpy` season-aggregates for European style.

For the 8-dim style vector this is the pivotal decision: FBref is where the MLS ↔ European column alignment lives. Without it we either (a) download CSVs by hand for the specific teams we match, or (b) redesign the style vector on ASA-compatible dimensions (which doesn't help for European analogs, where ASA is silent).

The originally-requested FBref column-overlap log (MLS ↔ Premier League) was skipped because neither page returned.

<!-- REVIEW(limitations vs hedging): Failure mode is clearly stated (403 → no overlap log). Good. If “skipped” reads as deferrable rather than blocked, tighten one clause when you resolve — not useless hedging, just scope clarity. -->

---

## 4. What we can measure, proxy, or not

Given the above, here's what the spec's measurement plan looks like in practice:

<!-- REVIEW(structure): There is no dedicated “Limitations” heading; constraints live under §4 and §3. Fine if intentional; otherwise consolidate so readers don’t miss honest limits scattered across sections. -->

### Measured directly (from available data)
- Inter Miami 2023 event-level: all spec metrics (build-up failure rate, post-regain retention, progressive-action timing) computable on 6 matches with 360.
- Bayer Leverkusen 2023/24 event-level: same, on 34 matches with 360.
- Season-aggregate xG, goals-added by action type, xPass metrics for all 31 MLS teams 2020–present (ASA).

<!-- REVIEW(dating + precision): “2020–present” should be anchored to ASA API reality (first season available, and what “present” meant on audit date). -->

<!-- REVIEW(overclaim risk): “All spec metrics … on 6 matches” may conflict with statistical power / missing game states / label stability; if you mean “technically computable from schema,” say so — avoids sounding like fully powered spec compliance. -->

### Proxied (weaker substitute; label outputs accordingly)
- **"MLS build-up pressure profile"** via ASA `passes_completed_over_expected` + `avg_vertical_distance` aggregates, labeled as season-level proxy, not event-level.
- **Post-regain efficiency** proxied via ASA goals-added Receiving + Passing splits. Weak — misses timing windows the spec wants.

<!-- REVIEW(proxy flag): Good “Weak” qualifier; ensure outputs label this as proxy not post-regain phase metric per §5. -->

- **European analog style vectors** — if FBref is recovered, this becomes direct; otherwise proxied via StatsBomb aggregates for the few available European seasons.

<!-- REVIEW(proxy clarity): “StatsBomb aggregates” is vague (team-season from events vs packaged aggregates vs non-360 leagues). Flag proxy type explicitly when you implement so §5 style vector isn’t silently half-direct. -->

### Out of scope (cannot do honestly with available data)
- **Philadelphia Union 2023 build-up collapse case study** — no StatsBomb coverage. Drop, or reframe as ASA-season narrative only.
- **Columbus Crew 2023 positive control** — same. Drop the internal control, or reframe.
- **Brighton / Bologna / Girona 2022–2024** as European analogs — no StatsBomb coverage in the requested years.
- **Pre/post metric movement around tactical pivots** at event level for any MLS team other than Inter Miami 2023.

---

## 5. Base year justification

The spec fixed 2023 as the base year "StatsBomb Open Data coverage constraint." That rationale no longer holds as stated.

<!-- REVIEW(base year defensibility): The rebuttal (“constraint no longer holds”) is clear given the MLS Open Data counts above. Also reconcile with project_spec.md §2 temporal framing bullet that still cites the same constraint — duplicate truth source until spec edits. -->

<!-- REVIEW(spec §3): §3 doesn’t name the base year; §2 does. “Why 2023 specifically” is defended here as spec legacy + Open Data patchiness; add explicit sentence if non-StatsBomb rationale for 2023 (e.g. ASA alignment, narrative pivot calendar) matters to stakeholders. -->

For MLS at the **event level**, Open Data provides only 6 Inter Miami fixtures in 2023 — not a team-wide or league-wide base. That's a useful *case record* (Messi's arrival), not a case study of either of the teams the spec names. For 2023 to remain the base year, the project must either:

- (a) Narrow the MLS event-level analysis to **Inter Miami 2023** specifically (with 360 data), and treat other MLS teams as season-aggregate only via ASA.
- (b) Keep Philly/Columbus 2023 as the stated case studies but **re-source their event data** (e.g., paid StatsBomb, Opta, Wyscout) — out of scope for an open-data project.
- (c) Shift the MLS base year to a season with broader Open Data coverage — there is none. MLS in Open Data is one season, 6 games.

On the European side, the only analog candidate with 2023/24 event coverage and 360 is **Bayer Leverkusen** (Xabi Alonso's title season). That's a strong single analog but not a set of four.

### Recommendation (for discussion, not decided)

Reframe the project as:
- **MLS trend context: 2020–present** via ASA team-season aggregates (all teams, all metrics).
- **MLS event-level deep dive: Inter Miami 2023** only (with 360).
- **European analog: Bayer Leverkusen 2023/24** as a single, high-quality comparator (with 360).
- **Narrative cameos** unchanged: Philly 2026 and San Diego 2025 per spec (prose only).

This preserves the project's two-failure-mode thesis but acknowledges that the original team-specific case studies are not executable on the chosen data stack.

<!-- REVIEW(tone): Recommendations are concrete (not hedged into mush). Ensure any adopted path updates spec so “honest limitations” stay contract-true — no extra rewrite here. -->

---

## 6. Action items

- [ ] Decide reframe (§5) or alternative data procurement before writing any modeling code.
- [ ] Decide FBref path: manual CSV, Playwright, partner access, or drop.
- [ ] If reframe is accepted, update `docs/project_spec.md` §2 (scope and framing) and §5 (metrics) accordingly. Hard don't #3 prevents metric changes without a spec edit.
- [ ] Pull ASA with explicit `season=2023` query param to align trend context with event-level analyses.
