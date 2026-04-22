# Pressured Progression: A Messi-Era Miami Meets Alonso's Leverkusen

## 1. Opening — one sequence, three lenses

Late in the first half of one of Inter Miami's six MLS fixtures packaged in StatsBomb's 2023 Messi-era Open Data drop—coverage runs across late summer and autumn 2023—the broadcast tells a familiar story in miniature. Miami wins the ball inside its own defensive third: a tackle-turned-interception, maybe a midfielder stepping into a lane to choke off a bounce pass before it ever reaches an opponent's forward line. For a beat, bodies tilt toward the attacking goal; the stadium noise lifts as if recognizing the chance to punish a team caught high; the director cuts from the recovering defender to whoever has taken the restart. What happens next is usually less cinematic than it sounds on television: rather than settling into a composed restart—two sideways touches to clean the slate, scan, then progress—the next few touches climb the pitch in a hurry.

On the feed you see it as tempo: midfield runners drag markers, wings pin back lines, commentators reach for verbs like **spring** or **bite**. Slow the clip and the geometry is thinner than the narration. A pass arrives under modest pressure—the sort that does not register as an error on the naked eye because it still finds a teammate—yet the teammate's first touch points back into traffic instead of widening the pitch. Possession breaks, or narrows into a hurried shot cue that barely registers as a scoring chance until the replay adds the x-axis label you never wanted to think about during live play.

The same passage, read through event data, is less about vibes and more about countable sequences. StatsBomb-style feeds record each touch as an on-ball event with coordinates on the competition's pitch grid; possessions stitch together while team IDs remain constant (`docs/data_reality.md` ingestion notes align with StatsBomb publishing practices for Open Data bundles). What the clip *felt* like—danger followed by relief or regret—is compatible with two project measurements that sit underneath the spectacle. First, whether the buildup segment qualified as a **failure** under an explicit defensive-third + pressure heuristic (labels described in `docs/project_spec.md`, implemented in code). Second, whether the regain that followed it became **waste** in the narrow operational sense used here: the ball returns, but the next handful of seconds do not reliably turn into territorial gain or a sober shot cue.

That gap—between broadcast intuition and countable events—is where this piece lives. Nothing here claims to explain *why* the sequence unfolded the way it did. The argument is narrower: Miami's finite Open Data corpus makes it possible to quantify how often similar patterns recur, with uncertainty shown on the plots produced for this project—and to contrast that snapshot with a Bundesliga analog season where full-season event coverage exists.

---

## 2. Two failure modes (plain language)

**Build-up collapse under pressure** names a situation most viewers already recognize under some nickname: chasing shadows in your own half when the opponent steps up line by line. In code, Miami's buildup chains are flagged when possession begins in Miami's defensive third while an opponent pressing action appears within the first three on-ball touches, then terminates in specified ways documented in `docs/project_spec.md`—for example turnovers, resets that shortly precede turnover, concession of a dangerous shot shortly downstream, or other failure-type endings captured in labeling rules. Plainly: you're trying to climb out while someone is leaning on you, and you don't manage it—without assigning blame to any single passer.

**Post-regain waste** shifts the clock forward one beat. Instead of evaluating the guarded exit from deep, this mode asks what Miami does with **newly won** possession: does the regain turn into steady progression, entry into the final third within a bounded window, or at least a shot attempt soon enough to count as decisive in this framework? If not—if the ball returns and the next moments behave like hurry rather than setup—the window is categorized as wasted for measurement purposes.

Both failure modes coexist in real football; neither implies the other logically. Collapse refers to buildup under harassment; waste refers to the impulse right after flipping the ball.

---

## 3. Why Inter Miami 2023—and why honesty about scope

StatsBomb released a six-match MLS 2023 bundle centered on Inter Miami—every Open Data row from that league-season belongs to those fixtures (`docs/data_reality.md`, catalog frozen at project audit). That release is not a league panel; it is a **Messi-hook dataset** that happens to carry enough touches for chain- and regain-level work on one club.

The honest scope statement is blunt: **six matches** are enough to illustrate methodology and show uncertainty bands; they are not enough to crown Miami as "typical MLS," nor to rank Miami against clubs that never appear in Open Data (Philadelphia Union and Columbus Crew have **zero** Open Data matches in 2023 per the audit). Treat the numbers that follow as **Inter Miami's 2023 Open Data profile**—associational, not representative of MLS as a whole.

American Soccer Analysis (ASA) supplies full-table team-season aggregates for MLS in the supplementary layer of the project ([ASA API v1](https://app.americansocceranalysis.com/api/v1/)); those surfaces matter for league-wide context elsewhere in the deliverables but cannot reconstruct single-touch chains.

---

## 4. Metrics—how they're built (without equations)

### Module A — supervised labeling + interpretable ML

Module A attaches a binary **buildup failure** label to qualifying possessions using deterministic rules documented in `docs/project_spec.md`: defensive-third origins, opponent pressure flagged within an early-action window, and discrete failure endings—turnovers, resets that coincide with giveaways, concessions of dangerous shots shortly downstream, among other enumerated cases implemented in labeling code. Features feeding the classifier summarize geometry of pressure, recipient roles, and timing derived from StatsBomb fields—examples saved in calibration logs include graph-style summaries of receiver positioning (`first_receiver_betweenness`), defensive-third pressure density (`defthird_pressure_density`), opponent press-height summaries with explicit opponent-attacking-frame coordinate conventions (`opp_press_height`; see `docs/project_spec.md` hard rule on notation), plus score-state buckets (`sd_*`) and minute buckets (`min_*`) captured because game script influences risk appetite even when causality stays off the table.

That classifier is **interpretable machine learning** in the plain sense recruiters use: gradient boosted trees (XGBoost) trained with cross-validation blocked by match so entire games stay together across folds—reducing leakage across correlated score trajectories within one fixture.

Why bother with trees if labels already exist? Two answers coexist in this project: (1) the supervised target stress-tests whether collapse-shaped possessions share measurable geometric signatures beyond the rule book—useful when expanding to leagues with richer coverage later; (2) inspection tools like SHAP reveal whether those signatures cluster on pressure geometry versus scoreboard bookkeeping at small *n*. Performance should be read modestly: pooled cross-validated ROC-AUC for the tuned gradient boosting model averages **0.548** across folds (`data/marts/cv_metrics.csv`), while a logistic baseline averages **0.586**—discrimination sits near coin-flip territory at this sample size. Calibration plots zigzag across reliability (`docs/figures/calibration_curve.png`). **SHAP** values summarize how each modeled feature pushes predictions above or below baseline risk on labeled examples—think of them as additive contributions rather than courtroom evidence; `docs/figures/shap_summary.png` collects the roll-up global picture alongside dependency plots `docs/figures/shap_dep_*.png`.

### Module B — six regenerate-facing composites

After regains are detected from defensive transition events (`sequences/regain.py`), Module B computes six team-season aggregates with **match-level bootstrap confidence intervals**: expected goals accumulated on shots soon after regain, median seconds-to-shot among attempts inside a capped horizon, rate of low-quality rushed attempts, rate of progression to the final third within the modeled chain, rate of rapid turnover after regain, and a patience composite blending shot-quality pressure against rushed-shot tendencies (`src/pressured_progression/features/post_regain.py`). League ASA moments enter only where documented—some composites z-score against ASA MLS 2023 season means when baselines exist (`data/marts/asa_mls_2023_baseline.csv`).

---

## 5. What Inter Miami's profile looks like (figures)

**Lead finding:** Among **353** tracked regains (`data/marts/team_post_regain.csv`), **319**—about **90.4%**—never convert into a modeled shot inside the fifteen-second horizon used for the Sankey accounting (derived from `data/marts/regain_events.parquet` splits matching `fig_post_regain_sankey` logic in `post_regain.py`). That proportion is less a verdict on Miami's attackers than a description of how often regain-to-shot links go missing inside the chosen counting window—a narrow, label-dependent fact that should be read alongside league-strength caveats later.

### Build-up failure rate with uncertainty

Across **73** qualifying buildup possessions (six matches), the raw failure rate is **0.342** with a bootstrap interval of **[0.253, 0.453]** (`data/marts/team_buildup_failure.csv`; bar treatment in **`docs/figures/case_study_03_buildup_failure_rate.png`**).

### Six Module B metrics with match-resampled CIs

Point estimates with 95% bootstrap bands—**xg per regain 0.0083** [0.0044, 0.0135], **median seconds to shot 5.5** [4.0, 9.0], **rushed shot rate 0.353** [0.136, 0.485], **final-third entry rate 0.669** [0.568, 0.772], **loss-within-four-actions rate 0.011** [0.0027, 0.024], **patience composite −10.88** [−11.32, −10.28]—are graphed horizontally in **`docs/figures/case_study_05_post_regain_metrics.png`** (pipeline twin: `docs/figures/post_regain_metrics_bar.png`).

### Regain Sankey

The Plotly Sankey **`docs/figures/case_study_06_sankey.png`** partitions regains into shot-within-window, final-third entries without an immediate shot, losses inside four touches, and residual paths—colors follow the violet/cyan palette declared in plotting code (`src/pressured_progression/features/post_regain.py`, `build_case_study_figures.py`). Counting logic lines up with logged splits from `data/marts/regain_events.parquet`: **34** regains register a modeled shot inside the capped horizon; **201** advance to the final third without that immediate shot; **4** satisfy the narrow "lost inside four actions" branch; **114** fall through to "other"—still within the operational accounting, still not a moral judgment about ambition.

Readers should treat the headline **90.4%** no-shot-within-window figure as a property of **this** counting rule (15-second shot linkage in the Sankey builder), not as proof that Miami never shoots after longer possessions. It simply states how rarely regains instantly cash into shot attempts under the plotting script's thresholds—the pattern is consistent with impatient subsequent chains even when longer-developing chances exist later in the possession.

Together, these visuals answer a narrow question: given six released matches, how concentrated are collapse-type endings and hurried regain outcomes under explicit definitions—not whether Miami "always" plays that way outside the sample.

---

## 6. Leverkusen as the analog—what the overlay actually contains

### Why Bayer Leverkusen 2023/24 was chosen directly

European analog candidates listed early in planning mostly failed Open Data availability checks (`docs/data_reality.md`): men's Brighton Premier League seasons never appear; Bologna and Girona lack requested campaigns; Bundesliga **2022/23 is absent entirely** from Open Data listing—only **2015/16** (no freeze-frame package) and **2023/24** (full Leverkusen season with freeze-frame metadata) qualify. Practical consequence: **there is no year-on-year Leverkusen forest plot inside this repository**. Module C reframed into a side-by-side overlay—same metrics on Leverkusen's 34-match **2023/24** sample and Miami's six-match slice—not because it is the philosophically perfect comparison, but because it is the **data-honest** comparison available without paid feeds.

The narrative clarity of Leverkusen's widely discussed pivot under Xabi Alonso belongs to journalism and broadcast archives ([Spielverlagerung](https://spielverlagerung.com/), [The Athletic](https://theathletic.com/), [Coaches' Voice](https://www.coachesvoice.com/))—associational context only. Nothing in the charts identifies coaching interventions as causal drivers.

### Two charts that carry the statistical contrast

**Bar overlay — `docs/figures/leverkusen_overlay_bar.png`:** Horizontal bars juxtapose Leverkusen 2023/24 (cyan error bars) against Inter Miami 2023 (violet) for Module A's raw failure rate and each Module B composite. This is intentionally busy—seven distinct scales share one canvas—so readers should lean on sign and overlap rather than mentally converting units across rows.

**Forest of differences — `docs/figures/leverkusen_diff_forest.png`:** Lines trace bootstrap percentile intervals for **Leverkusen minus Miami** on each metric; dots mark point differences; vertical zero reference sits at the center; cyan highlights intervals whose 95% range never crosses zero (`data/marts/leverkusen_overlay.csv`, column `ci_excludes_zero`). It is **not** a Leverkusen 2022/23 versus 2023/24 coaching panel—the Open Data catalog lacks Bundesliga 2022/23 entirely (`docs/data_reality.md`, Phase 5 appendix). Any journalistic story about year-on-year Bundesliga improvement belongs outside these axes.

How to read the forest without squinting at axis ticks: start at zero, walk metric-by-metric, ask whether the interval sits entirely on one side. When it does, the difference is large relative to bootstrap noise **within this pairing**; when it straddles zero, treat the contrast as ambiguous even if the point estimate looks juicy.

### Optional spatial companion

**Hexbin regain maps — `docs/figures/leverkusen_vs_miami_regain_heatmaps.png`:** Side-by-side density surfaces show where each team tends to recover the ball on the sampled pitches—associational texture for analysts who think spatially, not a substitute for the Module B timers above.

Reading the CSV directly—associational deltas, not causal attributions:

| Metric | Leverkusen mean | Miami mean | Difference (Leverkusen − Miami) | 95% diff interval |
| --- | ---: | ---: | ---: | --- |
| Raw buildup failure rate | 0.206 | 0.342 | −0.137 | [−0.262, −0.035] |
| xg per regain | 0.0178 | 0.00827 | +0.00958 | [+0.00322, +0.0159] |
| Time to shot (median seconds) | 5.0 | 5.5 | −0.50 | [−5.01, +1.5] |
| Rushed shot rate | 0.278 | 0.353 | −0.0748 | [−0.211, +0.123] |
| Final-third entry rate | 0.808 | 0.669 | +0.140 | [+0.020, +0.250] |
| Loss within four actions rate | 0.0131 | 0.0113 | +0.00181 | [−0.0116, +0.0128] |
| Patience composite | −9.79 | −10.88 | +1.09 | [+0.365, +1.80] |

Intervals **exclude zero** for raw failure rate, xg per regain, final-third entry, and patience composite—clear separations on those definitions within this observational pairing. Overlap remains for rushed shots, median time-to-shot, and regain-to-quick-loss incidence given Miami's uncertainty bands.

Where Miami sits relative to **both** Leverkusen seasons cannot be adjudicated here—Open Data simply omits Leverkusen's **2022/23** Bundesliga trace. Instead, Miami sits **below** Leverkusen 2023/24 on final-third entry rate and registers a **higher** raw buildup failure rate in this overlay—language kept deliberately comparative, not explanatory.

---

## 7. Short historical framing—money, roster skew, pressing fashion

MLS roster rules—Designated Players, Targeted Allocation Money, General Allocation Money—produce uneven squad depth even when stars shine ([MLS roster mechanism summaries via league documentation](https://www.mlssoccer.com/about/roster-rules-and-regulations)). Analysts outside the event pipeline nonetheless describe spending concentration and academy ROI challenges; CIES Football Observatory publications on MLS economics highlight ownership models and academy integration even when dollar figures shift year to year ([CIES "Spotlight on MLS"](https://www.cies.ch/fileadmin/documents/News_Agenda_Publications/Spotlight_on_MLS.pdf)). Separately, tactical writers argue MLS has leaned hard into coordinated pressing among elite clubs—a stylistic wave that stresses opponent buildup zones ([Backheeled on pressing prevalence](https://www.backheeled.com/mls-tactics-were-in-the-golden-age-of-pressing-in-mls/)).

European readers sometimes imagine MLS as monolithic; roster mechanics make asymmetry structural—stars under DP deals beside salary-budget scaffolding mean margin-for-error differs week to week even before tactics enter the chat. Nothing in this paragraph assigns Miami's overlay results to payroll arithmetic; it simply names why league commentators talk about uneven floors while pressing intensity rises league-wide.

These strands form **narrative context**, not coefficients in this project's models: they explain why uneven talent stacks under modern pressing profiles might coincide with brittle progression patterns without asserting that economics *cause* any Miami sequence observed in six matches.

---

## 8. Narrative cameos—cited journalism, not measurement

**Philadelphia Union 2026 (Philadelphia Inquirer):** Match coverage quotes head coach Bradley Carnell describing attacking breakdowns opponents forced near the penalty area—language about failing to **"organize...our feet"** amid attacking frustration appears in Inquirer reporting (for example [`https://www.inquirer.com/soccer/bradley-carnell-philadelphia-union-chicago-fire-gregg-berhalter-20260322`](https://www.inquirer.com/soccer/bradley-carnell-philadelphia-union-chicago-fire-gregg-berhalter-20260322)). That journalism is editorial context for the project's Philadelphia cameo—not an input into Miami's event labels.

**Columbus Crew 2023 MLS Cup:** Wilfried Nancy's Columbus side appears throughout MLS tactical writing as an internal **spirit-level** contrast—organized progression football—but **none** of Columbus's 2023 MLS minutes appear in StatsBomb Open Data (`docs/data_reality.md`). Referencing the championship arc stays prose-only.

**San Diego FC 2025–26 narrative (Guardian / broader press):** Guardian reporting on San Diego FC's expansion architecture—the club's academy-forward partnership story—provides organizational backdrop (`https://www.theguardian.com/football/2024/apr/24/san-diego-fc-soccer-team`). Separate tactical blogs detail expansion-side stylistic ambitions ([Backheeled San Diego tactical sketch](https://www.backheeled.com/mls-data-tactics-san-diego-fc-identity-vancouver-whitecaps-more/)). No ASA or StatsBomb figures in this repository validate those 2025 claims.

---

## 9. What the pattern suggests—and what it refuses to claim

First, **sample size**: six Miami matches shrink confidence for any Miami-specific conclusion; bootstrap bands already swallow several contrasts.

Second, **cross-league strength**: Bundesliga pace and MLS spacing differ; overlay differences may reflect competition environment as much as club habits.

Third, **observation, not intervention**: Metrics catalogue what co-occurred on released tapes—nothing identifies counterfactual Miami tactics or proves what Leverkusen would lose if dropped into MLS tomorrow.

Fourth, **360 availability mismatch**: StatsBomb catalogs advertise freeze-frame availability for Miami matches, yet GitHub-hosted freeze-frame JSON returned HTTP 404 during audit (`docs/data_reality.md`), leaving planned support-density features empty—SHAP rankings highlight **support_density_ff** at zero contribution while geometric pressure proxies carry signal (`project_info.md`).

Fifth, modeling humility: tuned XGBoost sits near baseline discrimination (`data/marts/cv_metrics.csv`); calibration error visible in **`docs/figures/calibration_curve.png`** counsels against deploying scores as scouting verdicts.

Sixth, **duplicate visualization paths**: Module B emits both legacy filenames (`docs/figures/post_regain_metrics_bar.png`, `docs/figures/post_regain_sankey.png`) and notebook-facing `case_study_*` variants—same underlying tables, alternate styling entry points for analysts versus article layout.

Nothing here says **Alonso caused** Leverkusen's statistical profile, nor that **Miami should copy** Leverkusen templates—those sentences violate the project's associational charter.

---

## 10. Closing — back to the sideline clip

Return to that regain deep in Miami's half: broadcast noise, urgent bodies, then a stalled progression. This project contributes something modest and, in the age of vibes, oddly useful—a named label set, reproducible pipelines, uncertainty on bars, and a Leverkusen overlay anchored in what Open Data actually stores. If the work has a payoff for the non-specialist reader, it is learning where the pretty story ends and the countable one begins—and how far apart they can sit when you only hold six league matches for one club.

One sentence worth the scroll: **the value is disciplined description under known data limits, not a playbook imported from Leverkusen.**

---

## Draft review (internal checklist)

**(a) Prescriptive language scrub:** Verified removal of imperative coaching advice; Leverkusen framed as comparative benchmark only.

**(b) Numeric anchoring:** Miami failure rate, bootstrap intervals, regain counts, Module B estimates, overlay differences, CV means, Sankey-derived no-shot percentage, and absence statements for Philly/Columbus tie to `data/marts/*.csv`, `docs/figures/*`, or explicit audit docs (`docs/data_reality.md`, `project_info.md`). Guardian / Inquirer / Backheeled / CIES linked where narrative claims appear.

**(c) Recruiter readability pass:** Specialist terms introduced once (SHAP as additive contribution; bootstrap as match-resampled uncertainty); causal verbs avoided throughout.
