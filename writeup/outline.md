# Writeup outline: v2.1 (Combine Game Equivalents)

Companion to `writeup/draft.md` (the text), `writeup/figure_plan.md` (figures and tables), `writeup/check_numbers.py` (the number checker) and `writeup/make_figures.py` (the figures).

- **v2.1 (2026-10-15)** applies two mock judges' reviews to v2: Football/Writeup (Football 6, Writeup 6.5) and Data Science/Data Viz (DS 7, Viz 6). Section 10 lists every judge item as applied, rejected or deferred.
- **v2** rebuilt the writeup around E012 (Combine Game Equivalents), as decided at Gate 2b (`experiments/log.md`).
- **v1** (the "Residual Combine" audit framing) is superseded.

## 1. Framing and names

**Question.** Week to week, how much should a club trust a rookie's own games over his pre-draft prior, and what is the Combine (now tracked at 10 Hz) worth once games arrive?

**One frame, one metric.** The frame is **the Combine Exchange Rate**: priors priced in the currency clubs think in, games. The metric, **Combine Game Equivalents (CGE)**, is named once (Section 2). After that the text says "worth N games".

| Term | Meaning |
|---|---|
| **Qualifying game** | A game with enough plays to measure the trait (WR top speed: ≥ 10 tracked plays). Defined in plain words in Section 2. |
| **λ** | Game-to-game noise / between-player spread: the qualifying games after which his own average is half signal. The position average is worth λ games. |
| **CGE** | The extra games of his own tracking that a Combine-based forecast is worth, out of sample. |
| **The trust rule** (the one equation) | w = k / (k + λ + CGE). The prior is worth λ + CGE games: its effective sample size (credibility weighting, as in baseball's "stabilization points"). Its standard error after k games is σ / √(k + λ + CGE). |
| **Trust table** (Table 1) | Per trait: prior worth (λ + CGE), when his games outweigh the prior (95% CI), about how many rookie appearances that is, and the weights after 1 / 4 / 8 games. |
| **Priced / not priced** | Priced: reliability and split-half ≥ 0.5 (9 traits). Not priced: T4, T8, T10 (rule PH7, post hoc). The figures use the same words. |
| **Receiver A** | The hook: E012's preregistered case-study pick (largest WR gap between prior and season). nfl_id 59278. Names withheld pending the user (section 7). |

**Information sets, as named in the text:**
- **Position average** = C0.
- **"plus stopwatch and measurements"** = C1.
- **Combine + draft forecast** = C2 = B2. Never "Combine forecast" alone, because it includes the draft slot (judge 1).
- **"+ Combine tracking"** = C3 (tracked-40 profile + the matched drill signature).
- The text never says C0–C3, B1 or B2.

## 2. Headline claims (each sourced in the draft's comments; all checked by `check_numbers.py`)

1. **Own games take over fast.**
   - For the 9 priced traits, a rookie's qualifying games outweigh the Combine + draft forecast after 2.0 (WR top speed) to 6.8 games; 95% CIs are in Table 1.
   - The position average alone is worth 1.5–6.8 games, and the Combine and draft add 0–1.2. That is the exchange rate in one line.
   - At 2025 rookies' rates this is 3–12 appearances.
2. **Outcome traits are slow.** Separation, pressure and pressure allowed over expected are half signal only after 11–23 qualifying games (about 14–39 appearances). A rookie pass rusher's pressure rate needs about 22.
3. **The Combine is worth a fraction of a game to about one.**
   - Point estimates: 1.2 / 0.5 / 0.2–0.3 / 0 for five traits.
   - The bootstrap intervals all include 0 but are biased low, covering about 80% in simulation.
   - Calibrated bounds (post hoc): WR top speed 0.08–1.6; DL rush distance 0.10–3.9; the rest below 1.7–3.6.
   - The estimate is conservative. Against the position average under the same CV, five traits reach 0.4–1.7 games (post hoc).
   - The DL value is role (roster position alone 2.4; the Combine adds −0.25 beyond it).
   - The WR value is fragile (0.14 in the 2023–24 fit; 0 without one class). It is partly usage (r 0.61 with deep share), yet the forecast adds 1.0 game beyond deep share.
4. **Combine tracking adds nothing measurable.**
   - ΔCGE ranges from −0.24 to +0.18.
   - H3: the estimate after one game is 2.1% worse (12 traits pooled).
   - For WR top speed alone the change is within noise: ratio 0.96 (0.80–1.09).
5. **Shrinkage does the work, and the rule is calibrated** (pseudo-prospective).
   - The rule cuts error 72% after one game (68% on priced traits) and 52% after two.
   - The position-average blend gets 99% of that gain.
   - The Combine + draft forecast adds 5% before game 1, 3.5% on priced traits and 2.5% after one game. All post hoc; none survives multiplicity.
   - The 80% intervals held 77–81%.
6. **Why the rate is low** (unchanged evidence).
   - 16/18 links null, beyond the stopwatch and beyond stopwatch + draft slot. Median draft-slot change 0.01 (intervals allow 0.37).
   - Nulls are bounds.
   - Drill reliability: 28/28, 2/22, 10/68, and 55–63% session variance in the OL wave drill.
7. **Side notes.**
   - The WR tracked top-speed signal "remains suggestive" (+0.48 / +0.42; fragile; q 0.06).
   - E010 gets one sentence. E011 moved to league item 3.

**Honesty rules from Gate 2b** (all applied):
- Shrinkage does almost all the in-season work.
- The Combine's ~5% is post hoc and not multiplicity-robust.
- Combine tracking adds nothing (H3 slightly worse).
- Every bootstrap CGE interval includes 0. v2.1 keeps this sentence but no longer leads with it, and pairs it with the calibrated bounds.
- T4, T8 and T10 are not priced.
- The 2025 test is pseudo-prospective.
- Nulls are bounds.

## 3. Structure and word budget

Counts come from `writeup/check_numbers.py`. They exclude HTML comments, image links and `[[TBD…]]`; captions and table cells count. Budget: **≤ 1,950 words, ≤ 8 figures + tables** (competition: ≤ 2,000 and < 10). **v2.1: 1,943 words, 6 items. Margin: 7 words.**

| Section | Purpose | Key content | Items | Words |
|---|---|---|---|---|
| Title | — | "The Combine Exchange Rate: how many NFL games is a Combine worth?" | — | 12 |
| 1. One rookie, fifteen games | Hook | 4.39; tracked finish 23.2 mph, rank 1/91; forecast 17.6 vs 17.5 (q90 explained); 18.3, 18.9 over 15; 2 SD, largest gap, preregistered pick; "no Combine failure" (worth 0.14 games; a 2-SD miss among 15 is a coin flip); official splits don't corroborate; tracking within noise (0.96); the rule caught up by game 2 | Figure 1 | 222 (caption 16) |
| 2. The Combine Exchange Rate | Question + method | Clubs' question (old Section 2, merged); qualifying game; λ; CGE; the one equation; worked example + standard error; prior art; four priors; prereg + A1; post hoc pricing rule | — | 258 |
| 3.1 Own games take over | Week-to-week deliverable | 2.0–6.8; λ 1.5–6.8 vs CGE 0–1.2; 3–12 appearances; outcome traits 11–23 games / 14–39 appearances / 22 for pass rushers | Table 1 | 324 (table 174, caption 19) |
| 3.2 Fraction of a game to about one | CGE | Values; biased-low bootstrap; calibrated bounds; conservative estimator (0.4–1.7 vs C0); DL role; WR usage and fragility; not priced, and what that rule set aside | Figure 2 | 254 (caption 20) |
| 3.3 Tracking adds nothing measurable | H3 | ΔCGE; 2.1% after one game, 12 traits pooled; scope sentence reconciling Section 5 | — | 62 |
| 3.4 Shrinkage works; the rule is calibrated | Validation | Pseudo-prospective; 72% (68% priced) / 52%; 77–81% coverage; 99% shrinkage; 5% / 3.5% / 2.5%, post hoc, none survives multiplicity | Figure 3 | 174 (caption 16) |
| 4. Why so little | Evidence | Transfer map (16/18; 0.01, at most 0.14, intervals allow 0.37; bounds; MDE); drill reliability (28/28, 2/22, 10/68, 55–63%) | Figures 4, 5 | 181 (captions 34) |
| 5. What tracking does show | Side notes | WR signal, suggestive and fragile; E010 | — | 104 |
| 6. What to do with it | Recommendations | Clubs: trust table with your best prior (C0 weights 0.40/0.73/0.84; DL roster role 2.4 → 6.0 games); shrink first (3.5×); don't weight Combine tracking (0–1.2, > 4 ruled out); be patient with outcome traits. League: identical repeat reps (40 0.66–0.79 vs pass-rush 0.23–0.51; 2–6 reps by Spearman–Brown); fixed tempo and frozen definitions; ball tracking with height (6–46 drills) | — | 241 |
| 7. Limitations + Code | No overclaiming | n 62–85; estimand (up to 15 in simulation); inference; game units (low-play games 2.0–3.5× noisier for 9 of 12; roles pooled); draft slot | — | 109 |
| **Total** | | | **6 of 8** | **1,943** |

**If words are needed** (for Figure 1's longer caption, +6, or anything else), cut in this order:
1. Section 5, the E010 sentence (24): Figure 5 already shows drill intensity.
2. Section 4, the parenthetical "(detectable effect 0.31–0.38 at 60–80 players)" (8): Figure 4 draws the MDE.
3. Section 3.2, the WR usage clause (17): keep it if at all possible, because it answers a club's first question.
4. Section 1, "(the 90th percentile of his per-play peaks, well below a sprint)" (11): Figure 1's y-axis now says it.

## 4. Wording guardrails

**Say:**
- "Combine + draft forecast"
- "worth N games"
- "qualifying games"
- "priced / not priced"
- "Combine tracking adds nothing measurable" / "slightly worse"
- "the bootstrap intervals all include 0 but are biased low"
- "simulation-calibrated bounds (post hoc)"
- "point estimates 0 to 1.2, more than about 4 ruled out"
- "pseudo-prospective"
- "post hoc" on every PH number
- "error-free measures"
- "remains suggestive" and "fragile" (the WR signal)
- "body type and role" (the DL CGE)
- "partly route usage" (WR top speed)

**Never say:**
- "Combine forecast" without "+ draft"
- "the Combine said average" or "the Combine failed" about Receiver A
- that his forecast was "nearly flat" or "called every receiver average" (not true: `post_hoc_review.validation_2.per_trait.T1.c2_prior_spread_tau` 0.55, `c2_prior_flat` false)
- "the Combine-based rule cut error 72%" without crediting shrinkage
- "frozen before the 2025 season" or "prospective"
- "the Combine is useless" / "worth zero" / "at most about one"
- "tracking adds nothing" without "Combine"
- "survives", "validated", "confirmed" or "replicated" for the WR signal
- "undervalued", "steal", "market inefficiency"
- CGE values for T4, T8, T10
- "every interval includes 0" as a lead or without the calibrated bounds
- "two reps" (the evidence supports identical reps, 2–6 by Spearman–Brown)
- "36/36" for validation 1
- B1/B2/C0–C3 in the text
- Receiver A's draft status

## 5. Mock-judging self-check (AGENTS.md checklist), v2.1

- **Baselines beaten?**
  - The rule beats raw early games (H1/H2).
  - The honest baseline is the position-average rule: the Combine + draft forecast beats it by 5% before game 1 (post hoc).
  - Combine tracking does not beat the Combine + draft forecast (H3; and for WR top speed alone, 0.96, 0.80–1.09).
  - The transfer map tests each link against both baselines.
- **Uncertainty reported?**
  - Half-weight CIs in Table 1.
  - Calibrated two-sided bounds next to the biased-low bootstrap.
  - The rule's 80% intervals hold 77–81%.
  - Leave-one-class-out and the 2023–24 fit for the WR CGE.
  - Bounds and MDE for the nulls.
  - p and q values on the figures.
- **Overclaiming?**
  - The hook now says "no Combine failure" and gives the chance of a 2-SD miss.
  - "At most about one" is gone.
  - The H3 wording is exact.
  - The pricing rule is labelled post hoc, and the text says what it set aside.
- **Usable week to week?**
  - Table 1 in club units (appearances, CIs, weights, outcome traits).
  - The worked example and standard error sit under the equation.
  - Prior-specific weights (Clubs item 1).
- **≤ 2,000 words and < 10 figures?** 1,943 words; 6 items.
- **Combine tracking used explicitly?** Yes: C3 is priced in games; the transfer map and drill reliability are all Combine tracking; Figure 1 animates the tracked 40 (and, per figure_plan, one of his game plays).

## 6. Provenance

- **Every number in the text** traces to a committed results file; `check_numbers.py` verifies it.
  - It recomputes each number from its JSON key at the quoted rounding.
  - It asserts the quoted phrase is in the text.
  - It fails on any numeric token that no check covers.
  - It requires a `<!-- src` comment near every number-bearing paragraph. On a comment-stripped copy it skips this check and still checks every number.
  - Last run (v2.1): **354 checks, 0 failures**, 1,943 words.
  - Mutation test: five altered numbers gave 30 failures; all five were caught.
- **Arithmetic on committed keys** (formula in each src comment; recomputed by the checker). The E012 owner should promote these into `E012.json` so the notebook prints them from keys:
  1. Rookie appearances = half-weight (or λ) / `qualifying_share_2025_rookies[T].share`.
  2. Priced-traits H1 cut, 68%: H1's own equal-weight pooling over the 9 priced traits of `secondary_per_trait[T]["1"]`. The same pooling over all 12 traits reproduces H1's pooled values exactly.
  3. Standard error after 4 games = √(σ² / (4 + λ + CGE)) = 0.33 mph, from `traits.T1.variance.sigma2` and `half_weight_game.C2`.
  4. Half-weight with the DL role prior = λ + `cge_role_reference` = 6.0.
  5. Reps needed for reliability 0.5 = (1 − icc) / icc from `drill_profile[*].icc_unit` (DL pass-rush): 2–6.
  6. Chance that a calibrated forecast misses at least one of 15 by 2 SD = 1 − 0.9545^15 ≈ 0.50.
  7. (Kept from v2) raw first game error / rule error = H1 `pooled_b / pooled_a` = 3.5.
- **Sources:** `E012.json`, `E012_prereg.json`, `E008.json`, `E009.json`, `E010.json`, `E011.json`, `E007_leaderboard.json`, `experiments/posthoc_writeup_v1/ph_pairs.json` (rank cross-check).
- **Allowlisted non-result constants** (with reasons, in the checker): data scope (2023–24, 2023–25, 2025, 10 Hz); drill names ("tracked 40", "tracked-40 speed profile", "his 40 peak"); "game 1"; the table's k values (1 / 4 / 8); CI labels ("(95% CI)", "(95% CIs)").
- **Prior art** (in the Section 2 src comment, for the notebook's references): Bühlmann credibility; sabermetric stabilization points; Morita, Thall & Müller (2008), "Determining the effective sample size of a parametric prior", *Biometrics*.
- **Not quoted, figure-only:** see `figure_plan.md`, "Not quoted in the text".

## 7. Open decisions for the user

1. **Framework name.** "The Combine Exchange Rate" (CGE named once). Keep it?
2. **Names.**
   - Receiver A/B are real 2025 rookies, named in `E012.json`. Receiver A went undrafted.
   - Judge 1 argues anonymity buys little (a staffer can likely identify "Receiver A, 4.39, 2025 Combine") and costs hook power.
   - Recommendation: real names are acceptable if no draft information appears anywhere. This is also the precondition for R2.
3. **R2: 2025 rookie trust board** (judge 1 item 4). It needs the names decision and about 80–120 words of cuts. The spec is in figure_plan.md.
4. **Figure 1 game-play panel.** Adding it lengthens the caption by 6 words (7 available).
5. **Receiver A's C1/C3 forecasts** (judge 2 item 3). Promote `prior_C1` and `prior_C3` for the case studies into `E012.json`. With them the hook could say how much of the forecast's "near average" came from the draft slot (stopwatch alone 17.76 mph vs + draft slot 17.61 vs + tracking 17.90, per judge 2 reading the gitignored parquet; unverified by this writer and not quotable until committed).

## 8. Pre-submission checklist

- Replace `[[TBD-NOTEBOOK-LINK]]`. Strip every `<!-- … -->` before pasting into Kaggle, then run `check_numbers.py <stripped copy>`: it must report 0 failures.
- Re-run `.venv/bin/python writeup/check_numbers.py` after every edit (exit 0).
- Re-run `writeup/make_figures.py` after any draft number changes. Its `same(...)` asserts hard-code the draft's strings, so update them together.
- The notebook must:
  - reproduce `E012.json` (about 4.5 h full run), `E008.json` and `E009.json`;
  - print the derived numbers in section 6 from keys;
  - include trait definitions and position averages (the Code line promises both).
- Re-run the mock judging on v2.1 with the final figures.
- Shared-module items from E012's review (log, GATE 2b):
  - RidgeCV's LOO alpha search on duplicated bootstrap rows. v2.1 quotes no E007–E011 bootstrap interval; its E008–E010 intervals are Fisher CIs.
  - `nfl_position` meaning. v2.1 quotes the role prior, with "roles may postdate the draft".

## 9. v2 change log (from v1, condensed)

- **Kept:** transfer map with bounds; drill reliability; the WR signal with fragility; E010/E011 one-liners; Combine-operations recommendations (now revised); guardrails.
- **Dropped:** the "Residual Combine" name and scorecard; the same-40 pair as the hook; finishing-speed detail; the 172-pairs statistic; v1's in-season WR forecast; the "standardize on the tracked 40" advice.
- **New:** CGE, the trust rule and table; the pseudo-prospective test with the shrinkage-vs-Combine split; E009's beyond-the-stopwatch column; `check_numbers.py`.

## 10. v2.1: judge items, applied / rejected / deferred

Judge 1 = Football/Writeup; Judge 2 = Data Science/Data Viz. "Figure builder" = `make_figures.py`. Several figure items were already done in the 19:39–19:40 renders; figure_plan.md lists each as done or to do.

### Judge 1, top improvements
1. **Role-aware priors; usage confounding.** **Partly applied.**
   - Clubs item 1 gives the DL roster-role prior: worth 2.4 games, outweighed after 6.0 (post hoc; roles may postdate the draft).
   - Section 3.2 states WR top speed's usage confound: r 0.61 with deep share; the forecast adds 1.0 game beyond it.
   - Limitations: position averages pool roles.
   - **Deferred:** re-fitting λ and Table 1 with role-specific averages. That needs an E012 re-run after verifying `nfl_position` (a Gate 2b open item), and the writer may not edit experiments. A slot/outside WR split has no committed key.
2. **Club decision units.** **Applied:**
   - (a) appearances column plus the 3–12 / 14–39 conversions (derived; owner should promote);
   - (b) λ beside CGE in Table 1 and in Section 3.1's exchange-rate sentence;
   - (d) the standard error σ/√(k + λ + CGE) with a worked value, plus 77–81% calibration.
   - **Deferred (c):** position averages in units. Only WR top speed (17.5 mph) has a meaningful raw-unit average in committed results; several traits are centred or "over expected". The Code line commits the notebook to definitions and position averages.
3. **Outcome-trait stabilization into Table 1 and the headline.** **Applied:** three Table 1 rows (λ with bootstrap CI, appearances, position-average weights); Section 3.1 paragraph; Clubs item 4. Seasons are given as appearances (14–39) rather than as fractions of a season.
4. **2025 rookie trust board (R2).** **Deferred:** it needs the names decision (section 7) and about 100 words of cuts. The spec is in figure_plan R2.
5. **Honest hook, positive punchline.** **Applied:**
   - "no Combine failure: in that fit the forecast was worth only 0.14 games";
   - chance of a 2-SD miss among 15 test receivers ≈ 50%;
   - official splits don't corroborate tracked finishing speed;
   - the q90 definition explains the gap to 23.2 mph;
   - ends on "What worked was the in-season rule … Table 1 gives that moment".
   - **Rejected in part:** "the forecast was nearly flat and called every receiver average". The committed data contradict it (`c2_prior_spread_tau` 0.55, `c2_prior_flat` false).
6. **Section 7 inconsistencies.** **Applied:**
   - weights by prior (C0 weights 0.40/0.73/0.84);
   - the league item now asks for identical reps, 2–6 by Spearman–Brown, with the like-for-like 40 vs pass-rush comparison;
   - "Don't weight Combine tracking".
7. **Prior art.** **Applied** (Section 2: credibility weighting, stabilization points, prior effective sample size; citations in the src comment).
8. **Readability.** **Applied:**
   - one frame name, CGE named once;
   - "Combine + draft forecast" throughout;
   - the worked example under the equation;
   - p, q and CI strings cut from Section 3.4 (one CI kept per headline effect; p/q on the figures);
   - "qualifying game" defined with an example;
   - the week-to-week question merged into Section 2.
9. **Drill-vs-game animation.** **Deferred to the figure builder:** figure_plan Figure 1 panel 2 (one typical game play on the field, selection rule fixed).

### Judge 1, overclaims or errors (11)
- Clubs item 3, "at most about one": **applied** ("0 to 1.2 (point estimates), with more than about 4 ruled out").
- Clubs item 1 vs Table 1 priors: **applied**.
- League item 1, "two reps": **applied**.
- H3 described as "forecasts": **applied** ("the estimate after one game … 12 traits pooled").
- Tracked-finish hint uncaveated: **applied**.
- "The Combine said average": **applied** in the text; the figure title change is in figure_plan (to do). The "flat" premise is rejected (above).
- "One signal survives": **applied** ("remains suggestive").
- DL prior should be role-specific: **applied** in Clubs item 1; the Table re-fit is **deferred**.
- T9/T11 weights under a failed assumption: **applied** (no weights; † footnote).
- "Largest miss" by absolute gap: **applied** ("largest gap"; the src comment says absolute).
- Prior-art omission: **applied**.

### Judge 1, figure issues (8)
- Figure 1 dots arrive together: **done** (speed-vs-distance curves).
- Figure 1 y-label: **done**.
- Figure 1 hollow diamond: **done** (removed until `prior_C3` is committed).
- Figure 2 λ bars, ticks, hollow glyph, stray glyph: **done**.
- Figure 3 jargon, five series, p/q density: **done**.
- Figure 4 outline boxes: **done** (removed). The six "≥ 0.5 if error-free" links get a small glyph (to do).
- Figure 5 OL one-rep row and the DL pass-rush vs "two reps" conflict: **done** (row to notebook; recommendation changed).
- Table 1 zeros, not-identified λ, identical-looking weights: **applied** ("0: no out-of-sample gain (clipped)"; λ + CGE column makes 4.8 = 4.8 + 0 = 3.6 + 1.2 visible).

### Judge 2, top improvements
1. **Calibrated two-sided bounds.** **Applied** in the text (Section 3.2) and **done** in Figure 2.
2. **Conservative estimator; retitle.** **Applied:**
   - Section 3.2 title "a fraction of a game to about one";
   - the "estimate is conservative" sentence with the same-CV values 0.4–1.7;
   - Clubs item 3 as a bound.
   - The simulation's p_zero (64.5% at population R² 0.2) is not quoted (words), and the CV estimand itself is 0 there.
3. **Hook attribution and statistics.** **Applied:** the T1-specific tracking test (0.96, 0.80–1.09) replaces the pooled H3 in the hook; the chance of a 2-SD miss is stated. The text uses 15, the committed count; judge 2's 11 (receivers with ≥ 10 games) has no committed key.
   - **Deferred:** the C1/C3 decomposition (section 7, item 5).
   - Leading with the WR case among 16 picks is an editorial choice. It is the preregistered WR pick, and WR top speed is the trait the forecast prices best. Not changed.
4. **Uncertainty in Table 1.** **Applied** (half-weight 95% CI column).
   - Weights stay at 2 dp: they are a plug-in rule, and the CI column carries the uncertainty.
   - **Deferred:** R1 trust curves (largely duplicate Table 1; no words left).
5. **Pricing rule is post hoc; what it removed.** **Applied:** "(post hoc)" in Section 2; Section 3.2 gives DL pressure's raw CV R² (0.21 vs WR top speed 0.22), its +0.40 exposure correlation, and "where a good prior matters most".
   - **Rejected for the text:** T8's calibrated lower bound (0.28) and leave-one-class-out range (0.58–2.72). The trait's true-score correction is unstable (reliability 0.43); these stay in the notebook.
6. **The game unit.** **Applied:** Limitations now quantify the noise ratio, correcting the judge's count: **9** of 12 traits (not 10) are ≥ 2.0×; T10 is 1.60. T9/T11 lose their weights.
   - **Deferred:** per-play weighting and season-centring (need an E012 re-run; PH11's season-centred CGE is 0 for both traits anyway). "CGE/λ, the threshold-free share" is shown implicitly by Table 1's λ + CGE column; there is no separate number.
7. **Reconcile Section 5 with 3.3.** **Applied** (one scope sentence; heading "nothing measurable").
8. **Report strengths.** **Applied:** calibration 77–81%; priced-traits H1 cut 68%, computed with H1's own pooling, which the checker verifies reproduces H1 exactly.
9. **Encoding consistency.** **Done** by the figure builder (hollow = post hoc only; Figure 4 navy only). Codified in figure_plan's house style.
10. **Operations advice and prior work.** **Applied:** identical reps; 2–6 by Spearman–Brown from single-rep ICC 0.15–0.39; the 40 (0.66–0.79) vs pass-rush (0.23–0.51) comparison, "mostly not" because 1 of 6 reaches 0.51; prior-art sentence and citation.

### Judge 2, overclaims or errors (15)
- "Combine said average" is mostly the draft slot: the text says "Combine + draft forecast". The decomposition is **deferred**; the figure title is **to do**.
- Largest of 11; a 2-SD miss is common: **applied** (with the committed 15).
- Pooled H3 answering a WR question: **applied**.
- Leading with "every interval includes 0": **applied**.
- "At most about one" is not a bound: **applied**.
- "Worth zero to about one" header: **applied**.
- Pricing rule placed after "preregistered": **applied**.
- "Adds nothing" vs +0.48: **applied**.
- Half-weight uncertainty: **applied**.
- "Not a draft-slot artifact": **applied** (clause removed; "intervals allow 0.37").
- "Two standardized reps": **applied**.
- T9/T11 weights: **applied**.
- 72% includes the unpriced separation trait: **applied** (68% on priced traits).
- Tracking-artifact alternative: **applied**.
- "About 11 catching drills": **applied** (6–46).

### Judge 2, figure issues (16)
- Done by the figure builder (19:39–19:40 renders):
  - Figure 1 still, start-line overprint, x-label, removed diamond, the 2023–24 vs 2023–25 note;
  - Figure 2 calibrated bounds, λ and axis title, and the stray glyph;
  - Figure 3 three lines, single linear axis and the pooled-traits note;
  - Figure 4 outlines and legend;
  - Figure 5 dot independence (split by source) and the like-for-like slope panel;
  - cross-figure encoding.
- To do (figure_plan):
  - Figure 1 alt text, GIF timing and the game-play panel;
  - Figure 2 raw CV R² in the not-priced rows;
  - Figure 3 calibration panel and priced-traits H1 row;
  - Figure 4 legibility check at 750 px;
  - Figure 5 reliability SEs.
- Rejected: Figure 3's "irreducible target noise" reference. No committed quantity defines it.
- Deferred: R1 trust curves.

### Errors found while revising (all fixed before the final check)
- The judges' noise-ratio count (10 of 12) is 9 of 12 in `E012.json`.
- The judges' "nearly flat forecast" premise is contradicted by `c2_prior_spread_tau` 0.55.
- My first Table 1 draft rounded OL pass-set acceleration's lower CI 3.7499 to 3.8; the checker caught it (now 3.7).
- The DL pass-rush drill has 1 of 6 first-step metrics at 0.514, so the text says "mostly not" rather than "not".
