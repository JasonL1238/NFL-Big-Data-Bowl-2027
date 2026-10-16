# Writeup outline: v2.2 (Combine Game Equivalents)

Companion to `writeup/draft.md` (the text), `writeup/figure_plan.md` (figures and tables), `writeup/check_numbers.py` (the number checker) and `writeup/make_figures.py` (the figures).

- **v2.2 (2026-10-15)** applies the final checker's five findings: word budget, hook accuracy, selection pools, drill-reliability framing, captions. Section 11 lists each.
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
| **Receiver A** | The hook: E012's preregistered case-study pick (largest WR gap between prior and season among the 11 of 15 test receivers with 10+ qualifying games). nfl_id 59278. Names withheld (orchestrator decision; section 7). |

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
   - For Receiver A it nudged the forecast 0.3 mph toward his season (post hoc export, `e012_case_priors.json`); across all 15 test receivers it left error after one game within noise (0.96, 0.80–1.09).
   - ΔCGE ranges from −0.24 to +0.18.
   - H3: the estimate after one game is 2.1% worse (12 traits pooled).
   - For WR top speed alone the change is within noise: ratio 0.96 (0.80–1.09).
5. **Shrinkage does the work, and the rule is calibrated** (pseudo-prospective).
   - The rule cuts error 72% after one game (68% on priced traits) and 52% after two.
   - The position-average blend gets 99% of that gain.
   - The Combine + draft forecast adds 5% before game 1, 3.5% on priced traits and 2.5% after one game. All post hoc; none survives multiplicity.
   - The 80% intervals held 77–81%.
6. **Why the rate is low** (Section 4).
   - 16/18 links null, beyond the stopwatch and beyond stopwatch + draft slot. Median draft-slot change 0.01 (intervals allow 0.37).
   - Unreliable measures do not explain the nulls: 13 of 16 pass the reliability gate (E008 `nulls_passing_c5`). Power and imperfect game targets limit the tests. Nulls are bounds.
   - **Separate finding (Section 5), not a reason for the nulls:** which Combine measurements are stable traits: 28/28, 2/22, 10/68, and 55–63% session variance in the OL wave drill; drill intensity is stable (0.74) yet predicts nothing.
7. **Side notes.**
   - The WR tracked top-speed signal "remains suggestive" (+0.48; fragile; q 0.06). The +0.42 beyond the stopwatch alone is now figure-only (Figure 4).
   - E010 is one sentence in the stable-traits paragraph. E011 is league item 3.

**Honesty rules from Gate 2b** (all applied):
- Shrinkage does almost all the in-season work.
- The Combine's ~5% is post hoc and not multiplicity-robust.
- Combine tracking adds nothing (H3 slightly worse).
- Every bootstrap CGE interval includes 0. v2.1 keeps this sentence but no longer leads with it, and pairs it with the calibrated bounds.
- T4, T8 and T10 are not priced.
- The 2025 test is pseudo-prospective.
- Nulls are bounds.

## 3. Structure and word budget

Counts come from `writeup/check_numbers.py`, after removing HTML comments and image links; captions and table cells count, and so does the `[[TBD…]]` placeholder (it becomes link text). Hard budgets (v2.2): **rendered words ≤ 1,900** (tokens with a letter or digit in the rendered text, table pipes as cell breaks) and **whitespace tokens ≤ 1,950**, counted both on the pasted markdown and on the rendered text; **≤ 8 figures + tables** (competition: ≤ 2,000 words and < 10 items). **v2.2: 1,888 rendered words; 1,881 / 1,914 whitespace tokens (markdown / rendered); 6 items.** v2.1 had 1,953 words by the old count and 2,137 whitespace tokens, over 2,000 for a counter that counts table pipes and symbols.

| Section | Purpose | Key content | Items | Rendered words |
|---|---|---|---|---|
| Title | — | "The Combine Exchange Rate: how many NFL games is a Combine worth?" | — | 12 |
| 1. One rookie, fifteen games | Hook | 4.39; tracked finish 23.2 mph, rank 1/91; forecast 17.6 vs 17.5 (q90 play peak); 18.3, 18.9 over 15; 2 SD, largest gap among the 11 of 15 test receivers with 10+ games, preregistered pick; "no Combine failure" (worth 0.14 games; a 2-SD miss among 11 ≈ 40%); tracking nudged him 0.3 mph toward his season, but across all 15 left error within noise (0.96); official splits don't corroborate; the rule caught up by game 2 | Figure 1 | 240 (caption 20) |
| 2. The Combine Exchange Rate | Question + method | Clubs' question; qualifying game; λ; CGE; the one equation; worked example + standard error; prior art; four priors; prereg + A1; post hoc pricing rule | — | 247 |
| 3.1 Own games take over | Week-to-week deliverable | 2.0–6.8; λ 1.5–6.8 vs CGE 0–1.2; 3–12 appearances; outcome traits 11–23 games / 14–39 appearances / 22 for pass rushers | Table 1 | 286 (cells 143, footnote 27, caption 19) |
| 3.2 Fraction of a game to about one | CGE | 0 (five traits) to 1.2; biased-low bootstrap; calibrated bounds; conservative estimator (0.4–1.7 vs C0); the two largest are soft (DL role; WR usage and fragility); not priced, and what that rule set aside | Figure 2 | 223 (caption 20) |
| 3.3 Tracking adds nothing measurable | H3 | ΔCGE; 2.1% after one game, 12 traits pooled; scope sentence reconciling Section 5 | — | 62 |
| 3.4 Shrinkage works; the rule is calibrated | Validation | Pseudo-prospective; 72% (95% CI 60–80%; 68% priced) / 52%; 77–81% coverage; 99% shrinkage; 5% / 3.5% / 2.5%, post hoc, none survives multiplicity | Figure 3 | 171 (caption 20) |
| 4. Why so little | Evidence | Transfer map (16/18; 0.01, at most 0.14, intervals allow 0.37); 13 of 16 nulls pass the reliability gate; power and imperfect game targets (bounds +0.09 to +0.38; MDE 0.31–0.38; six links ≥ 0.5 if error-free) | Figure 4 | 138 (caption 17) |
| 5. What tracking does show | Separate findings | Stability is a separate question from transfer (28/28, 2/22, 10/68, 55–63%; drill intensity stable 0.74 yet predicts nothing); WR signal, suggestive and fragile | Figure 5 | 169 (caption 19) |
| 6. What to do with it | Recommendations | Clubs: trust table with your best prior (C0 weights 0.40/0.73/0.84; DL roster role 2.4 → 6.0 games); shrink first (3.5×); don't weight Combine tracking (point estimates 0–1.2, > 4 ruled out); be patient with outcome traits. League: identical repeat reps (40 0.66–0.79 vs pass-rush 0.23–0.51; 2–6 reps by Spearman–Brown); fixed tempo and frozen definitions; ball tracking with height (6–46 drills) | — | 231 |
| 7. Limitations + Code | No overclaiming | n 62–85; estimand (up to 15 in simulation); inference; game units (low-play games 2.0–3.5× noisier for 9 of 12; roles pooled); draft slot | — | 107 |
| **Total** | | | **6 of 8** | **1,888** |

**If words are needed**, cut in this order (none of these is a caveat):
1. Section 5, "the kind Receiver A showed" (5).
2. Figure 1 caption: shorten "then his forecasts, rookie games and in-season estimate" to "then his season" (5).
3. Section 6, Clubs item 4 (14): Table 1's ‡ rows and Section 3.1 already say it.

Never cut: "fit on 2023–24", "pseudo-prospective", "post hoc" labels, "none survives multiplicity control", "That describes our model at this sample size", the calibrated bounds, the official-splits clause, or Limitations.

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
- "nudged his forecast 0.3 mph toward his season average" (Receiver A, Combine tracking), always next to the 15-receiver test
- "13 of the 16 pass the reliability gate" (the nulls); "stability is a separate question from transfer"
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
- Receiver A's draft status, or his stopwatch-only forecast (`prior_C1`, 17.8 mph): next to the 17.6 Combine + draft forecast it would show the draft slot lowered it. (Open: the figure builder's 20:25 Figure 1 render draws it as a post hoc marker; the orchestrator should either drop that marker or relax this rule. figure_plan, Figure 1.)
- "Nor would tracking have helped" or any claim that tracking did nothing for Receiver A (it moved him 0.3 mph the right way)
- a chance argument over a different pool from the selection rule (the rule chose among 11; the chance is over 11)
- that unreliable drills explain the transfer-map nulls (13 of 16 nulls pass the reliability gate)

## 5. Mock-judging self-check (AGENTS.md checklist), v2.2

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
  - The hook now says "no Combine failure" and gives the chance of a 2-SD miss, over the same 11 receivers the rule chose among.
  - The hook states what Combine tracking did for Receiver A (0.3 mph toward his season) and that it did not help across all 15.
  - The drill-reliability paragraph no longer reads as the reason for the nulls.
  - "At most about one" is gone.
  - The H3 wording is exact.
  - The pricing rule is labelled post hoc, and the text says what it set aside.
- **Usable week to week?**
  - Table 1 in club units (appearances, CIs, weights, outcome traits).
  - The worked example and standard error sit under the equation.
  - Prior-specific weights (Clubs item 1).
- **≤ 2,000 words and < 10 figures?** 1,888 rendered words; at most 1,914 whitespace tokens by any of the checker's counts; 6 items.
- **Combine tracking used explicitly?** Yes: C3 is priced in games; the transfer map and drill reliability are all Combine tracking; Figure 1 animates the tracked 40 (and, per figure_plan, one of his game plays).

## 6. Provenance

- **Every number in the text** traces to a committed results file; `check_numbers.py` verifies it.
  - It recomputes each number from its JSON key at the quoted rounding.
  - It asserts the quoted phrase is in the text.
  - It fails on any numeric token that no check covers.
  - It requires a `<!-- src` comment near every number-bearing paragraph. On a comment-stripped copy it skips this check and still checks every number.
  - Last run (v2.2): **374 checks, 0 failures** on the draft and on a comment-stripped copy; 1,888 rendered words, 1,881 / 1,914 whitespace tokens.
  - Mutation test (v2.2): seven altered numbers (the 0.3 mph nudge, 40%, 13 of 16, 11 of 15, all 15, a Table 1 cell, the H1 CI) each failed the run (2–11 failures each); padding with 80 symbol tokens tripped the whitespace-token check and padding with 20 words tripped the word check.
- **Arithmetic on committed keys** (formula in each src comment; recomputed by the checker). The E012 owner should promote these into `E012.json` so the notebook prints them from keys:
  1. Rookie appearances = half-weight (or λ) / `qualifying_share_2025_rookies[T].share`.
  2. Priced-traits H1 cut, 68%: H1's own equal-weight pooling over the 9 priced traits of `secondary_per_trait[T]["1"]`. The same pooling over all 12 traits reproduces H1's pooled values exactly.
  3. Standard error after 4 games = √(σ² / (4 + λ + CGE)) = 0.33 mph, from `traits.T1.variance.sigma2` and `half_weight_game.C2`.
  4. Half-weight with the DL role prior = λ + `cge_role_reference` = 6.0.
  5. Reps needed for reliability 0.5 = (1 − icc) / icc from `drill_profile[*].icc_unit` (DL pass-rush): 2–6.
  6. Chance that a calibrated forecast misses at least one of 11 by 2 SD = 1 − 0.9545^11 = 0.40. The 11 are the receivers the selection rule chose among (`e012_case_priors.json pools.T1.n_with_10plus_games`); v2.1 used 15 (≈ 0.50), a different pool from the rule's.
  8. Receiver A's tracking nudge = (`prior_C3` − `prior_C2`) × 3600/1760 = 0.29 mph (`e012_case_priors.json cases[59278].tracking_shift_C3_minus_C2` 0.142 yd/s).
  9. H1's CI in percent = 1 − `ratio_ci95` = 60–80%.
  7. (Kept from v2) raw first game error / rule error = H1 `pooled_b / pooled_a` = 3.5.
- **Sources:** `E012.json`, `E012_prereg.json`, `E008.json`, `E009.json`, `E010.json`, `E011.json`, `E007_leaderboard.json`, `experiments/posthoc_writeup_v1/ph_pairs.json` (rank cross-check), `experiments/posthoc_writeup_v2/e012_case_priors.json` (post hoc export of the stored E012 forecasts: case-study priors C0–C3 and the test pools; the checker cross-checks its C0, C2 and season mean against `E012.json`).
- **Allowlisted non-result constants** (with reasons, in the checker): data scope (2023–24, 2023–25, 2025, 10 Hz); drill names ("tracked 40", "tracked-40 speed profile", "his 40 peak", "40 vs DL pass-rush drill"); "game 1" and Figure 1's "game-1" panel; the table's k values (1/4/8); CI labels ("(95% CI)", "(95% CIs)").
- **Prior art** (in the Section 2 src comment, for the notebook's references): Bühlmann credibility; sabermetric stabilization points; Morita, Thall & Müller (2008), "Determining the effective sample size of a parametric prior", *Biometrics*.
- **Not quoted, figure-only:** see `figure_plan.md`, "Not quoted in the text".

## 7. Open decisions for the user

1. **Framework name.** "The Combine Exchange Rate" (CGE named once). Keep it?
2. **Names.**
   - Receiver A/B are real 2025 rookies, named in `E012.json`. Receiver A went undrafted.
   - Judge 1 argues anonymity buys little (a staffer can likely identify "Receiver A, 4.39, 2025 Combine") and costs hook power.
   - Recommendation: real names are acceptable if no draft information appears anywhere. This is also the precondition for R2.
3. **R2: 2025 rookie trust board** (judge 1 item 4). It needs the names decision and about 80–120 words of cuts. The spec is in figure_plan.md.
4. **Figure 1 game-play panel.** Done by the figure builder; the v2.2 caption names it.
5. **Receiver A's C1/C3 forecasts** (judge 2 item 3). **Resolved in v2.2** via the committed post hoc export `experiments/posthoc_writeup_v2/e012_case_priors.json` (17.76 / 17.61 / 17.90 mph confirmed). The text quotes only the tracking nudge (+0.29 mph, C3 − C2). C1 is checked (near average, far below his season) but not quoted, because next to C2 it shows the draft slot lowered his forecast, which hints at his draft status (guardrail). The E012 owner may still promote the keys into `E012.json`.
6. **Orchestrator decisions for v2.2** (the user may change them): names stay anonymized; the name "The Combine Exchange Rate" stays; no R2 trust board and no role-specific refit in this pass.

## 8. Pre-submission checklist

- Replace `[[TBD-NOTEBOOK-LINK]]`. Strip every `<!-- … -->` before pasting into Kaggle, then run `check_numbers.py <stripped copy>`: it must report 0 failures, including both budgets (rendered words ≤ 1,900; whitespace tokens ≤ 1,950 on the markdown and the rendered text).
- Check in the Kaggle writeup preview that Table 1's compact pipes (`|a|b|`, no spaces; valid GitHub-flavoured markdown) render as a table. Spaced pipes (`| a | b |`) would raise the markdown whitespace-token count by about 140, to about 2,020 (measured on v2.2), over the 1,950 budget, so do not re-space them without first cutting about 80 words.
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
3. **Hook attribution and statistics.** **Applied:** the T1-specific tracking test (0.96, 0.80–1.09) replaces the pooled H3 in the hook; the chance of a 2-SD miss is stated. The text uses 15, the committed count; judge 2's 11 (receivers with ≥ 10 games) has no committed key. (Superseded in v2.2: both counts are now committed and used; section 11.)
   - **Deferred:** the C1/C3 decomposition (section 7, item 5; resolved in v2.2).
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

## 11. v2.2: the final checker's findings, applied

1. **Word budget.** **Applied.**
   - v2.1 as committed counted 1,953 words by its own method (over its 1,950 limit) and 2,137 whitespace tokens: table pipes, "+", "/", "≥" and "×" are tokens to most word counters.
   - New hard checks in `check_numbers.py`: rendered words ≤ 1,900, and whitespace tokens ≤ 1,950 on both the pasted markdown and the rendered text. v2.2: 1,888 words; 1,881 / 1,914 tokens.
   - Table 1 separators compacted (`|1.5+0.5|`, `0.34/0.67/0.80`, `≥10 plays`).
   - Cuts, none a caveat: the trait name after "6.8 games"; the 0.67/0.33 restatement of w; "error ratio 0.28" (the CI is now in percent, as on Figure 3); Section 3.2's per-trait list (Table 1 has it); "These slow traits are where a good prior matters most"; the WR signal's +0.42 (Figure 4 shows it); E011's 54 receivers and 18 drills; "every logged test" in the Code line; shorter Section 2 phrasing and captions.
2. **Hook accuracy.** **Applied.** "Nor would tracking have helped" was wrong for Receiver A: Combine tracking moved his forecast +0.142 yd/s (0.29 mph) toward his season (`e012_case_priors.json`). The hook now says so, then that across all 15 test receivers tracking left error after one game within noise (0.96, 0.80–1.09), with the pooled H3 in Section 3.3. The checker verifies the nudge, its direction (C2 < C3 < season), and that C3 and C1 both stay more than 1 mph below his season. C1 is checked, not quoted (guardrail, section 4).
3. **Pools.** **Applied.** The hook selects among "the 11 of 15 rookie receivers in our 2025 test with ten or more qualifying games", and the chance argument uses the same 11: 1 − 0.9545^11 = 0.40 ("about 40% of the time"), replacing v2.1's 15-receiver 0.50. Both counts come from `e012_case_priors.json pools.T1`; the checker also confirms 15 = H1's and the T1 tracking test's n. I verified against the gitignored forecasts parquet that Receiver A has the largest absolute gap of the 11.
4. **Framing.** **Applied.** Section 4 now says unreliable measures do not explain the nulls: 13 of the 16 pass the reliability gate (E008 `summary.transfer.nulls_passing_c5`; the checker recomputes it from each link's `rel_x` and `rel_y`), and power and imperfect game targets limit the tests. The reliability paragraph moved to Section 5 under "Stability is a separate question from transfer", with drill intensity (stable, predicts nothing) as the example.
5. **Captions.** **Applied.** Figure 1 names the game-play panel and "his forecasts" (the figure builder added post hoc stopwatch-only and + tracking markers during this pass; see the open item in figure_plan, Figure 1); Figure 2 says the simulation-calibrated bounds are bold (primary) and the bootstrap thin; Figure 3 names the 80% coverage panel; Figure 5 names the first-step and session panels. Figure 4's caption already matched.

**Errors found (for LESSONS.md; the writer role may not edit it):**
- A word budget enforced with one counting method can pass while another common method fails (v2.1: 1,953 by the checker, 2,137 by whitespace) → enforce every plausible count, on both the markdown and the rendered text.
- A selection rule and the chance argument for it used different pools (11 vs 15) → state the selection pool and compute any "how surprising is this" figure over that same pool.
- A hook sentence ("Nor would tracking have helped") generalized a pooled result to an individual whose own value went the other way → check every individual-level claim against that individual's stored values.
