# Experiment log

One entry per idea. Newest first.

Template:
```
## YYYY-MM-DD — <idea>
- Hypothesis:
- Method:
- Result (with uncertainty):
- Decision: keep / kill / revisit — why
```

## 2026-10-19 — Correction to the E013 summary below (found by the writeup v2.6 final checker)
- "Once the Combine and draft are known, alignment share adds at most 0.16 games" is a difference involving a floored (0) fit for 5 of 9 traits (T2, T3, T5, T9, T11). It holds only for T1, T6, T7 and T12 and must not be quoted as a general bound.
- "The test could not detect effects of the size the internal fit implies" overstates the case: power at the 3.35% best case was about 0.59 (80%-power detectable cut 4.4%). The correct statement is that the 95% CI (−0.4% to +7.0%) excludes the 3% cut that exactly known alignment would bring (post hoc).
- The writeup (v2.6) quotes neither original statement; check_numbers.py fails if either returns.

## 2026-10-19 — E013 summary (user decisions 2026-10-18: real names, keep the title "The Combine Exchange Rate", run the role-aware analysis)
- **E013 (preregistered, committed before any result as 8d2d8ba): none of H1–H4 is supported.** Knowing a rookie's role from his first games (edge / tackle / corner / slot share) did not improve his 2025 in-season forecast (k = 1 ratio 1.028; 1.011 after correcting the one-game precision gap). Beyond role, Combine + draft (H3, p 0.077) and tracking (H4) were not significant.
- **Interpretation, written as bounds:**
  - Role does carry real between-player signal for WR hitch braking (29%) and DL rush distance (21%); DB top speed is weaker (10%).
  - Once the Combine and draft are known, alignment share adds at most 0.16 games.
  - For DL the role that matters is the roster designation (OLB vs DE) and body type, not edge alignment (descriptive).
  - Table 1 stays position-level; the writeup gets a short robustness note.
- **Writeup correction found by the E013 review:** "five traits reach 0.4–1.7 games" (E012 PH8) is contaminated by ridge shrinkage of the class dummies; it is 2 traits (WR top speed 0.63, DL rush distance 1.68).

## 2026-10-19 — E013 review response (two reviewers, both 'weakened'; post hoc PH4–PH13)

- **Unchanged.** The preregistered verdicts (none of H1–H4 supported) and every stored number. Re-run with `--reuse`: numbers match to within 9e-16; parquets byte-identical.
- **Prospective (post hoc)**
  - The one-game comparisons were biased against role: the coefficient was trained on whole-history shares but fed a share from one game.
    - Calibrated on training players: H1 1.011 [0.990, 1.036], H3 0.977 [0.937, 1.015].
    - Training shares at matched precision: H1 1.007, H3 0.983.
    - WR hitch braking per-trait ratio 1.33 → 1.14 / 1.04. The deep-share deterioration (1.110) falls to 1.015.
  - H3's 3.6% cut is mostly WR hitch braking (two rookies carry 78% of the role-aware blend's excess error).
    - H3 without that trait: 0.987.
    - Role-aware + Combine and draft vs the position average: 0.991 [0.949, 1.033] (E012's prior 0.986).
    - The WR test's team-cluster family q is 0.015 (player-level q 0.64).
  - Power: minimum detectable ratio 0.956 at one game; the best case the internal fit implies is about 0.967, so the test could not detect it.
- **Internal (post hoc)**
  - Role is real between players by the preregistered ANOVA (10,000 within-class permutations, BH over 9 traits):
    - WR hitch braking 29% of between-player variance (q 0.003);
    - DL rush distance 21% (q 0.003);
    - DB top speed 10% (q 0.026);
    - worth 1.89, 0.95 and 0.51 games if known exactly;
    - others ≤ 6%.
  - Learned out of fold, against a ridge on the class dummies, role is worth 0.90 (T3) and 1.14 (T6) games, about 0 elsewhere.
  - Once the Combine and draft are known, role adds −0.06 to +0.16 games for every priced trait.
  - DL rush distance's 1.23 games = 0.52 beyond role + 0.71 shared (edge share is 75% predictable from the Combine and draft). Edge share beyond the Combine and draft adds 0.06.
  - The DL role signal is the roster designation, not edge alignment: among pure edge players, OLB vs DE differ by 0.12 yd (p 0.0003); OLB is +0.110 beyond edge share and weight (p 4e-5). Descriptive; the label may postdate the draft.
  - E012's same-CV "relative to C0" values mixed ridge shrinkage of the class means into the Combine value. Against the dummies-only ridge, E012's Combine + draft forecast reaches ≥ 0.4 games for 2 traits (T1 0.63, T6 1.68), not 5.
- **Decision: keep the robustness note, rewritten as bounds.**
  - Table 1 stays position-level.
  - The note says:
    - role carries real between-player signal for WR hitch braking and DL rush distance (DB top speed weaker);
    - once the Combine and draft are known, alignment share adds ≤ 0.16 games;
    - it did not improve 2025 forecasts, even corrected (the test could not detect effects of the size the internal fit implies).
  - The DL wording becomes "roster designation (OLB vs DE) and body type, not edge alignment".
  - Fix the writeup's "five traits reach 0.4–1.7 games" (now two).
- **Revisit if** a draft-day OLB/DE (or CB/S) designation is preregistered as a k = 0 role prior.

## 2026-10-19 — E013: Role-aware Combine Exchange Rate (preregistered)

- **Hypothesis**
  - Prereg: `experiments/results/E013_prereg.json`, written before any role–trait association.
  - Motivation: the mock football judge said position averages pool EDGE with interior DL, CB with safeties, slot with outside WR, and tackles with interior OL.
  - On 2025 rookies:
    - H1/H2: a role-aware prior (role share from his first k games) beats the position-average blend at k = 1 and k = 2.
    - H3: beyond role, Combine + draft help at k = 1.
    - H4: Combine tracking helps beyond that at k = 1.
- **Method**
  - Code: `src/bdb27/probes/i_role_aware.py` and `experiments/E013_role_aware.py`. They reuse E012's tables and functions.
  - E012's internal CGE, B1 draws and 626 forecast rows are reproduced exactly.
  - Implementation choices were frozen before any real-data run (sha 9ace65fb…).
  - Post hoc: PH1–PH3 (figure layout, a descriptive ratio panel, headline wording) and BUG1 (a code fix).
  - Role share = his share of slot / edge / tackle / corner alignments in the trait's qualifying games.
  - 9 priced traits.
  - λ_role by stratified ANOVA with the role covariate (new; unbiased in simulation).
  - B1 with 1,000 resamples (E012's draws).
  - Calibrated bounds from 200 simulated datasets per grid point; B1 coverage checked at the T1 and T6 designs.
  - Prospective test: 100,000 sign flips, 10,000 bootstrap resamples, BH over H1–H4, team-cluster sensitivity, and a 67-test secondary family.
  - Run: 21,174 s wall; peak RSS 0.5 GB.
- **Result (with uncertainty)**
  - **Primaries: none supported.**
    - H1 ratio 1.028 [0.996, 1.070], p 0.92.
    - H2 1.002 [0.984, 1.022].
    - H3 0.964 [0.917, 1.008], p 0.077, q 0.31 (team-cluster p 0.055).
    - H4 1.004 [0.984, 1.022].
  - Role from his whole season does not help either: 1.013 [0.986, 1.046].
  - No secondary result survives the family BH (smallest q 0.64). WR Combine + draft beyond role at k = 1 is 0.900 [0.808, 0.969] (p 0.010, q 0.64).
  - **Role worth (2023–25 fit):** at least 0.1 games for 2 of 9 traits.
    - DL rush distance 0.77 (calibrated 0.11–2.08).
    - WR hitch braking 0.24.
    - 0 for the other 7; calibrated upper bounds 0.36–2.62.
  - **DL rush distance:** E012's Combine + draft value of 1.23 games is 0.52 beyond edge share (calibrated −0.12 to 2.13); the change is 0.71 [−0.02, 2.18].
  - **Prior worth** under the role-aware + Combine prior: 2.1–6.9 games (E012: 2.0–6.8).
  - **B1 coverage** for beyond role: 79–84% at nominal 95%.
- **Decision: keep as a robustness note (one sentence and a Limitations update), not a new section.**
  - Position-average priors are not hiding role for the priced traits, so Table 1 stays position-level.
  - The DL wording becomes "roster role and body type more than alignment".
  - Revisit if roster position is preregistered as role, or if traits are rebuilt without alignment centring.

## 2026-10-15 — GATE 2b: the framing becomes Combine Game Equivalents (E012), with the transfer map as its evidence
- **E012** (preregistered; amendment A1 fixed my blend-weight algebra before any real-data run; reviewed by 2 skeptics, both 'weakened'; the responder accepted almost every point):
  - In season, a player's own qualifying games outweigh any pre-season prior within 2–7 games for the 9 reliably measured traits.
  - On 2025 rookies (pseudo-prospective: fit on 2023–24 only), shrinkage toward the position average delivers 99% of the blend's 72% error cut after one game (H1 ratio 0.28 [0.20, 0.40]).
  - The stopwatch + draft slot prior adds ~5% before the first game (post hoc, BH q 0.29) and ~2.5% after one.
  - **Combine tracking adds nothing** (H3 ratio 1.021 [1.004, 1.039], slightly worse).
  - In game units the Combine is worth 0 to ~1.2 games where measurable; every interval includes 0. T4, T8 and T10 are not identified (reliability < 0.5).
- **Decision:** the writeup is rebuilt around CGE and the in-season rule (usable week to week: how much to trust a rookie's games after k games, per trait), with the transfer map, E009 and the drill reliability profile as the 'why'. WR tracked top speed stays a fragile side note. No more hypothesis rounds: the remaining work is figures, the public notebook, and the writeup (PLAN Gate 3).
- **Open shared-module items** from the E012 review:
  - RidgeCV's LOO alpha search sees duplicated bootstrap rows; check the E007–E011 bootstraps.
  - `linkage._design`'s dropped year dummy depends on row order.
  - The meaning of `nfl_position` is unverified.
  - The orphaned loky workers were killed on 2026-10-15.

## 2026-10-15 — E012: Combine Game Equivalents (preregistered; revised after two reviews)

- **Hypothesis**
  - Prereg: `experiments/results/E012_prereg.json`, plus amendment A1. A1 replaced the written blend weight with the Bayesian weight k/(k+λ/(1-R²)), recorded before any real-data run.
  - Per trait, CGE = λR²/(1-R²) counts how many games of a player's own tracking the Combine is worth.
  - H1/H2: on 2025 rookies, a blend with the C2 prior beats games-only at k=1 and k=2.
  - H3: the C3 prior (adds Combine tracking) beats the C2 prior.
- **Method**
  - Code: `src/bdb27/probes/h_game_equivalents.py` and `experiments/E012_game_equivalents.py`. Implementation choices were frozen before the data (sha 70c22471…).
  - 12 game traits.
  - Variance components with draft-year strata; 20×5 ridge CV for C0-C3; true-score R² = R²_CV / reliability.
  - B1 player bootstrap.
  - Validation 1: chronological crossover. Validation 2: fit on the 2023-24 seasons, applied to the 2025 rookies (pseudo-prospective).
  - Post hoc after review: PH5-PH12 (listed under Review response).
  - Full run: 16,086 s; peak RSS 3.2 GB.
- **Result**
  - **Preregistered tests**
    - H1 supported: error 2.143 vs 7.590, ratio 0.28 [0.20, 0.40], p 1e-5.
    - H2 supported: 1.901 vs 3.962, ratio 0.48.
    - H3 not supported: ratio 1.021 [1.004, 1.039], so tracking made the prior slightly worse.
  - **Shrinkage does the work**
    - The position-mean (C0) blend gets 99% of the H1 gain and 98% of the H2 gain.
    - The C2 prior vs C0 is post hoc: -5% error at k=0 (p 0.03, BH q 0.29; drivers T1 and T4) and -2.5% at k=1 (p 0.08).
  - **Own games dominate**
    - Half-weight on his own games comes after 2.0 (WR top speed) to 6.8 qualifying games for the identified traits.
  - **CGE for C2** (B1 interval; calibrated one-sided bounds; 2023-24 fit)
    - T1 WR top speed: 0.47 [0, 0.83]; calibrated [0.08, 1.6]; 2023-24 0.14; 0 without the 2023 class.
    - T6 DL rush burst: 1.23 [0, 2.22]; calibrated [0.10, 3.9]; 2023-24 0.08. NFL position alone is worth 2.36 games, and C2 adds -0.25 beyond it, so this is role and body type.
    - T7 0.24 and T12 0.25.
    - 0 for T2, T3, T5, T9 and T11; calibrated upper bounds 2.5-3.6.
    - T4, T8 and T10 are not identified (reliability or split-half below 0.5).
  - **Combine tracking increment (ΔCGE, C3 - C2):** 0 or within noise for every trait.
  - **Validation 1:** 36/36 by the frozen rule, but the check is weak. 13 cells could not disagree and 23 could only fail one way. The literal continuous check agrees in 33/36 (T1 fails).
- **Review response**
  - B1 restored as the reported interval (PH5 is now a sensitivity only).
  - Simulation at each trait's own design (PH6): B1 at 20 repeats covered 78-82% and PH5 79% overall, both at nominal 95%.
  - Reliability gate, leave-one-class-out, and C0/role/exposure references.
  - Stationarity: T9 and T11 shift up to 2.2 between-player SD from season to season.
  - Qualifying-game disclosures; outlier table relabelled; figures redrawn.
  - Headline rewritten; it supersedes `E012_audit/summary.txt`.
- **Decision: keep as a descriptive "in-season" section, not a hero metric.**
  - Message: own games take over fast; shrinkage does the work; the Combine adds a few percent before game 1; Combine tracking adds nothing.
  - Revisit if more classes arrive or a season-centred, per-play-weighted model is preregistered.

## 2026-10-15 — ROUND 3 summary (E009 beyond-stopwatch, E010 drill intensity, E011 ball tracking, writeup v1)
- **E009:** beyond the stopwatch alone (B1), no link passes, and the same 16 links are null. Adding the draft slot moves partial r by at most 0.14 (median 0.01), so the beyond-B2 nulls are not a draft-slot artifact. WR tracked top speed: +0.42 beyond B1, criterion 4 marginal (depends on the BH family), fails leave-3-out.
- **E010 (preregistered):** drill intensity is a reliable trait (0.74; 0.66 beyond B2) but shows no link to play intensity (−0.003 [−0.13, +0.12]) or snaps per season (−0.04 [−0.14, +0.06]); **kill**. OL "intensity" is mostly run-out speed when tracking stops.
- **E011:** Combine ball tracking exists only for 2025 (54 WR/TE). It is clean (catch timing within one frame for 87%), but no catch mechanic is a stable trait (best 0.42), so the gate stopped every game test. **Park.**
- **Writeup v1** (audit framing, 1,831 words, 174 number checks pass). Mock judges: Football 5/10, Writeup 6/10, Data Science 7/10, Data Viz 5/10 (final figures not built). New post hoc result: one game of a WR's own tracking forecasts his later game top speed about as well as his best Combine measure (r 0.56 vs 0.57); 2–8 games give 0.75–0.79 (n = 42). The three post hoc scripts were moved from the scratchpad into `experiments/posthoc_writeup_v1/`.
- Open shared-module items: the LOCO null SD is inflated by a few degenerate ridge fits (use median + MAD); FL p near a BH threshold needs its Monte Carlo SE / adaptive permutations; HistGradientBoosting ignores the n_jobs cap (OpenMP); `team_share` pooled across positions mixes cross-position pairs (pass team_col=[team, season, position]); `cross_drill_icc(zscore=True)` silently drops zero-variance groups.

## 2026-10-15 — Writeup v1 (audit framing): round-3 mock-judging fixes
- **What:** applied two mock judges' reviews (Football/Writeup 5/6; DS/Viz 7/5; both "weakened").
  - Every v0 number was upheld.
  - The problems were framing, consistency and missing analyses.
  - Files: `writeup/draft.md`, `writeup/outline.md` (§9 lists each issue as accepted, rejected or deferred), `writeup/figure_plan.md`.
- **Main changes:**
  1. **The WR signal follows Gate 2.**
     - "Fits a modest link to game top speed (about +0.3) as well as none, and no link to production".
     - "Tiebreaker" removed.
     - Renamed hidden → **finishing speed**, because clubs can approximate it from the official 20-yard split (not in our data).
  2. **Repeat is checked on the residual too.**
     - Table 1 shows metric / residual reliability.
     - DL 40 top-speed residual 0.29 fails; WR 0.62–0.75.
  3. **Bounds as measured and error-free** (up to +0.73), with population wording.
  4. **Fragility disclosed.**
     - Deep-route usage control: +0.21 [−0.10, +0.49] (post hoc).
     - The 3 supportive WRs are all 2024 draftees; 2023 shows nothing.
     - A tracking-artifact alternative is named, since official timing doesn't corroborate (r −0.11).
     - CV R² 0.18 → 0.23, excess +0.045 ± 0.029.
  5. **[[TBD-REV3]] resolved.** E006-P1 as computed (AS_R = 20) passes rev-3 criteria 1–6 on seeds 0–2.
     - Class holdout 2/3: 2023 and 2025 excess > 1 null SD; 2024 negative.
     - It still fails BH across ~90 round-one tests (q ≈ 0.06).
     - The AS_R = 1000 estimator (+0.51, post hoc) also passes; without 2025 +0.39, leave-three-out +0.35.
  6. **Week to week (new; post hoc; Figure 5).** One game of a WR's own tracking forecasts his later game top speed as well as his best Combine measure (r 0.56 vs 0.57); 2–8 games 0.75–0.79.
     - n = 42.
     - The gap's 95% CI excludes 0 only at games 5 and 8, hence "probably better".
     - The finishing-speed leaderboard moved to the notebook.
  7. **Other fixes.**
     - The 172-pairs 59% now carries a receiver-bootstrap CI of 43–73% (permutation p 0.062) and sits in the Figure 1 footer.
     - The hook pair is disclosed as the largest-gap pair.
     - "Drills run below full speed" removed (E010 tests it).
     - Cushion: alignment removes 73% of between-receiver variance.
     - Exposure check reported.
     - Labels and process wording fixed.
- **Error found:** v0 quoted "receiver top speed (0.92)" from the S0 target's reliability; observed top speed is 0.91 (E001-WR). Fixed.
- **Size:** 1,831 words before fill-ins. Fill-in budgets cut to ~125, for a ceiling of ~1,970; a 7-step cut list is in the outline. 6 of 9 figures/tables, 2 reserved.
- **Verification:** a scratch checker asserts every quoted number and string against E002/E006/E007/E007_leaderboard/E008 and the post hoc JSONs: 174 checks, 0 fails.
- **Rejected or deferred:**
  - Pulling the public 20-yard splits (external data; rules unchecked) goes to the user.
  - The AS_R = 1000 estimator is not made the headline, because preregistered definitions may not change; it is quoted as post hoc.
- **Action needed:** promote `ph_pairs`, `ph_rev3` and `ph_inseason` from the scratchpad (`round3/writeup_v1/`) into the notebook or a committed results JSON. The rev-3 pass claim and the in-season numbers currently trace only to them.
- **Decision:** v1 replaces v0 as the base. Fill E009/E010/E011 per outline §3, then build Figures 4 and 5 first (JSON-only / light data).

## 2026-10-15 — E011: Combine ball tracking at the catch point (feasibility, descriptive; 2025 class only; revised after review)

- **Hypothesis:** descriptive, with no preregistered primaries. Can the Combine BALL rows yield catch-point mechanics that are stable receiver traits?
  - Mechanics:
    - speed retention through the catch
    - deceleration into it
    - catch offset from the ball's path
    - adjustment distance and time in OVER_SHOULDER_ADJUST, CIRCUS and the fade drills
  - The plan was written at 2026-10-15T03:59:56Z, before any mechanic, reliability or association was computed. It is copied verbatim into `E011.json` as `plan`.
  - Only mechanics with reliability ≥ 0.5 would get at most 3 prelisted exploratory `linkage.evaluate` tests against rookie receiving outcomes: catch rate over expected, YAC over expected, and drops per target.
  - Power was stated up front: n ≈ 24, one-sided MDE ≈ 0.54, true-score MDE above 1.

- **Method:**
  - **Code:** `src/bdb27/probes/g_catch_point.py` and `experiments/E011_catch_point.py`. Runtime 1-8 min depending on load, peak 0.7 GB. Reruns are deterministic up to last-digit floating-point differences.
  - **Coverage:**
    - BALL rows exist only in 2025: 551 attempts, 54 players (37 WR, 17 TE), 18 drills.
    - That is 93% of the reps in those drills, with none in 2023-24 or any other position group.
    - x/y only, no ball height.
  - **Release:** first run of ≥ 3 frames with ball speed ≥ 12 yd/s. Before release the ball never exceeds 11.96 yd/s.
  - **Contact:** the first one-frame ball-speed drop to < 0.75 × the previous frame's, i.e. the ball's first impact with anything.
  - **Outcome:** inferred from post-contact ball–receiver distance (preregistered; no labels exist).
  - **Receiver kinematics:** SG.
  - **Reliability:** cross-drill ICC on player × drill means, z-scored within drill × session, with 2,000 player-bootstrap CIs.
  - **Descriptive:** Spearman correlations with 12 Combine measures.
  - **Game side** (exposure, reliability and power only): cross-fitted catch probability, NGS expected YAC, own drops (`dropped_pass` & target), EB shrinkage with `cluster=game_id`.
  - **Post hoc additions after review**, labelled `revision` in the JSON; no preregistered definition changed:
    - an unanchored contact-time check
    - a passed_him / ball_unrelated split of not_reached
    - retest and ICC bootstrap CIs
    - velocity matching to estimate the clock offset, plus a clock-shift sensitivity
    - the adjust_dist decomposition
  - **Figures:** `E011_{qc_contact,qc_outcomes,reliability}.png`.

- **Result (with uncertainty):**
  - **The data are clean and easy to segment.**
    - Contact time agrees with an unanchored geometric closest approach within 0.1 s, one 10 Hz frame, for 86.9% of 488 reached attempts (chance 11.6%; 92.2% within 0.15 s).
    - The first version of this check was bounded to [-0.05, +0.15] s by construction and is withdrawn.
    - Preregistered outcomes: 369 secured, 37 separated, 82 reached but unobserved, 34 not reached, 25 tracking ended in flight, 4 no flight.
    - Post hoc split of the 34 not reached:
      - 15 passed him: the ball came within 1.5 yd in flight and landed beyond him.
      - 4 have ball rows from another throw.
      - 15 remain, short or wide.
    - The separated share of observed outcomes is therefore 9-13% depending on the rule.
  - **The clock offset is not identified more tightly than about ±0.15 s.**
    - Velocity matching on carried balls gives +0.04 to +0.06 s or -0.11 to -0.13 s depending on the window.
    - The pooled slope test, -0.009 s, is confounded by ball handling: the within-attempt slope is +0.33 s.
  - **No mechanic is a stable trait.**

    | Mechanic | Reliability [95% CI] | ICC per drill [95% CI] |
    |---|---|---|
    | retention | 0.31 [0.00, 0.50] | 0.08 [-0.02, 0.16] |
    | decel_in | 0.33 [0.00, 0.54] | 0.06 [-0.01, 0.13] |
    | catch_offset | 0.42 [0.15, 0.56] | 0.08 [0.02, 0.14] |
    | adjust_dist | 0 | -0.33 |
    | adjust_time | not estimable | — |

    - adjust_time is 0 s in every OVER_SHOULDER_ADJUST and fade attempt.
    - adjust_dist mostly measures run-speed change in the fades and route curvature in CIRCUS.
    - Reliability 0.5 for catch offset would need about 6-46 distinct drills per player.
    - Within-drill retest ICCs cannot bound anything: CIs of about ±0.45 on 7-32 repeated cells, and 22 of 48 repeats followed a failed rep.
    - Post hoc checks all stay below 0.5: WR only (best 0.46), 0.3/0.7 s windows, drill-centred only, ±0.1/0.15 s clock shifts, and the separated rate (ICC ≈ 0).
  - **Descriptive:** 2 of 60 Combine-correlation CIs exclude 0, against about 3 expected by chance.
  - **Gate failed → 0 tests spent; no Combine–game association computed.**
    - 24 of the 54 have ≥ 10 rookie targets.
    - Target split-half reliability: 0.39 (catch rate over expected), 0.29 (YAC over expected), below 0 (drops).
    - The full B2 leaves resid_df = 6.
    - `dropped_pass` is play-level: 199 of the 251 tagged player-plays with a player_play row were not the target.

- **Decision:** **park.** Combine ball tracking is a clean, easily segmented data source, but the 2025 release cannot support a receiver trait or a game link.
  - **Revisit if** there are more classes with BALL rows, ball height or a higher sampling rate, repeated catches per drill, catch labels, or a verified ball–player clock.
  - **Writeup fill (29 words):** "The ball is tracked only in 2025 (54 receivers, 18 catching drills, no height). Catch timing is measurable to one frame, but no catch mechanic was reliable (best 0.42)."
  - No figure slot spent.

## 2026-10-15 — E010: Drill intensity (preregistered): how close to his own tracked-40 peak a player runs his position drills (revised after skeptic review)
- **Hypothesis** (prereg `experiments/results/E010_prereg.json`, written before any association; protocol rev 3): players who run their position drills closer to their own tracked-40 peak speed (P1) play closer to their own game ceiling on a typical play and (P2) earn more playing time, beyond B2. Expected sign + for both.
- **Method:**
  - **Feature** `c_drill_intensity`:
    - Per position-drill attempt (SKILL_DRILLS_{WR,TE,DB,DL,OL}; the LB group, 4 DL players, excluded): interior SG peak speed (speed.smax definition) / c40_smax.
    - z-scored within drill x Combine session (drill key = position group + drill name; 4 naming variants merged).
    - Drill-level mean of z, then the player mean over >= 2 drills.
    - QC (fixed from trace plots before any reliability was computed): keep attempts with path within [0.5, 2.5] x their drill x session median. This dropped 5 (2 aborted, 3 tracking overruns).
    - Population: 407 (WR 91, TE 34, DB 100, DL 88, OL 94).
  - **Stop rule:** `reliability.cross_drill_icc(zscore=True, center_by=[drill, session])` on player x drill values.
  - **Targets:**
    - P1 `g_play_intensity` = median / q95 of REG play max speed (q95 identical to E001's game_top_speed), >= 50 plays, n 277. **Coverage:** WR/TE tracked plays are pass plays only (their player_play REG rows contain no runs; tracked plays are a median 0.61 / 0.42 of career off+def snaps), so for WR/TE P1 is per route. For DB/DL/OL it is per snap (1.00 of snaps; 37-42% runs).
    - P2 `g_snaps_per_season` = (off + def snaps) / (2026 - draft_year), all 407 players.
  - **Harness:** `linkage.evaluate` rev 3, pooled (position dummies, within-position Freedman-Lane), one-sided +, 10,000 FL perms, 50x5 CV, 2,000 CV perms, seeds 0-2, n_jobs 3.
  - Every implementation choice is listed in `E010.json` (`implementation_choices`). The post hoc block added after the skeptic reviews is `game.post_hoc_review` (choices in `post_hoc_review.choices`; rerun alone with `--post-hoc`, ~7 min, 0.75 GB).
  - Files: `src/bdb27/probes/f_drill_intensity.py`, `experiments/E010_drill_intensity.py`, `experiments/results/E010.json`, `experiments/figures/E010_{qc_traces,qc_intensity,targets,results}.png`, `data/processed/e010_{players,play_max}.parquet`.
  - Runtime 25-65 min depending on CPU load, peak 1.46 GB.
  - A full top-to-bottom rerun after the review reproduced `E010.json`: 2,669 of 2,687 values are bit-identical. The 18 team-share values differ below 1e-13, from floating-point summation order in polars `group_by`. Both parquets are identical.
- **Result (with uncertainty):**
  - **Stop rule passes.** Cross-drill ICC 0.27, SB reliability 0.74 (mean 8.4 drills).
    - By position: WR 0.82, TE 0.55, DB 0.74, DL 0.73, OL 0.75.
    - The feature shares its 40-peak denominator (r -0.45 within position). Reliability beyond the 40 peak is 0.64, beyond B2 0.66, and beyond B2 + c40_smax + tau 0.59 (post hoc), so criterion 5 holds every way.
  - **What the feature is** (QC; post hoc where marked):
    - Position drills peak at a median 74% of the player's tracked-40 peak.
    - 41% of attempt peaks fall on the last tracked interior frame (run-out censored). Drills censored in >= 60% of attempts make up 85% of OL drill rows, 48% DL, 26% DB, 16% WR and 12% TE. So **OL "intensity" is essentially speed when tracking stops**: in the mirror, drop and wave drills the lateral phase runs at a median 2.3-3.6 yd/s (90th percentile of speed before the last 1.5 s), and the 6-7.4 yd/s peak is the run-out (post hoc check). TE block explosion is a ~2 yd/s blocking drill.
    - Post hoc: the censored-drill and true-peak-drill halves of the feature agree within WR (r +0.44), DB (+0.60) and DL (+0.54) but not TE (-0.11, n 34), so **the TE feature has no single construct**. In 19 of 21 drills where both kinds of attempt are common, censored attempts score higher z at near-identical durations. Peak-at-end therefore partly marks players accelerating through the finish, not only a tracking cut (player feature vs his session-centered censoring share: r +0.26).
    - Post hoc: **beyond B2 the feature is still partly sprint profile.** Partial r with c40_smax -0.36, vmax -0.28, tau -0.21. Within position, tau + vmax + weight explain R2 0.53 (WR), 0.61 (TE), 0.47 (OL), 0.20 (DB), 0.10 (DL). B2 + c40_smax + tau explain 34% of the feature in sample (B2 alone 24%).
  - **Targets:**
    - P1 split-half 0.73 (WR 0.66 to OL 0.78); exposure r +0.01; position explains 82% (CV R2 of B0).
    - **P1 is largely role within position** (post hoc). Alignment and route-depth mix explain CV R2 0.18-0.37 (in sample 0.31-0.43). Single covariates: OL tackle share r +0.61, DL EDGE share +0.61, WR deep-route share +0.55, DB CB share -0.49, DB man share -0.42. A play-context-adjusted P1 (play max minus its position x alignment x run/pass x route x coverage cell mean) correlates only 0.45 with raw P1 within position. Its split-half is WR 0.53, DB 0.54, DL 0.83, OL 0.72, TE 0.36.
    - **Team share (corrected).** The reported pooled -0.03 +- 0.04 used position-centered values across positions, so 2,122 of its 2,598 ordered teammate pairs are cross-position (e.g. a WR with an OL). It bounds a team-wide component, not scheme within a position. Same-position pairs only: +0.00 +- 0.10 (two-sided 95% upper about +0.19). Inverse-variance mean of the per-position estimates: -0.00 +- 0.09. Role-adjusted, same-position: +0.06 +- 0.12. So P1 is not a team-season effect; it is a stable player-or-role trait (team_share cannot separate the two).
    - P2: B2 CV R2 0.48 vs B1 0.02 (draft slot dominates). Season-1 vs season-2 player_play counts: r 0.73 given B0, 0.55 beyond B2 (post hoc; Spearman-Brown to available seasons 0.68).
  - **P1 (primary):**
    - Pooled within-position partial r -0.003 [-0.13, +0.12]; one-sided p 0.52 (seeds 0.52/0.52/0.52); excess dR2 -0.0006 +- 0.0011; LOCO 0/3; by class -0.07/+0.10/-0.05.
    - Leave-3-out: supportive -0.07 (p 0.84), contrary +0.04 (p 0.27).
    - Pooled observed r above +0.10 ruled out (one-sided 95%; position-stratified case bootstrap +0.11). One-sided MDE 0.16. Reliability-corrected (beyond-B2, both sides) [-0.19, +0.18].
  - **P2 (primary):**
    - Pooled partial r -0.042 [-0.14, +0.06]; p 0.79 (0.79/0.80/0.80); excess -0.0001 +- 0.0028; LOCO 0/3; by class -0.02/-0.09/-0.01.
    - Leave-3-out: -0.07 (p 0.91) / -0.01 (p 0.58).
    - Pooled observed r above +0.04 ruled out (bootstrap +0.04). MDE 0.13. Corrected for feature error only (snaps as recorded): [-0.17, +0.07]. If snaps per season is read as a playing-time trait (beyond-B2 stability 0.68, post hoc): [-0.21, +0.09], one-sided upper +0.06.
  - BH q (P1-P2) is 0.79 for both. Both fail criteria 1, 2, 3, 4 and 6 and pass 5.
  - **The bounds are pooled within-position averages.** By position (prespecified, descriptive):
    - P1: WR +0.10, DB +0.19 [-0.10, +0.45] (p 0.11), DL -0.12, OL -0.19; TE n 20 undefined.
    - P2: WR +0.05, TE -0.22, DB -0.15, DL -0.13, OL -0.05.
    - Per-position one-sided 95% upper bounds reach +0.41 (DB P1) and +0.34 (WR P1), and +0.24 for P2 (WR, TE). No detectable heterogeneity (post hoc feature x position F-test: P1 F(3,230) 1.29, p 0.28, without TE; P2 F(4,378) 0.74, p 0.56).
  - **Prespecified robustness, all null:**
    - Without 2025: P1 +0.02 [-0.13, +0.17]; P2 -0.05 [-0.18, +0.07].
    - Feature without division -0.01 / -0.02. Player-level median over drills -0.02 / -0.05.
    - The prereg's "drill-level median" was implemented as a player-level median, because a per-drill median over 1-2 attempts is nearly the mean. Run late after review, the literal version is near-identical to the primary (24 of 3,407 player-drill rows differ; r 0.9997): P1 -0.003 [-0.13, +0.12] p 0.52; P2 -0.043 [-0.14, +0.06] p 0.80.
    - P1 share of plays >= 80% of q95: -0.04. P2 log: -0.04. P2 with special-teams snaps: -0.005.
    - BH over the 21 originally reported p-valued tests: every q is 0.90.
  - **Prelisted exploratory:** starts per season -0.025 [-0.13, +0.07], p 0.69. Honors (descriptive): 8 players, mean within-position feature -0.24 vs +0.00 (no test).
  - **Post hoc after review (one-sided FL p; in no prespecified BH family; all null):**
    - Role-adjusted P1: -0.06 [-0.18, +0.06], p 0.83. P1 given B2 + position-specific role mix: -0.00 [-0.13, +0.13]. Role-adjusted by position: WR -0.07, DB +0.24 [-0.05, +0.49] (p 0.06), DL -0.15, OL -0.25.
    - B2 + c40_smax + tau: P1 +0.01 [-0.11, +0.14]; P2 -0.03 [-0.13, +0.07].
    - Feature from true-peak drills only (peak at end in < 30% of attempts; n 215 / 312): P1 +0.09 [-0.05, +0.23], p 0.13 (+0.10, p 0.11, with sprint controls); P2 -0.03 / -0.02. With a < 50% cut: P1 +0.04 [-0.10, +0.18], p 0.31; P2 -0.03. This is the largest post hoc estimate, and it moves with an arbitrary threshold.
    - BH over all 37 p-valued E010 tests including post hoc: min p 0.056, every q >= 0.92.
  - **Post hoc (descriptive, no test): where P2's raw r of -0.13 goes.** It is -0.13 given B0, -0.08 with the 40 time, -0.07 with c40_smax or the NGS athleticism score, -0.04 with all traditional Combine results (B1), and -0.02 with draft slot alone. Either Combine athleticism or draft slot absorbs it: slower players run drills at a higher fraction of their own top speed (r with the 40 time +0.34 given B0), and slower players are drafted later and play less (-0.16).
- **Decision: kill.**
  - Drill intensity is a reliable Combine measure (0.74; 0.66 beyond B2; 0.59 beyond B2 + sprint profile). It shows no detectable link beyond B2 to the speed profile of a player's game plays (P1) or to his playing time (P2).
  - Pooled within-position partial r above +0.10 (P1) and +0.04 (P2) is ruled out (one-sided 95%; bootstrap +0.11 / +0.04). Per-position effects up to about +0.4 (DB P1) are not.
  - **What the two sides measure:**
    - The feature is drill pace relative to the tracked-40 peak. It is partly sprint profile (acceleration shape and the 40 peak beyond the 40 time). In OL drills and other censored drills it is speed when tracking stops. Not effort or motor.
    - P1 is median/q95 of play max speed, per route for WR/TE and per snap for DB/DL/OL. Within position it is largely alignment and route depth. It does not measure "how close to his ceiling he plays".
  - **Transfer map:** list it as "a stable drill measure with no transfer". It partly overlaps E007's sprint signal, so do not count it as an independent drill trait.
  - **Revisit** only with a preregistered DB test (role-adjusted P1; DB is the one cell where a positive link could hide, at +0.19 to +0.24 post hoc). An effort measure needs an external anchor such as pursuit speed on plays away from the ball, which these data cannot isolate.
- **Verification (2 skeptics: 'upheld', 'weakened').** Both rebuilt the feature, targets and inference with independent code, and every number reproduced exactly. Attempts to break the null (running order, censored drills, shared denominator, rank-based r, availability, cleaner constructs) left it null.
  - Accepted and fixed (all post hoc, labeled in `post_hoc_review`; no prespecified test or definition changed):
    1. Same-position team share.
    2. P2 decomposition reworded.
    3. "Pooled within-position" qualifier, with heterogeneity, bootstrap and per-position bounds.
    4. P2 corrected interval labeled as feature-error only, plus the playing-time-trait version.
    5. Literal drill-level median run and flagged.
    6. P1 described by coverage and role, with role-adjusted P1 added.
    7. Feature described as partly sprint profile, with reliability beyond B2 + c40_smax + tau.
    8. Construct by drill type: OL run-out speed, TE incoherent, censoring partly behavioral.
    9. DB as the cell where an effect could hide.
  - Rejected: none.
  - **Corrected reading:** no detectable link, pooled. The P1 target measures role-dependent play speed, not effort. Kill stands.

## 2026-10-15 — E009: Beyond the stopwatch: the 18 transfer-map links against B1 (no draft slot) — revised after round-3 review
- **Hypothesis** (prereg `experiments/results/E009_prereg.json`; evidence labelled SECONDARY and NOT BLIND, because E008's beyond-B2 results were known and B1 is nested in B2):
  - Does a tracking signature add anything beyond the traditional Combine stopwatch and measurements (plus position and draft year)?
  - Same 18 links, features, targets, populations and expected signs as E008.
- **Method:**
  - Links: built with E008's builders, imported unchanged. n and raw r match round 1; beyond-B2 partial r matches E008.json exactly.
  - Prereg family (18, one BH family): `linkage.evaluate(draft_controls=[])` under rev 3. One-sided Freedman–Lane with 10,000 permutations; 50×5 CV; 2,000 CV permutations; seed 0, plus seeds 1–2 when p < 0.10.
  - Prespecified robustness:
    - without the 2025 class (all 18);
    - leave-3-out for links passing criterion 1;
    - B1 − B2 difference in partial r.
  - Post hoc:
    - paired bootstrap of that difference, plus a Frisch–Waugh–Lovell decomposition;
    - beyond-B1 bounds (E008's definitions);
    - two-sided p;
    - BH over the 36 B1 + B2 tests;
    - E008's beyond-B2 tests re-judged under rev 3.
  - Post hoc after review:
    - Freedman–Lane at 2,000,000 permutations for links with p < 0.10, with the Monte Carlo SE of q;
    - q over other families: seeds 1–2, E008's 14 primaries, the round-3 primaries (E009 + E010), all 59 round-3 tests;
    - criterion 3 on robust null scales (median + MAD-SD; 84th percentile), from an exact replay of the harness LOCO null (44 runs, error 0);
    - delete-one jackknife interval for the draft-slot difference;
    - null ranges signed to the expected direction.
  - Test count:
    - 18 prereg tests;
    - prespecified: 4 seed reruns, 18 without 2025, 2 leave-3-out;
    - 38 new p-values on distinct hypotheses or samples (in the round-3 BH with E010's 21; E011 has none).
  - Files: `experiments/E009_beyond_stopwatch.py`, `experiments/results/E009.json`, `experiments/figures/E009_transfer_map_two_baselines.png`, `data/processed/e009_{transfer_map,link_players}.parquet`.
  - Full rerun: ~40 min, peak 0.64 GB. Harness outputs match the first run to ≤ 3e-15.
- **Result (with uncertainty):**
  - **No link passes criteria 1–6 beyond B1.**
  - **Two links pass criterion 1; they are one WR speed signal** (61 shared WRs; beyond-B1 residual r 0.79 for the features, 0.82 for the targets).
    - **E006-P1** (fitted 40 vmax → game S0, n = 61):
      - +0.42 [+0.14, +0.63]; p 0.0025 (2M permutations: 0.00254 ± 0.00004).
      - Excess ΔR² +0.070 ± 0.043. LOCO 2/3, also on robust scales.
      - Without 2025: +0.35, p 0.033.
      - **Criterion 4 is marginal (post hoc).** q18 0.045 under the prereg computation, but 0.058 at seed 1 (10k permutations). With the precise p: 0.046 [0.044, 0.047] (±2 MC SE) over the prereg family (passes); 0.035 over E008's 14 primaries; **0.051 over the round-3 primaries (fails)**; 0.15 over all 59 round-3 tests.
      - **Fails criterion 6:** without its 3 most supportive WRs (46% of the partial r), +0.24 [−0.07, +0.51], p 0.066.
    - **E001-WR** (n = 64): +0.29 [−0.00, +0.53], p 0.028.
      - Fails criterion 3 under the harness rule (1/3; 2/3 on robust scales, post hoc).
      - Fails criterion 4 (q 0.25; 0.24 precise).
      - Fails criterion 6 (seeds, and leave-3-out p 0.15).
  - **The other 16 links are null beyond B1** (the same 16 as beyond B2).
    - Partial r in the expected direction: −0.17 to +0.12 (raw −0.17 to +0.14); p ≥ 0.20.
    - Per-link ceilings +0.09 to +0.38; joint Bonferroni ceilings +0.15 to +0.47.
    - Reliability-corrected true-score partial r up to +0.14 to +0.79 is not excluded; 0.5 or more for E002-P2, E004-P2, E005-P2, E001-DB and E001-DL.
    - One-sided MDE 0.18–0.37. Outcome targets without 2025: all p ≥ 0.45.
  - **What the draft slot absorbs:**
    - Δ = r(B1) − r(B2), signed: −0.08 to +0.14 (median |Δ| 0.013).
    - No jackknife or bootstrap 95% interval excludes 0. The largest upper limit is +0.37 (E001-DL), so the draft slot could absorb up to ~0.37 of DL top speed's partial r.
    - The bootstrap SE is 0.9–2.0× the jackknife SE.
    - For E006-P1 the draft slot sharpens the link: +0.42 → +0.48 (Δ −0.06 [−0.18, +0.06], jackknife).
  - **LOCO rule (criterion 3):**
    - The rev-3 rule passes 0 of 16 nulls under each baseline (rev 2 passed 11 of 16).
    - On robust scales, 0–1 of 16 pass (B1: E002-P2, median rule; B2: E003-P1). No verdict changes.
    - This is specificity on links judged null by the same data, not calibrated sensitivity. The harness's null-SD threshold, inflated by a few degenerate ridge permutations, costs sensitivity on the WR signal.
  - **Rev-3 re-judgment of E008 (post hoc):** no verdict changes. E006-P1 passes 1–6 beyond B2 (q 0.0093 over E008's family), but fails BH across all ~90 round-1 tests (q 0.063, E008; 0.060 at 2M permutations).
- **Decision:** **keep** as the transfer map's second column.
  - The beyond-B2 nulls are not an artifact of the draft-slot control: beyond the stopwatch alone the same 16 links show nothing detectable.
  - The only signal is WR tracked top speed (one signal). It is fragile: it fails leave-3-out, its criterion 4 depends on the family, and it fails BH across round 1.
  - No hero claim. Report it as "information beyond the stopwatch", never as "market inefficiency".
- **Writeup:** placeholder text suggested for draft lines 36 (0 words), 83 (38 words) and 140 (one clause).

## 2026-10-14 — GATE 2: hero candidate fails; the transfer map is the main finding
- **E007 (preregistered "hidden speed", WR): no primary passes.**
  - P1 (flying 20 → game top speed): +0.31 [0.00, 0.56], q 0.075, fails leave-3-out.
  - P2 (cushion): null; post hoc, cushion is mostly the receiver's own alignment.
  - P3 (YPRR): wrong sign, not robust; positive effects above about +0.1 are ruled out.
  - The 5 exploratory tests are null.
- **E008 (all 18 round-1 links under rev 2): 16 nulls.** Each rules out an observed partial r above +0.09 to +0.38, but true-score effects of 0.5 or more cannot be excluded for six. The only pass (E006-P1, WR vmax → extrapolated S0) is the same WR speed signal as E001, fragile (3 WRs = 42%), and fails BH across about 90 round-1 tests (q ≈ 0.06).
- **Drill profile, like for like:** 2 of 22 repeated position-drill metrics reach reliability 0.5, against 28 of 28 tracked 40-yd metrics. Cross-drill composites: 10 of 68. Unreliability does **not** explain the nulls (13 of 16 nulls pass reliability); low power (n ≈ 60–80 against ~20 B2 columns) and imperfect game targets do.
- **Decision (PLAN §6):** no hero metric survived. The project's original core idea (drill movement signatures transfer to the same movement in games beyond the stopwatch and draft slot) has no detectable support at this n. The direction of the writeup goes to the user (pending).
- **Technical debt found:** round-1 scripts E002–E006 no longer run against rev-2 `linkage` (removed keys, `_fit_resid`), so the public notebook must rebuild from the probe modules as E008 does. `probe_e_game_profile.parquet` is stale (AS_R = 20). Harness criterion 2 ignores sign and criterion 3 (LOCO) does not discriminate; both need fixing before any new round.

## 2026-10-14 — E008: Drill reliability + transfer map (all round-1 links re-estimated under protocol rev 2; revised after skeptic review)
- **Hypothesis:** no new links. Re-estimate the 14 round-1 primaries and the E001 sanity link (WR/DB/DL/OL; not preregistered) under rev 2, so the whole round reads on one footing. Bound the nulls with reliability-corrected intervals and profile every Combine signature the probes built (PLAN §6 fallback, now a core writeup section).
  - Expected signs: preregistered for E002–E005; inferred from the hypothesis text for E006 (no sign field); set after round 1 for E001 (two-sided p reported).
- **Method:**
  - Rebuild every feature and target from the round-1 probe parquets/functions. All 18 reproduce round-1 n and raw r.
  - Harness: `linkage.evaluate` rev 2 (B2 + year dummies; one-sided Freedman–Lane with 10,000 permutations; 50×5 CV with 2,000 permutations).
    - Seeds 1–2 when p < 0.10.
    - Without-2025 reruns.
    - Leave-3-out of the most supportive players.
    - Exposure check.
    - BH over the 14 primaries; E001 over all 18.
  - Reliability-corrected interval = CI / √(rel_x·rel_y), using the **beyond-B2 reliabilities** (the parts the partial r uses). The as-used version is an under-correction. Joint ceilings are Bonferroni over the 18 links.
  - E001 reliabilities: the feature's is computed for a max over attempts; the target's split-half is recomputed on the link samples with the exact target definition.
  - Drill profile: 142 signatures, grouped by reliability type (repeat attempts / cross-drill / within-attempt), with session and year shares.
  - Files: `experiments/E008_transfer_map.py`, `experiments/results/E008.json`, `experiments/figures/E008_{transfer_map,drill_reliability}.png`, `data/processed/e008_{transfer_map,drill_reliability,link_players}.parquet`. ~25–35 min, ≤1.8 GB, deterministic across runs.
- **Result (with uncertainty):**
  - **One link passes criteria 1–6: E006-P1, WR fitted 40 vmax → extrapolated game top-speed capacity S0.**
    - n = 61, partial r +0.48 [+0.20, +0.68], one-sided p = 0.0007 (two-sided 0.0013).
    - q = 0.010 over the 14 primaries; **q ≈ 0.06 across all ~90 round-1 tests (does not survive)**.
    - Excess ΔR² +0.063 ± 0.056 (3/3 seeds); LOCO 2/3 (2024 −0.07).
    - Fragile: 3 WRs carry 42% of the partial r; without them +0.31 [−0.01, +0.57], p 0.03. Without 2025: +0.38 [+0.01, +0.67]. By class: +0.29 / +0.50 / +0.67.
    - Corrected (beyond-B2) interval: [+0.26, +0.88].
    - **Not independent of E001-WR:** same 61 WRs; B2-residual r 0.77 (features), 0.83 (targets).
  - **E001-WR (observed game top speed), the same signal, fails criteria 4 and 6.** r +0.32 [+0.03, +0.56], p 0.016 (two-sided 0.03), q18 0.15; leave-3-out p 0.15.
  - **Other 16 links: no detectable link** (one-sided p ≥ 0.21; partial r −0.15 to +0.12).
    - Per link, observed partial r above +0.09 to +0.38 is excluded at 95% (jointly over 18 links: +0.15 to +0.48).
    - **For error-free measures, true-score partial r up to +0.14 to +0.73 is not excluded**, with ≥ 0.5 for E002-P2 (0.73), E005-P2, E004-P2, E003-P1, E001-DB and E004-P3.
    - One-sided MDE 0.31–0.38 at n ≈ 60–80. The +0.09 bound (a0 → A0) is a pooled within-position average.
  - **Drill profile, like for like:**
    - Repeated position-drill metrics: 2 of 22 reach 0.5. Tracked 40: 28 of 28 (top speed 0.95–0.98, start phase 0.67–0.81).
    - Cross-drill composites: 10 of 68 reach 0.5 (family medians 0.29–0.39), so signatures rarely generalize across drills.
    - WR same-drill retests: median ICC 0.70 (single attempts, same session), so route signatures are largely drill-specific.
    - The OL wave drill is the one clear session-tempo artifact.
    - 40 year shares: about 0 for top-speed metrics, 0.11–0.20 for start-phase metrics.
  - 13 of 16 nulls pass the reliability criterion, and 4 of the 5 position-drill features tested have reliability ≥ 0.55. **Unreliability does not explain these nulls**; low power and imperfect game targets do.
  - Harness: LOCO (criterion 3) passes for 11 of 16 nulls on ±0.001–0.02 excesses, so it does not discriminate.
- **Decision:** **keep** the transfer map and drill-reliability figures as the writeup's rigor section.
  - The 5 position-drill signatures we could test (WR route-break speed retention and stop braking, OL mirror reversal acceleration, DB backpedal speed and plant-and-drive) show no detectable transfer to the game measures we built. DL position drills and the OL wave drill were not tested.
  - The nulls bound the tested constructs, not the scouting concepts.
  - The only signal is WR tracked top-end speed, which is one signal (E001/E006) and fragile. E007 is the confirmatory test.
  - Revisit drill links only with repeated attempts or opponent tracking. Add a per-class noise threshold to criterion 3.
- **Verification (2 skeptics, both 'weakened'):** every number reproduced. Accepted fixes:
  - beyond-B2 correction made primary;
  - per-link vs joint bounds;
  - all-round BH;
  - like-for-like drill counts and literal target labels;
  - E001 reliabilities recomputed;
  - the unreliability-explains-nulls claim removed;
  - E006-P1 fragility shown.
  No go/no-go verdict changed.

**Plain-English summary (150 words).** We re-tested all 18 round-1 links. The question: does a movement signature in the Combine tracking data tell you anything about the same movement, or an outcome, in regular-season games that the 40 time, other Combine results, draft slot and draft year do not? One link holds up under our criteria. For wide receivers, top speed fitted from the tracked 40 predicts extrapolated in-game top-speed capacity beyond the 40 time: partial r +0.48, or +0.38 without the rookie-only 2025 class, and three receivers carry about 40% of it. It re-tests the signal first seen in E001, has no outcome link yet, and is weaker on directly observed game top speed (+0.32). It also fails multiplicity across all ~90 round-1 tests. The other 16 links show nothing detectable. Each rules out observed correlations above +0.09 to +0.38. But with 60 to 80 players, true effects of 0.5 or more remain possible for six of them. These nulls bound the game measures we built, not the scouting concepts.

## 2026-10-14 — E007: Hidden speed (WR): tracked flying 20 beyond the 40 time → game top speed, cushion, yards per route
- **Hypothesis** (prereg `experiments/results/E007_prereg.json`, amended before any association; protocol rev 2): among Combine WRs, the tracked flying 20 adds beyond B2 (expected sign + for all three tests).
  - Feature `c40_v_20_40` = 20 / (t40 − t20) from the official lines, mean over QC-passing 40 attempts.
  - (P1) It predicts game top speed (q95 of REG play max speed).
  - (P2) Defenders play off such receivers (cushion over expected).
  - (P3) It turns into production (yards per route over expected).
- **Method:**
  - Code: `src/bdb27/targets_wr.py` (route table, cross-fitted context expectations, half-difference diagnostic; 8 tests) and `experiments/E007_hidden_speed.py`, which reproduces everything top to bottom in about 22 min.
  - Data: REG WR+TE routes (43,162).
  - Expectation model: HGB with default hyperparameters, GroupKFold(5) by game. Context: alignment, formation, motion, man/zone, down, distance, field position, quarter, score, season, defense team-season.
  - Player targets: per WR `eb_mean(cluster='game_id')`.
  - Harness: `linkage.evaluate`, 50×5 CV, 10,000 Freedman–Lane perms, seeds 0–2.
  - Prespecified robustness and 5 prelisted exploratory tests.
  - Post hoc blocks 8b and 8c (8c added after the skeptic review; in no BH family).
  - Outputs: `experiments/results/E007.json`, `E007_leaderboard.json`, `experiments/figures/E007_{added_variable,concept,leaderboard}.png`.
- **Result (with uncertainty):**
  - **Reliability.**
    - Feature: attempt ICC 0.92. Only 16% of its variance is left beyond B2, with reliability 0.75.
    - Targets (split-half): P1 0.92, P2 0.71, P3 0.87.
    - EB targets are over-shrunk (half-difference z2 0.6–0.9).
  - **P1** (n = 61):
    - Partial r +0.31 [0.00, +0.56]; bootstrap [−0.06, +0.59].
    - One-sided FL p 0.025 (seeds 0.023–0.026). Excess ΔR² +0.045 ± 0.029. LOCO 3/3.
    - Fails BH (q 0.075 across the primaries; 0.20 across the 32 prespecified tests).
    - Fails leave-3-out (+0.12, p 0.24). By class −0.09 / +0.54 / +0.41.
    - The data fit a modest true effect (about 0.3) as well as none: leave-3-out has about 1/3 power at r 0.3, and the class differences are not significant (p 0.14).
    - Post hoc: partly deep-route usage. With deep share added +0.22 (p 0.09); with offense EPA added +0.37.
  - **P2** (n = 60):
    - Partial r +0.01 [−0.30, +0.31], p 0.48.
    - Post hoc check against tracking: `cushion` is the receiver-to-aligned-defender distance at the snap, so it includes the receiver's own depth and split. Depth is a stable personal habit (split-half 0.95).
    - With alignment added to the context: target split-half 0.47, P2 +0.16 [−0.15, +0.45], p 0.16.
    - So the prereg null does not bound 'defenders respect speed'. Effects up to about +0.45 (about +0.75 disattenuated) remain possible.
  - **P3** (n = 55):
    - Partial r −0.31 [−0.58, +0.02], the wrong sign; one-sided p 0.97.
    - Positive effects above about +0.1 are ruled out. Upper bounds are +0.06 to +0.10 under the bootstrap, offense/QB adjustment, and whole-population yards per season (−0.14 [−0.36, +0.10], n 91).
    - The negative estimate is not robust: dropping the 3 most negative WRs gives −0.08, and part of it is offense quality.
  - **Exploratory (5), all null:** deep share +0.13; aDOT +0.10; targets per route −0.21; DB top speed −0.01; career snaps −0.07.
  - **S0 target variant:**
    - +0.56 [+0.30, +0.74], p 0.0002. It is robust to usage controls (+0.53).
    - But it is a re-observation, not evidence: the feature equals E006's post hoc x31→50 split (r 0.999 beyond B2), which was chosen after it absorbed the vmax → S0 signal on these same WRs.
  - **Leaderboard (descriptive):**
    - 91 WRs. Hidden speed has reliability 0.72.
    - Attempt 1 vs attempt 2: r 0.70, and 8 of 15 top-quintile WRs replicate.
    - 7 of the 15 top-quintile late or undrafted WRs have P(truly top quintile) ≤ 0.5.
    - Those WRs reach the field no more often than other late WRs (8/15 vs 27/49).
    - Tracked hidden speed is not corroborated by the official electronic splits (r −0.11 beyond the 40).
- **Decision:**
  - **Kill** hidden speed as an outcome-linked hero. There is no detectable performance link at n ≈ 55–91 for:
    - cushion as defined (null, though it is mostly alignment)
    - yards per route (no positive effect above about +0.1)
    - targets, aDOT, deep share, career snaps
    Effects below these bounds remain possible. A cushion target net of receiver alignment was not powered or reliable enough here.
  - **Park** the P1/S0 kinematic link as a descriptive transfer result for the §6 map, not a claim. P1 fails BH and leave-3-out. S0 cannot be confirmed on these data; that needs a Combine class or season not yet examined, and none is available before the deadline.
  - **Leaderboard:** descriptive only, 'speed beyond the official 40', with per-player uncertainty. Never 'undervalued'. Draft position reflects non-football factors.
  - The post hoc 'production tracks the start' idea is withdrawn (untested; the negative P3 sign is not robust).
- **Verification (2 skeptics: both 'weakened'):** the arithmetic reproduced exactly, and the prereg, leakage and context semantics checks were clean. Issues raised:
  1. S0 re-observes a known post hoc result.
  2. 'Fragile' overstated P1, given the low leave-3-out power.
  3. Bounds were too precise (P3 +0.02 → about +0.1).
  4. The BH family wrongly included a post hoc test (fixed).
  5. Harness criterion 2 ignores sign.
  6. The leaderboard flagged noisy, unshrunk residuals as 'undervalued' (fixed).
  7. Cushion includes the receiver's own alignment.
  8. P3 is partly offense quality, and leave-3-out had been run one way only.
  9. P1 is partly usage.
  10. Hidden speed is not corroborated by the official splits.
  All were addressed in post hoc block 8c, except annotating a named player's reported off-field matter (rejected: unverifiable here and inappropriate). **Corrected reading: no detectable performance link; P1 is consistent with a modest effect or none.** Kill stands.

## 2026-10-14 — Infrastructure round 2: shared modules speed / shrinkage / reliability+segments (built, reviewed, fixed)
- What: promoted probe E's sprint code to `bdb27.speed`; added one empirical-Bayes definition in `bdb27.shrinkage`; added `split_half_estimator`, `team_share`, `cross_drill_icc(zscore, center_by)`, `segments.combine_sessions` and `segments.onset_time`. Each module got an adversarial reviewer (evidence required) and a fix pass. 77 fast + 3 slow tests pass.
- Changes that affect earlier numbers (log only; no decision changes):
  - **A-S profile draws R = 20 → 1000.** At R = 20, Monte Carlo noise moved a WR's S0 by up to 1.0 between-player SD between seeds (rank changes up to 16 of 70); at R = 1000 the seed-to-seed r is 0.9995. WR S0 (E006-P1 target) correlates 0.984 with the R = 20 version. Split-half A0/S0: WR 0.90/0.94, DB 0.80/0.85, DL 0.94/0.70.
  - The Combine 40 peak speed `smax` now ignores the Savitzky–Golay edge frames (ICC 0.942 → 0.946). The flying 20 (`c40_v_20_40`, the E007 feature) and `vmax` are bit-identical to before.
  - `reliability.icc1` reliability is 0 when ICC ≤ 0 (it used to blow up near −1/(n−1)); E002–E005 negative values would become 0.
  - `team_share` is now a cross-half covariance ratio with a jackknife SE (the old version grew with plays per player).
  - `shrinkage.eb_mean` / `eb_rate` take `cluster="game_id"`: plays within a game are correlated, roughly doubling the true sampling variance for DL get-off; without clustering the shrinkage was too weak.
- Open: OL pressure-over-expected variance still looks understated with clustering (half-difference ratio 1.5 ± 0.2, cause unknown), and HITCH decel is over-shrunk (the conservative direction). Run the half-difference diagnostic on new rate targets.
- E007 prereg amended (before any P1–P3 association): the flying 20 is measured from the official start line, because the onset-relative 40-yd mark falls past the end of tracking in 28% of attempts. P2/P3 shrinkage uses `cluster="game_id"`.

## 2026-10-14 — GATE 1: probe round 1 (E002–E006) summary and decision
- Round: 5 probes, 14 preregistered primary tests, ~76 exploratory tests; each probe built by one agent and attacked by two skeptics (reproduction/statistics and football/semantics). Numbers reproduced exactly in all five.
- BH across the 14 primaries: **only E006-P1 survives** (WR fitted 40 vmax → game top-speed capacity S0, q = 0.021). The next best is E006-P2, q = 0.15, with the wrong sign. Everything else has q ≥ 0.70.
- What we learned:
  - **Position-drill movement signatures (route breaks, DL burst, OL reversals, DB transitions) show no detectable transfer beyond Combine times + draft pick.** Partial r CIs are about ±0.3, so large effects are ruled out and modest ones are not.
  - Many drill metrics measure drill tempo or Combine session/year, not the athlete (wave tempo drifts 2.9 → 2.2 yd/s between 2023 and 2025; 2025 COMEBACK is a different movement).
  - **Tracked top-end (flying) speed is the one Combine-tracking signal that adds beyond the stopwatch**, for WRs only so far. It is kinematic, with no outcome link yet.
- Harness weaknesses found by the skeptics, to fix before round 2:
  - the ΔR² permutation test is sign-agnostic (E006-P2 'passed' p with the wrong sign);
  - LOCO > 0 counts ±1e-5 ridge noise as positive;
  - perm p swings with the CV seed (5-repeat subset);
  - low power with ~19 B2 columns at n ≈ 65;
  - no draft-year term for Combine session artifacts;
  - pooled ΔR² is normalized by between-position variance.
- Decision (PLAN §6):
  1. **Hero candidate: "hidden top speed"**, i.e. tracked flying speed beyond the 40 time. Deep build: robustness (session/year, influence, flying-split definition) and a preregistered search for a performance consequence.
  2. **Second track: the drill reliability + transfer map** (all probes, with reliability-corrected bounds). It becomes the writeup's rigor backbone and the 'so what' for clubs.
  3. Upgrade the harness first (sign-aware Freedman–Lane partial test, LOCO vs the null, year terms, power reporting).

## 2026-10-14 — E006: Sprint mechanics profile (tracked 40-yd velocity curve vs in-game acceleration-speed profile)
- Hypothesis: preregistered in `experiments/results/E006_prereg.json` before any Combine x game association. The shape of the tracked 40 adds beyond B2 (40 time, 10-yd split, other Combine times, draft capital):
  - (P1) fitted vmax -> WR in-game maximal speed S0
  - (P2) fitted a0 = vmax/tau (= F0/m) -> in-game maximal acceleration A0 (WR+TE+DB+DL pooled)
  - (P3) vmax -> WR/TE yards after catch over expected
- Method:
  - **Combine fit:** per 40 attempt, a mono-exponential v(t) = vmax(1 - e^-(t-t0)/tau) on Savitzky-Golay speed, fitted from peak dv/dt to peak speed. dv/dt rises for the first 0.3-0.8 s (sensor/filter onset), which a mono-exponential cannot fit. Also a Samozino 2016 F-V profile with air resistance (F0, V0, Pmax, Sfv).
  - **Combine QC:** path 38-43 yd, fit RMSE <= 0.15 yd/s (drops two-phase artifact traces), tau 0.3-3 s. 763/793 attempts pass; player value = mean of attempts.
  - **Game profile:** post-snap REG frames, SG a_tan. In-situ A-S profile (Morin 2021) made exposure-invariant: 20 draws of exactly 50 plays; per draw, q95 of a_tan in 0.5 yd/s bins from 3 to 10.5 yd/s, then a linear envelope gives A0 (intercept) and S0 (x-intercept).
  - **YAC outcome:** YACOE per catch winsorized at +-10 yd, DerSimonian-Laird EB per position, >= 10 catches.
  - **Evaluation:** `linkage.evaluate`, 50x5-fold, 2,000 Freedman-Lane perms, seeds 0-2, plus a run without the 2025 class. Exploratory: 16 planned + 4 post-hoc tests (20x5 folds, 500 perms).
  - **Files:** `experiments/E006_sprint_profile.py`, `src/bdb27/probes/e_speed_profile.py`, `experiments/results/E006.json`, `experiments/figures/E006_{forty_fits,game_as_profiles,primary_added_variable}.png`.
- Reliability:
  - **Combine** (icc1 / reliability as used): vmax 0.84-0.96 / 0.90-0.98; tau 0.53-0.67 / 0.66-0.78; a0 WR 0.43/0.57, others 0.61-0.69 / 0.73-0.81; t0 and the split residual are unreliable (<= 0.51). B2 explains only 21-41% of a0's within-position variance.
  - **Game** (split-half): A0 WR 0.88, TE 0.87, DB 0.80, DL 0.93, OL 0.42. S0 WR 0.92, DB 0.84 (TE 0.20, OL 0.24; DL/OL S0 is extrapolated). |corr with log plays| <= 0.26.
  - **YACOE:** split-half WR 0.55, TE 0.68; exposure-tied (corr with log catches +0.27 WR, +0.51 TE).
- Result (with uncertainty):
  - **P1 passes everything** (WR, n = 61): dR2 vs B2 +0.074 +- 0.033, excess +0.065 (seeds: +0.065/+0.071/+0.061), perm p 0.0015-0.002, partial r +0.38, LOCO 3/3 (2024 only +0.003), BH q 0.0045. Without 2025: excess +0.086 +- 0.066, p 0.006.
    - Not independent of E001: S0 correlates 0.85 with E001's target.
    - Post-hoc, fitted vmax adds little beyond E001's raw 40 peak speed (excess +0.025 +- 0.019, p 0.04).
    - Nothing for DB (excess -0.013) or TE (n = 20).
  - **P2 fails** (n = 218): excess +0.002 +- 0.002 (lower bound > 0 at 1/3 seeds), p 0.02-0.03, but the partial r is -0.14, the wrong direction. The dR2 permutation is sign-agnostic. Class partials -0.05/0.00/-0.16; without 2025, p 0.21. Within position: WR +0.11, DB -0.07, DL -0.06, OL -0.14, all p > 0.12. Stricter QC (post-hoc) keeps the wrong sign.
  - **P3 null** (n = 63): excess -0.010 +- 0.058, p 0.62; without 2025, p 0.50.
- Decision:
  - **Keep (narrow)** the WR "top speed beyond the 40 time" link. It replicates E001 on an independently estimated game target, and the scout story is: same 40 time, higher tracked top speed, faster on Sundays. Use it as a component of a WR game-speed metric at Gate 1. It still has no outcome link.
  - **Kill** acceleration-shape transfer (a0/tau/F0/Pmax -> game A0, all positions). Revisit only if a first-step probe (B/E003) finds start-phase signal.
  - **Kill** the YACOE outcome (unreliable, exposure-tied, null).
  - **Harness note:** add a sign check to `evaluate`, and a within-position dR2 normalization for pooled populations.
- Verification (2 skeptics: both 'weakened'): every number reproduces exactly from raw data and the prereg is valid. But: (1) P1 survives BH across the round's 14 primaries (q = 0.021) and **not** across all 88 tests in the round (q ≈ 0.08); (2) it is a near re-test of E001, which was seen before the prereg (S0 r = 0.86 with E001's target), so it is not independent confirmation; (3) it rests on few players: dropping the 3 most supportive WRs fails the go criteria, and LOCO 2024 hinges on one receiver; (4) the signal vanishes once a raw tracked flying split (time from x = 31 to x = 50 yd) is added to B2, so the right framing is **tracked flying speed**, not curve shape; (5) no performance consequence yet (deep-route separation and YACOE are null), and S0 extrapolates about 1 yd/s beyond observed game speeds. Keep, as a narrow candidate that needs a preregistered outcome link.

## 2026-10-14 — E005: DB backpedal → transition (probe D)
- Hypothesis (preregistered in `experiments/results/E005_prereg.json` before any feature-target association): DB drill movement transfers to the same movement in coverage beyond B2. **P1:** Combine backpedal speed → game backpedal speed. **P2:** Combine plant-and-reverse explosiveness → game post-throw transition explosiveness ("breaking on the ball"). Expected sign + for both.
- Method:
  - **Combine side** (122 DBs, `src/bdb27/probes/d_db_transition.py`):
    - Canonicalized drill attempts; breaks via `segments.breaks`; selection rules fixed after visual QC.
    - TA1 = the plant at the largest turn of 90° or more; TA2 = the stop-and-reverse; W = 3 reversals; BOX = 2 corners; BP90 = the 90° break.
    - `cmb_backpedal_speed` = peak speed in the opening backpedal leg of BP90/TA2/W/BOX/BP45, z-scored within drill and averaged over drills.
    - `cmb_transition_dv` = |velocity change| from 0.5 s before to 0.5 s after the plant's speed minimum in TA1/TA2/W/BOX. It is residualized on entry speed and turn angle within each drill-cut, z-scored, and averaged over drills.
  - **Game side** (92 DBs; REG pass plays aligned at CB/SLOT_CB/HIGH_SAFETY/BOX_SAFETY; snap-aligned):
    - `gm_backpedal_speed` = per-play peak speed while moving downfield and facing the LOS (orientation within 60° of 270) in the first 2.5 s before the throw. It is centered on its alignment × coverage-assignment cell, then EB-shrunk; players need at least 20 plays.
    - `gm_ballair_dv` = dv_10 at breaks out of the backpedal after `pass_forward`. It is residualized on turn angle (quadratic), entry speed, alignment and man/zone, then EB-shrunk; players need at least 10 events.
  - Harness: `linkage.evaluate` vs B2, 50×5-fold, 2,000 Freedman–Lane permutations, seeds 0/1/2, plus a rerun without 2025.
  - 7 labeled exploratory tests.
  - Script: `experiments/E005_db_backpedal_transition.py`. Results: `experiments/results/E005.json`. Figures: `experiments/figures/E005_qc_combine.png`, `E005_qc_game.png`, `E005_primary_added_variable.png`.
- Result (with uncertainty):
  - **Reliability (both pass ≥ 0.5):** Combine 0.55 / 0.55 (cross-drill, drill means); game 0.85 / 0.57 (split-half by game). Exposure: corr(trait, log plays) is +0.05 and −0.13.
  - **P1:** n = 82, raw r +0.08, partial r +0.03 (≈95% CI −0.22 to +0.28). Excess ΔR² vs B2 is +0.0007 ± 0.008, p = 0.25 (seeds 1 and 2: p 0.55 and 0.24). Without 2025: excess −0.004, p = 0.11.
  - **P2:** n = 69, raw r +0.07, partial r +0.07 (−0.21 to +0.34). Excess +0.005 ± 0.009, p = 0.55 (seeds 1 and 2: p 0.46 and 0.42). Without 2025: excess +0.04 ± 0.12, p = 0.65.
  - BH q = 0.49 and 0.55; 0/3 seeds pass. Corrected for reliability, the true-score r is below about 0.43 (P1) and 0.54 (P2) at the 95% bound.
  - **Exploratory:** all 7 null (drive_05, peak re-acceleration, pre-throw dv, man-only and zone-only backpedal, TA1-only and W-only dv). The largest is W-only dv with raw r +0.24, but excess −0.001 and p 0.73.
  - **Side findings:**
    - Combine braking (peak decel, speed lost in the 0.5 s before the plant) is drill tempo: with entry speed adjusted, reliability is about 0.
    - In-game braking, retention and min speed are not stable traits (split-half about 0).
    - Pre-throw transitions are not stable (receiver-driven). Post-throw drive (0.65 to 0.71) and backpedal speed (0.85) are.
    - The BP45 drill's two attempts agree on nothing (ICC about 0).
    - Traditional Combine times do not predict these game traits either (B1 CV R² ≤ B0).
- Decision: **kill** as a hero candidate. It fails go criteria 1, 2 and 6 with reliable measures on both sides, so the null is real at this n, not noise. Keep the reliability results ("Combine braking is tempo; game backpedal speed and post-throw drive are stable traits that neither drills nor stopwatch predict") for the Gate-2 "drill reliability + transfer map" fallback. Revisit only with Combine orientation (hip-flip timing) or opponent/ball tracking (a coverage outcome); neither is in the data. Harness note: the LOCO criterion counts ±1e-5 ridge ΔR² as "positive" (P1 was 3/3 on a null), so it needs a minimum-effect threshold.
- Verification (2 skeptics: both 'weakened'): exact reproduction, valid prereg, correct orientation/direction semantics. But: (1) the P1 game 'peak backpedal speed' is mostly retreat speed at the hip flip, the throw, or the 2.5 s cutoff, not backpedal speed; (2) the P2 target is generic post-throw redirection (often deep safeties pursuing short completions), not 'breaking on the ball'; (3) low power; (4) game backpedal reliability (0.85) is partly a team/scheme effect, so odd/even games cannot separate player from team. Four cleaner variants (CB-only, non-truncated backpedal, clean plant-and-drive) were null. **Corrected reading: no detectable link at this n; 'tells you essentially nothing' overclaims.** Kill stands.

## 2026-10-14 — E004: OL lateral reversals in Combine drills → pass sets and pressure allowed (probe C)
- **Hypothesis:** OL who swing their momentum harder through lateral reversals in the Combine pass-pro mirror drill accelerate into their pass sets faster and allow less pressure than expected, beyond B2. Preregistered in `experiments/results/E004_prereg.json` after the reliability screen and before any feature–target association.
- **Method:**
  - **Combine:** mirror drill (both naming variants merged), wave drill (one attempt each) and PASS_RUSH_DROP (two attempts).
    - Reversals are `segments.breaks` (min_turn 120°, entry ≥ 1 yd/s, shuffle phase only, run-out excluded).
    - `rev_accel` = |v_exit − v_entry| / Δt between the speed peaks around each reversal.
    - Centered by Combine session × reversal index. There are two sessions per year, and wave tempo drifts from 2.9 to 2.2 yd/s between 2023 and 2025.
  - **Game:** 46,352 REG pass-blocking snaps (T/G/C, ≥ 1 rusher).
    - Pass-set kinematics come from 28,424 non-play-action dropbacks of ≥ 2 s. `set_accel` = peak a_tan in 0–0.6 s after the snap, position-centered, averaged per player.
    - Outcomes are pressure_allowed and peak_pressure_probability_allowed relative to a play-level expectation (dropback spline, T/G/C, 2+ rushers, play action, season; AUC 0.78), averaged per player and EB-shrunk (normal-normal, method of moments).
  - **Tests:** 3 primary (mirror rev_accel → set_accel, → PPP over expected, → pressure over expected), each at 50×5-fold, 2,000 Freedman–Lane perms, seeds 0–2, plus no-2025 reruns. 12 exploratory tests.
  - **Files:** script `experiments/E004_ol_lateral_to_pass_pro.py`, module `src/bdb27/probes/c_ol_lateral.py`, results `experiments/results/E004.json` and `E004_reliability.json`, QC figures `experiments/figures/E004_qc_*.png`, result figure `E004_result.png`. A rerun from scratch reproduced every number exactly.
- **Result (with uncertainty):**
  - **Reliability screen:**
    - Mirror rev_accel: within-attempt 0.73. This is single-attempt, so the partner is confounded.
    - Wave drill: within-attempt ≈ 0.02. Mirror vs wave agreement: r = 0.25 (n = 105).
    - Drop-drill kick metrics: ICC ≈ 0 across two attempts (killed).
    - Game set_accel: split-half 0.68, teammate r 0.29.
    - Set speed, depth and getoff: 0.78–0.81, but teammate r 0.50–0.58, so mostly scheme.
    - Pressure over expected: 0.42. PPP over expected: 0.62. Sacks: 0.09.
  - **P1 → set_accel** (n = 67): partial r +0.04; excess ΔR² +0.0002 / +0.0012 / +0.0004 (SE 0.002–0.005); p 0.15 / 0.56 / 0.61; LOCO 0/3.
  - **P2 → PPP over expected** (n = 70): partial r −0.10; excess +0.002 / +0.001 / +0.003 (SE ≈ 0.01); p 0.76 / 0.49 / 0.016; LOCO 2/3. Without 2025 (n = 50): partial r −0.00, p 0.05–0.14.
  - **P3 → pressure over expected** (n = 70): partial r −0.00; excess −0.002 / +0.002 / +0.003; p 0.35 / 0.48 / 0.03; LOCO 1/3. Without 2025: p 0.38–0.46.
  - **Multiple testing:** BH q 0.45–0.76. The seed-2 p < 0.05 results have partial r ≈ 0 and excess < SE, so they are seed noise in the 5-repeat permutation statistic.
  - **Baselines:** B2 CV R² is negative for all targets.
  - **Exploratory (12, none pass):**
    - Mirror+wave composite → PPP over expected: partial r −0.25, LOCO 3/3, but excess −0.003 and p 0.31.
    - Game set_accel → pressure: partial r +0.12 to +0.21, the wrong sign.
- **Decision: kill.** It is a clean null. Combine OL drill reversal signatures add nothing beyond Combine times and draft pick for pass-set kinematics or pressure allowed. The drill signatures are also barely repeatable beyond one rep.
  - **Revisit if** repeated mirror attempts or partner identity become available.
  - **Reusable pieces:** the play-level pressure-over-expected targets (PPP over expected, reliability 0.62) and the session-centering finding for Combine drills.
- Verification (2 skeptics: both 'weakened'): exact reproduction, valid prereg, correct pass-block semantics. But: (1) low power (B2 CV R² negative; ridge shrinks a single added feature to ~0); (2) only the mirror metric was tested against outcomes, so 'OL drill signatures add nothing' over-generalizes. The wave-drill reversal metric, screened out on reliability, was post hoc the strongest association (expected sign, consistent across classes); (3) the P1 target (peak accel 0–0.6 s) is scheme/reaction, not the Combine construct; (4) the drop drill's ICC ≈ 0 is because its two attempts are kicks to opposite sides, not low repeatability. **Corrected reading: the mirror metric shows no detectable link; the other drills are untested.** Kill stands for this probe.

## 2026-10-14 — E003: DL first-step burst (Combine 40 start + DL drills → NFL burst, get-off, quick pressure)
- Hypothesis (preregistered in `experiments/results/E003_prereg.json` before any Combine–game association was computed): the 40-yd dash contains a stable burst trait the stopwatch misses, `c40_d10`. It is the yards covered in the first 1.0 s after motion onset, where onset is the moment speed reaches 1 yd/s. Prediction: it adds beyond B2 for three DL targets on NFL pass-rush snaps: (P1) the same onset-aligned burst, (P2) NGS get-off, (P3) quick-pressure rate.
- Method: script `experiments/E003_dl_burst.py`, module `src/bdb27/probes/b_dl_burst.py`.
  - Kinematics: Savitzky–Golay; onset interpolated, and an attempt whose first frame is already ≥ 1 yd/s counts as unobserved.
  - Pass-rush snap = `blitzing == True` & `sack` non-null & not nullified (verified semantics: pressure = `time_to_pressure` non-null; `quick_pressure` = pressure ≤ 2.4 s; `player_get_off` ≈ seconds from snap to 1.3 yd upfield, r = 0.90 with tracking).
  - Game values: per-play values centered by EDGE/INTERIOR alignment, then empirical-Bayes player means. Means use normal–normal shrinkage; rates use beta-binomial around each player's alignment-mix expectation.
  - Population: ≥ 30 pass-rush snaps → n = 67 DL with a tracked 40.
  - Linkage: `linkage.evaluate`, 50×5 CV, 2,000 Freedman–Lane permutations, seeds 0/1/2, plus a rerun without 2025.
  - Reliability screen first. QC plots: `experiments/figures/E003_qc_*.png`.
- Result (with uncertainty):
  - **Reliability.** Combine 40 burst: attempt ICC 0.61 (reliability 0.73). It correlates r = −0.38 with the 10-yd split, and its part not explained by B2 still has reliability 0.62. DL position drills do **not** measure burst reliably: cross-drill reliability 0.42 for 4 burst drills, only 0.21 beyond B2; FOUR_BAG and BODY_CONTROL do not load on burst. Game traits are reliable (split-half): burst 0.83, get-off 0.82, quick-pressure rate 0.72, pressure rate 0.57.
  - **Primary tests: all null on all 3 seeds.**

    | Test | Partial r | ΔR² vs B2 | Excess | Perm p | LOCO |
    |---|---|---|---|---|---|
    | P1 → game burst | +0.10 (95% CI −0.19…+0.37) | −0.005 ± 0.016 | −0.001 | 0.25 | 2/3 |
    | P2 → get-off | +0.07 (wrong sign) | — | −0.009 | 0.46 | 1/3 |
    | P3 → quick pressure | −0.00 | — | −0.007 | 0.97 | 0/3 |

    The other seeds give p 0.20–0.91. Without 2025 (n = 50), all three are null (p 0.82–0.92).
  - P1's raw r = +0.25 is fully absorbed by B2 (weight plus 40/10-yd times; B2 CV R² = 0.17).
  - Power: minimum detectable partial r is 0.39.
  - **Exploratory (25 tests):** 1 at p < 0.05 (INTERIOR-majority stratum, P1 target: n = 37, partial r +0.41, p = 0.010, LOCO 1/3, BH q = 0.25), as expected by chance. The negative control (Combine burst → reaction time) is null, as it should be. A post-hoc matched 0.5 s window is also null (p = 0.16).
  - **Game-side only:** game burst tracks get-off (r = −0.52) and pressure rate (+0.47) across 85 DL. Burst matters in games; the Combine just doesn't predict it beyond B2.
- Decision: **kill** as a hero candidate.
  - Keep two findings for the reliability + transfer map (PLAN §6): (1) the 40's first-second burst is a reliable trait the stopwatch misses but does not transfer for DL; (2) DL position drills are unreliable burst measures.
  - Revisit if pooled with probe E (40 speed-curve shape) across positions, or if more DL classes become available.
- Verification (skeptics: 'upheld', 'weakened'): bit-for-bit reproduction, valid prereg, no leakage. The 40-yd burst feature carries a Combine-year (tracking-session) artifact; with year removed its beyond-B2 reliability drops 0.62 → 0.51. The P1 game window runs into blocker contact and stunts, so it is not a pure acceleration measure. The null holds with draft-year dummies, season × alignment centering, and rookie-season-only targets. Kill stands.

## 2026-10-14 — E002: WR route-break mechanics (Combine route drills -> game OUT/IN/HITCH breaks, separation)
- Hypothesis: speed a WR carries through 90-degree breaks in Combine route drills transfers to OUT/IN breaks in games and to separation at the throw, beyond Combine times + draft pick (B2). Secondary: braking into Combine stop routes transfers to HITCH braking. Preregistered (3 primary tests) in `experiments/results/E002_prereg.json` before any Combine-vs-game association.
- Method: `src/bdb27/probes/a_route_breaks.py`, `experiments/E002_route_break_mechanics.py`. Combine: canonicalized attempts, one break per attempt. STOP events (curl/comeback/slot strike): stem peak -> first near-minimum speed. TURN events (speed out/dagger/slot sail/post-corner L): sharpest heading change in a per-drill path-distance window, break = slowest frame within +-half_window. Windows set from QC plots; slant/corner/post-corner R excluded (smooth arcs). Composites = drill-z mean over >= 2 drills. Game: REG WR/TE routes, snap-aligned, event must start before the throw (cap 4 s); HITCH = stop, OUT/IN = turn; per-play values centered by route, player mean EB-shrunk (>= 10 events). Outcome: separation over expected (cross-fitted GBM on route, coverage, alignment, formation, dropback, down/distance/field; cushion excluded), HITCH/OUT/IN plays, EB-shrunk. `linkage.evaluate`, 50x5-fold, 2,000 Freedman-Lane perms, seeds 0-2.
- Result (with uncertainty): reliability: Combine turn retention composite 0.56 (cross-drill ICC 0.27; drill-specific retest 0.62), stop braking 0.44; game OUT/IN retention 0.77, HITCH braking 0.79; SOE 0.15 all routes / 0.41 HITCH-OUT-IN. **P1 retention -> game OUT/IN retention: n=64, partial r -0.02 [-0.31, 0.27], excess dR2 -0.013 +- 0.024, p 0.78 (seeds 0.78/0.75/0.56). P2 retention -> SOE: n=67, partial r 0.00, excess -0.015 +- 0.024, p 0.53 (without 2025: n=51, p 0.31). P3 stop braking -> HITCH braking: n=62, partial r -0.02, excess +0.005 +- 0.023, p 0.61 (seeds 0.61/0.75/0.18).** BH q = 0.78 for all. 12 exploratory tests (top-speed control, WR+TE pooled, min speed, other SOE variants, game trait vs SOE, a post-hoc positive control, sideways acceleration): all null. Post-hoc sideways acceleration has partial r +0.24, p 0.25. Diagnostics: Combine drill stem speed is uncorrelated with the 40 (r -0.14), so drills are run at submaximal speed; game OUT/IN retention correlates 0.89 with turn radius (break shape / route style). Bug fixed after the primaries: EB prior estimated over 1-3-event players collapsed tau2 (HITCH decel 0.008 vs 0.14); rerun, run 1 preserved, also null. Lesson for LESSONS.md: estimate EB priors on the players actually used and check mean shrinkage weight against split-half reliability.
- Decision: **kill** as a hero candidate (fails criteria 1, 2 on all seeds; P3 also 5). Clean, well-measured null: Combine route-drill break mechanics are drill-specific and do not transfer. Park for the section 6 drill reliability + transfer map. Revisit only with a new prereg for sideways acceleration or a game break trait matched on turn angle and depth.
- Verification (2 skeptics: both 'weakened'): every number reproduces exactly and the prereg is valid (written before any linkage run). But the null is weaker than stated: (1) the P1 game target mostly measures how often a WR's OUT/IN event is a turnaround/sit-down, not speed carried through a 90° cut; (2) separation at `pass_forward` mostly precedes the break on timing routes; (3) the P3 Combine feature is confounded with draft class (the 2025 COMEBACK drill is a different movement; 37% of composite variance is between classes); (4) at n ≈ 64 with ~19 B2 design columns the harness has low power. **Corrected reading: no detectable link; true partial r ≳ 0.4 ruled out; small-to-moderate effects are not.** Eight cleaner post-hoc variants by the skeptic were also null. Kill stands.

## 2026-10-14 — E001 re-run after the infrastructure review (supersedes the entry below)
- Changes: target is now the **q95 of play max speed** (the top-5 mean grew with snap count, corr ≈ 0.5–0.78 with log plays, confounding draft capital). Harness now uses a **Freedman–Lane** permutation null, fold R² normalized by full-sample Var(y), the **excess** ΔR² over the null mean, and a leave-one-class-out criterion (see PLAN.md §4).
- Result (with uncertainty): pooled n = 285, raw r = 0.87, but **excess ΔR² vs B2 = +0.000** (p = 0.07) — the earlier pooled "pass" was an exposure artifact. **WR (n = 64): ΔR² vs B2 = +0.070 ± 0.024, excess over null +0.050, perm p = 0.0065, partial r = +0.33, LOCO ΔR² > 0 in 3/3 classes; stable across 5 CV seeds (excess +0.050…+0.061, p 0.004–0.012).** DL/OL/DB: nothing beyond B2. TE n = 20: too small (partial r undefined).
- Decision: **keep**. The WR "peak speed beyond the 40 time" signal survived a stricter harness and a cleaner target, so it becomes the anchor for probe E (speed-curve shape). Still one feature, one position, n = 64: not a claim yet.

## 2026-10-14 — E001: Combine 40-yd peak speed → game top speed (pipeline sanity link)
- Hypothesis: peak speed in the tracked 40 predicts regular-season game top speed (must hold, or the pipeline is broken). Beyond B2, which already contains the 40 time, expect ~0 gain.
- Method: `combine_top_speed` = max `s` across 40 attempts; `game_top_speed` = mean of a player's 5 fastest REG plays (play max `s`; plays > 12 yd/s dropped as glitches; ≥ 50 REG plays). `linkage.evaluate`: 50×5-fold ridge, 2,000 within-position permutations. Script: `experiments/E001_combine_top_speed_to_game_top_speed.py`, results: `experiments/results/E001.json`.
- Result (with uncertainty): corr(forty, combine_top_speed) = −0.978 (tracking matches the stopwatch). Pooled n = 285, raw r = 0.88, ΔR² vs B0 = +0.053 ± 0.014. Within position: WR raw r = 0.53, DL 0.66; OL/DB/TE weak. **Unexpected: WR ΔR² vs B2 = +0.055 ± 0.053, perm p = 0.002, partial r = +0.42, same sign 3/3 classes** — peak speed adds beyond 40 time + 10-yd split + NGS athleticism for WRs. Lower bound barely > 0; n = 64.
- Decision: **keep** (pipeline validated). WR result → a lead for probe E (speed-curve shape beats the stopwatch); re-test there with the full shape features, not as a standalone claim. TE n = 20 is too small alone → pool WR+TE in future. Attrition 418 → 285 (players with tracked 40s → ≥ 50 tracked REG plays) needs a breakdown.
