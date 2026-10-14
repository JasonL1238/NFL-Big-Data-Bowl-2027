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
