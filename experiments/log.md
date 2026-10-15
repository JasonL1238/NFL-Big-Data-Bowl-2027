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
