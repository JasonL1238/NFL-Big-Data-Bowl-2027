# PLAN.md — how we iterate on this project

`AGENTS.md` covers the what (competition, approach, conventions). This file covers the **how**: a system built to make testing and pivoting cheap, plus the decision gates that force commitment. Update it when a gate is passed or the direction changes.

---

## 0. Data facts that shape everything (verified 2026-10-14)

- **Game tracking contains only our 510 players.** There are no opponents, no ball and no QB; a play has 1–6 tracked players (mean 1.6). We **cannot compute separation, leverage, or pressure ourselves.** Game-side signals are (a) the player's **own kinematics** and (b) **outcomes already in `player_play`**.
- **There is no `play_direction` column in game tracking.** Join it from `player_play.play_direction` on (game_id, play_id).
- **127 of 510 players are undrafted** (draft_round = NA). That is useful: draft pick is not a complete control. Fix the "510 drafted" wording in `AGENTS.md`.
- **`games.csv` includes PRE and POST games.** The prompt says regular season, so filter `season_type == "REG"` for outcomes. Preseason games may be usable for feature *estimation*; decide that explicitly and log the decision.
- **`player_play` = 314,197 rows** (unique on game/play/player); 497 of 510 players appear in it.
- **Outcome coverage by Combine position:**

| Pos | n players | game plays | Best outcome fields | Matching Combine drills (~n) |
|---|---|---|---|---|
| WR | 106 | 43k (all routes) | `route_ran`, `separation_at_pass_forward` (38k), targets, rec yds, EPA | 15 named route drills (~100 each) incl. SLANT, CURL, COMEBACK, GO, POST_CORNER, SPEED_OUT |
| TE | 38 | 10k | same as WR (8.7k sep) | FLAT, HOOK, IN, WHEEL, CORNER routes (~40) |
| DL | 115 | 68k | `player_get_off` (43k), `time_to_pressure`, sacks, TFL | FRONT_START_WAVE, PASS_RUSH, RUN_AND_CLUB, FOUR_BAG (~112) |
| OL | 120 | 108k | `pressure_allowed` (60k), `time_to_pressure_allowed`, sack_allowed | FIVE_YARD_WAVE slide/shuffle, PASS_PRO_MIRROR, PASS_RUSH_DROP, pulls (~120) |
| DB | 118 | 85k | `coverage_assignment` only (no target/separation allowed) | BACK_PEDAL 45°/90°, W_DRILL, BOX, TERYL_AUSTIN (~120) |

- **Positions are not equally strong.** WR (route names match on both sides) and DL/OL (rich outcome fields) look strongest. DB outcomes are thin, so treat DB as kinematics-to-kinematics only.

---

## 1. Philosophy: a funnel with fixed kill criteria

```
many cheap probes (1–3 days each)  →  2 deep builds  →  1 hero metric  →  polish
```

- **Pivoting is cheap only if the infrastructure is shared.** Every idea is *one feature function + one call to the evaluation harness*. If a probe needs new plumbing, that plumbing goes into `src/bdb27/` so the next probe reuses it.
- **Decide kill criteria before looking at results** (section 4). Log every test, including the dead ones; this is how we keep the garden of forking paths honest.
- **Ship something every week:** one writeup-ready figure plus a paragraph, even for killed ideas. Pivoting then never costs us the writeup.
- **Keep a parking lot.** Killed or deferred ideas go into `experiments/log.md` with "revisit if …".

---

## 2. Infrastructure (week 1 — build once, reuse forever)

Build this in order. Each step has a verification.

1. **Environment.** Add `pyproject.toml` and install polars, duckdb, numpy, scipy, scikit-learn, statsmodels, matplotlib and pytest into `.venv`.
   → verify: `python -c "import bdb27"` works (not `bdb`: stdlib name clash).
2. **`src/bdb27/io.py`:** convert the CSVs once to Parquet in `data/processed/` with explicit schemas (`null_values="NA"`), and filter games to REG/PRE/POST as needed.
   → verify: row counts match section 0, and `player_play` = 314,197.
3. **`src/bdb27/standardize.py`:** apply the play-direction transform from `AGENTS.md` to game tracking (joined from `player_play`). Combine drills get their own canonicalization: rotate each attempt so the drill's main axis is +x, then mirror left/right drills.
   → verify: unit tests on synthetic trajectories, and a plot of 20 random plays/drills.
4. **`src/bdb27/segments.py`:** cut trajectories into analysis windows:
   - game: snap → +N s; snap → pass_forward; detect the route break (peak curvature / heading change)
   - combine: per drill attempt; detect the cut (heading change); detect the start (first frame where s > threshold)
   → verify: overlay detected breaks on 30 random routes and eyeball them.
5. **`src/bdb27/features.py`:** a registry of feature functions `f(segment_df) -> scalar`. Start with:
   - peak decel into break, speed retained through break, turn radius at speed
   - time to 90% top speed, first-step (0–0.5 s) accel, jerk/smoothness, lateral speed
   → verify: features are stable traits of the player. **This doubles as a reliability screen: drop features with reliability < ~0.5.** Most position drills have **one attempt per player**, so reliability (`src/bdb27/reliability.py`) comes in three forms:
     - `icc1`: across repeated attempts (40, shuttle, 3-cone, some DB drills)
     - `cross_drill_icc`: the same construct in different drills (e.g. a WR's break decel in CURL vs COMEBACK vs SPEED_OUT) or different cuts within an attempt, after centering by drill/cut
     - `split_half`: game traits, odd vs even games, Spearman–Brown corrected

   **Findings so far (2026-10-14):**
   - 40-yd top speed ICC = 0.98, peak accel 0.51, time to 90% speed 0.37.
   - Shuttle/3-cone cut kinematics are weak traits (best: shuttle entry speed 0.34, peak decel 0.27; others ≈ 0), and some attempts are incomplete → QC by path length.
   - Break detection catches sharp cuts. Rounded breaks (slant, post-corner) need a lower `min_turn` or a longer `half_window`. Game routes need `min_entry_speed ≈ 3` to drop heading wobble at the snap.
6. **Feature store.** Write `data/processed/combine_features.parquet` and `game_traits.parquet`, one row per nfl_id (wide), and append columns rather than rewriting.
7. **`src/bdb27/linkage.py`, the evaluation harness:** one function, used for every experiment:
   ```
   evaluate(feature, target, position, controls=[...]) ->
       ΔR² vs baselines (repeated K-fold CV, mean ± SE),
       partial r, permutation p, per-draft-class sign/size,
       leave-one-class-out ΔR², n
   ```
   → verify: on pure noise, ΔR² ≈ 0 and p ≈ uniform; on a planted signal, it recovers it.
8. **`experiments/`:** one script per test (`E001_forty_to_game_speed.py`, …) that calls the harness, writes `experiments/results/E001.json` and adds a log entry. Results JSONs are small, so commit them.

**Sanity link (E001, built as part of the infrastructure):** Combine 40-yd top speed → game max speed. It **must** work, or the pipeline is broken.

---

## 3. Candidate links (weeks 1–2 probes, ranked by prior)

| ID | Combine signature | Game trait (self-kinematics) | Game outcome (`player_play`) | Why promising |
|---|---|---|---|---|
| **A** | WR/TE route-drill break: decel, speed retention, turn radius, per route type | Same route type in games (SLANT↔SLANT, CURL/COMEBACK↔HITCH, POST_CORNER↔POST/CORNER, SPEED_OUT↔OUT) | `separation_at_pass_forward`, target rate, EPA/route | Same movement on both sides; separation is what scouts care about; largest outcome n |
| **B** | DL first-step burst / get-off in FRONT_START_WAVE, PASS_RUSH | Burst in first 1 s after snap | `player_get_off`, `time_to_pressure`, pressure/sack rate | NGS get-off is a known valued trait; 43k labeled plays |
| **C** | OL lateral slide: wave slide/shuffle, mirror drill (lateral speed, recovery after direction change) | Pass-set lateral velocity / kick-step after snap | `pressure_allowed`, `time_to_pressure_allowed` | 60k labeled plays; OL Combine tests are notoriously uninformative, so a hit here is non-obvious |
| **D** | DB backpedal → transition (45°/90° break): time to flip and drive | Backpedal-to-drive transitions in games | `coverage_assignment` only (outcome weak) | Kinematics-to-kinematics only; keep as a secondary |
| **E** | 40/shuttle speed-curve *shape* (not time) | Game accel profile | Snaps, EPA (all positions) | Cheap cross-position baseline; tells us whether "shape beats stopwatch" in general |

**Questions to ask about every link:**
- Does the signature predict the game trait **after** controlling for Combine times + draft pick + position?
- Does the game trait in turn connect to the outcome?
- Is the effect stable across draft classes?

---

## 4. Evaluation protocol (fixed now, don't tune after seeing results)

- **Unit = player.** Aggregate plays to player level, with shrinkage (empirical Bayes) toward the position mean so low-snap players don't dominate. Plays are not independent, so **never** use play-level p-values.
- **Baselines** (nested):
  - B0 = position + draft-year dummies. Year dummies absorb Combine session/tracking artifacts and class-level game differences.
  - B1 = + traditional Combine (40, 10-yd split, 3-cone, shuttle, vertical, broad, bench, height, weight, NGS athleticism score; missing values median-imputed with skipped-drill indicators)
  - B2 = + draft pick (with undrafted = separate indicator)
- **The candidate must beat B2.**
- **Inference (primary, seed-free):** Freedman–Lane permutation test (≥ 10,000 perms) of the in-sample partial correlation of the signature with the target given B2. Residuals of y under B2 are permuted within position. The test is **one-sided in the preregistered expected sign.** Report partial r with its Fisher 95% CI and the minimum detectable effect (80% power), and state nulls as bounds.
- **Prediction (secondary):** repeated (≥ 50×) 5-fold ridge ΔR² vs B2, with the Nadeau–Bengio corrected SE. Fold R² = 1 − MSE_fold / Var(y_all). Effects are reported as the **excess** over a Freedman–Lane permutation-null mean (≥ 2,000 perms), plus a within-position-normalized version for pooled populations. Also run leave-one-draft-class-out (LOCO), each class against its own null mean.
- **Exposure:** game traits must be exposure-invariant (quantiles, rates, fixed-size samples), never a top-k over a variable number of plays. Check corr(trait, log plays).
- **Multiple testing:** count every test in the log. Use BH-FDR across a round's primaries and also report it across all tests in the round. Use a max-statistic permutation for the final hero claim.
- **Go criteria for a test (all required):**
  1. Freedman–Lane one-sided p < 0.05 **and** partial r in the preregistered direction
  2. excess CV ΔR² over B2, minus 1 SE, > 0
  3. LOCO excess > 0 in ≥ 2 of 3 draft classes
  4. survives BH across the round's primaries
  5. feature and target reliability ≥ 0.5 (`reliability.py`)
  6. stable across ≥ 3 CV seeds, and not driven by ≤ 3 players (leave-3-out influence check)
  7. a scout can explain it in one sentence
- **Revision notes:**
  - 2026-10-14 (rev 1): after the infrastructure review and before round 1, the Freedman–Lane null, excess ΔR², and LOCO replaced sign consistency.
  - 2026-10-14 (rev 2): after round 1 and before round 2, made the inferential test directional, seed-free and in-sample; LOCO is now judged against its null; added year dummies; added the BH and influence criteria. Round-1 decisions stand under rev 1. See `experiments/log.md` GATE 1.
- **2025 class caveat:** only one season of games. Report results with and without 2025 for game-outcome targets.

---

## 5. Timeline with decision gates

| Dates | Phase | Gate / output |
|---|---|---|
| Oct 14–19 | Infrastructure (section 2) + E001 sanity link | Harness passes noise/planted tests; E001 works |
| Oct 20–26 | Probes A–E, 1–2 days each | **Gate 1 (Oct 26):** rank probes by criteria → keep top 2, park the rest |
| Oct 27–Nov 16 | Deep build on top 2: better break detection, route-type matching, sensitivity analyses, mixed models | **Gate 2 (Nov 16):** pick the hero link + name the metric. Fallback if nothing passes: see section 6 |
| Nov 17–Dec 7 | Leaderboards, "undervalued at the Combine" list, side-by-side drill vs game animations, robustness | **Gate 3 (Dec 7):** analysis frozen; only bug fixes after this |
| Dec 8–28 | Writeup drafts, mock judging every ~5 days, cut to 2,000 words / < 10 figures | Mock score ≥ target on all rubric items |
| Dec 29–Jan 5 | Rerun the public notebook top to bottom from raw data; submit **Jan 5** | Submitted a day early |

Mock judging every 2 weeks (from Oct 26) uses the checklist in `AGENTS.md`.

---

## 6. Pivot rules

- **Kill a probe** if it fails ≥ 2 go criteria after the time box. Do not extend the time box more than once.
- **Revisit** a probe if it fails only on n/power. Note what extra data would fix it (e.g., pooling WR+TE, pooling route types).
- **If nothing passes at Gate 2:** pivot to a **"drill reliability + transfer map"** framing:
  - which drill signatures are stable (ICC) and which transfer to game kinematics at all
  - a matrix of drill × game trait with honest CIs
  - this still satisfies the pass/fail rule and is a credible, useful finding for clubs. A good null result reported rigorously beats an overclaimed weak one.
- **If two links pass:** the hero is the one with the clearest scout story. The other becomes a one-figure "it generalizes" section.
- **Pivot inside a link before abandoning it:** different window, different aggregation, different route grouping. Each counts as a new logged test (multiple-testing budget).

---

## 7. Working cadence

- **Each experiment:** hypothesis in the log *before* running → script → results JSON → keep/kill/revisit entry.
- **Notebooks** (`notebooks/NN_*.ipynb`) are scratch. Anything reused gets promoted into `src/bdb27/` with a test.
- **The final public notebook** imports only from `src/bdb27/` (or inlines it) and regenerates every figure. Start the skeleton by Gate 2 so it never gets far out of date.
- **Weekly Sunday review:**
  - update this file's gate status
  - pick the week's figure
  - add an entry to `LESSONS.md` for any mistakes

---

## Gate status

- [x] Infrastructure + E001 (target Oct 19)
  - [x] env, `io` (Parquet cache), `standardize` (game), `linkage` harness + tests, feature store, E001 passed (2026-10-14)
  - [x] `kinematics`, `segments` (snap windows, break/cut detection), Combine drill canonicalization, `reliability` (2026-10-14)
  - [x] adversarial review of the infrastructure (21 agents): 10 confirmed defects fixed, regression tests added, E001 re-run (2026-10-14)
- [x] Gate 1 — probe ranking (done 2026-10-14; see experiments/log.md "GATE 1")
  - Only WR tracked top speed beyond the 40 time survived (E001/E006, BH q = 0.021 over the round's 14 primaries). Position-drill signatures (A–D) show no detectable transfer beyond B2 (CIs about ±0.3 on partial r).
  - Carried forward: (1) hero candidate **"hidden top speed"**; (2) the **drill reliability + transfer map** (PLAN §6 fallback, now a core section); (3) harness upgrade before round 2.
  - [x] Round 2: harness upgrade → hidden-speed deep build (E007) → transfer map (E008) (2026-10-14)
- [x] Gate 2 — hero metric (2026-10-14): **no hero survived.** E007 primaries all fail; E008 shows 16/18 links null, with bounds. See experiments/log.md "GATE 2". Direction of the writeup: pending the user's decision.
- [ ] Gate 3 — analysis frozen (Dec 7)
- [ ] Submitted (Jan 5)
