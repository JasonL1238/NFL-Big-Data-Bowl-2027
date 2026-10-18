"""E012 - Combine Game Equivalents (CGE): how many NFL games of a player's own tracking is his Combine worth, trait
by trait, and how should the two be blended in season? Prereg: experiments/results/E012_prereg.json (binding).

Runs top to bottom:
  1. Combine inputs: player table, tracked-40 set (speed.combine_speed_table), matched drill signatures (saved probe
     tables), and the validation-2 versions of the drill-z composites with 2023-24 Combine-class constants.
  2. Play-level values of the 12 traits, rebuilt from the probe modules (probe A route breaks and probe C pass sets
     are rebuilt from tracking and checked against their saved player-level tables), in two versions: internal
     (all seasons, the source experiments' definitions) and prospective (every per-play constant and expectation
     model refit on 2023-24 games). Per-game values with the prereg minimum-play rules.
  3. Per trait: variance components (draft-year strata), reliability, cross-validated R2 for C0-C3, true-score
     R2, CGE and Delta CGE with a 1,000-resample player bootstrap of the whole pipeline; exposure checks.
  4. Validation 1 (internal crossover) and validation 2 (prospective: 2023-24 fit -> 2025 rookies; H1-H3 with
     sign-flip permutation tests, bootstrap CIs, BH; secondary per trait / k = 3, 4; calibration).
  4c. Post hoc, after two reviews (POST_HOC PH5-PH12, REVIEW_RESPONSE): reliability gate, leave-one-class-out,
     C0-relative / role / exposure references, season-centred sensitivity and stationarity checks; simulation at each
     trait's own design (calibrated bounds for CGE(C2), B1 vs PH5 coverage); validation 1 and 2 re-reads.
  5. Deliverables: CGE table, in-season weight curves, rookie case studies, rookie outliers relative to the prior
     (the prereg's 'Combine surprises'); headline and summary; figures; E012.json.
Implementation choices were frozen before any CGE, variance component or forecast was computed (IMPLEMENTATION
CHOICES below, with the freeze time and hash); amendment A1 (the posterior blend weight) was recorded before any
forecast. Anything decided later is in POST_HOC.

Run: .venv/bin/python experiments/E012_game_equivalents.py            (everything; n_jobs 3; ~2 h)
     .venv/bin/python experiments/E012_game_equivalents.py --quick    (small bootstrap/permutation counts, scratch)
     .venv/bin/python experiments/E012_game_equivalents.py --reuse-values  (skip step 2: read data/processed/e012_*)
     .venv/bin/python experiments/E012_game_equivalents.py --reuse-sims    (read the PH6 simulations from parquet)
     .venv/bin/python experiments/E012_game_equivalents.py --figures-only  (redraw the figures from E012.json)
     .venv/bin/python experiments/E012_game_equivalents.py --rewrite  (rebuild cge_table, headline, summary, review
                                                                      notes and figures from E012.json)
Outputs: experiments/results/E012.json, experiments/figures/E012_*.png, data/processed/e012_*.parquet.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from bdb27 import io, linkage, reliability, segments, shrinkage, speed  # noqa: E402
from bdb27 import targets_wr as TW  # noqa: E402
from bdb27.probes import a_route_breaks as A  # noqa: E402
from bdb27.probes import b_dl_burst as B  # noqa: E402
from bdb27.probes import c_ol_lateral as C  # noqa: E402
from bdb27.probes import d_db_transition as D  # noqa: E402
from bdb27.probes import f_drill_intensity as F  # noqa: E402
from bdb27.probes import h_game_equivalents as H  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG, DATA = ROOT / "experiments" / "results", ROOT / "experiments" / "figures", io.PROCESSED
QUICK = "--quick" in sys.argv
REUSE = "--reuse-values" in sys.argv
REUSE_SIMS = "--reuse-sims" in sys.argv
FIGURES_ONLY = "--figures-only" in sys.argv
REWRITE = "--rewrite" in sys.argv  # rebuild cge_table, wording and figures from E012.json (no recomputation)
SCRATCH = Path("/private/tmp/claude-501/-Users-jasonli-Documents-GitHub-NFL-Big-Data-Bowl-2027/"
               "ca6dc0b9-7e4f-45b4-b4e4-1468819c8969/scratchpad/e012")
OUT_RES, OUT_FIG = (SCRATCH, SCRATCH) if QUICK else (RES, FIG)
N_JOBS = 3
N_BOOT = 30 if QUICK else 1000
BOOT_REPEATS = 2 if QUICK else H.N_REPEATS
N_FLIPS = 5_000 if QUICK else 100_000
N_BOOT_V = 1_000 if QUICK else 10_000
SEED = 20261015
TRAIN_SEASONS, TEST = (2023, 2024), 2025
KS = (1, 2, 3, 4)
# post hoc (review response) settings
REL_MIN = 0.5  # PLAN section 4 criterion 5: a trait's CGE is interpreted only if both reliabilities reach this
SIM_GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75)  # planted population R2 (PH6)
SIM_REPS = 20 if QUICK else 200  # simulated datasets per trait x grid point
SIM_PH5_REPS = 5 if QUICK else 50  # of which the first ones also get the PH5 interval (coverage check)
COV_DESIGNS = ("T1", "T6")  # designs for the B1-vs-PH5 coverage simulation
COV_DATASETS = 3 if QUICK else 80
COV_B1_RESAMPLES = 20 if QUICK else 250  # B1 resamples per simulated dataset (the real run used 1,000)
DEEP_ROUTES = ["GO", "POST", "CORNER", "WHEEL"]
MIN_SEASON_GAMES = 3  # qualifying games for a player-season mean in the stationarity check
HIGHER_MEANS = {  # direction cue for every trait (values of T2-T11 are centred on their route / alignment / slot mean)
    "T1": "faster", "T2": "more speed kept through the OUT/IN break", "T3": "harder braking into the HITCH",
    "T4": "more separation than expected", "T5": "more yards per route than expected",
    "T6": "more ground covered in the first second", "T7": "slower get-off (worse)",
    "T8": "more pressure than expected (better)", "T9": "quicker pass set", "T10": "more pressure allowed (worse)",
    "T11": "faster retreat", "T12": "faster"}
PREREG_PATH = RES / "E012_prereg.json"
PREREG = json.loads(PREREG_PATH.read_text())
T0 = time.time()

# =====================================================================================================
# Frozen implementation choices (copied verbatim from the freeze file; see module docstring)
# =====================================================================================================
CHOICES_FROZEN_AT = "2026-10-15T13:20:00Z"
CHOICES_SHA256 = "70c2247164400b394dce968867b56b4b2937938b18aa1dd5fe9e23980dc83a76"  # of the freeze file
AMENDMENT_A1_SHA256 = "1020ac059a8ad5201ee0233d1a6b36dd8187f553fa63c00dcd7a5b7d5126b627"
IMPLEMENTATION_CHOICES = json.loads(r'''
[
 {"id": "G1", "topic": "population", "choice": "Position = Combine position (linkage.player_table().position). WR traits use Combine WRs only (no TE), DB traits Combine DBs. Draft class = draft_year.", "reason": "Same population definition as every earlier experiment; the prereg names positions, not NFL roles."},
 {"id": "G2", "topic": "games", "choice": "Regular-season (REG) games only. Chronological order of a player's games = game_id order (game_id encodes YYYYMMDD; verified monotone in season and week).", "reason": "AGENTS.md: outcomes are regular season. game_id is the only date field."},
 {"id": "G3", "topic": "qualifying game", "choice": "A game qualifies for trait T if the player has at least the prereg minimum number of plays/events in it with a finite play-level value of T (after the trait's own play filters). The per-game value is the mean of those play values (T1/T12: their 0.9 quantile, polars default 'nearest' interpolation, as features.game_top_speed uses).", "reason": "The prereg gives minimum counts per game; counting only plays that carry a value makes the count and the value refer to the same plays."},
 {"id": "G4", "topic": "player sample", "choice": "A player enters trait T (variance components, reliability, R2, CGE) with >= 3 qualifying games; the same player set is used for every information set.", "reason": "Prereg: players with >= 3 qualifying games. A common sample keeps C0-C3 comparable."},
 {"id": "T1_T12", "topic": "play max speed", "choice": "Play max = max raw `s` over all tracked frames of a REG player-play (probes.f_drill_intensity.play_max_speeds, as E001/E010); plays > 12 yd/s dropped; recomputed for all Combine WRs and DBs (e010_play_max covers only E010's population).", "reason": "Prereg definition; e010_play_max.parquet lacks players without drill-intensity features."},
 {"id": "T2", "topic": "OUT/IN retention", "choice": "Rebuild E002 game events with probes.a_route_breaks (game_route_plays -> game_frames -> game_events). Play value = retention of found OUT/IN events minus the route-type mean over all found WR+TE events of that route (E002 game_trait centering). WR rows only.", "reason": "Prereg: route-centered E002 definitions. Verified against probe_a_game_traits raw means before use."},
 {"id": "T3", "topic": "HITCH braking", "choice": "Found HITCH events, peak_decel minus the HITCH mean over all found WR+TE HITCH events. WR rows only.", "reason": "E002 game_trait definition (route-centered)."},
 {"id": "T4", "topic": "separation over expected", "choice": "Play value = soe from probes.a_route_breaks.separation_over_expected (E002: HGB, GroupKFold(5) by player over all WR+TE route plays with a separation value, SCREEN excluded), on HITCH/OUT/IN routes, WR rows only. Fit inside threadpool_limits(3).", "reason": "Prereg: E002 soe definition."},
 {"id": "T5", "topic": "yards per route over expected", "choice": "Play value = yprr_oe from data/processed/e007_routes.parquet (E007 cross-fit), WR rows only.", "reason": "Prereg: E007 yprr_oe residuals."},
 {"id": "T6_T7", "topic": "DL first-second rush distance and get-off", "choice": "From probe_b_game_plays.parquet: pass-rush snaps (rush == True) lined up EDGE or INTERIOR_LINE. Play value = d10 (T6) or player_get_off (T7) minus its alignment mean plus the overall mean (b_dl_burst.center_by_alignment formula), means taken over all such snaps of study DL with a finite value. No >= 30-snap player filter (E012 has its own per-game rule).", "reason": "E003 alignment adjustment; E003's player filter (>= 30 snaps) belonged to its player-level traits, not to play values. T7 follows E003's g_getoff_rush, which is alignment-adjusted."},
 {"id": "T8", "topic": "DL pressure over expected", "choice": "Play value = pressure (time_to_pressure non-null) minus the alignment (EDGE / INTERIOR_LINE) base rate over all pass-rush snaps of study DL; per-game mean over pass-rush snaps (>= 10).", "reason": "E003 definition of pressure over expected (alignment base rate as the expectation)."},
 {"id": "T9", "topic": "OL pass-set acceleration", "choice": "Rebuild with probes.c_ol_lateral (pass_block_plays -> add_expectations -> pass_set_frames -> pass_set_metrics). Play value = set_accel_c (peak a_tan 0-0.6 s on non-play-action dropbacks >= 2 s with a full 0-2 s window, minus its T/G/C mean). Verified against probe_c_game before use.", "reason": "Prereg: E004 set_accel play values."},
 {"id": "T10", "topic": "OL pressure allowed over expected", "choice": "Play value = pr_oe = pressure_allowed - p_hat from c_ol_lateral.add_expectations (logistic on dropback spline, T/G/C, 2+ rushers, play action, season; in-sample over all pass-block snaps, as E004). Per-game mean over pass-block snaps (>= 15).", "reason": "Prereg: E004 pr_oe definitions."},
 {"id": "T11", "topic": "DB retreat speed", "choice": "From probe_d_game_backpedal_plays.parquet: d_db_transition.game_backpedal_events (bp_peak winsorized at the 1%/99% quantiles of all plays, minus its alignment x coverage-assignment cell mean). Per-game mean over plays (>= 3).", "reason": "Prereg: E005 gm_backpedal play values."},
 {"id": "VC1", "topic": "variance components with draft-year centering", "choice": "Stratified one-way random-effects ANOVA with draft-year strata: MSW = within-player mean square (df N - k); MSB = sum n_i (m_i - stratum game-weighted mean)^2 / (k - Y); n0 = (N - sum_y sum_{i in y} n_i^2 / N_y) / (k - Y); tau2 = (MSB - MSW) / n0, sigma2 = MSW, lambda = sigma2 / tau2. With one stratum this is exactly reliability.icc1's ANOVA estimator (unit-tested). tau2 <= 0 -> lambda = inf and CGE undefined (counted, reported).", "reason": "Prereg: icc1 machinery after draft-year centering; the stratified form gives the centering its correct degrees of freedom."},
 {"id": "VC2", "topic": "reliability used for R2", "choice": "rel = tau2 / (tau2 + mean_i sigma2 / n_i) over the trait's players (n_i = qualifying games). icc1-style mean Spearman-Brown reliability and odd/even-game split-half are reported alongside, descriptive.", "reason": "This is the ratio of true-score to observed variance of the player means that the CV MSE averages over, so R2_CV / rel is unbiased for the true-score R2 under the model."},
 {"id": "R1", "topic": "player-level trait and CV R2", "choice": "y_i = mean of the player's qualifying game values. Var_w = sum_i (y_i - player-weighted draft-year mean)^2 / (n - Y). CV = RepeatedKFold(5 folds, 20 repeats, random_state 0) over players sorted by nfl_id; per repeat R2 = 1 - mean_i (y_i - oof_i)^2 / Var_w; R2_CV = mean over repeats. R2_true = clip(R2_CV / rel, 0, 0.99).", "reason": "Prereg (20 repeats, 5-fold, player-level, true-score correction, clip). Full-sample within-year variance as denominator (LESSONS 2026-10-14: normalize by the full-sample variance)."},
 {"id": "R2", "topic": "information-set models", "choice": "C0 = the training-fold mean of y within the player's draft year (training-fold overall mean if the class is absent from the fold). C1-C3 = linkage._model() (median impute + indicators -> StandardScaler -> RidgeCV) on [draft-year dummies (all classes except the earliest, deterministic) + the set's columns]. C1 = + linkage.COMBINE_CONTROLS; C2 = C1 + linkage.DRAFT_CONTROLS; C3 = C2 + c40_smax, c40_v_20_40, c40_vmax, c40_tau, c40_a0 (speed.combine_speed_table, QC-passing attempts) + the matched signature (T2/T4 c_turn_retention, T3 c_stop_peak_decel, T6-T8 c40_d10, T9/T10 cmb_rev_accel_mirror, T11 cmb_backpedal_speed; none for T1, T5, T12). Fits run under threadpool_limits.", "reason": "Prereg information sets and model. linkage._design drops the first-APPEARING draft year (row-order dependent), so the dummy coding is written out explicitly."},
 {"id": "R3", "topic": "CGE", "choice": "CGE_C = lambda R2_C / (1 - R2_C) with R2_C = R2_true. Reported for C0-C3, Delta CGE = CGE(C3) - CGE(C2) (tracking), and descriptive CGE(C2) - CGE(C1) (draft slot) and CGE(C1) - CGE(C0) (stopwatch).", "reason": "Prereg estimand."},
 {"id": "B1", "topic": "bootstrap", "choice": "1,000 player resamples per trait (numpy default_rng seed 20261015 + trait index). Whole pipeline per resample: draft-year centering, variance components, rel, CV R2 (RepeatedKFold(5, 20, random_state = resample index) over the UNIQUE resampled players, so copies of a player never sit in train and test at once), CGE and Delta CGE. 95% percentile intervals; non-finite draws (tau2 <= 0 or rel <= 0) are counted, and an interval is reported as unbounded above when they exceed 2.5%.", "reason": "Prereg (player bootstrap of the whole pipeline). Grouped folds prevent duplicated players from leaking across CV folds."},
 {"id": "V1", "topic": "validation 1 (internal crossover)", "choice": "Players with >= 8 qualifying games. Game values centered by the full-sample game-weighted draft-year mean. X_k = mean of the chronologically first k games (k = 1..6). Target = mean of games 7..end (the games remaining after the longest window, so every comparison uses the same disjoint target). Combine forecast = the main CV's out-of-fold prediction averaged over its 20 repeats, minus the player-weighted draft-year mean of y. r = Pearson. Integer crossover = smallest k in 1..6 with r(X_k, target) >= r(C, target); 7 means '> 6' (censored). Agreement: ceil(formula CGE), clipped to [1, 7], lies inside the 95% percentile interval of the integer crossover over 1,000 player resamples. A continuous crossover (linear interpolation of r over k with r(0) = 0) is reported as descriptive.", "reason": "The prereg says 'remaining games'; a common target removes target-length differences between k. Integers avoid an arbitrary interpolation rule in the agreement test."},
 {"id": "P1", "topic": "validation 2 training data", "choice": "Training = REG games of the 2023 and 2024 SEASONS (which contain only 2023 and 2024 class players). The 2023/2024 classes' 2025-season games are not used anywhere in validation 2. Test = 2025 class, 2025 season.", "reason": "Strict reading of 'no 2025 data': the fit is what a club could have frozen before the 2025 season."},
 {"id": "P2", "topic": "validation 2 play pipeline refits", "choice": "Every data-derived per-play constant is refit on training plays and applied unchanged to 2025 plays: T2/T3 route means (WR+TE events of 2023-24); T4 SOE model: training values cross-fitted by player (GroupKFold 5) within 2023-24 plays, 2025 values predicted by the same HGB specification fitted on all 2023-24 plays (categories coded from training labels; unseen labels missing); T5 yprr: training values cross-fitted by game (GroupKFold 5, HGB default, seed 0) within 2023-24 routes, 2025 values from the model fitted on all 2023-24 routes with season -> 2024 and defense_team_season -> <team>_2024 (the latest known team-season); T6/T7 alignment means and T8 alignment base rates from 2023-24 snaps; T9 T/G/C means from 2023-24 sets; T10 logistic expectation (spline knots, encoders, coefficients) fitted on 2023-24 snaps, 2025 snaps scored with season -> 2024; T11 winsorization quantiles and alignment x coverage cell means from 2023-24 plays. 2025 plays in a cell or category unseen in training get no value. The minimum-play thresholds and the 12 yd/s cap are fixed constants of the prereg, not data-derived.", "reason": "Prereg leakage rule: no 2025 data may touch any expectation model, centering constant or threshold."},
 {"id": "P3", "topic": "validation 2 Combine inputs", "choice": "2025 Combine measurements are forecast inputs (a club has them before the season). Drill z-score composites (c_turn_retention, c_stop_peak_decel, cmb_backpedal_speed) are recomputed with within-drill means/SDs estimated on the 2023-24 Combine classes only and applied to everyone, so no 2025 value enters a constant of the training fit. cmb_rev_accel_mirror keeps its definitional per-session centering (each session centers on its own Combine reps; no game data). Raw measures (40 time etc., c40 set, c40_d10) are unchanged.", "reason": "Leakage rule applied to Combine-side constants as well; session centering is part of that feature's definition and uses only pre-season Combine data."},
 {"id": "P4", "topic": "validation 2 model fit", "choice": "On training players (>= 3 qualifying 2023-24 games): draft-year strata 2023/2024 for variance components; rel and R2_C via the same 20x5 CV; ridge refit on all training players. 2025 prior mu_C = ridge prediction with the draft-year dummy set to its training mean (a new class gets the training-average class level); C0 prior = training mean of y. Traits with training tau2 <= 0 are excluded from validation 2.", "reason": "A new class has no draft-year mean; the training average is the only leakage-free level."},
 {"id": "P5", "topic": "validation 2 forecasts and metric", "choice": "For each 2025 player and k in 1..4 with >= k + 3 qualifying 2025 games: target = mean of games k+1..end; games-only = mean of games 1..k; Combine-only = mu_C; blend = mu_C + w_k (ybar_k - mu_C), w_k = k / (k + lambda (1 - R2_C)). Standardized SE = (forecast - target)^2 / tau2_train. Pooled = equal-weight mean over traits (traits with >= 1 eligible player at that k) of the trait's mean standardized SE.", "reason": "Prereg."},
 {"id": "P6", "topic": "H1-H3 inference", "choice": "Player-level paired difference d_i = sum over his traits t of (SE_A,it - SE_B,it) / (T n_t), so sum_i d_i equals the pooled trait-weighted mean difference. One-sided sign-flip permutation test (100,000 random sign vectors, seed 20261015; p = (1 + #{S* <= S}) / (1 + B), expected sign negative). 95% CI: 10,000 player-bootstrap resamples (per-trait means recomputed on each resample). BH across H1-H3 at alpha 0.05. Effect sizes: pooled standardized MSE of each method, their difference, and the ratio.", "reason": "Prereg (100,000 flips, player bootstrap, BH); the prereg leaves the bootstrap count open."},
 {"id": "P7", "topic": "secondary and calibration", "choice": "Per-trait and k = 3, 4 comparisons are descriptive (bootstrap CIs; sign-flip p reported but in no BH family). Calibration: 80% predictive interval for the rest-of-season mean = blend +- 1.2816 sqrt(V_k + sigma2 / m), V_k = 1 / (1 / ((1 - R2_C) tau2) + k / sigma2), m = target games; all from training; coverage for C2 and C3 priors at k = 0..4, pooled (trait-weighted) and per trait.", "reason": "Prereg secondary analyses."},
 {"id": "D1", "topic": "case studies", "choice": "Trait per position = the first-listed trait of that position in the prereg (WR T1, DL T6, OL T9, DB T11). 2025 rookies with >= 10 qualifying 2025 games; rank by |C2 prior - full rookie-season mean|; the 2 largest and 2 smallest per position (16 players). Panels show the prior, games-only running mean and blend (C2) after each game, and the season mean.", "reason": "The prereg fixes the rule but not the trait; first-listed is a rule chosen without looking at results."},
 {"id": "D2", "topic": "Combine surprises", "choice": "2025 rookies with >= 10 qualifying 2025 games; z = (rookie-season mean - C2 prior) / sqrt((1 - R2_C2) tau2) with training quantities; top 3 by |z| per trait; the predictive z with sampling noise sqrt((1 - R2_C2) tau2 + sigma2 / n) is shown alongside. Descriptive.", "reason": "Prereg: prior-SD units; the predictive z shows how much of the surprise is sampling noise."},
 {"id": "D3", "topic": "exposure checks", "choice": "corr(draft-year-centered y_i, log n_i games) per trait; corr(per-game value, qualifying plays in that game) pooled and within player (both centered by player).", "reason": "Prereg exposure checks; the within-player version removes between-player role differences."},
 {"id": "X1", "topic": "compute", "choice": "joblib n_jobs <= 3 with one BLAS/OpenMP thread per worker (threadpool_limits(1)); HGB fits inside threadpool_limits(3); lazy scans filtered before collect.", "reason": "Project memory/CPU rules."}
]
''')
AMENDMENTS = [{
    "id": "A1", "recorded_at_utc": "2026-10-15T13:27:00Z", "sha256_of_record": AMENDMENT_A1_SHA256,
    "status": "Recorded before any CGE, variance component, R2 or forecast on real data (only the unit tests on "
              "simulated data had run).",
    "topic": "in-season blend weight (supersedes the weight formula in choice P5)",
    "finding": "The prereg's in_season_rule writes w_k = k / (k + lambda (1 - R2_C)). That is not the Bayesian posterior "
               "weight. With prior N(mu_C, (1 - R2_C) tau2) and ybar_k ~ N(theta, sigma2 / k) the posterior mean puts "
               "weight (1 - R2) tau2 / ((1 - R2) tau2 + sigma2 / k) = k / (k + lambda / (1 - R2_C)) = "
               "k / (k + lambda + CGE_C) on ybar_k, the only form consistent with the prereg's reading 'C is a prior "
               "worth CGE_C pseudo-games on top of shrinkage to the position mean'. The literal formula gives games "
               "MORE weight when the Combine prior is BETTER. The required unit test (blend weight = Bayesian "
               "posterior mean) fails on the literal formula.",
    "choice": "Primary (H1-H3, calibration, weight curves, case studies): w_k = k / (k + lambda / (1 - R2_C)). "
              "Sensitivity reported beside it: the literal prereg formula ('prereg-literal weight'). The posterior "
              "variance used for calibration is unchanged; it already matched the posterior.",
    "reason": "The prereg names the rule as the Bayesian in-season rule and defines CGE through it; the written "
              "weight is an algebra slip. Fixing it before any forecast keeps the test confirmatory; reporting the "
              "literal version keeps the prereg auditable."}]
POST_HOC: list[dict] = [  # decided after results were seen (quick-mode run of the frozen pipeline)
    {"id": "PH1", "what": "Figure layout and encoding (axis scales, legends, the H1-H3 panel drawn as error ratios on "
     "a log axis), after viewing the quick-run figures.", "affects_numbers": False},
    {"id": "PH2", "what": "Validation 1 sensitivity: Combine forecasts centered on the same folds' C0 forecast "
     "(f_C - f_C0) instead of the full-sample class mean. The frozen centering leaves a leave-fold-out artifact (the "
     "training-fold class mean moves opposite to the held-out players), which makes r(C0, target) strongly negative "
     "(validation_1.<trait>.sets.C0.r_combine) and leaks a small negative component into sets whose ridge forecast is "
     "near constant.", "affects_numbers": False},
    {"id": "PH3", "what": "Validation 2, descriptive: Combine prior (C2 / C3) against plain shrinkage to the position "
     "mean (C0 prior) at k = 0..4. H1/H2 compare the blend with raw game averages, which shrinkage alone would beat; "
     "this separates the Combine's share.", "affects_numbers": False},
    {"id": "PH4", "what": "Descriptive correlations of each player trait with each Combine column (what a nonzero CGE "
     "is made of).", "affects_numbers": False},
    {"id": "PH5", "what": "Fixed-forecast player bootstrap (ci95_fixed_forecast_PH5; cge_table ph5_lo / ph5_hi): the "
     "full-sample out-of-fold forecasts are held fixed and players are resampled (same draws as B1) to recompute "
     "variance components, reliability, R2, CGE and Delta CGE. The builder made it the display interval because the "
     "preregistered refit bootstrap's (B1) R2 draws centre below the point estimate (bootstrap_bias_diagnostic: "
     "RidgeCV's leave-one-out alpha search sees duplicated players, and a resample refits on ~63% distinct players). "
     "REVISED AFTER REVIEW: centring does not decide validity, coverage does, and a post hoc interval may not "
     "replace a preregistered one. The preregistered B1 is again the reported interval everywhere (figures, "
     "cge_table lo / hi, weight-curve bands) and PH5 is a labeled sensitivity that is never drawn. In simulation at "
     "the real designs (PH6) neither reaches 95%: PH5 covers the CV estimand about 79% of the time over all designs, "
     "and at the T1 / T6 designs B1 with its preregistered 20 CV repeats covers 78-83% against PH5's 86-88% "
     "(post_hoc_review.simulation). The simulation-calibrated bounds (PH6) are the coverage-checked statement.",
     "affects_numbers": True},
    {"id": "PH6", "what": "Simulation at each trait's own design (its players, classes, game counts and C2 design "
     "matrix; theta = a planted signal on 3 random Combine columns + noise, tau2 = 1; iid games with the trait's "
     "lambda; population R2 grid SIM_GRID). (a) Neyman-calibrated one-sided 95% bounds for CGE(C2) (the "
     "estimand is the CV estimand: the Combine as learned by the ridge from 80% of ~50-85 players), which replace "
     "degenerate [0, 0] intervals as the statement of what a zero estimate rules out; (b) coverage of B1 (250 "
     "resamples, 20 CV repeats) and PH5 for that estimand on the T1 and T6 designs, and PH5 coverage on every "
     "design; (c) how much of a population signal a ridge on this many players learns.", "affects_numbers": True},
    {"id": "PH7", "what": "Reliability gate and stability: a trait's CGE is interpreted only if both its model-based "
     "reliability (rel) and its odd/even split-half reach 0.5 (PLAN section 4 criterion 5); otherwise it is reported "
     "as not identified. Leave-one-class-out CGE for C1-C3, and the 2023-24-seasons fit (validation 2 training) "
     "quoted beside the 3-class values.", "affects_numbers": False},
    {"id": "PH8", "what": "Reference sensitivities: CGE measured against C0 under the same CV (the Combine side "
     "otherwise pays the out-of-fold cost of estimating class means, which the k-game benchmark does not); CGE of a "
     "role reference (NFL roster position for DL/OL/DB; deep-route share for WR, which is in-season usage) and of an "
     "exposure reference (log qualifying games), and what C2 adds beyond each (CGE(ref + C2) - CGE(ref)).",
     "affects_numbers": False},
    {"id": "PH9", "what": "Validation 1 re-read: which cells could have disagreed at all, the literal prereg check "
     "(formula CGE inside the continuous-crossover interval), and the CGE implied by the correlations under the "
     "model (implied true-score R2 = r(C, target)^2 / reliability of the target means), from the frozen and the "
     "PH2-centred forecasts.", "affects_numbers": False},
    {"id": "PH10", "what": "Validation 2 decomposition: the share of the H1/H2 gains that the position-mean blend "
     "(C0) already achieves, C0 blend vs games-only tests, BH across the 10 PH3 tests, PH3 per-trait drivers and PH3 "
     "without the not-identified traits, per-trait calibration and observed/expected C0 error, the 2025 class "
     "shift per trait, the spread of the C2 prior across 2025 rookies, and the share of rookie appearances that are "
     "qualifying games.", "affects_numbers": False},
    {"id": "PH11", "what": "Stationarity and the game unit: season-centred sensitivity (lambda, rel, CGE with per-game "
     "values minus their season mean), within-player season-to-season change by class and transition, within-season "
     "trend, the share of within-player-season variance common to one game, and within-player variance in the "
     "lowest vs highest plays-per-game quartile.", "affects_numbers": False},
    {"id": "PH12", "what": "Wording and figures after review: headline rewritten (E012.json headline; it supersedes "
     "E012_audit/summary.txt); the 'Combine surprises' deliverable is labeled 'rookie outliers relative to the prior' "
     "(selection rule unchanged) with the C0-prior z, the predictive z and NFL position; weight curves state their "
     "fit and use B1 bands; figures redrawn (linear CGE axis, not-identified traits separated, colour-vision-safe "
     "encodings, caveats in subtitles).", "affects_numbers": False},
]

INTERVAL_NOTE = (
    "cge_table lo / hi and every drawn interval: the preregistered whole-pipeline player bootstrap (B1, "
    "ci95_prereg_bootstrap); the post hoc fixed-forecast bootstrap (PH5, ph5_lo / ph5_hi) is a sensitivity that is "
    "never drawn. Neither is calibrated: in simulation at the real designs both cover the CV estimand roughly 80% of "
    "the time at nominal 95% (post_hoc_review.simulation.interval_coverage and calibrated_bounds."
    "ph5_coverage_pooled), so read the drawn intervals as narrower than 95%. Every B1 CGE interval includes 0. For a "
    "zero point estimate the B1 interval is [0, 0] and says nothing; the simulation-calibrated one-sided 95% bounds "
    "(cge_C2_calibrated_lower95 / upper95, PH6) state what an estimate rules out under the simulation model (signal "
    "on 3 Combine columns, iid games); they can sit above 0 where B1 does not.")

DETERMINISM_FIX = {  # 2026-10-17, found by the clean-room reproduction; no definition or statistical setting changed
    "date": "2026-10-17", "check": "PH11, stationarity()",
    "affected": "post_hoc_review.per_trait.<T>.stationarity_and_game_unit.season_change_within_player.<transition>.ci95",
    "what": "The bootstrap of each class/transition's season-to-season changes resampled rows by position, and the row "
            "order came from an unordered polars group_by and join, so the 36 intervals changed from run to run "
            "(across four earlier full runs, T1, T7 and T9 '2024 class 2024->2025' changed whether they include 0). "
            "Rows are now sorted by nfl_id before the bootstrap; seed, resample count (2,000) and every definition "
            "are unchanged. This run's intervals replace the earlier ones and are reproducible. mean_tau and n are "
            "unchanged (last-digit rounding aside); they are the only season-change values that the cge_table notes "
            "and the writeup use."}

REVIEW_RESPONSE = [  # two reviews (statistics, football) of the builder's run, 2026-10-15; numbers live in the keys named
    {"issue": "R1.1 PH5 replaced the preregistered B1 interval and under-covers", "status": "accepted (partly)",
     "action": "B1 restored as the reported interval everywhere (a post hoc interval may not replace a preregistered "
               "one); PH5 kept as a labeled sensitivity, never drawn; simulation-calibrated bounds added. Not "
               "reproduced: that B1 covers better. With its preregistered 20 CV repeats B1 covered 78-83% at the T1 "
               "and T6 designs against PH5's 86-88% (PH5 79% over all designs); the reviewer's B1 runs used 1 or 5 "
               "repeats, which make each draw noisier and the interval wider. Both are reported as under-covering",
     "where": ["interval_note", "cge_table[].lo/hi, ph5_lo/ph5_hi", "post_hoc_review.simulation.interval_coverage"]},
    {"issue": "R1.2 / R2.2 T4 and T8 CGEs are artifacts of the true-score correction", "status": "accepted",
     "action": "Rule (PLAN 4 criterion 5): a CGE is interpreted only if rel and split-half both reach 0.5; T4, T8 and "
               "T10 are reported as not identified, kept out of the headline and drawn apart in the figures",
     "where": ["cge_table[].identified, notes", "post_hoc_review.per_trait.<T>.identified"]},
    {"issue": "R1.3 no CGE excludes 0; headline values weaker out of sample", "status": "accepted",
     "action": "Point estimates are shown with B1 intervals (all include 0), the 2023-24-seasons fit, the "
               "leave-one-class-out range and the simulation-calibrated bounds (PH6) beside them; the headline "
               "quotes no CGE without this context",
     "where": ["cge_table[].cge_C2_2023_24_seasons_fit, cge_C2_leave_one_class_out_range"]},
    {"issue": "R1.4 / R2.7 validation 1 '36/36' could hardly fail", "status": "accepted",
     "action": "Reported as 'not contradicted (weak check)'; cells classified by whether they could disagree; the "
               "literal prereg check with the continuous crossover and the model-implied CGE added (PH9); '36/36' "
               "dropped from the headline",
     "where": ["validation_1_summary.reading_after_review", "post_hoc_review.validation_1"]},
    {"issue": "R1.5 / R2.1 H1/H2 credit the Combine with what shrinkage does; 'frozen before 2025'",
     "status": "accepted",
     "action": "Headline rewritten to lead with shrinkage; share of the H1/H2 gain achieved by the position-mean "
               "blend, C0-blend vs games-only tests, BH across the 10 PH3 tests, PH3 drivers; design relabeled "
               "pseudo-prospective; prospective figure shows the C0 blend and the PH3 comparisons",
     "where": ["headline", "summary", "supersedes", "validation_2.design_label", "post_hoc_review.validation_2"]},
    {"issue": "R1.6 asymmetric baseline; estimand specific to a ridge on ~50-85 players", "status": "accepted",
     "action": "C0-relative CGE added as a sensitivity; estimand defined (the Combine as learned by the ridge from 80% "
               "of the players); population-signal bound added from the same simulation",
     "where": ["cge_table[].cge_C2_vs_C0_same_cv, cge_C2_population_signal_upper95",
               "post_hoc_review.simulation.calibrated_bounds.estimand"]},
    {"issue": "R1.7 / R2.13 exposure entanglement (T5, T8, T10)", "status": "accepted (partly)",
     "action": "Flagged in the cge_table notes; log-games reference sensitivity (CGE of exposure, C2 beyond it). A "
               "fixed-games subsample was not run: too few players",
     "where": ["cge_table[].exposure_r_trait_log_games, notes", "post_hoc_review.per_trait.<T>.references.exposure"]},
    {"issue": "R1.8 variance components do not transfer trait by trait; season and game shocks", "status":
     "accepted (partly)",
     "action": "Per-trait calibration, observed/expected C0 error and the 2025 class shift reported; season-centred "
               "sensitivity of lambda / rel / CGE; same-game variance share. The preregistered model is unchanged; "
               "player resampling cannot capture season or class-shift uncertainty (stated)",
     "where": ["post_hoc_review.validation_2.per_trait", "post_hoc_review.per_trait.<T>.season_centred",
               "post_hoc_review.per_trait.<T>.stationarity_and_game_unit"]},
    {"issue": "R1.9 degenerate [0, 0] intervals read as bounds", "status": "accepted",
     "action": "Simulation-calibrated (Neyman) one-sided 95% bounds for CGE(C2) at each trait's own design (PH6)",
     "where": ["cge_table[].cge_C2_calibrated_upper95", "post_hoc_review.simulation.calibrated_bounds"]},
    {"issue": "R2.3 DL burst is role / body type, not athleticism", "status": "accepted",
     "action": "Role reference (NFL roster position) and the Combine's CGE beyond it reported for every trait; the "
               "DL value is described as body type and role",
     "where": ["cge_table[].cge_role_reference, cge_C2_beyond_role", "post_hoc_review.per_trait.T6.references.role"]},
    {"issue": "R2.4 non-stationary traits (season shifts, within-season trend, rookie drift)", "status":
     "accepted (partly)",
     "action": "Within-player season-to-season change by class and transition, within-season trend and a "
               "season-centred sensitivity reported; T9 and T11 flagged; outlier rows carry the 2025 class shift. "
               "Validation 2 was not re-run on season-centred values (a club would need contemporaneous league means; "
               "left as a limitation)",
     "where": ["post_hoc_review.per_trait.<T>.stationarity_and_game_unit", "cge_table[].notes"]},
    {"issue": "R2.5 a 'game' is a qualifying game; equal weights despite snap-dependent noise; opponent effects",
     "status": "accepted (partly)",
     "action": "Axes say 'qualifying games (>= N plays)'; share of 2025 rookie appearances that qualify, noise in the "
               "lowest vs highest plays quartile, same-game variance share reported. Per-play weighting and opponent "
               "adjustment change the preregistered model and were not implemented (stated as limitations)",
     "where": ["post_hoc_review.validation_2.qualifying_share_2025_rookies",
               "post_hoc_review.per_trait.<T>.stationarity_and_game_unit"]},
    {"issue": "R2.6 case studies and surprises mostly show rookies far from average", "status": "accepted",
     "action": "Table relabeled 'rookie outliers relative to the prior' (rule unchanged) with the C0-prior z, "
               "predictive z, NFL position and the 2023-24 CGE; flat C2 priors flagged; case-study panels say when "
               "the prior is the position average; axes say what they are centred on",
     "where": ["deliverables.rookie_outliers_vs_prior", "deliverables.c2_prior_flat_by_trait",
               "deliverables.case_studies"]},
    {"issue": "R2.8 WR top speed is fragile and partly usage", "status": "accepted",
     "action": "Leave-one-class-out range, the 2023-24 value and the deep-route-share correlation and reference "
               "reported", "where": ["cge_table[T1]", "post_hoc_review.per_trait.T1.references.role"]},
    {"issue": "R2.9 PH3 post hoc, multiplicity, drivers", "status": "accepted",
     "action": "BH q over the 10 PH3 tests, per-trait drivers, and PH3 on identified traits only",
     "where": ["post_hoc_review.validation_2.ph3_bh_q_over_10, ph3_drivers"]},
    {"issue": "R2.10 H3 shows tracking slightly worse; interval invisible", "status": "accepted",
     "action": "Worded as 'slightly worse' with its interval; the prospective figure draws the Combine-vs-shrinkage "
               "comparisons on their own linear axis", "where": ["headline", "figures/E012_prospective.png"]},
    {"issue": "R2.11 weight curves come from the full fit; 'C2 prior' where C2 = C0", "status": "accepted",
     "action": "Fit stated; B1 bands; 'position-average prior' wording where C2 = C0; not-identified traits greyed",
     "where": ["deliverables.weight_curves.<T>.fit", "figures/E012_weight_curves.png"]},
    {"issue": "R2.12 figure accessibility", "status": "accepted",
     "action": "Series carry marker and line-style encodings, colours validated (dataviz validator) for CVD and "
               ">= 3:1 contrast, caveats moved to subtitles in secondary ink, CGE axis linear with printed values",
     "where": ["figures/"]},
]

INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, VIOLET = "#2a78d6", "#eb6834", "#4a3aa7"  # categorical slots 1, 2, 7 (validated all-pairs, light)
# ordinal blue ramp for the nested information sets (light end 3.5:1 on the surface; validated --ordinal)
RAMP = {"C1": "#3987e5", "C2": "#1c5cab", "C3": "#0d366b"}
MARK = {"C1": "o", "C2": "s", "C3": "D"}  # secondary encoding: identity never rests on colour alone
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.titlecolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 9.5, "axes.titlelocation": "left", "legend.frameon": False, "savefig.facecolor": SURF})


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def clean(o):
    """JSON-safe: numpy -> python, NaN/inf -> None."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


def corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else np.nan


def maxdiff(a: pl.DataFrame, b: pl.DataFrame, on: str, ca: str, cb: str) -> dict:
    m = a.select(on, x=ca).join(b.select(on, y=cb), on=on, how="full", coalesce=True)
    both = m.filter(pl.col("x").is_not_null() & pl.col("y").is_not_null())
    return {"n_a": int(a[ca].is_not_null().sum()), "n_b": int(b[cb].is_not_null().sum()), "n_both": both.height,
            "max_abs_diff": float((both["x"] - both["y"]).abs().max()) if both.height else None}


def check_frozen_choices() -> dict:
    """If the freeze file exists (committed audit record), assert the copy above is identical to it."""
    f = RES / "E012_audit" / "implementation_choices_frozen.json"
    if not f.exists():
        return {"freeze_file_present": False}
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    same = json.loads(f.read_text())["choices"] == IMPLEMENTATION_CHOICES
    assert sha == CHOICES_SHA256 and same, "implementation choices differ from the frozen file"
    return {"freeze_file_present": True, "sha256_matches": True, "content_matches": True}


# =====================================================================================================
# 1. Combine inputs
# =====================================================================================================

def build_combine() -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    pt = linkage.player_table()
    c40 = speed.combine_speed_table().select("nfl_id", *H.TRACK40)
    a_sig = pl.read_parquet(DATA / "probe_a_combine_signatures.parquet").select("nfl_id", "c_turn_retention",
                                                                                "c_stop_peak_decel")
    b_sig = pl.read_parquet(DATA / "probe_b_combine.parquet").select("nfl_id", "c40_d10")
    c_sig = pl.read_parquet(DATA / "probe_c_combine.parquet").select("nfl_id", "cmb_rev_accel_mirror")
    ce, cb = (pl.read_parquet(DATA / f"probe_d_combine_{k}.parquet") for k in ("events", "backpedal"))
    d_sig = D.combine_signatures(ce, cb).select("nfl_id", "cmb_backpedal_speed")
    players = pt
    for t in (c40, a_sig, b_sig, c_sig, d_sig):
        players = players.join(t, on="nfl_id", how="left")
    e7 = pl.read_parquet(DATA / "e007_players.parquet").select("nfl_id", "c40_v_20_40", "c40_smax", "c40_vmax")
    checks = {f"{c}_vs_e007": maxdiff(c40, e7, "nfl_id", c, c) for c in ("c40_v_20_40", "c40_smax", "c40_vmax")}

    # validation-2 versions: drill-z constants from the 2023-24 Combine classes only (choice P3)
    cev = A.combine_events(A.combine_frames(list(A.DRILLS)))
    ref_ids = pt.filter(pl.col("draft_year") < TEST)["nfl_id"].to_list()
    new = {}
    for col, value, fam in (("c_turn_retention", "retention", "turn"), ("c_stop_peak_decel", "peak_decel", "stop")):
        full = H.route_composite(cev, value, fam).select("nfl_id", pl.col("comp").alias(col))
        checks[f"{col}_rebuilt_vs_saved"] = maxdiff(full, a_sig, "nfl_id", col, col)
        new[col] = H.route_composite(cev, value, fam, ref_ids).select("nfl_id", pl.col("comp").alias(col))
    full = H.backpedal_composite(cb, D.BACKPEDAL_DRILLS)
    checks["cmb_backpedal_speed_rebuilt_vs_saved"] = maxdiff(full, d_sig, "nfl_id", "cmb_backpedal_speed",
                                                             "cmb_backpedal_speed")
    new["cmb_backpedal_speed"] = H.backpedal_composite(cb, D.BACKPEDAL_DRILLS, ref_ids)
    players_p = players.drop(list(new))
    for t in new.values():
        players_p = players_p.join(t, on="nfl_id", how="left")
    for col in new:
        checks[f"{col}_strict_vs_full_r"] = corr(players.sort("nfl_id")[col], players_p.sort("nfl_id")[col])
    return players.sort("nfl_id"), players_p.sort("nfl_id"), checks


# =====================================================================================================
# 2. Play-level values and per-game tables
# =====================================================================================================

def build_values(players: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Internal and prospective per-game tables for the 12 traits (long: trait, nfl_id, game_id, season, value,
    n_plays [, split])."""
    checks: dict = {}
    seasons = io.load("games").filter(pl.col("season_type") == "REG").select("game_id", "season")
    pos = players.select("nfl_id", "position", "draft_year")
    train = pl.col("season") <= max(TRAIN_SEASONS)
    test = (pl.col("season") == TEST) & (pl.col("draft_year") == TEST)
    internal, prosp = {}, {}

    def add_season(df):
        return df.join(seasons, on="game_id", how="left") if "season" not in df.columns else df

    # ---- T1 / T12: play max speed
    ids = pos.filter(pl.col("position").is_in(["WR", "DB"]))["nfl_id"].to_list()
    pm = F.play_max_speeds(ids)
    e10 = pl.read_parquet(DATA / "e010_play_max.parquet")
    k = ["game_id", "play_id", "nfl_id"]
    both = e10.join(pm, on=k, how="inner", suffix="_new")
    checks["play_max_vs_e010"] = {"rows_e010_wr_db": int(e10.join(pos.filter(pl.col("nfl_id").is_in(ids)),
                                                                  on="nfl_id", how="semi").height),
                                  "rows_matched": both.height,
                                  "max_abs_diff": float((both["smax"] - both["smax_new"]).abs().max())}
    v = H.t_play_speed(pm)
    internal["T1"] = internal["T12"] = prosp["T1"] = prosp["T12"] = v
    log(f"play max: {pm.height:,} plays")

    # ---- T2 / T3 / T4: probe A route breaks and separation
    rplays = A.game_route_plays()
    gframes, gtimes = A.game_frames(rplays)
    gev = A.game_events(gframes, rplays, gtimes)
    del gframes
    gev = gev.join(rplays.select(*segments.PLAY_KEYS, "season"), on=segments.PLAY_KEYS, how="left")
    saved_a = pl.read_parquet(DATA / "probe_a_game_traits.parquet")
    for name, routes, col in (("g_turn_retention", ["OUT", "IN"], "retention"),
                              ("g_stop_peak_decel", ["HITCH"], "peak_decel")):
        t = A.game_trait(gev, col, routes, name)
        checks[f"{name}_raw_rebuilt_vs_saved"] = maxdiff(t, saved_a, "nfl_id", f"{name}_raw", f"{name}_raw")
        checks[f"{name}_n_rebuilt_vs_saved"] = maxdiff(t, saved_a, "nfl_id", f"{name}_n", f"{name}_n")
    for tr_, routes, col in (("T2", ["OUT", "IN"], "retention"), ("T3", ["HITCH"], "peak_decel")):
        internal[tr_] = H.t_route_break(gev, routes, col)
        prosp[tr_] = H.t_route_break(gev, routes, col, ref=train)
    log(f"route events: {gev.height:,}")
    hoi = ["HITCH", "OUT", "IN"]
    with threadpool_limits(3):
        soe = A.separation_over_expected(rplays)
    t = A.soe_trait(soe, hoi, "soe_hoi")
    checks["soe_hoi_raw_rebuilt_vs_saved"] = maxdiff(t, saved_a, "nfl_id", "soe_hoi_raw", "soe_hoi_raw")
    internal["T4"] = soe.filter(pl.col("route_ran").is_in(hoi)).select("nfl_id", "game_id", "season", value="soe")
    rp = rplays.join(pos.select("nfl_id", "draft_year"), on="nfl_id", how="left")
    soe_all = H.soe_prospective(rp, pl.lit(True), pl.lit(False))
    m = soe.select(*segments.PLAY_KEYS, a="sep_expected").join(
        soe_all.select(*segments.PLAY_KEYS, b="sep_expected"), on=segments.PLAY_KEYS)
    checks["soe_refit_machinery_all_seasons_vs_E002"] = {"n": m.height, "r": corr(m["a"], m["b"]),
                                                         "max_abs_diff": float((m["a"] - m["b"]).abs().max())}
    soe_p = H.soe_prospective(rp, train, test)
    prosp["T4"] = soe_p.filter(pl.col("route_ran").is_in(hoi)).select("nfl_id", "game_id", "season", value="soe")
    log("separation over expected built")

    # ---- T5: yards per route over expected
    routes = pl.read_parquet(DATA / "e007_routes.parquet")
    e7 = pl.read_parquet(DATA / "e007_players.parquet").select("nfl_id", "yprr_oe")
    eb = TW.player_target(routes, "yprr_oe", 100)
    checks["yprr_oe_eb_vs_e007"] = maxdiff(eb, e7, "nfl_id", "shrunk", "yprr_oe")
    internal["T5"] = routes.select("nfl_id", "game_id", "season", value="yprr_oe")
    rr = routes.join(pos.select("nfl_id", "draft_year"), on="nfl_id", how="left")
    y_all = H.yprr_prospective(rr, TW.CONTEXT_CAT, TW.CONTEXT_NUM, pl.lit(True), pl.lit(False))
    m = routes.select("row_id", a="yprr_exp").join(y_all.select("row_id", b="yprr_exp"), on="row_id")
    checks["yprr_refit_machinery_all_seasons_vs_E007"] = {"n": m.height, "r": corr(m["a"], m["b"]),
                                                          "max_abs_diff": float((m["a"] - m["b"]).abs().max())}
    y_p = H.yprr_prospective(rr, TW.CONTEXT_CAT, TW.CONTEXT_NUM, train, test)
    # test rows carry season "2024" inside the model only; the real season comes back from games for the split
    prosp["T5"] = y_p.select("nfl_id", "game_id", value="yprr_oe").join(seasons, on="game_id", how="left")
    log("yards per route over expected built")

    # ---- T6 / T7 / T8: DL
    gp = add_season(pl.read_parquet(DATA / "probe_b_game_plays.parquet"))
    gt = B.game_traits(gp.filter(pl.col("rush")), "rush", 30)
    saved_b = pl.read_parquet(DATA / "probe_b_game_traits.parquet")
    for c in ("g_d10_rush", "g_player_get_off_rush", "g_pressure_oe_rush"):
        checks[f"{c}_rebuilt_vs_saved"] = maxdiff(gt, saved_b, "nfl_id", c, c)
    el = B.eligible_plays(gp.filter(pl.col("rush")), 30)
    for tr_, fn in (("T6", lambda ref: H.t_dl(gp, "d10", ref)), ("T7", lambda ref: H.t_dl(gp, "player_get_off", ref)),
                    ("T8", lambda ref: H.t_dl_pressure(gp, ref))):
        internal[tr_], prosp[tr_] = fn(None), fn(train)
    # E003's play values on its own >= 30-snap population differ from E012's only by a constant per alignment
    chk = el.select(*segments.PLAY_KEYS, "lined_up_position", e3="d10_adj").join(
        _dl_keys(gp, "d10", None), on=segments.PLAY_KEYS, how="inner").with_columns(dv=pl.col("e3") - pl.col("value"))
    checks["d10_vs_E003_adj_diff_sd_within_alignment"] = float(
        chk.group_by("lined_up_position").agg(pl.col("dv").std())["dv"].max())
    log("DL values built")

    # ---- T9 / T10: OL
    olp = C.add_expectations(C.pass_block_plays())
    sf = C.pass_set_frames(olp)
    sm = C.pass_set_metrics(sf, olp)
    del sf
    saved_c = pl.read_parquet(DATA / "probe_c_game.parquet")
    gs, oc = C.game_set_traits(sm), C.outcome_traits(olp)
    for a, src in (("g_set_accel", gs), ("set_plays", gs), ("pr_oe", oc), ("pb_plays", oc)):
        checks[f"{a}_rebuilt_vs_saved"] = maxdiff(src, saved_c, "nfl_id", a, a)
    sm = add_season(sm)
    checks["set_accel_c_vs_c_ol_lateral_max_abs_diff"] = float(
        (H.center(sm, "set_accel", ["lined_up_position"])["value"] - sm["set_accel_c"]).abs().max())
    internal["T9"], prosp["T9"] = H.t_ol_set(sm), H.t_ol_set(sm, ref=train)
    internal["T10"] = olp.select("nfl_id", "game_id", "season", value="pr_oe")
    op = olp.join(pos.select("nfl_id", "draft_year"), on="nfl_id", how="left")
    allfit = H.ol_pressure_prospective(op, pl.lit(True), pl.lit(False))
    checks["ol_pressure_refit_machinery_all_seasons_vs_E004_max_abs_diff"] = float(
        (allfit["p_hat"] - olp["p_hat"]).abs().max())
    prosp["T10"] = H.ol_pressure_prospective(op, train, test).select(
        "nfl_id", "game_id", value="pr_oe").join(seasons, on="game_id", how="left")
    log("OL values built")

    # ---- T11: DB retreat speed
    bp = add_season(pl.read_parquet(DATA / "probe_d_game_backpedal_plays.parquet"))
    mine = H.t_db_retreat(bp)
    ref = D.game_backpedal_events(bp)
    checks["retreat_vs_d_db_transition_max_abs_diff"] = float((mine["value"] - ref["res"]).abs().max())
    internal["T11"], prosp["T11"] = mine, H.t_db_retreat(bp, ref=train)
    log("DB values built")

    # ---- per-game tables
    gi, gpro = [], []
    for t, spec in H.TRAITS.items():
        ids_t = pos.filter(pl.col("position") == spec["pos"]).select("nfl_id")
        for src, out in ((internal, gi), (prosp, gpro)):
            plays = src[t].join(ids_t, on="nfl_id", how="semi").select("nfl_id", "game_id",
                                                                         pl.col("season").cast(pl.Int64), "value")
            out.append(H.game_table(plays, spec["min_plays"], spec["agg"]).with_columns(trait=pl.lit(t)))
    games_i = pl.concat(gi).select("trait", "nfl_id", "game_id", "season", "value", "n_plays")
    games_p = pl.concat(gpro).select("trait", "nfl_id", "game_id", "season", "value", "n_plays")
    games_p = games_p.join(pos.select("nfl_id", "draft_year"), on="nfl_id").with_columns(
        split=pl.when(train).then(pl.lit("train")).when(test).then(pl.lit("test")).otherwise(pl.lit("unused"))
    ).drop("draft_year")
    return games_i, games_p, checks


def _dl_keys(gp, col, ref):
    """t_dl with the play keys kept (for the E003 comparison)."""
    p = gp.filter(pl.col("rush") & pl.col("lined_up_position").is_in(["EDGE", "INTERIOR_LINE"])
                  & pl.col(col).cast(pl.Float64).is_finite())
    return H.center(p, col, ["lined_up_position"], ref, add_overall=True).select(*segments.PLAY_KEYS, "value")


# =====================================================================================================
# 3. Internal analysis per trait: variance components, R2, CGE, bootstrap, exposure
# =====================================================================================================

def year_centered(g: pl.DataFrame) -> pl.DataFrame:
    """Game values minus the game-weighted mean of the player's draft class (needs draft_year)."""
    return g.with_columns(value=pl.col("value") - pl.col("value").mean().over("draft_year"))


def exposure(g: pl.DataFrame) -> dict:
    """g: qualifying games of the trait's analysis players (nfl_id, draft_year, value, n_plays)."""
    p = (g.group_by("nfl_id", "draft_year").agg(y=pl.col("value").mean(), n=pl.len())
         .with_columns(yc=pl.col("y") - pl.col("y").mean().over("draft_year")))
    w = g.with_columns(vc=pl.col("value") - pl.col("value").mean().over("nfl_id"),
                       nc=pl.col("n_plays") - pl.col("n_plays").mean().over("nfl_id"))
    return {"r_player_trait_log_games": corr(p["yc"], np.log(p["n"].to_numpy())),
            "r_game_value_plays_pooled": corr(g["value"], g["n_plays"]),
            "r_game_value_plays_within_player": corr(w["vc"], w["nc"]),
            "median_plays_per_game": float(g["n_plays"].median())}


def descriptive_reliability(g: pl.DataFrame) -> dict:
    gc = year_centered(g)
    r = reliability.icc1(gc["value"].to_numpy(), gc["nfl_id"].to_numpy())
    sh = reliability.split_half(gc, "value", unit="game_id", min_per_half=1)
    return {"icc1_single_game": r["icc"], "icc1_mean_spearman_brown": r["reliability"],
            "split_half_odd_even_games": sh["reliability"], "split_half_n": sh["n_players"]}


def ci_of(boot: pl.DataFrame, col: str) -> dict:
    return H.percentile_ci(boot[col].to_numpy()) if col in boot.columns else {}


def drivers(td, est, pcols: pl.DataFrame, cols: list[str]) -> dict:
    """POST HOC (PH4), descriptive: Pearson r of the draft-year-centered player trait with each Combine column
    (complete cases), to read what a nonzero CGE is made of."""
    y = est["y"].copy()
    for s in np.unique(td.year):
        y[td.year == s] -= y[td.year == s].mean()
    p = pcols.sort("nfl_id")
    assert np.array_equal(p["nfl_id"].to_numpy(), td.nfl_id)
    return {c: {"r": corr(p[c].cast(pl.Float64).to_numpy(), y), "n": int(p[c].is_not_null().sum())} for c in cols}


def internal_trait(t: str, i: int, games_i: pl.DataFrame, players: pl.DataFrame) -> tuple[dict, dict]:
    spec = H.TRAITS[t]
    sets = H.info_sets(spec["matched"])
    pp = players.filter(pl.col("position") == spec["pos"])
    g = games_i.filter(pl.col("trait") == t).select("nfl_id", "game_id", "season", "value", "n_plays")
    td = H.trait_data(g, pp, sets)
    est = H.estimate(td, keep_oof=True)
    boot = H.bootstrap(td, N_BOOT, SEED + i, N_JOBS, BOOT_REPEATS)  # preregistered (B1)
    bff = H.bootstrap_fixed_forecasts(td, est, N_BOOT, SEED + i)  # post hoc PH5, same player draws
    ga = g.join(pp.select("nfl_id", "draft_year"), on="nfl_id").filter(pl.col("nfl_id").is_in(td.nfl_id.tolist()))
    k_half = {s: est["lambda"] / (1 - est["r2"][s]) if np.isfinite(est["r2"][s]) else np.nan for s in sets}
    boot, bff = ([b.with_columns([(pl.col("lambda") / (1 - pl.col(f"r2_{s}"))).alias(f"khalf_{s}") for s in sets])
                  for b in (boot, bff)])
    pcols = pp.filter(pl.col("nfl_id").is_in(td.nfl_id.tolist()))

    def two(col: str, point: float) -> dict:
        d = boot[col].to_numpy()
        pre = {**ci_of(boot, col), "median_draw": float(np.nanmedian(d)) if np.isfinite(d).any() else None,
               "share_draws_below_est": float(np.mean(d < point - 1e-12)) if np.isfinite(point) else None}
        return {"est": point, "ci95_prereg_bootstrap": pre, "ci95_fixed_forecast_PH5": ci_of(bff, col)}
    res = {
        "position": spec["pos"], "name": spec["name"], "unit": spec["unit"], "min_plays_per_game": spec["min_plays"],
        "per_game_value": spec["agg"], "matched_signature": spec["matched"],
        "n_players": est["n"], "n_games": est["n_games"], "mean_games": est["mean_games"],
        "by_class": dict(pcols.group_by("draft_year").len().sort("draft_year").iter_rows()),
        "variance": {k: {"est": est[k], **ci_of(boot, k),
                         "same_draws_in_PH5": bool(np.allclose(boot[k].to_numpy(), bff[k].to_numpy(), equal_nan=True))}
                     for k in ("tau2", "sigma2", "lambda", "rel")},
        "icc_single_game": est["icc"], "var_w": est["var_w"],
        "reliability_descriptive": descriptive_reliability(ga),
        "exposure": exposure(ga),
        "r2_cv": est["r2_cv"],
        "r2_true": {s: two(f"r2_{s}", est["r2"][s]) for s in sets},
        "cge": {s: two(f"cge_{s}", est["cge"][s]) for s in sets},
        "delta_cge": {k: two(f"delta_{k}", est["delta"][k]) for k in est["delta"]},
        "half_weight_game": {s: two(f"khalf_{s}", k_half[s]) for s in sets},
        "c3_coverage": {c: float(pcols[c].is_not_null().mean()) for c in sets["C3"] if c not in sets["C2"]},
        "post_hoc_drivers": drivers(td, est, pcols, sets["C3"]),
        "bootstrap": {"n": N_BOOT, "cv_repeats": BOOT_REPEATS, "seed": SEED + i,
                      "share_tau2_le_0": float((boot["tau2"] <= 0).mean())},
    }
    return res, {"td": td, "est": est, "boot": boot, "boot_ff": bff, "games": ga}


def _weighted_r2cv(td, ids: np.ndarray, X: np.ndarray, n_repeats: int, seed: int) -> float:
    """R2_CV of one information set on a bootstrap resample fitted as unique players with multiplicity weights
    (StandardScaler and RidgeCV sample_weight), so RidgeCV's inner leave-one-out drops a whole player."""
    n = len(td.nfl_id)
    n_i = np.bincount(td.g_player, minlength=n).astype(float)
    y = np.bincount(td.g_player, weights=td.g_value, minlength=n) / n_i
    cnt = np.bincount(ids, minlength=n).astype(float)
    u = np.flatnonzero(cnt > 0)
    w, yu, yr, Xu = cnt[u], y[u], td.year[u], X[u]
    dev = np.zeros(len(u))
    for s_ in np.unique(yr):
        m = yr == s_
        dev[m] = yu[m] - np.average(yu[m], weights=w[m])
    var_w = np.sum(w * dev ** 2) / (w.sum() - len(np.unique(yr)))
    P = np.full((n_repeats, len(u)), np.nan)
    for rep, te in H.group_folds(np.arange(len(u)), n_repeats, seed):
        tr = ~te
        m = linkage._model().fit(Xu[tr], yu[tr], standardscaler__sample_weight=w[tr], ridgecv__sample_weight=w[tr])
        P[rep, te] = m.predict(Xu[te])
    return float(np.mean(1 - (np.sum(w * (yu - P) ** 2, axis=1) / w.sum()) / var_w))


def bootstrap_bias_diagnostic(arts: dict, traits=("T1", "T4", "T6", "T8"), n_draws: int = 100,
                              repeats: int = 3) -> dict:
    """POST HOC (PH5): C2 R2_CV on the first n_draws player resamples of each trait's bootstrap under three schemes:
    duplicated rows refitted (B1), unique players with multiplicity weights refitted, and fixed forecasts (PH5)."""
    out = {}
    with threadpool_limits(1):
        for i, t in enumerate(H.TRAITS):
            if t not in traits:
                continue
            td, est = arts[t]["td"], arts[t]["est"]
            idx = np.random.default_rng(SEED + i).integers(0, len(td.nfl_id), (n_draws, len(td.nfl_id)))
            dup = [H.estimate(H.resample(td, ids), n_repeats=repeats, seed=b, groups=ids)["r2_cv"]["C2"]
                   for b, ids in enumerate(idx)]
            wtd = [_weighted_r2cv(td, ids, td.X["C2"], repeats, b) for b, ids in enumerate(idx)]
            ff = arts[t]["boot_ff"]["r2cv_C2"].to_numpy()[:n_draws]
            pt = est["r2_cv"]["C2"]
            out[t] = {"point_r2cv_C2": pt, **{k: {"median": float(np.median(v)), "share_below_point": float(np.mean(
                np.asarray(v) < pt))} for k, v in (("duplicated_rows_refit_B1", dup), ("weighted_refit", wtd),
                                                   ("fixed_forecasts_PH5", ff))}}
    return {"n_draws": n_draws, "cv_repeats": repeats, "traits": out}


def prereg_bootstrap_summary(internal: dict) -> dict:
    """How far the preregistered refit bootstrap (B1) sits from the point estimates (PH5 evidence)."""
    excl, shares, both_ok = [], [], []
    for t, v in internal.items():
        for block in ("r2_true", "cge", "delta_cge"):
            for s_, d in v[block].items():
                e, pre, ff = d["est"], d["ci95_prereg_bootstrap"], d["ci95_fixed_forecast_PH5"]
                if e is None or not np.isfinite(e):
                    continue
                outside = lambda c: (c.get("lo") is not None and e < c["lo"] - 1e-9) or (  # noqa: E731
                    c.get("hi") is not None and e > c["hi"] + 1e-9)
                if outside(pre):
                    excl.append(f"{t}-{block}-{s_}")
                both_ok.append(not outside(ff))
        for s_ in ("C1", "C2", "C3"):
            d = v["r2_true"][s_]
            if d["est"] is not None and d["est"] > 0:
                shares.append(d["ci95_prereg_bootstrap"]["share_draws_below_est"])
    return {"n_cells_prereg_ci_excludes_point": len(excl), "cells": excl,
            "share_draws_below_point_range_where_r2_gt_0": [min(shares), max(shares)] if shares else None,
            "n_cells_checked": len(both_ok), "n_cells_PH5_ci_contains_point": int(sum(both_ok))}


# =====================================================================================================
# 4a. Validation 1: internal crossover
# =====================================================================================================

def validation1(t: str, art: dict, cge_point: dict) -> dict:
    td, est, ga = art["td"], art["est"], art["games"]
    fw = H.first_k_windows(year_centered(ga).select("nfl_id", "game_id", "value"), 6, 8)
    yr_mean = {s: est["y"][td.year == s].mean() for s in np.unique(td.year)}
    fc = pl.DataFrame({"nfl_id": td.nfl_id, **{f"f_{s}": v - np.array([yr_mean[y] for y in td.year])
                                                for s, v in est["oof_mean"].items()}})
    d = fw.join(fc, on="nfl_id", how="inner").sort("nfl_id")
    X = d.select([f"X{k}" for k in range(1, 7)]).to_numpy()
    Fc = {s: d[f"f_{s}"].to_numpy() for s in est["oof_mean"]}
    tg = d["target"].to_numpy()

    def stats(idx):
        rg = np.array([np.corrcoef(X[idx, k], tg[idx])[0, 1] for k in range(6)])
        rc = {s: np.corrcoef(f[idx], tg[idx])[0, 1] for s, f in Fc.items()}
        return rg, rc

    rg, rc = stats(np.arange(d.height))
    rng = np.random.default_rng(SEED + 100)
    bi, bc = {s: [] for s in Fc}, {s: [] for s in Fc}
    brg = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, d.height, d.height)
        g_, c_ = stats(idx)
        brg.append(g_)
        for s in Fc:
            bi[s].append(H.integer_crossover(g_, c_[s]))
            bc[s].append(H.continuous_crossover(g_, c_[s]))
    brg = np.array(brg)
    # POST HOC (PH2): forecasts centered on the same folds' C0 forecast, which removes the leave-fold-out artifact
    # (a training-fold class mean moves opposite to the held-out players' values; it makes r(C0) strongly negative)
    ph = {}
    for s in [s for s in Fc if s != "C0"]:
        fph = Fc[s] - Fc["C0"]
        rph = np.corrcoef(fph, tg)[0, 1]
        ints = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, d.height, d.height)
            ints.append(H.integer_crossover(np.array([np.corrcoef(X[idx, k], tg[idx])[0, 1] for k in range(6)]),
                                            np.corrcoef(fph[idx], tg[idx])[0, 1]))
        lo_, hi_ = (int(v) for v in np.quantile(ints, [0.025, 0.975], method="inverted_cdf"))
        cc = H.ceil_cge(cge_point[s])
        ph[s] = {"r_combine": float(rph), "crossover_int": H.integer_crossover(rg, rph),
                 "crossover_int_ci95": [lo_, hi_], "crossover_continuous": H.continuous_crossover(rg, rph),
                 "agree": bool(lo_ <= cc <= hi_) if np.isfinite(cge_point[s]) else None}
    m = d["n_target"].to_numpy().astype(float)
    out = {"n_players": d.height, "median_target_games": float(d["n_target"].median()),
           "rel_target_model": float(np.mean(m / (m + est["lambda"]))),  # PH9: reliability of the target means
           "post_hoc_fold_centered_forecasts": ph,
           "r_games_k": {str(k + 1): {"est": float(rg[k]), "lo": float(np.quantile(brg[:, k], 0.025)),
                                      "hi": float(np.quantile(brg[:, k], 0.975))} for k in range(6)},
           "sets": {}}
    for s in Fc:
        ints = np.array(bi[s])
        lo, hi = (int(v) for v in np.quantile(ints, [0.025, 0.975], method="inverted_cdf"))
        cc = H.ceil_cge(cge_point[s])
        out["sets"][s] = {"r_combine": float(rc[s]), "crossover_int": H.integer_crossover(rg, rc[s]),
                          "crossover_int_ci95": [lo, hi], "crossover_int_7_means_gt6": True,
                          "crossover_continuous": H.continuous_crossover(rg, rc[s]),
                          "crossover_continuous_ci95": H.percentile_ci(np.array(bc[s])),
                          "formula_cge": cge_point[s], "formula_ceil": cc,
                          "agree": bool(lo <= cc <= hi) if np.isfinite(cge_point[s]) else None}
    return out


# =====================================================================================================
# 4b. Validation 2: prospective
# =====================================================================================================

def validation2(games_p: pl.DataFrame, players_p: pl.DataFrame) -> tuple[dict, pl.DataFrame, dict]:
    fits, rows, summ = {}, [], {}
    for t, spec in H.TRAITS.items():
        sets = H.info_sets(spec["matched"])
        pp = players_p.filter(pl.col("position") == spec["pos"])
        g = games_p.filter(pl.col("trait") == t).select("nfl_id", "game_id", "season", "value")
        gtr, gte = H.split_prospective(g, pp, TRAIN_SEASONS, TEST, TEST)
        assert gtr["season"].max() <= max(TRAIN_SEASONS) and set(gte["season"].unique().to_list()) <= {TEST}
        fit = H.fit_prospective(gtr, pp, sets)
        n_test = gte.group_by("nfl_id").len()
        summ[t] = {"n_train_players": int(fit.vc["n"]), "n_train_games": int(fit.vc["n_games"]),
                   "tau2": fit.vc["tau2"], "sigma2": fit.vc["sigma2"], "lambda": fit.vc["lambda"],
                   "rel": fit.vc["rel"], "r2_cv": fit.r2_cv, "r2_true": fit.r2,
                   "cge": {s: H.cge(fit.vc["lambda"], r) for s, r in fit.r2.items()},
                   "n_test_players_any_game": n_test.height,
                   "n_test_players_ge": {str(k + 3): int((n_test["len"] >= k + 3).sum()) for k in KS},
                   "included": bool(fit.vc["tau2"] > 0)}
        if not fit.vc["tau2"] > 0:
            continue
        fits[t] = fit
        f = H.in_season_forecasts(fit, gte, pp.filter(pl.col("draft_year") == TEST), ks=KS)
        rows.append(f.with_columns(trait=pl.lit(t)))
    return summ, pl.concat(rows, how="diagonal_relaxed"), fits


def paired(Fk: pl.DataFrame, a: str, b: str, seed: int, n_flips: int = N_FLIPS) -> dict:
    d = Fk.select("nfl_id", "trait", a, b).filter(pl.col(a).is_finite() & pl.col(b).is_finite())
    ids, di = H.player_diffs(d, a, b)
    p = H.sign_flip_p(di, n_flips, seed, "less")
    bs = H.pooled_bootstrap(d, [a, b], N_BOOT_V, seed)
    pa, pb = H.pooled_mean(d, a), H.pooled_mean(d, b)
    q = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]  # noqa: E731
    return {"a": a, "b": b, "n_players": int(len(ids)), "n_player_traits": d.height,
            "n_by_trait": dict(d.group_by("trait").len().sort("trait").iter_rows()),
            "pooled_a": pa, "pooled_b": pb, "diff": pa - pb, "ratio": pa / pb if pb > 0 else None,
            "diff_ci95": q(bs[a] - bs[b]), "ratio_ci95": q(bs[a] / bs[b]), "pooled_a_ci95": q(bs[a]),
            "pooled_b_ci95": q(bs[b]),
            "p_one_sided": p, "expected_sign": "negative (a < b)", "n_flips": n_flips, "n_boot": N_BOOT_V}


def validation2_tests(Fc: pl.DataFrame) -> dict:
    k1, k2 = Fc.filter(pl.col("k") == 1), Fc.filter(pl.col("k") == 2)
    prim = {"H1": paired(k1, "se_blend_C2", "se_games", SEED + 1), "H2": paired(k2, "se_blend_C2", "se_games", SEED + 2),
            "H3": paired(k1, "se_blend_C3", "se_blend_C2", SEED + 3)}
    for h, claim in zip(prim, [x["claim"] for x in PREREG["validation_2_prospective"]["primary_tests"]]):
        prim[h]["claim"] = claim
    qs = H.bh([prim[h]["p_one_sided"] for h in prim])
    for h, q in zip(prim, qs):
        prim[h]["q_bh"] = q
        prim[h]["pass"] = bool(q < 0.05 and prim[h]["diff"] < 0)
    lit = {"H1": paired(k1, "se_blendlit_C2", "se_games", SEED + 11),
           "H2": paired(k2, "se_blendlit_C2", "se_games", SEED + 12),
           "H3": paired(k1, "se_blendlit_C3", "se_blendlit_C2", SEED + 13)}
    for h, q in zip(lit, H.bh([lit[h]["p_one_sided"] for h in lit])):
        lit[h]["q_bh"] = q
    # pooled curves (descriptive): every method at every k on that k's common rows
    methods = ["se_games", *[f"se_comb_{s}" for s in H.SETS], *[f"se_blend_{s}" for s in H.SETS],
               "se_blendlit_C2", "se_blendlit_C3"]
    curves = {}
    for k in (0, *KS):
        d = Fc.filter(pl.col("k") == k)
        cols = [m for m in methods if not (k == 0 and m in ("se_games", "se_blendlit_C2", "se_blendlit_C3"))
                and not (k == 0 and m.startswith("se_blend_"))]
        d = d.select("nfl_id", "trait", *cols).drop_nulls()
        bs = H.pooled_bootstrap(d, cols, N_BOOT_V, SEED + 20 + k)
        curves[str(k)] = {"n_players": d["nfl_id"].n_unique(), "n_traits": d["trait"].n_unique(),
                          **{m: {"est": H.pooled_mean(d, m), "lo": float(np.quantile(bs[m], 0.025)),
                                 "hi": float(np.quantile(bs[m], 0.975))} for m in cols}}
    sec_k = {f"k{k}": {"blend_C2_vs_games": paired(Fc.filter(pl.col("k") == k), "se_blend_C2", "se_games", SEED + 30 + k),
                       "blend_C3_vs_blend_C2": paired(Fc.filter(pl.col("k") == k), "se_blend_C3", "se_blend_C2",
                                                      SEED + 40 + k)} for k in (3, 4)}
    per_trait = {}
    for t in sorted(Fc["trait"].unique().to_list(), key=lambda s: int(s[1:])):
        per_trait[t] = {}
        for k in KS:
            d = Fc.filter((pl.col("trait") == t) & (pl.col("k") == k))
            if d.height < 3:
                per_trait[t][str(k)] = {"n": d.height}
                continue
            per_trait[t][str(k)] = {
                "n": d.height, **{m: float(d[m].mean()) for m in ("se_games", "se_comb_C2", "se_blend_C2", "se_blend_C3")},
                "blend_C2_vs_games": paired(d, "se_blend_C2", "se_games", SEED + 50 + k, 20_000),
                "blend_C3_vs_blend_C2": paired(d, "se_blend_C3", "se_blend_C2", SEED + 60 + k, 20_000)}
    calib = {}
    for s in ("C2", "C3"):
        calib[s] = {}
        for k in (0, *KS):
            d = Fc.filter(pl.col("k") == k).select("nfl_id", "trait", f"cov_{s}").drop_nulls()
            calib[s][str(k)] = {"pooled_coverage": H.pooled_mean(d, f"cov_{s}"),
                                "raw_coverage": float(d[f"cov_{s}"].mean()), "n_player_traits": d.height,
                                "by_trait": dict(d.group_by("trait").agg(pl.col(f"cov_{s}").mean())
                                                 .sort("trait").iter_rows())}
    # POST HOC (PH3): is the Combine prior better than plain shrinkage to the position mean (C0 prior)?
    vs_c0 = {}
    for k in (0, *KS):
        d = Fc.filter(pl.col("k") == k)
        if k == 0:
            vs_c0["k0"] = {"comb_C2_vs_C0": paired(d, "se_comb_C2", "se_comb_C0", SEED + 70),
                           "comb_C3_vs_C0": paired(d, "se_comb_C3", "se_comb_C0", SEED + 71)}
        else:
            vs_c0[f"k{k}"] = {"blend_C2_vs_C0": paired(d, "se_blend_C2", "se_blend_C0", SEED + 72 + k),
                              "blend_C3_vs_C0": paired(d, "se_blend_C3", "se_blend_C0", SEED + 76 + k)}
    return {"primary": prim, "alpha": 0.05, "post_hoc_combine_prior_vs_position_mean": vs_c0,
            "sensitivity_prereg_literal_weight": lit, "pooled_curves": curves, "secondary_k3_k4": sec_k,
            "secondary_per_trait": per_trait, "calibration_80pct": calib}


# =====================================================================================================
# 4c. Post hoc (review response): per-trait sensitivities, simulation, validation re-reads
# =====================================================================================================

def usage_tables() -> tuple[pl.DataFrame, pl.DataFrame]:
    """From REG player_play: appearances (nfl_id, season, game_id with >= 1 row) and the WR deep-route share
    (GO/POST/CORNER/WHEEL among routes run, all seasons; in-season usage, not known before the season)."""
    reg = io.load("games").filter(pl.col("season_type") == "REG").select("game_id", "season", "week")
    pp = (pl.scan_parquet(DATA / "player_play.parquet").select("nfl_id", "game_id", "route_ran")
          .join(reg.lazy(), on="game_id", how="inner").collect())
    app = pp.select("nfl_id", "season", "game_id").unique().sort("nfl_id", "game_id")
    deep = (pp.filter(pl.col("route_ran").is_not_null()).group_by("nfl_id")
            .agg(deep_share=pl.col("route_ran").is_in(DEEP_ROUTES).mean().cast(pl.Float64), n_routes=pl.len()))
    return app, deep


def _est_summary(e: dict, sets) -> dict:
    return {"n": e["n"], "lambda": e["lambda"], "rel": e["rel"], "r2_cv": {s: e["r2_cv"][s] for s in sets},
            "r2_true": {s: e["r2"][s] for s in sets}, "cge": {s: e["cge"][s] for s in sets}}


def _boot_mean_ci(x: np.ndarray, rng, n_boot: int = 2000) -> list[float]:
    b = np.array([rng.choice(x, len(x)).mean() for _ in range(n_boot)])
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def stationarity(ga: pl.DataFrame, tau2: float, weeks: pl.DataFrame, seed: int) -> dict:
    """POST HOC (PH11). ga: the trait's analysis games (nfl_id, game_id, season, value, n_plays, draft_year)."""
    rng = np.random.default_rng(seed)
    tau = np.sqrt(tau2)
    ps = ga.group_by("nfl_id", "draft_year", "season").agg(m=pl.col("value").mean(), n=pl.len()).filter(
        pl.col("n") >= MIN_SEASON_GAMES)
    nxt = ps.join(ps.with_columns(season=pl.col("season") - 1), on=["nfl_id", "draft_year", "season"], suffix="_next")
    nxt = nxt.with_columns(d=(pl.col("m_next") - pl.col("m")) / tau)
    change = {}
    for (cls, s), grp in sorted(nxt.group_by("draft_year", "season"), key=lambda x: x[0]):
        x = grp.sort("nfl_id")["d"].to_numpy()  # group_by / join order is arbitrary; the bootstrap resamples by row
        if len(x) >= 5:
            change[f"{cls} class {s}->{s + 1}"] = {"mean_tau": float(x.mean()), "ci95": _boot_mean_ci(x, rng),
                                                  "n": int(len(x))}
    w = ga.join(weeks, on="game_id", how="left").with_columns(
        r=pl.col("value") - pl.col("value").mean().over("nfl_id", "season"),
        wkc=pl.col("week") - pl.col("week").mean().over("nfl_id", "season"))
    ids, r, wkc = w["nfl_id"].to_numpy(), w["r"].to_numpy(), w["wkc"].to_numpy().astype(float)

    def slope(idx):
        return float(np.sum(wkc[idx] * r[idx]) / np.sum(wkc[idx] ** 2) * 16 / tau)
    uid, inv = np.unique(ids, return_inverse=True)
    rows_of = [np.flatnonzero(inv == j) for j in range(len(uid))]
    bs = [slope(np.concatenate([rows_of[j] for j in rng.integers(0, len(uid), len(uid))])) for _ in range(1000)]
    # within player-season deviations: share common to one game, and noise by plays-per-game quartile
    game_share = H.variance_components(r, w["game_id"].to_numpy())["icc"]
    rp = (ga["value"] - ga.with_columns(pl.col("value").mean().over("nfl_id"))["value"]).to_numpy()
    npl = ga["n_plays"].to_numpy()
    q1, q3 = np.quantile(npl, [0.25, 0.75])
    return {"season_change_within_player": change,
            "within_season_trend_tau_per_16_weeks": {"est": slope(np.arange(len(r))),
                                                     "ci95": [float(np.quantile(bs, 0.025)),
                                                              float(np.quantile(bs, 0.975))]},
            "share_of_within_player_season_variance_common_to_a_game": float(game_share),
            "noise_ratio_lowest_vs_highest_plays_quartile": float(np.mean(rp[npl <= q1] ** 2) / np.mean(rp[npl >= q3] ** 2)),
            "plays_quartiles": [float(q1), float(q3)]}


def trait_posthoc(t: str, i: int, art: dict, internal_t: dict, players: pl.DataFrame, deep: pl.DataFrame,
                  weeks: pl.DataFrame, fit_summ: dict) -> dict:
    """POST HOC (PH7, PH8, PH11) for one trait, all on the frozen internal pipeline (estimate())."""
    spec = H.TRAITS[t]
    sets = H.info_sets(spec["matched"])
    pp = players.filter(pl.col("position") == spec["pos"])
    td, est, ga = art["td"], art["est"], art["games"]
    g = ga.select("nfl_id", "game_id", "season", "value", "n_plays")
    rel, sh = est["rel"], internal_t["reliability_descriptive"]["split_half_odd_even_games"]
    out = {"identified": bool(rel >= REL_MIN and sh is not None and sh >= REL_MIN),
           "identification_rule": f"rel >= {REL_MIN} and odd/even split-half >= {REL_MIN} (PLAN 4, criterion 5)",
           "rel": rel, "split_half": sh}
    out["cge_2023_24_seasons_fit"] = {s: fit_summ["cge"][s] for s in ("C1", "C2", "C3")}
    out["r2_true_2023_24_seasons_fit"] = {s: fit_summ["r2_true"][s] for s in ("C1", "C2", "C3")}
    # PH8: CGE against C0 under the same CV
    out["c0_relative"] = {}
    for s in ("C1", "C2", "C3"):
        r2 = H.relative_true_r2(est["r2_cv"][s], est["r2_cv"]["C0"], rel)
        out["c0_relative"][s] = {"r2_true": r2, "cge": H.cge(est["lambda"], r2)}
    out["r2_cv_C0"] = est["r2_cv"]["C0"]
    # PH7: leave one draft class out
    out["leave_one_class_out"] = {}
    for y in td.levels:
        e = H.estimate(H.trait_data(g, pp.filter(pl.col("draft_year") != y), sets))
        out["leave_one_class_out"][f"without_{y}"] = _est_summary(e, ("C1", "C2", "C3"))
    lo_ = [v["cge"]["C2"] for v in out["leave_one_class_out"].values()]
    out["leave_one_class_out_cge_C2_range"] = [float(np.nanmin(lo_)), float(np.nanmax(lo_))]
    # PH11: season-centred per-game values
    gs = g.with_columns(value=pl.col("value") - pl.col("value").mean().over("season"))
    out["season_centred"] = _est_summary(H.estimate(H.trait_data(gs, pp, sets)), ("C1", "C2", "C3"))
    # PH8: role and exposure references
    ids = pl.DataFrame({"nfl_id": td.nfl_id, "log_games": np.log(np.bincount(td.g_player).astype(float))})
    if spec["pos"] == "WR":
        role = deep.select("nfl_id", "deep_share")
        role_cols, role_note = ["deep_share"], "deep-route share (GO/POST/CORNER/WHEEL) over all REG routes: in-season usage"
    else:
        lv = sorted(pp.join(ids, on="nfl_id", how="semi")["nfl_position"].drop_nulls().unique().to_list())
        role = pp.select("nfl_id", *[(pl.col("nfl_position") == v).cast(pl.Float64).alias(f"role_{v}") for v in lv[1:]])
        role_cols = [f"role_{v}" for v in lv[1:]]
        role_note = ("NFL roster position (players.csv nfl_position: " + "/".join(lv) + "); may be the latest roster "
                     "designation rather than the one known on draft day")
    pe = pp.join(ids, on="nfl_id", how="left").join(role, on="nfl_id", how="left")
    refs = {}
    for name, cols, note in (("role", role_cols, role_note), ("exposure", ["log_games"], "log qualifying games")):
        rs = {"C0": None, "REF": cols, "REF+C2": sets["C2"] + cols}
        e = H.estimate(H.trait_data(g, pe, rs))
        refs[name] = {"reference": note, "r2_cv": e["r2_cv"], "r2_true": e["r2"], "cge_ref": e["cge"]["REF"],
                      "cge_ref_plus_C2": e["cge"]["REF+C2"], "cge_C2": est["cge"]["C2"],
                      "combine_beyond_ref": e["cge"]["REF+C2"] - e["cge"]["REF"]}
    if spec["pos"] == "WR":
        y = est["y"].copy()
        for s_ in np.unique(td.year):
            y[td.year == s_] -= y[td.year == s_].mean()
        dsh = pl.DataFrame({"nfl_id": td.nfl_id}).join(deep, on="nfl_id", how="left")["deep_share"].to_numpy()
        refs["role"]["r_trait_deep_share"] = corr(y, dsh)
    out["references"] = refs
    out["stationarity_and_game_unit"] = stationarity(ga, est["tau2"], weeks, SEED + 200 + i)
    return out


# ---- PH6: simulation at the real designs

def _sim_td(td: H.TraitData) -> H.TraitData:
    return H.TraitData(nfl_id=td.nfl_id, year=td.year, X={"C0": None, "C2": td.X["C2"]}, g_player=td.g_player,
                       g_value=td.g_value, levels=td.levels)


def _sim_cell(t: str, i: int, td: H.TraitData, lam: float, gi: int, r2pop: float, reps: int, n_ph5: int) -> list:
    tds = _sim_td(td)
    Z = H.signal_matrix(td.X["C2"], len(td.levels) - 1)
    rows = []
    with threadpool_limits(1):
        for r in range(reps):
            rng = np.random.default_rng([SEED, i, gi, r])
            sim, theta = H.simulate_values(tds, Z, lam, r2pop, rng)
            e = H.estimate(sim, keep_oof=True)
            row = {"trait": t, "grid": gi, "r2pop": r2pop, "rep": r, "lam_hat": e["lambda"], "rel_hat": e["rel"],
                   "r2cv": e["r2_cv"]["C2"], "r2_true": e["r2"]["C2"], "cge": e["cge"]["C2"],
                   "r2_oracle": H.oracle_r2(theta, e["oof"]["C2"]), "ph5_lo": None, "ph5_hi": None}
            if r < n_ph5:
                ci = H.percentile_ci(H.bootstrap_fixed_forecasts(sim, e, 400, int(rng.integers(2**31)))["cge_C2"]
                                     .to_numpy())
                row["ph5_lo"], row["ph5_hi"] = ci["lo"], ci["hi"]
            rows.append(row)
    return rows


def sim_belt(arts: dict, internal: dict) -> pl.DataFrame:
    from joblib import Parallel, delayed
    jobs = [(t, i, arts[t]["td"], internal[t]["variance"]["lambda"]["est"], gi, g)
            for i, t in enumerate(H.TRAITS) for gi, g in enumerate(SIM_GRID)]
    res = Parallel(n_jobs=N_JOBS)(delayed(_sim_cell)(t, i, td, lam, gi, g, SIM_REPS, SIM_PH5_REPS)
                                  for t, i, td, lam, gi, g in jobs)
    return pl.DataFrame([r for rows in res for r in rows], infer_schema_length=None)


def _star(belt_t: pl.DataFrame, lam: float) -> pl.DataFrame:
    """Per grid point: the CV estimand CGE* (from the mean oracle R2), the population CGE, and estimator summaries."""
    s = (belt_t.group_by("grid", "r2pop").agg(r2_oracle=pl.col("r2_oracle").mean(), cge_mean=pl.col("cge").mean(),
                                              cge_median=pl.col("cge").median(),
                                              p_zero=(pl.col("cge") <= 1e-12).mean().cast(pl.Float64),
                                              n=pl.len()).sort("grid"))
    return s.with_columns(cge_star=pl.col("r2_oracle").map_elements(
        lambda r: H.cge(lam, float(np.clip(r, 0, H.R2_MAX))), return_dtype=pl.Float64),
        cge_pop=pl.col("r2pop").map_elements(lambda r: H.cge(lam, r), return_dtype=pl.Float64))


def _bound(v: list, fill: float) -> np.ndarray:
    """Interval ends as floats, None (unbounded / undefined) -> fill."""
    return np.array([fill if x is None or not np.isfinite(x) else x for x in v], float)


def _covered(lo: list, hi: list, x) -> np.ndarray:
    return (_bound(lo, -np.inf) <= np.asarray(x) + 1e-9) & (np.asarray(x) <= _bound(hi, np.inf) + 1e-9)


def belt_summary(belt: pl.DataFrame, internal: dict) -> dict:
    out, cov_rows = {}, []
    for t in H.TRAITS:
        lam = internal[t]["variance"]["lambda"]["est"]
        b = belt.filter(pl.col("trait") == t)
        st = _star(b, lam)
        draws = [b.filter(pl.col("grid") == gi)["cge"].to_numpy() for gi in st["grid"].to_list()]
        obs = internal[t]["cge"]["C2"]["est"]
        bounds = H.calibrated_bounds(obs, st["cge_star"].to_numpy(), draws)
        bpop = H.calibrated_bounds(obs, st["cge_pop"].to_numpy(), draws)
        star_of = dict(zip(st["grid"].to_list(), st["cge_star"].to_list()))
        ph = b.filter(pl.col("rep") < SIM_PH5_REPS).with_columns(
            cge_star=pl.col("grid").replace_strict(star_of, return_dtype=pl.Float64))
        cv = _covered(ph["ph5_lo"].to_list(), ph["ph5_hi"].to_list(), ph["cge_star"].to_numpy())
        pos = ph["cge_star"].to_numpy() > 0
        cov_rows.append((t, cv[pos]))
        out[t] = {"observed_cge_C2": obs, "lambda_used": lam,
                  "calibrated_one_sided_95": {"lower": bounds["lower"], "upper": bounds["upper"]},
                  "calibrated_one_sided_95_population_signal": {"lower": bpop["lower"], "upper": bpop["upper"]},
                  "grid": st.select("r2pop", "r2_oracle", "cge_star", "cge_pop", "cge_mean", "cge_median",
                                    "p_zero", "n").to_dicts(),
                  "p_estimate_le_observed": bounds["p_le_obs"], "p_estimate_ge_observed": bounds["p_ge_obs"],
                  "ph5_coverage_where_estimand_gt_0": float(cv[pos].mean()) if pos.any() else None,
                  "ph5_n": int(pos.sum())}
    allc = np.concatenate([c for _, c in cov_rows])
    by = {t: float(c.mean()) for t, c in cov_rows if len(c)}
    return {"traits": out, "ph5_coverage_pooled": float(allc.mean()), "ph5_coverage_n": int(len(allc)),
            "ph5_coverage_range_by_trait": [min(by.values()), max(by.values())],
            "design": f"grid {list(SIM_GRID)} (population R2), {SIM_REPS} datasets per point, PH5 on the first "
                      f"{SIM_PH5_REPS}; signal on 3 random C2 columns; games iid N(theta, lambda_hat)",
            "estimand": "CGE* = lambda R2* / (1 - R2*), R2* = the mean true-score R2 the CV forecasts achieve "
                        "(the Combine as learned by the ridge from 80% of the trait's players), clipped at 0. The "
                        "_population_signal bounds map the same Neyman set to the planted population CGE "
                        "(lambda R2pop / (1 - R2pop)): what a perfectly learned linear Combine signal of this "
                        "simulated form could be worth."}


def _cov_dataset(t: str, i: int, td: H.TraitData, lam: float, r2pop: float, d: int) -> dict:
    tds = _sim_td(td)
    Z = H.signal_matrix(td.X["C2"], len(td.levels) - 1)
    rng = np.random.default_rng([SEED, 99, i, d])
    with threadpool_limits(1):
        sim, theta = H.simulate_values(tds, Z, lam, r2pop, rng)
        e = H.estimate(sim, keep_oof=True)
        b1 = H.percentile_ci(H.bootstrap(sim, COV_B1_RESAMPLES, int(rng.integers(2**31)), 1, H.N_REPEATS)["cge_C2"]
                             .to_numpy())
        ph5 = H.percentile_ci(H.bootstrap_fixed_forecasts(sim, e, 400, int(rng.integers(2**31)))["cge_C2"].to_numpy())
    return {"trait": t, "dataset": d, "r2pop": r2pop, "cge": e["cge"]["C2"], "r2_oracle": H.oracle_r2(theta, e["oof"]["C2"]),
            "b1_lo": b1["lo"], "b1_hi": b1["hi"], "ph5_lo": ph5["lo"], "ph5_hi": ph5["hi"]}


def sim_coverage(arts: dict, internal: dict, belt_sum: dict) -> pl.DataFrame:
    """B1 vs PH5 coverage on the COV_DESIGNS designs, at the grid point whose CGE* is closest to the observed
    CGE(C2) (truth = estimate)."""
    from joblib import Parallel, delayed
    jobs = []
    for t in COV_DESIGNS:
        i = list(H.TRAITS).index(t)
        grid = belt_sum["traits"][t]["grid"]
        obs = internal[t]["cge"]["C2"]["est"]
        r2pop = min(grid, key=lambda g: abs(g["cge_star"] - obs))["r2pop"]
        jobs += [(t, i, arts[t]["td"], internal[t]["variance"]["lambda"]["est"], r2pop, d) for d in range(COV_DATASETS)]
    res = Parallel(n_jobs=N_JOBS)(delayed(_cov_dataset)(*j) for j in jobs)
    return pl.DataFrame(res, infer_schema_length=None)


def coverage_summary(cov: pl.DataFrame, internal: dict) -> dict:
    out = {}
    for t in COV_DESIGNS:
        c = cov.filter(pl.col("trait") == t)
        lam = internal[t]["variance"]["lambda"]["est"]
        star = H.cge(lam, float(np.clip(c["r2_oracle"].mean(), 0, H.R2_MAX)))
        res = {"r2pop": float(c["r2pop"][0]), "cge_star": star, "n_datasets": c.height,
               "mean_cge_hat": float(c["cge"].mean()), "share_cge_hat_0": float((c["cge"] <= 1e-12).mean())}
        for k in ("b1", "ph5"):
            lo, hi = c[f"{k}_lo"].to_list(), c[f"{k}_hi"].to_list()
            cv = _covered(lo, hi, np.full(c.height, star))
            res[k] = {"coverage": float(cv.mean()), "se": float(np.sqrt(cv.mean() * (1 - cv.mean()) / len(cv))),
                      "share_upper_below_truth": float(np.mean(_bound(hi, np.inf) < star - 1e-9)),
                      "median_width": float(np.median(_bound(hi, np.inf) - _bound(lo, 0.0)))}
        out[t] = res
    out["note"] = (f"B1 here: {COV_B1_RESAMPLES} resamples x {H.N_REPEATS} CV repeats per simulated dataset (the real "
                   "run: 1,000 x 20); PH5: 400 resamples. Nominal coverage 0.95 for the CV estimand CGE*.")
    return out


# ---- PH9: validation 1 re-read

def validation1_posthoc(v1: dict, internal: dict) -> dict:
    cells, out = [], {}
    for t, r in v1.items():
        lam = internal[t]["variance"]["lambda"]["est"]
        rt = r["rel_target_model"]
        out[t] = {"rel_target_model": rt, "r_games_k_model": {str(k): float(np.sqrt(k / (k + lam) * rt))
                                                              for k in range(1, 7)}, "sets": {}}
        for s in ("C1", "C2", "C3"):
            v, ph = r["sets"][s], r["post_hoc_fold_centered_forecasts"][s]
            imp = {}
            for name, rc in (("frozen", v["r_combine"]), ("ph2_centred", ph["r_combine"])):
                r2 = float(np.clip(max(rc, 0.0) ** 2 / rt, 0, H.R2_MAX))
                imp[name] = {"r_combine": rc, "implied_r2_true": r2, "implied_cge": H.cge(lam, r2)}
            cc = v["crossover_continuous_ci95"]
            f = v["formula_cge"]
            literal = bool(cc["lo"] is not None and cc["lo"] - 1e-9 <= f and (cc["hi"] is None or f <= cc["hi"] + 1e-9))
            lo, hi = v["crossover_int_ci95"]
            kind = ("could not disagree (interval 1 to > 6)" if [lo, hi] == [1, 7] else
                    "one-sided: fails only if the Combine reliably beats 1 game" if v["formula_ceil"] == 1 and lo == 1
                    else "informative")
            cells.append(kind)
            out[t]["sets"][s] = {"formula_cge": f, "implied": imp, "literal_prereg_check_continuous": literal,
                                 "cell_kind": kind}
    lit = [c["literal_prereg_check_continuous"] for r in out.values() for c in r["sets"].values()]
    return {"traits": out, "n_cells": len(cells),
            "cell_kinds": {k: cells.count(k) for k in sorted(set(cells))},
            "literal_prereg_check_agree": int(sum(lit)),
            "reading": "Not contradicted, but a weak check: no cell could flag a formula that overstates the "
                       "Combine, and cells above one game could not disagree at all."}


# ---- PH10: validation 2 decomposition

def validation2_posthoc(Fc: pl.DataFrame, v2: dict, fits: dict, app: pl.DataFrame, games_p: pl.DataFrame,
                        players_p: pl.DataFrame, identified: dict) -> dict:
    curves, out = v2["pooled_curves"], {}
    out["shrinkage_share_of_blend_gain"] = {
        str(k): (curves[str(k)]["se_games"]["est"] - curves[str(k)]["se_blend_C0"]["est"])
        / (curves[str(k)]["se_games"]["est"] - curves[str(k)]["se_blend_C2"]["est"]) for k in KS}
    out["blend_C0_vs_games"] = {f"k{k}": paired(Fc.filter(pl.col("k") == k), "se_blend_C0", "se_games", SEED + 80 + k)
                                for k in (1, 2)}
    ph3 = v2["post_hoc_combine_prior_vs_position_mean"]
    keys = [(k, n) for k, d in ph3.items() for n in d]
    qs = H.bh([ph3[k][n]["p_one_sided"] for k, n in keys])
    out["ph3_bh_q_over_10"] = {f"{k}:{n}": q for (k, n), q in zip(keys, qs)}
    drv = {}
    for k, a, b in ((0, "se_comb_C2", "se_comb_C0"), (1, "se_blend_C2", "se_blend_C0")):
        d = Fc.filter(pl.col("k") == k).select("trait", a, b).drop_nulls()
        per = d.group_by("trait").agg(diff=(pl.col(a) - pl.col(b)).mean(), n=pl.len())
        T = per.height
        drv[f"k{k}"] = {r["trait"]: {"mean_diff": r["diff"], "contribution_to_pooled": r["diff"] / T, "n": r["n"]}
                        for r in per.sort("trait").iter_rows(named=True)}
        keep = [t for t in identified if identified[t]]
        dk = Fc.filter((pl.col("k") == k) & pl.col("trait").is_in(keep))
        drv[f"k{k}_identified_traits_only"] = paired(dk, a, b, SEED + 90 + k)
    out["ph3_drivers"] = drv
    # calibration per trait and observed / expected C0 error before any game
    cal = {}
    for t, fit in fits.items():
        d0 = Fc.filter((pl.col("trait") == t) & (pl.col("k") == 0))
        lam = fit.vc["lambda"]
        expected = float(np.mean(1 + lam / d0["m"].to_numpy()))
        tau = np.sqrt(fit.vc["tau2"])
        shift = ((d0["target"] - d0["mu_C0"]) / tau).to_numpy()
        mu2 = d0["mu_C2"].to_numpy()
        cal[t] = {"coverage80_C2": {str(k): v2["calibration_80pct"]["C2"][str(k)]["by_trait"].get(t) for k in (0, 1, 2)},
                  "observed_over_expected_C0_error_k0": float(d0["se_comb_C0"].mean() / expected),
                  "class_2025_shift_tau": {"mean": float(shift.mean()),
                                           "se": float(shift.std(ddof=1) / np.sqrt(len(shift))), "n": len(shift)},
                  "c2_prior_spread_tau": float(np.std(mu2) / tau),
                  "c2_prior_flat": bool(np.std(mu2) / tau < 0.1)}
    out["per_trait"] = cal
    # share of 2025 rookies' REG appearances that are qualifying games
    q = {}
    for t, spec in H.TRAITS.items():
        rook = players_p.filter((pl.col("draft_year") == TEST) & (pl.col("position") == spec["pos"])).select("nfl_id")
        a = app.filter(pl.col("season") == TEST).join(rook, on="nfl_id", how="semi")
        qg = games_p.filter((pl.col("trait") == t) & (pl.col("split") == "test"))
        nq = qg.group_by("nfl_id").len()["len"].to_numpy()
        q[t] = {"appearances": a.height, "qualifying_games": qg.height,
                "share": qg.height / a.height if a.height else None,
                "rookies_with_appearances": a["nfl_id"].n_unique(),
                "rookies_with_ge_4_qualifying_games": int(np.sum(nq >= 4))}
    out["qualifying_share_2025_rookies"] = q
    return out

def weight_curves(internal: dict, arts: dict, identified: dict, kmax: int = 17) -> dict:
    ks = np.arange(kmax + 1)
    out = {}
    for t, r in internal.items():
        lam = r["variance"]["lambda"]["est"]
        boot = arts[t]["boot"]  # preregistered B1 draws (PH5 revised after review)
        out[t] = {"k": ks.tolist(), "k_unit": f"qualifying games (>= {H.TRAITS[t]['min_plays']} plays each)",
                  "fit": "internal fit, 2023-25 classes and seasons (not the 2023-24 fit tested in validation 2)",
                  "band": "95% preregistered player bootstrap (B1)", "identified": identified[t],
                  "c2_prior_equals_position_average": bool(r["r2_true"]["C2"]["est"] == 0)}
        for s in ("C0", "C2", "C3"):
            r2 = r["r2_true"][s]["est"]
            if not (np.isfinite(lam) and np.isfinite(r2)):
                continue
            w = H.blend_weight(ks, lam, r2)
            lb, r2b = boot["lambda"].to_numpy(), boot[f"r2_{s}"].to_numpy()
            ok = np.isfinite(lb) & np.isfinite(r2b)
            W = np.array([H.blend_weight(ks, a, b) for a, b in zip(lb[ok], r2b[ok])])
            out[t][s] = {"w": w.tolist(), "lo": np.quantile(W, 0.025, axis=0).tolist(),
                         "hi": np.quantile(W, 0.975, axis=0).tolist(),
                         "half_weight_game": float(lam / (1 - r2))}
    return out


def player_names() -> dict:
    p = io.load("players").select("nfl_id", "display_name", "draft_overall_pick")
    return {r["nfl_id"]: r for r in p.iter_rows(named=True)}


def case_studies(games_p: pl.DataFrame, players_p: pl.DataFrame, fits: dict) -> list[dict]:
    names = player_names()
    out = []
    for pos, t in (("WR", "T1"), ("DL", "T6"), ("OL", "T9"), ("DB", "T11")):
        fit = fits[t]
        lam, r2 = fit.lam, fit.r2["C2"]
        pp = players_p.filter((pl.col("position") == pos) & (pl.col("draft_year") == TEST))
        g = games_p.filter((pl.col("trait") == t) & (pl.col("split") == "test")).sort("nfl_id", "game_id")
        n = g.group_by("nfl_id").agg(n=pl.len(), season_mean=pl.col("value").mean()).filter(pl.col("n") >= 10)
        pp = pp.join(n, on="nfl_id", how="inner").sort("nfl_id")
        pri = H.prior_means(fit, pp)
        pp = pp.with_columns(prior=pl.Series(pri["C2"]), prior_c0=pl.Series(pri["C0"])).with_columns(
            gap=(pl.col("prior") - pl.col("season_mean")).abs())
        ranked = pp.sort("gap", descending=True)
        pick = [("largest", r) for r in ranked.head(2).iter_rows(named=True)] + \
               [("smallest", r) for r in ranked.tail(2).reverse().iter_rows(named=True)]
        sd_prior = float(np.sqrt((1 - r2) * fit.vc["tau2"]))
        for kind, r in pick:
            v = g.filter(pl.col("nfl_id") == r["nfl_id"])["value"].to_numpy()
            ks = np.arange(1, len(v) + 1)
            run = np.cumsum(v) / ks
            out.append({"position": pos, "trait": t, "rule": f"{kind} |C2 prior - season mean|",
                        "nfl_id": r["nfl_id"], "name": names[r["nfl_id"]]["display_name"],
                        "draft_pick": names[r["nfl_id"]]["draft_overall_pick"], "nfl_position": r["nfl_position"],
                        "n_games": len(v), "prior_C2": r["prior"], "prior_C0": r["prior_c0"],
                        "cge_C2_2023_24_fit": H.cge(lam, r2), "season_mean": r["season_mean"], "abs_gap": r["gap"],
                        "prior_sd": sd_prior,
                        "z_predictive": (r["season_mean"] - r["prior"]) / np.sqrt(sd_prior ** 2 + fit.vc["sigma2"] / len(v)),
                        "game_values": v.tolist(), "running_mean": run.tolist(),
                        "blend_C2": H.blend(r["prior"], run, ks, lam, r2).tolist(),
                        "weight_C2": H.blend_weight(ks, lam, r2).tolist()})
    return out


def surprises(games_p: pl.DataFrame, players_p: pl.DataFrame, fits: dict, top: int = 3) -> list[dict]:
    names = player_names()
    out = []
    for t, fit in fits.items():
        spec = H.TRAITS[t]
        pp = players_p.filter((pl.col("position") == spec["pos"]) & (pl.col("draft_year") == TEST))
        g = games_p.filter((pl.col("trait") == t) & (pl.col("split") == "test"))
        n = g.group_by("nfl_id").agg(n=pl.len(), season_mean=pl.col("value").mean()).filter(pl.col("n") >= 10)
        pp = pp.join(n, on="nfl_id", how="inner").sort("nfl_id")
        if pp.height == 0:
            continue
        r2, tau2, sigma2 = fit.r2["C2"], fit.vc["tau2"], fit.vc["sigma2"]
        pri = H.prior_means(fit, pp)
        pp = pp.with_columns(prior=pl.Series(pri["C2"]), prior_c0=pl.Series(pri["C0"])).with_columns(
            z=(pl.col("season_mean") - pl.col("prior")) / np.sqrt((1 - r2) * tau2),
            z_pred=(pl.col("season_mean") - pl.col("prior")) / ((1 - r2) * tau2 + sigma2 / pl.col("n")).sqrt(),
            z_c0=(pl.col("season_mean") - pl.col("prior_c0")) / np.sqrt(tau2))
        for r in pp.sort(pl.col("z").abs(), descending=True).head(top).iter_rows(named=True):
            out.append({"trait": t, "trait_name": spec["name"], "position": spec["pos"], "nfl_id": r["nfl_id"],
                        "name": names[r["nfl_id"]]["display_name"], "draft_pick": names[r["nfl_id"]]["draft_overall_pick"],
                        "nfl_position": r["nfl_position"], "n_games": r["n"], "prior_C2": r["prior"],
                        "prior_C0": r["prior_c0"], "season_mean": r["season_mean"],
                        "z_prior_sd": r["z"], "z_predictive": r["z_pred"], "z_vs_position_average_C0": r["z_c0"],
                        "cge_C2_2023_24_fit": H.cge(fit.lam, r2)})
    return out


# =====================================================================================================
# Figures
# =====================================================================================================
SET_LABEL = {"C0": "Position average only", "C1": "Stopwatch + measurements (C1)", "C2": "+ draft slot (C2)",
             "C3": "+ Combine tracking (C3)"}
AXIS_LABEL = {  # what the case-study y axes measure (direction cue; centred traits say what they are centred on)
    "T1": "Top speed, yd/s\n(higher = faster)", "T6": "First-second rush distance, yd\n(alignment-adjusted; higher = quicker)",
    "T9": "Set accel. vs T/G/C average, yd/s²\n(higher = quicker set)",
    "T11": "Retreat speed vs alignment x coverage\naverage, yd/s (higher = faster)"}
FOOT = dict(fontsize=8.5, color=INK2)


def _ends(est, lo, hi, a, b):
    """Interval clipped to [a, b]; flags for arrowheads where it continues."""
    return max(lo if lo is not None else a, a), min(hi if hi is not None else b, b), \
        lo is not None and lo < a, hi is None or hi > b


def _interval(ax, est, lo, hi, y, color, a, b, marker="o", ms=6.5, label_value=False):
    """Dot + interval line clipped to the axis, arrowheads where the interval continues (nothing if undefined)."""
    if est is None or not np.isfinite(est):
        return
    x0, x1, cut0, cut1 = _ends(est, lo, hi, a, b)
    ax.plot([x0, x1], [y, y], color=color, lw=1.8, solid_capstyle="round", zorder=2)
    if cut1:
        ax.plot(b, y, marker=">", ms=5, color=color, zorder=2, clip_on=False)
    if cut0:
        ax.plot(a, y, marker="<", ms=5, color=color, zorder=2, clip_on=False)
    ax.plot(min(max(est, a), b), y, marker=marker, ms=ms, color=color, mec=SURF, mew=1.2, zorder=3, ls="none")
    if label_value:
        ax.text(max(x1, min(max(est, a), b)) + 0.01 * (b - a), y, f"{est:.2f}", ha="left", va="center",
                fontsize=8, color=INK, bbox=dict(boxstyle="square,pad=0.1", fc=SURF, ec="none"), zorder=4)


def fig_cge(out: dict, path: Path) -> None:
    traits, tab = list(H.TRAITS), {r["trait"]: r for r in out["cge_table"]}
    ypos, y = {}, 0.0
    for i, t in enumerate(traits):
        if i and H.TRAITS[t]["pos"] != H.TRAITS[traits[i - 1]]["pos"]:
            y += 0.6
        ypos[t] = y
        y += 1.15
    xmax, dlim = 4.0, 1.5
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(11, 7.4), gridspec_kw={"width_ratios": [2.3, 1], "wspace": 0.08},
                                  sharey=True)
    off = {"C1": -0.3, "C2": 0.0, "C3": 0.3}
    for t in traits:
        r = tab[t]
        lam = r["lambda"]["est"]
        ax.text(-0.02, ypos[t], "inf" if lam is None else f"{lam:.1f}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=8.5, color=INK2)
        if not r["identified"]:
            c2 = r["cge_C2"]
            hi = "unbounded" if c2["hi"] is None else f"{c2['hi']:.1f}" if c2["hi"] < 10 else f"{c2['hi']:.0f}"
            ax.text(0.03, ypos[t], f"Not identified (reliability {r['rel_player_mean']:.2f}, split-half "
                    f"{r['split_half']:.2f}): C2 point {c2['est']:.1f} games, 95% interval 0 to {hi}",
                    va="center", fontsize=8.5, color=INK2, style="italic")
            axd.text(0.0, ypos[t], "-", va="center", ha="center", fontsize=9, color=INK2)
            continue
        for s_ in ("C1", "C2", "C3"):
            c = r[f"cge_{s_}"]
            _interval(ax, c["est"], c["lo"], c["hi"], ypos[t] + off[s_], RAMP[s_], -0.05, xmax, MARK[s_],
                      label_value=(s_ == "C2"))
        u, lo_ = r["cge_C2_calibrated_upper95"], r["cge_C2_calibrated_lower95"]
        inside = u is not None and u <= xmax
        ax.plot(min(u, xmax) if inside else xmax, ypos[t], marker="|" if inside else ">", ms=11 if inside else 7,
                mew=2, color=RAMP["C2"], zorder=3, clip_on=False)
        if lo_ is not None and lo_ > 0:
            ax.plot(lo_, ypos[t], marker="|", ms=11, mew=2, color=RAMP["C2"], zorder=3)
        d = r["delta_cge_tracking"]
        _interval(axd, d["est"], d["lo"], d["hi"], ypos[t], ORANGE, -dlim, dlim, "o")
    ax.text(-0.02, -0.95, "λ", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9,
            color=INK2, style="italic")
    ax.set_xlim(-0.05, xmax)
    ax.set_xticks(np.arange(0, xmax + 0.1, 0.5))
    ax.set_yticks([ypos[t] for t in traits])
    ax.set_yticklabels([f"{H.TRAITS[t]['pos']}   {H.TRAITS[t]['short']}" for t in traits], fontsize=9)
    for lab, t in zip(ax.get_yticklabels(), traits):
        lab.set_color(INK if tab[t]["identified"] else MUTED)
    ax.tick_params(axis="y", length=0, pad=30)
    ax.set_ylim(y - 0.5, -1.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Qualifying games of his own NFL tracking that forecast the trait as well as the Combine does")
    handles = [Line2D([], [], color=RAMP[s_], marker=MARK[s_], lw=1.8, ms=6.5, mec=SURF, label=SET_LABEL[s_])
               for s_ in ("C1", "C2", "C3")]
    handles.append(Line2D([], [], color=RAMP["C2"], marker="|", ms=11, mew=2, ls="none",
                          label="C2: simulation-calibrated one-sided 95% bounds (> = beyond the axis)"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=8.5, handlelength=1.8,
              borderaxespad=0.3)
    axd.set_xlim(-dlim, dlim)
    axd.axvline(0, color=BASE, lw=1.0, zorder=1)
    axd.grid(axis="y", visible=False)
    axd.tick_params(axis="y", length=0)
    axd.set_xlabel("Games added by Combine tracking (C3 - C2)")
    axd.set_title("Tracking's increment", fontsize=10.5, pad=34)
    fig.suptitle("What the Combine is worth, in games of a player's own tracking", x=0.01, ha="left", y=0.985,
                 fontsize=12, weight="bold")
    above = [f"{H.TRAITS[t]['pos']} {H.TRAITS[t]['short'].lower()}" for t in traits
             if tab[t]["identified"] and (tab[t]["cge_C2_calibrated_lower95"] or 0) > 0]
    cov = out["post_hoc_review"]["simulation"]["interval_coverage"]
    b1cov = [cov[t]["b1"]["coverage"] for t in COV_DESIGNS]
    c_lo, c_hi = min(b1cov), max(b1cov)
    fig.text(0.01, 0.955, "Point estimate (value printed for C2) and nominal 95% preregistered player-bootstrap "
             f"interval (simulated coverage {c_lo:.0%}-{c_hi:.0%}): every one includes 0 games.\nTicks: "
             "simulation-calibrated one-sided 95% bounds for C2" + (f" (only {' and '.join(above)} sit above 0)"
                                                                     if above else "") +
             ". The estimand is the Combine as learned by a ridge from ~50-85 same-position players.", va="top",
             **FOOT)
    fig.text(0.01, 0.012, "λ = between-game noise / between-player signal: qualifying games until his own average is "
             "half signal. Not identified: reliability or odd/even split-half below 0.5 (PLAN criterion 5).\n"
             "Arrowheads: the interval continues beyond the axis. Bootstrap intervals can sit below their point "
             "estimate (each resample relearns the ridge from ~63% distinct players).", **FOOT)
    fig.subplots_adjust(left=0.19, right=0.985, top=0.82, bottom=0.14)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_weights(out: dict, path: Path) -> None:
    wc, tab = out["deliverables"]["weight_curves"], {r["trait"]: r for r in out["cge_table"]}
    traits = list(H.TRAITS)
    fig, axes = plt.subplots(3, 4, figsize=(11, 7.9), sharex=True, sharey=True)
    for ax, t in zip(axes.flat, traits):
        w, ok = wc[t], tab[t]["identified"]
        k = np.array(w["k"])
        a = 1.0 if ok else 0.35
        if "C0" in w:
            ax.plot(k, w["C0"]["w"], color=MUTED, lw=1.4, alpha=a, label="Position-average prior (C0)")
        if "C2" in w:
            ax.fill_between(k, w["C2"]["lo"], w["C2"]["hi"], color=RAMP["C2"], alpha=0.13 * a, lw=0)
            ax.plot(k, w["C2"]["w"], color=RAMP["C2"], lw=2, alpha=a, label="Stopwatch + draft slot prior (C2)")
        if "C3" in w:
            ax.plot(k, w["C3"]["w"], color=RAMP["C3"], lw=1.4, ls=(0, (4, 2)), alpha=a, label="+ Combine tracking (C3)")
        ax.axhline(0.5, color=BASE, lw=0.8, zorder=0)
        kh = w.get("C2", {}).get("half_weight_game")
        prior = "position-average prior" if w["c2_prior_equals_position_average"] else "Combine (C2) prior"
        if not ok:
            txt = "Not identified\n(reliability < 0.5): do not use"
        elif kh is not None:
            prior = prior.replace("position-average", "position-\naverage").replace("Combine (C2)", "Combine\n(C2)")
            txt = f"his games outweigh the {prior}\nafter {kh:.1f} games" if kh <= 17 else \
                f"his games outweigh the {prior}\nafter more than 17 games"
        else:
            txt = ""
        ax.text(0.03 if not ok else 0.97, 0.95 if not ok else 0.04, txt, transform=ax.transAxes,
                ha="left" if not ok else "right", va="top" if not ok else "bottom", fontsize=8,
                color=INK if not ok else INK2, weight="bold" if not ok else "normal")
        ax.set_title(f"{H.TRAITS[t]['pos']}  {H.TRAITS[t]['short']}  ·  ≥{H.TRAITS[t]['min_plays']} plays",
                     fontsize=9, color=INK if ok else MUTED)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 17)
    for ax in axes[-1]:
        ax.set_xlabel("Qualifying games so far")
    for ax in axes[:, 0]:
        ax.set_ylabel("Weight on his own games")
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", ncol=3, fontsize=8.5, bbox_to_anchor=(0.005, 0.905))
    fig.suptitle("In season, a player's own games take over within a few qualifying games", x=0.01, ha="left",
                 y=0.993, fontsize=12, weight="bold")
    fig.text(0.01, 0.938, "Estimate = prior + weight x (his average so far - prior); weight = k / (k + λ / (1 - R²)) "
             "(amendment A1).\nInternal fit on the 2023-25 classes (validation 2 tested a 2023-24 fit). Band: 95% "
             "preregistered player bootstrap (C2; nominal: these bootstraps under-cover in simulation).", **FOOT)
    sh = [v["share"] for v in out["post_hoc_review"]["validation_2"]["qualifying_share_2025_rookies"].values()]
    fig.text(0.01, 0.008, "A qualifying game has at least the trait's minimum number of plays; for 2025 rookies "
             f"{min(sh):.0%}-{max(sh):.0%} of appearances qualify, by trait. Where the lines coincide the Combine adds "
             "nothing over the position average.", **FOOT)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.835, bottom=0.085, hspace=0.36, wspace=0.08)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _ratio_row(ax, y, r, label, col, fp):
    lo, hi = r["ratio_ci95"]
    ax.plot([lo, hi], [y, y], color=col, lw=1.8, solid_capstyle="round")
    ax.plot(r["ratio"], y, "o", ms=7, color=col, mec=SURF, mew=1.2, zorder=3)
    ax.text(0.02, y + 0.28, label, transform=ax.get_yaxis_transform(), fontsize=8.5, color=INK, va="bottom")
    ax.text(0.02, y - 0.28, f"ratio {r['ratio']:.3f} [{lo:.3f}, {hi:.3f}] · one-sided p {fp(r['p_one_sided'])}"
            + (f" · BH q {fp(r['q'])}" if r.get("q") is not None else ""), transform=ax.get_yaxis_transform(),
            fontsize=8, color=INK2, va="top")


def fig_prospective(out: dict, path: Path) -> None:
    v2, v2p = out["validation_2"], out["post_hoc_review"]["validation_2"]
    curves, prim, ph3 = v2["pooled_curves"], v2["primary"], v2["post_hoc_combine_prior_vs_position_mean"]
    q = v2p["ph3_bh_q_over_10"]
    fp = lambda v: "< 0.0001" if v < 1e-4 else (f"{v:.4f}" if v < 0.01 else f"{v:.2f}")  # noqa: E731
    fig = plt.figure(figsize=(12, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.14)
    ax, axs, axc = (fig.add_subplot(gs[0, i]) for i in range(3))
    series = [("se_games", "His first k games only", ORANGE, "s", "-"),
              ("se_comb_C2", "Combine + draft slot only (C2)", VIOLET, "^", ":"),
              ("se_blend_C0", "Blend, position-average prior (C0)", MUTED, "v", "-"),
              ("se_blend_C2", "Blend, C2 prior (preregistered)", RAMP["C2"], "o", "-"),
              ("se_blend_C3", "Blend, C3 prior (+ tracking)", RAMP["C3"], "D", (0, (4, 2)))]
    k0map = {"se_blend_C0": "se_comb_C0", "se_blend_C2": "se_comb_C2", "se_blend_C3": "se_comb_C3"}
    dx = dict(zip([s[0] for s in series], (-0.16, -0.08, 0.0, 0.08, 0.16)))
    for key, lab, col, mk, ls in series:
        ks = [k for k in (0, *KS) if k > 0 or key in k0map]
        xs = np.array(ks, float) + dx[key]
        ests = [curves[str(k)][key if k else k0map[key]] for k in ks]
        ax.vlines(xs, [e["lo"] for e in ests], [e["hi"] for e in ests], color=col, lw=1.3)
        ax.plot(xs, [e["est"] for e in ests], marker=mk, ls=ls, color=col, lw=1.8, ms=6, mec=SURF, mew=1.0, label=lab)
    ax.set_xticks([0, *KS])
    ax.set_xlim(-0.5, max(KS) + 0.4)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Qualifying games observed (k) before forecasting the rest of his rookie season")
    ax.set_ylabel("Forecast error (units of between-player variance)")
    ax.set_title("Pooled forecast error, 95% intervals", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, ncol=1)
    rows_s = [(prim["H1"], "H1 (prereg)  C2 blend vs his first game"),
              (prim["H2"], "H2 (prereg)  C2 blend vs his first 2 games"),
              (v2p["blend_C0_vs_games"]["k1"], "post hoc  C0 blend vs his first game"),
              (v2p["blend_C0_vs_games"]["k2"], "post hoc  C0 blend vs his first 2 games")]
    for i, (r, lab) in enumerate(rows_s):
        _ratio_row(axs, 3 - i, {**r, "q": r.get("q_bh")}, lab, RAMP["C2"] if "prereg" in lab else MUTED, fp)
    axs.set_xscale("log")
    tk = [0.2, 0.3, 0.5, 1]
    axs.set_xticks(tk)
    axs.set_xticklabels([f"{v:g}" for v in tk])
    axs.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axs.set_xlim(0.18, 1.3)
    axs.axvline(1, color=BASE, lw=1)
    axs.set_title("Shrinkage vs raw early games", fontsize=10)
    rows_c = [({**ph3["k0"]["comb_C2_vs_C0"], "q": q["k0:comb_C2_vs_C0"]}, "post hoc  C2 prior vs C0, before any game"),
              ({**ph3["k1"]["blend_C2_vs_C0"], "q": q["k1:blend_C2_vs_C0"]}, "post hoc  C2 blend vs C0 blend, 1 game"),
              ({**ph3["k0"]["comb_C3_vs_C0"], "q": q["k0:comb_C3_vs_C0"]}, "post hoc  C3 prior vs C0, before any game"),
              ({**prim["H3"], "q": prim["H3"]["q_bh"]}, "H3 (prereg)  C3 blend vs C2 blend, 1 game")]
    for i, (r, lab) in enumerate(rows_c):
        _ratio_row(axc, 3 - i, r, lab, RAMP["C3"] if "prereg" in lab else VIOLET, fp)
    axc.set_xlim(0.84, 1.1)
    axc.set_xticks([0.85, 0.9, 0.95, 1.0, 1.05, 1.1])
    axc.axvline(1, color=BASE, lw=1)
    axc.set_title("Combine prior vs position average", fontsize=10)
    for a_ in (axs, axc):
        a_.set_yticks([])
        a_.set_ylim(-0.7, 3.8)
        a_.grid(axis="y", visible=False)
        a_.spines["left"].set_visible(False)
        a_.set_xlabel("Error ratio, first / second (below 1 = first better)")
    fig.suptitle("2025 rookies forecast from a fit on the 2023-24 seasons (pseudo-prospective)", x=0.01, ha="left",
                 y=0.99, fontsize=12, weight="bold")
    fig.text(0.01, 0.925, "Shrinking early games toward the position average does almost all the work; the Combine "
             "prior trims error a few percent before the first game and Combine tracking adds nothing.", **FOOT)
    fig.text(0.01, 0.01, "Error pooled over traits with equal weight, each standardized by its 2023-24 between-player "
             "variance; 95% player-bootstrap intervals. At k = 0 each blend is its prior.\nBH q for post hoc rows: "
             "over the 10 Combine-vs-position-average comparisons. Blend weight = k / (k + λ / (1 - R²)) "
             "(amendment A1).", **FOOT)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.84, bottom=0.17)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_cases(out: dict, path: Path) -> None:
    cases, flat = out["deliverables"]["case_studies"], out["deliverables"]["c2_prior_flat_by_trait"]
    fig, axes = plt.subplots(4, 4, figsize=(12, 11.4))
    for ax, c in zip(axes.flat, cases):
        n = c["n_games"]
        k = np.arange(1, n + 1)
        ax.axhline(c["season_mean"], color=MUTED, lw=1.0, zorder=0)
        ax.plot(k, c["game_values"], "o", ms=3.2, color=MUTED, zorder=1, label="Single qualifying game")
        ax.plot(k, c["running_mean"], color=ORANGE, lw=1.6, ls=(0, (3, 1.5)), label="His average so far")
        ax.plot(np.r_[0, k], np.r_[c["prior_C2"], c["blend_C2"]], color=RAMP["C2"], lw=2,
                label="In-season blend (C2 prior)")
        ax.plot(0, c["prior_C2"], "s", ms=7, color=RAMP["C2"], mec=SURF, mew=1.2, zorder=3, label="Pre-season prior (C2)")
        pick = "undrafted" if c["draft_pick"] is None else f"pick {c['draft_pick']}"
        ax.set_title(f"{c['name']} ({c['nfl_position']}, {pick})", fontsize=8.5)
        ax.text(0.98, 0.04, f"miss {c['abs_gap'] / c['prior_sd']:.1f} prior SD (predictive z {c['z_predictive']:+.1f})",
                transform=ax.transAxes, ha="right", fontsize=7.5, color=INK2, zorder=5,
                bbox=dict(boxstyle="square,pad=0.15", fc=SURF, ec="none", alpha=0.85))
        ax.set_xlim(-0.5, n + 0.5)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=5))
        ax.tick_params(labelsize=7.5)
    for r, (pos, t) in enumerate((("WR", "T1"), ("DL", "T6"), ("OL", "T9"), ("DB", "T11"))):
        c = cases[4 * r]
        axes[r, 0].set_ylabel(f"{pos}  {AXIS_LABEL[t]}", fontsize=8.5)
        cg = c["cge_C2_2023_24_fit"]
        prior = ("prior = position average (the Combine adds nothing)" if flat.get(t) else
                 "C2 prior worth 0 games in the 2023-24 fit (close to the position average)" if cg == 0 else
                 f"C2 prior worth {cg:.2f} games in the 2023-24 fit")
        role = {r_["trait"]: r_ for r_ in out["cge_table"]}[t]
        if pos != "WR" and role["cge_role_reference"] > 0.5:
            prior += (f"; NFL position alone is worth {role['cge_role_reference']:.1f} games, the Combine adds "
                      f"{role['cge_C2_beyond_role']:+.2f} beyond it (internal fit)")
        axes[r, 0].text(0.0, 1.2, f"{pos} {H.TRAITS[t]['name'].lower()}: {prior}", transform=axes[r, 0].transAxes,
                        fontsize=9, color=INK, weight="bold")
    for ax in axes[-1]:
        ax.set_xlabel("Qualifying games in 2025")
    fig.suptitle("2025 rookies: pre-season prior, his own games, and the in-season blend", fontsize=12, x=0.01,
                 ha="left", y=0.997, weight="bold")
    db = out["post_hoc_review"]["validation_2"]["per_trait"]["T11"]["class_2025_shift_tau"]["mean"]
    fig.text(0.01, 0.951, "Per position the preregistered rule picks the 2 largest (left) and 2 smallest (right) "
             "|prior - season average| among rookies with >= 10 qualifying games. Where the prior is the position "
             "average,\nthe 'largest misses' are rookies far from average, not Combine errors; 2025 DB rookies' "
             f"retreat speed averaged {db:+.1f} between-player SD from the 2023-24 level.", **FOOT)
    h, lab = axes.flat[0].get_legend_handles_labels()
    h.append(Line2D([], [], color=MUTED, lw=1))
    lab.append("Rookie-season average")
    fig.legend(h, lab, loc="upper left", ncol=5, fontsize=8.5, bbox_to_anchor=(0.005, 0.925))
    fig.text(0.01, 0.006, "Prior SD = sqrt((1 - R²) τ²) from the 2023-24 fit; predictive z adds the sampling noise of "
             "his season average. Fits and priors use the 2023-24 seasons only.", **FOOT)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.85, bottom=0.06, hspace=0.62, wspace=0.27)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# =====================================================================================================
# Main
# =====================================================================================================

def trait_notes(t: str, r: dict, ph: dict) -> list[str]:
    """Caveats attached to a cge_table row, by fixed rules (post hoc PH7, PH8, PH11)."""
    notes = []
    if not ph["identified"]:
        notes.append(f"not identified: reliability {ph['rel']:.2f}, split-half {ph['split_half']:.2f} (< {REL_MIN}); "
                     "the true-score correction makes the CGE unstable, do not interpret it")
    ex = r["exposure"]["r_player_trait_log_games"]
    if abs(ex) >= 0.3:
        notes.append(f"exposure: the player trait correlates {ex:+.2f} with log qualifying games")
    rr = ph["references"]["role"]
    if rr["cge_ref"] > 0.5:
        what = "deep-route share, in-season usage" if r["position"] == "WR" else "NFL roster position"
        notes.append(f"role reference ({what}) alone is worth {rr['cge_ref']:.2f} games; C2 adds "
                     f"{rr['combine_beyond_ref']:+.2f} beyond it")
    lo, hi = ph["leave_one_class_out_cge_C2_range"]
    if r["cge"]["C2"]["est"] > 0 and lo == 0 and ph["identified"]:
        notes.append(f"CGE(C2) is 0 when one draft class is left out (range {lo:.2f}-{hi:.2f})")
    sc = ph["stationarity_and_game_unit"]["season_change_within_player"]
    big = max((abs(v["mean_tau"]) for v in sc.values()), default=0)
    if big >= 1:
        notes.append(f"non-stationary: within-player season-to-season shifts up to {big:.1f} between-player SD")
    return notes


def cge_table(internal: dict, ph: dict, belt_sum: dict) -> list[dict]:
    """One row per trait. lo / hi = the preregistered player bootstrap (B1); ph5_lo / ph5_hi = the post hoc
    fixed-forecast bootstrap (sensitivity, under-covers in simulation); calibrated bounds from PH6."""
    rows = []
    for t, r in internal.items():
        f = lambda d: {"est": d["est"], "lo": d["ci95_prereg_bootstrap"].get("lo"),  # noqa: E731
                       "hi": d["ci95_prereg_bootstrap"].get("hi"),
                       "ph5_lo": d["ci95_fixed_forecast_PH5"].get("lo"), "ph5_hi": d["ci95_fixed_forecast_PH5"].get("hi")}
        g = lambda d: {k: d.get(k) for k in ("est", "lo", "hi")}  # noqa: E731
        p, b = ph[t], belt_sum["traits"][t]
        rows.append({"trait": t, "position": r["position"], "name": r["name"], "higher_means": HIGHER_MEANS[t],
                     "n_players": r["n_players"], "mean_games": r["mean_games"],
                     "min_plays_per_game": r["min_plays_per_game"],
                     "lambda": g(r["variance"]["lambda"]), "rel_player_mean": r["variance"]["rel"]["est"],
                     "split_half": r["reliability_descriptive"]["split_half_odd_even_games"],
                     "identified": p["identified"], "r2_cv": r["r2_cv"],
                     **{f"r2_{s}": f(r["r2_true"][s]) for s in ("C1", "C2", "C3")},
                     **{f"cge_{s}": f(r["cge"][s]) for s in ("C1", "C2", "C3")},
                     "cge_C2_calibrated_upper95": b["calibrated_one_sided_95"]["upper"],
                     "cge_C2_calibrated_lower95": b["calibrated_one_sided_95"]["lower"],
                     "cge_C2_population_signal_upper95": b["calibrated_one_sided_95_population_signal"]["upper"],
                     "cge_C2_2023_24_seasons_fit": p["cge_2023_24_seasons_fit"]["C2"],
                     "cge_C2_leave_one_class_out_range": p["leave_one_class_out_cge_C2_range"],
                     "cge_C2_vs_C0_same_cv": p["c0_relative"]["C2"]["cge"],
                     "cge_C2_season_centred": p["season_centred"]["cge"]["C2"],
                     "cge_role_reference": p["references"]["role"]["cge_ref"],
                     "cge_C2_beyond_role": p["references"]["role"]["combine_beyond_ref"],
                     "exposure_r_trait_log_games": r["exposure"]["r_player_trait_log_games"],
                     "delta_cge_tracking": f(r["delta_cge"]["C3-C2"]), "notes": trait_notes(t, r, p)})
    return rows


def write_up(out: dict) -> dict:
    """Headline, plain summary and the review response, with every number taken from this run (PH12)."""
    v2, v2p, tab = out["validation_2"], out["post_hoc_review"]["validation_2"], {r["trait"]: r for r in out["cge_table"]}
    prim, curves, ph3 = v2["primary"], v2["pooled_curves"], v2["post_hoc_combine_prior_vs_position_mean"]
    h1, h2, h3 = prim["H1"], prim["H2"], prim["H3"]
    k0, k1 = ph3["k0"]["comb_C2_vs_C0"], ph3["k1"]["blend_C2_vs_C0"]
    q = v2p["ph3_bh_q_over_10"]
    ident = [t for t, r in tab.items() if r["identified"]]
    nid = [t for t, r in tab.items() if not r["identified"]]
    half = {t: out["deliverables"]["weight_curves"][t]["C2"]["half_weight_game"] for t in ident}
    pos_c = {t: tab[t]["cge_C2"]["est"] for t in ident if tab[t]["cge_C2"]["est"] > 0}
    top = sorted(pos_c, key=lambda t: -pos_c[t])[:2]
    up = {t: tab[t]["cge_C2_calibrated_upper95"] for t in ident}
    fin = [v for v in up.values() if v is not None]
    rules_out = (f"rule out more than {max(fin):.1f} games for every reliably measured trait"
                 if len(fin) == len(up) else f"rule out more than {max(fin):.1f} games for {len(fin)} of the "
                 f"{len(up)} reliably measured traits")
    above = {t: tab[t]["cge_C2_calibrated_lower95"] for t in ident if (tab[t]["cge_C2_calibrated_lower95"] or 0) > 0}
    pop_up = [tab[t]["cge_C2_population_signal_upper95"] for t in ident]
    drv = v2p["ph3_drivers"]["k0"]
    lead = sorted(drv, key=lambda t: drv[t]["mean_diff"])[:2]
    share1 = v2p["shrinkage_share_of_blend_gain"]["1"]
    lbl = lambda t: f"{H.TRAITS[t]['pos']} {H.TRAITS[t]['short'][0].lower()}{H.TRAITS[t]['short'][1:]}"  # noqa: E731
    and_ = lambda xs: " and ".join(xs) if len(xs) < 3 else ", ".join(xs[:-1]) + " and " + xs[-1]  # noqa: E731
    headline = (
        f"In season a rookie's own tracking quickly dominates: for the {len(ident)} reliably measured traits his "
        f"qualifying games outweigh any pre-season prior after {min(half.values()):.0f}-{max(half.values()):.0f} games. "
        f"On 2025 rookies (pseudo-prospective: fit on the 2023-24 seasons only), shrinking early games toward the "
        f"position average does almost all the work: it achieves {share1:.0%} of the preregistered blend's "
        f"{1 - h1['ratio']:.0%} error cut after one game (H1). Adding the stopwatch and draft slot to the prior lowers "
        f"error a further {1 - k0['ratio']:.0%} before the first game (post hoc, p {k0['p_one_sided']:.2f}, BH q "
        f"{q['k0:comb_C2_vs_C0']:.2f}) and {1 - k1['ratio']:.1%} after one game; Combine tracking adds nothing (H3 "
        f"error ratio {h3['ratio']:.3f} [{h3['ratio_ci95'][0]:.3f}, {h3['ratio_ci95'][1]:.3f}], slightly worse). "
        f"In game units the Combine is worth 0 to {max(pos_c.values()):.1f} games for every reliably measured trait.")
    n_zero = sum(1 for t in ident if tab[t]["cge_C2"]["est"] == 0)
    summary = (
        f"In season a player's own games quickly dominate: for the {len(ident)} reliably measured traits his "
        f"qualifying games outweigh any pre-season prior after {min(half.values()):.0f}-{max(half.values()):.0f} "
        f"games. Shrinkage does most of the work: on 2025 rookies (pseudo-prospective) the preregistered blend beat "
        f"raw early-game averages (H1 error ratio {h1['ratio']:.2f} after one game, H2 {h2['ratio']:.2f} after two), "
        f"but blending toward the position average alone achieves {share1:.0%} of that gain. The Combine (stopwatch "
        f"+ draft slot) adds a little before the first game: {1 - k0['ratio']:.0%} lower error (post hoc, p "
        f"{k0['p_one_sided']:.2f}, BH q {q['k0:comb_C2_vs_C0']:.2f}; mostly {lbl(lead[0])} and {lbl(lead[1])}), "
        f"{1 - k1['ratio']:.1%} after one game. Combine tracking adds nothing (H3). In games, the Combine is worth "
        f"about {pos_c[top[0]]:.1f} for {lbl(top[0])} and {pos_c[top[1]]:.1f} for {lbl(top[1])} (2023-24 fit: "
        f"{tab[top[0]]['cge_C2_2023_24_seasons_fit']:.2f} and {tab[top[1]]['cge_C2_2023_24_seasons_fit']:.2f}) and 0 "
        f"for {n_zero} traits. Every preregistered 95% bootstrap interval includes 0; simulation-calibrated bounds "
        + (f"put only {and_([f'{lbl(t)} ({v:.2f})' for t, v in above.items()])} above 0 and " if above else "")
        + f"{rules_out} (the Combine as learned from ~50-85 players; a perfectly learned signal of the simulated form "
        f"could be worth up to {max(v for v in pop_up if v is not None):.0f}). The DL value is mostly body type and "
        f"role (NFL position alone: {tab['T6']['cge_role_reference']:.1f} games; the Combine adds "
        f"{tab['T6']['cge_C2_beyond_role']:+.2f} beyond it). {and_([lbl(t) for t in nid])} cannot be measured "
        f"(reliability below 0.5).")
    return {"headline": headline, "summary": summary,
            "supersedes": {"file": "experiments/results/E012_audit/summary.txt",
                           "why": "Written by the builder before review. It credited the Combine-based blend with "
                                  "the 72% error cut that shrinkage to the position mean achieves on its own, called "
                                  "the test 'frozen before the 2025 season' (it is pseudo-prospective: the fit uses "
                                  "2023-24 seasons only, but trait definitions came after earlier experiments used "
                                  "2025 games), and left out the two unstable large CGEs (T4, T8). The audit "
                                  "record is read-only, so the correction lives here."}}


def main() -> None:
    out = {"experiment": "E012", "title": PREREG["experiment"], "run_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                                 time.gmtime()),
           "quick": QUICK, "prereg_file": str(PREREG_PATH.relative_to(ROOT)),
           "prereg_sha256": hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest(),
           "implementation_choices": {"frozen_at_utc": CHOICES_FROZEN_AT, "freeze_file_sha256": CHOICES_SHA256,
                                      "check": check_frozen_choices(), "choices": IMPLEMENTATION_CHOICES},
           "amendments": AMENDMENTS}
    names = ["e012_players", "e012_game_values", "e012_prospective_values", "e012_checks"]
    if REUSE:
        players = pl.read_parquet(DATA / "e012_players.parquet")
        v2cols = [c for c in players.columns if c.endswith("__v2")]
        players_p = players.drop([c[:-4] for c in v2cols]).rename({c: c[:-4] for c in v2cols})
        players = players.drop(v2cols)
        games_i, games_p = (pl.read_parquet(DATA / f"{n}.parquet") for n in names[1:3])
        checks = json.loads(pl.read_parquet(DATA / "e012_checks.parquet")["json"][0])
        log("reused e012 value tables")
    else:
        players, players_p, cchk = build_combine()
        log("Combine inputs built")
        games_i, games_p, vchk = build_values(players)
        checks = {"combine": cchk, "values": vchk}
        v2 = players_p.select("nfl_id", *[pl.col(c).alias(f"{c}__v2") for c in
                                          ("c_turn_retention", "c_stop_peak_decel", "cmb_backpedal_speed")])
        if not QUICK:
            players.join(v2, on="nfl_id", how="left").write_parquet(DATA / "e012_players.parquet")
            games_i.write_parquet(DATA / "e012_game_values.parquet")
            games_p.write_parquet(DATA / "e012_prospective_values.parquet")
            pl.DataFrame({"json": [json.dumps(clean(checks))]}).write_parquet(DATA / "e012_checks.parquet")
    out["data_checks"] = checks

    # ---- 3. internal CGE
    internal, arts, boots = {}, {}, []
    for i, t in enumerate(H.TRAITS):
        internal[t], arts[t] = internal_trait(t, i, games_i, players)
        boots.append(arts[t]["boot"].with_columns(trait=pl.lit(t), kind=pl.lit("prereg_refit_B1")))
        boots.append(arts[t]["boot_ff"].with_columns(trait=pl.lit(t), kind=pl.lit("fixed_forecast_PH5")))
        r = internal[t]
        log(f"{t} {r['position']} {r['name']}: n {r['n_players']}, lambda {r['variance']['lambda']['est']:.2f}, "
            f"rel {r['variance']['rel']['est']:.2f}, R2 C1/C2/C3 " +
            "/".join(f"{r['r2_true'][s]['est']:.2f}" for s in ("C1", "C2", "C3")) + ", CGE " +
            "/".join(f"{r['cge'][s]['est']:.2f}" for s in ("C1", "C2", "C3")) +
            f", dCGE {r['delta_cge']['C3-C2']['est']:+.2f}")
    out["traits"] = internal
    out["interval_note"] = INTERVAL_NOTE
    out["bootstrap_bias_diagnostic"] = bootstrap_bias_diagnostic(arts)
    out["bootstrap_bias_diagnostic"]["prereg_bootstrap_summary"] = prereg_bootstrap_summary(internal)
    out["bootstrap_bias_diagnostic"]["reading"] = (
        "Descriptive only (revised after review): B1's R2 draws centre below the point estimate because each resample "
        "refits the ridge on ~63% distinct players; centring does not decide an interval's validity, coverage does "
        "(post_hoc_review.simulation.interval_coverage).")
    log("bootstrap bias diagnostic: " + json.dumps(clean({t: {k: v["median"] for k, v in d.items() if k != "point_r2cv_C2"}
                                                         for t, d in out["bootstrap_bias_diagnostic"]["traits"].items()})))

    # ---- 4a. validation 1
    out["validation_1"] = {t: validation1(t, arts[t], {s: internal[t]["cge"][s]["est"] for s in H.SETS})
                           for t in H.TRAITS}
    agree = [(t, s, v["agree"]) for t, r in out["validation_1"].items() for s, v in r["sets"].items()
             if s != "C0" and v["agree"] is not None]
    full_range = [(t, s) for t, r in out["validation_1"].items() for s, v in r["sets"].items()
                  if s != "C0" and v["crossover_int_ci95"] == [1, 7]]
    out["validation_1_summary"] = {
        "agree": sum(a for *_, a in agree), "of": len(agree),
        "n_cells_ci_spans_1_to_gt6": len(full_range), "cells_ci_spans_1_to_gt6": [f"{t}-{s}" for t, s in full_range],
        "note": "C0 is excluded: its CGE is 0 by construction and its forecast is noise. A crossover interval of "
                "[1, 7] ('1 to more than 6 games') cannot disagree with any formula value, so those cells carry no "
                "information about the formula.",
        "reading_after_review": "Not contradicted, which is weak evidence: every other cell had formula CGE <= 1, "
                                "so it could fail only if the Combine reliably beat one game. See "
                                "post_hoc_review.validation_1 (PH9) for the literal prereg check and the "
                                "model-implied CGE."}
    log(f"validation 1: formula CGE inside the crossover CI for {out['validation_1_summary']['agree']} / {len(agree)}")

    # ---- 4b. validation 2
    summ, Fc, fits = validation2(games_p, players_p)
    v2 = validation2_tests(Fc)
    out["validation_2"] = {"training_fit": summ, "design_label": "pseudo-prospective: fit on the 2023-24 seasons only "
                           "and applied to the 2025 rookies' 2025 season; the trait definitions were fixed after "
                           "earlier experiments (E001-E011) had used 2025 games", **v2}
    for h, r in v2["primary"].items():
        log(f"{h}: {r['pooled_a']:.3f} vs {r['pooled_b']:.3f} (diff {r['diff']:+.3f} [{r['diff_ci95'][0]:+.3f}, "
            f"{r['diff_ci95'][1]:+.3f}]), p {r['p_one_sided']:.4f}, q {r['q_bh']:.4f}, n {r['n_players']}")

    # ---- 4c. post hoc (review response)
    app, deep = usage_tables()
    weeks = io.load("games").select("game_id", "week")
    ph = {t: trait_posthoc(t, i, arts[t], internal[t], players, deep, weeks, summ[t]) for i, t in enumerate(H.TRAITS)}
    identified = {t: ph[t]["identified"] for t in H.TRAITS}
    log("post hoc per-trait sensitivities done; not identified: " + ", ".join(t for t, v in identified.items() if not v))
    if REUSE_SIMS:
        belt = pl.read_parquet(DATA / "e012_sim_belt.parquet")
        log("reused simulation belt")
    else:
        belt = sim_belt(arts, internal)
        if not QUICK:
            belt.write_parquet(DATA / "e012_sim_belt.parquet")
        log(f"simulation belt: {belt.height:,} simulated datasets")
    belt_sum = belt_summary(belt, internal)
    if REUSE_SIMS:
        cov = pl.read_parquet(DATA / "e012_sim_coverage.parquet")
    else:
        cov = sim_coverage(arts, internal, belt_sum)
        if not QUICK:
            cov.write_parquet(DATA / "e012_sim_coverage.parquet")
    cov_sum = coverage_summary(cov, internal)
    log("interval coverage: " + json.dumps(clean({t: {k: cov_sum[t][k]["coverage"] for k in ("b1", "ph5")}
                                                 for t in COV_DESIGNS})))
    out["post_hoc_review"] = {
        "per_trait": ph,
        "simulation": {"calibrated_bounds": belt_sum, "interval_coverage": cov_sum},
        "validation_1": validation1_posthoc(out["validation_1"], internal),
        "validation_2": validation2_posthoc(Fc, v2, fits, app, games_p, players_p, identified),
        "determinism_fix": DETERMINISM_FIX}
    out["cge_table"] = cge_table(internal, ph, belt_sum)

    # ---- 5. deliverables
    wc = weight_curves(internal, arts, identified)
    cases = case_studies(games_p, players_p, fits)
    outl = surprises(games_p, players_p, fits)
    flat = {t: v["c2_prior_flat"] for t, v in out["post_hoc_review"]["validation_2"]["per_trait"].items()}
    out["deliverables"] = {
        "weight_curves": wc, "case_studies": cases, "rookie_outliers_vs_prior": outl,
        "rookie_outliers_note": "The prereg's 'Combine surprises' table (selection rule unchanged). Where the C2 prior "
                                "is flat across 2025 rookies (c2_prior_flat_by_trait) it is the position average, so "
                                "the rows show rookies far from average, not cases where the Combine was wrong.",
        "c2_prior_flat_by_trait": flat, "higher_means_by_trait": HIGHER_MEANS}
    if not QUICK:
        pl.concat(boots, how="diagonal_relaxed").write_parquet(DATA / "e012_bootstrap.parquet")
        Fc.write_parquet(DATA / "e012_forecasts_2025.parquet")

    out["post_hoc"] = POST_HOC
    out["review_response"] = REVIEW_RESPONSE
    out = clean(out)  # figures are drawn from exactly what is written (also what --figures-only reads)
    out = {**{k: out[k] for k in ("experiment", "title")}, **write_up(out), **out}
    out["figures"] = draw_figures(out)
    out["runtime_s"] = time.time() - T0
    (OUT_RES / "E012.json").write_text(json.dumps(clean(out), indent=1))
    log("headline: " + out["headline"])
    log(f"done: {OUT_RES / 'E012.json'}")


def draw_figures(out: dict) -> list[str]:
    figs = {"cge": OUT_FIG / "E012_cge.png", "weights": OUT_FIG / "E012_weight_curves.png",
            "prospective": OUT_FIG / "E012_prospective.png", "cases": OUT_FIG / "E012_case_studies.png"}
    fig_cge(out, figs["cge"])
    fig_weights(out, figs["weights"])
    fig_prospective(out, figs["prospective"])
    fig_cases(out, figs["cases"])
    return [str(p.relative_to(ROOT)) if not QUICK else str(p) for p in figs.values()]


def rewrite() -> None:
    """Deterministic post-run assembly from E012.json: cge_table, wording, post hoc / review records, figures."""
    out = json.loads((OUT_RES / "E012.json").read_text())
    ph = out["post_hoc_review"]["per_trait"]
    out["cge_table"] = clean(cge_table(out["traits"], ph, out["post_hoc_review"]["simulation"]["calibrated_bounds"]))
    out["post_hoc"], out["review_response"], out["interval_note"] = POST_HOC, REVIEW_RESPONSE, INTERVAL_NOTE
    out["deliverables"]["higher_means_by_trait"] = HIGHER_MEANS
    wu = clean(write_up(out))
    out = {**{k: out[k] for k in ("experiment", "title")}, **wu, **{k: v for k, v in out.items() if k not in wu}}
    out["figures"] = draw_figures(out)
    out["rewritten_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (OUT_RES / "E012.json").write_text(json.dumps(clean(out), indent=1))
    print(out["headline"])


if __name__ == "__main__":
    if FIGURES_ONLY:
        draw_figures(json.loads((OUT_RES / "E012.json").read_text()))
    elif REWRITE:
        rewrite()
    else:
        with threadpool_limits(H.MAX_THREADS):
            main()
