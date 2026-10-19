"""E013 - Role-aware Combine Exchange Rate: once a rookie's role is known from his first games, how much do the Combine
and draft add, and how fast do his own games take over? Prereg: experiments/results/E013_prereg.json (binding).

Runs top to bottom (reuses E012's stored tables and code; nothing of E012 is edited):
  1. Inputs: E012's player table (internal, and the strict-composite version for the prospective test), E012's
     per-game trait values (internal 2023-25; prospective with 2023-24 constants), and per player-game alignment
     counts from player_play (role share, WR deep-route share, 2025 rookies' teams).
  2. Internal (2023-25) per trait: E012's estimate() on C0 / C0_role / C2 / C2_role / C3 / C3_role; lambda_role,
     role worth, CGE beyond role, the change in the Combine + draft value once role is known, half-weight games and the
     role-aware trust table; the preregistered player bootstrap (B1, E012's draws).
  3. Simulation at each trait's own design: calibrated one-sided 95% bounds for role worth and for C2 / C3 beyond role
     (E012 PH6 method); B1 interval coverage at the T1 and T6 designs.
  4. Prospective (2023-24 fit -> 2025 rookies; role share from his first k games): H1-H4 with sign flips, bootstrap
     CIs, BH, team-cluster sensitivity; the secondary family; descriptive curves and calibration; leakage checks.
  5. Descriptive diagnostics, tables, figures, E013.json.
  6. Post hoc, after the E013 review (POST_HOC PH4-PH13; post_hoc_review in E013.json): prospective variants for the
     training-vs-test role-share precision mismatch, decompositions of H3, power; role worth from the preregistered
     ANOVA lambda_role with a permutation test; same-CV references against a dummies-only ridge; both orders of the
     role / Combine attribution; roster designation and role-share semantics. No preregistered number changes.
Implementation choices were frozen before any role-trait association, role-aware variance component, CGE or forecast
was computed (IMPLEMENTATION_CHOICES, freeze file experiments/results/E013_audit/, hash below). Anything decided later
is in POST_HOC.

Run: .venv/bin/python -B experiments/E013_role_aware.py            (everything, n_jobs 3; several hours)
     .venv/bin/python -B experiments/E013_role_aware.py --quick    (small counts; output to data/scratch/e013_quick)
     .venv/bin/python -B experiments/E013_role_aware.py --resume   (reuse finished bootstrap / simulation checkpoints)
     .venv/bin/python -B experiments/E013_role_aware.py --reuse    (read data/processed/e013_* draws; recompute the rest)
     .venv/bin/python -B experiments/E013_role_aware.py --figures-only  (redraw the figures from E013.json)
     .venv/bin/python -B experiments/E013_role_aware.py --rewrite  (headline, summary, post hoc record, figures from
                                                                   E013.json; no recomputation)
Outputs: experiments/results/E013.json, experiments/figures/E013_*.png, data/processed/e013_*.parquet (with --reuse,
the post hoc dummies-ridge draws are read from e013_posthoc_bootstrap.parquet when present).
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.patches  # noqa: F401  (legend patch)
import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from bdb27 import io  # noqa: E402
from bdb27.probes import h_game_equivalents as H  # noqa: E402
from bdb27.probes import i_role_aware as I  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG, DATA = ROOT / "experiments" / "results", ROOT / "experiments" / "figures", io.PROCESSED
AUDIT = RES / "E013_audit"
QUICK = "--quick" in sys.argv
RESUME = "--resume" in sys.argv
REUSE = "--reuse" in sys.argv
FIGURES_ONLY = "--figures-only" in sys.argv
REWRITE = "--rewrite" in sys.argv  # headline, summary, post hoc record and figures from E013.json
# Checkpoints (--resume) and --quick output. Since 2026-10-19 they live under the repository's gitignored data/, so
# the public notebook keeps them in its workspace and a later Kaggle session resumes from them (frozen choice X1 put
# them in the session scratchpad; only the location changed).
SCRATCH = ROOT / "data" / "scratch" / ("e013" + ("_quick" if QUICK else ""))
CKPT = SCRATCH / "ckpt"
OUT_RES, OUT_FIG = (SCRATCH, SCRATCH) if QUICK else (RES, FIG)
N_JOBS = 3
SEED = 20261018
E012_SEED = 20261015  # B1 player draws = E012's (seed + E012 trait index), so C0 / C2 / C3 draws reproduce E012's
N_BOOT = 20 if QUICK else 1000
BOOT_REPEATS = 2 if QUICK else H.N_REPEATS
N_FLIPS = 5_000 if QUICK else 100_000
N_BOOT_V = 1_000 if QUICK else 10_000
TRAIN_SEASONS, TEST = (2023, 2024), 2025
KS = (1, 2, 3, 4)
SIM_GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75)  # E012's grid (partial R2 beyond role)
ROLE_GRID = SIM_GRID + (0.9,)  # population R2 of role share
SIM_REPS = 6 if QUICK else 200
RHO_MAX = 0.95
COV_DESIGNS = ("T1", "T6")
COV_DATASETS = 3 if QUICK else 80
COV_B1_RESAMPLES = 10 if QUICK else 250
FAMILIES = {  # simulation family -> (sets kept, set carrying the planted Combine signal, estimated set)
    "role": (("C0", "C0_role"), None, "C0_role"),
    "beyond_C2": (("C0", "C0_role", "C2_role"), "C2_role", "C2_role"),
    "beyond_C3": (("C0", "C0_role", "C3_role"), "C3_role", "C3_role"),
}
POSITIONS = {"WR": ["T1", "T2", "T3", "T5"], "DL": ["T6", "T7"], "OL": ["T9"], "DB": ["T11", "T12"]}
PREREG_PATH = RES / "E013_prereg.json"
PREREG = json.loads(PREREG_PATH.read_text())
E012 = json.loads((RES / "E012.json").read_text())
T0 = time.time()

# =====================================================================================================
# Frozen implementation choices (copied verbatim from the freeze file)
# =====================================================================================================
CHOICES_FROZEN_AT = "2026-10-19T02:03:19Z"
CHOICES_SHA256 = "9ace65fb40fc4f1c5d362eea73c299ad8e9b5fee7678f8582a9f2073fed85645"  # of the freeze file
IMPLEMENTATION_CHOICES = json.loads(r'''
[
 {"id": "G1", "topic": "population, games and values", "choice": "Exactly E012's: Combine positions (WR traits use Combine WRs), REG games, chronological order = game_id order, qualifying games and per-game values as stored by E012 (data/processed/e012_game_values.parquet for the internal fit; e012_prospective_values.parquet, built with 2023-24 constants and expectation models, for the prospective test), players with >= 3 qualifying games. Player table = e012_players.parquet (internal: as stored; prospective: the strict 2023-24 Combine composites (__v2) swapped in, as E012 --reuse-values). The 9 traits of the prereg: T1, T2, T3, T5, T6, T7, T9, T11, T12.", "reason": "Prereg: values, qualifying rules, centering and expectation models exactly as E012, reusing its tables."},
 {"id": "R1", "topic": "role share counting", "choice": "Counting unit = one player_play row (REG games) of the player in a game. Every row counts, including plays nullified by penalty (the alignment is a pre-snap fact) and plays that carry no value of the trait (prereg: all plays of the qualifying games). Numerator / denominator alignments exactly as the prereg (WR slot: SLOT_WR / SLOT_WR+WR; DL edge: EDGE+OLB / EDGE+OLB+INTERIOR_LINE; OL tackle: T / T+G+C; DB corner: CB+SLOT_CB / CB+SLOT_CB+HIGH_SAFETY+BOX_SAFETY+OLB+MLB+EDGE+INTERIOR_LINE); null and other alignments ignored. Share = sum numerator / sum denominator over the games (play-weighted); undefined (null) if the denominator is 0, then median-imputed with a missing indicator inside linkage._model (and by i_role_aware.impute_cols in the ANOVA). Data facts seen before this freeze (vocabulary only, no trait values): Combine DL have 51 OLB rows against 28,884 EDGE and 23,030 INTERIOR_LINE (edge share is effectively EDGE / (EDGE + INTERIOR_LINE)); 503 DL rows at SLOT_CB are ignored.", "reason": "Prereg role_definition; play weighting makes the share the fraction of his snaps in the role."},
 {"id": "R2", "topic": "games behind each role share", "choice": "Internal fit: the player's qualifying games of the trait in e012_game_values (2023-25). Prospective training: his qualifying games of the trait in the train split (2023-24 seasons, 2023-24 classes). In season at k: his first k qualifying 2025 games of the trait (game_id order). Oracle (secondary): all his 2025 qualifying games of the trait. The prereg's 'role share over all of the player's 2023-24 games' is read with its role_definition.source: all plays of the qualifying games used for the trait.", "reason": "Prereg role_definition (source, training, in_season)."},
 {"id": "R3", "topic": "information sets", "choice": "C0 = E012's (draft-year mean within position; the training-fold class mean in CV). C0_role = linkage._model on [draft-year dummies (h_game_equivalents.design, earliest class the reference) | role_share]. C2_role = [dummies | linkage.COMBINE_CONTROLS + linkage.DRAFT_CONTROLS | role_share]. C3_role = [dummies | C2 columns + the tracked-40 set + the trait's matched signature (E012 C3) | role_share]. E012's C2 and C3 are carried in the same estimate (same folds) for reference and reproduction. Role share enters linearly.", "reason": "Prereg information_sets."},
 {"id": "E1", "topic": "internal estimate and descriptive estimands", "choice": "h_game_equivalents.estimate on the six sets (draft-year-strata variance components, rel, RepeatedKFold 5 x 20 seed 0 over players, R2_CV = mean over repeats of 1 - MSE / Var_w, R2_true = clip(R2_CV / rel, 0, 0.99), CGE = lambda R2 / (1 - R2) with E012's position-level lambda and rel). role_worth = CGE(C0_role) (E012 formula; like E012's CGE it is measured against the position average); sensitivity: C0-relative under the same CV (E012 PH8 convention). cge_beyond_role(C) = CGE(C) - CGE(C0_role) for C2_role and C3_role (i_role_aware derivation 2: equals lambda_role P / (1 - P) with P the partial true-score R2 beyond role; the unclipped difference of the two clipped CGEs, so it can be negative); tracking beyond C2_role = CGE(C3_role) - CGE(C2_role), descriptive. combine_change_once_role_known = CGE(C2) - cge_beyond_role(C2_role). lambda_role (primary) = sigma2 / tau2_role with tau2_role from the stratified one-way ANOVA with the role covariate (derivation 3, unit-tested against simulation), 'as E012's variance components'; reported beside the CV route lambda / (1 - R2_true(C0_role)) = lambda + CGE(C0_role), which is what the trust rule uses. half_weight_game = lambda / (1 - R2_C) and trust-table weights w_k = k / (k + lambda / (1 - R2_C)) at k = 1, 4, 8 (E012 amendment A1) for C0_role and C2_role, with C0 and C2 beside them; the 2023-24 fit's values (validation training fit) beside the internal ones. Rookie appearances = half-weight / E012's qualifying_share_2025_rookies share (derived, as writeup provenance item 1).", "reason": "Prereg estimands_descriptive; one estimator family with E012 so differences are attributable to role."},
 {"id": "E2", "topic": "bootstrap (B1)", "choice": "1,000 player resamples per trait with E012's seeds (20261015 + E012 trait index): the same player draws and per-resample CV seeds as E012's B1, so its C0 / C2 / C3 draws reproduce data/processed/e012_bootstrap.parquet (checked). The whole pipeline (i_role_aware.estimate_role: E012 estimate with grouped folds over the unique resampled players, 20 CV repeats, plus lambda_role) per resample; each player's role share travels with him. 95% percentile intervals (h_game_equivalents.percentile_ci, non-finite draws as +inf). Differences (beyond role, the change once role is known) are computed within each draw. Simulated coverage of these intervals is reported (S2), and they are read as nominal.", "reason": "Prereg: E012's B1 bootstrap."},
 {"id": "S1", "topic": "simulation-calibrated bounds", "choice": "E012 PH6 method at each trait's own design (its players, classes, game counts and design matrices; tau2 = 1, no class effects, games iid N(theta, lambda_hat) with the trait's internal lambda; 200 datasets per grid point; seeds default_rng([20261018, family, trait index, grid index, rep])). Neyman one-sided 95% bounds (h_game_equivalents.calibrated_bounds) for the CV estimand (each set's CGE from the mean oracle R2 of its out-of-fold forecasts against theta, clipped at 0), plus the population-signal version. Families: (a) role worth: theta = sqrt(rho) z + sqrt(1 - rho) u, z = standardized role share (median-imputed), rho in (0, .05, .1, .15, .2, .25, .3, .4, .5, .6, .75, .9); sets C0, C0_role; estimate CGE(C0_role); population CGE = lambda rho / (1 - rho). (b) C2 beyond role: theta = sqrt(rho_hat) z + sqrt(1 - rho_hat) (sqrt(g) s_perp + sqrt(1 - g) u), s = signed sum of 3 random columns of the C2 Combine design (h_game_equivalents.signal_matrix), residualized on z and standardized; rho_hat = the trait's observed internal R2_true(C0_role), clipped to [0, 0.95]; g in E012's grid (0 ... 0.75); sets C0, C0_role, C2_role; estimate CGE(C2_role) - CGE(C0_role); estimand CGE*(C2_role) - CGE*(C0_role); population = lambda / (1 - rho_hat) g / (1 - g). (c) C3 beyond role: as (b) with the C3 Combine design and C3_role.", "reason": "Prereg: 'the simulation-calibrated bounds method' for CGE beyond role; the task asks for it on every descriptive CGE estimand."},
 {"id": "S2", "topic": "bootstrap coverage in simulation", "choice": "At the T1 and T6 designs (E012's coverage designs): 80 simulated datasets each, theta planted at rho_hat (as S1 b) and at the grid g whose family-(b) estimand is closest to the observed C2 beyond-role value; per dataset B1 with 250 resamples x 20 CV repeats (h_game_equivalents.bootstrap); coverage of the 95% intervals for role worth and for C2 beyond role against the CV estimand from those datasets' mean oracle R2 (as E012's coverage_summary). Seeds default_rng([20261018, 99, trait index, dataset]).", "reason": "LESSONS 2026-10-15: check interval coverage by simulation before trusting any bootstrap."},
 {"id": "P1", "topic": "prospective fit (leakage rules as E012)", "choice": "E012 P1-P4 unchanged: the stored prospective values split by h_game_equivalents.split_prospective (train = 2023-24 seasons of the 2023-24 classes; test = 2025 class, 2025 season); h_game_equivalents.fit_prospective on the six sets (WR traits also the deep-share sets, P6); training role share from 2023-24 qualifying games only; the 2025 prior for a role set = ridge prediction with the draft-year dummy at its training mean and the role share from his first k qualifying 2025 games. Checked at run time: training seasons <= 2024 and classes < 2025; the fit is unchanged when every 2025 alignment is altered; the k-game priors are unchanged when the alignments of games after k are altered; the C0 / C2 / C3 forecast errors equal E012's stored e012_forecasts_2025 rows.", "reason": "Prereg leakage rules: nothing from 2025 in any fitted quantity; role share in the test from the first k qualifying games only."},
 {"id": "P2", "topic": "forecasts and metric", "choice": "h_game_equivalents.in_season_forecasts run separately at each k in 1..4 with that k's role shares (i_role_aware.in_season_role_forecasts): 2025 rookies with >= k + 3 qualifying games; target = mean of games k+1..end; blend = mu_C + w_k (ybar_k - mu_C), w_k = k / (k + lambda / (1 - R2_C)) (E012 A1) with lambda, R2_C and tau2 from the 2023-24 fit; standardized SE = (forecast - target)^2 / tau2_train (position-level). No role prior at k = 0. Pooled = equal-weight mean over the 9 traits with >= 1 eligible player at that k of the trait's mean standardized SE.", "reason": "Prereg prospective_test design and metric."},
 {"id": "P3", "topic": "primary tests H1-H4", "choice": "H1: se_blend_C0_role vs se_blend_C0 at k = 1; H2: the same at k = 2; H3: se_blend_C2_role vs se_blend_C0_role at k = 1; H4: se_blend_C3_role vs se_blend_C2_role at k = 1. Player-level paired difference d_i = sum over his traits of (SE_a - SE_b) / (T n_t) (E012 P6); one-sided sign-flip test, 100,000 flips, expected sign negative, seeds 20261018 + 1..4; player bootstrap (10,000) 95% CIs for both pooled means, the difference and the ratio; BH across H1-H4 at alpha 0.05; pass = q < 0.05 and difference < 0.", "reason": "Prereg primary_tests and inference."},
 {"id": "P4", "topic": "team-cluster sensitivity", "choice": "Team of a 2025 rookie = the team_abbr on most of his 2025 REG player_play rows (ties: alphabetical first; none: his own cluster). For every primary and secondary test: sign flips of team sums of d_i (100,000) and a team-cluster bootstrap (10,000 resamples of teams, each with all its players) for the difference and the ratio. BH across H1-H4 on the team-cluster p values is reported as a sensitivity; the decision uses the player-level test.", "reason": "Prereg inference: team-cluster sign flips as a sensitivity (E012 review)."},
 {"id": "P5", "topic": "secondary family", "choice": "Descriptive tests, each with a player-level one-sided sign-flip p, player-bootstrap CIs and the team-cluster versions; BH across the whole family (player-level p) reported as q_family. (i) per trait (9 x 4): C0_role vs C0 at k = 1 and k = 2, C2_role vs C0_role at k = 1, C3_role vs C2_role at k = 1; (ii) per position (4 x 4): the same four comparisons pooled with equal weight over the position's traits (WR T1/T2/T3/T5, DL T6/T7, OL T9, DB T11/T12); (iii) k = 3 and k = 4 (2 x 3): C0_role vs C0, C2_role vs C0_role, C3_role vs C2_role, pooled; (iv) oracle role (4): H1-H4 with role share from all his 2025 qualifying games (an upper bound: it uses the target games); (v) WR deep-route share (5): deep share (GO/POST/CORNER/WHEEL among routes run, over the same plays as role share; training over 2023-24 qualifying games, test over his first k) appended to every role set for the 4 WR traits: H1-H4 analogues on the WR traits, and C0_role+deep vs C0_role at k = 1. A test runs when >= 3 players are eligible (otherwise n is reported and the test is not in the family). Seeds 20261018 + 1000 + its index in this order.", "reason": "Prereg secondary list; one fixed family for multiplicity."},
 {"id": "P6", "topic": "descriptive prospective outputs", "choice": "Pooled curves at k = 1..4 for games-only, every prior alone and every blend (player-bootstrap 95% CIs, common rows); 80% interval coverage of the C0_role / C2_role / C3_role blends with E012's P7 formula (all quantities from the 2023-24 fit); per-trait mean SEs.", "reason": "E012 P7 analogue."},
 {"id": "D1", "topic": "diagnostics (descriptive, computed after this freeze)", "choice": "Per trait: role share distribution of the analysis players (mean, quartiles, shares >= 0.9 and <= 0.1, missing); correlation of the first-k role share with the oracle role share for test rookies (k = 1, 2); within-player correlation of per-game role share with per-game trait value (both centred by player; the same-game share of role that the internal fit can carry because role and trait come from the same games); Pearson r of the draft-year-centred player trait with role share; E012's NFL roster-position reference for context.", "reason": "Interpretation aids, fixed before results."},
 {"id": "D2", "topic": "known design facts (from E012's definitions, not results)", "choice": "T6 / T7 play values are centred by EDGE / INTERIOR_LINE alignment, T9 by T / G / C, T11 by alignment x coverage cell, and T2 / T3 by route type, so for those traits role share can add only what play-level centring leaves; T1, T5 and T12 are not alignment-centred.", "reason": "Stated before results so it cannot be used selectively."},
 {"id": "X1", "topic": "compute and files", "choice": "joblib n_jobs 3 with one BLAS/OpenMP thread per worker, threadpool_limits(3) at top; bootstrap and simulation checkpoints in the session scratchpad (removed at the end); outputs data/processed/e013_*.parquet (role shares, bootstrap draws, simulation belts, coverage, forecasts), experiments/results/E013.json, experiments/figures/E013_*.png. Figures: per trait, what the position average, the role-aware average and the Combine + draft beyond role are worth in games (B1 intervals, calibrated bounds, E012's CGE(C2) as reference) and the prospective test; layout decided on viewing (post hoc, no numbers).", "reason": "Project compute rules; prereg deliverables."}
]
''')
POST_HOC: list[dict] = [  # decided after results were seen (quick-mode run of the frozen pipeline, 2026-10-19)
    {"id": "PH1", "what": "Figure layout after viewing the quick-run figures: Figure 1 split into three panels (prior "
     "worth with the position-average interval only, since the three priors share lambda's uncertainty; role "
     "increment; Combine-and-draft-beyond-role increment) with calibrated bounds drawn as light bands; Figure 2's "
     "left panel shows paired error ratios against the position-average blend (PH2) instead of overlapping marginal "
     "curves; row spacing.", "affects_numbers": False},
    {"id": "PH2", "what": "Descriptive, for Figure 2: paired error ratio of each blend (E012 C2 / C3 and the three role "
     "sets) to the position-average (C0) blend at k = 1..4 with player-bootstrap 95% intervals "
     "(prospective.descriptive.post_hoc_ratio_vs_position_average_blend). No test.", "affects_numbers": False},
    {"id": "PH3", "what": "Headline and summary (write_up) are built from the result keys with wording rules chosen "
     "after the results were seen (which quantities to quote, 'did not improve' / 'not significant' conditionals). "
     "No number is changed.", "affects_numbers": False},
    {"id": "BUG1", "what": "Code fix found by the quick run (not a choice change): belt_bounds listed the C0_role oracle "
     "column twice for the role-worth family (polars DuplicateError).", "affects_numbers": False},
    # ---- after the E013 review (2026-10-19; two reviewers, both 'weakened'): section 6, post_hoc_review in E013.json
    {"id": "PH4", "what": "Prospective variants for the training-vs-test precision mismatch (the coefficient is trained "
     "on whole-history role shares but fed a first-k share): (a) calibrated: the test share mapped by a training-only "
     "regression of whole-history on first-k share; (b) matched: the fit refitted with each training player's first-k "
     "share; (c) all_games: the prereg's literal training share over all 2023-24 REG games. H1-H4 and the WR "
     "deep-share analogues under each, with the preregistered tests' machinery; descriptive (BH over the post hoc "
     "family).", "affects_numbers": False, "review": "stats 1"},
    {"id": "PH5", "what": "Decompositions of H3 and the WR secondary: without T3; the T3 influence (players' shares of "
     "the role-aware blend's excess error); the role-aware + Combine and draft blend and E012's C2 blend against the "
     "position-average blend as paired tests at k = 1, 2; the team-cluster family q of the WR test surfaced.",
     "affects_numbers": False, "review": "stats 2, football 4"},
    {"id": "PH6", "what": "Role worth from the preregistered ANOVA lambda_role: lambda_role - lambda (the population "
     "worth of an exactly known linear role effect, derivation 1) and the share of between-player variance role "
     "explains, with B1 draws (paired), a within-class permutation test (10,000), the estimator at each trait's design "
     "(500 datasets per rho), the CV estimator's detection floor from the role belt, and the population-signal "
     "calibrated bounds quoted beside the CV ones.", "affects_numbers": False, "review": "stats 3, football 1"},
    {"id": "PH7", "what": "Power: one-sided minimum detectable error ratio (80% power) for H1-H4 from the tests' own "
     "bootstrap draws, and the model-based best-case H1 / H2 ratio if role were learned exactly at the internal ANOVA "
     "share.", "affects_numbers": False, "review": "stats 4"},
    {"id": "PH8", "what": "Same-CV references against a ridge on the draft-year dummies alone (C0_ridge), replacing the "
     "C0 reference for every 'relative to C0' sensitivity (role worth; E012's C2 / C3; C2_role / C3_role): the C0 "
     "reference mixed ridge shrinkage of the class means into the role (or Combine) value. B1 intervals from a post "
     "hoc bootstrap on E012's draws and CV seeds (data/processed/e013_posthoc_bootstrap.parquet); CV seeds 0-4.",
     "affects_numbers": False, "review": "stats 5"},
    {"id": "PH9", "what": "Both orders of entry for role and the Combine + draft (and + tracking): role beyond Combine + "
     "draft, Combine + draft beyond role, and the shared part, for every trait with B1 draws; and how predictable "
     "role share is from the Combine and draft (CV R2).", "affects_numbers": False, "review": "stats 6, football 3"},
    {"id": "PH10", "what": "Roster designation (players.csv nfl_position; possibly the latest roster label) against role "
     "share, descriptive and in sample: DL OLB vs DE among pure edge players, OLS (HC1) of the centred trait on edge "
     "share, weight and OLB; DB CB vs safety; OL T vs G / C.", "affects_numbers": False, "review": "football 2"},
    {"id": "PH11", "what": "Role-share semantics: the alignment vocabulary of each position's REG rows, median role share "
     "by roster designation, and each trait's actual play-level alignment adjustment (corrects the frozen D2's "
     "grouping of T2 / T3 and T5).", "affects_numbers": False, "review": "football 5, 6"},
    {"id": "PH12", "what": "Figures: Figure 1 retitled (role measured from his games, not pre-season), the band labelled "
     "'each end one-sided 95%', hollow markers for the ANOVA role worth and for role beyond Combine + draft, a note "
     "that B1 is biased low; Figure 2 gains post hoc rows (calibrated H1 / H3, H3 without T3, role-aware + Combine and "
     "draft vs position average).", "affects_numbers": False, "review": "football 8"},
    {"id": "PH13", "what": "Headline and summary rewritten from the result keys after the review: the preregistered "
     "verdicts first, then bounded statements (the test as 'alignment share from his first qualifying games', role "
     "real between players where the ANOVA says so, both orders of the DL attribution, H3 beside the "
     "position-average comparison, power, qualifying games with rookie appearances). Supersedes PH3's wording.",
     "affects_numbers": False, "review": "both"},
]

INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, VIOLET = "#2a78d6", "#eb6834", "#4a3aa7"  # E012's validated categorical slots
RAMP = {"C0_role": "#3987e5", "C2_role": "#1c5cab", "C3_role": "#0d366b"}  # E012's validated ordinal blue ramp
MARK = {"C0": "o", "C0_role": "s", "C2_role": "D", "C3_role": "^"}
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


def check_frozen_choices() -> dict:
    """The freeze file must exist and match the copy above (sha and content)."""
    f = AUDIT / "implementation_choices_frozen.json"
    assert f.exists(), "freeze file missing: freeze the implementation choices before running"
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    same = json.loads(f.read_text())["choices"] == IMPLEMENTATION_CHOICES
    assert sha == CHOICES_SHA256 and same, "implementation choices differ from the frozen file"
    return {"freeze_file": str(f.relative_to(ROOT)), "sha256_matches": True, "content_matches": True}


def ckpt(name: str, build, *args) -> pl.DataFrame:
    """Checkpointed table: with --resume, read the scratch checkpoint if present; else build and save it."""
    p = CKPT / f"{name}.parquet"
    if RESUME and p.exists():
        return pl.read_parquet(p)
    d = build(*args)
    CKPT.mkdir(parents=True, exist_ok=True)
    d.write_parquet(p)
    return d


def tidx(t: str) -> int:
    """E012's trait index (seeds)."""
    return list(H.TRAITS).index(t)


# =====================================================================================================
# 1. Inputs
# =====================================================================================================

def load_inputs() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    players = pl.read_parquet(DATA / "e012_players.parquet")
    v2 = [c for c in players.columns if c.endswith("__v2")]
    players_p = players.drop([c[:-4] for c in v2]).rename({c: c[:-4] for c in v2}).sort("nfl_id")
    players = players.drop(v2).sort("nfl_id")
    keep = pl.col("trait").is_in(list(I.TRAITS))
    games_i = pl.read_parquet(DATA / "e012_game_values.parquet").filter(keep)
    games_p = pl.read_parquet(DATA / "e012_prospective_values.parquet").filter(keep)
    return players, players_p, games_i, games_p


def alignment_inputs(players: pl.DataFrame) -> tuple[dict, pl.DataFrame, dict]:
    """Per-position player-game alignment counts (WR also routes), 2025 rookies' teams, checks."""
    reg = io.load("games").filter(pl.col("season_type") == "REG").select("game_id", "season")
    pp = (pl.scan_parquet(DATA / "player_play.parquet")
          .select("nfl_id", "game_id", "play_id", "lined_up_position", "route_ran", "team_abbr")
          .join(reg.lazy(), on="game_id", how="inner").collect())
    n_rows = pp.height
    assert pp.select(pl.struct("game_id", "play_id", "nfl_id").is_unique().all()).item()
    pp = pp.join(players.select("nfl_id", "position", "draft_year"), on="nfl_id", how="inner")
    counts = {}
    for pos in I.ROLE_ALIGN:
        cols = ["nfl_id", "game_id", "lined_up_position"] + (["route_ran"] if pos == "WR" else [])
        counts[pos] = I.game_alignment_counts(pp.filter(pl.col("position") == pos).select(cols), pos).join(
            reg, on="game_id", how="left", maintain_order="left")
    teams = (pp.filter((pl.col("season") == TEST) & (pl.col("draft_year") == TEST) & pl.col("team_abbr").is_not_null())
             .group_by("nfl_id", "team_abbr").len()
             .sort(["nfl_id", "len", "team_abbr"], descending=[False, True, False])
             .group_by("nfl_id", maintain_order=True).first().select("nfl_id", "team_abbr"))
    checks = {"player_play_reg_rows": n_rows, "rows_study_players": pp.height,
              "player_games_by_position": {p: c.height for p, c in counts.items()},
              "rookies_2025_with_team": teams.height, "n_teams_2025_rookies": teams["team_abbr"].n_unique()}
    # post hoc (PH11): the alignment vocabulary of each Combine position's REG rows (counted, not used by any estimate)
    vocab = {pos: dict(pp.filter(pl.col("position") == pos).group_by("lined_up_position").len()
                       .sort("len", descending=True).iter_rows()) for pos in I.ROLE_ALIGN}
    return counts, teams, checks, {p: {str(k): int(v) for k, v in d.items()} for p, d in vocab.items()}


# =====================================================================================================
# 2. Internal (2023-25)
# =====================================================================================================

def two(boot: pl.DataFrame, col: str, point: float) -> dict:
    d = boot[col].to_numpy() if col in boot.columns else np.array([np.nan])
    ci = H.percentile_ci(d)
    fin = np.isfinite(d)
    return {"est": point, "lo": ci["lo"], "hi": ci["hi"], "share_nonfinite": ci["share_nonfinite"],
            "median_draw": float(np.median(d[fin])) if fin.any() else None,
            "share_draws_below_est": float(np.mean(d < point - 1e-12)) if point is not None and np.isfinite(point)
            else None}


def role_summary(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    f = x[np.isfinite(x)]
    return {"n": int(len(x)), "n_missing": int(np.sum(~np.isfinite(x))), "mean": float(f.mean()),
            "q25": float(np.quantile(f, 0.25)), "median": float(np.median(f)), "q75": float(np.quantile(f, 0.75)),
            "share_ge_0.9": float(np.mean(f >= 0.9)), "share_le_0.1": float(np.mean(f <= 0.1))}


def internal_trait(t: str, players: pl.DataFrame, games_i: pl.DataFrame, counts: dict,
                   e12_boot: pl.DataFrame | None) -> tuple[dict, dict]:
    spec = H.TRAITS[t]
    pos = spec["pos"]
    sets = I.role_sets(spec["matched"])
    g = games_i.filter(pl.col("trait") == t).select("nfl_id", "game_id", "season", "value", "n_plays")
    role = I.role_share(counts[pos], g)
    pp = players.filter(pl.col("position") == pos).join(role, on="nfl_id", how="left")
    td = H.trait_data(g, pp, sets)
    assert td.X["C0_role"].shape[1] == len(td.levels)  # dummies (levels - 1) + role share last
    est = I.estimate_role(td, keep_oof=True)
    if REUSE:
        boot = pl.read_parquet(DATA / "e013_bootstrap.parquet").filter(pl.col("trait") == t).drop("trait")
    else:
        boot = ckpt(f"boot_{t}", lambda: I.bootstrap_role(td, N_BOOT, E012_SEED + tidx(t), N_JOBS, BOOT_REPEATS))
    e12 = E012["traits"][t]
    rep = {"lambda": [est["lambda"], e12["variance"]["lambda"]["est"]], "rel": [est["rel"], e12["variance"]["rel"]["est"]],
           **{f"cge_{s}": [est["cge"][s], e12["cge"][s]["est"]] for s in ("C2", "C3")}}
    repro = {"point_max_abs_diff": float(max(abs(a - b) for a, b in rep.values())), "values_e013_e012": rep}
    if e12_boot is not None and not QUICK:
        b12 = e12_boot.filter((pl.col("trait") == t) & (pl.col("kind") == "prereg_refit_B1")).sort("b")
        repro["bootstrap_cge_C2_max_abs_diff"] = float(np.nanmax(np.abs(b12["cge_C2"].to_numpy()
                                                                        - boot.sort("b")["cge_C2"].to_numpy())))
        repro["bootstrap_lambda_max_abs_diff"] = float(np.nanmax(np.abs(b12["lambda"].to_numpy()
                                                                        - boot.sort("b")["lambda"].to_numpy())))
    S = list(sets)
    res = {
        "position": pos, "name": spec["name"], "role": I.ROLE_NAME[pos], "n_players": est["n"],
        "n_games": est["n_games"], "mean_games": est["mean_games"],
        "role_share": role_summary(I.role_column(td)),
        "variance": {"tau2": est["tau2"], "sigma2": est["sigma2"], "rel": est["rel"]},
        "lambda": two(boot, "lambda", est["lambda"]),
        "lambda_role_anova": two(boot, "lambda_role_anova", est["lambda_role_anova"]),
        "tau2_role_anova": two(boot, "tau2_role_anova", est["tau2_role_anova"]),
        "lambda_role_cv": two(boot, "lambda_role_cv", est["lambda_role_cv"]),
        "r2_cv": est["r2_cv"],
        "r2_true": {s: two(boot, f"r2_{s}", est["r2"][s]) for s in S},
        "cge": {s: two(boot, f"cge_{s}", est["cge"][s]) for s in S},
        "role_worth": two(boot, "role_worth", est["role_worth"]),
        "role_worth_vs_C0_same_cv": two(boot, "role_worth_vs_C0_same_cv", est["role_worth_vs_C0_same_cv"]),
        "beyond_role": {s: two(boot, f"beyond_{s}", v) for s, v in est["beyond_role"].items()},
        "combine_change_once_role_known": two(boot, "combine_change_once_role_known",
                                              est["combine_change_once_role_known"]),
        "half_weight": {s: two(boot, f"khalf_{s}", est["half_weight"][s]) for s in S},
        "weights": {s: {k: two(boot, f"w{k}_{s}", v) for k, v in est["weights"][s].items()} for s in S},
        "e012_reproduction": repro,
        "bootstrap": {"n": N_BOOT, "cv_repeats": BOOT_REPEATS, "seed": E012_SEED + tidx(t),
                      "share_tau2_le_0": float((boot["tau2"] <= 0).mean())},
    }
    return res, {"td": td, "est": est, "boot": boot, "games": g, "players": pp}


# =====================================================================================================
# 3. Simulation at each trait's own design
# =====================================================================================================

def _sim_cell(fam: str, t: str, des: I.SimDesign, lam: float, rho_hat: float, gi: int, val: float, reps: int) -> list:
    sets, _, est_set = FAMILIES[fam]
    fi = list(FAMILIES).index(fam)
    rows = []
    with threadpool_limits(1):
        for r in range(reps):
            rng = np.random.default_rng([SEED, fi, tidx(t), gi, r])
            rho, g = (val, 0.0) if fam == "role" else (rho_hat, val)
            sim, theta = I.simulate_role_values(des, lam, rho, g, rng)
            e = H.estimate(sim, keep_oof=True)
            row = {"family": fam, "trait": t, "grid": gi, "value": val, "rho": rho, "g": g, "rep": r,
                   "lam_hat": e["lambda"], "rel_hat": e["rel"]}
            for s in sets[1:]:
                row |= {f"r2cv_{s}": e["r2_cv"][s], f"cge_{s}": e["cge"][s],
                        f"oracle_{s}": H.oracle_r2(theta, e["oof"][s])}
            row["est"] = e["cge"]["C0_role"] if fam == "role" else I.cge_beyond(e["cge"][est_set], e["cge"]["C0_role"])
            rows.append(row)
    return rows


def sim_family(fam: str, t: str, art: dict, internal_t: dict) -> pl.DataFrame:
    from joblib import Parallel, delayed
    sets, sig, _ = FAMILIES[fam]
    des = I.sim_design(art["td"], sets, sig)
    lam = internal_t["lambda"]["est"]
    rho_hat = float(np.clip(internal_t["r2_true"]["C0_role"]["est"], 0, RHO_MAX))
    grid = ROLE_GRID if fam == "role" else SIM_GRID
    res = Parallel(n_jobs=N_JOBS)(delayed(_sim_cell)(fam, t, des, lam, rho_hat, gi, v, SIM_REPS)
                                  for gi, v in enumerate(grid))
    return pl.DataFrame([r for rows in res for r in rows], infer_schema_length=None)


def belt_bounds(belt: pl.DataFrame, fam: str, lam: float, obs: float) -> dict:
    """Neyman bounds for one family x trait: estimand per grid point from the mean oracle R2."""
    _, _, est_set = FAMILIES[fam]
    used = list(dict.fromkeys(("C0_role", est_set)))
    st = (belt.group_by("grid", "value", "rho", "g").agg(
        *[pl.col(f"oracle_{s}").mean() for s in used],
        est_mean=pl.col("est").mean(), est_median=pl.col("est").median(),
        p_zero=(pl.col("est").abs() <= 1e-12).mean().cast(pl.Float64), n=pl.len()).sort("grid"))
    c = lambda r: H.cge(lam, float(np.clip(r, 0, H.R2_MAX)))  # noqa: E731
    if fam == "role":
        star = np.array([c(r) for r in st["oracle_C0_role"].to_list()])
        pop = np.array([H.cge(lam, v) for v in st["value"].to_list()])
    else:
        star = np.array([c(a) - c(b) for a, b in zip(st[f"oracle_{est_set}"].to_list(), st["oracle_C0_role"].to_list())])
        pop = np.array([I.population_beyond(lam, r, g) for r, g in zip(st["rho"].to_list(), st["g"].to_list())])
    draws = [belt.filter(pl.col("grid") == gi)["est"].to_numpy() for gi in st["grid"].to_list()]
    b = H.calibrated_bounds(obs, star, draws)
    bp = H.calibrated_bounds(obs, pop, draws)
    grid = st.with_columns(estimand_cv=pl.Series(star), estimand_population=pl.Series(pop),
                           **{f"mean_oracle_r2_{s}": pl.col(f"oracle_{s}") for s in used}).drop(
        [f"oracle_{s}" for s in used])
    return {"observed": obs, "lambda_used": lam, "calibrated_one_sided_95": {"lower": b["lower"], "upper": b["upper"]},
            "calibrated_one_sided_95_population_signal": {"lower": bp["lower"], "upper": bp["upper"]},
            "grid": grid.to_dicts(), "p_estimate_le_observed": b["p_le_obs"], "p_estimate_ge_observed": b["p_ge_obs"]}


def _cov_dataset(t: str, des: I.SimDesign, lam: float, rho: float, g: float, d: int) -> dict:
    rng = np.random.default_rng([SEED, 99, tidx(t), d])
    with threadpool_limits(1):
        sim, theta = I.simulate_role_values(des, lam, rho, g, rng)
        e = H.estimate(sim, keep_oof=True)
        b = H.bootstrap(sim, COV_B1_RESAMPLES, int(rng.integers(2**31)), 1, H.N_REPEATS)
    ci_r = H.percentile_ci(b["cge_C0_role"].to_numpy())
    ci_b = H.percentile_ci((b["cge_C2_role"] - b["cge_C0_role"]).to_numpy())
    return {"trait": t, "dataset": d, "rho": rho, "g": g, "role_worth_hat": e["cge"]["C0_role"],
            "beyond_hat": I.cge_beyond(e["cge"]["C2_role"], e["cge"]["C0_role"]),
            "oracle_C0_role": H.oracle_r2(theta, e["oof"]["C0_role"]),
            "oracle_C2_role": H.oracle_r2(theta, e["oof"]["C2_role"]),
            "role_lo": ci_r["lo"], "role_hi": ci_r["hi"], "beyond_lo": ci_b["lo"], "beyond_hi": ci_b["hi"]}


def coverage_runs(t: str, art: dict, internal_t: dict, bounds_c2: dict) -> pl.DataFrame:
    from joblib import Parallel, delayed
    des = I.sim_design(art["td"], FAMILIES["beyond_C2"][0], "C2_role")
    lam = internal_t["lambda"]["est"]
    rho = float(np.clip(internal_t["r2_true"]["C0_role"]["est"], 0, RHO_MAX))
    obs = internal_t["beyond_role"]["C2_role"]["est"]
    g = min(bounds_c2["grid"], key=lambda r: abs(r["estimand_cv"] - obs))["g"]
    res = Parallel(n_jobs=N_JOBS)(delayed(_cov_dataset)(t, des, lam, rho, g, d) for d in range(COV_DATASETS))
    return pl.DataFrame(res, infer_schema_length=None)


def _covered(lo, hi, x) -> np.ndarray:
    lo = np.array([-np.inf if v is None or not np.isfinite(v) else v for v in lo], float)
    hi = np.array([np.inf if v is None or not np.isfinite(v) else v for v in hi], float)
    return (lo <= x + 1e-9) & (x <= hi + 1e-9)


def coverage_summary(cov: pl.DataFrame, lam: float) -> dict:
    c = lambda r: H.cge(lam, float(np.clip(r, 0, H.R2_MAX)))  # noqa: E731
    star_r = c(cov["oracle_C0_role"].mean())
    star_b = c(cov["oracle_C2_role"].mean()) - star_r
    out = {"rho": float(cov["rho"][0]), "g": float(cov["g"][0]), "n_datasets": cov.height}
    for k, star, hat in (("role_worth", star_r, "role_worth_hat"), ("beyond_C2_role", star_b, "beyond_hat")):
        pre = "role" if k == "role_worth" else "beyond"
        cv = _covered(cov[f"{pre}_lo"].to_list(), cov[f"{pre}_hi"].to_list(), np.full(cov.height, star))
        out[k] = {"estimand_cv": star, "mean_estimate": float(cov[hat].mean()), "coverage": float(cv.mean()),
                  "se": float(np.sqrt(cv.mean() * (1 - cv.mean()) / len(cv)))}
    return out


# =====================================================================================================
# 4. Prospective
# =====================================================================================================

def _late_flags(df: pl.DataFrame, late: pl.DataFrame) -> pl.Series:
    return df.join(late.with_columns(_late=pl.lit(True)), on=["nfl_id", "game_id"], how="left",
                   maintain_order="left")["_late"].fill_null(False)


def prospective(players_p: pl.DataFrame, games_p: pl.DataFrame, counts: dict) -> tuple[pl.DataFrame, dict, dict, dict]:
    rows, summ, checks, ctx = [], {}, {}, {}
    for t in I.TRAITS:
        spec = H.TRAITS[t]
        pos = spec["pos"]
        sets = I.role_sets(spec["matched"])
        if pos == "WR":
            sets |= {k: v for k, v in I.role_sets(spec["matched"], (I.DEEP,), "_deep").items() if k.endswith("_deep")}
        pp = players_p.filter(pl.col("position") == pos)
        g = games_p.filter(pl.col("trait") == t).select("nfl_id", "game_id", "season", "value")
        gtr, gte = H.split_prospective(g, pp, TRAIN_SEASONS, TEST, TEST)
        assert gtr["season"].max() <= max(TRAIN_SEASONS) and set(gte["season"].unique().to_list()) <= {TEST}
        cnt = counts[pos]
        role_tr = I.role_share(cnt, gtr)
        assert set(role_tr["nfl_id"].to_list()) <= set(pp.filter(pl.col("draft_year") < TEST)["nfl_id"].to_list())
        fit = H.fit_prospective(gtr, pp.join(role_tr, on="nfl_id", how="left"), sets)
        # leakage check 1: every 2025-season alignment (any class) altered -> the fit is unchanged
        is25 = pl.col("season") == TEST
        alt = cnt.with_columns(role_num=pl.when(is25).then(pl.col("role_den") - pl.col("role_num"))
                               .otherwise(pl.col("role_num")))
        assert alt.filter(is25).height > 0
        fit_alt = H.fit_prospective(gtr, pp.join(I.role_share(alt, gtr), on="nfl_id", how="left"), sets)
        same_fit = fit_alt.r2 == fit.r2 and fit_alt.vc == fit.vc
        tp = pp.filter(pl.col("draft_year") == TEST)
        by_k = {k: tp.join(I.role_share(cnt, I.first_k_games(gte, k)), on="nfl_id", how="left") for k in KS}
        F = I.in_season_role_forecasts(fit, gte, by_k, KS)
        # leakage check 2: alignments of games after k altered -> the k-game role priors are unchanged
        jj = gte.sort("nfl_id", "game_id").with_columns(j=pl.int_range(1, pl.len() + 1).over("nfl_id"))
        same_k = {}
        for k in (1, 2):
            late = jj.filter(pl.col("j") > k).select("nfl_id", "game_id")
            fl = _late_flags(cnt, late)
            alt_k = cnt.with_columns(role_num=pl.when(fl).then(pl.col("role_den") - pl.col("role_num"))
                                     .otherwise(pl.col("role_num")))
            bk = tp.join(I.role_share(alt_k, I.first_k_games(gte, k)), on="nfl_id", how="left")
            Fk = I.in_season_role_forecasts(fit, gte, {k: bk}, (k,))
            a, b = F.filter(pl.col("k") == k).sort("nfl_id"), Fk.sort("nfl_id")
            same_k[str(k)] = bool(all(np.allclose(a[f"mu_{s}"].to_numpy(), b[f"mu_{s}"].to_numpy(), equal_nan=True)
                                      for s in I.ROLE_SETS) and int(fl.sum()) > 0)
        oracle = tp.join(I.role_share(cnt, gte), on="nfl_id", how="left")
        Fo = I.in_season_role_forecasts(fit, gte, {k: oracle for k in KS}, KS).select(
            "nfl_id", "k", *[pl.col(f"se_blend_{s}").alias(f"se_blend_{s}_oracle") for s in I.ROLE_SETS])
        F = F.join(Fo, on=["nfl_id", "k"], how="left").with_columns(trait=pl.lit(t))
        rows.append(F)
        checks[t] = {"fit_unchanged_when_2025_alignments_altered": bool(same_fit),
                     "k_priors_unchanged_when_later_alignments_altered": same_k,
                     "n_train_players_with_role": int(role_tr[I.ROLE].is_not_null().sum())}
        lam = fit.vc["lambda"]
        cg = {s: H.cge(lam, r) for s, r in fit.r2.items()}
        summ[t] = {"n_train_players": int(fit.vc["n"]), "n_train_games": int(fit.vc["n_games"]),
                   "tau2": fit.vc["tau2"], "sigma2": fit.vc["sigma2"], "lambda": lam, "rel": fit.vc["rel"],
                   "r2_cv": fit.r2_cv, "r2_true": fit.r2, "cge": cg,
                   "role_worth": cg["C0_role"],
                   "beyond_role": {s: I.cge_beyond(cg[s], cg["C0_role"]) for s in ("C2_role", "C3_role")},
                   "half_weight": {s: lam / (1 - r) if np.isfinite(r) else None for s, r in fit.r2.items()},
                   "weights": {s: {str(k): float(H.blend_weight(k, lam, r)) for k in I.KS_TABLE}
                               for s, r in fit.r2.items() if np.isfinite(r)},
                   "n_test_players_any_game": gte["nfl_id"].n_unique()}
        ctx[t] = {"fit": fit, "sets": sets, "players": pp, "test_players": tp, "gtr": gtr, "gte": gte, "role_tr": role_tr,
                  "by_k": by_k}  # for the post hoc variants (section 6)
    return pl.concat(rows, how="diagonal_relaxed"), summ, checks, ctx


def e012_forecast_check(F: pl.DataFrame) -> dict:
    """The C0 / C2 / C3 errors of this run against E012's stored forecasts (same rows)."""
    e = pl.read_parquet(DATA / "e012_forecasts_2025.parquet").filter(
        pl.col("trait").is_in(list(I.TRAITS)) & (pl.col("k") >= 1))
    cols = ["se_games", "se_blend_C0", "se_blend_C2", "se_blend_C3", "se_comb_C2"]
    m = F.select("nfl_id", "trait", "k", *cols).join(e.select("nfl_id", "trait", "k", *cols), on=["nfl_id", "trait", "k"],
                                                     how="full", coalesce=True, suffix="_e012")
    out = {"rows_e013": F.height, "rows_e012": e.height, "rows_matched": int(m.drop_nulls().height)}
    for c in cols:
        out[f"{c}_max_abs_diff"] = float((m[c] - m[f"{c}_e012"]).abs().max())
    return out


def run_test(F: pl.DataFrame, a: str, b: str, seed: int, teams: dict, traits=None, k: int = 1) -> dict:
    d = F.filter(pl.col("k") == k)
    if traits is not None:
        d = d.filter(pl.col("trait").is_in(list(traits)))
    d = d.select("nfl_id", "trait", a, b).filter(pl.col(a).is_finite() & pl.col(b).is_finite())
    if d["nfl_id"].n_unique() < 3:
        return {"a": a, "b": b, "k": k, "n_players": d["nfl_id"].n_unique(), "tested": False}
    return {"k": k, "tested": True, **I.paired_test(d, a, b, seed, teams, N_FLIPS, N_BOOT_V)}


PRIMARY = [("H1", 1, "se_blend_C0_role", "se_blend_C0"), ("H2", 2, "se_blend_C0_role", "se_blend_C0"),
           ("H3", 1, "se_blend_C2_role", "se_blend_C0_role"), ("H4", 1, "se_blend_C3_role", "se_blend_C2_role")]


def prospective_tests(F: pl.DataFrame, teams: dict) -> dict:
    prim = {}
    for i, (h, k, a, b) in enumerate(PRIMARY):
        prim[h] = run_test(F, a, b, SEED + i + 1, teams, k=k)
        prim[h]["claim"] = PREREG["prospective_test"]["primary_tests"][i]["claim"]
    for h, q in zip(prim, H.bh([prim[h]["p_one_sided"] for h in prim])):
        prim[h]["q_bh"] = q
        prim[h]["pass"] = bool(q < 0.05 and prim[h]["diff"] < 0)
    for h, q in zip(prim, H.bh([prim[h]["team_cluster"]["p_one_sided"] for h in prim])):
        prim[h]["team_cluster"]["q_bh"] = q
    # secondary family (P5), in the frozen order
    comps = [("C0_role_vs_C0_k1", 1, "se_blend_C0_role", "se_blend_C0"),
             ("C0_role_vs_C0_k2", 2, "se_blend_C0_role", "se_blend_C0"),
             ("C2_role_vs_C0_role_k1", 1, "se_blend_C2_role", "se_blend_C0_role"),
             ("C3_role_vs_C2_role_k1", 1, "se_blend_C3_role", "se_blend_C2_role")]
    fam: dict = {}
    jobs = []
    for t in I.TRAITS:
        jobs += [(f"per_trait:{t}:{n}", k, a, b, [t]) for n, k, a, b in comps]
    for pos, ts in POSITIONS.items():
        jobs += [(f"per_position:{pos}:{n}", k, a, b, ts) for n, k, a, b in comps]
    for k in (3, 4):
        jobs += [(f"k{k}:C0_role_vs_C0", k, "se_blend_C0_role", "se_blend_C0", None),
                 (f"k{k}:C2_role_vs_C0_role", k, "se_blend_C2_role", "se_blend_C0_role", None),
                 (f"k{k}:C3_role_vs_C2_role", k, "se_blend_C3_role", "se_blend_C2_role", None)]
    jobs += [(f"oracle_role:{h}", k, a + "_oracle", (b + "_oracle") if b != "se_blend_C0" else b, None)
             for h, k, a, b in PRIMARY]
    wr = POSITIONS["WR"]
    jobs += [("deep_share:H1", 1, "se_blend_C0_role_deep", "se_blend_C0", wr),
             ("deep_share:H2", 2, "se_blend_C0_role_deep", "se_blend_C0", wr),
             ("deep_share:H3", 1, "se_blend_C2_role_deep", "se_blend_C0_role_deep", wr),
             ("deep_share:H4", 1, "se_blend_C3_role_deep", "se_blend_C2_role_deep", wr),
             ("deep_share:C0_role_deep_vs_C0_role_k1", 1, "se_blend_C0_role_deep", "se_blend_C0_role", wr)]
    for j, (name, k, a, b, ts) in enumerate(jobs):
        fam[name] = run_test(F, a, b, SEED + 1000 + j, teams, ts, k)
    tested = [n for n in fam if fam[n]["tested"]]
    for n, q in zip(tested, H.bh([fam[n]["p_one_sided"] for n in tested])):
        fam[n]["q_family"] = q
    for n, q in zip(tested, H.bh([fam[n]["team_cluster"]["p_one_sided"] for n in tested])):
        fam[n]["team_cluster"]["q_family"] = q
    return {"primary": prim, "alpha": 0.05, "secondary_family": fam,
            "secondary_family_size": len(tested), "secondary_not_tested": [n for n in fam if not fam[n]["tested"]]}


def descriptive_prospective(F: pl.DataFrame) -> dict:
    sets = ["C0", "C2", "C3", *I.ROLE_SETS]
    methods = ["se_games", *[f"se_comb_{s}" for s in sets], *[f"se_blend_{s}" for s in sets]]
    curves, calib, per_trait = {}, {}, {}
    for k in KS:
        d = F.filter(pl.col("k") == k).select("nfl_id", "trait", *methods).drop_nulls()
        bs = H.pooled_bootstrap(d, methods, N_BOOT_V, SEED + 500 + k)
        curves[str(k)] = {"n_players": d["nfl_id"].n_unique(), "n_traits": d["trait"].n_unique(),
                          **{m: {"est": H.pooled_mean(d, m), "lo": float(np.quantile(bs[m], 0.025)),
                                 "hi": float(np.quantile(bs[m], 0.975))} for m in methods}}
        calib[str(k)] = {}
        for s in sets:
            c = F.filter(pl.col("k") == k).select("nfl_id", "trait", f"cov_{s}").drop_nulls()
            calib[str(k)][s] = {"pooled": H.pooled_mean(c, f"cov_{s}"), "raw": float(c[f"cov_{s}"].mean()),
                                "n": c.height}
    for t in I.TRAITS:
        per_trait[t] = {}
        for k in KS:
            d = F.filter((pl.col("trait") == t) & (pl.col("k") == k))
            per_trait[t][str(k)] = {"n": d.height, **{m: float(d[m].mean()) for m in
                                                      ("se_games", *[f"se_blend_{s}" for s in sets]) if d.height}}
    # POST HOC (PH2): paired error ratio of each blend to the position-average (C0) blend, for the figure
    rat = {}
    for k in KS:
        rat[str(k)] = {}
        for j, s in enumerate(("C2", "C3", *I.ROLE_SETS)):
            a_, b_ = f"se_blend_{s}", "se_blend_C0"
            d = F.filter(pl.col("k") == k).select("nfl_id", "trait", a_, b_).drop_nulls()
            bs = H.pooled_bootstrap(d, [a_, b_], N_BOOT_V, SEED + 600 + 10 * k + j)
            rat[str(k)][s] = {"ratio": H.pooled_mean(d, a_) / H.pooled_mean(d, b_),
                              "lo": float(np.quantile(bs[a_] / bs[b_], 0.025)),
                              "hi": float(np.quantile(bs[a_] / bs[b_], 0.975)), "n_players": d["nfl_id"].n_unique()}
    return {"pooled_curves": curves, "calibration_80pct": calib, "per_trait_mean_se": per_trait,
            "post_hoc_ratio_vs_position_average_blend": rat}


# =====================================================================================================
# 5. Diagnostics, tables
# =====================================================================================================

def diagnostics(t: str, art: dict, counts: dict, games_p: pl.DataFrame, players_p: pl.DataFrame) -> dict:
    pos = H.TRAITS[t]["pos"]
    td, est, g = art["td"], art["est"], art["games"]
    y = est["y"].copy()
    for s in np.unique(td.year):
        y[td.year == s] -= y[td.year == s].mean()
    out = {"r_player_trait_role_share": corr(y, I.role_column(td))}
    pg = g.join(counts[pos].with_columns(rs=pl.when(pl.col("role_den") > 0).then(pl.col("role_num") / pl.col("role_den"))),
                on=["nfl_id", "game_id"], how="left").filter(pl.col("nfl_id").is_in(td.nfl_id.tolist()))
    pg = pg.filter(pl.col("rs").is_not_null()).with_columns(
        vc=pl.col("value") - pl.col("value").mean().over("nfl_id"), rc=pl.col("rs") - pl.col("rs").mean().over("nfl_id"))
    out["within_player_r_game_role_vs_game_value"] = corr(pg["vc"], pg["rc"])
    out["within_player_sd_game_role_share"] = float(pg["rc"].std())
    out["n_player_games"] = pg.height
    # first-k vs oracle role share for 2025 rookies with >= k + 3 qualifying games
    gp = games_p.filter((pl.col("trait") == t) & (pl.col("split") == "test")).select("nfl_id", "game_id")
    n = gp.group_by("nfl_id").len()
    orc = I.role_share(counts[pos], gp).select("nfl_id", o=I.ROLE)
    out["first_k_vs_oracle_r"] = {}
    for k in (1, 2):
        ids = n.filter(pl.col("len") >= k + 3).select("nfl_id")
        fk = I.role_share(counts[pos], I.first_k_games(gp.join(ids, on="nfl_id", how="semi"), k)).select("nfl_id", f=I.ROLE)
        m = fk.join(orc, on="nfl_id")
        out["first_k_vs_oracle_r"][str(k)] = {"r": corr(m["f"], m["o"]), "n": m.height,
                                              "mean_abs_diff": float((m["f"] - m["o"]).abs().mean())}
    e12 = {r["trait"]: r for r in E012["cge_table"]}[t]
    out["e012_roster_position_reference"] = {"cge_role_reference": e12["cge_role_reference"],
                                             "cge_C2_beyond_role": e12["cge_C2_beyond_role"]}
    return out


def trust_table(internal: dict, fit_summ: dict, ph: dict | None = None) -> list[dict]:
    share = E012["post_hoc_review"]["validation_2"]["qualifying_share_2025_rookies"]
    rows = []
    for t, r in internal.items():
        if ph is not None:  # post hoc (PH6, PH8, PH9): the role-aware prior's worth by the preregistered ANOVA lambda_role
            p = ph[t]
            lr = r["lambda_role_anova"]["est"]
            extra = {"post_hoc": {
                "role_worth_anova_lambda_role_minus_lambda": p["anova"]["worth_lambda_role_minus_lambda"],
                "role_share_of_between_player_variance_anova": p["anova"]["share_of_between_player_variance"],
                "role_aware_average_worth_anova_lambda_role": lr,
                "half_weight_change_vs_position_average_anova": lr / r["lambda"]["est"] - 1,
                "half_weight_rookie_appearances_anova_lambda_role": lr / share[t]["share"] if share[t]["share"] else None,
                "role_worth_vs_dummies_ridge_same_cv": p["same_cv_vs_dummies_ridge"]["role_worth"],
                "role_beyond_combine_draft": p["attribution_C2"]["role_beyond_c"],
                "shared_role_and_combine_draft": p["attribution_C2"]["shared"]}}
        else:
            extra = {}
        hw = r["half_weight"]
        sh = share[t]["share"]
        g3 = lambda d: {k: d.get(k) for k in ("est", "lo", "hi")}  # noqa: E731
        rows.append({
            "trait": t, "position": r["position"], "name": r["name"], "role": r["role"], "n_players": r["n_players"],
            "position_average_worth_lambda": g3(r["lambda"]),
            "role_aware_average_worth": g3(hw["C0_role"]),
            "role_aware_plus_combine_draft_worth": g3(hw["C2_role"]),
            "e012_combine_draft_worth": g3(hw["C2"]),
            "lambda_role_anova": g3(r["lambda_role_anova"]),
            "role_worth": g3(r["role_worth"]), "combine_draft_beyond_role": g3(r["beyond_role"]["C2_role"]),
            "tracking_beyond_role": g3(r["beyond_role"]["C3_role"]),
            "e012_cge_C2": g3(r["cge"]["C2"]),
            "combine_change_once_role_known": g3(r["combine_change_once_role_known"]),
            "role_worth_calibrated_one_sided_95": r.get("calibrated_bounds", {}).get("role", {}).get(
                "calibrated_one_sided_95"),
            "beyond_C2_role_calibrated_one_sided_95": r.get("calibrated_bounds", {}).get("beyond_C2", {}).get(
                "calibrated_one_sided_95"),
            "beyond_C3_role_calibrated_one_sided_95": r.get("calibrated_bounds", {}).get("beyond_C3", {}).get(
                "calibrated_one_sided_95"),
            "weights": {s: {k: g3(v) for k, v in r["weights"][s].items()} for s in ("C0", "C0_role", "C2", "C2_role")},
            "half_weight_rookie_appearances": {s: hw[s]["est"] / sh for s in ("C0", "C0_role", "C2_role")} if sh else None,
            "qualifying_share_2025_rookies_e012": sh,
            "fit_2023_24": {"lambda": fit_summ[t]["lambda"], "role_worth": fit_summ[t]["role_worth"],
                            "beyond_C2_role": fit_summ[t]["beyond_role"]["C2_role"],
                            "half_weight": {s: fit_summ[t]["half_weight"][s] for s in ("C0", "C0_role", "C2_role")}},
            **extra,
        })
    return rows


# =====================================================================================================
# 6. Post hoc: response to the E013 review (2026-10-19). Decided after the results were seen (POST_HOC PH4-PH13);
#    nothing here changes a preregistered number or verdict.
# =====================================================================================================
N_PERM = 500 if QUICK else 10_000
ANOVA_SIM_GRID = (0.0, 0.1, 0.2, 0.3)
ANOVA_SIM_REPS = 20 if QUICK else 500
CV_SEEDS = (0, 1, 2, 3, 4)
PH_BOOT = "e013_posthoc_bootstrap.parquet"
PH_FORECASTS = "e013_posthoc_forecasts_2025.parquet"
VARIANTS = ("calibrated", "matched", "all_games")


def two_d(d: np.ndarray, point: float) -> dict:
    """two() for an array of bootstrap draws (non-finite draws count as +inf, as in percentile_ci)."""
    return two(pl.DataFrame({"x": np.asarray(d, float)}), "x", point)


def anova_sim(t: str, td: H.TraitData, lam: float) -> dict:
    """PH6: the ANOVA role share at the trait's own design (its players, classes, game counts and role shares;
    theta = sqrt(rho) z + sqrt(1 - rho) u, games iid N(theta, lambda_hat)), ANOVA_SIM_REPS datasets per rho."""
    des = I.sim_design(td, ("C0", "C0_role"), None)
    out = {}
    for gi, rho in enumerate(ANOVA_SIM_GRID):
        s = []
        for r in range(ANOVA_SIM_REPS):
            sim, _ = I.simulate_role_values(des, lam, rho, 0.0, np.random.default_rng([SEED, 8, tidx(t), gi, r]))
            s.append(I.role_variance_share(sim)["share"])
        s = np.asarray(s)
        f = s[np.isfinite(s)]
        out[str(rho)] = {"mean": float(f.mean()), "median": float(np.median(f)), "sd": float(f.std(ddof=1)),
                         "q95": float(np.quantile(f, 0.95)), "share_undefined": float(np.mean(~np.isfinite(s))),
                         "n": int(len(s))}
    return out


def posthoc_internal_trait(t: str, art: dict, it: dict, phb: pl.DataFrame) -> dict:
    """PH6 (ANOVA role worth, permutation, simulation, CV-estimator detection), PH8 (same-CV references against a
    dummies-only ridge, with B1 draws and CV seeds), PH9 (both orders of attribution; role share from the Combine and
    draft), PH10 / PH11 (roster designation and role-share semantics) for one trait."""
    td, est, boot = art["td"], art["est"], art["boot"].sort("b")
    lam, rel, cg, r2cv = est["lambda"], est["rel"], est["cge"], est["r2_cv"]
    out: dict = {}
    # ---- PH6: the preregistered ANOVA lambda_role read as a role worth
    rv = I.role_variance_share(td)
    assert np.isclose(rv["lambda_role"], est["lambda_role_anova"], rtol=1e-10)
    t2, t2r, lb, lbr = (boot[c].to_numpy() for c in ("tau2", "tau2_role_anova", "lambda", "lambda_role_anova"))
    with np.errstate(invalid="ignore", divide="ignore"):
        share_b = np.where(t2 > 0, 1 - t2r / t2, np.nan)
        worth_b = np.where((t2 > 0) & (t2r > 0), lbr - lb, np.nan)
    out["anova"] = {
        "share_of_between_player_variance": two_d(share_b, rv["share"]),
        "worth_lambda_role_minus_lambda": two_d(worth_b, rv["worth"]),
        "lambda_role_over_lambda": rv["lambda_role"] / lam,
        "within_class_permutation": I.role_share_permutation(td, N_PERM, [SEED, 7, tidx(t)]),
        "simulation_at_design": anova_sim(t, td, lam)}
    cb = it["calibrated_bounds"]["role"]
    grid = [{"rho": r["value"], "population_worth": r["estimand_population"], "cv_estimand": r["estimand_cv"],
             "p_cv_estimate_zero": r["p_zero"]} for r in cb["grid"]]
    out["cv_estimator"] = {
        "role_worth": it["role_worth"], "calibrated_one_sided_95_cv": cb["calibrated_one_sided_95"],
        "calibrated_one_sided_95_population_signal": cb["calibrated_one_sided_95_population_signal"],
        "largest_population_worth_estimated_zero_half_the_time": max(
            [r["population_worth"] for r in grid if r["p_cv_estimate_zero"] >= 0.5], default=0.0),
        "grid": grid}
    # ---- PH8: same-CV references against the dummies-only ridge (B1 draws paired with the stored ones)
    dr = I.dummies_ridge(td)
    assert dr["lambda"] == lam and dr["rel"] == rel
    m = boot.join(phb.select("b", lambda_ph="lambda", rel_ph="rel", r2cv_C0_ridge="r2cv_C0_ridge"), on="b", how="left")
    same = lambda a, b_: bool(np.allclose(np.nan_to_num(a, posinf=9e9), np.nan_to_num(b_, posinf=9e9)))  # noqa: E731
    assert same(m["lambda"].to_numpy(), m["lambda_ph"].to_numpy()) and same(m["rel"].to_numpy(), m["rel_ph"].to_numpy())
    rc = lambda a, ref, l_, r_: H.cge(l_, H.relative_true_r2(a, ref, r_))  # noqa: E731

    def rel_draws(col: str, ref: str) -> np.ndarray:
        return np.array([rc(a, b_, l_, r_) for a, b_, l_, r_ in
                         zip(m[col].to_list(), m[ref].to_list(), m["lambda"].to_list(), m["rel"].to_list())], float)
    sets_rel = {"role_worth": "C0_role", "e012_C2": "C2", "e012_C3": "C3", "C2_role": "C2_role", "C3_role": "C3_role"}
    ref_e012 = {r["trait"]: r for r in E012["cge_table"]}[t]["cge_C2_vs_C0_same_cv"]
    assert np.isclose(rc(r2cv["C2"], r2cv["C0"], lam, rel), ref_e012, rtol=1e-9, atol=1e-12)
    seeds = {}
    for s in CV_SEEDS:
        if s == 0:
            r = dict(r2cv) | {"C0_ridge": dr["r2cv_C0_ridge"]}
        else:
            X = {k: td.X[k] for k in ("C0", "C0_role", "C2", "C3")} | {"C0_ridge": H.year_dummies(td.year, td.levels)}
            r = H.estimate(H.TraitData(nfl_id=td.nfl_id, year=td.year, X=X, g_player=td.g_player, g_value=td.g_value,
                                       levels=td.levels), seed=s)["r2_cv"]
        seeds[str(s)] = {"r2cv_C0": r["C0"], "r2cv_C0_ridge": r["C0_ridge"], "r2cv_C0_role": r["C0_role"],
                         "role_vs_dummies_ridge": rc(r["C0_role"], r["C0_ridge"], lam, rel),
                         "role_vs_C0": rc(r["C0_role"], r["C0"], lam, rel),
                         "e012_C2_vs_dummies_ridge": rc(r["C2"], r["C0_ridge"], lam, rel),
                         "e012_C2_vs_C0": rc(r["C2"], r["C0"], lam, rel)}
    out["same_cv_vs_dummies_ridge"] = {
        "r2cv_C0_ridge": dr["r2cv_C0_ridge"], "r2cv_C0": r2cv["C0"],
        **{k: two_d(rel_draws(f"r2cv_{s}", "r2cv_C0_ridge"), rc(r2cv[s], dr["r2cv_C0_ridge"], lam, rel))
           for k, s in sets_rel.items()},
        "against_C0_for_reference": {k: rc(r2cv[s], r2cv["C0"], lam, rel) for k, s in sets_rel.items()},
        "cv_seeds": seeds}
    # ---- PH9: both orders of entry (role and the Combine + draft; role and the + tracking set)
    for c, both in (("C2", "C2_role"), ("C3", "C3_role")):
        a = I.attribution(cg["C0_role"], cg[c], cg[both])
        ab = {k: np.asarray(v, float) for k, v in I.attribution(boot["cge_C0_role"].to_numpy(), boot[f"cge_{c}"].to_numpy(),
                                                              boot[f"cge_{both}"].to_numpy()).items()}
        out[f"attribution_{c}"] = {k: two_d(ab[k], a[k]) for k in a}
    assert np.isclose(out["attribution_C2"]["c_beyond_role"]["est"], it["beyond_role"]["C2_role"]["est"])
    assert np.isclose(out["attribution_C2"]["shared"]["est"], it["combine_change_once_role_known"]["est"])
    role = I.role_column(td)
    ok = np.isfinite(role)
    with threadpool_limits(N_JOBS):
        P = H.cv_oof({"C2": td.X["C2"][ok]}, role[ok], td.year[ok], None, H.N_REPEATS, 0)["C2"]
    pp = art["players"].filter(pl.col("nfl_id").is_in(td.nfl_id.tolist())).sort("nfl_id")
    assert np.array_equal(pp["nfl_id"].to_numpy(), td.nfl_id)
    out["role_share_from_combine_draft"] = {
        "cv_r2": float(np.mean(1 - np.mean((role[ok][None, :] - P) ** 2, axis=1) / role[ok].var())),
        "n": int(ok.sum()), "design": "E012's C2 design (draft-year dummies + traditional Combine + draft slot); "
                                      "linkage._model; RepeatedKFold 5 x 20 seed 0",
        **{f"r_{c}": corr(role, pp[c].to_numpy().astype(float)) for c in ("combine_weight", "combine_height", "forty")}}
    out["roster_designation"] = roster_designation(t, est, td, pp)
    return out


def _ols_r2(y: np.ndarray, *cols: np.ndarray) -> float:
    X = np.column_stack([np.ones(len(y)), *cols])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(1 - np.sum((y - X @ b) ** 2) / np.sum((y - y.mean()) ** 2))


def roster_designation(t: str, est: dict, td: H.TraitData, pp: pl.DataFrame) -> dict:
    """PH10 / PH11 (descriptive, in sample): the draft-year-centred player trait against role share and the roster
    designation (players.csv nfl_position, which may be the latest roster label rather than the draft-day one)."""
    import statsmodels.api as sm
    from scipy.stats import ttest_ind
    y = est["y"].copy()
    for s in np.unique(td.year):
        y[td.year == s] -= y[td.year == s].mean()
    role = I.role_column(td)
    r_imp = np.where(np.isfinite(role), role, np.nanmedian(role))
    pos = np.array([p if p is not None else "NA" for p in pp["nfl_position"].to_list()])
    w = pp["combine_weight"].to_numpy().astype(float)
    w = np.where(np.isfinite(w), w, np.nanmedian(w))
    lev = sorted(set(pos))
    Dm = np.column_stack([(pos == v).astype(float) for v in lev[1:]]) if len(lev) > 1 else np.zeros((len(y), 0))
    out = {"by_roster_position": {v: {"n": int(np.sum(pos == v)), "median_role_share": float(np.nanmedian(role[pos == v]))
                                      if np.isfinite(role[pos == v]).any() else None,
                                      "mean_centred_trait": float(y[pos == v].mean())} for v in lev},
           "r2_in_sample": {"role_share": _ols_r2(y, r_imp), "roster_position": _ols_r2(y, Dm),
                            "roster_position_and_role_share": _ols_r2(y, Dm, r_imp), "weight": _ols_r2(y, w),
                            "weight_and_role_share": _ols_r2(y, w, r_imp), "weight_and_roster_position": _ols_r2(y, w, Dm)},
           "r_trait_role_share": corr(y, role)}
    pd_ = H.TRAITS[t]["pos"]
    if pd_ == "DL":
        olb = (pos == "OLB").astype(float)
        fit = sm.OLS(y, sm.add_constant(np.column_stack([r_imp, w, olb]))).fit(cov_type="HC1")
        out["ols_hc1_role_weight_olb"] = {k: {"coef": float(fit.params[i + 1]), "p": float(fit.pvalues[i + 1])}
                                          for i, k in enumerate(("edge_share", "combine_weight", "olb"))}
        pure = np.isfinite(role) & (role >= 0.8)
        a, b = y[pure & (pos == "OLB")], y[pure & (pos == "DE")]
        out["pure_edge_olb_vs_de"] = {
            "edge_share_threshold": 0.8, "n_olb": int(len(a)), "n_de": int(len(b)), "mean_olb": float(a.mean()),
            "mean_de": float(b.mean()), "welch_p_two_sided": float(ttest_ind(a, b, equal_var=False).pvalue),
            "by_class": {str(s): {"olb": float(y[pure & (pos == "OLB") & (td.year == s)].mean()),
                                  "de": float(y[pure & (pos == "DE") & (td.year == s)].mean())}
                         for s in np.unique(td.year)}}
    if pd_ == "DB":
        cb = (pos == "CB").astype(float)
        a, b = y[pos == "FS"], y[pos == "SS"]
        out["cb_vs_safety"] = {"r2_cb_indicator": _ols_r2(y, cb), "r2_corner_share": _ols_r2(y, r_imp),
                               "fs_vs_ss_welch_p_two_sided": float(ttest_ind(a, b, equal_var=False).pvalue),
                               "n_fs": int(len(a)), "n_ss": int(len(b))}
    return out


def posthoc_variant_forecasts(ctx: dict, counts: dict) -> tuple[pl.DataFrame, dict]:
    """PH4: the prospective forecasts again under three post hoc readings of the role share (the preregistered fit
    and test design otherwise unchanged; training-only quantities throughout):
      calibrated: the test's first-k share mapped to the whole-history scale the coefficient was trained on, by a
                  regression of the training players' whole-history share on their own first-k share;
      matched:    the fit refitted with each training player's role share from his first k qualifying games (the
                  same precision as the test's);
      all_games:  the fit with the prereg's literal training share (all of his 2023-24 REG games, not only the
                  trait's qualifying games).
    WR traits carry the deep-share sets through each variant."""
    rows, calib, fits = [], {}, {}
    for t in I.TRAITS:
        c = ctx[t]
        cnt = counts[H.TRAITS[t]["pos"]]
        fit, sets, pp, gtr, gte, role_tr, by_k = (c[k] for k in ("fit", "sets", "players", "gtr", "gte", "role_tr",
                                                                   "by_k"))
        usage = [I.ROLE] + ([I.DEEP] if I.DEEP in role_tr.columns else [])
        tr_ids = pl.DataFrame({"nfl_id": fit.train_ids})
        gq = gtr.join(tr_ids, on="nfl_id", how="semi")
        assert gq["season"].max() <= max(TRAIN_SEASONS)
        full = role_tr.join(tr_ids, on="nfl_id", how="semi")
        calib[t], fits[t] = {}, {}
        for k in (1, 2):
            fk = I.role_share(cnt, I.first_k_games(gq, k)).join(full, on="nfl_id", how="inner", suffix="_full")
            assert set(fk["nfl_id"].to_list()) <= set(fit.train_ids.tolist())
            bk, cal = by_k[k], {}
            for col in usage:
                cal[col] = I.regression_calibration(fk[col].to_numpy(), fk[col + "_full"].to_numpy())
                bk = bk.with_columns((cal[col]["a"] + cal[col]["b"] * pl.col(col)).alias(col))
            calib[t][str(k)] = cal
            rows.append(I.in_season_role_forecasts(fit, gte, {k: bk}, (k,)).with_columns(
                trait=pl.lit(t), variant=pl.lit("calibrated")))
            fit_k = H.fit_prospective(gtr, pp.join(I.role_share(cnt, I.first_k_games(gtr, k)), on="nfl_id", how="left"),
                                      sets)
            fits[t][f"matched_k{k}"] = {"r2_true": fit_k.r2, "lambda": fit_k.vc["lambda"]}
            rows.append(I.in_season_role_forecasts(fit_k, gte, {k: by_k[k]}, (k,)).with_columns(
                trait=pl.lit(t), variant=pl.lit("matched")))
        allg = cnt.filter(pl.col("season") <= max(TRAIN_SEASONS)).select("nfl_id", "game_id")
        fit_a = H.fit_prospective(gtr, pp.join(I.role_share(cnt, allg), on="nfl_id", how="left"), sets)
        fits[t]["all_games"] = {"r2_true": fit_a.r2, "lambda": fit_a.vc["lambda"]}
        rows.append(I.in_season_role_forecasts(fit_a, gte, {k: by_k[k] for k in (1, 2)}, (1, 2)).with_columns(
            trait=pl.lit(t), variant=pl.lit("all_games")))
        fits[t]["prereg"] = {"r2_true": fit.r2, "lambda": fit.vc["lambda"]}
    return pl.concat(rows, how="diagonal_relaxed"), {"calibration": calib, "fits": fits}


def posthoc_prospective(F: pl.DataFrame, ctx: dict, counts: dict, tests: dict, cluster_of: dict,
                        internal_ph: dict) -> tuple[dict, pl.DataFrame]:
    """PH4 (variants), PH5 (decompositions of H3 and the WR test; T3 influence), PH7 (power)."""
    V, info = posthoc_variant_forecasts(ctx, counts)
    keys = ["nfl_id", "trait", "k"]
    chk = V.join(F.select(*keys, se0="se_blend_C0", sg="se_games"), on=keys, how="left")
    assert chk["se0"].null_count() == 0
    assert float((chk["se_blend_C0"] - chk["se0"]).abs().max()) == 0.0 and float((chk["se_games"] - chk["sg"]).abs().max()) == 0.0
    wr = POSITIONS["WR"]
    nt3 = [t for t in I.TRAITS if t != "T3"]
    jobs = []
    for v in VARIANTS:
        Fv = V.filter(pl.col("variant") == v)
        jobs += [(f"{v}:{h}", Fv, k, a, b, None) for h, k, a, b in PRIMARY]
    for v in ("calibrated", "matched"):
        Fv = V.filter(pl.col("variant") == v)
        jobs += [(f"{v}:deep_share:H1", Fv, 1, "se_blend_C0_role_deep", "se_blend_C0", wr),
                 (f"{v}:deep_share:H2", Fv, 2, "se_blend_C0_role_deep", "se_blend_C0", wr),
                 (f"{v}:deep_share:H3", Fv, 1, "se_blend_C2_role_deep", "se_blend_C0_role_deep", wr),
                 (f"{v}:deep_share:H4", Fv, 1, "se_blend_C3_role_deep", "se_blend_C2_role_deep", wr),
                 (f"{v}:deep_share:C0_role_deep_vs_C0_role_k1", Fv, 1, "se_blend_C0_role_deep", "se_blend_C0_role", wr)]
    Vc = V.filter(pl.col("variant") == "calibrated")
    jobs += [("prereg:H1_without_T3", F, 1, "se_blend_C0_role", "se_blend_C0", nt3),
             ("prereg:H3_without_T3", F, 1, "se_blend_C2_role", "se_blend_C0_role", nt3),
             ("prereg:WR_C2_role_vs_C0_role_k1_without_T3", F, 1, "se_blend_C2_role", "se_blend_C0_role",
              [t for t in wr if t != "T3"]),
             ("prereg:C2_role_vs_C0_k1", F, 1, "se_blend_C2_role", "se_blend_C0", None),
             ("prereg:C2_role_vs_C0_k2", F, 2, "se_blend_C2_role", "se_blend_C0", None),
             ("prereg:E012_C2_vs_C0_k1", F, 1, "se_blend_C2", "se_blend_C0", None),
             ("prereg:E012_C2_vs_C0_k2", F, 2, "se_blend_C2", "se_blend_C0", None),
             ("prereg:C2_role_vs_E012_C2_k1", F, 1, "se_blend_C2_role", "se_blend_C2", None),
             ("prereg:WR_C2_role_vs_C0_k1", F, 1, "se_blend_C2_role", "se_blend_C0", wr),
             ("calibrated:C2_role_vs_C0_k1", Vc, 1, "se_blend_C2_role", "se_blend_C0", None)]
    fam = {name: run_test(Fx, a, b, SEED + 3000 + j, cluster_of, ts, k) for j, (name, Fx, k, a, b, ts) in enumerate(jobs)}
    tested = [n for n in fam if fam[n]["tested"]]
    for n, q in zip(tested, H.bh([fam[n]["p_one_sided"] for n in tested])):
        fam[n]["q_posthoc_family"] = q
    for n, q in zip(tested, H.bh([fam[n]["team_cluster"]["p_one_sided"] for n in tested])):
        fam[n]["team_cluster"]["q_posthoc_family"] = q
    # per-trait H1 ratio (C0_role / C0 blend error at k = 1) under each reading
    per_trait = {}
    for v, Fx in (("prereg", F), *[(v, V.filter(pl.col("variant") == v)) for v in VARIANTS]):
        d = Fx.filter(pl.col("k") == 1).group_by("trait").agg(
            (pl.col("se_blend_C0_role").mean() / pl.col("se_blend_C0").mean()).alias("r")).sort("trait")
        per_trait[v] = dict(d.iter_rows())
    # PH5: T3 influence in H1 / H3 (k = 1)
    d = F.filter((pl.col("k") == 1) & (pl.col("trait") == "T3")).with_columns(
        excess=pl.col("se_blend_C0_role") - pl.col("se_blend_C0")).sort("excess", descending=True)
    top = d.head(2)["nfl_id"].to_list()
    cnt = counts["WR"]
    orc = I.role_share(cnt, ctx["T3"]["gte"]).select("nfl_id", season_share=I.ROLE)
    fk1 = ctx["T3"]["by_k"][1].select("nfl_id", first_game_share=I.ROLE)
    names = ctx["T3"]["test_players"].select("nfl_id", "display_name")  # real names (user decision, 2026-10-18)
    pl_rows = (d.head(2).select("nfl_id", "excess", "mu_C0", "mu_C0_role", "mu_C2_role", "ybar", "target")
               .join(names, on="nfl_id", how="left").join(fk1, on="nfl_id", how="left")
               .join(orc, on="nfl_id", how="left"))
    t3_rest = d.filter(~pl.col("nfl_id").is_in(top))
    influence = {
        "n_players": d.height, "total_excess_se": float(d["excess"].sum()), "top2_nfl_ids": top,
        "top2_share_of_excess": float(d.head(2)["excess"].sum() / d["excess"].sum()),
        "top2": pl_rows.to_dicts(),
        "t3_ratio_C0_role_vs_C0": float(d["se_blend_C0_role"].mean() / d["se_blend_C0"].mean()),
        "t3_ratio_without_top2": float(t3_rest["se_blend_C0_role"].mean() / t3_rest["se_blend_C0"].mean()),
        "t3_ratio_C2_role_vs_C0": float(d["se_blend_C2_role"].mean() / d["se_blend_C0"].mean()),
        "training_mean_slot_share": float(ctx["T3"]["role_tr"][I.ROLE].mean())}
    fam_prereg = tests["secondary_family"]
    wr_test = fam_prereg["per_position:WR:C2_role_vs_C0_role_k1"]
    # PH7: power of H1-H4 (MDE from the same bootstrap draws as the test) and the best case the internal fit implies
    power = {}
    for i, (h, k, a, b) in enumerate(PRIMARY):
        dd = F.filter(pl.col("k") == k).select("nfl_id", "trait", a, b).filter(pl.col(a).is_finite() & pl.col(b).is_finite())
        bs = H.pooled_bootstrap(dd, [a, b], N_BOOT_V, SEED + i + 1)
        rat = bs[a] / bs[b]
        ci = [float(np.nanquantile(rat, 0.025)), float(np.nanquantile(rat, 0.975))]
        assert np.allclose(ci, tests["primary"][h]["ratio_ci95"]), (h, ci)
        power[h] = {"sd_log_ratio": float(np.std(np.log(rat), ddof=1)), "mde_ratio_one_sided_80pct_power":
                    I.mde_ratio(np.log(rat))}
    best = {}
    for h, k in (("H1", 1), ("H2", 2)):
        dl, e0 = {}, {}
        for t in I.TRAITS:
            dd = F.filter((pl.col("k") == k) & (pl.col("trait") == t)).filter(
                pl.col("se_blend_C0_role").is_finite() & pl.col("se_blend_C0").is_finite())
            if not dd.height:
                continue
            lam_t = ctx[t]["fit"].vc["lambda"]
            rho = max(internal_ph[t]["anova"]["share_of_between_player_variance"]["est"], 0.0)
            mm = dd["m"].to_numpy().astype(float)
            E0, Er = I.expected_blend_se(k, mm, lam_t, 0.0), I.expected_blend_se(k, mm, lam_t, rho)
            dl[t], e0[t] = float(np.mean(E0 - Er)), float(np.mean(E0))
        delta = float(np.mean(list(dl.values())))
        best[h] = {"pooled_expected_reduction": delta, "per_trait_expected_reduction": dl,
                   "ratio_vs_observed_position_average_error": 1 - delta / tests["primary"][h]["pooled_b"],
                   "ratio_vs_model_position_average_error": 1 - delta / float(np.mean(list(e0.values()))),
                   "assumptions": "role learned exactly with the internal (2023-25) ANOVA share of between-player "
                                  "variance (clipped at 0); 2023-24 lambda; matching posterior weight; each test "
                                  "player's own number of target games; normal model"}
    res = {"tests": fam, "family_size": len(tested), "per_trait_h1_ratio_k1": per_trait,
           "variant_details": info, "t3_influence": influence,
           "wr_combine_draft_beyond_role_test": {
               "ratio": wr_test["ratio"], "ratio_ci95": wr_test["ratio_ci95"], "p_one_sided": wr_test["p_one_sided"],
               "q_family": wr_test["q_family"], "team_cluster_p": wr_test["team_cluster"]["p_one_sided"],
               "team_cluster_q_family": wr_test["team_cluster"]["q_family"]},
           "power": {"primary": power, "best_case_from_internal_fit": best}}
    return res, V


def posthoc_summary(ph: dict, internal: dict, tests: dict) -> dict:
    """Cross-trait summaries of the post hoc results, for the headline and the writeup."""
    it = ph["internal"]
    an = {t: it[t]["anova"] for t in I.TRAITS}
    sig = [t for t in I.TRAITS if an[t]["within_class_permutation"]["p_one_sided"] < 0.05]
    rbc = {t: it[t]["attribution_C2"]["role_beyond_c"]["est"] for t in I.TRAITS}
    ridge_role = {t: it[t]["same_cv_vs_dummies_ridge"]["role_worth"]["est"] for t in I.TRAITS}
    ridge_c2 = {t: it[t]["same_cv_vs_dummies_ridge"]["e012_C2"]["est"] for t in I.TRAITS}
    old_c2 = {t: it[t]["same_cv_vs_dummies_ridge"]["against_C0_for_reference"]["e012_C2"] for t in I.TRAITS}
    pq = dict(zip(I.TRAITS, H.bh([an[t]["within_class_permutation"]["p_one_sided"] for t in I.TRAITS])))
    return {
        "anova_share_permutation_p_lt_0.05": sig,
        "anova_bh_q": pq, "anova_share_bh_q_lt_0.05": [t for t in I.TRAITS if pq[t] < 0.05],
        "anova_share": {t: an[t]["share_of_between_player_variance"]["est"] for t in I.TRAITS},
        "anova_worth": {t: an[t]["worth_lambda_role_minus_lambda"]["est"] for t in I.TRAITS},
        "anova_permutation_p": {t: an[t]["within_class_permutation"]["p_one_sided"] for t in I.TRAITS},
        "cv_role_worth_ge_0.1": [t for t in I.TRAITS if internal[t]["role_worth"]["est"] >= 0.1],
        "role_worth_vs_dummies_ridge_same_cv": ridge_role,
        "role_worth_vs_dummies_ridge_ge_0.1": [t for t in I.TRAITS if ridge_role[t] >= 0.1],
        "role_beyond_combine_draft": rbc, "role_beyond_combine_draft_range": [min(rbc.values()), max(rbc.values())],
        "e012_C2_vs_dummies_ridge_same_cv": ridge_c2, "e012_C2_vs_C0_same_cv": old_c2,
        "e012_C2_vs_C0_same_cv_ge_0.4": [t for t in I.TRAITS if old_c2[t] >= 0.4],
        "e012_C2_vs_dummies_ridge_ge_0.4": [t for t in I.TRAITS if ridge_c2[t] >= 0.4],
        "role_share_from_combine_draft_cv_r2": {t: it[t]["role_share_from_combine_draft"]["cv_r2"] for t in I.TRAITS},
    }


TRAIT_ADJUSTMENT = {  # PH11: how each priced trait's play values already condition on alignment (from E012's code)
    "T1": "none (play max speed)", "T12": "none (play max speed)",
    "T2": "route-type centring (route_ran); does not remove slot vs outside differences",
    "T3": "route-type centring (route_ran); does not remove slot vs outside differences",
    "T5": "expectation model conditions on lined_up_position and receiver_alignment (targets_wr.CONTEXT_CAT)",
    "T6": "alignment centring (EDGE / INTERIOR_LINE snaps only)", "T7": "alignment centring (EDGE / INTERIOR_LINE "
    "snaps only)", "T9": "alignment centring (T / G / C)", "T11": "alignment x coverage-assignment centring",
}


# =====================================================================================================
# Figures
# =====================================================================================================
LBL = {"C0": "Position average", "C0_role": "Role-aware average (role from his games)",
       "C2_role": "Role-aware + Combine and draft"}


def _iv(ax, est, lo, hi, y, color, a, b, marker="o", ms=6.5, hollow=False, lw=1.6):
    if est is None or not np.isfinite(est):
        return
    x0 = a if lo is None else max(lo, a)
    x1 = b if hi is None else min(hi, b)
    ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="round", zorder=2)
    if hi is None or hi > b:
        ax.plot(b, y, marker=">", ms=4.5, color=color, zorder=2, clip_on=False)
    if lo is not None and lo < a:
        ax.plot(a, y, marker="<", ms=4.5, color=color, zorder=2, clip_on=False)
    ax.plot(min(max(est, a), b), y, marker=marker, ms=ms, color=SURF if hollow else color, mec=color if hollow else SURF,
            mew=1.4 if hollow else 1.1, zorder=3, ls="none")


def _ypos(traits):
    ypos, y = {}, 0.0
    for i, t in enumerate(traits):
        if i and H.TRAITS[t]["pos"] != H.TRAITS[traits[i - 1]]["pos"]:
            y += 0.55
        ypos[t] = y
        y += 1.2
    return ypos, y


def _band(ax, lo, hi, y, color, a, b, h=0.36):
    """Simulation-calibrated bounds as a light band (each end a one-sided 95% bound; open ends run to the axis)."""
    x0 = a if lo is None else max(lo, a)
    x1 = b if hi is None else min(hi, b)
    ax.fill_between([x0, x1], y - h / 2, y + h / 2, color=color, alpha=0.16, lw=0, zorder=1)


def fig_worth(out: dict, path: Path) -> None:
    tab = {r["trait"]: r for r in out["trust_table"]}
    ph = out.get("post_hoc_review", {}).get("internal", {})
    traits = list(I.TRAITS)
    ypos, ymax = _ypos(traits)
    fig, (ax, axr, axc) = plt.subplots(1, 3, figsize=(12.5, 7.0), sharey=True,
                                       gridspec_kw={"width_ratios": [1.3, 1.1, 1], "wspace": 0.08})
    xmax, dmin, dmax = 14.0, -1.5, 4.0
    col = {"C0": MUTED, "C0_role": RAMP["C0_role"], "C2_role": RAMP["C2_role"]}
    for t in traits:
        r, y = tab[t], ypos[t]
        lam, rw, cd = (r[k]["est"] for k in ("position_average_worth_lambda", "role_aware_average_worth",
                                             "role_aware_plus_combine_draft_worth"))
        v = r["position_average_worth_lambda"]
        _iv(ax, lam, v["lo"], v["hi"], y - 0.22, MUTED, 0, xmax, "o")
        ax.plot([lam, rw, cd], [y - 0.22, y, y + 0.22], color=BASE, lw=0.9, zorder=2)
        for s_, x, dy in (("C0_role", rw, 0.0), ("C2_role", cd, 0.22)):
            ax.plot(min(x, xmax), y + dy, marker=MARK[s_], ms=7, color=col[s_], mec=SURF, mew=1.1, zorder=4, ls="none")
        ax.text(min(max(lam, rw, cd), xmax) + 0.3, y + 0.22, f"{cd:.1f}", ha="left", va="center", fontsize=8,
                color=INK, zorder=5, bbox=dict(boxstyle="square,pad=0.08", fc=SURF, ec="none"))
        for a_, k_, cb_k, c, mk in ((axr, "role_worth", "role_worth_calibrated_one_sided_95", RAMP["C0_role"], "s"),
                                    (axc, "combine_draft_beyond_role", "beyond_C2_role_calibrated_one_sided_95",
                                     RAMP["C2_role"], "D")):
            cb = r[cb_k] or {}
            if cb:
                _band(a_, cb.get("lower"), cb.get("upper"), y, c, dmin, dmax)
            v = r[k_]
            _iv(a_, v["est"], v["lo"], v["hi"], y, c, dmin, dmax, mk)
        if t in ph:  # post hoc (PH6, PH9): role known exactly (ANOVA) and role beyond the Combine + draft
            a = ph[t]["anova"]["worth_lambda_role_minus_lambda"]
            _iv(axr, a["est"], a["lo"], a["hi"], y + 0.3, RAMP["C0_role"], dmin, dmax, "s", ms=6, hollow=True, lw=0.9)
            rb = ph[t]["attribution_C2"]["role_beyond_c"]["est"]
            axr.plot(min(max(rb, dmin), dmax), y - 0.3, marker="X", ms=6.5, color=VIOLET, mec=SURF, mew=0.6, zorder=4,
                     ls="none")
        e12 = r["e012_cge_C2"]["est"]
        axc.plot(min(max(e12, dmin), dmax), y - 0.3, marker="v", ms=6, color=ORANGE, mec=SURF, mew=0.8, zorder=4,
                 ls="none")
    for a_ in (ax, axr, axc):
        a_.grid(axis="y", visible=False)
        a_.tick_params(axis="y", length=0)
        a_.set_ylim(ymax - 0.4, -0.9)
    for a_ in (axr, axc):
        a_.axvline(0, color=BASE, lw=1.0, zorder=1)
        a_.set_xlim(dmin, dmax)
        a_.set_xlabel("Games added")
    ax.set_yticks([ypos[t] for t in traits])
    ax.set_yticklabels([f"{H.TRAITS[t]['pos']}  {H.TRAITS[t]['short']}  ({tab[t]['role']})" for t in traits],
                       fontsize=8.8)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Games of his own tracking the prior is worth\n(his games get half the weight after this many)")
    ax.set_title("Prior worth, in games", fontsize=10.5)
    axr.set_title("Role adds", fontsize=10.5)
    axc.set_title("Combine and draft add, beyond role", fontsize=10.5)
    h = [Line2D([], [], color=MUTED, marker="o", lw=1.6, ms=6.5, mec=SURF, label="Position average (95% interval)"),
         Line2D([], [], color=RAMP["C0_role"], marker="s", ls="none", ms=7, mec=SURF,
                label="+ role (share measured from his games)"),
         Line2D([], [], color=RAMP["C2_role"], marker="D", ls="none", ms=7, mec=SURF,
                label="+ role, Combine and draft (value printed)")]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0, 1.05), fontsize=8.3)
    band = matplotlib.patches.Patch(color=INK2, alpha=0.2, lw=0, label="calibrated bounds (each end one-sided 95%)")
    h2 = [Line2D([], [], color=RAMP["C0_role"], marker="s", lw=1.6, ms=6, mec=SURF,
                 label="learned out of fold (preregistered)"), band,
          Line2D([], [], color=RAMP["C0_role"], marker="s", lw=0.9, ms=6, mfc=SURF, mec=RAMP["C0_role"], mew=1.4,
                 label="known exactly: ANOVA λ_role − λ (post hoc)"),
          Line2D([], [], color=VIOLET, marker="X", ls="none", ms=6.5, mec=SURF,
                 label="beyond Combine and draft (post hoc)")]
    axr.legend(handles=h2, loc="lower left", bbox_to_anchor=(0, 1.05), fontsize=8.3)
    h3 = [Line2D([], [], color=RAMP["C2_role"], marker="D", lw=1.6, ms=6, mec=SURF, label="estimate (preregistered)"),
          band, Line2D([], [], color=ORANGE, marker="v", ls="none", ms=6, mec=SURF,
                       label="E012: Combine and draft, role ignored")]
    axc.legend(handles=h3, loc="lower left", bbox_to_anchor=(0, 1.05), fontsize=8.3)
    fig.suptitle("What a rookie's prior is worth in games: position average, role (measured from his own games), "
                 "Combine and draft", x=0.01, ha="left", y=0.995, fontsize=12, weight="bold")
    fig.text(0.01, 0.952, "Internal fit, 2023-25 classes. Lines: E012's preregistered player bootstrap, nominal 95% "
             "but biased low (several intervals exclude their own estimate); arrowheads = beyond the axis.\nBands: "
             "simulation-calibrated bounds, each end a one-sided 95% bound. Hollow and X markers: post hoc.",
             fontsize=8.5, color=INK2, va="top")
    fig.text(0.01, 0.01, "Role = his share of snaps in the slot (WR), on the edge (DL), at tackle (OL) or at corner "
             "(DB; slot corners count as corners), over his qualifying games. A prior worth N games keeps weight "
             "N / (k + N) after k qualifying games.", fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.19, right=0.985, top=0.735, bottom=0.115)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_prospective(out: dict, path: Path) -> None:
    p = out["prospective"]
    prim, fam = p["tests"]["primary"], p["tests"]["secondary_family"]
    pt = out.get("post_hoc_review", {}).get("prospective", {}).get("tests", {})
    fp = lambda v: "< 0.0001" if v < 1e-4 else (f"{v:.4f}" if v < 0.01 else f"{v:.2f}")  # noqa: E731
    fig = plt.figure(figsize=(12, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.12)
    ax, axr = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    rat = p["descriptive"]["post_hoc_ratio_vs_position_average_blend"]
    series = [("C2", "Combine and draft, role ignored (E012 prior)", ORANGE, "v", (0, (4, 2))),
              ("C0_role", "Role-aware prior", RAMP["C0_role"], "s", "-"),
              ("C2_role", "Role-aware + Combine and draft", RAMP["C2_role"], "D", "-"),
              ("C3_role", "Role-aware + Combine and draft + tracking", RAMP["C3_role"], "^", (0, (1, 1.5)))]
    dx = dict(zip([s_[0] for s_ in series], (-0.12, -0.04, 0.04, 0.12)))
    for key, lab, col, mk, ls in series:
        xs = np.array(KS, float) + dx[key]
        e = [rat[str(k)][key] for k in KS]
        ax.vlines(xs, [v["lo"] for v in e], [v["hi"] for v in e], color=col, lw=1.2)
        ax.plot(xs, [v["ratio"] for v in e], marker=mk, ls=ls, color=col, lw=1.6, ms=6, mec=SURF, mew=1.0, label=lab)
    ax.axhline(1, color=BASE, lw=1.1, zorder=1)
    ax.text(KS[-1] + 0.3, 1.0, "position-average\nblend", fontsize=8, color=INK2, va="center")
    ax.set_xticks(KS)
    ax.set_xlim(0.6, KS[-1] + 0.75)
    ax.set_xlabel("Qualifying games observed (k); role share from those games")
    ax.set_ylabel("Forecast error relative to the position-average blend\n(below 1 = better)")
    ax.set_title("Error vs the position-average blend (post hoc, descriptive)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    rows = [(prim["H1"], "H1  role-aware vs position average, 1 game", "pre"),
            (prim["H2"], "H2  role-aware vs position average, 2 games", "pre"),
            (prim["H3"], "H3  + Combine and draft vs role-aware, 1 game", "pre"),
            (prim["H4"], "H4  + Combine tracking vs + Combine and draft, 1 game", "pre"),
            (fam["oracle_role:H1"], "secondary  H1 with role from his whole season (upper bound)", "sec")]
    if pt:
        rows += [(pt["calibrated:H1"], "post hoc  H1, test role share calibrated on training players", "ph"),
                 (pt["calibrated:H3"], "post hoc  H3, test role share calibrated on training players", "ph"),
                 (pt["prereg:H3_without_T3"], "post hoc  H3 without WR hitch braking", "ph"),
                 (pt["prereg:C2_role_vs_C0_k1"], "post hoc  role-aware + Combine and draft vs position average, "
                  "1 game", "ph")]
    qkey = {"pre": ("q_bh", "BH q"), "sec": ("q_family", "family q"), "ph": ("q_posthoc_family", "post hoc q")}
    for i, (r, lab, kind) in enumerate(rows):
        y = 1.25 * (len(rows) - 1 - i)
        lo, hi = r["ratio_ci95"]
        c = RAMP["C2_role"] if kind == "pre" else MUTED
        axr.plot([lo, hi], [y, y], color=c, lw=1.8 if kind == "pre" else 1.3, solid_capstyle="round")
        axr.plot(r["ratio"], y, "o", ms=7, color=SURF if kind == "ph" else c, mec=c if kind == "ph" else SURF,
                 mew=1.4 if kind == "ph" else 1.2, zorder=3)
        axr.text(0.01, y + 0.2, lab, transform=axr.get_yaxis_transform(), fontsize=8.5, color=INK, va="bottom")
        qk, ql = qkey[kind]
        axr.text(0.01, y - 0.2, f"ratio {r['ratio']:.3f} [{lo:.3f}, {hi:.3f}] · p {fp(r['p_one_sided'])} · "
                 f"{ql} {fp(r[qk])} · team-cluster p {fp(r['team_cluster']['p_one_sided'])}",
                 transform=axr.get_yaxis_transform(), fontsize=8, color=INK2, va="top")
    lo_all = min(r["ratio_ci95"][0] for r, _, _ in rows)
    hi_all = max(r["ratio_ci95"][1] for r, _, _ in rows)
    axr.set_xlim(min(0.8, lo_all - 0.02), max(1.1, hi_all + 0.02))
    axr.axvline(1, color=BASE, lw=1)
    axr.set_yticks([])
    axr.set_ylim(-0.8, 1.25 * (len(rows) - 1) + 0.75)
    axr.grid(axis="y", visible=False)
    axr.spines["left"].set_visible(False)
    axr.set_xlabel("Error ratio, first / second (below 1 = first better)")
    axr.set_title("Preregistered tests (player sign flips, BH over H1-H4); post hoc rows hollow", fontsize=10)
    fig.suptitle("2025 rookies forecast from a 2023-24 fit, with alignment share from his first qualifying games "
                 "(pseudo-prospective)", x=0.01, ha="left", y=0.99, fontsize=12, weight="bold")
    fig.text(0.01, 0.01, "Error pooled over the 9 traits with equal weight, each standardized by its 2023-24 "
             "between-player variance; paired 95% player-bootstrap intervals. Blend weight = k / (k + λ / (1 - R²)) "
             "(E012 amendment A1).", fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.11)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def draw_figures(out: dict) -> list[str]:
    figs = {"worth": OUT_FIG / "E013_role_worth_in_games.png", "prospective": OUT_FIG / "E013_prospective.png"}
    fig_worth(out, figs["worth"])
    fig_prospective(out, figs["prospective"])
    return [str(p.relative_to(ROOT)) if not QUICK else str(p) for p in figs.values()]


# =====================================================================================================
# Headline and summary (built from the result keys; wording rules fixed after results: PH3, then PH13)
# =====================================================================================================

WORDS = {"T1": "WR top speed", "T2": "WR turn retention", "T3": "WR hitch braking", "T5": "WR yards per route over "
         "expected", "T6": "DL rush distance", "T7": "DL get-off", "T9": "OL set acceleration", "T11": "DB retreat "
         "speed", "T12": "DB top speed"}


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def write_up(out: dict) -> dict:
    """Headline and summary from the result keys. Wording rules: PH3, superseded after the review by PH13 (the
    preregistered verdicts first, then bounded post hoc statements)."""
    prim, it = out["prospective"]["tests"]["primary"], out["internal"]
    fam = out["prospective"]["tests"]["secondary_family"]
    ph = out["post_hoc_review"]
    pt, pin, sm, pro = ph["prospective"]["tests"], ph["internal"], ph["summary"], ph["prospective"]
    fq = lambda v: f"{v:.2f}" if v >= 0.01 else (f"{v:.3f}" if v >= 0.001 else f"{v:.4f}")  # noqa: E731
    ci = lambda r: f"{r['ratio']:.3f} [{r['ratio_ci95'][0]:.3f}, {r['ratio_ci95'][1]:.3f}]"  # noqa: E731
    h1, h2, h3, h4 = (prim[h] for h in ("H1", "H2", "H3", "H4"))
    orc = fam["oracle_role:H1"]
    s1 = ("Preregistered: knowing a 2025 rookie's alignment share (slot, edge, tackle or corner) from his first "
          "qualifying games " + ("improved" if h1["pass"] else "did not improve") + " his in-season forecast "
          f"(pseudo-prospective: fit on the 2023-24 seasons). Error relative to the position-average blend was {ci(h1)} "
          f"after one game (H1, BH q {fq(h1['q_bh'])}) and {ci(h2)} after two (H2, q {fq(h2['q_bh'])}); role from his "
          f"whole season gave {orc['ratio']:.3f} (secondary upper bound). Beyond role, the Combine and draft gave "
          f"{ci(h3)} (H3, p {fq(h3['p_one_sided'])}, q {fq(h3['q_bh'])}, "
          + ("supported" if h3["pass"] else "not significant") + f") and Combine tracking {ci(h4)} (H4"
          + (", supported" if h4["pass"] else "") + ").")
    c1, c3, m1, m3 = pt["calibrated:H1"], pt["calibrated:H3"], pt["matched:H1"], pt["matched:H3"]
    inf = pro["t3_influence"]
    pw = pro["power"]
    s2 = (" Post hoc (review): the one-game comparisons are biased against role, because the role coefficient was "
          "trained on whole-history shares but fed a one-game share; with the test share calibrated on training "
          f"players, H1 is {ci(c1)} and H3 {ci(c3)} (training shares at the test's precision: {m1['ratio']:.3f} and "
          f"{m3['ratio']:.3f}). H3's {1 - h3['ratio']:.1%} cut is mostly {WORDS['T3']}, where two rookies carry "
          f"{inf['top2_share_of_excess']:.0%} of the role-aware blend's excess error (H3 without that trait "
          f"{ci(pt['prereg:H3_without_T3'])}); against the position-average blend, the role-aware + Combine and draft "
          f"prior gave {ci(pt['prereg:C2_role_vs_C0_k1'])} after one game, E012's prior "
          f"{pt['prereg:E012_C2_vs_C0_k1']['ratio']:.3f}. The test could detect an error ratio of about "
          f"{pw['primary']['H1']['mde_ratio_one_sided_80pct_power']:.3f} or lower after one game; role learned exactly "
          f"at the internal effect sizes would give about "
          f"{pw['best_case_from_internal_fit']['H1']['ratio_vs_observed_position_average_error']:.3f}.")
    real = sorted(sm["anova_share_bh_q_lt_0.05"], key=lambda t: -sm["anova_share"][t])
    rest = [t for t in I.TRAITS if t not in real]
    cv01 = sorted(sm["cv_role_worth_ge_0.1"], key=lambda t: -it[t]["role_worth"]["est"])
    s3 = (" In the 2023-25 fit, role is real between players for some traits: by the preregistered ANOVA, alignment "
          "share explains " + _and([f"{sm['anova_share'][t]:.0%}" + (" of the between-player variance" if i == 0 else "")
                                    + f" for {WORDS[t]}" for i, t in enumerate(real)])
          + f" (within-class permutation, BH q {fq(max(sm['anova_bh_q'][t] for t in real))} or less over 9 traits), "
          "worth " + _and([f"{sm['anova_worth'][t]:.2f}" for t in real]) + " games if known exactly; for the other "
          f"{len(rest)} it explains at most {max(sm['anova_share'][t] for t in rest):.0%}. Learned out of fold (the "
          "preregistered CV estimator, which also pays for estimating class means out of fold) role is worth at least "
          f"0.1 games for {len(cv01)} of 9 traits ("
          + ", ".join(f"{WORDS[t]} {it[t]['role_worth']['est']:.2f}" for t in cv01)
          + "; against a ridge on the class dummies under the same CV, "
          + _and([f"{sm['role_worth_vs_dummies_ridge_same_cv'][t]:.2f}" for t in cv01]) + ").")
    t6 = pin["T6"]["attribution_C2"]
    rd = pin["T6"]["roster_designation"]
    pe = rd["pure_edge_olb_vs_de"]
    lo, hi = sm["role_beyond_combine_draft_range"]
    s4 = (f" Once the Combine and draft are known, alignment share adds {lo:+.2f} to {hi:+.2f} games for every priced "
          f"trait. E012's {t6['c_alone']['est']:.2f}-game {WORDS['T6']} value splits into "
          f"{t6['c_beyond_role']['est']:.2f} beyond edge share, {t6['role_beyond_c']['est']:.2f} edge share beyond the "
          f"Combine and draft, and {t6['shared']['est']:.2f} shared: edge share is "
          f"{sm['role_share_from_combine_draft_cv_r2']['T6']:.0%} predictable from the Combine and draft (CV R2). "
          f"Among pure edge players, roster OLBs out-rush roster DEs by {pe['mean_olb'] - pe['mean_de']:.2f} yd "
          f"(Welch p {fq(pe['welch_p_two_sided'])}; descriptive, in sample; the roster label may postdate the draft): "
          "for DL the role signal is in the roster designation (OLB vs DE), not in edge alignment.")
    tt = {r["trait"]: r for r in out["trust_table"]}
    hw = {t: it[t]["half_weight"]["C2_role"]["est"] for t in it}
    hw12 = {t: it[t]["half_weight"]["C2"]["est"] for t in it}
    ap = {t: tt[t]["half_weight_rookie_appearances"]["C2_role"] for t in it}
    s5 = (f" Under the role-aware + Combine and draft prior his own games still take over after "
          f"{min(hw.values()):.1f}-{max(hw.values()):.1f} qualifying games (about {min(ap.values()):.1f}-"
          f"{max(ap.values()):.1f} rookie appearances; E012's prior: {min(hw12.values()):.1f}-{max(hw12.values()):.1f} "
          "qualifying games).")
    headline = s1 + s2 + s3 + s4 + s5
    cb = {t: it[t]["calibrated_bounds"]["role"] for t in it}
    up = {t: cb[t]["calibrated_one_sided_95"]["upper"] for t in it}
    pop = {t: cb[t]["calibrated_one_sided_95_population_signal"] for t in it}
    fin_b = [it[t]["calibrated_bounds"]["beyond_C2"]["calibrated_one_sided_95"]["upper"] for t in it]
    fin_b = [v for v in fin_b if v is not None]
    cov = out["simulation"]["b1_coverage"]
    old, new = sm["e012_C2_vs_C0_same_cv_ge_0.4"], sm["e012_C2_vs_dummies_ridge_ge_0.4"]
    summary = (headline + " Bounds: the simulation-calibrated one-sided 95% bounds for the CV role worth have upper "
               f"ends {min(v for v in up.values() if v is not None):.2f}-{max(v for v in up.values() if v is not None):.2f} "
               "games; with a population-scale role signal they are "
               + _and([f"{WORDS[t]} ({pop[t]['lower']:.2f}, {pop[t]['upper']:.2f})" for t in real])
               + f"; Combine and draft beyond role is below {max(fin_b):.1f} games for {len(fin_b)} of 9 traits. The "
               "CV estimator is 0 in at least half of simulated datasets up to a population role worth of "
               + _and([f"{pin[t]['cv_estimator']['largest_population_worth_estimated_zero_half_the_time']:.2f}"
                       for t in real]) + f" games ({_and([WORDS[t] for t in real])}). The preregistered "
               "player bootstrap under-covers (Combine and draft beyond role: "
               + _and([f"{cov[t]['beyond_C2_role']['coverage']:.0%}" for t in cov]) + " at nominal 95%). Same-CV "
               "sensitivities now use a ridge on the class dummies as reference (the position average's unshrunk "
               "class means mixed ridge shrinkage into the value): E012's Combine + draft forecast reaches 0.4 games "
               f"for {len(new)} traits ({_and([WORDS[t] for t in new])}), not {len(old)}. Caveats: role share is "
               "measured on the trait's own games; corner share counts slot corners, so nickel safeties read as "
               "part-corners; DL edge share is effectively EDGE / (EDGE + interior) (almost no OLB-aligned rows); T2 "
               "and T3 are route-centred (not alignment-centred), T5's expectation model conditions on alignment, and "
               "T6, T7, T9 and T11 are alignment-centred; the 2025 test reuses E012's rookies; every per-trait, "
               "per-position and post hoc result is descriptive (BH over the secondary family of "
               f"{out['prospective']['tests']['secondary_family_size']} and the post hoc family of "
               f"{pro['family_size']}).")
    return {"headline": headline, "summary": summary}


def rewrite() -> None:
    """Deterministic re-assembly from E013.json: headline, summary, post hoc record and figures (no recomputation)."""
    out = json.loads((OUT_RES / "E013.json").read_text())
    out["post_hoc"] = POST_HOC
    wu = clean(write_up(out))
    out = {**{k: out[k] for k in ("experiment", "title")}, **wu, **{k: v for k, v in out.items() if k not in wu}}
    out["figures"] = draw_figures(out)
    out["rewritten_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (OUT_RES / "E013.json").write_text(json.dumps(clean(out), indent=1))
    print(out["headline"])


# =====================================================================================================
# Main
# =====================================================================================================

def main() -> None:
    out = {"experiment": "E013", "title": PREREG["experiment"], "quick": QUICK, "reuse_stored_draws": REUSE,
           "run_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "prereg_file": str(PREREG_PATH.relative_to(ROOT)),
           "prereg_sha256": hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest(),
           "implementation_choices": {"frozen_at_utc": CHOICES_FROZEN_AT, "freeze_file_sha256": CHOICES_SHA256,
                                      "check": check_frozen_choices(), "choices": IMPLEMENTATION_CHOICES}}
    players, players_p, games_i, games_p = load_inputs()
    counts, teams_df, achk, vocab = alignment_inputs(players)
    teams = dict(teams_df.iter_rows())
    out["data_checks"] = {"alignment": achk}
    log(f"inputs: {games_i.height:,} internal and {games_p.height:,} prospective game values; "
        f"{achk['player_play_reg_rows']:,} REG player_play rows")

    # ---- 4. prospective (fast; first, so a crash later still leaves it in the checkpoint)
    F, fit_summ, lchk, ctx = prospective(players_p, games_p, counts)
    cluster_of = {int(i): teams.get(int(i), f"none_{int(i)}") for i in F["nfl_id"].unique().to_list()}
    out["data_checks"]["leakage"] = lchk
    out["data_checks"]["e012_forecast_reproduction"] = e012_forecast_check(F)
    out["data_checks"]["test_rookies_without_team"] = sum(1 for v in cluster_of.values() if str(v).startswith("none_"))
    log("prospective forecasts built; E012 reproduction " + json.dumps(clean(out["data_checks"]["e012_forecast_reproduction"])))
    tests = prospective_tests(F, cluster_of)
    out["prospective"] = {"design_label": "pseudo-prospective: fit on the 2023-24 seasons only and applied to the 2025 "
                          "rookies' 2025 season, with role share from his first k qualifying games; the same 2025 "
                          "rookies as E012's tests, so these are new hypotheses on already-used data",
                          "training_fit": fit_summ, "tests": tests, "descriptive": descriptive_prospective(F)}
    for h, r in tests["primary"].items():
        log(f"{h}: {r['pooled_a']:.3f} vs {r['pooled_b']:.3f} ratio {r['ratio']:.3f} [{r['ratio_ci95'][0]:.3f}, "
            f"{r['ratio_ci95'][1]:.3f}] p {r['p_one_sided']:.4f} q {r['q_bh']:.4f} team-cluster p "
            f"{r['team_cluster']['p_one_sided']:.4f}, n {r['n_players']}")
    if not QUICK:
        F.write_parquet(DATA / "e013_forecasts_2025.parquet")

    # ---- 2. internal
    e12_boot = None if QUICK else pl.read_parquet(DATA / "e012_bootstrap.parquet")
    internal, arts, boots = {}, {}, []
    for t in I.TRAITS:
        internal[t], arts[t] = internal_trait(t, players, games_i, counts, e12_boot)
        boots.append(arts[t]["boot"].with_columns(trait=pl.lit(t)))
        r = internal[t]
        log(f"{t} {r['position']} {r['name']}: n {r['n_players']}, lambda {r['lambda']['est']:.2f}, lambda_role "
            f"{r['lambda_role_anova']['est']:.2f}, role worth {r['role_worth']['est']:.2f} "
            f"[{r['role_worth']['lo']}, {r['role_worth']['hi']}], beyond role C2 {r['beyond_role']['C2_role']['est']:+.2f} "
            f"C3 {r['beyond_role']['C3_role']['est']:+.2f}; E012 C2 {r['cge']['C2']['est']:.2f}; reproduction "
            + json.dumps(clean({k: v for k, v in r["e012_reproduction"].items() if k != "values_e013_e012"})))
    del e12_boot
    if not QUICK:
        pl.concat(boots, how="diagonal_relaxed").write_parquet(DATA / "e013_bootstrap.parquet")
        pl.concat([arts[t]["players"].select("nfl_id", "position", I.ROLE, "role_den", "n_games_role")
                   .with_columns(trait=pl.lit(t)) for t in I.TRAITS]).write_parquet(DATA / "e013_role_shares.parquet")

    # ---- 5a. diagnostics
    out["diagnostics"] = {t: diagnostics(t, arts[t], counts, games_p, players_p) for t in I.TRAITS}
    log("diagnostics done")

    # ---- 3. simulation belts and coverage
    belts = []
    for fam in FAMILIES:
        for t in I.TRAITS:
            if REUSE:
                b = pl.read_parquet(DATA / "e013_sim_belt.parquet").filter(
                    (pl.col("family") == fam) & (pl.col("trait") == t))
            else:
                b = ckpt(f"sim_{fam}_{t}", sim_family, fam, t, arts[t], internal[t])
            belts.append(b)
            obs = internal[t]["role_worth"]["est"] if fam == "role" else \
                internal[t]["beyond_role"][FAMILIES[fam][2]]["est"]
            internal[t].setdefault("calibrated_bounds", {})[fam] = belt_bounds(b, fam, internal[t]["lambda"]["est"], obs)
            cb = internal[t]["calibrated_bounds"][fam]["calibrated_one_sided_95"]
            log(f"sim {fam} {t}: observed {obs:+.2f}, calibrated [{cb['lower']}, {cb['upper']}]")
    if not QUICK:
        pl.concat(belts, how="diagonal_relaxed").write_parquet(DATA / "e013_sim_belt.parquet")
    covs, cov_sum = [], {}
    for t in COV_DESIGNS:
        if REUSE:
            c = pl.read_parquet(DATA / "e013_sim_coverage.parquet").filter(pl.col("trait") == t)
        else:
            c = ckpt(f"cov_{t}", coverage_runs, t, arts[t], internal[t], internal[t]["calibrated_bounds"]["beyond_C2"])
        covs.append(c)
        cov_sum[t] = coverage_summary(c, internal[t]["lambda"]["est"])
        log(f"coverage {t}: " + json.dumps(clean(cov_sum[t])))
    if not QUICK:
        pl.concat(covs, how="diagonal_relaxed").write_parquet(DATA / "e013_sim_coverage.parquet")
    out["internal"] = internal
    t6 = internal["T6"]
    out["deliverables"] = {"dl_rush_distance_combine_value_once_role_known": {
        "e012_cge_C2": t6["cge"]["C2"], "role_worth": t6["role_worth"], "beyond_role_C2": t6["beyond_role"]["C2_role"],
        "change": t6["combine_change_once_role_known"],
        "calibrated_beyond_C2": t6["calibrated_bounds"]["beyond_C2"]["calibrated_one_sided_95"],
        "e012_roster_position_reference": out["diagnostics"]["T6"]["e012_roster_position_reference"]}}
    out["simulation"] = {"b1_coverage": cov_sum,
                         "design": f"role grid {list(ROLE_GRID)}, beyond-role grid {list(SIM_GRID)}, {SIM_REPS} datasets "
                                   "per point; Combine signal on 3 random columns of the set's Combine design, "
                                   "residualized on role; games iid N(theta, lambda_hat)"}

    # ---- 6. post hoc: response to the E013 review (PH4-PH13; no preregistered number changes)
    ph_path = DATA / PH_BOOT
    if REUSE and not QUICK and ph_path.exists():
        phb = pl.read_parquet(ph_path)
    else:
        phb = pl.concat([ckpt(f"phboot_{t}", lambda t=t: I.bootstrap_dummies_ridge(
            arts[t]["td"], N_BOOT, E012_SEED + tidx(t), N_JOBS, BOOT_REPEATS).with_columns(trait=pl.lit(t)))
            for t in I.TRAITS])
        if not QUICK:
            phb.write_parquet(ph_path)
    ph_int = {}
    for t in I.TRAITS:
        ph_int[t] = posthoc_internal_trait(t, arts[t], internal[t], phb.filter(pl.col("trait") == t).drop("trait"))
        a = ph_int[t]["anova"]
        log(f"post hoc {t}: ANOVA role share {a['share_of_between_player_variance']['est']:.3f} "
            f"(perm p {a['within_class_permutation']['p_one_sided']:.4f}), worth {a['worth_lambda_role_minus_lambda']['est']:.2f}; "
            f"role vs dummies ridge {ph_int[t]['same_cv_vs_dummies_ridge']['role_worth']['est']:.2f}; "
            f"role beyond Combine+draft {ph_int[t]['attribution_C2']['role_beyond_c']['est']:+.2f}")
    ph_pro, V = posthoc_prospective(F, ctx, counts, tests, cluster_of, ph_int)
    if not QUICK:
        V.write_parquet(DATA / PH_FORECASTS)
    for n in ("calibrated:H1", "calibrated:H3", "matched:H1", "matched:H3", "prereg:H3_without_T3", "prereg:C2_role_vs_C0_k1"):
        r = ph_pro["tests"][n]
        log(f"post hoc {n}: ratio {r['ratio']:.3f} [{r['ratio_ci95'][0]:.3f}, {r['ratio_ci95'][1]:.3f}] p {r['p_one_sided']:.4f}")
    out["post_hoc_review"] = {
        "date": "2026-10-19",
        "scope": "Analyses added after the E013 review (two reviewers, both 'weakened'), decided after the results were "
                 "seen (POST_HOC PH4-PH13). The preregistered tests, estimands and verdicts are unchanged.",
        "internal": ph_int, "prospective": ph_pro, "summary": posthoc_summary({"internal": ph_int}, internal, tests),
        "alignment_vocabulary_reg_rows": vocab, "trait_alignment_adjustment": TRAIT_ADJUSTMENT,
        "note_on_frozen_choice_D2": "D2 grouped T2 / T3 (route-type centring) with the alignment-centred traits and "
                                    "listed T5 as not alignment-centred. Route centring does not remove slot vs outside "
                                    "differences, and T5's expectation model conditions on alignment; "
                                    "trait_alignment_adjustment gives each trait's actual adjustment."}
    out["deliverables"]["dl_rush_distance_combine_value_once_role_known"]["post_hoc_both_orders"] = \
        ph_int["T6"]["attribution_C2"]
    out["deliverables"]["dl_rush_distance_combine_value_once_role_known"]["post_hoc_roster_designation"] = \
        ph_int["T6"]["roster_designation"]

    # ---- 5b. tables, figures, json
    out["trust_table"] = trust_table(internal, fit_summ, ph_int)
    out["post_hoc"] = POST_HOC
    out = clean(out)
    out = {**{k: out[k] for k in ("experiment", "title")}, **clean(write_up(out)), **out}
    out["figures"] = draw_figures(out)
    out["runtime_s"] = time.time() - T0
    if REUSE:
        out["runtime_note"] = ("runtime_s is this --reuse run (stored bootstrap and simulation draws read from "
                               "data/processed/e013_*.parquet); the full run's log is experiments/results/E013_audit/full.log")
    (OUT_RES / "E013.json").write_text(json.dumps(clean(out), indent=1))
    log(f"done: {OUT_RES / 'E013.json'}")


if __name__ == "__main__":
    if FIGURES_ONLY:
        draw_figures(json.loads((OUT_RES / "E013.json").read_text()))
    elif REWRITE:
        rewrite()
    else:
        SCRATCH.mkdir(parents=True, exist_ok=True)
        with threadpool_limits(N_JOBS):
            main()
