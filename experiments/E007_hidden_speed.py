"""E007 — Hidden speed (WR): tracked flying-20 speed beyond the 40 time.

Feature: c40_v_20_40 = average speed (yd/s) from the official 20-yd line to the 40-yd line of the tracked Combine
40, mean over QC-passing attempts (bdb27.speed, E007 amendment). Primary targets (Combine WRs, expected sign +):
  P1 game_top_speed  q95 of REG play max speed (bdb27.features.game_top_speed, >= 50 plays)
  P2 cushion_oe      defender cushion at the snap over a cross-fitted context expectation, EB-shrunk (>= 50 routes)
  P3 yprr_oe         receiving yards per route run over a cross-fitted context expectation, EB-shrunk (>= 100 routes)
Preregistration (binding, incl. amendments): experiments/results/E007_prereg.json. Protocol: PLAN.md §4 rev 2.

Reproduces everything top to bottom (about 25-30 min with n_jobs=3; peak memory ~2 GB). Sections: Combine
feature, targets, primary tests, prespecified robustness, prelisted exploratory, a labelled post hoc block (8b),
a second labelled post hoc block added after the two-skeptic review (8c; sensitivity analyses of the
preregistered results, none of them claims, none in any BH family), BH, leaderboard, figures. Run:
    .venv/bin/python experiments/E007_hidden_speed.py
Outputs: experiments/results/E007.json, experiments/results/E007_leaderboard.json, experiments/figures/E007_*.png,
data/processed/e007_{forty_attempts,routes,players}.parquet.
"""

import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import f as f_dist
from scipy.stats import false_discovery_control, norm, spearmanr
from scipy.stats import t as t_dist
from sklearn.metrics import roc_auc_score

from bdb27 import features, io, linkage, reliability, shrinkage, speed
from bdb27 import targets_wr as T

HERE = Path(__file__).parent
RES, FIG = HERE / "results", HERE / "figures"
FIG.mkdir(exist_ok=True)
PREREG = json.loads((RES / "E007_prereg.json").read_text())

FEATURE = "c40_v_20_40"
PRIMARY = {"P1": "game_top_speed", "P2": "cushion_oe", "P3": "yprr_oe"}
PRIMARY_LABEL = {"P1": "game top speed (q95 play max, yd/s)", "P2": "cushion over expected (yd)",
                 "P3": "yards per route over expected"}
FULL = dict(n_repeats=50, n_perm=2000, n_perm_fl=10000, n_jobs=3)  # prereg harness settings
EXPL = dict(n_repeats=20, n_perm=500, n_perm_fl=2000, n_jobs=3, seed=0)  # brief: prelisted exploratory
SEEDS = (0, 1, 2)
MIN_ROUTES = {"P2": 50, "P3": 100}
LAST_SEASON = 2025
MPH = 3600 / 1760  # mph per yd/s
P1_HALF_MIN_PLAYS = 25
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"

# Fixed before any Combine-feature x target association was computed (see the brief and PLAN.md §4 rev 2).
IMPLEMENTATION_CHOICES = [
    {"item": "study receivers / route table",
     "choice": "Combine position WR and TE; player_play rows with route_ran non-null in games with season_type REG. "
               "All 43,162 such rows have play_nullified_by_penalty = 'N' (asserted), so no penalty filter. Routes on "
               "sacks, scrambles and throwaways (pass_result S/R/IN) are routes run and are kept.",
     "why": "Prereg/brief population; 'routes run' conventionally counts every dropback route."},
    {"item": "yards to the end zone",
     "choice": "100 - yardline_number when yardline_side == possession_team, yardline_number otherwise; "
               "yardline_side is null only at the 50 (571 rows, all yardline_number 50) -> 50.",
     "why": "Direct reading of the field-position fields; checked on the data before any modelling."},
    {"item": "defense team and score differential",
     "choice": "defense = the games.csv team (home/visitor) that is not possession_team; score_diff = possession "
               "team's pre-snap score minus the opponent's; defense team-season = '<team>_<season>' (96 levels).",
     "why": "Prereg context definition."},
    {"item": "context encoding",
     "choice": "Categorical (native HGB categories): lined_up_position, receiver_alignment, offense_formation, "
               "in_motion_at_ball_snap, team_coverage_man_zone, season, defense_team_season. Numeric: down, yards_to_go, "
               "yards_to_end_zone, quarter (OT = 5), score_diff. Nulls are HGB's missing value. Category codes = rank "
               "of the label among sorted labels (outcome-free).",
     "why": "Labels with no natural order are categorical; down/distance/field/quarter/score are ordered quantities."},
    {"item": "expectation model",
     "choice": "HistGradientBoostingRegressor(categorical_features=<7 categorical columns>, random_state=0), all other "
               "hyperparameters default (incl. early_stopping='auto', which is active here because each training fold "
               "has > 10,000 rows: 10% of the training fold is held out at random, fixed by random_state). "
               "GroupKFold(n_splits=5) by game_id with sklearn defaults (no shuffle). P2 model trained on all WR+TE "
               "routes with non-null cushion; P3 model on all WR+TE routes with rec_yards null -> 0. "
               "residual = observed - out-of-fold prediction.",
     "why": "Brief: default hyperparameters, fixed random_state, cross-fitted by game on all study WR+TE routes."},
    {"item": "binary exploratory expectations",
     "choice": "Deep-route share and targets per route: HistGradientBoostingClassifier with the same context, "
               "categories, folds and random_state; expectation = out-of-fold P(event); EB with shrinkage.eb_rate "
               "(cluster='game_id', min_n=100 routes, as P3).",
     "why": "eb_rate needs per-play probabilities in (0, 1); a squared-error regressor can leave that range."},
    {"item": "P2/P3 player targets",
     "choice": "shrinkage.eb_mean(WR rows only, cluster='game_id', min_n=50 (P2) / 100 (P3)); target = `shrunk`. "
               "The prior is fitted on the kept WRs, including WRs without a tracked 40.",
     "why": "Prereg + amendment (game-clustered sampling variance); 'prior estimated on eligible players'."},
    {"item": "P1 split-half reliability",
     "choice": "reliability.split_half_estimator on Combine WRs with >= 50 REG plays: odd vs even games (dense rank), "
               "estimator = q95 of play max speed (plays <= 12 yd/s, polars 'nearest' quantile as in "
               "features.game_top_speed), NaN if a half has < 25 plays; Pearson, Spearman-Brown.",
     "why": "Same estimator as the target; 25 = half the target's 50-play minimum."},
    {"item": "P2/P3 reliability and diagnostics",
     "choice": "split-half = shrinkage.check_prior's split-half (odd/even games, kept WRs, Spearman-Brown); "
               "half-difference diagnostic (targets_wr.half_difference): the same EB estimator refit on odd and even "
               "games, mean z2 = (raw0 - raw1)^2 / (se0^2 + se1^2) with SE (expected 1 when sampling variances are "
               "right), halves need >= 10 routes; team_share with team_col = [team_abbr, season] on WR routes; "
               "exposure = corr(shrunk target, log routes) over kept WRs.",
     "why": "Brief; the half-difference check was asked for after the OL pressure target looked under-dispersed."},
    {"item": "feature reliability (criterion 5)",
     "choice": "speed.attempt_icc on WR attempts (centered within draft year), `reliability` column "
               "(Spearman-Brown for each player's number of attempts, averaged).",
     "why": "Reliability of the player-level feature as used."},
    {"item": "go criterion 6",
     "choice": "Seed stability = criteria 1-3 hold at each of seeds 0, 1, 2. Leave-3-out = drop the 3 WRs with the "
               "largest positive products of B2-residualized feature and target (linkage.partial_residuals, i.e. the "
               "largest positive contributions to the partial correlation), rerun with the full harness at seed 0; "
               "passes if the one-sided Freedman-Lane p < 0.05 with the expected sign.",
     "why": "PLAN.md §4 criterion 6 made operational before results."},
    {"item": "BH",
     "choice": "Benjamini-Hochberg over the seed-0 one-sided Freedman-Lane p of P1-P3 (criterion 4). Also reported: "
               "BH over every distinct E007 test (primaries + every robustness variant except seed replicates + "
               "exploratory).",
     "why": "Prereg 'go' field."},
    {"item": "without 2025",
     "choice": "Same player targets (EB priors fitted with all eligible WRs); 2025 draftees dropped before evaluate.",
     "why": "Isolates the 2025 one-season class without changing target definitions."},
    {"item": "WR+TE pooled",
     "choice": "P1 = game_top_speed; P2/P3 from eb_mean(..., group='position') on WR+TE routes (one prior per "
               "position; the WR values equal the WR-only fit, asserted); the harness adds the position dummy.",
     "why": "Prereg 'WR+TE pooled with position dummies'."},
    {"item": "feature variants",
     "choice": "c40_smax (brief's 'raw tracked peak speed'), c40_vmax (fitted vmax), c40_v_30_40 (flying 10), each vs "
               "P1-P3; plus the literal E001 feature (combine_top_speed = max device `s` over 40 attempts) because "
               "the prereg wording names it.",
     "why": "Prereg + brief; c40_smax and the E001 feature differ slightly (SG interior peak, QC-passing mean vs raw max)."},
    {"item": "target variants",
     "choice": "P1: S0 from speed.game_as_profiles (AS_K = 50, AS_R = 1000, seed 0) on WR post-snap REG frames. "
               "P2: cushion minus its lined_up_position mean (over all WR+TE cushion routes), eb_mean min_n 50, "
               "cluster game. P3: rec_yards0 (raw yards per route), eb_mean min_n 100, cluster game.",
     "why": "Prereg; 'alignment' = the receiver's own alignment (lined_up_position: WR / SLOT_WR / TE ...), since "
            "receiver_alignment describes the whole formation (2x2, 3x1)."},
    {"item": "exploratory definitions (expected sign + for all)",
     "choice": "(E1) deep-route share over expected: route in {GO, POST, CORNER}; (E2) average depth of target: "
               "pass_length on the WR's targeted routes, raw, eb_mean min_n 20 targets, cluster game; (E3) targets "
               "per route over expected; (E4) DB analog: c40_v_20_40 -> DB game_top_speed; (E5) career offensive "
               "snaps per available season = career_offensive_snaps / (2026 - draft_year), all Combine WRs with a "
               "tracked 40. Harness n_repeats 20, n_perm 500, n_perm_fl 2000, seed 0.",
     "why": "Prelisted in the prereg; the details were left open and fixed here."},
    {"item": "hidden speed (leaderboard)",
     "choice": "Combine WRs with a tracked 40 and an official forty: center c40_v_20_40 and forty within draft year "
               "(WR means), OLS of centered flying-20 on centered forty, hidden speed = residual (yd/s; mph = x 3600/1760). "
               "Rank 1 = largest. Top quintile = hidden speed >= its 80th percentile. Late = draft round > 3 or undrafted.",
     "why": "Brief definition."},
    {"item": "concept figure pair",
     "choice": "Among WRs with a tracked 40, an official forty and >= 50 tracked REG plays: pairs with |delta forty| "
               "<= 0.01 s (widened to 0.02 / 0.03 only if none), pick the pair with the largest |delta hidden speed|. "
               "Uses Combine data only.",
     "why": "Avoids choosing an illustrative pair by its game outcome."},
]


t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else float("nan")


def evaluate(d, feature, target, **kw):
    r = linkage.evaluate(d, feature, target, expected_sign=1, **kw)
    ci = r["partial_r_ci95"]
    log(f"  {feature} -> {target}: n={r['n']} partial r={r['partial_r'] if r['partial_r'] is None else round(r['partial_r'], 3)} "
        f"{[None if c is None else round(c, 2) for c in ci]} FL p={r['fl_p']} excess={r['delta_r2_vs_B2']['excess']:+.4f}"
        f" +- {r['delta_r2_vs_B2']['se']:.4f} LOCO {r['checks']['loco_excess_positive_classes']}")
    return r


def brief(r: dict) -> dict:
    """The headline numbers of an evaluate result."""
    return {"n": r["n"], "raw_r": r["raw_r"], "partial_r": r["partial_r"], "partial_r_ci95": r["partial_r_ci95"],
            "partial_r_mde80": r["partial_r_mde80"], "fl_p": r["fl_p"],
            "excess": r["delta_r2_vs_B2"]["excess"], "excess_se": r["delta_r2_vs_B2"]["se"],
            "excess_within_position": r["delta_r2_vs_B2"]["excess_within_position"],
            "cv_perm_p": r["cv_perm_p"], "loco": r["checks"]["loco_excess_positive_classes"],
            "sign_matches": r["checks"]["sign_matches"]}


def c123(r: dict) -> tuple[bool, bool, bool]:
    c1 = bool(r["checks"]["fl_p_lt_05"] and r["checks"]["sign_matches"])
    c2 = bool(r["checks"]["excess_lb_gt_0"])
    c3 = int(r["checks"]["loco_excess_positive_classes"].split("/")[0]) >= 2
    return c1, c2, c3


def year_centered(col: str) -> pl.Expr:
    return pl.col(col) - pl.col(col).mean().over("draft_year")


# ================================================================== 1. Combine feature
log("Combine 40: attempts and player table")
frames40 = speed.forty_frames()
attempts = speed.forty_attempts(frames40)
ctab = speed.combine_speed_table(attempts)
attempts.write_parquet(io.PROCESSED / "e007_forty_attempts.parquet")
VARIANT_METRICS = ["v_20_40", "smax", "vmax", "v_30_40"]
attempt_icc = {}
for pos in ("WR", "TE", "DB"):
    t = speed.attempt_icc(attempts.filter(pl.col("position") == pos)).filter(pl.col("metric").is_in(VARIANT_METRICS))
    attempt_icc[pos] = {r["metric"]: {k: v for k, v in r.items() if k != "metric"} for r in t.iter_rows(named=True)}
e001_feature = features.combine_top_speed()

pt = linkage.player_table()
players = io.load("players").select("nfl_id", "draft_round", "draft_overall_pick")
combine_cols = ["nfl_id", FEATURE, "c40_smax", "c40_vmax", "c40_v_30_40", "c40_n_attempts"]
cwr = (pt.filter(pl.col("position") == "WR").join(ctab.select(combine_cols), on="nfl_id", how="inner")
       .with_columns(fc=year_centered("forty"), vc=year_centered(FEATURE)))
yr = cwr.group_by("draft_year").agg(pl.col(FEATURE).mean().alias("mean"), pl.len().alias("n")).sort("draft_year")
tot_var = float(cwr[FEATURE].var(ddof=0))
between = float(np.average((yr["mean"].to_numpy() - cwr[FEATURE].mean()) ** 2, weights=yr["n"].to_numpy()))
feature_summary = {
    "n_combine_wr": int(pt.filter(pl.col("position") == "WR").height),
    "n_wr_with_tracked_40": cwr.height,
    "n_attempts_wr": {"qc_pass": int(attempts.filter((pl.col("position") == "WR") & pl.col("qc_ok")).height),
                      "all": int(attempts.filter(pl.col("position") == "WR").height)},
    "mean_yds": float(cwr[FEATURE].mean()), "sd_yds": float(cwr[FEATURE].std()),
    "mean_mph": float(cwr[FEATURE].mean() * MPH), "sd_mph": float(cwr[FEATURE].std() * MPH),
    "corr_with_official_forty": corr(cwr[FEATURE], cwr["forty"]),
    "corr_with_official_forty_year_centered": corr(cwr["vc"], cwr["fc"]),
    "corr_with_ten_yd_split": corr(cwr[FEATURE], cwr["ten_yd_split"]),
    "corr_with_variants": {v: corr(cwr[FEATURE], cwr[v]) for v in ("c40_smax", "c40_vmax", "c40_v_30_40")},
    "draft_year_means": {int(r["draft_year"]): {"mean": r["mean"], "n": r["n"]} for r in yr.iter_rows(named=True)},
    "between_year_variance_share": between / tot_var,
    "attempt_icc": attempt_icc,
}
log(f"WR with tracked 40: {cwr.height}; WR flying-20 ICC {attempt_icc['WR']['v_20_40']}")

# ================================================================== 2. P1 target (+ S0 variant)
log("P1: game top speed")
gts = features.game_top_speed()
pos_all = pt.select("nfl_id", "position")
wr_ids = pos_all.filter(pl.col("position") == "WR")["nfl_id"].to_list()
play_max = (features.reg_season_tracking().filter(pl.col("nfl_id").is_in(wr_ids))
            .group_by("game_id", "play_id", "nfl_id").agg(smax=pl.col("s").max())
            .filter(pl.col("smax") <= features.MAX_PLAUSIBLE_SPEED).collect())
check = (play_max.group_by("nfl_id").agg(q=pl.col("smax").quantile(0.95), n=pl.len())
         .filter(pl.col("n") >= features.MIN_GAME_PLAYS).join(gts, on="nfl_id", how="inner"))
assert check.height == gts.join(pos_all.filter(pl.col("position") == "WR"), on="nfl_id").height
assert np.allclose(check["q"], check["game_top_speed"]) and (check["n"] == check["game_plays"]).all()
p1_pool = play_max.join(gts.select("nfl_id"), on="nfl_id", how="semi")


def q95_half(g: pl.DataFrame) -> float:
    return float(g["smax"].quantile(0.95)) if g.height >= P1_HALF_MIN_PLAYS else float("nan")


p1_rel = reliability.split_half_estimator(p1_pool, q95_half)
p1_wr = gts.join(pos_all.filter(pl.col("position") == "WR"), on="nfl_id")
p1_diag = {"n_wr_eligible": p1_wr.height, "split_half": p1_rel,
           "corr_log_plays": corr(p1_wr["game_top_speed"], np.log(p1_wr["game_plays"].to_numpy()))}
log(f"P1 split-half {p1_rel}")

log("P1 variant: A-S profile S0 (AS_R = 1000)")
wr_frames = pl.scan_parquet(speed.GAME_FRAMES).filter(pl.col("nfl_id").is_in(wr_ids)).collect()
s0 = speed.game_as_profiles(wr_frames, seed=0)
s0_sh = speed.game_as_split_half(wr_frames, seed=0)
s0_rel = speed.split_half_reliability(s0_sh, "S0")
del wr_frames, s0_sh
s0_ok = s0.filter(pl.col("S0").is_finite())
p1_diag["S0"] = {"n_wr": s0_ok.height, "split_half": s0_rel,
                 "corr_log_plays": corr(s0_ok["S0"], np.log(s0_ok["n_plays"].to_numpy())),
                 "corr_with_game_top_speed": corr(s0_ok.join(gts, on="nfl_id")["S0"],
                                                  s0_ok.join(gts, on="nfl_id")["game_top_speed"])}
log(f"S0 split-half {s0_rel}")

# ================================================================== 3. Route table and expectations
log("Routes: table and cross-fitted expectations")
routes = T.route_table()
assert routes.height == 43162 and (routes["play_nullified_by_penalty"] == "N").all()
assert routes["defense_team"].null_count() == 0 and routes["score_diff"].null_count() == 0
assert routes["yards_to_end_zone"].is_between(1, 99).all()
routes = routes.with_row_index("row_id")
cush_rows = routes.filter(pl.col("cushion").is_not_null())
cush_exp = cush_rows.select("row_id").with_columns(cushion_exp=pl.Series(T.cross_fit(cush_rows, "cushion")))
routes = (routes.join(cush_exp, on="row_id", how="left", maintain_order="left")
          .with_columns(yprr_exp=pl.Series(T.cross_fit(routes, "rec_yards0")),
                        p_deep=pl.Series(T.cross_fit(routes, "deep", kind="classification")),
                        p_target=pl.Series(T.cross_fit(routes, "targeted", kind="classification"))))
routes = routes.with_columns(cushion_oe=pl.col("cushion") - pl.col("cushion_exp"),
                             yprr_oe=pl.col("rec_yards0") - pl.col("yprr_exp"),
                             deep_oe=pl.col("deep") - pl.col("p_deep"),
                             target_oe=pl.col("targeted") - pl.col("p_target"),
                             cushion_c=pl.col("cushion") - pl.col("cushion").mean().over("lined_up_position"))
keep_cols = ["row_id", "game_id", "play_id", "nfl_id", "position", "season", "team_abbr", "defense_team_season",
             "route_ran", *T.CONTEXT_CAT, *T.CONTEXT_NUM, "cushion", "cushion_exp", "cushion_oe", "cushion_c",
             "rec_yards0", "yprr_exp", "yprr_oe", "deep", "p_deep", "deep_oe", "targeted", "p_target", "target_oe",
             "pass_length"]
routes = routes.select(list(dict.fromkeys(keep_cols)))  # written in section 12 with the post hoc columns


def model_summary(rows: pl.DataFrame, y: str, pred: str, binary: bool) -> dict:
    out = {"n_routes": rows.height}
    for name, d in (("all", rows), ("WR", rows.filter(pl.col("position") == "WR"))):
        yv, pv = d[y].to_numpy(), d[pred].to_numpy()
        m = {"n": d.height, "oof_r2": T.oof_r2(yv, pv), "mean_observed": float(yv.mean()), "mean_expected": float(pv.mean())}
        if binary:
            m |= {"auc": float(roc_auc_score(yv, pv)), "brier_skill": T.oof_r2(yv, pv)}
        out[name] = m
    return out


expectation_models = {
    "P2_cushion": model_summary(routes.filter(pl.col("cushion").is_not_null()), "cushion", "cushion_exp", False),
    "P3_yprr": model_summary(routes, "rec_yards0", "yprr_exp", False),
    "E1_deep_route": model_summary(routes, "deep", "p_deep", True),
    "E3_target_per_route": model_summary(routes, "targeted", "p_target", True),
}
log(f"OOF R2: cushion {expectation_models['P2_cushion']['all']['oof_r2']:.3f}, "
    f"yprr {expectation_models['P3_yprr']['all']['oof_r2']:.3f}")

# ================================================================== 4. Player targets + diagnostics
log("Player targets: EB shrinkage and diagnostics")
wr_routes = routes.filter(pl.col("position") == "WR")
wr_cush = wr_routes.filter(pl.col("cushion").is_not_null())
wr_targets = wr_routes.filter((pl.col("targeted") == 1) & pl.col("pass_length").is_not_null())


def eb_mean_fit(value: str, min_n: int):
    return lambda d: shrinkage.eb_mean(d, value, min_n=min_n, cluster="game_id")


def eb_rate_fit(event: str, expected: str, min_n: int):
    return lambda d: shrinkage.eb_rate(d, event, expected, min_n=min_n, cluster="game_id")


def diagnostics(rows: pl.DataFrame, eb: pl.DataFrame, value: str, half_fit) -> dict:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cp = shrinkage.check_prior(eb, rows, value)
    logn = np.log(eb["n"].to_numpy())
    return {"n_wr": eb.height, "routes_median": float(eb["n"].median()), "prior_mean": float(eb["prior_mean"][0]),
            "tau2": float(eb["tau2"][0]), "sd_shrunk": float(eb["shrunk"].std()),
            "check_prior": cp | {"warnings": [str(x.message) for x in w]},
            "half_difference": T.half_difference(rows, eb, half_fit),
            "team_share": reliability.team_share(rows, value, ["team_abbr", "season"]),
            "corr_log_routes_shrunk": corr(eb["shrunk"], logn), "corr_log_routes_raw": corr(eb["raw"], logn)}


TARGETS = {  # name -> (rows, eb, value, half_fit)
    "cushion_oe": (wr_cush, T.player_target(wr_cush, "cushion_oe", MIN_ROUTES["P2"]), "cushion_oe",
                   eb_mean_fit("cushion_oe", 1)),
    "yprr_oe": (wr_routes, T.player_target(wr_routes, "yprr_oe", MIN_ROUTES["P3"]), "yprr_oe",
                eb_mean_fit("yprr_oe", 1)),
    "cushion_c_eb": (wr_cush, T.player_target(wr_cush, "cushion_c", MIN_ROUTES["P2"]), "cushion_c",
                     eb_mean_fit("cushion_c", 1)),
    "yprr_raw_eb": (wr_routes, T.player_target(wr_routes, "rec_yards0", MIN_ROUTES["P3"]), "rec_yards0",
                    eb_mean_fit("rec_yards0", 1)),
    "deep_oe": (wr_routes, shrinkage.eb_rate(wr_routes, "deep", "p_deep", min_n=100, cluster="game_id"), "deep_oe",
                eb_rate_fit("deep", "p_deep", 1)),
    "target_oe": (wr_routes, shrinkage.eb_rate(wr_routes, "targeted", "p_target", min_n=100, cluster="game_id"),
                  "target_oe", eb_rate_fit("targeted", "p_target", 1)),
    "adot": (wr_targets, shrinkage.eb_mean(wr_targets, "pass_length", min_n=20, cluster="game_id"), "pass_length",
             eb_mean_fit("pass_length", 1)),
}
target_diag = {}
for name, (rows, eb, value, half_fit) in TARGETS.items():
    target_diag[name] = diagnostics(rows, eb, value, half_fit)
    d = target_diag[name]
    log(f"  {name}: n={d['n_wr']} split-half {d['check_prior']['split_half']:.2f} implied "
        f"{d['check_prior']['implied_reliability']:.2f} ok={d['check_prior']['ok']} half-diff z2 "
        f"{d['half_difference']['mean_z2']:.2f}+-{d['half_difference']['se']:.2f} team share "
        f"{d['team_share']['team_share']:.2f}+-{d['team_share']['se']:.2f} corr log n {d['corr_log_routes_shrunk']:+.2f}")

# Pooled WR+TE fits (one prior per position); WR values must equal the WR-only fits.
pooled_eb = {}
for name, rows_all, value, min_n in (("cushion_oe", routes.filter(pl.col("cushion").is_not_null()), "cushion_oe", 50),
                                     ("yprr_oe", routes, "yprr_oe", 100)):
    pe = T.player_target(rows_all, value, min_n, positions=("WR", "TE"), group="position")
    wr_only = TARGETS[name][1]
    chk = pe.filter(pl.col("position") == "WR").join(wr_only.select("nfl_id", s=pl.col("shrunk")), on="nfl_id")
    assert chk.height == wr_only.height and np.allclose(chk["shrunk"], chk["s"])
    pooled_eb[name] = pe
te_rows = {"cushion_oe": routes.filter((pl.col("position") == "TE") & pl.col("cushion").is_not_null()),
           "yprr_oe": routes.filter(pl.col("position") == "TE")}
te_diag = {}
for name, pe in pooled_eb.items():
    te = pe.filter(pl.col("position") == "TE").drop("position")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        te_diag[name] = {"n_te": te.height, "check_prior": shrinkage.check_prior(te, te_rows[name], name)}

# ================================================================== 5. Player-level analysis table
log("Player table")
career = (io.load("player_career_successes").select("nfl_id", "career_offensive_snaps")
          .join(pt.select("nfl_id", "draft_year"), on="nfl_id")
          .select("nfl_id", career_snaps_per_season=pl.col("career_offensive_snaps")
                  / (LAST_SEASON + 1 - pl.col("draft_year"))))


def eb_cols(eb: pl.DataFrame, name: str) -> pl.DataFrame:
    return eb.select("nfl_id", pl.col("shrunk").alias(name), pl.col("n").alias(f"{name}_n"))


def pooled_cols(name: str) -> pl.DataFrame:
    return pooled_eb[name].select("nfl_id", pl.col("shrunk").alias(name), pl.col("n").alias(f"{name}_n"))


df = (pt.join(ctab.select(combine_cols), on="nfl_id", how="left")
      .join(e001_feature, on="nfl_id", how="left")
      .join(players, on="nfl_id", how="left")
      .join(gts, on="nfl_id", how="left")
      .join(s0.select("nfl_id", "S0"), on="nfl_id", how="left")
      .join(pooled_cols("cushion_oe"), on="nfl_id", how="left")
      .join(pooled_cols("yprr_oe"), on="nfl_id", how="left")
      .join(eb_cols(TARGETS["cushion_c_eb"][1], "cushion_c_eb"), on="nfl_id", how="left")
      .join(eb_cols(TARGETS["yprr_raw_eb"][1], "yprr_raw_eb"), on="nfl_id", how="left")
      .join(eb_cols(TARGETS["deep_oe"][1], "deep_oe"), on="nfl_id", how="left")
      .join(eb_cols(TARGETS["target_oe"][1], "target_oe"), on="nfl_id", how="left")
      .join(eb_cols(TARGETS["adot"][1], "adot"), on="nfl_id", how="left")
      .join(career, on="nfl_id", how="left")
      .with_columns(S0=pl.col("S0").fill_nan(None)))
# WR-only rows of the EB-derived exploratory and variant columns are WR fits; TE rows of them are null by design.
# Written in section 12 with the post hoc columns.
wr = df.filter(pl.col("position") == "WR")
wrte = df.filter(pl.col("position").is_in(["WR", "TE"]))
db = df.filter(pl.col("position") == "DB")

has_feat = wr.filter(pl.col(FEATURE).is_not_null())
attrition = {
    "combine_wr": wr.height, "with_tracked_40": has_feat.height,
    "P1_ge_50_plays": int(has_feat["game_top_speed"].is_not_null().sum()),
    "P2_ge_50_cushion_routes": int(has_feat["cushion_oe"].is_not_null().sum()),
    "P3_ge_100_routes": int(has_feat["yprr_oe"].is_not_null().sum()),
    "no_tracked_REG_play": int(has_feat.join(play_max.select("nfl_id").unique(), on="nfl_id", how="anti").height),
    "by_class": {int(y): {"tracked_40": int(g.height), "P1": int(g["game_top_speed"].is_not_null().sum()),
                          "P2": int(g["cushion_oe"].is_not_null().sum()), "P3": int(g["yprr_oe"].is_not_null().sum())}
                 for (y,), g in has_feat.group_by(["draft_year"], maintain_order=False)},
}
log(f"attrition {attrition}")

# Reliability of the part of the feature the test uses (beyond the official 40 / beyond B2). Feature-only.
# Player value m_i = mean of n_i attempts with error variance sw2 / n_i (sw2 = pooled within-player attempt variance).
# For a residual maker R (projection off the design), E[e' R e] = sum_i R_ii sw2 / n_i, so
# reliability = 1 - sum_i R_ii sw2 / n_i / (x' R x).
wr_att = attempts.filter(pl.col("qc_ok") & (pl.col("position") == "WR")).select("nfl_id", "v_20_40")
g = wr_att.group_by("nfl_id").agg(n=pl.len(), ss=((pl.col("v_20_40") - pl.col("v_20_40").mean()) ** 2).sum())
sw2 = float(g["ss"].sum() / (g["n"] - 1).sum())


def beyond_reliability(d: pl.DataFrame, controls: list[str], year_effects: bool = True) -> dict:
    d = d.filter(pl.col(FEATURE).is_finite()).sort("nfl_id")
    if controls == ["forty"]:
        d = d.filter(pl.col("forty").is_not_null())
    R = linkage._residual_maker(linkage._imputed(linkage._design(d, controls, year_effects)))
    m, n_att = d[FEATURE].to_numpy(), d["c40_n_attempts"].to_numpy().astype(float)
    err = float(np.sum(np.diag(R) * sw2 / n_att))
    tot = float(m @ R @ m)
    return {"n": d.height, "reliability": 1 - err / tot, "share_of_feature_variance_left": tot / float(np.sum((m - m.mean()) ** 2))}


B2_COLS = linkage.COMBINE_CONTROLS + linkage.DRAFT_CONTROLS
feature_beyond = {"within_player_attempt_sd_yds": float(np.sqrt(sw2))}
for label, pop in (("all_wr_with_tracked_40", has_feat),
                   ("P1_population", has_feat.filter(pl.col("game_top_speed").is_not_null())),
                   ("P2_population", has_feat.filter(pl.col("cushion_oe").is_not_null())),
                   ("P3_population", has_feat.filter(pl.col("yprr_oe").is_not_null()))):
    feature_beyond[label] = {"beyond_year": beyond_reliability(pop, []),
                             "beyond_official_forty_and_year": beyond_reliability(pop, ["forty"]),
                             "beyond_B2": beyond_reliability(pop, B2_COLS)}
log(f"feature reliability beyond B2 (all WR): {feature_beyond['all_wr_with_tracked_40']['beyond_B2']}")

# EB calibration context for P2/P3 (target-only): the unclustered fit, and per-route variance heterogeneity.
eb_context = {}
for name, rows, mn in (("cushion_oe", wr_cush, 50), ("yprr_oe", wr_routes, 100)):
    plain = shrinkage.eb_mean(rows, name, min_n=mn)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        cp = shrinkage.check_prior(plain, rows, name)
    pv = (rows.group_by("nfl_id").agg(n=pl.len(), var=pl.col(name).var(), m=pl.col(name).mean())
          .filter(pl.col("n") >= mn))
    eb_context[name] = {
        "unclustered_check_prior": cp,
        "unclustered_half_difference": T.half_difference(rows, plain, lambda d, v=name: shrinkage.eb_mean(d, v, min_n=1)),
        "per_route_variance_by_player": {"p10": float(pv["var"].quantile(0.1)), "p90": float(pv["var"].quantile(0.9)),
                                         "corr_with_player_mean": corr(pv["var"], pv["m"]),
                                         "corr_with_log_routes": corr(pv["var"], np.log(pv["n"].to_numpy()))}}

# ================================================================== 6. Primary tests (first Combine x target association)
log("PRIMARY tests")
primary = {pid: {s: evaluate(wr, FEATURE, tgt, seed=s, **FULL) for s in SEEDS} for pid, tgt in PRIMARY.items()}
fl_p0 = [primary[p][0]["fl_p"] for p in PRIMARY]
bh_q = dict(zip(PRIMARY, false_discovery_control(fl_p0, method="bh")))


def leave3out(d: pl.DataFrame, feature: str, target: str) -> dict:
    pr = (linkage.partial_residuals(d, feature, target).with_columns(contrib=pl.col("x_res") * pl.col("y_res"))
          .sort("contrib", descending=True))
    drop = pr.head(3)
    r = evaluate(d.filter(~pl.col("nfl_id").is_in(drop["nfl_id"].to_list())), feature, target, seed=0, **FULL)
    return {"dropped": drop.select("nfl_id", "display_name", "draft_year", "contrib").to_dicts(), "result": r}


log("ROBUSTNESS: leave-3-out")
l3o = {pid: leave3out(wr, FEATURE, tgt) for pid, tgt in PRIMARY.items()}

target_rel = {"P1": p1_rel["reliability"], "P2": target_diag["cushion_oe"]["check_prior"]["split_half"],
              "P3": target_diag["yprr_oe"]["check_prior"]["split_half"]}
feat_rel = attempt_icc["WR"]["v_20_40"]["reliability"]
SCOUT = {"P1": "Two receivers with the same 40 time: the one who is faster from the 20 to the 40 is faster in games.",
         "P2": "Corners play off a receiver whose late-sprint speed is better than his 40 time says.",
         "P3": "A receiver whose late-sprint speed beats his 40 time produces more yards per route than his "
               "usage predicts."}
go = {}
for pid in PRIMARY:
    c1, c2, c3 = c123(primary[pid][0])
    seeds_ok = all(all(c123(primary[pid][s])) for s in SEEDS)
    lr = l3o[pid]["result"]
    l3o_ok = bool(lr["fl_p"] is not None and lr["fl_p"] < 0.05 and lr["checks"]["sign_matches"])
    crit = {"1_fl_p_lt_05_and_sign": c1, "2_excess_minus_se_gt_0": c2, "3_loco_excess_pos_ge_2_of_3": c3,
            "4_bh_q_lt_05": bool(bh_q[pid] < 0.05),
            "5_reliability_ge_05": bool(feat_rel >= 0.5 and target_rel[pid] >= 0.5),
            "6_seed_stable_and_not_3_players": bool(seeds_ok and l3o_ok), "6a_seeds_0_1_2": seeds_ok,
            "6b_leave_3_out": l3o_ok, "7_scout_sentence": SCOUT[pid]}
    core = [crit[k] for k in crit if k[0] in "123456" and not k.startswith("6a") and not k.startswith("6b")]
    # Per-seed record of criteria 1-3 (post hoc, skeptic review): criterion 2 as specified ignores the sign of the
    # partial correlation, so `2_with_sign` is shown beside it. It changes no verdict (criterion 1 requires the sign).
    per_seed = {}
    for s in SEEDS:
        s1, s2, s3 = c123(primary[pid][s])
        per_seed[s] = {"1": s1, "2": s2, "2_with_sign": bool(s2 and primary[pid][s]["checks"]["sign_matches"]),
                       "3": s3, "excess": primary[pid][s]["delta_r2_vs_B2"]["excess"],
                       "excess_se": primary[pid][s]["delta_r2_vs_B2"]["se"],
                       "loco": primary[pid][s]["checks"]["loco_excess_positive_classes"]}
    go[pid] = {"criteria": crit, "n_failed_1_6": int(sum(not c for c in core)), "go": bool(all(core)),
               "bh_q": float(bh_q[pid]), "feature_reliability": feat_rel, "target_reliability": target_rel[pid],
               "per_seed_criteria_1_3": per_seed}
    log(f"{pid}: go={go[pid]['go']} q={bh_q[pid]:.4f} {crit}")

# ================================================================== 7. Prespecified robustness
log("ROBUSTNESS: without 2025")
no2025 = {pid: evaluate(wr.filter(pl.col("draft_year") != 2025), FEATURE, tgt, seed=0, **FULL)
          for pid, tgt in PRIMARY.items()}
log("ROBUSTNESS: feature variants")
feature_variants = {f: {pid: evaluate(wr, f, tgt, seed=0, **FULL) for pid, tgt in PRIMARY.items()}
                    for f in ("c40_smax", "c40_vmax", "c40_v_30_40", "combine_top_speed")}
log("ROBUSTNESS: target variants")
target_variants = {"P1_S0": evaluate(wr, FEATURE, "S0", seed=0, **FULL),
                   "P2_alignment_centered_raw_cushion": evaluate(wr, FEATURE, "cushion_c_eb", seed=0, **FULL),
                   "P3_raw_yprr": evaluate(wr, FEATURE, "yprr_raw_eb", seed=0, **FULL)}
log("ROBUSTNESS: WR+TE pooled")
pooled = {pid: evaluate(wrte, FEATURE, tgt, seed=0, **FULL) for pid, tgt in PRIMARY.items()}

# ================================================================== 8. Prelisted exploratory
log("EXPLORATORY")
exploratory = {
    "E1_deep_route_share_oe": evaluate(wr, FEATURE, "deep_oe", **EXPL),
    "E2_average_depth_of_target": evaluate(wr, FEATURE, "adot", **EXPL),
    "E3_targets_per_route_oe": evaluate(wr, FEATURE, "target_oe", **EXPL),
    "E4_DB_game_top_speed": evaluate(db, FEATURE, "game_top_speed", **EXPL),
    "E5_career_off_snaps_per_season": evaluate(wr, FEATURE, "career_snaps_per_season", **EXPL),
}
db_p1 = gts.join(pos_all.filter(pl.col("position") == "DB"), on="nfl_id")
exploratory_diag = {"E4_DB_target_corr_log_plays": corr(db_p1["game_top_speed"], np.log(db_p1["game_plays"].to_numpy())),
                    "E4_DB_feature_attempt_icc": attempt_icc["DB"]["v_20_40"]}

# ================================================================== 8b. Post hoc (added after seeing the results above)
# Labelled post hoc: diagnostics plus one test, none of them claims.
log("POST HOC")
spearman_res = {}
for tgt in (*PRIMARY.values(), "S0"):
    pr = linkage.partial_residuals(wr, FEATURE, tgt)
    spearman_res[tgt] = float(spearmanr(pr["x_res"].to_numpy(), pr["y_res"].to_numpy()).correlation)
p3 = primary["P3"][0]
post_hoc = {
    "note": "Added after the primary, robustness and exploratory results were seen. Not claims.",
    "spearman_of_B2_residuals": spearman_res,  # is a partial correlation carried by a few extreme points?
    "P3_opposite_tail": {  # P3 came out with the opposite sign to the prereg; the prereg test is one-sided (+)
        "one_sided_p_negative_direction": 1 - p3["fl_p"] + 1 / (p3["n_perm_fl"] + 1),
        "two_sided_p_approx": 2 * min(p3["fl_p"], 1 - p3["fl_p"] + 1 / (p3["n_perm_fl"] + 1))},
    # S0 target variant: the largest partial r in E007, but a re-observation of an E006 post hoc result (see 8c)
    "S0_leave_3_out": leave3out(wr, FEATURE, "S0"),
}

# ================================================================== 8c. Post hoc after the skeptic review
# Added after two independent skeptics reviewed every result above. These are sensitivity analyses and diagnostics
# of the preregistered results, not new claims. The preregistered tests and their definitions are unchanged, and
# none of these tests is in any BH family. Harness for tests: PH (FL 10,000 perms, lighter CV).
log("POST HOC (skeptic review)")
PH = dict(n_repeats=20, n_perm=500, n_perm_fl=10000, n_jobs=3, seed=0)


def finite_rows(d: pl.DataFrame, *cols: str) -> pl.DataFrame:
    return d.filter(pl.all_horizontal([pl.col(c).is_finite() for c in cols])).sort("nfl_id")


def b2_matrix(d: pl.DataFrame) -> np.ndarray:
    return linkage._imputed(linkage._design(d, B2_COLS, True))


def ph_eval(d: pl.DataFrame, target: str, extra: list[str] | None = None) -> dict:
    """evaluate with the PH harness; `extra` columns are added to the B2 controls."""
    kw = {"combine_controls": linkage.COMBINE_CONTROLS + extra} if extra else {}
    return evaluate(d, FEATURE, target, **PH, **kw)


def leave3out_reverse(d: pl.DataFrame, target: str) -> dict:
    """Drop the 3 most NEGATIVE contributions x_res * y_res (the opposite direction to criterion 6b) and rerun."""
    pr = (linkage.partial_residuals(d, FEATURE, target).with_columns(contrib=pl.col("x_res") * pl.col("y_res"))
          .sort("contrib"))
    drop = pr.head(3)
    r = ph_eval(d.filter(~pl.col("nfl_id").is_in(drop["nfl_id"].to_list())), target)
    return {"dropped": drop.select("nfl_id", "display_name", "draft_year", "contrib").to_dicts(), "result": r}


def boot_partial_r(d: pl.DataFrame, target: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """Player bootstrap of the partial correlation, re-imputing and re-residualizing on B2 in every resample."""
    d = finite_rows(d, FEATURE, target)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        db = d[rng.integers(0, d.height, d.height)]
        R = linkage._residual_maker(b2_matrix(db))
        out[b] = linkage._corr(R @ db[FEATURE].to_numpy().astype(float), R @ db[target].to_numpy().astype(float))
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return {"n": d.height, "n_boot": n_boot, "ci95_percentile": [float(lo), float(hi)],
            "share_le_0": float(np.nanmean(out <= 0))}


def class_interaction_f(d: pl.DataFrame, target: str) -> dict:
    """F-test of feature x draft-class interactions added to B2 + feature (one common slope vs one per class)."""
    d = finite_rows(d, FEATURE, target)
    x, y, yrs = d[FEATURE].to_numpy().astype(float), d[target].to_numpy().astype(float), d["draft_year"].to_numpy()
    Xa = np.column_stack([b2_matrix(d), x])
    Xb = np.column_stack([Xa] + [x * (yrs == k) for k in np.unique(yrs)[1:]])

    def rss(X):
        return float(np.sum((y - X @ (np.linalg.pinv(X) @ y)) ** 2))

    q = np.linalg.matrix_rank(Xb) - np.linalg.matrix_rank(Xa)
    df_b = len(y) - np.linalg.matrix_rank(Xb)
    fst = ((rss(Xa) - rss(Xb)) / q) / (rss(Xb) / df_b)
    return {"n": d.height, "F": fst, "df": [int(q), int(df_b)], "p": float(f_dist.sf(fst, q, df_b))}


def drop_one_range(d: pl.DataFrame, target: str) -> dict:
    d = finite_rows(d, FEATURE, target)
    rs = []
    for i in range(d.height):
        di = d.filter(pl.int_range(pl.len()) != i)
        R = linkage._residual_maker(b2_matrix(di))
        rs.append(linkage._corr(R @ di[FEATURE].to_numpy().astype(float), R @ di[target].to_numpy().astype(float)))
    return {"n": d.height, "min": float(min(rs)), "max": float(max(rs))}


def leave3out_power(d: pl.DataFrame, target: str, rhos=(0.0, 0.2, 0.3, 0.4, 0.5), n_sim: int = 2000,
                    seed: int = 0) -> dict:
    """Power of the leave-3-out check (criterion 6b) on the actual design: y = beta x + N(0, 1) with true partial r
    rho given B2. Criterion 1 and the leave-3-out rerun are approximated by the one-sided t-test of the partial
    correlation (df = n - rank - 1) instead of the Freedman-Lane permutation; rows are dropped from the full-sample
    imputed design instead of re-imputing."""
    d = finite_rows(d, FEATURE, target)
    X, x = b2_matrix(d), d[FEATURE].to_numpy().astype(float)
    n = len(x)
    rx = linkage._residual_maker(X) @ x
    sx = np.sqrt(rx @ rx / (n - np.linalg.matrix_rank(X)))
    rng = np.random.default_rng(seed)

    def test(Xd, xv, yv):
        R = linkage._residual_maker(Xd)
        a, b = R @ xv, R @ yv
        r = linkage._corr(a, b)
        df_ = len(yv) - np.linalg.matrix_rank(Xd) - 1
        return r, float(t_dist.sf(r * np.sqrt(df_ / (1 - r * r)), df_)), a * b

    out = {}
    for rho in rhos:
        beta = rho / np.sqrt(1 - rho**2) / sx
        n1 = n2 = 0
        for _ in range(n_sim):
            yv = beta * x + rng.standard_normal(n)
            r, p, contrib = test(X, x, yv)
            if r > 0 and p < 0.05:
                n1 += 1
                keep = np.ones(n, bool)
                keep[np.argsort(-contrib)[:3]] = False
                r2, p2, _ = test(X[keep], x[keep], yv[keep])
                n2 += bool(r2 > 0 and p2 < 0.05)
        out[f"{rho:.1f}"] = {"p_criterion_1": n1 / n_sim, "p_leave_3_out_given_criterion_1": n2 / max(n1, 1)}
    return {"n": n, "n_sim": n_sim, "by_true_partial_r": out}


# (a) S0 provenance (skeptic A1): the E006 skeptic's post hoc flying split (raw x = 31 -> 50, i.e. 21 -> 40 yd past
# the start line) vs the E007 feature. The S0 target variant re-observes that E006 post hoc result on the same WRs.
qc_wr_traj = attempts.filter(pl.col("qc_ok") & (pl.col("position") == "WR")).select("traj", "nfl_id")
split_rows = []
for (traj,), g40 in frames40.join(qc_wr_traj, on="traj", how="semi").group_by(["traj"], maintain_order=True):
    t21, t40 = speed.crossing_times(g40["t"].to_numpy(), g40["x"].to_numpy() - speed.START_X, [21, 40])
    split_rows.append({"traj": traj, "v_21_40": 19 / (t40 - t21)})
e006_split = (pl.DataFrame(split_rows).join(qc_wr_traj, on="traj").group_by("nfl_id")
              .agg(e006_split=pl.col("v_21_40").fill_nan(None).mean()))
sp = finite_rows(has_feat.join(e006_split, on="nfl_id"), FEATURE, "e006_split")
R_sp = linkage._residual_maker(b2_matrix(sp))
s0_provenance = {
    "n": sp.height, "corr_raw": corr(sp[FEATURE], sp["e006_split"]),
    "corr_beyond_B2": corr(R_sp @ sp[FEATURE].to_numpy(), R_sp @ sp["e006_split"].to_numpy()),
    "note": "The E007 feature is, beyond B2, the same variable as the E006 skeptic's post hoc flying split, which "
            "was chosen after it absorbed the vmax -> S0 signal on these WRs (prereg 'already_seen' item 2). The "
            "S0 target variant is therefore a re-observation of a known post hoc association, not new evidence; "
            "its q-values carry no confirmatory weight, and a new prereg on the same WRs cannot change that."}
log(f"  S0 provenance: {s0_provenance['corr_raw']:.4f} raw, {s0_provenance['corr_beyond_B2']:.4f} beyond B2")

# (b) P1: heterogeneity across classes, drop-one range, and the power of the leave-3-out check (skeptic A2).
p1_power = {"class_interaction": class_interaction_f(wr, "game_top_speed"),
            "drop_one_partial_r": drop_one_range(wr, "game_top_speed"),
            "leave_3_out_power": leave3out_power(wr, "game_top_speed")}
log(f"  P1 interaction {p1_power['class_interaction']}, drop-one {p1_power['drop_one_partial_r']}")

# (c) Receiver alignment at the snap and what `cushion` measures (skeptic B1). Snap positions come from standardized
# REG tracking (event 'ball_snap'); the line of scrimmage is at x = 110 - yards_to_end_zone. Defenders = study
# players on the defensive team of a route play (most defenders are not tracked).
log("  receiver alignment at the snap")
route_keys = routes.select("game_id", "play_id").unique()
defenders = (io.load("player_play").filter(pl.col("team_abbr") != pl.col("possession_team"))
             .select("game_id", "play_id", def_id="nfl_id").join(route_keys, on=["game_id", "play_id"], how="semi"))
snap_ids = sorted(set(routes["nfl_id"].unique().to_list()) | set(defenders["def_id"].unique().to_list()))
snap = (features.reg_season_tracking()
        .filter(pl.col("nfl_id").is_in(snap_ids) & (pl.col("event") == "ball_snap"))
        .join(route_keys.lazy(), on=["game_id", "play_id"], how="semi")
        .group_by("game_id", "play_id", "nfl_id")
        .agg(x_snap=pl.col("x").sort_by("time").first(), y_snap=pl.col("y").sort_by("time").first()).collect())
cs = (routes.filter(pl.col("cushion").is_not_null()).select("row_id", "game_id", "play_id", "nfl_id", "cushion")
      .join(defenders, on=["game_id", "play_id"])
      .join(snap, on=["game_id", "play_id", "nfl_id"])
      .join(snap.rename({"nfl_id": "def_id", "x_snap": "xd", "y_snap": "yd"}), on=["game_id", "play_id", "def_id"])
      .with_columns(dist=((pl.col("x_snap") - pl.col("xd")) ** 2 + (pl.col("y_snap") - pl.col("yd")) ** 2).sqrt())
      .group_by("row_id").agg(pl.col("cushion").first(), nearest=pl.col("dist").min(),
                              matches_any=((pl.col("cushion") - pl.col("dist")).abs() < 0.05).any()))
close = cs.filter(pl.col("nearest") < 10)
cushion_semantics = {
    "n_cushion_routes_with_a_tracked_defender": cs.height, "n_with_a_tracked_defender_within_10yd": close.height,
    "share_cushion_equals_a_tracked_defender_distance_at_snap": float(close["matches_any"].mean()),
    "share_cushion_equals_nearest_tracked_defender": float(((close["cushion"] - close["nearest"]).abs() < 0.05).mean()),
    "share_a_tracked_defender_closer_than_cushion_by_0_2yd": float((close["cushion"] > close["nearest"] + 0.2).mean()),
    "reading": "cushion is the receiver-to-(aligned)-defender distance at the snap, so it contains the receiver's "
               "own depth off the line and split, which the prereg context does not include."}
routes_al = (routes.join(snap, on=["game_id", "play_id", "nfl_id"], how="left", maintain_order="left")
             .with_columns(rec_depth=(110 - pl.col("yards_to_end_zone")) - pl.col("x_snap"),
                           rec_sideline=pl.min_horizontal(pl.col("y_snap"), 160 / 3 - pl.col("y_snap")))
             .drop("x_snap", "y_snap"))
del snap, cs, close
cush_al = routes_al.filter(pl.col("cushion").is_not_null())
cush_al = cush_al.with_columns(cushion_exp_align=pl.Series(
    T.cross_fit(cush_al, "cushion", num=T.CONTEXT_NUM + ["rec_depth", "rec_sideline"])))
cush_al = cush_al.with_columns(cushion_oe_align=pl.col("cushion") - pl.col("cushion_exp_align"))
routes_al = routes_al.join(cush_al.select("row_id", "cushion_exp_align", "cushion_oe_align"), on="row_id", how="left",
                           maintain_order="left")
wr_routes_al = routes_al.filter(pl.col("position") == "WR")
wr_cush_al = wr_routes_al.filter(pl.col("cushion").is_not_null())
eb_al = T.player_target(wr_cush_al, "cushion_oe_align", MIN_ROUTES["P2"])
wr_cush_depth = wr_cush_al.filter(pl.col("rec_depth").is_not_null())
alignment = {
    "share_routes_with_snap_frame": float(routes_al["rec_depth"].is_not_null().mean()),
    "corr_route_level_cushion_oe_with_receiver_depth": corr(wr_cush_depth["cushion_oe"], wr_cush_depth["rec_depth"]),
    "corr_route_level_cushion_oe_with_receiver_sideline": corr(wr_cush_depth["cushion_oe"], wr_cush_depth["rec_sideline"]),
    "receiver_depth_trait": {"split_half": reliability.split_half(wr_routes_al, "rec_depth", min_per_half=25),
                             "team_share": reliability.team_share(wr_routes_al, "rec_depth", ["team_abbr", "season"])},
    "cushion_model_with_alignment": {"oof_r2": T.oof_r2(cush_al["cushion"].to_numpy(),
                                                        cush_al["cushion_exp_align"].to_numpy()),
                                     "oof_r2_prereg_context": expectation_models["P2_cushion"]["all"]["oof_r2"]},
    "cushion_oe_align_eb": diagnostics(wr_cush_al, eb_al, "cushion_oe_align", eb_mean_fit("cushion_oe_align", 1)),
}
log(f"  cushion semantics {cushion_semantics}; alignment OOF R2 {alignment['cushion_model_with_alignment']}")

# (d) Player-level post hoc covariates. Offense quality: mean EPA of the receiver's team-season dropbacks
# (REG plays in player_play with a pass_result), excluding the plays on which he was targeted (his own production);
# the inclusive version is the skeptic's. QB: passer parsed from play_description ("X.Name pass"), mean EPA of that
# passer-team-season's dropbacks without the receiver's own targets (>= 50 such plays). Both route-weighted over the
# receiver's REG routes. Role: deep-route and slot shares of his REG routes. Whole-population production: receiving
# yards and routes per available season (0 if none) for every Combine WR with a tracked 40.
reg_games = io.load("games").filter(pl.col("season_type") == "REG").select("game_id", "season")
pp_all = io.load("player_play")
dropbacks = (pp_all.filter(pl.col("pass_result").is_not_null()).unique(["game_id", "play_id"])
             .join(reg_games, on="game_id")
             .select("game_id", "play_id", "possession_team", "season", epa="expected_points_added",
                     passer=pl.col("play_description").str.extract(r"([A-Z][A-Za-z\.'\-]+)\s+pass", 1)))
own_targets = pp_all.filter(pl.col("target")).select("game_id", "play_id", "nfl_id").join(
    dropbacks, on=["game_id", "play_id"])
del pp_all
team_sum = dropbacks.group_by("possession_team", "season").agg(ts=pl.col("epa").sum(), tn=pl.len())
qb_sum = (dropbacks.drop_nulls("passer").group_by("passer", "possession_team", "season")
          .agg(qs=pl.col("epa").sum(), qn=pl.len()))
own_team = own_targets.group_by("nfl_id", "possession_team", "season").agg(os=pl.col("epa").sum(), on=pl.len())
own_qb = (own_targets.drop_nulls("passer").group_by("nfl_id", "passer", "possession_team", "season")
          .agg(oqs=pl.col("epa").sum(), oqn=pl.len()))
tkeys = ["team_abbr", "season"]
offense = (wr_routes_al.select("game_id", "play_id", "nfl_id", *tkeys)
           .join(dropbacks.select("game_id", "play_id", "passer"), on=["game_id", "play_id"], how="left")
           .join(team_sum, left_on=tkeys, right_on=["possession_team", "season"], how="left")
           .join(own_team, left_on=["nfl_id", *tkeys], right_on=["nfl_id", "possession_team", "season"], how="left")
           .join(qb_sum, left_on=["passer", *tkeys], right_on=["passer", "possession_team", "season"], how="left")
           .join(own_qb, left_on=["nfl_id", "passer", *tkeys], right_on=["nfl_id", "passer", "possession_team", "season"],
                 how="left")
           .with_columns(pl.col("os", "on", "oqs", "oqn").fill_null(0))
           .with_columns(off_epa=(pl.col("ts") - pl.col("os")) / (pl.col("tn") - pl.col("on")),
                         off_epa_incl=pl.col("ts") / pl.col("tn"),
                         qb_epa=pl.when(pl.col("qn") - pl.col("oqn") >= 50)
                         .then((pl.col("qs") - pl.col("oqs")) / (pl.col("qn") - pl.col("oqn"))))
           .group_by("nfl_id").agg(pl.col("off_epa", "off_epa_incl", "qb_epa").mean()))
offense_meta = {"team_season_dropbacks_median": float(team_sum["tn"].median()), "team_season_dropbacks_min": int(team_sum["tn"].min()),
                "share_dropbacks_with_parsed_passer": float(dropbacks["passer"].is_not_null().mean())}
role = (wr_routes_al.group_by("nfl_id")
        .agg(deep_share=pl.col("deep").mean(), slot_share=(pl.col("lined_up_position") == "SLOT_WR").mean(),
             depth_mean=pl.col("rec_depth").mean(), sideline_mean=pl.col("rec_sideline").mean(),
             rec_yards_reg=pl.col("rec_yards0").sum(), routes_reg=pl.len()))
ph_player = (has_feat.select("nfl_id", "draft_year")
             .join(eb_al.select("nfl_id", cushion_oe_align="shrunk"), on="nfl_id", how="left")
             .join(offense, on="nfl_id", how="left").join(role, on="nfl_id", how="left")
             .with_columns(pl.col("rec_yards_reg", "routes_reg").fill_null(0))
             .with_columns(rec_yards_per_season=pl.col("rec_yards_reg") / (LAST_SEASON + 1 - pl.col("draft_year")),
                           log_routes_per_season=(1 + pl.col("routes_reg") / (LAST_SEASON + 1 - pl.col("draft_year"))).log())
             .drop("draft_year"))
wr_ph = wr.join(ph_player, on="nfl_id", how="left")


def residual_corrs(target: str, cols: list[str]) -> dict:
    """Correlation of each covariate with the B2-residualized feature and target (who confounds what)."""
    pr = linkage.partial_residuals(wr_ph, FEATURE, target).join(wr_ph.select("nfl_id", *cols), on="nfl_id")
    out = {}
    for c in cols:
        ok = pr.filter(pl.col(c).is_finite())
        out[c] = {"n": ok.height, "with_feature_res": corr(ok["x_res"], ok[c]), "with_target_res": corr(ok["y_res"], ok[c])}
    return out


covariate_corrs = {"P1": residual_corrs("game_top_speed", ["deep_share", "slot_share", "off_epa"]),
                   "P2": residual_corrs("cushion_oe", ["depth_mean", "sideline_mean", "slot_share"]),
                   "P3": residual_corrs("yprr_oe", ["off_epa", "off_epa_incl", "qb_epa", "deep_share", "slot_share"]),
                   "S0": residual_corrs("S0", ["deep_share", "slot_share"])}

# (e) Post hoc tests (PH harness; expected sign + as in the prereg).
log("  post hoc tests")
ph_tests = {
    "P2_alignment_in_context": ph_eval(wr_ph, "cushion_oe_align"),
    "P3_plus_offense_epa_excl_own_targets": ph_eval(wr_ph, "yprr_oe", ["off_epa"]),
    "P3_plus_offense_epa_incl_own_targets": ph_eval(wr_ph, "yprr_oe", ["off_epa_incl"]),
    "P3_plus_offense_and_qb_epa": ph_eval(wr_ph, "yprr_oe", ["off_epa", "qb_epa"]),
    "whole_population_rec_yards_per_season": ph_eval(wr_ph, "rec_yards_per_season"),
    "whole_population_log_routes_per_season": ph_eval(wr_ph, "log_routes_per_season"),
    "P1_plus_deep_share": ph_eval(wr_ph, "game_top_speed", ["deep_share"]),
    "P1_plus_offense_epa": ph_eval(wr_ph, "game_top_speed", ["off_epa"]),
    "S0_plus_deep_share": ph_eval(wr_ph, "S0", ["deep_share"]),
    "S0_plus_deep_and_slot_share": ph_eval(wr_ph, "S0", ["deep_share", "slot_share"]),
}
reverse_l3o = {"P2": leave3out_reverse(wr, "cushion_oe"), "P3": leave3out_reverse(wr, "yprr_oe")}
log("  bootstrap")
bootstrap = {"P1": boot_partial_r(wr, "game_top_speed"), "P2": boot_partial_r(wr, "cushion_oe"),
             "P3": boot_partial_r(wr, "yprr_oe"), "S0": boot_partial_r(wr, "S0"),
             "P2_alignment_in_context": boot_partial_r(wr_ph, "cushion_oe_align")}
rel_al = alignment["cushion_oe_align_eb"]["check_prior"]["split_half"]
rel_x_p2 = feature_beyond["P2_population"]["beyond_B2"]["reliability"]
r_al = ph_tests["P2_alignment_in_context"]
disatt_al = np.sqrt(rel_x_p2 * max(rel_al, 1e-9))
p2_alignment_disattenuated = {"partial_r": r_al["partial_r"] / disatt_al,
                              "ci95": [c / disatt_al for c in r_al["partial_r_ci95"]],
                              "target_split_half": rel_al, "feature_reliability_beyond_B2": rel_x_p2}

# Upper 95% bounds on the P2 / P3 partial r under the alternatives above (how far 'ruled out' can be pushed).
upper_bounds = {
    "P2": {"prereg_fisher": primary["P2"][0]["partial_r_ci95"][1], "bootstrap": bootstrap["P2"]["ci95_percentile"][1],
           "alignment_in_context_fisher": r_al["partial_r_ci95"][1],
           "alignment_in_context_disattenuated": p2_alignment_disattenuated["ci95"][1],
           "reverse_leave_3_out_fisher": reverse_l3o["P2"]["result"]["partial_r_ci95"][1]},
    "P3": {"prereg_fisher": primary["P3"][0]["partial_r_ci95"][1], "bootstrap": bootstrap["P3"]["ci95_percentile"][1],
           "plus_offense_epa_fisher": ph_tests["P3_plus_offense_epa_excl_own_targets"]["partial_r_ci95"][1],
           "plus_offense_epa_incl_own_targets_fisher": ph_tests["P3_plus_offense_epa_incl_own_targets"]["partial_r_ci95"][1],
           "plus_offense_and_qb_epa_fisher": ph_tests["P3_plus_offense_and_qb_epa"]["partial_r_ci95"][1],
           "whole_population_rec_yards_per_season_fisher":
               ph_tests["whole_population_rec_yards_per_season"]["partial_r_ci95"][1],
           "reverse_leave_3_out_fisher": reverse_l3o["P3"]["result"]["partial_r_ci95"][1]},
}
post_hoc_review = {
    "note": "Added after the two-skeptic review of all results above. Sensitivity analyses and diagnostics of the "
            "preregistered results; not claims; in no BH family. Tests use expected sign + and the PH harness.",
    "harness": PH,
    "definitions": {
        "receiver_depth": "line of scrimmage x (110 - yards_to_end_zone) minus the receiver's standardized x at the "
                          "ball_snap frame (yd; positive = off the line)",
        "receiver_sideline": "min(y, 53.33 - y) at the ball_snap frame (yd)",
        "cushion_oe_align": "cushion minus a cross-fitted expectation from the prereg context plus receiver depth and "
                            "sideline distance (same HGB, folds and random_state), eb_mean WR min_n 50 cluster game",
        "off_epa": "route-weighted mean EPA of the receiver's team-season REG dropbacks in player_play, excluding "
                   "the dropbacks on which he was targeted (off_epa_incl keeps them)",
        "qb_epa": "route-weighted mean EPA of the parsed passer x team-season's dropbacks without the receiver's own "
                  "targets (>= 50), null otherwise (imputed with an indicator by the harness)",
        "deep_share / slot_share": "share of the receiver's REG routes that are GO/POST/CORNER / lined up SLOT_WR",
        "rec_yards_per_season / log_routes_per_season": "REG receiving yards (routes) / (2026 - draft_year), 0 if "
                                                        "none, all Combine WRs with a tracked 40",
        "reverse_leave_3_out": "drop the 3 most negative x_res * y_res contributions (criterion 6b drops the most "
                               "positive), rerun",
        "bootstrap": "2,000 player resamples (seed 0), re-imputed and re-residualized on B2, percentile interval",
    },
    "S0_provenance": s0_provenance,
    "P1_power_and_heterogeneity": p1_power,
    "cushion_semantics": cushion_semantics,
    "alignment": alignment,
    "offense_meta": offense_meta,
    "covariate_correlations_with_B2_residuals": covariate_corrs,
    "tests": ph_tests,
    "reverse_leave_3_out": reverse_l3o,
    "bootstrap_partial_r": bootstrap,
    "P2_alignment_disattenuated": p2_alignment_disattenuated,
    "upper_95_bounds": upper_bounds,
}

# ================================================================== 9. BH across all distinct E007 tests
all_tests = {f"primary/{p}": primary[p][0] for p in PRIMARY}
all_tests |= {f"leave3out/{p}": l3o[p]["result"] for p in PRIMARY}
all_tests |= {f"no2025/{p}": no2025[p] for p in PRIMARY}
all_tests |= {f"feature_variant/{f}/{p}": r for f, v in feature_variants.items() for p, r in v.items()}
all_tests |= {f"target_variant/{k}": r for k, r in target_variants.items()}
all_tests |= {f"pooled_WR_TE/{p}": pooled[p] for p in PRIMARY}
all_tests |= {f"exploratory/{k}": r for k, r in exploratory.items()}
# The family is the prespecified tests only (implementation choice 'BH'). Post hoc tests (8b, 8c) are in no family:
# an earlier version added the post hoc S0 leave-3-out here, which mechanically lowered every other q (skeptic A4).
# Also reported: the same family without the nested subset reruns of the primaries (leave-3-out, without 2025).
q_all = false_discovery_control([r["fl_p"] for r in all_tests.values()], method="bh")
nested = [k for k in all_tests if k.startswith(("leave3out/", "no2025/"))]
not_nested = [k for k in all_tests if k not in nested]
q_nn = dict(zip(not_nested, false_discovery_control([all_tests[k]["fl_p"] for k in not_nested], method="bh")))
bh_all = {k: {"fl_p": r["fl_p"], "q": float(q), "q_without_nested_reruns": None if k in nested else float(q_nn[k]),
              "partial_r": r["partial_r"]} for (k, r), q in zip(all_tests.items(), q_all)}
q_pe = false_discovery_control([primary[p][0]["fl_p"] for p in PRIMARY] + [r["fl_p"] for r in exploratory.values()],
                               method="bh")
bh_primary_plus_exploratory = dict(zip([*PRIMARY, *exploratory], map(float, q_pe)))

# Reliability-corrected (disattenuated) partial correlation for the primaries: the feature's reliability BEYOND B2
# in the test population (the part of the feature the partial correlation uses) and the target's split-half.
disattenuated = {}
for pid in PRIMARY:
    r0 = primary[pid][0]
    rel_x = feature_beyond[f"{pid}_population"]["beyond_B2"]["reliability"]
    rel = np.sqrt(max(rel_x, 1e-9) * max(target_rel[pid], 1e-9))
    disattenuated[pid] = {"partial_r": r0["partial_r"] / rel, "ci95": [c / rel for c in r0["partial_r_ci95"]],
                          "feature_reliability_beyond_B2": rel_x, "target_reliability": target_rel[pid],
                          "sqrt_reliability_product": float(rel)}

# ================================================================== 10. Leaderboard
log("Leaderboard")
board = wr.filter(pl.col(FEATURE).is_not_null())
fit_rows = (board.filter(pl.col("forty").is_not_null())
            .with_columns(fc=year_centered("forty"), vc=year_centered(FEATURE)))
fcv, vcv = fit_rows["fc"].to_numpy(), fit_rows["vc"].to_numpy()
slope = float(np.sum(fcv * vcv) / np.sum(fcv**2))
hv = vcv - slope * fcv
q80 = float(np.quantile(hv, 0.8))


def beyond_forty(d: pl.DataFrame, col: str) -> np.ndarray:
    """Residual of year-centered `col` on the year-centered official forty (the hidden-speed construction)."""
    t = d.with_columns(_f=year_centered("forty"), _v=year_centered(col))
    f_, v_ = t["_f"].to_numpy(), t["_v"].to_numpy()
    return v_ - np.sum(f_ * v_) / np.sum(f_**2) * f_


# Post hoc (skeptic review): measurement-error shrinkage of hidden speed, per-player uncertainty and P(truly top
# quintile). Each player's flying 20 is a mean of n_i attempts with error variance sw2 / n_i (sw2 = pooled
# within-player attempt variance, section 5). Normal-normal: tau2 = Var(hidden) - mean(sw2 / n_i); posterior mean
# m + w (h - m) with w = tau2 / (tau2 + sw2 / n_i), posterior SD sqrt(w sw2 / n_i); P(true top quintile) =
# P(true hidden >= m + z_0.8 tau). Error in the official forty and in the slope is ignored, so the intervals are,
# if anything, too narrow.
err_h = sw2 / fit_rows["c40_n_attempts"].to_numpy().astype(float)
m_h, var_h = float(hv.mean()), float(np.var(hv, ddof=1))
tau2_h = var_h - float(err_h.mean())
w_h = tau2_h / (tau2_h + err_h)
pm_h, psd_h = m_h + w_h * (hv - m_h), np.sqrt(w_h * err_h)
thr_true = m_h + float(norm.ppf(0.8)) * np.sqrt(tau2_h)
Z80 = float(norm.ppf(0.9))  # half-width of an 80% central interval, in SDs
hidden = fit_rows.select("nfl_id").with_columns(hidden=pl.Series(hv), hidden_post=pl.Series(pm_h),
                                                hidden_post_sd=pl.Series(psd_h),
                                                p_top=pl.Series(norm.sf((thr_true - pm_h) / psd_h)))
board = (board.join(hidden, on="nfl_id", how="left")
         .with_columns(rank=pl.col("hidden").rank("ordinal", descending=True).cast(pl.Int64),
                       rank_shrunk=pl.col("hidden_post").rank("ordinal", descending=True).cast(pl.Int64),
                       top_quintile=pl.col("hidden") >= q80,
                       late=pl.col("undrafted").cast(pl.Boolean) | (pl.col("draft_round") > 3))
         .with_columns(top_late=pl.col("top_quintile") & pl.col("late"))
         .sort("rank", nulls_last=True))
tl = board.filter(pl.col("top_late"))

# Post hoc (skeptic review): does the top-quintile call replicate from one 40 attempt to the next? (WRs with >= 2
# QC-passing attempts; hidden speed rebuilt from attempt 1 alone and from attempt 2 alone, same official forty.)
a12 = (attempts.filter(pl.col("qc_ok") & (pl.col("position") == "WR")).sort("nfl_id", "event_id", "attempt")
       .with_columns(k=pl.int_range(pl.len()).over("nfl_id").cast(pl.String)).filter(pl.col("k").is_in(["0", "1"]))
       .pivot(on="k", index="nfl_id", values="v_20_40").drop_nulls()
       .join(fit_rows.select("nfl_id", "draft_year", "forty"), on="nfl_id"))
h1, h2 = beyond_forty(a12, "0"), beyond_forty(a12, "1")
top1, top2 = h1 >= np.quantile(h1, 0.8), h2 >= np.quantile(h2, 0.8)
attempt_replication = {"n_wr_with_2_attempts": a12.height, "corr_hidden_attempt1_vs_attempt2": corr(h1, h2),
                       "n_top_quintile_by_attempt1": int(top1.sum()), "n_still_top_quintile_by_attempt2": int((top1 & top2).sum())}

# Post hoc (skeptic review): do late-round / undrafted WRs in the top quintile reach the field more often?
late_rows = board.filter(pl.col("late") & pl.col("hidden").is_not_null())
field = {}
for grp, sub in (("top_quintile", late_rows.filter(pl.col("top_quintile"))),
                 ("other", late_rows.filter(~pl.col("top_quintile")))):
    field[grp] = {"n": sub.height, "ge_50_tracked_REG_plays": int(sub["game_top_speed"].is_not_null().sum()),
                  "ge_100_REG_routes": int(sub["yprr_oe"].is_not_null().sum())}

# Post hoc (skeptic review): hidden speed against the official electronic splits. Tracked 10->40 time = sum of the
# tracked 10-20, 20-30 and 30-40 splits (mean over QC attempts); official 10->40 = forty - ten_yd_split.
t1040 = (attempts.filter(pl.col("qc_ok") & (pl.col("position") == "WR"))
         .group_by("nfl_id").agg(t10_40=(pl.col("split_10_20") + pl.col("split_20_30") + pl.col("split_30_40"))
                                 .fill_nan(None).mean()))
osd = (fit_rows.join(hidden.select("nfl_id", "hidden"), on="nfl_id").join(t1040, on="nfl_id")
       .filter(pl.col("ten_yd_split").is_not_null() & pl.col("t10_40").is_finite())
       .with_columns(off10_40=pl.col("forty") - pl.col("ten_yd_split"), ten_c=year_centered("ten_yd_split"))
       .with_columns(off_fly30=30 / pl.col("off10_40")))
off_hidden = beyond_forty(osd, "off_fly30")
osd_h = osd["hidden"].to_numpy()
tq_tr, tq_off = osd_h >= np.quantile(osd_h, 0.8), off_hidden >= np.quantile(off_hidden, 0.8)
official_splits = {
    "n": osd.height,
    "r2_hidden_on_official_ten_yd_split": corr(osd_h, osd["ten_c"]) ** 2,
    "corr_tracked_vs_official_10_40_time_raw": corr(osd["t10_40"], osd["off10_40"]),
    "corr_tracked_vs_official_10_40_time_beyond_forty": corr(beyond_forty(osd, "t10_40"), beyond_forty(osd, "off10_40")),
    "corr_hidden_vs_official_flying30_beyond_forty": corr(osd_h, off_hidden),
    "top_quintile_overlap_tracked_vs_official": [int((tq_tr & tq_off).sum()), int(tq_tr.sum())],
    "note": "The tracked component beyond the 40 is not corroborated by the official electronic 10-yd split, and "
            "the 20-yd split clubs receive is not in the data. 'Beyond the official 40', not 'beyond the stopwatch'."}

leaderboard_meta = {"n": board.height, "n_with_official_forty": fit_rows.height,
                    "slope_flying20_on_forty_year_centered": slope,
                    "r2_flying20_on_forty_year_centered": float(1 - np.var(hv) / np.var(vcv)),
                    "top_quintile_threshold_yds": q80, "top_quintile_threshold_mph": q80 * MPH,
                    "n_top_quintile_and_late": tl.height,
                    "post_hoc_uncertainty": {
                        "within_player_attempt_sd_yds": float(np.sqrt(sw2)), "sd_hidden_yds": float(np.sqrt(var_h)),
                        "sd_true_hidden_yds": float(np.sqrt(tau2_h)), "reliability_of_hidden": tau2_h / var_h,
                        "true_top_quintile_threshold_yds": thr_true,
                        "top_quintile_and_late_expected_truly_top_quintile": float(tl["p_top"].sum()),
                        "top_quintile_and_late_with_p_le_0_5": int((tl["p_top"] <= 0.5).sum())},
                    "post_hoc_attempt_replication": attempt_replication,
                    "post_hoc_late_wrs_reaching_the_field": field,
                    "post_hoc_official_splits": official_splits}
log(f"  leaderboard: {leaderboard_meta['post_hoc_uncertainty']}; {attempt_replication}; {field}")


def opt(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v


def mph(v):
    return None if v is None or not np.isfinite(v) else v * MPH


lb_rows = []
for r in board.iter_rows(named=True):
    has_h = r["hidden"] is not None
    lb_rows.append({
        "rank": r["rank"], "rank_shrunk": r["rank_shrunk"], "nfl_id": r["nfl_id"], "name": r["display_name"],
        "draft_year": r["draft_year"], "draft_round": r["draft_round"], "draft_pick": r["draft_overall_pick"],
        "undrafted": bool(r["undrafted"]), "official_forty": r["forty"], "flying20_yds": r[FEATURE],
        "flying20_mph": r[FEATURE] * MPH, "n_qc_attempts": r["c40_n_attempts"],
        "hidden_speed_yds": opt(r["hidden"]), "hidden_speed_mph": mph(r["hidden"]) if has_h else None,
        "hidden_speed_shrunk_mph": mph(r["hidden_post"]) if has_h else None,
        "hidden_speed_80pct_interval_mph": [mph(r["hidden_post"] - Z80 * r["hidden_post_sd"]),
                                            mph(r["hidden_post"] + Z80 * r["hidden_post_sd"])] if has_h else None,
        "p_true_top_quintile": opt(r["p_top"]), "top_quintile": r["top_quintile"],
        "late_round_or_undrafted": r["late"], "top_quintile_and_late": r["top_late"],
        "game_top_speed_yds": r["game_top_speed"], "game_top_speed_mph": mph(r["game_top_speed"]) if r["game_top_speed"] is not None else None,
        "game_plays": r["game_plays"], "cushion_oe": r["cushion_oe"], "cushion_routes": r["cushion_oe_n"],
        "yprr_oe": r["yprr_oe"], "routes": r["yprr_oe_n"]})
(RES / "E007_leaderboard.json").write_text(json.dumps(clean({
    "experiment": "E007",
    "definition": "hidden speed = residual of c40_v_20_40 (tracked flying 20, official 20->40 yd lines) on the "
    "official forty within Combine WRs, both centered within draft year: speed beyond the official 40 time. "
    "rank = by observed hidden speed; rank_shrunk = by the measurement-error-shrunk value. top_quintile = observed "
    "hidden speed >= its 80th percentile; p_true_top_quintile = posterior probability that the true value is in the "
    "top quintile (attempt-to-attempt error only). Game targets are the E007 P1-P3 targets where available "
    "(null = not enough game data).",
    "notes": [
        "Descriptive only. E007 found no game outcome that hidden speed predicts beyond B2 (cushion, yards per route, "
        "targets, depth of target, deep-route share, career snaps: null or negative), so neither a high rank nor "
        "top_quintile_and_late means 'undervalued'.",
        "Draft position also reflects medical, character, off-field and other non-football factors. "
        "late_round_or_undrafted is not evidence that a player's athleticism was missed.",
        "Hidden speed is noisy: reliability about 0.72; built from attempt 1 alone vs attempt 2 alone it correlates "
        "about 0.70, and about half of the top quintile changes. Players with one QC-passing attempt are the least "
        "certain. Error in the official forty is not included.",
        "The tracked component beyond the 40 is not corroborated by the official electronic 10-yd split, and the "
        "20-yd split that clubs receive is not in the data."],
    "meta": leaderboard_meta, "players": lb_rows}), indent=1))

# ================================================================== 11. Figures
log("Figures")
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK, "axes.facecolor": SURFACE,
                     "figure.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})

# (a) added-variable plots
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, pid in zip(axes, PRIMARY):
    tgt = PRIMARY[pid]
    pr = linkage.partial_residuals(wr, FEATURE, tgt)
    x, y = pr["x_res"].to_numpy(), pr["y_res"].to_numpy()
    drop_ids = {d["nfl_id"] for d in l3o[pid]["dropped"]}
    infl = np.isin(pr["nfl_id"].to_numpy(), list(drop_ids))
    ax.scatter(x[~infl], y[~infl], s=28, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    ax.scatter(x[infl], y[infl], s=40, facecolor="none", edgecolor=ORANGE, linewidth=1.8, zorder=4,
               label="3 most influential (leave-3-out)")
    b = float(np.sum(x * y) / np.sum(x**2))
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, b * xs, color=INK2, lw=2)
    r0 = primary[pid][0]
    ci = r0["partial_r_ci95"]
    ax.set_title(f"{pid}: {PRIMARY_LABEL[pid]}", fontsize=10, loc="left")
    ax.text(0.02, 0.97, f"n = {r0['n']}\npartial r = {r0['partial_r']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]\n"
            f"one-sided FL p = {r0['fl_p']:.4f}, BH q = {bh_q[pid]:.3f}", transform=ax.transAxes, va="top",
            fontsize=8, color=INK)
    ax.set_xlabel("flying-20 speed beyond B2 (yd/s)")
    ax.set_ylabel(f"{tgt} beyond B2")
axes[0].legend(loc="lower right", fontsize=7, frameon=False)
fig.suptitle("E007 added-variable plots: Combine WR flying-20 speed vs game targets, both residualized on B2 "
             "(Combine times, draft capital, draft year)", fontsize=10, color=INK, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(FIG / "E007_added_variable.png", dpi=110)
plt.close(fig)

# (b) concept: two WRs with nearly the same official forty, different hidden speed
cand = board.filter(pl.col("hidden").is_not_null() & pl.col("game_top_speed").is_not_null())
cv = cand.select("nfl_id", "display_name", "forty", "hidden", FEATURE, "draft_year").to_dicts()
pair, tol_used = None, None
for tol in (0.01, 0.02, 0.03):
    best = -1.0
    for i in range(len(cv)):
        for j in range(i + 1, len(cv)):
            if abs(cv[i]["forty"] - cv[j]["forty"]) <= tol + 1e-9:
                dh = abs(cv[i]["hidden"] - cv[j]["hidden"])
                if dh > best:
                    best, pair = dh, (cv[i], cv[j])
    if pair:
        tol_used = tol
        break
fast, slow = sorted(pair, key=lambda p: -p["hidden"])
# Post hoc (skeptic review): every matched pair (|delta forty| <= tol_used), not just the illustrated one. For each
# pair, the WR with more hidden speed vs the other: share with the higher game q95, and the game gap per hidden gap.
gaps = []
for i in range(len(cv)):
    for j in range(i + 1, len(cv)):
        if abs(cv[i]["forty"] - cv[j]["forty"]) <= tol_used + 1e-9:
            hi_, lo_ = (cv[i], cv[j]) if cv[i]["hidden"] >= cv[j]["hidden"] else (cv[j], cv[i])
            gaps.append((hi_["hidden"] - lo_["hidden"], hi_["nfl_id"], lo_["nfl_id"]))
gq = dict(zip(cand["nfl_id"].to_list(), cand["game_top_speed"].to_list()))
gh = np.array([g_[0] for g_ in gaps]) * MPH
gg = np.array([gq[g_[1]] - gq[g_[2]] for g_ in gaps]) * MPH
matched_pairs = {"n_pairs": len(gaps), "share_more_hidden_speed_has_higher_game_q95": float(np.mean(gg > 0)),
                 "mean_hidden_gap_mph": float(gh.mean()), "mean_game_gap_mph": float(gg.mean()),
                 "game_gap_per_hidden_gap_through_origin": float(np.sum(gh * gg) / np.sum(gh**2)),
                 "illustrated_pair_game_gap_mph": (gq[fast["nfl_id"]] - gq[slow["nfl_id"]]) * MPH,
                 "note": "post hoc; same WRs and target as P1, so not independent of the P1 test"}
concept = {"tolerance_s": tol_used, "fast": fast, "slow": slow, "post_hoc_all_matched_pairs": matched_pairs}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
qc = attempts.filter(pl.col("qc_ok")).select("traj")
for p, col in ((fast, ORANGE), (slow, BLUE)):
    fr = frames40.filter(pl.col("nfl_id") == p["nfl_id"]).join(qc, on="traj", how="semi").sort("traj", "time")
    for k, ((traj,), g) in enumerate(fr.group_by(["traj"], maintain_order=True)):
        ax1.plot(g["x"].to_numpy() - speed.START_X, g["speed"].to_numpy() * MPH, color=col, lw=2,
                 alpha=1.0 if k == 0 else 0.55,
                 label=f"{p['display_name']} ({p['draft_year']}): 40 = {p['forty']:.2f} s, "
                       f"flying 20 = {p[FEATURE] * MPH:.1f} mph" if k == 0 else None)
    pm = play_max.filter(pl.col("nfl_id") == p["nfl_id"])["smax"].sort().to_numpy() * MPH
    ax2.plot(pm, np.arange(1, len(pm) + 1) / len(pm), color=col, lw=2,
             label=f"{p['display_name']}: {len(pm)} plays, q95 = {np.quantile(pm, 0.95, method='nearest'):.1f} mph")
    ax2.axvline(np.quantile(pm, 0.95, method="nearest"), color=col, lw=1, ls="-", alpha=0.6)
ax1.axvspan(20, 40, color=GRID, alpha=0.5, lw=0, zorder=0)
ax1.set_xlim(0, 42)
ax1.set_ylim(0, 29)
ax1.text(30, 1.0, "flying 20 (20 -> 40 yd)", ha="center", color=INK2, fontsize=8)
ax1.set_xlabel("yards past the start line")
ax1.set_ylabel("speed (mph)")
ax1.set_title("Tracked Combine 40 (each QC-passing attempt)", loc="left", fontsize=10)
ax1.legend(loc="upper left", fontsize=7.5, frameon=False)
ax2.set_xlabel("max speed on a regular-season play (mph)")
ax2.set_ylabel("share of plays at or below")
ax2.set_title("Game play max speed (ECDF; vertical line = q95, the P1 target)", loc="left", fontsize=10)
ax2.legend(loc="upper left", fontsize=7.5, frameon=False)
fig.suptitle(f"Same official 40 (within {tol_used:.2f} s), different flying 20: hidden speed "
             f"{fast['hidden'] * MPH:+.2f} vs {slow['hidden'] * MPH:+.2f} mph", fontsize=10, color=INK, x=0.01, ha="left")
fig.text(0.01, 0.01, f"Pair chosen by a Combine-only rule (largest hidden-speed gap among WRs with 40 times within "
         f"{tol_used:.2f} s and >= 50 tracked plays).\nAcross all {matched_pairs['n_pairs']} such pairs, the WR with "
         f"more hidden speed had the higher game q95 in {matched_pairs['share_more_hidden_speed_has_higher_game_q95']:.0%}"
         f" ({matched_pairs['game_gap_per_hidden_gap_through_origin']:.2f} mph of game gap per mph of hidden gap; "
         f"same WRs and target as P1, not independent of it).", fontsize=7.5, color=INK2, ha="left", va="bottom")
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(FIG / "E007_concept.png", dpi=110)
plt.close(fig)

# (c) leaderboard figure (descriptive: no outcome supports an 'undervalued' reading)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.0), gridspec_kw={"width_ratios": [1.1, 1]})
bf = board.filter(pl.col("hidden").is_not_null())
fx = bf.select(year_centered("forty") + pl.col("forty").mean()).to_series().to_numpy()
fy = bf.select((year_centered(FEATURE) + pl.col(FEATURE).mean()) * MPH).to_series().to_numpy()
tlate, top = bf["top_late"].to_numpy(), bf["top_quintile"].to_numpy()
ax1.scatter(fx[~top], fy[~top], s=26, color=MUTED, edgecolor=SURFACE, linewidth=1, label="other WRs", zorder=3)
ax1.scatter(fx[top & ~tlate], fy[top & ~tlate], s=34, color=BLUE, edgecolor=SURFACE, linewidth=1.2,
            label="top-quintile hidden speed, drafted rounds 1-3", zorder=4)
ax1.scatter(fx[tlate], fy[tlate], s=40, color=ORANGE, edgecolor=SURFACE, linewidth=1.2,
            label="top-quintile hidden speed, round 4+ / undrafted", zorder=5)
xs = np.linspace(fx.min(), fx.max(), 50)
ax1.plot(xs, (np.mean(fy) / MPH + slope * (xs - np.mean(fx))) * MPH, color=INK2, lw=2)
placed = []  # label below-right instead of above-right when a labelled point sits just above
for i in np.argsort(-fy[tlate]):
    xv, yv, nm = fx[tlate][i], fy[tlate][i], bf["display_name"].to_numpy()[tlate][i]
    near = any(abs(xv - px) < 0.03 and abs(yv - py) < 0.15 for px, py in placed)
    ax1.annotate(nm, (xv, yv), xytext=(4, -10 if near else 3), textcoords="offset points", fontsize=7, color=INK)
    placed.append((xv, yv))
ax1.set_xlabel("official 40 time (s, draft-year adjusted)")
ax1.set_ylabel("tracked flying 20 (mph, draft-year adjusted)")
ax1.set_title("Hidden speed = distance above the line (beyond the official 40)", loc="left", fontsize=10)
ax1.legend(loc="upper right", fontsize=7.5, frameon=False)
tq = bf.filter(pl.col("top_quintile")).sort("hidden")
lab = [f"{n} ({y}, {'UDFA' if u else f'R{int(rd)} #{int(pk)}'}{', 1 att.' if na == 1 else ''})"
       for n, y, u, rd, pk, na in zip(tq["display_name"], tq["draft_year"], tq["undrafted"],
                                      tq["draft_round"].fill_null(0), tq["draft_overall_pick"].fill_null(0),
                                      tq["c40_n_attempts"])]
yy = np.arange(tq.height)
cols = [ORANGE if f else BLUE for f in tq["top_late"].to_list()]
post, psd = tq["hidden_post"].to_numpy() * MPH, tq["hidden_post_sd"].to_numpy() * MPH
ax2.hlines(yy, post - Z80 * psd, post + Z80 * psd, colors=cols, lw=2.2, zorder=3)
ax2.scatter(post, yy, s=36, c=cols, edgecolor=SURFACE, linewidth=1.2, zorder=4, label="shrunk estimate, 80% interval")
ax2.scatter(tq["hidden"].to_numpy() * MPH, yy, s=30, facecolor="none", edgecolor=INK2, linewidth=1.1, zorder=5,
            label="observed")
ax2.axvline(thr_true * MPH, color=INK2, lw=1, ls="--")
ax2.set_yticks(yy, lab, fontsize=7.5, color=INK2)
ax2.grid(axis="y", visible=False)
ax2r = ax2.twinx()
ax2r.set_ylim(ax2.get_ylim())
ax2r.set_yticks(yy, [f"{pt:.2f}" for pt in tq["p_top"].to_list()], fontsize=7.5, color=INK2)
ax2r.set_ylabel("P(true hidden speed in the top quintile)", fontsize=8, color=INK2)
ax2r.grid(False)
ax2r.spines["right"].set_visible(True)
ax2.set_xlabel("hidden speed (mph beyond the official 40; dashed = true top-quintile line)")
ax2.set_title("Observed top quintile: uncertainty from attempt-to-attempt noise", loc="left", fontsize=10)
ax2.legend(loc="lower right", fontsize=7, frameon=False)
fig.suptitle("E007 hidden-speed leaderboard (descriptive): Combine WRs with a tracked 40, 2023-25", fontsize=10,
             color=INK, x=0.01, ha="left")
fig.text(0.01, 0.01, "Not an 'undervalued' list: E007 found no game outcome that hidden speed predicts beyond Combine "
         "times and draft capital.\nDraft position also reflects medical, character and other non-football factors. "
         "Orange = drafted after round 3 or undrafted (descriptive). 1 att. = one QC-passing 40.",
         fontsize=7.5, color=INK2, ha="left", va="bottom")
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(FIG / "E007_leaderboard.png", dpi=110)
plt.close(fig)

# ================================================================== 12. Results JSON
log("Writing E007.json")
out = {
    "experiment": PREREG["experiment"], "date": "2026-10-14",
    "prereg": "experiments/results/E007_prereg.json (incl. amendments)", "script": "experiments/E007_hidden_speed.py",
    "harness": {"primary_and_robustness": FULL | {"seeds": list(SEEDS), "expected_sign": 1},
                "exploratory": EXPL | {"expected_sign": 1}},
    "implementation_choices": IMPLEMENTATION_CHOICES,
    "feature": {"name": FEATURE, "definition": "tracked 40: 20 / (t40 - t20) from the official 20- and 40-yd lines, "
                "mean over QC-passing attempts (bdb27.speed)", **feature_summary,
                "reliability_beyond_controls": feature_beyond},
    "attrition": attrition,
    "targets": {"P1": p1_diag, "expectation_models": expectation_models, "eb_diagnostics": target_diag,
                "eb_calibration_context": eb_context, "TE_pooled_eb": te_diag},
    "primary": {pid: {"target": PRIMARY[pid], "seeds": primary[pid], "bh_q": float(bh_q[pid]), "go": go[pid],
                      "disattenuated": disattenuated[pid]} for pid in PRIMARY},
    "robustness": {"leave_3_out": l3o, "without_2025": no2025, "feature_variants": feature_variants,
                   "target_variants": target_variants, "pooled_WR_TE": pooled},
    "exploratory": {"n_tests": len(exploratory), "results": exploratory, "diagnostics": exploratory_diag},
    "post_hoc": post_hoc,
    "post_hoc_review": post_hoc_review,
    "bh_all_tests": {"n_tests": len(all_tests), "n_tests_without_nested_reruns": len(not_nested),
                     "family": "every prespecified E007 test (primaries at seed 0, robustness except seed replicates, "
                               "exploratory); post hoc tests are in no family",
                     "note_S0": "target_variant/P1_S0 re-observes the E006 post hoc vmax/flying-split -> S0 "
                                "association on the same WRs (see post_hoc_review.S0_provenance); its q is not "
                                "confirmatory evidence.",
                     "tests": bh_all},
    "bh_primary_plus_exploratory": bh_primary_plus_exploratory,
    "leaderboard": leaderboard_meta | {"file": "experiments/results/E007_leaderboard.json"},
    "concept_pair": concept,
    "figures": ["experiments/figures/E007_added_variable.png", "experiments/figures/E007_concept.png",
                "experiments/figures/E007_leaderboard.png"],
    "runtime_s": time.time() - t0,
}
(RES / "E007.json").write_text(json.dumps(clean(out), indent=1))
df.join(ph_player, on="nfl_id", how="left").write_parquet(io.PROCESSED / "e007_players.parquet")
routes_al.write_parquet(io.PROCESSED / "e007_routes.parquet")
log("done")
