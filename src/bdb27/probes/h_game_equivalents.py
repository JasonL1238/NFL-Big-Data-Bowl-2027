"""Probe H (E012): Combine Game Equivalents (CGE). How many NFL games of a player's own tracking is his Combine
worth, trait by trait, and how should the two be blended in season? Prereg: experiments/results/E012_prereg.json.

Model, per trait T at one position: game-level value y_ig = theta_i + e_ig, between-player variance tau2 (within
position, after draft-year centering) and between-game within-player variance sigma2; lambda = sigma2 / tau2 is the
number of games of noise per unit of signal. An information set C forecasts theta with true-score R2_C
(cross-validated R2 on the player means divided by their reliability, clipped to [0, R2_MAX]). Then
  CGE_C = lambda R2_C / (1 - R2_C)
is the number of games of the player's own data that forecast theta as well as C does: k games used optimally
(shrunk to the mean) explain k / (k + lambda) of tau2, and solving k / (k + lambda) = R2_C gives CGE_C. The
in-season rule is the normal-normal posterior mean with prior N(mu_C, (1 - R2_C) tau2) and ybar_k ~ N(theta, sigma2/k):
  theta_hat = mu_C + w_k (ybar_k - mu_C),  w_k = k / (k + lambda / (1 - R2_C)) = k / (k + lambda + CGE_C).
The prereg wrote lambda (1 - R2_C) in w_k; that is not the posterior weight (E012 amendment A1). The literal formula
is kept as a sensitivity (blend_weight(..., literal=True)).

Contents:
  - trait specs (TRAITS) and information sets (info_sets)
  - play-level builders for the 12 traits, each taking an optional `ref` row filter: every data-derived constant
    (centering means, base rates, winsorization quantiles) is computed on the ref rows only and applied to all rows,
    which is how validation 2 keeps 2025 data out of every constant
  - prospective refits of the per-play expectation models (separation, yards per route, OL pressure)
  - Combine drill-z composites with constants from reference players
  - estimation: stratified variance components, cross-validated R2, CGE, the player bootstrap
  - validation 1 (internal crossover) and validation 2 (prospective fit, forecasts, sign-flip test, BH)
  - post hoc (review response): simulation at a trait's own design, simulation-calibrated (Neyman) bounds and
    reference-relative R2
Implementation choices are frozen in E012.json (implementation_choices).
"""

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, RepeatedKFold
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from threadpoolctl import threadpool_limits

from bdb27 import linkage

MIN_GAMES = 3
N_SPLITS = 5
N_REPEATS = 20
R2_MAX = 0.99
Z80 = 1.2815515655446004  # standard normal 0.9 quantile (two-sided 80% interval)
MAX_THREADS = 3
TRACK40 = ["c40_smax", "c40_v_20_40", "c40_vmax", "c40_tau", "c40_a0"]
SETS = ("C0", "C1", "C2", "C3")

TRAITS = {
    "T1": dict(pos="WR", name="Game top speed", short="Top speed", unit="yd/s", min_plays=10, agg="q90", matched=None),
    "T2": dict(pos="WR", name="OUT/IN turn speed retention", short="Turn retention", unit="ratio", min_plays=2,
               agg="mean", matched="c_turn_retention"),
    "T3": dict(pos="WR", name="HITCH peak braking", short="Hitch braking", unit="yd/s²", min_plays=2, agg="mean",
               matched="c_stop_peak_decel"),
    "T4": dict(pos="WR", name="Separation over expected (HITCH/OUT/IN)", short="Separation OE", unit="yd",
               min_plays=3, agg="mean", matched="c_turn_retention"),
    "T5": dict(pos="WR", name="Yards per route over expected", short="Yards/route OE", unit="yd", min_plays=10,
               agg="mean", matched=None),
    "T6": dict(pos="DL", name="First-second rush distance", short="Rush burst", unit="yd", min_plays=5, agg="mean",
               matched="c40_d10"),
    "T7": dict(pos="DL", name="Get-off (NGS)", short="Get-off", unit="s", min_plays=5, agg="mean",
               matched="c40_d10"),
    "T8": dict(pos="DL", name="Pressure rate over expected", short="Pressure OE", unit="per snap", min_plays=10,
               agg="mean", matched="c40_d10"),
    "T9": dict(pos="OL", name="Pass-set acceleration (0-0.6 s)", short="Set accel.", unit="yd/s²", min_plays=10,
               agg="mean", matched="cmb_rev_accel_mirror"),
    "T10": dict(pos="OL", name="Pressure allowed over expected", short="Pressure allowed OE", unit="per snap",
                min_plays=15, agg="mean", matched="cmb_rev_accel_mirror"),
    "T11": dict(pos="DB", name="Pre-throw peak retreat speed", short="Retreat speed", unit="yd/s", min_plays=3,
                agg="mean", matched="cmb_backpedal_speed"),
    "T12": dict(pos="DB", name="Game top speed", short="Top speed", unit="yd/s", min_plays=10, agg="q90",
                matched=None),
}


def info_sets(matched: str | None) -> dict[str, list[str] | None]:
    """Columns added on top of the draft-year dummies; None = C0 (draft-year means)."""
    c1 = list(linkage.COMBINE_CONTROLS)
    c2 = c1 + list(linkage.DRAFT_CONTROLS)
    return {"C0": None, "C1": c1, "C2": c2, "C3": c2 + TRACK40 + ([matched] if matched else [])}


# =====================================================================================================
# Play-level helpers (constants from `ref` rows only)
# =====================================================================================================

def _ref(df: pl.DataFrame, ref: pl.Expr | None) -> pl.DataFrame:
    return df if ref is None else df.filter(ref)


def center(df: pl.DataFrame, value: str, by: list[str], ref: pl.Expr | None = None, add_overall: bool = False,
           name: str = "value") -> pl.DataFrame:
    """Add `name` = value - mean(value | by) [+ overall mean], both means over the finite ref rows. Rows whose
    group never appears in the ref rows get null."""
    r = _ref(df, ref).filter(pl.col(value).cast(pl.Float64).is_finite())
    m = r.group_by(by).agg(_m=pl.col(value).cast(pl.Float64).mean())
    out = df.join(m, on=by, how="left", maintain_order="left").with_columns(
        (pl.col(value).cast(pl.Float64) - pl.col("_m")).alias(name)).drop("_m")
    if add_overall:
        out = out.with_columns(pl.col(name) + float(r[value].cast(pl.Float64).mean()))
    return out


def winsorize(df: pl.DataFrame, value: str, q: float, ref: pl.Expr | None = None) -> pl.DataFrame:
    """Clip `value` at the q / 1 - q quantiles of the finite ref rows (polars default interpolation)."""
    r = _ref(df, ref).filter(pl.col(value).is_finite())[value]
    return df.with_columns(pl.col(value).clip(r.quantile(q), r.quantile(1 - q)))


def game_table(plays: pl.DataFrame, min_n: int, agg: str = "mean") -> pl.DataFrame:
    """plays: nfl_id, game_id, value -> one row per qualifying player-game (>= min_n finite play values):
    nfl_id, game_id, value (mean, or 'q90' = 0.9 quantile, polars default 'nearest'), n_plays."""
    a = pl.col("value").mean() if agg == "mean" else pl.col("value").quantile(0.9)
    keys = ["nfl_id", "game_id"] + (["season"] if "season" in plays.columns else [])
    return (plays.filter(pl.col("value").is_finite()).group_by(keys)
            .agg(value=a, n_plays=pl.len()).filter(pl.col("n_plays") >= min_n).sort("nfl_id", "game_id"))


# =====================================================================================================
# Trait builders: play-level `value` (one row per player-play or event) for the trait's position
# =====================================================================================================

def t_play_speed(pm: pl.DataFrame) -> pl.DataFrame:
    """T1/T12: play max speed (f_drill_intensity.play_max_speeds rows; plays > 12 yd/s already dropped)."""
    return pm.select("nfl_id", "game_id", "season", value=pl.col("smax"))


def t_route_break(gev: pl.DataFrame, routes: list[str], col: str, ref: pl.Expr | None = None) -> pl.DataFrame:
    """T2/T3: found break events of `routes`, `col` minus its route-type mean over all (WR+TE) ref events
    (E002 game_trait centering). Returns WR and TE rows; the caller keeps the trait's position."""
    e = gev.filter(pl.col("route_ran").is_in(routes) & pl.col("found") & pl.col(col).is_finite())
    return center(e, col, ["route_ran"], ref).select("nfl_id", "game_id", "season", "position", "value")


def t_dl(gp: pl.DataFrame, col: str, ref: pl.Expr | None = None) -> pl.DataFrame:
    """T6/T7: pass-rush snaps lined up EDGE / INTERIOR_LINE; `col` minus its alignment mean plus the overall mean
    (b_dl_burst.center_by_alignment), over all ref snaps with a finite value."""
    p = gp.filter(pl.col("rush") & pl.col("lined_up_position").is_in(["EDGE", "INTERIOR_LINE"])
                  & pl.col(col).cast(pl.Float64).is_finite())
    return center(p, col, ["lined_up_position"], ref, add_overall=True).select("nfl_id", "game_id", "season", "value")


def t_dl_pressure(gp: pl.DataFrame, ref: pl.Expr | None = None) -> pl.DataFrame:
    """T8: pressure (time_to_pressure non-null) minus the alignment base rate over ref pass-rush snaps."""
    p = gp.filter(pl.col("rush") & pl.col("lined_up_position").is_in(["EDGE", "INTERIOR_LINE"]))
    p = p.with_columns(pl.col("pressure").cast(pl.Float64))
    return center(p, "pressure", ["lined_up_position"], ref).select("nfl_id", "game_id", "season", "value")


def t_ol_set(set_m: pl.DataFrame, ref: pl.Expr | None = None) -> pl.DataFrame:
    """T9: set_accel minus its T/G/C mean over ref qualifying sets (c_ol_lateral.pass_set_metrics' set_accel_c)."""
    s = set_m.filter(pl.col("set_accel").is_finite())
    return center(s, "set_accel", ["lined_up_position"], ref).select("nfl_id", "game_id", "season", "value")


def t_db_retreat(bp: pl.DataFrame, ref: pl.Expr | None = None, wins: float = 0.01) -> pl.DataFrame:
    """T11: d_db_transition.game_backpedal_events with ref-only constants: bp_peak winsorized at the ref
    quantiles, minus the ref mean of its alignment x coverage-assignment cell."""
    d = bp.filter(pl.col("bp_peak").is_finite()).with_columns(cov=pl.col("coverage_assignment").fill_null("NA"))
    d = winsorize(d, "bp_peak", wins, ref)
    return center(d, "bp_peak", ["lined_up_position", "cov"], ref).select("nfl_id", "game_id", "season", "value")


# =====================================================================================================
# Prospective refits of per-play expectation models (fit on train rows only, applied to test rows)
# =====================================================================================================

def codes_from_ref(d: pl.DataFrame, cols: list[str], is_ref: np.ndarray) -> np.ndarray:
    """Integer codes over the sorted labels present in the ref rows; null or unseen labels -> NaN (missing)."""
    out = []
    for c in cols:
        s = d[c].cast(pl.String)
        labels = sorted(s.filter(pl.Series(is_ref)).drop_nulls().unique().to_list())
        out.append(s.replace_strict(labels, list(range(len(labels))), default=None, return_dtype=pl.Float64)
                   .to_numpy())
    return np.column_stack(out).astype(float)


def _hgb_fit_predict(X, y, is_tr, is_te, groups, n_splits, params) -> np.ndarray:
    """Out-of-fold predictions for train rows (GroupKFold by `groups` within them) and predictions for test rows
    from a model fit on all train rows."""
    pred = np.full(len(y), np.nan)
    tr_idx, te_idx = np.flatnonzero(is_tr), np.flatnonzero(is_te)
    with threadpool_limits(MAX_THREADS):
        for a, b in GroupKFold(n_splits=n_splits).split(X[tr_idx], y[tr_idx], groups[tr_idx]):
            m = HistGradientBoostingRegressor(**params).fit(X[tr_idx[a]], y[tr_idx[a]])
            pred[tr_idx[b]] = m.predict(X[tr_idx[b]])
        if len(te_idx):
            m = HistGradientBoostingRegressor(**params).fit(X[tr_idx], y[tr_idx])
            pred[te_idx] = m.predict(X[te_idx])
    return pred


SOE_CAT = ["route_ran", "team_coverage_man_zone", "team_coverage_type", "receiver_alignment", "lined_up_position",
           "offense_formation", "in_motion_at_ball_snap"]
SOE_NUM = ["dropback_duration", "down", "yards_to_go", "yards_to_goal", "quarter"]


def soe_prospective(plays: pl.DataFrame, train: pl.Expr, test: pl.Expr, n_splits: int = 5,
                    seed: int = 0) -> pl.DataFrame:
    """a_route_breaks.separation_over_expected refit for validation 2: the same rows, context and HGB settings;
    train rows get predictions cross-fitted by player within the train rows, test rows from the model fitted on all
    train rows. Category codes come from train labels. Returns the train and test rows with sep_expected, soe."""
    d = (plays.filter((train | test) & pl.col("separation_at_pass_forward").is_not_null()
                      & (pl.col("route_ran") != "SCREEN"))
         .with_columns(yards_to_goal=pl.when(pl.col("yardline_side") == pl.col("possession_team"))
                       .then(100 - pl.col("yardline_number")).otherwise(pl.col("yardline_number")).cast(pl.Float64),
                       _tr=train))
    is_tr = d["_tr"].to_numpy()
    X = np.column_stack([codes_from_ref(d.with_columns(pl.col(SOE_CAT).cast(pl.String).fill_null("NA")), SOE_CAT,
                                        is_tr)] + [d[c].cast(pl.Float64).to_numpy() for c in SOE_NUM])
    y = d["separation_at_pass_forward"].to_numpy().astype(float)
    params = dict(categorical_features=list(range(len(SOE_CAT))), max_iter=300, learning_rate=0.05,
                  min_samples_leaf=50, random_state=seed)
    pred = _hgb_fit_predict(X, y, is_tr, ~is_tr, d["nfl_id"].to_numpy(), n_splits, params)
    return d.drop("_tr").with_columns(sep_expected=pl.Series(pred), soe=pl.Series(y - pred))


def yprr_prospective(routes: pl.DataFrame, cat: list[str], num: list[str], train: pl.Expr, test: pl.Expr,
                     n_splits: int = 5, seed: int = 0, latest: str = "2024") -> pl.DataFrame:
    """targets_wr.cross_fit refit for validation 2 (rec_yards0 on the E007 context, HGB defaults): train rows
    cross-fitted by game within the train rows; test rows predicted by the model fitted on all train rows, with
    season -> `latest` and defense_team_season -> <team>_<latest> (the latest team-season known before the test
    season). Returns train and test rows with yprr_exp, yprr_oe."""
    d = routes.filter(train | test).with_columns(_tr=train)
    team = pl.col("defense_team_season").str.split("_").list.first()
    d = d.with_columns(
        season=pl.when(pl.col("_tr")).then(pl.col("season").cast(pl.String)).otherwise(pl.lit(latest)),
        defense_team_season=pl.when(pl.col("_tr")).then(pl.col("defense_team_season"))
        .otherwise(team + "_" + latest))
    is_tr = d["_tr"].to_numpy()
    X = np.column_stack([codes_from_ref(d, cat, is_tr)] + [d[c].cast(pl.Float64).to_numpy() for c in num])
    y = d["rec_yards0"].to_numpy().astype(float)
    params = dict(categorical_features=list(range(len(cat))), random_state=seed)
    pred = _hgb_fit_predict(X, y, is_tr, ~is_tr, d["game_id"].to_numpy(), n_splits, params)
    return d.drop("_tr").with_columns(yprr_exp=pl.Series(pred), yprr_oe=pl.Series(y - pred))


def ol_pressure_prospective(plays: pl.DataFrame, train: pl.Expr, test: pl.Expr, latest: str = "2024") -> pl.DataFrame:
    """c_ol_lateral.add_expectations refit for validation 2: the same design (dropback cubic spline with 6 knots,
    one-hot T/G/C, 2+ rushers, play action, season; logistic C = 1e4) fitted on train rows only; test rows are scored
    with season -> `latest`. Train rows keep in-sample expectations, as E004. Adds p_hat, pr_oe."""
    d = plays.filter(train | test).with_columns(_tr=train)
    d = d.with_columns(season=pl.when(pl.col("_tr")).then(pl.col("season").cast(pl.String)).otherwise(pl.lit(latest)))
    X = d.select("dropback_duration", "lined_up_position", "rush2", pl.col("pa").cast(pl.Int8), "season").to_numpy()
    is_tr = d["_tr"].to_numpy()
    ct = ColumnTransformer([("db", SplineTransformer(n_knots=6, degree=3), [0]),
                            ("cat", OneHotEncoder(drop="first"), [1, 2, 3, 4])])
    with threadpool_limits(MAX_THREADS):
        Xt = ct.fit(X[is_tr]).transform(X)
        lr = LogisticRegression(C=1e4, max_iter=5000).fit(Xt[is_tr], d["pressure"].to_numpy()[is_tr])
        p_hat = lr.predict_proba(Xt)[:, 1]
    return d.drop("_tr").with_columns(p_hat=pl.Series(p_hat)).with_columns(pr_oe=pl.col("pressure") - pl.col("p_hat"))


# =====================================================================================================
# Combine drill-z composites with constants from reference players (validation 2)
# =====================================================================================================

def route_composite(events: pl.DataFrame, value: str, family: str, ref_ids: list[int] | None = None,
                    min_drills: int = 2) -> pl.DataFrame:
    """a_route_breaks.composite with the within-drill mean/SD of the player x drill attempt-means taken over the
    ref players only (ref_ids None: all players, which reproduces combine_signatures). nfl_id, comp, n_drills
    for players with >= min_drills drills."""
    e = events.filter(pl.col("found") & pl.col(value).is_finite() & (pl.col("family") == family))
    pdv = e.group_by("nfl_id", "drill").agg(v=pl.col(value).mean())
    r = pdv if ref_ids is None else pdv.filter(pl.col("nfl_id").is_in(ref_ids))
    k = r.group_by("drill").agg(_m=pl.col("v").mean(), _s=pl.col("v").std())
    z = pdv.join(k, on="drill", how="inner").with_columns(z=(pl.col("v") - pl.col("_m")) / pl.col("_s"))
    return (z.group_by("nfl_id").agg(comp=pl.col("z").mean(), n_drills=pl.len())
            .filter(pl.col("n_drills") >= min_drills).sort("nfl_id"))


def backpedal_composite(backpedal: pl.DataFrame, drills: list[str], ref_ids: list[int] | None = None,
                        clip: float = 3.0) -> pl.DataFrame:
    """d_db_transition.player_composite(bp_peak) with the within-drill attempt mean/SD (ddof 1) from ref players:
    z clipped at +-clip, averaged within player x drill, then over drills. nfl_id, cmb_backpedal_speed."""
    b = backpedal.filter(pl.col("drill_name").is_in(drills) & pl.col("bp_peak").is_finite())
    r = b if ref_ids is None else b.filter(pl.col("nfl_id").is_in(ref_ids))
    k = r.group_by("drill_name").agg(_m=pl.col("bp_peak").mean(), _s=pl.col("bp_peak").std())
    z = b.join(k, on="drill_name", how="inner").with_columns(
        z=((pl.col("bp_peak") - pl.col("_m")) / pl.col("_s")).clip(-clip, clip))
    per_drill = z.group_by("nfl_id", "drill_name").agg(z=pl.col("z").mean())
    return per_drill.group_by("nfl_id").agg(cmb_backpedal_speed=pl.col("z").mean()).sort("nfl_id")


# =====================================================================================================
# Estimation
# =====================================================================================================

def variance_components(values: np.ndarray, players: np.ndarray, strata: np.ndarray | None = None) -> dict:
    """Stratified one-way random-effects ANOVA (each player in one stratum, e.g. his draft year):
    MSW = sum (y - m_i)^2 / (N - k); MSB = sum_i n_i (m_i - stratum game-weighted mean)^2 / (k - Y);
    n0 = (N - sum_y sum_{i in y} n_i^2 / N_y) / (k - Y); tau2 = (MSB - MSW) / n0; sigma2 = MSW.
    With one stratum this is reliability.icc1's ANOVA estimator. lambda = sigma2 / tau2 (inf if tau2 <= 0);
    rel = tau2 / (tau2 + mean_i sigma2 / n_i), the reliability of the player means (0 if tau2 <= 0)."""
    values = np.asarray(values, float)
    players = np.asarray(players)
    strata = np.zeros(len(values), int) if strata is None else np.asarray(strata)
    ok = np.isfinite(values)
    values, players, strata = values[ok], players[ok], strata[ok]
    labels, inv = np.unique(players, return_inverse=True)
    k, N = len(labels), len(values)
    n_i = np.bincount(inv).astype(float)
    m_i = np.bincount(inv, weights=values) / n_i
    s_i = np.zeros(k, dtype=strata.dtype)
    s_i[inv] = strata
    if np.any(strata != s_i[inv]):
        raise ValueError("a player appears in more than one stratum")
    ssw = float(np.sum((values - m_i[inv]) ** 2))
    ssb, sum_sq = 0.0, 0.0
    ys = np.unique(s_i)
    for s in ys:
        m = s_i == s
        Ny = n_i[m].sum()
        grand = np.sum(n_i[m] * m_i[m]) / Ny
        ssb += float(np.sum(n_i[m] * (m_i[m] - grand) ** 2))
        sum_sq += float(np.sum(n_i[m] ** 2) / Ny)
    dfb, dfw = k - len(ys), N - k
    out = {"n_players": int(k), "n_games": int(N), "n_strata": int(len(ys)), "mean_games": float(n_i.mean())}
    if dfb < 2 or dfw < 1:
        return out | {"tau2": np.nan, "sigma2": np.nan, "lambda": np.nan, "icc": np.nan, "rel": np.nan}
    msb, msw = ssb / dfb, ssw / dfw
    n0 = (N - sum_sq) / dfb
    tau2, sigma2 = (msb - msw) / n0, msw
    tot = tau2 + sigma2
    lam = sigma2 / tau2 if tau2 > 0 else np.inf
    rel = tau2 / (tau2 + float(np.mean(sigma2 / n_i))) if tau2 > 0 else 0.0
    return out | {"tau2": float(tau2), "sigma2": float(sigma2), "lambda": float(lam),
                  "icc": float(tau2 / tot) if tot > 0 else np.nan, "rel": float(rel), "msb": msb, "msw": msw,
                  "n0": float(n0)}


def within_year_var(y: np.ndarray, year: np.ndarray) -> float:
    """sum (y - player-weighted year mean)^2 / (n - number of years)."""
    dev = y.copy()
    for s in np.unique(year):
        m = year == s
        dev[m] = y[m] - y[m].mean()
    return float(np.sum(dev ** 2) / (len(y) - len(np.unique(year))))


def year_dummies(year: np.ndarray, levels: list) -> np.ndarray:
    """Indicator columns for levels[1:] (the earliest class is the reference)."""
    return np.column_stack([(year == lv).astype(float) for lv in levels[1:]]) if len(levels) > 1 \
        else np.zeros((len(year), 0))


def design(players: pl.DataFrame, cols: list[str] | None, levels: list,
           dummy_means: np.ndarray | None = None) -> np.ndarray | None:
    """[draft-year dummies | cols] as float (NaN = missing); None for C0. dummy_means: replace the dummies by these
    values (a class absent from training gets the training-average class level)."""
    if cols is None:
        return None
    D = year_dummies(players["draft_year"].to_numpy(), levels)
    if dummy_means is not None:
        D = np.tile(dummy_means, (players.height, 1))
    C = players.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy() if cols else np.zeros((players.height, 0))
    return np.hstack([D, C])


def year_mean_forecast(y_tr: np.ndarray, yr_tr: np.ndarray, yr_te: np.ndarray) -> np.ndarray:
    """C0: the training mean of y within the class (training overall mean if the class is absent)."""
    out = np.full(len(yr_te), y_tr.mean())
    for s in np.unique(yr_te):
        m = yr_tr == s
        if m.any():
            out[yr_te == s] = y_tr[m].mean()
    return out


def group_folds(groups: np.ndarray, n_repeats: int = N_REPEATS, seed: int = 0):
    """Yield (repeat, test mask) of RepeatedKFold(N_SPLITS) over the unique `groups`: all rows of a group (e.g. the
    copies of one player in a bootstrap resample) fall in the same fold."""
    ug, ginv = np.unique(np.asarray(groups), return_inverse=True)
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=n_repeats, random_state=seed)
    for f, (_, te_g) in enumerate(rkf.split(ug.reshape(-1, 1))):
        yield f // N_SPLITS, np.isin(ginv, te_g)


def cv_oof(X: dict, y: np.ndarray, year: np.ndarray, groups: np.ndarray | None = None,
           n_repeats: int = N_REPEATS, seed: int = 0) -> dict[str, np.ndarray]:
    """Out-of-fold predictions (n_repeats x n) per information set, folds from group_folds (default: one group per
    row). X[s] None -> C0."""
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    oof = {s: np.full((n_repeats, len(y)), np.nan) for s in X}
    for rep, te in group_folds(groups, n_repeats, seed):
        tr = ~te
        for s, Xs in X.items():
            if Xs is None:
                oof[s][rep, te] = year_mean_forecast(y[tr], year[tr], year[te])
            else:
                oof[s][rep, te] = linkage._model().fit(Xs[tr], y[tr]).predict(Xs[te])
    return oof


def true_r2(r2_cv: float, rel: float) -> float:
    """clip(R2_CV / rel, 0, R2_MAX); NaN if the reliability is not positive."""
    if not (np.isfinite(r2_cv) and np.isfinite(rel)) or rel <= 0:
        return np.nan
    return float(np.clip(r2_cv / rel, 0.0, R2_MAX))


def cge(lam: float, r2: float) -> float:
    """Combine Game Equivalents: lambda R2 / (1 - R2)."""
    if not (np.isfinite(lam) and np.isfinite(r2)):
        return np.nan
    return float(lam * r2 / (1 - r2))


def blend_weight(k, lam: float, r2: float, literal: bool = False):
    """Weight on the k-game mean (0 at k = 0). Posterior (E012 amendment A1): w_k = k / (k + lambda / (1 - R2))
    = k / (k + lambda + CGE), the normal-normal posterior weight under the prior N(mu_C, (1 - R2) tau2).
    literal=True: the prereg's written formula k / (k + lambda (1 - R2)), kept only as a labeled sensitivity."""
    k = np.asarray(k, float)
    return k / (k + lam * (1 - r2)) if literal else k / (k + lam / (1 - r2))


def blend(mu, ybar, k, lam: float, r2: float, literal: bool = False):
    """In-season rule: mu + w_k (ybar_k - mu); mu at k = 0."""
    w = blend_weight(k, lam, r2, literal)
    return np.where(np.asarray(k) > 0, np.asarray(mu) + w * (np.nan_to_num(np.asarray(ybar, float)) - np.asarray(mu)),
                    np.asarray(mu))


def posterior_var(k, tau2: float, sigma2: float, r2: float):
    """Posterior variance of theta after k games under the prior N(mu_C, (1 - R2) tau2)."""
    return 1.0 / (1.0 / ((1 - r2) * tau2) + np.asarray(k, float) / sigma2)


@dataclass
class TraitData:
    """One trait's analysis sample: players (sorted by nfl_id) and their qualifying games, as arrays."""
    nfl_id: np.ndarray
    year: np.ndarray
    X: dict
    g_player: np.ndarray  # index into the players
    g_value: np.ndarray
    levels: list


def trait_data(games: pl.DataFrame, players: pl.DataFrame, sets: dict, min_games: int = MIN_GAMES,
               levels: list | None = None) -> TraitData:
    """games: nfl_id, game_id, value (qualifying games); players: nfl_id, draft_year + set columns.
    Keeps players with >= min_games games and a row in `players`."""
    n = games.group_by("nfl_id").len().filter(pl.col("len") >= min_games).select("nfl_id")
    p = players.join(n, on="nfl_id", how="semi").sort("nfl_id")
    g = games.join(p.select("nfl_id"), on="nfl_id", how="semi").sort("nfl_id", "game_id")
    ids = p["nfl_id"].to_numpy()
    levels = sorted(p["draft_year"].unique().to_list()) if levels is None else levels
    return TraitData(nfl_id=ids, year=p["draft_year"].to_numpy(),
                     X={s: design(p, c, levels) for s, c in sets.items()},
                     g_player=np.searchsorted(ids, g["nfl_id"].to_numpy()), g_value=g["value"].to_numpy(),
                     levels=levels)


def estimate(td: TraitData, n_repeats: int = N_REPEATS, seed: int = 0, groups: np.ndarray | None = None,
             keep_oof: bool = False) -> dict:
    """Variance components (draft-year strata), reliability, CV R2 per set, true-score R2, CGE, Delta CGE."""
    n = len(td.nfl_id)
    n_i = np.bincount(td.g_player, minlength=n).astype(float)
    y = np.bincount(td.g_player, weights=td.g_value, minlength=n) / n_i
    vc = variance_components(td.g_value, td.g_player, td.year[td.g_player])
    var_w = within_year_var(y, td.year)
    oof = cv_oof(td.X, y, td.year, groups, n_repeats, seed)
    out = {"n": int(n), **{k: vc[k] for k in ("tau2", "sigma2", "lambda", "rel", "icc", "mean_games", "n_games")},
           "var_w": var_w, "r2_cv": {}, "r2": {}, "cge": {}}
    for s, P in oof.items():
        r2cv = float(np.mean(1 - np.mean((y[None, :] - P) ** 2, axis=1) / var_w))
        out["r2_cv"][s] = r2cv
        out["r2"][s] = true_r2(r2cv, vc["rel"])
        out["cge"][s] = cge(vc["lambda"], out["r2"][s])
    c = out["cge"]
    out["delta"] = {"C3-C2": _diff(c, "C3", "C2"), "C2-C1": _diff(c, "C2", "C1"), "C1-C0": _diff(c, "C1", "C0")}
    if keep_oof:
        out["y"], out["oof"], out["oof_mean"] = y, oof, {s: P.mean(axis=0) for s, P in oof.items()}
    return out


def _diff(c: dict, a: str, b: str) -> float:
    return float(c[a] - c[b]) if a in c and b in c and np.isfinite(c[a]) and np.isfinite(c[b]) else np.nan


def resample(td: TraitData, ids: np.ndarray) -> TraitData:
    """Bootstrap copy: player j of the resample is original player ids[j], with all his games."""
    order = np.argsort(td.g_player, kind="stable")
    gp, gv = td.g_player[order], td.g_value[order]
    starts = np.searchsorted(gp, np.arange(len(td.nfl_id)))
    ends = np.searchsorted(gp, np.arange(len(td.nfl_id)), side="right")
    counts = ends[ids] - starts[ids]
    vals = np.concatenate([gv[starts[p]:ends[p]] for p in ids])
    return TraitData(nfl_id=np.arange(len(ids)), year=td.year[ids],
                     X={s: (None if Xs is None else Xs[ids]) for s, Xs in td.X.items()},
                     g_player=np.repeat(np.arange(len(ids)), counts), g_value=vals, levels=td.levels)


def _boot_chunk(td: TraitData, idx: np.ndarray, b0: int, n_repeats: int) -> list[dict]:
    out = []
    with threadpool_limits(1):
        for j, ids in enumerate(idx):
            e = estimate(resample(td, ids), n_repeats=n_repeats, seed=b0 + j, groups=ids)
            out.append({"lambda": e["lambda"], "tau2": e["tau2"], "sigma2": e["sigma2"], "rel": e["rel"],
                        **{f"r2_{s}": v for s, v in e["r2"].items()}, **{f"cge_{s}": v for s, v in e["cge"].items()},
                        **{f"delta_{k}": v for k, v in e["delta"].items()}})
    return out


def bootstrap(td: TraitData, n_boot: int, seed: int, n_jobs: int = 3, n_repeats: int = N_REPEATS) -> pl.DataFrame:
    """Player bootstrap of the whole estimate() pipeline; one row per resample."""
    from joblib import Parallel, delayed
    idx = np.random.default_rng(seed).integers(0, len(td.nfl_id), (n_boot, len(td.nfl_id)))
    chunks = np.array_split(np.arange(n_boot), max(1, n_jobs * 4))
    res = Parallel(n_jobs=n_jobs)(delayed(_boot_chunk)(td, idx[c], int(c[0]), n_repeats) for c in chunks if len(c))
    rows = [r for chunk in res for r in chunk]
    return pl.DataFrame(rows).with_row_index("b")


def bootstrap_fixed_forecasts(td: TraitData, est: dict, n_boot: int, seed: int) -> pl.DataFrame:
    """Player bootstrap with the out-of-fold forecasts held fixed (E012 post hoc PH5). Same player draws as
    bootstrap(td, n_boot, seed). Per resample: variance components and reliability from the resampled players' games,
    Var_w from their means, R2_CV = mean over CV repeats of 1 - mean (y - oof)^2 / Var_w with each player's own
    full-sample out-of-fold forecasts, then true-score R2, CGE and Delta CGE. No model is refitted, so the
    learning-curve bias of refitting on ~63% distinct players is absent; refit variability is not included either.
    `est` = estimate(td, keep_oof=True)."""
    n = len(td.nfl_id)
    idx = np.random.default_rng(seed).integers(0, n, (n_boot, n))
    y, oof = est["y"], est["oof"]
    rows = []
    for b, ids in enumerate(idx):
        rt = resample(td, ids)
        vc = variance_components(rt.g_value, rt.g_player, rt.year[rt.g_player])
        yb = y[ids]
        var_w = within_year_var(yb, td.year[ids])
        row = {"b": b, "lambda": vc["lambda"], "tau2": vc["tau2"], "sigma2": vc["sigma2"], "rel": vc["rel"]}
        c = {}
        for s, P in oof.items():
            r2cv = float(np.mean(1 - np.mean((yb[None, :] - P[:, ids]) ** 2, axis=1) / var_w))
            r2 = true_r2(r2cv, vc["rel"])
            c[s] = cge(vc["lambda"], r2)
            row |= {f"r2cv_{s}": r2cv, f"r2_{s}": r2, f"cge_{s}": c[s]}
        row |= {"delta_C3-C2": _diff(c, "C3", "C2"), "delta_C2-C1": _diff(c, "C2", "C1"),
                "delta_C1-C0": _diff(c, "C1", "C0")}
        rows.append(row)
    return pl.DataFrame(rows)


def percentile_ci(draws: np.ndarray, level: float = 0.95) -> dict:
    """Percentile interval treating NaN as +inf (an undefined CGE = a trait with no detectable player variance).
    Returns lo, hi (None when unbounded) and the share of non-finite draws."""
    d = np.where(np.isfinite(draws), draws, np.inf)
    a = (1 - level) / 2
    with np.errstate(invalid="ignore"):  # interpolating between inf draws gives nan = unbounded
        lo, hi = np.quantile(d, [a, 1 - a])
    return {"lo": float(lo) if np.isfinite(lo) else None, "hi": float(hi) if np.isfinite(hi) else None,
            "share_nonfinite": float(np.mean(~np.isfinite(draws)))}


# =====================================================================================================
# Validation 1: internal crossover
# =====================================================================================================

def first_k_windows(games: pl.DataFrame, k_max: int = 6, min_games: int = 8) -> pl.DataFrame:
    """games: nfl_id, game_id, value (centered). Players with >= min_games games: X1..X{k_max} = mean of the first
    k games (game_id order), target = mean of games k_max+1..end, n_target."""
    g = games.sort("nfl_id", "game_id").with_columns(j=pl.int_range(1, pl.len() + 1).over("nfl_id"))
    g = g.filter(pl.len().over("nfl_id") >= min_games)
    aggs = [pl.col("value").filter(pl.col("j") <= k).mean().alias(f"X{k}") for k in range(1, k_max + 1)]
    return (g.group_by("nfl_id").agg(*aggs, target=pl.col("value").filter(pl.col("j") > k_max).mean(),
                                     n_target=(pl.col("j") > k_max).sum()).sort("nfl_id"))


def integer_crossover(r_games: np.ndarray, r_c: float) -> int:
    """Smallest k (1-based) with r_games[k-1] >= r_c; len + 1 if none (censored)."""
    hit = np.flatnonzero(np.asarray(r_games) >= r_c)
    return int(hit[0] + 1) if len(hit) else len(r_games) + 1


def continuous_crossover(r_games: np.ndarray, r_c: float) -> float:
    """Linear interpolation of r over k = 0..K with r(0) = 0; 0 if r_c <= 0, inf if never reached."""
    if r_c <= 0:
        return 0.0
    r = np.r_[0.0, np.asarray(r_games, float)]
    for k in range(1, len(r)):
        if r[k] >= r_c:
            return float(k - 1 + (r_c - r[k - 1]) / (r[k] - r[k - 1])) if r[k] > r[k - 1] else float(k)
    return np.inf


def ceil_cge(c: float, k_max: int = 6) -> int:
    """The integer crossover the formula predicts: ceil(CGE) clipped to [1, k_max + 1]."""
    if not np.isfinite(c):
        return k_max + 1
    return int(np.clip(np.ceil(c - 1e-12), 1, k_max + 1))


# =====================================================================================================
# Validation 2: prospective fit and in-season forecasts
# =====================================================================================================

def split_prospective(games: pl.DataFrame, players: pl.DataFrame, train_seasons=(2023, 2024),
                      test_season: int = 2025, test_class: int = 2025) -> tuple[pl.DataFrame, pl.DataFrame]:
    """games: nfl_id, game_id, season, value (+ ...) -> (train, test). Train = games of the train seasons of players
    drafted before the test class; test = the test class's games in the test season. Nothing else is returned, so
    no later season reaches the fit."""
    g = games.join(players.select("nfl_id", "draft_year"), on="nfl_id", how="inner")
    tr = g.filter(pl.col("season").is_in(list(train_seasons)) & (pl.col("draft_year") < test_class))
    te = g.filter((pl.col("season") == test_season) & (pl.col("draft_year") == test_class))
    return tr.drop("draft_year"), te.drop("draft_year")


@dataclass
class ProspectiveFit:
    levels: list
    dummy_means: np.ndarray
    sets: dict
    vc: dict
    r2_cv: dict
    r2: dict
    models: dict
    y_mean: float
    train_ids: np.ndarray

    @property
    def lam(self) -> float:
        return self.vc["lambda"]


def fit_prospective(games_train: pl.DataFrame, players: pl.DataFrame, sets: dict, n_repeats: int = N_REPEATS,
                    seed: int = 0) -> ProspectiveFit:
    """Everything the in-season rule needs, from training games and the training players' rows only:
    variance components (draft-year strata), reliability, true-score R2 per set (same CV as estimate()) and the
    ridge models refitted on all training players."""
    td = trait_data(games_train, players, sets)
    e = estimate(td, n_repeats=n_repeats, seed=seed)
    n_i = np.bincount(td.g_player).astype(float)
    y = np.bincount(td.g_player, weights=td.g_value) / n_i
    models = {}
    with threadpool_limits(MAX_THREADS):
        for s, Xs in td.X.items():
            models[s] = float(y.mean()) if Xs is None else linkage._model().fit(Xs, y)
    D = year_dummies(td.year, td.levels)
    return ProspectiveFit(levels=td.levels, dummy_means=D.mean(axis=0), sets=sets,
                          vc={k: e[k] for k in ("tau2", "sigma2", "lambda", "rel", "icc", "n", "mean_games", "n_games")},
                          r2_cv=e["r2_cv"], r2=e["r2"], models=models, y_mean=float(y.mean()), train_ids=td.nfl_id)


def prior_means(fit: ProspectiveFit, players_test: pl.DataFrame) -> dict[str, np.ndarray]:
    """mu_C for test players (rows in the given order): ridge predictions with the draft-year dummies at their
    training means; C0 = training mean of y."""
    out = {}
    for s, cols in fit.sets.items():
        if cols is None:
            out[s] = np.full(players_test.height, fit.models[s])
        else:
            out[s] = fit.models[s].predict(design(players_test, cols, fit.levels, fit.dummy_means))
    return out


def in_season_forecasts(fit: ProspectiveFit, games_test: pl.DataFrame, players_test: pl.DataFrame,
                        ks=(1, 2, 3, 4), min_target: int = 3) -> pl.DataFrame:
    """One row per test player x k (players with >= k + min_target qualifying games): ybar_k (first k games),
    target (mean of games k+1..end), m (target games), mu_<C>, blend_<C>, sq-errors standardized by the training
    tau2 (se_games, se_comb_<C>, se_blend_<C>) and 80% interval coverage of the blend (cov_<C>). k = 0 rows
    (prior only, target = whole season) are included for calibration."""
    tau2, sigma2, lam = fit.vc["tau2"], fit.vc["sigma2"], fit.vc["lambda"]
    pt = players_test.join(games_test.select("nfl_id").unique(), on="nfl_id", how="semi").sort("nfl_id")
    mus = prior_means(fit, pt)
    mu_df = pl.DataFrame({"nfl_id": pt["nfl_id"], **{f"mu_{s}": v for s, v in mus.items()}})
    g = games_test.join(pt.select("nfl_id"), on="nfl_id", how="semi").sort("nfl_id", "game_id").with_columns(
        j=pl.int_range(1, pl.len() + 1).over("nfl_id"), n_games=pl.len().over("nfl_id"))
    rows = []
    for k in (0, *ks):
        w = (g.filter(pl.col("n_games") >= k + min_target).group_by("nfl_id")
             .agg(n_games=pl.col("n_games").first(),
                  ybar=pl.col("value").filter(pl.col("j") <= k).mean() if k else pl.lit(None, pl.Float64),
                  target=pl.col("value").filter(pl.col("j") > k).mean(), m=(pl.col("j") > k).sum())
             .with_columns(k=pl.lit(k)))
        rows.append(w)
    d = pl.concat(rows).join(mu_df, on="nfl_id", how="left").sort("k", "nfl_id")
    kk, yb, tg, m = (d[c].to_numpy().astype(float) for c in ("k", "ybar", "target", "m"))
    cols = {"se_games": np.where(kk > 0, (yb - tg) ** 2 / tau2, np.nan)}
    for s in fit.sets:
        mu, r2 = d[f"mu_{s}"].to_numpy(), fit.r2[s]
        if not np.isfinite(r2):
            continue
        b = blend(mu, yb, kk, lam, r2)
        b_lit = blend(mu, yb, kk, lam, r2, literal=True)
        half = Z80 * np.sqrt(posterior_var(kk, tau2, sigma2, r2) + sigma2 / m)
        cols |= {f"blend_{s}": b, f"w_{s}": blend_weight(kk, lam, r2), f"se_comb_{s}": (mu - tg) ** 2 / tau2,
                 f"se_blend_{s}": (b - tg) ** 2 / tau2, f"cov_{s}": (np.abs(tg - b) <= half).astype(float),
                 f"se_blendlit_{s}": (b_lit - tg) ** 2 / tau2}
    return d.with_columns(pl.Series(c, v) for c, v in cols.items())


# =====================================================================================================
# Pooled trait-weighted comparisons, sign-flip test, bootstrap, BH
# =====================================================================================================

def _matrices(df: pl.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """Players x traits value matrices (0 where absent) and the presence indicator, per column."""
    ids = np.sort(df["nfl_id"].unique().to_numpy())
    traits = sorted(df["trait"].unique().to_list(), key=lambda t: int(t[1:]))
    pi = np.searchsorted(ids, df["nfl_id"].to_numpy())
    ti = np.array([traits.index(t) for t in df["trait"].to_list()])
    M = np.zeros((len(ids), len(traits)))
    M[pi, ti] = 1.0
    V = {}
    for c in cols:
        A = np.zeros((len(ids), len(traits)))
        A[pi, ti] = df[c].to_numpy()
        V[c] = A
    return ids, M, V, np.array(traits)


def pooled_mean(df: pl.DataFrame, col: str) -> float:
    """Equal-weight mean over traits of the trait's mean of `col` (df: nfl_id, trait, col; one row per pair)."""
    return float(df.group_by("trait").agg(pl.col(col).mean())[col].mean())


def player_diffs(df: pl.DataFrame, a: str, b: str) -> tuple[np.ndarray, np.ndarray]:
    """d_i = sum over the player's traits of (a - b) / (T n_t): sum_i d_i = pooled_mean(a) - pooled_mean(b)."""
    ids, M, V, _ = _matrices(df, [a, b])
    n_t = M.sum(axis=0)
    T = len(n_t)
    return ids, ((V[a] - V[b]) / (T * n_t)).sum(axis=1)


def sign_flip_p(d: np.ndarray, n_flips: int = 100_000, seed: int = 0, alternative: str = "less",
                chunk: int = 5_000) -> float:
    """One-sided sign-flip permutation p of S = sum d (alternative 'less': S unusually negative)."""
    rng = np.random.default_rng(seed)
    S, count, done = float(d.sum()), 0, 0
    while done < n_flips:
        c = min(chunk, n_flips - done)
        Sp = (rng.integers(0, 2, (c, len(d))) * 2 - 1) @ d
        count += int(np.sum(Sp <= S + 1e-12)) if alternative == "less" else int(np.sum(Sp >= S - 1e-12))
        done += c
    return (1 + count) / (1 + n_flips)


def pooled_bootstrap(df: pl.DataFrame, cols: list[str], n_boot: int = 10_000, seed: int = 0) -> dict[str, np.ndarray]:
    """Player-bootstrap draws of pooled_mean for each column (per-trait means recomputed on each resample; a trait
    absent from a resample is dropped from that draw's average)."""
    ids, M, V, _ = _matrices(df, cols)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=n_boot).astype(float)
    cnt = W @ M
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for c in cols:
            out[c] = np.nanmean((W @ V[c]) / np.where(cnt > 0, cnt, np.nan), axis=1)
    return out


def bh(p: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (q), in input order."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out.tolist()


# =====================================================================================================
# Post hoc (review response): simulation at a trait's own design, calibrated bounds, relative R2
# =====================================================================================================

def relative_true_r2(r2cv: float, r2cv_ref: float, rel: float) -> float:
    """True-score R2 of a set measured against a reference forecast under the same CV: clip((R2_CV - R2_CV_ref) /
    rel, 0, R2_MAX). With the C0 reference this removes the cost of estimating class means out of fold, which the
    k-game benchmark k / (k + lambda) does not pay."""
    return true_r2(r2cv - r2cv_ref, rel)


def signal_matrix(X: np.ndarray, n_dummies: int) -> np.ndarray:
    """The Combine columns of a design (the leading draft-year dummies dropped), median-imputed and standardized;
    constant columns dropped. Used only to plant a known signal at the real design."""
    Z = np.asarray(X, float)[:, n_dummies:]
    Z = np.where(np.isnan(Z), np.nanmedian(Z, axis=0), Z)
    Z = Z[:, Z.std(axis=0) > 0]
    return (Z - Z.mean(axis=0)) / Z.std(axis=0)


def simulate_values(td: TraitData, Z: np.ndarray, lam: float, r2: float, rng: np.random.Generator,
                    n_signal: int = 3) -> tuple[TraitData, np.ndarray]:
    """A simulated copy of a trait at its own design (same players, classes, game counts and design matrices):
    theta = sqrt(r2) s + sqrt(1 - r2) u, with s a unit-variance signed sum of n_signal randomly chosen columns of Z
    and u ~ N(0, 1) (tau2 = 1, no class effects); game values = theta + N(0, lam). Returns (TraitData, theta)."""
    cols = rng.choice(Z.shape[1], min(n_signal, Z.shape[1]), replace=False)
    s = Z[:, cols] @ rng.choice([-1.0, 1.0], len(cols))
    s = (s - s.mean()) / s.std()
    theta = np.sqrt(r2) * s + np.sqrt(1 - r2) * rng.normal(size=len(s))
    vals = theta[td.g_player] + rng.normal(0, np.sqrt(lam), len(td.g_player))
    return TraitData(nfl_id=td.nfl_id, year=td.year, X=td.X, g_player=td.g_player, g_value=vals,
                     levels=td.levels), theta


def oracle_r2(theta: np.ndarray, oof: np.ndarray) -> float:
    """The true-score R2 that out-of-fold forecasts (n_repeats x n) achieve against the true theta (tau2 = 1, as in
    simulate_values): 1 - mean over repeats of mean_i (theta_i - oof_i)^2. This is the CV estimand that
    R2_CV / rel estimates: the Combine as learned by the ridge from (N_SPLITS - 1) / N_SPLITS of the players."""
    return float(1 - np.mean(np.mean((np.asarray(theta)[None, :] - oof) ** 2, axis=1)))


def calibrated_bounds(obs: float, est_star: np.ndarray, draws: list[np.ndarray], alpha: float = 0.05) -> dict:
    """Neyman construction on a simulation grid. est_star[g]: the estimand at grid point g (made non-decreasing);
    draws[g]: simulated estimates there. One-sided (1 - alpha) bounds for the estimand given the observed estimate:
    upper = the largest estimand with P(estimate <= obs) >= alpha, lower = the smallest with P(estimate >= obs)
    >= alpha, each interpolated linearly in that probability between neighbouring grid points. upper None: not
    bounded within the grid (the estimate is consistent with the largest grid value). Undefined draws (NaN, a
    simulated trait with no detectable player variance) count as +inf, as in percentile_ci."""
    e = np.maximum.accumulate(np.asarray(est_star, float))
    draws = [np.where(np.isfinite(np.asarray(d, float)), np.asarray(d, float), np.inf) for d in draws]
    p_le = np.array([np.mean(d <= obs + 1e-9) for d in draws])
    p_ge = np.array([np.mean(d >= obs - 1e-9) for d in draws])
    acc = np.flatnonzero(p_le >= alpha)
    if len(acc) == 0:
        upper = float(e[0])
    elif acc[-1] == len(e) - 1:
        upper = None
    else:
        a = acc[-1]
        f = (p_le[a] - alpha) / (p_le[a] - p_le[a + 1]) if p_le[a] > p_le[a + 1] else 0.0
        upper = float(e[a] + f * (e[a + 1] - e[a]))
    acc = np.flatnonzero(p_ge >= alpha)
    if len(acc) == 0:
        lower = None
    elif acc[0] == 0:
        lower = float(e[0])
    else:
        b = acc[0]
        f = (alpha - p_ge[b - 1]) / (p_ge[b] - p_ge[b - 1]) if p_ge[b] > p_ge[b - 1] else 1.0
        lower = float(e[b - 1] + f * (e[b] - e[b - 1]))
    return {"lower": lower, "upper": upper, "alpha_each_side": alpha, "p_le_obs": p_le.tolist(),
            "p_ge_obs": p_ge.tolist()}
