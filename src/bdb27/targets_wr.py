"""Receiver game targets (E007): play-context expectations and per-player 'over expected' traits.

Route table (route_table): one row per regular-season player_play row with a non-null `route_ran` for the study
receivers (Combine position WR and TE), with play context derived from player_play and games.csv:
  - yards_to_end_zone: 100 - yardline_number when the ball is on the possession team's own side, yardline_number
    on the opponent's side; at midfield yardline_side is null and the value is 50.
  - defense_team: the team in the game (games.csv home/visitor) that is not possession_team;
    defense_team_season = "<defense_team>_<season>".
  - score_diff: possession team's pre-snap score minus the opponent's.
  - rec_yards0: rec_yards with null (no catch) as 0, i.e. receiving yards on the route.

Expectation (cross_fit): the context is fixed by the E007 prereg (CONTEXT_CAT + CONTEXT_NUM): alignment,
formation, motion, man/zone, down, distance, yards to the end zone, quarter, score differential, season and the
defense's team-season. No route type (route choice is part of how a player is used), no Combine data and no player
identity. HistGradientBoosting (regressor, or classifier for binary outcomes) with native categorical support and
default hyperparameters, cross-fitted with GroupKFold(5) by game_id: a play's expectation comes from a model that
never saw any play of that game. Category codes depend only on the labels present, never on outcomes.
residual (`<value>_oe`) = observed - out-of-fold prediction.

Player targets (player_target): shrinkage.eb_mean of a per-route value with cluster='game_id' (plays in one game
are correlated); the prior is fitted on the kept players only.

half_difference: calibration check of the EB sampling variances (true means cancel in the odd-even game difference).
"""

from collections.abc import Callable

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from bdb27 import io, shrinkage

CONTEXT_CAT = ["lined_up_position", "receiver_alignment", "offense_formation", "in_motion_at_ball_snap",
               "team_coverage_man_zone", "season", "defense_team_season"]
CONTEXT_NUM = ["down", "yards_to_go", "yards_to_end_zone", "quarter", "score_diff"]
DEEP_ROUTES = ("GO", "POST", "CORNER")
N_SPLITS = 5
MAX_THREADS = 3  # project rule for the shared machine (LESSONS 2026-10-15)
SEED = 0


def yards_to_end_zone() -> pl.Expr:
    """Yards from the line of scrimmage to the end zone the offense attacks."""
    side, num = pl.col("yardline_side"), pl.col("yardline_number")
    return (pl.when(side.is_null()).then(num)  # midfield: no side
            .when(side == pl.col("possession_team")).then(100 - num)
            .otherwise(num)).cast(pl.Float64)


def add_context(plays: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """Join games (game_id, season, home_team_abbr, visitor_team_abbr; inner, so pass REG games only to keep REG
    plays) and add yards_to_end_zone, defense_team, defense_team_season and score_diff. Rows whose
    possession_team is neither home nor visitor get null defense/score fields."""
    home = pl.col("possession_team") == pl.col("home_team_abbr")
    away = pl.col("possession_team") == pl.col("visitor_team_abbr")
    hs, vs = pl.col("pre_snap_home_score"), pl.col("pre_snap_visitor_score")
    d = plays.join(games.select("game_id", "season", "home_team_abbr", "visitor_team_abbr"), on="game_id",
                   how="inner", maintain_order="left")
    d = d.with_columns(
        yards_to_end_zone=yards_to_end_zone(),
        defense_team=pl.when(home).then(pl.col("visitor_team_abbr")).when(away).then(pl.col("home_team_abbr")),
        score_diff=pl.when(home).then(hs - vs).when(away).then(vs - hs).cast(pl.Float64),
    )
    return d.with_columns(defense_team_season=pl.concat_str(pl.col("defense_team"), pl.col("season").cast(pl.String),
                                                            separator="_"))


def route_table(player_play: pl.DataFrame | None = None, games: pl.DataFrame | None = None,
                positions_df: pl.DataFrame | None = None, positions: tuple[str, ...] = ("WR", "TE")) -> pl.DataFrame:
    """REG-season route rows (route_ran non-null) of study players whose Combine position is in `positions`,
    with add_context columns, `position`, rec_yards0, `deep` (route in DEEP_ROUTES) and `targeted` (target as 0/1).
    Defaults load player_play, games and combine_results from the Parquet cache."""
    pp = io.load("player_play") if player_play is None else player_play
    g = io.load("games") if games is None else games
    pos = (io.load("combine_results").select("nfl_id", position="combine_position") if positions_df is None
           else positions_df.select("nfl_id", "position"))
    d = (pp.filter(pl.col("route_ran").is_not_null())
         .join(pos.filter(pl.col("position").is_in(list(positions))), on="nfl_id", how="inner", maintain_order="left"))
    d = add_context(d, g.filter(pl.col("season_type") == "REG"))
    return d.with_columns(rec_yards0=pl.col("rec_yards").fill_null(0).cast(pl.Float64),
                          deep=pl.col("route_ran").is_in(list(DEEP_ROUTES)).cast(pl.Float64),
                          targeted=pl.col("target").cast(pl.Float64))


def design(d: pl.DataFrame, cat: list[str] = CONTEXT_CAT, num: list[str] = CONTEXT_NUM) -> np.ndarray:
    """Context matrix: the `cat` columns first, as integer codes over their sorted labels (null -> NaN, which
    HistGradientBoosting treats as its own missing category), then `num` as float (null -> NaN)."""
    cols = []
    for c in cat:
        s = d[c].cast(pl.String)
        labels = sorted(s.drop_nulls().unique().to_list())
        codes = s.replace_strict(labels, list(range(len(labels))), default=None, return_dtype=pl.Float64)
        cols.append(codes.to_numpy())
    cols += [d[c].cast(pl.Float64).to_numpy() for c in num]
    return np.column_stack(cols).astype(float)


def cross_fit(d: pl.DataFrame, value: str, kind: str = "regression", n_splits: int = N_SPLITS, seed: int = SEED,
              cat: list[str] = CONTEXT_CAT, num: list[str] = CONTEXT_NUM) -> np.ndarray:
    """Out-of-fold expectation of `value` from context, GroupKFold(n_splits) by game_id. kind='regression':
    HistGradientBoostingRegressor prediction; kind='classification' (0/1 value): HistGradientBoostingClassifier
    P(value = 1). Default hyperparameters, random_state=seed. Rows with a null value must be dropped first."""
    if d[value].null_count():
        raise ValueError(f"{value!r} has nulls; filter them out before cross-fitting")
    X, y, groups = design(d, cat, num), d[value].cast(pl.Float64).to_numpy(), d["game_id"].to_numpy()
    cat_idx = list(range(len(cat)))
    pred = np.full(len(y), np.nan)
    with threadpool_limits(MAX_THREADS):  # OpenMP would otherwise take every core of the shared machine
        _fit_folds(X, y, groups, kind, n_splits, seed, cat_idx, pred)
    return pred


def _fit_folds(X, y, groups, kind, n_splits, seed, cat_idx, pred) -> None:
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if kind == "regression":
            m = HistGradientBoostingRegressor(categorical_features=cat_idx, random_state=seed).fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        elif kind == "classification":
            m = HistGradientBoostingClassifier(categorical_features=cat_idx, random_state=seed).fit(X[tr], y[tr])
            pred[te] = m.predict_proba(X[te])[:, list(m.classes_).index(1.0)]
        else:
            raise ValueError(f"kind must be 'regression' or 'classification', not {kind!r}")


def oof_r2(y: np.ndarray, pred: np.ndarray) -> float:
    """1 - SSE / SST of out-of-fold predictions."""
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def player_target(routes: pl.DataFrame, value: str, min_n: int, positions: tuple[str, ...] = ("WR",),
                  group: str | None = None) -> pl.DataFrame:
    """shrinkage.eb_mean of per-route `value` over the routes of players in `positions`, cluster='game_id'.
    group='position' fits one prior per position (pooled populations)."""
    d = routes.filter(pl.col("position").is_in(list(positions)))
    return shrinkage.eb_mean(d, value, group=group, min_n=min_n, cluster="game_id")


def half_difference(plays: pl.DataFrame, eb: pl.DataFrame, fit: Callable[[pl.DataFrame], pl.DataFrame],
                    player: str = "nfl_id", unit: str = "game_id", min_per_half: int = 10) -> dict:
    """Calibration of EB sampling variances. Each kept player's (rows of `eb`) plays are split into odd and even
    `unit`s (dense rank, as in reliability.split_half) and `fit` (the same estimator, returning raw and se per
    player) is applied to each half. With z2 = (raw_0 - raw_1)^2 / (se_0^2 + se_1^2), the player's true mean
    cancels, so E[z2] = 1 when the sampling variances are right; > 1 means they are understated (too little
    shrinkage), < 1 overstated. Players need >= min_per_half rows in each half. Returns the mean z2 with its SE,
    the pooled ratio sum (raw_0 - raw_1)^2 / sum (se_0^2 + se_1^2), and n."""
    d = (plays.join(eb.select(player), on=player, how="semi")
         .with_columns(_half=(pl.col(unit).rank("dense").over(player) % 2).cast(pl.Int8))
         .filter(pl.len().over(player, "_half") >= min_per_half))
    h = [fit(d.filter(pl.col("_half") == k).drop("_half")).select(player, pl.col("raw").alias(f"raw{k}"),
                                                                   (pl.col("se") ** 2).alias(f"v{k}"))
         for k in (0, 1)]
    m = h[0].join(h[1], on=player, how="inner")
    diff2 = ((m["raw0"] - m["raw1"]) ** 2).to_numpy()
    v = (m["v0"] + m["v1"]).to_numpy()
    z2 = diff2 / v
    return {"mean_z2": float(z2.mean()), "se": float(z2.std(ddof=1) / np.sqrt(len(z2))),
            "pooled_ratio": float(diff2.sum() / v.sum()), "n_players": int(len(z2))}
