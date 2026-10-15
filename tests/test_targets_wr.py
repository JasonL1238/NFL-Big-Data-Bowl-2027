import numpy as np
import polars as pl
import pytest

from bdb27 import targets_wr as T
from bdb27.shrinkage import eb_mean

GAMES = pl.DataFrame({"game_id": [1, 2], "season": [2023, 2024], "season_type": ["REG", "PRE"],
                      "home_team_abbr": ["KC", "KC"], "visitor_team_abbr": ["DET", "BUF"]})


def _plays(**over):
    """Four hand-built plays of game 1 (KC home 7, DET visitor 3) plus one PRE play of game 2."""
    base = {
        "game_id": [1, 1, 1, 1, 2], "play_id": [10, 11, 12, 13, 20], "nfl_id": [100, 100, 101, 100, 100],
        "possession_team": ["KC", "DET", "KC", "DET", "KC"],
        "yardline_side": ["KC", "KC", None, "DET", "KC"], "yardline_number": [20, 30, 50, 1, 40],
        "pre_snap_home_score": [7, 7, 7, 7, 0], "pre_snap_visitor_score": [3, 3, 3, 3, 0],
        "route_ran": ["GO", "HITCH", None, "POST", "GO"], "rec_yards": [None, 12, None, -2, 5],
        "target": [False, True, False, True, True],
    }
    return pl.DataFrame({**base, **over})


def test_yards_to_end_zone_defense_team_and_score_differential():
    d = T.add_context(_plays(), GAMES)
    assert d["yards_to_end_zone"].to_list() == [80.0, 30.0, 50.0, 99.0, 60.0]  # own 20, opp 30, midfield, own 1
    assert d["defense_team"].to_list() == ["DET", "KC", "DET", "KC", "BUF"]
    assert d["defense_team_season"].to_list() == ["DET_2023", "KC_2023", "DET_2023", "KC_2023", "BUF_2024"]
    assert d["score_diff"].to_list() == [4.0, -4.0, 4.0, -4.0, 0.0]


def test_add_context_leaves_unknown_possession_team_null():
    d = T.add_context(_plays(possession_team=["NYJ", "DET", "KC", "DET", "KC"]), GAMES)
    assert d["defense_team"][0] is None and d["score_diff"][0] is None and d["defense_team_season"][0] is None


def test_route_table_keeps_reg_routes_of_study_positions():
    pos = pl.DataFrame({"nfl_id": [100, 101], "position": ["WR", "DB"]})
    d = T.route_table(_plays(), GAMES, pos)
    assert d["play_id"].to_list() == [10, 11, 13]  # play 12 has no route, play 20 is preseason
    assert d["rec_yards0"].to_list() == [0.0, 12.0, -2.0]  # null (no catch) -> 0, negative yards kept
    assert d["deep"].to_list() == [1.0, 0.0, 1.0] and d["targeted"].to_list() == [0.0, 1.0, 1.0]
    pos_te = pos.with_columns(position=pl.lit("TE"))
    assert T.route_table(_plays(), GAMES, pos_te).height == 3
    assert T.route_table(_plays(), GAMES, pos_te, positions=("WR",)).height == 0


def test_design_codes_categories_by_sorted_label_and_keeps_missing():
    d = pl.DataFrame({"a": ["z", "b", None, "b"], "flag": [True, None, False, True], "n": [1, None, 3, 4]})
    X = T.design(d, cat=["a", "flag"], num=["n"])
    assert np.array_equal(X[:, 0], [1, 0, np.nan, 0], equal_nan=True)
    assert np.array_equal(X[:, 1], [1, np.nan, 0, 1], equal_nan=True)
    assert np.array_equal(X[:, 2], [1, np.nan, 3, 4], equal_nan=True)


def _synthetic_routes(n_games=60, per_game=40, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_games), per_game)
    n = len(g)
    align = rng.choice(["WR", "SLOT_WR", "TE"], n)
    ytg = rng.integers(1, 15, n).astype(float)
    game_noise = rng.normal(0, 2.0, n_games)[g]  # game-level shocks the context cannot explain
    y = np.where(align == "SLOT_WR", 2.0, 0.0) + 0.3 * ytg + game_noise + rng.normal(0, 1, n)
    return pl.DataFrame({"game_id": g, "align": align, "ytg": ytg, "y": y,
                         "yb": (y > np.median(y)).astype(float)})


def test_cross_fit_has_no_leakage_across_game_folds():
    d = _synthetic_routes()
    kw = dict(cat=["align"], num=["ytg"])
    pred = T.cross_fit(d, "y", **kw)
    assert np.isfinite(pred).all()
    assert np.array_equal(pred, T.cross_fit(d, "y", **kw))  # deterministic
    # Changing the outcomes of one game cannot change that game's own expectations (it is never in training).
    hit = d["game_id"] == 7
    d2 = d.with_columns(y=pl.when(hit).then(pl.col("y") + 100.0).otherwise(pl.col("y")))
    pred2 = T.cross_fit(d2, "y", **kw)
    assert np.array_equal(pred[hit.to_numpy()], pred2[hit.to_numpy()])
    assert not np.array_equal(pred[~hit.to_numpy()], pred2[~hit.to_numpy()])
    # Game shocks are not learnable out of fold: residuals keep them (in-sample fitting would absorb part).
    resid = d["y"].to_numpy() - pred
    game_mean = d.with_columns(r=pl.Series(resid)).group_by("game_id").agg(pl.col("r").mean())["r"].to_numpy()
    assert np.var(game_mean) > 2.0  # the game shock variance is 4; 1/40 of the play noise is ~0.03
    assert 0.2 < T.oof_r2(d["y"].to_numpy(), pred) < 0.6  # context explains ~0.3-0.4 of the variance


def test_cross_fit_classification_gives_probabilities_and_rejects_nulls():
    d = _synthetic_routes(seed=1)
    p = T.cross_fit(d, "yb", kind="classification", cat=["align"], num=["ytg"])
    assert ((p > 0) & (p < 1)).all()
    assert np.corrcoef(p, d["yb"].to_numpy())[0, 1] > 0.2
    with pytest.raises(ValueError, match="nulls"):
        T.cross_fit(d.with_columns(y=pl.lit(None, pl.Float64)), "y", cat=["align"], num=["ytg"])
    with pytest.raises(ValueError, match="kind"):
        T.cross_fit(d, "y", kind="ranking", cat=["align"], num=["ytg"])


def _game_plays(k=300, n_games=16, per_game=(3, 9), game_sd=0.6, seed=0):
    """Per-play values: theta_i ~ N(0, 0.25) + game effect N(0, game_sd^2) + N(0, 1)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(k):
        theta = rng.normal(0, 0.5)
        for g in range(n_games):
            m = rng.integers(*per_game)
            v = theta + rng.normal(0, game_sd) + rng.normal(0, 1, m)
            rows.append(pl.DataFrame({"nfl_id": i, "game_id": g, "value": v, "position": "WR"}))
    return pl.concat(rows)


def test_half_difference_is_calibrated_with_clustering_and_flags_ignored_game_effects():
    plays = _game_plays()
    eb = T.player_target(plays, "value", min_n=30)  # cluster='game_id'
    hd = T.half_difference(plays, eb, lambda d: eb_mean(d, "value", min_n=1, cluster="game_id"))
    assert hd["n_players"] == 300
    assert hd["mean_z2"] == pytest.approx(1.0, abs=4 * hd["se"])
    plain = T.half_difference(plays, eb, lambda d: eb_mean(d, "value", min_n=1))
    assert plain["mean_z2"] > 1.5  # unclustered variances miss the game effect (true ratio ~2.9)
    iid = _game_plays(game_sd=0.0, seed=1)
    hd0 = T.half_difference(iid, T.player_target(iid, "value", min_n=30),
                            lambda d: eb_mean(d, "value", min_n=1, cluster="game_id"))
    assert hd0["mean_z2"] == pytest.approx(1.0, abs=4 * hd0["se"])


def test_player_target_filters_positions_and_fits_one_prior_per_group():
    wr = _game_plays(k=40, seed=2)
    te = _game_plays(k=40, seed=3).with_columns(nfl_id=pl.col("nfl_id") + 1000, position=pl.lit("TE"),
                                                value=pl.col("value") + 3.0)
    both = pl.concat([wr, te])
    assert T.player_target(both, "value", min_n=30)["nfl_id"].max() < 1000
    pooled = T.player_target(both, "value", min_n=30, positions=("WR", "TE"), group="position")
    assert pooled.height == 80 and pooled["prior_mean"].n_unique() == 2
