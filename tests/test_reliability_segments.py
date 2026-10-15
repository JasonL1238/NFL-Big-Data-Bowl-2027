import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from bdb27.reliability import cross_drill_icc, icc1, split_half, split_half_estimator, team_share
from bdb27.segments import combine_sessions, onset_time

T0 = datetime(2024, 3, 1, 9)


def _games(n_players=150, n_games=20, per_game=3, sd=2.0, seed=2):
    """Rows of mu_player + N(0, sd); per-half mean of 30 rows -> r_half = 1 / (1 + sd^2 / 30)."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame([{"nfl_id": p, "game_id": g, "v": mu + rng.normal(0, sd)}
                         for p, mu in enumerate(rng.normal(size=n_players))
                         for g in range(n_games) for _ in range(per_game)])


# ---------- icc1 edge cases ----------

@pytest.mark.parametrize("values, groups", [
    (np.arange(5.0), np.arange(5)),                          # all singletons
    (np.ones(9), np.array([0, 0, 1, 1, 2, 2, 3, 4, 5])),     # no variance: 0 / 0
])
def test_icc1_degenerate_inputs_are_nan_without_warnings(values, groups):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = icc1(values, groups)
    assert np.isnan(res["icc"]) and np.isnan(res["reliability"])


@pytest.mark.parametrize("eps", [0, 1e-6, 1e-3])
def test_icc1_negative_icc_has_zero_reliability_at_the_singular_point(eps):
    # icc = -1 makes the Spearman-Brown denominator 1 + (n - 1) icc = 0; a nudge used to give -3e12
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = icc1(np.array([0, 1, 0, 1, 0, 1.0 + eps]), np.array([0, 0, 1, 1, 2, 2]))
    assert res["icc"] == pytest.approx(-1.0, abs=1e-3) and res["reliability"] == 0.0


def test_icc1_negative_icc_with_unbalanced_repeats_is_not_reliable():
    # anti-correlated pairs (2 repeats) + 8-repeat players: icc ~ -0.2, where 1 + (n - 1) icc changes sign
    # between n = 2 and n = 8 and the averaged Spearman-Brown used to read +1.46 (passing a >= 0.5 screen)
    rng = np.random.default_rng(0)
    vals, grp = [], []
    for p in range(50):
        a = rng.normal()
        vals += [a, -a]
        grp += [p, p]
    for p in range(50, 100):
        x = rng.normal(size=8)
        vals += list(x - x.mean() + rng.normal(0, 0.15))
        grp += [p] * 8
    res = icc1(np.array(vals), np.array(grp))
    assert -0.3 < res["icc"] < -0.1 and res["reliability"] == 0.0


# ---------- cross_drill_icc ----------

def test_cross_drill_icc_center_by_removes_session_offsets():
    rng = np.random.default_rng(4)
    rows = []
    for p, mu in enumerate(rng.normal(size=400)):
        for d, offs in (("A", (3, -3)), ("B", (-3, 3)), ("C", (2, -2))):
            s = int(rng.integers(2))  # each player attends one session per drill
            rows.append({"nfl_id": p, "drill_name": d, "session": s, "v": mu + offs[s] + rng.normal(0, 0.5)})
    df = pl.DataFrame(rows)
    assert cross_drill_icc(df, "v") == cross_drill_icc(df, "v", center_by=["drill_name"])
    # drill-only centering leaves the session offsets (var ~ 4-9) as noise; drill x session removes them
    assert cross_drill_icc(df, "v")["icc"] < 0.3
    assert abs(cross_drill_icc(df, "v", center_by=["drill_name", "session"])["icc"] - 0.8) < 0.05


def test_cross_drill_icc_zscore_equalizes_drill_scales():
    rng = np.random.default_rng(5)
    rows = [{"nfl_id": p, "drill_name": d, "v": scale * (mu + rng.normal(0, 0.5))}
            for p, mu in enumerate(rng.normal(size=1000)) for d, scale in (("A", 1.0), ("B", 10.0))]
    df = pl.DataFrame(rows)
    # raw pairs: ICC ~ 2 cov / (var_A + var_B) = 20 / 126.25 = 0.16; z-scored: 1 / 1.25 = 0.8
    assert abs(cross_drill_icc(df, "v")["icc"] - 0.16) < 0.05
    assert abs(cross_drill_icc(df, "v", zscore=True)["icc"] - 0.8) < 0.04


def test_cross_drill_icc_drops_rows_alone_in_their_centering_group():
    rng = np.random.default_rng(1)
    rows = [{"nfl_id": p, "drill_name": d, "session": p if d == "C" else 0, "v": mu + rng.normal(0, 0.5)}
            for p, mu in enumerate(rng.normal(size=300)) for d in "ABC"]  # drill C: every player in his own session
    df = pl.DataFrame(rows)
    ab = df.filter(pl.col("drill_name") != "C")
    assert abs(cross_drill_icc(ab, "v")["icc"] - 0.8) < 0.05
    for z in (False, True):  # with zscore=False an exact-0 singleton used to count as a repeat (icc 0.39)
        res = cross_drill_icc(df, "v", center_by=["drill_name", "session"], zscore=z)
        assert res["n_dropped"] == 300 and res["n_players"] == 300 and res["n_single"] == 0
        assert res["icc"] == pytest.approx(cross_drill_icc(ab, "v", zscore=z)["icc"])


# ---------- split_half / split_half_estimator ----------

def test_split_half_spearman_correlates_ranks():
    # one row per player-half (2 games): half means are the rows, corr 0.5 bivariate normal
    rng = np.random.default_rng(8)
    mu = rng.normal(size=4000)
    df = pl.DataFrame({"nfl_id": np.repeat(np.arange(4000), 2), "game_id": np.tile([0, 1], 4000),
                       "v": np.repeat(mu, 2) + rng.normal(size=8000)})
    lognormal = df.with_columns(v=(2 * pl.col("v")).exp())  # monotone: same ranks; population Pearson r 0.02
    sp = split_half(df, "v", min_per_half=1, method="spearman")
    assert abs(sp["r_half"] - 6 / np.pi * np.arcsin(0.5 / 2)) < 0.04  # 0.483, SE ~0.012
    assert split_half(lognormal, "v", min_per_half=1, method="spearman") == sp
    assert split_half(lognormal, "v", min_per_half=1)["r_half"] < sp["r_half"] - 0.2
    assert sp["reliability"] == pytest.approx(2 * sp["r_half"] / (1 + sp["r_half"]))


def test_split_half_rejects_an_unknown_method_even_with_few_players():
    small = _games(n_players=5)  # < 10 players: both used to return NaN without checking the method
    with pytest.raises(ValueError):
        split_half(small, "v", method="bogus")
    with pytest.raises(ValueError):
        split_half_estimator(small, lambda g: g["v"].mean(), method="bogus")


def test_split_half_spearman_resists_an_extreme_player():
    clean = _games()
    df = clean.with_columns(v=pl.when((pl.col("nfl_id") == 0) & (pl.col("game_id") == 0))
                            .then(1e6).otherwise(pl.col("v")))
    assert split_half(clean, "v")["r_half"] > 0.8 and split_half(df, "v")["r_half"] < 0.2
    # the extreme player moves one rank only
    assert abs(split_half(df, "v", method="spearman")["r_half"] - split_half(clean, "v", method="spearman")["r_half"]) < 0.02
    assert split_half(df, "v") == pytest.approx(split_half(df, "v", method="pearson"))


def test_split_half_estimator_with_the_mean_reproduces_split_half():
    df = _games()
    res = split_half_estimator(df, lambda g: g["v"].mean() if g.height >= 5 else np.nan)
    assert res == pytest.approx(split_half(df, "v"))  # row order differs: equal up to float rounding


def test_split_half_estimator_quantile_has_the_asymptotic_reliability():
    rng = np.random.default_rng(6)
    sd, n_half, q = 3.0, 100, 0.9
    rows = [(p, g, mu + rng.normal(0, sd)) for p, mu in enumerate(rng.normal(size=1000)) for g in range(2 * n_half)]
    rows += [(1000 + p, g, rng.normal(0, sd)) for p in range(50) for g in range(20)]  # too few games: dropped
    df = pl.DataFrame(rows, schema=["nfl_id", "game_id", "v"], orient="row")
    res = split_half_estimator(df, lambda g: g["v"].quantile(q) if g.height >= 50 else np.nan)
    # Var(sample q90) = q (1 - q) sd^2 / (n phi(z_q)^2) = 0.263 -> r_half = 1 / 1.263 = 0.79
    noise = q * (1 - q) * sd**2 / (n_half * (np.exp(-1.2816**2 / 2) / np.sqrt(2 * np.pi)) ** 2)
    assert res["n_players"] == 1000
    assert abs(res["r_half"] - 1 / (1 + noise)) < 0.04


# ---------- team_share ----------

def _team_plays(team_sd, player_sd, noise_sd, n_plays, shared_sd=0.0, n_teams=1200, per_team=4, n_games=16, seed=7):
    """Every player of a team is on each of its n_plays plays (spread evenly over n_games games);
    value = team + player + play noise shared by the teammates on the play (shared_sd) + own noise."""
    rng = np.random.default_rng(seed)
    team = np.repeat(np.arange(n_teams), per_team)
    mu = rng.normal(0, team_sd, n_teams)[team] + rng.normal(0, player_sd, len(team))
    game = np.arange(n_plays) % n_games
    v = mu[:, None] + rng.normal(0, shared_sd, (n_teams, n_plays))[team] + rng.normal(0, noise_sd, (len(team), n_plays))
    return pl.DataFrame({"nfl_id": np.repeat(np.arange(len(team)), n_plays), "team": np.repeat(team, n_plays),
                         "season": 2024, "game_id": np.tile(game, len(team)), "v": v.ravel()})


def _close(res, truth):
    return res["se"] < 0.05 and abs(res["team_share"] - truth) < 4 * res["se"]


@pytest.mark.parametrize("n_plays", [24, 400])
def test_team_share_does_not_depend_on_plays_per_player(n_plays):
    # team var 1, player var 1, play noise var 36: an ICC of player x team means reads 1 / (2 + 36 / n) = 0.29 at
    # 24 plays (7 se below 0.5) and 0.48 at 400; the cross-half ratio reads 0.5 at both
    res = team_share(_team_plays(1.0, 1.0, 6.0, n_plays), "v", ["team", "season"])
    assert _close(res, 0.5)
    r_half = 2 / (2 + 36 / (n_plays / 2))
    assert abs(res["reliability"] - 2 * r_half / (1 + r_half)) < 0.03
    assert res["n_teams"] == res["n_shared"] == 1200 and res["n_player_teams"] == 4800
    assert _close(team_share(_team_plays(0.0, 1.0, 6.0, n_plays), "v", ["team", "season"]), 0.0)


def test_team_share_ignores_play_noise_shared_by_teammates():
    # shared play noise var 36 over 32 plays adds 1.1 to teammates' covariance: the ICC of means reads 0.67
    assert _close(team_share(_team_plays(1.0, 1.0, 1.0, 32, shared_sd=6.0), "v", ["team", "season"]), 0.5)


def test_team_share_jackknife_se_matches_the_spread_across_samples():
    res = [team_share(_team_plays(1.0, 1.0, 4.0, 40, n_teams=60, per_team=3, seed=s), "v", ["team", "season"])
           for s in range(60)]
    est, se = np.array([r["team_share"] for r in res]), np.array([r["se"] for r in res])
    assert abs(est.mean() - 0.5) < 0.05 and 0.75 < np.median(se) / est.std() < 1.33


def test_team_share_counts_a_traded_player_once_per_team_and_needs_shared_teams():
    rows = [(pid, team, g, float(pid + g)) for pid, team, games in [(1, "A", range(4)), (1, "B", range(4, 8)),
                                                                    (2, "B", range(4, 8)), (3, "C", range(4)),
                                                                    (4, "C", [0])]  # one game: one half only
            for g in games for _ in range(3)]
    df = pl.DataFrame(rows, schema=["nfl_id", "team_season", "game_id", "v"], orient="row")
    res = team_share(df, "v", "team_season", min_per_half=5)
    assert res["n_player_teams"] == 4 and res["n_teams"] == 3 and res["n_shared"] == 1
    assert np.isnan(res["team_share"]) and np.isnan(res["se"])  # < 3 team-seasons with >= 2 study players


# ---------- segments: combine_sessions, onset_time ----------

def _attempts(starts):
    """starts: list of (drill_name, draft_year, seconds after T0); 5 frames per attempt, one traj each."""
    return pl.DataFrame([{"traj": i, "drill_name": d, "draft_year": y,
                          "time": T0 + timedelta(seconds=s, milliseconds=100 * f)}
                         for i, (d, y, s) in enumerate(starts) for f in range(5)])


def test_combine_sessions_splits_on_ten_minute_gaps_within_drill_and_year():
    starts = [("A", 2023, 0), ("A", 2023, 30), ("A", 2023, 590), ("A", 2023, 7200), ("A", 2023, 7260),
              ("B", 2023, 0), ("B", 2023, 300), ("B", 2023, 600), ("B", 2023, 900),
              ("A", 2024, 0), ("A", 2024, 601)]
    df = _attempts(starts).sample(fraction=1.0, shuffle=True, seed=0)
    out = combine_sessions(df)
    assert out.height == df.height and out["traj"].to_list() == df["traj"].to_list()
    got = dict(out.select("traj", "session").unique().iter_rows())
    assert [got[i] for i in range(len(starts))] == (["A_2023_0"] * 3 + ["A_2023_1"] * 2 + ["B_2023_0"] * 4
                                                    + ["A_2024_0", "A_2024_1"])


def test_onset_time_interpolates_between_frames():
    t, obs = onset_time(0.3 * np.arange(10))  # crosses 1.0 at frame 10/3
    assert obs and t == pytest.approx(1 / 3)
    blip = np.array([0, 1.5, 0.2, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8])  # one-frame blip is not a run of 3
    assert onset_time(blip)[0] == pytest.approx(0.1 * (5 + 0.1 / 0.3))
    assert onset_time(blip, hold=1)[0] == pytest.approx(0.1 * (1 / 1.5))
    assert onset_time(0.3 * np.arange(10), thresh=2.0)[0] == pytest.approx(0.1 * 20 / 3)


def test_onset_time_flags_unobserved_and_never():
    t, obs = onset_time(np.array([1.2, 0.5, 0.2, 1.0, 2.0, 3.0]))  # already moving at the first frame
    assert np.isnan(t) and not obs
    t, obs = onset_time(np.array([0.1, 0.5, 0.9, 0.5, 0.2]))
    assert np.isnan(t) and obs
