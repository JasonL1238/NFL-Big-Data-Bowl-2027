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

@pytest.mark.parametrize("values, groups, icc", [
    (np.arange(5.0), np.arange(5), np.nan),                          # all singletons
    (np.ones(9), np.array([0, 0, 1, 1, 2, 2, 3, 4, 5]), np.nan),     # no variance: 0 / 0
    (np.array([0, 1, 0, 1, 0, 1.0]), np.array([0, 0, 1, 1, 2, 2]), -1.0),  # icc = -1 -> SB denominator 0
])
def test_icc1_degenerate_inputs_are_nan_without_warnings(values, groups, icc):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = icc1(values, groups)
    assert np.isnan(res["reliability"])
    assert np.isnan(res["icc"]) if np.isnan(icc) else res["icc"] == pytest.approx(icc)


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


# ---------- split_half / split_half_estimator ----------

def test_split_half_spearman_matches_the_normal_rank_correlation():
    res = split_half(_games(n_players=2000), "v", method="spearman")
    r = 1 / (1 + 4 / 30)  # Pearson r_half 0.88 -> Spearman (6 / pi) asin(r / 2) = 0.873
    assert abs(res["r_half"] - 6 / np.pi * np.arcsin(r / 2)) < 0.03
    assert res["reliability"] == pytest.approx(2 * res["r_half"] / (1 + res["r_half"]))


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

def _team_plays(team_sd, n_teams=300, per_team=4, plays=200, seed=7):
    rng = np.random.default_rng(seed)
    rows, pid = [], 0
    for t in range(n_teams):
        eff = rng.normal(0, team_sd)
        for _ in range(per_team):
            mu = eff + rng.normal()
            n = plays if pid % 10 else 10  # every 10th player has too few plays
            rows += [(pid, f"T{t % 32}", 2023 + t // 32, mu + rng.normal(0, 2)) for _ in range(n)]
            pid += 1
    return pl.DataFrame(rows, schema=["nfl_id", "team", "season", "v"], orient="row")


def test_team_share_recovers_the_team_variance_share():
    df = _team_plays(team_sd=1.0)
    res = team_share(df, "v", ["team", "season"])
    # team var 1, player var 1, mean noise 4 / 200 -> 1 / 2.02 = 0.495
    assert abs(res["team_share"] - 0.495) < 0.07
    assert res["n_teams"] == 300 and res["n_player_teams"] == 1200 - 120
    assert abs(team_share(_team_plays(team_sd=0.0), "v", ["team", "season"])["team_share"]) < 0.07


def test_team_share_counts_a_traded_player_once_per_team_and_needs_shared_teams():
    df = pl.DataFrame({"nfl_id": [1] * 40 + [2] * 20 + [3] * 20, "team_season": ["A"] * 20 + ["B"] * 40 + ["C"] * 20,
                       "v": np.arange(80.0)})
    res = team_share(df, "v", "team_season")
    assert res["n_player_teams"] == 4 and res["n_teams"] == 3 and res["n_shared"] == 1
    assert np.isnan(res["team_share"])  # < 3 team-seasons with >= 2 study players


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
