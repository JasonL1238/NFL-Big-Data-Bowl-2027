import warnings

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from scipy.stats import spearmanr

from bdb27 import io, reliability
from bdb27.shrinkage import check_prior, eb_mean, eb_rate, reml


def _plays(k=1000, tau2=0.25, sigma2=4.0, mu=1.0, n_range=(10, 300), seed=0, per_game=4, game_var=0.0):
    """Play-level values: theta_i ~ N(mu, tau2), value = theta_i + game effect + N(0, sigma2), n_i log-uniform
    in n_range, `per_game` consecutive plays per game. Returns (plays, theta indexed by nfl_id)."""
    rng = np.random.default_rng(seed)
    n = np.exp(rng.uniform(np.log(n_range[0]), np.log(n_range[1] + 1), k)).astype(int)
    theta = mu + rng.normal(0, np.sqrt(tau2), k)
    pid = np.repeat(np.arange(k), n)
    game = np.concatenate([np.arange(m) // per_game for m in n])
    noise = rng.normal(0, np.sqrt(game_var), (k, game.max() + 1))[pid, game] + rng.normal(0, np.sqrt(sigma2), n.sum())
    return pl.DataFrame({"nfl_id": pid, "game_id": game, "value": theta[pid] + noise}), theta


def _restricted_loglik(t, y, v):
    w = 1 / (v + t)
    m = np.sum(w * y) / np.sum(w)
    return -0.5 * (np.sum(np.log(v + t)) + np.log(np.sum(w)) + np.sum(w * (y - m) ** 2))


def test_reml_matches_brute_force_maximum_and_is_scale_free():
    rng = np.random.default_rng(0)
    v = rng.uniform(0.05, 1.0, 60)
    y = 2 + rng.normal(0, np.sqrt(0.3 + v))
    mu, tau2 = reml(y, v)
    grid = np.linspace(0, 2, 20001)
    assert tau2 == pytest.approx(grid[np.argmax([_restricted_loglik(t, y, v) for t in grid])], abs=2e-4)
    w = 1 / (v + tau2)
    assert mu == pytest.approx(np.sum(w * y) / np.sum(w))
    mu_s, tau2_s = reml(1e-3 * y, 1e-6 * v)
    assert tau2_s == pytest.approx(1e-6 * tau2, rel=1e-4) and mu_s == pytest.approx(1e-3 * mu, rel=1e-6)


def test_reml_returns_zero_when_means_vary_less_than_sampling_noise():
    rng = np.random.default_rng(1)
    v = rng.uniform(0.5, 1.0, 60)
    y = rng.normal(0, np.sqrt(0.2 * v))
    assert reml(y, v)[1] == 0.0


def test_eb_mean_recovers_known_prior_with_unequal_n():
    plays, _ = _plays()
    eb = eb_mean(plays, "value", min_n=10)
    assert eb.height == 1000 and eb["n"].min() >= 10
    assert eb["tau2"][0] == pytest.approx(0.25, abs=0.05)  # sampling SD 0.015 over seeds
    assert eb["prior_mean"][0] == pytest.approx(1.0, abs=0.06)
    sigma2 = (eb["se"] ** 2 * eb["n"]).to_numpy()
    assert np.allclose(sigma2, sigma2[0]) and sigma2[0] == pytest.approx(4.0, rel=0.02)
    assert np.allclose(eb["weight"], eb["tau2"] / (eb["tau2"] + eb["se"] ** 2))
    raw = plays.group_by("nfl_id").agg(pl.col("value").mean()).sort("nfl_id")["value"]
    assert np.allclose(eb["raw"], raw)


def test_shrunk_means_have_the_expected_lower_mse():
    plays, theta = _plays(seed=2)
    eb = eb_mean(plays, "value", min_n=10)
    t = theta[eb["nfl_id"].to_numpy()]
    mse_raw = np.mean((eb["raw"].to_numpy() - t) ** 2)
    mse_eb = np.mean((eb["shrunk"].to_numpy() - t) ** 2)
    v, w = eb["se"].to_numpy() ** 2, eb["weight"].to_numpy()
    assert mse_raw == pytest.approx(v.mean(), rel=0.2)  # SD 0.06 over seeds
    assert mse_eb == pytest.approx(np.mean(w * v), rel=0.2)  # posterior variance under the true prior
    assert mse_eb < 0.7 * mse_raw


def test_weights_increase_with_n_and_low_n_players_shrink_most():
    eb = eb_mean(_plays(seed=3)[0], "value", min_n=10).sort("n")
    w = eb["weight"].to_numpy()
    assert np.all(np.diff(w) >= 0) and w[0] < 0.5 < w[-1]
    mu = eb["prior_mean"][0]
    assert np.allclose(eb["shrunk"] - mu, w * (eb["raw"] - mu))


def _e002_scenario(seed=0):
    """66 regulars (10-80 events, tau2 0.14, sigma2 2.07: E002 WR HITCH decel) + 400 fringe players with
    1-3 events whose events vary less (0.5). E002 run 1 fitted an unweighted moment estimator to everyone."""
    reg, _ = _plays(k=66, tau2=0.14, sigma2=2.07, mu=0.0, n_range=(10, 80), seed=seed)
    rng = np.random.default_rng(seed + 1)
    n = rng.integers(1, 4, 400)
    pid = np.repeat(np.arange(400), n)
    fringe = pl.DataFrame({"nfl_id": pid + 10_000, "game_id": np.zeros(n.sum(), dtype=np.int64),
                           "value": rng.normal(0, np.sqrt(0.14), 400)[pid] + rng.normal(0, np.sqrt(0.5), n.sum())})
    return reg, pl.concat([reg, fringe])


def _moments_tau2_all_players(plays):
    g = plays.group_by("nfl_id").agg(m=pl.col("value").mean(), n=pl.len(),
                                     ss=((pl.col("value") - pl.col("value").mean()) ** 2).sum())
    sigma2 = g["ss"].sum() / (g["n"] - 1).sum()
    var_m = float(np.var(g["m"].to_numpy(), ddof=1))
    return max(var_m - float(np.mean(sigma2 / g["n"].to_numpy())), 0.01 * var_m)


def test_e002_collapse_scenario_low_n_players_do_not_touch_the_prior():
    reg, plays = _e002_scenario()
    collapsed = _moments_tau2_all_players(plays)
    assert collapsed < 0.02  # the scenario reproduces the E002 run-1 collapse
    eb = eb_mean(plays, "value", min_n=10)
    assert eb.height == 66 and eb["nfl_id"].max() < 10_000
    assert_frame_equal(eb, eb_mean(reg, "value", min_n=10), check_exact=False)  # fringe players have no influence
    assert eb["tau2"][0] == pytest.approx(0.14, abs=0.13)  # SD 0.04 over seeds; collapsed is < 0.02
    assert check_prior(eb, plays, "value")["ok"]
    bad = eb.with_columns(tau2=pl.lit(collapsed), weight=collapsed / (collapsed + pl.col("se") ** 2))
    with pytest.warns(UserWarning, match="disagrees"):
        res = check_prior(bad, plays, "value")
    assert not res["ok"] and res["implied_reliability"] < res["split_half_ci"][0]


def test_separate_prior_per_group():
    a, _ = _plays(k=600, tau2=0.1, sigma2=1.0, mu=0.0, seed=4)
    b, _ = _plays(k=600, tau2=0.6, sigma2=4.0, mu=3.0, seed=5)
    plays = pl.concat([a.with_columns(align=pl.lit("EDGE")), b.with_columns(align=pl.lit("INTERIOR"))])
    eb = eb_mean(plays, "value", group="align", min_n=10)
    assert eb.height == 1200 and eb.select("nfl_id", "align").is_unique().all()  # a player has one row per group
    fit = eb.group_by("align").agg(pl.first("tau2", "prior_mean"), sigma2=(pl.col("se") ** 2 * pl.col("n")).mean())
    edge, interior = (fit.filter(pl.col("align") == a).row(0, named=True) for a in ("EDGE", "INTERIOR"))
    assert edge["tau2"] == pytest.approx(0.1, abs=0.025) and interior["tau2"] == pytest.approx(0.6, abs=0.12)
    assert edge["prior_mean"] == pytest.approx(0.0, abs=0.06) and interior["prior_mean"] == pytest.approx(3.0, abs=0.12)
    assert edge["sigma2"] == pytest.approx(1.0, rel=0.03) and interior["sigma2"] == pytest.approx(4.0, rel=0.03)
    with pytest.raises(ValueError, match="need >= 3"):
        eb_mean(plays.filter((pl.col("align") == "EDGE") | (pl.col("nfl_id") < 2)), "value", group="align", min_n=10)


def _rate_plays(k=400, sd_delta=0.04, n_range=(30, 600), seed=0, per_game=10):
    """Binary events: each player has a context mix (expected p per play around his own ctx_i) and a true
    rate over expected delta_i; event ~ Bernoulli(p + delta_i)."""
    rng = np.random.default_rng(seed)
    n = np.exp(rng.uniform(np.log(n_range[0]), np.log(n_range[1] + 1), k)).astype(int)
    ctx, delta = rng.uniform(0.15, 0.45, k), rng.normal(0, sd_delta, k)
    pid = np.repeat(np.arange(k), n)
    p = ctx[pid] + rng.uniform(-0.1, 0.1, n.sum())
    plays = pl.DataFrame({"nfl_id": pid, "game_id": np.concatenate([np.arange(m) // per_game for m in n]),
                          "pressure": rng.random(n.sum()) < p + delta[pid], "p_hat": p})
    return plays, ctx, delta


def test_eb_rate_recovers_heterogeneous_rates_over_expected():
    plays, ctx, delta = _rate_plays()
    eb = eb_rate(plays, "pressure", "p_hat", min_n=30)
    assert eb["tau2"][0] == pytest.approx(0.04**2, abs=7e-4)  # SD ~ 2e-4 over seeds
    assert eb["prior_mean"][0] == pytest.approx(0.0, abs=0.009)
    per = plays.group_by("nfl_id").agg(rate=pl.col("pressure").mean(), e=pl.col("p_hat").mean(),
                                       v=(pl.col("p_hat") * (1 - pl.col("p_hat"))).sum() / pl.len() ** 2).sort("nfl_id")
    assert np.allclose(eb["raw"], per["rate"] - per["e"]) and np.allclose(eb["se"] ** 2, per["v"])
    d = delta[eb["nfl_id"].to_numpy()]
    assert np.mean((eb["shrunk"].to_numpy() - d) ** 2) < 0.6 * np.mean((eb["raw"].to_numpy() - d) ** 2)
    assert spearmanr(eb["weight"], eb["n"]).statistic > 0.95  # se also depends on each player's expected mix
    oe = plays.with_columns(oe=pl.col("pressure").cast(pl.Float64) - pl.col("p_hat"))
    assert check_prior(eb, oe, "oe")["ok"]
    with pytest.raises(ValueError, match="do not reproduce"):  # the raw event is not what eb_rate averaged
        check_prior(eb, oe.with_columns(pl.col("pressure").cast(pl.Float64)), "pressure")


def test_eb_rate_without_expected_uses_the_base_rate_and_keeps_context_variance():
    plays, ctx, delta = _rate_plays(seed=1)
    eb = eb_rate(plays, "pressure", min_n=30)
    assert np.allclose(eb["expected"], plays["pressure"].mean())
    assert eb["tau2"][0] == pytest.approx(np.var(ctx + delta), abs=0.0015)  # context is now part of the trait
    assert_frame_equal(eb_rate(plays.with_columns(pl.col("pressure").cast(pl.Int8)), "pressure", min_n=30), eb,
                       check_exact=False)  # 0/1 integers work like booleans
    big = eb_rate(plays, "pressure", min_n=300)  # base rate and prior come from the kept players only
    kept = plays.filter(pl.len().over("nfl_id") >= 300)
    assert big["n"].min() >= 300 and np.allclose(big["expected"], kept["pressure"].mean())


def test_check_prior_well_specified_and_game_clustered():
    plays, _ = _plays(k=300, seed=6)
    eb = eb_mean(plays, "value", min_n=10)
    res = check_prior(eb, plays, "value")
    assert res["ok"] and res["n_players"] == res["n_cells"] == 300
    assert res["mean_weight"] > res["implied_reliability"]  # why the mean weight itself is not the test
    clustered, _ = _plays(k=150, n_range=(20, 300), sigma2=2.0, per_game=8, game_var=0.5, seed=7)
    eb = eb_mean(clustered, "value", min_n=20)
    with pytest.warns(UserWarning):
        res = check_prior(eb, clustered, "value")
    assert res["implied_reliability"] > res["split_half_ci"][1]  # plays within a game are not independent
    assert check_prior(eb_mean(clustered, "value", min_n=20, cluster="game_id"), clustered, "value")["ok"]


def _spanning_plays(seed, k=400, tau2=0.01, sigma2=0.12, n_range=(60, 600)):
    """Players whose plays split 20-80% between two alignments; one theta per player, alignment offset only."""
    rng = np.random.default_rng(seed)
    n = np.exp(rng.uniform(np.log(n_range[0]), np.log(n_range[1]), k)).astype(int)
    pid = np.repeat(np.arange(k), n)
    edge = rng.random(n.sum()) < rng.uniform(0.2, 0.8, k)[pid]
    value = 0.8 + 0.05 * edge + rng.normal(0, np.sqrt(tau2), k)[pid] + rng.normal(0, np.sqrt(sigma2), n.sum())
    return pl.DataFrame({"nfl_id": pid, "game_id": np.concatenate([np.arange(m) // 20 for m in n]),
                         "align": np.where(edge, "EDGE", "INT"), "value": value})


def test_check_prior_splits_each_cell_when_players_span_groups():
    plays = _spanning_plays(0)
    eb = eb_mean(plays, "value", group="align", min_n=30)
    res = check_prior(eb, plays, "value", group="align")
    assert res["ok"] and res["n_cells"] == eb.height > res["n_players"] == 400
    # The old recipe split each player's plays pooled over both alignments: a longer mean, so a higher
    # reliability than the per-alignment cells the prior describes (it failed half of correct fits at k = 110).
    kept = plays.join(eb.select("nfl_id", "align"), on=["nfl_id", "align"], how="semi")
    assert reliability.split_half(kept, "value", min_per_half=1)["reliability"] > res["split_half_ci"][1]


def test_eb_rate_drops_nan_events_and_expectations():
    plays, _, _ = _rate_plays(seed=3)
    bad = pl.int_range(pl.len()).is_in([5, 500, 5000])
    nan_p = plays.with_columns(p_hat=pl.when(bad).then(float("nan")).otherwise("p_hat"))
    assert_frame_equal(eb_rate(nan_p, "pressure", "p_hat", min_n=30),
                       eb_rate(plays.filter(~bad), "pressure", "p_hat", min_n=30), check_exact=False)
    nan_y = plays.with_columns(pressure=pl.when(bad).then(float("nan")).otherwise(pl.col("pressure").cast(pl.Float64)))
    eb = eb_rate(nan_y, "pressure", min_n=30)
    assert eb["shrunk"].is_finite().all() and eb["events"].sum() == plays.filter(~bad)["pressure"].sum()
    assert_frame_equal(eb, eb_rate(plays.filter(~bad), "pressure", min_n=30), check_exact=False)


def test_identical_calls_give_identical_results():
    plays, (rate, _, _) = _spanning_plays(1, k=200), _rate_plays(seed=2)
    for fit in (lambda: eb_mean(plays, "value", group="align", min_n=30),
                lambda: eb_mean(plays, "value", group="align", min_n=30, cluster="game_id"),
                lambda: eb_rate(rate, "pressure", "p_hat", min_n=30, cluster="game_id")):
        first = fit()
        for _ in range(10):  # group_by order is random; unsorted cells gave a new tau2 on most calls
            assert_frame_equal(fit(), first, check_exact=True)


def test_counts_above_65535_do_not_overflow():
    n = 70_000  # polars counts are UInt32, and n**2 wraps around above 65,535
    plays = pl.DataFrame({"nfl_id": np.repeat(np.arange(3), n), "game_id": np.tile(np.arange(n) // 50, 3),
                          "y": np.tile(np.arange(n) % 2 == 0, 3)})  # alternating within games: c floored at 0
    for cluster in (None, "game_id"):
        assert np.allclose(eb_rate(plays, "y", min_n=10, cluster=cluster)["se"], np.sqrt(0.25 / n))
        assert np.allclose(eb_mean(plays.with_columns(pl.col("y").cast(pl.Float64)), "y", min_n=10,
                                   cluster=cluster)["se"], np.sqrt(0.25 / (n - 1)))


def _game_plays(k=600, tau2=0.05, sig_e=1.0, sig_g=0.3, seed=0):
    """theta_i ~ N(0, tau2); 3-40 games per player, 1-15 plays per game; value = theta_i + game effect N(0, sig_g)
    + N(0, sig_e). Two plays in one game have covariance sig_g, so v_i = (sig_e + sig_g) / n + sig_g (s2 - n) / n^2,
    s2 = sum_g m_g^2. Returns (plays, theta, s2 indexed by nfl_id)."""
    rng = np.random.default_rng(seed)
    games = rng.integers(3, 41, k)
    pid, gid = np.repeat(np.arange(k), games), np.concatenate([np.arange(g) for g in games])
    m = rng.integers(1, 16, len(pid))
    theta = rng.normal(0, np.sqrt(tau2), k)
    game_value = theta[pid] + rng.normal(0, np.sqrt(sig_g), len(pid))
    plays = pl.DataFrame({"nfl_id": np.repeat(pid, m), "game_id": np.repeat(gid, m),
                          "value": np.repeat(game_value, m) + rng.normal(0, np.sqrt(sig_e), m.sum())})
    return plays, theta, np.bincount(pid, weights=m**2)


def _z2(eb, truth):
    """Mean squared z-score of the raw means; 1 when the sampling variances are right."""
    return np.mean((eb["raw"].to_numpy() - truth[eb["nfl_id"].to_numpy()]) ** 2 / eb["se"].to_numpy() ** 2)


def test_clustered_variance_matches_a_known_game_effect():
    plays, theta, s2 = _game_plays()
    eb = eb_mean(plays, "value", min_n=10, cluster="game_id")
    plain = eb_mean(plays, "value", min_n=10)
    n, s2 = eb["n"].to_numpy(), s2[eb["nfl_id"].to_numpy()]
    v_true = 1.3 / n + 0.3 * (s2 - n) / n**2
    assert np.mean(eb["se"].to_numpy() ** 2 / v_true) == pytest.approx(1.0, abs=0.05)  # SD 0.016 over seeds
    assert _z2(eb, theta) == pytest.approx(1.0, abs=0.2)  # SD 0.064 over seeds
    assert _z2(plain, theta) > 2.5  # ~3.2 (min 2.7 over 40 seeds): sigma2 / n ignores the game effect
    assert eb["tau2"][0] == pytest.approx(0.05, abs=0.015)  # SD 0.005 over seeds
    assert check_prior(eb, plays, "value")["ok"]
    with pytest.warns(UserWarning, match="disagrees"):
        check_prior(plain, plays, "value")


def test_cluster_reduces_to_the_default_without_within_game_covariance():
    one, _ = _plays(k=300, per_game=1, seed=8)  # one play per game: nothing to cluster
    assert_frame_equal(eb_mean(one, "value", min_n=10, cluster="game_id"), eb_mean(one, "value", min_n=10),
                       check_exact=False)
    rate, _, _ = _rate_plays(per_game=1, seed=8)
    assert_frame_equal(eb_rate(rate, "pressure", "p_hat", min_n=30, cluster="game_id"),
                       eb_rate(rate, "pressure", "p_hat", min_n=30), check_exact=False)
    pairs, _ = _plays(k=300, per_game=2, seed=9)  # the two plays of a game anti-correlated: covariance floored at 0
    sign = pl.int_range(pl.len()).over("nfl_id", "game_id") * 2 - 1
    pairs = (pairs.with_columns(u=pl.Series(np.random.default_rng(9).normal(0, 1, pairs.height)))
             .with_columns(pl.col("value") + sign * pl.col("u").first().over("nfl_id", "game_id")))
    assert_frame_equal(eb_mean(pairs, "value", min_n=10, cluster="game_id"), eb_mean(pairs, "value", min_n=10),
                       check_exact=False)
    iid, _ = _plays(seed=10)  # 4 plays per game, no game effect: c is ~0
    ratio = eb_mean(iid, "value", min_n=10, cluster="game_id")["se"] / eb_mean(iid, "value", min_n=10)["se"]
    assert ratio.min() >= 1 and ratio.max() < 1.03


def _rate_game_plays(k=400, sd_delta=0.04, sig_g=0.004, seed=0):
    """Binary events, 5-17 games per player, 2-30 plays per game, p_hat ~ U(0.2, 0.4) per play, and a game effect
    N(0, sig_g) on the probability: two plays in one game have covariance sig_g."""
    rng = np.random.default_rng(seed)
    games = rng.integers(5, 18, k)
    pid, gid = np.repeat(np.arange(k), games), np.concatenate([np.arange(g) for g in games])
    m = rng.integers(2, 31, len(pid))
    delta, gamma = rng.normal(0, sd_delta, k), rng.normal(0, np.sqrt(sig_g), len(pid))
    p = rng.uniform(0.2, 0.4, m.sum())
    y = rng.random(m.sum()) < p + np.repeat(delta[pid] + gamma, m)
    return pl.DataFrame({"nfl_id": np.repeat(pid, m), "game_id": np.repeat(gid, m), "pressure": y, "p_hat": p}), delta


def test_eb_rate_clustered_variance_matches_a_known_game_effect():
    plays, delta = _rate_game_plays()
    eb = eb_rate(plays, "pressure", "p_hat", min_n=30, cluster="game_id")
    plain = eb_rate(plays, "pressure", "p_hat", min_n=30)
    s2 = plays.group_by("nfl_id", "game_id").len().group_by("nfl_id").agg(s2=(pl.col("len") ** 2).sum())
    j = eb.join(s2, on="nfl_id", maintain_order="left").join(plain.select("nfl_id", v0=pl.col("se") ** 2), on="nfl_id",
                                                              maintain_order="left")
    c = ((j["se"] ** 2 - j["v0"]) * j["n"] ** 2 / (j["s2"] - j["n"])).to_numpy()  # the pooled covariance
    assert np.allclose(c, c[0]) and c[0] == pytest.approx(0.004, abs=0.0015)  # SD 0.0005 over seeds
    assert _z2(eb, delta) == pytest.approx(1.0, abs=0.25)  # SD 0.075 over seeds
    assert _z2(plain, delta) > 1.15  # 1.46 on average, SD ~0.1
    oe = plays.with_columns(oe=pl.col("pressure").cast(pl.Float64) - pl.col("p_hat"))
    assert check_prior(eb, oe, "oe")["ok"]


@pytest.mark.skipif(not (io.PROCESSED / "player_play.parquet").exists(), reason="needs the Parquet data cache")
def test_real_dl_get_off_prior_agrees_with_split_half_once_games_are_clustered():
    """Build-report flag: the DL get-off prior disagreed with split-half (pooled implied 0.95 vs 0.90 [0.84, 0.93];
    per alignment 0.93 vs [0.82, 0.91]) because a player's get-offs within one game are correlated."""
    reg = io.load("games").filter(pl.col("season_type") == "REG").select("game_id")
    pos = io.load("combine_results").select("nfl_id", position="combine_position")
    dl = (pl.scan_parquet(io.PROCESSED / "player_play.parquet")
          .select("nfl_id", "game_id", "play_id", "player_get_off", "lined_up_position")
          .join(reg.lazy(), on="game_id", how="semi").join(pos.lazy(), on="nfl_id")
          .filter((pl.col("position") == "DL") & pl.col("player_get_off").is_not_null()
                  & pl.col("lined_up_position").is_in(["EDGE", "INTERIOR_LINE"])).collect())
    for group in (None, "lined_up_position"):
        plain = eb_mean(dl, "player_get_off", group=group, min_n=30)
        with pytest.warns(UserWarning, match="disagrees"):
            assert not check_prior(plain, dl, "player_get_off", group=group)["ok"]
        eb = eb_mean(dl, "player_get_off", group=group, min_n=30, cluster="game_id")
        assert check_prior(eb, dl, "player_get_off", group=group)["ok"]
        assert (eb["se"] ** 2).mean() > 1.5 * (plain["se"] ** 2).mean()  # about doubles


@pytest.mark.slow
def test_tau2_unbiased_and_check_prior_calibrated():
    tau2, ok, ok_cl, ok_cells = [], [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in range(200):
            plays, _ = _plays(k=65, tau2=0.14, sigma2=2.07, mu=0.0, n_range=(10, 80), seed=100 + s)
            eb = eb_mean(plays, "value", min_n=10)
            tau2.append(eb["tau2"][0])
            ok.append(check_prior(eb, plays, "value")["ok"])
            plays, _, _ = _game_plays(k=80, seed=100 + s)
            ok_cl.append(check_prior(eb_mean(plays, "value", min_n=10, cluster="game_id"), plays, "value")["ok"])
            if s < 100:
                plays = _spanning_plays(100 + s, k=110)
                eb = eb_mean(plays, "value", group="align", min_n=30)
                ok_cells.append(check_prior(eb, plays, "value", group="align")["ok"])
    tau2 = np.array(tau2)
    assert abs(tau2.mean() - 0.14) < 3 * tau2.std() / np.sqrt(len(tau2))
    for rate in (ok, ok_cl, ok_cells):  # nominal 0.95
        assert 0.88 <= np.mean(rate) <= 1.0
