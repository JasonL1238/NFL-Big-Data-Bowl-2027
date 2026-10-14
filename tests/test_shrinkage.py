import warnings

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from scipy.stats import spearmanr

from bdb27 import reliability
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


def _kept_split_half(plays, eb, value="value"):
    return reliability.split_half(plays.join(eb.select("nfl_id"), on="nfl_id", how="semi"), value, min_per_half=1)


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
    sh = _kept_split_half(reg, eb)
    assert check_prior(eb, sh)["ok"]
    bad = eb.with_columns(tau2=pl.lit(collapsed), weight=collapsed / (collapsed + pl.col("se") ** 2))
    with pytest.warns(UserWarning, match="disagrees"):
        res = check_prior(bad, sh)
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
    assert check_prior(eb, _kept_split_half(oe, eb, "oe"))["ok"]


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
    res = check_prior(eb, _kept_split_half(plays, eb))
    assert res["ok"] and res["n_players"] == 300
    assert res["mean_weight"] > res["implied_reliability"]  # why the mean weight itself is not the test
    clustered, _ = _plays(k=150, n_range=(20, 300), sigma2=2.0, per_game=8, game_var=0.5, seed=7)
    eb = eb_mean(clustered, "value", min_n=20)
    with pytest.warns(UserWarning):
        res = check_prior(eb, _kept_split_half(clustered, eb))
    assert res["implied_reliability"] > res["split_half_ci"][1]  # plays within a game are not independent


@pytest.mark.slow
def test_tau2_unbiased_and_check_prior_calibrated():
    tau2, ok = [], []
    for s in range(200):
        plays, _ = _plays(k=65, tau2=0.14, sigma2=2.07, mu=0.0, n_range=(10, 80), seed=100 + s)
        eb = eb_mean(plays, "value", min_n=10)
        tau2.append(eb["tau2"][0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ok.append(check_prior(eb, _kept_split_half(plays, eb))["ok"])
    tau2 = np.array(tau2)
    assert abs(tau2.mean() - 0.14) < 3 * tau2.std() / np.sqrt(len(tau2))
    assert 0.88 <= np.mean(ok) <= 1.0  # nominal 0.95
