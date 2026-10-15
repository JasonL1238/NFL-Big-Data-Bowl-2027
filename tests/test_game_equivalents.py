"""E012 Combine Game Equivalents (bdb27.probes.h_game_equivalents): estimator recovery on simulated players with known
tau2, sigma2 and a planted Combine signal; the blend weight is the Bayesian posterior mean; the prospective machinery
(per-play constants, expectation refits, Combine composites, the fit and the forecasts) cannot see test data."""

import numpy as np
import polars as pl
import pytest
from statsmodels.stats.multitest import multipletests

from bdb27 import reliability
from bdb27.probes import h_game_equivalents as H

SETS = {"C0": None, "C1": ["x1"], "C2": ["x1", "x2"], "C3": ["x1", "x2", "x3"]}
SHARES = (0.2, 0.2, 0.2)  # share of tau2 explained by each of x1, x2, x3 -> true R2 0.2 / 0.4 / 0.6


def simulate(n=3000, tau2=1.0, sigma2=4.0, shares=SHARES, games=(3, 15), seed=0,
             years=(2023, 2024, 2025), year_fx=(0.0, 0.5, -0.3)) -> tuple[pl.DataFrame, pl.DataFrame, np.ndarray]:
    """Players with theta = year effect + sum_j sqrt(share_j tau2) x_j + u (Var u = (1 - sum shares) tau2) and
    n_i games with value = theta + N(0, sigma2). Returns games, players, theta."""
    rng = np.random.default_rng(seed)
    yi = rng.integers(0, len(years), n)
    X = rng.normal(size=(n, len(shares)))
    theta = (np.asarray(year_fx)[yi] + X @ np.sqrt(np.asarray(shares) * tau2)
             + rng.normal(0, np.sqrt(tau2 * (1 - sum(shares))), n))
    n_i = rng.integers(games[0], games[1] + 1, n)
    pid = np.repeat(np.arange(n), n_i)
    games_df = pl.DataFrame({"nfl_id": pid, "game_id": np.concatenate([np.arange(k) for k in n_i]),
                             "value": theta[pid] + rng.normal(0, np.sqrt(sigma2), len(pid))})
    players = pl.DataFrame({"nfl_id": np.arange(n), "draft_year": np.asarray(years)[yi],
                            **{f"x{j + 1}": X[:, j] for j in range(len(shares))}})
    return games_df, players, theta


# ---------------------------------------------------------------- variance components

def test_variance_components_single_stratum_is_icc1():
    games, _, _ = simulate(n=300, seed=1)
    vc = H.variance_components(games["value"].to_numpy(), games["nfl_id"].to_numpy())
    ref = reliability.icc1(games["value"].to_numpy(), games["nfl_id"].to_numpy())
    assert vc["icc"] == pytest.approx(ref["icc"], rel=1e-10)


def test_variance_components_remove_year_effects():
    games, players, _ = simulate(n=3000, year_fx=(0.0, 2.0, -2.0), seed=2)
    g = games.join(players.select("nfl_id", "draft_year"), on="nfl_id")
    strat = H.variance_components(g["value"].to_numpy(), g["nfl_id"].to_numpy(), g["draft_year"].to_numpy())
    pooled = H.variance_components(g["value"].to_numpy(), g["nfl_id"].to_numpy())
    assert strat["tau2"] == pytest.approx(1.0, abs=0.1) and strat["sigma2"] == pytest.approx(4.0, rel=0.03)
    assert pooled["tau2"] > 2.5  # without strata the large class effects count as player variance


def test_k_game_mean_explains_k_over_k_plus_lambda():
    """The premise of CGE: a k-game mean's squared correlation with theta is k / (k + lambda)."""
    rng = np.random.default_rng(3)
    tau2, sigma2, k, n = 1.0, 4.0, 3, 200_000
    theta = rng.normal(0, np.sqrt(tau2), n)
    ybar = theta + rng.normal(0, np.sqrt(sigma2 / k), n)
    assert np.corrcoef(theta, ybar)[0, 1] ** 2 == pytest.approx(k / (k + sigma2 / tau2), abs=0.005)


# ---------------------------------------------------------------- estimator recovery

def test_estimator_recovers_lambda_r2_and_cge():
    games, players, _ = simulate(n=3000, seed=4)
    td = H.trait_data(games, players, SETS)
    est = H.estimate(td, n_repeats=2, seed=0)
    lam = 4.0
    assert est["tau2"] == pytest.approx(1.0, abs=0.08)
    assert est["sigma2"] == pytest.approx(4.0, rel=0.03)
    assert est["lambda"] == pytest.approx(lam, rel=0.1)
    for s, r2 in zip(("C1", "C2", "C3"), (0.2, 0.4, 0.6)):
        assert est["r2"][s] == pytest.approx(r2, abs=0.05), s
        assert est["cge"][s] == pytest.approx(lam * r2 / (1 - r2), rel=0.2), s
    assert est["r2"]["C0"] == pytest.approx(0.0, abs=0.01) and est["cge"]["C0"] == pytest.approx(0.0, abs=0.05)
    assert est["delta"]["C3-C2"] == pytest.approx(lam * (0.6 / 0.4 - 0.4 / 0.6), rel=0.3)
    # reliability of the player means as used in R2_true
    n_i = np.bincount(td.g_player)
    assert est["rel"] == pytest.approx(1 / (1 + np.mean(4.0 / n_i)), abs=0.03)


def test_noise_feature_gets_no_games():
    games, players, _ = simulate(n=1500, shares=(0.0, 0.0, 0.0), seed=5)
    est = H.estimate(H.trait_data(games, players, SETS), n_repeats=2)
    assert all(est["cge"][s] < 0.15 for s in SETS)


def test_bootstrap_interval_covers_truth():
    games, players, _ = simulate(n=400, seed=6)
    td = H.trait_data(games, players, {"C0": None, "C2": ["x1", "x2"]})
    b = H.bootstrap(td, n_boot=40, seed=0, n_jobs=1, n_repeats=1)
    ci = H.percentile_ci(b["cge_C2"].to_numpy())
    assert ci["lo"] < 4 * 0.4 / 0.6 < ci["hi"]


def test_resample_and_group_folds_keep_copies_together():
    games, players, _ = simulate(n=50, seed=7)
    td = H.trait_data(games, players, SETS)
    ids = np.random.default_rng(0).integers(0, 50, 50)
    rt = H.resample(td, ids)
    for j in (0, 10, 49):  # copy j carries exactly the games of original player ids[j]
        assert np.allclose(np.sort(rt.g_value[rt.g_player == j]), np.sort(td.g_value[td.g_player == ids[j]]))
    for _, te in H.group_folds(ids, n_repeats=3, seed=1):
        assert not set(ids[te]) & set(ids[~te])


def test_design_dummies_and_dummy_means():
    p = pl.DataFrame({"draft_year": [2024, 2023, 2024], "a": [1.0, None, 3.0]})
    X = H.design(p, ["a"], [2023, 2024])
    assert np.array_equal(X[:, 0], [1, 0, 1]) and np.isnan(X[1, 1])
    Xm = H.design(p, ["a"], [2023, 2024], dummy_means=np.array([0.4]))
    assert np.allclose(Xm[:, 0], 0.4)
    assert H.design(p, None, [2023]) is None


# ---------------------------------------------------------------- blend rule

def test_blend_weight_is_bayesian_posterior_mean():
    rng = np.random.default_rng(8)
    for _ in range(50):
        tau2, lam, r2, k = rng.uniform(0.1, 3), rng.uniform(0.2, 10), rng.uniform(0, 0.9), rng.integers(1, 17)
        sigma2, v0 = lam * tau2, (1 - r2) * tau2
        mu, ybar = rng.normal(size=2)
        post_mean = (mu / v0 + k * ybar / sigma2) / (1 / v0 + k / sigma2)
        assert H.blend(mu, ybar, k, lam, r2) == pytest.approx(post_mean, rel=1e-10, abs=1e-12)
        assert H.posterior_var(k, tau2, sigma2, r2) == pytest.approx(1 / (1 / v0 + k / sigma2), rel=1e-12)
    assert H.blend(0.3, np.nan, 0, 2.0, 0.5) == pytest.approx(0.3)  # k = 0: the prior
    # the weight equals k / (k + lambda + CGE): the prior is worth lambda + CGE pseudo-games
    assert H.blend_weight(3, 4.0, 0.6) == pytest.approx(3 / (3 + 4.0 + H.cge(4.0, 0.6)))
    # amendment A1: the prereg's literal weight k / (k + lambda (1 - R2)) is NOT the posterior weight
    assert H.blend_weight(3, 4.0, 0.6, literal=True) != pytest.approx(float(H.blend_weight(3, 4.0, 0.6)))


def test_blend_weight_matches_regression_of_theta_on_data():
    """Monte Carlo: E[theta | mu, ybar_k] is linear with slope w_k on (ybar_k - mu)."""
    rng = np.random.default_rng(9)
    tau2, sigma2, r2, k, n = 1.0, 3.0, 0.35, 2, 400_000
    mu = rng.normal(0, np.sqrt(r2 * tau2), n)  # the forecast explains R2 of tau2
    theta = mu + rng.normal(0, np.sqrt((1 - r2) * tau2), n)
    ybar = theta + rng.normal(0, np.sqrt(sigma2 / k), n)
    slope = np.polyfit(ybar - mu, theta - mu, 1)[0]
    assert slope == pytest.approx(float(H.blend_weight(k, sigma2 / tau2, r2)), abs=0.01)


# ---------------------------------------------------------------- prospective machinery: no leakage

def test_center_and_winsorize_use_ref_rows_only():
    rng = np.random.default_rng(10)
    d = pl.DataFrame({"g": rng.choice(["a", "b"], 200), "season": rng.choice([2023, 2024, 2025], 200),
                      "v": rng.normal(size=200)})
    ref = pl.col("season") <= 2024
    base = H.center(H.winsorize(d, "v", 0.05, ref), "v", ["g"], ref, add_overall=True)
    bumped = d.with_columns(pl.when(pl.col("season") == 2025).then(pl.col("v") * 50 + 7).otherwise(pl.col("v")))
    out = H.center(H.winsorize(bumped, "v", 0.05, ref), "v", ["g"], ref, add_overall=True)
    tr = (d["season"] <= 2024).to_numpy()
    assert np.allclose(base["value"].to_numpy()[tr], out["value"].to_numpy()[tr])
    # the same constants shift the 2025 rows: their centered values move only through their own (clipped) values
    lo, hi = d.filter(ref)["v"].quantile(0.05), d.filter(ref)["v"].quantile(0.95)
    assert out.filter(pl.col("season") == 2025)["v"].is_between(lo, hi).all()
    leaky = H.center(bumped, "v", ["g"], None)
    assert not np.allclose(H.center(d, "v", ["g"], None)["value"].to_numpy()[tr], leaky["value"].to_numpy()[tr])


def _soe_frame(n=900, seed=11) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    cats = {c: rng.choice(["p", "q", "r"], n) for c in H.SOE_CAT if c != "route_ran"}
    return pl.DataFrame({
        **cats, "route_ran": rng.choice(["HITCH", "OUT", "IN", "GO"], n), "nfl_id": rng.integers(0, 30, n),
        "season": rng.choice([2023, 2024, 2025], n), "dropback_duration": rng.uniform(1, 4, n),
        "down": rng.integers(1, 5, n), "yards_to_go": rng.integers(1, 15, n), "quarter": rng.integers(1, 5, n),
        "yardline_side": rng.choice(["A", "B"], n), "possession_team": rng.choice(["A", "B"], n),
        "yardline_number": rng.integers(1, 50, n), "separation_at_pass_forward": rng.gamma(2, 1.5, n)})


def test_expectation_refits_ignore_test_outcomes():
    tr, te = pl.col("season") <= 2024, pl.col("season") == 2025
    d = _soe_frame()
    a = H.soe_prospective(d, tr, te)
    bump = pl.when(pl.col("season") == 2025).then(pl.col("separation_at_pass_forward") + 9).otherwise(
        pl.col("separation_at_pass_forward"))
    b = H.soe_prospective(d.with_columns(separation_at_pass_forward=bump), tr, te)
    assert np.allclose(a["sep_expected"].to_numpy(), b["sep_expected"].to_numpy())
    c = H.soe_prospective(d.with_columns(separation_at_pass_forward=pl.col("separation_at_pass_forward") + 9 * (
        pl.col("season") == 2023)), tr, te)
    assert not np.allclose(a["sep_expected"].to_numpy(), c["sep_expected"].to_numpy())  # sanity: train data matter

    rng = np.random.default_rng(12)
    n = 900
    r = pl.DataFrame({"season": rng.choice([2023, 2024, 2025], n), "align": rng.choice(["s", "w"], n),
                      "down": rng.integers(1, 5, n).astype(float), "game_id": rng.integers(0, 60, n),
                      "rec_yards0": rng.exponential(3, n)}).with_columns(
        defense_team_season=pl.lit("KC_") + pl.col("season").cast(pl.String))
    cat, num = ["season", "defense_team_season", "align"], ["down"]
    a = H.yprr_prospective(r, cat, num, tr, te)
    b = H.yprr_prospective(r.with_columns(rec_yards0=pl.col("rec_yards0") + 20 * (pl.col("season") == 2025)),
                           cat, num, tr, te)
    assert np.allclose(a["yprr_exp"].to_numpy(), b["yprr_exp"].to_numpy())
    assert a.filter(pl.col("defense_team_season").str.ends_with("2025")).height == 0  # test rows mapped to 2024

    o = pl.DataFrame({"dropback_duration": rng.uniform(1, 5, n), "lined_up_position": rng.choice(["T", "G", "C"], n),
                      "rush2": rng.integers(0, 2, n).astype(np.int8), "pa": rng.random(n) < 0.2,
                      "season": rng.choice([2023, 2024, 2025], n), "pressure": (rng.random(n) < 0.1).astype(float)})
    a = H.ol_pressure_prospective(o, tr, te)
    b = H.ol_pressure_prospective(o.with_columns(pressure=pl.when(pl.col("season") == 2025).then(1.0)
                                                 .otherwise(pl.col("pressure"))), tr, te)
    assert np.allclose(a["p_hat"].to_numpy(), b["p_hat"].to_numpy())


def test_combine_composites_use_ref_players_only():
    rng = np.random.default_rng(13)
    n = 400
    ev = pl.DataFrame({"nfl_id": rng.integers(0, 80, n), "drill": rng.choice(["d1", "d2", "d3"], n),
                       "family": "turn", "found": True, "retention": rng.normal(0.6, 0.1, n)})
    ref = list(range(50))
    a = H.route_composite(ev, "retention", "turn", ref)
    b = H.route_composite(ev.with_columns(pl.when(pl.col("nfl_id") >= 50).then(pl.col("retention") + 5)
                                          .otherwise(pl.col("retention"))), "retention", "turn", ref)
    m = a.join(b, on="nfl_id", suffix="_b").filter(pl.col("nfl_id") < 50)
    assert m.height > 20 and np.allclose(m["comp"], m["comp_b"])
    bp = pl.DataFrame({"nfl_id": rng.integers(0, 80, n), "drill_name": rng.choice(["W_DRILL", "BOX_DRILL"], n),
                       "bp_peak": rng.normal(5, 0.5, n)})
    a = H.backpedal_composite(bp, ["W_DRILL", "BOX_DRILL"], ref)
    b = H.backpedal_composite(bp.with_columns(pl.when(pl.col("nfl_id") >= 50).then(pl.col("bp_peak") * 3)
                                              .otherwise(pl.col("bp_peak"))), ["W_DRILL", "BOX_DRILL"], ref)
    m = a.join(b, on="nfl_id", suffix="_b").filter(pl.col("nfl_id") < 50)
    assert np.allclose(m["cmb_backpedal_speed"], m["cmb_backpedal_speed_b"])


def _prospective_data(seed=14):
    games, players, _ = simulate(n=900, seed=seed, years=(2023, 2024, 2025), games=(4, 16))
    season = (players["draft_year"].to_numpy()[games["nfl_id"].to_numpy()]
              + (games["game_id"].to_numpy() >= 8) * 1)  # first 8 games in the rookie season, then the next
    games = games.with_columns(season=pl.Series(np.minimum(season, 2025)), game_id=pl.col("game_id") + 1000 * season)
    return games, players


def test_prospective_fit_and_priors_ignore_test_data():
    games, players = _prospective_data()
    tr, te = H.split_prospective(games, players)
    assert tr["season"].max() <= 2024 and te["season"].unique().to_list() == [2025]
    assert set(te["nfl_id"].to_list()) <= set(players.filter(pl.col("draft_year") == 2025)["nfl_id"].to_list())
    fit = H.fit_prospective(tr, players, SETS, n_repeats=2)
    assert fit.levels == [2023, 2024] and set(fit.train_ids) <= set(tr["nfl_id"].to_list())
    test_players = players.filter(pl.col("draft_year") == 2025)
    pri = H.prior_means(fit, test_players)
    # any change to 2025-season data (outcomes of everyone, incl. older classes) leaves the fit untouched
    bumped = games.with_columns(pl.when(pl.col("season") == 2025).then(pl.col("value") * 10 + 3)
                                .otherwise(pl.col("value")))
    tr2, te2 = H.split_prospective(bumped, players)
    fit2 = H.fit_prospective(tr2, players, SETS, n_repeats=2)
    assert fit2.vc == fit.vc and fit2.r2 == fit.r2 and fit2.y_mean == fit.y_mean
    pri2 = H.prior_means(fit2, test_players)
    assert all(np.allclose(pri[s], pri2[s]) for s in SETS)
    # test players' Combine values do not enter the fit either
    fit3 = H.fit_prospective(tr, players.with_columns(
        x1=pl.when(pl.col("draft_year") == 2025).then(99.0).otherwise(pl.col("x1"))), SETS, n_repeats=2)
    assert fit3.r2 == fit.r2
    # the new class is placed at the training-average class level
    assert np.allclose(fit.dummy_means, [np.mean(players.filter(pl.col("nfl_id").is_in(fit.train_ids))
                                                 ["draft_year"].to_numpy() == 2024)])


def test_in_season_forecast_uses_only_first_k_games():
    games, players = _prospective_data(seed=15)
    tr, te = H.split_prospective(games, players)
    fit = H.fit_prospective(tr, players, SETS, n_repeats=1)
    tp = players.filter(pl.col("draft_year") == 2025)
    f = H.in_season_forecasts(fit, te, tp)
    late = te.sort("nfl_id", "game_id").with_columns(j=pl.int_range(1, pl.len() + 1).over("nfl_id"))
    late = late.with_columns(value=pl.when(pl.col("j") > 2).then(pl.col("value") + 5).otherwise(pl.col("value")))
    g = H.in_season_forecasts(fit, late.drop("j"), tp)
    for k in (1, 2):
        a, b = f.filter(pl.col("k") == k), g.filter(pl.col("k") == k)
        assert np.allclose(a["blend_C2"], b["blend_C2"]) and np.allclose(a["ybar"], b["ybar"])
    a, b = f.filter(pl.col("k") == 2), g.filter(pl.col("k") == 2)
    assert np.allclose(a["target"] + 5, b["target"])  # the target is games k+1..end only
    assert f.filter(pl.col("k") == 3)["m"].min() >= 3  # >= k + 3 games


# ---------------------------------------------------------------- inference helpers

def test_player_diffs_sum_to_pooled_difference_and_bootstrap_centres():
    rng = np.random.default_rng(16)
    rows = [(i, t, rng.normal(1, 1), rng.normal(1.2, 1)) for i in range(60) for t in ("T1", "T2", "T10")
            if rng.random() < (0.9 if t != "T10" else 0.2)]
    df = pl.DataFrame(rows, schema=["nfl_id", "trait", "a", "b"], orient="row")
    _, d = H.player_diffs(df, "a", "b")
    assert d.sum() == pytest.approx(H.pooled_mean(df, "a") - H.pooled_mean(df, "b"), rel=1e-10)
    bs = H.pooled_bootstrap(df, ["a", "b"], n_boot=4000, seed=0)
    assert np.mean(bs["a"] - bs["b"]) == pytest.approx(d.sum(), abs=0.05)


def test_sign_flip_is_calibrated_and_detects_shift():
    rng = np.random.default_rng(17)
    p = np.array([H.sign_flip_p(rng.normal(size=40), n_flips=2000, seed=s) for s in range(300)])
    assert 0.02 <= np.mean(p < 0.05) <= 0.09 and 0.42 <= p.mean() <= 0.58
    assert H.sign_flip_p(rng.normal(-0.8, 1, 40), n_flips=5000) < 0.01
    assert H.sign_flip_p(rng.normal(0.8, 1, 40), n_flips=5000) > 0.9  # one-sided: the wrong direction is not support


def test_bh_matches_statsmodels():
    p = [0.01, 0.04, 0.03, 0.2, 0.0005]
    assert np.allclose(H.bh(p), multipletests(p, method="fdr_bh")[1])


def test_crossover_helpers():
    r = np.array([0.3, 0.45, 0.55, 0.6, 0.62, 0.65])
    assert H.integer_crossover(r, 0.5) == 3 and H.integer_crossover(r, 0.2) == 1 and H.integer_crossover(r, 0.9) == 7
    assert H.continuous_crossover(r, 0.15) == pytest.approx(0.5)
    assert H.continuous_crossover(r, 0.5) == pytest.approx(2.5)
    assert H.continuous_crossover(r, -0.1) == 0.0 and H.continuous_crossover(r, 0.9) == np.inf
    assert H.ceil_cge(0.0) == 1 and H.ceil_cge(2.3) == 3 and H.ceil_cge(np.inf) == 7 and H.ceil_cge(40) == 7


def test_fixed_forecast_bootstrap_is_centred_and_shares_draws():
    """PH5: with the forecasts held fixed the draws centre on the point estimate (the refit bootstrap need not), and
    the variance-component draws equal the refit bootstrap's (same player draws)."""
    games, players, _ = simulate(n=300, seed=18)
    td = H.trait_data(games, players, {"C0": None, "C2": ["x1", "x2"]})
    est = H.estimate(td, n_repeats=2, keep_oof=True)
    ff = H.bootstrap_fixed_forecasts(td, est, n_boot=300, seed=3)
    assert 0.3 < np.mean(ff["r2_C2"].to_numpy() < est["r2"]["C2"]) < 0.7
    ci = H.percentile_ci(ff["cge_C2"].to_numpy())
    assert ci["lo"] < 4 * 0.4 / 0.6 < ci["hi"]
    rb = H.bootstrap(td, n_boot=6, seed=3, n_jobs=1, n_repeats=1)
    f6 = H.bootstrap_fixed_forecasts(td, est, n_boot=6, seed=3)
    assert np.allclose(rb["lambda"].to_numpy(), f6["lambda"].to_numpy())


# ---------------------------------------------------------------- post hoc (review response)

def test_calibrated_bounds_invert_a_known_estimator():
    """Estimate ~ N(estimand, 1): the one-sided 95% bounds are obs -+ 1.645; with an estimator clipped at 0, an
    observed 0 gives lower 0 and upper 1.645; an estimate beyond the grid is unbounded above."""
    rng = np.random.default_rng(20)
    grid = np.linspace(0, 10, 41)
    draws = [rng.normal(g, 1, 20_000) for g in grid]
    b = H.calibrated_bounds(5.0, grid, draws)
    assert b["lower"] == pytest.approx(5 - 1.645, abs=0.06) and b["upper"] == pytest.approx(5 + 1.645, abs=0.06)
    clipped = [np.maximum(d, 0) for d in draws]
    b0 = H.calibrated_bounds(0.0, grid, clipped)
    assert b0["lower"] == 0.0 and b0["upper"] == pytest.approx(1.645, abs=0.06)
    assert H.calibrated_bounds(30.0, grid, draws)["upper"] is None


def test_simulated_design_recovers_the_cv_estimand():
    """At a large simulated design the estimator's R2_true matches the R2 its own out-of-fold forecasts achieve
    against theta (the CV estimand), which approaches the planted population R2."""
    games, players, _ = simulate(n=1500, seed=21)
    td = H.trait_data(games, players, {"C0": None, "C2": ["x1", "x2", "x3"]})
    Z = H.signal_matrix(td.X["C2"], len(td.levels) - 1)
    assert Z.shape == (1500, 3) and np.allclose(Z.std(axis=0), 1)
    sim, theta = H.simulate_values(td, Z, lam=3.0, r2=0.4, rng=np.random.default_rng(0))
    assert np.array_equal(sim.g_player, td.g_player) and not np.allclose(sim.g_value, td.g_value)
    est = H.estimate(sim, n_repeats=2, keep_oof=True)
    orc = H.oracle_r2(theta, est["oof"]["C2"])
    assert est["r2"]["C2"] == pytest.approx(orc, abs=0.05) and orc == pytest.approx(0.4, abs=0.06)
    assert est["lambda"] == pytest.approx(3.0, rel=0.1)
    assert H.relative_true_r2(0.30, -0.05, 0.7) == pytest.approx(0.5) and H.relative_true_r2(-0.1, -0.05, 0.7) == 0
