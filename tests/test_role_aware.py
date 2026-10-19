"""E013 role-aware Combine Exchange Rate (bdb27.probes.i_role_aware), on simulated players with a planted role effect:
role share counting; variance components after year + role (derivation 3) against simulation; role-aware priors
recover the planted role and Combine-beyond-role values; the blend weight under a role-aware prior is the Bayesian
posterior; CGE beyond role equals the role-aware exchange rate (derivation 2); the in-season role share uses only the
first k games and the fit sees no test data; team-cluster sign flips; simulation at a design."""

import numpy as np
import polars as pl
import pytest

from bdb27.probes import h_game_equivalents as H
from bdb27.probes import i_role_aware as I

SETS = {"C0": None, "C0_role": [I.ROLE], "C2": ["x1", "x2"], "C2_role": ["x1", "x2", I.ROLE]}


def simulate(n=3000, tau2=1.0, sigma2=4.0, a=0.6, b1=0.3, b2=0.4, games=(3, 15), seed=0, years=(2023, 2024, 2025),
             year_fx=(0.0, 0.5, -0.3)):
    """Players with role share x = Phi(z) (z ~ N(0, 1), a share in (0, 1)), Combine x1 = 0.6 z + 0.8 e1 (correlated
    with role) and x2 ~ N(0, 1); theta = year effect + scale * (a z + b1 x1 + b2 x2) + u, scaled so Var theta (within
    year) = tau2, with u carrying the rest. Games = theta + N(0, sigma2). Returns games, players, and the true
    population R2 of role alone, of role + x1 + x2, and tau2_role (variance left after role)."""
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    yi = rng.integers(0, len(years), n)
    z = rng.normal(size=n)
    x1 = 0.6 * z + 0.8 * rng.normal(size=n)
    x2 = rng.normal(size=n)
    # population moments: Var(a z + b1 x1 + b2 x2) = (a + 0.6 b1)^2 + (0.8 b1)^2 + b2^2
    v_sig = (a + 0.6 * b1) ** 2 + (0.8 * b1) ** 2 + b2 ** 2
    r2_all = 0.6  # share of tau2 explained by the signal
    sc = np.sqrt(r2_all * tau2 / v_sig)
    theta = np.asarray(year_fx)[yi] + sc * (a * z + b1 * x1 + b2 * x2) + rng.normal(0, np.sqrt((1 - r2_all) * tau2), n)
    # role alone: E[signal | z] = (a + 0.6 b1) z (x1 = 0.6 z + noise); but the role SHARE is Phi(z), a nonlinear
    # transform, so a linear model in the share captures corr(z, Phi(z))^2 = 3 / pi of that part
    r2_role_z = sc ** 2 * (a + 0.6 * b1) ** 2 / tau2
    r2_role = r2_role_z * 3 / np.pi
    n_i = rng.integers(games[0], games[1] + 1, n)
    pid = np.repeat(np.arange(n), n_i)
    g = pl.DataFrame({"nfl_id": pid, "game_id": np.concatenate([np.arange(k) for k in n_i]),
                      "value": theta[pid] + rng.normal(0, np.sqrt(sigma2), len(pid))})
    players = pl.DataFrame({"nfl_id": np.arange(n), "draft_year": np.asarray(years)[yi], "x1": x1, "x2": x2,
                            I.ROLE: norm.cdf(z)})
    return g, players, {"r2_role": r2_role, "tau2_role": tau2 * (1 - r2_role)}


# ---------------------------------------------------------------- role share

def test_alignment_counts_and_role_share():
    plays = pl.DataFrame({
        "nfl_id": [1] * 6 + [2] * 3 + [3] * 2,
        "game_id": [10, 10, 10, 11, 11, 11, 10, 10, 10, 10, 10],
        "lined_up_position": ["SLOT_WR", "WR", "TE", "SLOT_WR", "SLOT_WR", None, "WR", "WR", "RB", "TE", "FB"],
        "route_ran": ["GO", "HITCH", None, "POST", "OUT", "WHEEL", "CORNER", None, "SCREEN", "GO", None]})
    c = I.game_alignment_counts(plays, "WR")
    r = c.filter((pl.col("nfl_id") == 1) & (pl.col("game_id") == 10)).row(0, named=True)
    assert (r["role_num"], r["role_den"], r["n_rows"], r["deep_num"], r["deep_den"]) == (1, 2, 3, 1, 2)
    s = I.role_share(c, pl.DataFrame({"nfl_id": [1, 1, 2, 3], "game_id": [10, 11, 10, 10]}))
    one = s.filter(pl.col("nfl_id") == 1).row(0, named=True)
    assert one[I.ROLE] == pytest.approx(3 / 4) and one["role_den"] == 4 and one["n_games_role"] == 2
    assert one[I.DEEP] == pytest.approx(3 / 5)  # GO, POST, WHEEL of 5 routes; TE / null alignments do not matter
    assert s.filter(pl.col("nfl_id") == 2)[I.ROLE].item() == 0.0
    assert s.filter(pl.col("nfl_id") == 3)[I.ROLE].item() is None  # no play at SLOT_WR or WR: undefined
    only10 = I.role_share(c, pl.DataFrame({"nfl_id": [1], "game_id": [10]}))
    assert only10[I.ROLE].item() == pytest.approx(0.5)
    for pos, (num, den) in I.ROLE_ALIGN.items():
        assert set(num) <= set(den), pos


def test_first_k_games_is_chronological():
    g = pl.DataFrame({"nfl_id": [1, 1, 1, 2, 2], "game_id": [30, 10, 20, 5, 7], "value": [3.0, 1.0, 2.0, 5.0, 7.0]})
    f = I.first_k_games(g, 2)
    assert f.filter(pl.col("nfl_id") == 1)["game_id"].to_list() == [10, 20]
    assert f.filter(pl.col("nfl_id") == 2)["game_id"].to_list() == [5, 7]


# ---------------------------------------------------------------- variance components after year + role

def test_variance_components_cov_reduces_to_e012():
    g, p, _ = simulate(n=400, seed=1)
    gg = g.join(p, on="nfl_id")
    v, pid, yr = gg["value"].to_numpy(), gg["nfl_id"].to_numpy(), gg["draft_year"].to_numpy()
    a = H.variance_components(v, pid, yr)
    b = I.variance_components_cov(v, pid, yr, None)
    c = I.variance_components_cov(v, pid, yr, np.ones((len(v), 1)))  # a constant covariate is dropped
    for x in (b, c):
        assert x["tau2"] == pytest.approx(a["tau2"], rel=1e-9) and x["sigma2"] == pytest.approx(a["sigma2"], rel=1e-9)


def test_variance_components_cov_is_unbiased_for_tau2_role():
    """Derivation 3 against simulation: over many small unbalanced datasets the estimator's mean matches the planted
    between-player variance left after year + role (and the year-only estimator does not)."""
    rng = np.random.default_rng(2)
    tau2_role, beta, sigma2 = 0.5, 1.2, 3.0
    est, est_year = [], []
    for _ in range(400):
        n = 60
        yr = rng.integers(0, 3, n)
        x = rng.uniform(0, 1, n)
        x[rng.random(n) < 0.1] = np.nan  # some undefined role shares: median + indicator
        theta = np.array([0.0, 0.7, -0.4])[yr] + beta * np.nan_to_num(x, nan=0.5) + rng.normal(0, np.sqrt(tau2_role), n)
        n_i = rng.integers(2, 17, n)
        pid = np.repeat(np.arange(n), n_i)
        v = theta[pid] + rng.normal(0, np.sqrt(sigma2), len(pid))
        est.append(I.variance_components_cov(v, pid, yr[pid], x[pid][:, None])["tau2"])
        est_year.append(H.variance_components(v, pid, yr[pid])["tau2"])
    se = np.std(est) / np.sqrt(len(est))
    assert abs(np.mean(est) - tau2_role) < 3 * se + 0.01
    assert np.mean(est_year) > tau2_role + 0.08  # year-only leaves role variance (beta^2 Var x ~ 0.11) in tau2


def test_variance_components_cov_rejects_varying_covariate():
    with pytest.raises(ValueError):
        I.variance_components_cov(np.arange(6.0), np.array([0, 0, 0, 1, 1, 1]), np.zeros(6),
                                  np.array([[0.1], [0.2], [0.1], [0.5], [0.5], [0.5]]))


# ---------------------------------------------------------------- role-aware priors recover a planted role effect

def test_role_aware_estimate_recovers_planted_role():
    g, p, truth = simulate(n=4000, seed=3)
    td = H.trait_data(g, p, SETS)
    e = I.estimate_role(td, n_repeats=1)
    lam = 4.0
    assert e["lambda"] == pytest.approx(lam, rel=0.08)
    assert e["r2"]["C0_role"] == pytest.approx(truth["r2_role"], abs=0.04)
    assert e["role_worth"] == pytest.approx(H.cge(lam, truth["r2_role"]), rel=0.2)
    # the variance role leaves, two ways: ANOVA with the covariate and the CV route lambda / (1 - R2_role)
    assert e["tau2_role_anova"] == pytest.approx(truth["tau2_role"], abs=0.06)
    assert e["lambda_role_anova"] == pytest.approx(lam / (1 - truth["r2_role"]), rel=0.12)
    assert e["lambda_role_cv"] == pytest.approx(e["lambda_role_anova"], rel=0.12)
    # Combine beyond role: x1 / x2 add what role leaves; the C2 + role set explains more than either
    r2c, r2r = e["r2"]["C2_role"], e["r2"]["C0_role"]
    assert r2c > max(e["r2"]["C2"], r2r) + 0.05
    assert e["beyond_role"]["C2_role"] == pytest.approx(I.beyond_identity(e["lambda"], r2c, r2r), rel=1e-9)
    assert e["combine_change_once_role_known"] == pytest.approx(e["cge"]["C2"] - e["beyond_role"]["C2_role"])
    # half-weight games = lambda + CGE, and the trust weights are E012's blend weights
    assert e["half_weight"]["C0_role"] == pytest.approx(e["lambda"] + e["cge"]["C0_role"], rel=1e-9)
    assert e["weights"]["C2_role"]["4"] == pytest.approx(4 / (4 + e["lambda"] + e["cge"]["C2_role"]), rel=1e-9)


def test_no_role_effect_gets_no_games():
    g, p, _ = simulate(n=1500, a=0.0, b1=0.0, b2=1.0, seed=4)
    p = p.with_columns(pl.Series(I.ROLE, np.random.default_rng(0).uniform(size=p.height)))  # role unrelated
    e = I.estimate_role(H.trait_data(g, p, SETS), n_repeats=2)
    assert e["role_worth"] < 0.1 and e["beyond_role"]["C2_role"] == pytest.approx(e["cge"]["C2"], abs=0.25)


def test_beyond_role_identity_and_posterior_equivalence():
    """Derivation 2: CGE_C - CGE_role = lambda_role P / (1 - P); and the posterior after k games under the richer
    prior equals the posterior under the role prior after k + (CGE_C - CGE_role) games."""
    rng = np.random.default_rng(5)
    for _ in range(200):
        lam, tau2 = rng.uniform(0.5, 12), rng.uniform(0.1, 3)
        r2r = rng.uniform(0, 0.8)
        r2c = rng.uniform(r2r, 0.95)
        b = H.cge(lam, r2c) - H.cge(lam, r2r)
        assert I.beyond_identity(lam, r2c, r2r) == pytest.approx(b, rel=1e-9)
        k = rng.integers(0, 17)
        sigma2 = lam * tau2
        assert H.posterior_var(k, tau2, sigma2, r2c) == pytest.approx(
            1 / (1 / ((1 - r2r) * tau2) + (k + b) / sigma2), rel=1e-9)


def test_role_prior_blend_weight_is_bayesian_posterior():
    """Monte Carlo: with mu = the role-predicted level (explains R2_role of tau2), E[theta | mu, ybar_k] has slope
    k / (k + lambda_role) on ybar_k - mu, lambda_role = sigma2 / tau2_role = lambda / (1 - R2_role)."""
    rng = np.random.default_rng(6)
    tau2, sigma2, r2r, n = 1.0, 3.0, 0.3, 400_000
    x = rng.uniform(0, 1, n)
    mu = (x - 0.5) / np.std(x) * np.sqrt(r2r * tau2)
    theta = mu + rng.normal(0, np.sqrt((1 - r2r) * tau2), n)
    for k in (1, 4):
        ybar = theta + rng.normal(0, np.sqrt(sigma2 / k), n)
        slope = np.polyfit(ybar - mu, theta - mu, 1)[0]
        lam_role = sigma2 / ((1 - r2r) * tau2)
        assert slope == pytest.approx(k / (k + lam_role), abs=0.01)
        assert float(H.blend_weight(k, sigma2 / tau2, r2r)) == pytest.approx(k / (k + lam_role), rel=1e-12)


# ---------------------------------------------------------------- bootstrap

def test_bootstrap_role_reproduces_e012_draws_and_covers():
    g, p, truth = simulate(n=300, seed=7)
    td = H.trait_data(g, p, SETS)
    a = I.bootstrap_role(td, n_boot=12, seed=11, n_jobs=1, n_repeats=1)
    b = H.bootstrap(td, n_boot=12, seed=11, n_jobs=1, n_repeats=1)
    assert np.allclose(a["cge_C2"].to_numpy(), b["cge_C2"].to_numpy(), equal_nan=True)
    assert np.allclose(a["lambda"].to_numpy(), b["lambda"].to_numpy())
    assert np.allclose(a["beyond_C2_role"].to_numpy(), (a["cge_C2_role"] - a["cge_C0_role"]).to_numpy())
    big = I.bootstrap_role(td, n_boot=60, seed=1, n_jobs=1, n_repeats=1)
    ci = H.percentile_ci(big["lambda_role_anova"].to_numpy())
    assert ci["lo"] < 4.0 / (1 - truth["r2_role"]) < ci["hi"]


# ---------------------------------------------------------------- in season: no leakage

def _prospective(seed=8, n=900):
    """Simulated classes 2023-25 with per-game alignment counts (slot plays drawn from the player's slot rate) and a
    trait whose level depends on the slot rate."""
    rng = np.random.default_rng(seed)
    yr = rng.choice([2023, 2024, 2025], n)
    rate = rng.beta(2, 2, n)
    x1 = rng.normal(size=n)
    theta = 1.5 * (rate - 0.5) + 0.3 * x1 + rng.normal(0, 0.6, n)
    n_i = rng.integers(4, 17, n)
    pid = np.repeat(np.arange(n), n_i)
    j = np.concatenate([np.arange(k) for k in n_i])
    season = np.minimum(yr[pid] + (j >= 8), 2025)
    gid = 1000 * season + j
    den = rng.integers(15, 40, len(pid)).astype(float)
    num = rng.binomial(den.astype(int), rate[pid]).astype(float)
    games = pl.DataFrame({"nfl_id": pid, "game_id": gid, "season": season,
                          "value": theta[pid] + rng.normal(0, 1.2, len(pid))})
    counts = pl.DataFrame({"nfl_id": pid, "game_id": gid, "role_num": num, "role_den": den})
    players = pl.DataFrame({"nfl_id": np.arange(n), "draft_year": yr, "x1": x1, "x2": rng.normal(size=n)})
    return games, counts, players


def _role_forecasts(games, counts, players, ks=(1, 2)):
    sets = {"C0": None, "C0_role": [I.ROLE], "C2_role": ["x1", "x2", I.ROLE]}
    tr, te = H.split_prospective(games, players)
    fit = H.fit_prospective(tr, players.join(I.role_share(counts, tr), on="nfl_id", how="left"), sets, n_repeats=1)
    tp = players.filter(pl.col("draft_year") == 2025)
    by_k = {k: tp.join(I.role_share(counts, I.first_k_games(te, k)), on="nfl_id", how="left") for k in ks}
    return fit, I.in_season_role_forecasts(fit, te, by_k, ks), te


def test_in_season_role_uses_only_first_k_games_and_fit_ignores_2025():
    games, counts, players = _prospective()
    fit, f, te = _role_forecasts(games, counts, players)
    assert fit.r2["C0_role"] > 0.1 and f["k"].unique().sort().to_list() == [1, 2]
    # alignments in games after k (and every 2025 outcome after k) do not move the k-game forecast
    late = te.sort("nfl_id", "game_id").with_columns(j=pl.int_range(pl.len()).over("nfl_id"))
    after2 = late.filter(pl.col("j") >= 2).select("nfl_id", "game_id").with_columns(_late=pl.lit(True))

    def flag(df):
        return df.join(after2, on=["nfl_id", "game_id"], how="left", maintain_order="left")["_late"].fill_null(False)
    c2 = counts.with_columns(role_num=pl.when(flag(counts)).then(pl.col("role_den")).otherwise(pl.col("role_num")))
    g2 = games.with_columns(value=pl.when(flag(games)).then(pl.col("value") + 9).otherwise(pl.col("value")))
    assert flag(counts).sum() > 100 and flag(games).sum() > 100
    fit2, f2, _ = _role_forecasts(g2, c2, players)
    assert fit2.r2 == fit.r2 and fit2.vc == fit.vc
    for k in (1, 2):
        a, b = f.filter(pl.col("k") == k), f2.filter(pl.col("k") == k)
        assert np.allclose(a["mu_C0_role"], b["mu_C0_role"]) and np.allclose(a["blend_C2_role"], b["blend_C2_role"])
        assert np.allclose(a["ybar"], b["ybar"])
    # ... but alignments within the first k games do
    first = late.filter(pl.col("j") == 0).select("nfl_id", "game_id").with_columns(_f=pl.lit(True))
    flag1 = counts.join(first, on=["nfl_id", "game_id"], how="left", maintain_order="left")["_f"].fill_null(False)
    c3 = counts.with_columns(role_num=pl.when(flag1).then(0.0).otherwise(pl.col("role_num")))
    _, f3, _ = _role_forecasts(games, c3, players)
    assert not np.allclose(f.filter(pl.col("k") == 1)["mu_C0_role"], f3.filter(pl.col("k") == 1)["mu_C0_role"])
    # 2025 role shares never reach the fit; the training role shares come from 2023-24 games only
    tr, _ = H.split_prospective(games, players)
    assert tr["season"].max() <= 2024
    roles_tr = I.role_share(counts, tr)
    assert set(roles_tr["nfl_id"].to_list()) <= set(players.filter(pl.col("draft_year") < 2025)["nfl_id"].to_list())


# ---------------------------------------------------------------- team-cluster inference

def test_cluster_sign_flip_singletons_equal_player_test_and_is_calibrated():
    rng = np.random.default_rng(9)
    d = rng.normal(-0.3, 1, 40)
    assert I.cluster_sign_flip_p(d, np.arange(40), 5000, 3) == H.sign_flip_p(d, 5000, 3)
    ps = []
    for s in range(300):
        cl = rng.integers(0, 12, 40)
        team_fx = rng.normal(0, 1, 12)[cl]  # correlated within team, mean 0 under the null
        ps.append(I.cluster_sign_flip_p(team_fx + rng.normal(0, 1, 40), cl, 2000, s))
    ps = np.array(ps)
    assert 0.02 <= np.mean(ps < 0.05) <= 0.09 and 0.42 <= ps.mean() <= 0.58


def test_paired_test_with_clusters():
    rng = np.random.default_rng(10)
    rows = [(i, t, rng.normal(1, 1) - 0.9, rng.normal(1, 1)) for i in range(80) for t in ("T1", "T6")
            if rng.random() < 0.8]
    df = pl.DataFrame(rows, schema=["nfl_id", "trait", "a", "b"], orient="row")
    cl = {i: i % 25 for i in range(80)}
    r = I.paired_test(df, "a", "b", seed=0, cluster_of=cl, n_flips=5000, n_boot=2000)
    assert r["diff"] == pytest.approx(H.pooled_mean(df, "a") - H.pooled_mean(df, "b"))
    assert r["p_one_sided"] < 0.01 and r["team_cluster"]["p_one_sided"] < 0.05
    lo, hi = r["team_cluster"]["diff_ci95"]
    assert lo < r["diff"] < hi and r["team_cluster"]["n_clusters"] == 25


# ---------------------------------------------------------------- simulation at a design

def test_simulated_design_plants_role_and_beyond_role():
    g, p, _ = simulate(n=2500, seed=12)
    td = H.trait_data(g, p, SETS)
    des = I.sim_design(td, ("C0", "C0_role", "C2_role"), "C2_role")
    assert des.Zc.shape == (2500, 2) and abs(np.corrcoef(des.z, I.role_column(td))[0, 1]) > 0.99
    sim, theta = I.simulate_role_values(des, lam=3.0, rho=0.3, g=0.4, rng=np.random.default_rng(0))
    assert theta.var() == pytest.approx(1.0, abs=0.08)
    assert np.corrcoef(theta, des.z)[0, 1] ** 2 == pytest.approx(0.3, abs=0.04)
    e = I.estimate_role(sim, n_repeats=1, keep_oof=True)
    orc = {s: H.oracle_r2(theta, e["oof"][s]) for s in ("C0_role", "C2_role")}
    assert orc["C0_role"] == pytest.approx(0.3, abs=0.04) and orc["C2_role"] == pytest.approx(0.3 + 0.7 * 0.4, abs=0.05)
    assert e["r2"]["C2_role"] == pytest.approx(orc["C2_role"], abs=0.05)
    star = H.cge(3.0, orc["C2_role"]) - H.cge(3.0, orc["C0_role"])
    assert e["beyond_role"]["C2_role"] == pytest.approx(star, rel=0.25)
    assert I.population_beyond(3.0, 0.3, 0.4) == pytest.approx(H.cge(3.0, 0.58) - H.cge(3.0, 0.3), rel=1e-9)


# ---------------------------------------------------------------- post hoc helpers (E013 review response)

def test_role_variance_share_and_permutation_test():
    """The ANOVA share of between-player variance that role explains recovers a planted role effect, its worth is
    lambda_role - lambda, and the within-class permutation test rejects with a planted effect and is calibrated
    without one."""
    g, p, truth = simulate(n=2000, seed=13)
    td = H.trait_data(g, p, {"C0": None, "C0_role": [I.ROLE]})
    rv = I.role_variance_share(td)
    assert rv["share"] == pytest.approx(truth["r2_role"], abs=0.05)
    vr = I.variance_components_cov(td.g_value, td.g_player, td.year[td.g_player], I.role_column(td)[td.g_player][:, None])
    assert rv["worth"] == pytest.approx(vr["lambda"] - H.variance_components(td.g_value, td.g_player,
                                                                              td.year[td.g_player])["lambda"], rel=1e-9)
    small = H.trait_data(*simulate(n=150, seed=14)[:2], {"C0": None, "C0_role": [I.ROLE]})
    perm = I.role_share_permutation(small, 300, 0)
    assert perm["observed"] == pytest.approx(I.role_variance_share(small)["share"], rel=1e-8)
    assert perm["p_one_sided"] < 0.01
    rng = np.random.default_rng(15)
    ps = []
    for s in range(60):
        g0, p0, _ = simulate(n=80, a=0.0, b1=0.0, b2=1.0, seed=100 + s)
        p0 = p0.with_columns(pl.Series(I.ROLE, rng.uniform(size=p0.height)))  # unrelated to the trait
        ps.append(I.role_share_permutation(H.trait_data(g0, p0, {"C0": None, "C0_role": [I.ROLE]}), 100, s)["p_one_sided"])
    ps = np.array(ps)
    assert 0.35 <= ps.mean() <= 0.65 and np.mean(ps < 0.05) <= 0.15


def test_expected_blend_se_matches_monte_carlo():
    rng = np.random.default_rng(16)
    n, lam = 400_000, 3.0
    for r2, k, m in ((0.0, 1, 8), (0.3, 1, 5), (0.5, 2, 12)):
        mu = rng.normal(0, np.sqrt(r2), n)
        theta = mu + rng.normal(0, np.sqrt(1 - r2), n)  # tau2 = 1; the prior's error variance is 1 - r2
        ybar = theta + rng.normal(0, np.sqrt(lam / k), n)
        target = theta + rng.normal(0, np.sqrt(lam / m), n)
        w = float(H.blend_weight(k, lam, r2))
        mc = np.mean((mu + w * (ybar - mu) - target) ** 2)
        assert float(I.expected_blend_se(k, m, lam, r2)) == pytest.approx(mc, rel=0.01)
    assert float(I.expected_blend_se(1, 6, lam, 0.2)) < float(I.expected_blend_se(1, 6, lam, 0.0))


def test_mde_ratio():
    d = np.random.default_rng(17).normal(0, 0.02, 200_000)
    assert I.mde_ratio(d) == pytest.approx(np.exp(-(1.6448536 + 0.8416212) * 0.02), rel=1e-3)


def test_regression_calibration_undoes_overdispersion():
    """A coefficient trained on full-history shares and fed a noisy short-window share over-disperses the forecast;
    the training-only calibration a + b x removes most of that error."""
    rng = np.random.default_rng(18)
    n = 4000
    s_tr, s_te = rng.beta(2, 2, n), rng.beta(2, 2, n)
    noisy = lambda s: rng.binomial(8, s) / 8  # noqa: E731  (a short window's plays; reliability about 2 / 3)
    x_tr, x_te = noisy(s_tr), noisy(s_te)
    beta = 2.0  # trait level = beta * full-history share (+ noise), learned exactly on full shares
    cal = I.regression_calibration(x_tr, s_tr)
    assert cal["b"] == pytest.approx(np.var(s_tr) / np.var(x_tr), abs=0.03) and cal["n"] == n
    raw_err = np.mean((beta * x_te - beta * s_te) ** 2)
    cal_err = np.mean((beta * (cal["a"] + cal["b"] * x_te) - beta * s_te) ** 2)
    assert cal_err < 0.8 * raw_err  # the calibrated error is (reliability) x the raw error
    with_nan = I.regression_calibration(np.r_[x_tr, np.nan], np.r_[s_tr, 0.5])
    assert with_nan["n"] == n and with_nan["b"] == pytest.approx(cal["b"], rel=1e-9)


def test_attribution_identity():
    a = I.attribution(0.8, 1.2, 1.3)
    assert a["role_beyond_c"] == pytest.approx(0.1) and a["c_beyond_role"] == pytest.approx(0.5)
    assert a["shared"] == pytest.approx(0.7) and a["shared"] == pytest.approx(a["c_alone"] - a["c_beyond_role"])
    assert a["role_beyond_c"] + a["shared"] == pytest.approx(a["role_alone"])


def test_dummies_ridge_matches_estimate_and_pairs_with_bootstrap():
    g, p, _ = simulate(n=300, seed=19)
    td = H.trait_data(g, p, SETS)
    D = H.year_dummies(td.year, td.levels)
    t2 = H.TraitData(nfl_id=td.nfl_id, year=td.year, X=td.X | {"C0_ridge": D}, g_player=td.g_player,
                     g_value=td.g_value, levels=td.levels)
    e = H.estimate(t2, n_repeats=2, seed=3)
    assert I.dummies_ridge(td, n_repeats=2, seed=3)["r2cv_C0_ridge"] == pytest.approx(e["r2_cv"]["C0_ridge"], rel=1e-12)
    a = I.bootstrap_dummies_ridge(td, n_boot=6, seed=11, n_jobs=1, n_repeats=1)
    b = I.bootstrap_role(t2, n_boot=6, seed=11, n_jobs=1, n_repeats=1)  # the same draws, with the ridge set inside
    assert a["b"].to_list() == b["b"].to_list()
    assert np.allclose(a["lambda"].to_numpy(), b["lambda"].to_numpy())
    assert np.allclose(a["r2cv_C0_ridge"].to_numpy(), b["r2cv_C0_ridge"].to_numpy(), rtol=1e-12)
