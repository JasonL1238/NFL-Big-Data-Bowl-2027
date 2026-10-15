"""Empirical-Bayes shrinkage of per-player game traits: one normal-normal definition for every experiment.

Model, per player i (separately per `group` if given): raw mean y_i ~ N(theta_i, v_i) with v_i treated as known,
theta_i ~ N(mu, tau2). Shrunk = mu + w_i (y_i - mu), w_i = tau2 / (tau2 + v_i); w_i is also the reliability of
player i's raw mean, and the posterior SD is sqrt(w_i) * se_i.
  - eb_mean: play-level values; v_i = sigma2 / n_i, sigma2 = pooled within-player variance of the group.
  - eb_rate: binary events minus a per-play expected probability; v_i = sum p (1 - p) / n_i^2 (binomial).
  - cluster="game_id" (both): a player's plays in one game share that game's conditions (opponent, plan, script),
    so they are correlated and the variances above understate v_i. With c = the covariance of two of a player's
    plays in the same game, v_i gains c sum_g m_g (m_g - 1) / n_i^2, m_g = his plays in game g. c (and, for eb_mean,
    the per-play variance, which sigma2 understates under clustering) are moment estimates from the within- and
    between-game sums of squares, pooled like sigma2; c is floored at 0, and c = 0 gives the unclustered result.
    Real DL get-off: v_i about doubles.
mu and tau2 are REML estimates over the players KEPT (n >= min_n) only. E002 run 1 fitted an unweighted
moment estimator over all players, many with 1-3 events, and tau2 collapsed (WR HITCH decel 0.008 vs 0.14).
check_prior compares the fitted prior with a split-half reliability to catch that kind of failure.
Cells are sorted before fitting, so identical inputs give bit-identical outputs.
"""

import warnings

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from bdb27 import reliability


def reml(y: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    """REML estimates (mu, tau2) of y_i ~ N(mu, tau2 + v_i) with known sampling variances v_i."""
    c, s = float(y.mean()), float(y.std()) or 1.0
    z, u = (y - c) / s, v / s**2  # unit scale, so the optimizer tolerance is scale-free

    def nll(t):  # -2 x restricted log-likelihood, up to a constant
        w = 1 / (u + t)
        m = np.sum(w * z) / np.sum(w)
        return np.sum(np.log(u + t)) + np.log(np.sum(w)) + np.sum(w * (z - m) ** 2)

    t = minimize_scalar(nll, bounds=(0.0, 10.0), method="bounded", options={"xatol": 1e-10}).x
    t = 0.0 if nll(0.0) <= nll(t) else t  # the bounded search never evaluates the boundary itself
    w = 1 / (v + t * s**2)
    return float(np.sum(w * y) / np.sum(w)), float(t * s**2)


def _over(expr: pl.Expr, group: str | None) -> pl.Expr:
    return expr.over(group) if group else expr


def _tot(expr: pl.Expr, group: str | None) -> pl.Expr:
    """Sum over the group's cells, on every cell."""
    return _over(expr.sum(), group)


def _shrink(cells: pl.DataFrame, keys: list[str], group: str | None) -> pl.DataFrame:
    """cells: one row per player (x group) with `raw` and sampling variance `v` -> adds the EB columns."""
    out = []
    for name, g in cells.sort(keys).group_by(group, maintain_order=True) if group else [(None, cells.sort(keys))]:
        if g.height < 3:
            raise ValueError(f"group {name}: {g.height} players with n >= min_n; need >= 3 to fit a prior")
        y, v = g["raw"].to_numpy(), g["v"].to_numpy()
        mu, tau2 = reml(y, v)
        w = tau2 / (tau2 + v)
        out.append(g.with_columns(se=pl.Series(np.sqrt(v)), shrunk=pl.Series(mu + w * (y - mu)),
                                  weight=pl.Series(w), prior_mean=pl.lit(mu), tau2=pl.lit(tau2)))
    return pl.concat(out).drop("v").sort(keys)


def _game_terms(d: pl.DataFrame, keys: list[str], cluster: str) -> pl.DataFrame:
    """Per cell, from per-play `_x` with variance `_nu`: G games, s2 = sum_g m_g^2, ssw / ssb = within / between-game
    sums of squares of _x, and k0, C such that E[ssb] = k0 + C c when two plays in one game have covariance c:
    k0 = sum_g nu_g / m_g - nu / n (nu_g = sum of _nu in game g), C = sum_g (m_g - 1)(n - m_g) / n."""
    g = (d.with_columns(_e=pl.col("_x") - pl.col("_x").mean().over(keys))
         .group_by(*keys, cluster).agg(m=pl.len().cast(pl.Float64), t=pl.col("_e").sum(), nu=pl.col("_nu").sum(),
                                       ssw=((pl.col("_e") - pl.col("_e").mean()) ** 2).sum())
         .sort(*keys, cluster))
    m, n = pl.col("m"), pl.col("m").sum()
    return g.group_by(keys).agg(G=pl.len(), s2=(m**2).sum(), ssw=pl.col("ssw").sum(),
                                ssb=(pl.col("t") ** 2 / m).sum(), k0=(pl.col("nu") / m).sum() - pl.col("nu").sum() / n,
                                C=((m - 1) * (n - m)).sum() / n)


def _cov(num: pl.Expr, den: pl.Expr, C: pl.Expr) -> pl.Expr:
    """Within-game covariance num / den floored at 0; 0 when no player has >= 2 plays in >= 2 games (C = 0)."""
    return pl.when(C > 0).then((num / den).clip(lower_bound=0.0)).otherwise(0.0)


def eb_mean(df: pl.DataFrame, value: str, player: str = "nfl_id", group: str | None = None, *,
            min_n: int, cluster: str | None = None) -> pl.DataFrame:
    """EB-shrunk per-player means of play-level `value`, for players with >= min_n finite values (rows are
    player x group cells if `group` varies within a player, e.g. alignment). The prior (and sigma2) is fitted
    on exactly these players. cluster: column of the game (or other unit) plays are correlated within; v_i then
    includes the within-game covariance. Columns: keys, n, raw, se, shrunk, weight, prior_mean, tau2."""
    keys = [player] + ([group] if group else [])
    d = df.filter(pl.col(value).is_finite()).filter(pl.len().over(keys) >= min_n).with_columns(_x=pl.col(value))
    cells = (d.group_by(keys).agg(n=pl.len(), raw=pl.col("_x").mean(),
                                  ss=((pl.col("_x") - pl.col("_x").mean()) ** 2).sum()).sort(keys))
    n, ss = pl.col("n"), pl.col("ss")
    if cluster is None:
        return _shrink(cells.with_columns(v=_tot(ss, group) / _tot(n - 1, group) / n).drop("ss"), keys, group)
    # Moments for the per-play variance s and within-game covariance c, summed over the group's cells (unit _nu,
    # so k0 = G - 1): E[ssw] = (s - c) sum(n - G), E[ssb] = s sum(k0) + c sum(C), and ss = ssw + ssb.
    t = cells.join(_game_terms(d.with_columns(_nu=pl.lit(1.0)), keys, cluster), on=keys, maintain_order="left")
    G, k0, C, nf = pl.col("G"), pl.col("k0"), pl.col("C"), n.cast(pl.Float64)
    within = _tot(pl.col("ssw"), group) / _tot(n - G, group)
    c = _cov(_tot(pl.col("ssb"), group) - within * _tot(k0, group), _tot(k0 + C, group), _tot(C, group))
    s = (_tot(ss, group) + c * _tot(n - G - C, group)) / _tot(n - 1, group)
    return _shrink(t.with_columns(v=s / nf + c * (pl.col("s2") - nf) / nf**2).select(*keys, "n", "raw", "v"),
                   keys, group)


def eb_rate(df: pl.DataFrame, event: str, expected: str | None = None, player: str = "nfl_id",
            group: str | None = None, *, min_n: int, cluster: str | None = None) -> pl.DataFrame:
    """EB-shrunk per-player rate over expected of a binary `event`, for players with >= min_n plays (plays with
    a null or NaN event or expectation are dropped). `expected` is a per-play probability (e.g. from a play-level
    model); without it, the base rate of the kept players' plays (per group). raw = rate - mean expected;
    sampling variance is binomial under the expectation, plus the within-game covariance if `cluster` is given.
    Columns: keys, n, events, rate, expected, raw, se, shrunk, weight, prior_mean, tau2 (raw/shrunk are rates
    over expected)."""
    keys = [player] + ([group] if group else [])
    ok = pl.col(event).cast(pl.Float64).is_finite()
    if expected:
        ok &= pl.col(expected).cast(pl.Float64).is_finite()
    d = df.filter(ok).filter(pl.len().over(keys) >= min_n).with_columns(_y=pl.col(event).cast(pl.Float64))
    d = d.with_columns(_p=pl.col(expected).cast(pl.Float64) if expected else _over(pl.col("_y").mean(), group))
    d = d.with_columns(_x=pl.col("_y") - pl.col("_p"), _nu=pl.col("_p") * (1 - pl.col("_p")))
    cells = (d.group_by(keys).agg(n=pl.len(), events=pl.col("_y").sum(), rate=pl.col("_y").mean(),
                                  expected=pl.col("_p").mean(), v=pl.col("_nu").sum() / pl.len().cast(pl.Float64) ** 2)
             .with_columns(raw=pl.col("rate") - pl.col("expected")).sort(keys))
    if cluster:  # E[ssb] = k0 + c C, with the binomial per-play variances in k0
        t = cells.join(_game_terms(d, keys, cluster), on=keys, maintain_order="left")
        n, C = pl.col("n").cast(pl.Float64), _tot(pl.col("C"), group)
        c = _cov(_tot(pl.col("ssb") - pl.col("k0"), group), C, C)
        cells = t.with_columns(v=pl.col("v") + c * (pl.col("s2") - n) / n**2).select(cells.columns)
    return _shrink(cells, keys, group)


def check_prior(eb: pl.DataFrame, plays: pl.DataFrame, value: str, player: str = "nfl_id",
                group: str | None = None, level: float = 0.95) -> dict:
    """Compare the fitted prior with the split-half reliability (odd vs even games, reliability.split_half) of
    the same cells. `plays` is the frame `eb` was fitted on and `value` the per-play quantity whose cell mean is
    eb["raw"]: the value for eb_mean, event - expected (or minus the base rate) for eb_rate; a mismatch raises.
    Split-half (Spearman-Brown) estimates Var(theta) / (Var(theta) + mean v_i) across cells; `implied` is that
    ratio under the fitted prior, and `ok` means it lies inside the split-half's Fisher-z interval. A player in
    several groups has one cell per group, and each is split separately (pooling his plays across groups would
    measure a longer, different mean). The mean weight is reported but not tested: w is convex in v, so with very
    unequal n it exceeds the split-half even when the prior is right. A collapsed prior gives implied <<
    split-half; plays correlated within games (v_i understated: use cluster=) give implied > split-half.
    Warns when not ok."""
    keys = [player] + ([group] if group else [])
    d = (plays.join(eb.select(keys), on=keys, how="semi").filter(pl.col(value).is_finite())
         .with_columns(_cell=pl.concat_str([pl.col(k).cast(pl.Utf8) for k in keys], separator="|")))
    m = d.group_by(keys).agg(m=pl.col(value).mean()).join(eb.select(*keys, "raw"), on=keys)
    if m.height != eb.height or not np.allclose(m["m"], m["raw"], rtol=1e-8, atol=1e-10):
        raise ValueError(f"cell means of {value!r} do not reproduce eb['raw']: pass the plays and the per-play "
                         "quantity the fit averaged (eb_rate: event - expected)")
    sh = reliability.split_half(d, value, player="_cell", min_per_half=1)
    var_theta = float(eb["tau2"].mean() + eb["prior_mean"].var(ddof=0))
    implied = var_theta / (var_theta + float((eb["se"] ** 2).mean()))
    half = np.tanh(np.arctanh(sh["r_half"]) + np.array([-1, 1]) * norm.ppf((1 + level) / 2)
                   / np.sqrt(sh["n_players"] - 3))
    lo, hi = 2 * half / (1 + half)
    out = {"mean_weight": float(eb["weight"].mean()), "implied_reliability": implied,
           "split_half": sh["reliability"], "split_half_ci": [float(lo), float(hi)],
           "ok": bool(lo <= implied <= hi), "n_cells": eb.height, "n_players": eb[player].n_unique()}
    if not out["ok"]:
        warnings.warn(f"EB prior disagrees with split-half reliability: implied {implied:.2f}, "
                      f"split-half {sh['reliability']:.2f} [{lo:.2f}, {hi:.2f}]")
    return out
