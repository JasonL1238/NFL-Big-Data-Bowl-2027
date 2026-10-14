"""Empirical-Bayes shrinkage of per-player game traits: one normal-normal definition for every experiment.

Model, per player i (separately per `group` if given): raw mean y_i ~ N(theta_i, v_i) with v_i treated as known,
theta_i ~ N(mu, tau2). Shrunk = mu + w_i (y_i - mu), w_i = tau2 / (tau2 + v_i); w_i is also the reliability of
player i's raw mean, and the posterior SD is sqrt(w_i) * se_i.
  - eb_mean: play-level values; v_i = sigma2 / n_i, sigma2 = pooled within-player variance of the group.
  - eb_rate: binary events minus a per-play expected probability; v_i = sum p (1 - p) / n_i^2 (binomial).
mu and tau2 are REML estimates over the players KEPT (n >= min_n) only. E002 run 1 fitted an unweighted
moment estimator over all players, many with 1-3 events, and tau2 collapsed (WR HITCH decel 0.008 vs 0.14).
check_prior compares the fitted prior with a split-half reliability to catch that kind of failure.
"""

import warnings

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar
from scipy.stats import norm


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


def _shrink(cells: pl.DataFrame, keys: list[str], group: str | None) -> pl.DataFrame:
    """cells: one row per player (x group) with `raw` and sampling variance `v` -> adds the EB columns."""
    out = []
    for name, g in cells.group_by(group) if group else [(None, cells)]:
        if g.height < 3:
            raise ValueError(f"group {name}: {g.height} players with n >= min_n; need >= 3 to fit a prior")
        y, v = g["raw"].to_numpy(), g["v"].to_numpy()
        mu, tau2 = reml(y, v)
        w = tau2 / (tau2 + v)
        out.append(g.with_columns(se=pl.Series(np.sqrt(v)), shrunk=pl.Series(mu + w * (y - mu)),
                                  weight=pl.Series(w), prior_mean=pl.lit(mu), tau2=pl.lit(tau2)))
    return pl.concat(out).drop("v").sort(keys)


def eb_mean(df: pl.DataFrame, value: str, player: str = "nfl_id", group: str | None = None, *,
            min_n: int) -> pl.DataFrame:
    """EB-shrunk per-player means of play-level `value`, for players with >= min_n finite values (rows are
    player x group cells if `group` varies within a player, e.g. alignment). The prior (and sigma2) is fitted
    on exactly these players. Columns: keys, n, raw, se, shrunk, weight, prior_mean, tau2."""
    keys = [player] + ([group] if group else [])
    cells = (df.filter(pl.col(value).is_finite())
             .group_by(keys).agg(n=pl.len(), raw=pl.col(value).mean(),
                                 ss=((pl.col(value) - pl.col(value).mean()) ** 2).sum())
             .filter(pl.col("n") >= min_n))
    sigma2 = _over(pl.col("ss").sum() / (pl.col("n") - 1).sum(), group)
    return _shrink(cells.with_columns(v=sigma2 / pl.col("n")).drop("ss"), keys, group)


def eb_rate(df: pl.DataFrame, event: str, expected: str | None = None, player: str = "nfl_id",
            group: str | None = None, *, min_n: int) -> pl.DataFrame:
    """EB-shrunk per-player rate over expected of a binary `event`, for players with >= min_n plays.
    `expected` is a per-play probability (e.g. from a play-level model); without it, the base rate of the
    kept players' plays (per group). raw = rate - mean expected; sampling variance is binomial under the
    expectation. Columns: keys, n, events, rate, expected, raw, se, shrunk, weight, prior_mean, tau2
    (raw/shrunk are rates over expected)."""
    keys = [player] + ([group] if group else [])
    d = (df.filter(pl.col(event).is_not_null() & (pl.col(expected).is_not_null() if expected else pl.lit(True)))
         .filter(pl.len().over(keys) >= min_n)
         .with_columns(_y=pl.col(event).cast(pl.Float64)))
    d = d.with_columns(_p=pl.col(expected) if expected else _over(pl.col("_y").mean(), group))
    cells = d.group_by(keys).agg(n=pl.len(), events=pl.col("_y").sum(), rate=pl.col("_y").mean(),
                                 expected=pl.col("_p").mean(),
                                 v=(pl.col("_p") * (1 - pl.col("_p"))).sum() / pl.len() ** 2)
    return _shrink(cells.with_columns(raw=pl.col("rate") - pl.col("expected")), keys, group)


def check_prior(eb: pl.DataFrame, split_half: dict, level: float = 0.95) -> dict:
    """Compare the fitted prior with `split_half` = reliability.split_half(...) of the SAME per-play values
    (for eb_rate: event - expected) and players, e.g.
        reliability.split_half(plays.join(eb.select(player), on=player, how="semi"), value, min_per_half=1)
    (the default min_per_half drops low-n players and inflates the split-half). Split-half (Spearman-Brown) estimates
    Var(theta) / (Var(theta) + mean v_i) across players; `implied` is that ratio under the fitted prior, and
    `ok` means it lies inside the split-half's Fisher-z interval. The mean weight is reported but not tested:
    w is convex in v, so with very unequal n it exceeds the split-half even when the prior is right.
    A collapsed prior gives implied << split-half; plays correlated within games (v_i understated) give
    implied > split-half. Warns when not ok."""
    var_theta = float(eb["tau2"].mean() + eb["prior_mean"].var(ddof=0))
    implied = var_theta / (var_theta + float((eb["se"] ** 2).mean()))
    half = np.tanh(np.arctanh(split_half["r_half"]) + np.array([-1, 1]) * norm.ppf((1 + level) / 2)
                   / np.sqrt(split_half["n_players"] - 3))
    lo, hi = 2 * half / (1 + half)
    out = {"mean_weight": float(eb["weight"].mean()), "implied_reliability": implied,
           "split_half": split_half["reliability"], "split_half_ci": [float(lo), float(hi)],
           "ok": bool(lo <= implied <= hi), "n_players": eb.height}
    if not out["ok"]:
        warnings.warn(f"EB prior disagrees with split-half reliability: implied {implied:.2f}, "
                      f"split-half {split_half['reliability']:.2f} [{lo:.2f}, {hi:.2f}]")
    return out
