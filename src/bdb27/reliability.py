"""Reliability screens: is a feature a stable trait of the player, or noise?

Forms (PLAN.md §2.5):
  - icc1: repeated attempts of the same drill (40, shuttle, 3-cone, some DB drills)
  - cross_drill_icc: the same construct measured in different drills (most position drills have 1 attempt)
  - split_half: game traits, odd vs even games per player, Spearman-Brown corrected
  - split_half_estimator: the same for any per-player estimator (quantiles, envelopes, draw-based fits)
  - team_share: how much of a game trait's between-player variance is team-season (scheme), not player
"""

from collections.abc import Callable

import numpy as np
import polars as pl
from scipy.stats import rankdata


def icc1(values: np.ndarray, groups: np.ndarray) -> dict:
    """One-way random-effects ICC(1) for unbalanced groups (ANOVA estimator), estimated from players
    with >= 2 repeats. `reliability` is the Spearman-Brown reliability of each player's mean over
    the player's own number of repeats, averaged over ALL players (singletons contribute the single-measure ICC):
    the reliability of the player-level feature as it will actually be used.
    NaN (no warnings) when fewer than 3 players repeat or there is no variance at all."""
    ok = np.isfinite(values)
    values, groups = values[ok], groups[ok]
    labels, inv, n_i = np.unique(groups, return_inverse=True, return_counts=True)
    n_all = n_i.copy()
    keep = n_i[inv] >= 2
    values, inv = values[keep], np.unique(inv[keep], return_inverse=True)[1]
    n_i = np.bincount(inv)
    k, N = len(n_i), len(values)
    base = {"n_players": int(len(n_all)), "n_single": int((n_all == 1).sum()), "n_repeated": int(k)}
    if k < 3:
        return {"icc": np.nan, "reliability": np.nan, **base}
    means = np.bincount(inv, weights=values) / n_i
    grand = values.mean()
    msb = np.sum(n_i * (means - grand) ** 2) / (k - 1)
    msw = np.sum((values - means[inv]) ** 2) / (N - k)
    n0 = (N - np.sum(n_i**2) / N) / (k - 1)
    den = msb + (n0 - 1) * msw
    if not den > 0:
        return {"icc": np.nan, "reliability": np.nan, **base}
    icc = (msb - msw) / den
    with np.errstate(divide="ignore", invalid="ignore"):  # 1 + (n-1) icc = 0 at icc = -1/(n-1)
        rel = float(np.mean(n_all * icc / (1 + (n_all - 1) * icc)))
    return {"icc": float(icc), "reliability": rel if np.isfinite(rel) else np.nan, **base}


def cross_drill_icc(df: pl.DataFrame, value: str, player: str = "nfl_id", drill: str = "drill_name",
                    zscore: bool = False, center_by: list[str] | None = None) -> dict:
    """ICC(1) across drills after removing each drill's mean (drills differ in difficulty).
    center_by: center within these groups instead of `drill` alone, e.g. [drill, session] or [drill, draft_year]
    (Combine drill tempo drifts by session and year). zscore: also scale to unit SD within the same groups,
    so drills measured on different scales count equally."""
    by = center_by or [drill]
    v = pl.col(value) - pl.col(value).mean().over(by)
    if zscore:
        v = v / pl.col(value).std().over(by)
    d = df.filter(pl.col(value).is_finite()).with_columns(_v=v)
    return icc1(d["_v"].to_numpy(), d[player].to_numpy())


def _half_corr(a: np.ndarray, b: np.ndarray, method: str) -> dict:
    """Correlation of per-player half estimates (players with both finite), Spearman-Brown to full length."""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 10:
        return {"r_half": np.nan, "reliability": np.nan, "n_players": len(a)}
    if method == "spearman":
        a, b = rankdata(a), rankdata(b)
    elif method != "pearson":
        raise ValueError(f"method must be 'pearson' or 'spearman', not {method!r}")
    r = float(np.corrcoef(a, b)[0, 1])
    return {"r_half": r, "reliability": 2 * r / (1 + r), "n_players": len(a)}


def split_half(df: pl.DataFrame, value: str, player: str = "nfl_id", unit: str = "game_id",
               min_per_half: int = 5, method: str = "pearson") -> dict:
    """Correlate per-player means over odd vs even `unit`s (in order); Spearman-Brown to full length.
    method='spearman' correlates ranks (robust to a few extreme players)."""
    d = df.filter(pl.col(value).is_finite()).with_columns(
        half=(pl.col(unit).rank("dense").over(player) % 2).cast(pl.Int8))
    m = (d.group_by(player, "half").agg(v=pl.col(value).mean(), n=pl.len())
         .filter(pl.col("n") >= min_per_half)
         .pivot("half", index=player, values="v").drop_nulls())
    if m.height < 10:
        return {"r_half": np.nan, "reliability": np.nan, "n_players": m.height}
    return _half_corr(m["0"].to_numpy(), m["1"].to_numpy(), method)


def split_half_estimator(df: pl.DataFrame, fn: Callable[[pl.DataFrame], float], player: str = "nfl_id",
                         unit: str = "game_id", method: str = "pearson") -> dict:
    """split_half for any per-player estimator: fn(rows of one player in one half) -> float, applied
    separately to the player's odd and even `unit`s (dense rank, as in split_half). For traits that are
    not a mean of row values: quantiles, envelopes, draw-based fits. fn returns NaN when a half has too
    little data; those players are dropped."""
    d = df.with_columns(_half=(pl.col(unit).rank("dense").over(player) % 2).cast(pl.Int8))
    est = {key: fn(g.drop("_half")) for key, g in d.partition_by(player, "_half", as_dict=True).items()}
    ids = sorted({p for p, _ in est})
    a, b = (np.array([est.get((p, h), np.nan) for p in ids], dtype=float) for h in (0, 1))
    return _half_corr(a, b, method)


def team_share(df: pl.DataFrame, value: str, team_col: str | list[str], player: str = "nfl_id",
               min_n: int = 20) -> dict:
    """Share of the between-player variance of a game trait that lies between team-seasons (scheme, QB,
    play calling) rather than between players: ICC(1) of player x team means of `value` (each from >= min_n
    rows), grouped by team. Pass team_col=[team, season], and one position (or position-centered values).
    Near 0: the trait belongs to the player; near 1: to the team. Only team-seasons with >= 2 study players
    inform the estimate (n_shared); the denominator includes the sampling noise of each player mean."""
    teams = [team_col] if isinstance(team_col, str) else team_col
    m = (df.filter(pl.col(value).is_finite()).group_by(player, *teams)
         .agg(v=pl.col(value).mean(), n=pl.len()).filter(pl.col("n") >= min_n)
         .with_columns(_team=pl.concat_str([pl.col(c).cast(pl.Utf8) for c in teams], separator="|")))
    r = icc1(m["v"].to_numpy(), m["_team"].to_numpy())
    return {"team_share": r["icc"], "n_player_teams": m.height, "n_teams": r["n_players"],
            "n_shared": r["n_repeated"]}
