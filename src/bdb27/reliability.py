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
    the reliability of the player-level feature as it will actually be used. `reliability` is 0 when icc <= 0
    (no detectable between-player variance; Spearman-Brown of a negative ICC changes sign and diverges).
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
    pos = max(icc, 0.0)
    rel = float(np.mean(n_all * pos / (1 + (n_all - 1) * pos)))
    return {"icc": float(icc), "reliability": rel, **base}


def cross_drill_icc(df: pl.DataFrame, value: str, player: str = "nfl_id", drill: str = "drill_name",
                    zscore: bool = False, center_by: list[str] | None = None) -> dict:
    """ICC(1) across drills after removing each drill's mean (drills differ in difficulty).
    center_by: center within these groups instead of `drill` alone, e.g. [drill, session] or [drill, draft_year]
    (Combine drill tempo drifts by session and year). zscore: also scale to unit SD within the same groups,
    so drills measured on different scales count equally. Rows alone in their group carry no player information
    and are dropped (n_dropped). Centering within groups of m rows also removes ~1/m of each player's signal
    (a true ICC of 0.8 reads ~0.45 at m = 2, ~0.72 at m = 10), so center by small groups only for large offsets."""
    by = center_by or [drill]
    v = pl.col(value) - pl.col(value).mean().over(by)
    if zscore:
        v = v / pl.col(value).std().over(by)
    d = df.filter(pl.col(value).is_finite()).with_columns(_m=pl.len().over(by))
    single = d["_m"] == 1
    d = d.filter(~single).with_columns(_v=v)
    return {**icc1(d["_v"].to_numpy(), d[player].to_numpy()), "n_dropped": int(single.sum())}


def _check_method(method: str) -> None:
    if method not in ("pearson", "spearman"):
        raise ValueError(f"method must be 'pearson' or 'spearman', not {method!r}")


def _half_corr(a: np.ndarray, b: np.ndarray, method: str) -> dict:
    """Correlation of per-player half estimates (players with both finite), Spearman-Brown to full length."""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 10:
        return {"r_half": np.nan, "reliability": np.nan, "n_players": len(a)}
    if method == "spearman":
        a, b = rankdata(a), rankdata(b)
    r = float(np.corrcoef(a, b)[0, 1])
    return {"r_half": r, "reliability": 2 * r / (1 + r), "n_players": len(a)}


def split_half(df: pl.DataFrame, value: str, player: str = "nfl_id", unit: str = "game_id",
               min_per_half: int = 5, method: str = "pearson") -> dict:
    """Correlate per-player means over odd vs even `unit`s (in order); Spearman-Brown to full length.
    method='spearman' correlates ranks (robust to a few extreme players)."""
    _check_method(method)
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
    _check_method(method)
    d = df.with_columns(_half=(pl.col(unit).rank("dense").over(player) % 2).cast(pl.Int8))
    est = {key: fn(g.drop("_half")) for key, g in d.partition_by(player, "_half", as_dict=True).items()}
    ids = sorted({p for p, _ in est})
    a, b = (np.array([est.get((p, h), np.nan) for p in ids], dtype=float) for h in (0, 1))
    return _half_corr(a, b, method)


def team_share(df: pl.DataFrame, value: str, team_col: str | list[str], player: str = "nfl_id",
               unit: str = "game_id", min_per_half: int = 10) -> dict:
    """Share of the true between-player variance of a game trait that lies between team-seasons (scheme, QB,
    play calling) rather than between players. Each team's `unit`s (games) are split odd/even, so teammates
    share halves; a, b are the centered half means of each player x team (>= min_per_half rows in each half):
        team_share = mean of a_i b_j over ordered teammate pairs i != j / mean of a_i b_i over player-teams
    The numerator estimates the team variance, the denominator team + player variance. Play noise, game effects
    and plays shared by teammates never fall in both halves, so the ratio does not depend on plays per player or
    per-play noise (an ICC of player x team means does). Pass team_col=[team, season] and one position (or
    position-centered values). Near 0: the trait belongs to the player; near 1: to the team. Uses team-seasons
    with >= 2 qualifying study players (n_shared); NaN if n_shared < 3 or the denominator is <= 0.
    `se`: leave-one-team-season-out jackknife SE. `reliability`: split-half reliability of the same player x team
    means; when it is low the denominator is near 0 and the ratio is unstable (check se)."""
    teams = [team_col] if isinstance(team_col, str) else team_col
    d = df.filter(pl.col(value).is_finite()).with_columns(_half=pl.col(unit).rank("dense").over(teams) % 2)
    m = (d.group_by(player, *teams)
         .agg(a=pl.col(value).filter(pl.col("_half") == 0).mean(), b=pl.col(value).filter(pl.col("_half") == 1).mean(),
              na=(pl.col("_half") == 0).sum(), nb=(pl.col("_half") == 1).sum())
         .filter((pl.col("na") >= min_per_half) & (pl.col("nb") >= min_per_half)))
    base = {"n_player_teams": m.height, "n_teams": m.n_unique(teams)}
    m = m.filter(pl.len().over(teams) >= 2)
    base["n_shared"] = m.n_unique(teams)
    if base["n_shared"] < 3:
        return {"team_share": np.nan, "se": np.nan, "reliability": np.nan, **base}
    t = (m.with_columns(pl.col("a") - pl.col("a").mean(), pl.col("b") - pl.col("b").mean())
         .group_by(teams).agg(sa=pl.col("a").sum(), sb=pl.col("b").sum(), own=(pl.col("a") * pl.col("b")).sum(),
                              k=pl.len().cast(pl.Float64)))
    own, k = t["own"].to_numpy(), t["k"].to_numpy()
    mates, pairs = (t["sa"] * t["sb"]).to_numpy() - own, k * (k - 1)
    r = float(np.corrcoef(m["a"], m["b"])[0, 1])
    res = {"team_share": np.nan, "se": np.nan, "reliability": 2 * r / (1 + r), **base}
    if own.sum() > 0:
        res["team_share"] = float(mates.sum() / pairs.sum() / (own.sum() / k.sum()))
        jack = (mates.sum() - mates) / (pairs.sum() - pairs) / ((own.sum() - own) / (k.sum() - k))
        res["se"] = float(np.sqrt((len(k) - 1) * np.var(jack)))
    return res
