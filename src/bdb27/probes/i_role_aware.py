"""Probe I (E013): role-aware Combine Exchange Rate. Once a rookie's role is known from his first games, how much do
the Combine and draft add, and how fast do his own games take over? Prereg: experiments/results/E013_prereg.json.

Builds on probe H (E012, h_game_equivalents): the same game values, information sets, cross-validation, CGE, in-season
blend (amendment A1 weight), bootstrap and inference helpers are imported, not re-derived. New here:
  - role share from player_play.lined_up_position (ROLE_ALIGN) over a given set of games: a trait's qualifying games
    (fit) or a rookie's first k qualifying games (in season)
  - role-aware information sets: C0_role / C2_role / C3_role = E012's C0 / C2 / C3 + role share (linear)
  - variance components after draft year + role share (stratified one-way ANOVA with player-level covariates)
  - CGE beyond role and its identity with the role-aware exchange rate
  - team-cluster sign flips and bootstrap (a sensitivity for the paired tests)
  - simulation at a trait's own design with a planted role effect and a planted Combine signal beyond role
  - post hoc helpers (E013 review, end of file): the ANOVA role share and its within-class permutation test, the
    expected blend error, the minimum detectable ratio, regression calibration of a short-window share, both orders
    of attribution, and the dummies-only ridge reference with its bootstrap on E012's draws

Derivations (each checked against simulation in tests/test_role_aware.py):

(1) Role-aware prior and its worth. Under E012's model y_ig = theta_i + e_ig (tau2 = Var theta within position after
    draft-year centering, sigma2 = Var e, lambda = sigma2 / tau2), an information set C that forecasts theta with
    true-score R2_C is a prior N(mu_C, (1 - R2_C) tau2). With ybar_k ~ N(theta, sigma2 / k) the posterior weight on
    ybar_k is (1 - R2_C) tau2 / ((1 - R2_C) tau2 + sigma2 / k) = k / (k + lambda / (1 - R2_C)), so the prior is worth
    lambda / (1 - R2_C) = lambda + CGE_C games (h_game_equivalents.blend_weight). C0_role is such a C: its prior is
    worth lambda_role = sigma2 / tau2_role games, tau2_role = (1 - R2_role) tau2 the variance role leaves.
(2) CGE beyond role. A richer set C (R2_C >= R2_role) has partial R2 beyond role P = (R2_C - R2_role) / (1 - R2_role).
    Measured against the role-aware prior, its worth is lambda_role P / (1 - P). Since 1 - P = (1 - R2_C) /
    (1 - R2_role):
        lambda_role P / (1 - P) = lambda (R2_C - R2_role) / ((1 - R2_role)(1 - R2_C)) = CGE_C - CGE_role.
    So the prior's worth splits additively: lambda (position average) + CGE_role (role) + [CGE_C - CGE_role] (Combine
    and draft beyond role) = lambda / (1 - R2_C). The reported 'beyond role' value is the difference of the two
    clipped CGEs, which can be negative when the extra columns make the out-of-sample forecast worse.
(3) tau2_role by ANOVA with covariates (the 'between-player variance left after year + role share'). Player means
    m_i = mu_s(i) + beta' x_i + u_i + ebar_i, Var u = tau2_role, Var ebar_i = sigma2 / n_i. Let X (k x p) be the
    stratum indicators plus the covariates, W = diag(n_i), M = W - W X (X'WX)^-1 X'W (so M X = 0). The weighted
    residual sum of squares SSB = m' M m has
        E[SSB] = tau2_role tr(M) + sigma2 tr(M W^-1) = tau2_role (N - tr((X'WX)^-1 X'W^2 X)) + sigma2 (k - p).
    With MSB = SSB / (k - p), n0 = (N - tr((X'WX)^-1 X'W^2 X)) / (k - p) and MSW the within-player mean square,
    tau2_role = (MSB - MSW) / n0. With no covariates tr(...) = sum_y sum_{i in y} n_i^2 / N_y: E012's estimator.
(4) Team-cluster sign flip. The paired statistic S = sum_i d_i = sum_c D_c, D_c = sum over the players of team c. If
    under the null the D_c are independent and symmetric about 0, flipping their signs gives an exact null; with one
    player per team it is the player-level test.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from bdb27.probes import h_game_equivalents as H

ROLE = "role_share"
DEEP = "deep_share"
DEEP_ROUTES = ("GO", "POST", "CORNER", "WHEEL")
TRAITS = ("T1", "T2", "T3", "T5", "T6", "T7", "T9", "T11", "T12")  # E012's 9 priced traits (prereg)
ROLE_ALIGN = {  # position -> (numerator alignments, denominator alignments); every other alignment is ignored
    "WR": (("SLOT_WR",), ("SLOT_WR", "WR")),
    "DL": (("EDGE", "OLB"), ("EDGE", "OLB", "INTERIOR_LINE")),
    "OL": (("T",), ("T", "G", "C")),
    "DB": (("CB", "SLOT_CB"), ("CB", "SLOT_CB", "HIGH_SAFETY", "BOX_SAFETY", "OLB", "MLB", "EDGE", "INTERIOR_LINE")),
}
ROLE_NAME = {"WR": "slot share", "DL": "edge share", "OL": "tackle share", "DB": "corner share"}
ROLE_SETS = ("C0_role", "C2_role", "C3_role")


def role_sets(matched: str | None, extra: tuple[str, ...] = (), suffix: str = "") -> dict[str, list[str] | None]:
    """E012's C0 / C2 / C3 plus the role-aware sets (columns added on top of the draft-year dummies; None = C0).
    extra: further usage columns appended to every role set (sets named C*_role<suffix>)."""
    base = H.info_sets(matched)
    add = [ROLE, *extra]
    out = {"C0": None, "C2": base["C2"], "C3": base["C3"]}
    for s, cols in (("C0", []), ("C2", base["C2"]), ("C3", base["C3"])):
        out[f"{s}_role{suffix}"] = cols + add
    return out


# =====================================================================================================
# Role share
# =====================================================================================================

def game_alignment_counts(plays: pl.DataFrame, pos: str) -> pl.DataFrame:
    """plays: one row per player-play (nfl_id, game_id, lined_up_position[, route_ran]) -> one row per player-game:
    role_num / role_den = plays at the position's numerator / denominator alignments (others, and null, ignored);
    with route_ran also deep_num = routes in DEEP_ROUTES, deep_den = routes run."""
    num, den = ROLE_ALIGN[pos]
    assert set(num) <= set(den)
    lp = pl.col("lined_up_position")
    aggs = [lp.is_in(list(num)).fill_null(False).sum().cast(pl.Float64).alias("role_num"),
            lp.is_in(list(den)).fill_null(False).sum().cast(pl.Float64).alias("role_den"), pl.len().alias("n_rows")]
    if "route_ran" in plays.columns:
        rr = pl.col("route_ran")
        aggs += [rr.is_in(list(DEEP_ROUTES)).fill_null(False).sum().cast(pl.Float64).alias("deep_num"),
                 rr.is_not_null().sum().cast(pl.Float64).alias("deep_den")]
    return plays.group_by("nfl_id", "game_id").agg(aggs).sort("nfl_id", "game_id")


def role_share(counts: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """Each player's shares over the given games (nfl_id, game_id): role_share = sum role_num / sum role_den (null if
    the denominator is 0), deep_share likewise when counts carry routes; plus the denominators and the number of
    those games found in counts. Games without player_play rows contribute nothing."""
    c = counts.join(games.select("nfl_id", "game_id").unique(), on=["nfl_id", "game_id"], how="semi")
    aggs = [pl.col("role_num").sum(), pl.col("role_den").sum(), pl.len().alias("n_games_role")]
    deep = "deep_num" in counts.columns
    if deep:
        aggs += [pl.col("deep_num").sum(), pl.col("deep_den").sum()]
    a = c.group_by("nfl_id").agg(aggs)
    out = a.with_columns(pl.when(pl.col("role_den") > 0).then(pl.col("role_num") / pl.col("role_den"))
                         .otherwise(None).alias(ROLE))
    if deep:
        out = out.with_columns(pl.when(pl.col("deep_den") > 0).then(pl.col("deep_num") / pl.col("deep_den"))
                               .otherwise(None).alias(DEEP))
    keep = ["nfl_id", ROLE, "role_den", "n_games_role"] + ([DEEP, "deep_den"] if deep else [])
    return out.select(keep).sort("nfl_id")


def first_k_games(games: pl.DataFrame, k: int) -> pl.DataFrame:
    """Each player's chronologically first k games (game_id order, E012 choice G2)."""
    return (games.sort("nfl_id", "game_id")
            .filter(pl.int_range(pl.len()).over("nfl_id") < k))


# =====================================================================================================
# Variance components after year + covariates
# =====================================================================================================

def impute_cols(C: np.ndarray) -> np.ndarray:
    """Median-impute each column and append a missing indicator for any column with missing values (the same
    convention as linkage._model's SimpleImputer(add_indicator=True)); constant columns are dropped."""
    C = np.asarray(C, float).reshape(len(C), -1)
    out = []
    for j in range(C.shape[1]):
        c = C[:, j]
        miss = ~np.isfinite(c)
        if miss.all():
            continue
        out.append(np.where(miss, np.median(c[~miss]), c))
        if miss.any():
            out.append(miss.astype(float))
    out = [c for c in out if np.ptp(c) > 0]
    return np.column_stack(out) if out else np.zeros((len(C), 0))


def variance_components_cov(values: np.ndarray, players: np.ndarray, strata: np.ndarray,
                            cov: np.ndarray | None = None) -> dict:
    """Stratified one-way random-effects ANOVA with player-level covariates (module docstring, derivation 3).
    values, players, strata: per game; cov: per game (n_games x q, constant within player), or None. Covariates are
    median-imputed with missing indicators (impute_cols). Returns tau2 (= tau2_role, the between-player variance left
    after strata + covariates), sigma2, lambda = sigma2 / tau2 (inf if tau2 <= 0), p (columns of X), n0, msb, msw."""
    values = np.asarray(values, float)
    players = np.asarray(players)
    strata = np.asarray(strata)
    ok = np.isfinite(values)
    values, players, strata = values[ok], players[ok], strata[ok]
    labels, inv = np.unique(players, return_inverse=True)
    k, N = len(labels), len(values)
    n_i = np.bincount(inv).astype(float)
    m_i = np.bincount(inv, weights=values) / n_i
    first = np.zeros(k, int)
    first[inv[::-1]] = np.arange(N)[::-1]
    s_i = strata[first]
    if np.any(strata != s_i[inv]):
        raise ValueError("a player appears in more than one stratum")
    D = np.column_stack([(s_i == s).astype(float) for s in np.unique(s_i)])
    if cov is not None:
        cov = np.asarray(cov, float)[ok].reshape(N, -1)
        c_i = cov[first]
        if not np.allclose(np.nan_to_num(cov, nan=-9e9), np.nan_to_num(c_i[inv], nan=-9e9)):
            raise ValueError("a covariate varies within a player")
        X = np.hstack([D, impute_cols(c_i)])
    else:
        X = D
    ssw = float(np.sum((values - m_i[inv]) ** 2))
    dfw = N - k
    XtWX = X.T @ (n_i[:, None] * X)
    p = int(np.linalg.matrix_rank(XtWX))
    dfb = k - p
    out = {"n_players": int(k), "n_games": int(N), "p": p}
    if dfb < 2 or dfw < 1:
        return out | {"tau2": np.nan, "sigma2": np.nan, "lambda": np.nan}
    G = np.linalg.pinv(XtWX)
    beta = G @ (X.T @ (n_i * m_i))
    ssb = float(np.sum(n_i * (m_i - X @ beta) ** 2))
    tr = float(np.trace(G @ (X.T @ ((n_i ** 2)[:, None] * X))))
    msb, msw = ssb / dfb, ssw / dfw
    n0 = (N - tr) / dfb
    tau2 = (msb - msw) / n0
    return out | {"tau2": float(tau2), "sigma2": float(msw), "lambda": float(msw / tau2) if tau2 > 0 else np.inf,
                  "n0": float(n0), "msb": float(msb), "msw": float(msw)}


# =====================================================================================================
# Estimation: E012's estimate() on the role-aware sets + the role estimands
# =====================================================================================================

KS_TABLE = (1, 4, 8)


def cge_beyond(cge_c: float, cge_ref: float) -> float:
    """Worth of C beyond the reference prior, in games: CGE_C - CGE_ref (module docstring, derivation 2)."""
    return float(cge_c - cge_ref) if np.isfinite(cge_c) and np.isfinite(cge_ref) else np.nan


def beyond_identity(lam: float, r2_c: float, r2_ref: float) -> float:
    """lambda_ref P / (1 - P) with lambda_ref = lambda / (1 - R2_ref), P = (R2_C - R2_ref) / (1 - R2_ref): the worth of
    C measured against the reference prior (equals CGE_C - CGE_ref; derivation 2)."""
    lam_ref = lam / (1 - r2_ref)
    P = (r2_c - r2_ref) / (1 - r2_ref)
    return float(lam_ref * P / (1 - P))


def role_column(td: H.TraitData) -> np.ndarray:
    """The players' role share (the last column of the C0_role design)."""
    return td.X["C0_role"][:, -1]


def estimate_role(td: H.TraitData, n_repeats: int = H.N_REPEATS, seed: int = 0, groups: np.ndarray | None = None,
                  keep_oof: bool = False) -> dict:
    """E012's estimate() on a TraitData whose sets include C0_role (and optionally C2/C3/C2_role/C3_role), plus:
    lambda_role by ANOVA with the role covariate (anova), lambda / (1 - R2_role) (cv), role worth CGE(C0_role),
    CGE beyond role for C2_role / C3_role, tracking beyond C2_role, the change in the Combine + draft value once role
    is known (CGE(C2) - beyond(C2_role)), half-weight games and the trust-table weights per set, and the C0-relative
    role worth under the same CV (E012 PH8 convention, sensitivity)."""
    e = H.estimate(td, n_repeats=n_repeats, seed=seed, groups=groups, keep_oof=keep_oof)
    role = role_column(td)
    vcr = variance_components_cov(td.g_value, td.g_player, td.year[td.g_player], role[td.g_player][:, None])
    lam, r2, c = e["lambda"], e["r2"], e["cge"]
    e["tau2_role_anova"], e["lambda_role_anova"] = vcr["tau2"], vcr["lambda"]
    e["lambda_role_cv"] = lam / (1 - r2["C0_role"]) if np.isfinite(r2["C0_role"]) else np.nan
    e["role_worth"] = c["C0_role"]
    e["beyond_role"] = {s: cge_beyond(c[s], c["C0_role"]) for s in ("C2_role", "C3_role") if s in c}
    if "C3_role" in c and "C2_role" in c:
        e["beyond_role"]["C3_role-C2_role"] = cge_beyond(c["C3_role"], c["C2_role"])
    if "C2" in c and "C2_role" in c:
        e["combine_change_once_role_known"] = cge_beyond(c["C2"], e["beyond_role"]["C2_role"])
    e["half_weight"] = {s: lam / (1 - v) if np.isfinite(v) and np.isfinite(lam) else np.nan for s, v in r2.items()}
    e["weights"] = {s: {str(k): float(H.blend_weight(k, lam, v)) if np.isfinite(v) and np.isfinite(lam) else np.nan
                        for k in KS_TABLE} for s, v in r2.items()}
    e["role_worth_vs_C0_same_cv"] = H.cge(lam, H.relative_true_r2(e["r2_cv"]["C0_role"], e["r2_cv"]["C0"], e["rel"]))
    return e


def flat_estimate(e: dict) -> dict:
    """One bootstrap row from estimate_role()."""
    row = {"lambda": e["lambda"], "tau2": e["tau2"], "sigma2": e["sigma2"], "rel": e["rel"],
           "lambda_role_anova": e["lambda_role_anova"], "tau2_role_anova": e["tau2_role_anova"],
           "lambda_role_cv": e["lambda_role_cv"], "role_worth": e["role_worth"],
           "role_worth_vs_C0_same_cv": e["role_worth_vs_C0_same_cv"],
           **{f"r2cv_{s}": v for s, v in e["r2_cv"].items()}, **{f"r2_{s}": v for s, v in e["r2"].items()},
           **{f"cge_{s}": v for s, v in e["cge"].items()}, **{f"beyond_{s}": v for s, v in e["beyond_role"].items()},
           **{f"khalf_{s}": v for s, v in e["half_weight"].items()},
           **{f"w{k}_{s}": v for s, d in e["weights"].items() for k, v in d.items()}}
    if "combine_change_once_role_known" in e:
        row["combine_change_once_role_known"] = e["combine_change_once_role_known"]
    return row


def _boot_chunk(td: H.TraitData, idx: np.ndarray, b0: int, n_repeats: int) -> list[dict]:
    out = []
    with threadpool_limits(1):
        for j, ids in enumerate(idx):
            e = estimate_role(H.resample(td, ids), n_repeats=n_repeats, seed=b0 + j, groups=ids)
            out.append({"b": b0 + j, **flat_estimate(e)})
    return out


def bootstrap_role(td: H.TraitData, n_boot: int, seed: int, n_jobs: int = 3,
                   n_repeats: int = H.N_REPEATS) -> pl.DataFrame:
    """E012's preregistered player bootstrap (B1) of the whole pipeline, run through estimate_role(): the same player
    draws as h_game_equivalents.bootstrap(td, n_boot, seed) and the same per-resample CV seeds (the resample index),
    so with E012's seed the C0 / C2 / C3 draws reproduce E012's exactly. One row per resample."""
    from joblib import Parallel, delayed
    idx = np.random.default_rng(seed).integers(0, len(td.nfl_id), (n_boot, len(td.nfl_id)))
    chunks = np.array_split(np.arange(n_boot), max(1, n_jobs * 4))
    res = Parallel(n_jobs=n_jobs)(delayed(_boot_chunk)(td, idx[c], int(c[0]), n_repeats) for c in chunks if len(c))
    return pl.DataFrame([r for chunk in res for r in chunk]).sort("b")


# =====================================================================================================
# In-season forecasts with a role share known from the first k games
# =====================================================================================================

def in_season_role_forecasts(fit: H.ProspectiveFit, games_test: pl.DataFrame, players_by_k: dict[int, pl.DataFrame],
                             ks=(1, 2, 3, 4)) -> pl.DataFrame:
    """h_game_equivalents.in_season_forecasts at each k separately, with the test players' role share (and any usage
    column) taken from players_by_k[k] (built from his first k qualifying games). Rows with k = 0 are dropped (no role
    prior before the first game). Columns as in_season_forecasts, for every set of the fit."""
    rows = []
    for k in ks:
        f = H.in_season_forecasts(fit, games_test, players_by_k[k], ks=(k,))
        rows.append(f.filter(pl.col("k") == k))
    return pl.concat(rows, how="diagonal_relaxed")


# =====================================================================================================
# Paired comparisons: player-level (E012) and team-cluster (sensitivity)
# =====================================================================================================

def cluster_sign_flip_p(d: np.ndarray, clusters: np.ndarray, n_flips: int = 100_000, seed: int = 0,
                        alternative: str = "less") -> float:
    """Sign-flip p of S = sum d with the signs flipped per cluster (module docstring, derivation 4)."""
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    return H.sign_flip_p(np.bincount(inv, weights=np.asarray(d, float)), n_flips, seed, alternative)


def cluster_pooled_bootstrap(df: pl.DataFrame, cols: list[str], cluster_of: dict, n_boot: int = 10_000,
                             seed: int = 0) -> dict[str, np.ndarray]:
    """h_game_equivalents.pooled_bootstrap with whole clusters resampled: each draw resamples the clusters with
    replacement and keeps every player of a drawn cluster (with multiplicity)."""
    ids, M, V, _ = H._matrices(df, cols)
    cl = np.array([cluster_of[int(i)] for i in ids])
    _, inv = np.unique(cl, return_inverse=True)
    nc = inv.max() + 1
    rng = np.random.default_rng(seed)
    W = rng.multinomial(nc, np.full(nc, 1 / nc), size=n_boot).astype(float)[:, inv]
    cnt = W @ M
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for c in cols:
            out[c] = np.nanmean((W @ V[c]) / np.where(cnt > 0, cnt, np.nan), axis=1)
    return out


def paired_test(df: pl.DataFrame, a: str, b: str, seed: int, cluster_of: dict | None = None,
                n_flips: int = 100_000, n_boot: int = 10_000) -> dict:
    """E012's paired comparison (pooled trait-weighted mean of a vs b; player-level one-sided sign flips, expected
    a < b; player-bootstrap 95% CIs) plus, with cluster_of, the team-cluster sign-flip p and cluster-bootstrap CIs."""
    d = df.select("nfl_id", "trait", a, b).filter(pl.col(a).is_finite() & pl.col(b).is_finite())
    ids, di = H.player_diffs(d, a, b)
    q = lambda v: [float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))]  # noqa: E731
    pa, pb = H.pooled_mean(d, a), H.pooled_mean(d, b)
    bs = H.pooled_bootstrap(d, [a, b], n_boot, seed)
    out = {"a": a, "b": b, "n_players": int(len(ids)), "n_player_traits": d.height,
           "n_by_trait": dict(d.group_by("trait").len().sort("trait").iter_rows()),
           "pooled_a": pa, "pooled_b": pb, "diff": pa - pb, "ratio": pa / pb if pb > 0 else None,
           "diff_ci95": q(bs[a] - bs[b]), "ratio_ci95": q(bs[a] / bs[b]),
           "pooled_a_ci95": q(bs[a]), "pooled_b_ci95": q(bs[b]),
           "p_one_sided": H.sign_flip_p(di, n_flips, seed, "less"), "expected_sign": "negative (a < b)",
           "n_flips": n_flips, "n_boot": n_boot}
    if cluster_of is not None:
        cl = np.array([cluster_of[int(i)] for i in ids])
        cb = cluster_pooled_bootstrap(d, [a, b], cluster_of, n_boot, seed + 1)
        out["team_cluster"] = {"n_clusters": int(len(np.unique(cl))),
                               "p_one_sided": cluster_sign_flip_p(di, cl, n_flips, seed + 2, "less"),
                               "diff_ci95": q(cb[a] - cb[b]), "ratio_ci95": q(cb[a] / cb[b])}
    return out


# =====================================================================================================
# Simulation at a trait's own design (calibrated bounds and interval coverage)
# =====================================================================================================

def standardize(x: np.ndarray) -> np.ndarray:
    """Median-impute and standardize (population SD); zeros if constant."""
    x = np.asarray(x, float)
    x = np.where(np.isfinite(x), x, np.nanmedian(x))
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def residualize(s: np.ndarray, z: np.ndarray) -> np.ndarray:
    """s minus its OLS fit on [1, z], standardized (in-sample orthogonal to z)."""
    A = np.column_stack([np.ones(len(z)), z])
    r = s - A @ np.linalg.lstsq(A, s, rcond=None)[0]
    return standardize(r)


@dataclass
class SimDesign:
    """A trait's own design for simulation: players, classes, game counts, the sets' design matrices, the standardized
    role share z and the Combine signal matrix Zc (columns of the set the signal is planted on)."""
    td: H.TraitData
    z: np.ndarray
    Zc: np.ndarray


def sim_design(td: H.TraitData, sets: tuple[str, ...], signal_set: str | None) -> SimDesign:
    """sets: the information sets to keep (e.g. ('C0', 'C0_role', 'C2_role')); signal_set: the set whose Combine
    columns carry the planted signal (e.g. 'C2_role'; the role column is excluded), or None (role only)."""
    t = H.TraitData(nfl_id=td.nfl_id, year=td.year, X={s: td.X[s] for s in sets}, g_player=td.g_player,
                    g_value=td.g_value, levels=td.levels)
    z = standardize(role_column(td))
    Zc = np.zeros((len(td.nfl_id), 0)) if signal_set is None else \
        H.signal_matrix(td.X[signal_set][:, :-1], len(td.levels) - 1)
    return SimDesign(td=t, z=z, Zc=Zc)


def simulate_role_values(des: SimDesign, lam: float, rho: float, g: float, rng: np.random.Generator,
                         n_signal: int = 3) -> tuple[H.TraitData, np.ndarray]:
    """theta = sqrt(rho) z + sqrt(1 - rho) (sqrt(g) s_perp + sqrt(1 - g) u): role explains rho of tau2 = 1 (population
    R2 of role) and a Combine signal explains g of what role leaves (partial R2 beyond role); s = signed sum of
    n_signal random columns of Zc, residualized on z; u ~ N(0, 1); no class effects; games = theta + N(0, lam)."""
    n = len(des.z)
    if g > 0 and des.Zc.shape[1]:
        cols = rng.choice(des.Zc.shape[1], min(n_signal, des.Zc.shape[1]), replace=False)
        s = residualize(des.Zc[:, cols] @ rng.choice([-1.0, 1.0], len(cols)), des.z)
    else:
        s = np.zeros(n)
    theta = np.sqrt(rho) * des.z + np.sqrt(1 - rho) * (np.sqrt(g) * s + np.sqrt(1 - g) * rng.normal(size=n))
    vals = theta[des.td.g_player] + rng.normal(0, np.sqrt(lam), len(des.td.g_player))
    return H.TraitData(nfl_id=des.td.nfl_id, year=des.td.year, X=des.td.X, g_player=des.td.g_player, g_value=vals,
                       levels=des.td.levels), theta


def population_beyond(lam: float, rho: float, g: float) -> float:
    """Population CGE beyond role of a perfectly learned signal: lambda_role g / (1 - g), lambda_role = lambda/(1-rho)."""
    return float(lam / (1 - rho) * g / (1 - g))


# =====================================================================================================
# Post hoc (E013 review response, 2026-10-19): helpers for analyses decided after the results were seen
# =====================================================================================================

def role_variance_share(td: H.TraitData) -> dict:
    """In-sample share of the between-player variance that role share explains, by the preregistered ANOVA
    (derivation 3): share = 1 - tau2_role / tau2 (tau2 after draft year only, tau2_role after year + role share; the
    within-player mean square is the same in both), and its worth in games, lambda_role - lambda = sigma2 / tau2_role
    - sigma2 / tau2 (derivation 1: the population worth of a perfectly learned role prior). Unlike CGE(C0_role) this
    does not pay for learning the class means and the role slope out of fold."""
    vc = H.variance_components(td.g_value, td.g_player, td.year[td.g_player])
    vr = variance_components_cov(td.g_value, td.g_player, td.year[td.g_player], role_column(td)[td.g_player][:, None])
    tau2, tau2r, s2 = vc["tau2"], vr["tau2"], vc["sigma2"]
    share = 1 - tau2r / tau2 if tau2 > 0 else np.nan
    worth = s2 / tau2r - s2 / tau2 if tau2 > 0 and tau2r > 0 else np.nan
    return {"tau2": float(tau2), "tau2_role": float(tau2r), "sigma2": float(s2), "lambda": vc["lambda"],
            "lambda_role": vr["lambda"], "share": float(share), "worth": float(worth)}


def _tau2_cov(m_i: np.ndarray, n_i: np.ndarray, msw: float, X: np.ndarray) -> float:
    """tau2 after the columns of X (derivation 3), from player means m_i, game counts n_i and the within-player mean
    square; the same algebra as variance_components_cov."""
    N, k = n_i.sum(), len(n_i)
    XtWX = X.T @ (n_i[:, None] * X)
    p = int(np.linalg.matrix_rank(XtWX))
    G = np.linalg.pinv(XtWX)
    beta = G @ (X.T @ (n_i * m_i))
    ssb = float(np.sum(n_i * (m_i - X @ beta) ** 2))
    tr = float(np.trace(G @ (X.T @ ((n_i ** 2)[:, None] * X))))
    return (ssb / (k - p) - msw) / ((N - tr) / (k - p))


def role_share_permutation(td: H.TraitData, n_perm: int, seed) -> dict:
    """Within-class permutation test of role_variance_share: the players' role shares are shuffled within each draft
    class (null: role share unrelated to the player's level within his class), and the share is recomputed with the
    same ANOVA. One-sided p = (1 + #{null share >= observed}) / (1 + n_perm)."""
    n = len(td.nfl_id)
    n_i = np.bincount(td.g_player, minlength=n).astype(float)
    m_i = np.bincount(td.g_player, weights=td.g_value, minlength=n) / n_i
    vc = H.variance_components(td.g_value, td.g_player, td.year[td.g_player])
    tau2, msw = vc["tau2"], vc["sigma2"]
    D = np.column_stack([(td.year == s).astype(float) for s in np.unique(td.year)])
    role = role_column(td)
    share = lambda r: 1 - _tau2_cov(m_i, n_i, msw, np.hstack([D, impute_cols(r[:, None])])) / tau2  # noqa: E731
    obs = share(role)
    ref = role_variance_share(td)["share"]
    assert np.isclose(obs, ref, rtol=1e-8, atol=1e-10), (obs, ref)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(td.year == s) for s in np.unique(td.year)]
    null = np.empty(n_perm)
    for b in range(n_perm):
        r = role.copy()
        for g in groups:
            r[g] = role[rng.permutation(g)]
        null[b] = share(r)
    return {"observed": float(obs), "p_one_sided": float((1 + np.sum(null >= obs - 1e-12)) / (1 + n_perm)),
            "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95)), "n_perm": int(n_perm)}


def expected_blend_se(k, m, lam: float, r2: float):
    """E[(blend_k - target)^2] / tau2 under E012's model (theta ~ prior with error variance (1 - r2) tau2, games iid
    N(theta, sigma2), lambda = sigma2 / tau2) when the blend uses the matching posterior weight
    w = k / (k + lambda / (1 - r2)) and the target is the mean of m later games:
    (1 - w)^2 (1 - r2) + w^2 lambda / k + lambda / m (the three error terms are independent)."""
    k, m = np.asarray(k, float), np.asarray(m, float)
    w = k / (k + lam / (1 - r2))
    return (1 - w) ** 2 * (1 - r2) + w ** 2 * lam / k + lam / m


def mde_ratio(log_ratio_draws: np.ndarray, alpha: float = 0.05, power: float = 0.8) -> float:
    """Approximate one-sided minimum detectable error ratio (below 1) for a paired ratio test whose log ratio has the
    bootstrap SD of the draws: exp(-(z_{1-alpha} + z_power) SD)."""
    from scipy.stats import norm
    d = np.asarray(log_ratio_draws, float)
    sd = float(np.std(d[np.isfinite(d)], ddof=1))
    return float(np.exp(-(norm.ppf(1 - alpha) + norm.ppf(power)) * sd))


def regression_calibration(noisy: np.ndarray, full: np.ndarray) -> dict:
    """Training-only regression calibration of a short-window share to the full-history share the coefficient was
    trained on: OLS of full on noisy over the pairs where both are finite, E[full | noisy] = a + b noisy. Feeding
    a + b x (not x) to a model trained on full-history shares undoes the over-dispersion a noisier test-time
    predictor otherwise causes."""
    x, y = np.asarray(noisy, float), np.asarray(full, float)
    ok = np.isfinite(x) & np.isfinite(y)
    b, a = np.polyfit(x[ok], y[ok], 1)
    return {"a": float(a), "b": float(b), "n": int(ok.sum()), "r": float(np.corrcoef(x[ok], y[ok])[0, 1])}


def attribution(cge_role: float, cge_c: float, cge_both: float) -> dict:
    """Both orders of entry for role and an information set C (e.g. C2): role beyond C = CGE(C + role) - CGE(C);
    C beyond role = CGE(C + role) - CGE(role); shared = CGE(role) + CGE(C) - CGE(C + role) (what either alone
    carries; equals CGE(C) - [C beyond role], the 'change once role is known')."""
    return {"role_alone": cge_role, "c_alone": cge_c, "both": cge_both, "role_beyond_c": cge_both - cge_c,
            "c_beyond_role": cge_both - cge_role, "shared": cge_role + cge_c - cge_both}


def dummies_ridge(td: H.TraitData, n_repeats: int = H.N_REPEATS, seed: int = 0, groups: np.ndarray | None = None) -> dict:
    """R2_CV of the ridge on the draft-year dummies alone (linkage._model; the folds of estimate() with the same seed
    and groups): the same-CV reference that separates role (or the Combine) from ridge shrinkage of the class means.
    Also lambda and rel, to check that a bootstrap draw is E012's."""
    n = len(td.nfl_id)
    n_i = np.bincount(td.g_player, minlength=n).astype(float)
    y = np.bincount(td.g_player, weights=td.g_value, minlength=n) / n_i
    vc = H.variance_components(td.g_value, td.g_player, td.year[td.g_player])
    var_w = H.within_year_var(y, td.year)
    P = H.cv_oof({"C0_ridge": H.year_dummies(td.year, td.levels)}, y, td.year, groups, n_repeats, seed)["C0_ridge"]
    return {"lambda": vc["lambda"], "rel": vc["rel"],
            "r2cv_C0_ridge": float(np.mean(1 - np.mean((y[None, :] - P) ** 2, axis=1) / var_w))}


def _ridge_chunk(td: H.TraitData, idx: np.ndarray, b0: int, n_repeats: int) -> list[dict]:
    out = []
    with threadpool_limits(1):
        for j, ids in enumerate(idx):
            out.append({"b": b0 + j, **dummies_ridge(H.resample(td, ids), n_repeats=n_repeats, seed=b0 + j, groups=ids)})
    return out


def bootstrap_dummies_ridge(td: H.TraitData, n_boot: int, seed: int, n_jobs: int = 3,
                            n_repeats: int = H.N_REPEATS) -> pl.DataFrame:
    """dummies_ridge on the player draws and CV seeds of bootstrap_role(td, n_boot, seed) (= E012's B1), so each row
    pairs with that draw's stored R2_CV of the other sets. One row per resample (b)."""
    from joblib import Parallel, delayed
    idx = np.random.default_rng(seed).integers(0, len(td.nfl_id), (n_boot, len(td.nfl_id)))
    chunks = np.array_split(np.arange(n_boot), max(1, n_jobs * 4))
    res = Parallel(n_jobs=n_jobs)(delayed(_ridge_chunk)(td, idx[c], int(c[0]), n_repeats) for c in chunks if len(c))
    return pl.DataFrame([r for chunk in res for r in chunk]).sort("b")
