"""Evaluation harness: does a Combine signature predict a game target beyond the baselines?

Every experiment calls `evaluate`, so all candidate links are judged by the same protocol (PLAN.md §4):
  B0 = position + draft-year dummies, B1 = B0 + traditional Combine, B2 = B1 + draft capital,
  full = B2 + feature. Unit = player.

Two complementary tests:
  - Inference (primary, seed-free): Freedman-Lane permutation test of the in-sample partial correlation
    of the feature with the target given B2 (OLS on the imputed design), directional when an
    expected sign is given. y's residuals under B2 are permuted within position.
  - Prediction (secondary): repeated K-fold ridge, Delta R^2 of full vs B2, Nadeau-Bengio corrected SE.
    Fold R^2 = 1 - MSE_fold / Var(y_all). Ridge gives a feature collinear with a control a small positive
    Delta R^2 even under the null, so the excess over a Freedman-Lane permutation-null mean is reported.

History: 2026-10-14 adversarial review (Freedman-Lane null, fold R^2 normalization, excess Delta R^2);
Gate 1 round-1 skeptics (sign-agnostic test, LOCO noise, seed-dependent p, low ridge power,
Combine session/year artifacts -> year dummies, pooled-variance normalization); Gate 2 skeptics
(criterion 2 now requires the expected sign; LOCO counts a class only if its held-out excess exceeds
one null SD and its partial r is not reversed; one-sided MDE when a sign is expected).
"""

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bdb27 import io

COMBINE_CONTROLS = ["combine_weight", "combine_height", "forty", "ten_yd_split", "vertical", "broad_jump",
                    "three_cone", "short_shuttle", "bench_reps", "ngs_athleticism_score"]
DRAFT_CONTROLS = ["log_pick", "undrafted"]
UNDRAFTED_PICK = 258  # one past the last pick in this data (257)


def player_table() -> pl.DataFrame:
    """One row per player: nfl_id, draft_year, position (Combine group), Combine and draft controls."""
    players = io.load("players").select(
        "nfl_id", "display_name", "nfl_position",
        undrafted=pl.col("draft_overall_pick").is_null().cast(pl.Int8),
        log_pick=pl.col("draft_overall_pick").fill_null(UNDRAFTED_PICK).log(),
    )
    combine = io.load("combine_results").select("nfl_id", "draft_year", position="combine_position",
                                                *COMBINE_CONTROLS)
    return combine.join(players, on="nfl_id", how="left")


def _model():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         RidgeCV(alphas=np.logspace(-3, 4, 30)))


def _dummies(d: pl.DataFrame, col: str) -> np.ndarray | None:
    if d[col].n_unique() < 2:
        return None
    return d.select(pl.col(col).cast(pl.String)).to_dummies(drop_first=True).cast(pl.Float64).to_numpy()


def _design(d: pl.DataFrame, cols: list[str], year_effects: bool = True) -> np.ndarray:
    """Position (and draft-year) dummies + the given columns, as float with NaN for missing."""
    parts = [p for p in (_dummies(d, "position"), _dummies(d, "draft_year") if year_effects else None)
             if p is not None]
    if cols:
        parts.append(d.select(cols).cast(pl.Float64).to_numpy())
    return np.hstack(parts) if parts else np.zeros((d.height, 1))


def _fold_r2(X, y, folds, var_y: float) -> np.ndarray:
    """Out-of-fold R^2 per fold, normalized by the full-sample variance of y."""
    out = []
    for tr, te in folds:
        m = _model().fit(X[tr], y[tr])
        out.append(1 - np.mean((y[te] - m.predict(X[te])) ** 2) / var_y)
    return np.array(out)


def _corrected_se(delta: np.ndarray, n_splits: int) -> float:
    """Nadeau-Bengio corrected SE for repeated k-fold differences (plain SE ignores fold overlap)."""
    ratio = 1 / (n_splits - 1)  # n_test / n_train
    return float(np.sqrt((1 / len(delta) + ratio) * np.var(delta, ddof=1)))


def _imputed(X) -> np.ndarray:
    """Median-imputed design + missingness indicators + intercept (for in-sample OLS)."""
    Xi = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True).fit_transform(X)
    return np.column_stack([np.ones(len(Xi)), Xi])


def _residual_maker(Xi: np.ndarray) -> np.ndarray:
    """R = I - H for the OLS projection onto the columns of Xi (pseudo-inverse handles collinearity)."""
    return np.eye(len(Xi)) - Xi @ np.linalg.pinv(Xi)


def _within_perm(rng, groups: np.ndarray) -> np.ndarray:
    """A permutation of row indices that only shuffles within each group."""
    idx = np.arange(len(groups))
    for g in np.unique(groups):
        m = np.flatnonzero(groups == g)
        idx[m] = rng.permutation(m)
    return idx


def _corr(a, b) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / den) if den > 0 else np.nan


def partial_residuals(df: pl.DataFrame, feature: str, target: str,
                      combine_controls: list[str] = COMBINE_CONTROLS, draft_controls: list[str] = DRAFT_CONTROLS,
                      year_effects: bool = True) -> pl.DataFrame:
    """Feature and target residualized on the B2 design (for added-variable plots)."""
    d = df.filter(pl.col(feature).is_finite() & pl.col(target).is_finite()).sort("nfl_id")
    R = _residual_maker(_imputed(_design(d, combine_controls + draft_controls, year_effects)))
    return d.select("nfl_id", *[c for c in ("display_name", "position", "draft_year") if c in d.columns]).with_columns(
        x_res=pl.Series(R @ d[feature].to_numpy().astype(float)),
        y_res=pl.Series(R @ d[target].to_numpy().astype(float)))


def evaluate(df: pl.DataFrame, feature: str, target: str, expected_sign: int | None = None,
             combine_controls: list[str] = COMBINE_CONTROLS, draft_controls: list[str] = DRAFT_CONTROLS,
             year_effects: bool = True, n_splits: int = 5, n_repeats: int = 50, n_perm: int = 2000,
             perm_repeats: int = 5, n_perm_fl: int = 10000, seed: int = 0, n_jobs: int = -1,
             min_df: int = 10) -> dict:
    """Score one feature -> target link. `df` is player-level with nfl_id, position, draft_year, controls,
    feature and target. `expected_sign` (+1/-1) makes the inferential test one-sided and adds a sign check."""
    d = df.filter(pl.col(feature).is_finite() & pl.col(target).is_finite()).sort("nfl_id")
    y = d[target].to_numpy().astype(float)
    x = d[feature].to_numpy().astype(float)
    pos = d["position"].to_numpy()
    years = d["draft_year"].to_numpy()
    n = len(y)
    var_y = float(np.var(y))
    pos_means = d.select(pl.col(target).mean().over("position")).to_series().to_numpy()
    var_within = float(np.var(y - pos_means))
    designs = {
        "B0": _design(d, [], year_effects),
        "B1": _design(d, combine_controls, year_effects),
        "B2": _design(d, combine_controls + draft_controls, year_effects),
    }
    rng = np.random.default_rng(seed)

    # ---- Inference: in-sample partial correlation, Freedman-Lane permutation (seed-free in practice).
    Xi = _imputed(designs["B2"])
    resid_df = int(n - np.linalg.matrix_rank(Xi) - 1)
    R = _residual_maker(Xi)
    rx, ry = R @ x, R @ y
    partial_r = _corr(rx, ry)
    fl_null = np.empty(n_perm_fl)
    for i in range(n_perm_fl):
        fl_null[i] = _corr(rx, R @ ry[_within_perm(rng, pos)])
    stat = partial_r if expected_sign is None else expected_sign * partial_r
    null_stat = np.abs(fl_null) if expected_sign is None else expected_sign * fl_null
    ok_df = resid_df >= min_df  # below this the partial correlation (and its test) is degenerate
    fl_p = (float((1 + np.sum(null_stat >= (abs(stat) if expected_sign is None else stat))) / (n_perm_fl + 1))
            if ok_df else None)
    if ok_df and np.isfinite(partial_r):
        z, se_z = np.arctanh(partial_r), 1 / np.sqrt(max(resid_df - 1, 1))
        ci = [float(np.tanh(z - 1.96 * se_z)), float(np.tanh(z + 1.96 * se_z))]
        z_a = 1.645 if expected_sign is not None else 1.96  # one-sided test when a sign is expected
        mde = float(np.tanh((z_a + 0.84) * se_z))  # 80% power at 5%
    else:
        ci, mde = [None, None], None
    by_class = {}
    for yr in np.unique(years):
        m = years == yr
        by_class[int(yr)] = {"n": int(m.sum()),
                             "partial_r": _corr(rx[m], ry[m]) if ok_df and m.sum() >= min_df else None}

    # ---- Prediction: repeated K-fold ridge.
    folds = list(RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed).split(y))
    r2 = {k: _fold_r2(X, y, folds, var_y) for k, X in designs.items()}
    full_X = np.column_stack([designs["B2"], x])
    r2["full"] = _fold_r2(full_X, y, folds, var_y)
    r2["B0+feature"] = _fold_r2(np.column_stack([designs["B0"], x]), y, folds, var_y)
    delta = r2["full"] - r2["B2"]
    delta_b0 = r2["B0+feature"] - r2["B0"]
    se = _corrected_se(delta, n_splits)

    # CV permutation null (Freedman-Lane on x: permute x's B2 residual within position, add back the fit).
    x_hat = x - rx
    pfolds = folds[: perm_repeats * n_splits]
    base_p = r2["B2"][: len(pfolds)]
    observed = float(np.mean(r2["full"][: len(pfolds)] - base_p))
    perms = [x_hat + rx[_within_perm(rng, pos)] for _ in range(n_perm)]
    null = np.array(Parallel(n_jobs=n_jobs)(
        delayed(lambda xp: float(np.mean(_fold_r2(np.column_stack([designs["B2"], xp]), y, pfolds, var_y) - base_p)))(xp)
        for xp in perms))
    cv_p = float((1 + np.sum(null >= observed)) / (n_perm + 1))
    null_mean = float(null.mean())
    excess = float(delta.mean() - null_mean)

    # Leave-one-draft-class-out, against its own permutation-null mean.
    loco = {}
    n_loco = min(n_perm, 300)
    for yr in np.unique(years):
        tr, te = np.flatnonzero(years != yr), np.flatnonzero(years == yr)
        base = _fold_r2(designs["B2"], y, [(tr, te)], var_y)[0]
        obs = _fold_r2(full_X, y, [(tr, te)], var_y)[0] - base
        nl = Parallel(n_jobs=n_jobs)(delayed(
            lambda xp: _fold_r2(np.column_stack([designs["B2"], xp]), y, [(tr, te)], var_y)[0] - base)(xp)
            for xp in perms[:n_loco])
        loco[int(yr)] = {"delta": float(obs), "excess": float(obs - np.mean(nl)), "null_sd": float(np.std(nl))}

    sign_ok = None if expected_sign is None or not np.isfinite(partial_r) else bool(np.sign(partial_r) == expected_sign)

    def _class_ok(yr, v):  # held-out gain clearly above its null, and not in the wrong direction
        pr = by_class[yr]["partial_r"]
        wrong = expected_sign is not None and pr is not None and np.sign(pr) != expected_sign
        return v["excess"] > v["null_sd"] and not wrong
    loco_pos = sum(_class_ok(yr, v) for yr, v in loco.items())
    scale = var_y / var_within if var_within > 0 else np.nan
    return {
        "feature": feature, "target": target, "n": n, "resid_df": resid_df,
        "positions": sorted(set(pos.tolist())), "expected_sign": expected_sign, "year_effects": year_effects,
        "raw_r": _corr(x, y),
        "partial_r": partial_r if ok_df else None, "partial_r_ci95": ci, "partial_r_mde80": mde,
        "fl_p": fl_p, "fl_one_sided": expected_sign is not None, "n_perm_fl": n_perm_fl,
        "by_class": by_class,
        "cv_r2": {k: float(v.mean()) for k, v in r2.items()},
        "delta_r2_vs_B0": {"mean": float(delta_b0.mean()), "se": _corrected_se(delta_b0, n_splits)},
        "delta_r2_vs_B2": {"mean": float(delta.mean()), "se": se, "null_mean": null_mean, "excess": excess,
                           "excess_within_position": float(excess * scale)},
        "cv_perm_p": cv_p, "n_perm": n_perm,
        "loco": loco,
        "checks": {  # automatable go criteria (PLAN.md §4); reliability, scout story and seed stability are separate
            "fl_p_lt_05": fl_p is not None and fl_p < 0.05,
            "sign_matches": sign_ok,
            "excess_lb_gt_0": bool(excess - se > 0) and sign_ok is not False,  # ridge Delta R^2 ignores sign
            "loco_classes_pass": f"{loco_pos}/{len(loco)}",  # excess > 1 null SD and class sign not reversed
            # Superseded rev-2 rule (excess > 0; passes most nulls), kept so E001/E007/E008 reproduce as logged.
            "loco_excess_positive_classes": f"{sum(v['excess'] > 0 for v in loco.values())}/{len(loco)}",
        },
    }
