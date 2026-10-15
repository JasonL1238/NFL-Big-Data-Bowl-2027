import numpy as np
import polars as pl
import pytest

from bdb27.linkage import evaluate

FAST = dict(n_repeats=10, n_perm=200, perm_repeats=2, n_perm_fl=2000, n_jobs=1)


def _players(n=120, signal=0.0, seed=0):
    """Synthetic player table: target depends on a control, draft capital, and (optionally) the feature."""
    rng = np.random.default_rng(seed)
    forty = rng.normal(4.6, 0.15, n)
    forty[rng.random(n) < 0.2] = np.nan  # skipped drills
    pick = rng.normal(0, 1, n)
    feature = 0.5 * np.nan_to_num(forty - 4.6) / 0.15 + rng.normal(0, 1, n)  # correlated with a control
    target = -np.nan_to_num(forty - 4.6, nan=0) / 0.15 + 0.5 * pick + signal * feature + rng.normal(0, 1, n)
    return pl.DataFrame({
        "nfl_id": np.arange(n), "position": rng.choice(["WR", "DB"], n), "draft_year": rng.choice([2023, 2024, 2025], n),
        "forty": forty, "pick": pick, "feature": feature, "target": target,
    })


def _eval(df, **kw):
    return evaluate(df, "feature", "target", combine_controls=["forty"], draft_controls=["pick"], **kw)


def test_planted_signal_is_detected():
    res = _eval(_players(signal=0.6), expected_sign=1, **FAST)
    assert res["delta_r2_vs_B2"]["mean"] > 0.05
    assert res["checks"]["excess_lb_gt_0"] and res["checks"]["fl_p_lt_05"] and res["checks"]["sign_matches"]
    assert res["fl_p"] < 0.01 and res["cv_perm_p"] < 0.01
    assert res["partial_r"] > 0.3 and res["partial_r_ci95"][0] > 0


def test_wrong_sign_effect_does_not_pass_a_directional_test():
    res = _eval(_players(signal=-0.6), expected_sign=1, **FAST)
    assert res["fl_p"] > 0.9 and res["checks"]["sign_matches"] is False
    assert res["checks"]["excess_lb_gt_0"] is False  # a wrong-sign gain in CV R^2 must not pass
    assert res["cv_perm_p"] < 0.01  # the sign-agnostic CV statistic would have "passed"


def test_pure_noise_shows_no_gain():
    res = _eval(_players(signal=0.0, seed=1), **FAST)
    assert abs(res["delta_r2_vs_B2"]["mean"]) < 0.03
    assert res["fl_p"] > 0.01 and res["cv_perm_p"] > 0.01


def test_feature_redundant_with_controls_adds_nothing_beyond_B2():
    """A feature that only proxies the 40 time should look useful vs B0 but not vs B2."""
    df = _players(signal=0.0, seed=2).filter(pl.col("forty").is_not_nan()).with_columns(feature=-pl.col("forty"))
    res = _eval(df, **FAST)
    assert res["delta_r2_vs_B0"]["mean"] > 0.1
    assert res["delta_r2_vs_B2"]["mean"] < 0.02


@pytest.mark.slow
def test_null_p_values_are_calibrated():
    res = [_eval(_players(seed=100 + s), n_repeats=2, n_perm=100, perm_repeats=1, n_perm_fl=1000, n_jobs=1, seed=s)
           for s in range(40)]
    for key in ("cv_perm_p", "fl_p"):
        ps = np.array([r[key] for r in res])
        assert np.mean(ps < 0.05) <= 0.15  # nominal 5%; generous bound for 40 draws
        assert 0.3 < np.mean(ps) < 0.7  # roughly uniform


@pytest.mark.slow
def test_null_p_values_are_calibrated_when_feature_is_collinear_with_a_control():
    """The review case: x = 0.9 * control + noise carries no information beyond B2 (10 controls)."""
    ps = []
    for s in range(60):
        rng = np.random.default_rng(1000 + s)
        n = 64
        C = rng.normal(size=(n, 10))
        df = pl.DataFrame({f"c{j}": C[:, j] for j in range(10)}).with_columns(
            nfl_id=pl.Series(np.arange(n)), position=pl.lit("WR"), draft_year=pl.Series(rng.choice([2023, 2024, 2025], n)),
            feature=pl.Series(0.9 * C[:, 0] + np.sqrt(1 - 0.81) * rng.normal(size=n)),
            target=pl.Series(0.45 * C[:, 0] + 0.2 * C[:, 1] + rng.normal(size=n)))
        res = evaluate(df, "feature", "target", combine_controls=[f"c{j}" for j in range(9)],
                       draft_controls=["c9"], n_repeats=2, n_perm=100, perm_repeats=1, n_perm_fl=1000,
                       n_jobs=1, seed=s, expected_sign=1)
        ps.append((res["cv_perm_p"], res["fl_p"]))
    for col in np.array(ps).T:
        assert np.mean(col < 0.05) <= 0.12  # raw-shuffle null gave ~0.2 here
        assert 0.35 < col.mean() < 0.65


def test_no_inference_when_residual_df_is_too_small():
    df = _players(n=14, signal=0.6, seed=4)
    res = evaluate(df, "feature", "target", expected_sign=1, combine_controls=["forty"], draft_controls=["pick"],
                   min_df=10, **FAST)
    assert res["resid_df"] < 10 and res["fl_p"] is None and res["partial_r"] is None
    assert res["checks"]["fl_p_lt_05"] is False


def test_loco_criterion_does_not_pass_pure_noise_often():
    passes = [_eval(_players(signal=0.0, seed=200 + s), expected_sign=1, n_repeats=2, n_perm=100,
                    perm_repeats=1, n_perm_fl=500, n_jobs=1, seed=s)["checks"]["loco_classes_pass"]
              for s in range(12)]
    assert sum(int(p.split("/")[0]) >= 2 for p in passes) <= 3  # old rule passed ~80% of nulls


def test_planted_signal_passes_loco():
    res = _eval(_players(n=240, signal=0.6, seed=5), expected_sign=1, **FAST)
    assert int(res["checks"]["loco_classes_pass"].split("/")[0]) >= 2
