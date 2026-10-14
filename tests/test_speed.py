import numpy as np
import polars as pl

from bdb27 import speed
from bdb27.kinematics import DT

T = np.arange(80) * DT
CENTERS = (speed.AS_BINS[:-1] + speed.AS_BINS[1:]) / 2
Z95 = 1.6448536269514722  # N(0, 1) 95th percentile


def _frames(sp, a, n_plays):
    """One player's frames, len(sp) / n_plays frames per play, 4 plays per game."""
    play = np.repeat(np.arange(n_plays), len(sp) // n_plays)
    return pl.DataFrame({"game_id": play // 4 + 1, "play_id": play, "speed": sp, "a_tan": a})


def test_fit_mono_exp_recovers_vmax_tau():
    vmax, tau, t0, t1 = 10.5, 1.1, 0.3, 0.8
    v = speed.mono_exp(T, vmax, tau, t0)
    # Real starts are not mono-exponential: dv/dt rises first. Convex ramp up to t1, where the curve takes over.
    ramp = T < t1 - 1e-9
    v[ramp] = v[np.isclose(T, t1)][0] * ((T[ramp] + 0.5) / (t1 + 0.5)) ** 2
    v += np.random.default_rng(0).normal(0, 0.03, len(T))
    f = speed.fit_mono_exp(T, v)
    assert abs(f["vmax"] - vmax) < 0.05 and abs(f["tau"] - tau) < 0.03 and abs(f["t0"] - t0) < 0.03
    assert np.isclose(f["a0"], f["vmax"] / f["tau"]) and f["rmse"] < 0.05


def test_splits_on_constant_acceleration_trace():
    a = 2.0
    x = speed.START_X + 0.5 * a * T**2  # from rest at the start line
    v = a * T
    s = speed.forty_splits(T, x, v)
    assert np.isclose(s["t_onset"], 1 / a)  # v is linear, so the onset interpolation is exact
    exact = {0: 1 / a, **{m: np.sqrt(2 * m / a) for m in speed.MARKS}}  # 0 = onset (v = 1 yd/s)
    got = speed.crossing_times(T, x - speed.START_X, speed.MARKS)
    # Linear interpolation of x(t) = a t^2 / 2 between frames errs by at most a DT^2 / 8 in yd, so a DT^2 / (8 v) in s.
    for m, t_m in zip(speed.MARKS, got):
        assert abs(t_m - exact[m]) <= a * DT**2 / (8 * a * exact[m]) + 1e-12
    d_on = 0.5 * a * exact[0] ** 2
    for lo, hi in speed.SPLITS:
        split = exact[hi] - exact[lo]
        assert abs(s[f"split_{lo}_{hi}"] - split) < 1e-3
        assert abs(s[f"v_{lo}_{hi}"] - (hi - (d_on if lo == 0 else lo)) / split) < 0.01
    assert np.isclose(speed.crossing_times(T, 3 * T, [10.0])[0], 10 / 3)  # exact for constant speed


def test_splits_missing_onset_and_unreached_marks_are_nan():
    x = speed.START_X + 0.7 + 8.0 * T[:50]  # already moving at the first frame; tracking ends 39.9 yd past the line
    s = speed.forty_splits(T[:50], x, np.full(50, 8.0))
    assert np.isnan(s["t_onset"]) and np.isnan(s["split_0_10"]) and np.isnan(s["v_0_10"])
    assert np.isclose(s["v_10_20"], 8.0) and np.isnan(s["split_30_40"]) and np.isnan(s["v_20_40"])


def test_combine_speed_table_means_qc_passing_attempts():
    base = dict.fromkeys(speed.METRICS, 1.0)
    rows = [{**base, "nfl_id": 1, "qc_ok": True, "vmax": 10.0, "split_0_10": np.nan},
            {**base, "nfl_id": 1, "qc_ok": True, "vmax": 11.0, "split_0_10": 1.8},
            {**base, "nfl_id": 1, "qc_ok": False, "vmax": 50.0},
            {**base, "nfl_id": 2, "qc_ok": False}]
    out = speed.combine_speed_table(pl.DataFrame(rows))
    assert out["nfl_id"].to_list() == [1]
    r = out.row(0, named=True)
    assert r["c40_vmax"] == 10.5 and r["c40_split_0_10"] == 1.8 and r["c40_n_attempts"] == 2


def test_envelope_recovers_A0_S0():
    a0, s0 = 6.0, 10.0
    assert np.allclose(speed.envelope(a0 - a0 / s0 * CENTERS), (a0, s0, -a0 / s0))
    rng = np.random.default_rng(3)
    n = 200 * 45
    sp = rng.choice(CENTERS, n)
    acc = a0 - a0 / s0 * sp + rng.normal(-Z95 * 0.5, 0.5, n)  # per-speed q95 of a_tan is the envelope
    p = speed.as_profile(_frames(sp, acc, 200))
    assert abs(p["A0"] - a0) < 0.15 and abs(p["S0"] - s0) < 0.15 and abs(p["slope"] + a0 / s0) < 0.02
    assert p["n_plays"] == 200 and np.isnan(speed.as_profile(_frames(sp, acc, 200), k=201)["A0"])


def test_draw_estimator_is_exposure_invariant_unlike_a_max():
    """Same per-play distribution, 1x vs 8x as many plays: A0/S0 from fixed-size draws keep their expectation;
    the max of a_tan over all frames grows with the number of plays."""
    rng = np.random.default_rng(0)
    k, fpp = 10, 30

    def player(n_plays, seed):
        cap = rng.uniform(5.0, 10.5, n_plays)  # each play reaches its own top speed
        sp = rng.uniform(3.0, np.repeat(cap, fpp))
        acc = 6.0 - 0.6 * sp + rng.normal(-Z95 * 0.5, 0.5, n_plays * fpp)
        p = speed.as_profile(_frames(sp, acc, n_plays), k=k, min_frames=5, seed=seed)
        return p["A0"], p["S0"], acc.max()

    est = {m: np.array([player(k * m, s) for s in range(80)]) for m in (1, 8)}
    diff = est[8].mean(0) - est[1].mean(0)
    se = np.sqrt(est[8].var(0, ddof=1) / 80 + est[1].var(0, ddof=1) / 80)
    assert abs(diff[0]) < 3 * se[0] and abs(diff[1]) < 3 * se[1]  # A0, S0: no trend
    assert diff[2] > 5 * se[2]  # max: grows with exposure
