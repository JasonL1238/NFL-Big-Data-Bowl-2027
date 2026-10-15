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
    """Same per-play distribution, 1x vs 8x as many plays: A0 (mean over fixed-size draws) keeps its expectation;
    the max of a_tan over all frames grows with the number of plays. S0 is a median over draws, so it is only
    approximately invariant (module docstring); this symmetric generator could not detect its bias, so no claim."""
    rng = np.random.default_rng(0)
    k, fpp = 10, 30

    def player(n_plays, seed):
        cap = rng.uniform(5.0, 10.5, n_plays)  # each play reaches its own top speed
        sp = rng.uniform(3.0, np.repeat(cap, fpp))
        acc = 6.0 - 0.6 * sp + rng.normal(-Z95 * 0.5, 0.5, n_plays * fpp)
        p = speed.as_profile(_frames(sp, acc, n_plays), k=k, r=50, min_frames=5, seed=seed)
        return p["A0"], acc.max()

    est = {m: np.array([player(k * m, s) for s in range(80)]) for m in (1, 8)}
    diff = est[8].mean(0) - est[1].mean(0)
    se = np.sqrt(est[8].var(0, ddof=1) / 80 + est[1].var(0, ddof=1) / 80)
    assert abs(diff[0]) < 3 * se[0]  # A0: no trend
    assert diff[1] > 5 * se[1]  # max: grows with exposure


def _envelope_frames(n_plays, a0=6.0, s0=10.0, fpp=45, seed=3):
    """Frames whose per-speed q95 of a_tan is the line A0 - (A0 / S0) S."""
    rng = np.random.default_rng(seed)
    sp = rng.choice(CENTERS, n_plays * fpp)
    return _frames(sp, a0 - a0 / s0 * sp + rng.normal(-Z95 * 0.5, 0.5, n_plays * fpp), n_plays)


def test_default_draw_count_makes_profiles_seed_stable():
    """The random-draw (Monte Carlo) noise does not shrink with more plays, only with more draws: at AS_R it must
    be a small fraction of what R = 20 gives (expected sqrt(20 / AS_R))."""
    fr = _envelope_frames(60, fpp=30)
    est = {r: np.array([[speed.as_profile(fr, r=r, seed=s)[m] for m in ("A0", "S0")] for s in range(8)])
           for r in (20, speed.AS_R)}
    sd = {r: e.std(0, ddof=1) for r, e in est.items()}
    assert (sd[speed.AS_R] < 0.4 * sd[20]).all()


def test_game_profiles_per_player_drop_glitches_and_split_by_game():
    a = _envelope_frames(120, seed=1)
    b = _envelope_frames(120, a0=7.0, s0=9.0, seed=2)
    glitch = pl.DataFrame({"game_id": [1] * 400, "play_id": [0] * 400, "speed": [5.25] * 400, "a_tan": [20.0] * 400})
    fr = pl.concat([a.with_columns(nfl_id=pl.lit(1)), b.with_columns(nfl_id=pl.lit(2)),
                    glitch.with_columns(nfl_id=pl.lit(1))]).select("game_id", "play_id", "nfl_id", "speed", "a_tan")
    p = speed.game_as_profiles(fr, r=50)
    assert p.columns == ["nfl_id", "A0", "S0", "slope", "n_plays"] and p["n_plays"].to_list() == [120, 120]
    assert np.allclose(p["A0"], [6.0, 7.0], atol=0.2) and np.allclose(p["S0"], [10.0, 9.0], atol=0.2)
    # odd games (dense rank 1, 3, ... -> half 1) from player 1's envelope, even games (half 0) from player 2's
    mixed = pl.concat([a.filter(pl.col("game_id") % 2 == 1), b.filter(pl.col("game_id") % 2 == 0)])
    sh = speed.game_as_split_half(mixed.with_columns(nfl_id=pl.lit(3)), r=50)
    assert sh["nfl_id"].to_list() == [3] and sh["n_plays_h0"].item() == sh["n_plays_h1"].item() == 60
    assert np.allclose(sh.select("A0_h1", "S0_h1").row(0), (6.0, 10.0), atol=0.3)
    assert np.allclose(sh.select("A0_h0", "S0_h0").row(0), (7.0, 9.0), atol=0.3)


def test_game_profiles_empty_inputs_give_empty_tables():
    empty = pl.DataFrame(schema={"game_id": pl.Int64, "play_id": pl.Int64, "nfl_id": pl.Int64,
                                 "speed": pl.Float64, "a_tan": pl.Float64})
    p = speed.game_as_profiles(empty)
    assert p.height == 0 and p.columns == ["nfl_id", "A0", "S0", "slope", "n_plays"]
    one_game = _envelope_frames(60, fpp=10).with_columns(game_id=pl.lit(1), nfl_id=pl.lit(7))
    sh = speed.game_as_split_half(one_game, r=5)  # every frame falls in one half
    assert sh.height == 0 and {"A0_h0", "S0_h1"} <= set(sh.columns)


def test_split_half_reliability_is_spearman_brown_of_the_half_correlation():
    rng = np.random.default_rng(6)
    n = 4000
    t = rng.normal(0, 1, n)
    sh = pl.DataFrame({"nfl_id": np.arange(n), "A0_h0": t + rng.normal(0, 1, n), "A0_h1": t + rng.normal(0, 1, n)})
    r = speed.split_half_reliability(sh, "A0")  # true half r = 1 / (1 + 1) = 0.5, full reliability 2/3
    assert abs(r["r_half"] - 0.5) < 0.03 and abs(r["reliability"] - 2 / 3) < 0.03 and r["n_players"] == n
    assert np.isclose(r["reliability"], 2 * r["r_half"] / (1 + r["r_half"]))
    small = sh.head(10).with_columns(A0_h0=pl.when(pl.col("nfl_id") == 0).then(np.nan).otherwise(pl.col("A0_h0")))
    out = speed.split_half_reliability(small, "A0")  # the NaN row is dropped, leaving 9 < 10 players
    assert out["n_players"] == 9 and np.isnan(out["reliability"])


def test_attempt_icc_recovers_planted_icc_after_centering():
    rng = np.random.default_rng(5)
    n = 600
    pos, year = rng.choice(["WR", "DL"], n), rng.choice([2023, 2024, 2025], n)
    true = rng.normal(0, 1, n) + np.where(pos == "WR", 10.0, 0.0) + 3.0 * (year - 2024)  # + position/year offsets
    pid = np.repeat(np.arange(n), 3)
    qc = np.tile([True, True, False], n)
    val = np.where(qc, true[pid] + rng.normal(0, 0.5, 3 * n), 1e6)  # failed attempts must not count
    a = pl.DataFrame({"nfl_id": pid, "qc_ok": qc, "position": pos[pid], "draft_year": year[pid],
                      **{m: val for m in speed.METRICS}})
    icc = speed.attempt_icc(a)
    assert icc["metric"].to_list() == speed.METRICS
    assert np.allclose(icc["icc"], 1 / (1 + 0.5**2), atol=0.05)


def test_add_qc_applies_each_limit():
    ok = dict(path=40.0, rmse=0.05, n_fit=20, tau=1.0, end_drop=0.2)
    edges = dict(path=38.0, rmse=0.15, n_fit=15, tau=3.0, end_drop=1.0)  # limits are inclusive
    bad = [("path", 37.9), ("path", 43.1), ("rmse", 0.16), ("n_fit", 14), ("tau", 0.29), ("tau", 3.01),
           ("end_drop", 1.01), ("rmse", np.nan), ("tau", np.nan), ("path", None)]
    a = speed.add_qc(pl.DataFrame([ok, edges, *({**ok, c: v} for c, v in bad)]))
    assert a["qc_ok"].to_list() == [True, True] + [False] * len(bad)
    assert a["two_phase"].to_list() == [False, False, False, False, True] + [False] * (len(bad) - 3)


def test_fv_profile_closed_form_without_air_resistance(monkeypatch):
    """With k = 0, F_H = m a = m vmax / tau - (m / tau) v exactly: F0 = m vmax / tau, V0 = vmax, Sfv = -m / tau."""
    monkeypatch.setattr(speed, "air_k", lambda *a, **kw: 0.0)
    m, vmax, tau = 100.0, 10.0, 1.1
    f = speed.fv_profile(vmax, tau, m, 1.88, 5.0)
    v = vmax * speed.YD
    assert np.isclose(f["F0"], m * v / tau) and np.isclose(f["V0"], v) and np.isclose(f["Sfv"], -m / tau)
    assert np.isclose(f["Pmax"], m * v / tau * v / 4) and np.isclose(f["F0_rel"], v / tau)
    assert np.isclose(f["Pmax_rel"], f["Pmax"] / m)
    for bad in (speed.fv_profile(np.nan, tau, m, 1.88, 5.0), speed.fv_profile(vmax, tau, m, 1.88, 0.0)):
        assert np.isnan(list(bad.values())).all()


def test_onset_needs_three_frames_and_a_clean_start():
    t = np.arange(40) * DT
    # a single fast first frame (SG edge estimate) is not 'already moving'
    assert np.isclose(speed.onset_time(t, np.r_[1.2, 0.5, 0.2, np.linspace(1.0, 8.0, 37)]), 0.3)
    assert np.isnan(speed.onset_time(t, np.linspace(1.0, 8.0, 40)))  # already moving
    assert np.isnan(speed.onset_time(t, np.full(40, 0.5)))  # never
    rise = np.r_[0.6, 1.2, 1.8, 2.1]
    assert np.isclose(speed.onset_time(t, np.r_[rise, 1.95, np.linspace(2.0, 8.0, 35)]), 0.4 / 0.6 * DT)  # dip 0.15
    v = np.r_[rise, 1.9, 1.6, np.linspace(1.6, 8.0, 34)]  # dip 0.5: pre-start movement, then the sprint
    assert np.isnan(speed.onset_time(t, v))
    v = np.r_[v, np.full(40, 8.0)]
    s = speed.forty_splits(T, speed.START_X + 0.7 + np.cumsum(v) * DT, v)
    assert np.isnan(s["split_0_10"]) and np.isnan(s["v_0_10"]) and np.isfinite(s["v_10_20"])
    # a dip after START_PHASE_SPEED is part of the run, not the start
    v = np.r_[rise, np.linspace(2.5, 5.0, 6), 4.4, np.linspace(4.6, 8.0, 29)]
    assert np.isclose(speed.onset_time(t, v), 0.4 / 0.6 * DT)


def test_smax_excludes_sg_edge_frames():
    v = speed.mono_exp(T, 10.0, 1.0, 0.0)
    v[[0, -1, -2]] += 0.5  # edge-polynomial overshoot
    x = speed.START_X + np.cumsum(v) * DT
    m = speed.attempt_metrics(T, x, v)
    assert m["smax"] == v[speed.EDGE:-speed.EDGE].max() < v[-1]
    assert np.isclose(m["end_drop"], 0.0) and m["dur"] == T[-1]
