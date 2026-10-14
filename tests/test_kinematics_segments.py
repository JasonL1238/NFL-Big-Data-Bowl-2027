from datetime import datetime, timedelta

import numpy as np
import polars as pl

from bdb27.kinematics import add_kinematics, kinematics
from bdb27.reliability import cross_drill_icc, icc1, split_half
from bdb27.segments import breaks, canonicalize, detect_breaks, snap_aligned

T0 = datetime(2024, 1, 1)


def _route(entry=7.0, low=3.0, exit_=8.0, turn=90.0, n=40):
    """Run along +x, brake into a turn of `turn` degrees (positive = clockwise/right), re-accelerate."""
    t = np.arange(n) * 0.1
    speed = np.interp(t, [0, 1.5, 2.0, 2.3, 4.0], [entry, entry, low, low, exit_])
    heading = 90 + turn * np.clip((t - 1.85) / 0.3, 0, 1)  # deg clockwise from +y
    vx, vy = speed * np.sin(np.radians(heading)), speed * np.cos(np.radians(heading))
    return np.cumsum(vx) * 0.1, np.cumsum(vy) * 0.1, speed


def _frames(x, y, traj_key=1, start=T0):
    return pl.DataFrame({"key": traj_key, "time": [start + timedelta(milliseconds=100 * i) for i in range(len(x))],
                         "x": x, "y": y})


def test_circle_clockwise():
    v, R = 6.0, 10.0
    t = np.arange(60) * 0.1
    w = v / R
    k = kinematics(R * np.sin(w * t), R * np.cos(w * t))
    mid = slice(5, -5)
    assert np.allclose(k["speed"][mid], v, rtol=0.01)
    assert np.allclose(k["curvature"][mid], 1 / R, rtol=0.02)
    assert np.allclose(k["turn_rate"][mid], np.degrees(w), rtol=0.02)  # clockwise -> positive
    assert np.allclose(k["a_lat"][mid], v**2 / R, rtol=0.02)  # rightward -> positive
    assert np.allclose(k["a_tan"][mid], 0, atol=0.05)
    assert np.isclose(k["heading"][10], (90 + np.degrees(w * t[10])) % 360, atol=1)


def test_constant_acceleration_along_y():
    t = np.arange(30) * 0.1
    k = kinematics(np.zeros_like(t), 0.5 * 3.0 * t**2)
    assert np.allclose(k["a_tan"][8:-3], 3.0, rtol=0.01)
    assert np.allclose(k["heading"][8:-3] % 360, 0, atol=0.5)
    assert np.allclose(k["turn_rate"][8:-3], 0, atol=0.5)


def test_too_short_trajectory_is_nan():
    k = kinematics(np.array([0.0, 0.1, 0.2]), np.array([0.0, 0.0, 0.0]))
    assert np.isnan(k["speed"]).all()


def test_route_break_detected_with_right_stats():
    x, y, _ = _route(turn=90)
    k = kinematics(x, y)
    br = detect_breaks(k["speed"], k["heading"], k["a_tan"], k["turn_rate"])
    assert len(br) == 1
    b = br[0]
    assert 80 < b["turn_deg"] < 100
    assert 1.8 <= b["t_break"] <= 2.3
    assert abs(b["entry_speed"] - 7) < 0.3 and abs(b["min_speed"] - 3) < 0.3
    assert 6 < b["peak_decel"] < 10  # 4 yd/s lost over 0.5 s = 8 yd/s^2
    assert np.isclose(b["retention"], b["min_speed"] / b["entry_speed"])


def test_left_turn_sign_and_shallow_turn_ignored():
    x, y, _ = _route(turn=-120)
    k = kinematics(x, y)
    br = detect_breaks(k["speed"], k["heading"], k["a_tan"], k["turn_rate"])
    assert len(br) == 1 and br[0]["turn_deg"] < -100
    x, y, _ = _route(turn=20)
    k = kinematics(x, y)
    assert detect_breaks(k["speed"], k["heading"], k["a_tan"], k["turn_rate"]) == []


def test_canonicalize_rotates_translates_and_mirrors():
    x, y, s = _route(turn=90)  # right turn -> path goes to -y before mirroring
    th = np.radians(37)
    xr, yr = 50 + np.cos(th) * x - np.sin(th) * y, 20 + np.sin(th) * x + np.cos(th) * y
    df = _frames(xr, yr).with_columns(s=pl.Series(s), traj=pl.lit(0, pl.Int64))
    out = canonicalize(df)
    xc, yc = out["x"].to_numpy(), out["y"].to_numpy()
    assert abs(xc[0]) < 1e-9 and abs(yc[0]) < 1e-9
    assert np.allclose(yc[:15], 0, atol=1e-6) and (np.diff(xc[:15]) > 0).all()
    assert out["mirrored"].all() and yc[-1] > 0


def test_add_kinematics_splits_on_keys_and_gaps_and_breaks_driver():
    x, y, _ = _route()
    a = _frames(x, y, 1)
    b = _frames(x, y, 2)
    gap = _frames(x[:20], y[:20], 1, start=T0 + timedelta(seconds=60))  # same key, after a time gap
    out = add_kinematics(pl.concat([b, a, gap]), ["key"])
    assert out["traj"].n_unique() == 3
    br = breaks(out, ["key"])
    assert br.height == 2 and sorted(br["key"].to_list()) == [1, 2]


def test_icc1_recovers_known_value_unbalanced():
    rng = np.random.default_rng(0)
    groups, vals = [], []
    for p in range(400):
        mu = rng.normal()
        for _ in range(rng.integers(2, 5)):
            groups.append(p)
            vals.append(mu + rng.normal())  # between var 1, within var 1 -> ICC 0.5
    res = icc1(np.array(vals), np.array(groups))
    assert abs(res["icc"] - 0.5) < 0.07
    assert res["reliability"] > res["icc"] and res["n_single"] == 0


def test_cross_drill_icc_removes_drill_offsets():
    rng = np.random.default_rng(1)
    rows = [{"nfl_id": p, "drill_name": d, "v": mu + off + rng.normal(0, 0.5)}
            for p, mu in enumerate(rng.normal(size=200)) for d, off in (("A", 0), ("B", 10), ("C", -5))]
    res = cross_drill_icc(pl.DataFrame(rows), "v")
    assert abs(res["icc"] - 0.8) < 0.07  # 1 / (1 + 0.25)


def test_split_half_reliability():
    rng = np.random.default_rng(2)
    rows = [{"nfl_id": p, "game_id": g, "v": mu + rng.normal(0, 2)}
            for p, mu in enumerate(rng.normal(size=150)) for g in range(20) for _ in range(3)]
    res = split_half(pl.DataFrame(rows), "v")
    # per-half mean of 30 obs: noise var 4/30 -> r_half = 1/(1+0.133) = 0.88; full-length SB = 0.94
    assert abs(res["r_half"] - 0.88) < 0.05 and res["reliability"] > res["r_half"]


def test_snap_aligned_drops_plays_without_snap():
    times = [T0 + timedelta(milliseconds=100 * i) for i in range(5)]
    df = pl.DataFrame({"game_id": [1] * 5 + [2] * 5, "play_id": 1, "nfl_id": 1, "time": times * 2,
                       "event": [None, "ball_snap", None, None, None] + [None] * 5})
    out = snap_aligned(df.lazy()).collect()
    assert out["game_id"].unique().to_list() == [1]
    assert np.allclose(out["t_rel"], [-0.1, 0, 0.1, 0.2, 0.3])


def _circle(v, omega_deg, n=60):
    """Clockwise circle at speed v (yd/s) and turn rate omega (deg/s)."""
    w = np.radians(omega_deg)
    t = np.arange(n) * 0.1
    R = v / w
    return R * np.sin(w * t), R * np.cos(w * t), R


def test_turn_radius_on_circle_and_turns_over_180_keep_sign():
    x, y, R = _circle(6.0, 60.0)
    k = kinematics(x, y)
    br = detect_breaks(k["speed"], k["heading"], k["a_tan"], k["turn_rate"])
    assert len(br) == 1 and abs(br[0]["turn_radius"] - R) / R < 0.05
    x, y, _ = _circle(5.0, 250.0)  # 250 deg per 1 s window: endpoint wrapping would report ~-110
    k = kinematics(x, y)
    br = detect_breaks(k["speed"], k["heading"], k["a_tan"], k["turn_rate"], min_turn=90)
    assert len(br) == 1 and 230 < br[0]["turn_deg"] < 270


def test_icc1_reliability_counts_single_attempt_players():
    rng = np.random.default_rng(3)
    groups, vals = [], []
    for p in range(600):
        mu = rng.normal()
        for _ in range(2 if p < 100 else 1):  # most players have one attempt
            groups.append(p)
            vals.append(mu + rng.normal())
    res = icc1(np.array(vals), np.array(groups))
    assert res["n_single"] == 500 and res["n_players"] == 600
    # 500 players at ICC (0.5) and 100 at the 2-attempt SB value (0.667) -> ~0.53, not 0.667
    assert abs(res["reliability"] - (500 * res["icc"] + 100 * 2 * res["icc"] / (1 + res["icc"])) / 600) < 1e-9


def test_snap_aligned_accepts_direct_snaps():
    times = [T0 + timedelta(milliseconds=100 * i) for i in range(3)]
    df = pl.DataFrame({"game_id": 1, "play_id": 1, "nfl_id": 1, "time": times, "event": [None, "snap_direct", None]})
    out = snap_aligned(df.lazy()).collect()
    assert np.allclose(out["t_rel"], [-0.1, 0, 0.1])
