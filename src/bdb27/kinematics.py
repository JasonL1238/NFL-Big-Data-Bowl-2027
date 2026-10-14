"""Smoothed kinematics from x/y at 10 Hz (Savitzky-Golay), per trajectory.

Conventions match the tracking data: heading is degrees clockwise from +y.
Sign conventions: a_tan > 0 speeding up, < 0 braking; turn_rate > 0 turning clockwise (right).
Heading-based quantities are NaN below MIN_HEADING_SPEED, where direction is undefined.
"""

import numpy as np
import polars as pl
from scipy.signal import savgol_filter

DT = 0.1
GAP_S = 0.15  # a time step longer than this starts a new trajectory
MIN_HEADING_SPEED = 0.5  # yd/s
KIN_COLS = ["vx", "vy", "speed", "a_tan", "a_lat", "heading", "turn_rate", "curvature"]


def kinematics(x: np.ndarray, y: np.ndarray, window: int = 7, poly: int = 2) -> dict[str, np.ndarray]:
    """Derivatives of one evenly sampled trajectory. Returns NaNs if it is too short to fit."""
    n = len(x)
    w = min(window, n if n % 2 else n - 1)
    if w < poly + 2:
        return {c: np.full(n, np.nan) for c in KIN_COLS}
    vx, vy = (savgol_filter(v, w, poly, deriv=1, delta=DT) for v in (x, y))
    ax, ay = (savgol_filter(v, w, poly, deriv=2, delta=DT) for v in (x, y))
    speed = np.hypot(vx, vy)
    moving = speed >= MIN_HEADING_SPEED
    with np.errstate(divide="ignore", invalid="ignore"):
        a_tan = (vx * ax + vy * ay) / speed
        cross = vx * ay - vy * ax  # > 0 for counterclockwise (left) turns
        a_lat = np.where(moving, -cross / speed, np.nan)
        turn_rate = np.where(moving, np.degrees(-cross / speed**2), np.nan)
        curvature = np.where(moving, np.abs(cross) / speed**3, np.nan)
    a_tan = np.where(speed > 1e-6, a_tan, 0.0)
    heading = np.where(moving, np.degrees(np.arctan2(vx, vy)) % 360, np.nan)
    return dict(vx=vx, vy=vy, speed=speed, a_tan=a_tan, a_lat=a_lat, heading=heading,
                turn_rate=turn_rate, curvature=curvature)


def trajectory_ids(df: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Sort by keys + time and add `traj` (int): a new id whenever keys change or there is a time gap."""
    df = df.sort([*keys, "time"])
    new = pl.any_horizontal([pl.col(k) != pl.col(k).shift(1) for k in keys])
    gap = (pl.col("time").diff().dt.total_milliseconds() / 1000) > GAP_S
    return df.with_columns(traj=(new | gap).fill_null(True).cum_sum().cast(pl.Int64) - 1)


def add_kinematics(df: pl.DataFrame, keys: list[str], window: int = 7, poly: int = 2) -> pl.DataFrame:
    """Add `traj` and KIN_COLS to frame-level data (one trajectory per keys, split at time gaps)."""
    df = trajectory_ids(df, keys)
    x, y, traj = df["x"].to_numpy(), df["y"].to_numpy(), df["traj"].to_numpy()
    bounds = np.flatnonzero(np.diff(traj)) + 1
    out = {c: np.empty(len(df)) for c in KIN_COLS}
    for s, e in zip(np.r_[0, bounds], np.r_[bounds, len(df)]):
        k = kinematics(x[s:e], y[s:e], window, poly)
        for c in KIN_COLS:
            out[c][s:e] = k[c]
    return df.with_columns(pl.Series(c, v) for c, v in out.items())
