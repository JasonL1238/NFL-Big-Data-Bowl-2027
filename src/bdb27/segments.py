"""Cut trajectories into analysis units: Combine drill attempts, snap-aligned game windows, and breaks.

Typical use:
  combine: attempts() -> canonicalize() -> kinematics.add_kinematics(df, ["traj"]) -> breaks()
  game:    snap_aligned(features.reg_season_tracking()) -> filter t_rel window -> add_kinematics -> breaks()
"""

import numpy as np
import polars as pl

from bdb27 import io
from bdb27.kinematics import DT, trajectory_ids

ATTEMPT_KEYS = ["event_id", "nfl_id", "drill_name", "attempt"]
PLAY_KEYS = ["game_id", "play_id", "nfl_id"]
SNAP_EVENTS = ["ball_snap", "snap_direct"]
SESSION_GAP_S = 600  # Combine attempts of a drill more than 10 min apart belong to different sessions


# ---------- Combine ----------

def attempts(drill_names: list[str] | None = None, drill_types: list[str] | None = None) -> pl.DataFrame:
    """Player frames of Combine drills with `traj` = one id per attempt (split at time gaps)."""
    lf = io.scan_combine_tracking().filter(pl.col("entity_type") == "PLAYER")
    if drill_names:
        lf = lf.filter(pl.col("drill_name").is_in(drill_names))
    if drill_types:
        lf = lf.filter(pl.col("drill_type").is_in(drill_types))
    return trajectory_ids(lf.collect(), ATTEMPT_KEYS)


def combine_sessions(df: pl.DataFrame, drill: str = "drill_name") -> pl.DataFrame:
    """Add `session` = "<drill>_<draft_year>_<k>" to attempt frames (needs `traj`, `time`, `draft_year`):
    within a drill and draft year, an attempt starting more than SESSION_GAP_S after the previous attempt's
    start opens session k + 1. Sessions share coaches, partners and tempo, so center drill metrics by session.
    Pass `drill` to merge naming variants of one drill. Sessions are inferred from the attempts in df only."""
    starts = (df.group_by("traj", drill, "draft_year").agg(t0=pl.col("time").min()).sort(drill, "draft_year", "t0")
              .with_columns(gap=pl.col("t0").diff().over(drill, "draft_year").dt.total_seconds().fill_null(0))
              .with_columns(k=(pl.col("gap") > SESSION_GAP_S).cum_sum().over(drill, "draft_year"))
              .select("traj", session=pl.concat_str([pl.col(c).cast(pl.Utf8) for c in (drill, "draft_year", "k")],
                                                    separator="_")))
    return df.join(starts, on="traj", how="left", maintain_order="left")


def onset_index(speed: np.ndarray, thresh: float = 1.0, hold: int = 3) -> int:
    """First frame where speed >= thresh for `hold` consecutive frames (0 if never)."""
    above = np.convolve(speed >= thresh, np.ones(hold, dtype=int), mode="valid") == hold
    idx = np.flatnonzero(above)
    return int(idx[0]) if len(idx) else 0


def onset_time(speed: np.ndarray, thresh: float = 1.0, hold: int = 3) -> tuple[float, bool]:
    """Motion onset, in seconds after the first frame: the interpolated time speed crosses `thresh` at the
    start of the first run of `hold` frames >= thresh. Returns (t, observed). observed is False (t NaN) when
    the trace already starts at or above thresh: the onset happened before tracking began. t is NaN with
    observed True when speed never holds above thresh."""
    if not speed[0] < thresh:
        return np.nan, False
    run = np.convolve(speed >= thresh, np.ones(hold, dtype=int), mode="valid") == hold
    idx = np.flatnonzero(run)
    if not len(idx):
        return np.nan, True
    i = int(idx[0])  # i >= 1 and speed[i - 1] < thresh <= speed[i]
    s0, s1 = speed[i - 1], speed[i]
    return float((i - 1 + (thresh - s0) / (s1 - s0)) * DT), True


def canonicalize(df: pl.DataFrame, init_dist: float = 2.0, mirror: bool = True) -> pl.DataFrame:
    """Per `traj`: origin at motion onset, initial travel direction along +x, and (if mirror)
    reflected so most of the path is at y >= 0. Adds `onset` (bool frame flag) and `mirrored`.
    Rewrites x/y; recompute kinematics afterwards (heading/turn signs change, magnitudes do not)."""
    df = df.sort("traj", "time")
    x, y, s, traj = (df[c].to_numpy().astype(float) for c in ("x", "y", "s", "traj"))
    bounds = np.flatnonzero(np.diff(traj)) + 1
    xc, yc = np.empty_like(x), np.empty_like(y)
    onset, mirrored = np.zeros(len(df), bool), np.zeros(len(df), bool)
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(df)]):
        i0 = a + onset_index(s[a:b])
        onset[i0] = True
        dx, dy = x[a:b] - x[i0], y[a:b] - y[i0]
        far = np.flatnonzero(np.hypot(dx[i0 - a:], dy[i0 - a:]) >= init_dist)
        j = i0 - a + (far[0] if len(far) else b - 1 - i0)
        theta = np.arctan2(dy[j], dx[j])
        c, sn = np.cos(-theta), np.sin(-theta)
        xr, yr = c * dx - sn * dy, sn * dx + c * dy
        flip = mirror and np.nanmean(yr[i0 - a:]) < 0
        xc[a:b], yc[a:b] = xr, (-yr if flip else yr)
        mirrored[a:b] = flip
    return df.with_columns(x=pl.Series(xc), y=pl.Series(yc), onset=pl.Series(onset), mirrored=pl.Series(mirrored))


# ---------- Game ----------

def snap_aligned(tracking: pl.LazyFrame) -> pl.LazyFrame:
    """Add `t_rel` (s since the first snap of each player-play); drop player-plays without a snap.
    Direct snaps (wildcat) are labeled 'snap_direct' instead of 'ball_snap'."""
    snaps = (tracking.filter(pl.col("event").is_in(SNAP_EVENTS))
             .group_by(PLAY_KEYS).agg(t_snap=pl.col("time").min()))
    return (tracking.join(snaps, on=PLAY_KEYS, how="inner", maintain_order="left")
            .with_columns(t_rel=(pl.col("time") - pl.col("t_snap")).dt.total_milliseconds() / 1000)
            .drop("t_snap"))


def event_times(aligned: pl.LazyFrame, events: list[str]) -> pl.LazyFrame:
    """One row per player-play with `t_<event>` = first t_rel of each event (null if absent)."""
    return aligned.group_by(PLAY_KEYS).agg(
        [pl.col("t_rel").filter(pl.col("event") == e).min().alias(f"t_{e}") for e in events])


# ---------- Breaks (cuts / route breaks) ----------

BREAK_COLS = ["i_break", "t_break", "turn_deg", "entry_speed", "min_speed", "exit_speed", "retention",
              "peak_decel", "peak_reaccel", "max_turn_rate", "turn_radius", "decel_time"]


def _wrap(deg):
    return (deg + 180) % 360 - 180


def detect_breaks(speed, heading, a_tan, turn_rate, min_turn: float = 45.0, half_window: float = 0.5,
                  context: float = 1.0, min_entry_speed: float = 0.0) -> list[dict]:
    """Breaks in one trajectory: runs of frames whose heading changes >= min_turn across +-half_window.
    The break frame is the speed minimum in the run. Entry/exit stats use +-context seconds.
    turn_deg > 0 means a clockwise (right) turn. turn_radius = path length over the +-half_window
    around the sharpest point / |turn in radians| (mean radius of the turn). Breaks with entry speed < min_entry_speed are dropped
    (use ~3 yd/s for game routes: heading wobbles while a player gets out of the stance at the snap)."""
    n, h, c = len(speed), int(round(half_window / DT)), int(round(context / DT))
    net = np.full(n, np.nan)
    if n > 2 * h:
        # Integrate wrapped frame-to-frame steps (so turns > 180 deg keep their sign); across low-speed
        # gaps where heading is undefined, the step is the jump between the last and next known heading.
        known = np.isfinite(heading)
        h_ff = heading[np.maximum.accumulate(np.where(known, np.arange(n), 0))]
        cum = np.r_[0.0, np.cumsum(np.nan_to_num(_wrap(np.diff(h_ff))))]
        ok = known[2 * h:] & known[:n - 2 * h]
        net[h:n - h] = np.where(ok, cum[2 * h:] - cum[:n - 2 * h], np.nan)
    cand = np.abs(np.nan_to_num(net)) >= min_turn
    out = []
    if not cand.any():
        return out
    edges = np.flatnonzero(np.diff(np.r_[0, cand.astype(int), 0]))
    for r0, r1 in zip(edges[::2], edges[1::2]):
        i = r0 + int(np.argmin(speed[r0:r1]))
        pre, post = slice(max(0, i - c), i + 1), slice(i, min(n, i + c + 1))
        entry = float(np.max(speed[pre]))
        if entry < min_entry_speed:
            continue
        i_entry = max(0, i - c) + int(np.argmax(speed[pre]))
        tr = np.abs(turn_rate[r0:r1])
        j = r0 + int(np.nanargmax(np.abs(net[r0:r1])))
        arc = float(np.sum(speed[max(0, j - h):j + h]) * DT)
        out.append(dict(
            i_break=int(i), t_break=i * DT,
            turn_deg=float(net[j]),
            entry_speed=entry, min_speed=float(speed[i]), exit_speed=float(np.max(speed[post])),
            retention=float(speed[i] / entry) if entry > 0 else np.nan,
            peak_decel=float(-np.min(a_tan[pre])), peak_reaccel=float(np.max(a_tan[post])),
            max_turn_rate=float(np.nanmax(tr)) if np.isfinite(tr).any() else np.nan,
            turn_radius=arc / abs(np.radians(net[j])),
            decel_time=(i - i_entry) * DT,
        ))
    return out


def breaks(df: pl.DataFrame, id_cols: list[str], **kw) -> pl.DataFrame:
    """Run detect_breaks over every `traj` of a frame table that already has kinematics.
    Returns one row per break with `traj`, id_cols, `k` (break order within traj) and BREAK_COLS."""
    df = df.sort("traj", "time")
    traj = df["traj"].to_numpy()
    arrs = {c: df[c].to_numpy() for c in ("speed", "heading", "a_tan", "turn_rate")}
    ids = df.select("traj", *id_cols).unique("traj", keep="first")
    bounds = np.flatnonzero(np.diff(traj)) + 1
    rows = []
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(df)]):
        for k, br in enumerate(detect_breaks(**{c: v[a:b] for c, v in arrs.items()}, **kw)):
            rows.append({"traj": int(traj[a]), "k": k, **br})
    schema = {"traj": pl.Int64, "k": pl.Int64, **{c: pl.Float64 for c in BREAK_COLS}}
    schema["i_break"] = pl.Int64
    return pl.DataFrame(rows, schema=schema).join(ids, on="traj", how="left")
