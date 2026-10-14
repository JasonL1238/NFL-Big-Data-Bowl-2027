"""Probe A (E002): WR/TE route-break mechanics in Combine route drills vs regular-season routes.

Two kinds of break events, measured with the same code on both sides:
  - STOP  (braking into a stop-and-turn): Combine CURL / COMEBACK / HOOK / SLOT_STRIKE, game HITCH.
          First frame after the stem where speed falls below STOP_FRAC x the running peak; the stem
          peak is the max speed before it, the stop is the min speed in the next STOP_SEARCH s.
  - TURN  (cut at speed): Combine SPEED_OUT / DAGGER / SLOT_SAIL / POST_CORNER / SLANT / IN / FLAT / CORNER,
          game OUT / IN / SLANT / CORNER / POST / CROSS.
          Anchor j = sharpest heading change over +-half_window inside the search window (|turn| >= min_turn);
          the break frame is the slowest frame within +-half_window of j.
segments.detect_breaks anchors the break at the speed minimum of the whole candidate run; on rounded,
accelerating breaks (slant) that is the first frame of the run, ~1 s before the turn, so this module
re-anchors at the sharpest point (see shared_module_requests in the E002 result).

Combine search windows are in path distance (yd) from motion onset, tuned per drill on QC plots
(experiments/figures/E002_combine_breaks_*.png). Game windows are in seconds since the snap: the event
must start before the throw (t_end = first of pass_forward / pass_shovel / sack / scramble, capped at 4 s).
"""

import numpy as np
import polars as pl

from bdb27 import features, io, kinematics, segments

DT = kinematics.DT
STOP_FRAC = 0.5      # a stop = speed below half of the running peak
STOP_SEARCH = 1.2    # s after the drop to look for the speed minimum
STOP_TOL = 0.2       # yd/s: the stop frame is the first frame this close to that minimum
MIN_ENTRY = 3.0      # yd/s, drops stance wobble at the snap / drill start
CONTEXT = 1.0        # s before/after the break for entry/exit stats

# Combine drills: family, path-distance window (yd from onset) for the break, min_turn, half_window.
# Excluded (no identifiable break: smooth arcs with peak turn rate ~25-40 deg/s spread over 10+ yd, see
# E002_combine_breaks_qc.png): SLANT_ROUTE_LEFT, CORNER_ROUTE, POST_CORNER_ROUTE_RIGHT, WHEEL_ROUTE_LEFT, GO_ROUTE_RIGHT.
DRILLS = {
    "CURL_ROUTE_RIGHT":        ("stop", 7.0, 19.0, None, None),
    "COMEBACK_ROUTE_RIGHT":    ("stop", 11.0, 23.0, None, None),
    "HOOK_ROUTE_RIGHT":        ("stop", 7.0, 20.0, None, None),
    "SLOT_STRIKE_ROUTE_LEFT":  ("stop", 3.0, 11.0, None, None),
    "SPEED_OUT_ROUTE_LEFT":    ("turn", 8.0, 17.0, 45.0, 0.5),
    "SPEED_OUT_LEFT":          ("turn", 8.0, 17.0, 45.0, 0.5),
    "DAGGER_ROUTE_LEFT":       ("turn", 13.0, 25.0, 40.0, 0.5),
    "SLOT_SAIL_ROUTE_RIGHT":   ("turn", 9.0, 18.0, 40.0, 0.5),
    "POST_CORNER_ROUTE_LEFT":  ("turn", 13.0, 25.0, 40.0, 0.5),
    "IN_ROUTE_LEFT":           ("turn", 9.0, 17.0, 45.0, 0.5),
    "FLAT_ROUTE_LEFT":         ("turn", 10.0, 22.0, 40.0, 0.8),
}
QC_EXCLUDED_DRILLS = ["SLANT_ROUTE_LEFT", "CORNER_ROUTE", "POST_CORNER_ROUTE_RIGHT"]
# Naming variants of the same drill are pooled for z-scoring.
DRILL_GROUP = {"SPEED_OUT_LEFT": "SPEED_OUT_ROUTE_LEFT"}

# Game routes: family, min_turn, half_window (turn routes only).
GAME_ROUTES = {
    "HITCH": ("stop", None, None),
    "OUT": ("turn", 45.0, 0.5), "IN": ("turn", 45.0, 0.5), "SLANT": ("turn", 40.0, 0.8),
    "CORNER": ("turn", 30.0, 0.5), "POST": ("turn", 30.0, 0.5), "CROSS": ("turn", 40.0, 0.8),
}
# Routes whose events build the game traits. SLANT/CORNER/POST/CROSS are mostly gradual bends with no
# identifiable break (detection 40-60%, see E002_game_breaks_qc.png): computed, used only as exploratory.
TRAIT_ROUTES = {"stop": ["HITCH"], "turn": ["OUT", "IN"]}
GAME_T_START = 0.5   # s after snap
GAME_T_CAP = 4.0
END_EVENTS = ["pass_forward", "pass_shovel", "qb_sack", "qb_strip_sack", "run"]

EVENT_COLS = ["i_break", "entry_speed", "min_speed", "exit_speed", "retention", "speed_drop", "peak_decel",
              "mean_decel", "decel_time", "brake_dist", "peak_reaccel", "turn_deg", "turn_radius", "peak_alat"]


# ---------- event detection on one trajectory ----------

def _wrap(deg):
    return (deg + 180) % 360 - 180


def net_turn(heading: np.ndarray, h: int) -> np.ndarray:
    """Heading change (deg, + clockwise) over +-h frames, integrating wrapped steps; NaN where undefined.
    Same construction as segments.detect_breaks."""
    n = len(heading)
    net = np.full(n, np.nan)
    if n <= 2 * h:
        return net
    known = np.isfinite(heading)
    h_ff = heading[np.maximum.accumulate(np.where(known, np.arange(n), 0))]
    cum = np.r_[0.0, np.cumsum(np.nan_to_num(_wrap(np.diff(h_ff))))]
    ok = known[2 * h:] & known[:n - 2 * h]
    net[h:n - h] = np.where(ok, cum[2 * h:] - cum[:n - 2 * h], np.nan)
    return net


def _stats(i: int, speed, a_tan, a_lat, net, h: int, i_entry: int | None = None) -> dict:
    """Entry/exit/braking stats around break frame i (the slowest frame of the break). The entry frame is
    `i_entry` if given (stop events: the stem peak), else the fastest frame in the CONTEXT s before i."""
    n, c = len(speed), int(round(CONTEXT / DT))
    if i_entry is None:
        i_entry = max(0, i - c) + int(np.argmax(speed[max(0, i - c):i + 1]))
    entry = float(speed[i_entry])
    brake = a_tan[i_entry:i + 1]
    post = slice(i, min(n, i + c + 1))
    lo, hi = max(0, i - h), min(n, i + h + 1)
    j = lo + int(np.nanargmax(np.abs(np.nan_to_num(net[lo:hi])))) if np.isfinite(net[lo:hi]).any() else i
    turn = float(net[j]) if np.isfinite(net[j]) else np.nan
    arc = float(np.sum(speed[max(0, j - h):j + h]) * DT)
    alat = np.abs(a_lat[lo:hi])
    return dict(
        i_break=int(i), entry_speed=entry, min_speed=float(speed[i]), exit_speed=float(np.max(speed[post])),
        retention=float(speed[i] / entry) if entry > 0 else np.nan, speed_drop=entry - float(speed[i]),
        peak_decel=float(max(0.0, -np.min(brake))) if len(brake) else 0.0,
        mean_decel=(entry - float(speed[i])) / ((i - i_entry) * DT) if i > i_entry else 0.0,
        decel_time=(i - i_entry) * DT, brake_dist=float(np.sum(speed[i_entry:i]) * DT),
        peak_reaccel=float(np.max(a_tan[post])),
        turn_deg=turn, turn_radius=arc / abs(np.radians(turn)) if np.isfinite(turn) and turn != 0 else np.nan,
        peak_alat=float(np.nanmax(alat)) if np.isfinite(alat).any() else np.nan,
    )


def stop_event(speed, a_tan, a_lat, heading, start: int, last_start: int) -> dict | None:
    """First stop after frame `start`: speed < STOP_FRAC x running peak (peak >= MIN_ENTRY). The stem
    peak must be at or before `last_start` (e.g. the throw). Break frame = first frame within STOP_TOL of
    the min speed in the STOP_SEARCH s after the drop. Entry = the stem peak."""
    n = len(speed)
    if start >= n - 2:
        return None
    s = speed[start:]
    run_max = np.maximum.accumulate(s)
    drop = np.flatnonzero((s < STOP_FRAC * run_max) & (run_max >= MIN_ENTRY))
    if not len(drop):
        return None
    k = start + int(drop[0])
    i_peak = start + int(np.argmax(speed[start:k + 1]))
    if i_peak > last_start:
        return None
    m = int(round(STOP_SEARCH / DT))
    seg = speed[k:min(n, k + m + 1)]
    i = k + int(np.argmax(seg <= seg.min() + STOP_TOL))  # first frame at (near) the bottom, not later drift
    h = int(round(0.5 / DT))
    return _stats(i, speed, a_tan, a_lat, net_turn(heading, h), h, i_entry=i_peak) | {"i_peak": i_peak}


def turn_event(speed, a_tan, a_lat, heading, window: np.ndarray, min_turn: float, half_window: float) -> dict | None:
    """Sharpest heading change (|net over +-half_window| >= min_turn) among frames where `window` is True;
    break frame = slowest frame within +-half_window of it; entry speed >= MIN_ENTRY."""
    h = int(round(half_window / DT))
    net = net_turn(heading, h)
    cand = np.where(window, np.abs(np.nan_to_num(net)), 0.0)
    if cand.max(initial=0.0) < min_turn:
        return None
    j = int(np.argmax(cand))
    lo, hi = max(0, j - h), min(len(speed), j + h + 1)
    i = lo + int(np.argmin(speed[lo:hi]))
    out = _stats(i, speed, a_tan, a_lat, net, h)
    out["turn_deg"] = float(net[j])
    arc = float(np.sum(speed[max(0, j - h):j + h]) * DT)
    out["turn_radius"] = arc / abs(np.radians(net[j]))
    out["i_sharp"] = j
    if out["entry_speed"] < MIN_ENTRY:
        return None
    return out


# ---------- Combine ----------

def combine_frames(drills: list[str] | None = None) -> pl.DataFrame:
    """Route-drill frames: canonicalized, kinematics, `tt` (s since onset) and `dist` (yd since onset)."""
    df = segments.attempts(drill_names=drills or list(DRILLS), drill_types=["SKILL_DRILLS_WR", "SKILL_DRILLS_TE"])
    df = segments.canonicalize(df).drop("traj")
    df = kinematics.add_kinematics(df, segments.ATTEMPT_KEYS)
    on = df.filter(pl.col("onset")).group_by("traj").agg(t_on=pl.col("time").min())
    df = df.join(on, on="traj", how="left", maintain_order="left").with_columns(
        tt=(pl.col("time") - pl.col("t_on")).dt.total_milliseconds() / 1000)
    return df.with_columns(dist=(pl.col("speed") * DT * (pl.col("tt") >= 0)).cum_sum().over("traj")).drop("t_on")


def combine_events(frames: pl.DataFrame) -> pl.DataFrame:
    """One row per attempt (traj) with the drill's break event (null stats if none found)."""
    frames = frames.sort("traj", "time")
    ids = frames.group_by("traj", maintain_order=True).agg(
        pl.first("event_id", "nfl_id", "drill_name", "attempt", "draft_year"))
    traj = frames["traj"].to_numpy()
    arr = {c: frames[c].to_numpy() for c in ("speed", "a_tan", "a_lat", "heading", "dist", "tt")}
    drills = frames["drill_name"].to_numpy()
    bounds = np.flatnonzero(np.diff(traj)) + 1
    rows = []
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(frames)]):
        if drills[a] not in DRILLS:
            continue
        fam, lo, hi, mt, hw = DRILLS[drills[a]]
        v = {c: x[a:b] for c, x in arr.items()}
        start = int(np.argmax(v["tt"] >= 0))
        if fam == "stop":
            ev = stop_event(v["speed"], v["a_tan"], v["a_lat"], v["heading"], start, len(v["speed"]) - 1)
            if ev is not None and not (lo <= v["dist"][ev["i_break"]] <= hi):
                ev = None
        else:
            win = (v["dist"] >= lo) & (v["dist"] <= hi) & (v["tt"] >= 0)
            ev = turn_event(v["speed"], v["a_tan"], v["a_lat"], v["heading"], win, mt, hw)
        row = {"traj": int(traj[a]), "family": fam, "found": ev is not None}
        if ev is not None:
            row |= {k: ev[k] for k in EVENT_COLS} | {"dist_break": float(v["dist"][ev["i_break"]]),
                                                     "t_break": float(v["tt"][ev["i_break"]])}
        rows.append(row)
    schema = {"traj": pl.Int64, "family": pl.String, "found": pl.Boolean,
              **{c: pl.Float64 for c in EVENT_COLS}, "dist_break": pl.Float64, "t_break": pl.Float64}
    schema["i_break"] = pl.Int64
    ev = pl.DataFrame(rows, schema=schema)
    return ids.join(ev, on="traj").with_columns(
        drill=pl.col("drill_name").replace(DRILL_GROUP))


def zscore_within(df: pl.DataFrame, value: str, by: str) -> pl.Expr:
    return ((pl.col(value) - pl.col(value).mean().over(by)) / pl.col(value).std().over(by))


def drill_values(events: pl.DataFrame, value: str, family: str | None = None) -> pl.DataFrame:
    """Per player x drill: attempt-mean of `value`, z-scored within drill (column `z`)."""
    e = events.filter(pl.col("found") & pl.col(value).is_finite())
    if family:
        e = e.filter(pl.col("family") == family)
    pd_ = e.group_by("nfl_id", "drill").agg(v=pl.col(value).mean(), n_att=pl.len())
    return pd_.with_columns(z=zscore_within(pd_, "v", "drill"))


def composite(events: pl.DataFrame, value: str, family: str | None, sign: float = 1.0) -> pl.DataFrame:
    """Player composite = mean over drills of the within-drill z-score (x sign), with n_drills."""
    dv = drill_values(events, value, family)
    return dv.group_by("nfl_id").agg(comp=(pl.col("z") * sign).mean(), n_drills=pl.len())


# ---------- Game ----------

def game_route_plays(positions=("WR", "TE")) -> pl.DataFrame:
    """REG-season route plays of study WR/TE (one row per player-play) with player_play context."""
    pos = io.load("combine_results").select("nfl_id", position="combine_position").filter(
        pl.col("position").is_in(list(positions)))
    reg = io.load("games").filter(pl.col("season_type") == "REG").select("game_id", "season", "week")
    return (io.load("player_play").join(reg, on="game_id").join(pos, on="nfl_id")
            .filter(pl.col("route_ran").is_not_null() & (pl.col("play_nullified_by_penalty") != "Y")))


def game_frames(plays: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Snap-aligned frames (t_rel in [-0.5, 6]) with kinematics, and per-play event times incl. t_end."""
    keys = plays.select(segments.PLAY_KEYS)
    lf = features.reg_season_tracking().join(keys.lazy(), on=segments.PLAY_KEYS, how="semi")
    al = segments.snap_aligned(lf)
    ev = segments.event_times(al, END_EVENTS).collect()
    ev = ev.with_columns(t_end=pl.min_horizontal(
        pl.min_horizontal([pl.col(f"t_{e}") for e in END_EVENTS]).fill_null(GAME_T_CAP), pl.lit(GAME_T_CAP)))
    fr = (al.filter(pl.col("t_rel").is_between(-0.5, 6.0))
          .select(*segments.PLAY_KEYS, "time", "t_rel", "x", "y", "s").collect())
    return kinematics.add_kinematics(fr, segments.PLAY_KEYS), ev


def game_events(frames: pl.DataFrame, plays: pl.DataFrame, ev: pl.DataFrame) -> pl.DataFrame:
    """One row per route play of a GAME_ROUTES type, with the route's break event (null if none)."""
    info = plays.select(*segments.PLAY_KEYS, "route_ran", "position").filter(pl.col("route_ran").is_in(list(GAME_ROUTES)))
    f = (frames.join(info, on=segments.PLAY_KEYS, how="inner", maintain_order="left")
         .join(ev.select(*segments.PLAY_KEYS, "t_end", "t_pass_forward"), on=segments.PLAY_KEYS, how="left",
               maintain_order="left").sort("traj", "time"))
    ids = f.group_by("traj", maintain_order=True).agg(pl.first(*segments.PLAY_KEYS, "route_ran", "position",
                                                                "t_end", "t_pass_forward"))
    traj = f["traj"].to_numpy()
    arr = {c: f[c].to_numpy() for c in ("speed", "a_tan", "a_lat", "heading", "t_rel", "t_end")}
    routes = f["route_ran"].to_numpy()
    bounds = np.flatnonzero(np.diff(traj)) + 1
    rows = []
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(f)]):
        fam, mt, hw = GAME_ROUTES[routes[a]]
        v = {c: x[a:b] for c, x in arr.items()}
        t, t_end = v["t_rel"], v["t_end"][0]
        if fam == "stop":
            start = int(np.argmax(t >= GAME_T_START)) if (t >= GAME_T_START).any() else len(t)
            last = int(np.flatnonzero(t <= t_end)[-1]) if (t <= t_end).any() else -1
            e = stop_event(v["speed"], v["a_tan"], v["a_lat"], v["heading"], start, last)
        else:
            win = (t >= GAME_T_START) & (t <= t_end + 0.5)
            e = turn_event(v["speed"], v["a_tan"], v["a_lat"], v["heading"], win, mt, hw)
        row = {"traj": int(traj[a]), "family": fam, "found": e is not None}
        if e is not None:
            row |= {k: e[k] for k in EVENT_COLS} | {"t_break": float(t[e["i_break"]])}
        rows.append(row)
    schema = {"traj": pl.Int64, "family": pl.String, "found": pl.Boolean,
              **{c: pl.Float64 for c in EVENT_COLS}, "t_break": pl.Float64}
    schema["i_break"] = pl.Int64
    return ids.join(pl.DataFrame(rows, schema=schema), on="traj")


# ---------- player-level aggregation ----------

def eb_shrink(df: pl.DataFrame, value: str, group: str = "nfl_id", by: str | None = "position",
              min_n: int = 1) -> pl.DataFrame:
    """Empirical-Bayes shrinkage of per-player means toward the `by`-group mean (normal-normal, method of
    moments): shrunk = mu + (xbar - mu) * tau2 / (tau2 + sigma2 / n), sigma2 = pooled within-player
    variance, tau2 = Var(xbar) - mean(sigma2 / n) (floored at 1% of Var(xbar)). Returns n, raw mean, shrunk.
    Only players with n >= min_n are kept AND used to estimate sigma2, tau2 and mu: with very unbalanced n,
    including 1-3-event players made the unweighted moment estimator collapse (HITCH peak decel: tau2 0.008
    with all players vs 0.14 among players with >= 10 events; E002 run 1 bug)."""
    d = df.filter(pl.col(value).is_finite())
    keys = [group] + ([by] if by else [])
    m = (d.group_by(keys).agg(n=pl.len(), raw=pl.col(value).mean(), ss=((pl.col(value) - pl.col(value).mean()) ** 2).sum())
         .filter(pl.col("n") >= min_n))
    out = []
    for _, g in (m.group_by(by) if by else [(None, m)]):
        sigma2 = float(g["ss"].sum() / max(1, (g["n"] - 1).sum()))
        xbar, n = g["raw"].to_numpy(), g["n"].to_numpy()
        tau2 = max(float(np.var(xbar, ddof=1) - np.mean(sigma2 / n)), 0.01 * float(np.var(xbar, ddof=1)))
        mu = float(np.average(xbar, weights=n / (sigma2 + tau2 * n)))
        w = tau2 / (tau2 + sigma2 / n)
        out.append(g.with_columns(shrunk=pl.Series(mu + (xbar - mu) * w), weight=pl.Series(w),
                                  tau2=pl.lit(tau2), sigma2=pl.lit(sigma2)))
    return pl.concat(out).drop("ss")


def separation_over_expected(plays: pl.DataFrame, n_splits: int = 5, seed: int = 0) -> pl.DataFrame:
    """Play-level residual of separation_at_pass_forward from a context model, cross-fitted by player
    (GroupKFold): a player's own plays never train the model that sets his expectation.
    Context: route, coverage (man/zone, type), alignment, lined-up position, formation, motion, dropback
    duration, down, distance, yards to goal, quarter. Excludes SCREEN, cushion (defenders' cushion reacts to
    the receiver's speed) and pass_length (a property of the target only)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    cat = ["route_ran", "team_coverage_man_zone", "team_coverage_type", "receiver_alignment",
           "lined_up_position", "offense_formation", "in_motion_at_ball_snap"]
    num = ["dropback_duration", "down", "yards_to_go", "yards_to_goal", "quarter"]
    d = (plays.filter(pl.col("separation_at_pass_forward").is_not_null() & (pl.col("route_ran") != "SCREEN"))
         .with_columns(yards_to_goal=pl.when(pl.col("yardline_side") == pl.col("possession_team"))
                       .then(100 - pl.col("yardline_number")).otherwise(pl.col("yardline_number")).cast(pl.Float64)))
    X = np.column_stack([d.select(pl.col(c).cast(pl.String).fill_null("NA")).to_series().cast(pl.Categorical)
                         .to_physical().to_numpy() for c in cat] +
                        [d[c].cast(pl.Float64).to_numpy() for c in num])
    y = d["separation_at_pass_forward"].to_numpy().astype(float)
    groups = d["nfl_id"].to_numpy()
    pred = np.empty_like(y)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = HistGradientBoostingRegressor(categorical_features=list(range(len(cat))), max_iter=300,
                                          learning_rate=0.05, min_samples_leaf=50, random_state=seed)
        pred[te] = m.fit(X[tr], y[tr]).predict(X[te])
    return d.with_columns(sep_expected=pl.Series(pred), soe=pl.Series(y - pred))


# ---------- player-level signatures / traits (E002 definitions, see experiments/results/E002_prereg.json) ----------

MIN_DRILLS = 2       # Combine composite needs >= 2 drills of the family
MIN_EVENTS = 10      # game break trait needs >= 10 detected events
MIN_SOE_PLAYS = 20   # separation over expected needs >= 20 plays


def combine_signatures(events: pl.DataFrame, feats: list[str]) -> pl.DataFrame:
    """Wide per-player Combine composites c_<family>_<feat> (drill-z averaged; null if < MIN_DRILLS drills)
    plus n_<family>_drills. Drills are position-specific, so WRs get WR-drill and TEs TE-drill composites."""
    out = None
    for fam in ("stop", "turn"):
        for f in feats:
            c = (composite(events, f, fam).filter(pl.col("n_drills") >= MIN_DRILLS)
                 .rename({"comp": f"c_{fam}_{f}", "n_drills": f"n_{fam}_drills"}))
            if out is None:
                out = c
            else:
                c = c.drop(f"n_{fam}_drills") if f"n_{fam}_drills" in out.columns else c
                out = out.join(c, on="nfl_id", how="full", coalesce=True)
    return out


def game_trait(events: pl.DataFrame, value: str, routes: list[str], name: str) -> pl.DataFrame:
    """Per-player trait: per-event `value` minus its route-type mean (all players' events), player mean
    EB-shrunk toward the position mean; players with >= MIN_EVENTS events. Columns: nfl_id, <name>,
    <name>_raw, <name>_n."""
    e = (events.filter(pl.col("route_ran").is_in(routes) & pl.col("found") & pl.col(value).is_finite())
         .with_columns(v=pl.col(value) - pl.col(value).mean().over("route_ran")))
    s = eb_shrink(e, "v", min_n=MIN_EVENTS)
    return s.select("nfl_id", pl.col("shrunk").alias(name), pl.col("raw").alias(f"{name}_raw"), pl.col("n").alias(f"{name}_n"))


def soe_trait(soe: pl.DataFrame, routes: list[str] | None, name: str) -> pl.DataFrame:
    """Per-player separation over expected (mean play residual), EB-shrunk; players with >= MIN_SOE_PLAYS."""
    d = soe if routes is None else soe.filter(pl.col("route_ran").is_in(routes))
    s = eb_shrink(d, "soe", min_n=MIN_SOE_PLAYS)
    return s.select("nfl_id", pl.col("shrunk").alias(name), pl.col("raw").alias(f"{name}_raw"), pl.col("n").alias(f"{name}_n"))
