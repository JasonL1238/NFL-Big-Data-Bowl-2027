"""Probe G (E011): catch-point mechanics from Combine ball tracking (2025 class only; descriptive feasibility).

Data (verified 2026-10-14/15):
  - BALL rows exist only for the 2025 class: 551 attempts by 54 players (37 WR, 17 TE) in 18 drill names
    (OVER_SHOULDER_ADJUST is run by both groups). A BALL row carries the receiver's nfl_id, event_id, drill_name and
    attempt, and shares the receiver's 10 Hz timestamps (every BALL row matches a receiver frame BY TIMESTAMP; in 4
    attempts the ball never comes within 5 yd of the receiver, so those rows belong to some other throw). x/y only:
    there is no ball height, so a high catch and a low catch at the same ground point look identical.
  - The ball's device speed `s` is smooth in flight (13-26 yd/s, slowing < 3% per frame from drag) and drops in one
    frame at contact; it ignores the one-frame position glitches seen while the QB holds the ball. Before release the
    ball never reaches V_FLIGHT (max 11.96 yd/s), so flight needs a speed no handling reaches. After contact the ball
    can exceed 10 yd/s again (deflections, balls past him); 3 attempts have a second >= V_FLIGHT run, and only the
    first is used.
  - The carried ball sits a median 0.57 yd from the receiver's tracked (shoulder-pad) point. A ball-vs-player clock
    offset is not identified more precisely than about +-0.15 s: offset-free velocity matching on the carried ball
    gives +0.04 to +0.06 s or -0.10 to -0.13 s depending on the window, because ball handling after the catch moves
    the ball relative to the body (E011.json clock_lag_check; attempt_table(clock_shift=...) for sensitivity).
  - Sessions (segments.combine_sessions): TE drills on 2025-02-28, WR drills in two groups on 2025-03-01.

Throw and contact (detect_throw), per attempt:
  - release: first frame of the first run of >= MIN_FLIGHT_FRAMES frames with ball s >= V_FLIGHT.
  - contact: first frame after release whose ball s < DROP_RATIO x the previous frame's, i.e. the ball's FIRST impact
    with anything (receiver, ground). The impact happened between it and the last in-flight frame, so t_contact = its
    time - DT / 2 (+-0.05 s). None when tracking ends in flight (mostly SLANT / WHEEL, whose tracking stops 0.1-0.3 s
    after the catch). Checked against an unanchored geometric closest approach (flight_approach).
Outcome (outcome; preregistered; INFERRED: the data have no catch labels; validated only on plots):
  - reached: the ball's first impact is within REACH_DIST of the receiver's tracked point (min frame distance over the
    last in-flight frame and the 3 after it; the ball moves ~2 yd per frame, so single-frame distances miss contact).
  - secured: 0.2-0.7 s after the first post-contact frame the ball stays with him: median distance <= SECURE_DIST and
    growth over the window <= SECURE_GROWTH. separated: reached, not secured (dropped or deflected; a ball that
    passes him with no impact near him is not_reached, see outcome_posthoc).
    reached_unobserved: reached, but tracking ends < 0.7 s after contact. Other classes: no_flight, no_contact
    (tracking ends in flight), not_reached. Post-contact distance is a continuum, so secured vs separated is
    threshold-dependent.
  - outcome_posthoc (POST HOC, added after review on 2026-10-15; the preregistered `outcome` is unchanged): a
    not_reached attempt is `ball_unrelated` when the ball never comes within UNRELATED_DIST of the receiver, and
    `passed_him` when the ball's in-flight path passed within PASS_DIST of his tracked point at least PASS_MIN_T
    before its first impact (it went past or over him and landed beyond: an incompletion). Both classes are
    not_reached, so no mechanic changes.
  - Mechanics other than retention are pre-contact quantities and use every reached attempt.
Mechanics (attempt_table), from receiver SG kinematics (bdb27.kinematics) interpolated at t_contact:
  - retention = speed 0.5 s after contact / 0.5 s before (secured, speed before >= MIN_SPEED_RETENTION, tracked
    through +0.5 s).
  - decel_in = (speed 0.5 s before - speed at contact) / 0.5 s (yd/s^2; > 0 = slowing into the catch).
  - catch_offset = perpendicular distance (yd) of his tracked point at contact from the ball's 2D flight line
    (through the last 3 in-flight ball positions): how far from his body line the ball arrived.
  - adjust_dist (ADJUST_DRILLS) = distance (yd) between his position at contact and his constant-velocity
    extrapolation from the release frame. Despite the name it mostly measures change in run speed during the
    flight in the two fades (the component along his release heading) and route curvature in CIRCUS (across it);
    OVER_SHOULDER_ADJUST is mixed. adjust_along / adjust_across hold the two components (E011.json
    mechanics.adjust_dist.decomposition).
  - adjust_time (ADJUST_DRILLS) = s from release until the angle between his velocity and the bearing to the
    eventual catch point is <= LOCK_DEG and stays so (frames > LOCK_MIN_DIST from the catch point, up to
    LOCK_END s before contact); NaN if he never locks on.
Player values (player_values): per mechanic, player x drill means z-scored within drill x session, averaged over
drills; reliability (mechanic_reliability) is the cross-drill ICC of the same player x drill means.
"""

import numpy as np
import polars as pl

from bdb27 import io, kinematics, reliability, segments, shrinkage, targets_wr
from bdb27.kinematics import DT

BALL_DRILL_TYPES = ["SKILL_DRILLS_WR", "SKILL_DRILLS_TE"]
ADJUST_DRILLS = ["OVER_SHOULDER_ADJUST", "CIRCUS_ROUTE_LEFT", "RED_ZONE_FADE_RIGHT", "END_ZONE_FADE_RIGHT"]
V_FLIGHT = 12.0          # yd/s; faster than any player, so the ball is in the air
MIN_FLIGHT_FRAMES = 3
DROP_RATIO = 0.75        # contact: ball speed falls below this fraction of the previous frame's
REACH_DIST = 2.0         # yd
SECURE_DIST = 2.0        # yd, median ball-receiver distance 0.2-0.7 s after contact
SECURE_GROWTH = 1.5      # yd, growth of that distance over the window
POST = (2, 7)            # frames after the first post-contact frame (0.2-0.7 s)
WIN = 0.5                # s before / after contact for retention and decel_in
MIN_SPEED_RETENTION = 3.0  # yd/s before contact for retention
LOCK_DEG, LOCK_MIN_DIST, LOCK_END = 20.0, 2.0, 0.3
# post hoc (after review, 2026-10-15): reclassification of not_reached attempts (outcome_posthoc)
PASS_DIST, PASS_MIN_T = 1.5, 0.15  # yd, s: in-flight path within PASS_DIST, >= PASS_MIN_T before the first impact
UNRELATED_DIST = 5.0     # yd: ball never this close to the receiver -> the rows belong to another throw
MECHANICS = ["retention", "decel_in", "catch_offset", "adjust_dist", "adjust_time"]


# ---------------------------------------------------------------- frames and events

def ball_frames() -> pl.DataFrame:
    """Receiver frames of every Combine attempt that has BALL rows, with the ball joined on time (bx, by, bs),
    receiver SG kinematics (bdb27.kinematics), `t` (s since the attempt's first frame), `session`
    (segments.combine_sessions) and `position` (Combine position)."""
    keys = segments.ATTEMPT_KEYS
    ball = (io.scan_combine_tracking().filter(pl.col("entity_type") == "BALL")
            .select(*keys, "time", bx="x", by="y", bs="s").collect())
    att = ball.select(keys).unique()
    rec = segments.attempts(drill_types=BALL_DRILL_TYPES).join(att, on=keys, how="semi")
    n_traj = rec["traj"].n_unique()
    if n_traj != att.height:
        raise ValueError(f"{n_traj} receiver trajectories for {att.height} ball attempts (time gaps split an attempt)")
    d = rec.join(ball, on=[*keys, "time"], how="left", maintain_order="left")
    d = kinematics.add_kinematics(d, keys)
    d = segments.combine_sessions(d)
    pos = io.load("combine_results").select("nfl_id", position="combine_position")
    d = d.join(pos, on="nfl_id", how="left", maintain_order="left").sort("traj", "time")
    return d.with_columns(t=(pl.col("time") - pl.col("time").min().over("traj")).dt.total_milliseconds() / 1000,
                          dist=((pl.col("bx") - pl.col("x")) ** 2 + (pl.col("by") - pl.col("y")) ** 2).sqrt())


def detect_throw(bs: np.ndarray) -> tuple[int | None, int | None]:
    """(release, contact) frame indices from the ball's device speed (NaN = no ball row). contact is None when the
    tracking ends in flight; both are None when no flight is found."""
    fl = np.where(np.isfinite(bs), bs, 0.0) >= V_FLIGHT
    run = np.flatnonzero(np.convolve(fl, np.ones(MIN_FLIGHT_FRAMES, dtype=int), mode="valid") == MIN_FLIGHT_FRAMES)
    if not len(run):
        return None, None
    i_r = int(run[0])
    for i in range(i_r + 1, len(bs)):
        if bs[i] < DROP_RATIO * bs[i - 1]:
            return i_r, i
    return i_r, None


def outcome(d: np.ndarray, k: int) -> tuple[str, float, float, float]:
    """(class, reach, post_median, post_growth) from ball-receiver distances d and the contact frame k. Classes:
    not_reached, secured, separated, or reached_unobserved (tracking ends < 0.7 s after contact)."""
    reach = float(np.nanmin(d[k - 1:k + 3]))
    a, b = k + POST[0], k + POST[1]
    if b >= len(d):
        return ("not_reached" if reach > REACH_DIST else "reached_unobserved"), reach, np.nan, np.nan
    med, grow = float(np.median(d[a:b + 1])), float(d[b] - d[a])
    if reach > REACH_DIST:
        return "not_reached", reach, med, grow
    return ("secured" if med <= SECURE_DIST and grow <= SECURE_GROWTH else "separated"), reach, med, grow


def flight_approach(t, x, y, bx, by, i_r: int, k: int, ahead: float = 0.5, step: float = 0.005) -> dict:
    """Ball-receiver closest approach on a fine time grid (post hoc, after review). The ball follows its tracked
    positions from release to the last in-flight frame k - 1 (linear interpolation), then is extrapolated at the
    velocity of its last in-flight step; the receiver follows his tracked positions. Searched from release to
    t_(k-1) + `ahead` (or the end of tracking), so the result is not bounded around the detected contact frame (only
    the switch to extrapolation and the end of the search depend on it).
    Returns t_ca / d_ca (overall closest approach: the unanchored contact-time check) and t_flight / d_flight (closest
    approach while the ball is on its tracked in-flight positions, release .. k - 1). NaN when k - 1 <= release."""
    nan = dict(t_ca=np.nan, d_ca=np.nan, t_flight=np.nan, d_flight=np.nan)
    if k - 1 <= i_r:
        return nan
    tt = np.arange(t[i_r], min(t[k - 1] + ahead, t[-1]) + 1e-9, step)
    v = np.array([bx[k - 1] - bx[k - 2], by[k - 1] - by[k - 2]]) / (t[k - 1] - t[k - 2])
    tf, xf, yf = t[i_r:k], bx[i_r:k], by[i_r:k]
    inf = tt <= t[k - 1]
    pbx = np.where(inf, np.interp(tt, tf, xf), bx[k - 1] + v[0] * (tt - t[k - 1]))
    pby = np.where(inf, np.interp(tt, tf, yf), by[k - 1] + v[1] * (tt - t[k - 1]))
    dd = np.hypot(pbx - np.interp(tt, t, x), pby - np.interp(tt, t, y))
    if not np.isfinite(dd).any():
        return nan
    j = int(np.nanargmin(dd))
    jf = int(np.nanargmin(np.where(inf, dd, np.inf)))
    return dict(t_ca=float(tt[j]), d_ca=float(dd[j]), t_flight=float(tt[jf]), d_flight=float(dd[jf]))


def _lock_time(t, x, y, vx, vy, i_r: int, k: int, t_c: float, pc: np.ndarray) -> float:
    """adjust_time: s from release until the velocity stays within LOCK_DEG of the bearing to the catch point."""
    idx = [i for i in range(i_r, k) if t[i] <= t_c - LOCK_END and np.hypot(pc[0] - x[i], pc[1] - y[i]) > LOCK_MIN_DIST]
    if not idx:
        return np.nan
    idx = np.array(idx)
    wx, wy = pc[0] - x[idx], pc[1] - y[idx]
    cosang = (vx[idx] * wx + vy[idx] * wy) / (np.hypot(vx[idx], vy[idx]) * np.hypot(wx, wy))
    ok = np.degrees(np.arccos(np.clip(cosang, -1, 1))) <= LOCK_DEG
    if not ok[-1]:
        return np.nan
    bad = np.flatnonzero(~ok)
    first = idx[bad[-1] + 1] if len(bad) else idx[0]
    return float(t[first] - t[i_r])


def attempt_table(frames: pl.DataFrame, win: float = WIN, clock_shift: float = 0.0) -> pl.DataFrame:
    """One row per ball attempt: ids, release / contact timing, outcome class (preregistered `outcome` and post hoc
    `outcome_posthoc`), detection diagnostics and the mechanics (NaN where an attempt does not qualify; see the
    module docstring). `win` (s) is the before/after window of retention and decel_in, and `clock_shift` (s) moves the
    time at which the receiver's position and speed are read around contact (pc, v_pre, v_contact, v_post: a
    ball-vs-player clock offset); both are for sensitivity checks only."""
    rows = []
    ids = ["traj", "nfl_id", "position", "draft_year", "event_id", "drill_name", "attempt", "session"]
    for g in frames.partition_by("traj", maintain_order=True):
        r = {c: g[c][0] for c in ids}
        t, bs, d = g["t"].to_numpy(), g["bs"].to_numpy(), g["dist"].to_numpy()
        x, y, vx, vy, sp = (g[c].to_numpy() for c in ("x", "y", "vx", "vy", "speed"))
        bx, by = g["bx"].to_numpy(), g["by"].to_numpy()
        i_r, k = detect_throw(bs)
        fl = np.where(np.isfinite(bs), bs, 0.0) >= V_FLIGHT
        runs = np.flatnonzero(np.convolve(fl, np.ones(MIN_FLIGHT_FRAMES, dtype=int), mode="valid") == MIN_FLIGHT_FRAMES)
        r.update(n_frames=g.height, i_release=i_r, i_contact=k, outcome="no_flight" if i_r is None else "no_contact",
                 n_flight_runs=int(len(runs) and 1 + np.sum(np.diff(runs) > 1)),
                 d_min_all=float(np.nanmin(d)) if np.isfinite(d).any() else np.nan,
                 bs_pre_max=float(np.nanmax(bs[:i_r])) if i_r and np.isfinite(bs[:i_r]).any() else np.nan)
        out = dict.fromkeys(MECHANICS, np.nan)
        if i_r is not None and k is not None:
            t_c = float(t[k] - DT / 2)
            cls, reach, med, grow = outcome(d, k)
            t_rc = t_c + clock_shift  # contact on the receiver's clock
            pc = np.array([np.interp(t_rc, t, x), np.interp(t_rc, t, y)])
            v = lambda tt: float(np.interp(tt, t, sp)) if t[0] <= tt <= t[-1] else np.nan  # noqa: E731
            v_pre, v_c, v_post = v(t_rc - win), v(t_rc), v(t_rc + win)
            ap = flight_approach(t, x, y, bx, by, i_r, k)
            r.update(t_release=float(t[i_r]), t_contact=t_c, flight_time=t_c - float(t[i_r]),
                     v_release=float(np.median(bs[i_r:i_r + MIN_FLIGHT_FRAMES])), v_arrive=float(bs[k - 1]),
                     outcome=cls, reach=reach, post_median=med, post_growth=grow,
                     catch_x=float(pc[0]), catch_y=float(pc[1]), v_pre=v_pre, v_contact=v_c, v_post=v_post,
                     rec_v_release=float(np.hypot(vx[i_r], vy[i_r])), tail=float(t[-1] - t_c),
                     t_cpa=ap["t_ca"], d_cpa=ap["d_ca"], d_flight_min=ap["d_flight"],
                     t_flight_min_before_drop=round(float(t[k]) - ap["t_flight"], 3),
                     bs_post_max=float(np.nanmax(bs[k:])) if np.isfinite(bs[k:]).any() else np.nan)
            if k - 3 >= i_r:
                u = np.array([bx[k - 1] - bx[k - 3], by[k - 1] - by[k - 3]])
                u /= np.linalg.norm(u)
                # ball minus receiver along the flight line at the last in-flight frame (< 0: ball still short of him)
                r["ball_along_last_flight"] = float(u @ np.array([bx[k - 1] - x[k - 1], by[k - 1] - y[k - 1]]))
            if cls != "not_reached":
                out["decel_in"] = (v_pre - v_c) / win
                if k - 3 >= i_r:
                    w = pc - np.array([bx[k - 1], by[k - 1]])
                    out["catch_offset"] = float(abs(u[0] * w[1] - u[1] * w[0]))
                if r["drill_name"] in ADJUST_DRILLS:
                    h = np.array([vx[i_r], vy[i_r]])
                    e = pc - (np.array([x[i_r], y[i_r]]) + h * (t_c - t[i_r]))
                    out["adjust_dist"] = float(np.linalg.norm(e))
                    out["adjust_time"] = _lock_time(t, x, y, vx, vy, i_r, k, t_c, pc)
                    hn = np.linalg.norm(h)
                    if hn > 0:  # components along / across his release heading (descriptive only)
                        r.update(adjust_along=float(e @ h / hn), adjust_across=float(abs(e[0] * h[1] - e[1] * h[0]) / hn))
            if cls == "secured" and v_pre >= MIN_SPEED_RETENTION and np.isfinite(v_post):
                out["retention"] = v_post / v_pre
        rows.append({**r, **out})
    att = pl.from_dicts(rows, infer_schema_length=None).sort("traj")
    nr = pl.col("outcome") == "not_reached"
    return att.with_columns(outcome_posthoc=pl.when(nr & (pl.col("d_min_all") > UNRELATED_DIST))
                            .then(pl.lit("ball_unrelated"))
                            .when(nr & (pl.col("d_flight_min") <= PASS_DIST)
                                  & (pl.col("t_flight_min_before_drop") >= PASS_MIN_T))
                            .then(pl.lit("passed_him")).otherwise(pl.col("outcome")))


# ---------------------------------------------------------------- player values and reliability

def drill_means(att: pl.DataFrame, mech: str) -> pl.DataFrame:
    """Player x drill (x session) means of one mechanic over qualifying attempts, with n attempts."""
    return (att.filter(pl.col(mech).is_finite())
            .group_by("nfl_id", "position", "drill_name", "session")
            .agg(pl.col(mech).mean(), n_att=pl.len()).sort("nfl_id", "drill_name"))


def player_values(att: pl.DataFrame) -> pl.DataFrame:
    """One row per ball-tracked player: each mechanic as the mean over his drills of the drill x session z-score of
    his drill mean, and <mech>_n_drills. Drill x session groups with < 3 players or no variance are skipped."""
    base = att.select("nfl_id", "position").unique().sort("nfl_id")
    for m in MECHANICS:
        dm = drill_means(att, m)
        grp = ["drill_name", "session"]
        z = (pl.col(m) - pl.col(m).mean().over(grp)) / pl.col(m).std().over(grp)
        p = (dm.filter((pl.len().over(grp) >= 3) & (pl.col(m).std().over(grp) > 0)).with_columns(z=z)
             .group_by("nfl_id").agg(pl.col("z").mean().alias(m), pl.len().alias(f"{m}_n_drills")))
        base = base.join(p, on="nfl_id", how="left")
    return base


def _cross_drill(dm: pl.DataFrame, m: str, center_by: list[str]) -> dict:
    """reliability.cross_drill_icc (zscore) on player x drill means, without zero-variance groups."""
    return reliability.cross_drill_icc(dm.filter(pl.col(m).std().over(center_by) > 0), m, zscore=True,
                                       center_by=center_by)


def mechanic_reliability(att: pl.DataFrame, n_boot: int = 0, seed: int = 0) -> dict:
    """Per mechanic: cross-drill reliability of player x drill means (z within drill x session, the as-used form;
    also drill-centered only), within-drill retest ICC across repeated attempts (centered by drill x session), and
    with n_boot > 0 percentile 95% CIs of the as-used reliability and the per-drill ICC (resampling players) and of
    the retest ICC (resampling repeated player x drill cells; separate RNG stream, seed + 1)."""
    out, rng, rng_rt = {}, np.random.default_rng(seed), np.random.default_rng(seed + 1)
    grp = ["drill_name", "session"]
    for m in MECHANICS:
        dm = drill_means(att, m)
        cd = _cross_drill(dm, m, grp)
        cd0 = _cross_drill(dm, m, ["drill_name"])
        a = att.filter(pl.col(m).is_finite()).with_columns(_c=pl.col(m) - pl.col(m).mean().over(grp))
        a = a.with_columns(_g=pl.concat_str([pl.col("nfl_id").cast(pl.Utf8), pl.col("drill_name")], separator="|"))
        rt = reliability.icc1(a["_c"].to_numpy(), a["_g"].to_numpy())
        res = {"n_attempts": int(att[m].is_finite().sum()), "n_players": int(dm["nfl_id"].n_unique()),
               "n_drills": int(dm["drill_name"].n_unique()),
               "median_drills_per_player": float(dm.group_by("nfl_id").len()["len"].median()),
               "cross_drill": cd, "cross_drill_drill_centered": cd0, "within_drill_retest": rt}
        if n_boot and np.isfinite(cd["reliability"]):
            ids = dm["nfl_id"].unique().sort().to_numpy()
            boot, boot_icc = [], []
            for _ in range(n_boot):  # resampled players get new ids so duplicates count as distinct players
                pick = pl.DataFrame({"nfl_id": rng.choice(ids, len(ids)), "_b": np.arange(len(ids))})
                b = dm.join(pick, on="nfl_id").with_columns(nfl_id=pl.col("_b")).drop("_b")
                cb = _cross_drill(b, m, grp)
                boot.append(cb["reliability"])
                boot_icc.append(cb["icc"])
            boot, boot_icc = np.array(boot, dtype=float), np.array(boot_icc, dtype=float)
            res["cross_drill_boot_ci95"] = [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))]
            res["cross_drill_icc_boot_ci95"] = [float(np.nanpercentile(boot_icc, 2.5)),
                                                float(np.nanpercentile(boot_icc, 97.5))]
            res["p_boot_ge_05"] = float(np.nanmean(boot >= 0.5))
        if n_boot and np.isfinite(rt["icc"]):
            rep = a.filter(pl.len().over("_g") >= 2).sort("_g", "attempt")
            cells = [c["_c"].to_numpy() for c in rep.partition_by("_g", maintain_order=True)]
            bt = []
            for _ in range(n_boot):
                pick = rng_rt.integers(0, len(cells), len(cells))
                bt.append(reliability.icc1(np.concatenate([cells[i] for i in pick]),
                                           np.repeat(np.arange(len(pick)), [len(cells[i]) for i in pick]))["icc"])
            res["within_drill_retest_icc_boot_ci95"] = [float(np.nanpercentile(bt, 2.5)),
                                                        float(np.nanpercentile(bt, 97.5))]
        out[m] = res
    return out


# ---------------------------------------------------------------- rookie receiving outcomes (exploratory)

CATCH_CAT = ["team_coverage_man_zone", "receiver_alignment", "season"]
CATCH_NUM = ["pass_length", "separation_at_pass_forward", "down", "yards_to_go", "yards_to_end_zone"]


def target_plays(season: int | None = None) -> pl.DataFrame:
    """REG targets of study WR/TE (targets_wr.route_table rows with target True and a pass result C / I / IN), with
    caught (0/1), p_catch (cross-fitted over all seasons' targets; rows missing pass_length are dropped first),
    drop (own dropped_pass: the event on a play where he is the target) and yacoe (catches with NGS xYAC).
    `season` filters the returned rows after cross-fitting."""
    rt = targets_wr.route_table()
    tg = (rt.filter((pl.col("target") == True) & pl.col("pass_result").is_in(["C", "I", "IN"])  # noqa: E712
                    & pl.col("pass_length").is_not_null())
          .with_columns(caught=(pl.col("pass_result") == "C").cast(pl.Float64)).sort("game_id", "play_id", "nfl_id"))
    tg = tg.with_columns(p_catch=pl.Series(targets_wr.cross_fit(tg, "caught", kind="classification",
                                                                cat=CATCH_CAT, num=CATCH_NUM)))
    seasons = tuple(sorted(tg["season"].unique().to_list()))
    ids = tg["nfl_id"].unique().implode()
    drops = (io.scan_game_tracking(seasons).filter((pl.col("event") == "dropped_pass") & pl.col("nfl_id").is_in(ids))
             .select("game_id", "play_id", "nfl_id").unique().collect().with_columns(drop=pl.lit(1.0)))
    tg = tg.join(drops, on=["game_id", "play_id", "nfl_id"], how="left", maintain_order="left").with_columns(
        drop=pl.col("drop").fill_null(0.0),
        yacoe=pl.when(pl.col("caught") == 1).then(pl.col("yards_after_catch") - pl.col("expected_yards_after_catch")))
    return tg if season is None else tg.filter(pl.col("season") == season)


def rookie_targets(tg: pl.DataFrame, min_targets: int = 10, min_catches: int = 5) -> pl.DataFrame:
    """Per player (all study WR/TE in `tg`, e.g. the 2025 season): EB-shrunk croe, drop_rate (rates over the
    expectation / base rate) and yacoe (mean), each with its n. Priors are fitted on every study WR/TE in tg with
    >= min_n, so a rookie's value is shrunk toward the season's WR/TE population."""
    croe = shrinkage.eb_rate(tg, "caught", expected="p_catch", min_n=min_targets, cluster="game_id")
    drp = shrinkage.eb_rate(tg, "drop", min_n=min_targets, cluster="game_id")
    yac = shrinkage.eb_mean(tg.filter(pl.col("yacoe").is_finite()), "yacoe", min_n=min_catches, cluster="game_id")
    return (croe.select("nfl_id", croe="shrunk", croe_raw="raw", croe_w="weight", n_targets="n")
            .join(drp.select("nfl_id", drop_rate="shrunk", drop_w="weight", n_drops="events"), on="nfl_id", how="full",
                  coalesce=True)
            .join(yac.select("nfl_id", yacoe="shrunk", yacoe_w="weight", n_catches="n"), on="nfl_id", how="full",
                  coalesce=True))
