"""Probe D (E005): DB backpedal -> transition. Combine DB drills vs regular-season coverage transitions.

Combine side (no orientation in Combine tracking, so the backpedal is identified by drill design):
  - TERYL_AUSTIN_DRILL_1: retreat ~10 yd, plant, drive back at ~125 deg (one reversal per attempt).
  - TERYL_AUSTIN_DRILL_2: backpedal, 90 deg turn, near-full stop, ~175 deg reversal.
  - W_DRILL: three low-speed reversals (~120 deg). BOX_DRILL: two ~120 deg corners.
  - BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL: ~5 yd backpedal, rounded 90 deg break.
  - BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION: two attempts per player; the 45 deg opening is too gradual
    to detect; the late U-turn reaction is detected but unreliable (ICC ~0). Used only for reliability.
Game side: DBs aligned at CB / SLOT_CB / HIGH_SAFETY / BOX_SAFETY on regular-season pass plays, snap-aligned,
breaks detected with segments.breaks; a "transition out of the backpedal" is a break whose entry phase
(0.8-0.3 s before the speed minimum) moves downfield (vx > 1 yd/s) while facing the line of scrimmage
(orientation within 60 deg of 270, i.e. facing -x after standardization).
"""

import numpy as np
import polars as pl

from bdb27 import features, io, kinematics, linkage, segments
from bdb27.kinematics import DT

COVERAGE_ALIGNMENTS = ["CB", "SLOT_CB", "HIGH_SAFETY", "BOX_SAFETY"]
PASS_RESULTS = ["C", "I", "S", "IN", "R"]
DB_DRILLS = ["TERYL_AUSTIN_DRILL_1", "TERYL_AUSTIN_DRILL_2", "W_DRILL", "BOX_DRILL",
             "BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL", "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION"]
REVERSAL_DRILLS = ["TERYL_AUSTIN_DRILL_1", "TERYL_AUSTIN_DRILL_2", "W_DRILL", "BOX_DRILL"]
BREAK_KW = dict(min_turn=45, half_window=0.5, context=1.0)
END_EVENTS = ["qb_sack", "qb_strip_sack", "run", "qb_slide", "out_of_bounds", "tackle", "fumble"]
ARRIVE_EVENTS = ["pass_arrived", "pass_outcome_caught", "pass_outcome_incomplete",
                 "pass_outcome_interception", "pass_outcome_touchdown"]
MAX_WINDOW = 5.0  # s after the snap


# ---------- shared helpers ----------

def window_features(frames: pl.DataFrame, brk: pl.DataFrame) -> pl.DataFrame:
    """Speed-change features over fixed windows around each break's speed minimum (i_break), which are
    less noisy than peak derivatives: drive_05 / drive_10 = speed gained 0.5 / 1.0 s after the minimum,
    brake_05 = speed lost in the 0.5 s before it, dv_10 = |velocity change| from 0.5 s before to 0.5 s after."""
    f = frames.sort("traj", "time").with_columns(fi=pl.int_range(pl.len()).over("traj"))
    f = f.select("traj", "fi", "speed", "vx", "vy")

    def at(off: int, name: str) -> pl.DataFrame:
        return (brk.select("traj", "k", fi=pl.col("i_break") + off)
                .join(f, on=["traj", "fi"], how="left")
                .select("traj", "k", **{f"sp_{name}": "speed", f"vx_{name}": "vx", f"vy_{name}": "vy"}))

    out = brk
    offsets = [(-5, "m05"), (0, "0"), (5, "p05"), (10, "p10")]
    for off, name in offsets:
        out = out.join(at(off, name), on=["traj", "k"], how="left")
    tmp = [f"{p}_{n}" for p in ("sp", "vx", "vy") for _, n in offsets]
    return out.with_columns(
        drive_05=pl.col("sp_p05") - pl.col("sp_0"),
        drive_10=pl.col("sp_p10") - pl.col("sp_0"),
        brake_05=pl.col("sp_m05") - pl.col("sp_0"),
        dv_10=((pl.col("vx_p05") - pl.col("vx_m05")) ** 2 + (pl.col("vy_p05") - pl.col("vy_m05")) ** 2).sqrt(),
    ).drop(tmp)


# ---------- Combine ----------

def combine_frames() -> pl.DataFrame:
    """DB drill attempts, canonicalized (origin at onset, initial direction +x, mirrored to y >= 0),
    trimmed to 0.5 s before onset, with kinematics. Adds t_on (s since onset)."""
    df = segments.attempts(drill_names=DB_DRILLS)
    c = segments.canonicalize(df)
    c = kinematics.add_kinematics(c.drop("traj"), segments.ATTEMPT_KEYS)
    c = c.with_columns(i=pl.int_range(pl.len()).over("traj").cast(pl.Int64),
                       i0=pl.col("onset").arg_max().over("traj").cast(pl.Int64))  # cast: u32 - 5 underflows
    c = c.filter(pl.col("i") >= pl.col("i0") - 5)
    c = c.with_columns(t_on=(pl.int_range(pl.len()).over("traj") - pl.col("onset").arg_max().over("traj").cast(pl.Int64)) * DT)
    return c.drop("i", "i0")


def combine_events(frames: pl.DataFrame) -> pl.DataFrame:
    """One row per drill transition used in the probe, with `dc` = drill + cut index.
    Selection rules (fixed after visual QC, before any linkage):
      TA1: the largest turn >= 90 deg at >= 1.5 s after onset (the plant-and-drive).
      TA2: the largest turn >= 120 deg at >= 1.5 s (the stop-and-reverse).
      W: the first three turns >= 90 deg. BOX: the first two turns >= 90 deg.
      BP90: the first break (>= 45 deg) within 3.5 s of onset (the 90 deg break from the backpedal).
      BP45: the largest turn at >= 5 s (late reaction U-turn; reliability only)."""
    br = segments.breaks(frames, ["nfl_id", "draft_year", "drill_name", "attempt", "event_id"],
                         min_entry_speed=1.0, **BREAK_KW)
    j0 = frames.group_by("traj").agg(j0=pl.col("onset").arg_max().cast(pl.Int64))
    br = br.join(j0, on="traj").with_columns(t_on=(pl.col("i_break") - pl.col("j0")) * DT,
                                             aturn=pl.col("turn_deg").abs()).drop("j0")
    dn, at, t = pl.col("drill_name"), pl.col("aturn"), pl.col("t_on")

    def largest(cond):
        return br.filter(cond).sort("aturn", descending=True).unique("traj", keep="first").with_columns(cut=pl.lit(0, pl.Int64))

    def first_n(cond, n):
        return (br.filter(cond).sort("traj", "k").with_columns(cut=pl.int_range(pl.len()).over("traj").cast(pl.Int64))
                .filter(pl.col("cut") < n))

    ev = pl.concat([
        largest((dn == "TERYL_AUSTIN_DRILL_1") & (at >= 90) & (t >= 1.5)),
        largest((dn == "TERYL_AUSTIN_DRILL_2") & (at >= 120) & (t >= 1.5)),
        first_n((dn == "W_DRILL") & (at >= 90), 3),
        first_n((dn == "BOX_DRILL") & (at >= 90), 2),
        first_n((dn == "BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL") & (t <= 3.5), 1),
        largest((dn == "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION") & (t >= 5)),
    ])
    ev = window_features(frames, ev)
    return ev.with_columns(dc=pl.col("drill_name") + "_" + pl.col("cut").cast(pl.String))


def combine_backpedal(frames: pl.DataFrame) -> pl.DataFrame:
    """Backpedal-phase features per attempt for drills that open with a backpedal along +x:
    bp_peak = max speed before the first transition window (BP90: t_on < 1.6 s; TA2: < 1.2 s; W/BOX: < 1.0 s;
    BP45: < 1.0 s), bp_v05 / bp_v10 = speed 0.5 / 1.0 s after onset (backpedal burst)."""
    lim = {"BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL": 1.6, "TERYL_AUSTIN_DRILL_2": 1.2, "W_DRILL": 1.0,
           "BOX_DRILL": 1.0, "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION": 1.0}
    f = frames.filter(pl.col("drill_name").is_in(list(lim)))
    f = f.with_columns(lim=pl.col("drill_name").replace_strict(lim, return_dtype=pl.Float64))
    return (f.group_by("traj", "nfl_id", "drill_name", "attempt", "event_id")
            .agg(bp_peak=pl.col("speed").filter((pl.col("t_on") >= 0) & (pl.col("t_on") <= pl.col("lim"))).max(),
                 bp_v05=pl.col("speed").filter((pl.col("t_on") - 0.5).abs() < 0.05).first(),
                 bp_v10=pl.col("speed").filter((pl.col("t_on") - 1.0).abs() < 0.05).first())
            .with_columns(dc=pl.col("drill_name")))


def adjusted_z(events: pl.DataFrame, value: str, adjust: list[str] | None = None, clip: float = 3.0) -> pl.DataFrame:
    """Within each drill-cut (`dc`): optionally residualize `value` on `adjust` columns (OLS with intercept),
    then z-score (clipped at +-clip). Adds `z`. Mirrors the game-side context adjustment."""
    out = []
    for _, g in events.filter(pl.col(value).is_finite()).group_by("dc", maintain_order=True):
        y = g[value].to_numpy().astype(float)
        if adjust:
            X = np.column_stack([np.ones(len(y))] + [g[c].to_numpy().astype(float) for c in adjust])
            y = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        sd = y.std(ddof=1)
        z = np.clip((y - y.mean()) / sd, -clip, clip) if sd > 0 else np.zeros_like(y)
        out.append(g.with_columns(z=pl.Series(z)))
    return pl.concat(out)


def player_composite(events: pl.DataFrame, value: str, drills: list[str], name: str,
                     adjust: list[str] | None = None) -> pl.DataFrame:
    """Per player: adjusted z within drill-cut, average the cuts (and attempts) of a drill, then average drills
    (equal weight per drill). Returns nfl_id, <name>, <name>_n_drills."""
    d = adjusted_z(events.filter(pl.col("drill_name").is_in(drills)), value, adjust)
    per_drill = d.group_by("nfl_id", "drill_name").agg(z=pl.col("z").mean())
    return per_drill.group_by("nfl_id").agg(pl.col("z").mean().alias(name), pl.len().alias(f"{name}_n_drills"))


# ---------- Game ----------

def game_frames() -> pl.DataFrame:
    """Regular-season pass plays of Combine DBs aligned in coverage, snap-aligned (-0.5..6 s), with kinematics
    and play context (lined_up_position, coverage, pass_result)."""
    pt = linkage.player_table()
    dbids = pt.filter(pl.col("position") == "DB")["nfl_id"]
    reg = io.load("games").filter(pl.col("season_type") == "REG").select("game_id")
    pp = (io.load("player_play").filter(pl.col("nfl_id").is_in(dbids.implode()))
          .join(reg, on="game_id", how="semi")
          .filter(pl.col("pass_result").is_in(PASS_RESULTS) & pl.col("lined_up_position").is_in(COVERAGE_ALIGNMENTS))
          .select(*segments.PLAY_KEYS, "lined_up_position", "coverage_assignment", "team_coverage_man_zone",
                  "team_coverage_type", "pass_result", "blitzing"))
    lf = features.reg_season_tracking().filter(pl.col("nfl_id").is_in(dbids.implode()))
    lf = lf.join(pp.lazy(), on=segments.PLAY_KEYS, how="inner", maintain_order="left")
    al = segments.snap_aligned(lf).filter((pl.col("t_rel") >= -0.5) & (pl.col("t_rel") <= 6.0))
    df = kinematics.add_kinematics(al.collect(), segments.PLAY_KEYS)
    return df


def _ctx(frames: pl.DataFrame, brk: pl.DataFrame, lo: int, hi: int, name: str) -> pl.DataFrame:
    w = (brk.select("traj", "k", "i_break").join(frames.select("traj", "fi", "vx", "o"), on="traj")
         .filter((pl.col("fi") >= pl.col("i_break") + lo) & (pl.col("fi") <= pl.col("i_break") + hi)))
    return w.group_by("traj", "k").agg(**{
        f"vx_{name}": pl.col("vx").mean(),
        f"o_{name}": pl.arctan2(pl.col("o").radians().sin().mean(), pl.col("o").radians().cos().mean()).degrees() % 360})


def game_events(frames: pl.DataFrame, min_entry_speed: float = 1.5) -> pl.DataFrame:
    """One row per detected break in the coverage window, with context and the transition class.
    Coverage window: 0.3 s after the snap until the first of pass arrival / sack / scramble / tackle /
    out of bounds, capped at 5 s. Breaks are detected on the whole -0.5..6 s trajectory so re-acceleration
    after a late break is not truncated."""
    frames = frames.sort("traj", "time").with_columns(fi=pl.int_range(pl.len()).over("traj"))
    br = segments.breaks(frames, segments.PLAY_KEYS, min_entry_speed=min_entry_speed, **BREAK_KW)
    ev = frames.group_by("traj").agg(
        t_pf=pl.col("t_rel").filter(pl.col("event") == "pass_forward").min(),
        t_pa=pl.col("t_rel").filter(pl.col("event").is_in(ARRIVE_EVENTS)).min(),
        t_stop=pl.col("t_rel").filter(pl.col("event").is_in(END_EVENTS)).min())
    tb = (br.select("traj", "k", "i_break").join(frames.select("traj", "fi", "t_rel"), left_on=["traj", "i_break"],
                                                  right_on=["traj", "fi"]).rename({"t_rel": "t_rel_break"}).drop("i_break"))
    ctx = frames.unique("traj").select("traj", "lined_up_position", "coverage_assignment", "team_coverage_man_zone",
                                       "team_coverage_type", "pass_result", "blitzing")
    B = (br.join(_ctx(frames, br, -8, -3, "in"), on=["traj", "k"], how="left")
         .join(_ctx(frames, br, 3, 8, "out"), on=["traj", "k"], how="left")
         .join(tb, on=["traj", "k"]).join(ev, on="traj").join(ctx, on="traj"))
    B = B.with_columns(
        aturn=pl.col("turn_deg").abs(),
        face_in=((pl.col("o_in") - 270 + 180) % 360 - 180).abs() <= 60,
        t_end=pl.min_horizontal(pl.col("t_pa").fill_null(99), pl.col("t_stop").fill_null(99), pl.lit(MAX_WINDOW)),
        mz=pl.col("team_coverage_man_zone").fill_null("NA"),
        phase=pl.when(pl.col("t_pf").is_null() | (pl.col("t_rel_break") < pl.col("t_pf")))
        .then(pl.lit("pre_pass")).otherwise(pl.lit("ball_in_air")),
    )
    B = B.filter((pl.col("t_rel_break") >= 0.3) & (pl.col("t_rel_break") <= pl.col("t_end")))
    B = B.with_columns(
        from_backpedal=pl.col("face_in") & (pl.col("vx_in") > 1),
        reversal=(pl.col("vx_in") > 1) & (pl.col("vx_out") < -1) & (pl.col("aturn") >= 100),
    )
    return window_features(frames, B)


def game_backpedal_plays(frames: pl.DataFrame, min_frames: int = 5, t_max: float = 2.5) -> pl.DataFrame:
    """Per player-play backpedal phase: frames in [0, min(pass_forward, t_max)] s after the snap where the DB moves
    downfield (vx > 0.5) while facing the line of scrimmage (orientation within 60 deg of 270).
    bp_peak = max speed over those frames (plays with >= min_frames backpedal frames only);
    bp_frames = number of backpedal frames."""
    t_pf = frames.group_by("traj").agg(t_pf=pl.col("t_rel").filter(pl.col("event") == "pass_forward").min())
    f = frames.join(t_pf, on="traj").filter(
        (pl.col("t_rel") >= 0) & (pl.col("t_rel") <= pl.min_horizontal(pl.col("t_pf").fill_null(99), pl.lit(t_max))))
    f = f.with_columns(bp=(pl.col("vx") > 0.5) & ((((pl.col("o") - 270 + 180) % 360) - 180).abs() <= 60))
    out = (f.group_by("traj", *segments.PLAY_KEYS, "lined_up_position", "coverage_assignment", "team_coverage_man_zone")
           .agg(bp_frames=pl.col("bp").sum(), bp_peak=pl.col("speed").filter("bp").max()))
    return out.filter(pl.col("bp_frames") >= min_frames).with_columns(mz=pl.col("team_coverage_man_zone").fill_null("NA"))


def context_design(d: pl.DataFrame, include_entry: bool = True) -> np.ndarray:
    """Event-level context: turn angle (quadratic), entry speed (optional), ball-in-air phase,
    alignment (vs CB) and man/zone (vs zone)."""
    a = d["aturn"].to_numpy() / 100
    cols = [np.ones(d.height), a, a ** 2]
    if include_entry:
        cols.append(d["entry_speed"].to_numpy())
    cols.append((d["phase"] == "ball_in_air").cast(pl.Float64).to_numpy())
    for p in ["SLOT_CB", "HIGH_SAFETY", "BOX_SAFETY"]:
        cols.append((d["lined_up_position"] == p).cast(pl.Float64).to_numpy())
    for m in ["MAN_COVERAGE", "NA"]:
        cols.append((d["mz"] == m).cast(pl.Float64).to_numpy())
    return np.column_stack(cols)


def residualize(d: pl.DataFrame, value: str, include_entry: bool = True, wins: float = 0.01) -> pl.DataFrame:
    """Winsorize `value` at the wins/1-wins quantiles, regress on context_design (OLS), add `res`."""
    d = d.filter(pl.col(value).is_finite())
    lo, hi = d[value].quantile(wins), d[value].quantile(1 - wins)
    d = d.with_columns(pl.col(value).clip(lo, hi))
    X = context_design(d, include_entry)
    y = d[value].to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return d.with_columns(res=pl.Series(y - X @ beta))


def eb_shrink(d: pl.DataFrame, value: str = "res", player: str = "nfl_id", min_n: int = 1) -> pl.DataFrame:
    """Empirical-Bayes (normal-normal) shrinkage of per-player means of event-level `value` toward 0
    (the context-adjusted population mean). sigma2 = pooled within-player variance; tau2 = method-of-moments
    between-player variance (var of player means minus mean sampling variance, floored at 1e-6).
    Returns nfl_id, mean, n, shrunk, weight = tau2 / (tau2 + sigma2 / n)."""
    g = d.group_by(player).agg(mean=pl.col(value).mean(), n=pl.len(), ss=((pl.col(value) - pl.col(value).mean()) ** 2).sum())
    g = g.filter(pl.col("n") >= min_n)
    sigma2 = float(g["ss"].sum() / max(1, (g["n"].sum() - g.height)))
    tau2 = max(1e-6, float(g["mean"].var() - (sigma2 / g["n"]).mean()))
    return (g.with_columns(weight=tau2 / (tau2 + sigma2 / pl.col("n")))
            .with_columns(shrunk=pl.col("weight") * pl.col("mean"))
            .drop("ss").with_columns(pl.lit(tau2).alias("tau2"), pl.lit(sigma2).alias("sigma2")))


# ---------- player-level signatures and traits ----------

BACKPEDAL_DRILLS = ["BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL", "TERYL_AUSTIN_DRILL_2", "W_DRILL", "BOX_DRILL",
                    "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION"]
MIN_BACKPEDAL_PLAYS = 20
MIN_BALLAIR_TRANSITIONS = 10


def combine_signatures(events: pl.DataFrame, backpedal: pl.DataFrame) -> pl.DataFrame:
    """Player-level Combine signatures (one row per DB with any of them).
    cmb_backpedal_speed: peak speed in the opening backpedal leg, z within drill, averaged over up to 5 drills.
    cmb_transition_dv: |velocity change| from 0.5 s before to 0.5 s after the plant (speed minimum) in the
      reversal drills (TA1, TA2, W x3, BOX x2), residualized on entry speed and turn angle within drill-cut,
      z-scored, averaged within drill, then across drills."""
    bp = player_composite(backpedal.with_columns(dc=pl.col("drill_name")), "bp_peak", BACKPEDAL_DRILLS,
                          "cmb_backpedal_speed")
    dv = player_composite(events, "dv_10", REVERSAL_DRILLS, "cmb_transition_dv", adjust=["entry_speed", "aturn"])
    return bp.join(dv, on="nfl_id", how="full", coalesce=True)


def game_ballair_events(events: pl.DataFrame, value: str) -> pl.DataFrame:
    """Transitions out of the backpedal after the throw, residualized on context (`res`)."""
    e = events.filter(pl.col("from_backpedal") & (pl.col("phase") == "ball_in_air"))
    return residualize(e, value, include_entry=(value != "entry_speed"))


def game_backpedal_events(bp_plays: pl.DataFrame, wins: float = 0.01) -> pl.DataFrame:
    """Per-play backpedal peak speed, winsorized, minus the mean of its alignment x coverage-assignment cell (`res`)."""
    d = bp_plays.filter(pl.col("bp_peak").is_finite()).with_columns(cov=pl.col("coverage_assignment").fill_null("NA"))
    lo, hi = d["bp_peak"].quantile(wins), d["bp_peak"].quantile(1 - wins)
    d = d.with_columns(pl.col("bp_peak").clip(lo, hi))
    return d.with_columns(res=pl.col("bp_peak") - pl.col("bp_peak").mean().over("lined_up_position", "cov"))


def game_traits(events: pl.DataFrame, bp_plays: pl.DataFrame, frames: pl.DataFrame) -> pl.DataFrame:
    """Player-level game traits (EB-shrunk means of context-adjusted event values) with exposure counts.
    gm_backpedal_speed: per-play backpedal peak speed (>= MIN_BACKPEDAL_PLAYS plays).
    gm_ballair_dv: dv_10 in post-throw transitions out of the backpedal (>= MIN_BALLAIR_TRANSITIONS events)."""
    plays = frames.group_by("nfl_id").agg(cov_pass_plays=pl.col("traj").n_unique())
    bp = eb_shrink(game_backpedal_events(bp_plays), min_n=MIN_BACKPEDAL_PLAYS).select(
        "nfl_id", gm_backpedal_speed="shrunk", gm_backpedal_speed_raw="mean", gm_backpedal_n="n",
        gm_backpedal_w="weight")
    dv = eb_shrink(game_ballair_events(events, "dv_10"), min_n=MIN_BALLAIR_TRANSITIONS).select(
        "nfl_id", gm_ballair_dv="shrunk", gm_ballair_dv_raw="mean", gm_ballair_n="n", gm_ballair_w="weight")
    return plays.join(bp, on="nfl_id", how="left").join(dv, on="nfl_id", how="left")


def game_event_trait(events: pl.DataFrame, value: str, cond: pl.Expr, name: str, min_n: int) -> pl.DataFrame:
    """Generic EB-shrunk player trait from game break events matching `cond` (context-residualized `value`)."""
    e = residualize(events.filter(cond), value, include_entry=(value != "entry_speed"))
    return eb_shrink(e, min_n=min_n).select("nfl_id", pl.col("shrunk").alias(name), pl.col("n").alias(f"{name}_n"))


def game_backpedal_trait(bp_plays: pl.DataFrame, cond: pl.Expr, name: str, min_n: int = MIN_BACKPEDAL_PLAYS) -> pl.DataFrame:
    """EB-shrunk backpedal-speed trait on the subset of plays matching `cond`."""
    return eb_shrink(game_backpedal_events(bp_plays.filter(cond)), min_n=min_n).select(
        "nfl_id", pl.col("shrunk").alias(name), pl.col("n").alias(f"{name}_n"))


# ---------- reliability screens ----------

def combine_reliability(events: pl.DataFrame, backpedal: pl.DataFrame) -> pl.DataFrame:
    """Reliability of Combine transition and backpedal features under several definitions:
    events-as-repeats (cross-drill ICC over all drill-cuts, centered), drill means across drills (stricter: removes
    within-attempt consistency), TA1 vs TA2, W cuts, BOX cuts, and BP45 attempts (icc1, 2 attempts/player)."""
    from bdb27 import reliability as R
    rows = []
    rev = events.filter(pl.col("drill_name").is_in(REVERSAL_DRILLS))
    b45 = events.filter(pl.col("drill_name") == "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION")
    feats = ["entry_speed", "peak_decel", "brake_05", "min_speed", "peak_reaccel", "drive_05", "drive_10", "dv_10",
             "decel_time", "retention", "turn_radius", "max_turn_rate"]
    for f in feats:
        for adj_name, adj in [("none", None), ("entry+turn", ["entry_speed", "aturn"])]:
            if f == "entry_speed" and adj:
                continue
            d = adjusted_z(rev, f, adj)
            rows.append(dict(feature=f, adjust=adj_name, scope="reversal events (TA1,TA2,Wx3,BOXx2)",
                             **R.cross_drill_icc(d, "z", drill="dc")))
            dm = d.group_by("nfl_id", "drill_name").agg(z=pl.col("z").mean())
            rows.append(dict(feature=f, adjust=adj_name, scope="reversal drill means (4 drills)",
                             **R.cross_drill_icc(dm, "z", drill="drill_name")))
            ta = d.filter(pl.col("drill_name").str.starts_with("TERYL"))
            rows.append(dict(feature=f, adjust=adj_name, scope="TA1 vs TA2", **R.cross_drill_icc(ta, "z", drill="dc")))
            for dn in ["W_DRILL", "BOX_DRILL"]:
                rows.append(dict(feature=f, adjust=adj_name, scope=f"{dn} cuts",
                                 **R.cross_drill_icc(d.filter(pl.col("drill_name") == dn), "z", drill="dc")))
        rows.append(dict(feature=f, adjust="none", scope="BP45 reaction U-turn, 2 attempts (icc1)",
                         **R.icc1(b45[f].to_numpy().astype(float), b45["nfl_id"].to_numpy())))
    bp = backpedal.with_columns(dc=pl.col("drill_name"))
    for f in ["bp_peak", "bp_v05", "bp_v10"]:
        d = adjusted_z(bp, f, None)
        rows.append(dict(feature=f, adjust="none", scope="backpedal attempts (5 drills)", **R.cross_drill_icc(d, "z", drill="dc")))
        dm = d.group_by("nfl_id", "drill_name").agg(z=pl.col("z").mean())
        rows.append(dict(feature=f, adjust="none", scope="backpedal drill means (5 drills)",
                         **R.cross_drill_icc(dm, "z", drill="drill_name")))
        b = bp.filter(pl.col("drill_name") == "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION")
        rows.append(dict(feature=f, adjust="none", scope="BP45 attempts (icc1)",
                         **R.icc1(b[f].to_numpy().astype(float), b["nfl_id"].to_numpy())))
    return pl.DataFrame(rows)


GAME_EVENT_SETS = {
    "from_backpedal": pl.col("from_backpedal"),
    "from_backpedal_ball_in_air": pl.col("from_backpedal") & (pl.col("phase") == "ball_in_air"),
    "from_backpedal_pre_pass": pl.col("from_backpedal") & (pl.col("phase") == "pre_pass"),
    "backpedal_reversal": pl.col("from_backpedal") & pl.col("reversal"),
    "all_breaks": pl.lit(True),
}


def game_reliability(events: pl.DataFrame, bp_plays: pl.DataFrame) -> pl.DataFrame:
    """Split-half (odd/even games, Spearman-Brown) reliability of context-residualized game traits."""
    from bdb27 import reliability as R
    rows = []
    feats = ["entry_speed", "peak_decel", "brake_05", "min_speed", "peak_reaccel", "drive_05", "drive_10", "dv_10",
             "decel_time", "retention", "turn_radius", "max_turn_rate"]
    for s, cond in GAME_EVENT_SETS.items():
        E = events.filter(cond)
        for f in feats:
            d = residualize(E, f, include_entry=(f != "entry_speed"))
            r5, r10 = R.split_half(d, "res", min_per_half=5), R.split_half(d, "res", min_per_half=10)
            rows.append(dict(event_set=s, feature=f, n_events=d.height, rel_min5=r5["reliability"], n_min5=r5["n_players"],
                             rel_min10=r10["reliability"], n_min10=r10["n_players"]))
    d = game_backpedal_events(bp_plays)
    r5, r10 = R.split_half(d, "res", min_per_half=5), R.split_half(d, "res", min_per_half=10)
    rows.append(dict(event_set="backpedal_plays", feature="bp_peak", n_events=d.height, rel_min5=r5["reliability"],
                     n_min5=r5["n_players"], rel_min10=r10["reliability"], n_min10=r10["n_players"]))
    return pl.DataFrame(rows)
