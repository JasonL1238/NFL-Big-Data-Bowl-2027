"""Probe C (E004): OL lateral movement in Combine drills -> pass protection in games.

Combine side (OL only):
  - PASS_PRO mirror drill (two naming variants, merged): pure lateral shuffles with ~4 reversals, then a run-out.
  - FIVE_YARD_WAVE drill: multidirectional slides/shuffles on a coach's cue, then a run-out.
  - PASS_RUSH_DROP: two attempts; a ~2-yard kick back, a stop, then a turn-and-sprint.
  Reversals = segments.breaks with min_turn=120 deg and entry speed >= 1 yd/s. Per reversal we add
  `rev_accel` = |v_exit - v_entry| / (t_exit - t_entry), the mean acceleration that swings the velocity
  vector from the entry speed peak to the exit speed peak (within +-1 s of the break).
  Values are centered by Combine session (drill x year x session; sessions are ~1.7 h apart and share
  coaches/partners), because drill tempo differs by session.

Game side (OL on REG pass-blocking snaps, lined up at T/G/C, >= 1 rusher encountered):
  - pass-set kinematics in the first ~1 s after the snap, on non-play-action dropbacks >= 2.0 s
    (offense moves to +x, so the set goes toward -x), centered by lined_up_position at the play level;
  - outcomes: pressure_allowed and peak_pressure_probability_allowed, each relative to a play-level
    expectation (dropback duration spline, T/G/C, 2+ rushers, play action, season), averaged per player
    and shrunk with empirical Bayes (method-of-moments normal-normal).
"""

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

from bdb27 import features, io, kinematics, linkage, segments

K = segments.PLAY_KEYS
MIRROR_NAMES = ["PASS_PRO-MIRROR_DRILL", "PASS_PRO_MIRROR_DRILL"]
WAVE_NAME = "FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE"
DROP_NAME = "PASS_RUSH_DROP"
SESSION_GAP_S = 600  # attempts more than 10 min apart belong to different Combine sessions
MIN_TURN = 120.0
MIN_ENTRY = 1.0  # yd/s; drops stance wobble at the start of an attempt
RUNOUT_SLOW = 1.5  # yd/s; the run-out (final sprint) starts at the last frame below this before peak speed
OL_POS = ["T", "G", "C"]
GAME_WINDOW = (-0.5, 4.5)  # s around the snap that we extract
MIN_DROPBACK_SET = 2.0  # s; pass-set kinematics only on dropbacks long enough to finish the set
MIN_PLAYS = 50


def ol_ids() -> pl.DataFrame:
    return linkage.player_table().filter(pl.col("position") == "OL").select("nfl_id")


# ---------------------------------------------------------------- Combine

def drill_frames(names: list[str]) -> pl.DataFrame:
    """OL frames of the named drills, with `drill` (mirror variants merged), `session`, kinematics."""
    a = segments.attempts(drill_names=names).join(ol_ids(), on="nfl_id", how="semi")
    a = a.with_columns(drill=pl.when(pl.col("drill_name").is_in(MIRROR_NAMES)).then(pl.lit("MIRROR"))
                       .when(pl.col("drill_name") == WAVE_NAME).then(pl.lit("WAVE"))
                       .when(pl.col("drill_name") == DROP_NAME).then(pl.lit("DROP"))
                       .otherwise(pl.col("drill_name")))
    starts = (a.group_by("traj", "drill", "draft_year").agg(t0=pl.col("time").min()).sort("drill", "draft_year", "t0")
              .with_columns(gap=pl.col("t0").diff().over("drill", "draft_year").dt.total_seconds().fill_null(0))
              .with_columns(sess=(pl.col("gap") > SESSION_GAP_S).cum_sum().over("drill", "draft_year"))
              .with_columns(session=pl.col("drill") + "_" + pl.col("draft_year").cast(pl.Utf8) + "_"
                            + pl.col("sess").cast(pl.Utf8))
              .select("traj", "session"))
    a = a.join(starts, on="traj", how="left", maintain_order="left")
    return kinematics.add_kinematics(a, ["traj"]).sort("traj", "time")


def _peak_frames(sp: np.ndarray, i: int, s: int, e: int, c: int = 10) -> tuple[int, int]:
    """Entry/exit speed-peak frames within +-c frames of break frame i (bounded by [s, e))."""
    ie = max(s, i - c) + int(np.argmax(sp[max(s, i - c):i + 1]))
    ix = i + int(np.argmax(sp[i:min(e, i + c + 1)]))
    return ie, ix


def add_rev_accel(frames: pl.DataFrame, br: pl.DataFrame) -> pl.DataFrame:
    """Add rev_accel (yd/s^2), rev_time (s, entry peak -> exit peak) and entry/exit velocity angles."""
    frames = frames.sort("traj", "time")
    traj = frames["traj"].to_numpy()
    bounds = np.flatnonzero(np.diff(traj)) + 1
    span = {int(traj[s]): (s, e) for s, e in zip(np.r_[0, bounds], np.r_[bounds, len(frames)])}
    sp, vx, vy = (frames[c].to_numpy() for c in ("speed", "vx", "vy"))
    out = {"rev_accel": [], "rev_time": [], "ent_ang": [], "ex_ang": []}
    for t, ib in zip(br["traj"].to_list(), br["i_break"].to_list()):
        s, e = span[t]
        ie, ix = _peak_frames(sp, s + ib, s, e)
        dt = (ix - ie) * kinematics.DT
        out["rev_accel"].append(np.hypot(vx[ix] - vx[ie], vy[ix] - vy[ie]) / dt if dt > 0 else np.nan)
        out["rev_time"].append(dt)
        out["ent_ang"].append(float(np.degrees(np.arctan2(vy[ie], vx[ie]))))
        out["ex_ang"].append(float(np.degrees(np.arctan2(vy[ix], vx[ix]))))
    return br.with_columns(pl.Series(k, v, dtype=pl.Float64) for k, v in out.items())


def runout_start(frames: pl.DataFrame, slow: float = RUNOUT_SLOW) -> pl.DataFrame:
    """Per traj: frame index where the final run-out begins = last frame before the attempt's peak
    speed with speed < `slow` (0 if none). Reversals from here on launch the sprint, not a shuffle."""
    rows = []
    for (t,), g in frames.sort("traj", "time").group_by("traj", maintain_order=True):
        sp = g["speed"].to_numpy()
        below = np.flatnonzero(sp[:int(np.argmax(sp))] < slow)
        rows.append((int(t), int(below[-1]) if len(below) else 0))
    return pl.DataFrame(rows, schema={"traj": pl.Int64, "i_runout": pl.Int64}, orient="row")


def combine_reversals(frames: pl.DataFrame, drop_runout: bool = True) -> pl.DataFrame:
    """One row per reversal (turn >= 120 deg, entry >= 1 yd/s) in the shuffle phase of mirror/wave
    attempts (if drop_runout: reversals within 0.3 s of the run-out start, or after it, are dropped)."""
    br = segments.breaks(frames, ["nfl_id", "drill", "session", "draft_year"], min_turn=MIN_TURN,
                         half_window=0.5, context=1.0, min_entry_speed=MIN_ENTRY)
    if drop_runout:
        br = (br.join(runout_start(frames), on="traj", how="left")
              .filter(pl.col("i_break") < pl.col("i_runout") - 3).drop("i_runout"))
    br = add_rev_accel(frames, br)
    lateral = lambda c: ((pl.col(c).abs() - 90).abs() < 35)  # raw field coords: drill lateral axis is y
    return br.with_columns(lateral=lateral("ent_ang") & lateral("ex_ang"),
                           cut=pl.col("session") + "_" + pl.col("k").cast(pl.Utf8))


REV_METRICS = ["rev_accel", "peak_decel", "peak_reaccel", "rev_time", "entry_speed", "exit_speed"]


def combine_reversal_features(rev: pl.DataFrame, min_rev: int = 2) -> pl.DataFrame:
    """Per player: mean of each reversal metric centered by cut (session x reversal index), per drill (wide), plus the
    mirror+wave composite of rev_accel (mean of within-drill z-scores, either drill if only one)."""
    c = rev.with_columns([(pl.col(m) - pl.col(m).mean().over("cut")).alias(m) for m in REV_METRICS])
    pm = (c.group_by("nfl_id", "drill").agg(n_rev=pl.len(), *[pl.col(m).mean() for m in REV_METRICS])
          .filter(pl.col("n_rev") >= min_rev))
    wide = pm.pivot("drill", index="nfl_id", values=["n_rev", *REV_METRICS])
    wide = wide.rename({c: f"cmb_{c.lower()}" for c in wide.columns if c != "nfl_id"})
    lat = (c.filter((pl.col("drill") == "WAVE") & pl.col("lateral")).group_by("nfl_id")
           .agg(n=pl.len(), cmb_rev_accel_wave_lat=pl.col("rev_accel").mean()).filter(pl.col("n") >= min_rev).drop("n"))
    z = lambda c: (pl.col(c) - pl.col(c).mean()) / pl.col(c).std()
    return wide.join(lat, on="nfl_id", how="left").with_columns(
        cmb_rev_accel_composite=pl.mean_horizontal(z("cmb_rev_accel_mirror"), z("cmb_rev_accel_wave")))


def drop_kick_metrics(frames: pl.DataFrame) -> pl.DataFrame:
    """PASS_RUSH_DROP kick phase per attempt (canonical frame): peak speed in the first 1.2 s after
    onset, peak accel around onset, peak brake in the 1 s after the kick peak, final sprint max speed."""
    c = kinematics.add_kinematics(segments.canonicalize(frames.drop(kinematics.KIN_COLS)), ["traj"])
    rows = []
    for (t,), g in c.group_by("traj", maintain_order=True):
        sp, at = g["speed"].to_numpy(), g["a_tan"].to_numpy()
        i0, n = int(np.flatnonzero(g["onset"].to_numpy())[0]), len(sp)
        ipk = i0 + int(np.argmax(sp[i0:min(n, i0 + 13)]))
        rows.append(dict(traj=t, nfl_id=g["nfl_id"][0], attempt=g["attempt"][0], session=g["session"][0],
                         kick_speed=float(sp[ipk]), kick_accel=float(np.max(at[max(0, i0 - 3):i0 + 7])),
                         kick_brake=float(-np.min(at[ipk:min(n, ipk + 11)])), sprint_max=float(sp.max())))
    return pl.DataFrame(rows)


# ---------------------------------------------------------------- Game

def pass_block_plays() -> pl.DataFrame:
    """REG-season pass-blocking snaps of study OL (T/G/C, >= 1 rusher encountered), with season, team
    and a play-action flag (tracking event 'play_action' on that player-play)."""
    reg = io.load("games").filter(pl.col("season_type") == "REG").select("game_id", "season")
    pp = (io.load("player_play").join(reg, on="game_id").join(ol_ids(), on="nfl_id", how="semi")
          .filter(pl.col("pressure_allowed").is_not_null() & (pl.col("pass_rushers_encountered") >= 1)
                  & pl.col("lined_up_position").is_in(OL_POS)))
    pa = (features.reg_season_tracking().join(pp.select(K).lazy(), on=K, how="semi")
          .group_by(K).agg(pa=(pl.col("event") == "play_action").any()).collect())
    return (pp.join(pa, on=K, how="left")
            .with_columns(pl.col("pa").fill_null(False), rush2=(pl.col("pass_rushers_encountered") >= 2).cast(pl.Int8),
                          pressure=pl.col("pressure_allowed").cast(pl.Float64),
                          ppp=pl.col("peak_pressure_probability_allowed"))
            .select(*K, "season", "team_abbr", "lined_up_position", "pa", "rush2", "dropback_duration",
                    "pressure", "ppp", "sack_allowed", "time_to_pressure_allowed"))


def pass_set_frames(plays: pl.DataFrame) -> pl.DataFrame:
    """Snap-aligned frames in GAME_WINDOW with kinematics, for the given player-plays."""
    tr = features.reg_season_tracking().join(plays.select(K).lazy(), on=K, how="semi")
    al = segments.snap_aligned(tr).filter(pl.col("t_rel").is_between(*GAME_WINDOW)).collect()
    al = kinematics.add_kinematics(al, K)
    return al.with_columns(tk=(pl.col("t_rel") * 10).round().cast(pl.Int32))


SET_METRICS = ["set_accel", "set_speed", "depth10", "getoff", "set_brake", "lat10"]


def pass_set_metrics(frames: pl.DataFrame, plays: pl.DataFrame) -> pl.DataFrame:
    """Per play (non-play-action, dropback >= MIN_DROPBACK_SET):
      set_accel = peak a_tan in [0, 0.6] s (first-step burst into the set, yd/s^2)
      set_speed = peak speed in [0, 1.0] s;  depth10 = x(0) - x(1.0 s) (yards retreated)
      getoff    = first t >= 0 with speed >= 1 yd/s;  lat10 = |y(1.0) - y(0)|
      set_brake = peak deceleration within 1 s after the speed peak in [0, 2.0] s (anchoring)."""
    q = plays.filter(~pl.col("pa") & (pl.col("dropback_duration") >= MIN_DROPBACK_SET)).select(*K, "lined_up_position")
    f = frames.join(q, on=K, how="inner")
    at = lambda tk, c: pl.col(c).filter(pl.col("tk") == tk).first()
    win = lambda a, b: pl.col("tk").is_between(a, b)
    m = f.group_by(K).agg(
        lined_up_position=pl.col("lined_up_position").first(),
        x0=at(0, "x"), y0=at(0, "y"), x10=at(10, "x"), y10=at(10, "y"),
        set_accel=pl.col("a_tan").filter(win(0, 6)).max(),
        set_speed=pl.col("speed").filter(win(0, 10)).max(),
        getoff=pl.col("tk").filter((pl.col("tk") >= 0) & (pl.col("speed") >= 1.0)).min() / 10,
        t_pk=pl.col("tk").filter(win(0, 20)).get(pl.col("speed").filter(win(0, 20)).arg_max()),
        n_win=win(0, 20).sum(),
    ).filter(pl.col("n_win") == 21)
    brake = (f.join(m.select(*K, "t_pk"), on=K)
             .filter((pl.col("tk") >= pl.col("t_pk")) & (pl.col("tk") <= pl.col("t_pk") + 10) & (pl.col("tk") <= 25))
             .group_by(K).agg(set_brake=(-pl.col("a_tan")).max()))
    return (m.join(brake, on=K, how="left")
            .with_columns(depth10=pl.col("x0") - pl.col("x10"), lat10=(pl.col("y10") - pl.col("y0")).abs())
            .with_columns([(pl.col(c) - pl.col(c).mean().over("lined_up_position")).alias(c + "_c") for c in SET_METRICS]))


def game_set_traits(metrics: pl.DataFrame, min_plays: int = MIN_PLAYS) -> pl.DataFrame:
    """Per player: mean of position-centered per-play set metrics (a mean, so exposure-invariant)."""
    return (metrics.group_by("nfl_id").agg(set_plays=pl.len(), *[pl.col(c + "_c").mean().alias("g_" + c) for c in SET_METRICS])
            .filter(pl.col("set_plays") >= min_plays))


def _expectation_design(plays: pl.DataFrame):
    X = plays.select("dropback_duration", "lined_up_position", "rush2",
                     pl.col("pa").cast(pl.Int8), pl.col("season").cast(pl.Utf8)).to_numpy()
    ct = ColumnTransformer([("db", SplineTransformer(n_knots=6, degree=3), [0]),
                            ("cat", OneHotEncoder(drop="first"), [1, 2, 3, 4])])
    return ct.fit_transform(X)


def add_expectations(plays: pl.DataFrame) -> pl.DataFrame:
    """Play-level expectations: logistic P(pressure) and OLS E[peak pressure prob] given dropback
    duration (cubic spline), T/G/C, 2+ rushers, play action, season. Adds *_oe residuals."""
    Xd = _expectation_design(plays)
    p_hat = LogisticRegression(C=1e4, max_iter=5000).fit(Xd, plays["pressure"].to_numpy()).predict_proba(Xd)[:, 1]
    ok = plays["ppp"].is_not_null().to_numpy()
    ols = LinearRegression().fit(Xd[ok], plays["ppp"].to_numpy()[ok])
    return plays.with_columns(p_hat=pl.Series(p_hat), ppp_hat=pl.Series(ols.predict(Xd))).with_columns(
        pr_oe=pl.col("pressure") - pl.col("p_hat"), ppp_oe=pl.col("ppp") - pl.col("ppp_hat"),
        pr_var=pl.col("p_hat") * (1 - pl.col("p_hat")))


def eb_shrink(mean: np.ndarray, samp_var: np.ndarray) -> tuple[np.ndarray, float]:
    """Normal-normal empirical Bayes: tau^2 = max(0, var(mean) - mean(samp_var)) (method of moments);
    shrunk = grand + (mean - grand) * tau^2 / (tau^2 + samp_var), grand = precision-weighted mean."""
    tau2 = max(0.0, float(np.var(mean, ddof=1) - np.mean(samp_var)))
    w = 1 / (samp_var + tau2) if tau2 > 0 else 1 / samp_var
    grand = float(np.sum(w * mean) / np.sum(w))
    return grand + (mean - grand) * tau2 / (tau2 + samp_var), tau2


def outcome_traits(plays: pl.DataFrame, min_plays: int = MIN_PLAYS) -> pl.DataFrame:
    """Per player: raw and EB-shrunk pressure-allowed rate over expected (binomial sampling variance
    sum p(1-p)/n^2) and peak-pressure-probability over expected (pooled within-player residual variance / n)."""
    s2 = float(plays.filter(pl.col("ppp_oe").is_not_null())
               .select((pl.col("ppp_oe") - pl.col("ppp_oe").mean().over("nfl_id")).pow(2).mean()).item())
    t = (plays.group_by("nfl_id").agg(
            pb_plays=pl.len(), pressure_rate=pl.col("pressure").mean(), pr_oe=pl.col("pr_oe").mean(),
            pr_var=pl.col("pr_var").sum() / pl.len() ** 2, ppp_oe=pl.col("ppp_oe").mean(),
            ppp_n=pl.col("ppp_oe").is_not_null().sum(), sack_rate=pl.col("sack_allowed").mean(),
            ttp=pl.col("time_to_pressure_allowed").mean(), n_pressures=pl.col("pressure").sum())
         .filter(pl.col("pb_plays") >= min_plays))
    pr_eb, tau_pr = eb_shrink(t["pr_oe"].to_numpy(), t["pr_var"].to_numpy())
    ppp_eb, tau_ppp = eb_shrink(t["ppp_oe"].to_numpy(), s2 / t["ppp_n"].to_numpy())
    return t.with_columns(pr_oe_eb=pl.Series(pr_eb), ppp_oe_eb=pl.Series(ppp_eb),
                          eb_tau2_pr=pl.lit(tau_pr), eb_tau2_ppp=pl.lit(tau_ppp))
