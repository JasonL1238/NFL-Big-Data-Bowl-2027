"""Probe B (E003): DL first-step burst / get-off. Combine drill burst vs regular-season burst and pass rush.

Burst = how fast a player builds speed AFTER he starts moving (motion onset). The Combine has no snap
stimulus, so Combine burst is onset-aligned; game burst is computed both onset-aligned (same definition)
and snap-aligned (reaction + burst, like NGS get-off).

Onset = interpolated time the smoothed speed first crosses ONSET_SPEED (1 yd/s), at the start of a run
of >= ONSET_HOLD frames at or above it. A trajectory whose first frame is already >= 1 yd/s has an
unobserved onset and gets no onset-aligned features.

Burst features (all relative to onset, speeds yd/s, 10 Hz, Savitzky-Golay kinematics):
  d05, d10   distance (integral of speed) in the first 0.5 / 1.0 s
  s05, s10   speed 0.5 / 1.0 s after onset
  amax06     peak tangential acceleration in the first 0.6 s
  t3, t5     time from onset to 3 / 5 yd/s (NaN if not reached within 2 s)

Game semantics (verified 2026-10-14 on DL REG plays, see E003 script):
  - pass-rush snap = blitzing == True on a dropback with defensive pass-rush fields (sack non-null),
    not nullified by penalty. `blitzing` is NGS "rushed the passer" (True on 94.7% of DL dropbacks).
  - pressure = time_to_pressure non-null (also implies time_to_qb_hurry non-null); quick_pressure is
    defined only on pressures and is True iff time_to_pressure <= 2.4 s.
  - player_get_off is in seconds (lower = faster), present on run plays too; it tracks the time from
    the snap until the player is ~1.3 yd upfield of his stance (r = 0.90 with that tracking time).
  - standardized game x: offense moves to +x, so a DL rushes toward -x; LOS = 10 + yardline_number if
    yardline_side == possession_team else 110 - yardline_number (DL stance median 0.6 yd past it).
"""

import numpy as np
import polars as pl

from bdb27 import features, io, kinematics, segments

DT = kinematics.DT
ONSET_SPEED = 1.0
ONSET_HOLD = 3
SEARCH_S = 2.0  # how far after onset to look for t3/t5

DL_DRILLS = ["FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION", "PASS_RUSH_DRILL", "RUN_AND_CLUB_DRILL",
             "FOUR_BAG_AGILITY_DRILL", "BODY_CONTROL_DRILL", "RUN_THE_HOOP_DRILL"]
FORTY = "FORTY_YARD_DASH"
BURST_FEATS = ["d05", "d10", "s05", "s10", "amax06", "t3"]
ALIGNMENTS = ["EDGE", "INTERIOR_LINE"]


# ---------- per-trajectory burst features ----------

def onset_time(speed: np.ndarray, start: int = 0) -> float:
    """Fractional frame index of motion onset at or after `start` (NaN if unobserved or never)."""
    if start >= len(speed) or not speed[start] < ONSET_SPEED:
        return np.nan
    above = speed[start:] >= ONSET_SPEED
    run = np.convolve(above, np.ones(ONSET_HOLD, dtype=int), mode="valid") == ONSET_HOLD
    idx = np.flatnonzero(run)
    if not len(idx):
        return np.nan
    i = start + int(idx[0])  # first frame of the run; speed[i-1] < ONSET_SPEED <= speed[i]
    s0, s1 = speed[i - 1], speed[i]
    return (i - 1) + (ONSET_SPEED - s0) / (s1 - s0)


def _cross_time(t: np.ndarray, speed: np.ndarray, t0: float, level: float, horizon: float) -> float:
    """Interpolated time after t0 that speed first reaches `level` (NaN if not within horizon)."""
    m = (t >= t0) & (t <= t0 + horizon)
    ti, si = t[m], speed[m]
    hit = np.flatnonzero(si >= level)
    if not len(hit):
        return np.nan
    k = hit[0]
    if k == 0:
        return 0.0
    return ti[k - 1] + (level - si[k - 1]) / (si[k] - si[k - 1]) * DT - t0


def burst_features(speed: np.ndarray, a_tan: np.ndarray, t_on: float, with_t5: bool = False) -> dict:
    """Onset-aligned burst features. `t_on` is a fractional frame index (from onset_time)."""
    out = {k: np.nan for k in BURST_FEATS + (["t5"] if with_t5 else [])}
    n = len(speed)
    if not np.isfinite(t_on) or t_on + 1.0 / DT > n - 1:
        return out
    t = np.arange(n) * DT
    t0 = t_on * DT
    fine = t0 + np.linspace(0, 1.0, 101)
    v = np.interp(fine, t, speed)
    out["d10"] = float(np.trapezoid(v, fine))
    out["d05"] = float(np.trapezoid(v[:51], fine[:51]))
    out["s05"], out["s10"] = float(v[50]), float(v[100])
    w = (t >= t0 - 1e-9) & (t <= t0 + 0.6 + 1e-9)
    out["amax06"] = float(np.max(a_tan[w]))
    out["t3"] = _cross_time(t, speed, t0, 3.0, SEARCH_S)
    if with_t5:
        out["t5"] = _cross_time(t, speed, t0, 5.0, SEARCH_S)
    return out


# ---------- Combine ----------

def combine_frames(nfl_ids: list[int]) -> pl.DataFrame:
    """DL drill + 40 attempts for the given players, with kinematics (traj = one attempt)."""
    att = segments.attempts(drill_names=DL_DRILLS + [FORTY])
    att = att.filter(pl.col("nfl_id").is_in(nfl_ids)).drop("traj")
    return kinematics.add_kinematics(att, segments.ATTEMPT_KEYS)


def combine_attempt_features(frames: pl.DataFrame) -> pl.DataFrame:
    """One row per attempt (traj): onset frame + burst features. Rows with unobserved onset have NaNs."""
    frames = frames.sort("traj", "time")
    traj = frames["traj"].to_numpy()
    speed, a_tan = frames["speed"].to_numpy(), frames["a_tan"].to_numpy()
    drill = frames["drill_name"].to_numpy()
    ids = frames.select("traj", *segments.ATTEMPT_KEYS, "draft_year").unique("traj", keep="first")
    bounds = np.flatnonzero(np.diff(traj)) + 1
    rows = []
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(frames)]):
        t_on = onset_time(speed[a:b])
        is_forty = drill[a] == FORTY
        f = burst_features(speed[a:b], a_tan[a:b], t_on, with_t5=True)
        if not is_forty:
            f["t5"] = np.nan
        rows.append({"traj": int(traj[a]), "t_on": t_on * DT if np.isfinite(t_on) else np.nan,
                     "n_frames": int(b - a), "speed0": float(speed[a]), **f})
    out = pl.DataFrame(rows, infer_schema_length=None)
    return out.with_columns(pl.col(pl.Float64).fill_nan(None)).join(ids, on="traj", how="left")


def combine_player_drill(att: pl.DataFrame, feats: list[str] = BURST_FEATS + ["t5"]) -> pl.DataFrame:
    """Per player x drill: mean over attempts, plus within-drill z-scores (z_<feat>)."""
    pdm = att.group_by("nfl_id", "drill_name").agg(
        [pl.col(f).filter(pl.col(f).is_finite()).mean().alias(f) for f in feats], n_att=pl.len())
    return pdm.with_columns(
        [((pl.col(f) - pl.col(f).mean().over("drill_name")) / pl.col(f).std().over("drill_name")).alias(f"z_{f}")
         for f in feats])


def composite(pdm: pl.DataFrame, feat: str, drills: list[str], sign: float = 1.0, name: str | None = None,
              min_drills: int = 1) -> pl.DataFrame:
    """Per player: mean of within-drill z-scores of `feat` across `drills` (sign flips e.g. times)."""
    name = name or f"cb_{feat}"
    return (pdm.filter(pl.col("drill_name").is_in(drills) & pl.col(f"z_{feat}").is_finite())
            .group_by("nfl_id").agg((sign * pl.col(f"z_{feat}")).mean().alias(name), pl.len().alias(f"{name}_k"))
            .filter(pl.col(f"{name}_k") >= min_drills))


# ---------- Game ----------

def dl_play_table(nfl_ids: list[int]) -> pl.DataFrame:
    """REG player_play rows for the given players, with play-type flags and LOS (standardized x)."""
    reg = io.load("games").filter(pl.col("season_type") == "REG").select("game_id")
    pp = io.load("player_play").filter(pl.col("nfl_id").is_in(nfl_ids)).join(reg, on="game_id", how="semi")
    ok = pl.col("play_nullified_by_penalty") == "N"
    return pp.select(
        *segments.PLAY_KEYS, "lined_up_position", "player_get_off", "time_to_pressure", "quick_pressure",
        "sack", "down", "yards_to_go",
        rush=ok & (pl.col("blitzing") == True) & pl.col("sack").is_not_null(),  # noqa: E712
        run=ok & pl.col("pass_result").is_null() & pl.col("sack").is_null() & pl.col("blitzing").is_null(),
        pressure=pl.col("time_to_pressure").is_not_null(),
        quick=pl.col("quick_pressure").fill_null(False),
        sacked=pl.col("sack").fill_null(0) > 0,
        los=pl.when(pl.col("yardline_side") == pl.col("possession_team")).then(10 + pl.col("yardline_number"))
        .otherwise(110 - pl.col("yardline_number")).cast(pl.Float64),
    )


def game_frames(nfl_ids: list[int], t_min: float = -1.0, t_max: float = 3.0) -> pl.DataFrame:
    """Snap-aligned REG tracking for the given players in [t_min, t_max] s, with kinematics."""
    lf = features.reg_season_tracking().filter(pl.col("nfl_id").is_in(nfl_ids))
    al = segments.snap_aligned(lf).filter(pl.col("t_rel").is_between(t_min, t_max)).collect()
    return kinematics.add_kinematics(al, segments.PLAY_KEYS)


UPFIELD_YD = 1.3  # player_get_off best matches the time to get this far upfield (-x) of the snap position


def _time_upfield(t_rel: np.ndarray, x: np.ndarray, i_snap: int, yd: float) -> float:
    """Interpolated t_rel at which x first drops `yd` below its snap value (DL rush toward -x)."""
    d = x[i_snap] - x[i_snap:]
    hit = np.flatnonzero(d >= yd)
    if not len(hit) or hit[0] == 0:
        return np.nan
    k = i_snap + hit[0]
    d0, d1 = x[i_snap] - x[k - 1], x[i_snap] - x[k]
    return float(t_rel[k - 1] + (yd - d0) / (d1 - d0) * DT)


def game_play_features(frames: pl.DataFrame, presnap_check: float = -0.5) -> pl.DataFrame:
    """One row per player-play: snap-aligned and onset-aligned burst features.
    Onset search starts at t_rel = presnap_check and requires the player to be set there (< 1 yd/s);
    t_move = onset time relative to the snap. Snap-aligned features: <feat>_snap.
    t_up13 = time from snap to UPFIELD_YD upfield of the snap position (a tracking analogue of get-off)."""
    frames = frames.sort("traj", "time")
    traj = frames["traj"].to_numpy()
    speed, a_tan = frames["speed"].to_numpy(), frames["a_tan"].to_numpy()
    trel, x = frames["t_rel"].to_numpy(), frames["x"].to_numpy()
    ids = frames.select("traj", *segments.PLAY_KEYS).unique("traj", keep="first")
    bounds = np.flatnonzero(np.diff(traj)) + 1
    rows = []
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(frames)]):
        sp, at, tr = speed[a:b], a_tan[a:b], np.round(trel[a:b], 3)
        i_chk = np.flatnonzero(np.isclose(tr, presnap_check))
        i_snap = np.flatnonzero(np.isclose(tr, 0.0))
        row = {"traj": int(traj[a]), "t_last": float(tr[-1])}
        if len(i_chk) and len(i_snap):
            t_on = onset_time(sp, int(i_chk[0]))
            row["t_move"] = float(tr[0] + t_on * DT) if np.isfinite(t_on) else np.nan
            row.update(burst_features(sp, at, t_on))
            # snap-aligned: same features with "onset" fixed at the snap frame
            snap = burst_features(sp, at, float(i_snap[0]))
            row.update({f"{k}_snap": v for k, v in snap.items()})
            row["x_snap"] = float(x[a + i_snap[0]])
            row["t_up13"] = _time_upfield(tr, x[a:b], int(i_snap[0]), UPFIELD_YD)
        rows.append(row)
    out = pl.DataFrame(rows, infer_schema_length=None)
    return out.with_columns(pl.col(pl.Float64).fill_nan(None)).join(ids, on="traj", how="left")


# ---------- player-level traits with empirical-Bayes shrinkage ----------

def eb_mean(df: pl.DataFrame, value: str, player: str = "nfl_id") -> pl.DataFrame:
    """Normal-normal EB shrinkage of per-player means toward the grand mean of player means.
    tau^2 (between-player variance) by method of moments: var(player means) - mean(sigma_w^2 / n_i)."""
    g = (df.filter(pl.col(value).is_finite()).group_by(player)
         .agg(m=pl.col(value).mean(), v=pl.col(value).var(), n=pl.len()).filter(pl.col("n") >= 2))
    sw2 = float(np.average(g["v"].to_numpy(), weights=g["n"].to_numpy() - 1))  # pooled within variance
    m, n = g["m"].to_numpy(), g["n"].to_numpy()
    mu = float(np.average(m, weights=n))
    tau2 = max(float(np.var(m, ddof=1) - np.mean(sw2 / n)), 1e-12)
    w = tau2 / (tau2 + sw2 / n)
    return g.select(player, pl.Series(value, mu + w * (m - mu)), pl.Series(f"{value}_n", n),
                    pl.Series(f"{value}_w", w))


def eb_rate(df: pl.DataFrame, event: str, expected: str, player: str = "nfl_id") -> pl.DataFrame:
    """Beta-binomial EB rate over expectation. Per play, `expected` is the alignment base rate.
    Player prior mean = mean expected rate over his snaps; prior strength kappa by method of moments on
    the ratio k/E (variance beyond binomial). Returns shrunk rate, its expectation and rate - expectation."""
    g = df.group_by(player).agg(k=pl.col(event).cast(pl.Float64).sum(), n=pl.len(),
                                e=pl.col(expected).mean())
    k, n, e = g["k"].to_numpy(), g["n"].to_numpy(), g["e"].to_numpy()
    r = k / n
    # between-player variance of true rates around their expectation (moment estimate)
    resid_var = np.average((r - e) ** 2, weights=n) - np.average(e * (1 - e) / n, weights=n)
    pbar = np.average(e, weights=n)
    tau2 = max(resid_var, 1e-8)
    kappa = max(pbar * (1 - pbar) / tau2 - 1, 1.0)
    shrunk = (k + kappa * e) / (n + kappa)
    return g.select(player, pl.Series(f"{event}_rate", shrunk), pl.Series(f"{event}_exp", e),
                    pl.Series(f"{event}_oe", shrunk - e), pl.Series(f"{event}_n", n),
                    pl.Series(f"{event}_kappa", np.full(len(k), kappa)))


def center_by_alignment(plays: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Subtract the alignment (EDGE / INTERIOR_LINE) mean from each per-play value, add back the overall mean."""
    return plays.with_columns(
        [(pl.col(c) - pl.col(c).mean().over("lined_up_position") + pl.col(c).mean()).alias(f"{c}_adj")
         for c in cols])


KIN_TRAITS = BURST_FEATS + [f"{k}_snap" for k in BURST_FEATS] + ["t_move", "t_up13", "player_get_off"]
RATE_EVENTS = ["pressure", "quick", "sacked"]


def eligible_plays(plays: pl.DataFrame, min_snaps: int = 30) -> pl.DataFrame:
    """EDGE/INTERIOR_LINE plays of players with >= min_snaps of them; adds <trait>_adj (alignment-centered)
    for KIN_TRAITS and <event>_adj = event - alignment base rate for RATE_EVENTS."""
    p = plays.filter(pl.col("lined_up_position").is_in(ALIGNMENTS))
    p = p.filter(pl.len().over("nfl_id") >= min_snaps)
    p = center_by_alignment(p, KIN_TRAITS)
    return p.with_columns([(pl.col(ev).cast(pl.Float64) - pl.col(ev).cast(pl.Float64).mean().over("lined_up_position"))
                           .alias(f"{ev}_adj") for ev in RATE_EVENTS])


def game_traits(plays: pl.DataFrame, suffix: str, min_snaps: int = 30) -> pl.DataFrame:
    """Player-level game traits from a set of player-plays (e.g. pass-rush snaps), EDGE/INTERIOR only.
    Kinematic traits: alignment-centered per-play values -> EB-shrunk player mean (g_<feat>_<suffix>).
    Rates: beta-binomial EB with per-play alignment base rates as expectation
    (g_<event>_rate_<suffix>, g_<event>_oe_<suffix> = shrunk rate - expectation).
    Only players with >= min_snaps plays are kept; also returns snap count and EDGE share."""
    p = eligible_plays(plays, min_snaps)
    out = p.group_by("nfl_id").agg(pl.len().alias(f"n_{suffix}"),
                                   (pl.col("lined_up_position") == "EDGE").mean().alias(f"edge_share_{suffix}"))
    for c in KIN_TRAITS:
        t = eb_mean(p, f"{c}_adj").select("nfl_id", pl.col(f"{c}_adj").alias(f"g_{c}_{suffix}"))
        out = out.join(t, on="nfl_id", how="left")
    for ev in RATE_EVENTS:
        q = p.with_columns(_e=pl.col(ev).cast(pl.Float64).mean().over("lined_up_position"))
        t = eb_rate(q, ev, "_e").select("nfl_id", pl.col(f"{ev}_rate").alias(f"g_{ev}_rate_{suffix}"),
                                          pl.col(f"{ev}_oe").alias(f"g_{ev}_oe_{suffix}"))
        out = out.join(t, on="nfl_id", how="left")
    return out.sort("nfl_id")
