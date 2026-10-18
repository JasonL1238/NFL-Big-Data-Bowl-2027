"""E011 - Combine ball tracking at the catch point: feasibility (DESCRIPTIVE; 2025 class only).

BALL rows exist only for the 2025 class (37 WR, 17 TE; 551 attempts in 18 drill names), whose players have one
rookie season. Any link to games is exploratory and badly underpowered, so the plan (copied verbatim into the JSON as
`plan`, written 2026-10-15T03:59:56Z before any mechanic, reliability or association was computed) gates every game
test on the mechanic's reliability:
  1. understand the ball data: coverage, alignment with the receiver's attempt, release, contact, catch vs not
     (inferred, validated on plots and on an unanchored geometric closest-approach check of the contact time);
  2. catch-point mechanics (bdb27.probes.g_catch_point): retention, decel_in, catch_offset, adjust_dist,
     adjust_time; cross-drill reliability (z within drill x Combine session) with player-bootstrap CIs;
  3. descriptive relation of the player mechanics to the 40 and other Combine measures (Spearman, Fisher CIs);
  4. at most 3 prelisted exploratory linkage.evaluate tests against rookie receiving outcomes, ONLY for mechanics
     with reliability >= 0.5 (power stated up front; reduced controls because the full B2 is degenerate at n ~ 24);
     target-side exposure, reliability and power are reported whether or not the gate passes, with no
     Combine-game association computed when it fails.
Post hoc robustness (labeled): WR only; retention / decel_in window 0.3 and 0.7 s; drill-centered only; the inferred
'separated' rate as a per-player value.
Revision after review (2026-10-15; all POST HOC, labeled in the JSON under `revision`; no preregistered definition or
test changed): the contact-time check is now unanchored (the first version was bounded to [-0.05, +0.15] s by
construction and is withdrawn); a post hoc outcome class for balls that passed him in flight and for ball rows of
another throw; retest ICCs and Spearman-Brown 'drills needed' with bootstrap CIs; an offset-free (velocity-matching)
clock-offset check; adjust_dist decomposed; dropped_pass attribution counted; ball-row coverage by year and drill.

Run: OMP_NUM_THREADS=3 .venv/bin/python experiments/E011_catch_point.py
     (1-8 min depending on load: 3 min on 2026-10-15 under shared load; peak ~0.7 GB; deterministic up to
     floating-point summation order, i.e. last-digit differences in a few split-half values)
Outputs: experiments/results/E011.json, experiments/figures/E011_{qc_contact,qc_outcomes,reliability}.png,
         data/processed/e011_{attempts,players}.parquet
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from bdb27 import io, kinematics, linkage, reliability, segments, speed
from bdb27.probes import g_catch_point as G

ROOT = Path(__file__).resolve().parents[1]
RES, FIG, P = ROOT / "experiments" / "results", ROOT / "experiments" / "figures", io.PROCESSED
PLAN = RES / "E011_prereg.json"
GATE = 0.5
N_BOOT = 2000
SEED = 0
COMBINE_MEASURES = ["forty", "ten_yd_split", "c40_v_20_40", "c40_smax", "vertical", "broad_jump", "three_cone",
                    "short_shuttle", "combine_weight", "combine_height", "hand_size", "arm_length"]
LABELS = {"retention": "speed retention", "decel_in": "decel into catch", "catch_offset": "catch offset",
          "adjust_dist": "adjust distance", "adjust_time": "adjust time"}
T0 = time.time()
REVISION = {
    "date": "2026-10-15", "label": "post hoc, after review; no preregistered definition, gate or test changed",
    "changes": [
        "detection.contact_time_check: unanchored closest approach replaces the first check, which was bounded to "
        "[-0.05, +0.15] s around t_contact by construction (withdrawn: its '91% within 0.1 s' compared with ~75% for a "
        "uniform time in that window)",
        "detection.outcome_posthoc / attempts.outcome_posthoc: not_reached split into passed_him (ball passed within "
        "1.5 yd in flight, >= 0.15 s before its first impact), ball_unrelated (ball never within 5 yd: rows of another "
        "throw) and not_reached; preregistered `outcome` unchanged; mechanics unchanged (all three are not_reached)",
        "detection.ball_speed_outside_flight: replaces 'handling and carrying stay <= 10 yd/s'",
        "reliability.*.within_drill_retest_icc_boot_ci95 and robustness_post_hoc.retest_cells: retest ICCs with CIs; "
        "repeats often follow a failed first rep",
        "reliability.*.cross_drill_icc_boot_ci95 and drills_needed_range_from_icc_ci: Spearman-Brown as ranges, in "
        "distinct drills",
        "clock_lag_check.within_attempt_slope_s and velocity_matching: the pooled slope does not identify a lag; "
        "offset-free estimates in two windows and two edge trims; robustness_post_hoc.clock_shift_*: mechanic "
        "reliability with the receiver read +-0.1 / 0.15 s off the ball's contact time",
        "mechanics.adjust_dist.decomposition: adjust_dist is mostly run-speed change in the two fades and route "
        "curvature in CIRCUS; mixed in OVER_SHOULDER_ADJUST",
        "rookie_outcomes.dropped_pass_attribution_54: corrected count of non-target dropped_pass player-plays",
        "data.coverage: BALL-row coverage by class, drill type and 2025 WR/TE drill name"]}


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def fnum(x):
    """JSON-safe float (None for NaN / inf / None)."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return fnum(o)
    return o


# =====================================================================================================
# 1. The ball data
# =====================================================================================================

def data_facts(frames: pl.DataFrame) -> dict:
    """Coverage of BALL rows and their alignment with the receiver's frames."""
    ct = io.scan_combine_tracking()
    ball = ct.filter(pl.col("entity_type") == "BALL").collect()
    rec_t = ct.filter(pl.col("entity_type") == "PLAYER").select("event_id", "nfl_id", "drill_name", "attempt", "time")
    matched = ball.select("event_id", "nfl_id", "drill_name", "attempt", "time").join(
        rec_t.collect(), on=["event_id", "nfl_id", "drill_name", "attempt", "time"], how="semi").height
    att = frames.group_by("traj").agg(pl.col("drill_name").first(), pl.col("position").first(),
                                       has=pl.col("bx").is_not_null().sum(), n=pl.len(),
                                       lead=pl.col("t").filter(pl.col("bx").is_not_null()).min())
    # Coverage (post hoc, for the writeup): share of Combine attempts (reps) with BALL rows, by class and drill type,
    # and by drill name for the 2025 WR/TE position drills.
    rep = (ct.group_by("draft_year", "drill_type", *segments.ATTEMPT_KEYS)
           .agg(ball=(pl.col("entity_type") == "BALL").any()).collect())
    by_type = (rep.group_by("draft_year", "drill_type").agg(attempts=pl.len(), with_ball=pl.col("ball").sum())
               .sort("draft_year", "drill_type").with_columns(share=pl.col("with_ball") / pl.col("attempts")))
    wrte25 = (rep.filter((pl.col("draft_year") == 2025) & pl.col("drill_type").is_in(G.BALL_DRILL_TYPES))
              .group_by("drill_type", "drill_name").agg(attempts=pl.len(), with_ball=pl.col("ball").sum())
              .sort("drill_type", "drill_name"))
    pos = rep.filter(pl.col("drill_type").str.starts_with("SKILL_DRILLS"))
    coverage = {
        "label": "post hoc (after review): coverage of BALL rows, for the writeup",
        "by_class_and_drill_type": by_type.to_dicts(),
        "position_drill_reps_all_classes": {"attempts": pos.height, "with_ball": int(pos["ball"].sum())},
        "position_drill_reps_2025": {"attempts": int(pos.filter(pl.col("draft_year") == 2025).height),
                                     "with_ball": int(pos.filter(pl.col("draft_year") == 2025)["ball"].sum())},
        "wr_te_2025_by_drill_name": wrte25.to_dicts(),
        "wr_te_2025_distinct_drill_names_with_ball": wrte25.filter(pl.col("with_ball") > 0)["drill_name"].n_unique(),
        "wr_te_2025_distinct_drill_names": wrte25["drill_name"].n_unique(),
        "wr_te_2025_reps_in_drills_with_ball": {
            "attempts": int(wrte25.filter(pl.col("with_ball") > 0)["attempts"].sum()),
            "with_ball": int(wrte25["with_ball"].sum())}}
    return {
        "ball_rows": ball.height, "draft_years": sorted(ball["draft_year"].unique().to_list()),
        "attempts": att.height, "players": frames["nfl_id"].n_unique(),
        "players_by_position": dict(frames.group_by("position").agg(pl.col("nfl_id").n_unique()).iter_rows()),
        "drill_names": int(att["drill_name"].n_unique()),
        "attempts_by_drill": dict(att.group_by("drill_name").len().sort("drill_name").iter_rows()),
        "sessions": sorted(frames["session"].unique().to_list()),
        "ball_rows_matching_a_receiver_frame_by_timestamp": matched,
        "receiver_frames_with_ball_share": float(att["has"].sum() / att["n"].sum()),
        "ball_track_start_after_receiver_s": {"median": fnum(att["lead"].median()), "max": fnum(att["lead"].max())},
        "columns": "x, y, s, a, dis, dir only: no ball height (z)",
        "coverage": coverage,
    }


LAG_TAUS = np.round(np.arange(-0.30, 0.3001, 0.01), 2)
LAG_WINDOWS = [(0.3, 1.5), (0.6, 2.0)]  # s after contact (post hoc, after review)
CLOCK_SHIFTS = (-0.15, -0.1, 0.1, 0.15)  # s, receiver clock vs ball clock (post hoc sensitivity)


def velocity_offset(frames: pl.DataFrame, att: pl.DataFrame, window: tuple[float, float], trim: int = 3,
                    n_boot: int = 1000) -> dict:
    """Offset-free clock check (post hoc, after review): over secured catches, the carried ball's SG velocity
    (kinematics.kinematics on its x/y within `window` s after contact; `trim` edge frames dropped) is compared with the
    receiver's SG velocity at time + tau. The tau minimizing the mean (over attempts) squared velocity difference
    is the best match: tau > 0 means the ball's record leads the receiver's. A constant carry offset cancels in
    velocities, but ball handling (tucking) does not, so the estimate depends on the window."""
    sec = att.filter(pl.col("outcome") == "secured").select("traj", "t_contact")
    E = []
    for g in frames.join(sec, on="traj").partition_by("traj", maintain_order=True):
        t, tc = g["t"].to_numpy(), g["t_contact"][0]
        bx, by = g["bx"].to_numpy(), g["by"].to_numpy()
        idx = np.flatnonzero((t >= tc + window[0] - 1e-9) & (t <= tc + window[1] + 1e-9) & np.isfinite(bx))
        if len(idx) < 2 * trim + 5 or np.any(np.diff(idx) != 1):
            continue
        kb = kinematics.kinematics(bx[idx], by[idx])
        tb, bvx, bvy = t[idx][trim:-trim], kb["vx"][trim:-trim], kb["vy"][trim:-trim]
        vx, vy = g["vx"].to_numpy(), g["vy"].to_numpy()
        e = []
        for tau in LAG_TAUS:
            tq = tb + tau
            ok = (tq >= t[0]) & (tq <= t[-1])
            d2 = (bvx - np.interp(tq, t, vx)) ** 2 + (bvy - np.interp(tq, t, vy)) ** 2
            e.append(float(np.mean(d2[ok])) if ok.any() else np.nan)
        E.append(e)
    E = np.array(E)
    best = float(LAG_TAUS[np.nanargmin(np.nanmean(E, 0))])
    rng = np.random.default_rng(SEED)
    bt = [LAG_TAUS[np.nanargmin(np.nanmean(E[rng.integers(0, len(E), len(E))], 0))] for _ in range(n_boot)]
    return {"window_s_after_contact": list(window), "trim_frames": trim, "n_attempts": len(E), "best_tau_s": best,
            "ci95_attempt_bootstrap": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
            "per_attempt_best_tau_median": float(np.median(LAG_TAUS[np.nanargmin(E, 1)]))}


def lag_check(frames: pl.DataFrame, att: pl.DataFrame, n_boot: int = 500) -> dict:
    """Carried ball (secured catches, 0.3-1.0 s after contact): offset of the ball from the receiver's tracked point
    along his velocity, regressed on his speed. A ball clock lagging by tau seconds gives slope -tau, but ball
    handling also moves the ball along his body as his speed changes, so the slope does not identify a lag (the
    within-attempt slope shows it). Post hoc: an offset-free velocity-matching check in two windows."""
    sec = att.filter(pl.col("outcome") == "secured").select("traj", "t_contact")
    f = frames.join(sec, on="traj").filter((pl.col("t") >= pl.col("t_contact") + 0.3)
                                           & (pl.col("t") <= pl.col("t_contact") + 1.0) & (pl.col("speed") >= 1))
    f = f.with_columns(lon=((pl.col("bx") - pl.col("x")) * pl.col("vx") + (pl.col("by") - pl.col("y")) * pl.col("vy"))
                       / pl.col("speed")).filter(pl.col("lon").is_finite())
    out = {"n_frames": f.height, "n_attempts": f["traj"].n_unique(),
           "carried_distance_yd": {q: fnum(f["dist"].quantile(v)) for q, v in (("q25", .25), ("median", .5), ("q75", .75))}}
    rng, rng_w = np.random.default_rng(SEED), np.random.default_rng(SEED + 1)
    fw = f.with_columns(lon_w=pl.col("lon") - pl.col("lon").mean().over("traj"),
                        sp_w=pl.col("speed") - pl.col("speed").mean().over("traj"))
    parts_w = fw.partition_by("traj", maintain_order=True)
    bw = []
    for _ in range(n_boot):
        s = pl.concat([parts_w[i] for i in rng_w.integers(0, len(parts_w), len(parts_w))])
        bw.append(np.polyfit(s["sp_w"], s["lon_w"], 1)[0])
    out["note"] = ("slope_lon_on_speed_s[*] is the original check (a ball clock lagging by tau gives slope -tau); ball "
                   "handling also moves the ball along his body as his speed changes, so it does not identify a lag "
                   "(see within_attempt_slope_s and the offset-free velocity_matching)")
    out["within_attempt_slope_s"] = {
        "label": "post hoc (after review): lon and speed demeaned within attempt",
        "slope": float(np.polyfit(fw["sp_w"], fw["lon_w"], 1)[0]),
        "ci95": [float(np.percentile(bw, 2.5)), float(np.percentile(bw, 97.5))]}
    out["velocity_matching"] = {"label": "post hoc (after review): offset-free clock check",
                                **{f"{a}-{b}s_trim{tr}": velocity_offset(frames, att, (a, b), trim=tr)
                                   for a, b in LAG_WINDOWS for tr in (3, 2)}}
    for name, g in [("all", f), *sorted(f.partition_by("session", as_dict=True).items())]:
        if g.height < 30:
            continue
        b = float(np.polyfit(g["speed"], g["lon"], 1)[0])
        trajs = g["traj"].unique().to_numpy()
        parts = {k[0]: v for k, v in g.partition_by("traj", as_dict=True).items()}
        bs = []
        for _ in range(n_boot):
            s = pl.concat([parts[t] for t in rng.choice(trajs, len(trajs))])
            bs.append(np.polyfit(s["speed"], s["lon"], 1)[0])
        key = name if isinstance(name, str) else name[0]
        out[f"slope_lon_on_speed_s[{key}]"] = {"slope": b, "ci95": [float(np.percentile(bs, 2.5)),
                                                                    float(np.percentile(bs, 97.5))], "n": g.height}
    return out


def contact_check(att: pl.DataFrame) -> dict:
    """Unanchored contact-time check (post hoc, after review): t of the ball-receiver closest approach
    (G.flight_approach, searched from release to t_(k-1) + 0.5 s) minus t_contact, over reached attempts. Chance =
    share passing 'within 0.1 s' if t were uniform over each attempt's search window."""
    r = att.filter(pl.col("outcome").is_in(["secured", "separated", "reached_unobserved"])
                   & pl.col("t_cpa").is_not_nan())
    dt = r["t_cpa"] - r["t_contact"]
    win = (pl.min_horizontal(pl.col("t_contact") - 0.05 + 0.5, pl.col("tail") + pl.col("t_contact"))
           - pl.col("t_release"))
    lo, hi = pl.col("t_contact") - 0.1 - pl.col("t_release"), pl.col("t_contact") + 0.1 - pl.col("t_release")
    chance = r.select((pl.min_horizontal(hi, win) - pl.max_horizontal(lo, 0)).clip(lower_bound=0) / win)
    short = r["ball_along_last_flight"].drop_nulls().drop_nans()
    return {
        "label": "post hoc (after review); replaces the first version, which searched only [t_(k-1), t_(k-1) + 0.2 s] "
                 "and so was bounded to [-0.05, +0.15] s around t_contact by construction (withdrawn)",
        "what": "t of the unanchored ball-receiver closest approach minus t_contact (ball-speed drop); the tracked "
                "point is the shoulder pads, not the hands, so +-1 frame is expected",
        "n": dt.len(), "median_s": fnum(dt.median()), "q10_s": fnum(dt.quantile(.1)), "q90_s": fnum(dt.quantile(.9)),
        "share_within_0.1s": fnum((dt.abs() <= 0.1).mean()), "share_within_0.15s": fnum((dt.abs() <= 0.15).mean()),
        "share_earlier_than_-0.15s": fnum((dt < -0.15).mean()), "share_later_than_0.15s": fnum((dt > 0.15).mean()),
        "chance_share_within_0.1s": fnum(chance.to_series().mean()),
        "ball_short_of_receiver_at_last_flight_frame": {
            "what": "ball minus receiver along the ball's flight line at frame k - 1 < 0 (contact not detected late)",
            "n": short.len(), "share": fnum((short < 0).mean()), "median_yd": fnum(short.median())}}


def outcome_posthoc_summary(att: pl.DataFrame) -> dict:
    """Post hoc reclassification of not_reached (after review), and the separated share under alternative rules."""
    obs = att.filter(pl.col("outcome").is_in(["secured", "separated"]))
    n_sep, n_obs = int((obs["outcome"] == "separated").sum()), obs.height
    nr = att.filter(pl.col("outcome_posthoc").is_in(["not_reached", "passed_him"]))
    rules = {}
    for d in (1.0, G.PASS_DIST, 2.0):
        n = int(nr.filter((pl.col("d_flight_min") <= d) & (pl.col("t_flight_min_before_drop") >= G.PASS_MIN_T)).height)
        rules[f"passed_within_{d}yd"] = {"n_passed": n, "separated_share_of_observed": (n_sep + n) / (n_obs + n)}
    rest = att.filter(pl.col("outcome_posthoc") == "not_reached")
    ph = att.filter(pl.col("outcome_posthoc") == "passed_him")
    un = att.filter(pl.col("outcome_posthoc") == "ball_unrelated")
    return {
        "label": "post hoc (after review): the preregistered `outcome` is unchanged; these classes are all not_reached "
                 "and carry no mechanic",
        "rules": {"PASS_DIST": G.PASS_DIST, "PASS_MIN_T": G.PASS_MIN_T, "UNRELATED_DIST": G.UNRELATED_DIST},
        "counts": dict(att["outcome_posthoc"].value_counts().sort("outcome_posthoc").iter_rows()),
        "separated_share_of_observed_prereg": n_sep / n_obs,
        "separated_share_with_passed_him": rules,
        "passed_him": {"by_drill": dict(ph.group_by("drill_name").len().sort("drill_name").iter_rows()),
                       "in_flight_min_dist_yd_median": fnum(ph["d_flight_min"].median()),
                       "ball_past_him_at_last_flight_frame_yd_median": fnum(ph["ball_along_last_flight"].median())},
        "ball_unrelated": {"trajs": un["traj"].to_list(), "drills": un["drill_name"].to_list(),
                           "min_dist_yd": fnum(un["d_min_all"].min())},
        "reached_meeting_the_pass_rule": {
            "counts": dict(att.filter(pl.col("outcome").is_in(["secured", "separated", "reached_unobserved"])
                                      & (pl.col("d_flight_min") <= G.PASS_DIST)
                                      & (pl.col("t_flight_min_before_drop") >= G.PASS_MIN_T))
                           .group_by("outcome").len().sort("outcome").iter_rows()),
            "note": "reached attempts whose 2D path also passed his tracked point >= 0.15 s before first impact (deep "
                    "balls over the shoulder): the rule identifies misses only among not_reached"},
        "remaining_not_reached": {"n": rest.height,
                                  "ball_ends_beside_him": int((rest["post_median"] <= G.SECURE_DIST).sum()),
                                  "note": "first impact > 2 yd from his tracked point without passing close in flight "
                                          "(short or wide throws); those where the ball ends beside him may be "
                                          "extended-arm catches or pick-ups"}}


def detection_summary(att: pl.DataFrame) -> dict:
    obs = att.filter(pl.col("outcome").is_in(["secured", "separated"]))
    by_drill = (att.group_by("drill_name").agg(
        n=pl.len(), **{c: (pl.col("outcome") == c).sum() for c in
                       ("secured", "separated", "reached_unobserved", "not_reached", "no_contact", "no_flight")},
        flight_time_med=pl.col("flight_time").median(), v_release_med=pl.col("v_release").median(),
        v_contact_med=pl.col("v_contact").median(), tail_med=pl.col("tail").median()).sort("drill_name"))
    return {
        "rules": {"V_FLIGHT": G.V_FLIGHT, "MIN_FLIGHT_FRAMES": G.MIN_FLIGHT_FRAMES, "DROP_RATIO": G.DROP_RATIO,
                  "REACH_DIST": G.REACH_DIST, "SECURE_DIST": G.SECURE_DIST, "SECURE_GROWTH": G.SECURE_GROWTH},
        "outcomes": dict(att["outcome"].value_counts().sort("outcome").iter_rows()),
        "separated_share_of_observed": fnum((obs["outcome"] == "separated").mean()),
        "release_speed_yds": {"median": fnum(att["v_release"].median()), "q05": fnum(att["v_release"].quantile(.05)),
                              "q95": fnum(att["v_release"].quantile(.95))},
        "flight_time_s": {"median": fnum(att["flight_time"].median()), "min": fnum(att["flight_time"].min()),
                          "max": fnum(att["flight_time"].max())},
        "ball_speed_outside_flight": {
            "label": "post hoc (after review)",
            "pre_release_max_yds": fnum(att["bs_pre_max"].max()),
            "pre_release_q99_yds": fnum(att["bs_pre_max"].drop_nulls().drop_nans().quantile(.99)),
            "attempts_post_contact_above_10yds": int((att["bs_post_max"].drop_nulls().drop_nans() > 10).sum()),
            "attempts_with_contact": int(att["i_contact"].is_not_null().sum()),
            "attempts_with_two_flight_runs": int((att["n_flight_runs"] >= 2).sum())},
        "contact_time_check": contact_check(att),
        "outcome_posthoc": outcome_posthoc_summary(att),
        "by_drill": by_drill.to_dicts(),
    }


# =====================================================================================================
# 2. Mechanics and reliability
# =====================================================================================================

def mechanic_summary(att: pl.DataFrame) -> dict:
    out = {}
    for m in G.MECHANICS:
        d = att.filter(pl.col(m).is_finite())
        out[m] = {"n_attempts": d.height, "n_players": d["nfl_id"].n_unique(),
                  "by_drill": d.group_by("drill_name").agg(n=pl.len(), mean=pl.col(m).mean(), sd=pl.col(m).std(),
                                                            median=pl.col(m).median()).sort("drill_name").to_dicts()}
    z = att.filter(pl.col("drill_name").is_in(G.ADJUST_DRILLS) & pl.col("adjust_time").is_finite())
    out["adjust_time"]["share_zero_by_drill"] = dict(
        z.group_by("drill_name").agg((pl.col("adjust_time") == 0).mean()).sort("drill_name").iter_rows())
    a = att.filter(pl.col("adjust_dist").is_finite() & pl.col("adjust_along").is_finite())
    out["adjust_dist"]["decomposition"] = {
        "label": "post hoc (after review): what adjust_dist measures",
        "what": "along = component of (catch point - release-time extrapolation) along his release heading; "
                "share_along = median of along^2 / adjust_dist^2; corr_along_speed_change = Pearson r of along with "
                "(his speed at contact - his speed at release)",
        "by_drill": a.group_by("drill_name").agg(
            n=pl.len(), adjust_dist_median=pl.col("adjust_dist").median(),
            across_median=pl.col("adjust_across").median(),
            share_along=(pl.col("adjust_along") ** 2 / pl.col("adjust_dist") ** 2).median(),
            corr_along_speed_change=pl.corr("adjust_along", pl.col("v_contact") - pl.col("rec_v_release")))
        .sort("drill_name").to_dicts()}
    return out


def drills_needed(icc: float, target: float) -> float | None:
    """Spearman-Brown: number of drills whose mean reaches `target` reliability at single-drill ICC `icc`."""
    if not (icc and np.isfinite(icc) and icc > 0):
        return None
    return float(target * (1 - icc) / (icc * (1 - target)))


FAILED = ["separated", "not_reached"]


def reliability_block(att: pl.DataFrame, frames: pl.DataFrame) -> tuple[dict, dict]:
    rel = G.mechanic_reliability(att, n_boot=N_BOOT, seed=SEED)
    for m, v in rel.items():
        icc = v["cross_drill"]["icc"]
        v["drills_needed_for_0.5"] = drills_needed(icc, 0.5)
        v["drills_needed_for_0.8"] = drills_needed(icc, 0.8)
        if "cross_drill_icc_boot_ci95" in v:  # post hoc (after review): range from the ICC's bootstrap CI
            lo, hi = v["cross_drill_icc_boot_ci95"]
            v["drills_needed_range_from_icc_ci"] = {
                "note": "distinct drills (cross-drill ICC), not repeated catches; None = unbounded (CI reaches <= 0)",
                **{f"for_{tg}": [drills_needed(hi, tg), drills_needed(lo, tg)] for tg in (0.5, 0.8)}}
        v["passes_gate"] = bool(np.isfinite(v["cross_drill"]["reliability"]) and v["cross_drill"]["reliability"] >= GATE)
    # What the repeated attempts are (post hoc, after review): re-throws after a failed first rep?
    cell = att.with_columns(n_cell=pl.len().over("nfl_id", "drill_name")).sort("nfl_id", "drill_name", "attempt")
    first = (cell.filter(pl.col("n_cell") >= 2).group_by("nfl_id", "drill_name", maintain_order=True)
             .agg(pl.col("outcome").first()))
    single = cell.filter(pl.col("n_cell") == 1)
    retest_cells = {
        "label": "post hoc (after review)",
        "repeated_player_drill_cells": first.height,
        "first_attempt_outcomes": dict(first["outcome"].value_counts().sort("outcome").iter_rows()),
        "first_attempt_failed_share": fnum(first["outcome"].is_in(FAILED).mean()),
        "single_attempt_cells": single.height,
        "single_attempt_failed_share": fnum(single["outcome"].is_in(FAILED).mean()),
        "failed": "outcome separated or not_reached (inferred)"}
    # Post hoc robustness: WR only, window 0.3 / 0.7 s, drill-centered (already in rel), separated rate.
    rob = {"label": "post hoc robustness (not in the plan)", "retest_cells": retest_cells}
    wr = G.mechanic_reliability(att.filter(pl.col("position") == "WR"))
    rob["WR_only"] = {m: {"reliability": v["cross_drill"]["reliability"], "icc": v["cross_drill"]["icc"],
                          "n_players": v["n_players"]} for m, v in wr.items()}
    for w in (0.3, 0.7):
        aw = G.attempt_table(frames, win=w)
        rw = G.mechanic_reliability(aw)
        rob[f"window_{w}s"] = {m: {"reliability": rw[m]["cross_drill"]["reliability"], "icc": rw[m]["cross_drill"]["icc"],
                                   "n_attempts": rw[m]["n_attempts"]} for m in ("retention", "decel_in")}
    # Post hoc (after review): the ball-vs-player clock offset is not identified within about +-0.15 s
    # (clock_lag_check), so read the receiver's position / speed shifted by that much.
    for sh in CLOCK_SHIFTS:
        rw = G.mechanic_reliability(G.attempt_table(frames, clock_shift=sh))
        rob[f"clock_shift_{sh:+.2f}s"] = {m: {"reliability": rw[m]["cross_drill"]["reliability"],
                                              "icc": rw[m]["cross_drill"]["icc"]}
                                          for m in ("retention", "decel_in", "catch_offset")}
    sep = att.filter(pl.col("outcome").is_in(["secured", "separated"])).with_columns(
        sep=(pl.col("outcome") == "separated").cast(pl.Float64))
    per = sep.group_by("nfl_id").agg(n=pl.len(), k=pl.col("sep").sum())
    rob["separated_rate"] = {
        "what": "per-player share of observed outcomes inferred 'separated' (drop / deflection)",
        "attempt_icc_across_players": reliability.icc1(sep["sep"].to_numpy(), sep["nfl_id"].to_numpy()),
        "players": per.height, "median_observed_per_player": fnum(per["n"].median()),
        "players_with_any": int((per["k"] > 0).sum()), "max_per_player": int(per["k"].max())}
    sp = att.filter(pl.col("outcome_posthoc").is_in(["secured", "separated", "passed_him"])).with_columns(
        sep=(pl.col("outcome_posthoc") != "secured").cast(pl.Float64))
    pp = sp.group_by("nfl_id").agg(k=pl.col("sep").sum())
    rob["separated_rate_with_passed_him"] = {
        "what": "as separated_rate, counting the post hoc 'passed_him' class (after review) as separated",
        "attempt_icc_across_players": reliability.icc1(sp["sep"].to_numpy(), sp["nfl_id"].to_numpy()),
        "players_with_any": int((pp["k"] > 0).sum()), "max_per_player": int(pp["k"].max())}
    return rel, rob


# =====================================================================================================
# 3. Descriptive relation to the 40 and other Combine measures
# =====================================================================================================

def combine_table(pv: pl.DataFrame) -> pl.DataFrame:
    cr = io.load("combine_results").select("nfl_id", *[c for c in COMBINE_MEASURES if not c.startswith("c40_")])
    sp = speed.combine_speed_table().select("nfl_id", "c40_v_20_40", "c40_smax", "c40_n_attempts")
    d = pv.join(cr, on="nfl_id", how="left").join(sp, on="nfl_id", how="left")
    # Combine measures centered within position (WR and TE differ by design); mechanics are already z within session.
    return d.with_columns([(pl.col(c) - pl.col(c).mean().over("position")).alias(f"{c}_c") for c in COMBINE_MEASURES])


def spearman_ci(x: np.ndarray, y: np.ndarray) -> dict:
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 10:
        return {"n": n, "rho": None, "ci95": [None, None]}
    r = float(spearmanr(x[ok], y[ok]).statistic)
    se = 1.06 / np.sqrt(n - 3)  # Fieller et al. (1957) SE for Fisher-z of Spearman's rho
    z = np.arctanh(r)
    return {"n": n, "rho": r, "ci95": [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]}


def descriptive_block(ct: pl.DataFrame) -> dict:
    out = {"label": "descriptive only (no tests, no multiplicity claims); Combine measures centered within position",
           "correlations": {}}
    for m in G.MECHANICS:
        out["correlations"][m] = {c: spearman_ci(ct[m].to_numpy().astype(float), ct[f"{c}_c"].to_numpy().astype(float))
                                  for c in COMBINE_MEASURES}
    mm = {}
    for i, a in enumerate(G.MECHANICS):
        for b in G.MECHANICS[i + 1:]:
            mm[f"{a}~{b}"] = spearman_ci(ct[a].to_numpy().astype(float), ct[b].to_numpy().astype(float))
    out["between_mechanics"] = mm
    cells = [v for m in out["correlations"].values() for v in m.values() if v["rho"] is not None]
    out["n_cells"] = len(cells)
    out["n_ci_excluding_0"] = int(sum(1 for v in cells if v["ci95"][0] > 0 or v["ci95"][1] < 0))
    out["expected_by_chance_at_5pct"] = 0.05 * len(cells)
    return out


# =====================================================================================================
# 4. Rookie receiving outcomes: exposure, reliability, power; tests only if the gate passes
# =====================================================================================================

def mde_one_sided(n: int, k_design: int) -> float | None:
    """The harness's one-sided MDE (80% power, alpha 0.05) of the observed partial r: resid_df = n - k - 1,
    where k counts the intercept and design columns (as in linkage.evaluate)."""
    df = n - k_design - 1
    if df < 3:
        return None
    return float(np.tanh((1.645 + 0.84) / np.sqrt(df - 1)))


def outcome_block(pv: pl.DataFrame, rel: dict) -> dict:
    with threadpool_limits(limits=3):  # HistGradientBoosting's OpenMP threads (shared machine)
        tg = G.target_plays()
    t25 = tg.filter(pl.col("season") == 2025)
    rk = G.rookie_targets(t25)
    ids = pv["nfl_id"].implode()
    mine = t25.filter(pl.col("nfl_id").is_in(ids))
    expo = (pv.select("nfl_id", "position").join(
        mine.group_by("nfl_id").agg(targets=pl.len(), catches=pl.col("caught").sum(), drops=pl.col("drop").sum()),
        on="nfl_id", how="left").fill_null(0))
    out = {"expectation_model": {
        "targets_all_seasons": tg.height, "brier": fnum(((tg["caught"] - tg["p_catch"]) ** 2).mean()),
        "brier_base_rate": fnum(((tg["caught"] - tg["caught"].mean()) ** 2).mean()),
        "catch_rate": fnum(tg["caught"].mean()), "drop_rate_per_target": fnum(tg["drop"].mean())}}
    out["exposure_54"] = {
        "targets_total": int(expo["targets"].sum()), "drops_total": int(expo["drops"].sum()),
        "median_targets": fnum(expo["targets"].median()),
        "with_ge_10_targets": dict(expo.filter(pl.col("targets") >= 10).group_by("position").len().iter_rows()),
        "with_ge_5_catches": dict(expo.filter(pl.col("catches") >= 5).group_by("position").len().iter_rows()),
        "with_any_drop": int((expo["drops"] > 0).sum())}
    # dropped_pass is a play-level event copied onto every tracked player of the play (post hoc count, after review)
    ev = (io.scan_game_tracking().filter((pl.col("event") == "dropped_pass") & pl.col("nfl_id").is_in(ids))
          .select("game_id", "play_id", "nfl_id").unique().collect())
    pp = (pl.scan_parquet(io.PROCESSED / "player_play.parquet").select("game_id", "play_id", "nfl_id", "target")
          .join(ev.lazy(), on=["game_id", "play_id", "nfl_id"], how="semi").collect())
    j = ev.join(pp, on=["game_id", "play_id", "nfl_id"], how="left").join(
        io.load("games").select("game_id", "season_type"), on="game_id", how="left")
    reg = j.filter((pl.col("season_type") == "REG") & pl.col("target").is_not_null())
    out["dropped_pass_attribution_54"] = {
        "label": "post hoc (after review)",
        "player_plays_with_event_all_seasons_and_types": ev.height,
        "with_player_play_row": int(j["target"].is_not_null().sum()),
        "not_the_target": int((j["target"] == False).sum()),  # noqa: E712
        "reg_with_player_play_row": reg.height, "reg_not_the_target": int((reg["target"] == False).sum())}  # noqa: E712
    # Target reliability: split-half (odd/even games) over all study WR/TE player-seasons (2023-25) with enough
    # volume, Spearman-Brown to full length; and the EB weights (reliability of each rookie's raw mean).
    ps = tg.with_columns(ps=pl.concat_str([pl.col("nfl_id").cast(pl.Utf8), pl.col("season").cast(pl.Utf8)],
                                          separator="_"),
                         croe_x=pl.col("caught") - pl.col("p_catch"))
    rel_t = {}
    for name, value, frame in (("croe", "croe_x", ps), ("drop_rate", "drop", ps),
                               ("yacoe", "yacoe", ps.filter(pl.col("yacoe").is_finite()))):
        sh = reliability.split_half(frame, value, player="ps", unit="game_id", min_per_half=10)
        n_per = frame.group_by("ps").len()
        rel_t[name] = {"split_half_player_seasons": sh,
                       "median_n_per_player_season_with_ge_20": fnum(n_per.filter(pl.col("len") >= 20)["len"].median())}
    r54 = rk.join(pv.select("nfl_id"), on="nfl_id", how="semi")
    rel_t["croe"]["eb_mean_weight_rookies"] = fnum(r54["croe_w"].mean())
    rel_t["drop_rate"]["eb_mean_weight_rookies"] = fnum(r54["drop_w"].mean())
    rel_t["yacoe"]["eb_mean_weight_rookies"] = fnum(r54["yacoe_w"].mean())
    out["target_reliability"] = rel_t
    n_croe = int(r54["croe"].is_finite().sum())
    n_yac = int(r54["yacoe"].is_finite().sum())
    k = 5  # intercept + position + forty + log_pick + undrafted (reduced controls; full B2 ~ 15 columns)
    k_b2 = 1 + 1 + len(linkage.COMBINE_CONTROLS) + len(linkage.DRAFT_CONTROLS) + 3  # + ~3 missingness indicators
    out["power"] = {
        "n_croe_drop": n_croe, "n_yacoe": n_yac,
        "mde_one_sided_observed_reduced_controls": {"croe_drop": mde_one_sided(n_croe, k), "yacoe": mde_one_sided(n_yac, k)},
        "full_B2_resid_df": n_croe - k_b2 - 1,
        "note": "full B2 leaves resid_df < 10 (harness min_df), so it cannot be used; true-score MDE = observed MDE / "
                "sqrt(rel_feature x rel_target), which exceeds 1 at the reliabilities found here"}
    best = max((v["cross_drill"]["reliability"] for v in rel.values() if np.isfinite(v["cross_drill"]["reliability"])),
               default=np.nan)
    rt = rel_t["croe"]["eb_mean_weight_rookies"] or np.nan
    m0 = out["power"]["mde_one_sided_observed_reduced_controls"]["croe_drop"]
    out["power"]["true_score_mde_best_case"] = fnum(m0 / np.sqrt(best * rt)) if m0 and best > 0 and rt > 0 else None
    passing = [m for m, v in rel.items() if v["passes_gate"]]
    out["gate"] = {"threshold": GATE, "mechanics_passing": passing,
                   "decision": ("run prelisted exploratory tests" if passing else
                                "STOP: no mechanic reaches reliability 0.5; per the plan no Combine-game association "
                                "was computed (0 tests spent)")}
    out["tests"] = []
    if passing:  # prelisted pairing / sign / selection (plan); never reached in the 2026-10-15 run
        plan = json.loads(PLAN.read_text())["exploratory_tests_if_gate_passes"]
        pairing = plan["pairing_and_sign"]
        order = sorted(passing, key=lambda m: -rel[m]["cross_drill"]["reliability"])[:3]
        jobs = [(m, *pairing[m]) for m in order]
        if len(jobs) < 3 and "catch_offset" in passing:
            jobs.append(("catch_offset", "drop_rate", -1))
        d = linkage.player_table().join(pv.drop("position"), on="nfl_id", how="inner").join(rk, on="nfl_id", how="left")
        for m, target, sign in jobs[:3]:
            r = linkage.evaluate(d, m, target, expected_sign=sign, combine_controls=["forty"],
                                 n_repeats=50, n_perm=2000, n_perm_fl=10000, seed=SEED, n_jobs=3)
            out["tests"].append({"label": "prelisted exploratory", "mechanic": m, "target": target, **r})
    return out


# =====================================================================================================
# 5. Figures
# =====================================================================================================

QC_DRILLS = ["CURL_ROUTE_RIGHT", "GO_ROUTE_RIGHT", "OVER_SHOULDER_ADJUST", "RED_ZONE_FADE_RIGHT", "CIRCUS_ROUTE_LEFT",
             "CORNER_ROUTE"]


def fig_qc_contact(frames: pl.DataFrame, att: pl.DataFrame, path: Path) -> None:
    """One secured attempt per drill family: paths (receiver line, ball dots by time), release (red triangle), the
    contact frame (black square) and catch point (star), the ball's flight line; below, speeds and distance."""
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(2, len(QC_DRILLS), figsize=(3.4 * len(QC_DRILLS), 7.2))
    for c, dn in enumerate(QC_DRILLS):
        cand = att.filter((pl.col("drill_name") == dn) & (pl.col("outcome") == "secured"))
        r = cand.row(int(rng.integers(cand.height)), named=True)
        g = frames.filter(pl.col("traj") == r["traj"])
        t = g["t"].to_numpy()
        i_r, k = r["i_release"], r["i_contact"]
        ax = axes[0, c]
        ax.plot(g["x"], g["y"], c="tab:blue", lw=1.6, label="receiver")
        ax.scatter(g["bx"], g["by"], s=7, c=t, cmap="viridis", label="ball (time)")
        ax.plot(g["bx"][i_r], g["by"][i_r], "r^", ms=8, label="release")
        ax.plot(g["bx"][k], g["by"][k], "ks", ms=6, mfc="none", label="first post-contact ball frame")
        ax.plot(r["catch_x"], r["catch_y"], "k*", ms=11, label="catch point (receiver at contact)")
        ax.set_aspect("equal")
        ax.set_title(f"{dn}\nflight {r['flight_time']:.2f} s, offset {r['catch_offset']:.2f} yd", fontsize=8)
        ax.tick_params(labelsize=7)
        ax = axes[1, c]
        ax.plot(t, g["speed"], c="tab:blue", label="receiver speed (SG)")
        ax.plot(t, g["bs"], c="tab:orange", label="ball speed (device s)")
        ax.plot(t, g["dist"], c="k", ls="--", lw=1, label="ball-receiver distance (yd)")
        ax.axvline(r["t_release"], c="r", lw=1)
        ax.axvline(r["t_contact"], c="k", lw=1)
        ax.axvspan(r["t_contact"] - G.WIN, r["t_contact"] + G.WIN, color="0.85", zorder=0)
        ax.set_ylim(0, 26)
        ax.set_xlabel("s since attempt start", fontsize=8)
        ax.tick_params(labelsize=7)
        if c == 0:
            axes[0, c].legend(fontsize=6, loc="best")
            ax.legend(fontsize=6, loc="upper left")
            ax.set_ylabel("yd/s, yd", fontsize=8)
    fig.suptitle("E011 QC: release (red) = first of >= 3 frames with ball speed >= 12 yd/s; contact (black) = one-frame "
                 "ball-speed drop; grey = +-0.5 s retention window", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_qc_outcomes(frames: pl.DataFrame, att: pl.DataFrame, cc: dict, path: Path) -> None:
    """Left: post-contact median distance vs its growth, by inferred class (post hoc 'passed him' marked). Middle:
    unanchored contact-time check. Right: a 'separated' example and a post hoc 'passed him' example."""
    fig = plt.figure(figsize=(16, 4.6))
    gs = fig.add_gridspec(1, 5)
    ax = fig.add_subplot(gs[0, 0:2])
    obs = att.filter(pl.col("outcome").is_in(["secured", "separated", "not_reached"]) & pl.col("post_median").is_finite()
                     & (pl.col("outcome_posthoc") != "ball_unrelated"))
    for cls, col, mk in (("secured", "tab:blue", "o"), ("separated", "tab:red", "o"), ("not_reached", "0.5", "o"),
                         ("passed_him", "tab:purple", "x")):
        d = obs.filter(pl.col("outcome_posthoc") == cls)
        lab = "not_reached: passed him in flight (post hoc)" if cls == "passed_him" else cls
        ax.scatter(d["post_median"].clip(upper_bound=10), d["post_growth"], s=12, c=col, marker=mk,
                   label=f"{lab} ({d.height} with a 0.7 s post-contact window)")
    ax.axvline(G.SECURE_DIST, c="k", ls=":", lw=1)
    ax.axhline(G.SECURE_GROWTH, c="k", ls=":", lw=1)
    ax.set_xlabel("median ball-receiver distance 0.2-0.7 s after contact (yd; clipped at 10)", fontsize=8)
    ax.set_ylabel("growth of that distance (yd)")
    ax.legend(fontsize=7)
    ax.set_title("Inferred outcome (no labels in the data): a continuum", fontsize=9)
    ax = fig.add_subplot(gs[0, 2])
    r = att.filter(pl.col("outcome").is_in(["secured", "separated", "reached_unobserved"]))
    dt = (r["t_cpa"] - r["t_contact"]).drop_nans().drop_nulls().to_numpy()
    ax.hist(np.clip(dt, -0.5, 0.5), bins=np.arange(-0.5, 0.526, 0.05), color="0.4")
    ax.axvspan(-0.1, 0.1, color="tab:green", alpha=0.12, zorder=0)
    ax.set_xlabel("closest approach - t_contact (s; clipped at +-0.5)", fontsize=8)
    ax.set_title(f"Contact-time check: {cc['share_within_0.1s']:.0%} within 0.1 s\n"
                 f"(chance {cc['chance_share_within_0.1s']:.0%}; {cc['share_within_0.15s']:.0%} within 0.15 s)",
                 fontsize=9)
    sep = att.filter(pl.col("outcome") == "separated").sort("post_median", descending=True)
    ph = att.filter(pl.col("outcome_posthoc") == "passed_him").sort("d_flight_min")
    picks = [("separated", sep.row(sep.height // 2, named=True)), ("passed him (post hoc)", ph.row(0, named=True))]
    for j, (name, rr) in enumerate(picks):
        ax = fig.add_subplot(gs[0, 3 + j])
        g = frames.filter(pl.col("traj") == rr["traj"])
        ax.plot(g["x"], g["y"], c="tab:blue", lw=1.5)
        ax.scatter(g["bx"], g["by"], s=7, c=g["t"], cmap="viridis")
        ax.plot(rr["catch_x"], rr["catch_y"], "k*", ms=11)
        ax.set_aspect("equal", adjustable="datalim")
        extra = (f"post-contact median {rr['post_median']:.1f} yd" if name == "separated" else
                 f"in flight {rr['d_flight_min']:.2f} yd from him, then\n{rr['ball_along_last_flight']:.1f} yd past him "
                 f"before first impact")
        ax.set_title(f"{name}: {rr['drill_name']}\n{extra}", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_reliability(rel: dict, desc: dict, path: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1, 1.6]})
    ms = G.MECHANICS
    for i, m in enumerate(ms):
        v = rel[m]
        r = v["cross_drill"]["reliability"]
        if np.isfinite(r):
            lo, hi = v.get("cross_drill_boot_ci95", [np.nan, np.nan])
            a1.errorbar(r, i, xerr=[[r - lo], [hi - r]], fmt="o", c="k", capsize=3)
            a1.text(1.02, i, f"{r:.2f} (ICC/drill {v['cross_drill']['icc']:.2f}, {v['median_drills_per_player']:.0f} drills)",
                    va="center", fontsize=8, transform=a1.get_yaxis_transform())
        else:
            a1.text(-0.8, i, "not estimable (0 s in 3 of 4 drills)", va="center", fontsize=8)
        rt = v["within_drill_retest"]["icc"]
        ci = v.get("within_drill_retest_icc_boot_ci95")
        if np.isfinite(rt) and ci and m != "adjust_time":
            a1.errorbar(rt, i - 0.28, xerr=[[rt - ci[0]], [ci[1] - rt]], fmt="s", c="tab:orange", ms=4, capsize=2,
                        lw=1)
            a1.text(ci[1] + 0.03, i - 0.28, f"{v['within_drill_retest']['n_repeated']} cells",
                    va="center", fontsize=6, color="tab:orange")
    a1.plot([], [], "s", c="tab:orange", ms=4,
            label="within-drill retest ICC, 95% CI (repeats often follow a failed rep)")
    a1.plot([], [], "o", c="k", label="cross-drill reliability, as used (95% player bootstrap)")
    a1.axvline(GATE, c="r", ls="--", lw=1, label="gate 0.5")
    a1.axvline(0, c="0.6", lw=0.6)
    a1.set_yticks(range(len(ms)), [LABELS[m] for m in ms])
    a1.set_xlim(-0.85, 1)
    a1.set_ylim(len(ms) + 0.9, -0.5)  # inverted; the empty band below the last row holds the legend
    a1.set_xlabel("reliability (black) / retest ICC (orange)")
    a1.legend(fontsize=7, loc="lower left")
    a1.set_title("No catch mechanic is a stable trait at this sample (54 players, 2025 only)", fontsize=9)
    cm = COMBINE_MEASURES
    mat = np.array([[desc["correlations"][m][c]["rho"] if desc["correlations"][m][c]["rho"] is not None else np.nan
                     for c in cm] for m in ms], dtype=float)
    im = a2.imshow(mat, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    for i in range(len(ms)):
        for j in range(len(cm)):
            v = desc["correlations"][ms[i]][cm[j]]
            if v["rho"] is not None:
                sig = v["ci95"][0] > 0 or v["ci95"][1] < 0
                a2.text(j, i, f"{v['rho']:+.2f}" + ("*" if sig else ""), ha="center", va="center", fontsize=7)
    a2.set_xticks(range(len(cm)), cm, rotation=45, ha="right", fontsize=8)
    a2.set_yticks(range(len(ms)), [LABELS[m] + (" (CIRCUS only)" if m == "adjust_time" else "") for m in ms], fontsize=8)
    fig.colorbar(im, ax=a2, shrink=0.8, label="Spearman rho (Combine measure centered within position)")
    a2.set_title(f"Descriptive: mechanics vs Combine measures\n(* = 95% CI excludes 0: "
                 f"{desc['n_ci_excluding_0']} of {desc['n_cells']}, ~{desc['expected_by_chance_at_5pct']:.0f} by chance)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# =====================================================================================================

def main() -> None:
    plan = json.loads(PLAN.read_text())
    log("frames")
    frames = G.ball_frames()
    att = G.attempt_table(frames)
    log(f"{att.height} attempts")
    facts = data_facts(frames)
    lag = lag_check(frames, att)
    det = detection_summary(att)
    mech = mechanic_summary(att)
    log("reliability")
    rel, rob = reliability_block(att, frames)
    pv = G.player_values(att)
    log("descriptive (40 speed table)")
    ct = combine_table(pv)
    desc = descriptive_block(ct)
    log("rookie outcomes")
    outc = outcome_block(pv, rel)
    log("figures")
    fig_qc_contact(frames, att, FIG / "E011_qc_contact.png")
    fig_qc_outcomes(frames, att, det["contact_time_check"], FIG / "E011_qc_outcomes.png")
    fig_reliability(rel, desc, FIG / "E011_reliability.png")
    att.write_parquet(P / "e011_attempts.parquet")
    ct.drop([f"{c}_c" for c in COMBINE_MEASURES]).write_parquet(P / "e011_players.parquet")
    res = {"experiment": "E011", "date": "2026-10-15", "protocol": "PLAN.md section 4 rev 3 (descriptive track)",
           "plan": plan, "data": facts, "clock_lag_check": lag, "detection": det, "mechanics": mech,
           "reliability": rel, "robustness_post_hoc": rob, "descriptive_combine": desc, "rookie_outcomes": outc,
           "test_count": {"primary": 0, "prespecified_robustness": 0, "prelisted_exploratory": len(outc["tests"]),
                          "post_hoc": 0, "note": "reliability/robustness/descriptive summaries are not tests"},
           "revision": REVISION, "runtime_s": time.time() - T0}
    (RES / "E011.json").write_text(json.dumps(clean(res), indent=1))
    log("done")


if __name__ == "__main__":
    main()
