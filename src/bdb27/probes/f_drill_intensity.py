"""Probe F (E010) - drill intensity: how close to his own top speed a player runs his position drills.

Preregistration (binding): experiments/results/E010_prereg.json. Implementation choices fixed before any
feature-target association are listed in IMPLEMENTATION_CHOICES and copied into experiments/results/E010.json.

Combine side
  - Position drills = drill_type SKILL_DRILLS_{WR,TE,DB,DL,OL} (the LB group, 4 DL players, is excluded; the 40,
    shuttle and 3-cone are not position drills). One attempt = segments.ATTEMPT_KEYS (one trajectory each: no
    tracking gaps inside an attempt, verified).
  - Drill key = position group + drill name with naming variants merged (DRILL_MERGE). The group is part of the key
    because GAUNTLET_DRILL / OVER_SHOULDER_ADJUST / BACK_PEDAL_AND_REACT / PASS_RUSH_DRILL are run by different
    position groups in different sessions.
  - Attempt peak speed `smax`: peak Savitzky-Golay speed (bdb27.kinematics, window 7) over interior frames, ignoring
    speed.EDGE frames at each end, exactly as bdb27.speed smax for the 40.
  - Ratio = smax / c40_smax (bdb27.speed.combine_speed_table: mean over QC-passing 40 attempts of the same peak).
  - z = ratio z-scored within drill x Combine session (segments.combine_sessions on the drill key); an attempt alone
    in its drill x session cannot be z-scored and is dropped.
  - Player feature c_drill_intensity = mean over drills of the player's drill-level mean z; >= 2 distinct drills.

Game side (regular season)
  - Play max speed = max raw `s` over the tracked frames of a player-play, exactly as features.game_top_speed
    (E001 target): standardized REG tracking, plays with max > features.MAX_PLAUSIBLE_SPEED dropped.
  - g_play_intensity = median(play max) / q95(play max), >= 50 plays. The q95 uses polars' default quantile
    interpolation ('nearest'), as features.game_top_speed; the median is the usual midpoint median.
    What it covers: WR/TE tracked plays are pass plays only (their player_play REG rows contain no runs), so for
    them P1 is per route; for DB/DL/OL it is per snap, 37-42% of them runs. Within position it largely reflects
    alignment and route-depth mix (post hoc, after review: see play_context / role_adjusted_intensity).
  - g_snaps_per_season = (career offensive + defensive snaps) / (2026 - draft_year), every population player.

Post hoc helpers (added 2026-10-15 after the skeptic review; not part of the preregistered definitions):
play_context, role_mix, role_adjusted_intensity, censoring_types.
"""

import numpy as np
import polars as pl

from bdb27 import features, io, kinematics, reliability, segments, speed

POSITIONS = ["WR", "TE", "DB", "DL", "OL"]
DRILL_TYPES = [f"SKILL_DRILLS_{p}" for p in POSITIONS]  # SKILL_DRILLS_LB excluded (prereg: n = 4)
# Naming variants of one drill (each pair occurs in different draft years, never in the same session).
DRILL_MERGE = {
    "LINE": "LINE_DRILL",  # DB, 2023 vs 2024-25
    "PASS_PRO-MIRROR_DRILL": "PASS_PRO_MIRROR_DRILL",  # OL, 2024-25 vs 2023
    "SPEED_OUT_LEFT": "SPEED_OUT_ROUTE_LEFT",  # WR, 2024 vs 2023/2025
    "OVER_THE_SHOULDER_ADJUST": "OVER_SHOULDER_ADJUST",  # TE, 2023 vs 2024-25
}
MIN_DRILLS = 2
MIN_PLAYS = 50
SEASONS_AVAILABLE_THROUGH = 2026  # seasons available = 2026 - draft_year (2023: 3, 2024: 2, 2025: 1)


def drill_key() -> pl.Expr:
    """'<group>:<drill name, naming variants merged>', e.g. 'WR:SPEED_OUT_ROUTE_LEFT'."""
    return pl.concat_str([pl.col("drill_type").str.replace("SKILL_DRILLS_", ""),
                          pl.col("drill_name").replace(DRILL_MERGE)], separator=":")


def drill_frames() -> pl.DataFrame:
    """Position-drill frames with `traj` (one per attempt), SG kinematics, `drill` and `t` (s since attempt start)."""
    d = segments.attempts(drill_types=DRILL_TYPES)
    d = kinematics.add_kinematics(d, segments.ATTEMPT_KEYS).sort("traj", "time")
    return d.with_columns(drill=drill_key(),
                          t=(pl.col("time") - pl.col("time").min().over("traj")).dt.total_milliseconds() / 1000)


def attempt_table(frames: pl.DataFrame | None = None) -> pl.DataFrame:
    """One row per position-drill attempt: ids, drill, session, n_frames, dur, path (x/y path length, yd), smax
    (interior SG peak), i_peak, peak_at_end (the interior peak is the last interior frame: speed still rising when
    tracking stops), s_dev_max (device `s` peak, all frames)."""
    d = drill_frames() if frames is None else frames
    rows = []
    for (traj,), g in d.group_by(["traj"], maintain_order=True):
        v, x, y = (g[c].to_numpy() for c in ("speed", "x", "y"))
        n, e = len(v), speed.EDGE
        i = e + int(np.argmax(v[e:n - e]))
        rows.append(dict(traj=traj, n_frames=n, dur=float(g["t"][-1]),
                         path=float(np.sum(np.hypot(np.diff(x), np.diff(y)))),
                         smax=float(v[i]), i_peak=i, peak_at_end=bool(i == n - 1 - e),
                         s_dev_max=float(g["s"].max())))
    ids = (d.group_by("traj").agg(pl.col("nfl_id", "draft_year", "drill_type", "drill_name", "drill", "event_id",
                                         "attempt").first(), time=pl.col("time").min()))
    a = pl.DataFrame(rows).join(ids, on="traj", how="left")
    a = segments.combine_sessions(a, drill="drill")
    pos = io.load("combine_results").select("nfl_id", position="combine_position")
    return a.join(pos, on="nfl_id", how="left").sort("drill", "session", "nfl_id", "attempt")


def add_ratio(att: pl.DataFrame, c40: pl.DataFrame) -> pl.DataFrame:
    """Add c40_smax and ratio = smax / c40_smax (null without a QC-passing tracked 40)."""
    return (att.join(c40.select("nfl_id", "c40_smax"), on="nfl_id", how="left")
            .with_columns(ratio=pl.col("smax") / pl.col("c40_smax")))


def zscore(att: pl.DataFrame, value: str, by: tuple[str, ...] = ("drill", "session")) -> pl.DataFrame:
    """Add z_<value>: `value` z-scored within drill x session over the given attempts (null when the group has
    one attempt, so no SD)."""
    v = pl.col(value)
    sd = v.std().over(*by)
    return att.with_columns(pl.when(sd > 0).then((v - v.mean().over(*by)) / sd).otherwise(None).alias(f"z_{value}"))


def drill_level(att: pl.DataFrame, value: str, agg: str = "mean") -> pl.DataFrame:
    """Player x drill rows: v = mean (or median) over the player's attempts of z_<value>, with session and
    draft_year."""
    z = f"z_{value}"
    v = pl.col(z).mean() if agg == "mean" else pl.col(z).median()
    return (att.filter(pl.col(z).is_finite()).sort("nfl_id", "drill", "attempt")
            .group_by("nfl_id", "drill")
            .agg(v=v, n_att=pl.len(), session=pl.col("session").first(),
                 draft_year=pl.col("draft_year").first(), position=pl.col("position").first())
            .sort("nfl_id", "drill"))


def player_feature(dl: pl.DataFrame, name: str, agg: str = "mean", min_drills: int = MIN_DRILLS) -> pl.DataFrame:
    """Player value = mean (or median) of the player's drill-level values; players with >= min_drills drills."""
    f = pl.col("v").mean() if agg == "mean" else pl.col("v").median()
    return (dl.group_by("nfl_id").agg(f.alias(name), n_drills=pl.len())
            .filter(pl.col("n_drills") >= min_drills).sort("nfl_id"))


def cross_drill_reliability(dl: pl.DataFrame, players: pl.Series | None = None) -> dict:
    """Prereg stop rule: reliability.cross_drill_icc(zscore=True, center_by=[drill, session]) on player x drill
    values; icc1 then Spearman-Browns each player's ICC to his own number of drills and averages over players."""
    d = dl if players is None else dl.filter(pl.col("nfl_id").is_in(players.implode()))
    return reliability.cross_drill_icc(d, "v", drill="drill", zscore=True, center_by=["drill", "session"])


# ---------------------------------------------------------------- game side

def play_max_speeds(nfl_ids: list[int]) -> pl.DataFrame:
    """One row per REG player-play of the given players: game_id, play_id, nfl_id, season, smax = max raw `s`
    over the play's tracked frames (as features.game_top_speed); plays above MAX_PLAUSIBLE_SPEED dropped."""
    lf = (features.reg_season_tracking().filter(pl.col("nfl_id").is_in(nfl_ids))
          .group_by("game_id", "play_id", "nfl_id").agg(smax=pl.col("s").max(), season=pl.col("season").first())
          .filter(pl.col("smax") <= features.MAX_PLAUSIBLE_SPEED))
    return lf.collect().sort("nfl_id", "game_id", "play_id")


def intensity_stats(smax: np.ndarray, frac: float = 0.8) -> dict:
    """median / q95 ('nearest' quantile, as polars' default) and the share of plays >= frac * q95."""
    q95 = float(pl.Series(smax).quantile(0.95))
    return {"q95": q95, "median": float(np.median(smax)), "ratio": float(np.median(smax) / q95),
            "share80": float(np.mean(smax >= frac * q95))}


def play_intensity(pm: pl.DataFrame, min_plays: int = MIN_PLAYS) -> pl.DataFrame:
    """Per player (>= min_plays plays): g_play_intensity = median / q95 of play max speed, g_share80 = share of plays
    with play max >= 80% of his own q95, q95, median and n_plays."""
    q = pl.col("smax").quantile(0.95)
    return (pm.group_by("nfl_id")
            .agg(n_plays=pl.len(), q95=q, median=pl.col("smax").median(),
                 g_share80=(pl.col("smax") >= 0.8 * q).mean())
            .filter(pl.col("n_plays") >= min_plays)
            .with_columns(g_play_intensity=pl.col("median") / pl.col("q95"))
            .sort("nfl_id"))


def half_intensity(g: pl.DataFrame, min_plays: int) -> float:
    """split_half_estimator fn: median / q95 of one player's plays in one half (NaN below min_plays)."""
    if g.height < min_plays:
        return np.nan
    return intensity_stats(g["smax"].to_numpy())["ratio"]


def career_targets(players: pl.DataFrame) -> pl.DataFrame:
    """P2 and its variants for the given players (nfl_id, draft_year): seasons, g_snaps_per_season,
    g_log_snaps_per_season = log(1 + snaps per season), g_snaps_st_per_season (special-teams snaps added),
    g_starts_per_season, g_honors = Pro Bowls + All-Pro (1st + 2nd team) selections."""
    cs = io.load("player_career_successes")
    seasons = (SEASONS_AVAILABLE_THROUGH - pl.col("draft_year")).cast(pl.Float64)
    snaps = (pl.col("career_offensive_snaps") + pl.col("career_defensive_snaps")).cast(pl.Float64)
    return (players.select("nfl_id", "draft_year").join(cs, on="nfl_id", how="left")
            .with_columns(seasons=seasons)
            .with_columns(g_snaps_per_season=snaps / pl.col("seasons"),
                          g_snaps_st_per_season=(snaps + pl.col("career_special_teams_snaps")) / pl.col("seasons"),
                          g_starts_per_season=pl.col("career_games_started") / pl.col("seasons"),
                          g_honors=(pl.col("pro_bowl_original_ballot") + pl.col("ap_all_pro_1st_team")
                                    + pl.col("ap_all_pro_2nd_team")).cast(pl.Float64))
            .with_columns(g_log_snaps_per_season=(1 + pl.col("g_snaps_per_season")).log())
            .select("nfl_id", "seasons", "g_snaps_per_season", "g_log_snaps_per_season", "g_snaps_st_per_season",
                    "g_starts_per_season", "g_honors", "career_games_active"))


# ---------------------------------------------------------------- post hoc (after the skeptic review, 2026-10-15)
DEEP_ROUTES = ["GO", "POST", "CORNER", "WHEEL"]
# Role-mix covariates per position (shares over the player's plays that have a player_play row). WR/TE rows contain
# no runs, so run share is not a WR/TE covariate.
ROLE_COVARIATES = {"WR": ["slot", "deep_of_routes"], "TE": ["slot", "deep_of_routes"],
                   "DB": ["run", "cb", "slot", "man"], "DL": ["run", "edge"], "OL": ["run", "tackle", "center"]}
CENSORED_MIN, TRUE_PEAK_MAX = 0.6, 0.3  # drill types by the share of attempts whose peak is the last interior frame
BOTH_KINDS = (0.15, 0.85)  # drills where censored and uncensored attempts are both common (z comparison)


def play_context(pm: pl.DataFrame, players: pl.DataFrame) -> pl.DataFrame:
    """Post hoc. Play max rows of `players` (nfl_id, position, q95) joined to the same player-play's player_play row:
    has_row, team_abbr, lined_up_position, run (row with pass_result null), route_ran, cov (man / rush (no
    assignment) / deepzone / shortzone / na) and the play-context cell 'position|alignment|run|route|cov'. Plays
    without a player_play row (0.9%) keep null context and form their own cells."""
    pp = (pl.scan_parquet(io.PROCESSED / "player_play.parquet")
          .select("game_id", "play_id", "nfl_id", "team_abbr", "lined_up_position", "pass_result", "route_ran",
                  "coverage_assignment").collect())
    ca = pl.col("coverage_assignment")
    cov = (pl.when(ca.is_null()).then(pl.lit("na")).when(ca == "MAN").then(pl.lit("man"))
           .when(ca == "NO_ASSIGNMENT").then(pl.lit("rush"))
           .when(ca.str.contains("DEEP|QUARTERS|HALF|PREVENT")).then(pl.lit("deepzone")).otherwise(pl.lit("shortzone")))
    d = (pm.join(players.select("nfl_id", "position", "q95"), on="nfl_id", how="inner")
         .join(pp, on=["game_id", "play_id", "nfl_id"], how="left", maintain_order="left")
         .with_columns(has_row=pl.col("team_abbr").is_not_null(), cov=cov)
         .with_columns(run=pl.col("has_row") & pl.col("pass_result").is_null()))
    return d.with_columns(cell=pl.concat_str([pl.col("position"), pl.col("lined_up_position").fill_null("NA"),
                                              pl.col("run").cast(pl.Utf8), pl.col("route_ran").fill_null("none"),
                                              pl.col("cov")], separator="|"))


def role_mix(ctx: pl.DataFrame) -> pl.DataFrame:
    """Post hoc. Per player, shares over his plays with a player_play row: run, slot (alignment contains SLOT), cb,
    edge, tackle (T), center (C); man = MAN among his coverage snaps (man or zone assignment); deep_of_routes =
    GO/POST/CORNER/WHEEL among routes run."""
    lp = pl.col("lined_up_position")
    share = lambda num, den: pl.col(num).sum().cast(pl.Float64) / pl.col(den).sum().cast(pl.Float64).clip(1)
    return (ctx.filter(pl.col("has_row"))
            .with_columns(slot=lp.str.contains("SLOT").fill_null(False), cb=(lp == "CB").fill_null(False),
                          edge=(lp == "EDGE").fill_null(False), tackle=(lp == "T").fill_null(False),
                          center=(lp == "C").fill_null(False), man=pl.col("cov") == "man",
                          covering=pl.col("cov").is_in(["man", "deepzone", "shortzone"]),
                          routed=pl.col("route_ran").is_not_null(),
                          deep=pl.col("route_ran").is_in(DEEP_ROUTES).fill_null(False))
            .group_by("nfl_id")
            .agg(pl.col("run", "slot", "cb", "edge", "tackle", "center").cast(pl.Float64).mean(),
                 man=share("man", "covering"), deep_of_routes=share("deep", "routed"))
            .sort("nfl_id"))


def add_role_adjusted(ctx: pl.DataFrame) -> pl.DataFrame:
    """Post hoc. Per-play smax_adj = smax - mean(smax | play-context cell) + mean(smax | position), with cell means
    over all plays in ctx."""
    return ctx.with_columns(smax_adj=pl.col("smax") - pl.col("smax").mean().over("cell")
                            + pl.col("smax").mean().over("position"))


def role_adjusted_intensity(ctx: pl.DataFrame, min_plays: int = MIN_PLAYS) -> pl.DataFrame:
    """Post hoc role-adjusted P1: median(smax_adj) / q95(raw smax) (the P1 denominator), >= min_plays plays."""
    return (add_role_adjusted(ctx).group_by("nfl_id")
            .agg(n=pl.len(), med_adj=pl.col("smax_adj").median(), q95=pl.col("smax").quantile(0.95))
            .filter(pl.col("n") >= min_plays)
            .select("nfl_id", g_p1_role_adjusted=pl.col("med_adj") / pl.col("q95")).sort("nfl_id"))


def censoring_types(att: pl.DataFrame) -> pl.DataFrame:
    """Post hoc. Per drill: pe = share of the given attempts whose interior peak is the last interior frame, and
    ctype = 'censored' (pe >= CENSORED_MIN), 'true_peak' (pe < TRUE_PEAK_MAX) or 'mixed'."""
    pe = pl.col("pe")
    return (att.group_by("drill").agg(pe=pl.col("peak_at_end").cast(pl.Float64).mean(), n_att=pl.len())
            .with_columns(ctype=pl.when(pe >= CENSORED_MIN).then(pl.lit("censored"))
                          .when(pe < TRUE_PEAK_MAX).then(pl.lit("true_peak")).otherwise(pl.lit("mixed")))
            .sort("drill"))
