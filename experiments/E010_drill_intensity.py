"""E010 - Drill intensity (preregistered): how close to his own top speed a player runs his position drills.

Preregistration (binding): experiments/results/E010_prereg.json. Protocol: PLAN.md section 4 rev 3.
Code: src/bdb27/probes/f_drill_intensity.py (feature and targets). IMPLEMENTATION_CHOICES below were fixed before
any feature-target association was computed and are copied into the results JSON.

Order of work (each stage only uses what the previous one allows):
  1. Combine side only: attempt table, QC rule, QC figures (speed traces; intensity by drill and session), the
     feature, and the preregistered reliability stop rule (+ feature-side diagnostics, no game data).
  2. STOP RULE: if the cross-drill reliability is below 0.5, write the result and stop (no targets, no tests).
  3. Otherwise: targets (P1 play intensity, P2 snaps per season), target reliability, primaries, prespecified
     robustness, prelisted exploratory, BH, figures.
  4. POST HOC (added 2026-10-15 after the two skeptic reviews; post_hoc_review, labeled in the JSON): what P1 measures
     (coverage, role mix, role-adjusted P1), same-position team share, pooled vs per-position bounds, P2 as a
     playing-time trait, where P2's raw r goes, sprint capacity inside the feature, drill censoring, the literal
     'drill-level median' prereg variant, and BH over every p-valued test including these.

Run: .venv/bin/python experiments/E010_drill_intensity.py              (everything; n_jobs = 3)
     .venv/bin/python experiments/E010_drill_intensity.py --qc-only    (QC figures only; how the QC rule was reviewed
                                                                          before any reliability was computed)
     .venv/bin/python experiments/E010_drill_intensity.py --combine    (stage 1-2 only, then exit)
Final run: ~65 min under a shared CPU (n_jobs = 3), peak RSS 1.46 GB; deterministic (seed-0 numbers reproduced exactly
across two full runs).
     .venv/bin/python experiments/E010_drill_intensity.py --quick      (small permutation counts, scratch output)
     .venv/bin/python experiments/E010_drill_intensity.py --post-hoc   (stage 4 only, on the saved results and
                                                                          parquets; ~7 min, 0.75 GB)
Outputs: experiments/results/E010.json, experiments/figures/E010_*.png, data/processed/e010_*.parquet.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import norm

from bdb27 import io, linkage, reliability, speed
from bdb27.probes import f_drill_intensity as F

ROOT = Path(__file__).resolve().parents[1]
RES, FIG, DATA = ROOT / "experiments" / "results", ROOT / "experiments" / "figures", io.PROCESSED
QUICK = "--quick" in sys.argv
COMBINE_ONLY = "--combine" in sys.argv
POST_HOC_ONLY = "--post-hoc" in sys.argv
SCRATCH = Path("/private/tmp/claude-501/-Users-jasonli-Documents-GitHub-NFL-Big-Data-Bowl-2027/"
               "ca6dc0b9-7e4f-45b4-b4e4-1468819c8969/scratchpad/round3/E010/quick")
if QUICK:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    RES = FIG = DATA = SCRATCH
N_JOBS = 3
EVAL = dict(n_repeats=50, n_perm=2000, n_perm_fl=10000) if not QUICK else dict(n_repeats=5, n_perm=50, n_perm_fl=500)
REL_STOP = 0.5
T0 = time.time()

IMPLEMENTATION_CHOICES = {
    "position_drills": "drill_type SKILL_DRILLS_{WR,TE,DB,DL,OL}; SKILL_DRILLS_LB excluded (its 4 players are DL who "
                       "ran only LB drills, so they leave the population). The 40, shuttle and 3-cone are excluded.",
    "drill_key": "position group + drill name, e.g. 'WR:GAUNTLET_DRILL' (GAUNTLET_DRILL, OVER_SHOULDER_ADJUST, "
                 "BACK_PEDAL_AND_REACT and PASS_RUSH_DRILL are run by different position groups).",
    "naming_variants_merged": F.DRILL_MERGE,
    "not_merged": "DL SHORT_ZONE_BREAKS (2023-24, several attempts per player) and the 2025 drills "
                  "SHORT_ZONE_BREAK_FLAT_WHEEL / _FORWARD / _INSIDE are kept as named: the 2025 names are different "
                  "movements, not spellings. Because z-scores are formed within drill x session and every naming "
                  "variant occurs in different draft years, a merge only changes the drill label and the distinct-drill "
                  "count, never a z-score.",
    "attempt": "segments.ATTEMPT_KEYS (event_id, nfl_id, drill_name, attempt); one trajectory each (no tracking gap "
               "inside any attempt; all frames 0.1 s apart; verified).",
    "attempt_peak": "max SG speed (kinematics.add_kinematics, window 7, poly 2) over frames [EDGE, n - EDGE), "
                    "EDGE = speed.EDGE = 3, as bdb27.speed smax for the 40.",
    "forty_peak": "c40_smax from speed.combine_speed_table(speed.forty_attempts()): mean of the same interior SG peak "
                  "over the player's QC-passing 40 attempts.",
    "attempt_qc": "keep an attempt if its x/y path length is within [0.5, 2.5] x the median path of its drill x "
                  "session (drops aborted attempts and tracking overruns that continue past the drill). Fixed from "
                  "Combine data only, before any game data were loaded.",
    "sessions": "segments.combine_sessions on the attempt table with drill = the drill key (> 10 min since the "
                "previous attempt start of the same drill and draft year opens a new session).",
    "z_scoring": "ratio z-scored within drill x session over QC-passing attempts of players with a QC-passing tracked "
                 "40; attempts alone in their drill x session (no SD) are dropped.",
    "aggregation": "drill-level value = mean of the player's attempt z-scores in that drill; player value = mean of "
                   "drill-level values; >= 2 distinct drills (drill keys).",
    "stop_rule_input": "player x drill rows (the drill-level values above) of population players, passed to "
                       "reliability.cross_drill_icc(value, drill='drill', zscore=True, center_by=['drill', 'session']); "
                       "its icc1 Spearman-Browns each player's ICC to his number of drills and averages over players.",
    "variant_no_division": "attempt peak speed smax z-scored within drill x session (same attempts and aggregation).",
    "variant_median": "player value = median (instead of mean) of his drill-level values. A per-drill median over "
                      "attempts would be nearly vacuous (most drills have 1-2 attempts, where median = mean).",
    "play_max_speed": "max raw `s` over all tracked frames of a REG player-play, from features.reg_season_tracking() "
                      "(standardized, REG only), plays with max > 12 yd/s dropped: the features.game_top_speed / E001 "
                      "definition.",
    "p1_quantiles": "q95 = polars quantile(0.95) default ('nearest', as features.game_top_speed); median = midpoint "
                    "median. Players need >= 50 such plays.",
    "p1_split_half": "reliability.split_half_estimator by odd/even game (dense rank), fn = median/q95 of the half's "
                     "plays, NaN below 25 plays in the half; Spearman-Brown to full length.",
    "p1_team_share": "reliability.team_share on the per-play value play max / player's own q95 (all his plays), "
                     "team_col = [team_abbr from player_play, season]; per position, and pooled on position-centered "
                     "values. The shared full-sample q95 enters both halves, so the player variance (denominator) is "
                     "slightly overstated and the team share slightly understated.",
    "p2_population": "every population player (0 snaps if he never played); seasons = 2026 - draft_year.",
    "p2_variants": "two separate variants: (a) log(1 + snaps per season); (b) (offensive + defensive + special-teams "
                   "snaps) per season.",
    "p2_reliability": "snaps are a recorded count (no measurement error); as context, the season-to-season correlation "
                      "of off+def snap counts from player_play (players with 2+ available seasons), Spearman-Brown to "
                      "the class's number of seasons.",
    "exploratory_starts": "career_games_started / seasons, linkage.evaluate with expected_sign=+1 (same settings).",
    "exploratory_honors": "Pro Bowl + All-Pro (1st and 2nd team) count; descriptive only (no test): number of honored "
                          "players and their mean feature vs the rest, within position.",
    "leave_3_out": "drop the 3 players contributing most to sign * x_res * y_res (supportive) and, separately, the 3 "
                   "contributing least (contrary), using the B2 residuals of the harness; rerun the seed-free "
                   "Freedman-Lane test (CV parts at minimal settings, not used). If a primary passes criterion 1, the "
                   "supportive drop is also rerun at full settings for criteria 2-3.",
    "by_position": "linkage.evaluate on each position alone (same settings), descriptive.",
    "multiplicity": "BH across P1-P2 (the primary family); BH across every p-valued E010 test reported alongside.",
}


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_json(obj: dict) -> None:
    (RES / "E010.json").write_text(json.dumps(_clean(obj), indent=1))
    log(f"wrote {RES / 'E010.json'}")


# =====================================================================================================
# 1. Combine side
# =====================================================================================================
QC_PATH = (0.5, 2.5)


def add_qc(att: pl.DataFrame) -> pl.DataFrame:
    med = pl.col("path").median().over("drill", "session")
    return att.with_columns(path_rel=pl.col("path") / med).with_columns(
        qc_ok=pl.col("path_rel").is_between(*QC_PATH))


def build_combine() -> dict:
    frames = F.drill_frames()
    att = add_qc(F.attempt_table(frames))
    c40 = speed.combine_speed_table(speed.forty_attempts())
    att = F.add_ratio(att, c40)
    log(f"attempts {att.height}, players {att['nfl_id'].n_unique()}, qc fails {att.filter(~pl.col('qc_ok')).height}, "
        f"without tracked 40 {att.filter(pl.col('c40_smax').is_null())['nfl_id'].n_unique()} players")
    use = att.filter(pl.col("qc_ok") & pl.col("ratio").is_finite())
    use = F.zscore(F.zscore(use, "ratio"), "smax")
    dl = F.drill_level(use, "ratio")
    feat = F.player_feature(dl, "c_drill_intensity")
    dl_speed = F.drill_level(use, "smax")
    feat_speed = F.player_feature(dl_speed, "c_drill_speed_z")
    feat_med = F.player_feature(dl, "c_drill_intensity_median", agg="median")
    pop = feat["nfl_id"]
    players = (linkage.player_table().join(feat, on="nfl_id", how="inner")
               .join(feat_speed.select("nfl_id", "c_drill_speed_z"), on="nfl_id", how="left")
               .join(feat_med.select("nfl_id", "c_drill_intensity_median"), on="nfl_id", how="left")
               .join(c40.select("nfl_id", "c40_smax"), on="nfl_id", how="left").sort("nfl_id"))
    assert players.height == feat.height, "population players missing from player_table"
    assert set(players["position"].unique()) <= set(F.POSITIONS)
    return dict(frames=frames, att=att, use=use, dl=dl.filter(pl.col("nfl_id").is_in(pop.implode())),
                dl_speed=dl_speed.filter(pl.col("nfl_id").is_in(pop.implode())), players=players, c40=c40)


def _resid_rows(dl: pl.DataFrame, players: pl.DataFrame, cols: list[str], year_effects: bool = True) -> pl.DataFrame:
    """Drill-level rows with v replaced by v minus the OLS fit on the player's design (position + draft-year dummies +
    `cols`, imputed as in the harness). Player-level covariates are constant within a player, so this removes the
    part of each player's true score they explain and leaves the within-player (cross-drill) noise unchanged."""
    p = players.sort("nfl_id")
    X = linkage._imputed(linkage._design(p, cols, year_effects))
    idx = {int(i): k for k, i in enumerate(p["nfl_id"].to_list())}
    rows = dl.filter(pl.col("nfl_id").is_in(p["nfl_id"].implode()))
    Xr = X[[idx[int(i)] for i in rows["nfl_id"].to_list()]]
    v = rows["v"].to_numpy()
    beta = np.linalg.lstsq(Xr, v, rcond=None)[0]
    return rows.with_columns(v=pl.Series(v - Xr @ beta))


def combine_stage(C: dict) -> dict:
    """Stop rule + feature-side diagnostics (no game data)."""
    att, use, dl, players = C["att"], C["use"], C["dl"], C["players"]
    out = {}
    # ---- the preregistered stop rule
    rel = F.cross_drill_reliability(dl)
    out["stop_rule"] = {**rel, "threshold": REL_STOP, "passes": bool(rel["reliability"] >= REL_STOP),
                        "n_rows": dl.height, "n_drills_mean": float(players["n_drills"].mean())}
    log(f"STOP RULE: cross-drill ICC {rel['icc']:.3f}, SB reliability {rel['reliability']:.3f} "
        f"({rel['n_players']} players, mean {players['n_drills'].mean():.1f} drills) -> "
        f"{'PASS' if rel['reliability'] >= REL_STOP else 'STOP'}")
    # ---- diagnostics (feature side only; none of these is the gate)
    diag = {}
    diag["by_position"] = {p: F.cross_drill_reliability(dl.filter(pl.col("position") == p)) for p in F.POSITIONS}
    diag["variant_no_division"] = F.cross_drill_reliability(C["dl_speed"])
    pc = players.with_columns(c40c=pl.col("c40_smax") - pl.col("c40_smax").mean().over("position"))
    diag["beyond_40_peak"] = reliability.cross_drill_icc(_resid_rows(dl, pc, ["c40c"]), "v", drill="drill",
                                                         zscore=True, center_by=["drill", "session"])
    b2 = linkage.COMBINE_CONTROLS + linkage.DRAFT_CONTROLS
    diag["beyond_B2"] = reliability.cross_drill_icc(_resid_rows(dl, players, b2), "v", drill="drill", zscore=True,
                                                    center_by=["drill", "session"])
    diag["beyond_B2_no_division"] = reliability.cross_drill_icc(_resid_rows(C["dl_speed"], players, b2), "v",
                                                                drill="drill", zscore=True,
                                                                center_by=["drill", "session"])
    # R^2 of the feature on the B2 design (in-sample), and correlation with the 40 peak within position
    R = linkage._residual_maker(linkage._imputed(linkage._design(players.sort("nfl_id"), b2)))
    f = players.sort("nfl_id")["c_drill_intensity"].to_numpy()
    fc = f - f.mean()
    diag["feature_r2_on_B2_in_sample"] = float(1 - (R @ f) @ (R @ f) / (fc @ fc))
    cen = lambda c: pl.col(c) - pl.col(c).mean().over("position")
    pcen = players.select(f=cen("c_drill_intensity"), s=cen("c40_smax"), v=cen("c_drill_speed_z"))
    diag["r_feature_c40_smax_within_position"] = float(np.corrcoef(pcen["f"], pcen["s"])[0, 1])
    diag["r_feature_variant_no_division_within_position"] = float(np.corrcoef(pcen["f"], pcen["v"])[0, 1])
    diag["r_variant_no_division_c40_smax_within_position"] = float(np.corrcoef(pcen["v"], pcen["s"])[0, 1])
    # between-year share of the feature (z-scoring is within session, so it should be ~0)
    yr = reliability.icc1(players["c_drill_intensity"].to_numpy(),
                          players.select(pl.concat_str(["position", pl.col("draft_year").cast(pl.Utf8)])).to_series().to_numpy())
    diag["feature_year_x_position_share"] = yr["icc"]
    out["diagnostics"] = diag
    # ---- QC summary
    out["qc"] = {
        "n_attempts": att.height, "n_players_any_drill": att["nfl_id"].n_unique(),
        "n_qc_fail": int((~att["qc_ok"]).sum()),
        "qc_fail_attempts": att.filter(~pl.col("qc_ok")).select("nfl_id", "drill", "session", "attempt", "path",
                                                                "path_rel", "smax").to_dicts(),
        "n_players_without_tracked_40": att.filter(pl.col("c40_smax").is_null())["nfl_id"].n_unique(),
        "n_attempts_used": use.height,
        "n_attempts_dropped_alone_in_session": int((~use["z_ratio"].is_finite().fill_null(False)).sum()),
        "share_peak_at_last_interior_frame": float(att["peak_at_end"].mean()),
        "peak_at_end_by_drill": dict(att.group_by("drill").agg(pl.col("peak_at_end").mean()).sort("drill").iter_rows()),
        "ratio_summary": use.select(pl.col("ratio").min().alias("min"), pl.col("ratio").median().alias("median"),
                                    pl.col("ratio").max().alias("max")).to_dicts()[0],
        "sg_vs_device_peak_r": float(np.corrcoef(att["smax"], att["s_dev_max"])[0, 1]),
        "sessions": {"n": use["session"].n_unique(),
                     "players_per_session_median": float(use.group_by("session").agg(pl.col("nfl_id").n_unique())["nfl_id"].median()),
                     "players_per_session_min": int(use.group_by("session").agg(pl.col("nfl_id").n_unique())["nfl_id"].min())},
        "population": {"n": players.height, "by_position": dict(players.group_by("position").len().sort("position").iter_rows()),
                       "by_year": dict(players.group_by("draft_year").len().sort("draft_year").iter_rows()),
                       "n_drills_median": float(players["n_drills"].median()),
                       "n_drills_by_position": dict(players.group_by("position").agg(pl.col("n_drills").median()).sort("position").iter_rows())},
    }
    return out


# ---------------------------------------------------------------- QC figures
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
YEAR_COLORS = {2023: BLUE, 2024: ORANGE, 2025: AQUA}


def _style():
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})


def figure_traces(frames: pl.DataFrame, att: pl.DataFrame, path: Path) -> None:
    """Speed traces of every position-drill attempt, one panel per drill; the interior peak marked; QC drops orange."""
    _style()
    drills = att["drill"].unique().sort().to_list()
    nc = 7
    nr = int(np.ceil(len(drills) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 2.6, nr * 1.9), sharey=True)
    a = att.select("traj", "drill", "qc_ok", "i_peak", "smax", "peak_at_end")
    fr = (frames.select("traj", "t", "speed").join(a.select("traj", "drill", "qc_ok"), on="traj", maintain_order="left")
          .sort("traj", "t"))
    for ax, d in zip(axes.flat, drills):
        sub = fr.filter(pl.col("drill") == d)
        for (traj,), g in sub.group_by(["traj"], maintain_order=True):
            ok = g["qc_ok"][0]
            ax.plot(g["t"], g["speed"], lw=0.35 if ok else 0.9, color="#8a8984" if ok else ORANGE,
                    alpha=0.35 if ok else 0.95, zorder=1 if ok else 3)
        pk = a.filter(pl.col("drill") == d).join(sub.group_by("traj").agg(pl.col("t").sort()), on="traj")
        tp = [t[i] for t, i in zip(pk["t"].to_list(), pk["i_peak"].to_list())]
        ax.scatter(tp, pk["smax"], s=3, color=BLUE, zorder=2, linewidths=0)
        n = pk.height
        ax.set_title(f"{d.split(':')[0]} {d.split(':')[1][:24].lower()}\nn={n}, peak at end {pk['peak_at_end'].mean():.0%}",
                     fontsize=6.5, color=INK, loc="left")
        ax.tick_params(labelsize=6)
        ax.set_ylim(0, 11)
    for ax in axes.flat[len(drills):]:
        ax.axis("off")
    fig.suptitle("E010 QC: SG speed (yd/s) vs time in attempt (s), every position-drill attempt. Blue dot = interior "
                 "peak (the attempt's drill speed); orange = dropped by the path QC rule", fontsize=9, x=0.01,
                 ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_intensity(use: pl.DataFrame, players: pl.DataFrame, path: Path) -> None:
    """Left: attempt intensity (peak / own 40 peak) by drill, dots coloured by draft year, session medians as ticks.
    Right: player feature by position x draft year, and feature vs the 40 peak."""
    _style()
    drills = use.select("drill").unique().sort("drill")["drill"].to_list()[::-1]
    fig = plt.figure(figsize=(14, 13))
    ax = fig.add_axes((0.25, 0.05, 0.42, 0.9))
    rng = np.random.default_rng(0)
    for k, d in enumerate(drills):
        sub = use.filter(pl.col("drill") == d)
        y = k + rng.uniform(-0.28, 0.28, sub.height)
        ax.scatter(sub["ratio"], y, s=3, c=[YEAR_COLORS[int(v)] for v in sub["draft_year"]], alpha=0.45,
                   linewidths=0)
        sm = sub.group_by("session", "draft_year").agg(pl.col("ratio").median()).sort("session")
        for s in sm.iter_rows(named=True):
            ax.plot([s["ratio"]] * 2, [k - 0.38, k + 0.38], color=YEAR_COLORS[int(s["draft_year"])], lw=1.6)
    ax.set_yticks(range(len(drills)), [d.replace("_", " ").lower()[:46] for d in drills], fontsize=6.5)
    ax.set_xlabel("attempt peak speed / own tracked-40 peak speed")
    ax.set_xlim(0, 1.02)
    ax.set_title("Drill intensity by drill (dots: attempts; ticks: session medians)", loc="left", fontsize=9, color=INK)
    for yv, c in YEAR_COLORS.items():
        ax.scatter([], [], s=12, color=c, label=str(yv))
    ax.legend(title="draft year", fontsize=7, title_fontsize=7, loc="lower left", frameon=False)
    # player feature by position x year
    ax2 = fig.add_axes((0.74, 0.56, 0.24, 0.39))
    pos = F.POSITIONS
    for i, p in enumerate(pos):
        for j, yv in enumerate((2023, 2024, 2025)):
            v = players.filter((pl.col("position") == p) & (pl.col("draft_year") == yv))["c_drill_intensity"].to_numpy()
            x = i + (j - 1) * 0.25
            ax2.scatter(x + rng.uniform(-0.07, 0.07, len(v)), v, s=4, color=YEAR_COLORS[yv], alpha=0.6, linewidths=0)
            if len(v):
                ax2.plot([x - 0.1, x + 0.1], [np.median(v)] * 2, color=INK, lw=1.2)
    ax2.set_xticks(range(len(pos)), pos)
    ax2.set_ylabel("c_drill_intensity (mean drill z)")
    ax2.set_title("Player feature by position and draft year", loc="left", fontsize=9, color=INK)
    ax3 = fig.add_axes((0.74, 0.07, 0.24, 0.39))
    cen = players.with_columns(f=pl.col("c_drill_intensity") - pl.col("c_drill_intensity").mean().over("position"),
                               s=pl.col("c40_smax") - pl.col("c40_smax").mean().over("position"))
    ax3.scatter(cen["s"], cen["f"], s=5, color=BLUE, alpha=0.6, linewidths=0)
    r = float(np.corrcoef(cen["s"], cen["f"])[0, 1])
    ax3.set_xlabel("tracked-40 peak speed, yd/s (position-centered)")
    ax3.set_ylabel("c_drill_intensity (position-centered)")
    ax3.set_title(f"Feature vs its denominator: r = {r:+.2f} within position", loc="left", fontsize=9, color=INK)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# =====================================================================================================
# 2. Game side (only after the stop rule passes)
# =====================================================================================================
FEATURE = "c_drill_intensity"
P1, P2 = "g_play_intensity", "g_snaps_per_season"
CTRL = linkage.COMBINE_CONTROLS + linkage.DRAFT_CONTROLS
HALF_MIN_PLAYS = 25


def build_targets(players: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    pm = F.play_max_speeds(players["nfl_id"].to_list())
    p1 = F.play_intensity(pm)
    p2 = F.career_targets(players)
    df = (players.join(p1.select("nfl_id", "n_plays", "q95", "median", P1, "g_share80"), on="nfl_id", how="left")
          .join(p2, on="nfl_id", how="left").sort("nfl_id"))
    assert df.height == players.height and df[P2].null_count() == 0
    log(f"targets: play-max rows {pm.height}; P1 n {df[P1].is_not_null().sum()}, P2 n {df.height}")
    return df, pm


def _share80(g: pl.DataFrame) -> float:
    if g.height < HALF_MIN_PLAYS:
        return np.nan
    return F.intensity_stats(g["smax"].to_numpy())["share80"]


def target_reliability(df: pl.DataFrame, pm: pl.DataFrame) -> dict:
    out = {"p1": {}, "p1_share80": {}, "p1_team_share": {}}
    s1 = df.filter(pl.col(P1).is_finite())
    pmx = pm.join(s1.select("nfl_id", "position"), on="nfl_id", how="inner")
    for p in F.POSITIONS:
        g = pmx.filter(pl.col("position") == p)
        out["p1"][p] = reliability.split_half_estimator(g, lambda h: F.half_intensity(h, HALF_MIN_PLAYS))
        out["p1_share80"][p] = reliability.split_half_estimator(g, _share80)
    n = dict(s1.group_by("position").len().iter_rows())
    for key in ("p1", "p1_share80"):
        ok = {p: v for p, v in out[key].items() if v["reliability"] is not None and np.isfinite(v["reliability"])}
        out[key]["weighted_by_sample"] = float(sum(ok[p]["reliability"] * n[p] for p in ok) / sum(n[p] for p in ok))
    # team share: per-play value = play max / the player's own q95 (all his plays)
    team = (pl.scan_parquet(io.PROCESSED / "player_play.parquet").select("game_id", "play_id", "nfl_id", "team_abbr")
            .collect())
    tv = (pmx.join(s1.select("nfl_id", "q95"), on="nfl_id").join(team, on=["game_id", "play_id", "nfl_id"], how="inner")
          .with_columns(rel=pl.col("smax") / pl.col("q95")))
    out["p1_team_share"]["n_plays_with_team"] = tv.height
    out["p1_team_share"]["n_plays_total"] = pmx.height
    for p in F.POSITIONS:
        out["p1_team_share"][p] = reliability.team_share(tv.filter(pl.col("position") == p), "rel", ["team_abbr", "season"])
    tvc = tv.with_columns(relc=pl.col("rel") - pl.col("rel").mean().over("position"))
    out["p1_team_share"]["pooled_position_centered"] = reliability.team_share(tvc, "relc", ["team_abbr", "season"])
    # exposure
    cen = lambda c: pl.col(c) - pl.col(c).mean().over("position")
    e = s1.with_columns(lp=pl.col("n_plays").cast(pl.Float64).log()).select(t=cen(P1), s=cen("g_share80"), lp=cen("lp"))
    out["p1_exposure"] = {"r_target_log_plays_within_position": float(np.corrcoef(e["t"], e["lp"])[0, 1]),
                          "r_share80_log_plays_within_position": float(np.corrcoef(e["s"], e["lp"])[0, 1]),
                          "median_plays": float(s1["n_plays"].median())}
    # P2 stability context: season-to-season r of REG off+def snap counts in player_play (position-centered)
    g = io.load("games").select("game_id", "season", "season_type")
    snaps = (team.join(g, on="game_id").filter(pl.col("season_type") == "REG")
             .join(df.select("nfl_id", "draft_year", "position"), on="nfl_id", how="inner")
             .group_by("nfl_id", "season").agg(n=pl.len()))
    rows = []
    for k in (1, 2):  # season k vs k + 1 of the player's career (rookie = 1)
        a = (df.filter(pl.col("draft_year") + k <= 2025).select("nfl_id", "position", "draft_year")
             .with_columns(s1=pl.col("draft_year") + k - 1, s2=pl.col("draft_year") + k)
             .join(snaps.rename({"season": "s1", "n": "n1"}), on=["nfl_id", "s1"], how="left")
             .join(snaps.rename({"season": "s2", "n": "n2"}), on=["nfl_id", "s2"], how="left").fill_null(0)
             .with_columns(c1=cen("n1"), c2=cen("n2")))
        rows.append({"seasons": f"{k}->{k + 1}", "n": a.height, "r": float(np.corrcoef(a["c1"], a["c2"])[0, 1])})
    out["p2_stability"] = {"season_to_season": rows,
                           "note": "snaps are a recorded count; this is stability, not measurement reliability"}
    return out


def ev(df: pl.DataFrame, feature: str, target: str, seed: int = 0, **kw) -> dict:
    return linkage.evaluate(df, feature, target, expected_sign=1, seed=seed, n_jobs=N_JOBS, **{**EVAL, **kw})


def summary(r: dict) -> dict:
    """The fields reported per test."""
    d = r["delta_r2_vs_B2"]
    return {"feature": r["feature"], "target": r["target"], "n": r["n"], "positions": r["positions"],
            "resid_df": r["resid_df"], "raw_r": r["raw_r"], "partial_r": r["partial_r"],
            "partial_r_ci95": r["partial_r_ci95"], "partial_r_mde80_one_sided": r["partial_r_mde80"], "fl_p": r["fl_p"],
            "by_class": r["by_class"], "cv_r2": r["cv_r2"], "delta_r2_vs_B2": d, "cv_perm_p": r["cv_perm_p"],
            "loco": r["loco"], "checks": r["checks"]}


def _sample(df: pl.DataFrame, feature: str, target: str) -> pl.DataFrame:
    return df.filter(pl.col(feature).is_finite() & pl.col(target).is_finite()).sort("nfl_id")


def influence(df: pl.DataFrame, feature: str, target: str, full: bool) -> dict:
    """Leave-3-out both directions (go criterion 6): drop the 3 most supportive / most contrary players by
    x_res * y_res (expected sign +1) under the harness's B2 design and rerun the seed-free Freedman-Lane test."""
    d = _sample(df, feature, target)
    R = linkage._residual_maker(linkage._imputed(linkage._design(d, CTRL)))
    xr, yr = R @ d[feature].to_numpy().astype(float), R @ d[target].to_numpy().astype(float)
    c = xr * yr
    contrib = c / np.sqrt((xr @ xr) * (yr @ yr))
    out = {"top3_supportive_share_of_partial_r": float(np.sort(contrib)[::-1][:3].sum() / contrib.sum())
           if contrib.sum() > 0 else None}
    for name, idx in (("drop3_supportive", np.argsort(-c)[:3]), ("drop3_contrary", np.argsort(c)[:3])):
        ids = d["nfl_id"].to_numpy()[idx].tolist()
        dd = d.filter(~pl.col("nfl_id").is_in(ids))
        r = ev(dd, feature, target, n_repeats=2, n_perm=20)
        out[name] = {"dropped_nfl_id": ids, "dropped_names": d["display_name"].to_numpy()[idx].tolist(), "n": r["n"],
                     "partial_r": r["partial_r"], "partial_r_ci95": r["partial_r_ci95"], "fl_p": r["fl_p"],
                     "passes_c1": bool(r["checks"]["fl_p_lt_05"] and r["checks"]["sign_matches"])}
        if name == "drop3_supportive" and full:
            out[name]["full_harness"] = summary(ev(dd, feature, target))
    return out


def bh(p) -> np.ndarray:
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def resid_reliability(d: pl.DataFrame, col: str, rel_by_pos: dict) -> float:
    """Reliability of the part of `col` left after the B2 design (E008 formula, classical error): per-player error
    variance (1 - rel_pos) Var_pos(col); rel_B2 = 1 - sum s2_i R_ii / ||R col||^2."""
    R = linkage._residual_maker(linkage._imputed(linkage._design(d, CTRL)))
    v = d[col].to_numpy().astype(float)
    pos = d["position"].to_numpy()
    s2 = np.zeros(len(v))
    for p in np.unique(pos):
        m = pos == p
        s2[m] = (1 - rel_by_pos[p]) * np.var(v[m], ddof=1)
    rv = R @ v
    return float(1 - np.sum(s2 * np.diag(R)) / (rv @ rv))


BOUND_SCOPE = {"P1": "feature and target measurement error (beyond-B2 reliabilities); bounds are pooled "
                     "within-position averages",
               "P2": "feature measurement error only (snaps as recorded, rel_y = 1); for snaps read as a playing-time "
                     "trait see post_hoc_review.p2_playing_time_trait; bounds are pooled within-position averages"}


def corrected_ci(ci: list, rel_x: float, rel_y: float) -> list:
    if ci[0] is None or not (rel_x > 0.05 and rel_y > 0.05):
        return [-1.0, 1.0]
    k = np.sqrt(rel_x * rel_y)
    return [float(np.clip(ci[0] / k, -1, 1)), float(np.clip(ci[1] / k, -1, 1))]


def game_stage(C: dict, combine_res: dict) -> dict:
    players = C["players"]
    df, pm = build_targets(players)
    df.write_parquet(DATA / "e010_players.parquet")
    pm.write_parquet(DATA / "e010_play_max.parquet")
    out = {"targets": {
        "p1_n": int(df[P1].is_finite().sum()),
        "p1_by_position": dict(df.filter(pl.col(P1).is_finite()).group_by("position").len().sort("position").iter_rows()),
        "p1_summary": df.select(pl.col(P1).min().alias("min"), pl.col(P1).median().alias("median"),
                                pl.col(P1).max().alias("max"), pl.col(P1).std().alias("sd")).to_dicts()[0],
        "p2_n": df.height, "p2_zero_share": float((df[P2] == 0).mean()),
        "p2_summary": df.select(pl.col(P2).median().alias("median"), pl.col(P2).mean().alias("mean"),
                                pl.col(P2).max().alias("max")).to_dicts()[0]}}
    trel = target_reliability(df, pm)
    out["target_reliability"] = trel
    log(f"P1 split-half (sample-weighted) {trel['p1']['weighted_by_sample']:.3f}; share80 "
        f"{trel['p1_share80']['weighted_by_sample']:.3f}; team share pooled "
        f"{trel['p1_team_share']['pooled_position_centered']['team_share']:.3f}; exposure r "
        f"{trel['p1_exposure']['r_target_log_plays_within_position']:+.3f}")
    tests = []  # flat list: every test, with its label

    def run(tid, label, data, feature, target, family=None, seeds=(0,), note=None):
        t = time.time()
        res = {s: ev(data, feature, target, seed=s) for s in seeds}
        r0 = res[seeds[0]]
        rec = {"id": tid, "label": label, "family": family, "note": note, **summary(r0)}
        if len(seeds) > 1:
            rec["seeds"] = {s: {"fl_p": r["fl_p"], "excess": r["delta_r2_vs_B2"]["excess"], "se": r["delta_r2_vs_B2"]["se"],
                                "cv_perm_p": r["cv_perm_p"], "checks": r["checks"]} for s, r in res.items()}
        tests.append(rec)
        pr = r0["partial_r"]
        log(f"{tid:28s} n={r0['n']:3d} raw={r0['raw_r']:+.3f} partial={pr if pr is None else round(pr, 3)} "
            f"ci={r0['partial_r_ci95']} p={r0['fl_p']} excess={r0['delta_r2_vs_B2']['excess']:+.4f}"
            f"±{r0['delta_r2_vs_B2']['se']:.4f} loco={r0['checks']['loco_classes_pass']} ({time.time() - t:.0f}s)")
        return rec

    # ---- primaries (seeds 0, 1, 2 are prespecified robustness; seed 0 is the reported value)
    prim = {"P1": run("P1", "primary", df, FEATURE, P1, family="primary", seeds=(0, 1, 2)),
            "P2": run("P2", "primary", df, FEATURE, P2, family="primary", seeds=(0, 1, 2))}
    q = bh([prim["P1"]["fl_p"], prim["P2"]["fl_p"]])
    prim["P1"]["bh_q_primaries"], prim["P2"]["bh_q_primaries"] = float(q[0]), float(q[1])
    # ---- prespecified robustness
    infl = {}
    for k, tgt in (("P1", P1), ("P2", P2)):
        c1 = bool(prim[k]["checks"]["fl_p_lt_05"] and prim[k]["checks"]["sign_matches"])
        infl[k] = influence(df, FEATURE, tgt, full=c1)
        log(f"{k} leave-3-out: supportive p {infl[k]['drop3_supportive']['fl_p']:.4f} "
            f"(r {infl[k]['drop3_supportive']['partial_r']:+.3f}); contrary p {infl[k]['drop3_contrary']['fl_p']:.4f}")
    no25 = df.filter(pl.col("draft_year") != 2025)
    for k, tgt in (("P1", P1), ("P2", P2)):
        run(f"{k}_without_2025", "prespecified robustness", no25, FEATURE, tgt)
    for k, tgt in (("P1", P1), ("P2", P2)):
        for p in F.POSITIONS:
            run(f"{k}_{p}", "prespecified robustness (by position, descriptive)", df.filter(pl.col("position") == p),
                FEATURE, tgt)
    for k, tgt in (("P1", P1), ("P2", P2)):
        run(f"{k}_feature_no_division", "prespecified robustness", df, "c_drill_speed_z", tgt)
        run(f"{k}_feature_median", "prespecified robustness", df, "c_drill_intensity_median", tgt)
    run("P1_target_share80", "prespecified robustness", df, FEATURE, "g_share80")
    run("P2_target_log", "prespecified robustness", df, FEATURE, "g_log_snaps_per_season")
    run("P2_target_with_st", "prespecified robustness", df, FEATURE, "g_snaps_st_per_season")
    # ---- prelisted exploratory
    run("X1_starts_per_season", "prelisted exploratory", df, FEATURE, "g_starts_per_season", family="exploratory")
    hon = df.with_columns(fc=pl.col(FEATURE) - pl.col(FEATURE).mean().over("position"))
    honored = hon.filter(pl.col("g_honors") > 0)
    out["X2_honors_descriptive"] = {
        "label": "prelisted exploratory (descriptive only, no test)",
        "n_honored": honored.height, "n_total": hon.height,
        "honored": honored.select("display_name", "position", "draft_year", "g_honors", "fc").sort("fc").to_dicts(),
        "mean_feature_within_position_honored": float(honored["fc"].mean()) if honored.height else None,
        "mean_feature_within_position_rest": float(hon.filter(pl.col("g_honors") == 0)["fc"].mean())}
    # ---- bounds: reliability-corrected intervals (beyond-B2 reliabilities)
    bounds = {}
    for k, tgt in (("P1", P1), ("P2", P2)):
        d = _sample(df, FEATURE, tgt)
        rx = reliability.cross_drill_icc(_resid_rows(C["dl"], d, CTRL), "v", drill="drill", zscore=True,
                                         center_by=["drill", "session"])["reliability"]
        if k == "P1":
            rel_pos = {p: trel["p1"][p]["reliability"] for p in F.POSITIONS}
            ry = resid_reliability(d, tgt, rel_pos)
        else:
            ry = 1.0  # a recorded count: no measurement error in the outcome as defined
        bounds[k] = {"rel_x_beyond_B2": rx, "rel_y_beyond_B2": ry, "corrected_ci95_scope": BOUND_SCOPE[k],
                     "corrected_ci95": corrected_ci(prim[k]["partial_r_ci95"], rx, ry),
                     "one_sided_upper_95": float(np.tanh(np.arctanh(prim[k]["partial_r"])
                                                         + norm.ppf(0.95) / np.sqrt(prim[k]["resid_df"] - 1)))}
    # ---- go criteria for the primaries
    rel_x = combine_res["stop_rule"]["reliability"]
    rel_y = {"P1": trel["p1"]["weighted_by_sample"], "P2": 1.0}
    crit = {}
    for k in ("P1", "P2"):
        r = prim[k]
        c = {"c1_fl_p_and_sign": bool(r["checks"]["fl_p_lt_05"] and r["checks"]["sign_matches"]),
             "c2_excess_lb_gt_0_signed": bool(r["checks"]["excess_lb_gt_0"]),
             "c3_loco_2of3": int(r["checks"]["loco_classes_pass"].split("/")[0]) >= 2,
             "c4_bh_q_lt_05": r["bh_q_primaries"] < 0.05,
             "c5_reliability_ge_05": bool(rel_x >= 0.5 and rel_y[k] >= 0.5)}
        seeds_ok = all(v["checks"]["fl_p_lt_05"] and v["checks"]["excess_lb_gt_0"]
                       and int(v["checks"]["loco_classes_pass"].split("/")[0]) >= 2 for v in r["seeds"].values())
        c["c6_seeds_and_leave3out"] = bool(seeds_ok and infl[k]["drop3_supportive"]["passes_c1"])
        c["c7_scout_sentence"] = "judgment; see decision"
        crit[k] = c
    # ---- POST HOC (added after the primaries were seen; descriptive, no p-values, in no BH family): in-sample partial r
    # of the feature with each target given nested baselines, to show where the raw correlation goes
    ph = {}
    for k, tgt in (("P1", P1), ("P2", P2)):
        d = _sample(df, FEATURE, tgt)
        x, y = d[FEATURE].to_numpy().astype(float), d[tgt].to_numpy().astype(float)
        row = {"raw_r": float(np.corrcoef(x, y)[0, 1])}
        for name, cols in (("B0_position_year", []), ("B1_plus_combine", linkage.COMBINE_CONTROLS), ("B2_plus_draft", CTRL)):
            R = linkage._residual_maker(linkage._imputed(linkage._design(d, cols)))
            row[name] = float(np.corrcoef(R @ x, R @ y)[0, 1])
        ph[k] = row
    out["post_hoc_nested_baselines"] = {"label": "post hoc, descriptive (no test)", **ph}
    # ---- multiplicity across every p-valued E010 test
    pv = [(t["id"], t["fl_p"]) for t in tests if t["fl_p"] is not None]
    qa = bh([p for _, p in pv])
    out.update({"tests": tests, "primaries": prim, "influence": infl, "bounds": bounds, "criteria": crit,
                "bh_all_e010_tests": {i: float(qq) for (i, _), qq in zip(pv, qa)}, "n_tests_with_p": len(pv)})
    return out, df


# =====================================================================================================
# 3. POST HOC, after the two skeptic reviews (2026-10-15). Everything here was added after the primaries and both
#    reviews were seen. Tests with a p-value are in no prespecified BH family; they are counted only in
#    bh_all_including_post_hoc. The one exception is the literal reading of the prereg's 'drill-level median'
#    variant, a prespecified robustness check that was run late.
# =====================================================================================================
POST_HOC_CHOICES = {
    "literal_median": "prereg robustness 'drill-level median instead of mean', literal reading: drill value = median "
                      "of the player's attempt z in that drill, player value = mean over >= 2 drills. Full harness "
                      "settings (seed 0). The implemented variant (player-level median over drills) is kept as run.",
    "post_hoc_tests": "linkage.evaluate, one-sided +, seed-free Freedman-Lane (n_perm_fl as the primaries); its CV parts "
                      "run at minimal settings (n_repeats 2, n_perm 20) and are not reported.",
    "team_share": "reliability.team_share on per-play play max / own q95: (a) the reported pooled construction "
                  "(position-centered values, team_col [team_abbr, season]: most teammate pairs are cross-position); "
                  "(b) same-position pairs only (team_col [team_abbr, season, position]); (c) the inverse-variance "
                  "mean of the per-position estimates; (d) (b) on the role-adjusted per-play value.",
    "play_context": "f_drill_intensity.play_context: cell = position | lined_up_position | run (player_play row with "
                    "pass_result null) | route_ran | coverage (man / rush / deepzone / shortzone / na), from player_play.",
    "role_adjusted_p1": "f_drill_intensity.role_adjusted_intensity: median(play max - cell mean + position mean) / q95 "
                        "(raw), >= 50 plays; cell means over the P1 population's plays.",
    "role_mix": "f_drill_intensity.ROLE_COVARIATES (shares over plays with a player_play row); as controls they enter "
                "position-specific (0 for other positions).",
    "sprint_controls": "c40_smax and c40_tau (speed.combine_speed_table) added to the B1 block.",
    "censoring_types": "per drill, share pe of used attempts whose interior peak is the last interior frame: "
                       "censored-type pe >= 0.6, true-peak pe < 0.3, mixed otherwise (the football skeptic's cuts). "
                       "True-peak feature = mean of the player's true-peak drill values, >= 2 such drills; threshold "
                       "sensitivity with pe < 0.5 (the statistics skeptic's cut). The censored-vs-uncensored z comparison "
                       "uses drills with 0.15 <= pe <= 0.85.",
    "heterogeneity": "OLS F-test of feature x position interactions added to B2 + feature (P1 without TE, n 20).",
    "bootstrap": "position-stratified case bootstrap of the pooled partial r, 2,000 replicates, seed 0; the design is "
                 "rebuilt (and re-imputed) in every replicate.",
    "p2_playing_time": "season-1 vs season-2 REG player_play row counts (classes 2023-24), correlated beyond B0 and "
                       "beyond B2, Spearman-Brown to each player's seasons available (2026 - draft_year) and averaged. "
                       "WR/TE rows are pass plays only, so this is a proxy for the snap count.",
    "p2_decomposition": "partial r of the feature with P2 given B0 plus one block at a time (descriptive).",
}
PH_EVAL = dict(n_repeats=2, n_perm=20)
B0_COLS: list[str] = []


def _upper(r: float, resid_df: int) -> float:
    """One-sided 95% upper bound of a partial r (Fisher z)."""
    return float(np.tanh(np.arctanh(r) + norm.ppf(0.95) / np.sqrt(resid_df - 1)))


def _partial(d: pl.DataFrame, x: str, y: str, cols: list[str]) -> dict:
    """In-sample partial r of x with y given position + draft-year dummies + cols (harness design), Fisher 95% CI;
    descriptive, no p."""
    d = d.filter(pl.col(x).is_finite() & pl.col(y).is_finite()).sort("nfl_id")
    X = linkage._imputed(linkage._design(d, cols))
    R = linkage._residual_maker(X)
    rx, ry = R @ d[x].to_numpy().astype(float), R @ d[y].to_numpy().astype(float)
    r = float(np.corrcoef(rx, ry)[0, 1])
    se = 1 / np.sqrt(d.height - np.linalg.matrix_rank(X) - 2)
    return {"n": d.height, "partial_r": r, "ci95": [float(np.tanh(np.arctanh(r) - 1.96 * se)),
                                                    float(np.tanh(np.arctanh(r) + 1.96 * se))]}


def _cen_r(d: pl.DataFrame, a: str, b: str) -> float:
    """Within-position correlation (both columns centered by position)."""
    d = d.filter(pl.col(a).is_finite() & pl.col(b).is_finite())
    c = d.select((pl.col(a) - pl.col(a).mean().over("position")).alias("a"),
                 (pl.col(b) - pl.col(b).mean().over("position")).alias("b"))
    return float(np.corrcoef(c["a"], c["b"])[0, 1])


def _resid(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]


def heterogeneity_f(d: pl.DataFrame, feature: str, target: str) -> dict:
    """F-test of feature x position interactions added to B2 + feature (OLS)."""
    from scipy.stats import f as f_dist
    X = linkage._imputed(linkage._design(d, CTRL))
    x, y, pos = d[feature].to_numpy().astype(float), d[target].to_numpy().astype(float), d["position"].to_numpy()
    lv = sorted(set(pos))
    X0 = np.column_stack([X, x])
    X1 = np.column_stack([X0] + [x * (pos == p) for p in lv[1:]])
    rss0, rss1 = (float(np.sum(_resid(M, y) ** 2)) for M in (X0, X1))
    q, dfr = len(lv) - 1, len(y) - np.linalg.matrix_rank(X1)
    F_ = ((rss0 - rss1) / q) / (rss1 / dfr)
    return {"positions": lv, "n": len(y), "F": F_, "df": [q, int(dfr)], "p": float(f_dist.sf(F_, q, dfr))}


def stratified_bootstrap(d: pl.DataFrame, feature: str, target: str, reps: int = 2000, seed: int = 0) -> dict:
    """Position-stratified case bootstrap of the pooled B2 partial r."""
    rng = np.random.default_rng(seed)
    pos = d["position"].to_numpy()
    groups = [np.flatnonzero(pos == p) for p in sorted(set(pos))]
    out = np.empty(reps)
    for b in range(reps):
        db = d[np.concatenate([rng.choice(g, len(g)) for g in groups])]
        X = linkage._imputed(linkage._design(db, CTRL))
        out[b] = np.corrcoef(_resid(X, db[feature].to_numpy().astype(float)),
                             _resid(X, db[target].to_numpy().astype(float)))[0, 1]
    return {"reps": reps, "p2_5": float(np.percentile(out, 2.5)), "p95_one_sided_upper": float(np.percentile(out, 95)),
            "p97_5": float(np.percentile(out, 97.5))}


def post_hoc_review(C: dict, df: pl.DataFrame, pm: pl.DataFrame, G: dict) -> dict:
    out = {"label": "post hoc, after the skeptic reviews (2026-10-15); descriptive unless a p is shown; no p here is "
                    "in a prespecified BH family", "choices": POST_HOC_CHOICES}
    tests = []

    def ph(tid, data, feature, target, extra=(), note=None):
        r = ev(data, feature, target, combine_controls=linkage.COMBINE_CONTROLS + list(extra), **PH_EVAL)
        pr = r["partial_r"]
        rec = {"id": tid, "label": "post hoc (after review)", "note": note, "feature": feature, "target": target,
               "extra_controls": list(extra), "n": r["n"], "resid_df": r["resid_df"], "partial_r": pr,
               "partial_r_ci95": r["partial_r_ci95"], "fl_p": r["fl_p"], "partial_r_mde80_one_sided": r["partial_r_mde80"],
               "one_sided_upper_95": _upper(pr, r["resid_df"]) if pr is not None else None}
        tests.append(rec)
        log(f"[post hoc] {tid:34s} n={r['n']:3d} partial={pr if pr is None else round(pr, 3)} ci={r['partial_r_ci95']} "
            f"p={r['fl_p']}")
        return rec

    c40 = C["c40"].select("nfl_id", "c40_tau", "c40_vmax")
    d = df.join(c40, on="nfl_id", how="left").sort("nfl_id")
    s1 = d.filter(pl.col(P1).is_finite())

    # ---- 1. literal 'drill-level median' (prespecified robustness, run late)
    use_pop = C["use"].filter(pl.col("nfl_id").is_in(d["nfl_id"].implode()))
    dl_med = F.drill_level(use_pop, "ratio", agg="median")
    lit = F.player_feature(dl_med, "c_drill_intensity_drill_median")
    d = d.join(lit.select("nfl_id", "c_drill_intensity_drill_median"), on="nfl_id", how="left")
    cmp = dl_med.join(C["dl"].select("nfl_id", "drill", vmean="v"), on=["nfl_id", "drill"])
    lit_res = {"label": "prespecified robustness, literal reading of the prereg's 'drill-level median' (run after review)",
               "n_player_drill_rows": cmp.height,
               "n_rows_median_differs_from_mean": int(((cmp["v"] - cmp["vmean"]).abs() > 1e-12).sum()),
               "r_with_primary_feature": float(np.corrcoef(*d.select(FEATURE, "c_drill_intensity_drill_median")
                                                           .to_numpy().T)[0, 1])}
    for k, tgt in (("P1", P1), ("P2", P2)):
        t0 = time.time()
        r = ev(d, "c_drill_intensity_drill_median", tgt)
        lit_res[k] = {"id": f"{k}_feature_drill_median_literal", **summary(r)}
        log(f"{k}_feature_drill_median_literal partial={r['partial_r']:+.3f} ci={r['partial_r_ci95']} p={r['fl_p']} "
            f"({time.time() - t0:.0f}s)")
    out["literal_drill_median"] = lit_res

    # ---- 2. what P1 measures: coverage, role mix, role-adjusted P1
    ctx = F.play_context(pm, s1)
    cs = io.load("player_career_successes").select(
        "nfl_id", snaps=(pl.col("career_offensive_snaps") + pl.col("career_defensive_snaps")).cast(pl.Float64))
    cov = (s1.join(cs, on="nfl_id").filter(pl.col("snaps") > 0)
           .with_columns(tracked_per_snap=pl.col("n_plays").cast(pl.Float64) / pl.col("snaps")))
    out["p1_coverage"] = {
        "share_tracked_plays_without_player_play_row": float((~ctx["has_row"]).mean()),
        "run_share_of_plays_with_row": dict(ctx.filter(pl.col("has_row")).group_by("position")
                                            .agg(pl.col("run").mean()).sort("position").iter_rows()),
        "tracked_plays_per_career_off_def_snap_median": dict(cov.group_by("position")
                                                             .agg(pl.col("tracked_per_snap").median())
                                                             .sort("position").iter_rows()),
        "note": "WR/TE player_play REG rows are pass plays only, so P1 is per route for WR/TE and per snap for DB/DL/OL"}
    roles = F.role_mix(ctx)
    d = d.join(roles, on="nfl_id", how="left")
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_predict
    rm = {}
    extra_roles = []
    for p, cols in F.ROLE_COVARIATES.items():
        s = d.filter((pl.col("position") == p) & pl.col(P1).is_finite())
        X, y = s.select(cols).fill_null(0).to_numpy(), s[P1].to_numpy()
        cv = [1 - np.mean((y - cross_val_predict(LinearRegression(), X, y, cv=KFold(5, shuffle=True, random_state=k)))
                          ** 2) / np.var(y) for k in range(20)]
        rm[p] = {"n": s.height, "covariates": cols, "r2_in_sample": float(LinearRegression().fit(X, y).score(X, y)),
                 "r2_cv_5fold_20rep": float(np.mean(cv)),
                 "r_with_p1": {c: float(np.corrcoef(s[c].fill_null(0), y)[0, 1]) for c in cols},
                 "r_with_feature": {c: float(np.corrcoef(s[c].fill_null(0), s[FEATURE])[0, 1]) for c in cols}}
        for c in cols:
            name = f"role_{p}_{c}"
            d = d.with_columns(pl.when(pl.col("position") == p).then(pl.col(c).fill_null(0)).otherwise(0.0).alias(name))
            extra_roles.append(name)
    out["p1_role_mix_within_position"] = rm
    ra = F.role_adjusted_intensity(ctx)
    d = d.join(ra, on="nfl_id", how="left")
    ctxa = F.add_role_adjusted(ctx)

    def half_adj(g: pl.DataFrame) -> float:
        if g.height < HALF_MIN_PLAYS:
            return np.nan
        return float(np.median(g["smax_adj"].to_numpy()) / pl.Series(g["smax"]).quantile(0.95))

    out["p1_role_adjusted"] = {
        "n": int(d["g_p1_role_adjusted"].is_finite().sum()),
        "n_cells": ctx["cell"].n_unique(),
        "r_with_p1_within_position": _cen_r(d, P1, "g_p1_role_adjusted"),
        "split_half_by_position": {p: reliability.split_half_estimator(ctxa.filter(pl.col("position") == p), half_adj)
                                   for p in F.POSITIONS}}
    ph("P1_role_adjusted", d, FEATURE, "g_p1_role_adjusted")
    ph("P1_given_role_mix", d, FEATURE, P1, extra=extra_roles, note="B2 + position-specific role-mix covariates")
    for p in F.POSITIONS:
        ph(f"P1_role_adjusted_{p}", d.filter(pl.col("position") == p), FEATURE, "g_p1_role_adjusted",
           note="by position, descriptive")

    # ---- 3. team share: same-position pairs
    team = (pl.scan_parquet(io.PROCESSED / "player_play.parquet").select("game_id", "play_id", "nfl_id", "team_abbr")
            .collect())
    tv = (pm.join(s1.select("nfl_id", "position", "q95"), on="nfl_id")
          .join(team, on=["game_id", "play_id", "nfl_id"], how="inner")
          .with_columns(rel=pl.col("smax") / pl.col("q95")))
    tv = tv.with_columns(relc=pl.col("rel") - pl.col("rel").mean().over("position"))
    tvh = (tv.with_columns(_h=pl.col("game_id").rank("dense").over("team_abbr", "season") % 2)
           .group_by("nfl_id", "team_abbr", "season", "position")
           .agg(na=(pl.col("_h") == 0).sum(), nb=(pl.col("_h") == 1).sum())
           .filter((pl.col("na") >= 10) & (pl.col("nb") >= 10)))
    k_all = tvh.group_by("team_abbr", "season").agg(k=pl.len().cast(pl.Float64))
    k_pos = tvh.group_by("team_abbr", "season", "position").agg(k=pl.len().cast(pl.Float64))
    per = G["target_reliability"]["p1_team_share"]
    w = {p: 1 / per[p]["se"] ** 2 for p in F.POSITIONS if per[p]["se"] is not None and np.isfinite(per[p]["se"])}
    # role-adjusted per-play value, centered within its play-context cell (team_share centers globally, so values
    # must carry no position level)
    adj = (ctx.filter(pl.col("has_row")).with_columns(rel=pl.col("smax") / pl.col("q95"))
           .with_columns(rel_adj=pl.col("rel") - pl.col("rel").mean().over("cell")))
    out["p1_team_share"] = {
        "pooled_position_centered_as_reported": per["pooled_position_centered"],
        "ordered_teammate_pairs_pooled": int((k_all["k"] * (k_all["k"] - 1)).sum()),
        "ordered_teammate_pairs_same_position": int((k_pos["k"] * (k_pos["k"] - 1)).sum()),
        "same_position_pairs": reliability.team_share(tv, "relc", ["team_abbr", "season", "position"]),
        "inverse_variance_mean_of_positions": {
            "team_share": float(sum(w[p] * per[p]["team_share"] for p in w) / sum(w.values())),
            "se": float(1 / np.sqrt(sum(w.values()))), "positions": list(w)},
        "role_adjusted_same_position_pairs": reliability.team_share(adj, "rel_adj", ["team_abbr", "season", "position"]),
        "note": "team_share cannot separate a player from his within-team role"}

    # ---- 4. pooled bounds vs positions: heterogeneity, bootstrap, per-position upper bounds
    pb = {}
    for k, tgt in (("P1", P1), ("P2", P2)):
        smp = _sample(d, FEATURE, tgt)
        het = smp.filter(pl.col("position") != "TE") if k == "P1" else smp
        byp = {}
        for p in F.POSITIONS:
            t = next(t for t in G["tests"] if t["id"] == f"{k}_{p}")
            if t["partial_r"] is not None:
                byp[p] = {"partial_r": t["partial_r"], "ci95": t["partial_r_ci95"], "fl_p": t["fl_p"],
                          "one_sided_upper_95": _upper(t["partial_r"], t["resid_df"])}
        pb[k] = {"heterogeneity_F": heterogeneity_f(het, FEATURE, tgt),
                 "bootstrap": stratified_bootstrap(smp, FEATURE, tgt),
                 "fisher_one_sided_upper_95": G["bounds"][k]["one_sided_upper_95"], "by_position": byp}
        log(f"[post hoc] {k} heterogeneity p {pb[k]['heterogeneity_F']['p']:.3f}; bootstrap upper "
            f"{pb[k]['bootstrap']['p95_one_sided_upper']:+.3f}")
    out["pooled_vs_position_bounds"] = pb

    # ---- 5. P2 read as a playing-time trait: beyond-B2 season-to-season stability, corrected interval
    g = io.load("games").select("game_id", "season", "season_type")
    sn = (team.join(g, on="game_id").filter(pl.col("season_type") == "REG")
          .group_by("nfl_id", "season").agg(n=pl.len()))
    st = (d.filter(pl.col("draft_year") <= 2024)
          .with_columns(s_1=pl.col("draft_year"), s_2=pl.col("draft_year") + 1)
          .join(sn.rename({"season": "s_1", "n": "n1"}), on=["nfl_id", "s_1"], how="left")
          .join(sn.rename({"season": "s_2", "n": "n2"}), on=["nfl_id", "s_2"], how="left")
          .with_columns(pl.col("n1", "n2").fill_null(0).cast(pl.Float64)).sort("nfl_id"))
    seasons = (F.SEASONS_AVAILABLE_THROUGH - d["draft_year"].to_numpy()).astype(float)
    p2s = {"n": st.height}
    for lab, cols in (("beyond_B0", B0_COLS), ("beyond_B2", CTRL)):
        r = _partial(st, "n1", "n2", cols)["partial_r"]
        p2s[lab] = {"r_season1_season2": r,
                    "spearman_brown_to_seasons_mean": float(np.mean(seasons * r / (1 + (seasons - 1) * r)))}
    rx = G["bounds"]["P2"]["rel_x_beyond_B2"]
    ry = p2s["beyond_B2"]["spearman_brown_to_seasons_mean"]
    pr2 = G["primaries"]["P2"]
    p2s["corrected_ci95_playing_time_trait"] = corrected_ci(pr2["partial_r_ci95"], rx, ry)
    p2s["corrected_one_sided_upper_95_playing_time_trait"] = float(G["bounds"]["P2"]["one_sided_upper_95"]
                                                                  / np.sqrt(rx * ry))
    p2s["corrected_ci95_feature_error_only"] = G["bounds"]["P2"]["corrected_ci95"]
    p2s["corrected_one_sided_upper_95_feature_error_only"] = float(G["bounds"]["P2"]["one_sided_upper_95"] / np.sqrt(rx))
    out["p2_playing_time_trait"] = p2s

    # ---- 6. where P2's raw negative r goes (descriptive)
    dec = {"raw_r": float(np.corrcoef(*_sample(d, FEATURE, P2).select(FEATURE, P2).to_numpy().T)[0, 1])}
    for lab, cols in (("B0", []), ("B0+forty", ["forty"]), ("B0+c40_smax", ["c40_smax"]),
                      ("B0+ngs_athleticism_score", ["ngs_athleticism_score"]),
                      ("B0+weight_height", ["combine_weight", "combine_height"]),
                      ("B0+B1_all_combine", linkage.COMBINE_CONTROLS), ("B0+draft_slot_only", linkage.DRAFT_CONTROLS),
                      ("B2", CTRL)):
        dec[lab] = _partial(d, FEATURE, P2, cols)["partial_r"]
    dec["path_r_feature_forty_given_B0"] = _partial(d, FEATURE, "forty", [])["partial_r"]
    dec["path_r_p2_forty_given_B0"] = _partial(d, P2, "forty", [])["partial_r"]
    out["p2_raw_r_decomposition"] = dec

    # ---- 7. sprint capacity inside the feature
    sp = {f"partial_r_feature_{c}_given_B2": _partial(d, FEATURE, c, CTRL) for c in ("c40_smax", "c40_vmax", "c40_tau")}
    sp["r2_feature_on_tau_vmax_weight_within_position"] = {}
    for p in F.POSITIONS:
        s = d.filter((pl.col("position") == p) & pl.col("c40_tau").is_finite() & pl.col("c40_vmax").is_finite())
        X, y = s.select("c40_tau", "c40_vmax", "combine_weight").fill_null(s["combine_weight"].median()).to_numpy(), \
            s[FEATURE].to_numpy()
        sp["r2_feature_on_tau_vmax_weight_within_position"][p] = {"n": s.height, "r2": float(
            LinearRegression().fit(X, y).score(X, y))}
    sprint = CTRL + ["c40_smax", "c40_tau"]
    Rs = linkage._residual_maker(linkage._imputed(linkage._design(d, sprint)))
    f = d[FEATURE].to_numpy()
    sp["feature_r2_on_B2_smax_tau_in_sample"] = float(1 - (Rs @ f) @ (Rs @ f) / ((f - f.mean()) @ (f - f.mean())))
    sp["feature_reliability_beyond_B2_smax_tau"] = reliability.cross_drill_icc(
        _resid_rows(C["dl"], d, sprint), "v", drill="drill", zscore=True, center_by=["drill", "session"])
    out["sprint_capacity"] = sp
    for k, tgt in (("P1", P1), ("P2", P2)):
        ph(f"{k}_given_sprint_profile", d, FEATURE, tgt, extra=["c40_smax", "c40_tau"], note="B2 + c40_smax + c40_tau")

    # ---- 8. censoring: what the feature is in each drill type
    ct = F.censoring_types(use_pop.filter(pl.col("z_ratio").is_finite()))
    dlt = C["dl"].join(ct.select("drill", "ctype", "pe"), on="drill", how="left")
    halves = (dlt.filter(pl.col("ctype") != "mixed").group_by("nfl_id", "ctype").agg(v=pl.col("v").mean(), k=pl.len())
              .pivot(on="ctype", index="nfl_id", values=["v", "k"]))
    hd = d.select("nfl_id", "position", FEATURE).join(halves, on="nfl_id", how="left")
    by_pos_halves = {}
    for p in F.POSITIONS:
        s = hd.filter((pl.col("position") == p) & pl.col("v_censored").is_finite() & pl.col("v_true_peak").is_finite())
        by_pos_halves[p] = {"n": s.height, "r": float(np.corrcoef(s["v_censored"], s["v_true_peak"])[0, 1])
                            if s.height > 10 else None}
    mixed = (use_pop.filter(pl.col("z_ratio").is_finite())
             .join(ct.filter(pl.col("pe").is_between(*F.BOTH_KINDS)).select("drill", "pe"), on="drill")
             .group_by("drill").agg(pe=pl.col("pe").first(), z_cens=pl.col("z_ratio").filter(pl.col("peak_at_end")).mean(),
                                    z_unc=pl.col("z_ratio").filter(~pl.col("peak_at_end")).mean(),
                                    dur_cens=pl.col("dur").filter(pl.col("peak_at_end")).median(),
                                    dur_unc=pl.col("dur").filter(~pl.col("peak_at_end")).median()).sort("drill"))
    cc = (use_pop.filter(pl.col("z_ratio").is_finite())
          .with_columns(c=pl.col("peak_at_end").cast(pl.Float64))
          .with_columns(cc=pl.col("c") - pl.col("c").mean().over("drill", "session"))
          .group_by("nfl_id").agg(cens_share_centered=pl.col("cc").mean()))
    tp = (dlt.filter(pl.col("ctype") == "true_peak").group_by("nfl_id")
          .agg(c_drill_intensity_true_peak=pl.col("v").mean(), k=pl.len()).filter(pl.col("k") >= F.MIN_DRILLS))
    d = d.join(tp.select("nfl_id", "c_drill_intensity_true_peak"), on="nfl_id", how="left")
    out["censoring"] = {
        "drill_types": ct.to_dicts(),
        "share_of_drill_rows_censored_type_by_position": dict(
            dlt.group_by("position").agg((pl.col("ctype") == "censored").mean()).sort("position").iter_rows()),
        "r_censored_half_vs_true_peak_half_within_position": by_pos_halves,
        "reliability_censored_type_only": reliability.cross_drill_icc(
            dlt.filter(pl.col("ctype") == "censored"), "v", drill="drill", zscore=True, center_by=["drill", "session"]),
        "reliability_true_peak_only": reliability.cross_drill_icc(
            dlt.filter(pl.col("ctype") == "true_peak"), "v", drill="drill", zscore=True, center_by=["drill", "session"]),
        "drills_with_both_attempt_kinds": {
            "definition": "drills whose share of peak-at-end attempts is within F.BOTH_KINDS (0.15-0.85)",
            "n": mixed.height, "n_censored_attempts_score_higher": int((mixed["z_cens"] > mixed["z_unc"]).sum()),
            "rows": mixed.to_dicts()},
        "r_feature_session_centered_censoring_share_within_position": _cen_r(d.join(cc, on="nfl_id", how="left"),
                                                                            FEATURE, "cens_share_centered"),
        "true_peak_feature": {"n": int(d["c_drill_intensity_true_peak"].is_finite().sum()),
                              "r_with_primary_feature_within_position": _cen_r(d, FEATURE, "c_drill_intensity_true_peak")}}
    # threshold sensitivity: true-peak drills = peak at end in < 50% of attempts (the statistics skeptic's cut)
    tp50 = (dlt.filter(pl.col("pe") < 0.5).group_by("nfl_id").agg(c_drill_intensity_peak_lt50=pl.col("v").mean(),
                                                                   k=pl.len()).filter(pl.col("k") >= F.MIN_DRILLS))
    d = d.join(tp50.select("nfl_id", "c_drill_intensity_peak_lt50"), on="nfl_id", how="left")
    for k, tgt in (("P1", P1), ("P2", P2)):
        ph(f"{k}_feature_true_peak_drills", d, "c_drill_intensity_true_peak", tgt)
        ph(f"{k}_feature_true_peak_given_sprint", d, "c_drill_intensity_true_peak", tgt, extra=["c40_smax", "c40_tau"],
           note="true-peak feature; B2 + c40_smax + c40_tau")
        ph(f"{k}_feature_peak_lt50_drills", d, "c_drill_intensity_peak_lt50", tgt,
           note="threshold sensitivity: drills with peak at end in < 50% of attempts")
    out["tests"] = tests

    # ---- 9. multiplicity: every p-valued E010 test, prespecified and post hoc
    pv = ([(t["id"], t["fl_p"]) for t in G["tests"] if t["fl_p"] is not None]
          + [(lit_res[k]["id"], lit_res[k]["fl_p"]) for k in ("P1", "P2")]
          + [(t["id"], t["fl_p"]) for t in tests if t["fl_p"] is not None])
    qa = bh([p for _, p in pv])
    out["bh_all_including_post_hoc"] = {"n": len(pv), "min_p": float(min(p for _, p in pv)),
                                        "q": {i: float(qq) for (i, _), qq in zip(pv, qa)}}
    return out


# ---------------------------------------------------------------- result figures

def figure_results(df: pl.DataFrame, G: dict, path: Path) -> None:
    """Added-variable plots for P1 and P2, and a forest plot of every test's partial r with its 95% CI: the
    preregistered tests (filled) and the post hoc checks added after review (hollow)."""
    _style()
    fig = plt.figure(figsize=(14, 9.6))
    for i, (k, tgt, lab) in enumerate((("P1", P1, "P1 median / q95 of play max speed\n(WR/TE per route; DB/DL/OL per snap)"),
                                       ("P2", P2, "P2 off+def snaps per available season"))):
        ax = fig.add_axes((0.075 + i * 0.24, 0.42, 0.17, 0.48))
        pr = linkage.partial_residuals(df, FEATURE, tgt)
        ax.scatter(pr["x_res"], pr["y_res"], s=7, color=BLUE, alpha=0.55, linewidths=0)
        b = np.polyfit(pr["x_res"], pr["y_res"], 1)
        xs = np.linspace(pr["x_res"].min(), pr["x_res"].max(), 2)
        ax.plot(xs, np.polyval(b, xs), color=INK, lw=1.5)
        r = G["primaries"][k]
        ax.set_title(f"{k}: pooled partial r {r['partial_r']:+.2f} [{r['partial_r_ci95'][0]:+.2f}, "
                     f"{r['partial_r_ci95'][1]:+.2f}]\none-sided p {r['fl_p']:.3f}, n {r['n']}", loc="left", fontsize=9,
                     color=INK)
        ax.set_xlabel("drill intensity beyond B2 (residual)")
        ax.set_ylabel(f"{lab}\nbeyond B2 (residual)", fontsize=7.5)
    ph = G["post_hoc_review"]
    rows = [(t["id"], t["n"], t["partial_r"], t["partial_r_ci95"],
             "primary" if t["label"] == "primary" else ("exploratory" if "exploratory" in t["label"] else "robustness"))
            for t in G["tests"] if t["partial_r"] is not None]
    lit = ph["literal_drill_median"]
    k_expl = next(j for j, r in enumerate(rows) if r[4] == "exploratory")
    rows[k_expl:k_expl] = [(lit[k]["id"] + " (run late)", lit[k]["n"], lit[k]["partial_r"], lit[k]["partial_r_ci95"],
                            "robustness") for k in ("P1", "P2")]
    rows += [(t["id"], t["n"], t["partial_r"], t["partial_r_ci95"], "post hoc")
             for t in ph["tests"] if t["partial_r"] is not None]
    style = {"primary": (BLUE, 2.0, 24, True), "robustness": ("#8a8984", 1.2, 12, True),
             "exploratory": (ORANGE, 1.2, 12, True), "post hoc": (INK, 1.0, 16, False)}
    ax = fig.add_axes((0.70, 0.05, 0.285, 0.89))
    for j, (tid, n, r, (lo, hi), kind) in enumerate(rows[::-1]):
        col, lw, size, filled = style[kind]
        ax.plot([lo, hi], [j, j], color=col, lw=lw)
        ax.scatter([r], [j], s=size, zorder=3, facecolors=col if filled else "#fcfcfb", edgecolors=col,
                   linewidths=0 if filled else 1.0)
    n_ph = sum(1 for r in rows if r[4] == "post hoc")
    ax.axhline(n_ph - 0.5, color=MUTED, lw=0.6, ls=":")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(rows)), [f"{r[0]} (n {r[1]})" for r in rows[::-1]], fontsize=6.3)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(-0.75, 0.85)
    ax.set_xlabel("partial r beyond B2 with 95% CI (expected sign +)")
    ax.set_title("Every E010 test with a partial r\n(pooled rows: within-position averages)", loc="left", fontsize=9,
                 color=INK)
    handles = []
    for kind, lab in (("primary", "preregistered primary"), ("robustness", "prespecified robustness"),
                      ("exploratory", "prelisted exploratory"), ("post hoc", "post hoc, after review (below the line)")):
        col, lw, size, filled = style[kind]
        handles.append(ax.scatter([], [], s=size, facecolors=col if filled else "#fcfcfb", edgecolors=col,
                                  linewidths=0 if filled else 1.0, label=lab))
    fig.legend(handles=handles, title="forest plot (right):", title_fontsize=7.5, fontsize=7.5, frameon=False, ncol=4,
               loc="upper left", bbox_to_anchor=(0.07, 0.995), alignment="left")
    # bottom-left: pooled bounds vs per-position bounds (one-sided 95% upper bound of the partial r)
    ax = fig.add_axes((0.075, 0.07, 0.41, 0.24))
    pos_order = ["pooled"] + F.POSITIONS
    lowest = 0.0
    for i, (k, col) in enumerate((("P1", BLUE), ("P2", ORANGE))):
        pb = ph["pooled_vs_position_bounds"][k]
        vals = [G["bounds"][k]["one_sided_upper_95"]] + [pb["by_position"].get(p, {}).get("one_sided_upper_95", np.nan)
                                                         for p in F.POSITIONS]
        x = np.arange(len(pos_order)) + (i - 0.5) * 0.3
        lowest = min(lowest, float(np.nanmin(vals)))
        ax.bar(x, vals, width=0.28, color=col, label=f"{k} (heterogeneity p {pb['heterogeneity_F']['p']:.2f}, post hoc)")
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(xi, v + 0.01, f"{v:+.2f}", ha="center", va="bottom", fontsize=6.3, color=INK)
            else:
                ax.text(xi, 0.01, "n/a", ha="center", va="bottom", fontsize=6.3, color=MUTED)
    ax.set_xticks(range(len(pos_order)), pos_order)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylim(min(0.0, lowest) - 0.03, 0.55)
    ax.set_ylabel("one-sided 95% upper bound\nof the partial r", fontsize=7.5)
    ax.set_title("Positive effects above these are ruled out: pooled vs by position (TE P1: n 20, undefined)",
                 loc="left", fontsize=9, color=INK)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_targets(df: pl.DataFrame, pm: pl.DataFrame, path: Path) -> None:
    """Target QC: P1 by position; per-play max speed relative to own q95 for 6 example players; P2 by position."""
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    rng = np.random.default_rng(1)
    s1 = df.filter(pl.col(P1).is_finite())
    for i, p in enumerate(F.POSITIONS):
        v = s1.filter(pl.col("position") == p)[P1].to_numpy()
        axes[0].scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=6, color=BLUE, alpha=0.6, linewidths=0)
        axes[0].plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=INK, lw=1.4)
    axes[0].set_xticks(range(5), F.POSITIONS)
    axes[0].set_title("P1: median / q95 of REG play max speed\n(WR/TE: routes only; DB/DL/OL: all snaps)", loc="left",
                      fontsize=9, color=INK)
    ex = s1.sort("n_plays", descending=True).group_by("position").head(1).sort("position")
    for i, r in enumerate(ex.iter_rows(named=True)):
        v = pm.filter(pl.col("nfl_id") == r["nfl_id"])["smax"].to_numpy() / r["q95"]
        axes[1].hist(v, bins=np.linspace(0, 1.15, 47), histtype="step", lw=1.1,
                     color=[BLUE, ORANGE, AQUA, "#e87ba4", "#4a3aa7"][i], label=f"{r['position']} (n {len(v)})")
    axes[1].axvline(0.8, color=MUTED, lw=0.8, ls="--")
    axes[1].set_xlabel("play max speed / own q95")
    axes[1].set_title("Per-play intensity, most-used player per position", loc="left", fontsize=9, color=INK)
    axes[1].legend(fontsize=7, frameon=False)
    for i, p in enumerate(F.POSITIONS):
        v = df.filter(pl.col("position") == p)[P2].to_numpy()
        axes[2].scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=6, color=BLUE, alpha=0.6, linewidths=0)
        axes[2].plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=INK, lw=1.4)
    axes[2].set_xticks(range(5), F.POSITIONS)
    axes[2].set_title("P2: off+def snaps per available season", loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


TARGET_TEXT = {"P1": "P1 (median / q95 of REG play max speed: per route for WR/TE, per snap for DB/DL/OL; within "
                     "position largely alignment and route-depth mix)",
               "P2": "P2 (off+def snaps per available season)"}


def decision(cmb: dict, G: dict) -> dict:
    """Verdict computed from the go criteria, the bounds and the post hoc bound checks (no hand-entered numbers)."""
    out = {}
    ph = G["post_hoc_review"]
    for k in ("P1", "P2"):
        c, r, b = G["criteria"][k], G["primaries"][k], G["bounds"][k]
        pb = ph["pooled_vs_position_bounds"][k]
        bp = pb["by_position"]
        top = max(bp, key=lambda p: bp[p]["one_sided_upper_95"])
        failed = [name for name, ok in c.items() if ok is False]
        out[k] = (f"{TARGET_TEXT[k]}: {'passes criteria 1-6' if not failed else 'fails ' + ', '.join(failed)}; pooled "
                  f"within-position partial r {r['partial_r']:+.3f} [{r['partial_r_ci95'][0]:+.2f}, "
                  f"{r['partial_r_ci95'][1]:+.2f}], one-sided p {r['fl_p']:.3f}, BH q {r['bh_q_primaries']:.2f}; pooled "
                  f"observed partial r above {b['one_sided_upper_95']:+.2f} ruled out (one-sided 95%; position-stratified "
                  f"bootstrap {pb['bootstrap']['p95_one_sided_upper']:+.2f}); per-position effects up to "
                  f"{bp[top]['one_sided_upper_95']:+.2f} ({top}) not excluded (feature x position heterogeneity p "
                  f"{pb['heterogeneity_F']['p']:.2f}, post hoc); one-sided MDE {r['partial_r_mde80_one_sided']:.2f}; "
                  f"reliability-corrected (beyond-B2) 95% interval [{b['corrected_ci95'][0]:+.2f}, "
                  f"{b['corrected_ci95'][1]:+.2f}]")
        if k == "P2":
            t = ph["p2_playing_time_trait"]
            ci = t["corrected_ci95_playing_time_trait"]
            out[k] += (f" for feature error only (snaps as recorded); [{ci[0]:+.2f}, {ci[1]:+.2f}] if snaps per season is "
                       f"read as a playing-time trait (beyond-B2 stability "
                       f"{t['beyond_B2']['spearman_brown_to_seasons_mean']:.2f}, post hoc)")
    any_pass = any(all(v for v in G["criteria"][k].values() if isinstance(v, bool)) for k in ("P1", "P2"))
    rx, rxb = cmb["stop_rule"]["reliability"], cmb["diagnostics"]["beyond_B2"]["reliability"]
    ry = G["target_reliability"]["p1"]["weighted_by_sample"]
    rel = (f"feature reliability {rx:.2f} (stop rule), {rxb:.2f} beyond B2; P1 split-half {ry:.2f}; P2 a recorded count")
    measured = min(rx, rxb, ry) >= REL_STOP
    out["verdict"] = ("KEEP: at least one primary passes criteria 1-6" if any_pass else
                      f"KILL: no primary passes. {rel}" + ("; all >= 0.5, so the null is not a measurement failure."
                                                           if measured else "; some < 0.5.")
                      + " Bounds are pooled within-position averages.")
    return out


def run_post_hoc(C: dict, res: dict, df: pl.DataFrame, pm: pl.DataFrame) -> None:
    """Post hoc review block, decision text, JSON and result figures (shared by the full run and --post-hoc)."""
    G = res["game"]
    for k in ("P1", "P2"):
        G["bounds"][k]["corrected_ci95_scope"] = BOUND_SCOPE[k]
    G["post_hoc_review"] = post_hoc_review(C, df, pm, G)
    res["decision"] = decision(res["combine"], G)
    write_json(res)
    figure_results(df, G, FIG / "E010_results.png")
    figure_targets(df, pm, FIG / "E010_targets.png")


def main() -> None:
    log(f"E010 start (quick={QUICK}, combine_only={COMBINE_ONLY}, post_hoc_only={POST_HOC_ONLY})")
    C = build_combine()
    if POST_HOC_ONLY:  # rerun only the post hoc block on the saved preregistered results (identical inputs)
        res = json.loads((RES / "E010.json").read_text())
        run_post_hoc(C, res, pl.read_parquet(DATA / "e010_players.parquet"),
                     pl.read_parquet(DATA / "e010_play_max.parquet"))
        log("done (post hoc only)")
        return
    res = {"experiment": "E010 drill intensity", "prereg": "experiments/results/E010_prereg.json",
           "protocol": "PLAN.md section 4 rev 3", "implementation_choices": IMPLEMENTATION_CHOICES, "eval_settings": EVAL}
    figure_traces(C["frames"], C["att"], FIG / "E010_qc_traces.png")
    figure_intensity(C["use"], C["players"], FIG / "E010_qc_intensity.png")
    log("QC figures written")
    if "--qc-only" in sys.argv:  # used once to review QC before any reliability was computed
        return
    res["combine"] = combine_stage(C)
    C["players"].write_parquet(DATA / "e010_players.parquet")
    if COMBINE_ONLY:
        (SCRATCH.parent / "E010_combine_stage.json").write_text(json.dumps(_clean(res), indent=1))
        log("combine stage only: wrote scratch JSON, exiting")
        return
    if not res["combine"]["stop_rule"]["passes"]:
        res["decision"] = "STOP (prereg reliability stop rule): cross-drill reliability below 0.5; no linkage tests run."
        write_json(res)
        return
    log("stop rule passed: building targets and running the preregistered tests")
    G, df = game_stage(C, res["combine"])
    res["game"] = G
    run_post_hoc(C, res, df, pl.read_parquet(DATA / "e010_play_max.parquet"))
    log("done")


if __name__ == "__main__":
    main()
