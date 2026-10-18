"""E004 — Probe C: OL lateral reversals in Combine drills -> pass sets and pressure allowed in games.

Run top to bottom:  python experiments/E004_ol_lateral_to_pass_pro.py          (everything)
                    python experiments/E004_ol_lateral_to_pass_pro.py --screen (features, reliability, QC only)

Stages
  1. features   Combine reversal signatures (mirror, wave), drop-drill kick metrics; game pass-set traits
                and pressure outcomes. Saved to data/processed/probe_c_*.parquet.
  2. screen     reliability of every candidate feature / trait -> results/E004_reliability.json
  3. QC         segmentation plots -> figures/E004_qc_*.png
  4. linkage    ONLY the tests listed in results/E004_prereg.json (written before any feature-target
                association was computed) + a labeled exploratory list -> results/E004.json
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bdb27 import io, linkage, reliability  # noqa: E402
from bdb27.probes import c_ol_lateral as C  # noqa: E402

HERE = Path(__file__).parent
RES, FIG = HERE / "results", HERE / "figures"
PREREG = RES / "E004_prereg.json"
SEEDS = (0, 1, 2)
OUTCOME_TARGETS = {"pr_oe_eb", "ppp_oe_eb", "pressure_rate", "sack_rate", "ttp", "pr_oe", "ppp_oe"}
N_JOBS = 3


def _f(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def _r(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


# ---------------------------------------------------------------- 1. features

def build_features():
    frames = C.drill_frames(C.MIRROR_NAMES + [C.WAVE_NAME])
    rev = C.combine_reversals(frames)
    cmb = C.combine_reversal_features(rev)
    drop_frames = C.drill_frames([C.DROP_NAME])
    drop = C.drop_kick_metrics(drop_frames)

    plays = C.add_expectations(C.pass_block_plays())
    set_frames = C.pass_set_frames(plays)
    set_m = C.pass_set_metrics(set_frames, plays)
    gset = C.game_set_traits(set_m)
    out = C.outcome_traits(plays)

    cmb.write_parquet(io.PROCESSED / "probe_c_combine.parquet")
    gset.join(out, on="nfl_id", how="full", coalesce=True).write_parquet(io.PROCESSED / "probe_c_game.parquet")
    return dict(frames=frames, rev=rev, cmb=cmb, drop_frames=drop_frames, drop=drop, plays=plays,
                set_frames=set_frames, set_m=set_m, gset=gset, out=out)


# ---------------------------------------------------------------- 2. reliability screen

def _teammate_r(play_df: pl.DataFrame, value: str, plays: pl.DataFrame, min_n: int = 30) -> dict:
    """Correlation of team-season means between study OL teammates (scheme/QB share of a game trait)."""
    d = play_df.join(plays.select(*C.K, "team_abbr", "season"), on=C.K, how="left") if "team_abbr" not in play_df.columns else play_df
    ts = (d.filter(pl.col(value).is_finite()).group_by("nfl_id", "team_abbr", "season")
          .agg(v=pl.col(value).mean(), n=pl.len()).filter(pl.col("n") >= min_n))
    xs, ys = [], []
    for _, g in ts.group_by("team_abbr", "season"):
        v = g["v"].to_list()
        for i in range(len(v)):
            for j in range(len(v)):
                if i != j:
                    xs.append(v[i]); ys.append(v[j])
    return {"teammate_r": _r(xs, ys) if len(xs) >= 6 else None, "n_pairs": len(xs) // 2}


def reliability_screen(F) -> dict:
    rev, cmb, drop, set_m, gset, plays, out = (F[k] for k in ("rev", "cmb", "drop", "set_m", "gset", "plays", "out"))
    R = {"combine_within_attempt": {}, "combine_cross_drill": {}, "drop_across_attempts": {}, "game_set": {}, "outcomes": {}}
    for drill in ("MIRROR", "WAVE"):
        d = rev.filter(pl.col("drill") == drill)
        R["combine_within_attempt"][drill] = {
            m: {k: _f(v) if isinstance(v, float) else v for k, v in reliability.cross_drill_icc(d, m, drill="cut").items()}
            for m in C.REV_METRICS}
        R["combine_within_attempt"][drill]["n_reversals"] = d.height
        R["combine_within_attempt"][drill]["n_lateral_share"] = float(d["lateral"].mean())
    for m in C.REV_METRICS:
        a, b = cmb[f"cmb_{m}_mirror"], cmb[f"cmb_{m}_wave"]
        r = _r(a, b)
        R["combine_cross_drill"][m] = {"r_mirror_wave": r, "composite_alpha": 2 * r / (1 + r) if r > -1 else None,
                                       "n": int((a.is_finite() & b.is_finite()).sum())}
    R["combine_cross_drill"]["rev_accel_mirror_vs_wave_lateral"] = {
        "r": _r(cmb["cmb_rev_accel_mirror"], cmb["cmb_rev_accel_wave_lat"]),
        "n": int((cmb["cmb_rev_accel_mirror"].is_finite() & cmb["cmb_rev_accel_wave_lat"].is_finite()).sum())}
    dd = drop.with_columns(grp=pl.col("session") + "_" + pl.col("attempt").cast(pl.Utf8))
    for m in ("kick_speed", "kick_accel", "kick_brake", "sprint_max"):
        v = dd.with_columns(v=pl.col(m) - pl.col(m).mean().over("grp"))
        R["drop_across_attempts"][m] = {k: _f(x) if isinstance(x, float) else x
                                        for k, x in reliability.icc1(v["v"].to_numpy(), v["nfl_id"].to_numpy()).items()}
    for m in C.SET_METRICS:
        sh = reliability.split_half(set_m, m + "_c")
        R["game_set"]["g_" + m] = {**{k: _f(x) if isinstance(x, float) else x for k, x in sh.items()},
                                   "corr_log_plays": _r(gset["g_" + m], np.log(gset["set_plays"].to_numpy())),
                                   **_teammate_r(set_m, m + "_c", plays)}
    play_vals = {"pr_oe": "pr_oe_eb", "ppp_oe": "ppp_oe_eb", "pressure": "pressure_rate", "ppp": None,
                 "sack_allowed": "sack_rate", "time_to_pressure_allowed": "ttp"}
    for pv, trait in play_vals.items():
        sh = reliability.split_half(plays, pv)
        R["outcomes"][pv] = {**{k: _f(x) if isinstance(x, float) else x for k, x in sh.items()},
                             **_teammate_r(plays, pv, plays)}
        if trait:
            R["outcomes"][pv]["player_trait"] = trait
            R["outcomes"][pv]["corr_log_plays"] = _r(out[trait], np.log(out["pb_plays"].to_numpy()))
    R["outcomes"]["eb"] = {"tau2_pr_oe": float(out["eb_tau2_pr"][0]), "tau2_ppp_oe": float(out["eb_tau2_ppp"][0]),
                           "method": "normal-normal EB on per-player mean residual; tau^2 by method of moments "
                                     "(var of means - mean sampling var); sampling var = sum p(1-p)/n^2 for pressure, "
                                     "pooled within-player residual var / n for peak pressure probability"}
    # Measurement sensitivity (feature-feature only): the run-out exclusion also drops some clean
    # shuffle reversals (QC figure); how much does it change the player-level feature?
    alt = C.combine_reversal_features(C.combine_reversals(F["frames"], drop_runout=False))
    both = cmb.select("nfl_id", "cmb_rev_accel_mirror").join(
        alt.select("nfl_id", alt_mirror="cmb_rev_accel_mirror"), on="nfl_id")
    R["combine_runout_sensitivity"] = {"r_mirror_rev_accel_with_vs_without_runout_exclusion":
                                       _r(both["cmb_rev_accel_mirror"], both["alt_mirror"]), "n": both.height,
                                       "reversals_dropped_by_runout_rule": int(
                                           C.combine_reversals(F["frames"], drop_runout=False).height - rev.height)}
    R["coverage"] = {"combine_mirror_players": int(cmb["cmb_rev_accel_mirror"].is_finite().sum()),
                     "combine_wave_players": int(cmb["cmb_rev_accel_wave"].is_finite().sum()),
                     "game_set_players": gset.height, "outcome_players": out.height,
                     "pass_block_plays": plays.height, "set_plays": set_m.height}
    return R


# ---------------------------------------------------------------- 3. QC figures

def qc_figures(F):
    FIG.mkdir(exist_ok=True)
    frames, rev = F["frames"], F["rev"]
    rng = np.random.default_rng(4)
    fig, axes = plt.subplots(4, 4, figsize=(20, 14))
    for col0, drill in ((0, "MIRROR"), (2, "WAVE")):
        trajs = rng.choice(frames.filter(pl.col("drill") == drill)["traj"].unique().sort().to_numpy(), 4, replace=False)
        for r, t in enumerate(trajs):
            g = frames.filter(pl.col("traj") == t).sort("time")
            b = rev.filter(pl.col("traj") == t)
            tt = np.arange(g.height) * 0.1
            ax = axes[r, col0]
            ax.plot(g["x"], g["y"], "-", c="0.6", lw=1)
            ax.scatter(g["x"], g["y"], c=tt, s=5, cmap="viridis")
            ax.plot(g["x"].gather(b["i_break"]), g["y"].gather(b["i_break"]), "rx", ms=9, mew=2)
            ax.set_aspect("equal"); ax.set_title(f"{drill} nfl_id {g['nfl_id'][0]}: path (color=time), reversals x", fontsize=9)
            ax = axes[r, col0 + 1]
            ax.plot(tt, g["speed"], "b-", label="speed (yd/s)")
            ax.plot(tt, g["a_tan"] / 2, "g-", lw=0.7, label="a_tan/2")
            for row in b.iter_rows(named=True):
                ax.axvline(row["t_break"], c="r", lw=1)
                ax.text(row["t_break"], 5.0, f"{abs(row['turn_deg']):.0f}°\n{row['rev_accel']:.1f}", fontsize=7)
            ax.set_title("speed; red = reversal (turn°, rev_accel yd/s²)", fontsize=9)
            if r == 0:
                ax.legend(fontsize=7, loc="upper left")
    plt.tight_layout(); plt.savefig(FIG / "E004_qc_combine_reversals.png", dpi=70); plt.close()

    sf, sm = F["set_frames"], F["set_m"]
    keys = sm.select(*C.K, "lined_up_position", "set_accel", "set_speed", "t_pk").sort(C.K).sample(12, seed=7)
    fig, axes = plt.subplots(4, 6, figsize=(24, 14))
    for i, row in enumerate(keys.iter_rows(named=True)):
        g = sf.filter((pl.col("game_id") == row["game_id"]) & (pl.col("play_id") == row["play_id"])
                      & (pl.col("nfl_id") == row["nfl_id"])).sort("t_rel")
        g0 = g.filter(pl.col("tk") == 0)
        x0, y0 = g0["x"][0], g0["y"][0]
        w = g.filter(pl.col("tk").is_between(0, 20))
        ax = axes[i // 3, (i % 3) * 2]
        ax.scatter(w["y"] - y0, w["x"] - x0, c=w["t_rel"], s=8, cmap="viridis")
        for tk, mk in ((6, "s"), (10, "o")):
            p = g.filter(pl.col("tk") == tk)
            ax.plot(p["y"][0] - y0, p["x"][0] - x0, "k" + mk, mfc="none", ms=8)
        ax.axhline(0, c="k", lw=0.4); ax.axvline(0, c="k", lw=0.4); ax.set_aspect("equal")
        ax.set_title(f"{row['lined_up_position']} {row['nfl_id']}: dy vs dx 0-2 s (□0.6 s, ○1.0 s)", fontsize=8)
        ax = axes[i // 3, (i % 3) * 2 + 1]
        ax.plot(g["t_rel"], g["speed"], "b-"); ax.plot(g["t_rel"], g["a_tan"], "g-", lw=0.7)
        ax.axvspan(0, 0.6, color="orange", alpha=0.15); ax.axvspan(0.6, 1.0, color="purple", alpha=0.08)
        ax.axhline(row["set_accel"], c="g", ls=":", lw=1); ax.axhline(row["set_speed"], c="b", ls=":", lw=1)
        ax.axvline(row["t_pk"] / 10, c="r", lw=0.8)
        ax.set_xlim(-0.5, 3.0)
        ax.set_title(f"speed(b), a_tan(g); set_accel={row['set_accel']:.2f}, set_speed={row['set_speed']:.2f}", fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "E004_qc_game_sets.png", dpi=60); plt.close()

    drop = F["drop"]
    two = drop.group_by("nfl_id").agg(n=pl.len()).filter(pl.col("n") == 2)["nfl_id"].sort().to_numpy()
    pick = rng.choice(two, 8, replace=False)
    df = F["drop_frames"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    for ax, pid in zip(axes.flat, pick):
        for (t,), g in df.filter(pl.col("nfl_id") == pid).group_by("traj", maintain_order=True):
            g = g.sort("time")
            ax.plot(np.arange(g.height) * 0.1, g["speed"], label=f"attempt {g['attempt'][0]}")
        ax.set_title(f"DROP drill nfl_id {pid}: speed, two attempts", fontsize=9); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "E004_qc_drop_attempts.png", dpi=60); plt.close()


# ---------------------------------------------------------------- 4. linkage

def analysis_table(F) -> pl.DataFrame:
    game = F["gset"].join(F["out"], on="nfl_id", how="full", coalesce=True)
    return (linkage.player_table().filter(pl.col("position") == "OL")
            .join(F["cmb"], on="nfl_id", how="left").join(game, on="nfl_id", how="left"))


def _slim(r: dict) -> dict:
    keep = ("feature", "target", "n", "raw_r", "partial_r", "delta_r2_vs_B0", "delta_r2_vs_B2", "cv_perm_p",
            "loco", "by_class", "checks", "cv_r2")
    return {k: r[k] for k in keep}


EXPLORATORY = [  # (feature, target, population note); evaluated only after the primary tests, labeled exploratory
    ("cmb_rev_accel_composite", "g_set_accel", "all"),
    ("cmb_rev_accel_wave", "g_set_accel", "all"),
    ("cmb_rev_accel_mirror", "g_set_speed", "all"),
    ("cmb_rev_accel_mirror", "g_set_brake", "all"),
    ("cmb_peak_decel_mirror", "g_set_accel", "all"),
    ("cmb_peak_reaccel_mirror", "g_set_accel", "all"),
    ("cmb_rev_accel_composite", "pr_oe_eb", "all"),
    ("cmb_rev_accel_composite", "ppp_oe_eb", "all"),
    ("cmb_rev_accel_mirror", "sack_rate", "all"),
    ("cmb_rev_accel_mirror", "pressure_rate", "all"),
    ("g_set_accel", "ppp_oe_eb", "game trait -> outcome (no Combine)"),
    ("g_set_accel", "pr_oe_eb", "game trait -> outcome (no Combine)"),
]


def run_linkage(F, R) -> dict:
    prereg = json.loads(PREREG.read_text())
    df = analysis_table(F)
    out = {"prereg_file": str(PREREG.relative_to(HERE.parent)), "primary": [], "exploratory": []}
    for t in prereg["primary_tests"]:
        f, y = t["feature"], t["target"]
        entry = {"feature": f, "target": y, "population": t["population"], "seeds": {}}
        for s in SEEDS:
            r = linkage.evaluate(df, f, y, n_repeats=50, n_perm=2000, seed=s, n_jobs=N_JOBS)
            entry["seeds"][s] = _slim(r)
            print(f"[primary] {f} -> {y} seed {s}: n={r['n']} raw_r={r['raw_r']:+.3f} partial_r={r['partial_r']} "
                  f"dR2_B2={r['delta_r2_vs_B2']['mean']:+.4f}±{r['delta_r2_vs_B2']['se']:.4f} "
                  f"excess={r['delta_r2_vs_B2']['excess']:+.4f} p={r['cv_perm_p']:.4f} loco={r['checks']['loco_excess_positive_classes']}", flush=True)
        if y in OUTCOME_TARGETS:
            d23 = df.filter(pl.col("draft_year") != 2025)
            entry["without_2025"] = {}
            for s in SEEDS:
                r = linkage.evaluate(d23, f, y, n_repeats=50, n_perm=2000, seed=s, n_jobs=N_JOBS)
                entry["without_2025"][s] = _slim(r)
                print(f"[primary no2025] {f} -> {y} seed {s}: n={r['n']} excess={r['delta_r2_vs_B2']['excess']:+.4f} "
                      f"p={r['cv_perm_p']:.4f} partial_r={r['partial_r']}", flush=True)
        out["primary"].append(entry)
    for f, y, note in EXPLORATORY:
        r = linkage.evaluate(df, f, y, n_repeats=20, n_perm=500, seed=0, n_jobs=N_JOBS)
        out["exploratory"].append({"note": note, **_slim(r)})
        print(f"[exploratory] {f} -> {y}: n={r['n']} raw_r={r['raw_r']:+.3f} partial_r={r['partial_r']} "
              f"excess={r['delta_r2_vs_B2']['excess']:+.4f} p={r['cv_perm_p']:.4f} loco={r['checks']['loco_excess_positive_classes']}", flush=True)
    out["n_exploratory"] = len(out["exploratory"])
    # BH-FDR over the primary tests (seed 0 p-values)
    ps = np.array([e["seeds"][0]["cv_perm_p"] for e in out["primary"]])
    order = np.argsort(ps); q = np.empty_like(ps); m = len(ps)
    q[order] = np.minimum.accumulate((ps[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    out["primary_bh_q_seed0"] = q.tolist()
    return out


def result_figure(F, L):
    """Writeup-style figure: reliability ladder + partial-residual scatter for each primary test."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    df = analysis_table(F)
    prim = L["primary"]
    fig, axes = plt.subplots(1, len(prim) + 1, figsize=(5.2 * (len(prim) + 1), 4.6))
    R = L["reliability"]
    labels, vals, colors = [], [], []
    for lab, v, c in [
        ("Mirror rev. accel\n(within attempt)", R["combine_within_attempt"]["MIRROR"]["rev_accel"]["reliability"], "#4C72B0"),
        ("Wave rev. accel\n(within attempt)", R["combine_within_attempt"]["WAVE"]["rev_accel"]["reliability"], "#4C72B0"),
        ("Mirror vs wave\n(cross-drill r)", R["combine_cross_drill"]["rev_accel"]["r_mirror_wave"], "#4C72B0"),
        ("Drop kick speed\n(2 attempts ICC)", R["drop_across_attempts"]["kick_speed"]["icc"], "#4C72B0"),
        ("Game set accel\n(split-half)", R["game_set"]["g_set_accel"]["reliability"], "#DD8452"),
        ("Pressure over exp.\n(split-half)", R["outcomes"]["pr_oe"]["reliability"], "#55A868"),
        ("Peak press. prob\nover exp. (split-half)", R["outcomes"]["ppp_oe"]["reliability"], "#55A868")]:
        labels.append(lab); vals.append(v if v is not None else np.nan); colors.append(c)
    ax = axes[0]
    ax.barh(range(len(vals)), vals, color=colors); ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0.5, c="k", ls="--", lw=0.8); ax.invert_yaxis(); ax.set_xlim(-0.1, 1); ax.set_title("Reliability screen", fontsize=10)
    for ax, e in zip(axes[1:], prim):
        f, y = e["feature"], e["target"]
        d = df.filter(pl.col(f).is_finite() & pl.col(y).is_finite())
        Xc = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True).fit_transform(
            d.select(linkage.COMBINE_CONTROLS + linkage.DRAFT_CONTROLS).cast(pl.Float64).to_numpy())
        x, yy = d[f].to_numpy(), d[y].to_numpy()
        rx = x - LinearRegression().fit(Xc, x).predict(Xc); ry = yy - LinearRegression().fit(Xc, yy).predict(Xc)
        for yr, mk in ((2023, "o"), (2024, "s"), (2025, "^")):
            msk = d["draft_year"].to_numpy() == yr
            ax.scatter(rx[msk], ry[msk], marker=mk, s=22, alpha=0.75, label=str(yr))
        b = np.polyfit(rx, ry, 1); xs = np.linspace(rx.min(), rx.max(), 10); ax.plot(xs, np.polyval(b, xs), "k-", lw=1)
        s0 = e["seeds"][0]
        ax.set_title(f"{f}\n→ {y} (partial, given B2)\nn={s0['n']}, partial r={s0['partial_r']:+.2f}, "
                     f"excess ΔR²={s0['delta_r2_vs_B2']['excess']:+.3f}, p={s0['cv_perm_p']:.3f}", fontsize=8)
        ax.set_xlabel("Combine feature residual"); ax.set_ylabel("game target residual"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "E004_result.png", dpi=80); plt.close()


if __name__ == "__main__":
    screen_only = "--screen" in sys.argv
    F = build_features()
    R = reliability_screen(F)
    RES.mkdir(exist_ok=True)
    (RES / "E004_reliability.json").write_text(json.dumps(R, indent=2, default=str))
    print(json.dumps(R, indent=1, default=str))
    qc_figures(F)
    if screen_only:
        sys.exit(0)
    if not PREREG.exists():
        sys.exit("E004_prereg.json missing: preregister the primary tests before running the linkage stage.")
    L = run_linkage(F, R)
    L["reliability"] = R
    (RES / "E004.json").write_text(json.dumps(L, indent=2, default=str))
    result_figure(F, L)
