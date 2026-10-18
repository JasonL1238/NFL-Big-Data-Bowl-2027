"""E003 - Probe B: DL first-step burst. Combine drill burst -> regular-season burst, NGS get-off, quick pressure.

Reproduces everything top to bottom: Combine features + reliability screen, game semantics checks, game
traits + reliability, visual QC figures, the 3 preregistered primary tests (seeds 0/1/2, with and without
the 2025 class) and every exploratory test (counted). The preregistration
(results/E003_prereg.json) was written before any Combine-game association was computed.

Outputs: results/E003.json, figures/E003_*.png, data/processed/probe_b_{combine,game_plays,game_traits}.parquet
"""

import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

from bdb27 import io, linkage, segments  # noqa: E402
from bdb27 import reliability as R  # noqa: E402
from bdb27.probes import b_dl_burst as B  # noqa: E402

HERE = Path(__file__).parent
RES, FIG = HERE / "results", HERE / "figures"
PREREG = RES / "E003_prereg.json"
assert PREREG.exists(), "preregister before running any linkage"
FIG.mkdir(exist_ok=True)
N_JOBS = 3
MIN_SNAPS = 30
BURST4 = ["FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION", "PASS_RUSH_DRILL", "RUN_AND_CLUB_DRILL", "RUN_THE_HOOP_DRILL"]
CTRL = linkage.COMBINE_CONTROLS + linkage.DRAFT_CONTROLS
FEATS = B.BURST_FEATS + ["t5"]
BLUE, ORANGE, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
T0 = time.time()
out: dict = {"prereg": json.loads(PREREG.read_text())}


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def fnum(v, nd=3):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


# =====================================================================================
# 1. Combine burst features
# =====================================================================================
pt = linkage.player_table()
dl = pt.filter(pl.col("position") == "DL")
ids = dl["nfl_id"].to_list()
cfr = B.combine_frames(ids)
catt = B.combine_attempt_features(cfr)
pdm = B.combine_player_drill(catt)
c40 = (pdm.filter(pl.col("drill_name") == B.FORTY)
       .select("nfl_id", *[pl.col(f).alias(f"c40_{f}") for f in FEATS], c40_n_att="n_att"))
comb = (dl.select("nfl_id")
        .join(c40, on="nfl_id", how="left")
        .join(B.composite(pdm, "d10", BURST4, name="cpos_d10"), on="nfl_id", how="left")
        .join(B.composite(pdm, "d10", BURST4 + [B.FORTY], name="call_d10"), on="nfl_id", how="left"))
comb.write_parquet(io.PROCESSED / "probe_b_combine.parquet")
out["combine_attempts"] = (catt.group_by("drill_name").agg(
    attempts=pl.len(), players=pl.col("nfl_id").n_unique(), onset_observed=pl.col("t_on").is_not_null().sum(),
    d10_median=pl.col("d10").median()).sort("drill_name").to_dicts())
log(f"Combine: {catt.height} attempts, c40_d10 for {comb['c40_d10'].is_not_null().sum()} DL")

# ---- reliability screen (before any linkage) ----
X_b2 = linkage._imputed(dl.select(CTRL).cast(pl.Float64).to_numpy())  # B2 design (median + indicators)
row_of = dict(zip(dl["nfl_id"].to_list(), range(dl.height)))


def resid_beyond_b2(nfl_id: np.ndarray, v: np.ndarray) -> np.ndarray:
    Xi = X_b2[[row_of[i] for i in nfl_id]]
    return v - LinearRegression().fit(Xi, v).predict(Xi)


def rel_attempts(drill, f):
    a = catt.filter((pl.col("drill_name") == drill) & pl.col(f).is_not_null())
    v, g = a[f].to_numpy().astype(float), a["nfl_id"].to_numpy()
    r = R.icc1(v, g)
    rr = R.icc1(resid_beyond_b2(g, v), g)
    return {"icc": fnum(r["icc"]), "reliability": fnum(r["reliability"]), "n_players": r["n_players"],
            "n_repeated": r["n_repeated"], "resid_b2_icc": fnum(rr["icc"]), "resid_b2_reliability": fnum(rr["reliability"])}


def rel_cross(drills, f):
    d = pdm.filter(pl.col("drill_name").is_in(drills) & pl.col(f"z_{f}").is_not_null())
    r = R.cross_drill_icc(d, f"z_{f}")
    v = d[f"z_{f}"].to_numpy() - d.with_columns(m=pl.col(f"z_{f}").mean().over("drill_name"))["m"].to_numpy()
    rr = R.icc1(resid_beyond_b2(d["nfl_id"].to_numpy(), v), d["nfl_id"].to_numpy())
    return {"icc": fnum(r["icc"]), "reliability": fnum(r["reliability"]), "n_players": r["n_players"],
            "resid_b2_icc": fnum(rr["icc"]), "resid_b2_reliability": fnum(rr["reliability"])}


crel = {"attempt_icc": {}, "cross_drill": {}}
for drill in [B.FORTY, "PASS_RUSH_DRILL"]:
    for f in FEATS:
        if drill != B.FORTY and f == "t5":
            continue
        crel["attempt_icc"][f"{drill}:{f}"] = rel_attempts(drill, f)
for name, ds in [("burst4", BURST4), ("burst4+40", BURST4 + [B.FORTY]),
                 ("all6+40", B.DL_DRILLS + [B.FORTY])]:
    for f in B.BURST_FEATS:
        crel["cross_drill"][f"{name}:{f}"] = rel_cross(ds, f)
crel["c40_vs_controls_r"] = {
    f: {c: fnum(comb.join(dl, on="nfl_id").select(pl.corr(f"c40_{f}", c)).item()) for c in
        ["ten_yd_split", "forty", "combine_weight", "ngs_athleticism_score", "log_pick"]} for f in FEATS}
out["combine_reliability"] = crel
log("Combine reliability: c40_d10 attempts " + json.dumps(crel["attempt_icc"][f"{B.FORTY}:d10"]))

# =====================================================================================
# 2. Game burst, get-off and pass-rush outcomes (DL, REG season)
# =====================================================================================
gfr = B.game_frames(ids)
gp = B.game_play_features(gfr).join(B.dl_play_table(ids), on=segments.PLAY_KEYS, how="left")
gp.drop("traj").write_parquet(io.PROCESSED / "probe_b_game_plays.parquet")
log(f"Game: {gp.height} DL player-plays, {gp['rush'].sum()} pass-rush snaps")

# ---- semantics checks ----
rp = gp.filter(pl.col("rush"))
sem = {
    "pass_rush_snaps": int(rp.height),
    "pressure_rate": fnum(rp["pressure"].mean()), "quick_pressure_rate": fnum(rp["quick"].mean()),
    "sack_rate": fnum(rp["sacked"].mean()),
    "quick_pressure_def": {"max_ttp_when_true": fnum(rp.filter(pl.col("quick_pressure") == True)["time_to_pressure"].max()),  # noqa: E712
                           "min_ttp_when_false": fnum(rp.filter(pl.col("quick_pressure") == False)["time_to_pressure"].min())},  # noqa: E712
}
m = gp.filter(pl.col("player_get_off").is_not_null() & pl.col("t_up13").is_not_null())
sem["get_off_vs_tracking_time_to_1p3yd_upfield"] = {
    "n": m.height, "r": fnum(np.corrcoef(m["player_get_off"], m["t_up13"])[0, 1]),
    "median_diff_s": fnum((m["player_get_off"] - m["t_up13"]).median()),
    "get_off_quantiles": {q: fnum(gp["player_get_off"].quantile(q)) for q in [0.05, 0.5, 0.95]}}
sem["t_move_quantiles_rush"] = {q: fnum(rp["t_move"].quantile(q)) for q in [0.05, 0.25, 0.5, 0.75, 0.95]}
out["game_semantics"] = sem
log("get-off check: " + json.dumps(sem["get_off_vs_tracking_time_to_1p3yd_upfield"]))

# ---- player traits ----
plays_by = {"rush": gp.filter(pl.col("rush")), "all": gp.filter(pl.col("rush") | pl.col("run"))}
gt = B.game_traits(plays_by["rush"], "rush", MIN_SNAPS).join(
    B.game_traits(plays_by["all"], "all", MIN_SNAPS), on="nfl_id", how="full", coalesce=True)
gt = gt.with_columns(g_getoff_rush=pl.col("g_player_get_off_rush"), g_quick_pressure_oe=pl.col("g_quick_oe_rush"))
gt.write_parquet(io.PROCESSED / "probe_b_game_traits.parquet")

grel = {}
for suffix, plays in plays_by.items():
    el = B.eligible_plays(plays, MIN_SNAPS)
    for c in B.KIN_TRAITS + B.RATE_EVENTS:
        r = R.split_half(el.with_columns(pl.col(f"{c}_adj").cast(pl.Float64)), f"{c}_adj")
        tcol = f"g_{c}_{suffix}" if c in B.KIN_TRAITS else f"g_{c}_oe_{suffix}"
        tt = gt.select(tcol, n=f"n_{suffix}").drop_nulls()
        grel[f"{tcol}"] = {"split_half": fnum(r["reliability"]), "n_players": r["n_players"],
                           "r_log_snaps": fnum(np.corrcoef(tt[tcol], np.log(tt["n"]))[0, 1])}
out["game_reliability"] = grel
log("game reliability (rush): d10 %.2f getoff %.2f quick %.2f pressure %.2f" % (
    grel["g_d10_rush"]["split_half"], grel["g_player_get_off_rush"]["split_half"],
    grel["g_quick_oe_rush"]["split_half"], grel["g_pressure_oe_rush"]["split_half"]))

# =====================================================================================
# 3. Visual QC
# =====================================================================================
cfr_s = cfr.sort("traj", "time")
drills_qc = ["PASS_RUSH_DRILL", "RUN_AND_CLUB_DRILL", "RUN_THE_HOOP_DRILL",
             "FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION", B.FORTY]
fig, axes = plt.subplots(len(drills_qc), 5, figsize=(17, 2.6 * len(drills_qc)), sharex=True)
rng = np.random.default_rng(3)
for i, d in enumerate(drills_qc):
    a = catt.filter(pl.col("drill_name") == d)
    for j, t in enumerate(rng.choice(a["traj"].to_numpy(), 5, replace=False)):
        f = cfr_s.filter(pl.col("traj") == t)
        r = a.filter(pl.col("traj") == t).row(0, named=True)
        ax, tt = axes[i, j], np.arange(f.height) * 0.1
        ax.plot(tt, f["speed"], color=BLUE, lw=1.6, label="speed (yd/s)")
        ax.plot(tt, f["a_tan"], color=ORANGE, lw=1, alpha=0.8, label="a_tan (yd/s²)")
        ax.axhline(1.0, color=INK2, lw=0.6, ls=":")
        if r["t_on"] is not None:
            ax.axvline(r["t_on"], color=INK, lw=1)
            ax.axvspan(r["t_on"], r["t_on"] + 1.0, color=BLUE, alpha=0.10)
            ttl = f"d10={r['d10']:.2f} yd"
        else:
            ttl = f"onset unobserved (v0={r['speed0']:.1f})"
        ax.set_title(f"{d.split('_DRILL')[0][:22]} | {ttl}", fontsize=8, color=INK)
        ax.set_xlim(0, 4)
    axes[i, 0].set_ylabel("yd/s, yd/s²")
axes[0, 0].legend(fontsize=7, loc="upper right", frameon=False)
for ax in axes[-1]:
    ax.set_xlabel("s from start of recording")
fig.suptitle("E003 QC: Combine motion onset (black line, speed crosses 1 yd/s) and the 1.0 s burst window (shaded)",
             color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "E003_qc_combine_onset.png", dpi=110)
plt.close(fig)

gfr_s = gfr.sort("traj", "time")
fig, axes = plt.subplots(3, 6, figsize=(19, 8.5), sharex=True)
for row, (lup, n) in enumerate([("EDGE", 9), ("INTERIOR_LINE", 9)]):
    pick = gp.filter(pl.col("rush") & (pl.col("lined_up_position") == lup)).sample(n, seed=11)
    for k, r in enumerate(pick.iter_rows(named=True)):
        ax = axes.flat[row * 9 + k]
        f = gfr_s.filter(pl.col("traj") == r["traj"])
        ax.plot(f["t_rel"], f["speed"], color=BLUE, lw=1.6, label="speed")
        ax.plot(f["t_rel"], f["a_tan"], color=ORANGE, lw=1, alpha=0.8, label="a_tan")
        ax.axvline(0, color=INK2, lw=1, ls="--", label="snap")
        ax.axhline(1.0, color=INK2, lw=0.6, ls=":")
        if r["t_move"] is not None:
            ax.axvline(r["t_move"], color=INK, lw=1, label="onset")
            ax.axvspan(r["t_move"], r["t_move"] + 1.0, color=BLUE, alpha=0.10)
        if r["player_get_off"] is not None:
            ax.plot([r["player_get_off"]], [0], marker="v", color=INK, ms=8, ls="none", label="NGS get-off")
        d10 = "n/a" if r["d10"] is None else f"{r['d10']:.2f}"
        ax.set_title(f"{lup[:8]} | d10={d10} yd | pressure={r['pressure']}", fontsize=8, color=INK)
        ax.set_xlim(-1, 3)
axes.flat[0].legend(fontsize=7, frameon=False, loc="upper left")
for ax in axes[-1]:
    ax.set_xlabel("s since snap")
fig.suptitle("E003 QC: NFL pass-rush snaps - snap, motion onset, 1.0 s burst window, NGS get-off (triangle)",
             color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "E003_qc_game_onset.png", dpi=110)
plt.close(fig)
log("QC figures written")

# =====================================================================================
# 4. Linkage: preregistered primary tests, then exploratory tests
# =====================================================================================
df = dl.join(comb, on="nfl_id", how="left").join(gt, on="nfl_id", how="inner")


def summarize(r, label=None, extra=None):
    d = r["delta_r2_vs_B2"]
    s = {"label": label, "feature": r["feature"], "target": r["target"], "n": r["n"], "resid_df": r["resid_df"],
         "raw_r": fnum(r["raw_r"]),
         "partial_r": fnum(r["partial_r"]) if r["partial_r"] is not None else None,
         "delta_r2_vs_B2": fnum(d["mean"], 4), "se": fnum(d["se"], 4), "null_mean": fnum(d["null_mean"], 4),
         "excess": fnum(d["excess"], 4), "cv_perm_p": fnum(r["cv_perm_p"], 4), "loco": r["loco"],
         "loco_positive": r["checks"]["loco_excess_positive_classes"], "by_class": r["by_class"],
         "cv_r2": {k: fnum(v, 4) for k, v in r["cv_r2"].items()},
         "delta_r2_vs_B0": {k: fnum(v, 4) for k, v in r["delta_r2_vs_B0"].items()}, "checks": r["checks"]}
    if extra:
        s.update(extra)
    return s


def show(s):
    pr = "n/a" if s["partial_r"] is None else f"{s['partial_r']:+.2f}"
    print(f"  {s['label'] or '':38s} {s['feature']:>10s} -> {s['target']:<22s} n={s['n']:3d} r={s['raw_r']:+.2f} "
          f"pr={pr} dR2={s['delta_r2_vs_B2']:+.3f}±{s['se']:.3f} excess={s['excess']:+.3f} p={s['cv_perm_p']:.4f} "
          f"loco+={s['loco_positive']}", flush=True)


primary = []
for test in out["prereg"]["primary_tests"]:
    seeds = []
    for seed in (0, 1, 2):
        r = linkage.evaluate(df, test["feature"], test["target"], n_repeats=50, n_perm=2000, seed=seed, n_jobs=N_JOBS)
        seeds.append(summarize(r, f"{test['id']} seed{seed}", {"seed": seed}))
        show(seeds[-1])
    r = linkage.evaluate(df.filter(pl.col("draft_year") != 2025), test["feature"], test["target"],
                         n_repeats=50, n_perm=2000, seed=0, n_jobs=N_JOBS)
    no25 = summarize(r, f"{test['id']} without 2025")
    show(no25)
    passes = [s["checks"]["excess_lb_gt_0"] and s["checks"]["fl_p_lt_05"]
              and int(s["loco_positive"].split("/")[0]) >= 2 for s in seeds]
    sign_ok = [(s["partial_r"] or 0) * (1 if test["expected_sign"] == "+" else -1) > 0 for s in seeds]
    # Descriptive precision of the (in-sample) partial r: Fisher-z 95% CI, the smallest partial r this n
    # detects with 80% power (two-sided 0.05), and the CI upper bound corrected for measurement error
    # (feature: reliability of its part beyond B2; target: split-half reliability).
    pr, dfr = seeds[0]["partial_r"], seeds[0]["resid_df"]
    zse = 1 / np.sqrt(dfr - 1)
    ci = [float(np.tanh(np.arctanh(pr) + k * 1.96 * zse)) for k in (-1, 1)]
    rel_x = crel["attempt_icc"][f"{B.FORTY}:d10"]["resid_b2_reliability"]
    rel_y = grel[{"g_d10_rush": "g_d10_rush", "g_getoff_rush": "g_player_get_off_rush",
                  "g_quick_pressure_oe": "g_quick_oe_rush"}[test["target"]]]["split_half"]
    precision = {"partial_r_ci95": [fnum(c) for c in ci], "mde_partial_r_80pct_power": fnum(np.tanh(2.8 * zse)),
                 "ci_upper_disattenuated": fnum(max(abs(c) for c in ci) / np.sqrt(rel_x * rel_y)),
                 "rel_x_beyond_B2": rel_x, "rel_y": rel_y}
    print(f"  {test['id']} partial r 95% CI {precision['partial_r_ci95']}, MDE {precision['mde_partial_r_80pct_power']}")
    primary.append({**test, "seeds": seeds, "without_2025": no25, "precision": precision,
                    "seeds_passing_all_checks": f"{sum(passes)}/3",
                    "expected_sign_matches": bool(all(sign_ok)),
                    "bonferroni_p_lt_0167_seed0": bool(seeds[0]["cv_perm_p"] < 0.05 / 3)})
out["primary"] = primary
log("primary tests done")

explo = []


def ex(feature, target, data=None, label=""):
    r = linkage.evaluate(df if data is None else data, feature, target, n_repeats=20, n_perm=500, seed=0,
                         n_jobs=N_JOBS)
    s = summarize(r, label)
    explo.append(s)
    show(s)


ex("c40_d10", "g_t_move_rush", label="E1 negative control: reaction time")
for f in ["d05", "s05", "s10", "amax06", "t3", "t5"]:
    ex(f"c40_{f}", "g_d10_rush", label=f"E 40-burst variant {f}")
for comp in ["cpos_d10", "call_d10"]:
    for tgt in ["g_d10_rush", "g_getoff_rush", "g_quick_pressure_oe"]:
        ex(comp, tgt, label=f"E composite {comp}")
ex("c40_d10", "g_d10_snap_rush", label="E snap-aligned game burst")
ex("c40_d10", "g_t_up13_rush", label="E tracking get-off (1.3 yd upfield)")
ex("c40_d10", "g_pressure_oe_rush", label="E overall pressure rate")
ex("c40_d10", "g_sacked_oe_rush", label="E sack rate")
ex("c40_d10", "g_d10_all", label="E game burst, all snaps")
edge = df.filter(pl.col("edge_share_rush") >= 0.5)
inter = df.filter(pl.col("edge_share_rush") < 0.5)
for name, sub in [("EDGE-majority", edge), ("INTERIOR-majority", inter)]:
    for tgt in ["g_d10_rush", "g_getoff_rush", "g_quick_pressure_oe"]:
        ex("c40_d10", tgt, data=sub, label=f"E stratum {name}")
# Post hoc (added AFTER seeing the primary results and the game QC plot, where interior DL meet blockers
# inside the 1.0 s window): matched shorter window. Labeled post hoc; counts as exploratory.
ex("c40_d05", "g_d05_rush", label="E POST HOC 0.5 s window both sides")


def bh(pvals):
    p = np.asarray(pvals, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out_q = np.empty_like(q)
    out_q[o] = np.minimum(q, 1)
    return out_q


for s_, q_ in zip(explo, bh([e["cv_perm_p"] for e in explo])):
    s_["bh_q"] = fnum(q_)
out["exploratory"] = explo
out["exploratory_count"] = len(explo)
out["exploratory_p_lt_05"] = sum(e["cv_perm_p"] < 0.05 for e in explo)
out["exploratory_min_bh_q"] = min(e["bh_q"] for e in explo)
log(f"{len(explo)} exploratory tests done")

# ---- game-side chain (descriptive; not a Combine linkage) ----
chain = {}
for a, b in [("g_d10_rush", "g_getoff_rush"), ("g_d10_rush", "g_quick_pressure_oe"),
             ("g_getoff_rush", "g_quick_pressure_oe"), ("g_d10_rush", "g_pressure_oe_rush"),
             ("g_getoff_rush", "g_pressure_oe_rush"), ("g_t_move_rush", "g_quick_pressure_oe")]:
    t = gt.select(a, b).drop_nulls()
    chain[f"{a}~{b}"] = {"n": t.height, "r": fnum(np.corrcoef(t[a], t[b])[0, 1])}
out["game_chain_descriptive"] = chain

# =====================================================================================
# 5. Figures for the log
# =====================================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
labels = {"g_d10_rush": "NFL burst: yards in 1st s after onset\n(pass rush, alignment-adj.)",
          "g_getoff_rush": "NGS get-off (s, lower = faster)", "g_quick_pressure_oe": "quick-pressure rate over expected"}
for ax, p in zip(axes, primary):
    d = df.filter(pl.col(p["feature"]).is_finite() & pl.col(p["target"]).is_finite()).sort("nfl_id")
    Xi = linkage._imputed(linkage._design(d, CTRL))
    rx = d[p["feature"]].to_numpy() - LinearRegression().fit(Xi, d[p["feature"]].to_numpy()).predict(Xi)
    ry = d[p["target"]].to_numpy() - LinearRegression().fit(Xi, d[p["target"]].to_numpy()).predict(Xi)
    yr = d["draft_year"].to_numpy()
    for y, mk in zip([2023, 2024, 2025], ["o", "s", "^"]):
        ax.scatter(rx[yr == y], ry[yr == y], s=34, marker=mk, color=BLUE, edgecolor=SURF, linewidth=1.5,
                   alpha=0.85, label=str(y))
    b = np.polyfit(rx, ry, 1)
    xs = np.linspace(rx.min(), rx.max(), 10)
    ax.plot(xs, np.polyval(b, xs), color=ORANGE, lw=2)
    s0 = p["seeds"][0]
    ax.set_title(f"{p['id']}: partial r = {s0['partial_r']:+.2f}, excess ΔR² = {s0['excess']:+.3f} ± {s0['se']:.3f}\n"
                 f"perm p = {s0['cv_perm_p']:.3f}, n = {s0['n']}", fontsize=9, color=INK)
    ax.set_xlabel("Combine 40 burst (yd in 1st s after onset)\nresidual given B2", color=INK)
    ax.set_ylabel(labels[p["target"]] + "\nresidual given B2", color=INK)
axes[0].legend(title="draft class", fontsize=7, title_fontsize=7, frameon=False)
fig.suptitle("E003 primary tests: Combine 40-yd burst vs NFL DL traits, both residualized on Combine times + draft pick (B2)",
             color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "E003_primary_partial.png", dpi=120)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 6.5))
items = [("40: d10 (attempts)", crel["attempt_icc"][f"{B.FORTY}:d10"]),
         ("40: s10 (attempts)", crel["attempt_icc"][f"{B.FORTY}:s10"]),
         ("40: amax06 (attempts)", crel["attempt_icc"][f"{B.FORTY}:amax06"]),
         ("PASS_RUSH: d10 (attempts)", crel["attempt_icc"]["PASS_RUSH_DRILL:d10"]),
         ("4 position drills: d10 (cross-drill)", crel["cross_drill"]["burst4:d10"]),
         ("4 drills + 40: d10 (cross-drill)", crel["cross_drill"]["burst4+40:d10"]),
         ("all 6 DL drills + 40: d10 (cross-drill)", crel["cross_drill"]["all6+40:d10"])]
yy = np.arange(len(items))[::-1]
ax.barh(yy + 0.2, [v["reliability"] for _, v in items], height=0.38, color=BLUE, label="reliability of player value")
ax.barh(yy - 0.2, [v["resid_b2_reliability"] for _, v in items], height=0.38, color=ORANGE,
        label="reliability of the part not explained by B2")
ax.set_yticks(yy, [k for k, _ in items])
ax.axvline(0.5, color=INK2, ls="--", lw=1)
ax.text(0.505, yy[0] + 0.45, "0.5 screen", color=INK2, fontsize=8)
ax.set_xlim(-0.1, 1)
ax.set_xlabel("reliability (Spearman-Brown, as used)")
ax.legend(frameon=False, loc="lower right", fontsize=8)
ax.set_title("E003 reliability: only the 40-yd start gives a stable burst signal beyond the stopwatch\n"
             f"(game traits, split-half: burst d10 {grel['g_d10_rush']['split_half']:.2f}, get-off "
             f"{grel['g_player_get_off_rush']['split_half']:.2f}, quick-pressure rate {grel['g_quick_oe_rush']['split_half']:.2f})",
             fontsize=9, color=INK)
fig.tight_layout()
fig.savefig(FIG / "E003_reliability.png", dpi=120)
plt.close(fig)

(RES / "E003.json").write_text(json.dumps(out, indent=2, default=str))
log("wrote results/E003.json")
