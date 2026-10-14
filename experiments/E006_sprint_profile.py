"""E006 — Probe E: sprint mechanics profile.

Combine 40-yd mono-exponential fit (vmax, tau, a0 = vmax/tau) + Samozino F-V profile  vs  the in-game
acceleration-speed (A-S) profile (A0, S0; Morin et al. 2021, made exposure-invariant with fixed-size play
draws) and WR/TE yards after catch over expected. Preregistration: experiments/results/E006_prereg.json.

Reproduces everything top to bottom: features, reliability screen, QC figures, primary tests (3 seeds +
without the 2025 class), exploratory tests, results JSON. Run: .venv/bin/python experiments/E006_sprint_profile.py
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import false_discovery_control

from bdb27 import features, io, linkage, reliability
from bdb27.probes import e_speed_profile as E

HERE = Path(__file__).parent
RES = HERE / "results"
FIG = HERE / "figures"
PREREG = RES / "E006_prereg.json"
OUT = RES / "E006.json"
FIG.mkdir(exist_ok=True)
POS = ["WR", "TE", "DB", "DL", "OL"]
N_JOBS = 3
t_start = time.time()


def log(msg):
    print(f"[{time.time() - t_start:6.0f}s] {msg}", flush=True)


# ------------------------------------------------------------------ Combine side
frames40 = E.forty_frames()
fits = E.combine_forty_fits(frames40)
prof = E.combine_profile(fits)
prof.write_parquet(io.PROCESSED / "probe_e_combine_profile.parquet")
pt = linkage.player_table()
pos_tbl = pt.select("nfl_id", "position")
log(f"40 attempts {fits.height}, QC pass {fits['qc_ok'].sum()}, players {prof.height}")

fq = fits.filter(pl.col("qc_ok")).join(pos_tbl, on="nfl_id")
combine_rel = {}
for c in E.COMBINE_PARAMS:
    combine_rel[c] = {}
    for p in POS:
        r = fq.filter(pl.col("position") == p)
        combine_rel[c][p] = reliability.icc1(r[c].to_numpy(), r["nfl_id"].to_numpy())
    cen = fq.with_columns(_c=pl.col(c) - pl.col(c).mean().over("position"))
    combine_rel[c]["pooled_position_centered"] = reliability.icc1(cen["_c"].to_numpy(), cen["nfl_id"].to_numpy())

# QC figure: Combine fits (8 random passing attempts + 4 failing ones)
rng = np.random.default_rng(6)
show = list(rng.choice(fits.filter(pl.col("qc_ok"))["traj"].to_numpy(), 8, replace=False))
bad = fits.filter(~pl.col("qc_ok"))["traj"].to_numpy()
show += list(rng.choice(bad, min(4, len(bad)), replace=False))
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for ax, tr in zip(axes.flat, show):
    g = frames40.filter(pl.col("traj") == tr)
    r = fits.filter(pl.col("traj") == tr).row(0, named=True)
    t, v = g["t"].to_numpy(), g["speed"].to_numpy()
    ax.plot(t, v, "k.", ms=3, label="SG speed")
    tt = np.linspace(0, t[-1], 200)
    ax.plot(tt, E.mono_exp(tt, r["vmax"], r["tau"], r["t0"]), "b-", label="mono-exp fit")
    ax.axvspan(t[r["i_fit0"]], t[r["i_fit1"] - 1], color="b", alpha=0.07, label="fit window")
    ax2 = ax.twinx()
    ax2.plot(t, np.gradient(v, 0.1), "g-", lw=0.8)
    ax2.set_ylim(-2, 10)
    ax2.set_ylabel("dv/dt (yd/s²)", color="g", fontsize=8)
    ax.set_ylim(0, 12.5)
    ax.set_xlabel("s since first tracked frame")
    ax.set_ylabel("yd/s")
    ax.set_title(f"{'PASS' if r['qc_ok'] else 'QC FAIL'} nfl {r['nfl_id']} att {r['attempt']}: vmax {r['vmax']:.1f} "
                 f"tau {r['tau']:.2f} a0 {r['a0']:.1f} rmse {r['rmse']:.2f}", fontsize=8,
                 color="k" if r["qc_ok"] else "r")
axes.flat[0].legend(fontsize=7, loc="lower right")
fig.suptitle("E006 QC: Combine 40-yd mono-exponential fits (fit from peak dv/dt to peak speed)")
fig.tight_layout()
fig.savefig(FIG / "E006_forty_fits.png", dpi=90)
plt.close(fig)

# ------------------------------------------------------------------ Game side
gframes = E.build_game_frames()
log(f"game frames {gframes.height}")
game = E.game_profiles(gframes)
game = game.with_columns(a_lo=pl.mean_horizontal("qa_3.0", "qa_3.5", "qa_4.0", "qa_4.5"))
sh = E.game_profiles_split_half(gframes)
sh = sh.with_columns(a_lo_h0=pl.mean_horizontal("qa_3.0_h0", "qa_3.5_h0", "qa_4.0_h0", "qa_4.5_h0"),
                     a_lo_h1=pl.mean_horizontal("qa_3.0_h1", "qa_3.5_h1", "qa_4.0_h1", "qa_4.5_h1"))
game = game.rename({"A0": "game_A0", "S0": "game_S0", "slope": "game_as_slope", "a_lo": "game_a_lo",
                    "n_plays": "game_reg_plays"})
game.write_parquet(io.PROCESSED / "probe_e_game_profile.parquet")
gts = features.game_top_speed()  # E001 target, recomputed (q95 of play max speed)
log(f"game profiles {game.filter(pl.col('game_A0').is_finite()).height}")

shp = sh.join(pos_tbl, on="nfl_id")
gp = game.join(pos_tbl, on="nfl_id")
game_rel = {}
for c, short in (("game_A0", "A0"), ("game_S0", "S0"), ("game_as_slope", "slope"), ("game_a_lo", "a_lo")):
    game_rel[c] = {}
    for p in POS:
        rel = E.sb_split_half(shp.filter(pl.col("position") == p), short)
        g = gp.filter((pl.col("position") == p) & pl.col(c).is_finite())
        rel["corr_log_plays"] = float(np.corrcoef(g[c].to_numpy(), np.log(g["game_reg_plays"].to_numpy()))[0, 1])
        rel["mean"], rel["sd"], rel["n"] = float(g[c].mean()), float(g[c].std()), g.height
        game_rel[c][p] = rel
# game top speed split-half (q95 of play max per half) for completeness
pm = (gframes.group_by("nfl_id", "game_id", "play_id").agg(smax=pl.col("s").max())
      .with_columns(half=pl.col("game_id").rank("dense").over("nfl_id") % 2))
hh = (pm.group_by("nfl_id", "half").agg(q=pl.col("smax").quantile(0.95), n=pl.len()).filter(pl.col("n") >= 25)
      .pivot("half", index="nfl_id", values="q").drop_nulls().join(pos_tbl, on="nfl_id"))
game_rel["game_top_speed"] = {}
for p in POS:
    x = hh.filter(pl.col("position") == p)
    r = float(np.corrcoef(x["0"], x["1"])[0, 1])
    game_rel["game_top_speed"][p] = {"r_half": r, "reliability": 2 * r / (1 + r), "n_players": x.height}

# QC figure: game A-S profiles, 3 random players per position
centers = (E.AS_BINS[:-1] + E.AS_BINS[1:]) / 2
fig, axes = plt.subplots(3, 5, figsize=(20, 11))
gfin = gp.filter(pl.col("game_A0").is_finite())
for j, p in enumerate(POS):
    ids = rng.choice(gfin.filter(pl.col("position") == p)["nfl_id"].to_numpy(), 3, replace=False)
    for i, pid in enumerate(ids):
        ax = axes[i, j]
        g = gframes.filter((pl.col("nfl_id") == pid) & (pl.col("a_tan").abs() <= E.MAX_A))
        r = gfin.filter(pl.col("nfl_id") == pid).row(0, named=True)
        sub = g.sample(min(20000, g.height), seed=1)
        ax.plot(sub["speed"], sub["a_tan"], ",", color="0.6", alpha=0.4)
        qa = np.array([r[f"qa_{b:.1f}"] for b in E.AS_BINS[:-1]])
        ax.plot(centers, qa, "ro", ms=4, label="q95 a_tan per bin (mean of 20 draws x 50 plays)")
        ss = np.linspace(0, 12, 50)
        ax.plot(ss, r["game_A0"] + r["game_as_slope"] * ss, "b-", label="linear envelope")
        ax.set_xlim(0, 12)
        ax.set_ylim(-6, 10)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("speed (yd/s)")
        ax.set_ylabel("a_tan (yd/s²)")
        ax.set_title(f"{p} {pid}: A0 {r['game_A0']:.2f} S0 {r['game_S0']:.1f}  REG plays {r['game_reg_plays']}", fontsize=9)
axes[0, 0].legend(fontsize=7, loc="lower left")
fig.suptitle("E006 QC: in-game acceleration-speed profiles (post-snap REG frames)")
fig.tight_layout()
fig.savefig(FIG / "E006_game_as_profiles.png", dpi=80)
plt.close(fig)

# ------------------------------------------------------------------ Outcome: YACOE
yac = E.yacoe_outcome(pos_tbl)
cat = E.catches().join(pos_tbl, on="nfl_id").with_columns(y=pl.col("yacoe").clip(-E.YACOE_WINSOR, E.YACOE_WINSOR))
yac_rel = {}
for p in ["WR", "TE"]:
    c = cat.filter(pl.col("position") == p)
    y = yac.join(pos_tbl, on="nfl_id").filter(pl.col("position") == p)
    yac_rel[p] = {"split_half_min5_per_half": reliability.split_half(c, "y", min_per_half=5),
                  "eb_mean_weight": float(y["yacoe_eb_weight"].mean()), "n_players": y.height,
                  "corr_log_catches": float(np.corrcoef(y["yacoe_eb"], np.log(y["yacoe_catches"]))[0, 1])}

# ------------------------------------------------------------------ Player table
df = (pt.join(prof, on="nfl_id", how="left").join(game, on="nfl_id", how="left")
      .join(gts, on="nfl_id", how="left").join(yac, on="nfl_id", how="left"))

# ------------------------------------------------------------------ Linkage (after preregistration only)
assert PREREG.exists(), "write the preregistration before any linkage"
prereg = json.loads(PREREG.read_text())


def population(pops):
    return df.filter(pl.col("position").is_in(pops))


def summarize(r):
    d = r["delta_r2_vs_B2"]
    return {k: r[k] for k in ("feature", "target", "n", "raw_r", "partial_r", "perm_p", "loco_delta_r2", "checks",
                              "by_class", "cv_r2")} | {"delta_r2_vs_B2": d, "delta_r2_vs_B0": r["delta_r2_vs_B0"]}


def fmt(r):
    d = r["delta_r2_vs_B2"]
    pr = "n/a" if r["partial_r"] is None else f"{r['partial_r']:+.2f}"
    return (f"{r['feature']}->{r['target']} n={r['n']} raw_r={r['raw_r']:+.2f} partial_r={pr} "
            f"dR2_B2={d['mean']:+.3f}±{d['se']:.3f} excess={d['excess']:+.3f} p={r['perm_p']:.4f} "
            f"loco+={r['checks']['loco_positive_classes']}")


PRIMARY = [("P1", "c40_vmax", "game_S0", ["WR"]),
           ("P2", "c40_a0", "game_A0", ["WR", "TE", "DB", "DL"]),
           ("P3", "c40_vmax", "yacoe_eb", ["WR", "TE"])]
for (pid, f, t, pops), pr in zip(PRIMARY, prereg["primary_tests"]):
    assert (pid, f, t) == (pr["id"], pr["feature"], pr["target"]), "script and prereg disagree"

primary = {}
for pid, f, t, pops in PRIMARY:
    d = population(pops)
    primary[pid] = {"population": "+".join(pops), "seeds": {}}
    for seed in (0, 1, 2):
        r = linkage.evaluate(d, f, t, n_repeats=50, n_perm=2000, seed=seed, n_jobs=N_JOBS)
        primary[pid]["seeds"][seed] = summarize(r)
        log(f"{pid} seed {seed}: {fmt(r)}")
    r = linkage.evaluate(d.filter(pl.col("draft_year") != 2025), f, t, n_repeats=50, n_perm=2000, seed=0, n_jobs=N_JOBS)
    primary[pid]["without_2025"] = summarize(r)
    log(f"{pid} without 2025: {fmt(r)}")
    s = primary[pid]["seeds"]
    rel = [combine_rel[f.removeprefix("c40_")][p]["reliability"] for p in pops]
    primary[pid]["feature_reliability_as_used_by_position"] = dict(zip(pops, rel))
    primary[pid]["seeds_passing_all_checks"] = sum(
        v["checks"]["excess_lb_gt_0"] and v["checks"]["perm_p_lt_05"]
        and int(v["checks"]["loco_positive_classes"].split("/")[0]) >= 2 for v in s.values())
p0 = np.array([primary[k]["seeds"][0]["perm_p"] for k in ("P1", "P2", "P3")])
bh = false_discovery_control(p0, method="bh")
for k, q in zip(("P1", "P2", "P3"), bh):
    primary[k]["bh_q_seed0"] = float(q)

EXPLORATORY = (
    [("c40_vmax", "game_S0", [p]) for p in ("TE", "DB")] + [("c40_vmax", "game_S0", ["WR", "TE", "DB"])]
    + [("c40_a0", "game_A0", [p]) for p in ("WR", "TE", "DB", "DL", "OL")]
    + [(f, "game_A0", ["WR", "TE", "DB", "DL"]) for f in ("c40_tau", "c40_F0", "c40_Pmax_rel", "c40_split10_resid")]
    + [("c40_vmax", "game_top_speed", ["WR"]), ("c40_a0", "game_a_lo", POS)]
    + [(f, "yacoe_eb", ["WR", "TE"]) for f in ("c40_a0", "c40_Pmax_rel")]
)
exploratory = []
for f, t, pops in EXPLORATORY:
    r = linkage.evaluate(population(pops), f, t, n_repeats=20, n_perm=500, seed=0, n_jobs=N_JOBS)
    exploratory.append({"population": "+".join(pops), "planned_in_prereg": True} | summarize(r))
    log(f"EXPL {'+'.join(pops)}: {fmt(r)}")

# Post-hoc exploratory (NOT in the prereg; added after seeing the primary results):
#  X1/X2: does the curve fit add anything beyond E001's observed 40 peak speed (c40_smax) for WR S0?
#  X3/X4: the two highest-leverage points in P2 have one two-phase artifact attempt each with RMSE 0.136-0.138,
#         just under the 0.15 QC limit; re-run P1/P2 with the stricter RMSE <= 0.12 QC considered pre-prereg.
STRICT_RMSE = 0.12
prof_strict = E.combine_profile(fits.with_columns(qc_ok=pl.col("qc_ok") & (pl.col("rmse") <= STRICT_RMSE)))
df_strict = df.drop([c for c in prof.columns if c != "nfl_id"]).join(prof_strict, on="nfl_id", how="left")
POSTHOC = [
    ("X1", df, "c40_vmax", "game_S0", ["WR"], dict(combine_controls=linkage.COMBINE_CONTROLS + ["c40_smax"])),
    ("X2", df, "c40_smax", "game_S0", ["WR"], {}),
    ("X3", df_strict, "c40_a0", "game_A0", ["WR", "TE", "DB", "DL"], {}),
    ("X4", df_strict, "c40_vmax", "game_S0", ["WR"], {}),
]
for xid, data, f, t, pops, kw in POSTHOC:
    r = linkage.evaluate(data.filter(pl.col("position").is_in(pops)), f, t, n_repeats=20, n_perm=500, seed=0,
                         n_jobs=N_JOBS, **kw)
    note = {"X1": "c40_smax added to the B1/B2 controls", "X3": f"QC rmse <= {STRICT_RMSE}",
            "X4": f"QC rmse <= {STRICT_RMSE}"}.get(xid, "")
    exploratory.append({"population": "+".join(pops), "planned_in_prereg": False, "posthoc_id": xid,
                        "note": note} | summarize(r))
    log(f"POSTHOC {xid} {'+'.join(pops)} {note}: {fmt(r)}")

# ------------------------------------------------------------------ Result figure: added-variable plots
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
ctrl = linkage.COMBINE_CONTROLS + linkage.DRAFT_CONTROLS
for ax, (pid, f, t, pops) in zip(axes, PRIMARY):
    d = population(pops).filter(pl.col(f).is_finite() & pl.col(t).is_finite())
    X = linkage._design(d, ctrl)
    Xi = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True).fit_transform(X)
    rx = d[f].to_numpy() - LinearRegression().fit(Xi, d[f].to_numpy()).predict(Xi)
    ry = d[t].to_numpy() - LinearRegression().fit(Xi, d[t].to_numpy()).predict(Xi)
    for p in pops:
        m = d["position"].to_numpy() == p
        ax.scatter(rx[m], ry[m], s=14, alpha=0.75, label=p)
    b = np.polyfit(rx, ry, 1)
    xs = np.linspace(rx.min(), rx.max(), 10)
    ax.plot(xs, np.polyval(b, xs), "k-", lw=1)
    s0 = primary[pid]["seeds"][0]
    dd = s0["delta_r2_vs_B2"]
    ax.set_title(f"{pid}: {f} -> {t} ({'+'.join(pops)}, n={s0['n']})\n"
                 f"excess dR2 vs B2 {dd['excess']:+.3f} ± {dd['se']:.3f}, perm p {s0['perm_p']:.3f}", fontsize=9)
    ax.set_xlabel(f"{f} | B2 (residual)")
    ax.set_ylabel(f"{t} | B2 (residual)")
    ax.axhline(0, color="0.8", lw=0.5)
    ax.axvline(0, color="0.8", lw=0.5)
    ax.legend(fontsize=7)
fig.suptitle("E006 primary tests: added-variable plots (in-sample residuals on B2 = Combine times + draft capital)")
fig.tight_layout()
fig.savefig(FIG / "E006_primary_added_variable.png", dpi=90)
plt.close(fig)

# ------------------------------------------------------------------ Save
results = {
    "experiment_id": "E006", "prereg": str(PREREG.relative_to(HERE.parent)),
    "combine_qc": {"attempts": fits.height, "qc_pass": int(fits["qc_ok"].sum()), "players": prof.height,
                   "qc": {k: list(v) if isinstance(v, tuple) else v for k, v in E.QC.items()}},
    "combine_reliability_icc1": combine_rel,
    "game_reliability_split_half": game_rel,
    "yacoe_reliability": yac_rel,
    "as_settings": {"bins": E.AS_BINS.tolist(), "q": E.AS_QUANTILE, "k_plays": E.AS_K, "draws": E.AS_R,
                    "min_frames_per_bin": E.AS_MIN_FRAMES, "min_bins": E.AS_MIN_BINS},
    "primary": primary,
    "exploratory": exploratory,
    "n_exploratory": len(exploratory),
    "runtime_s": time.time() - t_start,
}
OUT.write_text(json.dumps(results, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
log(f"wrote {OUT}")
