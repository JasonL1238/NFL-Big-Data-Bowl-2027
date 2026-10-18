"""E005 — Probe D: DB backpedal -> transition. Combine DB-drill movement vs the same movement in regular-season coverage.

Runs top to bottom: build events -> reliability screens -> segmentation QC figures -> player-level signatures/traits
-> (requires experiments/results/E005_prereg.json) primary tests at seeds 0/1/2 + without-2025 sensitivity
-> labeled exploratory tests -> summary figure -> experiments/results/E005.json.
Feature code: src/bdb27/probes/d_db_transition.py.
"""

import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bdb27 import io, linkage  # noqa: E402
from bdb27.probes import d_db_transition as D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "experiments" / "results", ROOT / "experiments" / "figures"
PREREG = RES / "E005_prereg.json"
N_JOBS = 3
INK, INK2, BLUE, GRID = "#0b0b0b", "#52514e", "#2a78d6", "#e4e3df"
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


# ---------- 1. events ----------
cf = D.combine_frames()
ce = D.combine_events(cf)
cb = D.combine_backpedal(cf)
gf = D.game_frames()
ge = D.game_events(gf)
gb = D.game_backpedal_plays(gf)
for name, df in [("combine_events", ce), ("combine_backpedal", cb), ("game_events", ge), ("game_backpedal_plays", gb)]:
    df.write_parquet(io.PROCESSED / f"probe_d_{name}.parquet")
log(f"events: combine {ce.height} transitions / {cb.height} backpedal legs; game {ge.height} breaks / {gb.height} backpedal plays")

# ---------- 2. reliability screens (before any linkage) ----------
crel = D.combine_reliability(ce, cb)
grel = D.game_reliability(ge, gb)
log("reliability screens done")

# ---------- 3. segmentation QC figures ----------
FIG.mkdir(parents=True, exist_ok=True)
BP_LIM = {"BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL": 1.6, "TERYL_AUSTIN_DRILL_2": 1.2, "W_DRILL": 1.0, "BOX_DRILL": 1.0,
          "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION": 1.0, "TERYL_AUSTIN_DRILL_1": None}
rng = np.random.default_rng(11)
fig, axes = plt.subplots(len(BP_LIM), 4, figsize=(20, 4.2 * len(BP_LIM)))
for r, dn in enumerate(BP_LIM):
    sub = cf.filter(pl.col("drill_name") == dn)
    for c, t in enumerate(rng.choice(sub["traj"].unique().to_numpy(), 4, replace=False)):
        ax = axes[r, c]
        tt = sub.filter(pl.col("traj") == t).sort("time")
        x, y, s, ton = (tt[k].to_numpy() for k in ("x", "y", "speed", "t_on"))
        ax.plot(x, y, "-", c="0.6", lw=0.8)
        ax.scatter(x, y, c=s, s=6, cmap="viridis", vmin=0, vmax=8)
        if BP_LIM[dn] is not None:
            m = (ton >= 0) & (ton <= BP_LIM[dn])
            ax.plot(x[m], y[m], "-", c="tab:orange", lw=3, alpha=0.6)
        for e in ce.filter(pl.col("traj") == t).iter_rows(named=True):
            i = e["i_break"]
            ax.plot(x[max(0, i - 5):i + 6], y[max(0, i - 5):i + 6], "-", c="tab:red", lw=3, alpha=0.7)
            ax.plot(x[i], y[i], "o", mfc="none", mec="k", ms=9)
            ax.annotate(f"cut {e['cut']}: dv={e['dv_10']:.1f}\nentry={e['entry_speed']:.1f}", (x[i], y[i]), fontsize=7)
        ax.set_aspect("equal")
        ax.set_title(f"{dn[:34]} (traj {t})", fontsize=8)
fig.suptitle("E005 Combine QC (canonical frame, color = speed): orange = backpedal window; red = +-0.5 s dv window "
             "around the plant (o)", y=1.0)
plt.tight_layout()
plt.savefig(FIG / "E005_qc_combine.png", dpi=60)
plt.close(fig)

gff = gf.sort("traj", "time").with_columns(fi=pl.int_range(pl.len()).over("traj"))
E = ge.filter(pl.col("from_backpedal") & (pl.col("phase") == "ball_in_air")).sample(16, seed=5)
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
for ax, e in zip(axes.flat, E.iter_rows(named=True)):
    t = gff.filter((pl.col("traj") == e["traj"]) & (pl.col("t_rel") >= 0) & (pl.col("t_rel") <= e["t_end"] + 1.0))
    x, y, o, tr, vx, fi = (t[k].to_numpy() for k in ("x", "y", "o", "t_rel", "vx", "fi"))
    ax.plot(x, y, "k-", lw=0.8)
    bpm = (vx > 0.5) & (np.abs(((o - 270 + 180) % 360) - 180) <= 60) & (tr <= min(2.5, e["t_pf"]))
    ax.plot(x[bpm], y[bpm], ".", c="tab:orange", ms=5)
    for j in range(0, len(x), 3):
        ax.arrow(x[j], y[j], 0.5 * np.sin(np.radians(o[j])), 0.5 * np.cos(np.radians(o[j])), head_width=0.12,
                 color="tab:blue", lw=0.4)
    ib = np.flatnonzero(fi == e["i_break"])
    if len(ib):
        ib = int(ib[0])
        ax.plot(x[max(0, ib - 5):ib + 6], y[max(0, ib - 5):ib + 6], "-", c="tab:red", lw=3, alpha=0.6)
        ax.plot(x[ib], y[ib], "o", mfc="none", mec="r", ms=10)
    ipf = np.flatnonzero(np.isclose(tr, e["t_pf"]))
    if len(ipf):
        ax.plot(x[ipf[0]], y[ipf[0]], "k*", ms=12)
    ax.set_aspect("equal")
    ax.set_title(f"{e['lined_up_position']} {e['coverage_assignment']} turn={e['turn_deg']:.0f}\n"
                 f"entry={e['entry_speed']:.1f} min={e['min_speed']:.1f} dv={e['dv_10']:.1f} t={e['t_rel_break']:.1f}s",
                 fontsize=8)
fig.suptitle("E005 game QC: post-throw transitions out of the backpedal (offense -> +x). orange = backpedal frames; "
             "red = dv window; * = pass_forward; arrows = orientation", y=1.0)
plt.tight_layout()
plt.savefig(FIG / "E005_qc_game.png", dpi=60)
plt.close(fig)
log("QC figures written")

# ---------- 4. player-level signatures and traits ----------
cs = D.combine_signatures(ce, cb)
gt = D.game_traits(ge, gb, gf)
pt = linkage.player_table().filter(pl.col("position") == "DB")
df = pt.join(cs, on="nfl_id", how="left").join(gt, on="nfl_id", how="left")

exposure = {}
for t, n in [("gm_backpedal_speed", "gm_backpedal_n"), ("gm_ballair_dv", "gm_ballair_n")]:
    d = df.filter(pl.col(t).is_finite())
    exposure[t] = {"n": d.height,
                   "corr_log_cov_pass_plays": float(np.corrcoef(d[t], np.log(d["cov_pass_plays"]))[0, 1]),
                   "corr_log_events": float(np.corrcoef(d[t], np.log(d[n]))[0, 1]),
                   "eb_weight_median": float(d[n.replace("_n", "_w")].median())}
feature_vs_controls = {}
for f in ["cmb_backpedal_speed", "cmb_transition_dv"]:
    d = df.filter(pl.col(f).is_finite())
    feature_vs_controls[f] = {c: float(d.select(pl.corr(f, c)).item()) for c in
                              ["forty", "ten_yd_split", "three_cone", "short_shuttle", "ngs_athleticism_score", "log_pick"]}
log(f"players: {df.height} DBs; exposure {exposure}")

# ---------- 5. preregistered primary tests ----------
assert PREREG.exists(), "write experiments/results/E005_prereg.json before running linkage"
prereg = json.loads(PREREG.read_text())
primary = {}
for p in prereg["primary_tests"]:
    f, tgt = p["feature"], p["target"]
    runs = {}
    for seed in (0, 1, 2):
        runs[seed] = linkage.evaluate(df, f, tgt, n_repeats=50, n_perm=2000, seed=seed, n_jobs=N_JOBS)
        r = runs[seed]
        log(f"{p['id']} seed {seed}: n={r['n']} raw_r={r['raw_r']:+.3f} partial_r={r['partial_r']} "
            f"dR2={r['delta_r2_vs_B2']['mean']:+.4f}+-{r['delta_r2_vs_B2']['se']:.4f} excess={r['delta_r2_vs_B2']['excess']:+.4f} "
            f"p={r['cv_perm_p']:.4f} loco={r['loco']}")
    no25 = linkage.evaluate(df.filter(pl.col("draft_year") != 2025), f, tgt, n_repeats=50, n_perm=2000, seed=0, n_jobs=N_JOBS)
    log(f"{p['id']} without 2025: n={no25['n']} excess={no25['delta_r2_vs_B2']['excess']:+.4f} p={no25['cv_perm_p']:.4f}")
    primary[p["id"]] = {"spec": p, "seeds": {str(k): v for k, v in runs.items()}, "without_2025": no25}

# BH-FDR across the primary tests (seed-0 p values)
ids = list(primary)
ps = np.array([primary[i]["seeds"]["0"]["cv_perm_p"] for i in ids])
order = np.argsort(ps)
q = np.empty_like(ps)
m = len(ps)
prev = 1.0
for rank, idx in reversed(list(enumerate(order, start=1))):
    prev = min(prev, ps[idx] * m / rank)
    q[idx] = prev
for i, qi in zip(ids, q):
    primary[i]["bh_q"] = float(qi)

# go criteria
def _rel(table, **eq):
    row = table.filter(pl.all_horizontal([pl.col(k) == v for k, v in eq.items()]))
    return row


rel_used = {  # reliability of each primary feature/target exactly as preregistered
    "P1": (_rel(crel, feature="bp_peak", scope="backpedal drill means (5 drills)")["reliability"].item(),
           _rel(grel, event_set="backpedal_plays", feature="bp_peak")["rel_min10"].item()),
    "P2": (_rel(crel, feature="dv_10", adjust="entry+turn", scope="reversal drill means (4 drills)")["reliability"].item(),
           _rel(grel, event_set="from_backpedal_ball_in_air", feature="dv_10")["rel_min5"].item()),
}
for i in ids:
    s = primary[i]["seeds"]
    seeds_pass = []
    for k in ("0", "1", "2"):
        r = s[k]
        ok = (r["checks"]["excess_lb_gt_0"] and r["checks"]["fl_p_lt_05"]
              and int(r["checks"]["loco_excess_positive_classes"].split("/")[0]) >= 2
              and (r["partial_r"] is not None and r["partial_r"] > 0))
        seeds_pass.append(ok)
    fr, tr_ = rel_used[i]
    primary[i]["go"] = {"seeds_passing_all_automatable_checks": f"{sum(seeds_pass)}/3",
                        "feature_reliability": fr, "target_reliability": tr_,
                        "reliability_ok": bool(fr >= 0.5 and tr_ >= 0.5),
                        "bh_q_lt_05": bool(primary[i]["bh_q"] < 0.05)}

# ---------- 6. exploratory tests (labeled; n_repeats=20, n_perm=500, seed 0) ----------
dx = df
cx = [D.player_composite(ce, "drive_05", D.REVERSAL_DRILLS, "cmb_drive05"),
      D.player_composite(ce, "peak_reaccel", D.REVERSAL_DRILLS, "cmb_reaccel"),
      D.player_composite(ce, "dv_10", ["TERYL_AUSTIN_DRILL_1"], "cmb_dv_ta1", adjust=["entry_speed", "aturn"]),
      D.player_composite(ce, "dv_10", ["W_DRILL"], "cmb_dv_w", adjust=["entry_speed", "aturn"])]
bair = pl.col("from_backpedal") & (pl.col("phase") == "ball_in_air")
prep = pl.col("from_backpedal") & (pl.col("phase") == "pre_pass")
gx = [D.game_event_trait(ge, "drive_05", bair, "gm_ballair_drive05", D.MIN_BALLAIR_TRANSITIONS),
      D.game_event_trait(ge, "peak_reaccel", bair, "gm_ballair_reaccel", D.MIN_BALLAIR_TRANSITIONS),
      D.game_event_trait(ge, "dv_10", prep, "gm_prepass_dv", D.MIN_BALLAIR_TRANSITIONS),
      D.game_backpedal_trait(gb, pl.col("team_coverage_man_zone") == "MAN_COVERAGE", "gm_backpedal_man"),
      D.game_backpedal_trait(gb, pl.col("team_coverage_man_zone") == "ZONE_COVERAGE", "gm_backpedal_zone")]
for x in cx + gx:
    dx = dx.join(x, on="nfl_id", how="left")
EXPLORATORY = [
    ("X1", "cmb_drive05", "gm_ballair_drive05", "drive (speed gained 0.5 s after plant), unadjusted Combine z"),
    ("X2", "cmb_reaccel", "gm_ballair_reaccel", "peak re-acceleration out of the plant"),
    ("X3", "cmb_transition_dv", "gm_prepass_dv", "dv in PRE-throw transitions (game trait unreliable)"),
    ("X4a", "cmb_backpedal_speed", "gm_backpedal_man", "backpedal speed, man-coverage plays only"),
    ("X4b", "cmb_backpedal_speed", "gm_backpedal_zone", "backpedal speed, zone-coverage plays only"),
    ("X5a", "cmb_dv_ta1", "gm_ballair_dv", "transition dv from TA1 only"),
    ("X5b", "cmb_dv_w", "gm_ballair_dv", "transition dv from W drill only"),
]
exploratory = {}
for xid, f, tgt, desc in EXPLORATORY:
    r = linkage.evaluate(dx, f, tgt, n_repeats=20, n_perm=500, seed=0, n_jobs=N_JOBS)
    exploratory[xid] = {"description": desc, "result": r}
    log(f"{xid} {f}->{tgt}: n={r['n']} raw_r={r['raw_r']:+.3f} partial_r={r['partial_r']} "
        f"excess={r['delta_r2_vs_B2']['excess']:+.4f}+-{r['delta_r2_vs_B2']['se']:.4f} p={r['cv_perm_p']:.4f} "
        f"loco+={r['checks']['loco_excess_positive_classes']}")

# ---------- 7. summary figure: added-variable plots for the primaries ----------
fig, axes = plt.subplots(1, len(ids), figsize=(6.2 * len(ids), 5.2))
for ax, i in zip(np.atleast_1d(axes), ids):
    spec, r = primary[i]["spec"], primary[i]["seeds"]["0"]
    d = df.filter(pl.col(spec["feature"]).is_finite() & pl.col(spec["target"]).is_finite()).sort("nfl_id")
    Xi = linkage._imputed(linkage._design(d, linkage.COMBINE_CONTROLS + linkage.DRAFT_CONTROLS))  # same B2 design as the harness
    xr = linkage._residual_maker(Xi) @ d[spec["feature"]].to_numpy().astype(float)
    yr = linkage._residual_maker(Xi) @ d[spec["target"]].to_numpy().astype(float)
    ax.scatter(xr, yr, s=36, c=BLUE, edgecolors="white", linewidths=1.0, zorder=3)
    b = np.polyfit(xr, yr, 1)
    xs = np.linspace(xr.min(), xr.max(), 50)
    ax.plot(xs, np.polyval(b, xs), "-", c=INK2, lw=2, zorder=2)
    ax.axhline(0, c=GRID, lw=1, zorder=1)
    ax.axvline(0, c=GRID, lw=1, zorder=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlabel(f"{spec['feature']} | B2 (residual, z units)", color=INK2)
    ax.set_ylabel(f"{spec['target']} | B2 (residual, yd/s)", color=INK2)
    pr = r["partial_r"]
    ax.set_title(f"{i}: Combine {spec['feature']} -> game {spec['target']}\n"
                 f"n={r['n']}, partial r={pr:+.2f}, excess dR2 vs B2={r['delta_r2_vs_B2']['excess']:+.3f} "
                 f"(SE {r['delta_r2_vs_B2']['se']:.3f}), perm p={r['cv_perm_p']:.3f}", fontsize=9.5, color=INK, loc="left")
fig.suptitle("E005: DB drill signatures vs the same movement in games, beyond Combine times + draft pick "
             "(added-variable plots)", fontsize=11, color=INK, x=0.01, ha="left")
plt.tight_layout()
plt.savefig(FIG / "E005_primary_added_variable.png", dpi=120)
plt.close(fig)

# ---------- 8. results ----------
out = {
    "experiment_id": "E005", "prereg": str(PREREG.relative_to(ROOT)),
    "counts": {"combine_transitions": ce.height, "combine_backpedal_legs": cb.height, "game_breaks_in_window": ge.height,
               "game_backpedal_plays": gb.height, "game_db_players": int(gf["nfl_id"].n_unique()),
               "game_ballair_backpedal_transitions": int(ge.filter(bair).height)},
    "reliability": {"combine": crel.to_dicts(), "game": grel.to_dicts()},
    "exposure": exposure, "feature_vs_controls_r": feature_vs_controls,
    "primary": primary, "exploratory": exploratory, "n_exploratory": len(exploratory),
    "figures": ["experiments/figures/E005_qc_combine.png", "experiments/figures/E005_qc_game.png",
                "experiments/figures/E005_primary_added_variable.png"],
}
(RES / "E005.json").write_text(json.dumps(out, indent=2, default=float))
log("wrote experiments/results/E005.json")
