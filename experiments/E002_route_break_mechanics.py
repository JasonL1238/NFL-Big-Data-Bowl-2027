"""E002 — Probe A: WR/TE route-break mechanics, Combine route drills -> regular-season routes.

Preregistration (written before any Combine-vs-game association): experiments/results/E002_prereg.json.
Runs top to bottom: break events (Combine + game), QC figures, reliability screen, player-level
signatures/traits, separation over expected, 3 primary tests x 3 seeds, labelled exploratory tests,
figures and experiments/results/E002.json. Feature code: src/bdb27/probes/a_route_breaks.py.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

from bdb27 import io, linkage, reliability, segments  # noqa: E402
from bdb27.probes import a_route_breaks as A  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "experiments" / "results", ROOT / "experiments" / "figures"
N_JOBS = 3
FEATS = ["entry_speed", "min_speed", "retention", "speed_drop", "peak_decel", "mean_decel", "decel_time",
         "peak_reaccel", "turn_radius", "peak_alat"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
assert (RES / "E002_prereg.json").exists(), "preregistration must exist before any linkage"
PREREG = json.loads((RES / "E002_prereg.json").read_text())

# ---------------------------------------------------------------- 1. break events
positions = io.load("combine_results").select("nfl_id", position="combine_position")
cframes = A.combine_frames(list(A.DRILLS) + A.QC_EXCLUDED_DRILLS)
cev = A.combine_events(cframes).join(positions, on="nfl_id", how="left")
plays = A.game_route_plays()
gframes, gtimes = A.game_frames(plays)
gev = A.game_events(gframes, plays, gtimes)
soe = A.separation_over_expected(plays)
detection = {
    "combine_found_rate": dict(cev.group_by("drill_name").agg(pl.col("found").mean()).sort("drill_name").iter_rows()),
    "game_found_rate": dict(gev.group_by("route_ran").agg(pl.col("found").mean()).sort("route_ran").iter_rows()),
    "n_combine_attempts": cev.height, "n_game_route_plays": plays.height, "n_game_events_rows": gev.height,
    "soe_plays": soe.height,
    "soe_context_model_play_r2": float(1 - soe["soe"].var() / soe["separation_at_pass_forward"].var()),
}


# ---------------------------------------------------------------- 2. QC figures
def qc_combine(path, n=12, seed=0):
    rng = np.random.default_rng(seed)
    drills = list(A.DRILLS) + A.QC_EXCLUDED_DRILLS
    fig, axes = plt.subplots(len(drills), 2, figsize=(13, 2.6 * len(drills)))
    for r, d in enumerate(drills):
        sub = cframes.filter(pl.col("drill_name") == d)
        e = cev.filter(pl.col("drill_name") == d)
        for k, t in enumerate(rng.choice(sub["traj"].unique().to_numpy(), size=min(n, sub["traj"].n_unique()), replace=False)):
            s = sub.filter(pl.col("traj") == t).sort("time")
            c = f"C{k % 10}"
            axes[r, 0].plot(s["x"], s["y"], lw=0.8, color=c)
            axes[r, 1].plot(s["tt"], s["speed"], lw=0.8, color=c)
            row = e.filter(pl.col("traj") == t)
            if row.height and row["found"][0]:
                i = row["i_break"][0]
                axes[r, 0].scatter(s["x"][i], s["y"][i], marker="x", s=40, color=c)
                axes[r, 1].scatter(s["tt"][i], s["speed"][i], marker="x", s=40, color=c)
        tag = "EXCLUDED (no identifiable break)" if d in A.QC_EXCLUDED_DRILLS else f"{A.DRILLS[d][0]} break"
        axes[r, 0].set_title(f"{d} - {tag}: canonical path (yd), x = break", fontsize=8)
        axes[r, 0].set_aspect("equal")
        axes[r, 1].set_title(f"{d}: speed (yd/s) vs s since onset", fontsize=8)
        axes[r, 1].set_xlim(-0.5, 7)
    fig.tight_layout()
    fig.savefig(path, dpi=55)
    plt.close(fig)


def qc_game(path, n=14, seed=5):
    rng = np.random.default_rng(seed)
    f = gframes.join(gev.select(*segments.PLAY_KEYS, "route_ran", "found", "i_break", "t_end"),
                     on=segments.PLAY_KEYS, how="inner").sort("traj", "time")
    x0 = f.filter(pl.col("t_rel") >= 0).group_by("traj").agg(x0=pl.col("x").first(), y0=pl.col("y").first())
    f = f.join(x0, on="traj", maintain_order="left").with_columns(dx=pl.col("x") - pl.col("x0"), dy=pl.col("y") - pl.col("y0"))
    routes = list(A.GAME_ROUTES)
    fig, axes = plt.subplots(len(routes), 2, figsize=(13, 2.8 * len(routes)))
    for r, rt in enumerate(routes):
        sub = f.filter(pl.col("route_ran") == rt)
        for k, t in enumerate(rng.choice(sub["traj"].unique().to_numpy(), n, replace=False)):
            s = sub.filter(pl.col("traj") == t)
            sgn = -1 if s.filter(pl.col("t_rel") <= pl.col("t_end") + 1)["dy"].mean() < 0 else 1
            c = f"C{k % 10}"
            show = s.filter(pl.col("t_rel") <= pl.col("t_end") + 1.0)
            pre = s.filter(pl.col("t_rel") <= pl.col("t_end"))
            axes[r, 0].plot(show["dx"], sgn * show["dy"], lw=0.5, color=c, alpha=0.6)
            axes[r, 0].plot(pre["dx"], sgn * pre["dy"], lw=1.2, color=c)
            axes[r, 1].plot(show["t_rel"], show["speed"], lw=0.8, color=c)
            if s["found"][0]:
                i = s["i_break"][0]
                axes[r, 0].scatter(s["dx"][i], sgn * s["dy"][i], marker="x", s=40, color=c)
                axes[r, 1].scatter(s["t_rel"][i], s["speed"][i], marker="x", s=40, color=c)
        used = "trait route" if any(rt in v for v in A.TRAIT_ROUTES.values()) else "exploratory only"
        axes[r, 0].set_title(f"{rt} ({used}): path from snap spot, bold = until throw, x = break", fontsize=8)
        axes[r, 0].set_aspect("equal")
        axes[r, 1].set_title(f"{rt}: speed (yd/s) vs s since snap", fontsize=8)
        axes[r, 1].set_xlim(-0.5, 5.5)
    fig.tight_layout()
    fig.savefig(path, dpi=55)
    plt.close(fig)


qc_combine(FIG / "E002_combine_breaks_qc.png")
qc_game(FIG / "E002_game_breaks_qc.png")

# ---------------------------------------------------------------- 3. reliability screen
rel_combine = []
for pos in ("WR", "TE"):
    e = cev.filter(pl.col("position") == pos)
    for fam in ("stop", "turn"):
        for f in FEATS:
            ef = e.filter((pl.col("family") == fam) & pl.col("found") & pl.col(f).is_finite())
            if ef.height == 0:
                continue
            c = ef.with_columns(v=pl.col(f) - pl.col(f).mean().over("drill"),
                                g=pl.concat_str([pl.col("nfl_id").cast(pl.String), pl.col("drill")]))
            rt = reliability.icc1(c["v"].to_numpy(), c["g"].to_numpy())
            dv = A.drill_values(e, f, fam)
            cd = reliability.icc1(dv["z"].to_numpy(), dv["nfl_id"].to_numpy())
            rel_combine.append(dict(position=pos, family=fam, feature=f, retest_icc=rt["icc"], retest_n=rt["n_repeated"],
                                    cross_drill_icc=cd["icc"], composite_reliability=cd["reliability"],
                                    n_players=cd["n_players"], n_drills=int(dv["drill"].n_unique())))
rel_combine = pl.DataFrame(rel_combine)

route_plays = plays.group_by("nfl_id").agg(route_plays=pl.len())
rel_game = []
for fam, routes in A.TRAIT_ROUTES.items():
    e = gev.filter(pl.col("route_ran").is_in(routes) & pl.col("found"))
    for f in FEATS:
        c = e.filter(pl.col(f).is_finite()).with_columns(v=pl.col(f) - pl.col(f).mean().over("route_ran"))
        for pos in ("WR", "TE"):
            sh = reliability.split_half(c.filter(pl.col("position") == pos), "v")
            tr = A.game_trait(e.filter(pl.col("position") == pos), f, routes, "t").join(route_plays, on="nfl_id")
            rel_game.append(dict(position=pos, family=fam, feature=f, split_half=sh["reliability"], n_players=sh["n_players"],
                                 r_trait_log_route_plays=float(np.corrcoef(tr["t"], np.log(tr["route_plays"]))[0, 1]),
                                 r_rawmean_log_events=float(np.corrcoef(tr["t_raw"], np.log(tr["t_n"]))[0, 1])))
rel_game = pl.DataFrame(rel_game)

SOE_VARIANTS = {"soe_all": None, "soe_nogo": ["HITCH", "OUT", "IN", "SLANT", "CORNER", "POST", "CROSS", "FLAT", "WHEEL", "ANGLE"],
                "soe_hoi": ["HITCH", "OUT", "IN"]}
rel_soe = {}
for name, routes in SOE_VARIANTS.items():
    for pos in ("WR", "TE"):
        d = soe.filter(pl.col("position") == pos)
        d = d if routes is None else d.filter(pl.col("route_ran").is_in(routes))
        tr = A.soe_trait(d, None, "t").join(route_plays, on="nfl_id")
        rel_soe[f"{name}_{pos}"] = reliability.split_half(d, "soe") | {
            "r_trait_log_route_plays": float(np.corrcoef(tr["t"], np.log(tr["route_plays"]))[0, 1])}
rel_soe["raw_separation_WR"] = reliability.split_half(soe.filter(pl.col("position") == "WR"), "separation_at_pass_forward")

# ---------------------------------------------------------------- 4. player tables
csig = A.combine_signatures(cev, FEATS)
traits = []
for fam, routes in A.TRAIT_ROUTES.items():
    for f in FEATS:
        traits.append(A.game_trait(gev, f, routes, f"g_{fam}_{f}"))
for name, routes in SOE_VARIANTS.items():
    traits.append(A.soe_trait(soe, routes, name))
gtab = traits[0]
for t in traits[1:]:
    gtab = gtab.join(t, on="nfl_id", how="full", coalesce=True)
gtab = gtab.join(route_plays, on="nfl_id", how="left")
csig.write_parquet(io.PROCESSED / "probe_a_combine_signatures.parquet")
gtab.write_parquet(io.PROCESSED / "probe_a_game_traits.parquet")
eb_info = {}
for name, d, v in [("g_turn_retention", gev.filter(pl.col("route_ran").is_in(["OUT", "IN"]) & pl.col("found")), "retention"),
                   ("g_stop_peak_decel", gev.filter((pl.col("route_ran") == "HITCH") & pl.col("found")), "peak_decel"),
                   ("soe_hoi", soe.filter(pl.col("route_ran").is_in(SOE_VARIANTS["soe_hoi"])), "soe")]:
    if v != "soe":
        d = d.with_columns(pl.col(v) - pl.col(v).mean().over("route_ran"))
    s = A.eb_shrink(d, v, min_n=A.MIN_SOE_PLAYS if v == "soe" else A.MIN_EVENTS)
    eb_info[name] = {p: {"tau2": float(g["tau2"][0]), "sigma2": float(g["sigma2"][0]), "mean_weight": float(g["weight"].mean()),
                         "n_players": g.height} for (p,), g in s.group_by("position")}

top = pl.read_parquet(io.PROCESSED / "combine_features.parquet").select("nfl_id", "combine_top_speed")
df = (linkage.player_table().join(csig, on="nfl_id", how="left").join(gtab, on="nfl_id", how="left")
      .join(top, on="nfl_id", how="left"))
WR = df.filter(pl.col("position") == "WR")
WRTE = df.filter(pl.col("position").is_in(["WR", "TE"]))


# ---------------------------------------------------------------- 5. primary tests
def run(d, feature, target, seed, n_repeats, n_perm, **kw):
    r = linkage.evaluate(d, feature, target, n_repeats=n_repeats, n_perm=n_perm, seed=seed, n_jobs=N_JOBS, **kw)
    pr = "n/a" if r["partial_r"] is None else f"{r['partial_r']:+.3f}"
    print(f"  {feature} -> {target} n={r['n']} seed={seed} raw_r={r['raw_r']:+.3f} partial_r={pr} "
          f"dR2vsB2={r['delta_r2_vs_B2']['mean']:+.4f}+-{r['delta_r2_vs_B2']['se']:.4f} excess={r['delta_r2_vs_B2']['excess']:+.4f} "
          f"p={r['perm_p']:.4f} loco={r['checks']['loco_positive_classes']}", flush=True)
    return r


primary = {}
for t in PREREG["primary_tests"]:
    print(f"PRIMARY {t['id']}: {t['feature']} -> {t['target']} ({t['population']})", flush=True)
    seeds = {s: run(WR, t["feature"], t["target"], s, 50, 2000) for s in (0, 1, 2)}
    entry = {"spec": t, "seeds": seeds}
    if t["target"].startswith("soe"):
        print("  without 2025 class:", flush=True)
        entry["without_2025"] = run(WR.filter(pl.col("draft_year") != 2025), t["feature"], t["target"], 0, 50, 2000)
    primary[t["id"]] = entry


def bh(pvals):
    p = np.asarray(pvals)
    order = np.argsort(p)
    q = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


bh_q = bh([primary[k]["seeds"][0]["perm_p"] for k in primary])
for k, q in zip(primary, bh_q):
    primary[k]["bh_q_seed0"] = float(q)

# ---------------------------------------------------------------- 6. exploratory tests (labelled, counted)
exploratory = {}
ctrl_top = linkage.COMBINE_CONTROLS + ["combine_top_speed"]
print("EXPLORATORY", flush=True)
for t in PREREG["primary_tests"]:
    exploratory[f"X1_{t['id']}_plus_combine_top_speed"] = run(WR, t["feature"], t["target"], 0, 20, 500, combine_controls=ctrl_top)
for t in PREREG["primary_tests"][:2]:  # no TE stop composite (1 TE stop drill), so P3 cannot be pooled
    exploratory[f"X2_{t['id']}_pooled_WR_TE"] = run(WRTE, t["feature"], t["target"], 0, 20, 500)
exploratory["X3_turn_min_speed"] = run(WR, "c_turn_min_speed", "g_turn_min_speed", 0, 20, 500)
exploratory["X4a_turn_retention_to_soe_all"] = run(WR, "c_turn_retention", "soe_all", 0, 20, 500)
exploratory["X4b_turn_retention_to_soe_nogo"] = run(WR, "c_turn_retention", "soe_nogo", 0, 20, 500)
# Added after the primary results (labelled post hoc): a pipeline positive control, and peak lateral acceleration,
# which (v^2 = a_lat * R) is not tied to the speed-vs-radius trade-off that dominates game retention (r = 0.88
# with game turn radius, see diagnostics).
exploratory["X6_posthoc_positive_control_turn_entry_speed"] = run(WR, "c_turn_entry_speed", "g_turn_entry_speed", 0, 20, 500)
exploratory["X7_posthoc_turn_peak_alat"] = run(WR, "c_turn_peak_alat", "g_turn_peak_alat", 0, 20, 500)


def boot_r(x, y, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    rs = np.array([np.corrcoef(x[i], y[i])[0, 1] for i in idx])
    return float(np.corrcoef(x, y)[0, 1]), [float(np.quantile(rs, 0.025)), float(np.quantile(rs, 0.975))]


for name, g in [("X5a_game_turn_retention_vs_soe_hoi", "g_turn_retention"), ("X5b_game_stop_peak_decel_vs_soe_hoi", "g_stop_peak_decel")]:
    d = WR.filter(pl.col(g).is_finite() & pl.col("soe_hoi").is_finite())
    r, ci = boot_r(d[g].to_numpy(), d["soe_hoi"].to_numpy())
    exploratory[name] = {"type": "game-game Pearson r, player bootstrap 95% CI", "n": d.height, "r": r, "ci95": ci}
    print(f"  {name}: n={d.height} r={r:+.3f} CI={ci[0]:+.3f}..{ci[1]:+.3f}", flush=True)



def corr(d, a, b):
    d = d.filter(pl.col(a).is_finite() & pl.col(b).is_finite())
    return {"r": float(np.corrcoef(d[a], d[b])[0, 1]), "n": d.height}


# Within-domain diagnostics (Combine-Combine, game-game; not linkage tests).
gts = pl.read_parquet(io.PROCESSED / "game_traits.parquet").select("nfl_id", "game_top_speed")
WRg = WR.join(gts, on="nfl_id", how="left").with_columns(log_route_plays=pl.col("route_plays").log())
diagnostics = {
    "combine": {f"{a}~{b}": corr(WR, a, b) for a, b in [
        ("c_turn_entry_speed", "forty"), ("c_turn_entry_speed", "combine_top_speed"), ("c_turn_retention", "forty"),
        ("c_turn_retention", "combine_top_speed"), ("c_stop_peak_decel", "short_shuttle"), ("c_stop_peak_decel", "forty"),
        ("c_turn_retention", "c_stop_peak_decel")]},
    "game": {f"{a}~{b}": corr(WRg, a, b) for a, b in [
        ("g_turn_entry_speed", "game_top_speed"), ("g_stop_entry_speed", "game_top_speed"), ("g_turn_retention", "g_turn_turn_radius"),
        ("g_turn_retention", "g_turn_entry_speed"), ("g_turn_retention", "game_top_speed"), ("g_stop_peak_decel", "game_top_speed"),
        ("g_turn_retention", "log_route_plays"), ("g_stop_peak_decel", "log_route_plays"), ("soe_hoi", "log_route_plays")]},
}
for k, v in {**diagnostics["combine"], **diagnostics["game"]}.items():
    print(f"  diag {k}: r={v['r']:+.3f} n={v['n']}", flush=True)

# ---------------------------------------------------------------- 7. figures
def partial_resid(d, feature, target):
    d = d.filter(pl.col(feature).is_finite() & pl.col(target).is_finite())
    X = SimpleImputer(strategy="median", add_indicator=True).fit_transform(
        d.select(linkage.COMBINE_CONTROLS + linkage.DRAFT_CONTROLS).cast(pl.Float64).to_numpy())
    res = [d[c].to_numpy() - LinearRegression().fit(X, d[c].to_numpy()).predict(X) for c in (feature, target)]
    return res[0], res[1], d["draft_year"].to_numpy()


labels = {"c_turn_retention": "Combine speed kept through 90° breaks\n(drill z, resid. on B2)",
          "c_stop_peak_decel": "Combine braking into stop routes\n(drill z, resid. on B2)",
          "g_turn_retention": "Game OUT/IN speed retention (resid.)", "soe_hoi": "Separation over expected,\nHITCH/OUT/IN (yd, resid.)",
          "g_stop_peak_decel": "Game HITCH peak decel (yd/s², resid.)"}
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, (k, ent) in zip(axes, primary.items()):
    sp = ent["spec"]
    x, y, yr = partial_resid(WR, sp["feature"], sp["target"])
    for cls, col in zip((2023, 2024, 2025), (C1, C2, C3)):
        m = yr == cls
        ax.scatter(x[m], y[m], s=22, color=col, edgecolor=SURF, linewidth=0.8, label=f"{cls} class", zorder=3)
    b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 10)
    ax.plot(xs, np.polyval(b, xs), color=INK2, lw=1.5, zorder=2)
    r0 = ent["seeds"][0]
    ax.set_title(f"{k}: partial r = {r0['partial_r']:+.2f}, excess ΔR² = {r0['delta_r2_vs_B2']['excess']:+.3f}\n"
                 f"perm p = {r0['perm_p']:.3f}, LOCO+ {r0['checks']['loco_positive_classes']}, n = {r0['n']}",
                 fontsize=8.5, color=INK)
    ax.set_xlabel(labels[sp["feature"]], fontsize=8.5)
    ax.set_ylabel(labels[sp["target"]], fontsize=8.5)
axes[0].legend(frameon=False, fontsize=8, loc="best")
fig.suptitle("E002 primary tests (WR): Combine route-break signatures vs regular-season traits, beyond Combine times + draft pick",
             fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(FIG / "E002_primary_partial_residuals.png", dpi=110)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1]})
rc = rel_combine.filter(pl.col("position") == "WR").sort("family", "feature")
lab = [f"{a} · {b}" for a, b in zip(rc["family"], rc["feature"])]
yy = np.arange(rc.height)
axes[0].barh(yy - 0.2, rc["composite_reliability"].fill_nan(0).to_numpy(), height=0.38, color=C1, label="cross-drill composite (Spearman-Brown)")
axes[0].barh(yy + 0.2, rc["retest_icc"].fill_nan(0).to_numpy(), height=0.38, color=C2, label="same-drill retest ICC (15-39 players)")
axes[0].set_yticks(yy, lab, fontsize=7.5)
axes[0].axvline(0.5, color=INK2, lw=1, ls="--")
axes[0].set_xlim(-0.2, 1)
axes[0].set_title("Combine WR break features: reliability", fontsize=9)
axes[0].legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2)
rg = rel_game.filter(pl.col("position") == "WR").sort("family", "feature")
lab = [f"{'HITCH' if a == 'stop' else 'OUT/IN'} · {b}" for a, b in zip(rg["family"], rg["feature"])]
yy = np.arange(rg.height)
axes[1].barh(yy, rg["split_half"].fill_nan(0).to_numpy().clip(-0.2, 1), height=0.6, color=C3)
axes[1].set_yticks(yy, lab, fontsize=7.5)
axes[1].axvline(0.5, color=INK2, lw=1, ls="--")
axes[1].set_xlim(-0.2, 1)
axes[1].set_title(f"Game WR break traits: split-half reliability (odd/even games)\n"
                  f"SOE split-half: all routes {rel_soe['soe_all_WR']['reliability']:.2f}, "
                  f"non-GO {rel_soe['soe_nogo_WR']['reliability']:.2f}, HITCH/OUT/IN {rel_soe['soe_hoi_WR']['reliability']:.2f}", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "E002_reliability.png", dpi=110, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------- 8. results JSON
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


n_exploratory = len(exploratory)
out = {
    "experiment_id": "E002", "prereg": "experiments/results/E002_prereg.json",
    "detection": detection,
    "reliability": {"combine": rel_combine.to_dicts(), "game": rel_game.to_dicts(), "soe": rel_soe, "eb_shrinkage": eb_info},
    "primary": primary, "exploratory": exploratory, "n_exploratory_tests": n_exploratory, "diagnostics": diagnostics,
    "sample": {"WR_with_c_turn": int(WR["c_turn_retention"].is_finite().sum()),
               "WR_with_g_turn": int(WR["g_turn_retention"].is_finite().sum()),
               "WR_with_soe_hoi": int(WR["soe_hoi"].is_finite().sum()),
               "WR_with_c_stop": int(WR["c_stop_peak_decel"].is_finite().sum()),
               "WR_with_g_stop": int(WR["g_stop_peak_decel"].is_finite().sum())},
}
(RES / "E002.json").write_text(json.dumps(clean(out), indent=2))
print("wrote", RES / "E002.json", "| exploratory tests:", n_exploratory)
