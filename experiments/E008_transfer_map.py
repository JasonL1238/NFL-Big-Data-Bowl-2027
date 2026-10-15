"""E008 - Drill reliability + transfer map (PLAN.md section 6; a core section of the writeup).

No new feature-target links. Every link preregistered in round 1 (E002-E006 primaries, 14 tests) plus the E001
sanity link (WR/DB/DL/OL) is re-estimated under protocol rev 2 (PLAN.md section 4), so the whole round is
reported on one footing:
  1. rebuild each link's player-level feature and target exactly as in round 1 (probe parquets + probe module
     functions) and check n and raw r against the round-1 results JSONs;
  2. linkage.evaluate with the preregistered expected sign (one-sided Freedman-Lane), BH across the 14
     round-1 primaries, seeds 1-2 when fl_p < 0.10, leave-3-out influence, without-2025 reruns, exposure;
  3. reliability of each feature and target (round-1 definitions, recomputed with the current
     bdb27.reliability; E001: reliability of a max over attempts, target split-half on the link samples) and a
     reliability-corrected (disattenuated) partial-r bound. The primary correction uses beyond-B2 reliabilities
     (the parts of x and y the partial correlation uses); the as-used version is reported as an under-correction;
  4. drill reliability profile: every Combine signature the probes built, its reliability (repeat or
     cross-drill ICC; drill-, session- and year-centered) and the share of its variance between Combine
     sessions / draft years (segments.combine_sessions);
  5. figures: transfer-map forest plot and drill reliability chart.
Revision after the skeptic review (same tests, same definitions): beyond-B2 correction primary, per-link vs joint
(Bonferroni) bounds, one-sided MDE, two-sided p for the links whose direction is not from a prereg, multiplicity
across all round-1 tests, E006-P1 fragility (top-3 share, without-2025), literal target labels, like-for-like
drill-profile counts.

Run: .venv/bin/python experiments/E008_transfer_map.py                  (everything: ~25 min, n_jobs=3, peak ~1.8 GB)
     .venv/bin/python experiments/E008_transfer_map.py --quick          (small permutation counts, scratch output)
     .venv/bin/python experiments/E008_transfer_map.py --figures-only   (redraw figures from the saved parquets)
Outputs: experiments/results/E008.json, experiments/figures/E008_*.png, data/processed/e008_*.parquet.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm

from bdb27 import features, io, linkage, reliability, segments, speed
from bdb27.probes import a_route_breaks as A
from bdb27.probes import b_dl_burst as B
from bdb27.probes import c_ol_lateral as C
from bdb27.probes import d_db_transition as D
from bdb27.probes import e_speed_profile as E

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "experiments" / "results", ROOT / "experiments" / "figures"
P = io.PROCESSED
QUICK = "--quick" in sys.argv
FIGURES_ONLY = "--figures-only" in sys.argv
SCRATCH = Path("/private/tmp/claude-501/-Users-jasonli-Documents-GitHub-NFL-Big-Data-Bowl-2027/"
               "ca6dc0b9-7e4f-45b4-b4e4-1468819c8969/scratchpad/round2/E008")
OUT_RES, OUT_FIG, OUT_DATA = (SCRATCH, SCRATCH, SCRATCH) if QUICK else (RES, FIG, P)
N_JOBS = 3
EVAL = dict(n_repeats=50, n_perm=2000, n_perm_fl=10000) if not QUICK else dict(n_repeats=5, n_perm=50, n_perm_fl=500)
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


# =====================================================================================================
# 1. Links: player-level feature and target exactly as in round 1
# =====================================================================================================

def _r1(exp: str) -> dict:
    return json.loads((RES / f"{exp}.json").read_text())


def _sign(s) -> int:
    return {"+": 1, "-": -1, 1: 1, -1: -1}[s]


def e002_table() -> pl.DataFrame:
    return (linkage.player_table().join(pl.read_parquet(P / "probe_a_combine_signatures.parquet"), on="nfl_id", how="left")
            .join(pl.read_parquet(P / "probe_a_game_traits.parquet"), on="nfl_id", how="left"))


def e003_table() -> pl.DataFrame:
    dl = linkage.player_table().filter(pl.col("position") == "DL")
    return (dl.join(pl.read_parquet(P / "probe_b_combine.parquet"), on="nfl_id", how="left")
            .join(pl.read_parquet(P / "probe_b_game_traits.parquet"), on="nfl_id", how="inner"))


def e004_table() -> pl.DataFrame:
    return (linkage.player_table().filter(pl.col("position") == "OL")
            .join(pl.read_parquet(P / "probe_c_combine.parquet"), on="nfl_id", how="left")
            .join(pl.read_parquet(P / "probe_c_game.parquet"), on="nfl_id", how="left"))


def e005_parts() -> dict:
    """Probe D event tables (saved by E005) -> player signatures/traits with the probe's own functions
    (D.combine_signatures; the two traits of D.game_traits, whose third input only counts coverage plays)."""
    ce = pl.read_parquet(P / "probe_d_combine_events.parquet")
    cb = pl.read_parquet(P / "probe_d_combine_backpedal.parquet")
    ge = pl.read_parquet(P / "probe_d_game_events.parquet")
    gb = pl.read_parquet(P / "probe_d_game_backpedal_plays.parquet")
    cs = D.combine_signatures(ce, cb)
    bp = D.eb_shrink(D.game_backpedal_events(gb), min_n=D.MIN_BACKPEDAL_PLAYS).select(
        "nfl_id", gm_backpedal_speed="shrunk", gm_backpedal_n="n")
    dv = D.eb_shrink(D.game_ballair_events(ge, "dv_10"), min_n=D.MIN_BALLAIR_TRANSITIONS).select(
        "nfl_id", gm_ballair_dv="shrunk", gm_ballair_n="n")
    return dict(ce=ce, cb=cb, ge=ge, gb=gb, cs=cs, bp=bp, dv=dv)


def e005_table(parts: dict) -> pl.DataFrame:
    return (linkage.player_table().filter(pl.col("position") == "DB").join(parts["cs"], on="nfl_id", how="left")
            .join(parts["bp"], on="nfl_id", how="left").join(parts["dv"], on="nfl_id", how="left"))


def e006_table() -> pl.DataFrame:
    pt = linkage.player_table()
    yac = E.yacoe_outcome(pt.select("nfl_id", "position"))
    return (pt.join(pl.read_parquet(P / "probe_e_combine_profile.parquet"), on="nfl_id", how="left")
            .join(pl.read_parquet(P / "probe_e_game_profile.parquet"), on="nfl_id", how="left")
            .join(yac, on="nfl_id", how="left"))


def e001_table() -> pl.DataFrame:
    return (linkage.player_table().join(pl.read_parquet(P / "combine_features.parquet"), on="nfl_id")
            .join(pl.read_parquet(P / "game_traits.parquet"), on="nfl_id"))


def round1_reference() -> dict:
    """Round-1 n and raw r (seed 0) per link, read from the round-1 results JSONs."""
    ref = {}
    r = _r1("E002")["primary"]
    for k in ("P1", "P2", "P3"):
        ref[f"E002-{k}"] = (r[k]["seeds"]["0"]["n"], r[k]["seeds"]["0"]["raw_r"])
    for t in _r1("E003")["primary"]:
        ref[f"E003-{t['id']}"] = (t["seeds"][0]["n"], t["seeds"][0]["raw_r"])
    for i, t in enumerate(_r1("E004")["primary"]):
        ref[f"E004-P{i + 1}"] = (t["seeds"]["0"]["n"], t["seeds"]["0"]["raw_r"])
    r = _r1("E005")["primary"]
    for k in ("P1", "P2"):
        ref[f"E005-{k}"] = (r[k]["seeds"]["0"]["n"], r[k]["seeds"]["0"]["raw_r"])
    r = _r1("E006")["primary"]
    for k in ("P1", "P2", "P3"):
        ref[f"E006-{k}"] = (r[k]["seeds"]["0"]["n"], r[k]["seeds"]["0"]["raw_r"])
    r = _r1("E001")
    for p in ("WR", "DB", "DL", "OL"):
        ref[f"E001-{p}"] = (r[p]["n"], r[p]["raw_r"])
    return ref


def links() -> tuple[list[dict], dict]:
    """Every link: id, experiment, test, population, feature, target, expected sign (and where it comes from), player
    table, exposure column, construct labels for the map. Labels name what each game target measures literally;
    dagger = the round-1 review found the target narrower than the scouting concept (see CAVEATS)."""
    pre = {e: json.loads((RES / f"{e}_prereg.json").read_text()) for e in ("E002", "E003", "E004", "E005", "E006")}
    t2, t3, t4, t6, t1 = e002_table(), e003_table(), e004_table(), e006_table(), e001_table()
    d5 = e005_parts()
    t5 = e005_table(d5)
    wr = t2.filter(pl.col("position") == "WR")
    L = []
    PREREG = "preregistered (expected_sign field)"

    def add(lid, exp, test, pops, table, feature, target, sign, exposure, construct, kind, label, outcome,
            dagger=False, sign_source=PREREG):
        L.append(dict(id=lid, experiment=exp, test=test, positions=pops, df=table.filter(pl.col("position").is_in(pops)),
                      feature=feature, target=target, expected_sign=sign, exposure=exposure, construct=construct,
                      kind=kind, label=label, outcome_target=outcome, primary14=exp != "E001", target_dagger=dagger,
                      sign_source=sign_source,
                      feature_source="40-yd dash" if feature.startswith("c40_") or feature == "combine_top_speed"
                      else "position drill"))

    s = {t["id"]: _sign(t["expected_sign"]) for t in pre["E002"]["primary_tests"]}
    add("E002-P1", "E002", "P1", ["WR"], wr, "c_turn_retention", "g_turn_retention", s["P1"], "g_turn_retention_n",
        "Route breaks", "drill -> same movement", "Route-drill speed kept → OUT/IN retention (route style)", False,
        dagger=True)
    add("E002-P2", "E002", "P2", ["WR"], wr, "c_turn_retention", "soe_hoi", s["P2"], "soe_hoi_n",
        "Route breaks", "drill -> outcome", "Route-drill speed kept → separation at throw over exp.", True, dagger=True)
    add("E002-P3", "E002", "P3", ["WR"], wr, "c_stop_peak_decel", "g_stop_peak_decel", s["P3"], "g_stop_peak_decel_n",
        "Route breaks", "drill -> same movement", "Stop-route drill braking → HITCH braking", False)
    s = {t["id"]: _sign(t["expected_sign"]) for t in pre["E003"]["primary_tests"]}
    add("E003-P1", "E003", "P1", ["DL"], t3, "c40_d10", "g_d10_rush", s["P1"], "n_rush",
        "First-step burst", "40 -> same movement", "40 first-second burst → first-second rush distance", False,
        dagger=True)
    add("E003-P2", "E003", "P2", ["DL"], t3, "c40_d10", "g_getoff_rush", s["P2"], "n_rush",
        "First-step burst", "40 -> NGS trait", "40 first-second burst → NGS get-off (incl. snap reaction)", False)
    add("E003-P3", "E003", "P3", ["DL"], t3, "c40_d10", "g_quick_pressure_oe", s["P3"], "n_rush",
        "First-step burst", "40 -> outcome", "40 first-second burst → quick pressures over exp.", True)
    s = {f"P{i + 1}": _sign(t["expected_sign"]) for i, t in enumerate(pre["E004"]["primary_tests"])}
    add("E004-P1", "E004", "P1", ["OL"], t4, "cmb_rev_accel_mirror", "g_set_accel", s["P1"], "set_plays",
        "Lateral reversals", "drill -> same movement", "Mirror reversal accel. → set accel. 0-0.6 s after snap", False,
        dagger=True)
    add("E004-P2", "E004", "P2", ["OL"], t4, "cmb_rev_accel_mirror", "ppp_oe_eb", s["P2"], "pb_plays",
        "Lateral reversals", "drill -> outcome", "Mirror reversal accel. → pressure prob. allowed over exp.", True)
    add("E004-P3", "E004", "P3", ["OL"], t4, "cmb_rev_accel_mirror", "pr_oe_eb", s["P3"], "pb_plays",
        "Lateral reversals", "drill -> outcome", "Mirror reversal accel. → pressures allowed over exp.", True)
    s = {t["id"]: _sign(t["expected_sign"]) for t in pre["E005"]["primary_tests"]}
    add("E005-P1", "E005", "P1", ["DB"], t5, "cmb_backpedal_speed", "gm_backpedal_speed", s["P1"], "gm_backpedal_n",
        "Backpedal / transition", "drill -> same movement", "Drill backpedal speed → pre-throw peak retreat speed", False,
        dagger=True)
    add("E005-P2", "E005", "P2", ["DB"], t5, "cmb_transition_dv", "gm_ballair_dv", s["P2"], "gm_ballair_n",
        "Backpedal / transition", "drill -> same movement", "Drill plant-and-drive dv → post-throw redirection dv", False,
        dagger=True)
    # E006: the prereg lists no sign field; '+' is inferred from its hypothesis and rationale text.
    e6 = "inferred from the E006 hypothesis text (the prereg has no expected_sign field)"
    add("E006-P1", "E006", "P1", ["WR"], t6, "c40_vmax", "game_S0", 1, "game_reg_plays",
        "Sprint speed", "40 -> same movement", "Fitted 40 vmax → extrapolated game top-speed capacity S0", False,
        sign_source=e6)
    add("E006-P2", "E006", "P2", ["WR", "TE", "DB", "DL"], t6, "c40_a0", "game_A0", 1, "game_reg_plays",
        "Sprint acceleration", "40 -> same movement", "Fitted 40 acceleration a0 → game max acceleration A0", False,
        sign_source=e6)
    add("E006-P3", "E006", "P3", ["WR", "TE"], t6, "c40_vmax", "yacoe_eb", 1, "yacoe_catches",
        "Sprint speed", "40 -> outcome", "Fitted 40 vmax → YAC over expected (WR + TE)", True, sign_source=e6)
    for p in ("WR", "DB", "DL", "OL"):
        add(f"E001-{p}", "E001", p, [p], t1, "combine_top_speed", "game_top_speed", 1, "game_plays",
            "Sprint speed", "40 -> same movement", "40 peak speed → game top speed (q95 of play max)", False,
            sign_source="not preregistered: '+' from the E001 hypothesis (raw link); its beyond-B2 expectation was "
                        "~0, and the one-sided direction was set after the round-1 results were seen")
    return L, d5


def check_reproduction(L: list[dict]) -> dict:
    """n and raw r of every link vs round 1 (must match before anything else runs)."""
    ref = round1_reference()
    out = {}
    for lk in L:
        d = lk["df"].filter(pl.col(lk["feature"]).is_finite() & pl.col(lk["target"]).is_finite())
        n, r = d.height, float(np.corrcoef(d[lk["feature"]], d[lk["target"]])[0, 1])
        n1, r1 = ref[lk["id"]]
        ok = n == n1 and abs(r - r1) < 5e-4  # E003 stored raw r rounded to 3 decimals
        out[lk["id"]] = {"n": n, "raw_r": r, "round1_n": n1, "round1_raw_r": r1, "match": bool(ok)}
        log(f"reproduce {lk['id']}: n {n} (r1 {n1}), raw r {r:+.4f} (r1 {r1:+.4f}) {'OK' if ok else 'MISMATCH'}")
    return out


# =====================================================================================================
# 2. Drill reliability profile: every Combine signature the probes built
# =====================================================================================================
# Naming variants of one drill share sessions (as in probes A and C).
DRILL_MERGE = {"PASS_PRO-MIRROR_DRILL": "PASS_PRO_MIRROR_DRILL", "SPEED_OUT_LEFT": "SPEED_OUT_ROUTE_LEFT"}
A_FEATS = ["entry_speed", "min_speed", "retention", "speed_drop", "peak_decel", "mean_decel", "decel_time",
           "peak_reaccel", "turn_radius", "peak_alat"]
B_BURST4 = ["FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION", "PASS_RUSH_DRILL", "RUN_AND_CLUB_DRILL", "RUN_THE_HOOP_DRILL"]
D_FEATS = ["entry_speed", "peak_decel", "brake_05", "min_speed", "peak_reaccel", "drive_05", "drive_10", "dv_10",
           "decel_time", "retention", "turn_radius", "max_turn_rate"]


def session_table() -> pl.DataFrame:
    """Combine session of every attempt (segments.combine_sessions: > 10 min since the previous attempt start of
    the same drill and draft year opens a new session). Keys: segments.ATTEMPT_KEYS."""
    att = (io.scan_combine_tracking().filter(pl.col("entity_type") == "PLAYER")
           .group_by(*segments.ATTEMPT_KEYS, "draft_year").agg(time=pl.col("time").min()).collect()
           .sort(segments.ATTEMPT_KEYS)
           .with_columns(traj=pl.int_range(pl.len()), drill=pl.col("drill_name").replace(DRILL_MERGE)))
    return segments.combine_sessions(att, drill="drill").select(*segments.ATTEMPT_KEYS, "session")


def rel_profile(tab: pl.DataFrame, unit: str, zscore: bool = True, share_tab: pl.DataFrame | None = None,
                share_unit: str | None = None) -> dict:
    """Reliability and session/year variance shares of one Combine signature.
    tab: one row per measurement (nfl_id, <unit>, session, draft_year, v). Rows sharing a player are repeats
    of the trait: repeated attempts (unit = the drill, or position for the 40), different drills/cuts
    (cross-drill; unit = drill or cut). With `zscore`, values are first z-scored within unit, so drills on
    different scales count equally (off where round 1 only centered, or values are already z), then
    reliability.cross_drill_icc centers them
      rel_unit:    within unit (the round-1 definition),
      rel_session: within unit x Combine session, rel_year: within unit x draft year.
    share_session / share_year: ICC(1) of player-level values (player x unit means, or `share_tab` with its own
    `share_unit`), z-scored within unit, grouped by unit x session / unit x draft year = the variance component between sessions
    (years) as a share of the within-unit variance; negative estimates are reported as 0 (raw value kept).
    A session share also contains real ability differences between groups that were tested together."""
    t = tab.filter(pl.col("v").is_finite())
    if zscore:
        t = t.with_columns(v=(pl.col("v") - pl.col("v").mean().over(unit)) / pl.col("v").std().over(unit))
        t = t.filter(pl.col("v").is_finite())
    out = {"n_players": t["nfl_id"].n_unique(), "n_rows": t.height, "n_units": t[unit].n_unique()}
    for name, cb in (("unit", [unit]), ("session", [unit, "session"]), ("year", [unit, "draft_year"])):
        r = reliability.cross_drill_icc(t, "v", drill=unit, center_by=cb)
        out[f"icc_{name}"], out[f"rel_{name}"], out[f"n_dropped_{name}"] = r["icc"], r["reliability"], r["n_dropped"]
    s = share_tab if share_tab is not None else (
        t.group_by("nfl_id", unit).agg(pl.col("v").mean(), pl.col("session").first(), pl.col("draft_year").first()))
    unit = share_unit or unit
    s = s.filter(pl.col("v").is_finite()).with_columns(
        z=(pl.col("v") - pl.col("v").mean().over(unit)) / pl.col("v").std().over(unit))
    for name, col in (("session", "session"), ("year", "draft_year")):
        g = s.select(pl.concat_str([pl.col(unit).cast(pl.Utf8), pl.col(col).cast(pl.Utf8)], separator="|")).to_series()
        r = reliability.icc1(s["z"].to_numpy(), g.to_numpy())
        out[f"share_{name}_raw"] = r["icc"]
        out[f"share_{name}"] = max(r["icc"], 0.0) if np.isfinite(r["icc"]) else np.nan
        out[f"n_{name}_groups"] = int(g.n_unique())
    return out


def _first_attempt_means(e: pl.DataFrame, value: str, unit: str) -> pl.DataFrame:
    """Player x unit attempt-means of `value` with the session / draft year of the player's first attempt."""
    return (e.filter(pl.col(value).is_finite()).sort("nfl_id", unit, "attempt")
            .group_by("nfl_id", unit).agg(v=pl.col(value).mean(), session=pl.col("session").first(),
                                          draft_year=pl.col("draft_year").first()))


def profile_specs(sess: pl.DataFrame, d5: dict) -> tuple[list[dict], dict]:
    """All Combine signatures built by probes A-E (+ E001), as rel_profile inputs. Returns (specs, extras) where
    extras holds attempt tables reused for the link reliabilities."""
    K = segments.ATTEMPT_KEYS
    pos = io.load("combine_results").select("nfl_id", position="combine_position")
    specs = []

    def spec(probe, position, metric, scope, kind, unit, tab, zscore=True, share_tab=None, share_unit=None,
             used_by=(), round1=None, **extra):
        specs.append(dict(probe=probe, position=position, metric=metric, scope=scope, kind=kind, unit=unit, tab=tab,
                          zscore=zscore, share_tab=share_tab, share_unit=share_unit, used_by=list(used_by),
                          round1=round1, extra=extra))

    # ---- Probe A (E002): WR/TE route-drill breaks; composite = drill-z mean over the family's drills
    r1 = {(r["position"], r["family"], r["feature"]): r for r in _r1("E002")["reliability"]["combine"]}
    cev = (A.combine_events(A.combine_frames(list(A.DRILLS) + A.QC_EXCLUDED_DRILLS))
           .join(pos, on="nfl_id", how="left").join(sess, on=K, how="left"))
    for p in ("WR", "TE"):
        for fam in ("stop", "turn"):
            for f in A_FEATS:
                e = cev.filter((pl.col("position") == p) & (pl.col("family") == fam) & pl.col("found")
                               & pl.col(f).is_finite())
                if e.height == 0:
                    continue
                c = e.with_columns(v=pl.col(f) - pl.col(f).mean().over("drill"),
                                   g=pl.concat_str([pl.col("nfl_id").cast(pl.String), pl.col("drill")]))
                rt = reliability.icc1(c["v"].to_numpy(), c["g"].to_numpy())
                used = {("WR", "turn", "retention"): ["E002-P1", "E002-P2"], ("WR", "stop", "peak_decel"): ["E002-P3"]}
                spec("A route breaks (E002)", p, f"{fam} {f}", f"{fam}-route drills, composite", "cross-drill", "drill",
                     _first_attempt_means(e, f, "drill"), used_by=used.get((p, fam, f), ()),
                     round1=(r1.get((p, fam, f)) or {}).get("composite_reliability"),
                     retest_icc=rt["icc"], retest_n_players=rt["n_repeated"])

    # ---- Probe B (E003): DL burst, onset-aligned (40 attempts, PASS_RUSH attempts, cross-drill composites)
    r1 = _r1("E003")["combine_reliability"]
    dl_ids = linkage.player_table().filter(pl.col("position") == "DL")["nfl_id"].to_list()
    catt = B.combine_attempt_features(B.combine_frames(dl_ids)).join(sess, on=K, how="left")
    for f in B.BURST_FEATS + ["t5"]:
        for drill in (B.FORTY, "PASS_RUSH_DRILL"):
            if f == "t5" and drill != B.FORTY:
                continue
            a = catt.filter((pl.col("drill_name") == drill) & pl.col(f).is_not_null()).select(
                "nfl_id", "drill_name", "session", "draft_year", v=pl.col(f))
            spec("B DL burst (E003)", "DL", f"{f}", f"{'40-yd dash' if drill == B.FORTY else 'pass-rush drill'} attempts",
                 "repeat attempts", "drill_name", a, used_by=["E003-P1", "E003-P2", "E003-P3"] if (drill, f) == (B.FORTY, "d10") else (),
                 round1=(r1["attempt_icc"].get(f"{drill}:{f}") or {}).get("reliability"))
        if f == "t5":
            continue
        for name, ds in [("burst4", B_BURST4)] + ([("burst4+40", B_BURST4 + [B.FORTY]), ("all6+40", B.DL_DRILLS + [B.FORTY])]
                                                 if f == "d10" else []):
            e = catt.filter(pl.col("drill_name").is_in(ds))
            spec("B DL burst (E003)", "DL", f"{f}", f"cross-drill {name}",
                 "cross-drill incl. 40" if "+40" in name else "cross-drill", "drill_name",
                 _first_attempt_means(e, f, "drill_name"), round1=(r1["cross_drill"].get(f"{name}:{f}") or {}).get("reliability"))

    # ---- Probe C (E004): OL mirror / wave lateral reversals; PASS_RUSH_DROP kick
    r1 = json.loads((RES / "E004_reliability.json").read_text())
    fr = C.drill_frames(C.MIRROR_NAMES + [C.WAVE_NAME])
    rev = C.combine_reversals(fr)
    cmb = C.combine_reversal_features(rev)
    psess = rev.group_by("nfl_id", "drill").agg(pl.col("session").first(), pl.col("draft_year").first())
    for drill in ("MIRROR", "WAVE"):
        for m in C.REV_METRICS:
            r = rev.filter(pl.col("drill") == drill).select("nfl_id", "cut", "session", "draft_year", v=pl.col(m))
            share = (rev.filter(pl.col("drill") == drill).group_by("nfl_id", "drill")
                     .agg(v=pl.col(m).mean(), session=pl.col("session").first(), draft_year=pl.col("draft_year").first()))
            spec("C OL lateral (E004)", "OL", f"{drill.lower()} {m}", "reversals within one attempt", "within-attempt", "cut",
                 r, zscore=False, share_tab=share, share_unit="drill", used_by=["E004-P1", "E004-P2", "E004-P3"] if (drill, m) == ("MIRROR", "rev_accel") else (),
                 round1=r1["combine_within_attempt"][drill][m]["reliability"])
    for m in C.REV_METRICS:
        w = (cmb.select("nfl_id", MIRROR=f"cmb_{m}_mirror", WAVE=f"cmb_{m}_wave")
             .unpivot(index="nfl_id", variable_name="drill", value_name="v").join(psess, on=["nfl_id", "drill"], how="left"))
        spec("C OL lateral (E004)", "OL", f"{m}", "mirror + wave cross-drill (session-cut-centered)", "cross-drill",
             "drill", w, round1=(r1["combine_cross_drill"].get(m) or {}).get("composite_alpha"))
    drop = C.drop_kick_metrics(C.drill_frames([C.DROP_NAME])).join(
        linkage.player_table().select("nfl_id", "draft_year"), on="nfl_id", how="left").with_columns(
        unit=pl.col("attempt").cast(pl.Utf8))
    for m in ("kick_speed", "kick_accel", "kick_brake", "sprint_max"):
        spec("C OL lateral (E004)", "OL", f"drop {m}", "pass-rush-drop attempts (opposite sides)", "repeat attempts",
             "unit", drop.select("nfl_id", "unit", "session", "draft_year", v=pl.col(m)), zscore=False,
             round1=r1["drop_across_attempts"][m]["reliability"], round1_variant="rel_session")

    # ---- Probe D (E005): DB drill transitions and backpedal (z within drill-cut as in round 1)
    r1 = {(r["feature"], r["adjust"], r["scope"]): r for r in _r1("E005")["reliability"]["combine"]}
    ce = d5["ce"].join(sess, on=K, how="left")
    cb = d5["cb"].join(sess, on=K, how="left").join(
        linkage.player_table().select("nfl_id", "draft_year"), on="nfl_id", how="left")
    rv = ce.filter(pl.col("drill_name").is_in(D.REVERSAL_DRILLS))
    for f in D_FEATS:
        for adj_name, adj in (("none", None), ("entry+turn", ["entry_speed", "aturn"])):
            if f == "entry_speed" and adj:
                continue
            dz = D.adjusted_z(rv, f, adj)
            dm = _first_attempt_means(dz, "z", "drill_name")
            used = ["E005-P2"] if (f, adj_name) == ("dv_10", "entry+turn") else []
            spec("D DB transition (E005)", "DB", f"{f}" + (" | entry+turn adj." if adj else ""),
                 "reversal drills (4), drill means", "cross-drill", "drill_name", dm, zscore=False, used_by=used,
                 round1=(r1.get((f, adj_name, "reversal drill means (4 drills)")) or {}).get("reliability"))
    b45 = ce.filter(pl.col("drill_name") == "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION")
    for f in D_FEATS:
        spec("D DB transition (E005)", "DB", f"BP45 {f}", "45-degree reaction U-turn, 2 attempts", "repeat attempts",
             "drill_name", b45.select("nfl_id", "drill_name", "session", "draft_year", v=pl.col(f)),
             round1=(r1.get((f, "none", "BP45 reaction U-turn, 2 attempts (icc1)")) or {}).get("reliability"))
    bpd = cb.with_columns(dc=pl.col("drill_name"))
    for f in ("bp_peak", "bp_v05", "bp_v10"):
        dz = D.adjusted_z(bpd.filter(pl.col("drill_name").is_in(D.BACKPEDAL_DRILLS)), f, None)
        spec("D DB transition (E005)", "DB", f"backpedal {f}", "backpedal drills (5), drill means", "cross-drill",
             "drill_name", _first_attempt_means(dz, "z", "drill_name"), zscore=False,
             used_by=["E005-P1"] if f == "bp_peak" else (),
             round1=(r1.get((f, "none", "backpedal drill means (5 drills)")) or {}).get("reliability"))

    # ---- 40-yd dash (bdb27.speed; probe E / E006 / E007 metrics), all positions, QC-passing attempts
    att40 = speed.forty_attempts().join(sess, on=["event_id", "nfl_id", "attempt"], how="left")
    q = att40.filter(pl.col("qc_ok"))
    used40 = {"vmax": ["E006-P1", "E006-P3"], "a0": ["E006-P2"]}
    for m in speed.METRICS:
        spec("40-yd dash (bdb27.speed)", "all", m, "40-yd attempts, centered by position", "repeat attempts", "position",
             q.select("nfl_id", "position", "session", "draft_year", v=pl.col(m)), zscore=False,
             used_by=used40.get(m, ()))
    # E001 feature: device `s` peak of each 40 attempt (all attempts, no QC); player value = max over attempts.
    smax = (io.scan_combine_tracking().filter((pl.col("entity_type") == "PLAYER") & (pl.col("drill_type") == "FORTY_YARD_DASH"))
            .group_by(*K, "draft_year").agg(v=pl.col("s").max()).collect()
            .join(sess, on=K, how="left").join(pos, on="nfl_id", how="left")
            .sort("nfl_id", "attempt", "event_id"))  # row order sets each player's first-attempt session (shares)
    spec("40-yd dash (bdb27.speed)", "all", "device peak s (E001)", "40-yd attempts, centered by position",
         "repeat attempts", "position", smax.select("nfl_id", "position", "session", "draft_year", "v"), zscore=False,
         used_by=[f"E001-{p}" for p in ("WR", "DB", "DL", "OL")])
    return specs, {"att40": att40, "smax40": smax}


def drill_profile(specs: list[dict]) -> pl.DataFrame:
    rows = []
    for s in specs:
        r = rel_profile(s["tab"], s["unit"], s["zscore"], s["share_tab"], s["share_unit"])
        rows.append({k: s[k] for k in ("probe", "position", "metric", "scope", "kind")}
                    | {"used_by": ",".join(s["used_by"]), "round1_reliability": s["round1"]} | r
                    | {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in s["extra"].items()})
    df = pl.DataFrame(rows, infer_schema_length=None)
    # the round-1 definition for the drop drill centered by session x attempt
    var = pl.col("round1_variant") if "round1_variant" in df.columns else pl.lit(None)
    return df.with_columns(rel_round1_def=pl.when(var == "rel_session").then(pl.col("rel_session"))
                           .otherwise(pl.col("rel_unit"))).drop("round1_variant", strict=False)


# =====================================================================================================
# 3. Link-level reliability (feature: round-1 definition recomputed, E001 as a max over attempts;
#    target: round-1 split-half, E001 recomputed on the link samples with the exact target definition)
# =====================================================================================================

def _icc_by_position(tab: pl.DataFrame, center_year: bool) -> dict:
    """icc1 reliability of per-attempt values within each position (optionally centered by draft year)."""
    out = {}
    for (p,), g in tab.filter(pl.col("v").is_finite()).group_by(["position"]):
        v = g["v"] - (g.select(pl.col("v").mean().over("draft_year")).to_series() if center_year else g["v"].mean())
        out[p] = reliability.icc1(v.to_numpy(), g["nfl_id"].to_numpy())["reliability"]
    return out


# Mean and variance of the max of k iid N(0, 1) draws (order statistics; k = attempts per player).
MAX_MU = {1: 0.0, 2: 0.5642, 3: 0.8463, 4: 1.0294}
MAX_VAR = {1: 1.0, 2: 0.6817, 3: 0.5595, 4: 0.4917}


def _max_reliability_by_position(tab: pl.DataFrame, center_year: bool) -> dict:
    """Reliability of each player's MAX over his attempts (the E001 feature), from the attempt ICC under normal errors:
    true variance = icc, error variance = (1 - icc) x (mean over players of Var(max of k) + variance over players of
    E(max of k)), with k the player's attempt count. Slightly below the mean-of-attempts reliability."""
    out = {}
    for (p,), g in tab.filter(pl.col("v").is_finite()).group_by(["position"]):
        v = g["v"] - (g.select(pl.col("v").mean().over("draft_year")).to_series() if center_year else g["v"].mean())
        icc = max(reliability.icc1(v.to_numpy(), g["nfl_id"].to_numpy())["icc"], 0.0)
        k = g.group_by("nfl_id").len()["len"].to_numpy()
        err = (1 - icc) * (np.mean([MAX_VAR[i] for i in k]) + np.var([MAX_MU[i] for i in k]))
        out[p] = float(icc / (icc + err)) if icc > 0 else 0.0
    return out


def e001_target_split_half(L: list[dict]) -> dict:
    """Split-half (odd/even REG games, Spearman-Brown) of the exact E001 target on each E001 link sample: q95 of play
    max speed, plays with max > 12 yd/s dropped (features.game_top_speed). A half needs >= 25 plays (half the
    target's 50-play minimum); >= 10 is a sensitivity. Round 1 took these values from E006's script (post-snap
    frame cache, no glitch filter, all players of the position)."""
    smp = {lk["id"]: _sample(lk)["nfl_id"].to_list() for lk in L if lk["experiment"] == "E001"}
    ids = sorted(set().union(*smp.values()))
    pm = (features.reg_season_tracking().filter(pl.col("nfl_id").is_in(ids))
          .group_by("game_id", "play_id", "nfl_id").agg(smax=pl.col("s").max())
          .filter(pl.col("smax") <= features.MAX_PLAUSIBLE_SPEED).collect())
    full = pm.group_by("nfl_id").agg(q=pl.col("smax").quantile(0.95)).join(
        pl.read_parquet(P / "game_traits.parquet").select("nfl_id", "game_top_speed"), on="nfl_id")
    assert full.height == len(ids) and float((full["q"] - full["game_top_speed"]).abs().max()) < 1e-9, \
        "rebuilt E001 target does not match game_traits.parquet"
    out = {}
    for lid, pids in smp.items():
        d = pm.filter(pl.col("nfl_id").is_in(pids))
        out[lid] = {f"min{m}_per_half": reliability.split_half_estimator(
            d, lambda g, m=m: float(g["smax"].quantile(0.95)) if g.height >= m else np.nan) for m in (25, 10)}
        log(f"E001 target split-half {lid}: {out[lid]}")
    return out


def target_reliability(e001_sh: dict) -> dict:
    """Split-half reliabilities of each target (odd/even games, Spearman-Brown); by position where pooled. Round-1
    values, except E001 (recomputed on the link samples with the exact target definition, >= 25 plays per half)."""
    e2, e3, e6 = _r1("E002")["reliability"], _r1("E003")["game_reliability"], _r1("E006")
    e4 = json.loads((RES / "E004_reliability.json").read_text())
    e5 = {(r["event_set"], r["feature"]): r for r in _r1("E005")["reliability"]["game"]}
    g2 = {(r["position"], r["family"], r["feature"]): r["split_half"] for r in e2["game"]}
    gr = e6["game_reliability_split_half"]
    return {
        "E002-P1": {"WR": g2[("WR", "turn", "retention")]},
        "E002-P2": {"WR": e2["soe"]["soe_hoi_WR"]["reliability"]},
        "E002-P3": {"WR": g2[("WR", "stop", "peak_decel")]},
        "E003-P1": {"DL": e3["g_d10_rush"]["split_half"]},
        "E003-P2": {"DL": e3["g_player_get_off_rush"]["split_half"]},
        "E003-P3": {"DL": e3["g_quick_oe_rush"]["split_half"]},
        "E004-P1": {"OL": e4["game_set"]["g_set_accel"]["reliability"]},
        "E004-P2": {"OL": e4["outcomes"]["ppp_oe"]["reliability"]},
        "E004-P3": {"OL": e4["outcomes"]["pr_oe"]["reliability"]},
        "E005-P1": {"DB": e5[("backpedal_plays", "bp_peak")]["rel_min10"]},
        "E005-P2": {"DB": e5[("from_backpedal_ball_in_air", "dv_10")]["rel_min5"]},
        "E006-P1": {"WR": gr["game_S0"]["WR"]["reliability"]},
        "E006-P2": {p: gr["game_A0"][p]["reliability"] for p in ("WR", "TE", "DB", "DL")},
        "E006-P3": {p: e6["yacoe_reliability"][p]["split_half_min5_per_half"]["reliability"] for p in ("WR", "TE")},
        **{f"E001-{p}": {p: e001_sh[f"E001-{p}"]["min25_per_half"]["reliability"]} for p in ("WR", "DB", "DL", "OL")},
    }


def target_reliability_round1_e001() -> dict:
    """Round-1 E001 target reliabilities (E006 script), kept for traceability."""
    gr = _r1("E006")["game_reliability_split_half"]["game_top_speed"]
    return {f"E001-{p}": gr[p]["reliability"] for p in ("WR", "DB", "DL", "OL")}


def feature_reliability(prof: pl.DataFrame, extras: dict) -> dict:
    """Per link and position: (as used = round-1 definition, recomputed; year-centered). Pooled links are
    weighted by the link sample's position counts later. E001: reliability of the max over attempts (the feature
    as used; round 1 reported the mean-of-attempts value)."""
    def row(probe, position, metric, scope):
        r = prof.filter((pl.col("probe") == probe) & (pl.col("position") == position) & (pl.col("metric") == metric)
                        & (pl.col("scope") == scope))
        assert r.height == 1, (probe, position, metric, scope, r.height)
        return r.row(0, named=True)

    out = {}
    a = row("A route breaks (E002)", "WR", "turn retention", "turn-route drills, composite")
    out["E002-P1"] = out["E002-P2"] = {"WR": (a["rel_round1_def"], a["rel_year"])}
    a = row("A route breaks (E002)", "WR", "stop peak_decel", "stop-route drills, composite")
    out["E002-P3"] = {"WR": (a["rel_round1_def"], a["rel_year"])}
    a = row("B DL burst (E003)", "DL", "d10", "40-yd dash attempts")
    out["E003-P1"] = out["E003-P2"] = out["E003-P3"] = {"DL": (a["rel_round1_def"], a["rel_year"])}
    a = row("C OL lateral (E004)", "OL", "mirror rev_accel", "reversals within one attempt")
    out["E004-P1"] = out["E004-P2"] = out["E004-P3"] = {"OL": (a["rel_round1_def"], a["rel_year"])}
    a = row("D DB transition (E005)", "DB", "backpedal bp_peak", "backpedal drills (5), drill means")
    out["E005-P1"] = {"DB": (a["rel_round1_def"], a["rel_year"])}
    a = row("D DB transition (E005)", "DB", "dv_10 | entry+turn adj.", "reversal drills (4), drill means")
    out["E005-P2"] = {"DB": (a["rel_round1_def"], a["rel_year"])}
    q = extras["att40"].filter(pl.col("qc_ok"))
    for lid, m, pops in (("E006-P1", "vmax", ["WR"]), ("E006-P2", "a0", ["WR", "TE", "DB", "DL"]), ("E006-P3", "vmax", ["WR", "TE"])):
        t = q.select("nfl_id", "position", "draft_year", v=pl.col(m))
        r0, ry = _icc_by_position(t, False), _icc_by_position(t, True)
        out[lid] = {p: (r0[p], ry[p]) for p in pops}
    t = extras["smax40"].select("nfl_id", "position", "draft_year", "v")
    r0, ry = _max_reliability_by_position(t, False), _max_reliability_by_position(t, True)
    for p in ("WR", "DB", "DL", "OL"):
        out[f"E001-{p}"] = {p: (r0[p], ry[p])}
    return out


# =====================================================================================================
# 4. Harness runs (protocol rev 2), influence, exposure
# =====================================================================================================
CTRL = linkage.COMBINE_CONTROLS + linkage.DRAFT_CONTROLS


def _sample(lk: dict) -> pl.DataFrame:
    """The rows evaluate() uses, in its order."""
    return lk["df"].filter(pl.col(lk["feature"]).is_finite() & pl.col(lk["target"]).is_finite()).sort("nfl_id")


def _eval(lk: dict, df: pl.DataFrame | None = None, seed: int = 0, **kw) -> dict:
    args = dict(EVAL, **kw)
    return linkage.evaluate(lk["df"] if df is None else df, lk["feature"], lk["target"],
                            expected_sign=lk["expected_sign"], seed=seed, n_jobs=N_JOBS, **args)


def _residuals(d: pl.DataFrame, cols: list[str]) -> tuple[np.ndarray, dict]:
    """Residual maker of the harness's B2 design (position + year dummies, imputed controls) and residuals."""
    R = linkage._residual_maker(linkage._imputed(linkage._design(d, CTRL)))
    return R, {c: R @ d[c].to_numpy().astype(float) for c in cols}


def influence(lk: dict, full: bool = False) -> dict:
    """Leave-3-out (go criterion 6): drop the 3 players contributing most to the partial correlation in the
    expected direction (sign * x_res * y_res), and, as a descriptive check on nulls, the 3 contributing most
    against it; re-run the seed-free Freedman-Lane inference (CV parts at minimal settings, not used). With `full`
    (links passing criterion 1), the supportive drop is also re-run at full settings for the CV criteria 2-3."""
    d = _sample(lk)
    _, res = _residuals(d, [lk["feature"], lk["target"]])
    xr, yr = res[lk["feature"]], res[lk["target"]]
    c = lk["expected_sign"] * xr * yr
    # each player's additive contribution to the (sign-adjusted) partial r: x_i y_i / sqrt(sum x^2 sum y^2); the share
    # carried by the top 3 is reported for links passing criterion 1 (`full`) only (meaningless near r = 0)
    contrib = c / np.sqrt((xr @ xr) * (yr @ yr))
    out = {"top3_supportive_share_of_partial_r": float(np.sort(contrib)[::-1][:3].sum() / contrib.sum())
           if full else None}
    for name, idx in (("drop3_supportive", np.argsort(-c)[:3]), ("drop3_contrary", np.argsort(c)[:3])):
        ids = d["nfl_id"].to_numpy()[idx].tolist()
        dd = d.filter(~pl.col("nfl_id").is_in(ids))
        r = _eval(lk, dd, n_repeats=2, n_perm=20)
        out[name] = {"dropped_nfl_id": ids, "dropped_names": d["display_name"].to_numpy()[idx].tolist(), "n": r["n"],
                     "partial_r": r["partial_r"], "partial_r_ci95": r["partial_r_ci95"], "fl_p": r["fl_p"],
                     "passes_c1": bool(r["checks"]["fl_p_lt_05"] and r["checks"]["sign_matches"])}
        if name == "drop3_supportive" and full:  # the CV criteria too, at full settings (links passing c1 only)
            rf = _eval(lk, dd)
            out[name]["full_harness"] = {"delta_r2_vs_B2": rf["delta_r2_vs_B2"], "loco": rf["loco"], "checks": rf["checks"]}
    return out


def exposure(lk: dict) -> dict:
    """corr(target, log exposure) and corr(feature, log exposure) in the link sample (position-centered when pooled)."""
    d = _sample(lk).filter(pl.col(lk["exposure"]) > 0)
    cen = lambda c: (pl.col(c) - pl.col(c).mean().over("position"))
    d = d.with_columns(pl.col(lk["exposure"]).cast(pl.Float64).log().alias("_log_exposure"))
    d = d.with_columns(_t=cen(lk["target"]), _f=cen(lk["feature"]), _e=cen("_log_exposure"))
    r = lambda a, b: float(np.corrcoef(d[a], d[b])[0, 1])
    return {"exposure_col": lk["exposure"], "n": d.height, "r_target_log_exposure": r("_t", "_e"),
            "r_feature_log_exposure": r("_f", "_e"), "median_exposure": float(d[lk["exposure"]].median())}


def run_links(L: list[dict]) -> dict:
    res = {}
    for lk in L:
        t = time.time()
        r = {"seed0": _eval(lk)}
        p0 = r["seed0"]["fl_p"]
        if p0 is not None and p0 < 0.10:
            r["seed1"], r["seed2"] = _eval(lk, seed=1), _eval(lk, seed=2)
        # two-sided Freedman-Lane p (same permutation stream; CV parts at minimal settings, not used): descriptive,
        # for the links whose direction is not from a prereg (E001, E006)
        r["two_sided"] = {k: v for k, v in linkage.evaluate(
            lk["df"], lk["feature"], lk["target"], expected_sign=None, seed=0, n_jobs=N_JOBS, n_repeats=2, n_perm=20,
            n_perm_fl=EVAL["n_perm_fl"]).items() if k in ("n", "partial_r", "fl_p", "fl_one_sided", "n_perm_fl")}
        r["without_2025"] = _eval(lk, lk["df"].filter(pl.col("draft_year") != 2025))
        r["influence"] = influence(lk, full=bool(r["seed0"]["checks"]["fl_p_lt_05"] and r["seed0"]["checks"]["sign_matches"]))
        r["exposure"] = exposure(lk)
        s0 = r["seed0"]
        log(f"{lk['id']:8s} n={s0['n']:3d} raw={s0['raw_r']:+.3f} partial={s0['partial_r']:+.3f} "
            f"[{s0['partial_r_ci95'][0]:+.2f},{s0['partial_r_ci95'][1]:+.2f}] mde={s0['partial_r_mde80']:.2f} "
            f"fl_p={s0['fl_p']:.4f} excess={s0['delta_r2_vs_B2']['excess']:+.4f}±{s0['delta_r2_vs_B2']['se']:.4f} "
            f"loco={s0['checks']['loco_excess_positive_classes']} | no2025 p={r['without_2025']['fl_p']} "
            f"| drop3 p={r['influence']['drop3_supportive']['fl_p']:.4f} ({time.time() - t:.0f}s)")
        res[lk["id"]] = r
    return res


# =====================================================================================================
# 5. Bounds: BH, reliability-corrected partial r, go criteria
# =====================================================================================================

def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def _weighted(rel_by_pos: dict, d: pl.DataFrame) -> float:
    """Link-sample-weighted mean of per-position reliabilities."""
    n = dict(d.group_by("position").len().iter_rows())
    return float(sum(rel_by_pos[p] * n[p] for p in n) / sum(n.values()))


def resid_reliability(d: pl.DataFrame, col: str, rel_by_pos: dict, R: np.ndarray) -> float:
    """Reliability of the part of `col` not explained by the B2 design (what the partial correlation uses),
    under classical error: player error variance s2_i = (1 - rel_pos) Var_pos(col); E||R e||^2 = sum s2_i R_ii;
    rel_B2 = 1 - E||R e||^2 / ||R col||^2. Lower than rel when B2 explains much of the true variance."""
    v = d[col].to_numpy().astype(float)
    pos = d["position"].to_numpy()
    s2 = np.zeros(len(v))
    for p in np.unique(pos):
        m = pos == p
        s2[m] = (1 - rel_by_pos[p]) * np.var(v[m], ddof=1)
    rv = R @ v
    return float(1 - np.sum(s2 * np.diag(R)) / (rv @ rv))


def corrected_ci(ci: list, rel_x: float, rel_y: float) -> list:
    """Disattenuated partial-r interval: ci / sqrt(rel_x rel_y), clipped to [-1, 1]; unbounded if rel <= 0.05."""
    if ci[0] is None or not (rel_x > 0.05 and rel_y > 0.05):
        return [-1.0, 1.0]
    k = np.sqrt(rel_x * rel_y)
    return [float(np.clip(ci[0] / k, -1, 1)), float(np.clip(ci[1] / k, -1, 1))]


# All tests in round 1 (experiments/log.md GATE 1): 14 primaries + 76 exploratory (E002 12, E003 25, E004 12,
# E005 7, E006 16 planned + 4 post hoc). The round-1 skeptics counted ~88.
N_ROUND1_TESTS = 90
# Joint (family-wise) one-sided ceilings: Bonferroni over the 18 links (a fixed family, not the post hoc nulls).
JOINT_Z = float(norm.ppf(1 - 0.05 / 18))
Z80 = float(norm.ppf(0.80))
C_NAMES = {"c2_excess_lb_gt_0": "CV excess", "c3_loco_2of3": "LOCO", "c4_bh_q_lt_05": "BH",
           "c5_reliability_ge_05": "reliability"}


def _verdict(crit: dict, seeds_ok: bool) -> str:
    """Short verdict for the map: what fails, in words."""
    if all(crit.values()):
        return "passes 1-6; see caveats"
    if not crit["c1_fl_p_and_sign"]:
        return "no detectable link" + ("" if crit["c5_reliability_ge_05"] else "; rel. < 0.5")
    fails = [v for k, v in C_NAMES.items() if not crit[k]]
    if not crit["c6_seeds_and_leave3out"]:
        fails.append("leave-3-out" if seeds_ok else "seeds")
    return "p < .05; fails " + " + ".join(fails)


def transfer_map(L: list[dict], res: dict, frel: dict, trel: dict) -> list[dict]:
    p14 = [res[lk["id"]]["seed0"]["fl_p"] for lk in L if lk["primary14"]]
    q14 = dict(zip([lk["id"] for lk in L if lk["primary14"]], bh(p14)))
    q18 = dict(zip([lk["id"] for lk in L], bh([res[lk["id"]]["seed0"]["fl_p"] for lk in L])))
    rows = []
    for lk in L:
        lid, r = lk["id"], res[lk["id"]]
        s0, sg = r["seed0"], lk["expected_sign"]
        d = _sample(lk)
        R, _ = _residuals(d, [])
        fx = {p: v[0] for p, v in frel[lid].items()}
        fx_year = {p: v[1] for p, v in frel[lid].items()}
        rel_x, rel_x_year, rel_y = _weighted(fx, d), _weighted(fx_year, d), _weighted(trel[lid], d)
        rx_b2 = resid_reliability(d, lk["feature"], fx, R)
        ry_b2 = resid_reliability(d, lk["target"], trel[lid], R)
        ci = s0["partial_r_ci95"]
        dis = corrected_ci(ci, rel_x, rel_y)        # as-used reliabilities (under-corrects a partial r)
        dis_b2 = corrected_ci(ci, rx_b2, ry_b2)     # beyond-B2 reliabilities (primary)
        # "ceiling" = the largest effect in the predicted direction the 95% CI allows (sign-adjusted units)
        ceil = lambda c: max(sg * c[0], sg * c[1])
        se_z = 1 / np.sqrt(max(s0["resid_df"] - 1, 1))  # as in linkage.evaluate
        ceil_joint = float(np.tanh(np.arctanh(sg * s0["partial_r"]) + JOINT_Z * se_z))
        ceil_joint_b2 = (float(min(ceil_joint / np.sqrt(rx_b2 * ry_b2), 1.0)) if rx_b2 > 0.05 and ry_b2 > 0.05 else 1.0)
        seeds = [r[k] for k in ("seed0", "seed1", "seed2") if k in r]
        seed_ok = lambda x: bool(x["checks"]["fl_p_lt_05"] and x["checks"]["sign_matches"] and x["checks"]["excess_lb_gt_0"]
                                 and int(x["checks"]["loco_excess_positive_classes"].split("/")[0]) >= 2)
        q = q14[lid] if lk["primary14"] else q18[lid]
        crit = {
            "c1_fl_p_and_sign": bool(s0["checks"]["fl_p_lt_05"] and s0["checks"]["sign_matches"]),
            "c2_excess_lb_gt_0": bool(s0["checks"]["excess_lb_gt_0"]),
            "c3_loco_2of3": int(s0["checks"]["loco_excess_positive_classes"].split("/")[0]) >= 2,
            "c4_bh_q_lt_05": bool(q < 0.05),
            "c5_reliability_ge_05": bool(rel_x >= 0.5 and rel_y >= 0.5),
            "c6_seeds_and_leave3out": bool(len(seeds) == 3 and all(seed_ok(x) for x in seeds)
                                           and r["influence"]["drop3_supportive"]["passes_c1"]),
        }
        inf, no25 = r["influence"]["drop3_supportive"], r["without_2025"]
        mde1 = float(np.tanh((norm.ppf(0.95) + Z80) * se_z))
        positive = ci[0] is not None and min(sg * ci[0], sg * ci[1]) > 0
        rows.append({
            "id": lid, "experiment": lk["experiment"], "test": lk["test"], "positions": "+".join(lk["positions"]),
            "construct": lk["construct"], "kind": lk["kind"], "label": lk["label"], "outcome_target": lk["outcome_target"],
            "target_dagger": lk["target_dagger"], "feature_source": lk["feature_source"], "sign_source": lk["sign_source"],
            "primary14": lk["primary14"], "feature": lk["feature"], "target": lk["target"], "expected_sign": sg,
            "n": s0["n"], "resid_df": s0["resid_df"], "raw_r": s0["raw_r"], "partial_r": s0["partial_r"],
            "ci_lo": ci[0], "ci_hi": ci[1], "mde80": s0["partial_r_mde80"], "mde80_one_sided": mde1,
            "fl_p": s0["fl_p"], "fl_p_two_sided": r["two_sided"]["fl_p"],
            "bh_q14": q14.get(lid), "bh_q18": q18[lid],
            "excess": s0["delta_r2_vs_B2"]["excess"], "excess_se": s0["delta_r2_vs_B2"]["se"],
            "excess_within_position": s0["delta_r2_vs_B2"]["excess_within_position"],
            "cv_perm_p": s0["cv_perm_p"], "loco_pos": s0["checks"]["loco_excess_positive_classes"],
            **{f"loco_excess_{y}": v["excess"] for y, v in s0["loco"].items()},
            **{f"partial_r_{y}": v["partial_r"] for y, v in s0["by_class"].items()},
            "seeds_run": len(seeds), "seeds_passing": sum(seed_ok(x) for x in seeds), "c6_evaluated": len(seeds) == 3,
            "excess_seeds": [x["delta_r2_vs_B2"]["excess"] for x in seeds],
            "no2025_n": no25["n"], "no2025_partial_r": no25["partial_r"],
            "no2025_ci_lo": no25["partial_r_ci95"][0], "no2025_ci_hi": no25["partial_r_ci95"][1],
            "no2025_fl_p": no25["fl_p"], "no2025_excess": no25["delta_r2_vs_B2"]["excess"],
            "drop3_top3_share": r["influence"]["top3_supportive_share_of_partial_r"],
            "drop3_n": inf["n"], "drop3_partial_r": inf["partial_r"],
            "drop3_ci_lo": inf["partial_r_ci95"][0], "drop3_ci_hi": inf["partial_r_ci95"][1], "drop3_fl_p": inf["fl_p"],
            "drop3_full_excess": (inf.get("full_harness") or {}).get("delta_r2_vs_B2", {}).get("excess"),
            "drop3_full_excess_se": (inf.get("full_harness") or {}).get("delta_r2_vs_B2", {}).get("se"),
            "drop3_full_loco": (inf.get("full_harness") or {}).get("checks", {}).get("loco_excess_positive_classes"),
            "drop3contra_partial_r": r["influence"]["drop3_contrary"]["partial_r"],
            "drop3contra_fl_p": r["influence"]["drop3_contrary"]["fl_p"],
            "r_target_log_exposure": r["exposure"]["r_target_log_exposure"],
            "rel_x": rel_x, "rel_x_year_centered": rel_x_year, "rel_y": rel_y, "rel_x_b2": rx_b2, "rel_y_b2": ry_b2,
            "dis_lo": dis[0], "dis_hi": dis[1], "dis_b2_lo": dis_b2[0], "dis_b2_hi": dis_b2[1],
            "ceiling": ceil(ci), "ceiling_dis": ceil(dis), "ceiling_dis_b2": ceil(dis_b2),
            "ceiling_joint": ceil_joint, "ceiling_joint_dis_b2": ceil_joint_b2,
            **crit, "passes_1to6": all(crit.values()), "verdict": _verdict(crit, all(seed_ok(x) for x in seeds)),
            "bound_statement": (
                f"partial r {s0['partial_r']:+.2f}, 95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}] excludes 0 in the predicted "
                f"direction (reliability-corrected with beyond-B2 reliabilities [{dis_b2[0]:+.2f}, {dis_b2[1]:+.2f}]; "
                f"as-used [{dis[0]:+.2f}, {dis[1]:+.2f}])" if positive else
                f"per link at 95%: observed partial r above {ceil(ci):+.2f} in the predicted direction is excluded "
                f"(jointly over the 18 links, Bonferroni: above {ceil_joint:+.2f}); for error-free measures "
                f"(beyond-B2 reliability-corrected) a true-score partial r up to {ceil(dis_b2):+.2f} is not excluded "
                f"({ceil(dis):+.2f} with as-used reliabilities, an under-correction); one-sided 80%-power MDE {mde1:.2f}"),
        })
    return rows


# Smallest p-values of the other round-1 tests (round-1 results JSONs; rev-1 dR2 permutation p, not re-estimated):
# E006 X4 (vmax -> S0 with stricter QC: the same link; at its 500-permutation floor), E006 X2, X3, tau -> A0,
# E006 vmax -> game_top_speed, E003 INTERIOR stratum. All other round-1 tests have p >= 0.016.
ROUND1_SMALLEST_OTHER_P = [0.0020, 0.0040, 0.0040, 0.0060, 0.0100, 0.0100]


def multiplicity(tm: list[dict]) -> dict:
    """BH q across all ~90 round-1 tests for a link that passes criterion 1 with the smallest p of the round:
    min over j of p_(j) m / j on the sorted list of the link's rev-2 p and the smallest other round-1 p-values (with
    every other p >= 0.016, no larger j can give a smaller ratio). Not defined here for links ranked lower."""
    out = {}
    for r in tm:
        if r["c1_fl_p_and_sign"] and r["fl_p"] <= min(ROUND1_SMALLEST_OTHER_P):
            ps = np.sort([r["fl_p"]] + ROUND1_SMALLEST_OTHER_P)
            q = float(min(1.0, np.min(ps * N_ROUND1_TESTS / np.arange(1, len(ps) + 1))))
            out[r["id"]] = {"fl_p": r["fl_p"], "bh_q14": r["bh_q14"], "bh_q18": r["bh_q18"],
                            "m_round1_tests": N_ROUND1_TESTS, "bh_q_all_round1_tests": q,
                            "note": "the other round-1 p-values are rev-1 (sign-agnostic dR2 permutation) values"}
    return out


# =====================================================================================================
# 6. Post hoc: E006 targets with the shared-module A-S estimator (AS_R = 1000 instead of 20)
# =====================================================================================================

def e006_r1000(L: list[dict]) -> dict:
    """Shared bdb27.speed A-S profiles (R = 1000 draws; round 1 used R = 20, whose Monte Carlo noise moved WR S0
    by up to 1 between-player SD) for the E006-P1/P2 samples; same harness, seed 0. Post hoc, labeled as such."""
    out = {}
    ids = sorted(set().union(*[set(_sample(lk)["nfl_id"].to_list()) for lk in L if lk["id"] in ("E006-P1", "E006-P2")]))
    fr = (pl.scan_parquet(speed.GAME_FRAMES).filter(pl.col("nfl_id").is_in(ids))
          .select("game_id", "play_id", "nfl_id", "speed", "a_tan").collect())
    prof = speed.game_as_profiles(fr).select("nfl_id", game_A0_r1000="A0", game_S0_r1000="S0")
    del fr
    for lk in L:
        if lk["id"] not in ("E006-P1", "E006-P2"):
            continue
        tgt = {"game_S0": "game_S0_r1000", "game_A0": "game_A0_r1000"}[lk["target"]]
        d = lk["df"].join(prof, on="nfl_id", how="left")
        both = d.filter(pl.col(lk["target"]).is_finite() & pl.col(tgt).is_finite())
        r = linkage.evaluate(d, lk["feature"], tgt, expected_sign=lk["expected_sign"], seed=0, n_jobs=N_JOBS, **EVAL)
        out[lk["id"]] = {"target": tgt, "r_with_round1_target": float(np.corrcoef(both[lk["target"]], both[tgt])[0, 1]),
                         "result": r}
        log(f"post hoc R=1000 {lk['id']}: n={r['n']} partial={r['partial_r']:+.3f} {r['partial_r_ci95']} "
            f"fl_p={r['fl_p']:.4f} excess={r['delta_r2_vs_B2']['excess']:+.4f}±{r['delta_r2_vs_B2']['se']:.4f}")
    return out


# =====================================================================================================
# 7. Caveats (round-1 skeptics, experiments/log.md GATE 1 + E00x verification bullets) and scout sentences
# =====================================================================================================
ALL_2025 = "2025 class has one season of games (rookie year only)."
CAVEATS = {
    "E002-P1": ["Game OUT/IN 'retention' mostly measures how often the break is a turnaround/sit-down (route style); "
                "r = 0.89 with game turn radius, so it is not speed carried through a 90-degree cut.",
                "Combine route drills are run at submaximal speed (drill stem speed uncorrelated with the 40, r = -0.14)."],
    "E002-P2": ["Separation at pass_forward mostly precedes the break on timing routes.",
                "Target reliability 0.41 < 0.5 (separation over expected, HITCH/OUT/IN).",
                "Outcome target: also run without the 2025 class."],
    "E002-P3": ["Combine stop-braking composite is confounded with draft class (2025 COMEBACK is a different movement); "
                "reliability 0.44 < 0.5, and 0.17 within draft year.",
                "Low power: n ~ 62 with ~19 B2 columns + year dummies."],
    "E003-P1": ["Game burst window runs into blocker contact and stunts (not pure acceleration).",
                "40-yd burst carries a Combine-year artifact (beyond-B2 reliability 0.62 -> 0.51 with year removed).",
                "Game burst correlates +0.31 with log pass-rush snaps (exposure)."],
    "E003-P2": ["NGS get-off mixes reaction to the snap (not measurable at the Combine) with burst."],
    "E003-P3": ["Quick-pressure rate over expected is exposure-tied (r = +0.39 with log snaps in round 1).",
                "Outcome target: also run without the 2025 class."],
    "E004-P1": ["Target (peak acceleration 0-0.6 s after the snap) is scheme/reaction, not the Combine construct; "
                "teammate r = 0.29.",
                "Mirror-drill reliability is within one attempt (single attempt per player; the partner/tempo of that rep "
                "is confounded)."],
    "E004-P2": ["Outcome traits correlate -0.36 with log pass-block snaps (exposure).",
                "Only the mirror metric was tested; the wave-drill metric was screened out on reliability (untested).",
                "B2 cross-validated R^2 is negative: very low power."],
    "E004-P3": ["Target reliability 0.42 < 0.5; trait correlates -0.35 with log snaps.",
                "Only the mirror metric was tested (the wave drill is untested)."],
    "E005-P1": ["Game 'peak backpedal speed' is mostly retreat speed at the hip flip, the throw or the 2.5 s cutoff, "
                "not backpedal speed.",
                "Game backpedal reliability (0.85) is partly a team/scheme effect (odd/even games cannot separate player "
                "from team)."],
    "E005-P2": ["Target is generic post-throw redirection (often deep safeties pursuing short completions), not "
                "'breaking on the ball'."],
    "E006-P1": ["Near re-test of E001 (S0 r ~ 0.85 with E001's target; E001 was seen before the prereg): not independent "
                "confirmation. Same 61 WRs as E001-WR; B2-residual feature r 0.77, target r 0.83 (wr_speed_signal).",
                "Rests on few players: the 3 most supportive WRs carry ~42% of the partial r; without them r +0.31 with "
                "95% CI including 0 (one-sided p 0.03 passes criterion 6 as defined). Without the rookie-only 2025 class "
                "r +0.38; by class +0.29 / +0.50 / +0.67.",
                "Passes BH over the 14 primaries (q 0.010), not across all ~90 round-1 tests (q ~ 0.06).",
                "The directly observed game top speed (E001-WR, q95 of play max) is weaker: fails BH and leave-3-out.",
                "Expected sign '+' inferred from the hypothesis text (the prereg has no sign field); two-sided p ~ 0.001.",
                "Vanishes once a raw tracked flying split is added to B2: the signal is tracked flying speed, not curve shape.",
                "S0 extrapolates ~1 yd/s beyond observed game speeds; round-1 S0 used R = 20 Monte Carlo draws "
                "(shared module now R = 1000; see post hoc).",
                "No performance consequence yet (deep-route separation and YACOE null)."],
    "E006-P2": ["Round 1 found the partial r in the wrong direction (pooled, -0.14); A0 at R = 20 draws.",
                "The bound is for the pooled within-position partial r (WR/TE/DB/DL together); a position-specific effect "
                "(round-1 WR partial r +0.11 at n ~ 60) is not excluded by it.",
                "WR a0 reliability 0.57 (0.40 within draft year)."],
    "E006-P3": ["YACOE reliability ~0.55 (WR) and exposure-tied (r = +0.27 WR, +0.51 TE with log catches).",
                "Outcome target: also run without the 2025 class."],
    "E001-WR": ["Seen before E006/E007 preregistrations (the lead that started the hidden-speed track).",
                "Not preregistered: the one-sided '+' direction comes from the E001 hypothesis (raw link) and was set "
                "after round 1; its stated beyond-B2 expectation was ~0. Two-sided p reported alongside.",
                "Feature is the max over attempts (device s; reliability computed for a max); target q95 of play max "
                "speed is exposure-invariant (split-half recomputed on the link sample)."],
    "E001-DB": ["Game top speed reliability 0.63 for DB."],
    "E001-DL": ["B2 (40 time, weight) explains most of DL top speed: little residual variance left."],
    "E001-OL": ["OL rarely reach top speed in games; target reliability 0.78."],
}
SCOUT = {
    "E002-P1": "A receiver who keeps more speed through Combine route-drill breaks keeps more speed through game breaks.",
    "E002-P2": "...and so separates more at the throw on breaking routes.",
    "E002-P3": "A receiver who brakes harder into Combine curls/comebacks brakes harder on hitches.",
    "E003-P1": "A lineman who covers more ground in the first second of the 40 bursts harder off the ball.",
    "E003-P2": "...and gets off the ball faster (NGS get-off).",
    "E003-P3": "...and wins quicker pressures.",
    "E004-P1": "A lineman who swings his momentum harder in the mirror drill accelerates harder into his pass set.",
    "E004-P2": "...and allows lower pressure probability.",
    "E004-P3": "...and allows fewer pressures.",
    "E005-P1": "A DB with a faster drill backpedal backpedals faster in coverage.",
    "E005-P2": "A DB with a sharper drill plant-and-drive transitions faster once the ball is thrown.",
    "E006-P1": "Same 40 time, higher tracked top speed: the receiver is faster at top end on Sundays.",
    "E006-P2": "A faster-accelerating 40 (beyond the 10-yd split) means more explosive in-game acceleration.",
    "E006-P3": "Faster receivers gain more yards after the catch than expected.",
    "E001-WR": "Same 40 time, higher tracked peak speed: a faster receiver in games.",
    "E001-DB": "Same 40 time, higher tracked peak speed: a faster DB in games.",
    "E001-DL": "Same 40 time, higher tracked peak speed: a faster lineman in games.",
    "E001-OL": "Same 40 time, higher tracked peak speed: a faster lineman in games.",
}
ROW_ORDER = ["E001-WR", "E006-P1", "E006-P3", "E002-P1", "E002-P3", "E002-P2",
             "E001-DL", "E003-P1", "E003-P2", "E003-P3",
             "E001-OL", "E004-P1", "E004-P2", "E004-P3",
             "E001-DB", "E005-P1", "E005-P2", "E006-P2"]


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def link_players(L: list[dict]) -> pl.DataFrame:
    """Long table: one row per link x player with feature, target and their B2 residuals (added-variable plots)."""
    out = []
    for lk in L:
        d = _sample(lk)
        _, res = _residuals(d, [lk["feature"], lk["target"]])
        out.append(d.select("nfl_id", "display_name", "position", "draft_year",
                            feature=pl.col(lk["feature"]).cast(pl.Float64), target=pl.col(lk["target"]).cast(pl.Float64))
                   .with_columns(link_id=pl.lit(lk["id"]), x_res=pl.Series(res[lk["feature"]]),
                                 y_res=pl.Series(res[lk["target"]])))
    return pl.concat(out)


# Top-speed vs start-phase metrics of the tracked 40 (bdb27.speed + E001's device peak).
TOP_SPEED_40 = {"vmax", "V0", "smax", "device peak s (E001)", "split_10_20", "split_20_30", "split_30_40", "split_20_40",
                "v_10_20", "v_20_30", "v_30_40", "v_20_40"}


def _profile_groups(prof: pl.DataFrame) -> pl.DataFrame:
    """Defined profile rows with their comparison group: tracked 40-yd dash (all repeat attempts), position drill, or a
    DL drill composite that includes the 40 (in neither group)."""
    is40 = pl.col("probe").str.starts_with("40") | (pl.col("scope") == "40-yd dash attempts")
    return (prof.filter(pl.col("rel_round1_def").is_finite() & pl.col("share_session").is_finite())
            .with_columns(group=pl.when(is40).then(pl.lit("40-yd dash"))
                          .when(pl.col("kind") == "cross-drill incl. 40").then(pl.lit("drill composite incl. 40"))
                          .otherwise(pl.lit("position drill"))))


def profile_summary(prof: pl.DataFrame) -> dict:
    """Like-for-like counts and medians behind the drill-reliability statements (figure title, log, writeup)."""
    d = _profile_groups(prof)
    ge = lambda t: int((t["rel_round1_def"] >= 0.5).sum())
    rng = lambda t, c: [float(t[c].min()), float(t[c].max())] if t.height else None
    out = {"n_signatures": prof.height, "n_defined": d.height, "by_group_kind": {}}
    for (g, k), t in sorted(d.group_by("group", "kind"), key=lambda x: x[0]):
        out["by_group_kind"][f"{g} | {k}"] = {"n": t.height, "n_ge_05": ge(t), "median": float(t["rel_round1_def"].median())}
    pdr = d.filter(pl.col("group") == "position drill")
    cx = pdr.filter(pl.col("kind") == "cross-drill")
    fam_med = {f: float(t["rel_round1_def"].median()) for (f,), t in cx.group_by(["probe"])}
    out["position_drill"] = {"n": pdr.height, "n_ge_05": ge(pdr), "cross_drill_median_by_family": fam_med,
                             "note": "counts include derivative variants of the same construct (e.g. entry speed, "
                                     "min speed and retention of one break)"}
    f40 = d.filter(pl.col("group") == "40-yd dash")
    sp = f40.filter(pl.col("probe").str.starts_with("40"))
    top, start = sp.filter(pl.col("metric").is_in(TOP_SPEED_40)), sp.filter(~pl.col("metric").is_in(TOP_SPEED_40))
    dl40 = f40.filter(~pl.col("probe").str.starts_with("40"))
    out["forty"] = {"n": f40.height, "n_ge_05": ge(f40), "median_all": float(f40["rel_round1_def"].median()),
                    "median_speed_module": float(sp["rel_round1_def"].median()), "n_speed_module": sp.height,
                    "n_dl_starts": dl40.height,
                    "top_speed": {"n": top.height, "rel": rng(top, "rel_round1_def"), "share_session": rng(top, "share_session"),
                                  "share_year": rng(top, "share_year")},
                    "start_phase": {"n": start.height, "rel": rng(start, "rel_round1_def"),
                                    "share_session": rng(start, "share_session"), "share_year": rng(start, "share_year")},
                    "dl_starts": {"rel": rng(dl40, "rel_round1_def"), "share_session": rng(dl40, "share_session"),
                                  "share_year": rng(dl40, "share_year")}}
    wr = prof.filter((pl.col("probe") == "A route breaks (E002)") & (pl.col("position") == "WR"))
    out["wr_route_same_drill_retest"] = {
        "retest_icc_median": float(wr["retest_icc"].median()), "composite_rel_median": float(wr["rel_round1_def"].median()),
        "retest_n_player_drill_pairs": sorted({int(v) for v in wr["retest_n_players"].drop_nulls()}),
        "note": "single-attempt ICC of repeated attempts of the same route drill (player x drill pairs with >= 2 "
                "attempts; profile column retest_n_players counts these pairs), centered by drill only. All repeats "
                "fall in the same Combine session, so session effects inflate it; composite = Spearman-Brown "
                "cross-drill reliability"}
    used = d.filter(pl.col("used_by") != "")
    out["used_features"] = used.select("probe", "metric", "scope", "kind", "used_by", "rel_round1_def", "rel_year",
                                       "share_session", "share_year").to_dicts()
    return out


def wr_speed_signal(lp: pl.DataFrame) -> dict:
    """E001-WR vs E006-P1: one signal measured twice on the same receivers (correlations of the features, targets and
    their B2 residuals over the shared players; residuals from each link's own sample)."""
    a, b = (lp.filter(pl.col("link_id") == k) for k in ("E006-P1", "E001-WR"))
    j = a.join(b, on="nfl_id", suffix="_e001")
    r = lambda u, v: float(np.corrcoef(j[u], j[v])[0, 1])
    return {"n_e006_p1": a.height, "n_e001_wr": b.height, "n_shared": j.height,
            "r_feature_raw": r("feature", "feature_e001"), "r_feature_b2_resid": r("x_res", "x_res_e001"),
            "r_target_raw": r("target", "target_e001"), "r_target_b2_resid": r("y_res", "y_res_e001")}


def transfer_summary(tm: list[dict]) -> dict:
    """Ranges and counts behind the headline statements, over the links that fail criterion 1 ('nulls')."""
    nl = [r for r in tm if not r["c1_fl_p_and_sign"]]
    rng = lambda k: [float(min(r[k] for r in nl)), float(max(r[k] for r in nl))]
    return {
        "n_links": len(tm), "passes_1to6": [r["id"] for r in tm if r["passes_1to6"]],
        "passes_c1": [r["id"] for r in tm if r["c1_fl_p_and_sign"]], "n_nulls": len(nl),
        "nulls_min_fl_p": float(min(r["fl_p"] for r in nl)),
        "nulls_signed_partial_r": [float(min(r["partial_r"] for r in nl)), float(max(r["partial_r"] for r in nl))],
        "ceiling_per_link_95": rng("ceiling"), "ceiling_joint_bonferroni18": rng("ceiling_joint"),
        "ceiling_corrected_beyond_b2": rng("ceiling_dis_b2"), "ceiling_corrected_as_used": rng("ceiling_dis"),
        "ceiling_joint_corrected_beyond_b2": rng("ceiling_joint_dis_b2"),
        "corrected_beyond_b2_ge_05": sorted([r["id"] for r in nl if r["ceiling_dis_b2"] >= 0.5]),
        "corrected_as_used_ge_05": sorted([r["id"] for r in nl if r["ceiling_dis"] >= 0.5]),
        "mde80_one_sided": rng("mde80_one_sided"), "mde80_two_sided": rng("mde80"),
        "nulls_passing_c5": sum(r["c5_reliability_ge_05"] for r in nl),
        "nulls_passing_c3": sum(r["c3_loco_2of3"] for r in nl),
        "nulls_position_drill_feature": [r["id"] for r in nl if r["feature_source"] == "position drill"],
        "nulls_40_feature": [r["id"] for r in nl if r["feature_source"] == "40-yd dash"],
        "position_drill_features_tested": sorted({r["feature"] for r in tm if r["feature_source"] == "position drill"}),
    }


# =====================================================================================================
# 8. Figures
# =====================================================================================================
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, BLUE_WASH, ORANGE_WASH = "#2a78d6", "#eb6834", "#cde2fb", "#fbd9c9"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF,
                         "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": False, "text.color": INK})
    return plt


def _fp(p: float) -> str:
    return f"{p:.4f}" if p < 0.001 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def figure_transfer_map(tm: pl.DataFrame, prof: pl.DataFrame, lp: pl.DataFrame, path: Path) -> None:
    """Forest plot: partial r beyond B2 (sign-adjusted so right = the expected direction), 95% CI, and the
    reliability-corrected interval (beyond-B2 reliabilities), grouped by position. Marker shape = feature source,
    fill = one-sided p < 0.05, ring = passes go criteria 1-6; E006-P1 also shown without 2025 and without its top 3."""
    plt = _style()
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    rows = {r["id"]: r for r in tm.iter_rows(named=True)}
    rel = lambda probe, metric, scope: prof.filter((pl.col("probe") == probe) & (pl.col("metric") == metric)
                                                   & (pl.col("scope") == scope))["rel_round1_def"][0]
    burst4 = rel("B DL burst (E003)", "d10", "cross-drill burst4")
    wave = rel("C OL lateral (E004)", "wave rev_accel", "reversals within one attempt")
    ws = wr_speed_signal(lp)
    mult = multiplicity(tm.to_dicts())
    items = ([("head", "Wide receivers"), ("link", "E001-WR"), ("link", "E006-P1"), ("sub", "E006-P1", "no2025"),
              ("sub", "E006-P1", "drop3"), ("link", "E006-P3"), ("link", "E002-P1"), ("link", "E002-P3"),
              ("link", "E002-P2"), ("gap",), ("head", "Defensive line")] + [("link", i) for i in ROW_ORDER[6:10]]
             + [("note", f"DL position drills: not tested against games (4-drill burst composite reliability {burst4:.2f})"),
                ("gap",), ("head", "Offensive line")] + [("link", i) for i in ROW_ORDER[10:14]]
             + [("note", f"OL wave drill: screened out on reliability ({wave:.2f}), not tested"), ("gap",),
                ("head", "Defensive backs")] + [("link", i) for i in ROW_ORDER[14:17]]
             + [("gap",), ("head", "Pooled WR + TE + DB + DL"), ("link", "E006-P2")])
    y, placed = 0.0, []
    for it in items:
        if it[0] == "head":
            placed.append((it, y + 0.12)); y += 0.85
        elif it[0] == "link":
            placed.append((it, y)); y += 1.0
        elif it[0] == "sub":
            placed.append((it, y - 0.42)); y += 0.5
        elif it[0] == "note":
            placed.append((it, y - 0.25)); y += 0.7
        else:
            y += 0.45
    top_in, bot_in = 1.95, 1.75
    H = 0.33 * y + top_in + bot_in
    fig = plt.figure(figsize=(14, H))
    top, bot = 1 - top_in / H, bot_in / H
    ax = fig.add_axes([0.385, bot, 0.32, top - bot])
    ax.set_ylim(y - 0.4, -0.5)
    ax.set_xlim(-1, 1)
    yf = lambda v: bot + (top - bot) * (y - 0.4 - v) / (y - 0.4 + 0.5)  # data y -> figure fraction
    for xv in (-0.5, 0.5):
        ax.axvline(xv, color=GRID, lw=0.8, zorder=0)
    ax.axvline(0, color=BASE, lw=1.0, zorder=1)
    cols = [(0.718, "n"), (0.752, "p (1-sided)"), (0.812, "BH q"), (0.853, "verdict")]
    for xf, h in cols:
        fig.text(xf, top + 0.004, h, ha="left", va="bottom", fontsize=8, color=INK2, weight="bold")
    mk = lambda r: "o" if r["feature_source"] == "position drill" else "s"
    sgn = lambda r, a, b: sorted((r["expected_sign"] * a, r["expected_sign"] * b))
    yy_of = {}
    for it, yy in placed:
        if it[0] == "head":
            fig.text(0.010, yf(yy), it[1], ha="left", va="center", fontsize=9.5, weight="bold", color=INK)
        elif it[0] == "note":
            fig.text(0.030, yf(yy), it[1], ha="left", va="center", fontsize=7.8, style="italic", color=MUTED)
        elif it[0] == "link":
            r = rows[it[1]]
            yy_of[it[1]] = yy
            col, wash = (ORANGE, ORANGE_WASH) if r["outcome_target"] else (BLUE, BLUE_WASH)
            lo, hi = sgn(r, r["ci_lo"], r["ci_hi"])
            dlo, dhi = sgn(r, r["dis_b2_lo"], r["dis_b2_hi"])
            est = r["expected_sign"] * r["partial_r"]
            ax.add_patch(plt.Rectangle((dlo, yy - 0.27), dhi - dlo, 0.54, color=wash, lw=0, zorder=1))
            ax.plot([lo, hi], [yy, yy], color=col, lw=2, solid_capstyle="round", zorder=3)
            ax.plot([r["mde80_one_sided"]], [yy], "|", color=INK2, ms=9, mew=1.2, zorder=2)
            ax.plot([est], [yy], mk(r), ms=8 if mk(r) == "o" else 7.2, mfc=col if r["c1_fl_p_and_sign"] else SURF,
                    mec=col, mew=2, zorder=4)
            if r["passes_1to6"]:
                ax.plot([est], [yy], "o", ms=15, mfc="none", mec=INK, mew=1.3, zorder=5)
            fig.text(0.010, yf(yy), f"{r['experiment']} {r['test']}", ha="left", va="center", fontsize=7.6, color=MUTED)
            fig.text(0.375, yf(yy), r["label"] + (" †" if r["target_dagger"] else ""), ha="right", va="center",
                     fontsize=8.5, color=INK)
            q = r["bh_q14"] if r["primary14"] else r["bh_q18"]
            for (xf, _), txt in zip(cols, (f"{r['n']}", _fp(r["fl_p"]), f"{q:.3f}" + ("*" if not r["primary14"] else ""),
                                           r["verdict"])):
                fig.text(xf, yf(yy), txt, ha="left", va="center", fontsize=8.2, color=INK,
                         weight="bold" if r["passes_1to6"] and xf > 0.85 else "normal")
        else:  # sub-row: the GO link without the 2025 class / without its 3 most supportive players
            r = rows[it[1]]
            lo, hi = sgn(r, r[f"{it[2]}_ci_lo"], r[f"{it[2]}_ci_hi"])
            est = r["expected_sign"] * r[f"{it[2]}_partial_r"]
            ax.plot([lo, hi], [yy, yy], color=MUTED, lw=1.3, solid_capstyle="round", zorder=3)
            ax.plot([est], [yy], mk(r), ms=5, mfc=MUTED, mec=MUTED, zorder=4)
            lab = ("without the 2025 class (rookie season only)" if it[2] == "no2025" else
                   f"without its 3 most supportive WRs ({r['drop3_top3_share']:.0%} of the partial r)")
            fig.text(0.375, yf(yy), lab, ha="right", va="center", fontsize=7.6, color=MUTED)
            for (xf, _), txt in zip(cols[:2], (f"{r[f'{it[2]}_n']}", _fp(r[f"{it[2]}_fl_p"]))):
                fig.text(xf, yf(yy), txt, ha="left", va="center", fontsize=7.6, color=MUTED)
    # bracket: E001-WR and E006-P1 are one signal on the same receivers
    y1, y2 = yy_of["E001-WR"], yy_of["E006-P1"]
    ax.plot([-0.95, -0.97, -0.97, -0.95], [y1, y1, y2, y2], color=INK2, lw=0.9, zorder=3)
    ax.text(-0.93, (y1 + y2) / 2, f"one signal: {ws['n_shared']} WRs in both;\nB2-residual r of the features "
            f"{ws['r_feature_b2_resid']:.2f},\nof the targets {ws['r_target_b2_resid']:.2f}", ha="left", va="center",
            fontsize=7, color=INK2, linespacing=1.25)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1], ["-1", "-0.5", "0", "0.5", "1"])
    ax.set_xlabel("partial r beyond B2, signed so right = the expected direction", fontsize=8.6)
    fig.text(0.010, 1 - 0.28 / H, "Transfer map: does a Combine tracking signature add anything beyond the stopwatch "
             "and draft slot?", fontsize=11.5, weight="bold", color=INK, va="top")
    fig.text(0.010, 1 - 0.60 / H, "14 preregistered round-1 primaries + the E001 sanity link (not preregistered), "
             "re-estimated under protocol rev 2. B2 = position, draft year, traditional Combine times and measurements, "
             "draft pick.", fontsize=8.6, color=INK2, va="top")
    gray = dict(color=INK2, lw=0, mew=1.5)
    h = [Line2D([], [], color=BLUE, lw=2, marker="o", ms=7, mfc=BLUE, label="target: a game movement trait"),
         Line2D([], [], color=ORANGE, lw=2, marker="o", ms=7, mfc=ORANGE, label="target: a game outcome"),
         Line2D([], [], marker="o", ms=7, mfc=SURF, **gray, label="feature: position drill"),
         Line2D([], [], marker="s", ms=6.5, mfc=SURF, **gray, label="feature: tracked 40-yd dash"),
         Line2D([], [], marker="o", ms=7, mfc=INK2, **gray, label="filled = one-sided p < 0.05"),
         Line2D([], [], marker="o", ms=11, mfc="none", mec=INK, lw=0, mew=1.3, label="ring = passes go criteria 1-6"),
         (Patch(color=BLUE_WASH), Patch(color=ORANGE_WASH)),
         Line2D([], [], color=INK2, lw=0, marker="|", ms=9, label="min. detectable r (80% power, one-sided)")]
    labels = [x.get_label() if not isinstance(x, tuple) else "error-free (reliability-corrected) interval" for x in h]
    fig.legend(handles=h, labels=labels, loc="upper left", bbox_to_anchor=(0.004, 1 - 0.82 / H), ncol=4, frameon=False,
               fontsize=8, handlelength=1.8, columnspacing=1.6, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})
    nl = tm.filter(~pl.col("c1_fl_p_and_sign"))
    dj = (nl["ceiling_joint"] - nl["ceiling"])
    two = {k: rows[k]["fl_p_two_sided"] for k in ("E001-WR", "E006-P1")}
    qall = (f" Across all ~{N_ROUND1_TESTS} round-1 tests, E006-P1's q = {mult['E006-P1']['bh_q_all_round1_tests']:.2f}."
            if "E006-P1" in mult else "")
    fig.text(0.010, 0.12 / H, "\n".join([
        "Line: 95% CI of the partial correlation (Fisher z). Its ceiling bounds each link on its own; jointly over the 18 "
        f"links (Bonferroni) the ceilings rise by {dj.min():.2f}-{dj.max():.2f}.",
        "Band: CI / sqrt(rel_x x rel_y) with beyond-B2 reliabilities (of the parts of feature and target left after B2), "
        "classical measurement error, clipped at +-1: what error-free measures could show.",
        "p: Freedman-Lane, 10,000 permutations, one-sided in the expected direction: preregistered for E002-E005, inferred "
        f"from the hypothesis for E006, set after round 1 for E001 (two-sided p: E001-WR {_fp(two['E001-WR'])}, "
        f"E006-P1 {_fp(two['E006-P1'])}).",
        f"BH q over the 14 round-1 primaries (* = E001, over all 18 links).{qall} Seeds 1-2 (criterion 6) were run "
        "only when p < 0.10.",
        "Go criteria: 1 p < 0.05, 2 CV excess dR2 - SE > 0, 3 leave-one-class-out excess > 0 in 2 of 3 classes, "
        "4 BH q < 0.05, 5 reliability >= 0.5, 6 three CV seeds + leave-3-out.",
        "† Round-1 review: the game target is narrower than its name (E002-P1 route style; E002-P2 separation mostly "
        "precedes the break; E003-P1 window runs into blockers; E004-P1 scheme/reaction; E005-P1 retreat at the flip or "
        "throw; E005-P2 generic redirection)."]), fontsize=7.1, color=INK2, va="bottom", linespacing=1.45)
    fig.savefig(path, dpi=200)
    plt.close(fig)


FAMILIES = [("A route breaks (E002)", "WR/TE route-drill breaks (E002)"), ("B DL burst (E003)", "DL burst: 40 start and drills (E003)"),
            ("C OL lateral (E004)", "OL mirror, wave and drop drills (E004)"),
            ("D DB transition (E005)", "DB backpedal and transition drills (E005)"),
            ("40-yd dash (bdb27.speed)", "40-yd dash curve (E001, E006, E007)")]
HIGHLIGHT = {  # (probe, position, metric, scope) -> (short label, optional (dx, dy) label offset in axis units)
    ("A route breaks (E002)", "WR", "turn retention", "turn-route drills, composite"): ("speed kept through breaks (E002 P1-2)", None),
    ("A route breaks (E002)", "WR", "stop peak_decel", "stop-route drills, composite"): ("stop braking (E002 P3)", None),
    ("B DL burst (E003)", "DL", "d10", "40-yd dash attempts"): ("40 first-second burst (E003)", None),
    ("B DL burst (E003)", "DL", "d10", "cross-drill burst4"): ("drill burst, 4 drills", None),
    ("C OL lateral (E004)", "OL", "mirror rev_accel", "reversals within one attempt"): ("mirror reversal accel. (E004)", None),
    ("C OL lateral (E004)", "OL", "wave rev_accel", "reversals within one attempt"): ("wave reversal accel.", None),
    ("D DB transition (E005)", "DB", "backpedal bp_peak", "backpedal drills (5), drill means"): ("backpedal speed (E005 P1)", (0.01, 0.12)),
    ("D DB transition (E005)", "DB", "dv_10 | entry+turn adj.", "reversal drills (4), drill means"): ("plant-and-drive dv (E005 P2)", (0.06, -0.10)),
    ("40-yd dash (bdb27.speed)", "all", "vmax", "40-yd attempts, centered by position"): ("fitted top speed vmax (E006)", None),
    ("40-yd dash (bdb27.speed)", "all", "a0", "40-yd attempts, centered by position"): ("fitted acceleration a0 (E006)", None),
    ("40-yd dash (bdb27.speed)", "all", "v_20_40", "40-yd attempts, centered by position"): ("flying 20 (E007)", None),
    ("40-yd dash (bdb27.speed)", "all", "split_0_10", "40-yd attempts, centered by position"): ("0-10 yd split (tracked)", None),
}
KIND_MARK = {"repeat attempts": "o", "cross-drill": "^", "within-attempt": "D", "cross-drill incl. 40": "v"}
KIND_LABEL = {"repeat attempts": "repeat attempts of the same drill", "cross-drill": "cross-drill: agreement between drills",
              "within-attempt": "within one attempt (OL; overstates a trait)", "cross-drill incl. 40": "drill composite incl. the 40"}


def figure_drill_reliability(prof: pl.DataFrame, path: Path) -> None:
    """Small multiples, one per drill family: reliability (y) vs between-session variance share (x) of every Combine
    signature; marker shape = how the reliability was measured; the family in blue over all others in gray;
    highlighted metrics have a line to their reliability within draft year (hollow marker)."""
    plt = _style()
    from matplotlib.lines import Line2D
    d = _profile_groups(prof)
    ps = profile_summary(prof)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.815, bottom=0.08, wspace=0.08, hspace=0.24)
    for ax, (fam, title) in zip(axes.flat, FAMILIES):
        ax.axhline(0.5, color=MUTED, lw=0.8, zorder=1)
        ax.text(0.695, 0.51, "0.5 screen", ha="right", va="bottom", fontsize=7, color=MUTED)
        for k, m in KIND_MARK.items():
            g = d.filter(pl.col("kind") == k)
            ax.scatter(g["share_session"], g["rel_round1_def"], s=14, marker=m, color=GRID, lw=0, zorder=2)
        f = d.filter(pl.col("probe") == fam)
        for k, m in KIND_MARK.items():
            g = f.filter(pl.col("kind") == k)
            ax.scatter(g["share_session"], g["rel_round1_def"], s=28, marker=m, color=BLUE, edgecolor=SURF, linewidth=0.9,
                       zorder=3, alpha=0.55)
        hl = [(r, *HIGHLIGHT[(r["probe"], r["position"], r["metric"], r["scope"])]) for r in f.iter_rows(named=True)
              if (r["probe"], r["position"], r["metric"], r["scope"]) in HIGHLIGHT]
        for r, _, _ in hl:
            x, y0, y1 = r["share_session"], r["rel_round1_def"], r["rel_year"]
            ax.plot([x, x], [y0, y1], color=INK2, lw=1, zorder=4)
            ax.plot([x], [y1], KIND_MARK[r["kind"]], ms=5.5, mfc=SURF, mec=INK2, mew=1.2, zorder=5)
            ax.plot([x], [y0], KIND_MARK[r["kind"]], ms=8, mfc=BLUE, mec=SURF, mew=1.5, zorder=6)
        _label_points(ax, [(r["share_session"], r["rel_round1_def"], lab, off) for r, lab, off in hl])
        ax.set_title(f"{title}  ·  {f.height} metrics", fontsize=9.5, loc="left", color=INK)
        ax.set_xlim(-0.02, 0.72)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(False)
        for yv in (0.25, 0.75):
            ax.axhline(yv, color=GRID, lw=0.6, zorder=0)
    wr = ps["wr_route_same_drill_retest"]
    axes.flat[0].text(0.985, 0.985, f"WR same-drill retests (single attempt,\n{min(wr['retest_n_player_drill_pairs'])}-"
                      f"{max(wr['retest_n_player_drill_pairs'])} player-drill pairs, same session):\nmedian ICC "
                      f"{wr['retest_icc_median']:.2f} vs "
                      f"cross-drill median {wr['composite_rel_median']:.2f}.\nRoute-drill signatures are largely "
                      "drill-specific.", transform=axes.flat[0].transAxes, ha="right", va="top", fontsize=7.2, color=INK2,
                      linespacing=1.3, bbox=dict(facecolor=SURF, edgecolor=GRID, boxstyle="round,pad=0.35"))
    for ax in axes[1]:
        ax.set_xlabel("between-session share (upper bound: includes group composition)", fontsize=8.6)
    for ax in axes[:, 0]:
        ax.set_ylabel("reliability of the player value")
    note = axes.flat[5]
    note.axis("off")
    note.text(0.02, 0.99, "How to read", fontsize=9.5, weight="bold", va="top", color=INK, transform=note.transAxes)
    hk = [Line2D([], [], color=INK2, lw=0, marker=m, ms=6.5, mfc=INK2 if k != "within-attempt" else INK2,
                 label=KIND_LABEL[k]) for k, m in KIND_MARK.items()]
    note.legend(handles=hk, loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False, fontsize=7.8,
                handlelength=1.2, borderaxespad=0, title="marker = how reliability was measured", title_fontsize=7.8,
                alignment="left")
    note.text(0.02, 0.60,
              "y: reliability as used in round 1 (Spearman-Brown;\n"
              "   0 when the ICC is negative).\n"
              "x: variance between Combine sessions (> 10 min apart)\n"
              "   as a share of within-drill variance. An upper bound\n"
              "   on session artifact: it also holds real differences\n"
              "   between groups tested together (interior vs edge DL,\n"
              "   40 run by position group). The OL wave drill is the\n"
              "   clear artifact (tempo drift 2.9 -> 2.2 yd/s, 2023-25).\n"
              "Labeled: features used in round-1 tests (+ contrasts).\n"
              "   Line to the hollow marker = reliability within draft\n"
              "   year (lower for most; WR turn retention rises).\n"
              "Gray: all other families. Counts include derivative\n"
              "variants of one construct.",
              fontsize=7.8, va="top", color=INK2, transform=note.transAxes, linespacing=1.33)
    bg = ps["by_group_kind"]
    rep, f40 = bg["position drill | repeat attempts"], ps["forty"]
    cx, wa = bg["position drill | cross-drill"], bg["position drill | within-attempt"]
    fm = ps["position_drill"]["cross_drill_median_by_family"].values()
    fig.text(0.06, 0.965, f"Drill reliability: {rep['n_ge_05']} of {rep['n']} repeated position-drill metrics reach 0.5; "
             f"all {f40['n']} tracked 40-yd metrics do", fontsize=11.5, weight="bold", color=INK)
    fig.text(0.06, 0.943, f"Position-drill signatures rarely agree across different drills: {cx['n_ge_05']} of {cx['n']} "
             f"cross-drill composites reach 0.5 (family medians {min(fm):.2f}-{max(fm):.2f}).\nOL mirror values "
             f"({wa['n_ge_05']} of {wa['n']} within-attempt metrics >= 0.5) come from a single attempt. The two DL drill "
             "composites that include the 40 are in neither count.\n"
             f"Tracked 40: top-speed metrics {f40['top_speed']['rel'][0]:.2f}-{f40['top_speed']['rel'][1]:.2f}, "
             f"start-phase metrics {f40['start_phase']['rel'][0]:.2f}-{f40['start_phase']['rel'][1]:.2f} (median of all "
             f"{f40['n']}: {f40['median_all']:.2f}; {f40['n_dl_starts']} are DL 40-start variants).",
             fontsize=8.4, color=INK2, va="top", linespacing=1.4)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _label_points(ax, pts):
    """Place labels beside points, stepping them vertically apart (in axis units) with a thin leader to the point;
    a label with an explicit (dx, dy) offset is placed there instead."""
    if not pts:
        return
    pts = sorted(pts, key=lambda p: -p[1])
    placed = []
    for x, y, lab, off in pts:
        if off is not None:
            lx, ly, right = x + off[0], y + off[1], off[0] >= 0
            placed.append(ly - 0.035)
        else:
            ly = y
            while any(abs(ly - q) < 0.09 for q in placed):
                ly -= 0.09
            placed.append(ly)
            right = x < 0.42
            lx, ly = x + (0.045 if right else -0.045), ly + 0.035
        ax.annotate(lab, (x, y), (lx, ly), fontsize=7.6, color=INK, ha="left" if right else "right",
                    va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=1, shrinkB=4,
                                                 relpos=(0, 0.5) if right else (1, 0.5)), zorder=7)


def make_figures() -> None:
    tm = pl.read_parquet(OUT_DATA / "e008_transfer_map.parquet")
    prof = pl.read_parquet(OUT_DATA / "e008_drill_reliability.parquet")
    lp = pl.read_parquet(OUT_DATA / "e008_link_players.parquet")
    figure_transfer_map(tm, prof, lp, OUT_FIG / "E008_transfer_map.png")
    figure_drill_reliability(prof, OUT_FIG / "E008_drill_reliability.png")
    log("figures written")


# =====================================================================================================
# 9. Main
# =====================================================================================================

def main() -> None:
    L, d5 = links()
    rep = check_reproduction(L)
    assert all(v["match"] for v in rep.values()), "round-1 n / raw r not reproduced"
    sess = session_table()
    specs, extras = profile_specs(sess, d5)
    prof = drill_profile(specs)
    e001_sh = e001_target_split_half(L)
    frel, trel = feature_reliability(prof, extras), target_reliability(e001_sh)
    e001_mean_rel = _icc_by_position(extras["smax40"].select("nfl_id", "position", "draft_year", "v"), False)
    log(f"drill profile: {prof.height} signatures")
    res = run_links(L)
    tm = transfer_map(L, res, frel, trel)
    post = e006_r1000(L)
    tmdf = pl.DataFrame(tm).with_columns(pl.col("excess_seeds").list.eval(pl.element().cast(pl.Float64)))
    tmdf.write_parquet(OUT_DATA / "e008_transfer_map.parquet")
    prof.write_parquet(OUT_DATA / "e008_drill_reliability.parquet")
    lp = link_players(L)
    lp.write_parquet(OUT_DATA / "e008_link_players.parquet")
    spec_out = {lk["id"]: {k: lk[k] for k in ("experiment", "test", "positions", "feature", "target", "expected_sign",
                                              "sign_source", "feature_source", "target_dagger", "exposure", "construct",
                                              "kind", "label", "outcome_target", "primary14")}
                | {"caveats": CAVEATS[lk["id"]] + ([ALL_2025] if lk["outcome_target"] else []),
                   "scout_sentence": SCOUT[lk["id"]]} for lk in L}
    out = {
        "experiment_id": "E008", "title": "Drill reliability + transfer map", "date": "2026-10-14",
        "quick_mode": QUICK, "protocol": "PLAN.md section 4 rev 2",
        "settings": {**EVAL, "seeds": "0 (+1, 2 when fl_p < 0.10)", "n_jobs": N_JOBS,
                     "controls_B2": CTRL + ["position dummies", "draft-year dummies"]},
        "test_labels": {
            "primary": "14 round-1 primaries (E002 P1-3, E003 P1-3, E004 P1-3, E005 P1-2, E006 P1-3), re-estimated; BH over these 14",
            "prespecified_sanity": "E001 WR/DB/DL/OL (round-1 sanity link, re-estimated; BH over all 18 reported)",
            "prespecified_robustness": "seeds 1-2 when fl_p < 0.10; without the 2025 class (all links); leave-3-out "
                                       "dropping the 3 most supportive players (go criterion 6)",
            "post_hoc": "leave-3-out dropping the 3 most contrary players (descriptive); E006-P1/P2 with the shared "
                        "AS_R = 1000 A-S estimator",
            "post_hoc_after_review": "beyond-B2 reliability correction made primary (as-used kept as an "
                                     "under-correction); joint Bonferroni ceilings over the 18 links; one-sided MDE; "
                                     "two-sided Freedman-Lane p (descriptive, all links); BH across all ~90 round-1 tests "
                                     "for the passing link; top-3 share of the partial r; E001 feature reliability "
                                     "for a max over attempts and E001 target split-half on the link samples "
                                     "(replacing E006's differently defined values)",
        },
        "summary": {"transfer": transfer_summary(tm), "multiplicity": multiplicity(tm), "wr_speed_signal": wr_speed_signal(lp),
                    "drill_profile": profile_summary(prof),
                    "primary_reliability_correction": "beyond-B2 (dis_b2_*, ceiling_dis_b2); as-used (dis_*, "
                                                      "ceiling_dis) under-corrects a partial r"},
        "reproduction_round1": rep,
        "links": spec_out,
        "transfer_map": tmdf.to_dicts(),
        "link_results": {k: {kk: vv for kk, vv in v.items()} for k, v in res.items()},
        "feature_reliability_by_position": {k: {p: {"as_used": a, "within_draft_year": b} for p, (a, b) in v.items()}
                                            for k, v in frel.items()},
        "target_reliability_by_position": trel,
        "e001_reliability_detail": {
            "feature_max_of_attempts": {p: frel[f"E001-{p}"][p][0] for p in ("WR", "DB", "DL", "OL")},
            "feature_mean_of_attempts_round1_definition": e001_mean_rel,
            "target_split_half_link_samples": e001_sh,
            "target_split_half_round1_e006_script": target_reliability_round1_e001(),
        },
        "disattenuation_assumptions": [
            "Classical measurement error: feature and target errors independent of each other, of the true scores and of "
            "the B2 controls. Violated if attempt-level noise in a tracked 40 metric is shared with the official 40 time "
            "(same attempts).",
            "Reliabilities are estimates with their own sampling error (n ~ 60-120 players); dividing by a noisy small "
            "reliability inflates the bound, so intervals are clipped at +-1 and left unbounded when a reliability <= 0.05.",
            "The partial correlation uses the parts of x and y left after B2, so the primary correction uses their "
            "reliabilities ('b2'): player error variance (1 - rel) Var(v) projected through the B2 residual maker. It is "
            "lower (wider bound) when B2 or the year dummies explain much of the true score (e.g. 40 speed vs the 40 "
            "time). The as-used rel_x / rel_y version is reported as a secondary under-correction.",
            "Corrects for error in x and y only, not for error in the B2 controls (a noisy 40 time leaves residual "
            "confounding that this does not address).",
            "Targets are EB-shrunk player means; their reliability is taken from the split-half of raw per-play values.",
        ],
        "post_hoc_e006_r1000": post,
        "drill_profile": prof.to_dicts(),
    }
    (OUT_RES / "E008.json").write_text(json.dumps(_clean(out), indent=1))
    log(f"wrote {OUT_RES / 'E008.json'}")
    make_figures()


if __name__ == "__main__":
    if FIGURES_ONLY:
        make_figures()
    else:
        main()
