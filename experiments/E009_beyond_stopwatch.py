"""E009 - Beyond the stopwatch: the 18 transfer-map links against B1 (traditional Combine only, no draft slot).

Implements experiments/results/E009_prereg.json. The prereg labels this evidence SECONDARY and NOT BLIND: E008's
beyond-B2 results and the raw correlations were known, and B1 is nested in B2 (B2 = B1 + draft pick), so the
results are partly predictable. Read them as a second column of the transfer map ("information beyond the
stopwatch"), never as "market inefficiency".
  1. The 18 E008 links (E002 P1-3, E003 P1-3, E004 P1-3, E005 P1-2, E006 P1-3, E001 WR/DB/DL/OL), built by importing
     E008's link builders unchanged (E008_transfer_map.links): same features, targets, populations, expected
     signs. Checks: n and raw r vs round 1, and the beyond-B2 partial r vs E008.json.
  2. Primary (the E009 family): linkage.evaluate(draft_controls=[]) under protocol rev 3. Baseline B1 = position +
     draft-year dummies + linkage.COMBINE_CONTROLS; one-sided Freedman-Lane (10,000 perms), 50x5 CV, 2,000 CV
     perms, seed 0, seeds 1-2 when fl_p < 0.10. BH across the 18. Go criteria 1-6 (rev 3) with B1 for B2.
     NOTE: with draft_controls=[], evaluate's keys named "B2" (delta_r2_vs_B2, cv_r2["B2"]) are the B1 baseline.
  3. Prespecified extras: without the 2025 class (all 18), leave-3-out for links passing criterion 1, and the
     difference in partial r between B1 and B2 (how much the draft slot absorbs).
  4. Post hoc (labeled as such in the JSON): a paired-bootstrap interval for that difference and its
     decomposition (which side of the link the draft slot explains); E008's bound definitions for B1
     (beyond-B1 reliability-corrected interval, joint Bonferroni ceilings, one-sided MDE); two-sided p
     (descriptive); BH across the 36 B1 + B2 tests for links passing criterion 1; E008's beyond-B2 tests
     re-judged under rev 3 (inference identical, asserted; CV criteria 2-3 recomputed) so both columns of the
     figure use one rule.
  4b. Post hoc after the round-3 review (labeled post_hoc_after_review in the JSON; no prereg test or definition
     changes): (i) Monte Carlo precision of criterion 4: the Freedman-Lane p at 2,000,000 permutations for links with
     p < 0.10, the MC SE of q, and q over other families (seeds 1-2, E008's 14 primaries, round-3 primaries E009 +
     E010, all round-3 tests); (ii) criterion 3 on a robust null scale (median + MAD-SD, null 84th percentile), from an
     exact, asserted replay of the harness's LOCO null; (iii) a delete-one jackknife interval for the draft-slot
     difference next to the bootstrap; (iv) null ranges signed to the expected direction.
  5. Figure: experiments/figures/E009_transfer_map_two_baselines.png.

Run: .venv/bin/python experiments/E009_beyond_stopwatch.py                  (everything: 1.5-3 h depending on machine load, n_jobs=3, peak ~0.7 GB)
     .venv/bin/python experiments/E009_beyond_stopwatch.py --quick          (small counts, scratch output)
     .venv/bin/python experiments/E009_beyond_stopwatch.py --figures-only   (redraw from the saved outputs)
Outputs: experiments/results/E009.json, experiments/figures/E009_transfer_map_two_baselines.png,
         data/processed/e009_{transfer_map,link_players}.parquet.
"""

import json
import sys
import time
from pathlib import Path

import hashlib

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from scipy import stats
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import E008_transfer_map as E8  # noqa: E402  (link builders, bound helpers and figure style, used unchanged)
from bdb27 import io, linkage  # noqa: E402

RES, FIG = ROOT / "experiments" / "results", ROOT / "experiments" / "figures"
QUICK = "--quick" in sys.argv
FIGURES_ONLY = "--figures-only" in sys.argv
SCRATCH = Path("/private/tmp/claude-501/-Users-jasonli-Documents-GitHub-NFL-Big-Data-Bowl-2027/"
               "ca6dc0b9-7e4f-45b4-b4e4-1468819c8969/scratchpad/round3/E009")
OUT_RES, OUT_FIG, OUT_DATA = (SCRATCH, SCRATCH, SCRATCH) if QUICK else (RES, FIG, io.PROCESSED)
N_JOBS = 3
EVAL = dict(n_repeats=50, n_perm=2000, n_perm_fl=10000) if not QUICK else dict(n_repeats=5, n_perm=50, n_perm_fl=500)
N_BOOT = 2000 if not QUICK else 100
BOOT_SEED = 9
SEED_RULE_P = 0.10  # seeds 1-2 when the seed-0 one-sided p is below this (prereg)
B1_CTRL = list(linkage.COMBINE_CONTROLS)
B2_CTRL = list(linkage.COMBINE_CONTROLS) + list(linkage.DRAFT_CONTROLS)
Z80, Z95_1, Z95_2 = float(norm.ppf(0.80)), float(norm.ppf(0.95)), float(norm.ppf(0.975))
N_PERM_PRECISE = 2_000_000 if not QUICK else 20_000  # post hoc: Freedman-Lane precision for links with p < 0.10
Q84 = float(norm.cdf(1.0))  # post hoc robust LOCO scale: the null's 84.13th percentile (= mean + 1 SD if normal)
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


# =====================================================================================================
# 1. Harness calls (rev 3)
# =====================================================================================================

def _eval(lk: dict, base: str, df: pl.DataFrame | None = None, seed: int = 0, **kw) -> dict:
    """linkage.evaluate against B1 (draft_controls=[]) or B2 (the default draft controls)."""
    dc = [] if base == "B1" else list(linkage.DRAFT_CONTROLS)
    return linkage.evaluate(lk["df"] if df is None else df, lk["feature"], lk["target"],
                            expected_sign=lk["expected_sign"], draft_controls=dc, seed=seed, n_jobs=N_JOBS,
                            **dict(EVAL, **kw))


def _loco_n(r: dict) -> int:
    return int(r["checks"]["loco_classes_pass"].split("/")[0])


def seed_ok(r: dict, loco_n: int | None = None) -> bool:
    """Criteria 1-3 under rev 3 for one seed (`loco_n` overrides the harness's LOCO count: post hoc robust scale)."""
    c = r["checks"]
    return bool(c["fl_p_lt_05"] and c["sign_matches"] and c["excess_lb_gt_0"]
                and (_loco_n(r) if loco_n is None else loco_n) >= 2)


def _resid(d: pl.DataFrame, ctrl: list[str], cols: list[str]) -> tuple[np.ndarray, dict]:
    """Residual maker of the harness design (position + year dummies, imputed `ctrl`) and residuals of `cols`."""
    R = linkage._residual_maker(linkage._imputed(linkage._design(d, ctrl)))
    return R, {c: R @ d[c].to_numpy().astype(float) for c in cols}


def leave3(lk: dict, base: str) -> dict:
    """Leave-3-out (criterion 6): drop the 3 players contributing most to the partial r in the expected direction
    (sign * x_res * y_res under this baseline) and re-run the full harness."""
    d = E8._sample(lk)
    _, res = _resid(d, B1_CTRL if base == "B1" else B2_CTRL, [lk["feature"], lk["target"]])
    xr, yr = res[lk["feature"]], res[lk["target"]]
    c = lk["expected_sign"] * xr * yr
    contrib = c / np.sqrt((xr @ xr) * (yr @ yr))
    idx = np.argsort(-c)[:3]
    ids = d["nfl_id"].to_numpy()[idx].tolist()
    r = _eval(lk, base, d.filter(~pl.col("nfl_id").is_in(ids)))
    return {"dropped_nfl_id": ids, "dropped_names": d["display_name"].to_numpy()[idx].tolist(),
            "top3_supportive_share_of_partial_r": float(np.sort(contrib)[::-1][:3].sum() / contrib.sum()),
            "n": r["n"], "partial_r": r["partial_r"], "partial_r_ci95": r["partial_r_ci95"], "fl_p": r["fl_p"],
            "passes_c1": bool(r["checks"]["fl_p_lt_05"] and r["checks"]["sign_matches"]),
            "delta_r2_vs_baseline": r["delta_r2_vs_B2"], "loco": r["loco"], "checks": r["checks"]}


def run_b1(L: list[dict]) -> dict:
    """Primary family (beyond B1) and its prespecified robustness."""
    res = {}
    for lk in L:
        t = time.time()
        r = {"seed0": _eval(lk, "B1")}
        s0 = r["seed0"]
        if s0["fl_p"] is not None and s0["fl_p"] < SEED_RULE_P:
            r["seed1"], r["seed2"] = _eval(lk, "B1", seed=1), _eval(lk, "B1", seed=2)
        r["without_2025"] = _eval(lk, "B1", lk["df"].filter(pl.col("draft_year") != 2025))
        c1 = bool(s0["checks"]["fl_p_lt_05"] and s0["checks"]["sign_matches"])
        r["leave3"] = leave3(lk, "B1") if c1 else None
        # two-sided Freedman-Lane p (same permutation stream; CV parts at minimal settings, not used): descriptive
        r["two_sided"] = {k: v for k, v in linkage.evaluate(
            lk["df"], lk["feature"], lk["target"], expected_sign=None, draft_controls=[], seed=0, n_jobs=N_JOBS,
            n_repeats=2, n_perm=20, n_perm_fl=EVAL["n_perm_fl"]).items()
            if k in ("n", "partial_r", "fl_p", "fl_one_sided", "n_perm_fl")}
        log(f"B1 {lk['id']:8s} n={s0['n']:3d} partial={s0['partial_r']:+.3f} "
            f"[{s0['partial_r_ci95'][0]:+.2f},{s0['partial_r_ci95'][1]:+.2f}] fl_p={s0['fl_p']:.4f} "
            f"excess={s0['delta_r2_vs_B2']['excess']:+.4f}±{s0['delta_r2_vs_B2']['se']:.4f} "
            f"loco={s0['checks']['loco_classes_pass']} | no2025 r={r['without_2025']['partial_r']:+.3f} "
            f"p={r['without_2025']['fl_p']:.3f}" + (f" | drop3 p={r['leave3']['fl_p']:.4f}" if c1 else "")
            + f" ({time.time() - t:.0f}s)")
        res[lk["id"]] = r
    return res


# Keys of evaluate() whose values do not depend on the rev-2 -> rev-3 change (inference and CV estimates).
_SAME_KEYS = ("n", "resid_df", "raw_r", "partial_r", "partial_r_ci95", "fl_p", "by_class", "cv_r2", "delta_r2_vs_B0",
              "delta_r2_vs_B2", "cv_perm_p")


def _max_diff(a, b) -> float:
    """Largest absolute difference between two nested JSON-like structures (inf if their shapes differ)."""
    if isinstance(a, dict):
        if set(map(str, a)) != set(map(str, b)):
            return float("inf")
        bb = {str(k): v for k, v in b.items()}
        return max([_max_diff(v, bb[str(k)]) for k, v in a.items()] + [0.0])
    if isinstance(a, (list, tuple)):
        return float("inf") if len(a) != len(b) else max([_max_diff(u, v) for u, v in zip(a, b)] + [0.0])
    if a is None or b is None:
        return 0.0 if a is None and b is None else float("inf")
    return abs(float(a) - float(b))


def run_b2_rev3(L: list[dict], e8: dict) -> tuple[dict, dict]:
    """Post hoc: E008's beyond-B2 tests re-run with the rev-3 harness (seed 0; seeds 1-2 where E008 ran them), to
    re-judge criteria 2, 3 and 6 under rev 3. Asserts that everything rev 3 did not change reproduces E008.json."""
    res, diffs = {}, {}
    for lk in L:
        lr = e8["link_results"][lk["id"]]
        r = {k: _eval(lk, "B2", seed=s) for k, s in (("seed0", 0), ("seed1", 1), ("seed2", 2)) if k in lr}
        dmax = 0.0
        for k, v in r.items():
            ref = _clean(lr[k])
            got = _clean(v)
            dmax = max([dmax] + [_max_diff(got[kk], ref[kk]) for kk in _SAME_KEYS]
                       + [_max_diff({y: {"delta": u["delta"], "excess": u["excess"]} for y, u in got["loco"].items()},
                                    {y: {"delta": u["delta"], "excess": u["excess"]} for y, u in ref["loco"].items()})])
        diffs[lk["id"]] = dmax
        if not QUICK:
            assert dmax < 1e-12, f"{lk['id']}: rev-3 rerun differs from E008.json by {dmax}"
        s0 = r["seed0"]
        log(f"B2(rev3) {lk['id']:8s} partial={s0['partial_r']:+.3f} fl_p={s0['fl_p']:.4f} "
            f"c2={s0['checks']['excess_lb_gt_0']} loco_rev3={s0['checks']['loco_classes_pass']} "
            f"(rev2 {s0['checks']['loco_excess_positive_classes']}) max|diff vs E008|={dmax:.1e}")
        res[lk["id"]] = r
    return res, diffs


# =====================================================================================================
# 2. How much of each partial r the draft slot absorbs (prespecified quantity; bootstrap + decomposition post hoc)
# =====================================================================================================

def _pr(X_raw: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """In-sample partial r of x and y given the (imputed) design, as in linkage.evaluate."""
    Xi = linkage._imputed(X_raw)
    v = np.column_stack([x, y])
    e = v - Xi @ np.linalg.lstsq(Xi, v, rcond=None)[0]
    return linkage._corr(e[:, 0], e[:, 1])


def draft_absorption(L: list[dict]) -> dict:
    """delta = sign * (partial r beyond B1 - partial r beyond B2): > 0 means adding the draft slot shrinks the link in
    the expected direction (the draft slot absorbs part of it); < 0 means it strengthens it.
    Decomposition (Frisch-Waugh-Lovell; B2 = B1 + D, D = log pick and undrafted): with x1, y1 the B1 residuals and P the
    projection onto D's B1 residuals,
        r_B2 = (r_B1 - path) / sqrt((1 - R2_x)(1 - R2_y)),
    path = x1'P y1 / (|x1| |y1|) (the part of r_B1 that runs through the draft slot), R2_x and R2_y = shares of the
    feature's and target's beyond-B1 variance that the draft slot explains. The interval is a paired percentile
    bootstrap over players (designs re-imputed per resample).
    Post hoc after review: a delete-one jackknife SE and normal 95% interval next to the bootstrap. Resampling with
    duplicated rows in ~20-column designs at n ~ 65 inflates the bootstrap spread of a nested-model difference (up to
    ~2x the jackknife SE where the draft slot explains almost nothing), so the jackknife is the tighter reading and the
    percentile bootstrap the conservative one. The jackknife draws no random numbers (bootstrap stream unchanged)."""
    rng = np.random.default_rng(BOOT_SEED)
    out = {}
    for lk in L:
        d = E8._sample(lk)
        sg, n = lk["expected_sign"], d.height
        x, y = d[lk["feature"]].to_numpy().astype(float), d[lk["target"]].to_numpy().astype(float)
        X1, X2 = linkage._design(d, B1_CTRL), linkage._design(d, B2_CTRL)
        r1, r2 = _pr(X1, x, y), _pr(X2, x, y)
        R1 = linkage._residual_maker(linkage._imputed(X1))
        x1, y1 = R1 @ x, R1 @ y
        D1 = R1 @ d.select(linkage.DRAFT_CONTROLS).cast(pl.Float64).to_numpy()
        P = D1 @ np.linalg.pinv(D1)
        nx, ny = np.sqrt(x1 @ x1), np.sqrt(y1 @ y1)
        r2x, r2y, path = (x1 @ P @ x1) / nx ** 2, (y1 @ P @ y1) / ny ** 2, (x1 @ P @ y1) / (nx * ny)
        r2_fwl = (r1 - path) / np.sqrt((1 - r2x) * (1 - r2y))
        early = -(R1 @ d["log_pick"].to_numpy().astype(float))  # beyond-B1 'drafted earlier' (log pick only)
        boot = np.empty((N_BOOT, 2))
        for b in range(N_BOOT):
            i = rng.integers(0, n, n)
            boot[b] = _pr(X1[i], x[i], y[i]), _pr(X2[i], x[i], y[i])
        db = sg * (boot[:, 0] - boot[:, 1])
        delta = float(sg * (r1 - r2))
        keep = np.ones(n, dtype=bool)
        jk = np.empty(n)
        for i in range(n):  # delete-one jackknife (post hoc)
            keep[i] = False
            jk[i] = sg * (_pr(X1[keep], x[keep], y[keep]) - _pr(X2[keep], x[keep], y[keep]))
            keep[i] = True
        se_jk = float(np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2)))
        out[lk["id"]] = {
            "partial_r_b1": float(r1), "partial_r_b2": float(r2), "delta_signed": delta,
            "delta_boot_ci95": [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))],
            "delta_boot_se": float(np.std(db, ddof=1)), "delta_boot_share_gt0": float(np.mean(db > 0)),
            "delta_jack_se": se_jk, "delta_jack_ci95": [delta - Z95_2 * se_jk, delta + Z95_2 * se_jk],
            "delta_boot_se_over_jack_se": float(np.std(db, ddof=1) / se_jk) if se_jk > 0 else None,
            "path_signed": float(sg * path), "r2_feature_on_draft": float(r2x), "r2_target_on_draft": float(r2y),
            "r_feature_resid_earlier_pick": linkage._corr(x1, early),
            "r_target_resid_earlier_pick": linkage._corr(y1, early),
            "fwl_identity_abs_err": float(abs(r2_fwl - r2)),
            "boot_partial_r_b1_ci95": [float(np.percentile(boot[:, 0], 2.5)), float(np.percentile(boot[:, 0], 97.5))],
            "boot_partial_r_b2_ci95": [float(np.percentile(boot[:, 1], 2.5)), float(np.percentile(boot[:, 1], 97.5))],
        }
        a = out[lk["id"]]
        log(f"draft slot {lk['id']:8s} r_B1 {r1:+.3f} r_B2 {r2:+.3f} delta {delta:+.3f} "
            f"boot [{a['delta_boot_ci95'][0]:+.3f},{a['delta_boot_ci95'][1]:+.3f}] "
            f"jack [{a['delta_jack_ci95'][0]:+.3f},{a['delta_jack_ci95'][1]:+.3f}] R2x {r2x:.2f} R2y {r2y:.2f} "
            f"path {sg * path:+.3f} fwl_err {a['fwl_identity_abs_err']:.1e}")
    return out


# =====================================================================================================
# 2b. Post hoc after the round-3 review: criterion 4's Monte Carlo precision and family, criterion 3's null scale
# =====================================================================================================

def precise_fl(L: list[dict], b1: dict, e8: dict) -> dict:
    """The one-sided Freedman-Lane p at N_PERM_PRECISE permutations for every link with seed-0 p < 0.10 under either
    baseline, through the harness (linkage.evaluate, seed 0: its first 10,000 permutations are the prereg's; the CV
    parts run at minimal settings and are not used), with its Monte Carlo SE and the parametric t p for reference."""
    e8r = {r["id"]: r for r in e8["transfer_map"]}
    out = {}
    for lk in L:
        lid = lk["id"]
        for base, p0, r_ref in (("B1", b1[lid]["seed0"]["fl_p"], b1[lid]["seed0"]["partial_r"]),
                                ("B2", e8r[lid]["fl_p"], e8r[lid]["partial_r"])):
            if p0 >= SEED_RULE_P:
                continue
            t = time.time()
            r = linkage.evaluate(lk["df"], lk["feature"], lk["target"], expected_sign=lk["expected_sign"],
                                 draft_controls=[] if base == "B1" else list(linkage.DRAFT_CONTROLS), seed=0,
                                 n_jobs=N_JOBS, n_repeats=2, n_perm=20, n_perm_fl=N_PERM_PRECISE)
            assert abs(r["partial_r"] - r_ref) < 1e-12, (lid, base)
            p, dfr = r["fl_p"], r["resid_df"]
            tstat = lk["expected_sign"] * r["partial_r"] * np.sqrt(dfr) / np.sqrt(1 - r["partial_r"] ** 2)
            out.setdefault(lid, {})[base] = {
                "n_perm_fl": N_PERM_PRECISE, "fl_p": p, "fl_p_mc_se": float(np.sqrt(p * (1 - p) / N_PERM_PRECISE)),
                "fl_p_10k_seed0": p0, "fl_p_10k_mc_se": float(np.sqrt(p0 * (1 - p0) / EVAL["n_perm_fl"])),
                "partial_r": r["partial_r"], "resid_df": dfr, "p_parametric_t_one_sided": float(stats.t.sf(tstat, dfr))}
            log(f"precise FL {lid:8s} {base}: p {p:.5f} (MC SE {out[lid][base]['fl_p_mc_se']:.1e}; 10k seed 0 {p0:.5f}; "
                f"t {out[lid][base]['p_parametric_t_one_sided']:.5f}) ({time.time() - t:.0f}s)")
    return out


def round3_inputs() -> dict:
    """The other round-3 experiments' tests, read from their results JSONs (E010: game.tests with a p; E011: its
    test_count). Recorded with a hash, since E010.json may be revised after this run."""
    raw = (RES / "E010.json").read_bytes()
    e10 = json.loads(raw)
    tests = [t for t in e10["game"]["tests"] if t.get("fl_p") is not None]
    e11 = json.loads((RES / "E011.json").read_text()).get("test_count", {})
    n11 = sum(v for v in e11.values() if isinstance(v, (int, float)))
    if n11:
        log(f"WARNING: E011 lists {n11} tests; they have no p in a machine-readable place and are not in the round-3 BH")
    return {"e010_sha256": hashlib.sha256(raw).hexdigest(),
            "e010_primaries": {t["id"]: t["fl_p"] for t in tests if t["label"] == "primary"},
            "e010_all_tests_with_p": {t["id"]: t["fl_p"] for t in tests},
            "e010_labels": {t["id"]: t["label"] for t in tests},
            "e011_test_count": e11, "e011_tests_with_p_included": 0}


def multiplicity_posthoc(L: list[dict], b1: dict, e8: dict, prec: dict, r3: dict) -> dict:
    """Criterion 4 for the links run at high precision: q under the prereg computation (10k permutations, seed 0),
    at seeds 1-2, at N_PERM_PRECISE (and at +-2 MC SE), over E008's family (14 round-1 primaries; E001 over 18), over
    the round-3 primaries (E009's 18 + E010's primaries; E011 has no tests) and over all round-3 tests with a p on a
    distinct hypothesis or sample (E009: 18 beyond B1, 18 without 2025, leave-3-out; E010: all its tests with a p).
    c4 status = 'pass' / 'fail' if every criterion-4 variant agrees (all except the all-round-3-tests family, which
    is reported only), else 'marginal'. The prereg's c4 is not changed."""
    ids = [lk["id"] for lk in L]
    e8r = {r["id"]: r for r in e8["transfer_map"]}
    p1 = [b1[i]["seed0"]["fl_p"] for i in ids]
    p2 = [e8r[i]["fl_p"] for i in ids]
    prim14 = [i for i in ids if e8r[i]["primary14"]]
    e9_all = ({f"E009/{i}": b1[i]["seed0"]["fl_p"] for i in ids}
              | {f"E009/{i}/without_2025": b1[i]["without_2025"]["fl_p"] for i in ids}
              | {f"E009/{i}/leave3": b1[i]["leave3"]["fl_p"] for i in ids if b1[i]["leave3"]})
    r3_all = e9_all | {f"E010/{k}": v for k, v in r3["e010_all_tests_with_p"].items()}
    r3_prim = list(r3["e010_primaries"].values())

    def q_in(ps: list[float], i: int) -> float:
        return float(E8.bh(np.asarray(ps, float))[i])

    def with_p(ps: list[float], i: int, p: float) -> list[float]:
        v = list(ps)
        v[i] = p
        return v

    out = {"definitions": "q = BH q of this link's test in the named family, with this link's p replaced as stated "
                          "and every other p as in the prereg run (10k permutations, seed 0)",
           "families": {"q18": "E009 prereg family (18 links beyond B1)",
                        "q_e008_family": "E008's family: the 14 round-1 primaries (E001 links over all 18)",
                        "q_round3_primaries": f"E009's 18 + E010's {len(r3_prim)} primaries (E011: no tests)",
                        "q_round3_all_tests": f"{len(r3_all)} round-3 tests with a p ({len(e9_all)} E009 + "
                                              f"{len(r3['e010_all_tests_with_p'])} E010; E011 none)",
                        "q_all_round1_tests": "E008's BH across all ~90 round-1 tests (beyond B2 only)"},
           "m_round3_all_tests": len(r3_all), "links": {}}
    for lid, pr in prec.items():
        i = ids.index(lid)
        res = {}
        if "B1" in pr:
            pp, se = pr["B1"]["fl_p"], pr["B1"]["fl_p_mc_se"]
            v = {f"q18_10k_{k}": q_in(with_p(p1, i, b1[lid][k]["fl_p"]), i)
                 for k in ("seed0", "seed1", "seed2") if k in b1[lid]}
            v["q18_precise"] = q_in(with_p(p1, i, pp), i)
            fam14 = prim14 if lid in prim14 else ids
            j = fam14.index(lid)
            pf = [p1[ids.index(k)] for k in fam14]
            v["q_e008_family_10k_seed0"] = q_in(pf, j)
            v["q_e008_family_precise"] = q_in(with_p(pf, j, pp), j)
            v["q_round3_primaries_10k_seed0"] = q_in(p1 + r3_prim, i)
            v["q_round3_primaries_precise"] = q_in(with_p(p1, i, pp) + r3_prim, i)
            keys = list(r3_all)
            k0 = keys.index(f"E009/{lid}")
            v["q_round3_all_tests_10k_seed0"] = q_in(list(r3_all.values()), k0)
            v["q_round3_all_tests_precise"] = q_in(with_p(list(r3_all.values()), k0, pp), k0)
            band = [q_in(with_p(p1, i, max(pp - 2 * se, 0.0)), i), q_in(with_p(p1, i, pp + 2 * se), i)]
            passes = {k: bool(q < 0.05) for k, q in v.items()}
            passes["q18_precise_plus_2mc_se"] = bool(band[1] < 0.05)
            # status over the criterion-4 families only (all round-3 tests is reported, as PLAN section 4 asks, but
            # criterion 4 is defined over primaries)
            c4v = [ok for k, ok in passes.items() if not k.startswith("q_round3_all_tests")]
            status = "pass" if all(c4v) else "fail" if not any(c4v) else "marginal"
            res["B1"] = {**v, "q18_precise_pm2_mc_se": band, "q18_precise_mc_se": float((band[1] - band[0]) / 4),
                         "passes_005": passes, "c4_prereg": bool(v["q18_10k_seed0"] < 0.05), "c4_status": status}
        if "B2" in pr:
            pp = pr["B2"]["fl_p"]
            fam = prim14 if lid in prim14 else ids
            j = fam.index(lid)
            pf = [p2[ids.index(k)] for k in fam]
            ps = np.sort([pp] + E8.ROUND1_SMALLEST_OTHER_P)
            v = {"q_e008_10k_seed0": e8r[lid]["bh_q14"] if lid in prim14 else e8r[lid]["bh_q18"],
                 "q_e008_precise": q_in(with_p(pf, j, pp), j)}
            v["q_all_round1_tests_e008"] = (e8["summary"]["multiplicity"].get(lid, {}).get("bh_q_all_round1_tests"))
            v["q_all_round1_tests_precise"] = (float(min(1.0, np.min(ps * E8.N_ROUND1_TESTS / np.arange(1, len(ps) + 1))))
                                               if pp <= min(E8.ROUND1_SMALLEST_OTHER_P) else None)
            res["B2"] = {**v, "passes_005": {k: bool(q < 0.05) for k, q in v.items() if q is not None}}
        out["links"][lid] = res
        log(f"c4 {lid}: " + "; ".join(f"{b}: " + ", ".join(f"{k} {q:.4f}" for k, q in d.items()
                                                            if isinstance(q, float)) for b, d in res.items()))
    return out


def _loco_replay(lk: dict, base: str, r: dict) -> dict:
    """Replays the harness's leave-one-class-out null exactly: same sample and design, same RNG stream (seed; the
    n_perm_fl Freedman-Lane permutations come first, then the CV permutations, of which LOCO uses the first
    min(n_perm, 300)), same ridge fits. Asserts it reproduces the stored delta, excess and null SD per class, then
    judges each class on two robust null scales (class partial r not reversed, as in rev 3):
      median_mad: held-out gain - null median > 1.4826 x null MAD;  q84: held-out gain > the null's 84.13th percentile."""
    d = E8._sample(lk)
    sg = lk["expected_sign"]
    x, y = d[lk["feature"]].to_numpy().astype(float), d[lk["target"]].to_numpy().astype(float)
    pos, years = d["position"].to_numpy(), d["draft_year"].to_numpy()
    var_y = float(np.var(y))
    D = linkage._design(d, B1_CTRL if base == "B1" else B2_CTRL)
    R = linkage._residual_maker(linkage._imputed(D))
    rx = R @ x
    x_hat = x - rx
    rng = np.random.default_rng(r["seed_used"])
    for _ in range(r["n_perm_fl"]):
        linkage._within_perm(rng, pos)
    perms = [x_hat + rx[linkage._within_perm(rng, pos)] for _ in range(min(r["n_perm"], 300))]
    full = np.column_stack([D, x])
    classes, n_h, n_m, n_q = {}, 0, 0, 0
    for yr in np.unique(years):
        tr, te = np.flatnonzero(years != yr), np.flatnonzero(years == yr)
        b = linkage._fold_r2(D, y, [(tr, te)], var_y)[0]
        obs = linkage._fold_r2(full, y, [(tr, te)], var_y)[0] - b
        nl = np.array(Parallel(n_jobs=N_JOBS)(delayed(
            lambda xp: linkage._fold_r2(np.column_stack([D, xp]), y, [(tr, te)], var_y)[0] - b)(xp) for xp in perms))
        st = r["loco"][int(yr)] if int(yr) in r["loco"] else r["loco"][str(int(yr))]
        err = max(abs(obs - st["delta"]), abs(obs - nl.mean() - st["excess"]), abs(nl.std() - st["null_sd"]))
        assert err < 1e-9, f"{lk['id']} {base} seed {r['seed_used']} {yr}: LOCO replay differs by {err}"
        bc = r["by_class"][int(yr)] if int(yr) in r["by_class"] else r["by_class"][str(int(yr))]
        wrong = bc["partial_r"] is not None and np.sign(bc["partial_r"]) != sg
        med = float(np.median(nl))
        mad_sd = float(1.4826 * np.median(np.abs(nl - med)))
        q84 = float(np.quantile(nl, Q84))
        h, m, q = (bool(obs - nl.mean() > nl.std() and not wrong), bool(obs - med > mad_sd and not wrong),
                   bool(obs > q84 and not wrong))
        n_h, n_m, n_q = n_h + h, n_m + m, n_q + q
        classes[int(yr)] = {"obs": float(obs), "null_mean": float(nl.mean()), "null_sd": float(nl.std()),
                            "null_median": med, "null_mad_sd": mad_sd, "null_q84": q84, "null_min": float(nl.min()),
                            "null_kurtosis": float(stats.kurtosis(nl)), "class_partial_r": bc["partial_r"],
                            "pass_harness": h, "pass_median_mad": m, "pass_q84": q, "replay_max_abs_err": float(err)}
    assert f"{n_h}/{len(classes)}" == r["checks"]["loco_classes_pass"], (lk["id"], base, r["seed_used"])
    return {"harness": n_h, "median_mad": n_m, "q84": n_q, "classes": classes}


def robust_loco(L: list[dict], b1: dict, b2: dict) -> dict:
    """Criterion 3 on robust null scales for every harness run behind a criterion (beyond B1 and beyond B2, every seed
    run). The harness's per-class threshold is the SD of 300 null permutations, which a few degenerate ridge fits
    (held-out dR2 down to ~ -1.7) can inflate several-fold."""
    out = {}
    for lk in L:
        t = time.time()
        for base, res in (("B1", b1[lk["id"]]), ("B2", b2[lk["id"]])):
            for k in ("seed0", "seed1", "seed2"):
                if k in res:
                    out.setdefault(lk["id"], {}).setdefault(base, {})[k] = _loco_replay(lk, base, {**res[k], "seed_used": int(k[-1])})
        o = out[lk["id"]]
        log(f"robust LOCO {lk['id']:8s} " + " | ".join(
            f"{b} {s}: harness {v['harness']}/3 mad {v['median_mad']}/3 q84 {v['q84']}/3"
            for b in o for s, v in o[b].items()) + f" ({time.time() - t:.0f}s)")
    return out


# =====================================================================================================
# 3. Transfer-map rows: B1 (primary family) next to B2 (E008 estimates, criteria re-judged under rev 3)
# =====================================================================================================

def _crit(s0: dict, seeds: list[dict], q: float, rel_ok: bool, drop3_c1: bool | None,
          loco_ns: list[int] | None = None) -> dict:
    """Go criteria 1-6 (rev 3). Keys as in E008; c3 = checks.loco_classes_pass (or, post hoc, `loco_ns`: LOCO counts
    per seed on a robust null scale, in the order of `seeds`)."""
    ln = [None] * len(seeds) if loco_ns is None else loco_ns
    return {
        "c1_fl_p_and_sign": bool(s0["checks"]["fl_p_lt_05"] and s0["checks"]["sign_matches"]),
        "c2_excess_lb_gt_0": bool(s0["checks"]["excess_lb_gt_0"]),
        "c3_loco_2of3": (_loco_n(s0) if ln[0] is None else ln[0]) >= 2,
        "c4_bh_q_lt_05": bool(q < 0.05),
        "c5_reliability_ge_05": bool(rel_ok),
        "c6_seeds_and_leave3out": bool(len(seeds) == 3 and all(seed_ok(x, k) for x, k in zip(seeds, ln))
                                       and bool(drop3_c1)),
    }


def _verdict(crit: dict, seeds: list[dict], drop3_c1: bool | None) -> str:
    """Short verdict listing every failed criterion; for criterion 6 both of its parts (seeds, leave-3-out) are named
    when both fail (E008's _verdict named only one)."""
    if all(crit.values()):
        return "passes 1-6; see caveats"
    if not crit["c1_fl_p_and_sign"]:
        return "no detectable link" + ("" if crit["c5_reliability_ge_05"] else "; rel. < 0.5")
    fails = [v for k, v in E8.C_NAMES.items() if not crit[k]]
    if not crit["c6_seeds_and_leave3out"]:
        if not (len(seeds) == 3 and all(seed_ok(x) for x in seeds)):
            fails.append("seeds")
        if not drop3_c1:
            fails.append("leave-3-out")
    return "p < .05; fails " + " + ".join(fails)


def transfer_rows(L: list[dict], b1: dict, b2: dict, e8: dict, absorb: dict, prec: dict, mult: dict,
                  rloco: dict) -> list[dict]:
    e8rows = {r["id"]: r for r in e8["transfer_map"]}
    frel = {k: {p: v["as_used"] for p, v in pv.items()} for k, pv in e8["feature_reliability_by_position"].items()}
    trel = e8["target_reliability_by_position"]
    ids = [lk["id"] for lk in L]
    q18 = dict(zip(ids, E8.bh([b1[i]["seed0"]["fl_p"] for i in ids])))
    # post hoc: beyond-B1 q over E008's family (14 round-1 primaries; E001 links over all 18), for comparability
    prim14 = [i for i in ids if e8rows[i]["primary14"]]
    q14 = dict(zip(prim14, E8.bh([b1[i]["seed0"]["fl_p"] for i in prim14])))
    rows = []
    for lk in L:
        lid, sg = lk["id"], lk["expected_sign"]
        r, r8 = b1[lid], e8rows[lid]
        s0 = r["seed0"]
        d = E8._sample(lk)
        R1, _ = _resid(d, B1_CTRL, [])
        R2, _ = _resid(d, B2_CTRL, [])
        fx, ty = frel[lid], trel[lid]
        rel_x, rel_y = E8._weighted(fx, d), E8._weighted(ty, d)
        rx1, ry1 = E8.resid_reliability(d, lk["feature"], fx, R1), E8.resid_reliability(d, lk["target"], ty, R1)
        rx2, ry2 = E8.resid_reliability(d, lk["feature"], fx, R2), E8.resid_reliability(d, lk["target"], ty, R2)
        ci = s0["partial_r_ci95"]
        dis1, dis = E8.corrected_ci(ci, rx1, ry1), E8.corrected_ci(ci, rel_x, rel_y)
        ceil = lambda c: max(sg * c[0], sg * c[1])
        se_z = 1 / np.sqrt(max(s0["resid_df"] - 1, 1))
        ceil_joint = float(np.tanh(np.arctanh(sg * s0["partial_r"]) + E8.JOINT_Z * se_z))
        ceil_joint_dis1 = float(min(ceil_joint / np.sqrt(rx1 * ry1), 1.0)) if rx1 > 0.05 and ry1 > 0.05 else 1.0
        mde1 = float(np.tanh((Z95_1 + Z80) * se_z))
        seeds = [r[k] for k in ("seed0", "seed1", "seed2") if k in r]
        l3 = r["leave3"]
        l3c1 = l3["passes_c1"] if l3 else None
        crit1 = _crit(s0, seeds, q18[lid], rel_x >= 0.5 and rel_y >= 0.5, l3c1)
        # B2 column: E008 estimates; criteria 2, 3 and 6 recomputed with the rev-3 harness; c4 as in E008 (BH over the
        # 14 round-1 primaries, E001 over all 18); c5 as in E008; leave-3-out from E008 (inference unchanged by rev 3)
        s2 = b2[lid]["seed0"]
        seeds2 = [b2[lid][k] for k in ("seed0", "seed1", "seed2") if k in b2[lid]]
        q2 = r8["bh_q14"] if r8["primary14"] else r8["bh_q18"]
        l3c1_2 = e8["link_results"][lid]["influence"]["drop3_supportive"]["passes_c1"]
        crit2 = _crit(s2, seeds2, q2, r8["c5_reliability_ge_05"], l3c1_2)
        # post hoc: criterion 3 (and criterion 6's per-seed LOCO) on the robust null scales; nothing else changes
        rl1, rl2 = rloco[lid]["B1"], rloco[lid]["B2"]
        rob = {f"{b}_{rule}": all(_crit(s, sd, q, rel, dc, [rl[k][rule] for k in ("seed0", "seed1", "seed2") if k in rl])
                                  .values())
               for b, s, sd, q, rel, dc, rl in (("b1", s0, seeds, q18[lid], rel_x >= 0.5 and rel_y >= 0.5, l3c1, rl1),
                                                ("b2", s2, seeds2, q2, r8["c5_reliability_ge_05"], l3c1_2, rl2))
               for rule in ("median_mad", "q84")}
        m1 = mult["links"].get(lid, {}).get("B1")
        c4_status = m1["c4_status"] if m1 else ("pass" if crit1["c4_bh_q_lt_05"] else "fail")
        no25 = r["without_2025"]
        positive = ci[0] is not None and min(sg * ci[0], sg * ci[1]) > 0
        ab = absorb[lid]
        ex = E8.exposure(lk)
        row = {
            **{k: r8[k] for k in ("id", "experiment", "test", "positions", "construct", "kind", "label",
                                   "outcome_target", "target_dagger", "feature_source", "sign_source", "primary14",
                                   "feature", "target", "expected_sign")},
            "n": s0["n"], "raw_r": s0["raw_r"], "rel_x": rel_x, "rel_y": rel_y,
            "r_target_log_exposure": ex["r_target_log_exposure"], "r_feature_log_exposure": ex["r_feature_log_exposure"],
            # ---- beyond B1 (primary family)
            "b1_resid_df": s0["resid_df"], "b1_partial_r": s0["partial_r"], "b1_ci_lo": ci[0], "b1_ci_hi": ci[1],
            "b1_mde80_one_sided": mde1, "b1_mde80_harness": s0["partial_r_mde80"],
            "b1_fl_p": s0["fl_p"], "b1_fl_p_two_sided": r["two_sided"]["fl_p"], "b1_bh_q18": q18[lid],
            # post hoc after review: Monte Carlo error and family sensitivity of criterion 4
            "b1_fl_p_mc_se": float(np.sqrt(s0["fl_p"] * (1 - s0["fl_p"]) / s0["n_perm_fl"])),
            "b1_fl_p_precise": prec.get(lid, {}).get("B1", {}).get("fl_p"),
            "b1_fl_p_precise_mc_se": prec.get(lid, {}).get("B1", {}).get("fl_p_mc_se"),
            "b1_bh_q_e008_family": q14.get(lid, q18[lid]), "b1_c4_status": c4_status,
            "b1_excess": s0["delta_r2_vs_B2"]["excess"], "b1_excess_se": s0["delta_r2_vs_B2"]["se"],
            "b1_excess_within_position": s0["delta_r2_vs_B2"]["excess_within_position"],
            "b1_cv_r2_baseline": s0["cv_r2"]["B2"], "b1_cv_perm_p": s0["cv_perm_p"],
            "b1_loco_pass": s0["checks"]["loco_classes_pass"],
            "b1_loco_pass_median_mad": f"{rl1['seed0']['median_mad']}/3", "b1_loco_pass_q84": f"{rl1['seed0']['q84']}/3",
            **{f"b1_loco_excess_{y}": v["excess"] for y, v in s0["loco"].items()},
            **{f"b1_loco_null_sd_{y}": v["null_sd"] for y, v in s0["loco"].items()},
            **{f"b1_partial_r_{y}": v["partial_r"] for y, v in s0["by_class"].items()},
            "b1_seeds_run": len(seeds), "b1_seeds_passing": sum(seed_ok(x) for x in seeds),
            "b1_excess_seeds": [x["delta_r2_vs_B2"]["excess"] for x in seeds],
            "b1_no2025_n": no25["n"], "b1_no2025_partial_r": no25["partial_r"],
            "b1_no2025_ci_lo": no25["partial_r_ci95"][0], "b1_no2025_ci_hi": no25["partial_r_ci95"][1],
            "b1_no2025_fl_p": no25["fl_p"], "b1_no2025_excess": no25["delta_r2_vs_B2"]["excess"],
            "b1_drop3_top3_share": l3["top3_supportive_share_of_partial_r"] if l3 else None,
            "b1_drop3_n": l3["n"] if l3 else None, "b1_drop3_partial_r": l3["partial_r"] if l3 else None,
            "b1_drop3_ci_lo": l3["partial_r_ci95"][0] if l3 else None,
            "b1_drop3_ci_hi": l3["partial_r_ci95"][1] if l3 else None, "b1_drop3_fl_p": l3["fl_p"] if l3 else None,
            "b1_drop3_excess": l3["delta_r2_vs_baseline"]["excess"] if l3 else None,
            "b1_drop3_loco_pass": l3["checks"]["loco_classes_pass"] if l3 else None,
            "b1_rel_x_resid": rx1, "b1_rel_y_resid": ry1,
            "b1_dis_lo": dis1[0], "b1_dis_hi": dis1[1], "b1_dis_as_used_lo": dis[0], "b1_dis_as_used_hi": dis[1],
            "b1_ceiling": ceil(ci), "b1_ceiling_dis": ceil(dis1), "b1_ceiling_dis_as_used": ceil(dis),
            "b1_ceiling_joint": ceil_joint, "b1_ceiling_joint_dis": ceil_joint_dis1,
            **{f"b1_{k}": v for k, v in crit1.items()}, "b1_passes_1to6": all(crit1.values()),
            "b1_verdict": _verdict(crit1, seeds, l3c1),
            "b1_passes_1to6_robust_loco_median_mad": rob["b1_median_mad"],
            "b1_passes_1to6_robust_loco_q84": rob["b1_q84"],
            "b1_bound_statement": (
                f"partial r {s0['partial_r']:+.2f}, 95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}] excludes 0 in the predicted "
                f"direction (reliability-corrected with beyond-B1 reliabilities [{dis1[0]:+.2f}, {dis1[1]:+.2f}])"
                if positive else
                f"per link at 95%: observed partial r beyond B1 above {ceil(ci):+.2f} in the predicted direction is "
                f"excluded (jointly over the 18 links, Bonferroni: above {ceil_joint:+.2f}); for error-free measures "
                f"(beyond-B1 reliability-corrected) a true-score partial r up to {ceil(dis1):+.2f} is not excluded; "
                f"one-sided 80%-power MDE {mde1:.2f}"),
            # ---- beyond B2 (E008 estimates; rev-3 criteria)
            "b2_resid_df": r8["resid_df"], "b2_partial_r": r8["partial_r"], "b2_ci_lo": r8["ci_lo"], "b2_ci_hi": r8["ci_hi"],
            "b2_mde80_one_sided": r8["mde80_one_sided"], "b2_fl_p": r8["fl_p"], "b2_fl_p_two_sided": r8["fl_p_two_sided"],
            "b2_bh_q14": r8["bh_q14"], "b2_bh_q18": r8["bh_q18"], "b2_bh_q_as_e008": q2,
            "b2_excess": r8["excess"], "b2_excess_se": r8["excess_se"],
            "b2_fl_p_precise": prec.get(lid, {}).get("B2", {}).get("fl_p"),
            "b2_fl_p_precise_mc_se": prec.get(lid, {}).get("B2", {}).get("fl_p_mc_se"),
            "b2_loco_pass_rev3": s2["checks"]["loco_classes_pass"], "b2_loco_pass_rev2": r8["loco_pos"],
            "b2_loco_pass_median_mad": f"{rl2['seed0']['median_mad']}/3", "b2_loco_pass_q84": f"{rl2['seed0']['q84']}/3",
            **{f"b2_loco_null_sd_{y}": v["null_sd"] for y, v in s2["loco"].items()},
            "b2_seeds_run": len(seeds2), "b2_seeds_passing_rev3": sum(seed_ok(x) for x in seeds2),
            **{f"b2_{k}": r8[k] for k in ("no2025_n", "no2025_partial_r", "no2025_ci_lo", "no2025_ci_hi", "no2025_fl_p",
                                         "drop3_top3_share", "drop3_n", "drop3_partial_r", "drop3_ci_lo", "drop3_ci_hi",
                                         "drop3_fl_p", "dis_b2_lo", "dis_b2_hi", "ceiling", "ceiling_dis_b2",
                                         "ceiling_joint", "ceiling_joint_dis_b2")},
            "b2_rel_x_resid": rx2, "b2_rel_y_resid": ry2,
            **{f"b2_{k}": v for k, v in crit2.items()}, "b2_passes_1to6": all(crit2.values()),
            "b2_verdict": _verdict(crit2, seeds2, l3c1_2),
            "b2_passes_1to6_robust_loco_median_mad": rob["b2_median_mad"],
            "b2_passes_1to6_robust_loco_q84": rob["b2_q84"],
            "b2_passes_1to6_e008_rev2": r8["passes_1to6"], "b2_verdict_e008_rev2": r8["verdict"],
            "b2_criteria_changed_rev2_to_rev3": [k for k in crit2 if crit2[k] != r8[k]],
            # ---- draft slot (B1 - B2)
            **{f"draft_{k}": v for k, v in ab.items() if k not in ("partial_r_b1", "partial_r_b2")},
        }
        # consistency with E008 (same sample, same B2 design and per-position reliabilities)
        assert abs(rx2 - r8["rel_x_b2"]) < 1e-9 and abs(ry2 - r8["rel_y_b2"]) < 1e-9, lid
        assert abs(rel_x - r8["rel_x"]) < 1e-9 and abs(rel_y - r8["rel_y"]) < 1e-9, lid
        assert abs(ex["r_target_log_exposure"] - r8["r_target_log_exposure"]) < 1e-9, lid
        assert abs(ab["partial_r_b1"] - s0["partial_r"]) < 1e-9 and abs(ab["partial_r_b2"] - r8["partial_r"]) < 1e-9, lid
        rows.append(row)
    return rows


def link_players(L: list[dict]) -> pl.DataFrame:
    """One row per link x player: feature, target and their residuals beyond B1 and beyond B2."""
    out = []
    for lk in L:
        d = E8._sample(lk)
        _, r1 = _resid(d, B1_CTRL, [lk["feature"], lk["target"]])
        _, r2 = _resid(d, B2_CTRL, [lk["feature"], lk["target"]])
        out.append(d.select("nfl_id", "display_name", "position", "draft_year", "log_pick", "undrafted",
                            feature=pl.col(lk["feature"]).cast(pl.Float64), target=pl.col(lk["target"]).cast(pl.Float64))
                   .with_columns(link_id=pl.lit(lk["id"]),
                                 x_res_b1=pl.Series(r1[lk["feature"]]), y_res_b1=pl.Series(r1[lk["target"]]),
                                 x_res_b2=pl.Series(r2[lk["feature"]]), y_res_b2=pl.Series(r2[lk["target"]])))
    return pl.concat(out)


def wr_speed_signal(lp: pl.DataFrame) -> dict:
    """E001-WR vs E006-P1 (one signal on the same receivers): correlations of features, targets and their residuals
    beyond B1 and beyond B2 over the shared players (residuals from each link's own sample, as in E008)."""
    a, b = (lp.filter(pl.col("link_id") == k) for k in ("E006-P1", "E001-WR"))
    j = a.join(b, on="nfl_id", suffix="_e001")
    r = lambda u, v: float(np.corrcoef(j[u], j[v])[0, 1])
    return {"n_e006_p1": a.height, "n_e001_wr": b.height, "n_shared": j.height,
            "r_feature_raw": r("feature", "feature_e001"), "r_target_raw": r("target", "target_e001"),
            "r_feature_b1_resid": r("x_res_b1", "x_res_b1_e001"), "r_target_b1_resid": r("y_res_b1", "y_res_b1_e001"),
            "r_feature_b2_resid": r("x_res_b2", "x_res_b2_e001"), "r_target_b2_resid": r("y_res_b2", "y_res_b2_e001")}


def summarize(tm: list[dict], e8: dict, mult: dict, rloco: dict) -> dict:
    """Counts, ranges and the multiplicity checks behind the headline statements."""
    nl = [r for r in tm if not r["b1_c1_fl_p_and_sign"]]
    nl2 = [r for r in tm if not r["b2_c1_fl_p_and_sign"]]
    rng = lambda rows, k: [float(min(r[k] for r in rows)), float(max(r[k] for r in rows))] if rows else None
    rng_s = lambda rows, k: ([float(min(r["expected_sign"] * r[k] for r in rows)),
                              float(max(r["expected_sign"] * r[k] for r in rows))] if rows else None)
    jk_hi = max(tm, key=lambda r: r["draft_delta_jack_ci95"][1])
    bt_hi = max(tm, key=lambda r: r["draft_delta_boot_ci95"][1])
    ratio = [r["draft_delta_boot_se_over_jack_se"] for r in tm if r["draft_delta_boot_se_over_jack_se"] is not None]

    def loco_counts(base: str, rows: list[dict]) -> dict:
        return {rule: sum(rloco[r["id"]][base]["seed0"][rule] >= 2 for r in rows)
                for rule in ("harness", "median_mad", "q84")}
    runs = [v for lid in rloco for b in rloco[lid] for v in rloco[lid][b].values()]
    cls = [c for v in runs for c in v["classes"].values()]
    robust = {
        "label": "post hoc after review (criterion 3's null scale); the rev-3 harness rule stays binding",
        "rules": {"harness": "excess (gain - null mean) > null SD (rev 3, binding)",
                  "median_mad": "gain - null median > 1.4826 x null MAD", "q84": "gain > null 84.13th percentile",
                  "all": "class partial r not reversed"},
        "replay": f"exact: {len(runs)} harness runs x 3 classes, max |replay - stored| "
                  f"{max(c['replay_max_abs_err'] for c in cls):.1e}",
        "b1_nulls_passing_c3": loco_counts("B1", nl), "b2_nulls_passing_c3": loco_counts("B2", nl2),
        "n_nulls_b1": len(nl), "n_nulls_b2": len(nl2),
        "b1_nulls_passing_c3_ids": {rule: sorted(r["id"] for r in nl if rloco[r["id"]]["B1"]["seed0"][rule] >= 2)
                                    for rule in ("median_mad", "q84")},
        "b2_nulls_passing_c3_ids": {rule: sorted(r["id"] for r in nl2 if rloco[r["id"]]["B2"]["seed0"][rule] >= 2)
                                    for rule in ("median_mad", "q84")},
        "c1_links_by_seed": {r["id"]: {b: {s: {k: v[k] for k in ("harness", "median_mad", "q84")}
                                           for s, v in rloco[r["id"]][b].items()} for b in ("B1", "B2")}
                             for r in tm if r["b1_c1_fl_p_and_sign"] or r["b2_c1_fl_p_and_sign"]},
        "null_sd_over_mad_sd_median_q90_classes_with_mad_sd_ge_001": [
            float(np.median([c["null_sd"] / c["null_mad_sd"] for c in cls if c["null_mad_sd"] >= 0.01])),
            float(np.quantile([c["null_sd"] / c["null_mad_sd"] for c in cls if c["null_mad_sd"] >= 0.01], 0.9))],
        "most_inflated_class_c1_links_seed0": max(
            ({"link": lid, "baseline": b, "class": y, **{k: c[k] for k in ("null_sd", "null_mad_sd", "null_kurtosis",
                                                                           "null_min", "obs", "null_median")}}
             for lid in rloco for b in ("B1", "B2") for y, c in rloco[lid][b]["seed0"]["classes"].items()
             if any(r["id"] == lid and r[f"{b.lower()}_c1_fl_p_and_sign"] for r in tm)),
            key=lambda c: c["null_sd"] / c["null_mad_sd"]),
        "null_min_over_all_runs": float(min(c["null_min"] for c in cls)),
        "null_kurtosis_median_max": [float(np.median([c["null_kurtosis"] for c in cls])),
                                     float(max(c["null_kurtosis"] for c in cls))],
        "verdict_changes": {f"{b}_{rule}": sorted(r["id"] for r in tm if r[f"{b}_passes_1to6_robust_loco_{rule}"]
                                                  != r[f"{b}_passes_1to6"])
                            for b in ("b1", "b2") for rule in ("median_mad", "q84")},
    }
    c1 = [r for r in tm if r["b1_c1_fl_p_and_sign"]]
    # post hoc: BH over the 36 tests (18 beyond B1 + E008's 18 beyond B2) for the links passing criterion 1 beyond B1
    p36 = [r["b1_fl_p"] for r in tm] + [r["b2_fl_p"] for r in tm]
    q36 = E8.bh(p36)
    d = np.array([r["draft_delta_signed"] for r in tm])
    return {
        "n_links": len(tm),
        "b1_passes_1to6": [r["id"] for r in tm if r["b1_passes_1to6"]],
        "b1_passes_c1": [r["id"] for r in c1],
        "b1_fl_p_lt_10": [r["id"] for r in tm if r["b1_fl_p"] < 0.10],
        "b1_criteria_by_link_passing_c1": {r["id"]: {k[3:]: r[k] for k in r if k.startswith("b1_c") and k[4].isdigit()}
                                           for r in c1},
        "b2_rev3_passes_1to6": [r["id"] for r in tm if r["b2_passes_1to6"]],
        "b2_e008_rev2_passes_1to6": [r["id"] for r in tm if r["b2_passes_1to6_e008_rev2"]],
        "b2_criteria_changes_rev2_to_rev3": {r["id"]: r["b2_criteria_changed_rev2_to_rev3"] for r in tm
                                             if r["b2_criteria_changed_rev2_to_rev3"]},
        "b2_verdict_text_changes_vs_e008": {r["id"]: [r["b2_verdict_e008_rev2"], r["b2_verdict"]] for r in tm
                                            if r["b2_verdict_e008_rev2"] != r["b2_verdict"]},
        "b1_n_nulls": len(nl),
        "b1_nulls_partial_r_expected_direction": rng_s(nl, "b1_partial_r"),
        "b1_nulls_raw_partial_r": rng(nl, "b1_partial_r"),
        "b2_nulls_partial_r_expected_direction": rng_s(nl2, "b2_partial_r"),
        "b2_nulls_raw_partial_r": rng(nl2, "b2_partial_r"),
        "note_null_ranges": "expected_direction = sign x partial r (+ = predicted direction); raw = as estimated. "
                            "The first E009 version reported the raw range under the name 'signed' (as E008's "
                            "summary.transfer.nulls_signed_partial_r does; there the two ranges coincide).",
        "b1_nulls_min_fl_p": float(min(r["b1_fl_p"] for r in nl)) if nl else None,
        "b1_ceiling_per_link_95": rng(nl, "b1_ceiling"), "b1_ceiling_joint_bonferroni18": rng(nl, "b1_ceiling_joint"),
        "b1_ceiling_corrected_beyond_b1": rng(nl, "b1_ceiling_dis"),
        "b1_corrected_beyond_b1_ge_05": sorted(r["id"] for r in nl if r["b1_ceiling_dis"] >= 0.5),
        "b1_mde80_one_sided": rng(tm, "b1_mde80_one_sided"),
        "b1_nulls_passing_loco_rev3": sum(r["b1_c3_loco_2of3"] for r in nl),
        "b2_nulls_passing_loco_rev3": sum(r["b2_c3_loco_2of3"] for r in tm if not r["b2_c1_fl_p_and_sign"]),
        "b2_nulls_passing_loco_rev2": sum(int(r["b2_loco_pass_rev2"].split("/")[0]) >= 2 for r in tm
                                          if not r["b2_c1_fl_p_and_sign"]),
        "b2_n_nulls": sum(not r["b2_c1_fl_p_and_sign"] for r in tm),
        "draft_delta_signed_range": [float(d.min()), float(d.max())],
        "draft_delta_abs_median": float(np.median(np.abs(d))),
        "draft_delta_ci_excludes_0": [r["id"] for r in tm if r["draft_delta_boot_ci95"][0] > 0
                                      or r["draft_delta_boot_ci95"][1] < 0],
        "draft_delta_jack_ci_excludes_0": [r["id"] for r in tm if r["draft_delta_jack_ci95"][0] > 0
                                           or r["draft_delta_jack_ci95"][1] < 0],
        "draft_delta_largest_upper_limit": {"jackknife": [jk_hi["id"], jk_hi["draft_delta_jack_ci95"][1]],
                                            "bootstrap": [bt_hi["id"], bt_hi["draft_delta_boot_ci95"][1]]},
        "draft_delta_boot_se_over_jack_se_range": [float(min(ratio)), float(max(ratio))],
        "draft_r2_feature_range": rng(tm, "draft_r2_feature_on_draft"),
        "draft_r2_target_range": rng(tm, "draft_r2_target_on_draft"),
        "c1_status_changes_b1_vs_b2": {r["id"]: {"b1": r["b1_c1_fl_p_and_sign"], "b2": r["b2_c1_fl_p_and_sign"]}
                                       for r in tm if r["b1_c1_fl_p_and_sign"] != r["b2_c1_fl_p_and_sign"]},
        "post_hoc_bh36": {r["id"]: {"b1_fl_p": r["b1_fl_p"], "b1_q18": r["b1_bh_q18"], "q36_b1": float(q36[i]),
                                    "q36_b2": float(q36[len(tm) + i])} for i, r in enumerate(tm) if r["b1_c1_fl_p_and_sign"]},
        "e008_all_round1_multiplicity_beyond_b2": e8["summary"]["multiplicity"],
        "post_hoc_c4_precision_and_families": mult,
        "b1_bh_q_e008_family_c1_links": {r["id"]: r["b1_bh_q_e008_family"] for r in tm if r["b1_c1_fl_p_and_sign"]},
        "b1_c4_status": {r["id"]: r["b1_c4_status"] for r in tm if r["b1_c1_fl_p_and_sign"]},
        "post_hoc_robust_loco": robust,
        "outcome_targets_without_2025": {r["id"]: {"b1_partial_r": r["b1_partial_r"], "b1_fl_p": r["b1_fl_p"],
                                                    "b1_no2025_partial_r": r["b1_no2025_partial_r"],
                                                    "b1_no2025_fl_p": r["b1_no2025_fl_p"], "b1_no2025_n": r["b1_no2025_n"]}
                                         for r in tm if r["outcome_target"]},
    }


def _clean(o):
    return E8._clean(o)


def test_count(L: list[dict], b1: dict, b2: dict, prec: dict, rloco: dict, mult: dict) -> dict:
    """Every test and estimate in E009, by label (PLAN section 4: count every test)."""
    ids = [lk["id"] for lk in L]
    n_seed = sum(k in b1[i] for i in ids for k in ("seed1", "seed2"))
    n_l3 = sum(b1[i]["leave3"] is not None for i in ids)
    n_b2 = sum(k in b2[i] for i in ids for k in ("seed0", "seed1", "seed2"))
    return {
        "prereg_family_beyond_b1": len(ids),
        "prespecified_robustness": {"seed_1_2_reruns_same_hypothesis": n_seed, "without_2025": len(ids),
                                    "leave3out": n_l3, "draft_slot_difference_estimates_no_test": len(ids)},
        "post_hoc": {"two_sided_p_descriptive_same_hypothesis": len(ids),
                     "draft_slot_difference_bootstrap_intervals": len(ids),
                     "b2_rev3_rejudgments_of_e008_tests_no_new_inference": n_b2, "bh36_family": 1},
        "post_hoc_after_review": {
            "precise_fl_p_reestimates_of_existing_tests": sum(len(v) for v in prec.values()),
            "robust_loco_replays_no_p": sum(len(s) for v in rloco.values() for s in v.values()),
            "draft_slot_difference_jackknife_intervals": len(ids),
            "bh_families_added": ["E008's 14-primary family for beyond-B1 p", "round-3 primaries",
                                  "all round-3 tests"]},
        "new_p_values_on_distinct_hypotheses_or_samples": 2 * len(ids) + n_l3,
        "m_round3_all_tests_bh": mult["m_round3_all_tests"],
        "note": f"seed reruns, two-sided p and precise p re-test hypotheses already counted; the "
                f"{2 * len(ids) + n_l3} = {len(ids)} beyond B1 + {len(ids)} without 2025 + {n_l3} leave-3-out enter "
                f"the round-3 all-tests BH with E010's tests",
    }


# =====================================================================================================
# 4. Figure: the two baselines side by side + the draft-slot difference
# =====================================================================================================

def _fp(p: float) -> str:
    """p-value label with two significant digits below 0.01 (so p and q = m p / rank read consistently)."""
    return f"{p:.4f}" if p < 0.01 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def _verdict_text(r: dict, b2_round1_fail: set) -> str:
    """Verdict per baseline, with markers: '‡' criterion 4 beyond B1 marginal (post hoc); '§' passes beyond B2 but
    fails BH across all ~90 round-1 tests. Wrapped to the column width."""
    import textwrap
    a = r["b1_verdict"] + ("; BH\u00a0marginal\u00a0‡" if r["b1_c4_status"] == "marginal" else "")
    b = r["b2_verdict"] + (" §" if r["id"] in b2_round1_fail else "")
    if a == b:
        return f"{a} (both)"
    return "\n".join(textwrap.fill(s, 44, subsequent_indent="      ") for s in (f"B1: {a}", f"B2: {b}"))


def _footnotes(rows: dict, sm: dict, mult: dict, b2_round1_fail: set) -> str:
    """Footnote text under the figure, each paragraph wrapped to the figure width."""
    import textwrap
    tw = {k: (rows[k]["b1_fl_p_two_sided"], rows[k]["b2_fl_p_two_sided"]) for k in ("E001-WR", "E006-P1")}
    changes = [k for k in E8.ROW_ORDER if rows[k]["b2_passes_1to6"] != rows[k]["b2_passes_1to6_e008_rev2"]]
    n_c3 = sum("c3_loco_2of3" in rows[k]["b2_criteria_changed_rev2_to_rev3"] for k in E8.ROW_ORDER)
    notes = []
    for lid, v in mult.items():  # criterion 4 beyond B1, marginal (post hoc)
        b = v.get("B1")
        if b and b["c4_status"] == "marginal":
            band = b["q18_precise_pm2_mc_se"]
            notes.append(
                f"‡ Criterion 4 beyond B1 for {lid} is marginal (post hoc): q18 = {b['q18_10k_seed0']:.3f} with the "
                f"prereg's 10,000 permutations (seed 0), {b.get('q18_10k_seed1', float('nan')):.3f} / "
                f"{b.get('q18_10k_seed2', float('nan')):.3f} at seeds 1 / 2, {b['q18_precise']:.3f} "
                f"[{band[0]:.3f}, {band[1]:.3f}] (+-2 MC SE) at {N_PERM_PRECISE:,} permutations; "
                f"{b['q_e008_family_precise']:.3f} over E008's 14 primaries, {b['q_round3_primaries_precise']:.3f} over "
                f"the round-3 primaries (E009 + E010), {b['q_round3_all_tests_precise']:.2f} over all "
                f"{sm['post_hoc_c4_precision_and_families']['m_round3_all_tests']} round-3 tests.")
    for lid in sorted(b2_round1_fail):
        b = mult[lid]["B2"]
        notes.append(f"§ {lid} beyond B2 fails BH across all ~90 round-1 tests: q = {b['q_all_round1_tests_e008']:.3f} "
                     f"(E008)" + (f", {b['q_all_round1_tests_precise']:.3f} with {N_PERM_PRECISE:,} permutations."
                                  if b["q_all_round1_tests_precise"] is not None else "."))
    rb = sm["post_hoc_robust_loco"]
    wr = rb["c1_links_by_seed"].get("E001-WR", {}).get("B1", {})
    ex = rb["most_inflated_class_c1_links_seed0"]
    notes.append(
        "LOCO null scale (post hoc): a few degenerate ridge permutations (held-out dR2 down to "
        f"{rb['null_min_over_all_runs']:.1f}) inflate criterion 3's per-class null SD (e.g. {ex['link']} beyond "
        f"{ex['baseline']}, {ex['class']}: SD {ex['null_sd']:.3f} vs MAD-based {ex['null_mad_sd']:.3f}). On a robust "
        "scale (median + MAD-SD / 84th percentile), "
        + (f"E001-WR beyond B1 passes {wr['seed0']['median_mad']}/3 / {wr['seed0']['q84']}/3 classes "
           f"(harness {wr['seed0']['harness']}/3), " if wr else "")
        + f"{rb['b1_nulls_passing_c3']['median_mad']} / {rb['b1_nulls_passing_c3']['q84']} of {rb['n_nulls_b1']} "
          f"nulls pass beyond B1 and {rb['b2_nulls_passing_c3']['median_mad']} / {rb['b2_nulls_passing_c3']['q84']} "
          f"of {rb['n_nulls_b2']} beyond B2 (harness {rb['b1_nulls_passing_c3']['harness']} and "
          f"{rb['b2_nulls_passing_c3']['harness']}); "
        + ("no verdict changes." if not any(rb["verdict_changes"].values()) else
           "verdict changes: " + ", ".join(f"{k} {v}" for k, v in rb["verdict_changes"].items() if v) + "."))
    rr = sm["draft_delta_boot_se_over_jack_se_range"]
    paras = ["B1 = position and draft-year dummies + traditional Combine (40, 10-yd split, 3-cone, shuttle, vertical, broad, "
        "bench, height, weight, NGS athleticism score; median-imputed with missing indicators). B2 = B1 + draft pick "
        "(log pick, undrafted). Beyond-B2 estimates are E008's (reproduced exactly).",
        "Line: 95% CI of the partial correlation (Fisher z). Band: CI / sqrt(rel_x x rel_y) with the reliabilities of the "
        "parts of feature and target left after that baseline (classical error, clipped at +-1): what error-free "
        "measures could show.",
        "p: Freedman-Lane, 10,000 permutations, one-sided in the expected direction: preregistered for E002-E005, "
        "inferred from the hypothesis for E006, set after round 1 for E001 (two-sided p beyond B1 / B2: E001-WR "
        f"{_fp(tw['E001-WR'][0])} / {_fp(tw['E001-WR'][1])}, E006-P1 {_fp(tw['E006-P1'][0])} / "
        f"{_fp(tw['E006-P1'][1])}).",
        "BH q, different families: beyond B1 over the 18 links (the E009 prereg family); beyond B2 as in E008, over the "
        "14 round-1 primaries (* = E001, over all 18). Seeds 1-2 (criterion 6) were run only when p < 0.10.",
        "Go criteria (rev 3, each baseline): 1 p < 0.05; 2 CV excess dR2 - SE > 0 in the expected direction; 3 in 2 of 3 "
        "held-out classes the excess exceeds 1 null SD and the class r is not reversed; 4 BH q < 0.05; 5 reliability "
        ">= 0.5; 6 three CV seeds + leave-3-out."
        + (f" Beyond-B2 verdicts re-judged under rev 3 (E008 logged rev 2; verdicts changed: {', '.join(changes)})."
           if changes else f" Beyond-B2 verdicts re-judged under rev 3: no verdict changes from E008's rev 2 (criterion "
                           f"3 now fails for the {n_c3} null links the rev-2 rule passed)."),
        *notes,
        "Draft-slot panel: partial r beyond B1 minus beyond B2, signed; + = the draft slot absorbs part of the link, "
        "- = adding it strengthens the link. Dark: delete-one jackknife, normal 95%; light: paired percentile bootstrap "
        f"(2,000 resamples), {rr[0]:.1f}-{rr[1]:.1f}x the jackknife SE (widest where the draft slot explains little). "
        "Both post hoc.",
        "† Round-1 review: the game target is narrower than its name (E002-P1 route style; E002-P2 separation mostly "
        "precedes the break; E003-P1 window runs into blockers; E004-P1 scheme/reaction; E005-P1 retreat at the flip or "
        "throw; E005-P2 generic redirection)."]
    return "\n".join(textwrap.fill(t, 310) for t in paras)


def figure_two_baselines(tm: pl.DataFrame, prof: pl.DataFrame, sm: dict, path: Path) -> None:
    """Forest plots of the partial r beyond B1 and beyond B2 (sign-adjusted: right = expected direction), with 95% CI,
    the reliability-corrected interval of each (residual reliabilities beyond that baseline) and the one-sided MDE;
    a third panel shows how much the draft slot moves each partial r (jackknife and paired-bootstrap 95%)."""
    ws = sm["wr_speed_signal"]
    mult = sm["post_hoc_c4_precision_and_families"]["links"]
    b2_round1_fail = {k for k, v in mult.items() if "B2" in v and v["B2"]["q_all_round1_tests_e008"] is not None
                      and v["B2"]["q_all_round1_tests_e008"] >= 0.05}
    plt = E8._style()
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    INK, INK2, MUTED, GRID, BASE, SURF = E8.INK, E8.INK2, E8.MUTED, E8.GRID, E8.BASE, E8.SURF
    BLUE, ORANGE, BLUE_WASH, ORANGE_WASH = E8.BLUE, E8.ORANGE, E8.BLUE_WASH, E8.ORANGE_WASH
    rows = {r["id"]: r for r in tm.iter_rows(named=True)}
    rel = lambda probe, metric, scope: prof.filter((pl.col("probe") == probe) & (pl.col("metric") == metric)
                                                   & (pl.col("scope") == scope))["rel_round1_def"][0]
    burst4 = rel("B DL burst (E003)", "d10", "cross-drill burst4")
    wave = rel("C OL lateral (E004)", "wave rev_accel", "reversals within one attempt")
    subs = [i for i in E8.ROW_ORDER if rows[i]["b1_c1_fl_p_and_sign"] or rows[i]["b2_c1_fl_p_and_sign"]]

    def link_items(ids):
        out = []
        for i in ids:
            out.append(("link", i))
            if i in subs:
                out += [("sub", i, "no2025"), ("sub", i, "drop3")]
        return out
    items = ([("head", "Wide receivers")] + link_items(E8.ROW_ORDER[:2]) + [("signal",)] + link_items(E8.ROW_ORDER[2:6])
             + [("gap",), ("head", "Defensive line")] + link_items(E8.ROW_ORDER[6:10])
             + [("note", f"DL position drills: not tested against games (4-drill burst composite reliability {burst4:.2f})"),
                ("gap",), ("head", "Offensive line")] + link_items(E8.ROW_ORDER[10:14])
             + [("note", f"OL wave drill: screened out on reliability ({wave:.2f}), not tested"), ("gap",),
                ("head", "Defensive backs")] + link_items(E8.ROW_ORDER[14:17])
             + [("gap",), ("head", "Pooled WR + TE + DB + DL")] + link_items(E8.ROW_ORDER[17:]))
    y, placed = 0.0, []
    for it in items:
        if it[0] == "head":
            placed.append((it, y + 0.12)); y += 0.85
        elif it[0] == "link":
            placed.append((it, y)); y += 1.0
        elif it[0] == "sub":
            placed.append((it, y - 0.42)); y += 0.5
        elif it[0] in ("note", "signal"):
            placed.append((it, y - 0.25)); y += 0.7
        else:
            y += 0.45
    foot = _footnotes(rows, sm, mult, b2_round1_fail)
    W, top_in = 19.0, 2.30
    bot_in = 0.95 + (foot.count("\n") + 1) * 7.1 * 1.45 / 72  # axis labels + footnote lines
    H = 0.33 * y + top_in + bot_in
    fig = plt.figure(figsize=(W, H))
    top, bot = 1 - top_in / H, bot_in / H
    yf = lambda v: bot + (top - bot) * (y - 0.4 - v) / (y - 0.4 + 0.5)  # data y -> figure fraction
    PANELS = {"b1": (0.250, 0.180), "b2": (0.505, 0.180), "d": (0.760, 0.075)}
    TXT = {"b1": [(0.436, "p"), (0.466, "q (18)")], "b2": [(0.691, "p"), (0.721, "q (E008)")]}
    X_N, X_VERDICT = 0.842, 0.864
    axes = {}
    for k, (x0, w) in PANELS.items():
        ax = fig.add_axes([x0, bot, w, top - bot])
        ax.set_ylim(y - 0.4, -0.5)
        lim = (-1, 1) if k != "d" else (-0.45, 0.45)
        ax.set_xlim(*lim)
        for xv in ((-0.5, 0.5) if k != "d" else (-0.25, 0.25)):
            ax.axvline(xv, color=GRID, lw=0.8, zorder=0)
        ax.axvline(0, color=BASE, lw=1.0, zorder=1)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        axes[k] = ax
    axes["b1"].set_xticks([-1, -0.5, 0, 0.5, 1], ["-1", "-0.5", "0", "0.5", "1"])
    axes["b2"].set_xticks([-1, -0.5, 0, 0.5, 1], ["-1", "-0.5", "0", "0.5", "1"])
    axes["d"].set_xticks([-0.25, 0, 0.25], ["-0.25", "0", "0.25"])
    axes["b1"].set_xlabel("partial r beyond B1, signed: right = expected direction", fontsize=8.4)
    axes["b2"].set_xlabel("partial r beyond B2, signed: right = expected direction", fontsize=8.4)
    axes["d"].set_xlabel("r(B1) - r(B2), signed", fontsize=8.4)
    for k, title, sub in (("b1", "Beyond B1: the stopwatch", "position, draft year, Combine times + measurements"),
                          ("b2", "Beyond B2: stopwatch + draft slot", "B1 + draft pick (E008 estimates)"),
                          ("d", "What the draft slot absorbs", "r(B1) - r(B2): + absorbs, - sharpens")):
        x0, w = PANELS[k]
        fig.text(x0, top + 0.30 / H, title, ha="left", va="bottom", fontsize=9.6, weight="bold", color=INK)
        fig.text(x0, top + 0.14 / H, sub, ha="left", va="bottom", fontsize=7.8, color=INK2)
    for k in ("b1", "b2"):
        for xf, h in TXT[k]:
            fig.text(xf, top + 0.04 / H, h, ha="left", va="bottom", fontsize=8, color=INK2, weight="bold")
    fig.text(X_N, top + 0.04 / H, "n", ha="left", va="bottom", fontsize=8, color=INK2, weight="bold")
    fig.text(X_VERDICT, top + 0.04 / H, "verdict (go criteria 1-6, rev 3)", ha="left", va="bottom", fontsize=8,
             color=INK2, weight="bold")
    mk = lambda r: "o" if r["feature_source"] == "position drill" else "s"
    sgn = lambda r, a, b: sorted((r["expected_sign"] * a, r["expected_sign"] * b))
    yy_of = {}
    for it, yy in placed:
        if it[0] == "head":
            fig.text(0.008, yf(yy), it[1], ha="left", va="center", fontsize=9.5, weight="bold", color=INK)
        elif it[0] == "note":
            fig.text(0.026, yf(yy), it[1], ha="left", va="center", fontsize=7.8, style="italic", color=MUTED)
        elif it[0] == "signal":
            fig.text(0.026, yf(yy), f"E001-WR and E006-P1 are one signal: {ws['n_shared']} WRs in both; residual r of the "
                     f"features {ws['r_feature_b1_resid']:.2f} (B1) / {ws['r_feature_b2_resid']:.2f} (B2), of the targets "
                     f"{ws['r_target_b1_resid']:.2f} / {ws['r_target_b2_resid']:.2f}", ha="left", va="center",
                     fontsize=7.6, style="italic", color=INK2)
        elif it[0] == "link":
            r = rows[it[1]]
            yy_of[it[1]] = yy
            col, wash = (ORANGE, ORANGE_WASH) if r["outcome_target"] else (BLUE, BLUE_WASH)
            for k, band, mde in (("b1", ("b1_dis_lo", "b1_dis_hi"), "b1_mde80_one_sided"),
                                 ("b2", ("b2_dis_b2_lo", "b2_dis_b2_hi"), "b2_mde80_one_sided")):
                ax = axes[k]
                lo, hi = sgn(r, r[f"{k}_ci_lo"], r[f"{k}_ci_hi"])
                dlo, dhi = sgn(r, r[band[0]], r[band[1]])
                est = r["expected_sign"] * r[f"{k}_partial_r"]
                ax.add_patch(plt.Rectangle((dlo, yy - 0.27), dhi - dlo, 0.54, color=wash, lw=0, zorder=1))
                ax.plot([lo, hi], [yy, yy], color=col, lw=2, solid_capstyle="round", zorder=3)
                ax.plot([r[mde]], [yy], "|", color=INK2, ms=9, mew=1.2, zorder=2)
                ax.plot([est], [yy], mk(r), ms=8 if mk(r) == "o" else 7.2,
                        mfc=col if r[f"{k}_c1_fl_p_and_sign"] else SURF, mec=col, mew=2, zorder=4)
                if r[f"{k}_passes_1to6"]:
                    ax.plot([est], [yy], "o", ms=15, mfc="none", mec=INK, mew=1.3, zorder=5)
            # draft-slot panel: point + jackknife 95% (dark) over the paired-bootstrap 95% (light)
            dl, dh = r["draft_delta_boot_ci95"]
            axes["d"].plot([max(dl, -0.45), min(dh, 0.45)], [yy, yy], color=GRID, lw=4.0, solid_capstyle="round", zorder=2)
            jl, jh = r["draft_delta_jack_ci95"]
            axes["d"].plot([max(jl, -0.45), min(jh, 0.45)], [yy, yy], color=INK2, lw=1.6, solid_capstyle="round", zorder=3)
            axes["d"].plot([r["draft_delta_signed"]], [yy], "D", ms=5.5, mfc=INK2, mec=SURF, mew=1.0, zorder=4)
            fig.text(0.008, yf(yy), f"{r['experiment']} {r['test']}", ha="left", va="center", fontsize=7.6, color=MUTED)
            fig.text(0.243, yf(yy), r["label"] + (" †" if r["target_dagger"] else ""), ha="right", va="center",
                     fontsize=8.4, color=INK)
            q2 = r["b2_bh_q_as_e008"]
            vals = {"b1": (_fp(r["b1_fl_p"]), f"{r['b1_bh_q18']:.3f}" + (" ‡" if r["b1_c4_status"] == "marginal" else "")),
                    "b2": (_fp(r["b2_fl_p"]), f"{q2:.3f}" + ("*" if not r["primary14"] else "")
                           + (" §" if r["id"] in b2_round1_fail else ""))}
            for k in ("b1", "b2"):
                for (xf, _), txt in zip(TXT[k], vals[k]):
                    fig.text(xf, yf(yy), txt, ha="left", va="center", fontsize=8.1, color=INK)
            fig.text(X_N, yf(yy), f"{r['n']}", ha="left", va="center", fontsize=8.1, color=INK)
            pas = r["b1_passes_1to6"] or r["b2_passes_1to6"]
            fig.text(X_VERDICT, yf(yy), _verdict_text(r, b2_round1_fail), ha="left", va="center", fontsize=7.7,
                     color=INK, weight="bold" if pas else "normal", linespacing=1.15)
        else:  # sub-row: without the 2025 class / without the 3 most supportive players (per baseline)
            r = rows[it[1]]
            pos = "WRs" if r["positions"] == "WR" else "players"
            for k, pre in (("b1", "b1_"), ("b2", "b2_")):
                if r[f"{pre}{it[2]}_partial_r"] is None:  # leave-3-out run only where criterion 1 passed
                    continue
                lo, hi = sgn(r, r[f"{pre}{it[2]}_ci_lo"], r[f"{pre}{it[2]}_ci_hi"])
                est = r["expected_sign"] * r[f"{pre}{it[2]}_partial_r"]
                axes[k].plot([lo, hi], [yy, yy], color=MUTED, lw=1.3, solid_capstyle="round", zorder=3)
                axes[k].plot([est], [yy], mk(r), ms=5, mfc=MUTED, mec=MUTED, zorder=4)
                fig.text(TXT[k][0][0], yf(yy), _fp(r[f"{pre}{it[2]}_fl_p"]), ha="left", va="center", fontsize=7.4,
                         color=MUTED)
            if it[2] == "no2025":
                lab = "without the 2025 class (rookie season only)"
            else:
                s1, s2 = r["b1_drop3_top3_share"], r["b2_drop3_top3_share"]
                lab = (f"without its 3 most supportive {pos} (" + (f"B1 {s1:.0%} / B2 {s2:.0%}" if s1 is not None and s2 is not None
                                                                   else "") + " of the partial r)")
            fig.text(0.243, yf(yy), lab, ha="right", va="center", fontsize=7.5, color=MUTED)
            nn = r[f"b1_{it[2]}_n"] if r[f"b1_{it[2]}_n"] is not None else r[f"b2_{it[2]}_n"]
            fig.text(X_N, yf(yy), f"{nn}", ha="left", va="center", fontsize=7.4, color=MUTED)
    # bracket in the left margin: E001-WR and E006-P1 are one signal on the same receivers
    y1, y2 = yf(yy_of["E001-WR"]), yf(yy_of["E006-P1"])
    xb = 0.0035
    fig.add_artist(Line2D([xb + 0.003, xb, xb, xb + 0.003], [y1, y1, y2, y2], color=INK2, lw=0.9,
                          transform=fig.transFigure))
    n_b1 = int(tm["b1_c1_fl_p_and_sign"].sum())
    n_b2 = int(tm["b2_c1_fl_p_and_sign"].sum())
    d = tm["draft_delta_signed"]
    up = sm["draft_delta_largest_upper_limit"]["jackknife"]
    n_ex = len(sm["draft_delta_ci_excludes_0"]) + len(sm["draft_delta_jack_ci_excludes_0"])
    fig.text(0.008, 1 - 0.26 / H, "Transfer map, two baselines: does a Combine tracking signature add anything beyond "
             "the stopwatch (B1), and beyond the stopwatch and draft slot (B2)?", fontsize=11.5, weight="bold",
             color=INK, va="top")
    fig.text(0.008, 1 - 0.58 / H,
             f"The same 18 links (E008) against two nested baselines. One-sided p < 0.05: {n_b1} of 18 beyond B1, "
             f"{n_b2} of 18 beyond B2. Adding the draft slot changes a partial r by at most {d.abs().max():.2f} "
             f"(median {d.abs().median():.2f}); " + ("no jackknife or bootstrap interval of that change excludes 0"
                                                     if n_ex == 0 else "see the draft-slot panel")
             + f" (largest upper limit {up[1]:+.2f}, {up[0]}).\n"
             "E009 is preregistered but not blind (the beyond-B2 results were known). Read the B1 column as information "
             "beyond the stopwatch, not as market inefficiency.", fontsize=8.6, color=INK2, va="top", linespacing=1.4)
    gray = dict(color=INK2, lw=0, mew=1.5)
    h = [Line2D([], [], color=BLUE, lw=2, marker="o", ms=7, mfc=BLUE, label="target: a game movement trait"),
         Line2D([], [], color=ORANGE, lw=2, marker="o", ms=7, mfc=ORANGE, label="target: a game outcome"),
         Line2D([], [], marker="o", ms=7, mfc=SURF, **gray, label="feature: position drill"),
         Line2D([], [], marker="s", ms=6.5, mfc=SURF, **gray, label="feature: tracked 40-yd dash"),
         Line2D([], [], marker="o", ms=7, mfc=INK2, **gray, label="filled = one-sided p < 0.05"),
         Line2D([], [], marker="o", ms=11, mfc="none", mec=INK, lw=0, mew=1.3, label="ring = passes go criteria 1-6"),
         (Patch(color=BLUE_WASH), Patch(color=ORANGE_WASH)),
         Line2D([], [], color=INK2, lw=0, marker="|", ms=9, label="min. detectable r (80% power, one-sided)"),
         (dh := (Line2D([], [], color=GRID, lw=4.0),
                 Line2D([], [], color=INK2, lw=1.6, marker="D", ms=5, mfc=INK2, mec=SURF)))]
    labels = [x.get_label() if not isinstance(x, tuple) else
              ("error-free (reliability-corrected) interval" if isinstance(x[0], Patch) else
               "draft slot: r(B1) - r(B2); jackknife 95% (dark), bootstrap 95% (light)") for x in h]
    fig.legend(handles=h, labels=labels, loc="upper left", bbox_to_anchor=(0.004, 1 - 1.05 / H), ncol=5, frameon=False,
               fontsize=8, handlelength=1.8, columnspacing=1.6,
               handler_map={dh: HandlerTuple(ndivide=1), tuple: HandlerTuple(ndivide=None, pad=0)})
    fig.text(0.008, 0.12 / H, foot, fontsize=7.1, color=INK2, va="bottom", linespacing=1.45)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_figures() -> None:
    tm = pl.read_parquet(OUT_DATA / "e009_transfer_map.parquet")
    prof = pl.read_parquet(io.PROCESSED / "e008_drill_reliability.parquet")
    sm = json.loads((OUT_RES / "E009.json").read_text())["summary"]
    figure_two_baselines(tm, prof, sm, OUT_FIG / "E009_transfer_map_two_baselines.png")
    log("figure written")


# =====================================================================================================
# 5. Main
# =====================================================================================================

def main() -> None:
    OUT_RES.mkdir(parents=True, exist_ok=True)
    prereg = json.loads((RES / "E009_prereg.json").read_text())
    e8 = json.loads((RES / "E008.json").read_text())
    L, _ = E8.links()
    rep = E8.check_reproduction(L)
    assert all(v["match"] for v in rep.values()), "round-1 n / raw r not reproduced"
    assert [lk["id"] for lk in L] == list(e8["links"]), "link set differs from E008"
    for lk in L:  # same features, targets, populations and expected signs as E008
        s = e8["links"][lk["id"]]
        assert (lk["feature"], lk["target"], lk["positions"], lk["expected_sign"]) == \
               (s["feature"], s["target"], s["positions"], s["expected_sign"]), lk["id"]
    b2, b2_diffs = run_b2_rev3(L, e8)  # first: it also verifies the harness reproduces E008 (aborts on mismatch)
    absorb = draft_absorption(L)
    b1 = run_b1(L)
    # post hoc after the round-3 review
    prec = precise_fl(L, b1, e8)
    r3 = round3_inputs()
    mult = multiplicity_posthoc(L, b1, e8, prec, r3)
    rloco = robust_loco(L, b1, b2)
    tm = transfer_rows(L, b1, b2, e8, absorb, prec, mult, rloco)
    lp = link_players(L)
    ws = wr_speed_signal(lp)
    ws8 = e8["summary"]["wr_speed_signal"]
    assert abs(ws["r_feature_b2_resid"] - ws8["r_feature_b2_resid"]) < 1e-9 and \
           abs(ws["r_target_b2_resid"] - ws8["r_target_b2_resid"]) < 1e-9
    tmdf = pl.DataFrame(tm, infer_schema_length=None).with_columns(
        pl.col("b1_excess_seeds").list.eval(pl.element().cast(pl.Float64)))
    tmdf.write_parquet(OUT_DATA / "e009_transfer_map.parquet")
    lp.write_parquet(OUT_DATA / "e009_link_players.parquet")
    out = {
        "experiment_id": "E009", "title": "Information value beyond the stopwatch (B1) for the 18 transfer-map links",
        "date": "2026-10-14", "quick_mode": QUICK, "protocol": "PLAN.md section 4 rev 3",
        "prereg": {"file": "experiments/results/E009_prereg.json", "status_of_evidence": prereg["status_of_evidence"]},
        "settings": {**EVAL, "seeds": f"0 (+1, 2 when fl_p < {SEED_RULE_P})", "n_jobs": N_JOBS, "n_boot": N_BOOT,
                     "n_perm_fl_precise_post_hoc": N_PERM_PRECISE,
                     "boot_seed": BOOT_SEED, "controls_B1": B1_CTRL + ["position dummies", "draft-year dummies"],
                     "controls_B2": B2_CTRL + ["position dummies", "draft-year dummies"],
                     "key_note": "linkage.evaluate was called with draft_controls=[] for B1, so in link_results_b1 every "
                                 "key named 'B2' (delta_r2_vs_B2, cv_r2['B2']) refers to the B1 baseline; cv_r2['B1'] "
                                 "and cv_r2['B2'] are identical there."},
        "test_labels": {
            "prereg_family_secondary_not_blind": "18 links beyond B1 (linkage.evaluate draft_controls=[], rev 3), one "
                       "BH family of 18. This is E009's preregistered family (the protocol's 'primary' tier for this "
                       "experiment), but its prereg labels the evidence SECONDARY and NOT BLIND (E008's beyond-B2 "
                       "results were known; B1 is nested in B2): not confirmatory, and not to be pooled as confirmatory "
                       "primaries into a hero claim. It is counted with the round-3 primaries only in the conservative "
                       "post hoc BH (summary.post_hoc_c4_precision_and_families).",
            "prespecified_robustness": "seeds 1-2 when fl_p < 0.10; without the 2025 class (all 18); leave-3-out "
                                       "(3 most supportive players beyond B1) for links passing criterion 1; the "
                                       "difference in partial r between B1 and B2 (draft_delta_signed)",
            "post_hoc": "paired-bootstrap interval and FWL decomposition of the B1-B2 difference; beyond-B1 "
                        "reliability-corrected intervals, joint Bonferroni ceilings and one-sided MDE (E008's bound "
                        "definitions); two-sided Freedman-Lane p (descriptive); BH over the 36 B1 + B2 tests for links "
                        "passing criterion 1; E008's beyond-B2 tests re-judged under rev 3 (b2_*: inference identical "
                        "to E008, asserted; criteria 2, 3, 6 recomputed); exposure re-checked (identical targets)",
            "post_hoc_after_review": "Freedman-Lane p at 2,000,000 permutations for links with p < 0.10 (precise_fl) "
                                     "and criterion 4 under other permutation seeds and BH families, incl. the round-3 "
                                     "primaries and all round-3 tests (summary.post_hoc_c4_precision_and_families; "
                                     "b1_c4_status); criterion 3 on robust null scales from an exact replay of the "
                                     "harness LOCO null (robust_loco; summary.post_hoc_robust_loco; "
                                     "*_passes_1to6_robust_loco_*); delete-one jackknife interval of the draft-slot "
                                     "difference (draft_delta_jack_*); beyond-B1 q over E008's family "
                                     "(b1_bh_q_e008_family); null ranges signed to the expected direction; verdict "
                                     "text listing every failed part of criterion 6",
        },
        "test_count": test_count(L, b1, b2, prec, rloco, mult),
        "criteria_definitions": {
            "c1_fl_p_and_sign": "one-sided Freedman-Lane p < 0.05 and partial r in the expected direction",
            "c2_excess_lb_gt_0": "checks.excess_lb_gt_0 (rev 3: excess - SE > 0 and sign not reversed)",
            "c3_loco_2of3": "checks.loco_classes_pass >= 2/3 (rev 3: held-out excess > 1 null SD and class partial r "
                            "not reversed)",
            "c4_bh_q_lt_05": "beyond B1: BH over the 18 links (prereg; 10,000 permutations, seed 0); beyond B2: as in "
                             "E008 (14 round-1 primaries; E001 over 18). Post hoc b1_c4_status: 'marginal' when the "
                             "pass/fail of q < 0.05 differs across permutation seeds, 2,000,000 permutations (+2 MC "
                             "SE) or BH family (E008's 14, round-3 primaries, all round-3 tests)",
            "c5_reliability_ge_05": "as-used feature and target reliability >= 0.5 (E008 values; baseline-independent)",
            "c6_seeds_and_leave3out": "seeds 0-2 each pass c1-c3 (rev 3; seeds run only when fl_p < 0.10) and the "
                                      "leave-3-out sample passes c1 (beyond B1: run for links passing c1; beyond B2: "
                                      "E008's leave-3-out)",
        },
        "verification": {
            "reproduction_round1": rep,
            "b2_partial_r_matches_e008": True,
            "b2_rev3_rerun_max_abs_diff_vs_e008_on_unchanged_keys": b2_diffs,
            "notes": "transfer_rows asserts beyond-B2 residual reliabilities, as-used reliabilities, exposure "
                     "correlations and both partial r against E008.json; draft_absorption reports the FWL identity "
                     "error; wr_speed_signal beyond B2 asserted against E008",
        },
        "summary": {**summarize(tm, e8, mult, rloco), "wr_speed_signal": ws},
        "links": {lk["id"]: e8["links"][lk["id"]] for lk in L},
        "transfer_map": tmdf.to_dicts(),
        "draft_absorption": absorb,
        "link_results_b1": b1,
        "link_results_b2_rev3": b2,
        "precise_fl": prec,
        "round3_inputs": r3,
        "robust_loco": rloco,
    }
    (OUT_RES / "E009.json").write_text(json.dumps(_clean(out), indent=1))
    log(f"wrote {OUT_RES / 'E009.json'}")
    make_figures()


if __name__ == "__main__":
    if FIGURES_ONLY:
        make_figures()
    else:
        main()
