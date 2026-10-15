"""Re-score the WR speed links under protocol rev 3 (writeup [[TBD-REV3]]). Scratch; nothing in the repo is written.

E008 ran under rev 2, whose criterion 3 (LOCO excess > 0) is superseded; E008.json stores no per-class null SD, so the
rev-3 criterion 3 (`checks.loco_classes_pass`: held-out excess > 1 null SD and the class partial r not reversed) cannot
be read from it. This script re-runs the working-tree rev-3 `linkage.evaluate` (same settings as E008: 50x5 CV,
2,000 CV permutations, 10,000 Freedman-Lane permutations, n_jobs=3) on:
  A. E006-P1 as preregistered and as computed in round 1 (target = probe-E S0, AS_R = 20), seeds 0-2.
     -> re-scoring of a primary under the current protocol (rev 3), not a new test.
  B. E001-WR (prespecified sanity link), seed 0.
  C. E006-P1 on the shared-module S0 (AS_R = 1000): post hoc in E008 already; here re-scored under rev 3, plus the
     without-2025 and leave-3-out (3 most supportive WRs) partial r. Post hoc.
Features/targets for A and B come from data/processed/e008_link_players.parquet (E008's exact link samples); controls
from linkage.player_table(). Each result is checked against E008.json before use.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from bdb27 import linkage, speed

ROOT = Path("/Users/jasonli/Documents/GitHub/NFL-Big-Data-Bowl-2027")
OUT = Path(__file__).with_name("ph_rev3.json")
EVAL = dict(n_repeats=50, n_perm=2000, n_perm_fl=10000, n_jobs=3)
T0 = time.time()
e8 = json.loads((ROOT / "experiments/results/E008.json").read_text())
tm = {r["id"]: r for r in e8["transfer_map"]}
lp = pl.read_parquet(ROOT / "data/processed/e008_link_players.parquet")
pt = linkage.player_table()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def table(link):
    s = lp.filter(pl.col("link_id") == link).select("nfl_id", feature="feature", target="target")
    return pt.join(s, on="nfl_id", how="inner")


def slim(r):
    keep = ("n", "partial_r", "partial_r_ci95", "fl_p", "by_class", "delta_r2_vs_B2", "loco", "checks", "cv_r2")
    return {k: r[k] for k in keep if k in r}


def leave3(d, feature, target):
    """Drop the 3 WRs contributing most to the partial r in the expected (+) direction (as E008.influence)."""
    pr = linkage.partial_residuals(d, feature, target)
    c = (pr["x_res"] * pr["y_res"]).to_numpy()
    ids = pr["nfl_id"].to_numpy()[np.argsort(-c)[:3]].tolist()
    r = linkage.evaluate(d.filter(~pl.col("nfl_id").is_in(ids)), feature, target, expected_sign=1, seed=0,
                         n_jobs=3, n_repeats=2, n_perm=20)
    return {"dropped_nfl_id": ids, "n": r["n"], "partial_r": r["partial_r"], "partial_r_ci95": r["partial_r_ci95"],
            "fl_p": r["fl_p"]}


out = {"label": "rev-3 re-scoring (A: primary as computed; B: prespecified sanity) and post hoc (C)",
       "linkage_protocol": "working-tree src/bdb27/linkage.py (rev 3: checks.loco_classes_pass, sign-aware excess_lb_gt_0)",
       "settings": EVAL}

# A. E006-P1 as computed (AS_R = 20), seeds 0-2
dA = table("E006-P1")
out["A_E006_P1_r20"] = {}
for seed in (0, 1, 2):
    r = linkage.evaluate(dA, "feature", "target", expected_sign=1, seed=seed, **EVAL)
    if seed == 0:
        assert r["n"] == tm["E006-P1"]["n"] and abs(r["partial_r"] - tm["E006-P1"]["partial_r"]) < 1e-9, r["partial_r"]
    out["A_E006_P1_r20"][str(seed)] = slim(r)
    log(f"A seed {seed}: n={r['n']} pr={r['partial_r']:+.3f} p={r['fl_p']:.4f} checks={r['checks']} "
        f"loco={ {k: (round(v['excess'], 3), round(v['null_sd'], 3)) for k, v in r['loco'].items()} }")

# B. E001-WR
dB = table("E001-WR")
r = linkage.evaluate(dB, "feature", "target", expected_sign=1, seed=0, **EVAL)
assert r["n"] == tm["E001-WR"]["n"] and abs(r["partial_r"] - tm["E001-WR"]["partial_r"]) < 1e-9, r["partial_r"]
out["B_E001_WR"] = {"0": slim(r)}
log(f"B: n={r['n']} pr={r['partial_r']:+.3f} p={r['fl_p']:.4f} checks={r['checks']}")

# C. E006-P1 on S0 with AS_R = 1000 (shared module), post hoc
ids = dA["nfl_id"].to_list()
fr = (pl.scan_parquet(speed.GAME_FRAMES).filter(pl.col("nfl_id").is_in(ids))
      .select("game_id", "play_id", "nfl_id", "speed", "a_tan").collect())
log(f"C: frames {fr.height:,} rows for {len(ids)} WRs")
prof = speed.game_as_profiles(fr).select("nfl_id", s0_r1000="S0")
del fr
dC = dA.join(prof, on="nfl_id", how="left")
r = linkage.evaluate(dC, "feature", "s0_r1000", expected_sign=1, seed=0, **EVAL)
ref = e8["post_hoc_e006_r1000"]["E006-P1"]["result"]
assert r["n"] == ref["n"] and abs(r["partial_r"] - ref["partial_r"]) < 1e-6, (r["partial_r"], ref["partial_r"])
out["C_E006_P1_r1000"] = {"0": slim(r)}
log(f"C: n={r['n']} pr={r['partial_r']:+.3f} p={r['fl_p']:.4f} checks={r['checks']}")
no25 = linkage.evaluate(dC.filter(pl.col("draft_year") != 2025), "feature", "s0_r1000", expected_sign=1, seed=0,
                        n_jobs=3, n_repeats=2, n_perm=20)
out["C_E006_P1_r1000"]["without_2025"] = {k: no25[k] for k in ("n", "partial_r", "partial_r_ci95", "fl_p")}
out["C_E006_P1_r1000"]["leave_3_out"] = leave3(dC, "feature", "s0_r1000")
log(f"C no2025 {out['C_E006_P1_r1000']['without_2025']}; drop3 {out['C_E006_P1_r1000']['leave_3_out']}")
out["runtime_s"] = time.time() - T0
OUT.write_text(json.dumps(out, indent=1, default=float))
log("done")
