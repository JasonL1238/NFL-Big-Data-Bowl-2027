"""E001 — pipeline sanity link: Combine 40-yd peak speed -> regular-season game top speed (q95 of play max).

This must work, or the pipeline is broken. Beyond B0 it should be large; beyond B2 (which already
contains the 40 time and 10-yd split) a small or zero gain is expected and fine.
"""

import json
from pathlib import Path

import polars as pl

from bdb27 import features, linkage

OUT = Path(__file__).parent / "results" / "E001.json"

combine = features.combine_top_speed()
game = features.game_top_speed()
features.update_store(combine, "combine_features")
features.update_store(game, "game_traits")

df = linkage.player_table().join(combine, on="nfl_id").join(game, on="nfl_id")
results = {"pooled": linkage.evaluate(df, "combine_top_speed", "game_top_speed", expected_sign=1)}
for pos in sorted(df["position"].unique()):
    results[pos] = linkage.evaluate(df.filter(pl.col("position") == pos), "combine_top_speed", "game_top_speed",
                                    expected_sign=1)
results["sanity_forty_vs_combine_top_speed_r"] = df.select(pl.corr("forty", "combine_top_speed")).item()

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(results, indent=2))
for k, r in results.items():
    if isinstance(r, dict):
        pr = "n/a" if r["partial_r"] is None else f"{r['partial_r']:+.2f} {[round(c, 2) if c is not None else None for c in r['partial_r_ci95']]}"
        print(f"{k:6s} n={r['n']:3d} raw_r={r['raw_r']:.2f} partial_r={pr} fl_p={'n/a' if r['fl_p'] is None else format(r['fl_p'], '.4f')} "
              f"dR2_vs_B2={r['delta_r2_vs_B2']['mean']:+.3f}±{r['delta_r2_vs_B2']['se']:.3f} "
              f"(excess {r['delta_r2_vs_B2']['excess']:+.3f}) cv_p={r['cv_perm_p']:.4f} "
              f"loco_excess+={r['checks']['loco_excess_positive_classes']} "
              f"cvR2 B0/B1/B2/full={r['cv_r2']['B0']:.2f}/{r['cv_r2']['B1']:.2f}/{r['cv_r2']['B2']:.2f}/{r['cv_r2']['full']:.2f}")
print("corr(forty, combine_top_speed) =", round(results["sanity_forty_vs_combine_top_speed_r"], 3))
