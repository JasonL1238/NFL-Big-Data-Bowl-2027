"""POST HOC (writeup v1; answers the mock judges' 'usable week to week?' gap). Descriptive; not a protocol test.

Question: once a receiver starts playing, how many games of his own game tracking does it take before they forecast his
later game top speed better than his Combine does?

Target (same definition as E001/E007-P1, bdb27.features.game_top_speed): per REG play, max tracked speed `s`; plays
above 12 yd/s dropped; a player's value = the 95th percentile of his play maxima.
Population: Combine WRs (linkage.player_table position == 'WR'). A 'game' = a REG game in which he has >= MIN_PLAYS
tracked plays, in chronological order from his first such game (rookie season first).
Fixed sample: WRs with >= M_MAX such games AND >= MIN_LATER tracked plays after his M_MAX-th game.
  Y   = q95 over all plays after the M_MAX-th game (the 'later' top speed; disjoint from every X_m).
  X_m = q95 over the plays of his first m games, m = 1..M_MAX.
Combine comparators on the same WRs: official 40 time (sign flipped), tracked flying 20 (E007 feature, from
E007_leaderboard.json flying20_mph), tracked 40 peak speed (combine_features.parquet combine_top_speed).
Reports Pearson r with Fisher 95% CIs, and a receiver bootstrap of r(X_m, Y) - r(best Combine measure, Y).
Validation: full-career q95 reproduces E007_leaderboard.json game_top_speed_mph.
"""
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from bdb27 import features, linkage

ROOT = Path("/Users/jasonli/Documents/GitHub/NFL-Big-Data-Bowl-2027")
OUT = Path(__file__).with_name("ph_inseason.json")
MIN_PLAYS, M_MAX, MIN_LATER = 10, 8, 50
MPH = 3600 / 1760
T0 = time.time()

wr = linkage.player_table().filter(pl.col("position") == "WR").select("nfl_id", "draft_year", "forty")
ids = wr["nfl_id"].to_list()
play_max = (features.reg_season_tracking().filter(pl.col("nfl_id").is_in(ids))
            .group_by("game_id", "play_id", "nfl_id").agg(smax=pl.col("s").max())
            .filter(pl.col("smax") <= features.MAX_PLAUSIBLE_SPEED).collect())
print(f"[{time.time() - T0:.0f}s] play maxima: {play_max.height:,} plays, {play_max['nfl_id'].n_unique()} WRs", flush=True)

# validation against E007 leaderboard (career q95, >= 50 plays)
lb = {p["nfl_id"]: p for p in json.loads((ROOT / "experiments/results/E007_leaderboard.json").read_text())["players"]}
car = play_max.group_by("nfl_id").agg(q=pl.col("smax").quantile(0.95), n=pl.len()).filter(pl.col("n") >= 50)
chk = [(r["q"] * MPH, lb[r["nfl_id"]]["game_top_speed_mph"]) for r in car.iter_rows(named=True)
       if r["nfl_id"] in lb and lb[r["nfl_id"]]["game_top_speed_mph"] is not None]
max_dev = max(abs(a - b) for a, b in chk)
assert max_dev < 1e-6, max_dev
print(f"validation: {len(chk)} WRs reproduce E007 game_top_speed_mph (max dev {max_dev:.2e})")

# per-game play counts; qualifying games in chronological order (game_id encodes the date)
gp = play_max.group_by("nfl_id", "game_id").agg(n=pl.len()).filter(pl.col("n") >= MIN_PLAYS).sort("nfl_id", "game_id")
gp = gp.with_columns(k=pl.col("game_id").rank("ordinal").over("nfl_id").cast(pl.Int64))
kth = gp.filter(pl.col("k") == M_MAX).select("nfl_id", g_cut="game_id")
pm = play_max.join(kth, on="nfl_id", how="inner")
later = pm.filter(pl.col("game_id") > pl.col("g_cut")).group_by("nfl_id").agg(Y=pl.col("smax").quantile(0.95), n_later=pl.len())
sample = later.filter(pl.col("n_later") >= MIN_LATER).select("nfl_id", "Y", "n_later")
rows = {"nfl_id": sample["nfl_id"].to_list()}
for m in range(1, M_MAX + 1):
    first = gp.filter(pl.col("k") <= m).select("nfl_id", "game_id")
    xm = (play_max.join(first, on=["nfl_id", "game_id"], how="semi").group_by("nfl_id")
          .agg(pl.col("smax").quantile(0.95).alias(f"X{m}"), pl.len().alias(f"nX{m}")))
    sample = sample.join(xm, on="nfl_id", how="left")
cf = pl.read_parquet(ROOT / "data/processed/combine_features.parquet")
fly = pl.DataFrame({"nfl_id": list(lb), "flying20": [lb[i]["flying20_mph"] for i in lb]}, schema={"nfl_id": pl.Int64, "flying20": pl.Float64})
sample = (sample.join(wr, on="nfl_id", how="left").join(cf, on="nfl_id", how="left").join(fly, on="nfl_id", how="left")
          .with_columns(neg_forty=-pl.col("forty")))
full = sample.drop_nulls(["neg_forty", "flying20", "combine_top_speed"])


def r_ci(a, b):
    r = float(np.corrcoef(a, b)[0, 1])
    z, se = np.arctanh(r), 1 / np.sqrt(len(a) - 3)
    return {"r": r, "ci95": [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))], "n": int(len(a))}


Y = full["Y"].to_numpy()
res = {"label": "post hoc (writeup v1, after mock judging); descriptive", "definitions": __doc__,
       "min_plays_per_game": MIN_PLAYS, "m_max": M_MAX, "min_later_plays": MIN_LATER,
       "validation_n_wr_reproduced": len(chk),
       "n_sample_all": sample.height, "n_sample_with_all_combine": full.height,
       "by_class": full.group_by("draft_year").len().sort("draft_year").to_dicts(),
       "median_plays_per_game_first_m": {}, "game_first_m": {}, "combine": {}}
for c in ("neg_forty", "flying20", "combine_top_speed"):
    res["combine"][c] = r_ci(full[c].to_numpy(), Y)
best = max(res["combine"], key=lambda c: res["combine"][c]["r"])
res["best_combine"] = best
rng = np.random.default_rng(20261015)
B = 10000
idx = rng.integers(0, full.height, (B, full.height))
cb = full[best].to_numpy()
for m in range(1, M_MAX + 1):
    x = full[f"X{m}"].to_numpy()
    d = r_ci(x, Y)
    diffs = np.array([np.corrcoef(x[i], Y[i])[0, 1] - np.corrcoef(cb[i], Y[i])[0, 1] for i in idx])
    d["minus_best_combine"] = float(d["r"] - res["combine"][best]["r"])
    d["minus_best_combine_ci95_boot"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    d["p_boot_le_0"] = float(np.mean(diffs <= 0))
    res["game_first_m"][str(m)] = d
    res["median_plays_per_game_first_m"][str(m)] = float((full[f"nX{m}"] / m).median())
res["median_later_plays"] = float(full["n_later"].median())
res["runtime_s"] = time.time() - T0
OUT.write_text(json.dumps(res, indent=1, default=str))
print(json.dumps({k: v for k, v in res.items() if k != "definitions"}, indent=1, default=str))
