"""Post hoc export (writeup v2): every prior for E012's preregistered case-study players, plus the
pool sizes the hook cites. Reads E012 outputs only (E012.json and the gitignored forecasts parquet it
wrote); computes no new statistics beyond differences and squared errors of stored values."""

import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("e012_case_priors.json")

e012 = json.loads((ROOT / "experiments/results/E012.json").read_text())
fc = pl.read_parquet(ROOT / "data/processed/e012_forecasts_2025.parquet")
cases = e012["deliverables"]["case_studies"]

out = {"source": "E012.json deliverables.case_studies + data/processed/e012_forecasts_2025.parquet (k=0 rows)",
       "units": "trait units as stored (WR top speed: yd/s)", "cases": [], "pools": {}}
for c in cases:
    row = fc.filter((pl.col("trait") == c["trait"]) & (pl.col("nfl_id") == c["nfl_id"]) & (pl.col("k") == 0))
    assert row.height == 1, (c["trait"], c["nfl_id"], row.height)
    r = row.row(0, named=True)
    assert abs(r["mu_C2"] - c["prior_C2"]) < 1e-9 and abs(r["mu_C0"] - c["prior_C0"]) < 1e-9
    priors = {f"prior_{s}": r[f"mu_{s}"] for s in ("C0", "C1", "C2", "C3")}
    out["cases"].append({"trait": c["trait"], "nfl_id": c["nfl_id"], "season_mean": c["season_mean"], **priors,
                         **{f"sq_err_{s}": (r[f"mu_{s}"] - c["season_mean"]) ** 2 for s in ("C0", "C1", "C2", "C3")},
                         "tracking_shift_C3_minus_C2": r["mu_C3"] - r["mu_C2"]})
for t in sorted(fc["trait"].unique()):
    k0 = fc.filter((pl.col("trait") == t) & (pl.col("k") == 0))
    out["pools"][t] = {"n_test_players_k0": k0.height, "n_with_10plus_games": int((k0["n_games"] >= 10).sum())}
OUT.write_text(json.dumps(out, indent=1))
print(json.dumps({"case0": out["cases"][0], "T1": out["pools"]["T1"]}, indent=1))
