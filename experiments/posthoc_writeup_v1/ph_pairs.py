"""POST HOC (writeup v1, after the round-3 mock judging). Uncertainty for E007's 172 same-40 pairs statistic.

Reproduces E007.json concept_pair.post_hoc_all_matched_pairs (172 pairs, share 0.593) from E007_leaderboard.json, then adds:
  - a receiver-cluster bootstrap 95% CI for the share (pairs share receivers, so pairs are not independent),
  - a one-sided permutation p (shuffle hidden speed across receivers, keep the pair structure),
  - the hook pair's rank among the 172 pairs (by hidden-speed gap) and the percentile of its game gap.
Writes ph_pairs.json next to this script. Descriptive; not a hypothesis test of the protocol.
"""
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2] / "experiments" / "results"
OUT = Path(__file__).with_name("ph_pairs.json")
lb = json.loads((R / "E007_leaderboard.json").read_text())
e7 = json.loads((R / "E007.json").read_text())
ref = e7["concept_pair"]["post_hoc_all_matched_pairs"]

P = [p for p in lb["players"] if p["game_top_speed_mph"] is not None and (p["game_plays"] or 0) >= 50]
hid = np.array([p["hidden_speed_mph"] for p in P])
gam = np.array([p["game_top_speed_mph"] for p in P])
forty = np.array([p["official_forty"] for p in P])
pairs = [(a, b) for a, b in itertools.combinations(range(len(P)), 2) if abs(forty[a] - forty[b]) <= 0.01 + 1e-9]


def _agree(h, a, b):
    """E007 definition: the receiver with more hidden speed has a strictly higher game q95 (a game tie counts as no)."""
    hi, lo = (a, b) if h[a] > h[b] else (b, a)
    return gam[hi] - gam[lo] > 0


def share(h, pr=pairs):
    return float(np.mean([_agree(h, a, b) for a, b in pr]))


obs = share(hid)
assert len(pairs) == ref["n_pairs"], (len(pairs), ref["n_pairs"])
assert abs(obs - ref["share_more_hidden_speed_has_higher_game_q95"]) < 1e-9
assert not any(hid[a] == hid[b] for a, b in pairs)  # no hidden-speed ties; one game tie (counted as no, as in E007)

rng = np.random.default_rng(20261015)
B = 10000
agree = np.array([_agree(hid, a, b) for a, b in pairs], dtype=float)
ia = np.array([a for a, _ in pairs])
ib = np.array([b for _, b in pairs])
bs = np.empty(B)
for i in range(B):
    w = np.bincount(rng.integers(0, len(P), len(P)), minlength=len(P)).astype(float)
    wt = w[ia] * w[ib]  # a pair appears once per combination of its two receivers' copies
    bs[i] = np.sum(wt * agree) / np.sum(wt) if wt.sum() > 0 else np.nan
bs = bs[np.isfinite(bs)]
null = np.array([share(rng.permutation(hid)) for _ in range(B)])

gap_h = np.array([abs(hid[a] - hid[b]) for a, b in pairs])
gap_g = np.array([(gam[a] - gam[b]) * (1 if hid[a] > hid[b] else -1) for a, b in pairs])  # game gap, hi minus lo hidden
k = int(np.argmax(gap_h))
hook_ids = {P[pairs[k][0]]["nfl_id"], P[pairs[k][1]]["nfl_id"]}
res = {
    "label": "post hoc (writeup v1, after mock judging); descriptive; same WRs and target as E007-P1",
    "n_receivers_with_game_q95": len(P),
    "n_pairs": len(pairs),
    "n_receivers_in_pairs": len(set(ia) | set(ib)),
    "max_pairs_per_receiver": max(Counter(list(ia) + list(ib)).values()),
    "share": obs,
    "share_ci95_receiver_bootstrap": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
    "p_bootstrap_share_le_05": float(np.mean(bs <= 0.5)),
    "perm_p_one_sided": float((1 + np.sum(null >= obs)) / (B + 1)),
    "n_boot": int(len(bs)), "n_perm": B, "seed": 20261015,
    "mean_game_gap_mph": float(np.mean(gap_g)),
    "hook_pair_is_largest_hidden_gap": hook_ids == {e7["concept_pair"]["fast"]["nfl_id"], e7["concept_pair"]["slow"]["nfl_id"]},
    "hook_pair_game_gap_mph": float(gap_g[k]),
    "hook_pair_game_gap_percentile": float(np.mean(gap_g <= gap_g[k])),
    "hook_fast_rank_hidden_of_91": int(1 + sum(p["hidden_speed_mph"] > [q for q in lb["players"] if q["nfl_id"] == e7["concept_pair"]["fast"]["nfl_id"]][0]["hidden_speed_mph"] for p in lb["players"])),
}
assert abs(res["mean_game_gap_mph"] - ref["mean_game_gap_mph"]) < 1e-9
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
