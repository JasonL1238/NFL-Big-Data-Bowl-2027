"""Regenerate every number, table and figure in writeup/draft.md from the raw competition data.

This is the pipeline behind notebooks/final_writeup.ipynb (the public Kaggle notebook). The notebook calls these
functions cell by cell; the same run works from the command line:

    python notebooks/run_all.py --workspace DIR                # every step, in dependency order (resumable)
    python notebooks/run_all.py --workspace DIR --stage A      # one Kaggle stage (notebooks/README.md)
    python notebooks/run_all.py --workspace DIR --compare      # only the final comparison
    python notebooks/run_all.py bundle --out DIR               # the code bundle uploaded as a Kaggle dataset

What it does:
  1. Environment: CPython 3.13.5 with the exact versions in notebooks/requirements-kaggle.txt. On Kaggle it builds
     that interpreter with uv; anything else fails loudly before a step runs.
  2. Workspace: a fresh copy of the repository's code and its frozen inputs (preregistrations, E012 audit files),
     plus links to the raw CSVs. Every script derives its paths from its own location, so all outputs land in the
     workspace and the committed experiments/results is only ever read, by the final comparison.
  3. Steps S01-S14 in dependency order, each in its own process, with a log, wall time, CPU time and peak memory.
     A finished step is recorded in _run/progress.json and skipped on a rerun, so an interrupted run resumes.
  4. Comparison: every regenerated results JSON against the committed one, leaf by leaf (compare()), and E008's
     constants transcribed from round 1 against their committed source (transcribed_constants()).

No statistical setting is changed: every script runs with its committed permutation, bootstrap, simulation and CV
counts and seeds. A few scripts needed small, non-statistical compatibility edits to run on a fresh machine. They are
committed to the scripts and only verified here (R1_KEYS and COMMITTED_EDITS below), so the workspace copies are byte
copies of the repository's scripts.

This file uses only the standard library, so it runs in any notebook kernel; the analysis runs in the pinned
interpreter.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PYTHON_VERSION = "3.13.5"
UV_VERSION = "0.12.20"  # released 2026-09-28; only used to build the pinned interpreter on Kaggle
COMPETITION = "nfl-big-data-bowl-2027"
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
HERE = Path(__file__).resolve().parent

RAW_FILES = ("players.csv", "combine_results.csv", "player_career_successes.csv", "games.csv", "player_play.csv",
             "combine_tracking.csv", "game_tracking_2023.csv", "game_tracking_2024.csv", "game_tracking_2025.csv")
# Row counts verified with a CSV parser on 2026-10-14 (AGENTS.md). A different data release fails step S01.
ROW_COUNTS = {"players": 510, "combine_results": 510, "player_career_successes": 510, "games": 1002,
              "player_play": 314197, "combine_tracking": 463189, "game_tracking (2023-25)": 22654290}

# ------------------------------------------------------------------------------------------------ steps
CACHE = f"""
import polars as pl
from bdb27 import io
io.build_cache()
n = {{t: pl.scan_parquet(io.PROCESSED / f"{{t}}.parquet").select(pl.len()).collect().item()
     for t in ("players", "combine_results", "player_career_successes", "games", "player_play", "combine_tracking")}}
n["game_tracking (2023-25)"] = sum(pl.scan_parquet(io.PROCESSED / f"game_tracking_{{s}}.parquet").select(pl.len())
                                  .collect().item() for s in io.SEASONS)
want = {ROW_COUNTS!r}
for k, v in want.items():
    print(f"{{k:28s}} {{n[k]:>12,}} rows (expected {{v:,}})")
assert n == want, "the raw data differ from the release this analysis used"
"""
FRAMES = "from bdb27 import speed; print(speed.build_game_frames().shape)"
# The writeup's Code line promises trait definitions and position averages; no results key holds the averages, so
# they are computed here from E012's own analysis sample (asserted to match its player counts). Not quoted in the text.
TRAITS = """
import json
import polars as pl
from bdb27.probes import h_game_equivalents as H
e = json.load(open("experiments/results/E012.json"))
pre = {t["id"]: t for t in json.load(open("experiments/results/E012_prereg.json"))["traits"]}
ct = {r["trait"]: r for r in e["cge_table"]}
games = pl.read_parquet("data/processed/e012_game_values.parquet")
players = pl.read_parquet("data/processed/e012_players.parquet").select("nfl_id", "position", "draft_year")
rows = []
for T, t in e["traits"].items():
    y = (games.filter(pl.col("trait") == T).join(players, on="nfl_id").filter(pl.col("position") == t["position"])
         .group_by("nfl_id", "draft_year").agg(y=pl.col("value").mean(), n=pl.len()).filter(pl.col("n") >= H.MIN_GAMES))
    by = {str(k): v for k, v in y.group_by("draft_year").agg(pl.len()).rows()}
    assert y.height == ct[T]["n_players"] and by == t["by_class"], (T, y.height, by)
    cls = {str(k): v for k, v in y.group_by("draft_year").agg(pl.col("y").mean()).sort("draft_year").rows()}
    rows.append({"trait": T, "position": t["position"], "name": t["name"], "unit": t["unit"],
                 "definition": pre[T]["game_value"], "min_plays_per_game": t["min_plays_per_game"],
                 "priced": ct[T]["identified"], "n_players": y.height, "average": y["y"].mean(), "class_means": cls})
    print(f"{T:4s} {t['position']} {t['name']:45s} n={y.height:3d} average={y['y'].mean():+.4f} {t['unit']}")
json.dump(rows, open("_run/trait_table.json", "w"), indent=1)
"""


@dataclass(frozen=True)
class Step:
    id: str
    stage: str
    title: str
    argv: tuple  # arguments to the pinned interpreter, run from the workspace root
    writes: str  # the outputs a reader cares about


STEPS = (
    Step("S01", "A", "Parquet cache of the raw CSVs", ("-c", CACHE),
         "data/processed/{players,...,game_tracking_*}.parquet"),
    Step("S02", "A", "Post-snap game frames", ("-c", FRAMES), "data/processed/game_frames_post_snap.parquet"),
    Step("S03", "A", "E001 feature store and sanity link",
         ("experiments/E001_combine_top_speed_to_game_top_speed.py",), "E001.json, combine_features, game_traits"),
    Step("S04a", "A", "E002 WR route breaks (round 1)", ("experiments/E002_route_break_mechanics.py",),
         "E002.json, probe_a_*.parquet"),
    Step("S04b", "A", "E003 DL burst (round 1)", ("experiments/E003_dl_burst.py",), "E003.json, probe_b_*.parquet"),
    Step("S04c", "A", "E004 OL lateral movement (round 1)", ("experiments/E004_ol_lateral_to_pass_pro.py",),
         "E004.json, E004_reliability.json, probe_c_*.parquet"),
    Step("S04d", "A", "E005 DB backpedal transition (round 1)", ("experiments/E005_db_backpedal_transition.py",),
         "E005.json, probe_d_*.parquet"),
    Step("S04e", "A", "E006 sprint profile (round 1)", ("experiments/E006_sprint_profile.py",),
         "E006.json, probe_e_*.parquet"),
    Step("S05", "A", "E007 hidden top speed", ("experiments/E007_hidden_speed.py",),
         "E007.json, E007_leaderboard.json, e007_*.parquet"),
    Step("S06", "A", "E010 drill intensity", ("experiments/E010_drill_intensity.py",), "E010.json, e010_*.parquet"),
    Step("S07", "A", "E011 Combine ball tracking", ("experiments/E011_catch_point.py",), "E011.json, e011_*.parquet"),
    Step("S08", "B", "E008 drill reliability and transfer map", ("experiments/E008_transfer_map.py",),
         "E008.json, e008_*.parquet"),
    Step("S09", "B", "E009 transfer map against two baselines", ("experiments/E009_beyond_stopwatch.py",),
         "E009.json, e009_*.parquet"),
    Step("S10", "C", "E012 Combine Game Equivalents", ("experiments/E012_game_equivalents.py",),
         "E012.json, e012_*.parquet"),
    Step("S11a", "D", "Case-study export (post hoc)", ("experiments/posthoc_writeup_v2/e012_case_priors.py",),
         "e012_case_priors.json"),
    Step("S11b", "D", "Same-40 pairs export (post hoc)", ("experiments/posthoc_writeup_v1/ph_pairs.py",),
         "ph_pairs.json"),
    Step("S12", "D", "Writeup figures", ("writeup/make_figures.py",), "writeup/figures/fig1-fig5"),
    Step("S13", "D", "Number checker on the draft", ("writeup/check_numbers.py",),
         "exit code 0 = every quoted number matches"),
    Step("S14", "D", "Trait definitions and position averages", ("-c", TRAITS), "_run/trait_table.json"),
)
STEP = {s.id: s for s in STEPS}
# S01/S02 rebuild the raw tables and per-frame tracking in Parquet. A saved Kaggle stage output never contains them
# (the competition data must not be republished), so every stage rebuilds them first (seconds to minutes).
# probe_e_game_frames (E006's per-frame cache) is also per-frame tracking; no later step reads it.
CACHE_STEPS = ("S01", "S02")
CACHE_FILES = tuple(f"{t}.parquet" for t in ("players", "combine_results", "player_career_successes", "games",
                                              "player_play", "combine_tracking", "game_tracking_2023",
                                              "game_tracking_2024", "game_tracking_2025", "game_frames_post_snap",
                                              "probe_e_game_frames"))
STAGES = {"A": "features, round-1 probes, E007, E010, E011", "B": "E008, E009", "C": "E012",
          "D": "exports, figures, number checker, comparison"}
UPSTREAM = {"A": "", "B": "A", "C": "A", "D": "ABC"}  # B and C can run at the same time

# ------------------------------------------------------------------------------------------------ patches
# Non-statistical compatibility edits. R1_KEYS and COMMITTED_EDITS are committed to the repository scripts
# (2026-10-17), so they are only verified: each workspace copy (a byte copy of the script) must no longer contain
# the old text and must contain the new text exactly the expected number of times, or the run stops.
# LITERAL_PATCHES (none at present) would be applied to the workspace copies only and printed in _run/patches.diff:
# each must match exactly the expected number of times; if the repository script was already fixed (no match,
# replacement present) the patch is a no-op; anything else stops the run.
R1_KEYS = (  # rev-1 harness result keys -> the names linkage.evaluate() returns since rev 2/3 (as quoted strings)
    ("perm_p", "cv_perm_p"),
    ("perm_p_lt_05", "fl_p_lt_05"),
    ("loco_delta_r2", "loco"),
    ("loco_positive_classes", "loco_excess_positive_classes"),
)
R1_COUNTS = {  # occurrences of each new R1_KEYS key in the committed script (the old keys: none)
    "experiments/E002_route_break_mechanics.py": (3, 0, 0, 2),
    "experiments/E003_dl_burst.py": (7, 1, 2, 1),  # one of the two "loco" keys predates the renames
    "experiments/E004_ol_lateral_to_pass_pro.py": (6, 0, 1, 2),
    "experiments/E005_db_backpedal_transition.py": (5, 1, 1, 2),
    "experiments/E006_sprint_profile.py": (4, 1, 1, 2),
}
COMMITTED_EDITS = (  # (file, old text (now absent), new text, expected count of the new text, why)
    ("experiments/E005_db_backpedal_transition.py", "linkage._fit_resid(",
     '    xr = linkage._residual_maker(Xi) @ d[spec["feature"]].to_numpy().astype(float)\n'
     '    yr = linkage._residual_maker(Xi) @ d[spec["target"]].to_numpy().astype(float)\n', 1,
     "linkage._fit_resid was removed in rev 2; _residual_maker(X) @ v gives the same OLS residuals (figure only)"),
    ("experiments/E009_beyond_stopwatch.py", '                                    ref["loco"])])',
     '                                    {y: {"delta": u["delta"], "excess": u["excess"]} '
     'for y, u in ref["loco"].items()})])', 1,
     "E009 asserts that its rerun reproduces E008.json's leave-one-class-out delta and excess. The committed "
     "E008.json predates harness rev 3; a regenerated one also carries the rev-3 key null_sd, so the key sets "
     "differed and the check failed on structure. Both sides now keep the same two fields; no value changes"),
    ("experiments/E011_catch_point.py",
     'PLAN = Path("/private/tmp/claude-501/-Users-jasonli-Documents-GitHub-NFL-Big-Data-Bowl-2027/"\n'
     '            "ca6dc0b9-7e4f-45b4-b4e4-1468819c8969/scratchpad/round3/E011_prereg_scratch.json")',
     'PLAN = RES / "E011_prereg.json"', 1,
     "E011 read its frozen plan from a temporary scratch file; the same plan, stored verbatim as "
     "E011.json['plan'], is committed as experiments/results/E011_prereg.json (setup checks the two are equal)"),
    ("experiments/posthoc_writeup_v1/ph_pairs.py",
     'R = Path("/Users/jasonli/Documents/GitHub/NFL-Big-Data-Bowl-2027/experiments/results")',
     'R = Path(__file__).resolve().parents[2] / "experiments" / "results"', 1,
     "absolute path to one machine's repository; read the workspace's own results instead"),
    ("experiments/E012_game_equivalents.py", 'f = SCRATCH / "implementation_choices_frozen.json"',
     'f = RES / "E012_audit" / "implementation_choices_frozen.json"', 1,
     "the freeze file E012 checks its implementation choices against moved from a scratch folder to the "
     "committed experiments/results/E012_audit/ (same SHA-256)"),
)
LITERAL_PATCHES = ()  # (file, old, new, expected count, why); none at present
R1_WHY = ("round-1 script written against the rev-1 harness keys (perm_p, perm_p_lt_05, loco_delta_r2, "
          "loco_positive_classes); renamed to the keys linkage.evaluate() returns now. Only round-1 bookkeeping "
          "reads them; E008 uses these scripts' n, raw r and game-side reliabilities, which no key name affects")

# Files copied into a fresh workspace (relative to the bundle root).
CODE_GLOBS = ("src/bdb27/**/*.py", "experiments/*.py", "experiments/posthoc_writeup_v1/ph_pairs.py",
              "experiments/posthoc_writeup_v2/e012_case_priors.py", "writeup/check_numbers.py",
              "writeup/make_figures.py", "writeup/draft.md")
FROZEN_GLOBS = ("experiments/results/*_prereg.json", "experiments/results/E012_audit/*")

# ------------------------------------------------------------------------------------------------ comparison
COMPARED = ("experiments/results/E001.json", "experiments/results/E002.json", "experiments/results/E003.json",
            "experiments/results/E004.json", "experiments/results/E004_reliability.json",
            "experiments/results/E005.json", "experiments/results/E006.json", "experiments/results/E007.json",
            "experiments/results/E007_leaderboard.json", "experiments/results/E008.json",
            "experiments/results/E009.json", "experiments/results/E010.json", "experiments/results/E011.json",
            "experiments/results/E012.json", "experiments/posthoc_writeup_v1/ph_pairs.json",
            "experiments/posthoc_writeup_v2/e012_case_priors.json")
NOT_REGENERATED = {
    "experiments/results/*_prereg.json": "frozen inputs (preregistrations), copied into the workspace",
    "experiments/results/E012_audit/*": "frozen audit record of E012 (amendment, freeze file, logs)",
    "experiments/results/E002_run1_eb_bug.json": "historical record of E002's first, buggy run; no script writes it",
    "experiments/posthoc_writeup_v1/ph_inseason.json, ph_rev3.json": "writeup v1 leftovers; draft v2.2 uses neither",
}
REL_TOL, ABS_TOL = 1e-9, 1e-12  # floats only: last-digit differences from BLAS/OpenMP summation order
# Values never printed (writeup/outline.md section 4): Receiver A's stopwatch-only forecast, draft status, names.
REDACT = re.compile(r"prior_C1|name|draft_(round|pick|status)|undrafted|college", re.I)

# Documented differences between a fresh and a committed JSON, by cause. Every other difference fails the comparison.
DOCUMENTED = {
    "volatile": "Run metadata: wall-clock runtime and timestamps.",
    "harness-rev1": (
        "Round-1 test results. E002-E006 were written for the rev-1 test harness, replaced before the first commit; "
        "the current harness adds draft-year terms to every baseline (rev 2), tests one-sided with Freedman-Lane "
        "permutations and judges leave-one-class-out against its null (rev 3), and renamed four result keys (the "
        "compatibility edit). Round-1 decisions stand as logged under rev 1 (experiments/log.md, GATE 1); E008 "
        "re-estimates every round-1 link under the current harness, and those are the numbers the writeup uses. "
        "Sample sizes, raw correlations and feature/target names are still compared exactly."),
    "harness-rev3": (
        "Harness rev 3 (src/bdb27/linkage.py, commit 598dc08) came after this file was committed. It added each "
        "class's leave-one-class-out null SD and the loco_classes_pass check, made the minimum detectable effect "
        "one-sided (partial_r_mde80 smaller), and made excess_lb_gt_0 also require the expected sign (copies of "
        "that flag follow it). E008 copies the harness value into transfer_map[*].mde80 and "
        "summary.transfer.mde80_two_sided; the writeup quotes E008's own mde80_one_sided, which is unchanged."),
    "upstream-hash": (
        "E009 records the SHA-256 of the E010.json it read. The fresh E010.json differs from the committed one only "
        "in last-digit floats (within tolerance), so its hash differs."),
    "reliability-rev": (
        "src/bdb27/reliability.py was revised in commit d11dee7, after the round-1 results were committed "
        "(7260d39): a negative ICC now gives reliability 0 instead of a negative Spearman-Brown value, and "
        "cross-drill ICCs drop rows alone in their centering group (new key n_dropped). These Combine-side "
        "blocks feed no writeup number; E008 recomputes every Combine reliability it uses and copies these values "
        "only into its unquoted comparison column drill_profile[*].round1_reliability."),
}
DEFECTS = set()  # rules that document a reproducibility bug: such a file shows PASS* and the verdict names it; none now
_ADDED = lambda x, y: y is KeyError and x is not KeyError  # noqa: E731  key new in the fresh file
_CLIPPED = lambda x, y: x == 0.0 and isinstance(y, (int, float)) and y < 0  # noqa: E731  negative -> 0
_ROUND1 = ("E002", "E003", "E004", "E005", "E006")
# (file, JSON-path regex, rule, optional check on (fresh, committed)); "*" = any file. A difference is documented
# only if its path matches and the check (when given) holds.
EXPECTED = (
    ("*", r"^/runtime_s$", "volatile", None),
    ("experiments/results/E012.json", r"^/(run_started_utc|rewritten_utc)$", "volatile", None),
    ("experiments/results/E009.json", r"^/round3_inputs/e010_sha256$", "upstream-hash",
     lambda x, y: isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x) is not None),
    *((f"experiments/results/{e}.json", r"^/(primary|exploratory)/(?!(.*/)?(n|raw_r|feature|target)$)",
       "harness-rev1", None) for e in _ROUND1),
    *((f"experiments/results/{e}.json", r"^/(exploratory_p_lt_05|exploratory_min_bh_q|primary_bh_q_seed0)(/|$)",
       "harness-rev1", None) for e in _ROUND1),
    *((f"experiments/results/{e}.json", r"/(null_sd|loco_classes_pass)$", "harness-rev3", _ADDED)
      for e in ("E001", "E007", "E008")),
    *((f"experiments/results/{e}.json", r"/partial_r_mde80$", "harness-rev3", lambda x, y: x < y)
      for e in ("E001", "E007", "E008")),
    ("experiments/results/E008.json", r"^/(transfer_map/\d+/mde80|summary/transfer/mde80_two_sided/\d+)$",
     "harness-rev3", lambda x, y: x < y),
    ("experiments/results/E008.json", r"^/drill_profile/\d+/round1_reliability$", "reliability-rev", None),
    ("experiments/results/E007.json",
     r"^/primary/P\d/(seeds/\d+/checks/excess_lb_gt_0|go/per_seed_criteria_1_3/\d+/\d+)$",
     "harness-rev3", lambda x, y: x is False and y is True),
    *((f"experiments/results/{f}", pat, "reliability-rev", _CLIPPED) for f, pat in (  # Combine blocks only
        ("E002.json", r"^/reliability/combine/\d+/composite_reliability$"),
        ("E003.json", r"^/combine_reliability/(attempt_icc|cross_drill)/[^/]+/resid_b2_reliability$"),
        ("E004.json", r"^/reliability/drop_across_attempts/\w+/reliability$"),
        ("E005.json", r"^/reliability/combine/\d+/reliability$"))),
    ("experiments/results/E005.json", r"^/reliability/combine/\d+/n_dropped$", "reliability-rev", _ADDED),
    *((f"experiments/results/{f}", rf"^{pre}/combine_within_attempt/\w+/\w+/(icc|reliability|n_dropped)$",
       "reliability-rev", None) for f, pre in (("E004.json", "/reliability"), ("E004_reliability.json", ""))),
    ("experiments/results/E004_reliability.json", r"^/drop_across_attempts/\w+/reliability$", "reliability-rev",
     _CLIPPED),
)


# ------------------------------------------------------------------------------------------------ helpers
def on_kaggle() -> bool:
    return KAGGLE_INPUT.is_dir() and (bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or KAGGLE_WORKING.is_dir())


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _fmt_s(s: float) -> str:
    return f"{s:.0f} s" if s < 90 else (f"{s / 60:.1f} min" if s < 5400 else f"{s / 3600:.2f} h")


def _walk_dirs(root: Path, depth: int):
    """Directories under root up to the given depth (breadth first, sorted, symlinks not followed)."""
    level = [root]
    for _ in range(depth + 1):
        nxt = []
        for d in level:
            yield d
            try:
                nxt += sorted(p for p in d.iterdir() if p.is_dir() and not p.is_symlink())
            except OSError:
                pass
        level = nxt


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy(src: Path, dst: Path) -> None:
    """Copy with metadata, then make the copy writable (Kaggle inputs are read-only; later steps rewrite logs)."""
    shutil.copy2(src, dst)
    dst.chmod(dst.stat().st_mode | 0o200)


def _write_json(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(p)


def find_bundle() -> Path:
    """The repository snapshot: this file's parent repo, or a Kaggle dataset holding the bundle."""
    root = HERE.parent
    if (root / "src" / "bdb27" / "__init__.py").exists():
        return root
    raise FileNotFoundError("run_all.py is not inside a repository snapshot (no src/bdb27 next to notebooks/)")


def find_data(explicit: str | None = None) -> Path:
    """The folder holding the nine competition CSVs: an explicit path, $BDB27_DATA, the Kaggle competition mount
    (possibly nested), or data/raw of the repository."""
    def ok(d: Path) -> bool:
        return all((d / f).is_file() for f in RAW_FILES)
    for c in (explicit, os.environ.get("BDB27_DATA")):
        if c:
            if not ok(Path(c)):
                raise FileNotFoundError(f"{c} does not contain all of {RAW_FILES}")
            return Path(c)
    if KAGGLE_INPUT.is_dir():
        roots = [p for p in (KAGGLE_INPUT / COMPETITION, KAGGLE_INPUT / "competitions" / COMPETITION) if p.is_dir()]
        for root in roots + [KAGGLE_INPUT]:
            for d in _walk_dirs(root, 4):
                if ok(d):
                    return d
        raise FileNotFoundError(f"no folder under {KAGGLE_INPUT} holds the competition CSVs: add the "
                                f"{COMPETITION} data to the notebook (Add Input -> Competitions)")
    d = find_bundle() / "data" / "raw"
    if not ok(d):
        raise FileNotFoundError(f"{d} lacks the competition CSVs; download them (README.md) or pass --data")
    return d


# ------------------------------------------------------------------------------------------------ environment
def read_pins(bundle: Path) -> dict:
    pins = {}
    for line in (bundle / "notebooks" / "requirements-kaggle.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            name, _, ver = line.partition("==")
            pins[name.strip()] = ver.strip()
    return pins


ENV_CHECK = """
import importlib.metadata as md, json, platform, sys
pins = json.loads(sys.argv[1]); want_py = sys.argv[2]
got = {k: (md.version(k) if any(True for _ in md.distributions(name=k)) else None) for k in pins}
bad = {k: (pins[k], got[k]) for k in pins if got[k] != pins[k]}
py = platform.python_version()
print(f"python {py} ({sys.executable})")
for k in sorted(pins):
    print(f"  {k:20s} {str(got[k]):14s}" + ("" if got[k] == pins[k] else f"  <-- expected {pins[k]}"))
assert py == want_py and not bad, f"environment mismatch: python {py} (want {want_py}); packages {bad}"
import bdb27, matplotlib, numpy, polars, scipy, sklearn, joblib, threadpoolctl, PIL  # noqa: the closure imports
print("bdb27 from", bdb27.__file__)
"""


def verify_env(py: Path, bundle: Path, ws: Path) -> str:
    """Run the pin check inside the pinned interpreter; raises on any mismatch."""
    env = _step_env(ws, 3)
    r = subprocess.run([str(py), "-c", ENV_CHECK, json.dumps(read_pins(bundle)), PYTHON_VERSION],
                       capture_output=True, text=True, env=env, cwd=ws)
    if r.returncode != 0:
        raise RuntimeError(f"pinned environment check failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout


def build_env(venv: Path, bundle: Path) -> Path:
    """CPython 3.13.5 + the pinned wheels in a fresh venv, built with uv (needs internet). Used on Kaggle, whose
    preinstalled Python and packages differ from ours."""
    py = venv / "bin" / "python"
    if py.exists():
        return py
    req = bundle / "notebooks" / "requirements-kaggle.txt"
    uv = [sys.executable, "-m", "uv"]
    cmds = ([sys.executable, "-m", "pip", "install", "--quiet", f"uv=={UV_VERSION}"],
            uv + ["python", "install", "--no-bin", PYTHON_VERSION],  # --no-bin: leave ~/.local/bin alone
            uv + ["venv", "--python", PYTHON_VERSION, "--python-preference", "only-managed", str(venv)],
            uv + ["pip", "install", "--python", str(py), "--no-deps", "--only-binary", ":all:", "-r", str(req)],
            uv + ["pip", "check", "--python", str(py)])
    for c in cmds:
        print("$", " ".join(c[2:] if c[:2] == [sys.executable, "-m"] else c), flush=True)
        r = subprocess.run(c, capture_output=True, text=True)
        if r.returncode != 0:
            shutil.rmtree(venv, ignore_errors=True)
            raise RuntimeError(f"environment build failed (is the notebook's internet on?):\n{r.stdout}\n{r.stderr}")
    return py


def default_python(bundle: Path) -> Path:
    if on_kaggle():
        return build_env(Path("/tmp/bdb27-venv"), bundle)
    venv = bundle / ".venv" / "bin" / "python"
    return venv if venv.exists() else Path(sys.executable)


def _step_env(ws: Path, workers: int) -> dict:
    env = dict(os.environ)
    for k in ("PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV"):
        env.pop(k, None)
    env.update(PYTHONPATH=str(ws / "src"), MPLBACKEND="Agg", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="3",  # as in the committed runs (E011's documented setting; LESSONS.md)
               LOKY_MAX_CPU_COUNT=str(workers))  # caps joblib's n_jobs=-1 (E001); every other script fixes n_jobs=3
    return env


# ------------------------------------------------------------------------------------------------ workspace
@dataclass
class Context:
    bundle: Path
    ws: Path
    data: Path
    py: Path
    stage: str
    workers: int
    steps: tuple = field(default_factory=tuple)

    @property
    def run_dir(self) -> Path:
        return self.ws / "_run"


def _files(root: Path, globs) -> list:
    out = set()
    for g in globs:
        out |= {p for p in root.glob(g) if p.is_file() and "__pycache__" not in p.parts}
    return sorted(out)


def code_fingerprint(bundle: Path) -> str:
    """Hash of everything that determines the workspace's results: code, frozen inputs, pins and patches."""
    h = hashlib.sha256(repr((R1_KEYS, R1_COUNTS, COMMITTED_EDITS, LITERAL_PATCHES)).encode())
    for p in _files(bundle, CODE_GLOBS + FROZEN_GLOBS + ("notebooks/requirements-kaggle.txt",)):
        h.update(str(p.relative_to(bundle)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def apply_patches(ws: Path) -> list:
    """Verify the committed edits (R1_COUNTS, COMMITTED_EDITS) in the workspace copies and apply LITERAL_PATCHES to
    them; return [(file, status, why, diff)]."""
    done = []
    edits = {}
    for f, counts in R1_COUNTS.items():
        edits.setdefault(f, []).append(("r1", counts))
    for f, old, new, n, why in COMMITTED_EDITS:
        edits.setdefault(f, []).append(("committed", (old, new, n, why)))
    for f, old, new, n, why in LITERAL_PATCHES:
        edits.setdefault(f, []).append(("lit", (old, new, n, why)))
    for f, todo in edits.items():
        p = ws / f
        before = text = p.read_text()
        whys = []
        for kind, spec in todo:
            if kind == "r1":
                old_k = [len(re.findall(rf"(['\"]){a}\1", text)) for a, _ in R1_KEYS]
                new_k = [len(re.findall(rf"(['\"]){b}\1", text)) for _, b in R1_KEYS]
                if any(old_k) or tuple(new_k) != spec:
                    raise RuntimeError(f"{f}: lacks the committed rev-1 key renames (old keys found {old_k}, new keys "
                                       f"{new_k}, expected {list(spec)}); review R1_COUNTS")
                whys.append("committed in the repository: " + R1_WHY)
            elif kind == "committed":
                old, new, n, why = spec
                if old in text or text.count(new) != n:
                    raise RuntimeError(f"{f}: lacks the committed edit (found {text.count(old)} of {old!r}, expected "
                                       f"0; {text.count(new)} of {new!r}, expected {n}); review COMMITTED_EDITS")
                whys.append("committed in the repository: " + why)
            else:
                old, new, n, why = spec
                k = text.count(old)
                if k == n:
                    text = text.replace(old, new)
                    whys.append(why)
                elif k == 0 and new in text:
                    whys.append("already fixed in the repository (no-op)")
                else:
                    raise RuntimeError(f"{f}: expected {n} match(es) of {old!r}, found {k}; review PATCHES")
        if text != before:
            p.write_text(text)
        diff = "".join(difflib.unified_diff(before.splitlines(True), text.splitlines(True),
                                            f"repository/{f}", f"workspace/{f}", n=0))
        status = "patched" if text != before else ("verified" if all(k != "lit" for k, _ in todo) else "unchanged")
        done.append((f, status, "; ".join(whys), diff))
    return done


def init_workspace(ctx: Context) -> dict:
    """Create the workspace once (code, frozen inputs, data links, patches); later calls reuse it."""
    meta_p = ctx.run_dir / "workspace.json"
    fp = code_fingerprint(ctx.bundle)
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if meta["code_sha256"] != fp:
            raise RuntimeError(f"{ctx.ws} was built from different code (fingerprint {meta['code_sha256'][:12]} vs "
                               f"{fp[:12]}); use a new workspace")
        link_data(ctx)
        return meta
    if ctx.ws.exists() and any(ctx.ws.iterdir()):
        raise RuntimeError(f"{ctx.ws} exists and is not a workspace; choose an empty or new folder")
    for p in _files(ctx.bundle, CODE_GLOBS + FROZEN_GLOBS):
        dst = ctx.ws / p.relative_to(ctx.bundle)
        dst.parent.mkdir(parents=True, exist_ok=True)
        _copy(p, dst)
    for d in ("experiments/figures", "data/processed", "writeup/figures", "_run/logs"):
        (ctx.ws / d).mkdir(parents=True, exist_ok=True)
    # E011's frozen plan is committed twice: E011_prereg.json (copied above; E011 reads it) and E011.json["plan"]
    plan = json.loads((ctx.bundle / "experiments/results/E011.json").read_text())["plan"]
    if json.loads((ctx.ws / "experiments/results/E011_prereg.json").read_text()) != plan:
        raise RuntimeError("experiments/results/E011_prereg.json differs from the plan stored in E011.json")
    patches = apply_patches(ctx.ws)
    (ctx.run_dir / "patches.diff").write_text("".join(d for *_, d in patches))
    meta = {"created_utc": _utc(), "code_sha256": fp, "bundle": str(ctx.bundle), "git": _git_state(ctx.bundle),
            "patches": [{"file": f, "status": s, "why": w} for f, s, w, _ in patches]}
    _write_json(meta_p, meta)
    link_data(ctx)
    return meta


def link_data(ctx: Context) -> None:
    raw = ctx.ws / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for f in RAW_FILES:
        dst, src = raw / f, ctx.data / f
        if dst.is_symlink() and Path(os.readlink(dst)) == src:
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(src)
        except OSError:
            _copy(src, dst)


def clear_raw(ctx: Context) -> None:
    """Remove the raw-data links and the S01/S02 caches, so a saved Kaggle output holds no competition data; the
    next stage rebuilds the caches (S01/S02 are deterministic)."""
    for f in RAW_FILES:
        p = ctx.ws / "data" / "raw" / f
        if p.is_symlink():
            p.unlink()
    for f in CACHE_FILES:
        (ctx.ws / "data" / "processed" / f).unlink(missing_ok=True)
    prog = load_progress(ctx)
    for sid in CACHE_STEPS:
        if sid in prog:
            prog[sid]["status"] = "cleared"
    _save_progress(ctx, prog)


def _git_state(bundle: Path) -> dict:
    manifest = bundle / "BUNDLE_MANIFEST.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        return {"commit": m.get("git_commit"), "dirty": m.get("git_dirty")}
    try:
        head = subprocess.run(["git", "-C", str(bundle), "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(bundle), "status", "--porcelain"], capture_output=True,
                               text=True).stdout.split("\n")
        return {"commit": head or None, "dirty": [d for d in dirty if d.strip()]}
    except OSError:
        return {"commit": None, "dirty": None}


# ------------------------------------------------------------------------------------------------ progress
def load_progress(ctx: Context) -> dict:
    p = ctx.run_dir / "progress.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _save_progress(ctx: Context, prog: dict) -> None:
    _write_json(ctx.run_dir / "progress.json", prog)


def _log(ctx: Context, msg: str) -> None:
    with open(ctx.run_dir / "run.log", "a") as f:
        f.write(f"{_utc()} {msg}\n")


def import_workspaces(ctx: Context, sources: list) -> list:
    """Merge the workspaces saved by earlier Kaggle stages (attached as notebook inputs) into this one."""
    merged = []
    prog = load_progress(ctx)
    for src in sources:
        sp = json.loads((src / "_run" / "progress.json").read_text())
        meta = json.loads((src / "_run" / "workspace.json").read_text())
        if meta["code_sha256"] != json.loads((ctx.run_dir / "workspace.json").read_text())["code_sha256"]:
            raise RuntimeError(f"{src} was produced by different code; rerun that stage with this bundle")
        for p in sorted(src.rglob("*")):
            if not p.is_file() or p.is_symlink():
                continue
            rel = p.relative_to(src)
            if rel.parts[0] == "_run" and rel.name in ("progress.json", "workspace.json", "run.log"):
                continue
            dst = ctx.ws / rel
            if dst.exists():
                if dst.stat().st_size == p.stat().st_size and _sha256(dst) == _sha256(p):
                    continue
                if rel.parts[0] == "_run":
                    continue
                raise RuntimeError(f"conflicting outputs for {rel} between imported stages")
            dst.parent.mkdir(parents=True, exist_ok=True)
            _copy(p, dst)
        for k, v in sp.items():
            if v.get("status") == "done" and prog.get(k, {}).get("status") != "done":
                prog[k] = v
        merged.append(str(src))
    _save_progress(ctx, prog)
    return merged


def find_stage_outputs() -> list:
    """Saved workspaces of earlier stages among the Kaggle inputs (a folder with _run/progress.json)."""
    if not KAGGLE_INPUT.is_dir():
        return []
    return [d for d in _walk_dirs(KAGGLE_INPUT, 5) if (d / "_run" / "progress.json").is_file()
            and (d / "_run" / "workspace.json").is_file()]


# ------------------------------------------------------------------------------------------------ running
def setup(stage: str = "all", workspace: str | None = None, data: str | None = None, python: str | None = None,
          workers: int | None = None, imports: list | None = None, quiet: bool = False) -> Context:
    """Locate code and data, build or check the pinned interpreter, create (or reopen) the workspace, and merge
    the saved workspaces of earlier stages (on Kaggle: found among the notebook's inputs)."""
    bundle = find_bundle()
    stage = stage.upper() if stage.lower() != "all" else "all"
    if stage != "all" and stage not in STAGES:
        raise ValueError(f"stage must be 'all' or one of {sorted(STAGES)}")
    workspace = workspace or os.environ.get("BDB27_WORKSPACE")
    ws = Path(workspace) if workspace else (KAGGLE_WORKING / "ws" if on_kaggle() else bundle / "data" / "notebook_ws")
    ws = ws.resolve()
    workers = workers or (os.cpu_count() if on_kaggle() else 3)
    py = Path(python) if python else default_python(bundle)
    ctx = Context(bundle=bundle, ws=ws, data=find_data(data), py=py, stage=stage, workers=workers,
                  steps=tuple(s.id for s in STEPS if stage == "all" or s.stage == stage or s.id in CACHE_STEPS))
    meta = init_workspace(ctx)
    sources = [Path(d) for d in imports] if imports is not None else find_stage_outputs()
    imported = import_workspaces(ctx, [d for d in sources if d.resolve() != ws])
    env_report = verify_env(py, bundle, ws)
    _write_json(ctx.run_dir / f"env_{stage}.json", {"utc": _utc(), "platform": platform.platform(),
                                                    "kaggle": on_kaggle(), "cpu_count": os.cpu_count(),
                                                    "workers": workers, "report": env_report})
    _log(ctx, f"setup stage={stage} kaggle={on_kaggle()} python={py} data={ctx.data}")
    if not quiet:
        print(f"code bundle : {bundle}  (git {str(meta['git'].get('commit'))[:10]}, "
              f"{len(meta['git'].get('dirty') or [])} uncommitted paths)")
        print(f"data        : {ctx.data}")
        print(f"workspace   : {ws}")
        print(f"stage       : {stage} -> steps {', '.join(ctx.steps)}")
        print(f"workers     : {workers} (joblib n_jobs=-1 cap); every other script uses its own n_jobs=3")
        if imported:
            print("imported    :", *imported, sep="\n  ")
        print(env_report, end="")
        print("patches     :")
        for p in meta["patches"]:
            print(f"  {p['file']}: {p['status']} - {p['why']}")
    missing = [s.id for s in STEPS if s.stage in UPSTREAM.get(stage, "") and s.id not in CACHE_STEPS
               and not done(ctx, s.id)]
    if missing:
        raise RuntimeError(f"stage {stage} needs the outputs of {', '.join(missing)}: attach the saved workspaces "
                           f"of stage(s) {', '.join(UPSTREAM[stage])} as notebook inputs")
    return ctx


def _summary(sid: str, rec: dict) -> str:
    s = STEP[sid]
    where = f"; ran on {rec['where']}" if rec.get("where") else ""
    return (f"{sid} {s.title}: {rec['status']} in {_fmt_s(rec['wall_s'])} (CPU {_fmt_s(rec['cpu_s'])}, "
            f"peak memory {rec['peak_rss_gb']:.1f} GB{where}); log _run/logs/{sid}.log")


def run(ctx: Context, sid: str, heartbeat_s: int = 900) -> None:
    """Run one step in the pinned interpreter (skipped if already done); raises if it fails."""
    s = STEP[sid]
    rec = load_progress(ctx).get(sid)
    if rec and rec.get("status") == "done":
        print(_summary(sid, rec) + "  [already done]")
        return
    if sid not in ctx.steps:
        print(f"{sid} {s.title}: belongs to stage {s.stage}; not run in stage {ctx.stage}")
        return
    log_p = ctx.run_dir / "logs" / f"{sid}.log"
    _log(ctx, f"start {sid} {s.title}")
    print(f"{sid} {s.title}: running ...", flush=True)
    t0 = time.time()
    with open(log_p, "w") as log:
        log.write(f"# {_utc()} {ctx.py} {' '.join(s.argv) if s.argv[0] != '-c' else '-c <inline>'}\n")
        log.flush()
        proc = subprocess.Popen([str(ctx.py), *s.argv], cwd=ctx.ws, env=_step_env(ctx.ws, ctx.workers),
                                stdout=log, stderr=subprocess.STDOUT)
        last = t0
        while True:
            pid, status, ru = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                break
            time.sleep(2)
            if time.time() - last > heartbeat_s:
                last = time.time()
                print(f"  ... {sid} still running ({_fmt_s(last - t0)})", flush=True)
    rc = os.waitstatus_to_exitcode(status)
    proc.returncode = rc
    rss = ru.ru_maxrss / (1 << 30 if sys.platform == "darwin" else 1 << 20)
    rec = {"status": "done" if rc == 0 else "failed", "rc": rc, "finished_utc": _utc(), "wall_s": time.time() - t0,
           "cpu_s": ru.ru_utime + ru.ru_stime, "peak_rss_gb": rss, "stage_run": ctx.stage,
           "where": "Kaggle" if on_kaggle() else f"local {platform.system()} {platform.machine()}"}
    prog = load_progress(ctx)
    prog[sid] = rec
    _save_progress(ctx, prog)
    _log(ctx, f"end {sid} rc={rc} wall={rec['wall_s']:.0f}s cpu={rec['cpu_s']:.0f}s rss={rss:.2f}GB")
    print(_summary(sid, rec))
    if rc != 0:
        tail = log_p.read_text().splitlines()[-40:]
        raise RuntimeError(f"{sid} failed (exit {rc}). Last lines of {log_p}:\n" + "\n".join(tail))


def run_stage(ctx: Context) -> None:
    for sid in ctx.steps:
        run(ctx, sid)


def done(ctx: Context, *ids: str) -> bool:
    prog = load_progress(ctx)
    return all(prog.get(i, {}).get("status") == "done" for i in ids)


# Machine-speed calibration (PROVENANCE.md 7.3): E001's WR link at full settings, n_jobs 3 as every script runs.
CALIBRATION = """
import json, time
import polars as pl
from bdb27 import io, linkage
d = (linkage.player_table().join(pl.read_parquet(io.PROCESSED / "combine_features.parquet"), on="nfl_id")
     .join(pl.read_parquet(io.PROCESSED / "game_traits.parquet"), on="nfl_id").filter(pl.col("position") == "WR"))
t = time.time()
r = linkage.evaluate(d, "combine_top_speed", "game_top_speed", expected_sign=1, n_jobs=3)
print(json.dumps({"seconds": time.time() - t, **{k: r[k] for k in ("partial_r", "fl_p", "cv_perm_p")}}))
"""
CALIBRATION_OUTPUT = {"partial_r": 0.321878, "fl_p": 0.016198, "cv_perm_p": 0.006997}  # rounded to 6 places
# Stage wall time per second of the calibration call, the largest over the four local full runs: clean-room run
# 2026-10-17 (call timed in the same conditions, 38.4 s): A 177, B 140, C 218; builder runs 2026-10-16 (call 31.9 s,
# timed 2026-10-15): A 165-171, B 112-119, C 197-229.
STAGE_PER_CALIBRATION_S = {"A": 177, "B": 140, "C": 229}
SETUP_H = 0.25  # per Kaggle stage: environment build and the S01/S02 cache rebuild (A's multiplier already has S01/S02)
SESSION_H, MARGIN_H = 12.0, 1.5  # Kaggle's session limit, and the margin a stage should keep below it


def calibrate(ctx: Context) -> dict:
    """Time the calibration call in the pinned interpreter, check its output, and predict each stage's wall time on
    this machine. On Kaggle, run it in stage A and read the prediction before starting stages B and C."""
    if not done(ctx, "S01", "S03"):
        raise RuntimeError("the calibration call needs the outputs of S01 and S03")
    r = subprocess.run([str(ctx.py), "-c", CALIBRATION], cwd=ctx.ws, env=_step_env(ctx.ws, ctx.workers),
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"calibration call failed:\n{r.stdout}\n{r.stderr}")
    out = json.loads(r.stdout.strip().splitlines()[-1])
    same = all(round(out[k], 6) == v for k, v in CALIBRATION_OUTPUT.items())
    pred = {s: m * out["seconds"] / 3600 + SETUP_H for s, m in STAGE_PER_CALIBRATION_S.items()}
    rec = {"utc": _utc(), "where": "Kaggle" if on_kaggle() else f"local {platform.system()} {platform.machine()}",
           "cpu_count": os.cpu_count(), **out, "output_matches": same, "predicted_stage_hours": pred}
    _write_json(ctx.run_dir / f"calibration_{ctx.stage}.json", rec)
    print(f"calibration call: {out['seconds']:.1f} s ({out['seconds'] / 31.9:.2f} x the 31.9 s reference); output "
          + ("matches" if same else f"DIFFERS: {({k: out[k] for k in CALIBRATION_OUTPUT})} vs {CALIBRATION_OUTPUT}"))
    for s, h in pred.items():
        verdict = ("fits" if h <= SESSION_H - MARGIN_H else
                   f"within {MARGIN_H:g} h of the limit" if h <= SESSION_H else "does NOT fit")
        print(f"  stage {s} ({STAGES[s]}): about {h:.1f} h per session here -> {verdict}")
    if pred["C"] > SESSION_H - MARGIN_H:
        print(f"Stage C (E012) is predicted above {SESSION_H - MARGIN_H:g} h. Do not start it in one session: E012 "
              f"must first be split across sessions (notebooks/README.md, Run times).")
    return rec


def finish(ctx: Context) -> None:
    """End of a run. On Kaggle, remove the raw data and its caches so the saved output holds only our own files."""
    if on_kaggle():
        clear_raw(ctx)
    todo = [s.id for s in STEPS if not done(ctx, s.id) and s.id not in CACHE_STEPS]
    print("all steps done" if not todo else f"workspace {ctx.ws} saved; steps still to run: {', '.join(todo)} "
                                            f"(attach this notebook's output to the next stage)")


# ------------------------------------------------------------------------------------------------ display
def _ipy():
    from IPython.display import Image, Markdown, display
    return Image, Markdown, display


def _draft(ctx: Context) -> str:
    return re.sub(r"<!--.*?-->", "", (ctx.ws / "writeup" / "draft.md").read_text(), flags=re.S)


def show_figures(ctx: Context) -> None:
    """Figures 1-5 as regenerated in this workspace, with the writeup's captions."""
    Image, Markdown, display = _ipy()
    caps = dict(re.findall(r"\*\*(Figure \d)\.\*\* (.*)", _draft(ctx)))
    for i, f in enumerate(("fig1_receiver_a.gif", "fig2_cge.png", "fig3_rookie_test.png", "fig4_transfer_map.png",
                           "fig5_drill_reliability.png"), 1):
        p = ctx.ws / "writeup" / "figures" / f
        display(Markdown(f"**Figure {i}.** {caps.get(f'Figure {i}', '')}  \n`writeup/figures/{f}`"))
        display(Image(data=p.read_bytes(), format=p.suffix[1:], width=900))


def show_table1(ctx: Context) -> None:
    """Table 1 exactly as it appears in the writeup (every cell is asserted by the number checker, S13)."""
    _, Markdown, display = _ipy()
    lines = _draft(ctx).splitlines()
    i = next(k for k, s in enumerate(lines) if s.startswith("**Table 1.**"))
    j = next(k for k in range(i + 2, len(lines)) if lines[k].strip() and not lines[k].startswith("|"))
    display(Markdown("\n".join(lines[i:j + 1])))


def show_checker(ctx: Context) -> None:
    """Summary of the number checker's run on the fresh results (full output: Appendix B)."""
    out = (ctx.run_dir / "logs" / "S13.log").read_text().splitlines()
    ok = sum(s.startswith("OK  ") for s in out)
    fail = [s for s in out if s.startswith("FAIL")]
    print(f"{ok} checks OK, {len(fail)} failed")
    for s in fail:
        print(" ", s)
    for s in out:
        if s.startswith(("rendered words", "open placeholders")) or re.fullmatch(r"\d+ failures", s):
            print(s)


def show_traits(ctx: Context) -> None:
    _, Markdown, display = _ipy()
    rows = json.loads((ctx.run_dir / "trait_table.json").read_text())
    mph = 3600 / 1760
    t = ["| Trait | Definition (preregistered) | Minimum per qualifying game | Players | Position average | "
         "Class means 2023 / 2024 / 2025 |",
         "|-|-|-|-|-|-|"]
    for r in rows:
        unit = r["unit"]
        avg = f"{r['average']:+.3f} {unit}" + (f" ({r['average'] * mph:+.2f} mph)" if unit == "yd/s" else "")
        cls = " / ".join(f"{v:+.3f}" for v in r["class_means"].values())
        flag = "" if r["priced"] else " (not priced)"
        t.append(f"| {r['trait']} {r['position']} {r['name']}{flag} | {r['definition']} | "
                 f"≥ {r['min_plays_per_game']} | {r['n_players']} | {avg} | {cls} |")
    display(Markdown("\n".join(t)))


def show_log(ctx: Context, sid: str) -> None:
    print((ctx.run_dir / "logs" / f"{sid}.log").read_text())


def timing_table(ctx: Context) -> str:
    prog = load_progress(ctx)
    rows = ["| Step | Stage | What | Wall | CPU | Peak memory | Ran on |", "|-|-|-|-|-|-|-|"]
    for s in STEPS:
        r = prog.get(s.id)
        if r:
            rows.append(f"| {s.id} | {s.stage} | {s.title} | {_fmt_s(r['wall_s'])} | {_fmt_s(r['cpu_s'])} | "
                        f"{r['peak_rss_gb']:.1f} GB | {r.get('where', '')} |")
        else:
            rows.append(f"| {s.id} | {s.stage} | {s.title} | not run | | | |")
    return "\n".join(rows)


# ------------------------------------------------------------------------------------------------ comparison
def _leaves(a, b, path=""):
    """Yield (path, fresh, committed) for every leaf of the union of two JSON trees (missing side = KeyError)."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in list(b) + [k for k in a if k not in b]:
            yield from _leaves(a.get(k, KeyError), b.get(k, KeyError), f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        for i in range(max(len(a), len(b))):
            yield from _leaves(a[i] if i < len(a) else KeyError, b[i] if i < len(b) else KeyError, f"{path}/{i}")
    else:
        yield path, a, b


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _same(a, b) -> str:
    """'exact', 'float' (within REL_TOL/ABS_TOL) or 'differs'."""
    if a is KeyError or b is KeyError:
        return "differs"
    if _num(a) and _num(b):
        if a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)):
            return "exact"
        if isinstance(a, float) and isinstance(b, float) and math.isfinite(a) and math.isfinite(b) and \
                abs(a - b) <= ABS_TOL + REL_TOL * max(abs(a), abs(b)):
            return "float"
        return "differs"
    return "exact" if (type(a) is type(b) and a == b) else "differs"


def _rule(rel: str, path: str, fresh=None, committed=None):
    for f, pat, rid, check in EXPECTED:
        if (f == "*" or f == rel) and re.search(pat, path) and (check is None or check(fresh, committed)):
            return rid
    return None


def _show(path: str, v) -> str:
    if v is KeyError:
        return "(absent)"
    if REDACT.search(path):
        return "(withheld)"
    s = json.dumps(v)
    return s if len(s) <= 60 else s[:57] + "..."


def compare(ctx: Context, reference: Path | None = None, verbose: int = 12) -> dict:
    """Every regenerated JSON vs the committed one: exact for ints, strings, booleans and structure; floats equal
    within REL_TOL (last-digit summation-order noise); documented differences listed by rule; anything else FAILS."""
    ref_root = reference or ctx.bundle
    rows, details = [], {}
    for rel in COMPARED:
        fresh_p, ref_p = ctx.ws / rel, ref_root / rel
        row = {"file": rel, "status": "MISSING", "leaves": 0, "exact": 0, "float": 0, "max_rel": 0.0,
               "tol_used": 0.0, "documented": {}, "unexplained": 0}
        if not fresh_p.exists():
            rows.append(row)
            continue
        a, b = json.loads(fresh_p.read_text()), json.loads(ref_p.read_text())
        bad = []
        for path, x, y in _leaves(a, b):
            row["leaves"] += 1
            k = _same(x, y)
            if k == "exact":
                row["exact"] += 1
            elif k == "float":
                row["float"] += 1
                m = max(abs(x), abs(y))
                # share of the allowed difference used (1 = at the limit): the margin a new platform has
                row["tol_used"] = max(row["tol_used"], abs(x - y) / (ABS_TOL + REL_TOL * m))
                if m >= 1e-6:  # smaller values are numerical zeros, matched by ABS_TOL
                    row["max_rel"] = max(row["max_rel"], abs(x - y) / m)
            else:
                rid = _rule(rel, path, x, y)
                if rid:
                    row["documented"][rid] = row["documented"].get(rid, 0) + 1
                else:
                    row["unexplained"] += 1
                    bad.append((path, x, y))
        row["status"] = "FAIL" if bad else ("PASS*" if DEFECTS & set(row["documented"]) else "PASS")
        rows.append(row)
        details[rel] = bad
    report = {"utc": _utc(), "rel_tol": REL_TOL, "abs_tol": ABS_TOL, "rows": rows,
              "failures": {f: [(p, _show(p, x), _show(p, y)) for p, x, y in v] for f, v in details.items() if v},
              "rules": DOCUMENTED, "defects": sorted(DEFECTS), "not_regenerated": NOT_REGENERATED}
    _write_json(ctx.run_dir / "compare.json", report)
    print_compare(report, verbose)
    return report


# E008 computes the writeup's "about 90 first-round tests (q 0.06)" from constants transcribed into the script: the
# smallest rev-1 p-values of the other round-1 tests. The rev-1 harness no longer exists, so they cannot be
# regenerated; this checks them against the committed round-1 results they were copied from (PROVENANCE.md N7).
TRANSCRIBED = (("experiments/results/E006.json", 19), ("experiments/results/E006.json", 17),
               ("experiments/results/E006.json", 18), ("experiments/results/E006.json", 8),
               ("experiments/results/E006.json", 12), ("experiments/results/E003.json", 21))


def transcribed_constants(ctx: Context) -> bool:
    """E008's ROUND1_SMALLEST_OTHER_P, each against exploratory[i].perm_p of the committed round-1 file."""
    import ast
    src = (ctx.ws / "experiments" / "E008_transfer_map.py").read_text()
    const = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(src).body
             if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
             and n.targets[0].id in ("ROUND1_SMALLEST_OTHER_P", "N_ROUND1_TESTS")}
    ok = True
    print(f"E008 constants transcribed from round 1 (N_ROUND1_TESTS = {const['N_ROUND1_TESTS']}, a count from "
          f"experiments/log.md GATE 1):")
    for v, (f, i) in zip(const["ROUND1_SMALLEST_OTHER_P"], TRANSCRIBED):
        e = json.loads((ctx.bundle / f).read_text())["exploratory"][i]
        same = round(e["perm_p"], 4) == v
        ok &= same
        print(f"  {v:.4f}  {'matches' if same else 'DIFFERS FROM'} {Path(f).name} exploratory[{i}].perm_p "
              f"{e['perm_p']:.6f} ({e['feature']} -> {e['target']})")
    print("transcribed constants:", "PASS" if ok else "FAIL")
    return ok


def print_compare(report: dict, verbose: int = 12) -> None:
    w = max(len(r["file"]) for r in report["rows"]) - len("experiments/")
    print(f"{'file':{w}s} {'status':6s} {'leaves':>7s} {'exact':>7s} {'float*':>7s} {'tol used':>8s} "
          f"{'documented':>10s} {'FAIL':>5s}")
    for r in report["rows"]:
        doc = sum(r["documented"].values())
        print(f"{r['file'].replace('experiments/', ''):{w}s} {r['status']:6s} {r['leaves']:7d} {r['exact']:7d} "
              f"{r['float']:7d} {r.get('tol_used', 0.0):8.1%} {doc:10d} {r['unexplained']:5d}")
    n_fail = sum(r["status"] not in ("PASS", "PASS*") for r in report["rows"])
    n_def = sum(r["status"] == "PASS*" for r in report["rows"])
    print(f"\n* float* = floats that differ only in the last digits: |fresh - committed| <= {report['abs_tol']:g} + "
          f"{report['rel_tol']:g} x max(|fresh|, |committed|).\n"
          f"  tol used = the largest such difference as a share of that limit (100% = at the limit).\n"
          f"  Everything else (integers, strings, booleans, nulls, keys, list lengths) must be identical.")
    used = {}
    for r in report["rows"]:
        for rid, n in r["documented"].items():
            used.setdefault(rid, []).append(f"{Path(r['file']).name} {n}")
    if used:
        print("\nDocumented differences (not failures; DEFECT = a known reproducibility bug, file marked PASS*):")
        for rid, where in used.items():
            tag = " DEFECT" if rid in report["defects"] else ""
            print(f"  [{rid}]{tag} {report['rules'][rid]}\n      " + ", ".join(where))
    for f, bad in report["failures"].items():
        print(f"\nUnexplained differences in {f} ({len(bad)}); first {min(verbose, len(bad))}:")
        for p, x, y in bad[:verbose]:
            print(f"  {p}: fresh {x} | committed {y}")
    print("\nNot regenerated:", *(f"  {k}: {v}" for k, v in report["not_regenerated"].items()), sep="\n")
    verdict = "FAIL" if n_fail else "PASS"
    extra = f"; {n_def} with a documented defect (PASS*)" if n_def else ""
    print(f"\nOVERALL: {verdict} ({len(report['rows']) - n_fail}/{len(report['rows'])} files pass{extra})")


# ------------------------------------------------------------------------------------------------ bundle
def make_bundle(out: Path) -> Path:
    """OUT/bdb27-bundle.zip: the tracked repository files plus notebooks/ (never data/), for a Kaggle dataset. The
    notebook finds either the zip or its extracted tree among its inputs."""
    import zipfile
    root = find_bundle()
    tracked = subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True, text=True,
                             check=True).stdout.split("\n")
    files = {root / f for f in tracked if f} | {p for p in (root / "notebooks").glob("*")
                                                 if p.is_file() and not p.name.startswith(".")}
    files = sorted(p for p in files if p.exists() and p.relative_to(root).parts[0] != "data")
    git = _git_state(root)
    manifest = {"created_utc": _utc(), "git_commit": git["commit"], "git_dirty": git["dirty"],
                "code_sha256": code_fingerprint(root), "files": len(files)}
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "bdb27-bundle.zip"
    if dst.exists():
        raise FileExistsError(dst)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, f"bdb27-bundle/{p.relative_to(root)}")
        z.writestr("bdb27-bundle/BUNDLE_MANIFEST.json", json.dumps(manifest, indent=1))
    print(f"wrote {dst} ({len(files)} files, {dst.stat().st_size / 1e6:.1f} MB); git {git['commit']}, "
          f"{len(git['dirty'] or [])} uncommitted paths")
    return dst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("bundle", help="write the code bundle for a Kaggle dataset")
    b.add_argument("--out", required=True)
    ap.add_argument("--workspace")
    ap.add_argument("--stage", default="all")
    ap.add_argument("--data")
    ap.add_argument("--python")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--compare", action="store_true", help="only rerun the final comparison")
    ap.add_argument("--import", dest="imports", nargs="*", help="saved workspaces of earlier stages to merge")
    a = ap.parse_args(argv)
    if a.cmd == "bundle":
        make_bundle(Path(a.out))
        return 0
    ctx = setup(a.stage, a.workspace, a.data, a.python, a.workers, a.imports)
    if not a.compare:
        run_stage(ctx)
    ok = True
    if ctx.stage in ("all", "D"):
        rep = compare(ctx)
        print()
        ok = all(r["status"] in ("PASS", "PASS*") for r in rep["rows"]) & transcribed_constants(ctx)
        print("\n" + timing_table(ctx))
    finish(ctx)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
