# notebooks/: the public reproducibility notebook

| File | What it is |
|---|---|
| `final_writeup.ipynb` | The public Kaggle notebook attached to the writeup. It rebuilds every number, table and figure in `writeup/draft.md` from the raw competition data. It then checks them two ways: the number checker, and a value-by-value comparison with the committed results. Markdown cells explain each step; the plumbing is in `run_all.py`. |
| `run_all.py` | The pipeline. The notebook calls it cell by cell, and it also runs from the command line. It uses only the standard library; the analysis runs in the pinned interpreter. |
| `requirements-kaggle.txt` | Exact package versions: the full dependency closure of what the code imports. |
| `PROVENANCE.md` | Each writeup number, figure and table, with the results key and the script and function that write it. |
| `PUBLISHING.md` | The numbered checklist for publishing on Kaggle: bundle, dataset, the stage notebooks (stage E resumes across sessions), calibration, run order, checks, the writeup link and the Kaggle Writeup. |

## Run it locally

```
.venv/bin/python notebooks/run_all.py --workspace /path/to/new/folder
```

- **Workspace.** The run writes everything into the workspace. The default is `data/notebook_ws/`, which is gitignored; `$BDB27_WORKSPACE` also sets it. The run never writes to the repository's `experiments/`, `data/processed/` or `writeup/`. Every script finds its paths from its own location, and the workspace holds copies of the scripts, so `writeup/check_numbers.py` and `writeup/make_figures.py` read the fresh results with no new argument.
- **Resuming.** Each finished step is recorded in `<workspace>/_run/progress.json`. If the run is interrupted (for example, the machine sleeps), rerun the same command and it resumes at the first unfinished step.
  - Step S10b (E013), the longest, also resumes inside the step.
  - It keeps each finished bootstrap and simulation table as a checkpoint in `<workspace>/data/scratch/e013/ckpt`.
  - It always runs with `--resume`, so a rerun reads those checkpoints instead of drawing them again.
- **Logs.** Each step's log is `<workspace>/_run/logs/<step>.log`. `_run/run.log` records step start and end times.
- **Exit code.** It is 0 only if three things hold: every step ran (the number checker is step S13), every regenerated JSON passed the comparison, and E008's transcribed constants matched.
- **Comparison only.** Rerun the final comparison alone with `--compare`.
- **Notebook.** Alternatively, open `final_writeup.ipynb` with `STAGE = "all"` and run all cells. The notebook and the command line share the same workspace logic.
- **Run time.** See [Run times](#run-times).

## Run it on Kaggle

One 12-hour Kaggle CPU session is not enough for the whole chain (see [Run times](#run-times)). The same notebook therefore runs as five stages, each a Kaggle notebook. Stage E may need two or more notebooks, one per session ([Stage E across sessions](#stage-e-across-sessions)). Each notebook saves its workspace, `/kaggle/working/ws`, as notebook output, and the next attaches that output as an input.

| Notebook | `STAGE` | Steps | Inputs to add |
|---|---|---|---|
| bdb27 stage A | `"A"` | S01–S07: caches, E001, round-1 probes E002–E006, E007, E010, E011 | competition data, code bundle |
| bdb27 stage B | `"B"` | S08–S09: E008, E009 | competition data, code bundle, output of stage A |
| bdb27 stage C | `"C"` | S10: E012 | competition data, code bundle, output of stage A |
| bdb27 stage E (and part 2, part 3 if needed) | `"E"` | S10b: E013 (resumes across sessions) | competition data, code bundle, output of stage C (a later part: output of the previous part) |
| **The Combine Exchange Rate: reproducibility** (public, linked from the writeup) | `"D"` | S11–S14 and the comparison | competition data, code bundle, outputs of stage B and the last stage E part |

Stage D's title has no digits on purpose: its URL goes into the writeup, and the number checker fails on any digit in the text that no check covers (PUBLISHING.md, step 15).

Stages B and C can run at the same time, and B can still be running while E runs. **The exact checklist, including what only the account owner can do on Kaggle, is [PUBLISHING.md](PUBLISHING.md).** In outline:

1. **Build the code bundle** from a clean commit:
   ```
   .venv/bin/python notebooks/run_all.py bundle --out /tmp/bdb27-dataset
   ```
   - It writes `bdb27-bundle.zip`, about 14 MB: every git-tracked file plus `notebooks/`, and a `BUNDLE_MANIFEST.json` with the commit and any uncommitted paths. It never includes `data/`.
   - Upload it as a public Kaggle dataset: on the website, or with `kaggle datasets init -p /tmp/bdb27-dataset`, then edit the metadata and run `kaggle datasets create -p /tmp/bdb27-dataset --public`.
   - The notebook accepts either the zip or its extracted tree.
2. **Create the stage notebooks** by uploading `final_writeup.ipynb` once per stage, and once more for each extra stage E session. In each, set `STAGE` in the first code cell and add the inputs in the table above.
3. **Notebook settings:**
   - Internet **on**. It is used only during setup, to download uv, CPython 3.13.5 and the pinned wheels.
   - Accelerator **none**.
   - Run each stage with **Save Version → Save & Run All (Commit)**, so it runs unattended and its output is saved.
4. **Calibrate first.** Before committing anything long, time the calibration call in an interactive session of stage A (PUBLISHING.md, step 7). It times one fixed calculation on Kaggle's CPU and predicts each stage's hours there.
5. **Run the stages:** A, then B and C, then E once C is done (as many sessions as it needs), then D. Stage D's version is the one to make public and to link from the writeup.
   - **Before starting B and C, read stage A's own calibration output** (end of Section 3 of the notebook) and apply the [stage C rule](#the-stage-c-rule) to the larger of the two measurements: start C only if it is predicted at 10.5 h or less (call time T ≤ 161 s).

**What each stage saves.** Each stage's output holds the workspace: our code, our results, figures, logs and derived Parquet files. From stage E on it also holds E013's checkpoints (`data/scratch/e013/ckpt`: bootstrap and simulation draws). Before saving, `ra.finish()` removes two kinds of files: the raw-data links, and the caches that copy the competition data. Those caches are the S01 Parquet tables, the S02 per-frame game tracking, and E006's per-frame copy, which no later step reads. The competition data is never republished. Each stage rebuilds the caches first, in a few minutes; the rebuild is deterministic.

**Why a dataset bundle**, rather than the other options:
- *Writing `src/` out from notebook cells* would put the library into the notebook. It would still miss the `experiments/` and `writeup/` scripts that produce the numbers.
- *`pip install` from GitHub* needs a public repository and pins the code only by a commit hash.

A dataset version instead freezes, in one place:
- the exact code;
- the frozen inputs (preregistrations, the E012 and E013 audit records);
- the committed results that the final comparison checks against.

The notebook prints the bundle's git commit. Stages refuse to merge workspaces built from different code: each workspace records a fingerprint of the code, the frozen inputs, the pins and the compatibility edits.

## Environment

- **Interpreter.** Kaggle's preinstalled Python and packages differ from ours, and numpy 2.5.4, scipy 1.18.1 and contourpy 1.4.0 need Python 3.12 or later. On Kaggle, `run_all.build_env()` therefore takes these steps:
  - installs `uv==0.12.20`;
  - installs CPython 3.13.5 (python-build-standalone, `--no-bin`);
  - creates a venv in `/tmp/bdb27-venv`, outside the saved output;
  - installs `requirements-kaggle.txt` with `--no-deps --only-binary :all:`;
  - runs `uv pip check`.
- **Check before every step.** Every stage checks the interpreter before any step runs: Python must be 3.13.5 and every pinned package must have its pinned version, or the run stops. Locally the project `.venv` passes the same check.
- **Verified on 2026-10-15/16:**
  - `uv pip compile` resolves the pins to exactly these 19 packages as `manylinux_2_28` x86-64 wheels for CPython 3.13.5, and uv offers CPython 3.13.5 for Linux x86-64.
  - `build_env()` builds the interpreter end to end on macOS, and `uv pip check` passes.
  - That uv-built interpreter gives bit-identical results to the project `.venv` on a full-setting harness run: E001's WR link, with 50×5 CV, 2,000 CV permutations and 10,000 Freedman–Lane permutations.
- **Threads.** Steps run with `OMP_NUM_THREADS=3`, as in the committed runs. The scripts fix `n_jobs=3`; E001's `n_jobs=-1` is capped by `LOKY_MAX_CPU_COUNT`, at 3 locally and the CPU count on Kaggle. Results do not depend on the worker count, which was verified bit-identical at 1, 3 and 4 (PROVENANCE.md §9 R5).

## Compatibility edits

Ten scripts needed small, non-statistical edits to run on a fresh machine. Until 2026-10-17 `run_all.py` applied them to the workspace copies only. They are now made in the repository's scripts themselves:
- E002–E006, E005, E009, E011 and `ph_pairs.py`, with `experiments/results/E011_prereg.json` committed as a verbatim copy of `E011.json["plan"]`;
- E012's freeze-file path, together with its row-order fix ([Known issues](#known-issues-and-follow-ups), item 1);
- E013's checkpoint folder (2026-10-19), made in the repository from the start.

`run_all.py` now verifies each edit instead of applying it: the workspace copy must already contain the edited text, or the run stops. Appendix A of the notebook prints any remaining diff between the repository and the workspace copies. The table records what each edit is and why it was needed.

| Script | Edit | Why |
|---|---|---|
| E002, E003, E004, E005, E006 | Rename the rev-1 harness result keys: `perm_p`→`cv_perm_p`, `perm_p_lt_05`→`fl_p_lt_05`, `loco_delta_r2`→`loco`, `loco_positive_classes`→`loco_excess_positive_classes`. In E005's figure, the `linkage._fit_resid(Xi, v)` calls become `linkage._residual_maker(Xi) @ v`, the same OLS residuals. | The scripts crash on the current harness (`KeyError: 'perm_p'`) after writing their probe files. E008 needs their JSONs: n and raw r for its reproduction check, and the game-side reliabilities behind "13 of the 16" and "six links". For E002, E003, E005 and E006 the renames reproduce the fix tested in PROVENANCE.md (§0 B1, run R2) byte for byte. E004 gets the same renames, so its linkage stage runs too. |
| E009 | Its check that its rerun reproduces E008's leave-one-class-out delta and excess now compares the same two fields on both sides. | The check was written against the committed E008.json, which predates harness rev 3. A regenerated E008.json also carries the rev-3 key `null_sd`, so the check failed on structure (`inf`) before computing anything. No value changes. |
| E011 | `PLAN = RES / "E011_prereg.json"` | E011 read its frozen plan from a temporary scratch file. The same plan, stored verbatim in `E011.json["plan"]`, is now committed as `experiments/results/E011_prereg.json`, and setup checks that the two are equal. |
| E012 | The freeze file is read from `experiments/results/E012_audit/` | E012 checks its implementation choices against a freeze file. The file has moved from a scratch folder into the committed audit record, with the same SHA-256. |
| ph_pairs.py | The results path is taken relative to the script | It was an absolute path on one machine. On that machine it silently read the committed results instead of the fresh ones. |
| E013 | `SCRATCH = ROOT / "data" / "scratch" / ("e013" + ("_quick" if QUICK else ""))`, so its checkpoints go to `data/scratch/e013/ckpt` | E013 kept its bootstrap and simulation checkpoints in one machine's session scratchpad, outside the workspace. On Kaggle they would not be saved, so stage E could not resume in a later session. Only the folder moves; no value changes (How this was verified, the E013 item). Its frozen implementation choice X1 still names the scratchpad, and a comment beside `SCRATCH` records the move. |

These edits also resolve PROVENANCE.md's blockers B1 (round-1 crash), B2 (E011's scratch plan) and B4 (hard-coded paths); B3 (no script built the game frames) is resolved by step S02.

## What the comparison checks

`run_all.compare()` walks every leaf of the 17 regenerated JSON files alongside the committed ones (E013.json since 2026-10-19):
- **Must be identical:** integers, strings, booleans, nulls, keys and list lengths.
- **Floats** must agree within a relative 1e-9 (absolute 1e-12): |fresh − committed| ≤ 1e-12 + 1e-9 × max(|fresh|, |committed|). That allows only last-digit summation-order noise from BLAS and OpenMP threads. PROVENANCE.md N3 covers the cross-platform case; E011's docstring covers the same effect within one machine.
  - The table reports, per file, how many floats needed the tolerance, and **tol used**: the largest of those differences as a share of its limit (100% = at the limit). This is the margin a new platform has.
  - In the four local runs, tol used never exceeded 4.6% of the limit, in any pair of runs or against the committed files (worst: an E007 by-class partial r and an E009 null kurtosis). The largest *relative* differences, up to 3.5e-10, are on values near 1e-6 (leave-one-class-out deltas). Their absolute differences, about 2e-16, pass the absolute part of the rule with a wide margin.
- **Anything else** is unexplained, and the file FAILs, unless the path matches a documented difference (`EXPECTED` in `run_all.py`). Each documented difference has a recorded cause and, where one is given, a check on the values: for example "fresh 0, committed negative", or "key new in the fresh file".

The documented differences:

| Rule | Files | What differs | Cause | Evidence |
|---|---|---|---|---|
| `volatile` | E006, E007, E011, E012, E013 | `runtime_s`. In E012 and E013, `run_started_utc`. | wall-clock metadata | — |
| `harness-rev1` | E002–E006 | Test results under `primary/` and `exploratory/`, and their BH summaries. | The round-1 scripts were written for the rev-1 harness, which changed before the first commit: rev 2 added draft-year terms, and rev 3 added the one-sided Freedman–Lane test and the LOCO-vs-null rule. Round-1 decisions stand as logged (experiments/log.md GATE 1); E008 re-estimates every round-1 link under the current harness, and those are the numbers the writeup uses. | n, raw r, feature and target names are still compared exactly, and are identical. E008 and E009 also assert them at run time. |
| `harness-rev3` | E001, E007, E008 | New keys `null_sd` and `loco_classes_pass`; a smaller (one-sided) `partial_r_mde80`, and E008's copies of it in `transfer_map[*].mde80` and `summary.transfer.mde80_two_sided`; `excess_lb_gt_0` True→False for E007 P3 seed 2. | `src/bdb27/linkage.py` rev 3 (commit 598dc08) came after these files were committed. The writeup quotes E008's own `mde80_one_sided`, which is unchanged. | E001 rerun with `linkage.py` from 7260d39: identical to the committed E001.json in all 275 leaves. E008 rerun with the code of its own commit (d11dee7) and the committed round-1 files: all 8,738 leaves of the committed E008.json reproduced (exact or float). |
| `reliability-rev` | E002–E005, E004_reliability, E008 `drill_profile[*].round1_reliability` | Combine-side reliabilities: negative values become 0; new key `n_dropped`; E004's OL within-attempt ICCs move. | `src/bdb27/reliability.py` was revised in d11dee7, after the round-1 results were committed (7260d39): it clips a negative ICC before Spearman–Brown and drops rows alone in their centering group. No writeup number uses these blocks; E008 recomputes every Combine reliability itself and copies these only into an unquoted comparison column. | `E004 --screen` rerun with `reliability.py` from 7260d39: identical to the committed E004_reliability.json (0 differences). |
| `upstream-hash` | E009 | `round3_inputs/e010_sha256` | E009 records the SHA-256 of the E010.json it read. The fresh E010.json differs from the committed one only in last-digit floats, so its hash differs. | E010.json itself passes the comparison. |
| `e013-reuse` | E013 | `reuse_stored_draws` (fresh false, committed true); `runtime_note` (absent in the fresh file) | The committed E013.json was written by a `--reuse` rerun after the E013 review. That rerun read the bootstrap and simulation draws the full run had stored in `data/processed/e013_*.parquet`. The notebook draws them itself. | Every value is compared, and none differs (2026-10-19 run, How this was verified). |

The game-side reliabilities that E008 reads from the round-1 files fall under no rule, so they must match, and they do. So does every other value in E007_leaderboard, E009, E010, E011, E012, E013 and the two exports, apart from the rows above.

The comparison never prints a stopwatch-only forecast (`prior_C1`), draft details or player names (`REDACT`), per the wording guardrails in `writeup/outline.md` §4.

**Transcribed constants.** "About 90 first-round tests (q 0.06)" uses p-values from the rev-1 harness, which E008 holds as constants (PROVENANCE.md N7). `run_all.transcribed_constants()` checks each one against the committed round-1 entry it was copied from.

## Run times

**Local.** Four full runs (stage `all`) on a 14-core Apple M4 Pro, 2026-10-16/17:

| Run | Conditions | Wall | CPU | E012 alone |
|-|-|-|-|-|
| Builder 1 | 3 workers; other jobs kept the load average at 5–8 | 4.27 h | 12.0 h | 1.75 h |
| Builder 2, through the notebook | same | 4.54 h | 12.9 h | 2.02 h |
| Clean room 1 | fresh clone and venv; 4-thread caps, `--workers 4`; other jobs running | 6.74 h | 15.3 h | 3.55 h |
| Clean room 2 | same; machine otherwise quiet | 5.71 h | 16.3 h | 2.33 h (6.88 CPU-h) |

Clean room 2 is the reference measurement, under Kaggle-like caps:
- **Setup.** A fresh clone and venv. `--workers 4` sets the cap for joblib's `n_jobs=-1`; OMP is 3 per step, and BLAS and polars are capped at 4 threads.
- **Total.** 5.71 h wall and 16.3 CPU-hours. `/usr/bin/time` gave 20,582 s real and 55,248 + 3,362 s user + sys.
- **Per stage, wall / CPU.** A 1.88 h / 5.21 h; B 1.49 h / 4.19 h; C 2.33 h / 6.88 h; D 36 s / 28 s.
- **E012's phases.** About 48 min building values and running the bootstraps and post hoc checks, then 40 min of belt simulations and 51 min of coverage simulations.

Per step (builder 1 / clean room 2):

| Step | Stage | Local wall | Local CPU | Peak memory |
|---|---|---|---|---|
| S01 cache | A* | 8 s / 8 s | 16 s / 14 s | 3.0 GB |
| S02 game frames | A* | 2.0 / 2.1 min | 2.2 / 2.2 min | 5.2 GB |
| S03 E001 | A | 3.1 / 3.3 min | 8.9 / 12.1 min | 0.8 GB |
| S04a–e round 1 (E002–E006) | A | 45 / 55 min | 2.0 / 2.5 h | 5.8 GB |
| S05 E007 | A | 21 / 26 min | 59 / 72 min | 1.6 GB |
| S06 E010 | A | 19 / 27 min | 52 / 76 min | 1.3 GB |
| S07 E011 | A | 32 / 16 s | 2.8 / 0.6 min | 0.7 GB |
| S08 E008 | B | 24 / 35 min | 1.1 / 1.7 h | 1.5 GB |
| S09 E009 | B | 36 / 55 min | 1.6 / 2.5 h | 0.6 GB |
| S10 E012 | C | 1.75 / 2.33 h | 5.1 / 6.88 h | 3.0 GB |
| S11a–S14 exports, figures, checker, traits | D | 36 / 36 s | 30 / 28 s | 3.1 GB |
| S10b E013 (added 2026-10-19; one run, below) | E | 4.36 h | 12.6 h | 0.5 GB |

\* Every stage rebuilds S01 and S02 (a few minutes on Kaggle), because no saved output contains them.

**E013 (S10b), measured 2026-10-19** in the proof run (How this was verified, E013). The setup was 3 workers, OMP 3 and a load average of about 5.
- **Totals.** Uninterrupted, it would have taken 15,685 s (4.36 h) and about 12.6 CPU-hours.
  - The run itself was stopped after 18 min, keeping two of its 47 checkpoints, then resumed. The resumed part took 14,825 s wall and 11.9 CPU-hours.
  - The stop cost about 225 s in all: the third trait's unfinished bootstrap, and the prospective test run twice.
- **Phases.**
  - The prospective test: 26 s.
  - The nine 1,000-resample bootstraps: 3,780 s, about 420 s each.
  - The 27 simulation belts: 7,163 s.
  - The two coverage simulations: 2,115 and 2,120 s.
  - The post hoc block, including its bootstrap: 480 s.
- **The committed full run** (`experiments/results/E013_audit/full.log`, 21,077 s) logged 5,938 s for its first trait's bootstrap; the other traits took about 290 s each. Its coverage simulations took 3,805 and 3,422 s.
- **The `--reuse` rerun** that wrote the committed E013.json took 99 s. It reads stored draws, so the notebook cannot use it (Stage E across sessions).

Run times swing with the machine's state: E012 took 1.75 h to 3.55 h in these runs, on the same code.

**The calibration call.** Kaggle does not publish its CPU, so each stage's time is tied to one fixed calculation. That is PROVENANCE.md §7.3's calibration call: E001's WR link at full settings with `n_jobs=3`, as every script runs. `ra.calibrate()` times it in the pinned interpreter, checks its output, and prints each stage's predicted hours. The notebook runs it at the end of stage A; PUBLISHING.md also runs it before stage A.

| Where | `n_jobs=3` | `n_jobs=4` | Output |
|-|-|-|-|
| Reference, 2026-10-15 (load average about 5) | 31.9 s | 24.9 s | partial r 0.321878, FL p 0.016198, CV p 0.006997 |
| Clean room 2, same conditions as its full run | 38.4 s (1.20×) | 30.0 s (1.20×) | identical |
| Local, other jobs running (2026-10-17) | 75.6–266 s | — | identical |
| Local, 2026-10-19: just before / just after the E013 run / 2 min later | 49.1 / 31.0 / 30.8 s | — | identical |

- **Worker count.** The clean room's slowdown was the same at both worker counts. A time measured at `n_jobs=4` therefore converts to the `n_jobs=3` scale by × 31.9 / 24.9 = 1.28.
- **Stage-to-call ratios.** In the local runs, each stage took at most 177 (A), 140 (B) and 229 (C) times as long as the call. The ratios come from clean room 2 (A 177, B 140, C 218, against its own 38.4 s) and from the builder runs (C up to 229, against the 31.9 s reference).
  - **Stage E: 510.** That is its compute over all sessions, from the E013 run's 15,685 s against the 31.0 s call just after it (506).
  - The committed full run, with its first trait at the others' pace, gives about 16,200 s against the 31.9 s reference (508).
  - The call just before the run took 49.1 s, which would give 319. The larger ratio is used, as for the other stages.
- **Prediction.** A stage on Kaggle takes about ratio × T seconds plus 0.25 h, where T is the call's time there. The 0.25 h covers the environment build and the cache rebuild.
- **Checked locally.** With T = 38.4 s the formula gives 229 × 38.4 / 3600 = 2.44 h for E012; clean room 2 took 2.33 h. Because the ratios are the largest of the local runs, the formula over-predicts every local run.

**Kaggle estimate.** Until T is measured on Kaggle, PROVENANCE.md §7.1's assumptions about Kaggle's CPU put it at about 3.5 × the 31.9 s reference (range 2.1–5.0×):

| Stage | Central (T ≈ 112 s) | Range (T ≈ 67–160 s) |
|---|---|---|
| A | 5.7 h | 3.5–8.1 h |
| B | 4.6 h | 2.9–6.5 h |
| C | 7.4 h | 4.5–10.4 h |
| E (E013) | 16.1 h of compute: 2 sessions | 9.7–22.9 h: 1–3 sessions |
| D | 0.3 h | — |

- **Total:** about 34 h of compute (21–49 h), counting 0.25 h of setup for each extra stage E session.
- **Elapsed:** about 30 h (18–42 h), running A, then C, then E's sessions back to back, then D, with B beside C and E.
- **Without E013** it was about 18 h of compute (11–25 h) and about 13.4 h elapsed.
- **Every stage except E fits one 12-hour session even at the pessimistic end, but only just.** At T ≈ 160 s, stage C has 1.6 h to spare, just inside the rule below.
- **Stage E fits one session only if T ≤ 72 s.** Otherwise it resumes across sessions, by design ([Stage E across sessions](#stage-e-across-sessions)).
- **The clean-room report's estimate was higher:** stage C at 8.5 h, up to 12.2 h. It applied the same §7.1 factor to the CPU time of clean room 2 (6.88 CPU-h). That run's calibration call took 1.2 × the reference (38.4 vs 31.9 s), so its CPU time overstates the reference machine's. Anchoring on the call removes that difference.
- Both are models. Only the calibration call on Kaggle decides.

### The stage C rule

This is the only rule; stage E needs none ([Stage E across sessions](#stage-e-across-sessions)). `ra.calibrate()` applies it to the call it times, and prints a verdict per stage; the notebook's calibration cell says the same; PUBLISHING.md applies it to every measurement.

1. **T** is the calibration call's time on Kaggle, in seconds, at `n_jobs=3`, as `ra.calibrate()` prints it. If you have two measurements (PUBLISHING.md's pre-check and stage A's own call), use the larger. Do not re-measure until a number passes.
2. **Predicted hours** for a stage = m × T / 3600 + 0.25, with m = 177 (A), 140 (B) or 229 (C).
3. **Start stage C only if its prediction is at most 10.5 h.** That is the 12-hour session limit minus a 1.5-hour margin:
   229 × T / 3600 + 0.25 ≤ 10.5 ⇔ T ≤ 10.25 × 3600 / 229 = 161.1 s ⇔ r = T / 31.9 s ≤ 5.05.

What the rule implies:
- **Stage C binds first.** At T = 161 s, stage A is predicted at 8.2 h and stage B at 6.5 h, so all three stages fit whenever C does.
- **The margin protects stage C.** C runs later than the calibration, possibly on another host. Locally, the same call took 38.4 s on a quiet machine and up to 266 s under heavy load.
- **Without the margin,** stage C stops fitting at T = 11.75 × 3600 / 229 = 184.7 s (r = 5.79). Between 161 and 185 s it would probably finish, but with under 1.5 h in hand, so the rule does not start it.
- **Above 161 s, do not start stage C.** E012 must first be split across two sessions (PROVENANCE.md §7.2). That changes E012 itself, so the code bundle changes. Stages refuse to merge outputs built from different code, so stages A and B would have to be rerun with the new bundle. That is why PUBLISHING.md applies the rule before stage A is committed.

**The older thresholds, now superseded:**
- **PROVENANCE.md §7.3 (2026-10-15):** "fits if r ≤ 3.5, split if r > 4.5", with r = (the call at `n_jobs=4`) / 31.9 s.
  - It assumed an `n_jobs` 4 override in the scripts, which was never made (the scripts run at `n_jobs=3`). It also divided an `n_jobs=4` time by the `n_jobs=3` reference, and it left 3.5–4.5 undecided.
  - On the `n_jobs=3` scale (× 1.28) its cut-offs are r = 4.48 and 5.77. Those predict stage C at 9.3 h and 12.0 h, so its split threshold sat at the 12-hour limit itself.
  - The single cut above, r = 5.05, is 3.94 on that rule's scale, inside its undecided band.
- **"About 6 × 31.9 s" (the notebook builder) and "r = 6 means about 11.8 h" (the clean-room report):** these describe where stage C stops fitting at all, now r = 5.79. They are not a split threshold.
- **"Stage C at 8.5 h, up to 12.2 h" (the clean-room report):** a CPU-time model; see the Kaggle estimate above.

### Stage E across sessions

Stage E runs one step, S10b: E013 in full. Its `--reuse` mode cannot serve here. It reads E013's own stored bootstrap and simulation draws (`data/processed/e013_*.parquet`), which are gitignored and so not in the bundle. Reading them would also reproduce nothing.

At 510 × T seconds, the step is longer than one Kaggle session whenever T exceeds about 72 s. That includes the central estimate. Instead of splitting E013's code, `run_all.run()` lets the step stop and resume.

**Checkpoints.** E013 writes each finished piece of its Monte Carlo work as a checkpoint in `data/scratch/e013/ckpt`:
- each trait's 1,000-resample bootstrap (`boot_T*`, 9 files);
- each trait's simulation belts, per family (`sim_{role,beyond_C2,beyond_C3}_T*`, 27 files);
- the two coverage simulations (`cov_T1`, `cov_T6`);
- each trait's post hoc dummies-ridge bootstrap (`phboot_T*`, 9 files).

That is 47 files. Run with `--resume`, it reads the checkpoints that exist and computes only the rest. The prospective test and the point estimates are recomputed every time; they take a few minutes.

**The deadline.** On Kaggle, a resumable step must stop 10.5 h (`SESSION_H − MARGIN_H`) after the notebook imported `run_all`. At that point `run()` sends SIGTERM to the step's process group, which holds E013 and its worker processes; SIGKILL follows a minute later if anything is left. It then:
1. removes any checkpoint cut short (a complete Parquet file starts and ends with `PAR1`);
2. records the step as `partial`, with the number of checkpoints kept;
3. lets the notebook finish normally, so the version saves.

`finish()` then says to attach this output to another stage E notebook. There, S10b is not done, so it runs again with `--resume` and picks up the checkpoints. Locally there is no deadline; `$BDB27_STOP_AFTER_H` sets one, to test this.

**Why the results cannot depend on where it stops.**
- Each checkpoint draws from its own fixed seeds: per trait for the bootstraps, and per trait, family, grid point and dataset for the simulations. No random stream is shared across checkpoints.
- A checkpoint is exactly the table the uninterrupted run would have built. A resumed run reads it back from Parquet, which preserves every value and type.
- Verified: the 2026-10-19 run was stopped part-way, then resumed, and its E013.json matched the committed file in every leaf (How this was verified, E013).

**Each session must make progress.** The largest single checkpoint is one coverage simulation: about 120 × T seconds, 5.4 h at T = 161 s. The 120 comes from the committed full run's 3,805 s against the 31.9 s reference; the 2026-10-19 run gives 68 (2,120 s against 31.0 s). Whenever the stage C rule passes, every stage E session therefore finishes at least one checkpoint. `ra.calibrate()` prints the expected number of sessions: ⌈(predicted hours − 0.25) / 10.25⌉.

## How this was verified (2026-10-15/19)

- **Full local run, from the raw CSVs.** Every step S01–S14 ran. This run found the E009 incompatibility: E009 stopped on its own check, the E009 edit was added, and the run resumed at S09.
  - **Checker:** 374 checks, 0 failures; 1,888 rendered words.
  - **Comparison:** all 16 files pass. E012.json was PASS\* at the time (the `row-order` defect, fixed 2026-10-18).
  - **Transcribed constants:** all six match.
  - **Floats:** 7,793 of 54,145 leaves needed the float tolerance. The largest relative difference was 3e-10, on a value of order 1e-6 (absolute 4e-16).
  - **Figures:** five of the six files were byte-identical to the committed ones. Figure 3 differs in 118 anti-aliased pixels of one drawn line, from last-digit float differences in E012's pooled curves.
- **Stage chaining.** The full run's workspace was split by step into the stage A, B and C outputs, with the competition data removed as `finish()` does on Kaggle. Stage D then ran from an extracted `bdb27-bundle.zip`, importing B and C. It rebuilt S01/S02, ran S11a–S14, and passed the comparison. Stage B without stage A's output refuses to start.
- **The documented causes.**
  - E001 rerun with the old `linkage.py`: identical to the committed file.
  - E004 screen rerun with the old `reliability.py`: identical to the committed file.
  - E008 rerun at its own commit: identical to the committed file.
- **Environment.** `build_env()` ran end to end, and the uv interpreter gave bit-identical results to `.venv` (see [Environment](#environment)).
- **Second full run, through the notebook.** `final_writeup.ipynb` (`STAGE = "all"`, `$BDB27_WORKSPACE` set to a new folder) was executed top to bottom with a Jupyter kernel from a fresh workspace, on the final code. It took 4.5 h of wall time and 12.9 CPU-hours, with no cell errors.
  - The results matched the first run: checker 374/0; comparison 16/16 (E012 PASS\*); transcribed constants PASS.
  - Compared directly, the two runs differ only in run metadata, E009's hash of E010.json, and the 72 E012 intervals of the `row-order` rule (since fixed). This confirmed that defect was run-to-run nondeterminism, not a code change.
- **Independent clean-room run (2026-10-17).**
  - Setup: a fresh `git clone` at 1a1e8be plus `notebooks/`, and a new venv installed straight from PyPI with `notebooks/requirements-kaggle.txt` (`pip check` clean). It ran under Kaggle-like 4-thread caps with `--workers 4`, from start to finish, with exit code 0.
  - Results:
    - checker 374/0, on the draft and on a comment-stripped copy (the text pasted into Kaggle);
    - comparison 16/16 (E012 PASS\*);
    - transcribed constants PASS.
  - Its own leaf-by-leaf diff, independent of the rule table, found no unexplained difference.
  - It re-verified the `harness-rev3` and `reliability-rev` causes by rerunning E001, E004's screen, E007 and E008 with the code of the earlier commits.
  - The raw CSVs were unchanged afterwards, and nothing was written outside the workspace.
  - Two later checks used a copy of its workspace: the comparison rerun with the `tol used` column, and `ra.calibrate()`, whose output matched.
- **E013 in the pipeline (2026-10-19).** The notebook's own path for step S10b ran end to end in a scratch workspace.
  - **The workspace.** `run_all.setup()` built it: code, frozen inputs and the E013 checkpoint edit, verified. It held the committed E012 outputs: E012.json from git, and the five e012 parquets of the run that wrote it. It also held the two S01 tables E013 reads.
  - **The run.** S10b ran through `run_all.run()`, exactly as the notebook calls it.
  - **First session.** `$BDB27_STOP_AFTER_H=0.3` stopped it after 18 min, in its third trait's bootstrap.
    - SIGTERM reached E013 and its three workers; no process was left.
    - The step was recorded as `partial`, with 2 checkpoints kept and none cut short, and `run()` printed the resume instruction.
  - **Second run.** It resumed with `--resume`: it read both checkpoints in about 2 s and finished in 4.12 h.
  - **Comparison with the committed E013.json:**
    - 16,477 leaves: 16,340 identical and 133 floats within tolerance (tol used 0.01%; largest absolute difference 4e-16);
    - 4 documented differences: `runtime_s` and `run_started_utc` (`volatile`), `reuse_stored_draws` and `runtime_note` (`e013-reuse`);
    - nothing unexplained. `compare()` gave PASS, and an independent leaf-by-leaf diff agreed.
  - **The draws themselves.** All seven e013 parquets the run wrote are byte-identical to the ones the committed E013.json was built from: the bootstrap, belts, coverage, post hoc bootstrap, both forecast tables and the role shares.
  - **Stage mechanics** were tested on small workspaces:
    - stage E refuses to start without stage C's output;
    - a partial stage E output imports into the next stage E part, with its checkpoints and log, and S10b is not marked done;
    - stage D refuses until S10b is done, then starts from stage B's output and the last stage E part;
    - `finish()` prints the resume instruction for a partial S10b;
    - `ra.calibrate()` prints the stage E line.
  - **Not yet run:** E013 from a freshly regenerated E012 (only from the committed E012 outputs), and stage E on Kaggle.

## Known issues and follow-ups

1. **Fixed (2026-10-18): E012's PH11 season-change intervals were not reproducible run to run** (the `row-order` rule). No writeup number uses them, but the movement was more than last-digit noise:
   - Across five versions (the committed file and four fresh runs), interval endpoints moved by up to 0.157 between-player SD.
   - For 3 of the 36 intervals (T1, T7 and T9; 2024 class, 2024→2025), whether the 95% interval excludes 0 changed from run to run.
   - The cause is in `experiments/E012_game_equivalents.py`, `stationarity()`: `nxt` comes from an unordered `group_by` and join, and `_boot_mean_ci` resamples by position.
   - The fix, now in the repository's E012: `x = grp.sort("nfl_id")["d"].to_numpy()`, recorded in `E012.json["post_hoc_review"]["determinism_fix"]`. E012 is rerun in full and recommitted, and its 36 intervals change once, to reproducible values. The `row-order` rule then comes out of `run_all.py`.
   - **Committed in f673643 (2026-10-18).** The clean-room results above describe the old E012. A bundle built from f673643 or later shows E012.json as PASS, not PASS\* (PUBLISHING.md, step 1).
2. **Done (2026-10-17): the compatibility edits are committed** to the scripts, and `experiments/results/E011_prereg.json` is committed (see [Compatibility edits](#compatibility-edits)). `run_all.py` now verifies them instead of applying them.
3. **The first Kaggle run** may show float differences larger than 1e-9 from OpenBLAS, against Apple Accelerate here.
   - Locally the comparison used at most 4.6% of its float limit (`tol used`), so there is room. The cross-platform margin is unmeasured.
   - If a FAIL lists only such leaves (absolute differences near machine precision of the operands), widen `REL_TOL` and record why. Anything larger is a real difference.
   - Stage D takes minutes, so a rerun with a new bundle is cheap. `REL_TOL` is not part of the code fingerprint, so the saved stage A–C outputs still attach (PUBLISHING.md, step 12).
4. **Some intermediate tables are not reproducible row for row:**
   - `probe_d_combine_backpedal`, `probe_d_combine_events` and `probe_d_game_backpedal_plays` (written by E005 through `src/bdb27/probes/d_db_transition.py`) come out in a different row order each run;
   - `probe_c_combine` (E004, `c_ol_lateral.combine_reversal_features`) comes out with a different column order.

   They are not compared. E008 and E012 read them, and no compared value moves beyond the float tolerance. Sorting before writing (LESSONS.md) would remove one source of last-digit noise.
5. **A Kaggle stage that stops early may save competition data.** `finish()` removes the raw-data links and the S01/S02 caches only in the notebook's last cell. If a stage fails or hits the 12-hour limit, its saved output may still hold them. Do not make such a version public; delete it.
6. **Stage E has not run on Kaggle.** Its stop-and-resume path was verified locally (How this was verified, E013).
   - **Where the clock starts.** On Kaggle the deadline counts from the moment the notebook imports `run_all`, not from the session's own start. A slow kernel start eats into the 1.5 h margin.
   - **Debris from a stop.** A stop leaves joblib's temporary memmap folders behind. They sit in the system temp folder, outside the saved output. loky's resource tracker may print `FileNotFoundError` warnings about them; they are harmless.
7. **E013 reads only its own checkpoints.** `--resume` trusts every file in `data/scratch/e013/ckpt`, and the workspace fingerprint keeps checkpoints from different code apart. Running E013 by hand in the repository with `--resume` would read stale checkpoints from an earlier code version. Delete `data/scratch/e013/ckpt` first.
