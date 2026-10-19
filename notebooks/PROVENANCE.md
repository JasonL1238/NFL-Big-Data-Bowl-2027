# PROVENANCE: every writeup number, figure and table → results key → script

Scope: `writeup/draft.md` **v2.2** (commit `1a1e8be`), its 5 figures and Table 1, extended on 2026-10-19 to the E013 numbers of draft v2.4 (§2.12). Built 2026-10-15 for Gate 3 (PLAN.md §5: "public notebook reproducing every writeup number"). AGENTS.md requires an attached **public Kaggle notebook** from which every writeup result is reproducible, run top to bottom.

**Status (2026-10-19): E013 is in the pipeline.**
- Step **S10b** (stage E) reruns E013, the role-aware check, from the workspace's own E012 outputs, never from committed copies. `E013.json` is the 17th compared file.
- **§2.12** maps every E013 number in draft v2.4 (writer, 2026-10-19): the Section 3.4 Roles note, the Section 3.2 DL wording, and the Section 3.2 correction of §2.4's "five traits reach 0.4–1.7 games" to two traits. The rows of §2.1–§2.11 are those of v2.2; rows whose wording changed in v2.4 say so.
- **Writeup v2.3** uses real player names in place of the earlier versions' "Receiver A" and "Receiver B" (`writeup/outline.md` §1); §2.1 and §3 use them too. No number changed.
- **E013 needs its full run in the notebook.** Its `--reuse` mode reads E013's own stored bootstrap and simulation draws (`data/processed/e013_*.parquet`). Those are gitignored, so they are not in the bundle, and reading them would not reproduce anything. The notebook therefore draws them itself.
- **The notebook path is proven** (§9, run R7). A scratch workspace held the committed E012 outputs. E013 ran through `run_all.run()`, was stopped part-way and resumed, and its E013.json matched the committed file in every leaf.

**Status (2026-10-17).** This file was written on 2026-10-15, before the notebook was built. It remains the provenance map (§2–§5, §8). For the current state, read `notebooks/README.md`; to publish, `notebooks/PUBLISHING.md`.
- **B1–B4 are resolved, in the repository itself** (§0 says how). The compatibility edits are committed to the scripts: E002–E006, E005, E009, E011 and `ph_pairs.py`, plus `experiments/results/E011_prereg.json`. E012's freeze-file path is committed with its row-order fix. `run_all.py` verifies these edits instead of applying them, and its step S02 builds the game frames (B3).
- **B5 and B6 are handled by the pipeline.** B5: the pinned interpreter. B6: four chained stage notebooks.
- **The full-count runs that §9 lacked have been made.** Four full local runs reproduce the writeup: checker 374/0, comparison 16/16. The clean-room run, under Kaggle-like caps, took 5.71 h wall and 16.3 CPU-hours, E012 6.88 CPU-hours (§7.0).
- **The E012 row-order defect is fixed (2026-10-18).** The notebook runs found it (README, Known issues 1). `stationarity()` now sorts by `nfl_id` before its bootstrap; E012 was rerun in full and recommitted, and the comparison's `row-order` rule is removed.
- **Parts of §7.2's plan were not implemented:** the `n_jobs` 4 override, skipping unused work, and the stage C split. The scripts run at `n_jobs=3`.
- **§7.3's decision rule is replaced** by the single stage C rule in README.md (Run times), which `ra.calibrate()` applies.

How this was checked: dry runs in isolated copies under the session scratchpad, rebuilt from `data/raw` (the repo, its results JSONs and `data/` were never written). §9 lists every run and its outcome.

**Bottom line (2026-10-15).**
- From a fresh rebuild of raw data with the committed code, every *deterministic* quoted number reproduces (to ≤ 1e-13). That covers 338 of the checker's assertions.
- The only failures were the Monte Carlo quantities, which `--quick` mode deliberately under-samples.
- Three things blocked a top-to-bottom notebook: crashing round-1 scripts (B1), a scratch-file dependency (B2) and an uncalled cache builder (B3). All three have since been resolved (§0).
- The full chain does not fit one 12-h Kaggle session. The central estimate was ≈ 15 h; anchored on the calibration call (2026-10-17) it is ≈ 18 h of compute (README, Run times). Four chained notebooks fit without changing any statistical setting (§7).

---

## 0. What blocked a top-to-bottom notebook on 2026-10-15, and how each was resolved

**Resolutions (2026-10-17).** No statistical setting changed in any of them.

| # | Resolution |
|---|---|
| **B1** | **Fixed in the repository**, with fix (a) below. E002–E006 use the current harness keys, and E005 computes its residuals as `linkage._residual_maker(X) @ v`. `run_all.py` checks that the edits are present. The full round-1 runs are inside the measured stage A time (§7.0). |
| **B2** | **Fixed in the repository.** `E011.json["plan"]` is committed verbatim as `experiments/results/E011_prereg.json`, and E011 reads `PLAN = RES / "E011_prereg.json"`. |
| **B3** | **Fixed in the pipeline.** `run_all.py` step S02 runs `speed.build_game_frames()` before E007, in every stage. |
| **B4** | **Fixed in the repository** for every path the pipeline uses. `ph_pairs.py` finds its results from its own location. E012 reads its freeze file from `experiments/results/E012_audit/`, a change committed with its row-order fix. The other `/private/tmp` scratch paths (E008, E009, E010, E012) serve only `--quick` and `--combine` runs, which the pipeline never makes. `ph_inseason.py`, `ph_rev3.py` and `check_numbers_v1.py` still hold absolute paths, but nothing runs them. |
| B5 | `run_all.build_env()` builds CPython 3.13.5 with the pins on Kaggle; every step checks the interpreter first. |
| B6 | Four chained stage notebooks (README, Run it on Kaggle; PUBLISHING.md). The runtime plan is §7, with measurements in §7.0. |
| Later (2026-10-16) | **E009 failed its reproduction check** on a regenerated E008.json. The check compared structures, and the fresh file has harness rev 3's extra key. It is fixed in the repository: both sides now keep only `delta` and `excess`, and no value changes. |

The table below is the 2026-10-15 record.

| # | Problem | Writeup numbers affected | Fix (no statistical change) |
|---|---|---|---|
| **B1** | **The round-1 scripts E002, E003, E005 and E006 crash** with `KeyError: 'perm_p'` at their first `linkage.evaluate` result. They were written against the rev-1 harness keys: `perm_p`, `loco_delta_r2`, `checks.loco_positive_classes` and `checks.perm_p_lt_05`. E005's figure code also calls the removed `linkage._fit_resid`. Each script writes its probe parquets and then dies, so **E002/E003/E005/E006.json cannot be regenerated**. E008 reads the game-target reliabilities `rel_y` from those JSONs (`E008_transfer_map.target_reliability`). | §4 "**13 of the 16** pass the reliability gate" (`E008 summary.transfer.nulls_passing_c5`); §4 "effects of 0.5 or more remain possible for **six** links" (`summary.transfer.corrected_beyond_b2_ge_05`); Figure 4's six "○" glyphs. All other E008 outputs are regenerable. | **Verified fix (a):** rename the four keys (`perm_p`→`cv_perm_p`, `perm_p_lt_05`→`fl_p_lt_05`, `loco_delta_r2`→`loco`, `loco_positive_classes`→`loco_excess_positive_classes`) and replace `_fit_resid(X, v)` with `(None, linkage._residual_maker(X) @ v)`. The patched scripts then ran end to end and regenerated every `rel_y` input and every n/raw-r reference **exactly** (§9, run R2). Cost ≈ 1.9 h on Kaggle, because they also rerun the unused round-1 linkage tests. **Cheaper fix (b):** compute only those split-half reliabilities, inside E008 or a small module, and assert them against the committed values. Keys in §2.8. |
| **B2** | **E011 reads its plan from the session scratchpad**: `PLAN = /private/tmp/claude-501/…/scratchpad/round3/E011_prereg_scratch.json` (line 48; read unconditionally at line 727). The file is temporary and absent on Kaggle, so E011 cannot run there. | §6 league item 3: "87%", "best 0.42", "6–46 catching drills"; "The 2025 ball data"; Figure 5's E011 point (0.42 and its CI). | The plan is stored verbatim in `E011.json["plan"]` (identical to the scratch file). Commit it as `experiments/results/E011_prereg.json` and point `PLAN` at it. With the file present, E011 reproduced E011.json exactly (§9). |
| **B3** | **No script calls `speed.build_game_frames()`**, yet E007 (line 281) and E008 (line 827) scan `data/processed/game_frames_post_snap.parquet` directly. On a fresh machine E007 fails. | Everything from E007 and E008 (hook, §4, §5, Figures 1, 4, 5). | Notebook step S2: `python -c "from bdb27 import speed; speed.build_game_frames()"` before E007. It takes 120 s here and its output matched the repo cache exactly. |
| **B4** | **Hard-coded absolute paths.** `experiments/posthoc_writeup_v1/ph_pairs.py` reads `/Users/jasonli/Documents/GitHub/…`; so do the unused `ph_inseason.py`, `ph_rev3.py` and `check_numbers_v1.py`. E008/E009/E010/E012 hard-code `/private/tmp/…` scratch dirs, but only for `--quick` output and E012's optional freeze-file check. E008 `--quick` also fails on a fresh machine because it never creates that dir. E009 `--quick`'s figure needs `data/processed/e008_drill_reliability.parquet`, which only a full E008 run writes. | §1 rank "1 of 91" cross-check (`ph_pairs.json hook_fast_rank_hidden_of_91`). The primary source, `E007_leaderboard.json`, is unaffected. | `ROOT = Path(__file__).resolve().parents[2]` in ph_pairs.py. Make the scratch dirs relative and `mkdir` them. |
| **B5** | **Our pins need Python ≥ 3.12.** numpy 2.5.4, scipy 1.18.1 and contourpy 1.4.0 all declare `requires_python >=3.12` on PyPI. Our venv runs CPython 3.13.5. On an older Kaggle image, `pip install` of the pins fails. | All. | §6: assert the interpreter version. If it is < 3.12, install CPython 3.13.5 with `uv` and run every step as a subprocess of that interpreter. |
| **B6** | **One 12-h Kaggle session is not enough.** At 4 vCPUs the central estimate for the whole chain is ≈ 15 h; E012 alone is ≈ 7 h. | All. | §7: four chained stage notebooks, with B and C running concurrently. Use n_jobs 4 (verified result-invariant) and skip computations no number uses. No permutation, bootstrap or simulation count changes. |

Not blocking, but read before comparing outputs:
- **N1. Some library code changed after results were written.**
  - `linkage.py` gained rev 3 (commit `598dc08`) after E001/E007/E008.json were committed. `reliability.py` changed after `E004_reliability.json` was committed.
  - Reruns therefore differ in fields nobody quotes:
    - `partial_r_mde80` is now one-sided (≈ 10% smaller).
    - New keys appear: `loco.*.null_sd` and `checks.loco_classes_pass`.
    - `excess_lb_gt_0` now requires the expected sign (one E007 seed-2 flag flips). E008's `c2` and `c6` flags inherit that rule.
    - In E004_reliability.json, negative Combine within-attempt ICCs are now clipped to 0, and `n_dropped` is added.
  - Verified (§9): E007_leaderboard.json reproduces exactly. E007.json and E001.json differ only in those fields, plus floats at ≤ 3e-11 relative. Every quoted E008 key reproduces to ≤ 1e-15. The E004 target reliabilities E008 uses reproduce to 1e-16.
- **N2. E012.json was written by a full run plus a `--rewrite` pass** (`rewritten_utc` 2026-10-15T22:21:16Z). The rewrite rebuilds `cge_table`, wording and figures from stored values, using the same functions a full run calls. A single full run produces the same keys.
- **N3. Cross-platform floating point.** numpy here links Apple Accelerate; the Linux wheels link OpenBLAS.
  - Every Monte Carlo draw is seeded (`default_rng`/PCG64, platform-independent), so expect last-digit differences only.
  - Ten quoted numbers sit within 5e-4 of a rounding edge (§5). Compare those first.
- **N4. Figures.** `make_figures.py` asks for Helvetica Neue or Arial and falls back to DejaVu Sans, the only one of the three on Kaggle. The layout holds in both, but the PNG/GIF bytes will differ. Here, the rerun reproduced all 6 committed files byte for byte.
- **N5. Writes go under the repo root.** Every script derives `ROOT` from `__file__` and writes `experiments/results`, `experiments/figures` and `data/processed` under it. On Kaggle, copy the repo to `/kaggle/working/` (`/kaggle/input` is read-only). Then symlink the competition CSVs into `data/raw/`, detecting a nested folder under `/kaggle/input/nfl-big-data-bowl-2027/`.
- **N6. Missing keys for the Code line.** "Trait definitions and position averages" are promised by the draft's Code line, but no committed key holds position averages in units. Only per-player `mu_C0` (in `e012_forecasts_2025.parquet`) and `case_studies[*].prior_C0` exist. The notebook must compute them, e.g. the C0 mean per trait, and label them as not quoted in the text.
- **N7. Transcribed constants inside E008.** The "about 90 first-round tests (q 0.06)" figure is computed from constants in `E008_transfer_map.py` (lines 691–800), not read from files:
  - `N_ROUND1_TESTS = 90`, a count taken from experiments/log.md GATE 1.
  - `ROUND1_SMALLEST_OTHER_P = [0.0020, 0.0040, 0.0040, 0.0060, 0.0100, 0.0100]`, the smallest rev-1 p-values of the other round-1 tests, copied from the round-1 results JSONs.

  These are reproducible because they live in the script. But no current script regenerates them: rev-1 p-values cannot be produced by the current harness, and even B1-patched round-1 scripts emit rev-3 `cv_perm_p` instead. I matched each constant to a committed JSON entry:

  | Constant | Committed source |
  |---|---|
  | 0.0020 | `E006.json exploratory[19].perm_p` (c40_vmax → game_S0) |
  | 0.0040 | `exploratory[17]` (c40_smax → S0) |
  | 0.0040 | `exploratory[18]` (c40_a0 → A0) |
  | 0.0060 | `exploratory[8]` (c40_tau → A0) |
  | 0.0100 | `exploratory[12]` (c40_vmax → game_top_speed, WR) |
  | 0.0100 | `E003.json exploratory[21].perm_p` (c40_d10 → g_d10_rush) |

  The notebook should assert the constants against these entries and say that they are transcribed.

---

## 1. Pipeline and dependency order

```
S0  env (Python 3.13.5 + pins, §6)
S1  data cache       python -m bdb27.io                    raw CSV → data/processed/{players, combine_results, player_career_successes, games,
                                                            player_play, combine_tracking, game_tracking_2023..2025}.parquet
S2  game frames      speed.build_game_frames()             → game_frames_post_snap.parquet            (B3: no script calls it)
S3  E001 store       features.update_store in E001         → combine_features.parquet, game_traits.parquet
                                                            (E001's five evaluate() calls feed no number)
S4  round-1 probes   E002, E003, E004 --screen, E005, E006 → probe_{a,b,c,d,e}_*.parquet, E004_reliability.json
                                                            (+ B1: E002/E003/E005/E006.json; E006 builds probe_e_game_frames.parquet)
S5  E007   (S2)                                            → E007.json, E007_leaderboard.json, e007_{forty_attempts, players, routes}.parquet
S6  E010   (S1)                                            → E010.json, e010_{players, play_max}.parquet
S7  E011   (S1; B2)                                        → E011.json, e011_{attempts, players}.parquet
S8  E008   (S2, S3, S4 + round-1 JSONs + E00x_prereg)     → E008.json, e008_{transfer_map, drill_reliability, link_players}.parquet
S9  E009   (imports E008 link builders; E008.json, E010.json, E011.json, E009_prereg) → E009.json, e009_*.parquet
S10 E012   (S4 probe_a..d, S5 e007_players/routes, S6 e010_play_max, E012_prereg)  → E012.json, e012_*.parquet
            (e012_forecasts_2025 and e012_game_values: full run only)
S10b E013  (S1 games/player_play; S10 E012.json, e012_{players, game_values, prospective_values, forecasts_2025,
            bootstrap}; E013_prereg, E013_audit freeze file; full run, --resume)  → E013.json, e013_*.parquet
            (added 2026-10-19; stage E; checkpoints in data/scratch/e013/ckpt)
S11 exports: posthoc_writeup_v2/e012_case_priors.py (S10 full); posthoc_writeup_v1/ph_pairs.py (S5; B4)
S12 writeup/make_figures.py (JSONs of S5–S11 + combine_tracking, e007_forty_attempts, e012_game_values, game_tracking_2025, games, player_play)
S13 writeup/check_numbers.py (374 checks, must exit 0) + print the §4 arithmetic from keys
```

Ordering constraints beyond the arrows:
- **E010 and E011 run before E009.** E009 hashes E010.json, pulls E010's p-values into its post hoc multiplicity, and reads E011's test count.
- **E008 runs before E009.** E009 copies E008.json's beyond-B2 columns into the Figure 4 table and asserts that its rev-3 rerun reproduces them to 1e-12.
- **E012 full (not `--quick`) runs before S11/S12.** Only a full run writes `e012_forecasts_2025.parquet` and `e012_game_values.parquet`.
- E008 asserts the rebuilt links' n/raw r against the committed round-1 JSONs before doing anything else.
- **E012 full runs before E013 (S10b), and E013 before S12/S13** (added 2026-10-19).
  - E013 reads E012.json and five e012 parquets.
  - It asserts that it reproduces E012's internal CGE, its 1,000 B1 draws per trait and its 626 stored 2025 forecast rows (`E013 internal[T].e012_reproduction`, `data_checks.e012_forecast_reproduction`).
  - The checker reads E013.json once the draft quotes E013.

| Step | Command (repo root) | Ran end to end on 2026-10-15? (§9) | Local wall, M4 Pro, n_jobs 3 | CPU-s (user+sys) | Peak RSS |
|---|---|---|---|---|---|
| S1 | `python -m bdb27.io` | yes; all 9 parquets identical to the repo cache | 7 s | 17 | 3.2 GB |
| S2 | `python -c "from bdb27 import speed; speed.build_game_frames()"` | yes; identical | 120 s | 131 | 5.6 GB |
| S3 | `python experiments/E001_combine_top_speed_to_game_top_speed.py` | yes; parquets identical, E001.json differs only by rev-3 fields (N1) | 1,285 s (feature store alone: seconds) | (workers not counted) | 0.8 GB |
| S4 | E002 / E003 / `E004 --screen` / E005 / E006 | E004 yes. **E002/3/5/6 crash after writing parquets (B1)**. All 13 probe parquets identical (≤ 1e-15). | to the crash: 59 / 61 / 26 / 57 / 163 s | 132 / 126 / 34 / 128 / 247 | ≤ 6.2 GB |
| S4 (B1 fix a) | patched E002 / E003 / E005 / E006, full | yes; every `rel_y` input and n/raw r identical | 476 / 638 / 339 / 601 s | 1,200 / 1,740 / 932 / 1,693 | ≤ 3.9 GB |
| S5 | `python experiments/E007_hidden_speed.py` | **yes**; E007_leaderboard.json identical, E007.json differs only by rev-3 fields | 1,339 s | 3,627 | 1.6 GB |
| S6 | `python experiments/E010_drill_intensity.py` | yes (`--quick`; all quoted keys identical) | full 1,487 s (log) | 3,862 (log) | 1.4 GB |
| S7 | `OMP_NUM_THREADS=3 python experiments/E011_catch_point.py` | yes here (identical); **no on Kaggle (B2)** | 25 s (186 s in the committed log) | 174 | 0.7 GB |
| S8 | `python experiments/E008_transfer_map.py` | yes (`--quick`; every quoted key identical) | full 2,047 s (log) | 5,132 (log) | 1.5 GB |
| S9 | `python experiments/E009_beyond_stopwatch.py` | yes (`--quick`; quoted keys identical) | full 2,390 s (log) | 6,580 (log) | 0.6 GB |
| S10 | `python experiments/E012_game_equivalents.py` | yes (`--quick`; every deterministic quoted key identical) | full 16,086 s on a shared machine (log); 2,162 s without the PH6 simulations | 20,730 (log) | 3.4 GB |
| S10b (2026-10-19) | `python experiments/E013_role_aware.py --resume` | yes, 2026-10-19 (§9 R7): stopped part-way, resumed, every leaf of E013.json reproduced | see README.md (Run times) | — | — |
| S11 | `python experiments/posthoc_writeup_v2/e012_case_priors.py`; `python experiments/posthoc_writeup_v1/ph_pairs.py` | yes; both JSONs identical to the committed ones | 0.1 s; 0.5 s | — | small |
| S12 | `python writeup/make_figures.py` | yes; all 6 files byte-identical | 27 s | 28 | 3.5 GB |
| S13 | `python writeup/check_numbers.py` | yes: 374 checks, 0 failures | < 1 s | — | small |

The status and times in this table come from the 2026-10-15 sandbox runs (§9). Every step now runs end to end (§0). Measured full-run times are in §7.0 and README.md (Run times).

---

## 2. Number-by-number provenance (draft.md v2.2; E013 rows for v2.4 in §2.12)

**Wording.** §2.1–§2.11 quote the v2.2 wording. v2.3 and v2.4 reworded or cut some of those sentences (`writeup/outline.md` §12, item 6). Examples: the League items on mirror-side reps and the 87% timing check, Limitations' role examples, and "five traits reach 0.4–1.7" (superseded, §2.12). No key or value of a remaining number changed. `writeup/check_numbers.py` is the authority on the current wording.

**Abbreviations used in the tables.**

| Short | Means |
|---|---|
| `E012`, `E008`, … | `experiments/results/E012.json`, and so on |
| `CP` | `experiments/posthoc_writeup_v2/e012_case_priors.json` |
| `PP` | `experiments/posthoc_writeup_v1/ph_pairs.json` |
| `ct[T]` | the `E012 cge_table` row for trait T |
| `tr[T]` | `E012 traits[T]` |
| `CS` | `E012 deliverables.case_studies[nfl_id 59278, T1]` (Isaiah Bond) |
| `V2` | `E012 validation_2` |
| `PHR` | `E012 post_hoc_review` |
| `TM8[id]` | `E008 transfer_map[id]` |
| mph | yd/s × 3600/1760 |
| "Arith" | arithmetic on committed keys; §4 lists the formulas, and the notebook must print them |

Every row is asserted by `writeup/check_numbers.py` at the quoted rounding. **Status** (last column):

| Status | Means |
|---|---|
| **OK** | the producer ran end to end in §9 and reproduced the value |
| **OK-MC** | the value depends on a Monte Carlo count. Its code path ran in `--quick` mode, but the value itself needs a full run to reproduce |
| **B1 / B2** | blocked on 2026-10-15; see §0 |

Since then, every row has been reproduced by four full local runs, OK-MC and B1/B2 rows included: checker 374/0 (README.md, How this was verified). The Status column records the 2026-10-15 state.

**Where each key is written.**

| Results file | Script | Keys → function or constant that writes them |
|---|---|---|
| E012 | `experiments/E012_game_equivalents.py` + `src/bdb27/probes/h_game_equivalents.py` | `traits` ← `internal_trait()` (H.estimate, H.bootstrap with 1,000 resamples)<br>`cge_table` ← `cge_table()`<br>`validation_2.*` ← `validation2()` + `validation2_tests()`<br>`post_hoc_review.per_trait` ← `trait_posthoc()`<br>`post_hoc_review.simulation` ← `sim_belt()`/`belt_summary()` + `sim_coverage()`/`coverage_summary()` (PH6)<br>`post_hoc_review.validation_2` ← `validation2_posthoc()`<br>`deliverables.weight_curves` ← `weight_curves()`<br>`deliverables.case_studies` ← `case_studies()`<br>`amendments` ← `AMENDMENTS`; `post_hoc` ← `POST_HOC` |
| E008 | `experiments/E008_transfer_map.py` | `transfer_map` ← `transfer_map()`<br>`summary.transfer` ← `transfer_summary()`<br>`summary.multiplicity` ← `multiplicity()`<br>`summary.drill_profile` ← `profile_summary()`<br>`drill_profile` ← `drill_profile()`<br>`links[*].caveats` ← `CAVEATS` in `main()` |
| E009 | `experiments/E009_beyond_stopwatch.py` | `summary` ← `summarize()`<br>`transfer_map` ← `transfer_rows()`; its `b2_*` columns are copied from E008.json |
| E010 | `experiments/E010_drill_intensity.py` | `combine.stop_rule` ← `combine_stage()`<br>`game.primaries` ← `game_stage()` |
| E011 | `experiments/E011_catch_point.py` | `data` ← `data_facts()`<br>`detection.contact_time_check` ← `contact_check()`<br>`reliability` ← `reliability_block()` |
| E007 | `experiments/E007_hidden_speed.py` | `E007_leaderboard.json players[*]` and `meta` ← the leaderboard block, lines ~995–1090<br>`E007.json concept_pair` ← line ~1303 |
| CP | `e012_case_priors.py` | all keys; it reads `E012.json` and the k = 0 rows of `data/processed/e012_forecasts_2025.parquet` |
| PP | `ph_pairs.py` | all keys; it reads E007.json and E007_leaderboard.json |

Hand-written, frozen inputs (not script output, never written by the notebook):
- `E012_prereg.json`: `information_sets`, `deliverables[2]`, `traits[T].game_value`, `framework.*`
- `E009_prereg.json`
- `E010_prereg.json`
- `experiments/results/E012_audit/*`

### 2.1 Title and §1 (hook: Isaiah Bond, nfl_id 59278)

| Quoted | Value | Source key | Producer | Status |
|---|---|---|---|---|
| "4.39 forty at the 2025 Combine" | 4.39; 2025 | `E007_leaderboard players[59278].official_forty`, and its Combine-class field (2025) | E007 | OK |
| "last 20 yards at 23.2 mph" | 23.15 | `players[59278].flying20_mph` | E007 | OK |
| "fastest finish for his 40 time of 91" | rank 1; 91 | `players[59278].rank` (ranked by `hidden_speed_mph`); `meta.n`; cross-check `PP hook_fast_rank_hidden_of_91` | E007; ph_pairs.py | OK (B4 for the cross-check) |
| "fit on 2023–24" | — | `V2.design_label` | E012 | OK |
| "90th-percentile play peak" | q90 | `tr[T1].per_game_value` | E012 | OK |
| "17.6 mph" / "17.5 receiver average" | 8.607 / 8.568 yd/s | `CS.prior_C2` / `CS.prior_C0` | E012 | OK |
| "18.3 in his first qualifying game" (also §2 "18.3 mph debut") | 8.95 yd/s | `CS.game_values[0]` | E012 | OK |
| "averaged 18.9 over 15" (heading "fifteen games") | 9.239 yd/s; 15 | `CS.season_mean`, `CS.n_games` | E012 | OK |
| "two forecast standard deviations" | 2.009 | `CS.z_predictive` | E012 | OK |
| "the 11 of 15 rookie receivers … ten or more qualifying games" | 11; 15 | `CP pools.T1.n_with_10plus_games`, `.n_test_players_k0`. Cross-checks: 15 = `V2.primary.H1.n_by_trait.T1` = `V2.secondary_per_trait.T1["1"].n` | e012_case_priors.py | OK (needs the full-run parquet) |
| "largest gap … (a rule fixed in advance chose him)" | abs_gap 0.632 | `CS.abs_gap` (max over WR case studies), `CS.rule`; rule text in `E012_prereg.json deliverables[2]` | E012; prereg | OK |
| "worth only 0.14 games" | 0.1355 | `CS.cge_C2_2023_24_fit` (= `ct[T1].cge_C2_2023_24_seasons_fit`) | E012 | OK (rounding edge, §5) |
| "about 40% of the time" | 0.40 | Arith A6 with 11 = `CP pools.T1.n_with_10plus_games` | arith | OK |
| "nudged his forecast 0.3 mph toward his season average" | 0.142 yd/s | `CP cases[59278,T1].tracking_shift_C3_minus_C2` (= prior_C3 − prior_C2) | e012_case_priors.py | OK |
| "ratio 0.96, 95% CI 0.80–1.09" (all 15) | 0.960 [0.798, 1.089] | `V2.secondary_per_trait.T1["1"].blend_C3_vs_blend_C2.{ratio, ratio_ci95, n_players}` | E012 | ratio OK; CI OK-MC |
| "official electronic splits do not corroborate" | r −0.11 | `E007_leaderboard meta.post_hoc_official_splits.{corr_tracked_vs_official_10_40_time_beyond_forty, note}` | E007 | OK |
| "by his second game his own record outweighed the forecast" | w 0.538 > 0.5 | `CS.weight_C2[1]`; also `tr[T1].half_weight_game.C2.est` 1.95 < 2 | E012 | OK |
| "12 traits" | 12 | `len(E012 traits)` | E012 | OK |

### 2.2 §2 The Combine Exchange Rate

| Quoted | Value | Source key | Producer | Status |
|---|---|---|---|---|
| "12 game traits at WR, DL, OL and DB" | 12; 4 positions | `E012 traits`, `traits[*].position` | E012 | OK |
| "(WR top speed: 10 tracked plays)" | 10 | `tr[T1].min_plays_per_game` | E012 | OK |
| "w = k/(k + λ + CGE)" | — | `E012 amendments[0].finding` (amendment A1) | E012 (`AMENDMENTS`) | OK |
| "λ 1.5, CGE 0.5" | 1.480; 0.471 | `ct[T1].lambda.est`; `ct[T1].cge_C2.est` | E012 | OK |
| "four games give w = 0.67" | 0.672 | `E012 deliverables.weight_curves.T1.C2.w[4]` | E012 | OK |
| "standard error … = 0.33 mph (σ = 0.8 mph)" | 0.327; 0.80 | Arith A3 from `tr[T1].variance.sigma2.est` and `tr[T1].half_weight_game.C2.est` | arith | OK |
| "We price four priors" | 4 | `E012_prereg.json information_sets` (C0–C3) | prereg | — |
| "one amendment, made before any real-data run" | — | `E012 amendments[0].status` | E012 | OK |
| "reliability or odd/even split-half below 0.5 (post hoc)" | — | `PHR.per_trait[*].identification_rule`; `E012 post_hoc` contains PH7 | E012 | OK |

### 2.3 §3.1 and Table 1

| Quoted | Value | Source key | Producer | Status |
|---|---|---|---|---|
| "nine reliably measured traits" | 9 | count of `ct[*].identified` | E012 | OK |
| "after 2.0 (WR top speed) to 6.8 games"; heading "two to seven" | 1.951 (T1); 6.784 (T2) | `tr[T].half_weight_game.C2.est` over identified T (= λ + CGE = `deliverables.weight_curves[T].C2.half_weight_game`) | E012 | OK |
| "the position average is worth 1.5 to 6.8 games" | 1.480; 6.784 | `ct[T].lambda.est` over identified T | E012 | OK |
| "the Combine and draft 0 to 1.2 more" | 0; 1.227 (T6) | `ct[T].cge_C2.est` | E012 | OK |
| "(47–92% of appearances qualified)" | 0.469; 0.917 | `PHR.validation_2.qualifying_share_2025_rookies[*].share` | E012 | OK |
| "three to 12 appearances" | 3.3; 12.4 | Arith A1 | arith | OK |
| "half signal only after 11–23 qualifying games" | 10.84; 22.80 | `ct[T10, T8, T4].lambda.est` (not identified) | E012 | OK |
| "about 14–39 appearances"; "a rookie pass rusher's pressure rate needs about 22" | 13.6, 38.9; 22.1 | Arith A1 | arith | OK |
| Table caption "fit on 2023–25" | — | `deliverables.weight_curves[T].fit` | E012 | OK |

**Table 1**, one row per trait (T1, T7, T3, T6, T5, T12, T2, T9, T11, T10, T8, T4):

| Column | Source key | Producer | Status |
|---|---|---|---|
| Trait (qualifying game) | `tr[T].min_plays_per_game`; units from `E012_prereg.json traits[T].game_value` | E012; prereg | OK |
| Prior worth, λ + CGE | `ct[T].lambda.est` + `ct[T].cge_C2.est`; ‡ rows show "—" because their CGE is not priced | E012 | OK |
| Outweighed after (95% CI) | priced rows: `tr[T].half_weight_game.C2.{est, ci95_prereg_bootstrap.lo, .hi}`<br>‡ rows: `ct[T].lambda.{est, lo, hi}` (= `weight_curves[T].C0.half_weight_game`) | E012 (B1 bootstrap, 1,000 resamples) | est OK; **CI OK-MC** |
| ≈ Appearances | Arith A1 | arith | OK |
| Weight after 1/4/8 | priced rows: `weight_curves[T].C2.w[1,4,8]`<br>‡ rows: `weight_curves[T].C0.w[1,4,8]`<br>† rows: none | E012 | OK |
| Footnote "0: no out-of-sample gain (clipped)" | `ct[T].r2_cv.C2 ≤ 0` for the 5 zero rows | E012 | OK |
| Footnote † "up to 2.2 between-player SD" | max \|`PHR.per_trait[T9,T11].stationarity_and_game_unit.season_change_within_player[*].mean_tau`\| = 2.194; † rows = `ct[T9,T11].notes` containing "non-stationary" | E012 | OK |
| Footnote ‡ "Not priced" | `ct[T4,T8,T10].identified` false | E012 | OK |

### 2.4 §3.2 The Combine is worth a fraction of a game to about one

| Quoted | Value | Source key | Producer | Status |
|---|---|---|---|---|
| "runs from 0 (five traits) to 1.2 games" | 5 zeros; 1.227 | `ct[T].cge_C2.est` over identified T | E012 | OK |
| "bootstrap intervals all include 0 but are biased low" | — | `ct[*].cge_C2.lo ≤ 0`; `tr[T1].cge.C2.ci95_prereg_bootstrap.share_draws_below_est` 0.909; `PHR.simulation.interval_coverage.{T1,T6}.b1.share_upper_below_truth` | E012 | OK-MC (held in `--quick`) |
| "covering about 80% in simulation" | 0.775; 0.825 | `PHR.simulation.interval_coverage.{T1,T6}.b1.coverage` (PH6: 80 datasets × 250 B1 resamples each) | E012 | **OK-MC** |
| "WR top speed at 0.08–1.6 games, DL rush distance at 0.10–3.9, and the rest below 1.7–3.6" | 0.079–1.597; 0.105–3.900; uppers 1.669–3.553 | `ct[T].cge_C2_calibrated_lower95` and `_upper95` (PH6 belt: 26,400 simulated datasets) | E012 | **OK-MC** |
| "(the Combine as learned from 62–85 players …)" (also §7) | 62; 85 | `ct[*].n_players` | E012 | OK |
| "five traits reach 0.4–1.7 games (post hoc)" | 0.430–1.746 | `ct[T].cge_C2_vs_C0_same_cv` (PH8) | E012 | OK, but **superseded 2026-10-19**: the C0 reference credits ridge shrinkage of the class dummies to the Combine. Against a ridge on the dummies alone it is **two traits**, WR top speed 0.63 and DL rush distance 1.68 (§2.12) |
| "DL rush distance (1.2) … roster position alone is worth 2.4 games, the Combine −0.25 beyond it" (v2.4: "is body type and roster designation (OLB vs DE), not edge alignment: designation alone is worth 2.4 games", with E013 evidence in §2.12) | 1.227; 2.362; −0.247 | `ct[T6].cge_C2.est`, `.cge_role_reference`, `.cge_C2_beyond_role` | E012 | OK |
| "WR top speed (0.5) falls to 0.14 in the 2023–24 fit and to 0 without one draft class" | 0.471; 0.135; 0.0 | `ct[T1].cge_C2.est`, `.cge_C2_2023_24_seasons_fit`, `.cge_C2_leave_one_class_out_range[0]` | E012 | OK |
| "(r 0.61 with deep-route share) … adds 1.0 game beyond deep share" | 0.612; 1.045 | `PHR.per_trait.T1.references.role.{r_trait_deep_share, combine_beyond_ref}` (= `ct[T1].cge_C2_beyond_role`) | E012 | OK |
| "reliability 0.34–0.53, split-half 0.20–0.28" | — | `ct[T4,T8,T10].{rel_player_mean, split_half}` | E012 | OK |
| "cross-validated R² 0.21 … WR top speed (0.22)" | 0.209; 0.215 | `ct[T8].r2_cv.C2`; `ct[T1].r2_cv.C2` | E012 | OK |
| "(r +0.40 with qualifying games)" | 0.399 | `ct[T8].exposure_r_trait_log_games` | E012 | OK |

### 2.5 §3.3 Combine tracking adds nothing measurable

| Quoted | Value | Source key | Producer | Status |
|---|---|---|---|---|
| "−0.24 to +0.18 games across the nine traits, every interval including 0" | −0.242 (T6); 0.176 (T2) | `ct[identified].delta_cge_tracking.{est, lo, hi}` | E012 | est OK; intervals OK-MC |
| "2.1% worse (95% CI 0.4% to 3.9%; preregistered, 12 traits pooled)" | 1.0206 [1.0040, 1.0391] | `V2.primary.H3.{ratio, ratio_ci95, pass, claim, n_by_trait}` | E012 (10,000-resample pooled bootstrap; 100,000 sign flips) | ratio OK; CI OK-MC |

### 2.6 §3.4 2025 rookies

| Quoted | Value | Source key | Producer | Status |
|---|---|---|---|---|
| "(81 rookies, 222 player-traits)" | 81; 222 | `V2.primary.H1.{n_players, n_player_traits}` | E012 | OK |
| "pseudo-prospective: …" | — | `V2.design_label` | E012 | OK |
| "cut squared error 72% … (95% CI 60–80%" | ratio 0.2823 [0.2032, 0.3952] | `V2.primary.H1.{ratio, ratio_ci95}`; Arith A9 | E012 | ratio OK; **CI OK-MC** |
| "68% on the nine priced traits" | 0.321 | Arith A2 | arith | OK |
| "after two 52% (both preregistered)" | 0.4798 | `V2.primary.H2.ratio`, `.pass` | E012 | OK |
| "80% intervals held 77–81%" | 0.770–0.809 | `V2.calibration_80pct.C2[k].pooled_coverage`, k = 0..4 | E012 | OK |
| "captures 99% of that gain" | 0.990 | `PHR.validation_2.shrinkage_share_of_blend_gain["1"]` | E012 | OK |
| "beats the position average by 5% (95% CI 0.3% to 9.6%)" | 0.9507 [0.9037, 0.9970] | `V2.post_hoc_combine_prior_vs_position_mean.k0.comb_C2_vs_C0` | E012 | ratio OK; **CI OK-MC** |
| "mostly via WR top speed and the unpriced separation trait" | T1 −0.054; T4 −0.040 | `PHR.validation_2.ph3_drivers.k0[T].contribution_to_pooled` | E012 | OK |
| "on priced traits by 3.5%" | 0.9652 | `PHR.validation_2.ph3_drivers.k0_identified_traits_only.ratio` | E012 | OK |
| "after one game by 2.5%" | 0.9754 | `V2.post_hoc_combine_prior_vs_position_mean.k1.blend_C2_vs_C0.ratio` | E012 | OK |
| "none survives multiplicity control" | min q 0.29 | `PHR.validation_2.ph3_bh_q_over_10` | E012 | OK-MC (held in `--quick`) |

### 2.7 §4 Why the Combine is worth so little (E008, E009)

| Quoted | Value | Source key | Producer | Status |
|---|---|---|---|---|
| "Of 18 links" | 18 | `E008 summary.transfer.n_links` | E008 | OK |
| "16 show nothing … and the same 16 beyond the stopwatch alone" | 16; 16 | `E008 summary.transfer.{n_nulls, passes_c1}`; `E009 summary.{b1_n_nulls, b1_passes_c1}` | E008; E009 | OK-MC (held in `--quick`) |
| "median of 0.01 (at most 0.14; intervals allow 0.37)" | 0.013; 0.139; 0.367 | `E009 summary.draft_delta_abs_median`, `.draft_delta_signed_range`, `.draft_delta_largest_upper_limit.jackknife[1]` | E009 | OK |
| "**13 of the 16** pass the reliability gate (… both ≥ 0.5)" | 13 | `E008 summary.transfer.nulls_passing_c5`; per link `TM8[*].{rel_x, rel_y, c5_reliability_ge_05}` | E008 | **B1**: `rel_y` comes from the round-1 JSONs |
| "excludes partial correlations above +0.09 to +0.38" | 0.095; 0.384 | `E008 summary.transfer.ceiling_per_link_95` | E008 | OK |
| "(detectable effect 0.31–0.38 at 60–80 players)" | 0.313–0.378; n 61–82 | `TM8[single-position].{mde80_one_sided, n}` | E008 | OK |
| "effects of 0.5 or more remain possible for **six** links" | 6 | `E008 summary.transfer.corrected_beyond_b2_ge_05`, built from `TM8[*].ceiling_dis_b2`, which divides by the beyond-B2 target reliability | E008 | **B1** |

### 2.8 The round-1 inputs E008 reads (B1 detail)

E008 reads the committed round-1 JSONs in two ways. Only (b) feeds writeup numbers.

**(a) Assertion only** (`check_reproduction`). E008 compares each link's `n` and `raw_r` against the frozen historical record:
- `E001.json [WR,DB,DL,OL]`
- `E002.json primary.P1–P3.seeds["0"]`
- `E003.json primary[*].seeds[0]`
- `E004.json primary[*].seeds["0"]`
- `E005.json primary.P1–P2.seeds["0"]`
- `E006.json primary.P1–P3.seeds["0"]`

Reading these from the repo is legitimate. The rebuilt links matched them in every run in §9.

**(b) Game-target reliability `rel_y`** (`target_reliability`). It feeds criterion 5 ("13 of 16") and the reliability-corrected ceilings ("six links").

| Link | Key read | Regenerable on 2026-10-15? | With B1 fix (a), §9 run R2 (now committed, §0) |
|---|---|---|---|
| E002-P1 / E002-P3 | `E002.json reliability.game[(WR, turn, retention)]` and `[(WR, stop, peak_decel)]` → `split_half` (0.766 / 0.792) | no | identical |
| E002-P2 | `E002.json reliability.soe.soe_hoi_WR.reliability` (0.412) | no | identical |
| E003-P1..P3 | `E003.json game_reliability.{g_d10_rush, g_player_get_off_rush, g_quick_oe_rush}.split_half` (0.833 / 0.823 / 0.724, stored at 3 dp) | no | identical |
| E004-P1..P3 | `E004_reliability.json game_set.g_set_accel.reliability`, `outcomes.ppp_oe.reliability`, `outcomes.pr_oe.reliability` | **yes**, via `E004 --screen` (equal to 1e-16) | — |
| E005-P1 / P2 | `E005.json reliability.game[(backpedal_plays, bp_peak)].rel_min10` (0.847), `[(from_backpedal_ball_in_air, dv_10)].rel_min5` (0.574) | no | identical |
| E006-P1 / P2 / P3 | `E006.json game_reliability_split_half.game_S0.WR.reliability` (0.919); `.game_A0[p].reliability`; `yacoe_reliability[p].split_half_min5_per_half.reliability` | no | identical (≤ 1e-16) |
| E001-WR/DB/DL/OL | recomputed inside E008 (`e001_target_split_half`) | yes | — |

The Combine-feature reliabilities `rel_x` are recomputed inside E008 (`drill_profile`, `rel_round1_def`).

The round-1 Combine reliability blocks only fill the drill profile's `round1` comparison column, which nothing quotes:
- `E002 reliability.combine`
- `E003 combine_reliability`
- `E004_reliability combine_*`
- `E005 reliability.combine`

Their regenerated values differ wherever an ICC was negative, because the current `reliability.py` clips negative ICCs to 0 (N1).

### 2.9 §5 What tracking does show (E008, E010)

| Quoted | Value | Source key | Producer | Status |
|---|---|---|---|---|
| "All 28 tracked-40 metrics are stable traits (reliability ≥ 0.5; top speed 0.95–0.98)" | 28/28; 0.945–0.982 | `E008 summary.drill_profile.by_group_kind["40-yd dash \| repeat attempts"].{n, n_ge_05}`; `summary.drill_profile.forty.top_speed.rel` | E008 | OK |
| "Of 22 position-drill metrics with repeat reps, 2 are" | 22; 2 | `E008 drill_profile` rows with kind "repeat attempts" and a scope other than the 40; count with `rel_round1_def ≥ 0.5` | E008 | OK |
| "of 68 composites … 10 are" | 68; 10 | `summary.drill_profile.by_group_kind["position drill \| cross-drill"].{n, n_ge_05}` | E008 | OK |
| "55–63% of OL wave-drill variance lies between Combine sessions" | 0.548–0.626 | `E008 drill_profile[metric in wave rev_accel/peak_decel/peak_reaccel/exit_speed].share_session` | E008 | OK |
| "Drill intensity … is stable (0.74)" | 0.744 | `E010 combine.stop_rule.reliability` | E010 | OK |
| "(partial r 0.00 and −0.04; preregistered)" | −0.003; −0.042 (CIs include 0) | `E010 game.primaries.{P1,P2}.{partial_r, partial_r_ci95}` | E010 | OK |
| "(partial r +0.48, 95% CI +0.20 to +0.68)" | 0.478 [0.204, 0.683] | `TM8[E006-P1].{partial_r, ci_lo, ci_hi}` | E008 | OK |
| "without its three most supportive receivers it falls to +0.31 (−0.01 to +0.57)" | n − drop3_n = 3; 0.305 [−0.011, 0.566] | `TM8[E006-P1].{n, drop3_n, drop3_partial_r, drop3_ci_lo, drop3_ci_hi}` | E008 | OK |
| "about 90 first-round tests (q 0.06)" | 90; 0.063 | `E008 summary.multiplicity["E006-P1"].{m_round1_tests, bh_q_all_round1_tests}` | E008 `multiplicity()`, using the hard-coded constants `N_ROUND1_TESTS = 90` and `ROUND1_SMALLEST_OTHER_P` (N7) | OK-MC: q = 90 × the link's own FL p (0.0007 at 10,000 permutations). In `--quick`, p = 0.004 exceeds the smallest other round-1 p, so q is left undefined. |

### 2.10 §6 What to do with it

| Quoted | Value | Source key | Producer | Status |
|---|---|---|---|---|
| "weights become 0.40/0.73/0.84 after 1/4/8 games" | 0.403/0.730/0.844 | `E012 deliverables.weight_curves.T1.C0.w[1,4,8]` | E012 | OK |
| "roster role (DE, DT, NT, OLB) beats the Combine: worth 2.4 games, outweighed after 6.0 (post hoc; roles may postdate the draft)" (v2.4: "a roster-designation prior beats the Combine: outweighed after 6.0 games (post hoc; designations may postdate the draft)") | 2.362; 5.97 | `PHR.per_trait.T6.references.role.{reference, cge_ref, cge_ref_plus_C2}`; Arith A4 | E012; arith | OK |
| "3.5 times the forecast error" | 3.54 | Arith A7 | arith | OK |
| "point estimates 0 to 1.2, more than about 4 ruled out" | 0–1.227; 3.900 | `ct[identified].cge_C2.est`; max `ct[identified].cge_C2_calibrated_upper95` | E012 | estimates OK; "about 4" OK-MC |
| "stable traits in the 40 (reliability 0.66–0.79)" | 0.659–0.785 | `E008 summary.drill_profile.forty.dl_starts.rel` | E008 | OK |
| "mostly not in the DL pass-rush drill (0.23–0.51)" | 0.227–0.514; 1 of 6 ≥ 0.5 | `E008 drill_profile[scope "pass-rush drill attempts"].rel_round1_def` (metrics d05, d10, s05, s10, amax06, t3) | E008 | OK |
| "Most existing repeats are mirror-side or reaction reps" | 16 of 22 | `E008 drill_profile` rows of kind "repeat attempts" whose `scope` contains "reaction" or "opposite sides" | E008 | OK |
| "single-rep reliabilities of 0.15–0.39, two to six identical reps" | 0.147–0.390; 2–6 | `E008 drill_profile[scope "pass-rush drill attempts"].icc_unit`; Arith A5 | E008; arith | OK |
| "Session effects dominate the OL wave drill" | > 0.5 | wave `share_session` (as in §5) | E008 | OK |
| "the 2025 comeback drill was a different movement" | — | `E008 links["E002-P3"].caveats` | E008 | OK |
| "The 2025 ball data (no height)" | [2025] | `E011 data.draft_years` | E011 | **B2** |
| "time 87% of reached balls within one frame" (not quoted since v2.4) | 0.869 | `E011 detection.contact_time_check["share_within_0.1s"]` | E011 | **B2** |
| "no catch mechanic is yet a stable trait (best 0.42)" | 0.423 | max `E011 reliability[*].cross_drill.reliability` (catch_offset) | E011 | **B2** |
| "about 6–46 catching drills per player" | [6.1, 46.2] | `E011 reliability.catch_offset.drills_needed_range_from_icc_ci["for_0.5"]` | E011 | **B2** |

### 2.11 §7 Limitations

| Quoted | Value | Source key | Producer | Status |
|---|---|---|---|---|
| "62–85 players per trait" | 62; 85 | `ct[*].n_players` | E012 | OK |
| "three classes, one season for 2025 rookies" | 2023/2024/2025 | `tr[T1].by_class` | E012 | OK |
| "could be worth up to 15 games in simulation" | 14.93 (T2) | max `ct[*].cge_C2_population_signal_upper95` (PH6) | E012 | **OK-MC** |
| "low-play games are 2.0–3.5 times noisier (9 of 12 traits)" | 2.01–3.45; 9 | `PHR.per_trait[T].stationarity_and_game_unit.noise_ratio_lowest_vs_highest_plays_quartile` | E012 | OK |
| "position averages pool roles (edge with interior, corner with safety)" | — | `PHR.per_trait[T6,T12].references.role.reference` | E012 | OK |

Allowlisted non-result constants (the checker only confirms they appear):
- drill and feature names: "tracked 40", "tracked-40 speed profile", "his 40 peak", "40 vs DL pass-rush drill"
- data scope: "tracked at 10 Hz", "fit on 2023–25", "we fit on 2023–24", "2025 Combine", "2025 rookie(s)", "2025 games", "the 2025 test"
- illustrative game numbers: "game 1", "a typical game-1 top-speed play"
- table and CI labels: "Weight after 1/4/8", "(95% CI)", "(95% CIs)"

### 2.12 E013 numbers in draft v2.4 (added 2026-10-19)

Draft v2.4 (writer, 2026-10-19) quotes E013 in three places:
- the Section 3.2 correction;
- the Section 3.2 DL wording;
- the Section 3.4 **Roles** note.

`writeup/check_numbers.py` asserts each number below from its key.

**Abbreviations.**
- `E13` = `experiments/results/E013.json`.
- `S13` = `E13 post_hoc_review.summary`.
- `I13` = `E13 post_hoc_review.internal`.
- `PHR13` = `E13 post_hoc_review` (PH4–PH13, post hoc: decided after the E013 review).

**Producer.** Step S10b, `experiments/E013_role_aware.py` (function named per row), with `src/bdb27/probes/i_role_aware.py`. It runs from the fresh E012 outputs of S10. Status as in §2: OK = deterministic given E012's tables; OK-MC = Monte Carlo with fixed seeds.

**Proven.** Every key below reproduced in the 2026-10-19 notebook-path run (§9, R7): the whole E013.json matched the committed file.

**Quoted in the text**

| Quoted (v2.4) | Value | Source key | Producer | Status |
|---|---|---|---|---|
| §3.2 "against draft-class averages under the same cross-validation, WR top speed reaches 0.63 games and DL rush distance 1.68" | 0.6281; 1.6762 | `S13.e012_C2_vs_dummies_ridge_same_cv.{T1, T6}` (= `I13[T].same_cv_vs_dummies_ridge.e012_C2.est`) | `posthoc_internal_trait()` PH8 (same CV folds as E012, seed 0), `posthoc_summary()` | OK |
| §3.2 "(intervals include 0)" | [0, 1.651]; [0, 4.172] | `I13.{T1, T6}.same_cv_vs_dummies_ridge.e012_C2.{lo, hi}` | PH8. The post hoc 1,000-resample bootstrap on E012's resamples and CV seeds, `data/processed/e013_posthoc_bootstrap.parquet`, is regenerated in S10b | OK-MC |
| §3.2 "the rest under 0.4 (post hoc)" | ["T1", "T6"] | `S13.e012_C2_vs_dummies_ridge_ge_0.4` | PH8 | OK |
| §3.2 "may be conservative" | 0.63 > 0.47; 1.68 > 1.23 | `S13.e012_C2_vs_dummies_ridge_same_cv` against E012 `cge_table[T].cge_C2.est` | PH8; E012 | OK |
| §3.4 "explains 29%, 21% and 10% of between-player variance in WR hitch braking, DL rush distance and DB top speed" | 0.2894; 0.2075; 0.0955 | `S13.anova_share.{T3, T6, T12}` (= `I13[T].anova.share_of_between_player_variance.est`); traits `E13 internal[T].{position, name, role}` | PH6 (the preregistered ANOVA `lambda_role`) | OK |
| (the three are the BH q < 0.05 ones; the other six ≤ 6.5%) | q 0.0027 / 0.0027 / 0.0261; max other 0.064 (T7) | `S13.anova_share_bh_q_lt_0.05`, `S13.anova_bh_q`; `S13.anova_share` | PH6: 10,000 within-class permutations | q OK-MC; shares OK |
| §3.4 "yet, cross-validated, adds −0.06 to +0.16 games beyond the Combine and draft (post hoc)" | −0.0567 (T7); +0.1595 (T1) | `S13.role_beyond_combine_draft_range` (= min/max of `S13.role_beyond_combine_draft` = `I13[T].attribution_C2.role_beyond_c.est`) | PH9 | OK |
| §3.4 "Alignment from a rookie's first games did not improve 2025 forecasts (preregistered)" | H1 1.0279 [0.9959, 1.0695], H2 1.0023 [0.9843, 1.0220]; `pass` false | `E13 prospective.tests.primary.{H1, H2}.{ratio, ratio_ci95, pass, k}`; claims in `E013_prereg.json prospective_test.primary_tests` | `prospective_tests()`: 100,000 sign flips, 10,000-resample player bootstrap, BH over H1–H4 | ratio OK; CI and `pass` OK-MC |
| §3.4 "even after a post hoc precision correction" | 1.0115 [0.9901, 1.0362], p 0.826 | `PHR13.prospective.tests["calibrated:H1"].{ratio, ratio_ci95, p_one_sided}` | `posthoc_prospective()` (PH4) | ratio OK; CI and p OK-MC |
| §3.4 "the test's detectable effect (4%)" | 1 − 0.9558 = 4.4% | `PHR13.prospective.power.primary.H1.mde_ratio_one_sided_80pct_power` (arithmetic: outline §6, item 10) | PH7, from the test's own bootstrap draws | OK-MC |
| §3.4 "exceeds the 3% gain the fit implies" | 1 − 0.9665 = 3.3% | `PHR13.prospective.power.best_case_from_internal_fit.H1.ratio_vs_observed_position_average_error` | PH7 | OK |

**In the source comments and checker conditions (not printed as numbers in the text)**

| Statement | Value | Source key | Producer | Status |
|---|---|---|---|---|
| §3.2 DL wording "body type and roster designation (OLB vs DE), not edge alignment": OLB +0.110 beyond edge share and weight (p 4e-5); edge share −0.028 (p 0.57) | | `I13.T6.roster_designation.ols_hc1_role_weight_olb.{olb, edge_share, combine_weight}.{coef, p}` | `roster_designation()` (PH10). Descriptive and in sample. `nfl_position` may be the latest roster label, so it may postdate the draft | OK |
| Among pure edge players OLBs out-rush DEs, in all three classes | 0.1045 vs −0.0149; Welch p 0.00031; n 21 / 19 | `I13.T6.roster_designation.pure_edge_olb_vs_de.{mean_olb, mean_de, welch_p_two_sided, n_olb, n_de, by_class}` | PH10 | OK |
| Edge share adds < 0.1 games beyond the Combine and draft, and is 75% predictable from them | 0.0649; CV R² 0.7517 | `S13.role_beyond_combine_draft.T6`; `S13.role_share_from_combine_draft_cv_r2.T6` | PH9 | OK |
| The old "five traits" (the claim the correction removes) | T1, T2, T6, T7, T12 | `S13.e012_C2_vs_C0_same_cv_ge_0.4`, equal to E012 `cge_table[T].cge_C2_vs_C0_same_cv` (asserted inside E013) | PH8 | OK |

**Not quoted** (outline §12; they stay in E013.json and the notebook's printed E013 summary):
- H3 (0.964, p 0.077) and H4 (1.004);
- the CV role worth (0.77, 0.24) and the role-aware trust table (2.1–6.9 qualifying games);
- the matched-precision variants (1.007, 0.983), and role from his whole season (1.013);
- the influence table. `PHR13.prospective.t3_influence.top2[*].display_name` names two other rookies, who must not be named. The comparison never prints a `name` field (`REDACT`).

---

## 3. Figures (`writeup/make_figures.py`)

The script prints a manifest of every drawn or printed number with its source key. It also asserts, via `same(...)`, that every string the draft also quotes matches the draft's rounding.

| Figure | JSON keys | Parquets (`data/processed`) | Upstream steps |
|---|---|---|---|
| **1** `fig1_receiver_a.gif/.png` | `E012 deliverables.case_studies[59278,T1]`: prior_C2, prior_C0, season_mean, game_values, running_mean, blend_C2, weight_C2, z_predictive, n_games<br>`E012 deliverables.weight_curves.T1.C2.w[1]`<br>`E007_leaderboard players[59278, 58274]`: official_forty, flying20_mph, rank; `meta.n`; `meta.post_hoc_official_splits.note`<br>`E007 concept_pair.{fast,slow}.nfl_id`<br>`CP cases[59278,T1].prior_C3` (17.9 mph, post hoc marker); asserts on `tracking_shift_C3_minus_C2` and `pools.T1` | `e007_forty_attempts` (E007); `combine_tracking`, `game_tracking_2025`, `games`, `player_play` (S1); `e012_game_values` (**E012 full run only**) | S1, S5, S10 (full), S11 |
| **2** `fig2_cge.png` | `ct[T]`: lambda.est/lo/hi, cge_C2.est/lo/hi, cge_C2_calibrated_lower95/upper95, rel_player_mean, split_half, r2_cv.C2, n_players, cge_C2_2023_24_seasons_fit, cge_role_reference, cge_C2_beyond_role, exposure_r_trait_log_games, delta_cge_tracking (right panel)<br>`PHR.simulation.interval_coverage.{T1,T6}.b1`<br>`PHR.per_trait.T1.references.role.{r_trait_deep_share, combine_beyond_ref}` | — | S10 |
| **3** `fig3_rookie_test.png` | `V2.primary.{H1,H2,H3}`: ratio, ratio_ci95, n_players, pooled_a, pooled_b<br>`V2.pooled_curves[k]`: se_games, se_blend_C0, se_blend_C2<br>`V2.calibration_80pct.C2[0..4].pooled_coverage`<br>`V2.post_hoc_combine_prior_vs_position_mean.{k0.comb_C2_vs_C0, k1.blend_C2_vs_C0}`<br>`PHR.validation_2`: shrinkage_share_of_blend_gain["1"], blend_C0_vs_games.k1, ph3_drivers.k0_identified_traits_only, ph3_bh_q_over_10<br>`V2.secondary_per_trait[priced T]["1"]` (the 68%) | — | S10 |
| **4** `fig4_transfer_map.png` | `E009 transfer_map[*]`: n, b1_partial_r, b1_ci_*, b2_partial_r, b2_ci_*, b2_mde80_one_sided, b2_drop3_* (the b2_* columns are copied from E008.json)<br>`E009 summary`: b1_n_nulls, draft_delta_abs_median, b1_/b2_nulls_partial_r_expected_direction<br>`E008 summary.transfer`: n_links, n_nulls, corrected_beyond_b2_ge_05 (**B1**)<br>`E008 summary.multiplicity["E006-P1"].bh_q_all_round1_tests` | — | S8, S9 |
| **5** `fig5_drill_reliability.png` | `E008 drill_profile[*]`: rel_round1_def, n_players, share_session<br>`E008 summary.drill_profile`: by_group_kind, forty.top_speed.rel, wr_route_same_drill_retest.retest_icc_median<br>`E010 combine.stop_rule.reliability`<br>`E011 reliability.catch_offset.{cross_drill.reliability, cross_drill_boot_ci95}` (**B2**) | — | S6, S7, S8 |
| **Table 1** | markdown in draft.md; keys in §2.3 | — | S10 |

---

## 4. Derived arithmetic the notebook must print from keys (outline.md §6)

| ID | Number(s) | Formula |
|---|---|---|
| A1 | Table 1 appearances; "three to 12"; "14–39"; "about 22" | (`tr[T].half_weight_game.C2.est`, or `ct[T].lambda.est` for T4/T8/T10) / `PHR.validation_2.qualifying_share_2025_rookies[T].share` |
| A2 | "68% on the nine priced traits" | 1 − mean over identified T of `V2.secondary_per_trait[T]["1"].se_blend_C2`, divided by the same mean of `.se_games`. The same pooling over all 12 traits reproduces `H1.pooled_a`/`pooled_b` exactly. |
| A3 | "0.33 mph", "σ = 0.8 mph" | σ = √`tr[T1].variance.sigma2.est` × 3600/1760; SE(4) = σ / √(4 + `tr[T1].half_weight_game.C2.est`) |
| A4 | "outweighed after 6.0" (DL role prior) | `ct[T6].lambda.est` + `ct[T6].cge_role_reference` |
| A5 | "two to six identical reps" | ⌈(1 − icc)/icc⌉ over `E008 drill_profile[pass-rush drill attempts].icc_unit` |
| A6 | "about 40%" | 1 − (1 − 2(1 − Φ(2)))^n, with n = `CP pools.T1.n_with_10plus_games` (11) |
| A7 | "3.5 times" | `V2.primary.H1.pooled_b / pooled_a` |
| A8 | "0.3 mph" nudge | `CP cases[59278,T1].tracking_shift_C3_minus_C2` × 3600/1760 |
| A9 | "60–80%" | 1 − `V2.primary.H1.ratio_ci95`, ends reversed |
| A10 | every mph value | yd/s × 3600/1760 |

---

## 5. Numbers within 5e-4 of a rounding edge (compare these first after any change of platform, library version or code)

Method: re-run every numeric comparison in the checker and measure the stored value's distance to the nearest rounding boundary at the quoted precision. These values are deterministic given their seeds.

| Quoted | Stored value | Distance to edge | Key | Note |
|---|---|---|---|---|
| Table 1 "3.7" (OL pass-set lower CI) | 3.749910 | 9.0e-5 | `tr[T9].half_weight_game.C2.ci95_prereg_bootstrap.lo` | bootstrap; the quick run gave 3.77 |
| §7 "3.5" (noise ratio, max) | 3.450185 | 1.9e-4 | `PHR.per_trait[T11].stationarity_and_game_unit.noise_ratio_lowest_vs_highest_plays_quartile` | |
| §5 "+0.31" (drop-3 partial r) | 0.305204 | 2.0e-4 | `TM8[E006-P1].drop3_partial_r` | |
| §3.2 "0.10" (DL calibrated lower) | 0.104751 | 2.5e-4 | `ct[T6].cge_C2_calibrated_lower95` | simulation |
| Table 1 "0.42" (OL pressure, k = 8) | 0.424681 | 3.2e-4 | `weight_curves.T10.C0.w[8]` | |
| §6 "0.79" (40 first-step, max) | 0.785367 | 3.7e-4 | `E008 summary.drill_profile.forty.dl_starts.rel[1]` | |
| §4 "+0.09" (ceiling, min) | 0.094567 | 4.3e-4 | `E008 summary.transfer.ceiling_per_link_95[0]` | |
| §5 "0.95" (40 top speed, min) | 0.945483 | 4.8e-4 | `E008 summary.drill_profile.forty.top_speed.rel[0]` | |
| §3.2 "0.22" (T1 raw CV R²) | 0.215484 | 4.8e-4 | `ct[T1].r2_cv.C2` | |
| §1 and §3.2 "0.14" (2023–24 fit) | 0.135496 | 5.0e-4 | `ct[T1].cge_C2_2023_24_seasons_fit` | |

Also near a decision threshold, though not a rounding edge:
- `TM8[E004-P3].ceiling_dis_b2` = 0.51 and `TM8[E004-P1].ceiling_dis_b2` = 0.49 decide the "six links" count.
- `TM8[E005-P1/P2].rel_x` = 0.55 and `TM8[E002-P1].rel_x` = 0.56 sit near the 0.5 reliability gate.

---

## 6. Package versions (what the pipeline imports)

**Interpreter:** CPython **3.13.5** (Anaconda build, macOS arm64).

**Direct third-party imports** in `src/`, `experiments/` and `writeup/`: polars, numpy, scipy, scikit-learn, joblib, threadpoolctl, matplotlib, PIL (pillow).

**Listed in `pyproject.toml` but never imported:** duckdb, statsmodels, pandas.

`pip freeze`, filtered to the runtime closure of those imports:

```
cloudpickle==3.1.2
contourpy==1.4.0          # requires Python >= 3.12
cycler==0.12.1
fonttools==4.67.0
joblib==1.6.0
kiwisolver==1.5.1
matplotlib==3.11.2
narwhals==2.27.1          # scikit-learn 1.9 dependency
numpy==2.5.4              # requires Python >= 3.12
packaging==26.3
pillow==12.3.0
polars==2.0.0
polars-runtime-32==2.0.0  # polars 2.0 ships its engine as a separate package; versions must match
pyparsing==3.3.3
python-dateutil==2.9.0.post0
scikit-learn==1.9.1
scipy==1.18.1             # requires Python >= 3.12
six==1.17.0
threadpoolctl==3.7.0
```

- **Wheels:** all have manylinux x86_64 wheels on PyPI (checked 2026-10-15).
- **BLAS:** numpy/scipy here use Apple Accelerate; the Linux wheels use OpenBLAS (N3).
- **Not needed on Kaggle (dev only):** pytest, ipykernel, jupytext, kaggle.

**Install and fail-loudly pattern for the notebook** (internet on). Kaggle preinstalls its own numpy, polars and scikit-learn, and the kernel may already have imported them. So:
1. Install the pins.
2. Run every pipeline step as a **subprocess**, so each step starts in a fresh interpreter.
3. Run the assertion below inside that interpreter.

```python
PINS = {"numpy": "2.5.4", "scipy": "1.18.1", "scikit-learn": "1.9.1", "joblib": "1.6.0", "threadpoolctl": "3.7.0",
        "narwhals": "2.27.1", "cloudpickle": "3.1.2", "polars": "2.0.0", "polars-runtime-32": "2.0.0",
        "matplotlib": "3.11.2", "contourpy": "1.4.0", "cycler": "0.12.1", "fonttools": "4.67.0",
        "kiwisolver": "1.5.1", "packaging": "26.3", "pillow": "12.3.0", "pyparsing": "3.3.3",
        "python-dateutil": "2.9.0.post0", "six": "1.17.0"}
# pip install --no-deps --force-reinstall <k==v ...>   (or: uv pip install ... into a 3.13.5 venv, see B5)
import sys, importlib.metadata as md
bad = {k: (v, md.version(k)) for k, v in PINS.items() if md.version(k) != v}
assert sys.version_info[:2] >= (3, 12) and not bad, f"environment mismatch: python {sys.version}, {bad}"
```

Make `bdb27` importable with `PYTHONPATH=<repo>/src` or `pip install --no-deps -e .`. A plain `pip install -e .` would also pull duckdb and statsmodels and re-resolve the pins.

---

## 7. Kaggle runtime (4 vCPUs, ~30 GB RAM, 12 h per session) and how to stay well under 12 h

### 7.0 Measured: the clean-room run (2026-10-17)

§7.1–§7.3 were written before any full run. This is the measurement that replaces their estimates.

**Setup.** A fresh clone and venv on the 14-core M4 Pro, run under Kaggle-like caps:
- `--workers 4` (joblib's `n_jobs=-1` cap); OMP 3 per step; BLAS and polars capped at 4 threads;
- the machine otherwise quiet;
- the scripts at their committed `n_jobs=3`, with every statistical count as committed.

| Stage | Steps | Wall | CPU | Wall ÷ calibration call (38.4 s) |
|---|---|---|---|---|
| A | S01–S07 | 1.88 h | 5.21 h | 177 |
| B | S08–S09 | 1.49 h | 4.19 h | 140 |
| C | S10 (E012) | 2.33 h | 6.88 h | 218 |
| D | S11a–S14 | 36 s | 28 s | — |
| **Total** | | **5.71 h** | **16.3 h** | |

- `/usr/bin/time` for the whole run: 20,582 s real, and 55,248 + 3,362 s user + sys. Peak memory was 4.5 GB, in E006.
- **E012's phases:** about 48 min of values, bootstraps, validation and per-trait post hoc (C1 below), then 40 min of belt and 51 min of coverage simulations (C2). The simulations are about 65% of E012.
- **The calibration call (§7.3), timed in the same conditions:**
  - `n_jobs=3`: 38.4 s, against the 2026-10-15 reference of 31.9 s;
  - `n_jobs=4`: 30.0 s, against 24.9 s;
  - serial ×20: 19.1 s, against about 26 s.
  
  The outputs were identical. The parallel slowdown, 1.20×, was the same at both worker counts.
- The stage-to-call ratios here and in the two builder runs (C up to 229) are the multipliers `ra.calibrate()` uses (README.md, Run times).
- Against §7.1: E012 used 6.88 CPU-h, against the 5.8 h (20,730 s) assumed there, and stage A ran the full round-1 scripts (fix (a)) and E001's evaluate calls.

### 7.1 Estimate per step

> Superseded (2026-10-17) by §7.0's measurements and the calibration-anchored estimate in README.md (Run times): about 18 h of compute, 7.4 h for stage C (range 4.5–10.4 h). Kept for the record.

**Inputs.**
- For parallel steps: local CPU-seconds C (user + sys from `/usr/bin/time -l`). In these runs it counted the joblib workers: user/wall ≈ 2.5–2.9 at n_jobs 3.
- For serial steps: local wall time W.

**Two assumptions about Kaggle CPU sessions.**
- s: how many times slower a Kaggle vCPU thread is than an M4 Pro performance core.
- e: how many single-thread equivalents 3–4 busy workers get on 4 vCPUs (2 physical cores × 2 hyperthreads).

| Scenario | s | e | Parallel steps | Serial steps |
|---|---|---|---|---|
| Central | 3 | 2.5 | Kaggle wall ≈ 1.2 × C | ≈ 3 × W |
| Optimistic | 2.2 | 3.0 | ≈ 0.73 × C | ≈ 2.2 × W |
| Pessimistic | 4 | 2.3 | ≈ 1.74 × C | ≈ 4 × W |

Kaggle does not document its CPU model. Measure the real ratio with §7.3 before scheduling.

| Step | Local | Kaggle central | Kaggle range |
|---|---|---|---|
| S1 data cache | W 7 s (I/O-bound) | 1–3 min | ≤ 5 min |
| S2 game frames | W 120 s (serial) | 6 min | 4–9 min |
| S3 E001 feature store only | W < 30 s | 1–2 min | — |
| (S3 E001's five evaluate() calls; nothing quoted) | W 1,285 s | ≈ 1.3 h | 0.8–1.9 h |
| S4 round-1 probe features, crash versions | W 366 s in total | 18 min | 12–27 min |
| S4 with B1 fix (a): patched E002/3/5/6 run fully | C 5,566 s | 1.9 h | 1.1–2.7 h |
| S4 with B1 fix (b): reliabilities only | W ≈ S4 crash versions + seconds | ≈ 20 min | — |
| S5 E007 | C 3,627–3,747 s | 1.25 h | 0.75–1.8 h |
| S6 E010 (incl. post hoc) | C 3,862 s | 1.3 h | 0.8–1.9 h |
| S7 E011 | W 25–186 s (mostly I/O and OpenMP) | 2–10 min | ≤ 15 min |
| S8 E008 | C 5,132 s | 1.7 h | 1.0–2.5 h |
| S9 E009 | C 6,580 s | 2.2 h | 1.3–3.2 h |
| S10 E012 | C 20,730 s | **6.9 h** | 4.2–10.0 h |
| → of which the PH6 simulations (26,400 belt datasets + 160 coverage datasets) | C ≈ 14,300 s | 4.8 h | 2.9–6.9 h |
| → of which values, 12 × 1,000-resample bootstraps, validation 1–2 and post hoc | C ≈ 6,430 s | 2.1 h | 1.3–3.1 h |
| S11–S13 exports, figures, checker | W ≈ 30 s | 3–5 min | ≤ 10 min |
| **Total, fix (a), without E001's evaluate calls** | | **≈ 15.5 h** | **≈ 9.5–22 h** |

About E012's C:
- It comes from the committed run, made on a heavily shared machine (16,086 s wall, user/wall 1.3).
- The same code without the simulations (`E012_audit/full3.log`: 2,162 s wall, 6,263 user-s, user/wall 2.9) is consistent with the split above.
- Uncontended, E012 should take ≈ 2.0 h here (C/2.9).

Memory: the largest step peaks at 6.2 GB (E006 crash version) and S2 at 5.6 GB, both far below 30 GB.

### 7.2 Plan: stay well under 12 h without touching a statistical setting

No permutation, bootstrap, simulation or CV-repeat count changes. Only scheduling, parallelism and caching change.

> **What was implemented (2026-10-17).**
> - **Item 1:** the four stages, with B and C in parallel. The C1/C2 split was not built.
> - **Item 4, in part:** stage outputs pass on the probe parquets and results. The S1/S2 caches and E006's frame cache are rebuilt in each stage, because a saved output must not contain competition data.
> - **Not implemented: items 2, 3 and 5.** The scripts run at `n_jobs=3`, E001's evaluate calls and the full round-1 scripts run, and item 5's `OMP_NUM_THREADS=1` was only for `n_jobs` 4.
> - **Item 6 would gain little** while the scripts fix `n_jobs=3`.
> - The stage times above are 2026-10-15 estimates; see §7.0 and README.md (Run times).

**1. Chain four Kaggle notebooks.** Each is public and runs top to bottom; each saves its outputs, which the next attaches as input. B and C run concurrently in separate sessions, so the elapsed time is ≈ A + max(B, C) + D, and the longest single session is C.

| Stage | Contents | Central (range) | Outputs and notes |
|---|---|---|---|
| **A "features"** | S1–S7: cache, game frames, E001 feature store, round-1 probes + B1 fix, E007, E010, E011 | ≈ 4.6 h (2.8–6.7) with fix (a); ≈ 3.1 h (1.9–4.6) with fix (b) | `data/processed/*.parquet` (≈ 1.6 GB incl. both frame caches); E00x round-1, E004_reliability, E007, E010 and E011 JSONs |
| **B "transfer map"** | S8 E008 → S9 E009 | ≈ 3.9 h (2.4–5.7) | |
| **C "CGE"** | S10 E012 | ≈ 6.9 h at n_jobs 3 (2026-10-15 estimate; now 7.4 h, range 4.5–10.4 h, from the calibration call) | Split into C1 and C2 (below) only if the stage C rule says so (§7.3) |
| **D "writeup"** (the notebook the writeup links) | S11–S13: prints every §4 formula from keys, shows the figures and Table 1, runs `check_numbers.py` (must exit 0) | minutes | |

Splitting stage C, if needed:
- **C1** = values + bootstraps + validation + per-trait post hoc.
- **C2** = the PH6 simulations, run by trait block.
- This needs plumbing changes in E012 only: persist and reload `arts`/`internal` (or rebuild each trait's `TraitData` and λ without the bootstrap); run `sim_belt`/`sim_coverage` for a subset of traits; then finish with `--reuse-sims`.
- Results are identical under any split. Each simulated dataset has its own seed, independent of order and worker: `default_rng([SEED, i, gi, r])` for the belt and `[SEED, 99, i, d]` for coverage.

**2. Use all 4 vCPUs.**
- Make `N_JOBS` an env-var override (3 → 4) in E007–E012, and pass `n_jobs` explicitly in E001 (its harness default is −1).
- Results do not depend on n_jobs. `linkage.evaluate` draws every permutation in the parent. `H.bootstrap` draws all resample indices in the parent and seeds each resample's CV by its global index (`seed = c[0] + j`).
- Verified here: bit-identical results at n_jobs 1, 3 and 4 (§9, run R5). The E001 rerun at n_jobs 3 also matched the committed n_jobs −1 values.
- Local gain from 3 → 4 workers was 22% (§7.3). Expect less on 2 hyperthreaded cores.

**3. Skip work that no number uses.**
- E001's five `evaluate()` calls: E008 re-estimates the E001 links, and E001.json is read only for the n/raw-r assertion. Skipping saves ≈ 1.3 h.
- The round-1 linkage stages: fix (b) avoids them entirely and saves ≈ 1.5 h against fix (a).

**4. Cache once, pass on.** Build the S1/S2 caches, E006's frame cache and the probe parquets once in A, and attach them to B, C and D.

**5. Avoid oversubscription.**
- Keep BLAS/OpenMP at 1 thread inside workers. E012 already uses `threadpool_limits(1)`; set `OMP_NUM_THREADS=1` for subprocesses at n_jobs 4.
- E011's HistGradientBoosting already runs inside `threadpool_limits(3)`.

**6. Optional: a TPU-VM session.** If the account has one, its host CPU is large: with n_jobs ≥ 32, stage C would finish in well under an hour. Session limits and quotas differ from CPU sessions; verify them before relying on this.

### 7.3 Calibration: run first on Kaggle and compare with these local times

Measured here 2026-10-15 22:00 with the repo's `data/processed`, on a machine at load average ≈ 5 from other work. The script is reproduced below so it can be pasted into a Kaggle cell.

| Workload | Local time | Output to match |
|---|---|---|
| serial: `speed.forty_attempts()`, 793 attempts (repeat ×20 for a stable timing) | 1.3 s per call (clean room: 19.1 s for ×20) | — |
| parallel: `linkage.evaluate(E001-WR, full settings: 50×5 CV, 2,000 CV perms, 10,000 FL perms)` | n_jobs 1: 76.0 s<br>n_jobs 3: 31.9 s (clean room, 2026-10-17: 38.4 s)<br>n_jobs 4: 24.9 s (clean room: 30.0 s) | partial_r 0.321878, fl_p 0.016198, cv_perm_p 0.006997 (clean room: identical) |

**Decision rule (2026-10-17: one rule, the stage C rule in README.md, Run times).**
- Let T be the call's time on Kaggle at `n_jobs=3`, as `ra.calibrate()` measures it. If there are two measurements (the pre-check and stage A's own call), use the larger.
- Start stage C in one session only if 229 × T / 3600 + 0.25 ≤ 10.5 h. That is T ≤ 161 s, or r = T / 31.9 s ≤ 5.05; stages A and B then fit too.
- Above that, split C into C1 and C2 (§7.2) before running anything. The split changes E012, so the bundle changes, and stages A and B must be rerun with it.
- README.md gives the arithmetic and how the older thresholds map onto this one.

> **Superseded 2026-10-15 rule, kept for the record:** let r = the Kaggle time of the parallel workload at `n_jobs` 4, divided by 31.9 s. A parallel step then takes ≈ r × (C / 2.9) on Kaggle; r ≤ 3.5 means stage C fits in one session (≈ 7 h), and r > 4.5 means split C into C1 and C2.
> - It assumed §7.2's `n_jobs` 4 override, which was not implemented, and it divided an `n_jobs=4` time by the `n_jobs=3` reference.
> - On the `n_jobs=3` scale its cut-offs are r = 4.48 and 5.77 (× 31.9 / 24.9 = 1.28; §7.0 shows the slowdown is the same at both worker counts). They predict stage C at 9.3 h and 12.0 h.

```python
import time, polars as pl
from bdb27 import io, linkage, speed
d = (linkage.player_table().join(pl.read_parquet(io.PROCESSED / "combine_features.parquet"), on="nfl_id")
     .join(pl.read_parquet(io.PROCESSED / "game_traits.parquet"), on="nfl_id").filter(pl.col("position") == "WR"))
t = time.time(); [speed.forty_attempts() for _ in range(20)]; print("serial x20", time.time() - t)
for nj in (3, 4):
    t = time.time(); r = linkage.evaluate(d, "combine_top_speed", "game_top_speed", expected_sign=1, n_jobs=nj)
    print(nj, time.time() - t, r["partial_r"], r["fl_p"], r["cv_perm_p"])
```

---

### 7.4 Stage E (E013), added 2026-10-19

E013 needs about 510 × T seconds of compute over all its sessions: 4.36 h here at T = 31.0 s; about 16 h on Kaggle at the central T ≈ 112 s.
- **It has a stage of its own**, E, after C. Stage E has no split rule: `run_all.run()` stops it 10.5 h into a Kaggle session with its finished checkpoints kept, and the next stage E notebook resumes from them (README.md, Stage E across sessions).
- **This plan avoids the E012-style code split of §7.2.** It changes no E013 statistic and needs only the checkpoint-folder edit (README.md, Compatibility edits).

## 8. Other provenance notes

- `experiments/posthoc_writeup_v1/ph_inseason.json` and `ph_rev3.json` are **not used** by draft v2.2 or by the checker; they are leftovers from writeup v1.
- `check_numbers.py` checks draft text against JSON keys only; it does not check figures. `make_figures.py` checks the figures' copies of drafted numbers with `same(...)`, so together the two cover both.
- `E012_prereg.json`, `E009_prereg.json`, `E010_prereg.json` and `E012_audit/` are frozen inputs. The notebook reads them and never writes them.
- **No stopwatch-only forecast (`CP prior_C1`) is ever shown:** not in the text, a figure, or the notebook's output (outline.md §4). The checker tests it only as a true/false condition. The comparison withholds every `prior_C1` value (`REDACT` in `run_all.py`).

---

## 9. Verification log (2026-10-15, local M4 Pro 14-core, n_jobs 3; all writes went to sandbox copies under the session scratchpad)

| Run | What | Result |
|---|---|---|
| R0 | `check_numbers.py` on the committed draft | 374 OK, 0 failures; 1,888 rendered words |
| R1 | **Sandbox A**, rebuilt from `data/raw` with the committed code. Only change in the copy: scratch dirs redirected inside the sandbox, and n_jobs=3 passed in E001. Order: S1 → S2 → E001 → E002, E003, E004 --screen, E005, E006 → E007 (full) → E008, E010 `--quick` → E011 (full, plan file present) → E009 `--quick` → E012 `--quick` | **S1, S2, E001 store, all 13 probe parquets, e007_*, e011_*:** identical to the repo's `data/processed` (≤ 1e-15).<br>**Round-1 scripts:** E002/E003/E005/E006 crash (B1); E004 `--screen` OK.<br>**E007:** E007_leaderboard.json identical; E007.json and E001.json differ only by rev-3 fields (N1).<br>**E011:** identical (runtime aside).<br>**E008 `--quick`:** first attempt failed only at writing (B4: its quick output dir did not exist); rerun with the dir created ran clean.<br>**E009 `--quick`:** first attempt computed everything, then failed drawing its figure (B4: needs a full-run E008 parquet); rerun passed after copying it in. |
| R2 | **Sandbox C**: patched round-1 scripts (B1 fix a: four key renames + `_fit_resid`) run in full | All four rc 0.<br>Every `rel_y` input in §2.8 and every n/raw-r reference identical to the committed JSONs.<br>Wall: 476 / 638 / 339 / 601 s. |
| R3 | **Sandbox D**: `check_numbers.py` (exception-tolerant copy) against R1's results: E007 full, E011 full, E008/E009/E010/E012 `--quick` | **338 checks OK.**<br>All 39 failures are Monte Carlo quantities that `--quick` under-samples:<br>• Table 1 bootstrap CIs (30 vs 1,000 resamples)<br>• coverage ≈ 80% and the calibrated bounds (20 vs 200 simulations per cell)<br>• the population-signal bound<br>• H1 and k0 bootstrap CIs (1,000 vs 10,000)<br>• E008's q for E006-P1 and its "about 90": at 500 FL permutations p = 0.004, above the smallest transcribed round-1 p, so q is undefined (N7)<br>**Deterministic quoted keys vs committed, max \|diff\|:** E008 transfer_map 4e-16; E008 drill_profile 1.3e-15; E009 partial r and CIs 2.5e-16; E010 identical; E012 cge_table 9e-14; weight curves 6e-15; case studies 2e-14; H1–H3 ratios 1e-16.<br>Also identical: E008 `n_nulls` 16, `nulls_passing_c5` 13, `corrected_beyond_b2_ge_05` (six links), ceilings; E009 draft deltas 0.013 / 0.139 / 0.367 jackknife. |
| R4 | **Sandbox B**, read-only on the repo's `data/processed`: `make_figures.py`, `e012_case_priors.py`, `ph_pairs.py` | All 6 figure files byte-identical (27 s, 3.5 GB). Both post hoc JSONs identical. |
| R5 | n_jobs invariance: `linkage.evaluate` and `H.bootstrap` (T1) at n_jobs 1, 3 and 4 | bit-identical |
| R6 | Rounding-edge scan (§5) and the PyPI metadata of every pin (§6) | as reported |
| R7 (2026-10-19) | **E013 through the notebook's own path.** A `run_all` workspace held the committed E012 outputs: E012.json from git, the five e012 parquets of the run that wrote it, and the two S01 tables. Step S10b ran via `run_all.run()`: it was stopped at 18 min (`$BDB27_STOP_AFTER_H`), keeping 2 checkpoints, then resumed with `--resume` to the end (4.12 h; 4.36 h uninterrupted). | **E013.json:** 16,477 leaves. 16,340 are identical and 133 floats are within 1e-9 (largest absolute difference 4e-16). The 4 others are the documented metadata (`runtime_s`, `run_started_utc`, `reuse_stored_draws`, `runtime_note`). `compare()` PASS.<br>**Draws:** all seven e013 parquets are byte-identical to the stored draws behind the committed file.<br>**So every §2.12 key reproduces,** OK-MC rows included. Details: README.md, How this was verified. |

**Done since (2026-10-16/17): four full local runs**, which reproduce every value (README.md, How this was verified; runtimes in §7.0). Original note: **not done here: full-count runs of E008, E009, E010 and E012** (≈ 7 h together on this machine). With the counts unchanged and the seeds fixed, those should reproduce the committed OK-MC values. A full dry run of each is still the last check before Gate 3, and also the way to measure Kaggle times (§7.3). Sandbox logs are in the session scratchpad (`prov/logs/`), which is temporary.
