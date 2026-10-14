# LESSONS.md — mistakes log

Agents: read this at the start of every session. When you make a mistake, append an entry.

Format: `- YYYY-MM-DD — what went wrong → rule to prevent it`

- 2026-10-14 — Reported player_play as 327,019 rows from `wc -l`; the true count is 314,197 because `play_description` contains embedded newlines → count CSV rows with a real parser (polars/duckdb), never `wc -l`.
- 2026-10-14 — Named the package `bdb` as AGENTS.md said; it collides with the stdlib debugger module `bdb` (silently imported instead, and would break `pdb`) → package is `bdb27`; check `python -c "import X; print(X.__file__)"` before naming a package.
- 2026-10-14 — polars joins do not preserve row order by default; a test caught shuffled tracking frames → pass `maintain_order="left"` on joins of frame data, and always `sort(time)` before any diff/rolling over frames.
- 2026-10-14 — Built a game trait as the mean of a player's top-5 plays; a top-k over a variable number of plays measures exposure (snap count), which is confounded with draft capital → per-player game traits must be exposure-invariant (quantiles, rates, or fixed-size samples); check corr(trait, log plays).
- 2026-10-14 — Permutation null shuffled the raw feature, which also broke its correlation with the controls → false positives 2–4× nominal for features collinear with Combine times. Use Freedman–Lane (permute the part of the feature not explained by the controls), and test calibration with a collinear feature, not an independent one.
- 2026-10-14 — Per-fold R² on ~13-player test folds made go/no-go depend on the CV seed → normalize fold MSE by the full-sample variance, and check key results across several seeds.
- 2026-10-14 — (E002) Empirical-Bayes prior variance was estimated over all players, including those with 1–3 events, and collapsed (tau² 0.008 vs 0.14) → estimate EB priors on the players actually used, and check mean shrinkage weight against split-half reliability.
- 2026-10-14 — (E005) `onset.arg_max()` returns UInt32; subtracting from it underflowed and silently dropped 57% of attempts → cast polars index results to Int64 before arithmetic, and assert row counts after every filter.
- 2026-10-14 — (E003/E004/E002) Combine drill metrics carry large session/draft-year artifacts (tempo drift, a drill redefined in 2025) → center Combine drill metrics by session/year and include draft-year terms in baselines; check between-class variance of every Combine feature.
- 2026-10-14 — (E006) The ΔR² permutation test is sign-agnostic: a feature "passed" p = 0.02 with the opposite sign to the hypothesis → every preregistered test states an expected sign, and the inferential test must be directional.
- 2026-10-14 — (Gate 1) Probe builders wrote "clean null / tells you nothing" for underpowered tests (n ≈ 65, ~19 control columns) → state nulls as bounds ("partial r > X ruled out"), with the minimum detectable effect.
