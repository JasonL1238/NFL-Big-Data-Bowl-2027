# AGENTS.md — NFL Big Data Bowl 2027

This is the single source of truth for any AI agent working in this repo (Claude Code, Codex/OpenAI, Cursor, Gemini, Copilot, …). `CLAUDE.md` imports this file. Do not duplicate content elsewhere.

**At the start of every session, read `LESSONS.md` and `PLAN.md`** (iteration plan, evaluation protocol, decision gates).

---

## Working rules (from the user — follow always)

- **Never add yourself (the agent) as a contributor.** No `Co-Authored-By` lines, no "Generated with …" trailers, no agent attribution in commits, PRs, or files.
- **Ignore engineering cost.** I don't care how expensive certain procedures may seem. Quality over speed/cost.
- **If you make a mistake, note it down so you don't make it again.** Append an entry to `LESSONS.md` (date, what went wrong, the rule that prevents it).

### Behavioral guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

#### 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

#### 2. Simplicity First

Minimum code that solves the problem. Nothing speculative.
- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

#### 3. Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

#### 4. Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## The competition

- **Kaggle:** https://www.kaggle.com/competitions/nfl-big-data-bowl-2027
- **Prompt:** Build an analytical framework that uncovers *non-obvious, actionable* linkages between **10 Hz Combine sensor tracking data** and **future regular-season NFL game performance**.
- **Format:** judged analytics writeup only (no leaderboard). Kaggle Writeup + attached **public Kaggle Notebook**.
- **Hard limits:** ≤ 2,000 words, < 10 figures/tables, code in the notebook/appendix.
- **Pass/fail:** the writeup **must use the Combine tracking data** to link to regular-season performance, or it is not scored.
- **Deadline:** **Jan 6, 2027 23:59 UTC** (entry, team merge, and final). Judging Jan 7–25. Finalists present at the 2027 Scouting Combine.
- **Tracks:** Undergraduate/Graduate and Open. Teams of up to 5.
- **Prizes:** $100k total (each track: 3 × $10k finalists + 3 × $5k runners-up; $10k grand prize).
- **Judges:** NFL club analytics staff. 2027 weights are unconfirmed. In 2026 they were **Football 30% / Data Science 30% / Writeup 20% / Data Viz 20%**:
  - Football: is it usable week to week, does it handle football's complexity, is it novel?
  - Data Science: is it correct, are claims backed by data, are the models appropriate, is it innovative?

### Data (510 WR/TE/DB/OL/DL from the 2023–25 Combines: 383 drafted, 127 undrafted; ~2.3 GB; CC BY-NC 4.0)

Row counts verified with a CSV parser on 2026-10-14.

| File | Rows | Contents |
|---|---|---|
| `players.csv` | 510 | bio, college, draft round/pick (NA = undrafted), position |
| `combine_results.csv` | 510 | measurements, drill times, NGS scores |
| `combine_tracking.csv` | 463,189 | 10 Hz drill tracking (x, y, s, a, dis, dir), PLAYER and BALL rows; 40yd, 3-cone, shuttle, 59 position-drill names (some are naming variants of the same drill) |
| `player_career_successes.csv` | 510 | career snaps, games, Pro Bowl/All-Pro |
| `games.csv` | 1,002 | 2023–25 games: 816 REG, 147 PRE, 39 POST |
| `player_play.csv` | 314,197 | per player per play: EPA, routes, separation, pressure, coverage, `play_direction` |
| `game_tracking_{2023,2024,2025}.csv` | 22,654,290 | 10 Hz in-game tracking (x, y, s, a, dis, o, dir, event) |

Verified caveats:
- **Game tracking contains only these 510 players.** There are no opponents, ball, or QB (1–6 tracked players per play). Separation and pressure cannot be computed from tracking; use the `player_play` fields.
- Game tracking has **no direction column**. Join `play_direction` from `player_play` on (game_id, play_id).
- `games.csv` includes PRE and POST games. Filter `season_type == "REG"` for regular-season outcomes.
- `dir`/`o` are degrees clockwise from +y; `s` is yd/s and matches frame-to-frame motion. Combine `dis` under-counts distance (~0.58× the x/y step), so use x/y differences, never `dis`.
- Missing values are the string `NA`. 3-cone and shuttle are missing for 50–80% of players.
- 2025 draftees have only one season of game data.
- Combine **BALL** rows exist only for the 2025 class (54 WR/TE, 18 catching drills, x/y only, no height).
- For WR/TE, tracked game plays and their `player_play` rows are **pass plays only** (no runs); DB/DL/OL are tracked on every snap.
- `player_play.cushion` is the receiver-to-aligned-defender distance at the snap: it includes the receiver's own depth and split.
- Pass events in game tracking (`dropped_pass`, `pass_outcome_*`) are play-level, copied onto every tracked player of the play.

---

## Approach

**What wins (2022–2026 winners):**
- One **named, interpretable metric** that captures something the box score misses (e.g. Missed Tackle Opportunities, Continuous Pocket Pressure, CPAA).
- Credit attributed to players, rolled up into **leaderboards**.
- **Animated plays** with the metric drawn on the field.
- A clear football **"so what?"** that a coach or scout recognizes.
- **Validated two ways.**

**Judge advice (Ron Yurko):**
- Beat simple baselines (base rates, logistic regression).
- Report uncertainty (CV standard errors).
- Don't overclaim: the sample is small and plays aren't independent.
- The novelty should be in how the model is *used*, not its complexity.
- No gratuitous 3D graphics.

**Core idea:** don't just predict "career success" from drill times. That is crowded, confounded by draft position, and has n = 510. Instead, link **movement signatures inside the drills** to **the same movement in games**:
1. **Combine signatures** from drill tracking. These are things the stopwatch hides:
   - braking deceleration into cuts
   - speed retained through change of direction
   - turn radius at speed
   - time to top speed
   - first-step burst
   - acceleration smoothness
2. **Game traits** measuring the same quantities by position:
   - WR/TE/DB: route-break deceleration (own tracking) and separation at pass forward (`player_play`)
   - DL: get-off and burst
   - OL: lateral recovery in pass sets
3. **Linkage:** do Combine signatures predict game traits and outcomes (separation, pressure rate, EPA) **beyond traditional Combine times and draft pick**?
   - partial correlations or mixed models
   - cross-validated incremental R²
   - permutation tests
   - stability across draft classes
4. **Deliverables:**
   - a named metric
   - position leaderboards
   - an "undervalued at the Combine" list
   - side-by-side drill vs. game animations

Explore 3–5 candidate links in weeks 1–2, then commit to the strongest one that a scout would actually find useful.

---

## Workflow

- **Weekly loop:** hypothesis → feature → validation → log keep/kill in `experiments/log.md`. Every week ends with one writeup-ready figure and paragraph.
- **Mock judging (every 2 weeks):** score the draft against the rubric above, then run the checklist:
  - baselines beaten?
  - uncertainty reported?
  - any overclaiming?
  - ≤ 2,000 words and < 10 figures?
  - Combine tracking data used explicitly?
- **Compute:** Azure credits are available. Cost is not a constraint. Use as much compute as rigor needs (big bootstrap or permutation runs, GPU transformer embeddings, broad sweeps).

## Conventions

- **Never commit data.** `data/` is gitignored; the data license is CC BY-NC. Download it with:
  ```
  kaggle competitions download -c nfl-big-data-bowl-2027 -p data/raw
  ```
- **Standardize play direction** for all game tracking (`bdb27.standardize`). If `player_play.play_direction == "left"`:
  - `x → 120 − x`
  - `y → 160/3 − y`
  - rotate `o` and `dir` by 180°
- Use **polars/duckdb** for large tracking files, not pandas.
- Every result that goes in the writeup must be reproducible from the final public notebook, run top to bottom.

## Stack

- Python, polars, duckdb, scikit-learn, statsmodels, matplotlib
- `nflreadpy` (nfl_data_py is archived)
- Optional: SumerSports [SportsTrackingTransformer](https://github.com/SumerSports/SportsTrackingTransformer)

## Repo layout

```
AGENTS.md          # this file (source of truth)
CLAUDE.md          # imports AGENTS.md
LESSONS.md         # mistakes log — read every session
PLAN.md            # iteration plan, evaluation protocol, gates — read every session
data/raw, data/processed   # gitignored; processed = Parquet cache + feature store
src/bdb27/         # library code (io, standardize, features, linkage, viz). Not `bdb`: that name clashes with the stdlib debugger
tests/             # pytest; `-m "not slow"` skips calibration tests
notebooks/         # 01_eda … final_writeup
experiments/       # log.md (one entry per idea), E###_*.py scripts, results/E###.json
writeup/           # draft.md, figures/
```

## Timeline

| Weeks | Dates | Focus |
|---|---|---|
| 1–2 | Oct 13–26 | Setup, EDA, direction standardization, candidate links |
| 3–5 | Oct 27–Nov 16 | Build and validate the core metric |
| 6–8 | Nov 17–Dec 7 | Leaderboards, case studies, animations, robustness |
| 9–11 | Dec 8–28 | Writeup drafts, mock judging, cut to 2,000 words |
| 12 | Dec 29–Jan 5 | Freeze, rerun from scratch, submit a day early |
