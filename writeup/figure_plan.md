# Figure and table plan: writeup v2.1 (Combine Game Equivalents)

The limit is **fewer than 10 figures and tables combined**; our budget is **at most 8**. v2.1 uses **6** (Figures 1–5 and Table 1) and keeps **2 reserve slots** (R1, R2). Numbering follows first mention in `writeup/draft.md` (v2.1 renumbered the sections, not the figures).

Every final figure is drawn by `writeup/make_figures.py` from committed results and the `data/processed` cache, and the public notebook must regenerate it. `make_figures.py` registers every printed number with its source key and asserts it against the draft's rounding (`same(...)`). The `experiments/figures/*.png` files are research figures, not final ones.

**Status line.** Each figure below lists the v2.1 judge items as **done** (visible in the renders of 2026-10-15 19:39–19:40) or **to do** (a precise change for the figure builder). Judge items that change only the text are in `writeup/outline.md` section 10.

## House style (every item)

- **One meaning per encoding, in every figure** (judge 2: hollow had five meanings; blue and orange had two each).

  | Series | Colour | Marker | Line |
  |---|---|---|---|
  | Position average | grey | ▼ | solid |
  | Combine + draft forecast | blue | ■ | solid |
  | + Combine tracking | navy | ◆ | dashed |
  | His own games | orange | ● or ■ | solid / dashed running average |

  - **Hollow = post hoc, and nothing else.**
  - Figure 4 has no prior series, so it uses navy only: ● beyond the stopwatch, ◆ beyond stopwatch + draft slot. It no longer uses blue/orange for target type.
  - Every series carries a marker and a line style, so colour is never the only cue. Re-run the `dataviz` validator on any new colour.
- **Words on figures.**
  - Trait names exactly as in Table 1.
  - No E-numbers, B1/B2 or C0–C3. Say "position average", "Combine + draft forecast" (never "Combine forecast" alone: it includes the draft slot), "+ Combine tracking", "stopwatch", "stopwatch + draft slot".
  - "Not priced" for T4, T8 and T10 (the text uses the same word).
  - Titles state the finding and must not imply more than the text. In particular, no title may say the Combine "failed" for Receiver A.
  - Every post hoc element says "post hoc" on the figure.
- **Uncertainty.**
  - Draw 95% intervals. For E012 CGE, the simulation-calibrated bounds are the primary statement and the bootstrap interval is secondary. The bootstrap intervals are biased low: every simulated miss is low (`post_hoc_review.simulation.interval_coverage.*.b1.share_upper_below_truth`).
  - p, q and CI strings belong on the figures and in the notebook, not in the text (judge 1). The text keeps effect sizes.
- **Units.** Speeds in mph (yd/s × 3600/1760). Games are "qualifying games". Partial r is signed so that right = the preregistered direction.
- **Output.**
  - PNG at 2,400 px wide (8 in at 300 dpi). The type scale in `make_figures.py` (body 10 pt, notes ≥ 9.5 pt) renders at about 12–13 px in a 750 px Kaggle column. Check every figure at 750 px before freezing.
  - Figure 1: GIF ≤ 5 MB plus a PNG still (its last frame).
- **Captions** live in `draft.md`, count toward the word budget and stay at 25 words or fewer. Any caption change must keep the draft at ≤ 1,950 words (`check_numbers.py`; 7 words of margin in v2.1).
- **Alt text** for every item lives in the notebook and in the Kaggle image alt field (`save(..., alt)` in `make_figures.py`).

---

## Figure 1: Receiver A (Section 1, hook; animated)

- **Purpose.**
  - The hook, and the explicit Combine-tracking-to-game moment the pass/fail rule asks for.
  - One rookie, picked by a preregistered rule. His Combine + draft forecast sat near the position average, his games ran fast, and the trust rule caught up by game 2.
- **Layout (v2.1 target).** 16:10, three panels: two on top, one full-width below.
  1. **Top left, the tracked 40 (animated; done).** Speed vs distance (mph vs yd, 0–40) for Receiver A and Receiver B (same-40 receiver, nfl_id 58274, 4.40). The curves draw in at real time with a dot at the head. The last 20 yards are shaded, and the gap between the curves is filled. Table inset: 40 time and last-20 speed (23.2 / 22.2 mph). Note: "Fastest finish for his 40 time of 91 Combine receivers; official electronic splits do not corroborate it (post hoc)."
  2. **Top right, NEW (to do; judge 1 item 9, judge 2 "missing visuals"): one of his game plays on the field.** This is the drill-vs-game side-by-side that AGENTS.md lists under "what wins".
     - Field strip in standardized direction (`bdb27.standardize`; `play_direction` from `player_play`), 0–60 yd downfield.
     - His path for one play, coloured by speed (sequential ramp from the validated palette, with a colour bar in mph). The play's peak speed is labelled.
     - Animated in real time after the 40 finishes, so the reader sees the drill and the game run side by side.
     - **Selection rule (fix before drawing, and state it in the notebook):** from his first qualifying game, the play whose max speed is closest to that game's 90th-percentile play max speed (the value E012 records, `case_studies[59278].game_values[0]` = 18.3 mph). The play is typical of the trait, not his fastest.
     - Label it "a typical top-speed play, game 1 (18.3 mph game value)". The play's peak is a figure-only value; the text does not quote it.
  3. **Bottom, his rookie season (done).**
     - x = qualifying game (0–15); y = "90th-percentile play top speed (mph)" (done: the label explains why it sits below the 23.2 mph Combine finish).
     - Orange dots: single games. Orange dashed line: his running average. Blue line: the in-season estimate. Blue ■ at x = 0: the forecast with its 95% range for a 15-game average (predictive SD, so z = 2.01 as in the text; done).
     - Grey line: position average 17.5. Black line: season average 18.9. Callout at game 2: "his games now outweigh the forecast".
     - The hollow navy ◆ (forecast with Combine tracking) is **removed (done)**. Its value lives only in the gitignored `data/processed/e012_forecasts_2025.parquet` (`mu_C3`). Restore it only after the E012 owner commits `case_studies[*].prior_C3` (and `prior_C1`) to `E012.json`. Then label it "+ Combine tracking (post hoc)" and add the T1 test: "across 15 rookie receivers, tracking changed error by a ratio of 0.96 (0.80–1.09)".
     - Footer (done): "Forecast and in-season estimate fit on 2023–24 (weight on his first game 0.37); Table 1 refits on 2023–25 (0.34)."
- **Title (to do).** The current title, "His Combine + draft forecast said average. Two games said otherwise.", reads as a Combine failure, which v2.1's text explicitly denies. Use: **"His forecast sat near average; by game 2 his own games outweighed it."**
- **Data (file → keys).**
  - `experiments/results/E012.json` → `deliverables.case_studies[nfl_id=59278, trait=T1]`: `prior_C2`, `prior_C0`, `game_values`, `running_mean`, `blend_C2`, `weight_C2`, `season_mean`, `abs_gap`, `z_predictive`, `n_games`, `cge_C2_2023_24_fit` (2023–24 fit).
  - `experiments/results/E007_leaderboard.json` → `players[nfl_id ∈ {59278, 58274}]`: `official_forty`, `flying20_mph`, `rank`; `meta.n`; `meta.post_hoc_official_splits`.
  - 40 traces: `data/processed/e007_forty_attempts.parquet` + `combine_tracking.parquet` (as now).
  - Game play (new): `data/processed` game tracking for 2025 REG, nfl_id 59278, first qualifying game (≥ 10 tracked plays); `player_play.play_direction`.
- **Caption (draft, 15 words; change only if panel 2 is added).** Current: "Receiver A: tracked 40 finish (animated), then his rookie games and the in-season estimate." With panel 2: "Receiver A: tracked 40 and a typical top-speed game play (animated), then his rookie games and the in-season estimate." That adds 6 words; 7 are available.
- **Alt text (to do: the old "Receiver A pulls ahead" was wrong).** "Animation of two receivers with the same 40 time: speed curves rise together, and Receiver A's keeps climbing over the last 20 yards (23.2 vs 22.2 mph). Next, one of his game plays is drawn on a field and coloured by speed. Below, his 15 rookie games: the forecast of 17.6 mph sits by the 17.5 mph position average, his games average 18.9 mph, and the in-season estimate rises toward his games, giving them more than half the weight from game 2."
- **Accessibility.** No flashing; at most 3 loops, then hold the final frame; speeds printed as text, not only shown by motion. The right-panel build starts within 1 s of the 40's finish (judge 2: it took about 6 s).
- **Names.** "Receiver A / B" until the user decides (outline section 7). With real names, show no draft information anywhere.

## Table 1: Trust table (Section 3.1)

- **Purpose.** The week-to-week deliverable, in club units: for each trait, what the position average and the Combine + draft forecast are worth (λ + CGE games), when his own games outweigh the prior (with a 95% interval), roughly how many rookie appearances that is, and the weight on his own average after 1, 4 and 8 qualifying games.
- **Form.** A markdown table in `draft.md`, 5 columns × 12 rows plus a footnote. `check_numbers.py` asserts every cell, including the derived ones.
  - Rows: 7 priced traits sorted by half-weight; then OL pass-set acceleration and DB retreat speed (†: levels shift between seasons, so no weights); then the 3 not-priced outcome traits (‡: position-average prior, sorted by λ).
- **Columns (file → keys; internal fit on 2023–25).**
  - Prior worth = `cge_table[T].lambda.est` + `cge_table[T].cge_C2.est` (1 dp each). For not-priced rows: λ + "—".
  - Outweighed after = `traits[T].half_weight_game.C2.est`, with `ci95_prereg_bootstrap`. Not-priced rows: λ with `cge_table[T].lambda.lo/hi`.
  - ≈ Appearances = outweighed-after / `post_hoc_review.validation_2.qualifying_share_2025_rookies[T].share`, rounded. This is arithmetic on committed keys. The E012 owner should promote it into `E012.json` so the notebook can print it from a key.
  - Weights = `deliverables.weight_curves[T].C2.w[k]` at k = 1, 4, 8. Not-priced rows use `weight_curves[T].C0.w[k]`.
  - Zeros: CGE clipped at 0 (footnote "0: no out-of-sample gain (clipped)").
- **Caption (draft, 19 words):** "Trust table (fit on 2023–25), prior = Combine + draft forecast. With only the position average, use k / (k + λ)."

## Figure 2: What priors are worth, in games (Section 3.2–3.3)

- **Purpose.** The exchange rate in one picture. Per trait, the position average is worth λ = 1.5 to 6.8 games (10.8–22.8 for the not-priced outcome traits), and the Combine + draft forecast adds 0 to 1.2, with the calibrated bounds. A second panel shows what Combine tracking adds (about 0).
- **Done (19:39 render).**
  - Grey ▼ λ per row with its bootstrap interval (judge 1 figure item; judge 2 item 7).
  - Simulation-calibrated two-sided brackets around each Combine value (judge 2 item 1).
  - Title "The position average is worth 1.5 to 6.8 games of a player's own record; the Combine and draft add 0 to 1.2".
  - Trait-agnostic x-axis ("Qualifying games of his own NFL record").
  - Not-priced rows print λ and the reliability numbers.
  - The hollow "outside its own interval" glyph is replaced by an asterisk and a one-line footnote.
  - The stray glyph on the WR top-speed tracking row is gone.
- **To do.**
  1. In each not-priced row, add the Combine + draft forecast's raw cross-validated R² before the reliability correction (`cge_table[T].r2_cv.C2`: WR separation 0.14, DL pressure 0.21, OL pressure allowed −0.09). For DL pressure, also print its exposure correlation (+0.40). The text now makes this point (Section 3.2, judge 2 item 5); the figure should show where the forecast fits but is not priced.
  2. Subtitle: add "every simulated miss of the bootstrap is low" so the figure and the text say the same thing about the bias.
  3. On the WR top-speed row, add one note: "partly route usage: r 0.61 with deep-route share; the forecast adds 1.0 game beyond it (post hoc)" (`post_hoc_review.per_trait.T1.references.role`). Without it, the figure answers judge 1's first question (fast, or running go routes?) nowhere.
  4. Height: 2,700 px is long. Drop the duplicated "Combine: 0.14 if fit on 2023–24…" note to one line if needed, and check the figure at 750 px.
- **Data.** `experiments/results/E012.json` → `cge_table[*]`: `lambda.{est,lo,hi}`, `cge_C2.{est,lo,hi}`, `cge_C2_calibrated_lower95/upper95`, `cge_role_reference`, `cge_C2_beyond_role`, `cge_C2_2023_24_seasons_fit`, `cge_C2_leave_one_class_out_range`, `delta_cge_tracking.{est,lo,hi}`, `identified`, `rel_player_mean`, `split_half`, `r2_cv.C2`, `exposure_r_trait_log_games`. Also `post_hoc_review.simulation.interval_coverage`.
- **Caption (draft, 20 words):** "Worth in games per trait: position average, and Combine + draft forecast with calibrated bounds (left); tracking's increment (right)."
- **Alt text.** "Dot-and-interval chart of 12 game traits. For each, the position average is worth 1.5 to 6.8 games of a player's own record (10.8 to 22.8 for three outcome traits that are not priced), and the Combine + draft forecast adds 0 to 1.2 games, with simulation-calibrated bounds: WR top speed 0.08 to 1.6, DL rush distance 0.10 to 3.9, others below 1.7 to 3.6. A second panel shows Combine tracking adding −0.24 to +0.18 games."

## Figure 3: 2025 rookies (Section 3.4)

- **Purpose.** The pseudo-prospective test, decomposed honestly, plus the evidence that the rule's intervals are usable.
- **Done (19:39 render).**
  - Left panel cut to three lines: his own games only, the rule with the position-average prior, the rule with the Combine + draft prior (judge 1).
  - Right panel on one linear "change in squared error" axis with percent wording ("72% less error (60–80%)"). The stacked log/linear axes are gone (judge 2).
  - "Same, on the 9 priced traits" row added.
  - Hollow = post hoc only.
  - Footer says the pooling includes the 3 not-priced traits.
- **To do.**
  1. **Calibration panel (judge 2 item 8; the text's Section 3.4 heading now claims "the rule is calibrated").** Add a small bottom-right panel: 80% interval coverage vs k = 0–4 for the Combine + draft rule, with an 80% reference line. Data: `validation_2.calibration_80pct.C2[k].pooled_coverage` (0.79, 0.77, 0.81, 0.81, 0.80). Title: "Its 80% intervals held 77–81%".
  2. **Priced-traits H1 row (judge 2 item 8).** Under "Rule (Combine + draft prior) vs his first game", add a post hoc row "on the 9 priced traits: 68% less error". Compute it with H1's own pooling (equal-weight mean over traits of `validation_2.secondary_per_trait[T]["1"].se_blend_C2` and `.se_games`, priced T only). `check_numbers.py` reproduces H1's pooled values exactly with this pooling. No interval is committed, so draw a point only, or have the notebook bootstrap it (labelled post hoc).
  3. Keep "q 0.29" on the figure. The text dropped p and q strings by design (judge 1).
- **Caption (draft, 15 words):** "2025 rookies: forecast error by games observed (left); shrinkage's and the Combine's shares (right)."
- **Alt text.** "Left: forecast error for 2025 rookies after 0 to 4 games. A rookie's own games alone are far worse early (about 7.6 after one game vs about 2.1 for either rule), and the two rules overlap. Right: the rule cuts error 72% after one game and 52% after two, and a position-average rule does nearly the same; the Combine + draft forecast trims 5% before game 1 (post hoc); adding Combine tracking is 2.1% worse. Bottom right: the rule's 80% intervals hold 77 to 81% of outcomes."

## Figure 4: Transfer map, two baselines (Section 4)

- **Purpose.** The first "why": beyond the stopwatch alone and beyond stopwatch + draft slot, 16 of 18 Combine-tracking links show nothing detectable. The draft slot changes almost nothing (median 0.01). Nulls are bounds.
- **Done (19:39 render).**
  - One navy colour with ● / ◆ for the two baselines (no blue/orange; no hollow).
  - The post hoc "error-free reach" outline boxes are removed. They were the largest marks and read as CIs (both judges).
  - Legend "smallest effect detectable (80% power)".
  - Footnote cut to three lines.
  - The fragile E006-P1 row is daggered and has its "without its 3 most supportive receivers" sub-row.
- **To do.**
  1. The error-free statement survives only in the text ("effects of 0.5 or more remain possible for six links"). Mark those six links (`E008.json summary.transfer.corrected_beyond_b2_ge_05`) with a small "≥ 0.5 possible if error-free (post hoc)" glyph in the n column.
  2. Check legibility at 750 px (2,400 × 2,700).
- **Caption (draft, 16 words):** "Transfer map: partial correlations beyond the stopwatch, and beyond stopwatch plus draft slot (95% CIs)."

## Figure 5: Drill reliability (Section 4)

- **Purpose.** The second "why": the tracked 40 is a trait (28/28); position-drill signatures mostly are not (2/22 with repeat reps; 10/68 across drills). The like-for-like comparison backs the league recommendation for identical repeat reps.
- **Done (19:40 render).**
  - Repeat-rep dots split into DL pass-rush (the same movement each rep) and DB reaction / OL opposite side. 16 of the 22 are the latter (judge 2).
  - New right panel: the same six first-step metrics for the same DL players, in the 40 (0.66–0.79) vs in the pass-rush drill (0.23–0.51). This is judge 2's suggested slope comparison.
  - OL within-one-rep row moved to the notebook (judge 1: it suggested OL drills are traits).
  - Session-effect inset kept.
- **To do.**
  1. Reliability SEs. With 72–118 players, SE ≈ 0.1, so counts against 0.5 are fragile (judge 2). Either add ±1 SE whiskers to the slope panel (notebook bootstrap, post hoc) or print n per row with "SE about 0.1 at these n" in the footer.
  2. In the slope panel, mark the single pass-rush metric at 0.51 (speed at 1.0 s) as "the 1 of 6 at 0.5". The text says "mostly not".
- **Caption (draft, 16 words):** "Reliability by measurement type: the tracked 40 is a trait; drill signatures mostly are not."

---

## Reserve slots (2; the maximum count stays 8)

Neither reserve slot is used in v2.1. The draft is at 1,943 of 1,950 words, and a new item needs a caption (15–25 words) plus cuts elsewhere.

| Slot | Use if... | Form | Data |
|---|---|---|---|
| R1 | Judges or the user want the trust rule as a picture (judge 2 item 4) | **Trust curves.** One panel; 9 lines of weight on his own games vs qualifying games (0–17), direct-labelled at each half-weight crossing, with B1 bands; 0.5 reference. Largely duplicates Table 1's weight and CI columns, which is why it stays in reserve. | `E012.json deliverables.weight_curves[T].C2.{w, lo, hi, half_weight_game}` |
| R2 | The user decides on names (outline section 7, decision 2) | **2025 rookie trust board** (judge 1 item 4). Per trait, 2 rookies: prior, k, his average, the rule's estimate ± 80% interval after k, and "games already overrule the prior?" (k > half-weight). Rows where the Combine + draft prior is flat are labelled "far from average", never "Combine wrong". No draft status. Costs about 80–120 words, so it needs cuts. | `E012.json deliverables.rookie_outliers_vs_prior`, `c2_prior_flat_by_trait`, `traits[T].variance` (σ², τ²), `weight_curves` |

**Notebook-only figures.** E012's full case-study grid (16 rookies), the research QC figures, v1's WR-signal robustness figure and the finishing-speed leaderboard.

**Cut order, if needed.**
1. Figure 1's game-play panel: keep the 40 and the season panel.
2. Figure 5's session inset.
3. Figure 3's right panel: move the ratios to the notebook.

Never cut Table 1, Figure 2 or Figure 4.

## Not quoted in the text (figure-only values)

- Receiver B's 22.2 mph and 4.40 (Figure 1; `E007_leaderboard.json`).
- The game-play peak speed (Figure 1 panel 2, new; recomputed from tracking in the notebook).
- λ intervals (Figure 2; they are in Table 1 only for the not-priced rows).
- The position-average rule vs games-only ratio (71%; Figure 3).
- Per-metric reliabilities (Figure 5).
- Receiver A's forecast with Combine tracking (`mu_C3`): not drawn until it is committed to `E012.json`.
