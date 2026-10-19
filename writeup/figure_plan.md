# Figure and table plan: writeup v2.6 (Combine Game Equivalents)

The limit is **fewer than 10 figures and tables combined**; our budget is **at most 8**. v2.1 uses **6** (Figures 1–5 and Table 1) and keeps **2 reserve slots** (R1, R2). Numbering follows first mention in `writeup/draft.md` (v2.1 renumbered the sections, not the figures).

Every final figure is drawn by `writeup/make_figures.py` from committed results and the `data/processed` cache, and the public notebook must regenerate it. `make_figures.py` registers every printed number with its source key and asserts it against the draft's rounding (`same(...)`). The `experiments/figures/*.png` files are research figures, not final ones.

**Status line.** Each figure below lists the v2.1 judge items as **done** (visible in the renders of 2026-10-15 19:39–19:40, updated by the 19:58 renders) or **to do** (a precise change for the figure builder). v2.2 (final checker) re-synced every caption with the 19:58 renders and moved Figure 5 to Section 5 as a separate finding. Judge items that change only the text are in `writeup/outline.md` section 10. v2.3 (2026-10-18) uses real names (user decision; outline sections 4 and 7): Isaiah Bond and Kyle Williams replace Receiver A and B in Figure 1. v2.4 (2026-10-19) integrates E013 in the text only (no new figure; outline section 12), shortens the Figure 1 caption, makes every mention of a named player in this plan neutral (outline section 4), and adds one optional wording change to Figure 2's DL note. **v2.5 (2026-10-19)** applies two mock judges' reviews (outline section 13). The figure builder's renders of 10:54–10:56 already carry most of their figure items; this plan marks each as done or to do (checked against the renders), updates the Figure 2, 3 and 5 captions, and lists two new to-dos for Figure 5 that follow from text cuts. **v2.6 (2026-10-19)** applies the final checker's findings (outline section 14) and re-syncs this plan with the figure builder's renders of 11:23 (`make_figures.py` 11:23): Figure 3's title and row header name the Combine + draft forecast and label its trim post hoc (F6); Figure 5's title says "repeatable within a session", its 0.5 line is the "reliability gate" and the drill-intensity row and footer carry its preregistered game null (F5); Figure 1's footer says the band is the rule's 80% interval for his level, narrower than the remaining-games interval Figure 3 checks, and the GIF's inset now holds the fixed last-20 averages. The Figure 3 caption names the Combine + draft forecast (v2.6). Figures 2 and 4 are unchanged since 10:58.

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
  - Titles state the finding and must not imply more than the text. In particular, no title may say the Combine "failed" for Bond.
  - Every post hoc element says "post hoc" on the figure.
- **Uncertainty.**
  - Draw 95% intervals. For E012 CGE, the simulation-calibrated bounds are the primary statement and the bootstrap interval is secondary. The bootstrap intervals are biased low: every simulated miss is low (`post_hoc_review.simulation.interval_coverage.*.b1.share_upper_below_truth`).
  - p, q and CI strings belong on the figures and in the notebook, not in the text (judge 1). The text keeps effect sizes.
- **Units.** Speeds in mph (yd/s × 3600/1760). Games are "qualifying games". Partial r is signed so that right = the preregistered direction.
- **Output.**
  - PNG at 2,400 px wide (8 in at 300 dpi). The type scale in `make_figures.py` (body 10 pt, notes ≥ 9.5 pt) renders at about 12–13 px in a 750 px Kaggle column. Check every figure at 750 px before freezing.
  - Figure 1: GIF ≤ 5 MB plus a PNG still (its last frame).
- **Captions** live in `draft.md`, count toward the word budget and stay at 25 words or fewer. Any caption change must keep the draft within both budgets in `check_numbers.py`: rendered words ≤ 1,900 and whitespace tokens ≤ 1,950 (v2.6: 1,882 words, at most 1,910 tokens; 18 words of margin. v2.5 had 2.)
- **Alt text** for every item lives in the notebook and in the Kaggle image alt field (`save(..., alt)` in `make_figures.py`).

---

## Figure 1: Isaiah Bond (Section 1, hook; animated)

- **Purpose.**
  - The hook, and the explicit Combine-tracking-to-game moment the pass/fail rule asks for.
  - One rookie, chosen by a preregistered rule. His Combine + draft forecast sat near the position average, his games ran fast, and the trust rule caught up by game 2.
- **Layout (v2.1 target).** 16:10, three panels: two on top, one full-width below.
  1. **Top left, the tracked 40 (animated; done).** Speed vs distance (mph vs yd, 0–40) for Isaiah Bond and Kyle Williams (same-40 receiver, nfl_id 58274, 4.40). The curves draw in at real time with a dot at the head. The last 20 yards are shaded, and the gap between the curves is filled. Table inset: 40 time and last-20 speed (23.2 / 22.2 mph). Note: "Fastest finish for his 40 time of 91 Combine receivers; official electronic splits do not corroborate it (post hoc)."
  2. **Top right (done in the 19:58 render; judge 1 item 9, judge 2 "missing visuals"): one of his game plays on the field.** This is the drill-vs-game side-by-side that AGENTS.md lists under "what wins".
     - Field strip in standardized direction (`bdb27.standardize`; `play_direction` from `player_play`), 0–60 yd downfield.
     - His path for one play, coloured by speed (sequential ramp from the validated palette, with a colour bar in mph). The play's peak speed is labelled.
     - Animated in real time after the 40 finishes, so the reader sees the drill and the game run side by side.
     - **Selection rule (fix before drawing, and state it in the notebook):** from his first qualifying game, the play whose max speed is closest to that game's 90th-percentile play max speed (the value E012 records, `case_studies[59278].game_values[0]` = 18.3 mph). The play is typical of the trait, not his fastest.
     - Label it "a typical top-speed play, game 1 (18.3 mph game value)". The play's peak is a figure-only value; the text does not quote it.
  3. **Bottom, his rookie season (done).**
     - x = qualifying game (0–15); y = "90th-percentile play top speed (mph)" (done: the label explains why it sits below the 23.2 mph Combine finish).
     - Orange dots: single games. Orange dashed line: his running average. Blue line: the in-season estimate. Blue ■ at x = 0: the forecast with its 95% range for a 15-game average (predictive SD, so z = 2.01 as in the text; done).
     - Grey line: position average 17.5. Black line: season average 18.9. Callout at game 2: "his games now outweigh the forecast".
     - **v2.2 (final):** one hollow navy ◆ just left of game 0 shows his forecast "+ Combine tracking 17.9" (post hoc, from the committed export `experiments/posthoc_writeup_v2/e012_case_priors.json`, `prior_C3`). The text quotes only the tracking nudge (+0.3 mph toward his season).
     - **Never drawn:** any forecast for him from another information set than the Combine + draft forecast and the + Combine tracking marker (outline section 4; a hard rule with real names). An early render drew one; the v2.2 commit removed it.
     - Footer (done): "Forecast and in-season estimate fit on 2023–24 (weight on his first game 0.37); Table 1 refits on 2023–25 (0.34)."
- **Title (done in the 19:58 render).** **"His forecast sat near average; by game 2 his own games outweighed it."** (The old title read as a Combine failure, which the text denies.) The subtitle's "rookie receivers with 10+ qualifying games" matches the text's selection pool (the 11 of 15).
- **Data (file → keys).**
  - `experiments/results/E012.json` → `deliverables.case_studies[nfl_id=59278, trait=T1]`: `prior_C2`, `prior_C0`, `game_values`, `running_mean`, `blend_C2`, `weight_C2`, `season_mean`, `abs_gap`, `z_predictive`, `n_games`, `cge_C2_2023_24_fit` (2023–24 fit).
  - `experiments/results/E007_leaderboard.json` → `players[nfl_id ∈ {59278, 58274}]`: `official_forty`, `flying20_mph`, `rank`; `meta.n`; `meta.post_hoc_official_splits`.
  - 40 traces: `data/processed/e007_forty_attempts.parquet` + `combine_tracking.parquet` (as now).
  - Game play (new): `data/processed` game tracking for 2025 REG, nfl_id 59278, first qualifying game (≥ 10 tracked plays); `player_play.play_direction`.
- **Caption (v2.4, 15 words incl. "Figure 1."):** "Isaiah Bond (animated): tracked 40, a typical game-1 top-speed play, then his season." The GIF animates all three panels in turn; "his season" covers the bottom panel: the Combine + draft forecast, the hollow post hoc + Combine tracking marker, his games and the in-season estimate. (v2.3's 20-word caption was cut for the budget: outline section 12.)
- **Alt text** (the current PNG metadata is authoritative; it also covers the 80% band, the post hoc + Combine tracking diamond, the 95% range and the GO route; earlier summary:) "Animation of Isaiah Bond and Kyle Williams, two receivers with the same 40 time: speed curves rise together, and Bond's keeps climbing over the last 20 yards (23.2 vs 22.2 mph). Next, one of his game plays is drawn on a field and coloured by speed. Below, his 15 rookie games: the forecast of 17.6 mph sits by the 17.5 mph position average, his games average 18.9 mph, and the in-season estimate rises toward his games, giving them more than half the weight from game 2."
- **v2.5 judge items.** Done in the 10:54–10:56 renders: the rule's 80% band around the in-season estimate (judge 2; what it is: see "Band" below); the play labelled with its route and down and distance ("GO route, 4th & 5") and the line of scrimmage (judge 1); "peak 18.3 mph ≈ his game-1 value" with the peak marked and the colour ramp at 8–20 mph (judge 2); the hollow + Combine tracking diamond in the legend (judge 1). **Done (checked in the 11:23 GIF, a mid-run frame):** the inset holds the fixed 40 times and last-20 averages; the live "speed now" column is gone (judge 1).
- **Band (v2.6, final checker).** The shaded band is the rule's 80% posterior interval for his level after k games (stored prior SD × √(1 − weight), `make_figures.py`), not the calibrated 80% predictive interval for a rookie's remaining-games average that Section 3.4 quotes (77–81%) and Figure 3's coverage panel checks; the predictive interval is wider (it adds σ²/m). The 11:23 footer says so: "Shaded band: the rule's 80% interval for his level (narrower than the remaining-games interval Figure 3 checks)". No text, caption, alt text or planning note may call the band calibrated or cite coverage for it; `check_numbers.py` fails on "calibrat", "coverage" or "held N" in any text sentence about Figure 1 or a band.
- **Accessibility.** No flashing; at most 3 loops, then hold the final frame; speeds printed as text, not only shown by motion. The right-panel build starts within 1 s of the 40's finish (judge 2: it took about 6 s).
- **Names (user decision, 2026-10-18).** Real names, exactly as in the committed results for their nfl_id: "Isaiah Bond" (59278) and "Kyle Williams" (58274) (`E007.json concept_pair[*].display_name` = `E007_leaderboard.json players[*].name` = `E012.json case_studies[*].name`). Nothing about either player's draft status, draft position or off-field matters anywhere in the figure: title, subtitle, labels, footer, alt text or file metadata (outline section 4). Say the selection rule "chose" him (outline section 4 lists the words to avoid).

## Table 1: Trust table (Section 3.1)

- **Purpose.** The week-to-week deliverable, in club units: for each trait, what the position average and the Combine + draft forecast are worth (λ + CGE games), when his own games outweigh the prior (with a 95% interval), roughly how many rookie appearances that is, and the weight on his own average after 1, 4 and 8 qualifying games.
- **Form.** A markdown table in `draft.md`, 5 columns × 12 rows plus a footnote. `check_numbers.py` asserts every cell, including the derived ones. v2.2 compacts the separators for the word budget: no spaces around pipes, `1.5+0.5`, `0.34/0.67/0.80`, `≥10 plays` (spaced pipes would add 140 markdown tokens).
  - Rows: 7 priced traits sorted by half-weight; then OL pass-set acceleration and DB retreat speed (†: levels shift between seasons, so no weights); then the 3 not-priced outcome traits (‡: position-average prior, sorted by λ).
- **Columns (file → keys; internal fit on 2023–25).**
  - Prior worth = `cge_table[T].lambda.est` + `cge_table[T].cge_C2.est` (1 dp each). For not-priced rows: λ + "—".
  - Outweighed after = `traits[T].half_weight_game.C2.est`, with `ci95_prereg_bootstrap`. Not-priced rows: λ with `cge_table[T].lambda.lo/hi`.
  - ≈ Appearances = outweighed-after / `post_hoc_review.validation_2.qualifying_share_2025_rookies[T].share`, rounded. This is arithmetic on committed keys. The E012 owner should promote it into `E012.json` so the notebook can print it from a key.
  - Weights = `deliverables.weight_curves[T].C2.w[k]` at k = 1, 4, 8. Not-priced rows use `weight_curves[T].C0.w[k]`.
  - Zeros: CGE clipped at 0 (footnote "0: no out-of-sample gain (clipped)").
- **Caption (draft, 19 words):** "Trust table (fit on 2023–25), prior = Combine + draft forecast. With only the position average, use k/(k + λ)."

## Figure 2: What priors are worth, in games (Section 3.2–3.3)

- **Purpose.** The exchange rate in one picture. Per trait, the position average is worth λ = 1.5 to 6.8 games (10.8–22.8 for the not-priced outcome traits), and the Combine + draft forecast adds 0 to 1.2, with the calibrated bounds. A second panel shows what Combine tracking adds (about 0).
- **Done (19:39 render).**
  - Grey ▼ λ per row with its bootstrap interval (judge 1 figure item; judge 2 item 7).
  - Simulation-calibrated two-sided brackets around each Combine value (judge 2 item 1).
  - Title "For priced traits, the position average is worth 1.5 to 6.8 games of a player's own record; the Combine and draft add 0 to 1.2".
  - Trait-agnostic x-axis ("Qualifying games of his own NFL record").
  - Not-priced rows print λ and the reliability numbers.
  - The hollow "outside its own interval" glyph is replaced by an asterisk and a one-line footnote.
  - The stray glyph on the WR top-speed tracking row is gone.
- **Done in the 19:58 render (checked by the v2.2 writer):** items 1–3 below (uncorrected R² 0.14 / 0.21 / −0.09 and r +0.40 in the not-priced rows; "every miss is low" in the subtitle; the WR route-usage note as footnote ᵃ). Item 4 remains to do (the render is 2,400 × 3,005 px).
- **To do (v2.1 list).**
  1. In each not-priced row, add the Combine + draft forecast's raw cross-validated R² before the reliability correction (`cge_table[T].r2_cv.C2`: WR separation 0.14, DL pressure 0.21, OL pressure allowed −0.09). For DL pressure, also print its exposure correlation (+0.40). The text now makes this point (Section 3.2, judge 2 item 5); the figure should show where the forecast fits but is not priced.
  2. Subtitle: add "every simulated miss of the bootstrap is low" so the figure and the text say the same thing about the bias.
  3. On the WR top-speed row, add one note: "partly route usage: r 0.61 with deep-route share; the forecast adds 1.0 game beyond it (post hoc)" (`post_hoc_review.per_trait.T1.references.role`). Without it, the figure answers judge 1's first question (fast, or running go routes?) nowhere.
  4. Height: 2,700 px is long. Drop the duplicated "Combine: 0.14 if fit on 2023–24…" note to one line if needed, and check the figure at 750 px.
  5. **(v2.4, optional; done in the 10:54 render)** Note ᵇ (DL rush distance) said "NFL roster position (may postdate the draft) alone is worth 2.4". That agrees with the text: Section 3.2 says "tracks body type and roster designation (descriptive)" (v2.5), and the Section 3.4 Roles note says edge share adds nothing detectable once roster designation (OLB vs DE) and weight are controlled (v2.6; E013 PH10). "OLB vs DE" never explains the 2.4 games (the label is four-level). If the line is touched, use "roster designation" and keep "may postdate the draft"; never "role" (the text uses role for alignment share).
- **Data.** `experiments/results/E012.json` → `cge_table[*]`: `lambda.{est,lo,hi}`, `cge_C2.{est,lo,hi}`, `cge_C2_calibrated_lower95/upper95`, `cge_role_reference`, `cge_C2_beyond_role`, `cge_C2_2023_24_seasons_fit`, `cge_C2_leave_one_class_out_range`, `delta_cge_tracking.{est,lo,hi}`, `identified`, `rel_player_mean`, `split_half`, `r2_cv.C2`, `exposure_r_trait_log_games`. Also `post_hoc_review.simulation.interval_coverage`.
- **v2.5 judge items.** Done in the 10:54 render: legend "90% interval: simulation-calibrated bounds (post hoc)" and "95% bootstrap interval (biased low)" (judge 2); rows in Table 1's order with † on OL pass-set acceleration and DB retreat speed (both judges); note ᵇ "roster designation (DE/DT/NT/OLB; may postdate the draft and reflect scheme and stance)" (judge 1; matches the text); the tracking panel's footnote ("bootstrap only; [0, 0] intervals not drawn; preregistered test in Figure 3"). (b) **Done** (checked in the 11:23 render): the thin bootstrap lines sit just below the bold brackets (judge 2). **To do:** (a) the render is 2,400 × 3,029 px with a seven-line footer; check legibility at 750 px and move duplicated notes (the subtitle and footer both explain the bootstrap) to the notebook if needed. The text no longer quotes tracking's −0.24 to +0.18 (v2.5), so the right panel is its only display.
- **Caption (v2.5, 19 words incl. "Figure 2."):** "Worth in games: position average; Combine + draft forecast (calibrated 90% bounds bold, 95% bootstrap thin); tracking's increment." It matches the render's legend.
- **Alt text** (the current PNG metadata is authoritative; earlier summary:) "Dot-and-interval chart of 12 game traits. For each, the position average is worth 1.5 to 6.8 games of a player's own record (10.8 to 22.8 for three outcome traits that are not priced), and the Combine + draft forecast adds 0 to 1.2 games, with simulation-calibrated bounds: WR top speed 0.08 to 1.6, DL rush distance 0.10 to 3.9, others below 1.7 to 3.6. A second panel shows Combine tracking adding −0.24 to +0.18 games."

## Figure 3: 2025 rookies (Section 3.4)

- **Purpose.** The pseudo-prospective test, decomposed honestly, plus the evidence that the rule's intervals are usable.
- **Done (19:39 render).**
  - Left panel cut to three lines: his own games only, the rule with the position-average prior, the rule with the Combine + draft prior (judge 1).
  - Right panel on one linear "change in squared error" axis with percent wording ("72% less error (60–80%)"). The stacked log/linear axes are gone (judge 2).
  - "Same, on the 9 priced traits" row added.
  - Hollow = post hoc only.
  - Footer says the pooling includes the 3 not-priced traits.
- **Done in the 19:58 render (checked by the v2.2 writer):** items 1–3 below (calibration panel "Its 80% intervals held 77–81%"; the "9 priced traits only: 68% less error · post hoc, no CI" row; q 0.29 kept).
- **To do (v2.1 list, now done).**
  1. **Calibration panel (judge 2 item 8; the text's Section 3.4 heading now claims "the rule is calibrated").** Add a small bottom-right panel: 80% interval coverage vs k = 0–4 for the Combine + draft rule, with an 80% reference line. Data: `validation_2.calibration_80pct.C2[k].pooled_coverage` (0.79, 0.77, 0.81, 0.81, 0.80). Title: "Its 80% intervals held 77–81%".
  2. **Priced-traits H1 row (judge 2 item 8).** Under "Rule (Combine + draft prior) vs his first game", add a post hoc row "on the 9 priced traits: 68% less error". Compute it with H1's own pooling (equal-weight mean over traits of `validation_2.secondary_per_trait[T]["1"].se_blend_C2` and `.se_games`, priced T only). `check_numbers.py` reproduces H1's pooled values exactly with this pooling. No interval is committed, so draw a point only, or have the notebook bootstrap it (labelled post hoc).
  3. Keep "q 0.29" on the figure. The text dropped p and q strings by design (judge 1).
- **v2.5 judge items.** Done in the 10:54 render: the alignment rows (E013 H1, "2.8% more error (0.4% less to 7.0% more) · preregistered", and the post hoc precision-corrected row, "1.1% more error (1.0% less to 3.6% more)"), the x-axis to +10% so the tracking and alignment intervals are legible (judge 1 item 8), the y-axis in units of between-player variance (judge 1), the 95% binomial band and per-trait dots in the coverage panel (judge 2). The text's Roles note quotes the same H1 numbers and points to Figure 3. **v2.6 (final checker F6), done in the 11:23 render:** the title is "On 2025 rookies, shrinkage does the work; against the position average, the Combine + draft forecast trims a few percent (post hoc)" (v2.5's "the Combine trims a few percent" named the forecast by one input and carried no post hoc label), and the middle row header is "The Combine + draft forecast vs the position average". **To do:** check the right panel at 750 px (10 labelled rows; judge 2 called it dense; the render is 2,400 × 2,902).
- **Caption (v2.6, 19 words incl. "Figure 3."):** "2025 rookies: error by games observed; shrinkage, the Combine + draft forecast, tracking and alignment; 80% interval coverage." (v2.5 said "the Combine" for the forecast; `check_numbers.py` now fails on it.) The text quotes H1 as "72% (95% CI 60–80%)", the figure's wording.
- **Alt text** (the 11:23 render's PNG metadata is authoritative; summary): left, squared error after 0 to 4 qualifying games (his own games alone about 7.6 after one game vs about 2.1 for either rule; the two rules overlap); right, the rule cuts error 72% after one game and 52% after two (68% on the priced traits, post hoc), a position-average rule does nearly the same, the Combine + draft forecast beats the position average by 5% before game 1 (post hoc), 3.5% on the priced traits and 2.5% after one game (both post hoc, intervals including 0), Combine tracking is 2.1% worse and first-game alignment 2.8% worse (preregistered; 1.1% when corrected, post hoc); bottom left, the rule's 80% intervals held 77 to 81% of outcomes.

## Figure 4: Transfer map, two baselines (Section 4)

- **Purpose.** The "why": beyond the stopwatch alone and beyond stopwatch + draft slot, 16 of 18 Combine-tracking links show nothing detectable. The draft slot changes almost nothing (median 0.01). Nulls are bounds. The text adds (v2.2) that 13 of the 16 nulls pass the reliability gate, so power and imperfect game targets, not unreliable drills, limit them (`E008.json summary.transfer.nulls_passing_c5`). The figure needs no change for this.
- **Done (19:39 render).**
  - One navy colour with ● / ◆ for the two baselines (no blue/orange; no hollow).
  - The post hoc "error-free reach" outline boxes are removed. They were the largest marks and read as CIs (both judges).
  - Legend "smallest effect detectable (80% power)".
  - Footnote cut to three lines.
  - The fragile E006-P1 row is daggered and has its "without its 3 most supportive receivers" sub-row.
- **Done in the 19:58 render:** item 1 below (open circles in the n column, legend "0.5 or more possible if error-free (post hoc)"). Item 2 remains to do.
- **To do (v2.1 list).**
  1. The error-free statement survives only in the text ("effects of 0.5 or more remain possible for six links"). Mark those six links (`E008.json summary.transfer.corrected_beyond_b2_ge_05`) with a small "≥ 0.5 possible if error-free (post hoc)" glyph in the n column.
  2. Check legibility at 750 px (2,400 × 2,700).
- **v2.5 judge items.** Done: the ‡ row (E001-WR) is explained in the footer as the pipeline sanity check that fails false-discovery control and the leave-3-out check (judge 1, judge 2); the drop-3 sub-row is navy, not grey (judge 2). The text now says both non-null links are fragile WR speed links (Section 4) and names the ceiling as "game top-speed ceiling", as the figure does. The detectable effect (0.31–0.38) and the draft slot's median change (0.01) are now figure-only.
- **Caption (draft, 17 words incl. "Figure 4."; unchanged in v2.2, it matches the render):** "Transfer map: partial correlations beyond the stopwatch, and beyond stopwatch plus draft slot (95% CIs)."

## Figure 5: Drill reliability (Section 5 since v2.2)

- **Purpose (reframed in v2.2).** A separate finding, not a reason for the transfer-map nulls (13 of 16 nulls pass the reliability gate): which Combine measurements are repeatable within a session. The tracked 40 is (28/28); position-drill signatures mostly are not (2/22 with repeat reps; 10/68 across drills). The like-for-like comparison backs the league recommendation for identical repeat reps. (v2.6: the title now says "repeatable within a session", not "a trait"; see below.)
- **Done (19:40 render).**
  - Repeat-rep dots split into DL pass-rush (the same movement each rep) and DB reaction / OL opposite side. 16 of the 22 are the latter (judge 2).
  - New right panel: the same six first-step metrics for the same DL players, in the 40 (0.66–0.79) vs in the pass-rush drill (0.23–0.51). This is judge 2's suggested slope comparison.
  - OL within-one-rep row moved to the notebook (judge 1: it suggested OL drills are traits).
  - Session-effect inset kept.
- **Done in the 19:58 render:** item 2 below ("1 of 6 at 0.5"); item 1 as a footer note ("No per-metric intervals were stored … counts against 0.5 are approximate"). Whiskers remain optional.
- **To do (v2.1 list).**
  1. Reliability SEs. With 72–118 players, SE ≈ 0.1, so counts against 0.5 are fragile (judge 2). Either add ±1 SE whiskers to the slope panel (notebook bootstrap, post hoc) or print n per row with "SE about 0.1 at these n" in the footer.
  2. In the slope panel, mark the single pass-rush metric at 0.51 (speed at 1.0 s) as "the 1 of 6 at 0.5". The text says "mostly not".
- **v2.5 judge items.** Done: the WR route-drills row has its count and "one-session retests: inflated" (judge 1); the footer explains it. **v2.6 (final checker F5), done in the 11:23 render (2,400 × 2,625):** (1) the title is "The tracked 40 is repeatable within a session; position-drill signatures mostly are not" (v2.5's "is a trait" contradicted the text's wording); the 0.5 line is labelled "reliability gate" (the text's term), not "trait threshold"; the footer says repeat reps are same-day retests that session effects can inflate, and "overstate repeatability" replaces "overstate a trait"; (2) the "Drill intensity" row reads "speed vs 40 peak; not counted / no game link (preregistered)", and the footer gives its preregistered game tests beyond stopwatch + draft slot: partial r 0.00 with typical-play intensity and −0.04 with snaps per season, 95% CIs ending at +0.12 and +0.06 (`E010.json game.primaries.P1/P2.partial_r` −0.003 and −0.042, `partial_r_ci95` upper ends 0.120 and 0.058; both CIs include 0). Optional: the session-effect bars have no intervals (judge 2); the footer already says none were stored.
- **Caption (v2.5, 16 words incl. "Figure 5."):** "Reliability by measurement type; first-step metrics, 40 vs DL pass-rush drill; OL session effects."

---

## Reserve slots (2; the maximum count stays 8)

Neither reserve slot is used in v2.5. The draft is at 1,882 of 1,900 rendered words (at most 1,910 of 1,950 whitespace tokens; v2.6), and a new item needs a caption (15–25 words) plus cuts elsewhere. E013's role-aware figures (`experiments/results/E013.json` `figures`) stay in the notebook: the text carries E013 as a 115-word note (v2.6), and Figure 3 draws its prospective rows. R2 (judge 1 item 5 again in v2.5) is deferred: it needs a user decision to name more rookies (outline section 7, item 3).

| Slot | Use if... | Form | Data |
|---|---|---|---|
| R1 | Judges or the user want the trust rule as a picture (judge 2 item 4) | **Trust curves.** One panel; 9 lines of weight on his own games vs qualifying games (0–17), direct-labelled at each half-weight crossing, with B1 bands; 0.5 reference. Largely duplicates Table 1's weight and CI columns, which is why it stays in reserve. | `E012.json deliverables.weight_curves[T].C2.{w, lo, hi, half_weight_game}` |
| R2 | The user wants it (names are decided: outline section 7, item 2; it still needs about 80–120 words of cuts) | **2025 rookie trust board** (judge 1 item 4). Per trait, 2 rookies: prior, k, his average, the rule's estimate ± 80% interval after k, and "games already overrule the prior?" (k > half-weight). Rows where the Combine + draft prior is flat are labelled "far from average", never "Combine wrong". No draft status. Costs about 80–120 words, so it needs cuts. | `E012.json deliverables.rookie_outliers_vs_prior`, `c2_prior_flat_by_trait`, `traits[T].variance` (σ², τ²), `weight_curves` |

**Notebook-only figures.** E012's full case-study grid (16 rookies), the research QC figures, v1's WR-signal robustness figure and the finishing-speed leaderboard.

**Cut order, if needed.**
1. Figure 1's game-play panel: keep the 40 and the season panel.
2. Figure 5's session inset.
3. Figure 3's right panel: move the ratios to the notebook.

Never cut Table 1, Figure 2 or Figure 4.

## Not quoted in the text (figure-only values)

- Kyle Williams's 22.2 mph and 4.40 (Figure 1; `E007_leaderboard.json`).
- The game-play peak speed (Figure 1 panel 2, new; recomputed from tracking in the notebook).
- λ intervals (Figure 2; they are in Table 1 only for the not-priced rows).
- The position-average rule vs games-only ratio (71%; Figure 3).
- Per-metric reliabilities (Figure 5).
- Bond's forecast with Combine tracking (17.9 mph), drawn in Figure 1: the text quotes only the +0.3 mph nudge. No other information set's forecast for him is drawn or quoted (Figure 1 section).
- The WR signal beyond the stopwatch alone (+0.42; Figure 4), cut from the text in v2.2.
- The DL get-off and DB top-speed CGE values (0.2, 0.3) are now quoted only in Table 1, not in Section 3.2's prose.
- (v2.5) Combine tracking's ΔCGE (−0.24 to +0.18; Figure 2 right panel); the draft slot's median change in partial r (0.01; Figure 4 subtitle); the detectable effect 0.31–0.38 (Figure 4); drill intensity's 0.74 and its game null (Figure 5 row and footer, done in the 11:23 render); the WR sanity link's +0.32 (Figure 4 ‡); E013's precision-corrected H1 interval (Figure 3; the text says only that it "agrees"); WR top speed's position-average weights 0.40/0.73/0.84 (Table 1's caption gives k/(k + λ)).
- (v2.6) Not quoted and not drawn anywhere (notebook only): E013's out-of-fold alignment worth (0.2–0.9 games for WR hitch braking, 0.8–1.1 for DL rush distance) and its "beyond the Combine and draft" range (−0.06 to +0.16). The first set a floored 0 against an in-sample value whose prospective counterpart went the other way, or credited edge share with what tracks roster designation; the second is a difference of floored fits for 5 of 9 traits (outline section 14).
