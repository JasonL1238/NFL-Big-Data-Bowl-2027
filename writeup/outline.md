# Writeup outline: v1 (audit framing)

Companion to `writeup/draft.md` (the text) and `writeup/figure_plan.md` (figures and tables). v0 was written 2026-10-14/15 after Gate 2. v1 (2026-10-15) applies the round-3 mock-judging fixes (section 9) and still precedes the round-3 results (E009, E010, E011).

## 1. Framing and names

**Question (the user's chosen audit framing):** what does the Combine's 10 Hz tracking tell you about NFL games that the stopwatch and the draft slot don't?

**Name: the Residual Combine.** Two checks applied to any tracked metric:
1. **Repeat:** is it a trait of the player (reliability ≥ 0.5), not of the rep, drill or session, **and is its residual beyond the scouting baseline still a trait?** (v1: the residual check is what Beyond actually tests; the judges caught that v0 checked only the raw metric.)
2. **Beyond:** after the **scouting baseline** (protocol B2: every official Combine result, draft slot, position, draft year), does the residual show up in games? Reported as a partial r with a CI, a bound (as measured, and error-free) and the detectable effect.

Why this name:
- It says literally what the method computes: the part of a tracked metric left after the baseline.
- It reads as a scouting term and promises an audit, not a prediction score.

**The one candidate is named finishing speed** (v0: "hidden speed"): a receiver's tracked flying 20 beyond his 40 time. Renamed because the flying 20 is a standard scouting number that clubs can approximate from the official 20-yard split, so "hidden" overclaimed. E007's JSONs keep the field name `hidden_speed_mph`; the draft's header comment records the mapping.

Alternatives considered: "Beyond the Stopwatch" (generic), "The Second Stopwatch" (implies tracking is only timing), "Stopwatch-Plus" (sounds like a product).

**Three headline claims** (each sourced in the draft's comments):
1. **Repeat.** The tracked 40 is an excellent measurement (all 28 metrics ≥ 0.5; top speed 0.95–0.98), but its residual is noisier (0.29 for DL to 0.80). Position-drill signatures are drill-specific: 2 of 22 repeated-rep metrics reach 0.5 (1 of 6 in the one clean comparison), 10 of 68 cross-drill composites.
2. **Beyond.** 16 of 18 links show no detectable link. As measured, partial correlations above +0.09 to +0.38 are excluded per link (above +0.22 for six); for error-free measures, true effects of 0.5 or more cannot be excluded for six links.
3. **The one candidate.** WR finishing speed fits a modest link to game top speed (about +0.3) as well as none. Three overlapping versions on largely the same 61 WRs; fragile (leave-three-out, carried by three 2024 receivers, null in 2023, partly route usage, uncorroborated by official timing); no production link (positive effects on yards per route above about +0.10 ruled out).

**Plus a week-to-week deliverable (post hoc, new in v1):** one game of a receiver's own tracking forecasts his later game top speed as well as his best Combine measure; two or more games probably forecast it better.

**Audience.** NFL club analytics staff. Rubric (2026 weights): Football 30, Data Science 30, Writeup 20, Data Viz 20.

## 2. Structure and word budget

Counts exclude HTML comments, image links and placeholder text. Scratch script: `/private/tmp/claude-501/-Users-jasonli-Documents-GitHub-NFL-Big-Data-Bowl-2027/ca6dc0b9-7e4f-45b4-b4e4-1468819c8969/scratchpad/round3/wordcount.py` (temporary, not in the repo). It strips comments, image links and `[[TBD…]]`, then counts tokens containing a letter or digit (captions and table cells included).

| Section | Purpose | Key content | Figures | Words now |
|---|---|---|---|---|
| Title | — | "The Residual Combine: what 10 Hz tracking tells you that the stopwatch and the draft slot don't" | — | 17 |
| 1. Same 40, different finish | Hook, question, framework, answer | Two 2025 WRs, same 40 (largest gap of 172 pairs), flying 20 23.2 vs 22.2 mph, rookie top speeds 19.7 vs 19.3, fewer yards per route; 81% of the flying 20 explained by the 40; Repeat (incl. the residual) and Beyond; answer | Figure 1 (animated), Table 1 | 424 (incl. table 118, Table 1 caption 26, Figure 1 caption 14) |
| 2. Data and method | Data Science credibility | Data limits; signatures; exposure check (≤ 0.32 movement, ≤ 0.40 outcomes); Repeat on metric and residual; Beyond = one-sided Freedman–Lane partial r given the baseline, CI, MDE, CV R² ± SE; prereg (direction; one in hypothesis text), re-estimated under a tightened protocol, > 120 tests, BH, two internal adversarial reviews | — | 228 |
| 3.1 Repeat | Reliability first | 28/28; residuals 0.62–0.80 (WR/OL/DB) vs 0.29 (DL); 2/22 but 16 are reaction or opposite-side, 1/6 in the clean DL comparison; burst 0.38 vs 0.73; 10/68; route signatures drill-specific; drill vs game break speed r 0.01 vs 0.47; wave 55–63% session | Figure 2 | 177 + [[TBD-E010]] |
| 3.2 Beyond | Transfer map with bounds | 16/18 null; population-bound wording; above +0.22 excluded for six; joint; error-free up to +0.73; MDE 0.31–0.38; no outcome link without 2025 | Figure 3 | 140 + [[TBD-E009]] |
| 3.3 The one candidate | The positive result, told honestly | Finishing speed defined; clubs can approximate it; three versions with honest labels; primary passes criteria 1–6 under rev 3 but not all-round BH; +0.51 post hoc estimator; CV R² 0.18 → 0.23, excess ± SE; one signal; fragile (leave-three-out + 2024 draftees, null 2023, usage control, timing + tracking-artifact alternative); modest or none; no production link; leaderboard pointer | Figure 4 | 380 |
| 3.4 Why the rest did not transfer | Diagnosis | 13/16 (12 on the residual) pass, and the carried drill signatures were the more reliable ones; proxies (route shape 0.88, separation precedes break, alignment three quarters of cushion, DB retreat partly scheme, DL blockers); small n | — | 132 |
| 4. What to do with it | Actionable recommendations | Clubs: standardize on the tracked 40 (adds at most a small WR signal; the 20-yard split may hold it); drill movements don't generalize, run Repeat (metric + residual) and Beyond; game traits in games, week to week (0.56 vs 0.57 after one game; probably better, about 0.77, after two). Combine operations: two reps, fixed tempo and frozen definitions, ball tracking, orientation, reconciled timing | Figure 5 (new) | 240 + [[TBD-E011]] |
| 5. Limitations | No overclaiming | Small samples; selection on playing time; no independent confirmation; demanding control; many tests | — | 91 + [[TBD-E009]] |
| Code | Reproducibility | Notebook link | — | ~10 |
| **Total** | | | Table 1 + Figures 1–5 = **6** of 9 | **1,831** |

**Round-3 fill-in budget (v1, reduced): at most 125 words.**
- E009: ≤ 40 in 3.2, plus ≤ 10 in Limitations, plus a Table 1 column only if a verdict changes (≤ 15).
- E010: ≤ 45.
- E011: ≤ 30.
- REV3: resolved in v1 (0 words).

That gives **at most about 1,956–1,971 words**, leaving only about 30–45 words of margin. **If a fill-in runs over, cut in this order** (about 230 words available):
1. Render Table 1 as a scorecard graphic (frees about 140; recommended by the DS judge; the user's call, section 7).
2. The leaderboard sentence in 3.3 (about 20; the leaderboard lives in the notebook).
3. The "Signatures" sentence in Section 2 (about 20).
4. Combine-operations item 5, "reconciled timing" (about 8; also covered in 3.3).
5. The DL "first-second window runs into blockers" clause in 3.4 (about 9).
6. The exposure clause in Section 2 (about 15; move it to the notebook).
7. The "fewer yards per route" clause in the hook (about 6).

## 3. Round-3 placeholders: what to write, by outcome

All round-3 tests are labeled in the text. E009 is "secondary, not blind" (its prereg says so). E010 is "preregistered primary (round 3)". E011 is "feasibility" (descriptive).

**[[TBD-E009]] Beyond-B1 column** (3 places: Table 1 note, 3.2, Limitations). Source: `experiments/results/E009.json` (not yet written; E009 was running at v1 time).
- **Nothing changes** (same links pass; partial r barely moves): one sentence in 3.2, "Controlling only for the official Combine results (not the draft slot) changes no verdict; the draft slot absorbs a median of X in partial r." Drop the Table 1 column. Limitations: "removing it changes no verdict".
- **More links pass with B1:** report them as "information beyond the stopwatch that the draft slot already reflects". Never "market inefficiency" or "clubs missed it". Add the column to Table 1 and the hollow markers to Figure 3; take the words from the cut list (Table 1 as a graphic first).
- **The WR signal weakens or strengthens:** update 3.3's bullet list with the B1 estimate; keep the "one candidate" framing.

**[[TBD-E010]] Drill intensity** (after Figure 2). Source: `E010.json`, `E010_prereg.json` (E010 was running at v1 time).
- **Repeat fails** (cross-drill reliability < 0.5, the prereg stop rule): "stopped at Repeat (reliability X); not tested against games." Add one point to Figure 2.
- **Repeat passes, primaries null:** one sentence with the partial r, 95% CI and the bound, and a Table 1 row "Position-drill intensity".
- **Repeat passes, a primary passes the rev-3 criteria:** it becomes a second candidate. Change Section 1 ("one candidate" → "two candidates"), Table 1 and 3.1; use reserved slot R1; re-check the word budget with the cut list.
- **Always:** report the without-2025 result (P2, snaps per season, is an outcome).
- E010 also directly tests the "position drills run below full speed" reading that v0 asserted without evidence; v1 removed that claim. If E010 confirms it, one clause can come back in 3.1.

**[[TBD-E011]] Ball-tracking feasibility** (Combine operations, item 3). Source: `experiments/results/E011.json`. **It now exists** (written 2026-10-15) but is not yet in `experiments/log.md` and not yet reviewed, so v1 keeps the placeholder. What E011.json already states (`plan.already_seen_before_this_plan`): ball rows exist for the **2025 class only**, 551 attempts, 54 players (37 WR, 17 TE), 18 drill names, x/y only (no ball height). Once E011 is logged, a ≤ 30-word fill could read: "In our data the ball is tracked only in 2025 (54 receivers, 18 drills, no height); [catch-point timing is / is not] measurable reliably (reliability X)." Use reserved slot R2 only for a compact coverage table.

**[[TBD-REV3]]: resolved in v1.** The 3.3 primary bullet now reads "It passes go criteria 1–6 under our current rules, but not false-discovery control across ~90 round-one tests (q ≈ 0.06)".
- Basis: `ph_rev3.py` (scratch) re-ran the working-tree rev-3 `linkage.evaluate` on E006-P1 exactly as computed in round 1 (E008's link sample, AS_R = 20 target; same settings as E008), seeds 0–2. `checks.loco_classes_pass` = 2/3 on every seed: the 2023 and 2025 held-out excesses (+0.16, +0.10) exceed one null SD (0.02–0.05); 2024 is negative. Criteria 1, 2, 4–6 are unchanged from E008.
- The same script re-scores E001-WR under rev 3 (criteria 1–3 pass, LOCO 3/3; it still fails criteria 4 and 6, as in E008) and the post hoc AS_R = 1000 target, which reproduces E008's +0.507 exactly and passes criteria 1–3 under rev 3 (LOCO 2/3); without 2025 +0.39 [+0.02, +0.67]; leave-three-out +0.35 [+0.04, +0.60] (all post hoc; for Figure 4).
- **Action for the owner of E008:** promote this re-scoring into a results JSON (e.g. an E008 addendum or the notebook) so the claim is traceable to the repo, not to the scratchpad.

**[[TBD-NOTEBOOK-LINK]]:** the public Kaggle notebook URL.

## 4. Wording guardrails (from LESSONS.md, the E007/E008 reviews and the round-3 mock judging)

**Say:**
- "no detectable link";
- "for the measures as built, partial correlations above X are excluded" (a CI excludes population values, not observed ones);
- "true effects up to X remain possible (error-free measures)";
- "fits a modest effect (about 0.3) as well as none";
- "possible small speed link, not production";
- "descriptive";
- "beyond the 40 time" (finishing speed) or "beyond the scouting baseline" (tests);
- "Repeat on the metric and on its residual".

**Never say:**
- "useless", "tells you nothing", "clean null" without a bound;
- "undervalued", "steal", "market inefficiency";
- "confirmed", "replicated" or "validated" for the WR signal (the re-test reused the same receivers); "three tests agree";
- "tiebreaker" or any board-move language for finishing speed (Gate 2: "no hero survived"; E007: "park ... not a claim");
- "predicts production";
- "hidden" speed, or "beyond the stopwatch" for finishing speed alone (the official 20-yard split, which clubs receive and our data lack, may carry it);
- "large effects are ruled out" without "as measured";
- "drills run below full speed" (untested until E010);
- the 172-pairs 59% without its 95% CI (43–73%);
- "B2" in the text (use "scouting baseline").

**Label every test:**
- **primary:** the 14 round-1 primaries; E007 P1–P3. The E006-P1 primary is labeled "preregistered after the sanity link was seen; same receivers".
- **prespecified sanity:** E001, labeled "where the signal was first seen; direction set after round one".
- **robustness:** without 2025, leave-three-out, seeds, feature variants.
- **exploratory:** E007's deep share, aDOT, targets per route, DB top speed, snaps.
- **post hoc:** reliability-corrected and joint bounds; the AS_R = 1000 estimator (+0.51); the usage control (+0.21); the timing check; the 172 pairs and their CI; the leave-three-out power simulation; the review diagnostics in 3.4; the leaderboard uncertainty; the in-season forecast (Figure 5).

**Draft slot.** Mention that it reflects medical and off-field factors whenever late or undrafted players appear.

## 5. Mock-judging self-check (AGENTS.md checklist), v1

- **Baselines beaten?** Every test is against the scouting baseline (official Combine results + draft slot + position + year), stricter than base rates or a logistic regression. Out of sample (v1, DS judge): the re-test's CV R² rises from 0.18 (baseline) to 0.23, excess +0.045 ± 0.029 SE. Only the WR candidate beats the baseline, fragilely; nulls are bounds against it.
- **Uncertainty reported?** 95% CIs, one-sided permutation p, MDE, as-measured and error-free bounds, BH q, by-class estimates, leave-three-out, CV excess ± SE, and (v1) a receiver-bootstrap CI for the 172-pairs share.
- **Overclaiming?** Guardrails above. v1 removed: "WR tiebreaker", "hidden", "the stopwatch can't see", "large gains are ruled out", "Not unreliability", "drills run below full speed", "observed partial r ... excluded", "two reviewers ... from raw data", "every tracked Combine signature", "10 Hz tracking of every drill".
- **Usable week to week? (v1, football judge's rubric gap)** Clubs item 3 and Figure 5: one game of a receiver's tracking ≈ his best Combine measure as a forecast of later game top speed (0.56 vs 0.57); two or more games, probably better, about 0.77 (post hoc, n = 42; the gap's 95% CI excludes zero only at games 5 and 8).
- **≤ 2,000 words and < 10 figures/tables?** 1,831 now plus at most about 125–140 for round 3; 6 items plus at most 2 reserved.
- **Combine tracking used explicitly?** Yes, throughout: every feature is a tracked-drill kinematic; Figure 1 animates the 10 Hz 40; Figure 5 compares tracked Combine measures with in-season tracking.

**Likely judge criticisms, and where the draft answers them:**
1. "Mostly nulls." The audit framing, bounds, Table 1 verdicts, the week-to-week deliverable and two recommendation lists.
2. "No outcome link." Stated up front (Section 1) and bounded (3.3).
3. "Small n." MDE (3.2) and Limitations.
4. "One finding tested three times." Stated explicitly (3.3).
5. "Is the game target the same construct?" 3.4 (proxies, role, scheme).
6. "Clubs already have the 20-yard split." Hook, 3.3 definition and Clubs item 1 say so (v1).
7. "Couldn't the tracked residual be a tracking artifact?" 3.3 Timing bullet (v1).

**Rubric gaps still open:**
- **Data Viz:** no final figure exists yet (writeup/figures/ is empty). Build Figure 4 and Figure 5 first (JSON-only or light data), then Figure 3's redraw, then Figure 1's animation.
- **Football:** a short scout-voiced example in Section 4, if words allow after round 3.
- **Novelty vs the 20-yard split:** only external data (public official 20-yard splits for the 91 WRs) would settle whether finishing speed is beyond what clubs already see (section 7, decision 6).

## 6. Numbers not taken from a committed results JSON (flagged in their comments)

**Post hoc writeup analyses (v1; scratch scripts, must be promoted to the notebook or a results JSON before submission):**
- `ph_pairs.py` → `ph_pairs.json`: the 172-pairs share 0.593 reproduced exactly from `E007_leaderboard.json` (E007's tie rule: a game tie counts as "no"); receiver-cluster bootstrap 95% CI [0.428, 0.735] (10,000 resamples); one-sided permutation p 0.062; the hook pair is the largest-gap pair (rank 1 of 91) with a typical game gap (57th percentile). Used only in the Figure 1 footer and comments.
- `ph_rev3.py` → `ph_rev3.json`: rev-3 re-scoring of E006-P1 (seeds 0–2), E001-WR, and the post hoc AS_R = 1000 target. Supports "passes go criteria 1–6 under our current rules".
- `ph_inseason.py` → `ph_inseason.json`: the in-season forecast (Clubs item 3, Figure 5). Definitions in its docstring; validated against all 61 E007 game top speeds (exact).
- All three live in `/private/tmp/claude-501/-Users-jasonli-Documents-GitHub-NFL-Big-Data-Bowl-2027/ca6dc0b9-7e4f-45b4-b4e4-1468819c8969/scratchpad/round3/writeup_v1/`.

**Other non-JSON facts:**
- **510 players:** competition data (`players.csv`; AGENTS.md data table, verified 2026-10-14). Regenerate in the notebook.
- **"Most position drills have one rep per player":** PLAN.md section 2.
- **Preregistration and the review process:** `experiments/log.md` and `experiments/results/E00x_prereg.json`. Process facts.
- **Orientation absent in Combine tracking:** the data schema (AGENTS.md data table).
- **That the OL wave-drill session effect is tempo:** the E004 analysis in `experiments/log.md`. v1 no longer asserts tempo in the text or the Figure 2 caption; the 2.9 → 2.2 yd/s drift is not quoted.
- **0.5 reliability bar:** PLAN.md section 4, criterion 5 (a protocol constant).
- **S0 definition** ("where his top in-game accelerations extrapolate to zero"): `src/bdb27/speed.py` module docstring.

**Discrepancy found (v0, still open):** `E008.json links[E002-P1].caveats` and `experiments/log.md` say the game OUT/IN retention vs turn-radius correlation is "r = 0.89"; the computed value is 0.8848 (`E002.json diagnostics.game["g_turn_retention~g_turn_turn_radius"].r`), which rounds to 0.88. The draft quotes 0.88.

**Correction made in v1:** v0's Clubs item 3 quoted "receiver top speed (0.92)" from `target_reliability_by_position.E006-P1` (0.919), which is the reliability of the extrapolated S0 capacity target, not of observed top speed. v1 quotes E001-WR's 0.913 (q95 of play max) as 0.91.

## 7. Open decisions for the user

1. **Framework name:** "the Residual Combine" (checks "Repeat" and "Beyond"; the candidate is now "finishing speed"). Keep, or pick an alternative?
2. **Figure 1 names.** The faster finisher went undrafted; draft status reflects non-football factors. Recommendation: "Receiver A / B", or real names with no draft information.
3. **Table 1 as a graphic?** Recommended by the DS judge for Data Viz; frees about 140 words. Text inside images is a grey area for the word limit.
4. **Figure 5: in-season forecast (v1) vs the finishing-speed leaderboard (v0).** v1 swapped them: the week-to-week figure answers a rubric question, while the leaderboard ranks a metric with no production link (reliability 0.72). The leaderboard now lives in the notebook with a one-sentence pointer. Revert if you want a leaderboard in the writeup (it could also take reserved slot R1 or R2 if unused).
5. **Title:** keep the question form, or use a shorter one such as "The Residual Combine"?
6. **External 20-yard splits (football judge's "best fix").** Pulling the public official 20-yard splits for the 91 Combine WRs and adding them to the baseline would test whether finishing speed is beyond what clubs already see. Not done: it needs data outside the competition dataset, and the competition's external-data rules have not been checked. If allowed, run it as a labeled post hoc test.

## 8. Pre-submission checklist (writeup-specific)

- Replace every `[[TBD-…]]`, and strip every `<!-- … -->` comment, before pasting into Kaggle.
- Promote the three `ph_*` analyses (section 6) into the public notebook, and ideally into a committed results JSON, so every quoted number is traceable to the repo.
- Regenerate all figures from the public notebook (figure_plan.md). Re-run the word count with captions and the table included.
- Re-verify every quoted number. The scratch script `…/scratchpad/round3/writeup_v1/check_numbers_v1.py` (temporary, not in the repo) checks every number in v1 against the JSONs and the `ph_*` outputs at the quoted rounding, and asserts each quoted string appears in the text. Promote an equivalent check into the notebook.
- **Notebook wrapper (optional).** v1 deleted v0's "our notebook runs both checks". If the notebook ships one call that returns Repeat (metric and residual reliability) and Beyond (partial r, CI, bounds) for a new metric, wrapping `bdb27.reliability` and `bdb27.linkage.evaluate`, the clause can come back in Clubs item 2 (about 5 words).
- Re-run the mock judging (AGENTS.md) on the filled-in draft.

## 9. v1 change log: round-3 mock judging (2 judges, both "weakened")

Judges: Football + Writeup (5/10, 6/10) and Data Science + Data Viz (7/10, 5/10). Every number in v0 was upheld; the issues were framing, consistency and missing analyses. Each issue was verified against the JSONs before acting.

**Accepted and fixed in the text:**
- **WR candidate oversold** (both): "one candidate survives", the Table 1 verdict "WR tiebreaker" and the Clubs "tiebreaker" advice are gone. Now: "fits a modest link ... as well as none, and no link to production"; verdict "possible small speed link, not production". The untested "official 20-yard split" substitute is gone from Clubs item 1.
- **Novelty vs the 20-yard split** (football): the hook now says tracking sees it "and so may the official 20-yard split, which clubs receive and our data lack"; renamed hidden → finishing speed; 3.3 says clubs can approximate it.
- **Repeat on the residual** (both): Table 1 Repeat is metric / residual; 3.1 adds the residual bullet (DL 40 top speed 0.29 fails); Section 1, Section 2 and Clubs item 2 define Repeat on both.
- **Bounds** (football): Table 1 shows as-measured and error-free bounds; "partial correlations above X are excluded"; "large gains are ruled out" replaced by "above +0.22 for six links".
- **Usage confound and 2024 concentration** (football): the 3.3 usage bullet (+0.21 [−0.10, +0.49], post hoc); the three supportive receivers are 2024 draftees; 2023 shows nothing. Role and scheme are folded into 3.4 reason 1 rather than added as a third reason.
- **Hook pair and 59%** (both): the hook says the pair is the largest gap among 172; the 59% now carries a receiver-bootstrap CI (43–73%) and the mean gap (0.2 mph), and moved to the Figure 1 footer (cut from the text for words). Optional "yet he gained fewer yards per route" added.
- **Week to week** (football): a new post hoc in-season forecast analysis instead of a Spearman–Brown projection (it answers "when do the games overtake the Combine?" directly); Clubs item 3 and the new Figure 5.
- **Repeat overreach and the 3.1/3.4 tension** (both): "drill-specific ... mostly measured once"; 16 of 22 are reaction or opposite-side; 1 of 6 in the clean comparison; "the drill signatures carried into games were the more reliable ones (0.44–0.73)".
- **Jargon** (football): "scouting baseline" replaces B2 and stands for "the stopwatch"; S0 defined in one clause; "three overlapping versions"; "re-test by class"; the alignment-repeatability number replaced by "about three quarters".
- **Reproducibility and process** (both): the preregistered primary stays at its as-computed value (+0.48, AS_R = 20; preregistered definitions may not change), with the less noisy estimator quoted as post hoc (+0.51); "two internal adversarial reviews reproduced every number"; exposure check reported (≤ 0.32 / ≤ 0.40).
- **CV baseline comparison** (DS): re-test R² 0.18 → 0.23, excess +0.045 ± 0.029 SE.
- **Tracking-artifact alternative** (DS): added to the 3.3 Timing bullet.
- **"Not unreliability"** (both): "Unreliability alone doesn't explain it: 13 of 16 ... (12 on the residual)".
- **Evidence fixes** (both): "drills run below full speed" removed (left to E010); r −0.14 replaced by the like-for-like 0.01 (drill) vs 0.47 (game); the cushion claim now cites the 73% drop in between-receiver variance.
- **Labels and provenance** (both): sanity and primary labels; "one only in its hypothesis text"; "re-estimated under a protocol tightened after round one"; DB "retreat speed" (dropped from Clubs item 3 for words); the WR Table 1 cell gives 95% lower limits; the mixed q family is noted in a comment.
- **Smaller overclaims** (football): "every tracked Combine signature" → "18 Combine-to-game links"; "10 Hz tracking of every drill" → "(coverage varies by drill and player)"; "Use the tracked 40" → "Standardize on the tracked 40 ... adds at most a small receiver top-speed signal"; "Survivorship" → "Selection ... selecting on playing time usually attenuates links"; Clubs item 2 reworded, with the notebook clause removed.
- **Clarity** (DS): the ambiguous "three drills" sentence is rewritten; the hook's selection rule is stated in the text.

**Accepted and fixed in figure_plan.md:** the Figure 3 hierarchy (bold primary CI, thin post hoc outline); Figure 1's footer CI; Figure 2's residual arrows; Figure 4's usage rows and CV strip; Table 1 as a graphic (recommended, user's call).

**Rejected or deferred:**
- **Pull external 20-yard splits** (football's "best fix"): deferred to the user (section 7, decision 6). It needs data outside the competition set and an external-data rules check.
- **Make the AS_R = 1000 estimator the headline** (DS): partly rejected. The preregistered primary keeps its as-computed value because preregistered definitions may not change. The +0.51 is quoted as post hoc, both are re-scored under rev 3, and the notebook must reproduce both (the probe-E path for AS_R = 20; the shared module for 1,000).
- **At most about 2 numbers per sentence in Sections 1 and 4** (football): partly addressed. The hook still carries 6 numbers; they are the hook's content.

**Found while revising (not raised by the judges):** v0's "receiver top speed (0.92)" used the S0 target's reliability (fixed to 0.91, section 6).
