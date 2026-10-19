# Writeup outline: v2.5 (Combine Game Equivalents)

Companion to `writeup/draft.md` (the text), `writeup/figure_plan.md` (figures and tables), `writeup/check_numbers.py` (the number checker) and `writeup/make_figures.py` (the figures).

- **v2.5 (2026-10-19)** applies two mock judges' reviews of v2.4 (Football/Writeup 6.5/6.0; Data Science/Data Viz 7.5/7). Section 13 lists every judge item as applied, rejected or deferred. Main changes: the Roles note is rewritten as like-for-like worth with intervals and selection, and the preregistered H1 interval replaces the floored "beyond" range and the detectable-effect clause; the DL wording is descriptive ("tracks body type and roster designation"); Section 3.2 states the estimand (learned from 62–85 players), the worth without the draft slot (it adds at most 0.1) and what a perfectly learned signal could be worth (4–15 games), and calls the outcome traits not priceable; a scope sentence joins Section 2; Section 3.3's heading follows H3 ("made estimates slightly worse"); the two non-null transfer links are both named as fragile WR speed links; Clubs items 1 and 3 are rewritten; "repeatable within a session" replaces "stable traits" for same-day retests. The figure builder's renders of 10:54–10:56 already carry most judge figure items (figure_plan.md).
- **v2.4 (2026-10-19)** integrates E013 (the role-aware re-fit) and finishes the names pass. Section 12 lists every change: a short role-aware robustness note at the end of Section 3.4 (bounds; preregistered and post hoc labelled), the DL wording "body type and roster designation (OLB vs DE), not edge alignment", the correction of v2.3's "five traits reach 0.4–1.7 games" (two traits: WR top speed 0.63, DL rush distance 1.68), neutral wording for every mention of a named player in the text, comments and planning files, an extended named-player guard in `check_numbers.py`, and cuts that keep the budget.
- **v2.3 (2026-10-18)** uses real player names (user decision; section 7, item 2): Isaiah Bond (nfl_id 59278) in the text and Figure 1, Kyle Williams (nfl_id 58274) in Figure 1 only, with the named-player guardrail of section 4. `check_numbers.py` checks the names and scans the sentences about them. Sections 10–11 keep their v2.1/v2.2 label "Receiver A" (= Isaiah Bond).
- **v2.2 (2026-10-15)** applies the final checker's five findings: word budget, hook accuracy, selection pools, drill-reliability framing, captions. Section 11 lists each.
- **v2.1 (2026-10-15)** applies two mock judges' reviews to v2: Football/Writeup (Football 6, Writeup 6.5) and Data Science/Data Viz (DS 7, Viz 6). Section 10 lists every judge item as applied, rejected or deferred.
- **v2** rebuilt the writeup around E012 (Combine Game Equivalents), as decided at Gate 2b (`experiments/log.md`).
- **v1** (the "Residual Combine" audit framing) is superseded.

## 1. Framing and names

**Question.** Week to week, how much should a club trust a rookie's own games over his pre-draft prior, and what is the Combine (now tracked at 10 Hz) worth once games arrive?

**One frame, one metric.** The frame is **the Combine Exchange Rate**: priors priced in the currency clubs think in, games. The metric, **Combine Game Equivalents (CGE)**, is named once (Section 2). After that the text says "worth N games".

| Term | Meaning |
|---|---|
| **Qualifying game** | A game with enough plays to measure the trait (WR top speed: ≥ 10 tracked plays). Defined in plain words in Section 2. |
| **λ** | Game-to-game noise / between-player spread: the qualifying games after which his own average is half signal. The position average is worth λ games. |
| **CGE** | The extra games of his own tracking that a Combine-based forecast is worth, out of sample. |
| **The trust rule** (the one equation) | w = k / (k + λ + CGE). The prior is worth λ + CGE games: its effective sample size (credibility weighting, as in baseball's "stabilization points"). Its standard error after k games is σ / √(k + λ + CGE). |
| **Trust table** (Table 1) | Per trait: prior worth (λ + CGE), when his games outweigh the prior (95% CI), about how many rookie appearances that is, and the weights after 1 / 4 / 8 games. |
| **Priced / not priced** | Priced: reliability and split-half ≥ 0.5 (9 traits). Not priced: T4, T8, T10 (rule PH7, post hoc). The figures use the same words. |
| **Isaiah Bond** | The hook: E012's preregistered WR case study (largest gap between prior and season among the 11 of 15 test receivers with 10+ qualifying games). nfl_id 59278. Full name at first mention (text and Figure 1 caption), "Bond" after. |
| **Kyle Williams** | Figure 1's same-40 comparison receiver (4.40; E007 `concept_pair`). nfl_id 58274. Named in the figure only. |
| **Alignment / role (E013)** | His share of slot / edge / tackle / corner alignments in the trait's qualifying games (`E013_prereg.json` `role_definition`). The text says "alignment" and names the share per position: receivers' slot share, DL edge share, DBs' corner share (v2.5; no OL trait is quoted). Used in the Section 3.4 note and Clubs item 1. |
| **Roster designation** | `players.csv` `nfl_position` (the four-level DE / DT / NT / OLB label for DL; the 2.4 games come from all four levels, not from OLB vs DE); possibly the latest roster label, so it may postdate the draft, and it may carry scheme and stance. The text says "designation", never "role", for this (Sections 3.2 and 6), with "may postdate the draft" next to every value (v2.5). |
| **Typical game top speed / top-speed ceiling** | v2.5 (judge 1): "typical game top speed" = the per-game q90 (E012 T1, the hook) or q95 (E001) play peak; "game top-speed ceiling" = S0, extrapolated from in-game accelerations (E006-P1, Section 5; Figure 4 uses the same word). |

**Information sets, as named in the text:**
- **Position average** = C0.
- **"plus stopwatch and measurements"** = C1.
- **Combine + draft forecast** = C2 = B2. Never "Combine forecast" alone, because it includes the draft slot (judge 1). v2.5: wherever C2 is meant the text says "Combine + draft forecast" (Section 3.2's "the Combine + draft forecast −0.25 beyond it"; Clubs item 1).
- **"stopwatch and measurements alone" / "without the draft slot"** = C1, quoted only pooled by trait (v2.5, Section 3.2: the draft slot adds at most 0.1 game), never for a named player (section 4).
- **"+ Combine tracking"** = C3 (tracked-40 profile + the matched drill signature).
- The text never says C0–C3, B1 or B2.

## 2. Headline claims (each sourced in the draft's comments; all checked by `check_numbers.py`)

0. **Scope (Section 2, v2.5).** The exchange rate prices forecasts of a playing rookie's trait level, not who plays or is drafted; worth scales with λ, so priors matter least for fast-stabilizing traits.
1. **Own games take over fast.**
   - For the 9 priced traits, a rookie's qualifying games outweigh the Combine + draft forecast after 2.0 (WR top speed) to 6.8 games; 95% CIs are in Table 1 (preregistered bootstrap, CGE part biased low: footnote, v2.5).
   - The position average alone is worth 1.5–6.8 games, and the Combine and draft add 0–1.2. That is the exchange rate in one line.
   - At 2025 rookies' qualifying rates this is 3–12 appearances.
2. **Outcome traits are slow.** Separation, pressure and pressure allowed over expected are half signal only after 10.8–22.8 qualifying games (about 14–39 appearances; per trait in Table 1). v2.5 quotes λ at 1 dp, as Table 1 does.
3. **On priced traits, the Combine is worth a fraction of a game to about one.**
   - Learned from 62–85 players per trait, the Combine + draft forecast is worth 0 to 1.2 games; without the draft slot nearly the same (the slot adds at most 0.1; v2.5, pooled C1 values).
   - The bootstrap intervals all include 0 but are biased low, covering about 80% in simulation.
   - Calibrated 90% bounds (two one-sided 95% bounds; post hoc): WR top speed 0.08–1.6; DL rush distance 0.10–3.9; the rest below 1.7–3.6.
   - The estimate may be conservative: against draft-class averages under the same CV (a ridge on the draft-year dummies alone, E013 PH8), WR top speed reaches 0.63 games and DL rush distance 1.68 (intervals include 0), the rest under 0.4; a perfectly learned Combine signal could be worth up to 4–15 games (population-signal bounds; both post hoc). **v2.4 correction:** v2.3 said "five traits reach 0.4–1.7" from E012's `cge_C2_vs_C0_same_cv`, whose reference mixed ridge shrinkage of the class means into the Combine value.
   - The DL value tracks body type and roster designation (descriptive): designation alone (which may postdate the draft) is worth 2.4 games, the Combine + draft forecast −0.25 beyond it. v2.5 dropped "(OLB vs DE), not edge alignment" (both judges: the DL alignment vocabulary is two-level, T6 is already alignment-centred, OLB vs DE may carry scheme and stance, and 0.71 of the forecast's 1.23 games is shared with edge share).
   - The WR value is fragile (0.14 in the 2023–24 fit; 0 without one class). It is partly usage (r 0.61 with deep share), yet the forecast adds 1.0 game beyond deep share.
   - **Not priceable (v2.5):** the three outcome traits cannot be priced; simulation cannot bound their worth above. Before the reliability correction the forecast fit DL pressure (R² 0.21) about as well as WR top speed (0.22), partly via playing time. No CGE value is quoted for them (E012 notes: do not interpret).
4. **Combine tracking made estimates slightly worse** (v2.5 heading, judge 1).
   - H3: the estimate after one game is 2.1% worse (95% CI 0.4–3.9%; 12 traits pooled), as modeled (five or six tracked features, 62–85 players).
   - For Bond it nudged the forecast 0.3 mph toward his season (post hoc export, `e012_case_priors.json`); across all 15 test receivers it left error after one game within noise (0.96, 0.80–1.09).
   - ΔCGE (−0.24 to +0.18) is figure-only since v2.5 (judge 2: several bootstrap intervals are degenerate or exclude their own estimate).
5. **Shrinkage does the work, and the rule is calibrated** (pseudo-prospective).
   - The rule cuts error 72% after one game (68% on priced traits) and 52% after two.
   - The position-average blend gets 99% of that gain.
   - The Combine + draft forecast adds 5% before game 1 (0.3–9.6%), 3.5% on priced traits and 2.5% after one game (both intervals include 0; v2.5). All post hoc; none survives multiplicity.
   - The 80% intervals held 77–81%.
6. **Why the rate is low** (Section 4).
   - 16/18 links null, beyond the stopwatch and beyond stopwatch + draft slot; the other two are both WR speed links and both fragile (E006-P1 in Section 5; E001-WR, the sanity link, fails BH and leave-3-out; Figure 4's ‡ row). v2.5 cut the draft-slot median (0.01) from the text; Figure 4's subtitle keeps it.
   - Unreliable measures do not explain the nulls: 13 of 16 pass the reliability gate. Power and imperfect targets limit the tests. Nulls are bounds (+0.09 to +0.38; six links could reach 0.5 if error-free). The detectable effect (0.31–0.38) is on Figure 4.
   - **Separate finding (Section 5), not a reason for the nulls:** which Combine measurements are repeatable: 28/28 within a session, 2/22 (16 of the 22 reaction or opposite-side reps), 10/68, and 55–63% session variance in the OL wave drill. Drill intensity (E010) is figure-only since v2.5 (Figure 5).
7. **Roles (E013, Section 3.4 note; rewritten in v2.5).**
   - Post hoc, 2023–25 fit: alignment (receivers' slot share, DL edge share, DBs' corner share) explains 29% / 21% / 10% of between-player differences in WR hitch braking / DL rush distance / DB top speed (95% CIs 4–57%, 4–44%, −1% to 28%; the only three of nine passing BH).
   - Learned out of fold, alignment is worth 0.2–0.9 games for hitch braking (two estimators: the preregistered CV estimator 0.24; against the dummies-only ridge 0.90), where the Combine + draft forecast is worth 0 under both; and 0.8–1.1 for rush distance, mostly shared with the forecast (0.71 of 0.77). Both orders are in the src comment; no floored "beyond" difference is quoted.
   - Preregistered: alignment from a rookie's first games did not improve 2025 forecasts: H1 error +2.8% (95% CI −0.4% to +7.0%), which excludes even the 3% cut exactly known alignment would bring (PH7 best case); a post hoc precision correction agrees (1.1%, −1.0% to +3.6%; Figure 3 draws both).
   - Clubs item 1: intended deployment, known before game 1, is a prior E013 did not test (k = 0; the log's "Revisit if").
   - Table 1 stays position-level.
8. **Side notes.**
   - The WR tracked top-speed signal "remains suggestive" (+0.48; fragile; q 0.06), named as the stronger of Section 4's two WR links. The +0.42 beyond the stopwatch alone is figure-only (Figure 4).
   - E010 (drill intensity) is figure-only (Figure 5). E011 is league item 3.

**Honesty rules from Gate 2b** (all applied):
- Shrinkage does almost all the in-season work.
- The Combine's ~5% is post hoc and not multiplicity-robust.
- Combine tracking adds nothing (H3 slightly worse).
- Every bootstrap CGE interval includes 0. v2.1 keeps this sentence but no longer leads with it, and pairs it with the calibrated bounds.
- T4, T8 and T10 are not priced.
- The 2025 test is pseudo-prospective.
- Nulls are bounds (v2.4: including E013's preregistered role null; v2.5 states it with its interval, which excludes the best case, instead of the 80%-power detectable effect).
- (v2.5) A floored zero is a bound, not a null: no difference of clipped CV values is quoted as "adds X beyond" (LESSONS 2026-10-19).
- (v2.5) The outcome traits are "not priceable", not "small"; the headline is scoped to priced traits and to the Combine as learned from 62–85 players.

## 3. Structure and word budget

Counts come from `writeup/check_numbers.py`, after removing HTML comments and image links; captions and table cells count, and so does the `[[TBD…]]` placeholder (it becomes link text). Hard budgets (v2.2): **rendered words ≤ 1,900** (tokens with a letter or digit in the rendered text, table pipes as cell breaks) and **whitespace tokens ≤ 1,950**, counted both on the pasted markdown and on the rendered text; **≤ 8 figures + tables** (competition: ≤ 2,000 words and < 10 items). **v2.2: 1,888 rendered words; 1,881 / 1,914 whitespace tokens (markdown / rendered); 6 items. v2.3: 1,887; 1,880 / 1,913; 6 items. v2.4: 1,898; 1,892 / 1,925; 6 items** (the 71-word Roles note, the DL wording and the corrected sensitivity paid for by the cuts in section 12). **v2.5: 1,898; 1,893 / 1,927; 6 items** (the judges' additions paid for by the cuts in section 13). v2.1 had 1,953 words by the old count and 2,137 whitespace tokens, over 2,000 for a counter that counts table pipes and symbols.

| Section | Purpose | Key content | Items | Rendered words |
|---|---|---|---|---|
| Title | — | "The Combine Exchange Rate: how many NFL games is a Combine worth?" | — | 12 |
| 1. One rookie, fifteen games | Hook | 4.39; tracked finish 23.2 mph, rank 1/91; Combine + draft forecast 17.6 vs 17.5 (typical game top speed = q90 play peak); 18.3, 18.9 over 15; 2 SD, largest gap among the 11 of 15 test receivers with 10+ games, chosen by a preregistered rule; "no Combine failure" (worth 0.14 games; a 2-SD miss among 11 ≈ 40%); tracking nudged him 0.3 mph toward his season, but across all 15 left error within noise (0.96); official splits don't corroborate; the rule caught up by game 2 | Figure 1 | 223 (caption 15) |
| 2. The Combine Exchange Rate | Question + method | Clubs' question; qualifying game; λ; CGE; the one equation; worked example + standard error; credibility weighting; **scope sentence (v2.5)**; four priors; prereg + A1; post hoc pricing rule | — | 249 |
| 3.1 Own games take over (incl. the "3. Results" heading, 2) | Week-to-week deliverable | 2.0–6.8; λ 1.5–6.8 vs CGE 0–1.2; 3–12 appearances; outcome traits 10.8–22.8 games / 14–39 appearances | Table 1 | 280 (cells 143, footnote 34, caption 19) |
| 3.2 On priced traits, a fraction of a game to about one | CGE | learned from 62–85 players: 0 to 1.2; without the draft slot nearly the same (≤ 0.1); biased-low bootstrap; calibrated 90% bounds; may be conservative (0.63, 1.68, the rest under 0.4; a perfectly learned signal 4–15); the two largest are soft (DL tracks body type and roster designation, descriptive; WR usage and fragility); outcome traits not priceable | Figure 2 | 273 (caption 19) |
| 3.3 Tracking made estimates slightly worse | H3 | 2.1% after one game, 12 traits pooled; as modeled (five or six features, 62–85 players); Section 4's two WR links are single-feature, in-sample | — | 49 |
| 3.4 Shrinkage works; the rule is calibrated; roles | Validation + robustness | Pseudo-prospective; 72% (95% CI 60–80%; 68% priced) / 52%; 77–81% coverage; 99% shrinkage; 5% / 3.5% / 2.5% (both intervals include 0), post hoc, none survives multiplicity; **Roles** note (118 words; v2.4: 71): 29/21/10% with CIs and selection; worth 0.2–0.9 (hitch braking, forecast 0) and 0.8–1.1 (rush distance, mostly shared); H1 +2.8% (−0.4% to +7.0%) excludes the 3% best case; precision correction agrees (Figure 3) | Figure 3 | 288 (caption 17) |
| 4. Why so little | Evidence | Transfer map (16/18; the other two fragile WR speed links); 13 of 16 nulls pass the reliability gate; power and imperfect targets (bounds +0.09 to +0.38; six links ≥ 0.5 if error-free) | Figure 4 | 119 (caption 17) |
| 5. What tracking does show | Separate findings | Repeatability is a separate question from transfer (28/28 within a session, 2/22 with 16 non-identical, 10/68, 55–63%); the stronger WR link, suggestive and fragile | Figure 5 | 140 (caption 16) |
| 6. What to do with it | Recommendations | Clubs: trust table with your best prior (deployment before game 1 untested; alignment outvalues the forecast for hitch braking; DL designation prior outweighed after 6.0, not 4.8); shrink first (3.5×); don't weight Combine tracking as modeled. League: enough identical repeat reps (40 0.66–0.79 vs pass-rush 0.23–0.51; 2–6 reps); fixed tempo and frozen definitions; ball tracking with height (best 0.42; 6–46 drills) | — | 183 |
| 7. Limitations + Code | No overclaiming | n 62–85; inference; game units (low-play games 2.0–3.5× noisier for 9 of 12; roles pooled, Section 3.4); draft slot (the estimand bullet moved to Section 3.2 in v2.5) | — | 82 |
| **Total** | | | **6 of 8** | **1,898** |

**If words are needed** (v2.5 used most of the v2.4 list; none of these is a caveat), cut in this order:
1. Clubs item 1, ", not 4.8" (2) and Section 3.4's "(Figure 3)" in the Roles note (2).
2. Section 1, "in our 2025 test" (4): the pool is defined by "rookie receivers ... with ten or more qualifying games".
3. Section 5, "(extrapolated from in-game accelerations; …)" → keep the numbers, cut the definition (4): Figure 4 names the ceiling.

Never cut: "fit on 2023–24", "pseudo-prospective", "post hoc" labels, "none survives multiplicity control", "That describes our model (five or six tracked features, 62–85 players)", the calibrated bounds, the estimand ("Learned from 62–85 players"; "a perfectly learned Combine signal could be worth up to 4–15 games"), the official-splits clause, the scope sentence, "simulation cannot bound their worth above", Limitations, or the Roles note's labels ("(preregistered)", "post hoc"), its intervals and selection, and its H1 interval clause ("its 95% CI (−0.4% to +7.0%) excludes even the 3% cut exactly known alignment would bring").

## 4. Wording guardrails

**Say:**
- "Combine + draft forecast"
- "worth N games"
- "qualifying games"
- "priced / not priced"
- "Combine tracking made estimates slightly worse" (v2.5 heading; H3), "as we modeled it" / "our model (five or six tracked features, 62–85 players)"
- "the bootstrap intervals all include 0 but are biased low"
- "simulation-calibrated 90% bounds (post hoc)" (two one-sided 95% bounds)
- "Learned from 62–85 players per trait" and "a perfectly learned Combine signal could be worth up to 4–15 games" beside the Combine + draft forecast's worth (v2.5 estimand)
- "without the draft slot, nearly the same (it adds at most 0.1)" (pooled by trait only)
- "cannot be priced" / "simulation cannot bound their worth above" for the outcome traits
- "typical game top speed" (q90/q95 play peak) vs "game top-speed ceiling" (S0)
- "repeatable within a session" for the tracked 40 and other same-day retests
- "pseudo-prospective"
- "post hoc" on every PH number
- "error-free measures"
- "remains suggestive" and "fragile" (the WR signal)
- "nudged his forecast 0.3 mph toward his season average" (Bond, Combine tracking), always next to the 15-receiver test
- **Real names** (user decision, 2026-10-18): "Isaiah Bond" at first mention and in the Figure 1 caption, then "Bond"; "Kyle Williams" only in Figure 1. Each name must match the committed results for his nfl_id (`check_numbers.py`).
- "a rule fixed in advance chose him" (the selection rule)
- "A Combine + draft forecast … put his typical game top speed …" (the forecast named by its information set, never by a possessive list of its inputs; v2.4, "typical" v2.5)
- "13 of the 16 pass the reliability gate" (the nulls); "stability is a separate question from transfer"
- "tracks body type and roster designation (descriptive)" with "designation alone (which may postdate the draft) is worth 2.4 games" (the DL CGE; v2.5); "roster designation" / "designation", not "role", for `nfl_position`
- "alignment (receivers' slot share, DL edge share, DBs' corner share)" for E013's role, with the shares' intervals and selection; worth "learned out of fold" with both estimators ("0.2–0.9", "0.8–1.1"); "did not improve 2025 forecasts (preregistered)" next to its interval ("excludes even the 3% cut exactly known alignment would bring")
- "Intended deployment, known before game 1, is a prior we did not test" (Clubs item 1)
- "the estimate may be conservative" with the draft-class-average reference (0.63, 1.68; intervals include 0; the rest under 0.4) and the population-signal bounds (4–15)
- "partly route usage" (WR top speed)

**Never say:**
- "Combine forecast" without "+ draft"
- "the Combine said average" or "the Combine failed" about Bond
- that his forecast was "nearly flat" or "called every receiver average" (not true: `post_hoc_review.validation_2.per_trait.T1.c2_prior_spread_tau` 0.55, `c2_prior_flat` false)
- "the Combine-based rule cut error 72%" without crediting shrinkage
- "frozen before the 2025 season" or "prospective"
- "the Combine is useless" / "worth zero" / "at most about one"; the headline without "on priced traits" or without the estimand (v2.5)
- "tracking adds nothing" without "Combine"
- "survives", "validated", "confirmed" or "replicated" for the WR signal
- "undervalued", "steal", "market inefficiency"
- CGE values for T4, T8, T10
- "every interval includes 0" as a lead or without the calibrated bounds; for Combine tracking's ΔCGE at all (v2.5: degenerate and own-estimate-excluding bootstrap intervals)
- "two reps" (the evidence supports identical reps, 2–6 by Spearman–Brown)
- "36/36" for validation 1
- B1/B2/C0–C3 in the text
- **Anything about a named player's draft status, draft position or off-field matters**, in the text, captions, figures, alt text, notebook output, planning files or comments. No "undrafted", "drafted", "draft slot", "round", "pick…", "select…" next to a draft word, "signed", "free agent", "UDFA" or off-field word ("accus…", "assault", "medical", "arrest", "charged", …) in any sentence about a named player. `check_numbers.py` scans every text sentence that names him and every he/his/him sentence in a section that names him, and the same in the HTML comments (a comment sentence that names him or his nfl_id, or a he/his/him sentence in such a comment). "Draft" stays allowed only in the forecast's name, "Combine + draft forecast", never as his draft slot.
- Any forecast for a named player from another information set than the Combine + draft forecast and the + Combine tracking nudge, anywhere (text, figures, alt text, comments, planning files), and any statement of how one information set's forecast for him compares with another's beyond the quoted tracking nudge. `check_numbers.py` fails on "stopwatch-only", "prior_C1" or "C1" in any text section or comment that names a player. Rationale written generically: comparing information sets for one named player can reveal how the draft slot moved a forecast.
- "Nor would tracking have helped" or any claim that tracking did nothing for Bond (it moved him 0.3 mph the right way)
- a chance argument over a different pool from the selection rule (the rule chose among 11; the chance is over 11)
- that unreliable drills explain the transfer-map nulls (13 of 16 nulls pass the reliability gate)
- "five traits reach 0.4–1.7 games", or any "relative to the position average under the same CV" value from E012's `cge_C2_vs_C0_same_cv` (contaminated reference; E013 review)
- that role "does not matter" or "is worth nothing" (it explains 29% / 21% / 10% for three traits), or that a role-aware prior "improves" forecasts (preregistered H1–H4 not supported); the role null without its interval
- (v2.5) "adds −0.06 to +0.16 games beyond the Combine and draft" or any other difference of floored CV values as an "adds beyond" result (T3's "0 beyond" is the floored model with both; LESSONS 2026-10-19); report both orders and the shared part, or like-for-like worth
- (v2.5) "not edge alignment", "OLB vs DE" as the source of the 2.4 games (the label is four-level), "is body type and roster designation" (a decomposition the descriptive regressions cannot carry), or "a roster-designation prior beats the Combine" (a possibly post-draft label against a pre-draft forecast); the 2.4 without "may postdate the draft"
- that DL rush distance is "edge alignment" or "edge vs interior"
- (v2.5) any CGE value, calibrated lower bound or leave-one-class-out range for T4, T8, T10 (E012 notes: "do not interpret"); the outcome traits are "not priceable", never "small"
- (v2.5) "the Combine" for the Combine + draft forecast (C2) in a sentence that compares it; "stable traits" for same-day retests (say "repeatable within a session")
- (v2.5) "the test's detectable effect (4%)" for E013 H1: after the data the interval is the statement (it excludes the 3% best case)
- "role" for the roster designation (E013 uses role for alignment share)
- any name from E013's influence table (`t3_influence`), or any other player named in the committed results

## 5. Mock-judging self-check (AGENTS.md checklist), v2.5

- **Baselines beaten?**
  - The rule beats raw early games (H1/H2).
  - The honest baseline is the position-average rule: the Combine + draft forecast beats it by 5% before game 1 (post hoc).
  - Combine tracking does not beat the Combine + draft forecast (H3; and for WR top speed alone, 0.96, 0.80–1.09).
  - The transfer map tests each link against both baselines.
  - A role-aware prior (alignment from his first games) does not beat the position-average rule (E013 H1/H2, preregistered); its interval excludes even the best case the internal fit implies (v2.5).
- **Uncertainty reported?**
  - Half-weight CIs in Table 1.
  - Calibrated two-sided bounds next to the biased-low bootstrap.
  - The rule's 80% intervals hold 77–81%.
  - Leave-one-class-out and the 2023–24 fit for the WR CGE.
  - Bounds for the nulls (MDE on Figure 4); E013's role null as its interval; the alignment shares with their intervals (v2.5).
  - The estimand next to the headline: learned from 62–85 players; a perfectly learned signal up to 4–15 games (v2.5).
  - Intervals for 3.5% and 2.5% ("both intervals include 0"; v2.5); 90% labelled on the calibrated bounds (text and Figure 2).
  - p and q values on the figures.
- **Overclaiming?**
  - The hook now says "no Combine failure" and gives the chance of a 2-SD miss, over the same 11 receivers the rule chose among.
  - The hook states what Combine tracking did for Bond (0.3 mph toward his season) and that it did not help across all 15.
  - The drill-reliability paragraph no longer reads as the reason for the nulls.
  - "At most about one" is gone.
  - The H3 wording is exact.
  - The pricing rule is labelled post hoc, and the text says what it set aside.
  - v2.4: the "five traits reach 0.4–1.7" sensitivity is corrected to two traits (0.63, 1.68; intervals include 0); "conservative" became "may be conservative".
  - v2.4: the DL value is no longer called "role". v2.5: "tracks body type and roster designation (descriptive)"; "not edge alignment" and "OLB vs DE" are gone (judges).
  - v2.5: the floored "−0.06 to +0.16 beyond" and the 4% detectable-effect clause are gone from the Roles note; like-for-like worth and the H1 interval replace them.
  - v2.5: Section 3.3's heading follows H3 ("made estimates slightly worse"); the tracking advice is scoped "as we modeled it".
  - v2.5: the outcome traits are "not priceable" (no upper bound), not small; the headline says "on priced traits".
  - v2.5: both non-null transfer links are named (fragile WR speed links); "repeatable within a session" for same-day retests.
- **Usable week to week?**
  - Table 1 in club units (appearances, CIs, weights, outcome traits).
  - The worked example and standard error sit under the equation.
  - Prior-specific advice (Clubs item 1): deployment known before game 1 (untested), the DL designation prior outweighed after 6.0 games (v2.5).
  - The scope sentence (Section 2) says what the rate prices and what it does not (v2.5).
- **≤ 2,000 words and < 10 figures?** 1,898 rendered words; at most 1,927 whitespace tokens by any of the checker's counts; 6 items.
- **Combine tracking used explicitly?** Yes: C3 is priced in games; the transfer map and drill reliability are all Combine tracking; Figure 1 animates the tracked 40 (and, per figure_plan, one of his game plays).

## 6. Provenance

- **Every number in the text** traces to a committed results file; `check_numbers.py` verifies it.
  - It recomputes each number from its JSON key at the quoted rounding.
  - It asserts the quoted phrase is in the text.
  - It fails on any numeric token that no check covers.
  - It requires a `<!-- src` comment near every number-bearing paragraph. On a comment-stripped copy it skips this check and still checks every number.
  - Last run (v2.5, 2026-10-19): **430 checks, 0 failures** on the draft and on a comment-stripped copy; 1,898 rendered words, 1,893 / 1,927 whitespace tokens. (v2.4: 415 checks; v2.3: 386; v2.2: 374.)
  - Names (v2.3; extended in v2.4): each name in the checker's `NAMED` must equal every committed record for that nfl_id (`E007.json` `concept_pair` `display_name`, `E007_leaderboard.json` and `E012.json` `name`; `players.csv` `display_name` agrees, checked 2026-10-18). No other player in those files or in `E013.json` may be named in the text or the comments, no "Receiver A/B" may remain, first mentions are full names, near-miss spellings fail, and draft-status, off-field, draft-slot or stopwatch-only words fail in any sentence about a named player, in the text and (v2.4) in the HTML comments. v2.4 adds "pick…", "select…" next to a draft word (not "selection rule", not next to "draft forecast"), "draft slot", "accus…", "assault", "medical", and a section- and comment-level ban on "stopwatch-only", "prior_C1" and "C1" wherever a player is named.
  - Mutation test (v2.3): inserting a draft-status word in a sentence about a named player ("undrafted"; "signed as a free agent" in a pronoun-only sentence; "late-round"; "picked"; "draft pick" in the caption), an off-field sentence, a misspelled name (three variants), a bare surname before the full name (either player), a leftover "Receiver A" or another committed player's name each failed the run (1–3 failures each); "Kyle Williams" in full passed.
  - Mutation test (v2.4, 22 cases + the stripped copy, all as expected): in a hook sentence, "Picking", "Selected in the fifth round", "a seventh-round selection", "a late draft selection", "Selected 120th overall", "accused", "assault", "medical", "Undrafted", "free agent", "his draft slot" and "stopwatch-only" each failed the run, as did "late pick" in the Figure 1 caption; in comments that name him, "draft slot", "prior_C1" (in a sentence without his name), "stopwatch-only", "undrafted", "medical" and an E013 influence-table name each failed. Legitimate text passed: "selection rule" in a hook sentence, "selected" next to "Combine + draft forecast", and "Selection rule:" in a comment that names him. The v2.3 draft fails three v2.4 guard checks with 11 hits: in the text, the old hook forecast wording and Section 5's back-reference to the hook player (both in a sentence with "draft slot"); in the comments, the version note's "selection"/"picked" sentence, "Combine results + draft slot" beside his forecast, "his draft status, draft position or off-field matters", and "prior_C1 … (stopwatch and measurements only)". Altering any new number (0.63, 1.68, 29/21/10%, −0.06, +0.16, 4%, 3%, "under 0.4", 2.4, 6.0, "include 0", "did not improve") failed the run (2–5 failures each).
  - Mutation test (v2.5, 26 cases, all caught with 1–10 failures each): every new number (10.8–22.8; "at most 0.1"; 4–15; "90%"; 21%; the share CIs; 0.2–0.9; 0.8–1.1; +2.8%; −0.4%; 3%; "not 4.8"; 62–85; 2.1%; "five or six"; "16 of them"), every new qualitative claim ("both intervals include 0"; "biased low"; "the only three of nine"; "cannot bound their worth above"; "both WR speed links"; "is a prior we did not test"), the return of v2.4's DL overreach ("OLB vs DE, not edge alignment"), of the floored "−0.06 to +0.16" and of the ΔCGE sentence with "every interval including 0", and a draft-status phrase in the hook sentence.
  - Mutation test (v2.2): seven altered numbers (the 0.3 mph nudge, 40%, 13 of 16, 11 of 15, all 15, a Table 1 cell, the H1 CI) each failed the run (2–11 failures each); padding with 80 symbol tokens tripped the whitespace-token check and padding with 20 words tripped the word check.
- **Arithmetic on committed keys** (formula in each src comment; recomputed by the checker). The E012 owner should promote these into `E012.json` so the notebook prints them from keys:
  1. Rookie appearances = half-weight (or λ) / `qualifying_share_2025_rookies[T].share`.
  2. Priced-traits H1 cut, 68%: H1's own equal-weight pooling over the 9 priced traits of `secondary_per_trait[T]["1"]`. The same pooling over all 12 traits reproduces H1's pooled values exactly.
  3. Standard error after 4 games = √(σ² / (4 + λ + CGE)) = 0.33 mph, from `traits.T1.variance.sigma2` and `half_weight_game.C2`.
  4. Half-weight with the DL role prior = λ + `cge_role_reference` = 6.0.
  5. Reps needed for reliability 0.5 = (1 − icc) / icc from `drill_profile[*].icc_unit` (DL pass-rush): 2–6.
  6. Chance that a calibrated forecast misses at least one of 11 by 2 SD = 1 − 0.9545^11 = 0.40. The 11 are the receivers the selection rule chose among (`e012_case_priors.json pools.T1.n_with_10plus_games`); v2.1 used 15 (≈ 0.50), a different pool from the rule's.
  8. Bond's tracking nudge = (`prior_C3` − `prior_C2`) × 3600/1760 = 0.29 mph (`e012_case_priors.json cases[59278].tracking_shift_C3_minus_C2` 0.142 yd/s).
  9. H1's CI in percent = 1 − `ratio_ci95` = 60–80%.
  7. (Kept from v2) raw first game error / rule error = H1 `pooled_b / pooled_a` = 3.5.
  10. (v2.4, E013; v2.5) the gain exactly known alignment would bring = 1 − `post_hoc_review.prospective.power.best_case_from_internal_fit.H1.ratio_vs_observed_position_average_error` = 3%; H1 in percent = (`prospective.tests.primary.H1.ratio` − 1) = +2.8%, CI (`ratio_ci95` − 1) = −0.4% to +7.0%. The detectable effect (4%) is no longer quoted. The E013 owner may add these as percent keys.
  11. (v2.5) the draft slot's increment on priced traits = `cge_table[T].cge_C2.est` − `cge_C1.est`, at most 0.104 (T12) → "at most 0.1"; pooled by trait, never per player.
  12. (v2.5) the alignment worth ranges "0.2–0.9" and "0.8–1.1" = (`internal[T].attribution_C2.role_alone`, `summary.role_worth_vs_dummies_ridge_same_cv`) for T3 and T6; "mostly shared" = `attribution_C2.shared` 0.705 of `role_alone` 0.770 (T6).
- **Sources:** `E012.json`, `E012_prereg.json`, `E013.json` and `E013_prereg.json` (v2.4: the Section 3.2 correction and DL wording, the Section 3.4 Roles note), `E008.json`, `E009.json`, `E010.json`, `E011.json`, `E007_leaderboard.json`, `experiments/posthoc_writeup_v1/ph_pairs.json` (rank cross-check), `experiments/posthoc_writeup_v2/e012_case_priors.json` (post hoc export of the stored E012 forecasts: case-study priors C0–C3 and the test pools; the checker cross-checks its C0, C2 and season mean against `E012.json`).
- **Allowlisted non-result constants** (with reasons, in the checker): data scope (2023–24, 2023–25, 2025, 10 Hz); drill names ("tracked 40", "tracked-40 speed profile", "his 40 peak", "40 vs DL pass-rush drill"); "game 1" and Figure 1's "game-1" panel; the table's k values (1/4/8); CI labels ("(95% CI)", "(95% CIs)").
- **Prior art** (in the Section 2 src comment, for the notebook's references): Bühlmann credibility; sabermetric stabilization points; Morita, Thall & Müller (2008), "Determining the effective sample size of a parametric prior", *Biometrics*.
- **Not quoted, figure-only:** see `figure_plan.md`, "Not quoted in the text".

## 7. Open decisions for the user

1. **Framework name.** **Decided (user, 2026-10-18):** keep "The Combine Exchange Rate" (CGE named once).
2. **Names.** **Decided (user, 2026-10-18):** real names. Isaiah Bond (nfl_id 59278) is the hook, named in the text and Figure 1; Kyle Williams (nfl_id 58274) is Figure 1's same-40 comparison, named in the figure only. The guardrail stays: nothing about any named player's draft status, draft position or off-field matters, anywhere (section 4). Judge 1 had argued that anonymity bought little and cost hook power. Real names were also the precondition for R2.
3. **R2: 2025 rookie trust board** (judge 1 item 4 in v2.1; judge 1 item 5 in v2.5). **Still deferred.** It needs (a) a user decision to name 2025 rookies beyond the two approved names (the named-player guardrail and `check_numbers.py`'s `NAMED` would extend to each), (b) about 80–120 words, which v2.5 spent on the judges' accuracy items, and (c) a seventh item (7 of 8). The spec is in figure_plan.md.
4. **Figure 1 game-play panel.** Done by the figure builder; the v2.2 caption names it (v2.4 caption: "then his season").
5. **The hook's forecast with Combine tracking** (judge 2 item 3). **Resolved in v2.2** via the committed post hoc export `experiments/posthoc_writeup_v2/e012_case_priors.json`. The text quotes only the tracking nudge (+0.29 mph, C3 − C2); no other information set's forecast for a named player is quoted, drawn or described anywhere (section 4). The E012 owner may still promote the keys into `E012.json`.
6. **Orchestrator decisions for v2.2.** Names anonymized: superseded by the user's decision (item 2). The title stays (confirmed, item 1). No R2 trust board in v2.2.
7. **Role-aware analysis (E013).** **Done (user decision 2026-10-18 to run it; integrated in v2.4; note rewritten in v2.5).** Preregistered H1–H4 not supported; Table 1 stays position-level; the Section 3.4 Roles note carries like-for-like worth and the H1 interval. Open, for the E013 owner: a preregistered draft-day deployment (slot/outside, OLB/DE, CB/S) as a k = 0 prior (log, "Revisit if"); Clubs item 1 now names it as untested.
8. **Outcome-trait pricing (v2.5, judge 1 item 3).** The text now calls the outcome traits not priceable (no upper bound) and quotes no CGE for them, because E012's notes say "do not interpret". Quoting DL pressure's calibrated lower bound (0.28), its leave-one-class-out range (0.58–2.72) or an uncorrected-R² price (about 3 games, not a committed key) would need either a decision to override that note or a new preregistered, exposure-adjusted outcome-trait analysis (E012 owner).
9. **Budget cap (v2.5, judge 2).** The self-imposed cap is 1,900 rendered words (the competition allows 2,000). Judge 2 notes about 75 words of real headroom if the cap were raised; v2.5 kept 1,900 as instructed. Raising it would fund R2 or the T8 sentence.

## 8. Pre-submission checklist

- Replace `[[TBD-NOTEBOOK-LINK]]`. Strip every `<!-- … -->` before pasting into Kaggle, then run `check_numbers.py <stripped copy>`: it must report 0 failures, including both budgets (rendered words ≤ 1,900; whitespace tokens ≤ 1,950 on the markdown and the rendered text).
- Check in the Kaggle writeup preview that Table 1's compact pipes (`|a|b|`, no spaces; valid GitHub-flavoured markdown) render as a table. Spaced pipes (`| a | b |`) would raise the markdown whitespace-token count by about 140, to about 2,020 (measured on v2.2), over the 1,950 budget, so do not re-space them without first cutting about 80 words.
- Re-run `.venv/bin/python writeup/check_numbers.py` after every edit (exit 0).
- Re-run `writeup/make_figures.py` after any draft number changes. Its `same(...)` asserts hard-code the draft's strings, so update them together. (v2.4 changed no number that `make_figures.py` asserts. v2.5 changed none either: the values the text stopped quoting, such as tracking's −0.24 to +0.18 and the draft-slot median 0.01, stay correct on Figures 2 and 4; the 10:54–10:56 renders already use v2.5's words: "90% interval", "roster designation … may postdate the draft and reflect scheme and stance", "game top-speed ceiling", the alignment rows in Figure 3.)
- The notebook must:
  - reproduce `E012.json` (about 4.5 h full run), `E008.json`, `E009.json` and the `E013.json` keys the text quotes (v2.4; `--reuse` reads the stored draws). v2.5 quotes from `E013.json`: `post_hoc_review.summary.anova_share`, `anova_share_bh_q_lt_0.05`, `role_worth_vs_dummies_ridge_same_cv`, `e012_C2_vs_dummies_ridge_same_cv`; `post_hoc_review.internal[T3, T6, T12].anova.share_of_between_player_variance` (intervals), `internal[T3, T6].attribution_C2` (`role_alone`, `c_alone`, `shared`); `prospective.tests.primary.H1` (ratio, `ratio_ci95`); `post_hoc_review.prospective.power.best_case_from_internal_fit.H1`; `post_hoc_review.prospective.tests["calibrated:H1"]`. From `E012.json` (new in v2.5): `cge_table[*].cge_C1`, `cge_C2_population_signal_upper95`, `post_hoc_review.simulation.calibrated_bounds.traits[*].calibrated_one_sided_95` (the not-priced traits' null upper bounds);
  - print the derived numbers in section 6 from keys;
  - include trait definitions and position averages (the Code line promises both).
- Re-run the mock judging on v2.1 with the final figures.
- Shared-module items from E012's review (log, GATE 2b):
  - RidgeCV's LOO alpha search on duplicated bootstrap rows. v2.1 quotes no E007–E011 bootstrap interval; its E008–E010 intervals are Fisher CIs.
  - `nfl_position` meaning. E013 PH10 calls it "possibly the latest roster label"; v2.4 quotes it as the "roster designation", with "designations may postdate the draft" (Section 6) and as descriptive in the src comments (Section 3.2).

## 9. v2 change log (from v1, condensed)

- **Kept:** transfer map with bounds; drill reliability; the WR signal with fragility; E010/E011 one-liners; Combine-operations recommendations (now revised); guardrails.
- **Dropped:** the "Residual Combine" name and scorecard; the same-40 pair as the hook; finishing-speed detail; the 172-pairs statistic; v1's in-season WR forecast; the "standardize on the tracked 40" advice.
- **New:** CGE, the trust rule and table; the pseudo-prospective test with the shrinkage-vs-Combine split; E009's beyond-the-stopwatch column; `check_numbers.py`.

## 10. v2.1: judge items, applied / rejected / deferred

Judge 1 = Football/Writeup; Judge 2 = Data Science/Data Viz. "Figure builder" = `make_figures.py`. Several figure items were already done in the 19:39–19:40 renders; figure_plan.md lists each as done or to do.

### Judge 1, top improvements
1. **Role-aware priors; usage confounding.** **Partly applied.**
   - Clubs item 1 gives the DL roster-role prior: worth 2.4 games, outweighed after 6.0 (post hoc; roles may postdate the draft).
   - Section 3.2 states WR top speed's usage confound: r 0.61 with deep share; the forecast adds 1.0 game beyond it.
   - Limitations: position averages pool roles.
   - **Deferred:** re-fitting λ and Table 1 with role-specific averages. That needs an E012 re-run after verifying `nfl_position` (a Gate 2b open item), and the writer may not edit experiments. A slot/outside WR split has no committed key.
   - **v2.4: done as E013** (preregistered role-aware re-fit with alignment share). H1–H4 not supported; Table 1 stays position-level; the Section 3.4 Roles note gives the bounds, and the DL prior is now called the roster designation (section 12).
2. **Club decision units.** **Applied:**
   - (a) appearances column plus the 3–12 / 14–39 conversions (derived; owner should promote);
   - (b) λ beside CGE in Table 1 and in Section 3.1's exchange-rate sentence;
   - (d) the standard error σ/√(k + λ + CGE) with a worked value, plus 77–81% calibration.
   - **Deferred (c):** position averages in units. Only WR top speed (17.5 mph) has a meaningful raw-unit average in committed results; several traits are centred or "over expected". The Code line commits the notebook to definitions and position averages.
3. **Outcome-trait stabilization into Table 1 and the headline.** **Applied:** three Table 1 rows (λ with bootstrap CI, appearances, position-average weights); Section 3.1 paragraph; Clubs item 4. Seasons are given as appearances (14–39) rather than as fractions of a season.
4. **2025 rookie trust board (R2).** **Deferred:** it needs the names decision (section 7) and about 100 words of cuts. The spec is in figure_plan R2.
5. **Honest hook, positive punchline.** **Applied:**
   - "no Combine failure: in that fit the forecast was worth only 0.14 games";
   - chance of a 2-SD miss among 15 test receivers ≈ 50%;
   - official splits don't corroborate tracked finishing speed;
   - the q90 definition explains the gap to 23.2 mph;
   - ends on "What worked was the in-season rule … Table 1 gives that moment".
   - **Rejected in part:** "the forecast was nearly flat and called every receiver average". The committed data contradict it (`c2_prior_spread_tau` 0.55, `c2_prior_flat` false).
6. **Section 7 inconsistencies.** **Applied:**
   - weights by prior (C0 weights 0.40/0.73/0.84);
   - the league item now asks for identical reps, 2–6 by Spearman–Brown, with the like-for-like 40 vs pass-rush comparison;
   - "Don't weight Combine tracking".
7. **Prior art.** **Applied** (Section 2: credibility weighting, stabilization points, prior effective sample size; citations in the src comment).
8. **Readability.** **Applied:**
   - one frame name, CGE named once;
   - "Combine + draft forecast" throughout;
   - the worked example under the equation;
   - p, q and CI strings cut from Section 3.4 (one CI kept per headline effect; p/q on the figures);
   - "qualifying game" defined with an example;
   - the week-to-week question merged into Section 2.
9. **Drill-vs-game animation.** **Deferred to the figure builder:** figure_plan Figure 1 panel 2 (one typical game play on the field, selection rule fixed).

### Judge 1, overclaims or errors (11)
- Clubs item 3, "at most about one": **applied** ("0 to 1.2 (point estimates), with more than about 4 ruled out").
- Clubs item 1 vs Table 1 priors: **applied**.
- League item 1, "two reps": **applied**.
- H3 described as "forecasts": **applied** ("the estimate after one game … 12 traits pooled").
- Tracked-finish hint uncaveated: **applied**.
- "The Combine said average": **applied** in the text; the figure title change is in figure_plan (to do). The "flat" premise is rejected (above).
- "One signal survives": **applied** ("remains suggestive").
- DL prior should be role-specific: **applied** in Clubs item 1; the Table re-fit is **deferred** (v2.4: E013 ran it; Table 1 stays position-level).
- T9/T11 weights under a failed assumption: **applied** (no weights; † footnote).
- "Largest miss" by absolute gap: **applied** ("largest gap"; the src comment says absolute).
- Prior-art omission: **applied**.

### Judge 1, figure issues (8)
- Figure 1 dots arrive together: **done** (speed-vs-distance curves).
- Figure 1 y-label: **done**.
- Figure 1 hollow diamond: **done** (removed until `prior_C3` is committed).
- Figure 2 λ bars, ticks, hollow glyph, stray glyph: **done**.
- Figure 3 jargon, five series, p/q density: **done**.
- Figure 4 outline boxes: **done** (removed). The six "≥ 0.5 if error-free" links get a small glyph (to do).
- Figure 5 OL one-rep row and the DL pass-rush vs "two reps" conflict: **done** (row to notebook; recommendation changed).
- Table 1 zeros, not-identified λ, identical-looking weights: **applied** ("0: no out-of-sample gain (clipped)"; λ + CGE column makes 4.8 = 4.8 + 0 = 3.6 + 1.2 visible).

### Judge 2, top improvements
1. **Calibrated two-sided bounds.** **Applied** in the text (Section 3.2) and **done** in Figure 2.
2. **Conservative estimator; retitle.** **Applied:**
   - Section 3.2 title "a fraction of a game to about one";
   - the "estimate is conservative" sentence with the same-CV values 0.4–1.7 (**corrected in v2.4**: that reference was contaminated; "may be conservative", two traits, 0.63 and 1.68; section 12);
   - Clubs item 3 as a bound.
   - The simulation's p_zero (64.5% at population R² 0.2) is not quoted (words), and the CV estimand itself is 0 there.
3. **Hook attribution and statistics.** **Applied:** the T1-specific tracking test (0.96, 0.80–1.09) replaces the pooled H3 in the hook; the chance of a 2-SD miss is stated. The text uses 15, the committed count; judge 2's 11 (receivers with ≥ 10 games) has no committed key. (Superseded in v2.2: both counts are now committed and used; section 11.)
   - **Deferred:** decomposing the hook's forecast by information set (section 7, item 5; v2.2 quotes only the Combine-tracking nudge).
   - Leading with the WR case among 16 case studies is an editorial choice. It is the preregistered WR case study, and WR top speed is the trait the forecast prices best. Not changed.
4. **Uncertainty in Table 1.** **Applied** (half-weight 95% CI column).
   - Weights stay at 2 dp: they are a plug-in rule, and the CI column carries the uncertainty.
   - **Deferred:** R1 trust curves (largely duplicate Table 1; no words left).
5. **Pricing rule is post hoc; what it removed.** **Applied:** "(post hoc)" in Section 2; Section 3.2 gives DL pressure's raw CV R² (0.21 vs WR top speed 0.22), its +0.40 exposure correlation, and "where a good prior matters most".
   - **Rejected for the text:** T8's calibrated lower bound (0.28) and leave-one-class-out range (0.58–2.72). The trait's true-score correction is unstable (reliability 0.43); these stay in the notebook.
6. **The game unit.** **Applied:** Limitations now quantify the noise ratio, correcting the judge's count: **9** of 12 traits (not 10) are ≥ 2.0×; T10 is 1.60. T9/T11 lose their weights.
   - **Deferred:** per-play weighting and season-centring (need an E012 re-run; PH11's season-centred CGE is 0 for both traits anyway). "CGE/λ, the threshold-free share" is shown implicitly by Table 1's λ + CGE column; there is no separate number.
7. **Reconcile Section 5 with 3.3.** **Applied** (one scope sentence; heading "nothing measurable").
8. **Report strengths.** **Applied:** calibration 77–81%; priced-traits H1 cut 68%, computed with H1's own pooling, which the checker verifies reproduces H1 exactly.
9. **Encoding consistency.** **Done** by the figure builder (hollow = post hoc only; Figure 4 navy only). Codified in figure_plan's house style.
10. **Operations advice and prior work.** **Applied:** identical reps; 2–6 by Spearman–Brown from single-rep ICC 0.15–0.39; the 40 (0.66–0.79) vs pass-rush (0.23–0.51) comparison, "mostly not" because 1 of 6 reaches 0.51; prior-art sentence and citation.

### Judge 2, overclaims or errors (15)
- "Combine said average": the text says "Combine + draft forecast". The decomposition is **deferred**; the figure title is **to do**.
- Largest of 11; a 2-SD miss is common: **applied** (with the committed 15).
- Pooled H3 answering a WR question: **applied**.
- Leading with "every interval includes 0": **applied**.
- "At most about one" is not a bound: **applied**.
- "Worth zero to about one" header: **applied**.
- Pricing rule placed after "preregistered": **applied**.
- "Adds nothing" vs +0.48: **applied**.
- Half-weight uncertainty: **applied**.
- "Not a draft-slot artifact": **applied** (clause removed; "intervals allow 0.37").
- "Two standardized reps": **applied**.
- T9/T11 weights: **applied**.
- 72% includes the unpriced separation trait: **applied** (68% on priced traits).
- Tracking-artifact alternative: **applied**.
- "About 11 catching drills": **applied** (6–46).

### Judge 2, figure issues (16)
- Done by the figure builder (19:39–19:40 renders):
  - Figure 1 still, start-line overprint, x-label, removed diamond, the 2023–24 vs 2023–25 note;
  - Figure 2 calibrated bounds, λ and axis title, and the stray glyph;
  - Figure 3 three lines, single linear axis and the pooled-traits note;
  - Figure 4 outlines and legend;
  - Figure 5 dot independence (split by source) and the like-for-like slope panel;
  - cross-figure encoding.
- To do (figure_plan):
  - Figure 1 alt text, GIF timing and the game-play panel;
  - Figure 2 raw CV R² in the not-priced rows;
  - Figure 3 calibration panel and priced-traits H1 row;
  - Figure 4 legibility check at 750 px;
  - Figure 5 reliability SEs.
- Rejected: Figure 3's "irreducible target noise" reference. No committed quantity defines it.
- Deferred: R1 trust curves.

### Errors found while revising (all fixed before the final check)
- The judges' noise-ratio count (10 of 12) is 9 of 12 in `E012.json`.
- The judges' "nearly flat forecast" premise is contradicted by `c2_prior_spread_tau` 0.55.
- My first Table 1 draft rounded OL pass-set acceleration's lower CI 3.7499 to 3.8; the checker caught it (now 3.7).
- The DL pass-rush drill has 1 of 6 first-step metrics at 0.514, so the text says "mostly not" rather than "not".

## 11. v2.2: the final checker's findings, applied

1. **Word budget.** **Applied.**
   - v2.1 as committed counted 1,953 words by its own method (over its 1,950 limit) and 2,137 whitespace tokens: table pipes, "+", "/", "≥" and "×" are tokens to most word counters.
   - New hard checks in `check_numbers.py`: rendered words ≤ 1,900, and whitespace tokens ≤ 1,950 on both the pasted markdown and the rendered text. v2.2: 1,888 words; 1,881 / 1,914 tokens.
   - Table 1 separators compacted (`|1.5+0.5|`, `0.34/0.67/0.80`, `≥10 plays`).
   - Cuts, none a caveat: the trait name after "6.8 games"; the 0.67/0.33 restatement of w; "error ratio 0.28" (the CI is now in percent, as on Figure 3); Section 3.2's per-trait list (Table 1 has it); "These slow traits are where a good prior matters most"; the WR signal's +0.42 (Figure 4 shows it); E011's 54 receivers and 18 drills; "every logged test" in the Code line; shorter Section 2 phrasing and captions.
2. **Hook accuracy.** **Applied.** "Nor would tracking have helped" was wrong for Receiver A: Combine tracking moved his forecast +0.142 yd/s (0.29 mph) toward his season (`e012_case_priors.json`). The hook now says so, then that across all 15 test receivers tracking left error after one game within noise (0.96, 0.80–1.09), with the pooled H3 in Section 3.3. The checker verifies the nudge, its direction (C2 < C3 < season), and that C3 stays more than 1 mph below his season. (v2.4 removed every other information set's forecast for him from the checker and the comments: section 4.)
3. **Pools.** **Applied.** The hook selects among "the 11 of 15 rookie receivers in our 2025 test with ten or more qualifying games", and the chance argument uses the same 11: 1 − 0.9545^11 = 0.40 ("about 40% of the time"), replacing v2.1's 15-receiver 0.50. Both counts come from `e012_case_priors.json pools.T1`; the checker also confirms 15 = H1's and the T1 tracking test's n. I verified against the gitignored forecasts parquet that Receiver A has the largest absolute gap of the 11.
4. **Framing.** **Applied.** Section 4 now says unreliable measures do not explain the nulls: 13 of the 16 pass the reliability gate (E008 `summary.transfer.nulls_passing_c5`; the checker recomputes it from each link's `rel_x` and `rel_y`), and power and imperfect game targets limit the tests. The reliability paragraph moved to Section 5 under "Stability is a separate question from transfer", with drill intensity (stable, predicts nothing) as the example.
5. **Captions.** **Applied.** Figure 1 names the game-play panel and "his forecasts" (the figure builder added a post hoc + tracking marker during this pass; see figure_plan, Figure 1); Figure 2 says the simulation-calibrated bounds are bold (primary) and the bootstrap thin; Figure 3 names the 80% coverage panel; Figure 5 names the first-step and session panels. Figure 4's caption already matched.

**Errors found (for LESSONS.md; the writer role may not edit it):**
- A word budget enforced with one counting method can pass while another common method fails (v2.1: 1,953 by the checker, 2,137 by whitespace) → enforce every plausible count, on both the markdown and the rendered text.
- A selection rule and the chance argument for it used different pools (11 vs 15) → state the selection pool and compute any "how surprising is this" figure over that same pool.
- A hook sentence ("Nor would tracking have helped") generalized a pooled result to an individual whose own value went the other way → check every individual-level claim against that individual's stored values.

## 12. v2.4 (2026-10-19): E013 integrated, names pass finished

Sources: `experiments/results/E013.json` (`post_hoc_review.summary`, `post_hoc_review.internal[T]`, `prospective.tests.primary`, `post_hoc_review.prospective`), `E013_prereg.json`, and the log entries "E013 summary" and "E013 review response" (2026-10-19). Every new number is checked by `check_numbers.py` (section 6).

1. **Roles note (end of Section 3.4, 71 words).** Applied as the log's decision: a robustness note, not a section; Table 1 stays position-level.
   - Post hoc, 2023–25 fit: alignment (slot, edge, tackle, corner) explains 29% / 21% / 10% of between-player variance in WR hitch braking / DL rush distance / DB top speed (`anova_share`; the three with BH q < 0.05 over 9 traits), "yet, cross-validated, adds −0.06 to +0.16 games beyond the Combine and draft" (`role_beyond_combine_draft_range`; CV point estimates, the floored estimator, so the src comment calls it a bound on what these players teach, not a null; bootstrap upper ends ≤ 0.42).
   - Preregistered: alignment from a rookie's first games did not improve 2025 forecasts (H1 1.028 at k = 1, H2 1.002 at k = 2; not supported), "even after a post hoc precision correction" (calibrated H1 1.011).
   - Bound: "the test's detectable effect (4%) exceeds the 3% gain the fit implies" (PH7 power keys; arithmetic item 10 in section 6).
   - Not quoted: H3 (0.964, p 0.077) and H4, the CV role worth (0.77, 0.24), the influence table (it names two other rookies), the role-aware trust table. They stay in E013 and the notebook.
2. **DL wording (Sections 3.2 and 6).** "DL rush distance (1.2) is body type and roster designation (OLB vs DE), not edge alignment: designation alone is worth 2.4 games, the Combine −0.25 beyond it." Evidence in the src comment (E013 PH10, descriptive, in sample; the label may postdate the draft): among pure edge players OLBs out-rush DEs (Welch p 0.0003, all three classes); OLB +0.110 beyond edge share and weight (p 4e-5) while edge share is −0.028 (p 0.57); edge share adds 0.06 games beyond the Combine and draft and is 75% predictable from them. Clubs item 1 now says "a roster-designation prior … (post hoc; designations may postdate the draft)": "role" in the text now means alignment only.
3. **Correction (Section 3.2).** v2.3's "The estimate is conservative …: against the position average under the same cross-validation, five traits reach 0.4–1.7 games (post hoc)" used E012's `cge_C2_vs_C0_same_cv`, whose reference credited ridge shrinkage of the class means to the Combine (found by the E013 review; LESSONS 2026-10-19). v2.4: "The estimate may be conservative …: against draft-class averages under the same cross-validation, WR top speed reaches 0.63 games and DL rush distance 1.68 (intervals include 0), the rest under 0.4 (post hoc)" (`e012_C2_vs_dummies_ridge_same_cv`; intervals `internal[T].same_cv_vs_dummies_ridge.e012_C2`, [0, 1.65] and [0, 4.17]). "May be": both exceed the main estimates (0.47, 1.23), but DB top speed's 0.18 is below its 0.25. Sections 2, 3, 5 and 10 of this outline carry the correction; the checker fails if "five traits reach" returns.
4. **Names pass.** No draft-slot or stopwatch-only reference is tied to a named player anywhere in the writer's files.
   - Text: the hook's forecast is now named by its information set, "A Combine + draft forecast" (the old wording attached the forecast's draft input to the player); Section 5's back-reference to the hook player is cut (that sentence also names the stopwatch and draft slot as baselines).
   - Comments: the version note no longer describes the hook's wording in draft terms; the Section 1 comment's guardrail line is generic ("Named-player guardrail: header comment"); his forecast is described as "information set C2 as defined in the header comment"; `prior_C1` and its value are gone; "draft_year 2025" became "2025 Combine class".
   - `check_numbers.py`: the two checks that compared another information set for the hook player, and their comment, are removed; the "A draft year" label is "A Combine class 2025".
   - Planning files: outline sections 1, 3, 4, 6, 7, 10 and 11 and figure_plan's Figure 1 section, "not quoted" list and status line no longer name or compare any other information set's forecast for a named player; "picked"/"pick" about the hook player became "chosen"/"case study". The section 4 guardrail states the rule generically.
5. **Guard extension (`check_numbers.py`).** "pick…"; "select…" only next to a draft word ("draft…" other than "draft forecast", "round", "overall"); "draft slot"; "accus…", "assault", "medical"; the scan now also covers the HTML comments; "stopwatch-only", "prior_C1" and "C1" fail anywhere in a text section or comment that names a player; `E013.json` joins the name universe (its influence table names two other rookies). Mutation results in section 6.
6. **Cuts that paid for it** (rendered words; none is a caveat): Section 1 forecast name −5 and "in the unit clubs think in" −5; Figure 1 caption "then his season" −5; Section 2 "now" −1; Section 3.1 "priced" −1 and the pass-rusher example −9 (Table 1 has it); Section 5's back-reference to the hook player −4 and the drill-intensity definition −3; Clubs item 1 −5 (the 2.4 games are in Section 3.2) and item 4 −16 (Table 1's ‡ rows and Section 3.1 say it); League item 1's mirror-side sentence −8; League item 3 −13 (the 87% timing check is no longer quoted); Limitations' role examples −4 (now a pointer to Section 3.4). Additions: Roles note +71, Section 3.2 correction +13, DL wording +6. Net +11: 1,887 → 1,898 rendered words; 1,880 / 1,913 → 1,892 / 1,925 whitespace tokens.

**Errors found (for LESSONS.md; the writer role may not edit it):**
- The hook's forecast sentence attached the draft-slot input to a named player with a possessive, and the v2.3 checker allowed it because "draft slot" was whitelisted as a forecast input → name the forecast by its information set ("Combine + draft forecast") and scan for "draft slot" in every sentence about a named player.
- The guardrail's own rationale, repeated in three planning files and a comment, recorded the very comparison the guardrail hides (two information sets' forecasts for a named player, with values) → write guardrails generically; never record the hidden comparison, even to justify hiding it.
- The v2.3 guard scanned only the comment-stripped text, so comments could tie the draft slot or another information set's forecast to a named player → scan comments with the same rules, with nfl_ids counted as names.
- (this pass) My first v2.4 comment scan was sentence-level only: a mutation that put "prior_C1 …" in its own sentence, without his name or a pronoun, inside a comment that names him passed → ban the information-set tokens at the level of the whole section or comment that names a player, and mutation-test each guard with the forbidden token in a sentence of its own.

## 13. v2.5 (2026-10-19): two mock judges on v2.4, applied / rejected / deferred

Judge 1 = Football/Writeup (Football 6.5, Writeup 6.0); judge 2 = Data Science/Data Viz (DS 7.5, Viz 7). "Figure builder" = `make_figures.py`, whose renders of 2026-10-19 10:54–10:56 (made in parallel with this pass) already carry most figure items; I checked Figures 1–5 against each item. Budget: 1,898 → 1,898 rendered words (1,892 / 1,925 → 1,893 / 1,927 tokens); additions of about 200 words were paid for by the cuts listed at the end.

### Judge 1, top improvements
1. **Roles note: the club's question, both orders, floored zeros.** **Applied.** The "−0.06 to +0.16 beyond" range is gone (for WR hitch braking it was a difference of two clipped zeros). The note now gives like-for-like worth learned out of fold under both estimators: 0.2–0.9 games for hitch braking, where the Combine + draft forecast is worth 0 under both, and 0.8–1.1 for rush distance, mostly shared with the forecast (0.71 of 0.77; both orders and the shared part in the src comment). It then gives the preregistered H1 with its interval (judge 2's version of the power point). The judge's "about the same words" was not reachable with the intervals and selection judge 2 asked for: 118 words (v2.4: 71).
2. **Deployment before game 1 (k = 0).** **Applied** in Clubs item 1: "Intended deployment, known before game 1, is a prior we did not test; in our 2023–25 fit, alignment outvalues the Combine + draft forecast for WR hitch braking, and a DL roster-designation prior is outweighed after 6.0 games, not 4.8 (post hoc; designations may postdate the draft)." For DL rush distance alignment is worth less than the forecast (0.8–1.1 vs 1.2–1.7), so the DL half of the judge's sentence is carried by the designation prior, with its leakage caveat.
3. **A bounded value for DL pressure (T8).** **Partly applied.** Section 3.2 now says the outcome traits cannot be priced and that simulation cannot bound their worth above, and keeps the R² 0.21 and playing-time points: "not priceable", not "small" (also judge 2, overclaim 11). **Rejected for the text:** T8's calibrated lower bound (0.28), its leave-one-class-out range (0.58–2.72) and λR²/(1 − R²) on the uncorrected R² (about 3.0). E012's notes say "do not interpret" for the not-priced traits; the full-sample estimate (10.6) lies far outside the leave-one-class-out range, which is the instability the gate is for; the 3.0 is not a committed key and is partly playing time (+0.40) and the draft slot (uncorrected R² 0.136 without it vs 0.209). Open decision: section 7, item 8.
4. **Scope sentence (Section 2).** **Applied:** "It prices forecasts of a playing rookie's trait level, not who plays or is drafted; worth scales with λ, so priors matter least for fast-stabilizing traits."
5. **Player-level deliverable (R2).** **Deferred** (section 7, item 3): naming more rookies needs a user decision, and the 80–120 words went to items 1–4 and judge 2's accuracy items. Of the judge's suggested cuts, v2.5 used the Section 4 detectable-effect clause, the drill-intensity sentence and the repeated C0 weights; it kept Section 3.2's sensitivity sentence (the corrected E013 numbers) in shortened form.
6. **DL wording.** **Applied** (Sections 3.2 and 6): "DL rush distance (1.2) tracks body type and roster designation (descriptive): designation alone (which may postdate the draft) is worth 2.4 games, the Combine + draft forecast −0.25 beyond it." "Not edge alignment", "OLB vs DE" and "beats the Combine" are gone.
7. **Readability.** **Partly applied:** "Combine + draft forecast" wherever C2 is compared; "typical game top speed" (hook) vs "game top-speed ceiling" (Section 5, as on Figure 4); Clubs item 3 is a full sentence; the Section 3.2 heading is scoped ("On priced traits"); fewer numbers repeated from Table 1 and the figures (C0 weights, ΔCGE, MDE, the draft-slot median and the drill-intensity values cut). **Rejected:** a dagger for post hoc (Table 1 and Figure 4 already use †; "post hoc" in words is a never-cut label). Plain-English opening sentences beyond the headings: not done (no words; each heading states its finding).
8. **E013 in Figure 3.** **Done by the figure builder** (alignment rows, preregistered and precision-corrected; x-axis to +10%). v2.5 names tracking and alignment in the caption and points the Roles note to Figure 3.
9. **Hook: why the forecast sat near average.** **Rejected.** A generic spread fact (`c2_prior_spread_tau` 0.55, over all test receivers) does not answer a question about one player's forecast, and the writeup explains no named player's forecast by its inputs (section 4). "Worth only 0.14 games" stays the explanation, and the words went to accuracy items.
10. **Pre-submission.** **Partly applied:** the checker passes on the comment-stripped copy (0 failures); section 6 item 10 lists the E013 percent figures now quoted (+2.8%, −0.4% to +7.0%, 3%). **Deferred:** `[[TBD-NOTEBOOK-LINK]]` (at submission). **Rejected for the text:** a low-play rule of thumb (no committed key; per-play weighting needs an E012 re-run; Limitations keep the 2.0–3.5× fact).

### Judge 1, overclaims or errors (11)
- Roles "−0.06 to +0.16": **applied** (removed; like-for-like worth instead).
- "Not edge alignment": **applied.**
- "Is body type and roster designation"; 2.4 without its caveat in Section 3.2: **applied** ("tracks … (descriptive)"; "which may postdate the draft").
- "A roster-designation prior beats the Combine": **applied** (removed).
- Naming drift ("the Combine" for C2): **applied** ("the Combine + draft forecast −0.25 beyond it"; the Section 3.2 heading is now also true of the stopwatch and measurements alone, worth nearly the same).
- Section 3.3 heading: **applied** ("Combine tracking made estimates slightly worse").
- Clubs item 3 "ruled out": **applied** (the bound sentence left Clubs item 3; Section 3.2 gives the post hoc 90% bounds).
- "11–23" vs Table 1's 10.8: **applied** ("10.8–22.8").
- "Tackle" in the alignment list: **applied** (per-position shares).
- Two WR top-speed targets: **applied** ("typical game top speed" vs "game top-speed ceiling").
- Title and Section 3.2 generalize to "the Combine"; T8 omitted: **partly applied** ("On priced traits"; the estimand; outcome traits not priceable). The title stays (user decision); T8 values rejected (item 3).

### Judge 1, figure issues (9)
- Figure 3 E013 row and axis: **done** (figure builder).
- Figure 2 note ᵇ: **done** ("roster designation (DE/DT/NT/OLB; may postdate the draft and reflect scheme and stance)").
- Figure 2 row order: **done** (Table 1's order). Height (2,400 × 3,029) and footer length: **to do** (750 px check; figure_plan).
- Figure 1 live "speed now" column: **to check** in the GIF frames (figure_plan).
- Figure 1 football context: **done** ("GO route, 4th & 5", line of scrimmage).
- Figure 1 diamond legend: **done.**
- Figure 3 y-axis units: **done** ("1 = between-player variance").
- Figure 5 WR route-drills row: **done** ("median 0.70, 20 metrics"; "one-session retests: inflated").
- Figure 4 "(same signal)": **done** (‡ footnote: the pipeline sanity check).

### Judge 2, top improvements
1. **Roles note (selection, intervals, like-for-like, CI not MDE).** **Applied** (item 1 above): "(95% CIs 4–57%, 4–44%, −1% to 28%; the only three of nine passing false-discovery control)"; 0.2–0.9 and 0.8–1.1 (both estimators); "its 95% CI (−0.4% to +7.0%) excludes even the 3% cut exactly known alignment would bring; a post hoc precision correction agrees".
2. **DL sentence vs the attribution.** **Applied** (item 6 above; the 0.71 shared and the four-level label are in the src comment; "may postdate the draft" is in Section 3.2).
3. **Estimand in the headline; population bounds.** **Applied** in Section 3.2 ("Learned from 62–85 players per trait"; "a perfectly learned Combine signal could be worth up to 4–15 games (both post hoc)"; heading "On priced traits"). Clubs item 3 no longer states the forecast's worth, so it needs no estimand; the Limitations estimand bullet moved into Section 3.2.
4. **Combine-only worth, pooled.** **Applied:** "without the draft slot, nearly the same (it adds at most 0.1)" (C2 − C1 over priced traits, max 0.104). Pooled only; the guard forbids it for a named player.
5. **Tracking claim on H3; scope; the second non-null link.** **Applied:** heading and text rest on H3; the ΔCGE range and "every interval including 0" are gone; "as we modeled it" and "five or six tracked features, 62–85 players"; Section 4 names both non-null links as fragile WR speed links (Figure 4's ‡ row is E001-WR). **Deferred:** a notebook check of the T1 increment with c40_vmax alone (post hoc; E012 owner).
6. **Figure 2: 90%, † flags, row order.** **Applied** in the text ("90% bounds"; caption "calibrated 90% bounds bold, 95% bootstrap thin"); **done** in the render.
7. **Table 1 footnote.** **Applied:** "CIs: preregistered bootstrap, CGE part biased low."
8. **Figure 1 band, colour ramp, "≈".** **Done** by the figure builder.
9. **Intervals for 3.5% and 2.5%.** **Applied** ("both intervals include 0"). **Deferred:** a bootstrap for the 68% (notebook; Figure 3 says "post hoc, no CI").
10. **"Repeatable within a session"; 16 of the 22.** **Applied** (Section 5 and League item 1). The 75 words of headroom under 2,000: not used (the instructed cap is 1,900; section 7, item 9).

### Judge 2, overclaims or errors (12)
- 1 (floored "beyond" range), 2 (MDE vs the CI), 3 (shares without intervals or selection): **applied.**
- 4 (DL "not edge alignment"; four-level label; leakage caveat; stance): **applied.**
- 5 (heading and Clubs item 3 vs the estimand; population bounds 3.8–14.9): **applied** (quoted as 4–15).
- 6 (the title's "a Combine" vs C2): **applied** (Combine-only worth, pooled).
- 7 (calibrated bounds read as 95%): **applied.**
- 8 ("every interval including 0" for ΔCGE; general tracking advice): **applied.**
- 9 (E001-WR unexplained): **applied.**
- 10 ("stable traits" for same-day reps): **applied.**
- 11 (outcome traits "not priceable", not "small"): **applied.**
- 12 (missing uncertainty): 3.5% and 2.5% **applied**; calibration band **done** (Figure 3 shading and per-trait dots); 68%, the designation's 2.4 and 6.0, and the 1.0 beyond deep share **deferred** (no committed intervals; notebook bootstrap, post hoc).

### Judge 2, figure issues (10)
- Figure 1 band; colour ramp, peak, "≈", diamond: **done.**
- Figure 2 90% legend; row order and † flags; tracking panel note: **done.** Bootstrap lines hidden under the brackets and the 750 px legibility check: **to do.**
- Figure 3 density: **partly done** (axis to +10%, units, binomial band); still dense (to do: check at 750 px).
- Figure 4 second non-null row; grey drop-3 diamond: **done** (‡; navy).
- Figure 5: WR row and footer **done**; title "is a trait" vs the text's "repeatable within a session", and the drill-intensity row without its game null (the text no longer quotes it): **to do** (figure_plan).
- No figure for the Roles numbers: Figure 3 now draws the prospective rows; the shares and worth carry their intervals in the text.

### Cuts that paid for it (approximate rendered words; none is a caveat)
Section 1 "in games," −2; Section 2 "(WR top speed: 10 tracked plays)" −5 (Table 1 has it), the effective-sample-size clause and "and tracking traits, in that currency" −9, the amendment sentence −2; Section 3.1 "(Table 1)" and the qualifying-rate range −7; Section 3.2 "(five traits)" −2 (Table 1 shows the zeros), "negative fits clipped to 0" −6 (Table 1's footnote keeps "clipped"); Section 3.3 the ΔCGE sentence −14; Section 3.4 "(after 0–4 games)" −3; Section 4 the draft-slot median sentence −18 (Figure 4's subtitle keeps it), the detectable-effect clause −8, "regular-season" −1, the gate definition −4; Section 5 the drill-intensity sentence −23 (Figure 5 shows it), "beyond the stopwatch and draft slot" −6; Clubs item 1's C0 weights −13 (Table 1's caption gives k/(k + λ)), Clubs item 3's bound sentence; League item 1 heading −2, item 3 −2; Limitations' estimand bullet −16 (now Section 3.2); captions of Figures 2, 3 and 5 −8.

### Errors found in this pass (for LESSONS.md; the writer role may not edit it)
- v2.4's Roles note repeated the floored-zero error that LESSONS 2026-10-19 had just recorded ("adds −0.06 to +0.16 beyond", whose T3 value is a difference of two clipped zeros) → before quoting any cross-validated "beyond" or "adds" value, check whether either model is clipped at 0; if so, quote like-for-like worth (each set alone, both orders, the shared part) instead.
- My first estimate of the v2.5 additions (about +100 words) was half the measured +200 → measure each block of additions with the checker's counter as it is written, and plan the cuts before drafting.
- While merging two Section 3.2 sentences I attached a reason the source does not give ("may be conservative because negative fits are clipped to 0", for a sensitivity whose reason is the reference's out-of-fold cost) and caught it on rereading → after merging sentences, re-check every "because" and every colon-joined reason against the src comment.
- The v2.4 checker asserted "five traits" was gone but still quoted "every interval including 0" for tracking, a claim its own src comment showed rested on degenerate intervals → a checker that confirms a number is present should also assert the claim's premise (here: non-degenerate intervals that contain the estimate), or the claim should not be quoted.
