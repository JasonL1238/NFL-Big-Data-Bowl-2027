# Publishing on Kaggle: the checklist

This is the exact order of work to publish the reproducibility notebook and the writeup. Background is in `README.md`: why the pipeline runs in stages, the [stage C rule](README.md#the-stage-c-rule), how stage E resumes across sessions, and what the comparison checks.

**What gets published:**
- a public Kaggle dataset holding the code bundle;
- the stage notebooks, all made from `notebooks/final_writeup.ipynb`: A, B, C, D, and one or more for stage E (E013), which resumes across sessions;
- the public stage D notebook, which the writeup links;
- the Kaggle Writeup itself.

Kaggle's button and menu names change from time to time. Where a name here does not match the screen, look for the nearest equivalent.

## Who does what

**Only you, the account owner, can do these:**
- sign in to Kaggle;
- join the competition and accept its rules;
- verify a phone number, if Kaggle asks before turning on a notebook's internet;
- create an API token (only for the optional CLI route);
- create and upload the dataset;
- create the notebooks, attach their inputs and change their settings;
- run (commit) each version;
- delete a failed version;
- make the dataset and notebooks public;
- create and submit the writeup.

Claude cannot sign in to your account and does not make anything public for you.

**You or Claude, on this machine:**
- build the bundle and run the number checker;
- put the notebook link into the draft;
- make any code change that the stage C rule or a failed comparison calls for.

## Before you start

**Time.**
- **Pre-check:** about 30–45 min, interactive (step 7).
- **Stage A:** about 5.7 h at the central estimate (3.5–8.1 h).
- **Stages B and C, in parallel:** C is the longer, about 7.4 h (4.5–10.4 h).
- **Stage E (E013), after C:** about 16 h of compute at the central estimate, in two sessions. It fits one session if T ≤ 72 s and needs three at T ≈ 160 s (9.7–22.9 h). B can still be running.
- **Stage D:** about 0.3 h.
- **In all:** about 30 h elapsed at the central estimate, from about 18 h to about 42 h (A, then C, then E's sessions, then D, with B beside C and E), plus your time between steps. Each session runs unattended.
- These are estimates. Step 8 replaces them with predictions from Kaggle's own CPU.

**Decide two things before anything is public.**
1. **Draft details in the published results files.**
   - **What the files hold.** The bundle (step 4) and every stage's output contain the committed results files.
     - `experiments/results/E007_leaderboard.json` and `E012.json` give every player's draft round and pick next to his name.
     - `E013.json` names two more 2025 rookies (`post_hoc_review.prospective.t3_influence`).
   - **What the writeup and notebook say.** The writeup names players (user decision, 2026-10-18), but it never discusses a named player's draft status, draft position or off-field matters (`writeup/outline.md` §4). The notebook never prints names or draft fields (`REDACT` in `run_all.py`). The files themselves, though, are readable.
   - **Why this may be acceptable.** The competition data (`players.csv`) holds the same draft fields for every player, so the files show nothing a reader of the data cannot see. If it is not acceptable, ask for a change before step 4.
2. **Whether stages A, B, C and E are public too.** Recommended (step 14). Stage D attaches their outputs, and a reader can open those inputs only if they are public.

## The checklist

### Part 1. Freeze the code (this machine)

1. **Commit everything.** The bundle is built from git, and every stage must use the same bundle.
   - The 2026-10-17 changes must be in the commit: the compatibility edits, and E012's row-order fix with its rerun results (README.md, Known issues 1 and 2).
   - So must the 2026-10-19 changes: E013 in the pipeline (step S10b, stage E), E013's checkpoint-folder edit, and the writeup's v2.4 draft, checker and figures. The draft and both writeup scripts are part of the code fingerprint, so they must be final before the bundle is built.
   - Then run, from the repository root:
     ```
     git status --short                                    # must print nothing
     .venv/bin/python writeup/check_numbers.py             # must end with "0 failures"
     .venv/bin/pytest -q -p no:cacheprovider -m "not slow" # must pass
     ```
2. **Build the code bundle:**
   ```
   .venv/bin/python notebooks/run_all.py bundle --out /tmp/bdb27-dataset
   ```
   - **Expected last line:** `wrote /tmp/bdb27-dataset/bdb27-bundle.zip (… files, about 14 MB); git <commit>, 0 uncommitted paths`. Note the commit.
   - **If it lists uncommitted paths,** go back to step 1.
   - **If it stops with `FileExistsError`,** a bundle from an earlier build is in the folder. Delete that zip, or use a new folder.
   - **Contents:** every git-tracked file plus `notebooks/`, and never `data/`.

### Part 2. Your Kaggle account (you)

3. **Get the account ready.**
   - **Sign in** at kaggle.com, with the account that will submit (the team leader's, if you are a team).
   - **Join the competition and accept its rules,** at https://www.kaggle.com/competitions/nfl-big-data-bowl-2027. Without this, a notebook cannot attach the competition data.
   - **Phone number.** If Kaggle asks you to verify one before it turns a notebook's internet on, do it. Every stage needs internet during setup, to download uv, CPython 3.13.5 and the pinned wheels.
   - **Optional, CLI route only (step 4):** create an API token (Settings → API). Keep it out of the repository and out of chat.

### Part 3. The code bundle as a public dataset (you)

4. **Upload the bundle.**
   - **Web:** Datasets → New Dataset. Upload `/tmp/bdb27-dataset/bdb27-bundle.zip`, give it a title such as "Combine Exchange Rate code bundle", and create it.
     - Kaggle may unpack the zip and list its files instead. The notebook accepts either form.
   - **CLI, instead of the web:**
     ```
     .venv/bin/kaggle datasets init -p /tmp/bdb27-dataset
     # edit /tmp/bdb27-dataset/dataset-metadata.json: "title", "id": "<your-username>/<slug>", and "licenses"
     .venv/bin/kaggle datasets create -p /tmp/bdb27-dataset --public
     ```
5. **Make it public and set its license.**
   - **Visibility:** set the dataset to Public in its settings. The CLI's `--public` already did this.
   - **License:** the bundle contains no raw data, but it does contain results derived from the competition data, which is CC BY-NC 4.0. Choose a license no more permissive than that, for example CC BY-NC-SA 4.0 (`CC-BY-NC-SA-4.0` in the metadata).
   - **Version:** every notebook must attach this same version.
     - A later bundle goes up as a new version: New Version in the web UI, or `.venv/bin/kaggle datasets version -p /tmp/bdb27-dataset -m "<what changed>"`.
     - Then switch the input to that version in each notebook that needs it.

### Part 4. The stage notebooks (you)

6. **Create the five notebooks.** For each one: Code → New Notebook, then File → Import Notebook, and upload `notebooks/final_writeup.ipynb`. Do this five times. Step 10 adds one more stage E notebook for each extra session that stage E needs.

   | Title | First code cell | Inputs to add now | Input to add later |
   |---|---|---|---|
   | bdb27 stage A | `STAGE = "A"` | the competition data (NFL Big Data Bowl 2027); your code bundle dataset | — |
   | bdb27 stage B | `STAGE = "B"` | the same two | bdb27 stage A, at step 10 |
   | bdb27 stage C | `STAGE = "C"` | the same two | bdb27 stage A, at step 10 |
   | bdb27 stage E | `STAGE = "E"` | the same two | bdb27 stage C, at step 10 |
   | The Combine Exchange Rate: reproducibility | `STAGE = "D"` | the same two | bdb27 stage B and the last stage E notebook, at step 11 |

   - **Title of stage D.** Keep the title without digits: its URL goes into the writeup, where the number checker rejects digits (step 15).
   - **Edit only the first code cell** (`STAGE = "all"` becomes the stage letter). Leave every other cell as it is.
   - **Settings, in each notebook:**
     - Accelerator: **None** (CPU).
     - Internet: **On**.
     - Leave the other settings at their defaults. The notebook builds its own interpreter, so the Kaggle environment version does not matter.
   - **Inputs:** use Add Input. The competition is under Competitions, and the bundle under Datasets, Your Work.
   - **Later inputs:** a notebook's output can be attached only once that notebook has a completed version.

### Part 5. Calibrate first (you, about 30–45 minutes)

7. **Time the calibration call before committing anything long.** Open **bdb27 stage A** in the editor and start an interactive session with the inputs and settings from step 6. Then:
   1. Run the `STAGE = "A"` cell, then the setup cell below it. Setup builds the pinned interpreter, which takes several minutes, and prints:
      - the bundle's git commit, which must match step 2's;
      - the environment check;
      - the compatibility edits, all already in the repository.
   2. Run the code cell under **"1. Raw data to a Parquet cache (S01, S02)"**.
   3. Add a temporary code cell anywhere below it, and run this in it:
      ```
      ra.run(ctx, "S03")
      ```
   4. Run the calibration cell: the code cell under **"Calibration: how long will the next stages take on this machine?"**, at the end of section 3.
   5. Delete the temporary cell and stop the session. Do not save a version here.

8. **Apply the stage C rule.** The calibration cell prints:
   ```
   calibration call: T s (r x the 31.9 s reference); output matches
     stage A (...): about X h per session here -> fits
     stage B (...): about Y h per session here -> fits
     stage C (E012): about Z h per session here -> fits
     stage E (E013): about W h of compute here -> about N sessions: it stops 10.5 h into a session and the next stage E notebook resumes from its checkpoints
   ```
   - **The output says `DIFFERS`:** stop, and report the printed values. Something computes differently on Kaggle; do not run the stages until it is understood.
   - **Stage C's line says `fits`:** go on to step 9. That means a prediction of 10.5 h or less, i.e. T ≤ 161 s. Write T down.
   - **Otherwise** (T > 161 s; stage C shows "within 1.5 h of the limit" or "does NOT fit", plus a warning): stop.
     - Do not commit stage A. E012 must first be split across two sessions.
     - The split changes the code, and therefore the bundle. A stage A built now could not be merged with the later stages.
     - Ask for the split, then start again at step 1.
     - Do not re-measure in the hope of a lower number: the rule uses the largest measurement.

   The rule's arithmetic: predicted stage C hours = 229 × T / 3600 + 0.25. With a 1.5 h margin below the 12-hour session limit, that gives T ≤ 10.25 × 3600 / 229 = 161 s. Stages A and B fit whenever C does.

   **Stage E has no rule of its own.** Its line says `fits in one session` or `about N sessions`; write N down for step 10. Stage E needs only that its largest single piece of work fits in one session. That piece is one trait's coverage simulation, about 120 × T seconds: 5.4 h at T = 161 s. So whenever stage C fits, stage E progresses in every session.

### Part 6. Run the stages (you)

Commit each stage with **Save Version → Save & Run All (Commit)**, keeping output saving on (the default). A committed run needs no attention, and you can close the browser.

9. **Stage A.** Commit **bdb27 stage A**. It takes about 177 × T / 3600 + 0.25 hours. When the version completes:
   - **Section 12** shows the timing table, then `workspace /kaggle/working/ws saved; steps still to run: S08, S09, S10, S10b, S11a, S11b, S12, S13, S14 (attach this notebook's output to the next stage)`. The appendices follow it.
   - **Stage A's own calibration:** read it at the end of section 3, and apply step 8 again with the larger of the two values of T. If stage C no longer fits, stop there, as in step 8.
   - **If the version failed or hit the 12-hour limit:**
     - Do not use it or make it public, and delete that version: its output may still hold competition-data caches (README.md, Known issues 5).
     - The failing cell names the step and its log, `_run/logs/<step>.log`. Report it.
10. **Stages B and C at the same time, then stage E.**
    - **B and C.** Attach the completed **bdb27 stage A** to both **bdb27 stage B** and **bdb27 stage C**: Add Input → Notebooks → Your Work.
      - Commit both. B takes about 140 × T / 3600 + 0.25 hours, C about 229 × T / 3600 + 0.25.
      - **Use the same versions in both:** the same stage A version and the same bundle version. Otherwise stage D refuses to merge them, with "conflicting outputs" or "produced by different code".
      - **One session at a time?** If Kaggle will not run two CPU sessions at once, run C first, then E, then B.
      - **In each, section 12 ends with** `workspace /kaggle/working/ws saved; steps still to run: ...`. Handle failures as in step 9.
    - **Stage E, once C is done.** Attach the completed **bdb27 stage C** to **bdb27 stage E** and commit it. B may still be running.
      - Stage E runs E013 (step S10b) for about 510 × T / 3600 hours of compute.
      - **What it does at the deadline.** 10.5 hours into a session, run_all stops it. Its finished bootstrap and simulation checkpoints are kept, and the version saves normally.
      - **Expect** about as many sessions as step 8 printed for stage E.
    - **When a stage E version completes, read section 6 and the end of section 12.**
      - **S10b done:** stage E is finished. Go to step 11.
      - **`S10b ... stopped at the session deadline with N finished checkpoints kept`, and section 12 ends with `S10b (...) stopped at the session deadline with its checkpoints saved`:** create one more notebook, titled for example **bdb27 stage E, part 2**, exactly as in step 6 (`STAGE = "E"`, the competition data and the bundle). Attach the previous stage E version as its only notebook input; it already contains stage C's output. Commit it. It resumes from the checkpoints.
      - Repeat with a part 3 if part 2 also stops. Each part must attach the previous part, never an older one.
      - **This is not a failure.** A stopped stage E version is a normal, complete version, and it holds no competition data (`finish()` ran). Keep it: the next part needs it.
      - **Where it stops changes no number.** Every checkpoint draws from its own fixed seeds, and a resumed run reads the saved draws (README.md, Stage E across sessions).
11. **Stage D.**
    - Attach the completed **bdb27 stage B** and the **last stage E notebook** (the one whose S10b is done) to the stage D notebook. Stages A and C need not be attached: their outputs are already inside B and E.
    - Commit it. It takes about 0.3 h: the environment, the S01/S02 rebuild, steps S11a–S14 and the comparison.

12. **Check stage D's output.** Read the version's rendered notebook.
    - **Setup:** the `code bundle` line shows step 2's commit and `0 uncommitted paths`.
    - **Section 9:** `N checks OK, 0 failed`, then `0 failures`. N must equal the count that `.venv/bin/python writeup/check_numbers.py` prints on this machine for the committed draft.
      - It also prints `open placeholders: ['[[TBD-NOTEBOOK-LINK]]']`. That is expected: the link is added after this run (step 15).
    - **Section 11, the comparison table:** every file is **PASS**, and the verdict reads `OVERALL: PASS (17/17 files pass)`.
      - E012.json must be PASS, not PASS\*. A PASS\* means the bundle predates E012's row-order fix: go back to step 1.
      - E013.json lists two documented differences, `volatile` and `e013-reuse`. They are run metadata and the run mode, not values.
    - **Below the table:** `transcribed constants: PASS`.
    - **Section 12:** all 20 steps, each run on Kaggle, and then `all steps done`.
    - **If the checker reports a failure:** stop, and report it.
    - **If a file FAILs only on floats** that differ near machine precision (Kaggle's OpenBLAS against Apple Accelerate; README.md, Known issues 3):
      1. widen `REL_TOL` in `notebooks/run_all.py`, and record why in README.md;
      2. commit, and rebuild the bundle (step 2);
      3. publish it as a new dataset version, and switch stage D's bundle input to it;
      4. rerun stage D only. The stage A, B, C and E outputs still attach, because `REL_TOL` is not part of the code fingerprint.
    - **If a file FAILs on anything else:** stop, and report the listed paths. It is a real difference.

### Part 7. Make it public (you)

13. **Before making a version public, check what it contains.**
    - **Only a version that ran to the end.** Its section 12 ends with `all steps done` or `workspace ... saved; ...`, and the appendices follow. A stage E version stopped at the session deadline also ran to the end. A failed or timed-out version may hold competition-data caches: delete it instead.
    - **No competition data in its output** (the Output tab, under `ws/`):
      - `data/raw` is empty;
      - `data/processed` holds none of these Parquet files: `players`, `combine_results`, `player_career_successes`, `games`, `player_play`, `combine_tracking`, `game_tracking_2023`/`2024`/`2025`, `game_frames_post_snap`, `probe_e_game_frames`.
      - `data/scratch/e013/ckpt` (stage E onward) holds only E013's checkpoints: bootstrap and simulation draws, no competition data.
14. **Make stage D public.** Use the notebook's sharing settings.
    - Recommended: also make stages A, B, C and every stage E part public (see "Before you start"). Kaggle may ask you to make private inputs public when you publish stage D.
    - Copy stage D's URL, which looks like `https://www.kaggle.com/code/<your-username>/the-combine-exchange-rate-reproducibility`.

### Part 8. Link the notebook from the writeup (this machine)

15. **Replace the placeholder.** In `writeup/draft.md`, on the **Code.** line near the end, replace `[[TBD-NOTEBOOK-LINK]]` with `[Kaggle notebook](<stage D URL>)`.
    - **Digits.** The number checker fails on any digit in the text that no check covers, and the URL is part of the text. Stage D's suggested title gives a URL slug without digits.
    - **If your Kaggle username contains a digit,** step 16 will report `FAIL uncovered number` for it. Do not ignore that. Ask for the checker to treat the notebook URL as structure: a one-line change in `writeup/check_numbers.py`, like its existing exception for figure paths.
    - **Word budget.** The link adds one word. Step 16's checker run must still pass its budget lines.
16. **Re-run the number checker** on the draft, and on the comment-stripped copy that you will paste into Kaggle:
    ```
    .venv/bin/python writeup/check_numbers.py
    .venv/bin/python -c "import re, pathlib; raw = pathlib.Path('writeup/draft.md').read_text(); pathlib.Path('/tmp/bdb27-writeup-kaggle.md').write_text(re.sub(r'<!--.*?-->', '', raw, flags=re.S))"
    .venv/bin/python writeup/check_numbers.py /tmp/bdb27-writeup-kaggle.md
    ```
    - **Both runs** must end with `0 failures` and show `open placeholders: []`.
    - **The second run** also prints `stripped copy (no HTML comments): src-comment check skipped`.
    - **Then commit the draft.**
    - **Do not rebuild or re-upload the bundle for this edit.** `writeup/draft.md` is part of the code fingerprint, so a new bundle would no longer merge with the stage A, B, C and E outputs. Stage D checks the numbers, which the link does not change.

### Part 9. The Kaggle Writeup (you)

17. **Create the writeup** from the competition's Writeups page.
    - **Title:** the draft's first line without the `# `, "The Combine Exchange Rate: how many NFL games is a Combine worth?". If the title field is too short, use "The Combine Exchange Rate", and put the rest in the subtitle.
    - **Body:** paste `/tmp/bdb27-writeup-kaggle.md` without its title line.
18. **Embed the five figures.** Each `![Figure N](figures/...)` line marks where a figure goes.
    - **Upload** the file with the editor's image tool at that point, then delete the local-path line.
    - **Keep** the `**Figure N.** …` caption line under each image: it is part of the checked text.
    - **Use the committed files** in `writeup/figures/`:

      | Figure | File |
      |---|---|
      | 1 | `fig1_receiver_a.gif` (animated) |
      | 2 | `fig2_cge.png` |
      | 3 | `fig3_rookie_test.png` |
      | 4 | `fig4_transfer_map.png` |
      | 5 | `fig5_drill_reliability.png` |

    - **Alt text** for each figure is stored in its PNG. Figure 1's is in `fig1_receiver_a.png`, since the GIF has none. Print them all with:
      ```
      .venv/bin/python -c "from PIL import Image; [print(f, Image.open('writeup/figures/' + f).text['Description'], sep='\n', end='\n\n') for f in ('fig1_receiver_a.png', 'fig2_cge.png', 'fig3_rookie_test.png', 'fig4_transfer_map.png', 'fig5_drill_reliability.png')]"
      ```
    - **The checker's verdict still holds:** it ran with the local image paths, and the image lines carry no words.
19. **Preview the writeup and check it:**
    - **Table 1 renders as a table.** Do not re-space its compact pipes: spaced pipes add about 140 whitespace tokens, which breaks the 1,950 budget (`writeup/outline.md` §8).
    - **Figure 1** animates.
    - **Each caption** sits under its figure.
    - **The Code line's link** opens the public stage D notebook.
20. **Attach and submit.**
    - Attach the public stage D notebook to the writeup. The competition requires an attached public notebook.
    - Fill any other required fields, such as the track (AGENTS.md, The competition).
    - Submit. The deadline is **Jan 6, 2027, 23:59 UTC**; PLAN.md plans submission for Jan 5.
    - Afterwards, do not save a failing version of stage D: the link opens the latest version.
