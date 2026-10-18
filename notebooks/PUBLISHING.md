# Publishing on Kaggle: the checklist

This is the exact order of work to publish the reproducibility notebook and the writeup. Background is in `README.md`: why there are four stages, the [stage C rule](README.md#the-stage-c-rule), and what the comparison checks.

**What gets published:**
- a public Kaggle dataset holding the code bundle;
- four Kaggle notebooks (stages A–D), all made from `notebooks/final_writeup.ipynb`;
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
- **Stage D:** about 0.3 h.
- **In all:** about 13.4 h elapsed at the central estimate, up to about 19 h, plus your time between steps. Each stage runs unattended.
- These are estimates. Step 8 replaces them with predictions from Kaggle's own CPU.

**Decide two things before anything is public.**
1. **Player names in the published results files.**
   - The bundle (step 4) and every stage's output contain the committed results files. `experiments/results/E007_leaderboard.json` and `E012.json` name players and give their draft round and pick, and Receiver A is named in five results files.
   - So anyone who opens those files can identify Receiver A and see his draft status. The writeup deliberately withholds both (`writeup/outline.md` §4 and §7, item 2).
   - The notebook never prints these fields (`REDACT` in `run_all.py`), but the files themselves are readable.
   - The competition data already lets a determined reader identify him (outline.md §7, item 2), so this may be acceptable. If it is not, ask for a change before step 4.
2. **Whether stages A–C are public too.** Recommended (step 14). Stage D attaches their outputs, and a reader can open those inputs only if they are public.

## The checklist

### Part 1. Freeze the code (this machine)

1. **Commit everything.** The bundle is built from git, and every stage must use the same bundle.
   - The 2026-10-17 changes must be in the commit: the compatibility edits, and E012's row-order fix with its rerun results (README.md, Known issues 1 and 2).
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

### Part 4. The four notebooks (you)

6. **Create the four notebooks.** For each one: Code → New Notebook, then File → Import Notebook, and upload `notebooks/final_writeup.ipynb`. Do this four times.

   | Title | First code cell | Inputs to add now | Input to add later |
   |---|---|---|---|
   | bdb27 stage A | `STAGE = "A"` | the competition data (NFL Big Data Bowl 2027); your code bundle dataset | — |
   | bdb27 stage B | `STAGE = "B"` | the same two | bdb27 stage A, at step 10 |
   | bdb27 stage C | `STAGE = "C"` | the same two | bdb27 stage A, at step 10 |
   | The Combine Exchange Rate: reproducibility | `STAGE = "D"` | the same two | bdb27 stage B and bdb27 stage C, at step 11 |

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
   ```
   - **The output says `DIFFERS`:** stop, and report the printed values. Something computes differently on Kaggle; do not run the stages until it is understood.
   - **Stage C's line says `fits`:** go on to step 9. That means a prediction of 10.5 h or less, i.e. T ≤ 161 s. Write T down.
   - **Otherwise** (T > 161 s; stage C shows "within 1.5 h of the limit" or "does NOT fit", plus a warning): stop.
     - Do not commit stage A. E012 must first be split across two sessions.
     - The split changes the code, and therefore the bundle. A stage A built now could not be merged with the later stages.
     - Ask for the split, then start again at step 1.
     - Do not re-measure in the hope of a lower number: the rule uses the largest measurement.

   The rule's arithmetic: predicted stage C hours = 229 × T / 3600 + 0.25. With a 1.5 h margin below the 12-hour session limit, that gives T ≤ 10.25 × 3600 / 229 = 161 s. Stages A and B fit whenever C does.

### Part 6. Run the stages (you)

Commit each stage with **Save Version → Save & Run All (Commit)**, keeping output saving on (the default). A committed run needs no attention, and you can close the browser.

9. **Stage A.** Commit **bdb27 stage A**. It takes about 177 × T / 3600 + 0.25 hours. When the version completes:
   - **Section 11** shows the timing table, then `workspace /kaggle/working/ws saved; steps still to run: S08, S09, S10, S11a, S11b, S12, S13, S14 (attach this notebook's output to the next stage)`. The appendices follow it.
   - **Stage A's own calibration:** read it at the end of section 3, and apply step 8 again with the larger of the two values of T. If stage C no longer fits, stop there, as in step 8.
   - **If the version failed or hit the 12-hour limit:**
     - Do not use it or make it public, and delete that version: its output may still hold competition-data caches (README.md, Known issues 5).
     - The failing cell names the step and its log, `_run/logs/<step>.log`. Report it.
10. **Stages B and C, at the same time.**
    - Attach the completed **bdb27 stage A** to both **bdb27 stage B** and **bdb27 stage C**: Add Input → Notebooks → Your Work.
    - Commit both. B takes about 140 × T / 3600 + 0.25 hours, C about 229 × T / 3600 + 0.25.
    - **Use the same versions in both:** the same stage A version and the same bundle version. Otherwise stage D refuses to merge them, with "conflicting outputs" or "produced by different code".
    - **One session at a time?** If Kaggle will not run two CPU sessions at once, run C first, then B.
    - **In each, section 11 ends with** `workspace /kaggle/working/ws saved; steps still to run: ...`. Handle failures as in step 9.
11. **Stage D.**
    - Attach the completed **bdb27 stage B** and **bdb27 stage C** to the stage D notebook. Stage A need not be attached: its output is already inside both.
    - Commit it. It takes about 0.3 h: the environment, the S01/S02 rebuild, steps S11a–S14 and the comparison.

12. **Check stage D's output.** Read the version's rendered notebook.
    - **Setup:** the `code bundle` line shows step 2's commit and `0 uncommitted paths`.
    - **Section 8:** `374 checks OK, 0 failed`, then `0 failures`.
      - It also prints `open placeholders: ['[[TBD-NOTEBOOK-LINK]]']`. That is expected: the link is added after this run (step 15).
    - **Section 10, the comparison table:** every file is **PASS**, and the verdict reads `OVERALL: PASS (16/16 files pass)`.
      - E012.json must be PASS, not PASS\*. A PASS\* means the bundle predates E012's row-order fix: go back to step 1.
    - **Below the table:** `transcribed constants: PASS`.
    - **Section 11:** all 19 steps, each run on Kaggle, and then `all steps done`.
    - **If the checker reports a failure:** stop, and report it.
    - **If a file FAILs only on floats** that differ near machine precision (Kaggle's OpenBLAS against Apple Accelerate; README.md, Known issues 3):
      1. widen `REL_TOL` in `notebooks/run_all.py`, and record why in README.md;
      2. commit, and rebuild the bundle (step 2);
      3. publish it as a new dataset version, and switch stage D's bundle input to it;
      4. rerun stage D only. The stage A–C outputs still attach, because `REL_TOL` is not part of the code fingerprint.
    - **If a file FAILs on anything else:** stop, and report the listed paths. It is a real difference.

### Part 7. Make it public (you)

13. **Before making a version public, check what it contains.**
    - **Only a version that ran to the end.** Its section 11 ends with `all steps done` or `workspace ... saved; ...`, and the appendices follow. A failed or timed-out version may hold competition-data caches: delete it instead.
    - **No competition data in its output** (the Output tab, under `ws/`):
      - `data/raw` is empty;
      - `data/processed` holds none of these Parquet files: `players`, `combine_results`, `player_career_successes`, `games`, `player_play`, `combine_tracking`, `game_tracking_2023`/`2024`/`2025`, `game_frames_post_snap`, `probe_e_game_frames`.
14. **Make stage D public.** Use the notebook's sharing settings.
    - Recommended: also make stages A, B and C public (see "Before you start"). Kaggle may ask you to make private inputs public when you publish stage D.
    - Copy stage D's URL, which looks like `https://www.kaggle.com/code/<your-username>/the-combine-exchange-rate-reproducibility`.

### Part 8. Link the notebook from the writeup (this machine)

15. **Replace the placeholder.** In `writeup/draft.md`, on the **Code.** line near the end, replace `[[TBD-NOTEBOOK-LINK]]` with `[Kaggle notebook](<stage D URL>)`.
    - **Digits.** The number checker fails on any digit in the text that no check covers, and the URL is part of the text. Stage D's suggested title gives a URL slug without digits.
    - **If your Kaggle username contains a digit,** step 16 will report `FAIL uncovered number` for it. Do not ignore that. Ask for the checker to treat the notebook URL as structure: a one-line change in `writeup/check_numbers.py`, like its existing exception for figure paths.
    - **Word budget.** The link adds one word. The draft is at 1,888 of 1,900 rendered words, so it fits.
16. **Re-run the number checker** on the draft, and on the comment-stripped copy that you will paste into Kaggle:
    ```
    .venv/bin/python writeup/check_numbers.py
    .venv/bin/python -c "import re, pathlib; raw = pathlib.Path('writeup/draft.md').read_text(); pathlib.Path('/tmp/bdb27-writeup-kaggle.md').write_text(re.sub(r'<!--.*?-->', '', raw, flags=re.S))"
    .venv/bin/python writeup/check_numbers.py /tmp/bdb27-writeup-kaggle.md
    ```
    - **Both runs** must end with `0 failures` and show `open placeholders: []`.
    - **The second run** also prints `stripped copy (no HTML comments): src-comment check skipped`.
    - **Then commit the draft.**
    - **Do not rebuild or re-upload the bundle for this edit.** `writeup/draft.md` is part of the code fingerprint, so a new bundle would no longer merge with the stage A–C outputs. Stage D checks the numbers, which the link does not change.

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
