# NFL Big Data Bowl 2027

Our entry for the [NFL Big Data Bowl 2027](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027). The goal is to link 10 Hz Combine sensor tracking data to regular-season NFL performance.

See `AGENTS.md` for the competition details, approach, workflow, and timeline, and `PLAN.md` for how we iterate.

## Setup
1. Accept the competition rules on Kaggle.
2. Put your Kaggle API token in `~/.kaggle/access_token` (or a legacy `~/.kaggle/kaggle.json`) and run `chmod 600` on it.
3. Create the environment:
   ```
   python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
   ```
4. Download the data. The zip unpacks into a subfolder, so move the CSVs up into `data/raw/`:
   ```
   .venv/bin/kaggle competitions download -c nfl-big-data-bowl-2027 -p data/raw
   cd data/raw && unzip -q nfl-big-data-bowl-2027.zip && mv nfl-big-data-bowl-2027/*.csv . && rmdir nfl-big-data-bowl-2027 && rm nfl-big-data-bowl-2027.zip && cd ../..
   ```
5. Build the Parquet cache (~10 s) and run the tests:
   ```
   .venv/bin/python -m bdb27.io
   .venv/bin/pytest -m "not slow"
   ```
