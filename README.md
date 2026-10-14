# NFL Big Data Bowl 2027

Our entry for the [NFL Big Data Bowl 2027](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027). The goal is to link 10 Hz Combine sensor tracking data to regular-season NFL performance.

See `AGENTS.md` for the competition details, approach, workflow, and timeline.

## Setup
1. Accept the competition rules on Kaggle.
2. Put your Kaggle API token in `~/.kaggle/kaggle.json`.
3. Run `kaggle competitions download -c nfl-big-data-bowl-2027 -p data/raw && unzip data/raw/*.zip -d data/raw`.
