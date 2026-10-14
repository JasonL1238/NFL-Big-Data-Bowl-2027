#!/usr/bin/env bash
# Setup script for a Claude Code cloud environment.
# Needs env var KAGGLE_API_TOKEN (or KAGGLE_USERNAME + KAGGLE_KEY) and network access to kaggle.com.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
.venv/bin/pip install -q -e ".[dev]" kaggle

if [ ! -f data/raw/player_play.csv ]; then
  .venv/bin/kaggle competitions download -c nfl-big-data-bowl-2027 -p data/raw
  (cd data/raw && unzip -oq ./*.zip && rm -f ./*.zip)
fi
.venv/bin/python -m bdb27.io   # CSV -> Parquet cache; skips files that exist
