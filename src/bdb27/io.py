"""Load the raw competition CSVs once, cache them as Parquet, and read from the cache."""

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

TABLES = ("players", "combine_results", "player_career_successes", "games", "player_play")
SEASONS = (2023, 2024, 2025)

_ID = pl.Int64
_F = pl.Float64

GAME_TRACKING_SCHEMA = {
    "game_id": _ID, "play_id": _ID, "nfl_id": _ID, "time": pl.String,
    "x": _F, "y": _F, "s": _F, "a": _F, "dis": _F, "o": _F, "dir": _F, "event": pl.String,
}
COMBINE_TRACKING_SCHEMA = {
    "draft_year": pl.Int32, "event_id": pl.String, "nfl_id": _ID, "entity_type": pl.String,
    "time": pl.String, "drill_type": pl.String, "drill_name": pl.String, "attempt": pl.Int32,
    "x": _F, "y": _F, "s": _F, "a": _F, "dis": _F, "dir": _F,
}


def _clean(df: pl.DataFrame) -> pl.DataFrame:
    """Empty strings become null; id columns become Int64 so joins never hit dtype mismatches."""
    ids = [c for c in ("nfl_id", "game_id", "play_id") if c in df.columns]
    return df.with_columns(pl.col(pl.String).replace("", None), pl.col(ids).cast(_ID))


def _read_tracking(path: Path, schema: dict) -> pl.DataFrame:
    df = pl.read_csv(path, schema=schema, null_values="NA")
    return _clean(df).with_columns(pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", time_unit="ms"))


def build_cache(overwrite: bool = False) -> None:
    """Convert every raw CSV to Parquet in data/processed/. Safe to rerun."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    jobs = {name: lambda n=name: _clean(pl.read_csv(RAW / f"{n}.csv", null_values="NA", infer_schema_length=None))
            for name in TABLES}
    jobs["combine_tracking"] = lambda: _read_tracking(RAW / "combine_tracking.csv", COMBINE_TRACKING_SCHEMA)
    for season in SEASONS:
        jobs[f"game_tracking_{season}"] = lambda s=season: _read_tracking(
            RAW / f"game_tracking_{s}.csv", GAME_TRACKING_SCHEMA)
    for name, read in jobs.items():
        out = PROCESSED / f"{name}.parquet"
        if out.exists() and not overwrite:
            continue
        read().write_parquet(out)
        print(f"wrote {out.name}")


def load(name: str) -> pl.DataFrame:
    """One of TABLES, from the Parquet cache."""
    return pl.read_parquet(PROCESSED / f"{name}.parquet")


def scan_combine_tracking() -> pl.LazyFrame:
    return pl.scan_parquet(PROCESSED / "combine_tracking.parquet")


def scan_game_tracking(seasons: tuple[int, ...] = SEASONS) -> pl.LazyFrame:
    """Raw (unstandardized) game tracking, with a `season` column. Use bdb27.standardize for analysis."""
    return pl.concat([
        pl.scan_parquet(PROCESSED / f"game_tracking_{s}.parquet").with_columns(pl.lit(s, pl.Int32).alias("season"))
        for s in seasons
    ])


if __name__ == "__main__":
    build_cache()
