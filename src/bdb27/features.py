"""Player-level features. Combine signatures and game traits, one row per nfl_id.

Speeds come from the `s` field (yd/s), which matches frame-to-frame motion. Never use `dis`:
in Combine tracking it under-counts distance (~0.58x the x/y step, verified 2026-10-14).
"""

import polars as pl

from bdb27 import io, standardize

MAX_PLAUSIBLE_SPEED = 12.0  # yd/s (~24.5 mph); above this is a tracking glitch
MIN_GAME_PLAYS = 50


def combine_top_speed() -> pl.DataFrame:
    """Peak speed (yd/s) in the player's best 40-yard dash attempt."""
    return (io.scan_combine_tracking()
            .filter((pl.col("entity_type") == "PLAYER") & (pl.col("drill_type") == "FORTY_YARD_DASH"))
            .group_by("nfl_id").agg(combine_top_speed=pl.col("s").max())
            .collect())


def reg_season_tracking() -> pl.LazyFrame:
    """Standardized game tracking, regular-season games only."""
    reg = io.load("games").filter(pl.col("season_type") == "REG").select("game_id")
    return standardize.scan_game_tracking().join(reg.lazy(), on="game_id", how="semi")


def game_top_speed(q: float = 0.95) -> pl.DataFrame:
    """The q-quantile of the player's regular-season play max speeds, with play count.
    A quantile does not grow with the number of plays; a top-k mean does (it measured snap count)."""
    play_max = (reg_season_tracking()
                .group_by("game_id", "play_id", "nfl_id").agg(smax=pl.col("s").max())
                .filter(pl.col("smax") <= MAX_PLAUSIBLE_SPEED))
    return (play_max.group_by("nfl_id")
            .agg(game_top_speed=pl.col("smax").quantile(q), game_plays=pl.len())
            .filter(pl.col("game_plays") >= MIN_GAME_PLAYS)
            .collect())


def update_store(features: pl.DataFrame, name: str) -> None:
    """Add/replace columns in data/processed/<name>.parquet (one row per nfl_id)."""
    path = io.PROCESSED / f"{name}.parquet"
    if path.exists():
        old = pl.read_parquet(path)
        keep = [c for c in old.columns if c == "nfl_id" or c not in features.columns]
        features = old.select(keep).join(features, on="nfl_id", how="full", coalesce=True)
    features.sort("nfl_id").write_parquet(path)
