"""Standardize game tracking so the offense always moves toward +x.

Conventions (verified on this data, 2026-10-14): `dir`/`o` are degrees clockwise from +y,
`s` is yards/sec. Game tracking has no direction column, so it is joined from player_play.
Plays without a known direction (~0.06% of tracked plays) are dropped.
"""

import polars as pl

from bdb27 import io

FIELD_LENGTH = 120.0
FIELD_WIDTH = 160 / 3


def play_directions(player_play: pl.DataFrame) -> pl.DataFrame:
    """One row per (game_id, play_id) with a non-null play_direction."""
    return (player_play.select("game_id", "play_id", "play_direction")
            .drop_nulls("play_direction").unique(["game_id", "play_id"]))


def standardize(tracking: pl.LazyFrame, directions: pl.DataFrame) -> pl.LazyFrame:
    """Flip 'left' plays: x -> 120 - x, y -> 160/3 - y, o/dir rotated 180 degrees."""
    left = pl.col("play_direction") == "left"
    return (tracking.join(directions.lazy(), on=["game_id", "play_id"], how="inner", maintain_order="left")
            .with_columns(
                x=pl.when(left).then(FIELD_LENGTH - pl.col("x")).otherwise(pl.col("x")),
                y=pl.when(left).then(FIELD_WIDTH - pl.col("y")).otherwise(pl.col("y")),
                o=pl.when(left).then((pl.col("o") + 180) % 360).otherwise(pl.col("o")),
                dir=pl.when(left).then((pl.col("dir") + 180) % 360).otherwise(pl.col("dir")),
            ))


def scan_game_tracking(seasons: tuple[int, ...] = io.SEASONS) -> pl.LazyFrame:
    """Standardized game tracking for analysis."""
    return standardize(io.scan_game_tracking(seasons), play_directions(io.load("player_play")))
