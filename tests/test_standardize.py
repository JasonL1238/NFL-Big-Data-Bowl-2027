import numpy as np
import polars as pl

from bdb27.standardize import FIELD_WIDTH, play_directions, standardize


def _play(game_id, play_id, x0, vx, vy, n=10):
    """A straight-line run at 10 Hz with dir/o consistent with the motion (clockwise from +y)."""
    t = np.arange(n) * 0.1
    heading = np.degrees(np.arctan2(vx, vy)) % 360
    return pl.DataFrame({
        "game_id": game_id, "play_id": play_id, "nfl_id": 1,
        "x": x0 + vx * t, "y": 20 + vy * t, "o": heading, "dir": heading,
    })


def _heading_from_motion(df):
    dx, dy = np.diff(df["x"].to_numpy()), np.diff(df["y"].to_numpy())
    return np.degrees(np.arctan2(dx, dy)) % 360


def test_left_play_is_flipped_and_angles_stay_consistent_with_motion():
    tracking = pl.concat([_play(1, 1, 80, -5, 2), _play(1, 2, 30, 5, 2)])
    directions = pl.DataFrame({"game_id": [1, 1], "play_id": [1, 2], "play_direction": ["left", "right"]})
    out = standardize(tracking.lazy(), directions).collect().sort("play_id")

    left = out.filter(pl.col("play_id") == 1)
    assert np.allclose(left["x"], 120 - tracking.filter(pl.col("play_id") == 1)["x"])
    assert np.allclose(left["y"], FIELD_WIDTH - tracking.filter(pl.col("play_id") == 1)["y"])
    assert (np.diff(left["x"].to_numpy()) > 0).all()  # offense now moves toward +x
    assert np.allclose(left["dir"][:-1], _heading_from_motion(left))
    assert np.allclose(left["o"], left["dir"])

    right = out.filter(pl.col("play_id") == 2)
    assert right.select("x", "y", "dir").equals(tracking.filter(pl.col("play_id") == 2).select("x", "y", "dir"))


def test_angles_wrap_into_0_360():
    tracking = pl.DataFrame({"game_id": 1, "play_id": 1, "nfl_id": 1, "x": 10.0, "y": 10.0,
                             "o": [0.0, 179.9, 180.0, 359.9], "dir": [0.0, 179.9, 180.0, 359.9]})
    directions = pl.DataFrame({"game_id": [1], "play_id": [1], "play_direction": ["left"]})
    out = standardize(tracking.lazy(), directions).collect()
    assert np.allclose(out["dir"], [180.0, 359.9, 0.0, 179.9])


def test_plays_without_direction_are_dropped():
    tracking = pl.concat([_play(1, 1, 50, 5, 0), _play(1, 2, 50, 5, 0)])
    player_play = pl.DataFrame({"game_id": [1, 1, 1], "play_id": [1, 1, 2],
                                "play_direction": ["right", "right", None]})
    out = standardize(tracking.lazy(), play_directions(player_play)).collect()
    assert out["play_id"].unique().to_list() == [1]
