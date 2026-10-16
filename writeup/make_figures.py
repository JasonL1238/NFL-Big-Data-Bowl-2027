"""Regenerate every writeup figure in writeup/figures/ from committed results and the data/processed cache.

No statistic is recomputed here. Every drawn or printed value is read from a results key (named in the comment or
the `num(...)` call that prints it). The only arithmetic is presentation: yd/s -> mph, signing partial r by the
preregistered direction (sign x r), ratios and shares printed as percents ((ratio - 1) x 100), 1.96 x a stored SD
for a 95% range (Figure 1's predictive SD is the stored gap divided by the stored predictive z, which reproduces that
z exactly), the 20- and 40-yard crossings of the animated 40 (asserted to reproduce its stored flying 20), and
interpolating the 10 Hz frames for drawing and animation. Text is wrapped by its rendered width, so layouts hold in
Helvetica/Arial and DejaVu Sans.

Outputs (figure numbers follow writeup/draft.md):
  fig1_receiver_a.gif / .png   Receiver A: tracked 40 speed curves and one typical game play (animated), then his
                               rookie season and the in-season rule
  fig2_cge.png                 what the position average, the Combine + draft forecast and Combine tracking are worth
  fig3_rookie_test.png         2025 rookies: shrinkage does the work
  fig4_transfer_map.png        transfer map against two baselines
  fig5_drill_reliability.png   drill reliability by measurement type, with the like-for-like first-step comparison
Table 1 is a markdown table in draft.md (checked by writeup/check_numbers.py), not a figure.

Encoding, the same in every figure: grey triangle = position average; blue square = Combine + draft forecast;
navy diamond = Combine tracking; orange = the player's own games. Hollow = post hoc, and nothing else.

Every printed number is registered with its source key and, where the draft quotes the same quantity, asserted to
match the draft's rounding (`same(...)`). The run ends by printing that manifest; any mismatch raises.

Sources: experiments/results/E007.json, E007_leaderboard.json, E008.json, E009.json, E010.json, E011.json,
E012.json; experiments/posthoc_writeup_v2/e012_case_priors.json (Figure 1: Receiver A's stopwatch-only and
+ Combine tracking forecasts, a post hoc export of stored E012 values); data/processed/{combine_tracking,
e007_forty_attempts, e012_game_values, game_tracking_2025, games, player_play}.parquet (Figure 1 only: the 40 traces
and the game play).

Run: .venv/bin/python writeup/make_figures.py [--only fig1 fig2 ...] [--no-gif]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "writeup" / "figures"
MPH = 3600 / 1760  # yd/s -> mph
DPI = 300  # figures are W = 8 in wide: 2,400 px
W = 8.0
# Type scale. At a ~750 px Kaggle column an 8 in figure renders 1 pt as ~1.3 px: body 10 pt ~ 13 px, notes 9.5 pt
# ~ 12.4 px. Nothing smaller than FS_N is drawn.
FS, FS_N, FS_P, FS_T = 10.0, 9.5, 10.5, 13.0

# ---------------------------------------------------------------------------------------------------- style
# Palette (dataviz validator, light, surface #ffffff, --pairs all): BLUE/ORANGE are categorical slots 1/2 (CVD dE
# 24.7, normal 33.6, contrast >= 3:1). NAVY is the dark ordinal step of the same blue ("+ Combine tracking", nested on
# the forecast; blue->navy normal-vision dE >= 15, CVD >= 9.8) and GREY the neutral reference ("position average").
# Every series also carries its own marker and line style, so colour is never the only cue. Text uses INK (19.7:1)
# or INK2 (7.9:1 on white, >= 6.9:1 on BAND): WCAG AA. GREY (3.6:1) is never used for text.
SURF, INK, INK2, GREY, GRID, BASE, BAND = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f2f1ed"
BLUE, NAVY, ORANGE = "#2a78d6", "#0d366b", "#eb6834"
plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],  # per-glyph fallback (e.g. the triangle) to DejaVu
    "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS, "axes.edgecolor": BASE,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURF,
    "figure.facecolor": SURF, "savefig.facecolor": SURF, "axes.grid": False, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "legend.frameon": False, "axes.unicode_minus": True,
    "lines.solid_capstyle": "round"})
MS = 7  # marker size (points)

# Names on every figure, exactly as in draft.md
FC = "Combine + draft forecast"  # information set C2: stopwatch, measurements, draft slot
PA = "position average"  # C0
TRK = "+ Combine tracking"  # C3 - C2


def load(name: str) -> dict:
    return json.loads((RES / name).read_text())


E7LB, E8, E9, E10, E11, E12 = (load(f) for f in ("E007_leaderboard.json", "E008.json", "E009.json", "E010.json",
                                                   "E011.json", "E012.json"))

# ------------------------------------------------------------------------------------------- printed numbers
PRINTED: list[tuple[str, str, str]] = []  # (figure, printed text, source key)


def num(fig: str, value: float, spec: str, src: str, signed: bool = False) -> str:
    """Format a value for printing on a figure and register it with its source key."""
    s = format(value, ("+" if signed else "") + spec).replace("-", "−")
    PRINTED.append((fig, s, src))
    return s


def same(printed: str, draft: str, what: str) -> None:
    """The figure prints the same quantity the draft quotes: the strings must match."""
    if printed != draft:
        raise AssertionError(f"{what}: figure prints {printed!r}, draft quotes {draft!r}")


def minus(s: str) -> str:
    """Typographic minus signs on numbers only."""
    return re.sub(r"(?<![\w])-(?=\d)", "−", s)


# ------------------------------------------------------------------------------------------------ helpers
def new_fig(h: float, w: float = W):
    return plt.figure(figsize=(w, h))


def axes_in(fig, x: float, y: float, w: float, h: float, **kw):
    """Axes placed in inches from the figure's bottom-left corner."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes([x / fw, y / fh, w / fw, h / fh], **kw)


def ftext(fig, x: float, y: float, s: str, **kw):
    """Figure text placed in inches from the bottom-left corner."""
    fw, fh = fig.get_size_inches()
    kw.setdefault("color", INK2)
    kw.setdefault("size", FS)
    return fig.text(x / fw, y / fh, s, **kw)


def text_w(fig, s: str, size: float, weight: str = "normal") -> float:
    """Rendered width of a string in inches, in the font the machine actually resolves."""
    from matplotlib.font_manager import FontProperties
    fp = FontProperties(family=plt.rcParams["font.family"], size=size, weight=weight)
    return fig.canvas.get_renderer().get_text_width_height_descent(s, fp, ismath=False)[0] / fig.dpi


def wrap_in(fig, text: str, size: float, width: float, weight: str = "normal") -> list[str]:
    """Greedy word wrap by rendered width (inches) in the active font, so lines fit whatever font the machine
    resolves (Helvetica Neue / Arial locally, often DejaVu Sans elsewhere). Non-breaking spaces never split."""
    lines, cur = [], ""
    for word in text.split(" "):
        cand = f"{cur} {word}".strip()
        if cur and text_w(fig, cand, size, weight) > width:
            lines.append(cur)
            cur = word
        else:
            cur = cand
    return lines + [cur]


def header(fig, title: str, subtitle: str) -> float:
    """Left-aligned title (the finding) and subtitle (method and caveats); returns the y (in) below them."""
    fw, fh = fig.get_size_inches()
    t = wrap_in(fig, title, FS_T, fw - 0.3, "bold")
    ftext(fig, 0.1, fh - 0.1, "\n".join(t), size=FS_T, weight="bold", color=INK, va="top", linespacing=1.2)
    y = fh - 0.1 - len(t) * FS_T * 1.2 / 72 - 0.1
    sub = wrap_in(fig, subtitle, FS, fw - 0.3)
    ftext(fig, 0.1, y, "\n".join(sub), size=FS, va="top", linespacing=1.35)
    return y - len(sub) * FS * 1.35 / 72 - 0.08


def footer(fig, text: str, y: float = 0.08) -> float:
    """Notes along the bottom; returns the y (in) of their top."""
    lines = wrap_in(fig, text, FS_N, fig.get_size_inches()[0] - 0.3)
    ftext(fig, 0.1, y, "\n".join(lines), size=FS_N, va="bottom", linespacing=1.35)
    return y + len(lines) * FS_N * 1.35 / 72


def key_layout(fig, items: list, x0: float = 0.1, gap: float = 0.32) -> list:
    """Legend entries placed left to right, wrapping when the rendered width would pass the right edge."""
    rows, x = [[]], x0
    for draw, lab in items:
        w_ = 0.46 + text_w(fig, lab, FS)
        if rows[-1] and x + w_ > W - 0.1:
            rows.append([])
            x = x0
        rows[-1].append((x, draw, lab))
        x += w_ + gap
    return rows


def key_row(fig, y: float, items: list, x0: float = 0.1, gap: float = 0.32, dy: float = 0.3) -> float:
    """Legend entries (draw(ax, x_centre), label) from height y (in), wrapping onto further rows when the rendered
    width would pass the right edge; returns the y of the last row."""
    rows = key_layout(fig, items, x0, gap)
    for r, row in enumerate(rows):
        kax = axes_in(fig, 0, y - r * dy - 0.12, W, 0.24)
        kax.set_axis_off()
        kax.set_xlim(0, W)
        kax.set_ylim(-1, 1)
        for x, draw, lab in row:
            draw(kax, x + 0.2)
            kax.text(x + 0.46, 0, lab, va="center", size=FS, color=INK2)
    return y - (len(rows) - 1) * dy


def k_point(marker, color, ms=MS, hollow=False, line=None, lw=2.4):
    """Legend glyph: optional interval line plus a marker."""
    def draw(kax, xc):
        if line:
            kax.plot([xc - 0.18, xc + 0.18], [0, 0], color=color, lw=lw, solid_capstyle="butt", ls=line)
        if marker:
            kax.plot([xc], [0], marker=marker, ms=ms, ls="none", mfc=SURF if hollow else color,
                     mec=color if hollow else SURF, mew=1.5 if hollow else 1.0)
    return draw


def check_inside(fig, name: str) -> None:
    """Every visible text lies inside the figure (catches font-dependent clipping at the edges)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W_px, H_px = fig.bbox.width, fig.bbox.height
    bad = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        bb = t.get_window_extent(r)
        if bb.x0 < -1 or bb.x1 > W_px + 1 or bb.y0 < -1 or bb.y1 > H_px + 1:
            bad.append(t.get_text()[:60])
    if bad:
        raise AssertionError(f"{name}: text outside the figure: {bad}")


def save(fig, name: str, alt: str) -> Path:
    """PNG at DPI with the alt text in the file's Description metadata."""
    check_inside(fig, name)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=DPI, metadata={"Title": name, "Description": alt})
    plt.close(fig)
    print(f"wrote {path}")
    return path


def interval(ax, est, lo, hi, y, color, marker, lw=2.4, ms=MS, hollow=False, ring=True, arrow=False):
    """Point + interval line clipped to the x-limits (an arrowhead marks a clipped upper end when arrow=True).
    Hollow only for post hoc estimates. Returns True when the point lies outside its own interval."""
    a, b = ax.get_xlim()
    if hi - lo > 1e-9 and lo < b:
        ax.plot([max(lo, a), min(hi, b)], [y, y], color=color, lw=lw, zorder=3, solid_capstyle="butt")
        if arrow and hi > b:
            ax.plot([b], [y], marker=">", ms=4.5, color=color, mec=color, ls="none", zorder=3, clip_on=False)
    if est <= b:
        ax.plot([est], [y], marker=marker, ms=ms, ls="none", zorder=4, mfc=SURF if hollow else color,
                mec=color if hollow else (SURF if ring else color), mew=1.5 if hollow else (1.0 if ring else 0.6))
    return not (lo - 1e-9 <= est <= hi + 1e-9)


# =============================================================================================================
# Shared E012 names (exactly as in draft.md Table 1)
# =============================================================================================================
NAME = {"T1": "WR top speed", "T2": "WR speed kept through OUT/IN breaks", "T3": "WR HITCH braking",
        "T4": "WR separation over expected", "T5": "WR yards per route over expected",
        "T6": "DL first-second rush distance", "T7": "DL get-off", "T8": "DL pressure over expected",
        "T9": "OL pass-set acceleration", "T10": "OL pressure allowed over expected", "T11": "DB retreat speed",
        "T12": "DB top speed"}
CGE = {r["trait"]: r for r in E12["cge_table"]}
assert set(CGE) == set(NAME)
assert {t for t, r in CGE.items() if not r["identified"]} == {"T4", "T8", "T10"}  # draft: three traits not priced
IDENT = [t for t in NAME if CGE[t]["identified"]]


def one_dp(v: float) -> str:
    """Table 1's rounding: one decimal, and a plain 0 for zero."""
    return "0" if round(v, 1) == 0 else f"{v:.1f}"


# =============================================================================================================
# Figure 1: Receiver A (animated GIF + still PNG)
# =============================================================================================================
A_ID, B_ID = 59278, 58274  # E007.json concept_pair fast / slow (same official 40 within 0.01 s); names withheld


def forty_trace(nfl_id: int) -> dict:
    """The player's QC-passing tracked 40 whose flying 20 (v_20_40) is closest to his mean over QC-passing attempts
    (ties: the lower attempt number): raw x (yd past the start line) and device speed s, 10 Hz."""
    from bdb27 import speed  # START_X: start line in the Combine frame
    att = (pl.read_parquet(PROC / "e007_forty_attempts.parquet")
           .filter((pl.col("nfl_id") == nfl_id) & pl.col("qc_ok")).sort("attempt"))
    mean = att["v_20_40"].mean()
    att = att.with_columns(dist=((pl.col("v_20_40") - mean).abs() * 1e9).round()).sort("dist", "attempt")
    r = att.row(0, named=True)
    f = (pl.scan_parquet(PROC / "combine_tracking.parquet")
         .filter((pl.col("event_id") == r["event_id"]) & (pl.col("nfl_id") == nfl_id)
                 & (pl.col("attempt") == r["attempt"]) & (pl.col("entity_type") == "PLAYER")
                 & (pl.col("drill_type") == "FORTY_YARD_DASH"))
         .sort("time").collect())
    t = (f["time"] - f["time"][0]).dt.total_milliseconds().to_numpy() / 1000
    d = f["x"].to_numpy() - speed.START_X
    cross = dict(zip((20, 40), speed.crossing_times(t, d, [20, 40])))  # first crossing, interpolated
    assert abs(20 / (cross[40] - cross[20]) - r["v_20_40"]) < 1e-9  # reproduces the stored flying 20
    s = f["s"].to_numpy() * MPH
    # the drawn run: frames up to the 40-yd crossing, ending exactly at 40 yd
    k = int(np.searchsorted(t, cross[40]))
    tr, dr, sr = np.r_[t[:k], cross[40]], np.r_[d[:k], 40.0], np.r_[s[:k], np.interp(cross[40], t, s)]
    yl = f["y"].to_numpy()
    yr = np.r_[yl[:k], np.interp(cross[40], t, yl)]
    return {"t": tr, "d": dr, "s": sr, "y": yr, "t20": cross[20], "t40": cross[40], "attempt": r["attempt"]}


def game_play(nfl_id: int, case: dict) -> dict:
    """One of his game plays, typical of the trait (figure_plan.md v2.1 selection rule, fixed before drawing): from
    his first qualifying game, the play whose max device speed is closest to that game's E012 value (the 90th
    percentile of his play maxima; ties: the lower play_id). Play max = max raw `s` over all tracked frames of a
    regular-season player-play, plays above 12 yd/s dropped (E012 implementation choice T1_T12). Frames are
    standardized so the offense moves toward +x (bdb27.standardize)."""
    from bdb27 import standardize as st
    gv = (pl.read_parquet(PROC / "e012_game_values.parquet")
          .filter((pl.col("trait") == "T1") & (pl.col("nfl_id") == nfl_id) & (pl.col("season") == 2025))
          .sort("game_id"))
    assert gv["value"].to_list() == case["game_values"]  # same games, same order as the committed case study
    g0, v0 = gv["game_id"][0], gv["value"][0]
    reg = pl.read_parquet(PROC / "games.parquet").filter(pl.col("game_id") == g0)
    assert reg["season_type"][0] == "REG"
    frames = pl.scan_parquet(PROC / "game_tracking_2025.parquet").filter(
        (pl.col("game_id") == g0) & (pl.col("nfl_id") == nfl_id))
    pm = (frames.group_by("play_id").agg(smax=pl.col("s").max()).filter(pl.col("smax") <= 12.0)
          .sort("play_id").collect())
    assert pm.height == gv["n_plays"][0]
    assert abs(float(np.quantile(pm["smax"].to_numpy(), 0.9)) - v0) < 1e-9  # reproduces the stored game value
    pm = pm.with_columns(gap=(pl.col("smax") - v0).abs()).sort("gap", "play_id")
    play = pm["play_id"][0]
    pp = pl.read_parquet(PROC / "player_play.parquet").filter(
        (pl.col("game_id") == g0) & (pl.col("play_id") == play))
    f = (st.standardize(frames.filter(pl.col("play_id") == play), st.play_directions(pp))
         .sort("time").collect())
    ev = f["event"].to_list()
    i0 = ev.index("ball_snap")
    t = (f["time"] - f["time"][i0]).dt.total_milliseconds().to_numpy() / 1000
    x, y = f["x"].to_numpy(), f["y"].to_numpy()
    s = f["s"].to_numpy() * MPH
    assert abs(s.max() - pm["smax"][0] * MPH) < 1e-9
    assert np.argmax(s) >= i0  # the play's max comes after the snap, so drawing from the snap keeps it
    t, x, y, s = t[i0:], x[i0:] - x[i0], y[i0:] - y[i0], s[i0:]
    return {"t": t, "x": x, "y": y, "s": s, "y_snap": f["y"][i0], "x_snap": f["x"][i0],
            "peak": int(np.argmax(s)), "value": v0, "play_id": play, "game_id": g0}


def fig1(gif: bool = True) -> None:
    F = "fig1"
    from matplotlib import patheffects
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from PIL import Image

    case = next(c for c in E12["deliverables"]["case_studies"] if c["nfl_id"] == A_ID and c["trait"] == "T1")
    lb = {p["nfl_id"]: p for p in E7LB["players"] if p["nfl_id"] in (A_ID, B_ID)}
    n_wr = E7LB["meta"]["n"]
    pair = load("E007.json")["concept_pair"]
    assert pair["fast"]["nfl_id"] == A_ID and pair["slow"]["nfl_id"] == B_ID
    assert lb[A_ID]["rank"] == 1

    prior, pos_avg, season = case["prior_C2"] * MPH, case["prior_C0"] * MPH, case["season_mean"] * MPH
    games = np.array(case["game_values"]) * MPH
    run = np.array(case["running_mean"]) * MPH
    blend = np.r_[case["prior_C2"], case["blend_C2"]] * MPH
    n = case["n_games"]
    half = 1 + int(np.argmax(np.array(case["weight_C2"]) > 0.5))  # first k with weight on his games > 0.5
    # predictive SD of his season average under the forecast: the stored gap / the stored predictive z
    assert case["z_predictive"] > 0
    sd_pred = (case["season_mean"] - case["prior_C2"]) / case["z_predictive"]
    rng95 = 1.96 * sd_pred * MPH
    PRINTED.append((F, "drawn", "E012.deliverables.case_studies[59278,T1]: 95% range = prior_C2 +- 1.96 x "
                                "(season_mean - prior_C2) / z_predictive; game_values, running_mean, blend_C2"))
    p_txt = num(F, prior, ".1f", "E012.deliverables.case_studies[59278,T1].prior_C2 x mph")
    a_txt = num(F, pos_avg, ".1f", "E012.deliverables.case_studies[59278,T1].prior_C0 x mph")
    s_txt = num(F, season, ".1f", "E012.deliverables.case_studies[59278,T1].season_mean x mph")
    g1_txt = num(F, games[0], ".1f", "E012.deliverables.case_studies[59278,T1].game_values[0] x mph")
    same(p_txt, "17.6", "Receiver A forecast")
    same(a_txt, "17.5", "position average")
    same(s_txt, "18.9", "Receiver A season average")
    same(g1_txt, "18.3", "Receiver A first game")
    same(str(half), "2", "his games outweigh the forecast by game 2")
    same(str(n), "15", "qualifying games")
    w_2324 = num(F, case["weight_C2"][0], ".2f", "E012.deliverables.case_studies[59278,T1].weight_C2[0] (2023-24 fit)")
    w_all = num(F, E12["deliverables"]["weight_curves"]["T1"]["C2"]["w"][1], ".2f",
                "E012.deliverables.weight_curves.T1.C2.w[1] (2023-25 fit, Table 1)")
    same(w_all, "0.34", "Table 1 WR top speed weight after 1 game")
    f40 = {i: num(F, lb[i]["official_forty"], ".2f", f"E007_leaderboard.players[{i}].official_forty")
           for i in (A_ID, B_ID)}
    same(f40[A_ID], "4.39", "Receiver A 40")
    v20 = {i: num(F, lb[i]["flying20_mph"], ".1f", f"E007_leaderboard.players[{i}].flying20_mph")
           for i in (A_ID, B_ID)}
    same(v20[A_ID], "23.2", "Receiver A last 20 yards")
    tr_ = {i: forty_trace(i) for i in (A_ID, B_ID)}
    for i, tr in tr_.items():  # the animated attempt's own flying 20 prints the same as the player mean
        same(f"{20 / (tr['t40'] - tr['t20']) * MPH:.1f}", v20[i], f"{i} animated attempt flying 20")
    PRINTED.append((F, f"{n_wr}", "E007_leaderboard.meta.n (rank 1: players[59278].rank)"))
    PRINTED.append((F, "curves", "data/processed/combine_tracking.parquet device speed s (attempt per forty_trace)"))
    note = E7LB["meta"]["post_hoc_official_splits"]["note"]
    assert note.startswith("The tracked component beyond the 40 is not corroborated by the official electronic")
    gp = game_play(A_ID, case)
    pk_txt = num(F, gp["s"][gp["peak"]], ".1f", "data/processed/game_tracking_2025.parquet: max device speed of the "
                                                "selected play (figure-only; equals the game's E012 value)")
    same(pk_txt, g1_txt, "the selected play's peak equals his game-1 value")
    # His game-0 forecast with Combine tracking added (C3 = C2 + tracking; post hoc export of the stored E012 k = 0
    # forecasts, 2023-24 fit as the case study). The stopwatch-only forecast (C1) is deliberately NOT drawn: beside
    # C2 it would reveal how the draft slot moved his forecast, i.e. hint at his draft status (outline.md section 4).
    cp = json.loads((ROOT / "experiments" / "posthoc_writeup_v2" / "e012_case_priors.json").read_text())
    cpa = next(c for c in cp["cases"] if c["nfl_id"] == A_ID and c["trait"] == "T1")
    for k in ("prior_C0", "prior_C2", "season_mean"):  # the export is the same case study as E012.json
        assert cpa[k] == case[k], k
    for k, v in (("prior_C0", 8.568), ("prior_C1", 8.683), ("prior_C2", 8.607), ("prior_C3", 8.749),
                 ("season_mean", 9.239), ("tracking_shift_C3_minus_C2", 0.142)):
        assert round(cpa[k], 3) == v, (k, cpa[k])  # the values the brief quotes (yd/s)
    assert cpa["tracking_shift_C3_minus_C2"] == cpa["prior_C3"] - cpa["prior_C2"]
    assert cp["pools"]["T1"] == {"n_test_players_k0": 15, "n_with_10plus_games": 11}
    CPK = "experiments/posthoc_writeup_v2/e012_case_priors.json cases[nfl_id=59278,T1]"
    c3 = cpa["prior_C3"] * MPH
    c3_txt = num(F, c3, ".1f", f"{CPK}.prior_C3 x mph (+ Combine tracking; post hoc export)")
    same(c3_txt, "17.9", "Receiver A forecast + Combine tracking")
    same(f"{(cpa['prior_C3'] - cpa['prior_C2']) * MPH:.1f}", "0.3", "draft: tracking nudged his forecast 0.3 mph")

    # ---------------------------------------------------------------- layout: two panels on top, one below
    fh = 7.43  # v2.1 7.0 + the footer's fourth line (hollow-marker note) + 0.25 in for the season panel
    fig = new_fig(fh)
    top = header(fig, "His forecast sat near average; by game 2 his own games outweighed it.",
                 "Receiver A, a 2025 rookie receiver, picked by a rule fixed in advance: the largest gap between "
                 f"the {FC} and the season average among rookie receivers with 10+ qualifying games.")
    fb = footer(fig, f"Fastest finish for his 40 time of {n_wr} Combine receivers; official electronic splits do "
                     f"not corroborate it (post hoc). Forecast and in-season estimate fit on 2023–24 (weight on "
                     f"his first game {w_2324}); Table 1 refits on 2023–25 ({w_all}). The 95% range is for a "
                     "15-game average under the forecast. Hollow diamond, just left of game 0 (post hoc): the same "
                     "forecast with Combine tracking added.")
    leg_items = [(k_point("o", ORANGE, ms=4.6), "his single games"),
                 (k_point(None, ORANGE, line=(0, (4, 2)), lw=1.8), "his average so far"),
                 (k_point(None, BLUE, line="-", lw=2.2), "in-season estimate (the rule)"),
                 (k_point("s", BLUE, line="-", lw=2.0), "forecast, 95% range")]
    n_rows = len(key_layout(fig, leg_items, gap=0.28))
    ky = fb + 0.26 + (n_rows - 1) * 0.3
    key_row(fig, ky, leg_items, gap=0.28)
    s_bot = ky + 0.12 + 0.55
    s_h = 1.9  # v2.1 1.65; +0.25 in so the two hollow-marker labels stack above the 95% range
    t_bot = s_bot + s_h + 0.32 + 0.55
    t_h = top - 0.42 - t_bot
    axL = axes_in(fig, 0.62, t_bot, 3.0, t_h)
    axF = axes_in(fig, 4.35, t_bot, 2.9, t_h)
    axR = axes_in(fig, 0.62, s_bot, 6.2, s_h)
    axL.set_title("Tracked 40: speed along the run", loc="left", size=FS_P, color=INK, weight="bold", pad=7)
    axF.set_title("A typical top-speed play, game 1", loc="left", size=FS_P, color=INK, weight="bold", pad=7)
    axR.set_title("Rookie season: 90th-percentile play top speed per game", loc="left", size=FS_P, color=INK,
                  weight="bold", pad=7)

    # ---- top left: speed vs distance on the 40, last 20 yards shaded
    axL.set_xlim(0, 40.5)
    axL.set_ylim(0, 27)
    axL.set_xticks([0, 10, 20, 30, 40])
    axL.set_yticks([0, 10, 20])
    axL.set_xlabel("Distance from the start line (yd)", labelpad=3)
    axL.set_ylabel("Speed (mph)", labelpad=3)
    axL.grid(axis="y", color=GRID, lw=0.8)
    axL.set_axisbelow(True)
    axL.axvspan(20, 40, color=BAND, lw=0, zorder=0)
    axL.text(30, 26.6, "last 20 yards", ha="center", va="top", size=FS_N, color=INK2)
    col = {A_ID: ORANGE, B_ID: INK2}
    sty = {A_ID: "-", B_ID: (0, (3.5, 2))}
    name = {A_ID: "Receiver A", B_ID: "Receiver B"}
    curves = {i: axL.plot([], [], color=col[i], lw=2.2, ls=sty[i], zorder=3)[0] for i in tr_}
    heads = {i: axL.plot([], [], "o", ms=6, color=col[i], mec=SURF, mew=1.0, zorder=4)[0] for i in tr_}
    dg = np.linspace(20, 40, 161)
    for tr in tr_.values():
        assert np.all(np.diff(tr["d"][tr["d"] >= 15]) > 0)  # monotone over the last 25 yd: interp is valid
    sA, sB = (np.interp(dg, tr_[i]["d"], tr_[i]["s"]) for i in (A_ID, B_ID))
    gap = axL.fill_between(dg, sB, sA, where=sA >= sB, color=ORANGE, alpha=0.28, lw=0, zorder=2, visible=False)
    tx = axL.transAxes
    cx = (0.27, 0.64, 0.88)
    y_hdr, y_row = 0.47, (0.33, 0.19)
    hdr_speed = axL.text(cx[2], y_hdr, "speed now", transform=tx, size=FS_N, color=INK2, ha="center", va="center")
    axL.text(cx[1], y_hdr, "40 time", transform=tx, size=FS_N, color=INK2, ha="center", va="center")
    reads = {}
    for j, i in enumerate(tr_):
        yy = y_row[j]
        axL.plot([cx[0] - 0.09, cx[0] - 0.02], [yy, yy], transform=tx, color=col[i], lw=2.2, ls=sty[i],
                 solid_capstyle="butt")
        axL.text(cx[0], yy, name[i], transform=tx, size=FS, color=INK, ha="left", va="center")
        axL.text(cx[1], yy, f"{f40[i]} s", transform=tx, size=FS, color=INK, ha="center", va="center")
        reads[i] = axL.text(cx[2], yy, "", transform=tx, size=FS, color=INK, ha="center", va="center")

    # ---- top right: the game play on the field, coloured by speed (one sequential orange ramp, light -> dark)
    cmap = LinearSegmentedColormap.from_list("speed", ["#f7c3a8", ORANGE, "#7d2c0e"])
    norm = Normalize(0, 24)
    x0g, x1g = -3.0, max(30.0, gp["x"].max() + 2)
    box_w, box_h = 2.9, t_h
    yc_ = (gp["y"].min() + gp["y"].max()) / 2
    y_half = (x1g - x0g) * box_h / box_w / 2
    axF.set_xlim(x0g, x1g)
    axF.set_ylim(yc_ - y_half, yc_ + y_half)
    axF.set_aspect("equal", adjustable="box")
    axF.set_yticks([])
    axF.spines["left"].set_visible(False)
    axF.set_xticks([0, 10, 20, 30])
    axF.set_xlabel("Yards downfield from his spot at the snap", labelpad=3)
    for xa in np.arange(10, 111, 5):  # yard lines every 5 yd (standardized frame: goal lines at 10 and 110)
        axF.axvline(xa - gp["x_snap"], color=GRID, lw=0.9, zorder=0)
    ysl = -gp["y_snap"]  # the near sideline (y = 0) relative to his snap spot
    if yc_ - y_half < ysl:
        axF.axhspan(yc_ - y_half, ysl, color=BAND, lw=0, zorder=0.5)
        axF.axhline(ysl, color=INK2, lw=1.0, zorder=1)
        axF.text(x1g - 0.5, ysl - 0.5, "sideline", ha="right", va="top", size=FS_N, color=INK2)
    ysl2 = 160 / 3 - gp["y_snap"]  # the far sideline
    if yc_ + y_half > ysl2:
        axF.axhspan(ysl2, yc_ + y_half, color=BAND, lw=0, zorder=0.5)
        axF.axhline(ysl2, color=INK2, lw=1.0, zorder=1)
    axF.plot([0], [0], marker="x", ms=6, mew=1.4, color=INK2, ls="none", zorder=2)
    axF.text(0.3, 0.7, "snap", size=FS_N, color=INK2, ha="left", va="bottom")
    pts = np.c_[gp["x"], gp["y"]]
    seg = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection([], cmap=cmap, norm=norm, lw=4.2, zorder=3, capstyle="round",
                        path_effects=[patheffects.Stroke(linewidth=5.6, foreground=INK2), patheffects.Normal()])
    axF.add_collection(lc)
    g_head = axF.plot([], [], "o", ms=6.5, color=INK, mec=SURF, mew=1.0, zorder=4)[0]
    g_read = axF.text(0.98, 0.95, "", transform=axF.transAxes, size=FS, color=INK, ha="right", va="top")
    pk = gp["peak"]
    pk_note = axF.annotate(f"peak {pk_txt} mph = his game-1 value", xy=(gp["x"][pk], gp["y"][pk]),
                           xytext=(0.03, 0.93), textcoords="axes fraction", ha="left", va="top", size=FS_N, color=INK,
                           arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, shrinkA=0, shrinkB=3), visible=False)
    cax = axes_in(fig, 4.35 + 2.9 + 0.12, t_bot + 0.1, 0.1, t_h - 0.2)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=[0, 6, 12, 18, 24])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=FS_N, length=2)
    cb.set_label("mph", size=FS_N, color=INK2, labelpad=2)

    # ---- bottom: the season
    axR.set_xlim(-1.1, n + 0.5)  # room left of game 0 for the tracking forecast
    lo_y = min(games.min(), prior - rng95) - 0.35
    hi_y = max(games.max(), prior + rng95) + 0.35
    # headroom for the hollow-marker label above the 95% range (5 + label + 4 pt)
    probe = fig.text(0, 0, f"+ Combine\ntracking {c3_txt}", size=FS_N, linespacing=1.15)
    lab_pt = probe.get_window_extent(fig.canvas.get_renderer()).height * 72 / fig.dpi
    probe.remove()
    f_need = (5 + lab_pt + 4) / (s_h * 72)  # share of the panel height above the range's top
    hi_y = max(hi_y, (prior + rng95 - f_need * lo_y) / (1 - f_need))
    axR.set_ylim(lo_y, hi_y)
    axR.set_xticks([0, 5, 10, 15])
    axR.yaxis.set_major_locator(plt.MultipleLocator(2))  # 2-mph grid as in v2.1 (the label headroom added 21)
    axR.set_xlabel("Qualifying games (10+ tracked plays)", labelpad=3)
    axR.set_ylabel("mph", labelpad=3)
    axR.grid(axis="y", color=GRID, lw=0.8)
    axR.set_axisbelow(True)
    axR.axhline(pos_avg, color=GREY, lw=1.4, zorder=1)
    axR.text(n + 0.7, pos_avg - 0.05, f"{a_txt} position\naverage", size=FS, color=INK2, va="top", ha="left",
             linespacing=1.15)
    axR.plot([0, 0], [prior - rng95, prior + rng95], color=BLUE, lw=2.0, zorder=3, solid_capstyle="butt")
    axR.plot([0], [prior], "s", ms=MS, color=BLUE, mec=SURF, mew=1.0, zorder=5)
    axR.annotate(f"Forecast {p_txt}", xy=(0.12, prior - 0.06), xytext=(0.8, lo_y + 0.3), size=FS, color=INK,
                 va="center", ha="left", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, shrinkA=0, shrinkB=3))
    # His tracking forecast (post hoc, so hollow), set just left of game 0 because it sits within 0.3 mph of the
    # forecast, with a vertical leader up to its label above the forecast's 95% range.
    rend = fig.canvas.get_renderer()
    mph_per_pt = (hi_y - lo_y) / (axR.get_window_extent(rend).height * 72 / fig.dpi)
    yl = prior + rng95 + 5 * mph_per_pt  # bottom of the lower label, 5 pt above the range's top
    for xm, ym, mk, ms_, colr, txt in ((-0.5, c3, "D", 5.0, NAVY, f"+ Combine\ntracking {c3_txt}"),):
        axR.plot([xm], [ym], marker=mk, ms=ms_, mfc=SURF, mec=colr, mew=1.5, ls="none", zorder=5)
        axR.plot([xm, xm], [ym + 5.5 * mph_per_pt, yl - 1.5 * mph_per_pt], color=INK2, lw=0.8, zorder=2,
                 solid_capstyle="butt")
        lab = axR.text(xm - 0.1, yl, txt, size=FS_N, color=INK, ha="left", va="bottom", linespacing=1.15)
        lab_top = axR.transData.inverted().transform(lab.get_window_extent(rend))[1, 1]  # rendered top (mph)
        yl = lab_top + 6 * mph_per_pt  # the next label stacks 6 pt above
    assert lab_top < hi_y - 3 * mph_per_pt, (lab_top, hi_y)  # the label sits inside the panel
    g_dots = axR.plot([], [], "o", ms=4.6, color=ORANGE, mec=SURF, mew=0.6, zorder=2)[0]
    g_run = axR.plot([], [], color=ORANGE, lw=1.8, ls=(0, (4, 2)), zorder=3)[0]
    g_est = axR.plot([], [], color=BLUE, lw=2.2, zorder=4)[0]
    season_line = axR.axhline(season, color=INK, lw=0.9, zorder=1, visible=False)
    season_txt = axR.text(n + 0.7, season + 0.05, f"{s_txt} his season\naverage", size=FS, color=INK, va="bottom",
                          ha="left", linespacing=1.15, visible=False)
    call = axR.annotate(f"Game {half}: his games now\noutweigh the forecast", xy=(half, blend[half]),
                        xytext=(2.5, hi_y - 0.2), size=FS, color=INK, va="top", ha="left",
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, shrinkA=0, shrinkB=4), visible=False)

    # ---------------------------------------------------------------- state setters
    t_end = max(lb[i]["official_forty"] for i in tr_) + 1e-6  # the 40 stops once both have reached 40 yards

    def set_run(ta: float) -> None:
        """40 curves and readouts at animation time ta (s; 0 = official start, finish at his official 40)."""
        done = True
        for i, tr in tr_.items():
            t = ta + tr["t40"] - lb[i]["official_forty"]
            m = tr["t"] <= t
            if t >= tr["t"][0]:
                xd = np.r_[tr["d"][m], np.interp(t, tr["t"], tr["d"])]
                ys = np.r_[tr["s"][m], np.interp(t, tr["t"], tr["s"])]
                curves[i].set_data(xd, ys)
                heads[i].set_data([xd[-1]], [ys[-1]])
            else:
                curves[i].set_data([], [])
                heads[i].set_data([], [])
            fin = t >= tr["t"][-1]
            done &= fin
            reads[i].set_text(f"{v20[i]} mph" if fin else f"{np.interp(t, tr['t'], tr['s']):4.1f} mph")
        hdr_speed.set_text("last 20 yd" if done else "speed now")
        gap.set_visible(done)

    def set_play(tp: float | None) -> None:
        """The game play at time tp (s from the snap; None = not started; inf = whole play)."""
        if tp is None:
            lc.set_segments([])
            g_head.set_data([], [])
            g_read.set_text("")
            pk_note.set_visible(False)
            return
        k = int(np.searchsorted(gp["t"], tp, side="right"))  # frames shown so far
        k = max(1, min(k, len(gp["t"])))
        lc.set_segments(seg[: k - 1])
        lc.set_array(((gp["s"][:-1] + gp["s"][1:]) / 2)[: k - 1])
        g_head.set_data([gp["x"][k - 1]], [gp["y"][k - 1]])
        fin = k == len(gp["t"])
        g_read.set_text("" if fin else f"{gp['s'][k - 1]:4.1f} mph")
        g_head.set_visible(not fin)
        pk_note.set_visible(k - 1 >= pk)

    def set_games(k: int) -> None:
        g_dots.set_data(np.arange(1, k + 1), games[:k])
        g_run.set_data(np.arange(1, k + 1), run[:k])
        g_est.set_data(np.arange(0, k + 1), blend[: k + 1])
        call.set_visible(k >= half)
        season_line.set_visible(k == n)
        season_txt.set_visible(k == n)

    # ---------------------------------------------------------------- still PNG (last frame)
    set_run(t_end)
    set_play(np.inf)
    set_games(n)
    alt = (f"Top left: speed along the tracked 40 for two receivers with the same official time ({f40[A_ID]} and "
           f"{f40[B_ID]} s). The curves rise together, and Receiver A's keeps climbing over the last 20 yards "
           f"({v20[A_ID]} vs {v20[B_ID]} mph), the fastest finish for his 40 time of {n_wr} Combine receivers, though "
           f"official electronic splits do not corroborate it. Top right: a typical top-speed play from his first "
           f"game drawn on the field and coloured by speed, a deep route peaking at {pk_txt} mph. Bottom: his {n} "
           f"rookie qualifying games. The forecast of {p_txt} mph sits by the {a_txt} mph {PA}; beside it, a hollow "
           f"diamond (post hoc) shows the forecast with Combine tracking added ({c3_txt}). His games average {s_txt} mph, just above the forecast's 95% range; the in-season estimate "
           f"rises toward his games and gives them more than half the weight from game {half}.")
    check_inside(fig, "fig1_receiver_a.png")
    fig.savefig(OUT / "fig1_receiver_a.png", dpi=DPI, metadata={"Title": "fig1_receiver_a.png", "Description": alt})
    print(f"wrote {OUT / 'fig1_receiver_a.png'}")
    if not gif:
        plt.close(fig)
        return

    # ---------------------------------------------------------------- GIF frames
    # The 40 runs in real time; 0.4 s after it finishes, the game play (real time, from the first tracked frame)
    # and the season (one game per 0.25 s) build together, then everything holds.
    gdpi = 200  # 1,600 px wide
    frames, durs = [], []

    def grab(ms_: int) -> None:
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        durs.append(ms_)

    fig.set_dpi(gdpi)
    set_play(None)
    set_games(0)
    set_run(-0.6)
    grab(700)
    for ta in [*np.arange(-0.6, t_end, 0.05), t_end]:  # real time, 20 frames per second
        set_run(float(ta))
        grab(50)
    grab(400)
    t0p = gp["t"][0]
    play_len = gp["t"][-1] - t0p
    steps = int(np.ceil(play_len / 0.05))
    for j in range(steps + 1):  # 20 frames per second; season games every 5 frames
        set_play(t0p + j * 0.05)
        set_games(min(n, j // 5))
        grab(50)
    for k in range(min(n, steps // 5) + 1, n + 1):
        set_games(k)
        grab(250)
    set_play(np.inf)
    set_games(n)
    grab(4500)
    # One palette from the final frame (median cut), then exact nearest-colour mapping: PIL's own palette mapping
    # is approximate and turns the white background into an off-white entry.
    pal = frames[-1].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    P = np.array(pal.getpalette()[: 3 * 255], dtype=np.int32).reshape(-1, 3)
    P[np.argmin(((P - 255) ** 2).sum(1))] = 255  # the entry nearest white is pure white
    # Every house colour gets an exact entry: median cut drops colours with few pixels (the navy diamond's outline
    # came out slate grey). Each missing one replaces the least-used entry of the final frame (ties: lower index).
    house = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)]
                      for c in (INK, INK2, GREY, GRID, BASE, BAND, BLUE, NAVY, ORANGE)])
    has = lambda c: bool((P == c).all(1).any())  # noqa: E731
    keep = {int(i) for c in [*house, [255, 255, 255]] for i in np.flatnonzero((P == c).all(1))}
    free = [int(i) for i in np.argsort(np.bincount(np.asarray(pal).ravel(), minlength=255)[:255], kind="stable")
            if int(i) not in keep]
    for c, i in zip([c for c in house if not has(c)], free):
        P[i] = c
    assert all(has(c) for c in house)
    flat_pal = P.flatten().tolist()

    def to_p(frame):
        a = np.asarray(frame, dtype=np.int32)
        key = (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]
        u, inv = np.unique(key, return_inverse=True)
        rgb = np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], 1)
        idx = ((rgb[:, None, :] - P[None, :, :]) ** 2).sum(-1).argmin(1).astype(np.uint8)
        out = Image.fromarray(idx[inv].reshape(a.shape[:2]), "P")
        out.putpalette(flat_pal)
        return out

    q = [to_p(f) for f in frames]
    assert q[0].convert("RGB").getpixel((2, 2)) == (255, 255, 255)
    path = OUT / "fig1_receiver_a.gif"
    q[0].save(path, save_all=True, append_images=q[1:], duration=durs, loop=2, optimize=True, disposal=1)
    plt.close(fig)
    size_mb = path.stat().st_size / 1e6
    assert size_mb <= 5.0, f"GIF {size_mb:.1f} MB > 5 MB"
    print(f"wrote {path} ({len(frames)} frames, {sum(durs) / 1000:.1f} s, {size_mb:.2f} MB, "
          f"plays 3 times then holds)")


# =============================================================================================================
# Figure 2: what the position average, the Combine + draft forecast and Combine tracking are worth, in games
# =============================================================================================================
def fig2() -> None:
    F = "fig2"
    groups = [["T1", "T2", "T3", "T5", "T4"], ["T6", "T7", "T8"], ["T9", "T10"], ["T12", "T11"]]
    lam = {t: CGE[t]["lambda"] for t in NAME}
    lam_lo = min(lam[t]["est"] for t in IDENT)
    lam_hi = max(lam[t]["est"] for t in IDENT)
    cge_hi = max(CGE[t]["cge_C2"]["est"] for t in IDENT)
    l1, l2, c2 = (num(F, v, ".1f", s) for v, s in ((lam_lo, "cge_table[identified].lambda.est min"),
                                                     (lam_hi, "cge_table[identified].lambda.est max"),
                                                     (cge_hi, "cge_table[identified].cge_C2.est max")))
    same(f"{l1} to {l2}", "1.5 to 6.8", "position average worth (draft 3.1)")
    same(c2, "1.2", "Combine and draft add up to (draft 3.1)")
    cov = E12["post_hoc_review"]["simulation"]["interval_coverage"]
    for t in ("T1", "T6"):  # "about 80%", and every simulated miss is low (all misses are upper bounds below truth)
        b = cov[t]["b1"]
        assert 0.75 <= b["coverage"] <= 0.85 and abs((1 - b["coverage"]) - b["share_upper_below_truth"]) < 1e-12
    PRINTED.append((F, "about 80%; every miss low", "post_hoc_review.simulation.interval_coverage.{T1,T6}.b1"
                                                    ".coverage, share_upper_below_truth"))
    n_lo = min(r["n_players"] for r in CGE.values())
    n_hi = max(r["n_players"] for r in CGE.values())
    PRINTED.append((F, f"{n_lo}–{n_hi}", "cge_table[*].n_players min-max"))
    same(f"{n_lo}–{n_hi}", "62–85", "players per trait")
    t1, t6 = CGE["T1"], CGE["T6"]
    role1 = E12["post_hoc_review"]["per_trait"]["T1"]["references"]["role"]
    assert abs(role1["combine_beyond_ref"] - t1["cge_C2_beyond_role"]) < 1e-12
    f2324 = num(F, t1["cge_C2_2023_24_seasons_fit"], ".2f", "cge_table[T1].cge_C2_2023_24_seasons_fit")
    same(f2324, "0.14", "WR CGE 2023-24 fit")
    assert t1["cge_C2_leave_one_class_out_range"][0] == 0.0  # "0 without one draft class"
    r_deep = num(F, role1["r_trait_deep_share"], ".2f", "post_hoc_review.per_trait.T1.references.role.r_trait_deep_share")
    beyond1 = num(F, role1["combine_beyond_ref"], ".1f", "post_hoc_review.per_trait.T1.references.role.combine_beyond_ref")
    same(r_deep, "0.61", "WR top speed vs deep-route share")
    same(beyond1, "1.0", "forecast beyond deep share")
    rtxt = num(F, t6["cge_role_reference"], ".1f", "cge_table[T6].cge_role_reference")
    same(rtxt, "2.4", "DL NFL position alone")
    btxt = num(F, t6["cge_C2_beyond_role"], ".2f", "cge_table[T6].cge_C2_beyond_role")
    same(btxt, "−0.25", "DL Combine beyond role")
    notes = {"T1": ("ᵃ", f"WR top speed (post hoc): {f2324} if fit on 2023–24 and 0 with one class left out; partly "
                          f"route usage (r\u00a0{r_deep} with deep-route share), and the forecast adds {beyond1} game "
                          "beyond it."),
             "T6": ("ᵇ", f"DL rush distance (post hoc): NFL roster position alone is worth {rtxt}; the Combine adds "
                          f"{btxt} beyond it.")}

    def unpriced_lines(t):
        r = CGE[t]
        rel = num(F, r["rel_player_mean"], ".2f", f"cge_table[{t}].rel_player_mean")
        sh = num(F, r["split_half"], ".2f", f"cge_table[{t}].split_half")
        r2 = num(F, r["r2_cv"]["C2"], ".2f", f"cge_table[{t}].r2_cv.C2")
        txt = f"Not priced: reliability\u00a0{rel}, split-half\u00a0{sh}; uncorrected forecast R²\u00a0{r2}"
        if r["exposure_r_trait_log_games"] > 0.3:  # the one trait whose forecast fit partly tracks playing time
            ex = num(F, r["exposure_r_trait_log_games"], "+.2f", f"cge_table[{t}].exposure_r_trait_log_games")
            txt += f", partly playing time (r\u00a0{ex} with games)"
        return [], txt
    same(f"{CGE['T8']['r2_cv']['C2']:.2f}", "0.21", "DL pressure forecast R2 (draft 3.2)")
    same(f"{CGE['T8']['exposure_r_trait_log_games']:+.2f}", "+0.40", "DL pressure exposure r")

    # ---- measurement pass: every height in inches, so a row with a wrapped note gets exactly its lines
    mfig = new_fig(5)
    lab_w = max(text_w(mfig, NAME[t], FS) for t in NAME)
    gap_, d_w = 0.22, 1.35
    ax_x = 0.1 + lab_w + 0.15
    ax_w = W - 0.15 - d_w - gap_ - ax_x
    LH = FS_N * 1.25 / 72
    wrapped, ypos, bands, y = {}, {}, {}, 0.1
    for gi, g in enumerate(groups):
        if gi:
            y += 0.1
        for t in g:
            if CGE[t]["identified"]:
                ypos[t] = y + 0.2
                y += 0.36
            else:
                first, rest = unpriced_lines(t)
                wrapped[t] = first + wrap_in(mfig, rest, FS_N, ax_w * 0.97)
                ypos[t] = y + 0.2
                bands[t] = (y + 0.04, ypos[t] + 0.075 + len(wrapped[t]) * LH + 0.07)
                y = bands[t][1] + 0.04
    H_plot = y + 0.04
    title = (f"For priced traits, the {PA} is worth {l1} to {l2} games of a player’s own record; the Combine and draft add 0 to "
             f"{c2}")
    sub = ("Qualifying games of his own NFL record that each prior is worth when forecasting his level (fit on "
           "2023–25). The Combine’s estimate is clipped at 0, so it reads low when its true worth is small. Its "
           "bootstrap intervals cover only about 80% in simulation and every miss is low, so the "
           "simulation-calibrated bounds are the primary statement.")
    foot = ("Bold blue line and brackets: simulation-calibrated bounds, each a one-sided 95% bound (post hoc). Thin "
            "lines: 95% player-bootstrap intervals; * marks an estimate outside its own bootstrap interval (each "
            "resample relearns the forecast from a subset of players). Not priced: reliability or odd/even "
            f"split-half below 0.5 (post hoc rule); uncorrected R²: before the reliability correction. {n_lo}–{n_hi} "
            f"players per trait. {notes['T1'][0]} "
            f"{notes['T1'][1]} {notes['T6'][0]} {notes['T6'][1]}")
    keys1 = [(k_point("v", GREY, ms=6.5, line="-", lw=1.4), f"{PA} alone"),
             (k_point("s", BLUE, line="-", lw=2.4), f"{FC} adds"),
             (k_point("D", NAVY, ms=6.2, line="-", lw=1.4), "Combine tracking adds")]

    def bracket_glyph(kax, xc):
        kax.plot([xc - 0.15, xc + 0.15], [0, 0], color=BLUE, lw=2.2, solid_capstyle="butt")
        for xb, sg in ((xc - 0.15, 1), (xc + 0.15, -1)):
            kax.plot([xb, xb], [-0.6, 0.6], color=BLUE, lw=1.4)
            kax.plot([xb, xb + sg * 0.04], [0.6, 0.6], color=BLUE, lw=1.4)
            kax.plot([xb, xb + sg * 0.04], [-0.6, -0.6], color=BLUE, lw=1.4)
    keys2 = [(bracket_glyph, "simulation-calibrated bounds (post hoc)"),
             (k_point(None, INK2, line="-", lw=1.0), "95% bootstrap interval (biased low)")]
    n_k1, n_k2 = len(key_layout(mfig, keys1)), len(key_layout(mfig, keys2))
    t_lines = len(wrap_in(mfig, title, FS_T, W - 0.3, "bold"))
    s_lines = len(wrap_in(mfig, sub, FS, W - 0.3))
    f_lines = len(wrap_in(mfig, foot, FS_N, W - 0.3))
    plt.close(mfig)
    head_h = 0.1 + t_lines * FS_T * 1.2 / 72 + 0.1 + s_lines * FS * 1.35 / 72 + 0.08
    key_h = 0.2 + (n_k1 - 1) * 0.3 + 0.3 + (n_k2 - 1) * 0.3
    foot_h = 0.08 + f_lines * FS_N * 1.35 / 72
    fh = head_h + key_h + 0.55 + H_plot + 0.72 + foot_h

    fig = new_fig(fh)
    top = header(fig, title, sub)
    fb = footer(fig, foot)
    ky = key_row(fig, top - 0.2, keys1)
    ky = key_row(fig, ky - 0.3, keys2)
    ax_y = fb + 0.72
    assert abs((ky - 0.55) - (ax_y + H_plot)) < 1e-6
    ax = axes_in(fig, ax_x, ax_y, ax_w, H_plot)
    axd = axes_in(fig, ax_x + ax_w + gap_, ax_y, d_w, H_plot, sharey=ax)
    xmax, dlim = 8.0, 1.5
    ax.set_xlim(-0.2, xmax)
    axd.set_xlim(-dlim, dlim)
    ax.set_ylim(H_plot, 0)  # data y = inches from the top of the plot
    for a in (ax, axd):
        a.tick_params(axis="y", length=0)
        a.spines["left"].set_visible(False)
        a.grid(axis="x", color=GRID, lw=0.8)
        a.set_axisbelow(True)
    ax.set_title("Worth, in games", loc="left", size=FS_P, color=INK, weight="bold", pad=7)
    axd.set_title("Tracking adds", loc="left", size=FS_P, color=INK, weight="bold", pad=7)
    axd.axvline(0, color=INK2, lw=1.0, zorder=2)
    plt.setp(axd.get_yticklabels(), visible=False)
    tr = ax.get_yaxis_transform()
    dy = 0.075

    def brackets(yy, lo, hi):
        """Calibrated bounds: a bold line with square-bracket ends on the Combine row."""
        h, w_ = 0.06, 0.07
        if hi - lo > 1e-9:
            ax.plot([lo, hi], [yy, yy], color=BLUE, lw=2.2, zorder=3, solid_capstyle="butt")
        for xb, sg in ((lo, 1), (hi, -1)):
            ax.plot([xb, xb], [yy - h, yy + h], color=BLUE, lw=1.4, zorder=4, solid_capstyle="butt")
            ax.plot([xb, xb + sg * w_], [yy - h, yy - h], color=BLUE, lw=1.4, zorder=4, solid_capstyle="butt")
            ax.plot([xb, xb + sg * w_], [yy + h, yy + h], color=BLUE, lw=1.4, zorder=4, solid_capstyle="butt")

    outside = []
    for t, yy in ypos.items():
        r, L = CGE[t], lam[t]
        yl, yc = yy - dy, yy + dy
        interval(ax, L["est"], L["lo"], L["hi"], yl, GREY, "v", lw=1.4, ms=6.5, arrow=True)
        PRINTED.append((F, "drawn", f"cge_table[{t}].lambda.lo/hi"))
        if not r["identified"]:
            for a in (ax, axd):
                a.axhspan(*bands[t], color=BAND, lw=0, zorder=0)
            if L["lo"] >= xmax:  # the whole interval lies beyond the axis: an arrowhead at the edge
                ax.plot([xmax], [yl], marker=">", ms=4.5, color=GREY, ls="none", zorder=3, clip_on=False)
            ltxt = num(F, L["est"], ".1f", f"cge_table[{t}].lambda.est")
            ax.text(min(L["lo"], xmax) - 0.15, yl, f"{PA} alone: {ltxt} games", ha="right", va="center",
                    size=FS_N, color=INK2)
            ax.text(0.01, yc, "\n".join(wrapped[t]), transform=tr, va="top", ha="left", size=FS_N, color=INK2,
                    linespacing=1.25, style="italic")
            continue
        ax.text(L["est"], yl - 0.05, num(F, L["est"], ".1f", f"cge_table[{t}].lambda.est"), ha="center",
                va="bottom", size=FS_N, color=INK2)
        # Combine + draft forecast (CGE): calibrated bounds (bold, primary), B1 bootstrap (thin, just below)
        c = r["cge_C2"]
        brackets(yc, r["cge_C2_calibrated_lower95"], r["cge_C2_calibrated_upper95"])
        PRINTED.append((F, "brackets", f"cge_table[{t}].cge_C2_calibrated_lower95/upper95"))
        if c["hi"] - c["lo"] > 1e-9:
            ax.plot([c["lo"], c["hi"]], [yc + 0.07] * 2, color=INK2, lw=1.0, zorder=3, solid_capstyle="butt")
        PRINTED.append((F, "drawn", f"cge_table[{t}].cge_C2.lo/hi (B1)"))
        ax.plot([c["est"]], [yc], marker="s", ms=MS, color=BLUE, mec=SURF, mew=1.0, ls="none", zorder=5)
        if not (c["lo"] - 1e-9 <= c["est"] <= c["hi"] + 1e-9):
            outside.append(t)
        if round(c["est"], 1) != 0:  # zeros read off the axis (Table 1 prints them)
            vtxt = num(F, c["est"], ".1f", f"cge_table[{t}].cge_C2.est")
            same(vtxt, one_dp(c["est"]), f"{t} CGE as in Table 1")
            ax.text(c["est"], yc - 0.06, vtxt + ("*" if t in outside else "") + (notes[t][0] if t in notes else ""),
                    va="bottom", ha="center", size=FS_N, color=INK, zorder=6)
        d = r["delta_cge_tracking"]
        if interval(axd, d["est"], d["lo"], d["hi"], yy, NAVY, "D", lw=1.4, ms=6.0, ring=False):
            axd.text(d["est"] + (0.13 if d["est"] >= 0 else -0.13), yy - 0.05, "*", ha="center", va="center",
                     size=FS, color=INK)
            outside.append(t + " (tracking)")
    # row labels
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels([NAME[t] for t in ypos], size=FS)
    for lab, t in zip(ax.get_yticklabels(), ypos):
        lab.set_color(INK if CGE[t]["identified"] else INK2)
        if not CGE[t]["identified"]:
            lab.set_style("italic")
    ax.set_xticks(range(0, 9))
    axd.set_xticks([-1, 0, 1])
    axd.set_xticklabels(["−1", "0", "+1"])
    ax.set_xlabel("Qualifying games of his own NFL record", labelpad=4)
    axd.set_xlabel("Games", labelpad=4)
    # checks of quoted values
    same(one_dp(t6["cge_C2"]["est"]), "1.2", "DL rush CGE")
    same(one_dp(t1["cge_C2"]["est"]), "0.5", "WR top speed CGE")
    same(f"{t1['cge_C2_calibrated_lower95']:.2f}–{t1['cge_C2_calibrated_upper95']:.1f}", "0.08–1.6",
         "WR calibrated bounds")
    same(f"{t6['cge_C2_calibrated_lower95']:.2f}–{t6['cge_C2_calibrated_upper95']:.1f}", "0.10–3.9",
         "DL rush calibrated bounds")
    rest = [CGE[t]["cge_C2_calibrated_upper95"] for t in IDENT if t not in ("T1", "T6")]
    same(f"{min(rest):.1f}–{max(rest):.1f}", "1.7–3.6", "other calibrated upper bounds")
    assert all(CGE[t]["cge_C2"]["lo"] <= 0 <= CGE[t]["cge_C2"]["hi"] for t in CGE)  # every B1 interval includes 0
    assert sorted(outside) == ["T12", "T12 (tracking)", "T2 (tracking)"], outside
    nl_lo = min(lam[t]["est"] for t in NAME if t not in IDENT)
    nl_hi = max(lam[t]["est"] for t in NAME if t not in IDENT)
    same(f"{nl_lo:.1f}–{nl_hi:.1f}", "10.8–22.8", "not-priced traits: position average worth (Table 1)")
    dmin = min(CGE[t]["delta_cge_tracking"]["est"] for t in IDENT)
    dmax = max(CGE[t]["delta_cge_tracking"]["est"] for t in IDENT)
    same(f"{dmin:+.2f} to {dmax:+.2f}", "-0.24 to +0.18", "tracking adds")
    alt = minus(
        f"Dot-and-interval chart of 12 game traits, in games of a player's own record. For the nine priced traits "
        f"the {PA} alone is worth {l1} to {l2} games, and the {FC} adds {c2} games for DL first-second rush "
        f"distance, {one_dp(t1['cge_C2']['est'])} for WR top speed, about {one_dp(CGE['T7']['cge_C2']['est'])}-"
        f"{one_dp(CGE['T12']['cge_C2']['est'])} for DL get-off and DB top speed, and 0 for five traits. "
        f"Simulation-calibrated bounds put WR top speed at {t1['cge_C2_calibrated_lower95']:.2f} to "
        f"{t1['cge_C2_calibrated_upper95']:.1f} games and DL rush distance at "
        f"{t6['cge_C2_calibrated_lower95']:.2f} to {t6['cge_C2_calibrated_upper95']:.1f}, and rule out more than "
        f"{min(rest):.1f} to {max(rest):.1f} for the rest; the bootstrap intervals all include 0 but are biased "
        f"low. Three outcome traits are not priced; the {PA} alone is worth {nl_lo:.1f} to {nl_hi:.1f} games for "
        f"them, though for DL pressure over expected the forecast's cross-validated R² before correction is "
        f"{CGE['T8']['r2_cv']['C2']:.2f}. A second panel shows Combine tracking adding {dmin:+.2f} to {dmax:+.2f} "
        f"games, every interval including 0.")
    save(fig, "fig2_cge.png", alt)


# =============================================================================================================
# Figure 3: 2025 rookies, shrinkage does the work
# =============================================================================================================
def pct(F: str, r: dict, src: str, spec: str = ".0f", ci_spec: str = ".1f") -> tuple[str, float, float, float]:
    """An error ratio as a percent change in squared error: (ratio - 1) x 100, worded for club readers."""
    est, lo, hi = ((v - 1) * 100 for v in (r["ratio"], *r["ratio_ci95"]))
    e = format(abs(est), spec)
    if hi < 0:
        s = f"{e}% less error ({format(-hi, ci_spec)}–{format(-lo, ci_spec)}%)"
    elif lo > 0:
        s = f"{e}% more error ({format(lo, ci_spec)}–{format(hi, ci_spec)}%)"
    else:
        s = f"{e}% less error ({format(-lo, ci_spec)}% less to {format(hi, ci_spec)}% more)"
    PRINTED.append((F, s, src + ".ratio, ratio_ci95"))
    return s, est, lo, hi


def fig3() -> None:
    F = "fig3"
    v2, v2p = E12["validation_2"], E12["post_hoc_review"]["validation_2"]
    curves, prim = v2["pooled_curves"], v2["primary"]
    ph3, q10 = v2["post_hoc_combine_prior_vs_position_mean"], v2p["ph3_bh_q_over_10"]
    c0raw = v2p["blend_C0_vs_games"]["k1"]
    k0id = v2p["ph3_drivers"]["k0_identified_traits_only"]
    share = v2p["shrinkage_share_of_blend_gain"]["1"]
    H1, H2, H3 = prim["H1"], prim["H2"], prim["H3"]
    k0, k1 = ph3["k0"]["comb_C2_vs_C0"], ph3["k1"]["blend_C2_vs_C0"]
    # the draft quotes these; the figure prints the same quantities
    same(f"{share:.0%}", "99%", "shrinkage share of the blend gain after 1 game")
    same(f"{1 - H1['ratio']:.0%}", "72%", "H1 error cut")
    same(f"{1 - H2['ratio']:.0%}", "52%", "H2 error cut")
    same(f"{(1 - k0['ratio']) * 100:.0f}%", "5%", "Combine vs position average before game 1")
    same(f"{(1 - k0['ratio_ci95'][1]) * 100:.1f}% to {(1 - k0['ratio_ci95'][0]) * 100:.1f}%", "0.3% to 9.6%",
         "its CI")
    same(f"{(1 - k0id['ratio']) * 100:.1f}%", "3.5%", "identified traits only")
    same(f"{(1 - k1['ratio']) * 100:.1f}%", "2.5%", "Combine vs position average after 1 game")
    same(f"{(H3['ratio'] - 1) * 100:.1f}%", "2.1%", "H3 tracking worse")
    assert H1["n_players"] == H3["n_players"] == 81
    times = H1["pooled_b"] / H1["pooled_a"]
    same(f"{times:.1f}", "3.5", "raw first game vs rule error")
    PRINTED.append((F, f"{H1['n_players']}", "validation_2.primary.H1.n_players"))
    PRINTED.append((F, f"{share:.0%}", "post_hoc_review.validation_2.shrinkage_share_of_blend_gain['1']"))
    # H1 on the 9 priced traits (post hoc, figure_plan v2.1): H1's own pooling, an equal-weight mean over traits of
    # the per-trait mean squared errors; over all 12 traits it reproduces H1's committed pooled values exactly.
    sp = v2["secondary_per_trait"]
    pool = {key: {ts: float(np.mean([sp[t]["1"][key] for t in ts])) for ts in (tuple(NAME), tuple(IDENT))}
            for key in ("se_blend_C2", "se_games")}
    assert abs(pool["se_blend_C2"][tuple(NAME)] - H1["pooled_a"]) < 1e-9
    assert abs(pool["se_games"][tuple(NAME)] - H1["pooled_b"]) < 1e-9
    h1p = pool["se_blend_C2"][tuple(IDENT)] / pool["se_games"][tuple(IDENT)]
    h1p_txt = num(F, (1 - h1p) * 100, ".0f", "validation_2.secondary_per_trait[priced T]['1'].se_blend_C2 / se_games "
                                              "(equal-weight mean over the 9 priced traits)")
    same(f"{h1p_txt}%", "68%", "H1 on the nine priced traits")
    cal = v2["calibration_80pct"]["C2"]
    cov80 = [cal[str(k)]["pooled_coverage"] for k in range(5)]
    c_lo, c_hi = (num(F, v * 100, ".0f", "validation_2.calibration_80pct.C2[k].pooled_coverage min/max")
                  for v in (min(cov80), max(cov80)))
    same(f"{c_lo}–{c_hi}%", "77–81%", "80% interval coverage")

    title = "On 2025 rookies, shrinkage does the work; the Combine trims a few percent"
    sub = (f"Fit on the 2023–24 seasons, applied to {H1['n_players']} rookies in 2025 (pseudo-prospective: trait "
           "definitions were chosen after earlier analyses had used 2025 games). Blending toward the "
           f"{PA} alone captures {share:.0%} of the rule’s gain over his raw early games (post hoc).")
    foot = ("Error pooled over all 12 traits with equal weight (including the 3 not priced) unless noted, each "
            "scaled by how much players differ (2023–24 between-player variance); 95% player-bootstrap intervals. At "
            "0 games each rule is its prior. Filled: preregistered; hollow: post hoc; q: false-discovery-adjusted "
            "over 10 post hoc comparisons. Coverage: share of rookies’ remaining-games averages inside the rule’s "
            "80% predictive interval.")
    # forest rows in inches (top = 0): headers, full rows (label, value line, interval), one point-only row
    PITCH, HDR, PT = 0.52, 0.34, 0.36
    n_full, n_hdr = 7, 2
    H_r = 0.1 + n_hdr * HDR + 0.12 + n_full * PITCH + PT + 0.12
    mfig = new_fig(5)
    head_h = (0.1 + len(wrap_in(mfig, title, FS_T, W - 0.3, "bold")) * FS_T * 1.2 / 72 + 0.1
              + len(wrap_in(mfig, sub, FS, W - 0.3)) * FS * 1.35 / 72 + 0.08)
    foot_h = 0.08 + len(wrap_in(mfig, foot, FS_N, W - 0.3)) * FS_N * 1.35 / 72
    plt.close(mfig)
    c_h = 1.05
    fh = foot_h + 0.62 + c_h + 0.85 + H_r + 0.45 + head_h
    fig = new_fig(fh)
    top = header(fig, title, sub)
    fb = footer(fig, foot)
    # ---- left: pooled error by k (three series; the forecast-only and tracking series are in the right panel)
    y0 = fb + 1.25
    ax = axes_in(fig, 0.62, y0, 2.65, top - 0.45 - y0)
    series = [("se_games", None, "His own games only", ORANGE, "s"),
              ("se_blend_C0", "se_comb_C0", "Rule, position-average prior", GREY, "v"),
              ("se_blend_C2", "se_comb_C2", "Rule, Combine + draft prior", BLUE, "s")]
    dx = dict(zip([s[0] for s in series], (0.0, -0.1, 0.1)))
    for key, k0key, lab, col, mk in series:
        ks = [k for k in range(5) if k > 0 or k0key]
        xs = np.array(ks, float) + dx[key]
        ests = [curves[str(k)][key if k else k0key] for k in ks]
        ax.vlines(xs, [e["lo"] for e in ests], [e["hi"] for e in ests], color=col, lw=1.4, zorder=2)
        ax.plot(xs, [e["est"] for e in ests], color=col, lw=2.0, zorder=3, label=lab, marker=mk,
                ms=6.5, mec=SURF, mfc=col, mew=1.0)
        PRINTED.append((F, "drawn", f"validation_2.pooled_curves[k].{key} (k=0: {k0key})"))
    ax.set_xticks(range(5))
    ax.set_xlim(-0.4, 4.45)
    ax.set_ylim(0, 11.5)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Qualifying games observed")
    ax.set_ylabel("Squared forecast error (scaled)")
    ax.set_title("Error forecasting his remaining games", loc="left", size=FS_P, color=INK, weight="bold", pad=7)
    g1 = curves["1"]["se_games"]
    ax.annotate(f"His first game alone:\n{num(F, times, '.1f', 'validation_2.primary.H1.pooled_b / pooled_a')}"
                "× the rule’s error", xy=(1 + 0.08, g1["est"]), xytext=(1.4, 9.9), size=FS, color=INK,
                va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9, shrinkA=2, shrinkB=5))
    ax.text(4.3, 0.95, "both rules overlap", size=FS_N, color=INK2, ha="right", va="top")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", fontsize=FS, handlelength=2.0,
               labelspacing=0.35, borderaxespad=0, bbox_to_anchor=(0.1 / W, (y0 - 0.5) / fig.get_size_inches()[1]))

    # ---- right bottom: calibration of the rule's 80% intervals (preregistered secondary analysis)
    rx, rw = 3.75, 4.1
    c_bot = fb + 0.62
    axc = axes_in(fig, rx + 0.45, c_bot, rw - 0.45, c_h)
    axc.set_title(f"Its 80% intervals held {c_lo}–{c_hi}%", loc="left", size=FS_P, color=INK,
                  weight="bold", pad=7)
    axc.axhline(80, color=INK2, lw=1.0, zorder=1)
    axc.text(4.35, 80.4, "80% target", ha="right", va="bottom", size=FS_N, color=INK2)
    axc.plot(range(5), np.array(cov80) * 100, color=BLUE, lw=2.0, marker="s", ms=6.2, mec=SURF, mew=1.0, zorder=3)
    PRINTED.append((F, "drawn", "validation_2.calibration_80pct.C2[0..4].pooled_coverage"))
    axc.set_xlim(-0.3, 4.45)
    axc.set_ylim(70, 90)
    axc.set_yticks([70, 80, 90])
    axc.set_yticklabels(["70%", "80%", "90%"])
    axc.set_xticks(range(5))
    axc.set_xlabel("Qualifying games observed", labelpad=3)
    axc.grid(axis="y", color=GRID, lw=0.8)
    axc.set_axisbelow(True)
    # ---- right top: one linear axis of percent change in squared error
    r_bot = c_bot + c_h + 0.85
    assert abs((r_bot + H_r) - (top - 0.45)) < 1e-6
    axr = axes_in(fig, rx, r_bot, rw, H_r)
    axr.set_xlim(-88, 12)
    S, P = "validation_2.primary", "validation_2.post_hoc_combine_prior_vs_position_mean"
    qtxt = num(F, q10["k0:comb_C2_vs_C0"], ".2f", "post_hoc_review.validation_2.ph3_bh_q_over_10[k0:comb_C2_vs_C0]")
    same(qtxt, "0.29", "BH q over 10")
    rows = [
        ("hdr", "Shrinkage: the rule vs his raw early games"),
        (H1, "Rule (Combine + draft prior) vs his first game", "preregistered", f"{S}.H1", BLUE, "s", False, ".0f", ".0f"),
        ("pt", h1p, f"9 priced traits only: {h1p_txt}% less error · post hoc, no CI"),
        (H2, "Same rule vs his first two games", "preregistered", f"{S}.H2", BLUE, "s", False, ".0f", ".0f"),
        (c0raw, "Rule (position-average prior) vs his first game", "post hoc",
         "post_hoc_review.validation_2.blend_C0_vs_games.k1", GREY, "v", True, ".0f", ".0f"),
        ("hdr", f"The Combine vs the {PA}"),
        (k0, "Combine + draft vs position average, before game 1", f"post hoc, q {qtxt}", f"{P}.k0.comb_C2_vs_C0", BLUE, "s", True,
         ".0f", ".1f"),
        (k0id, "Same, on the 9 priced traits", "post hoc",
         "post_hoc_review.validation_2.ph3_drivers.k0_identified_traits_only", BLUE, "s", True, ".1f", ".1f"),
        (k1, "Same comparison inside the rule, after game 1", "post hoc", f"{P}.k1.blend_C2_vs_C0", BLUE, "s", True,
         ".1f", ".1f"),
        (H3, f"{TRK} vs without, game 1: not supported", "preregistered", f"{S}.H3", NAVY, "D", False,
         ".1f", ".1f"),
    ]
    assert sum(r[0] == "hdr" for r in rows) == n_hdr and sum(r[0] == "pt" for r in rows) == 1
    assert len(rows) - n_hdr - 1 == n_full
    y = 0.1  # inches from the top of the forest
    tr = axr.get_yaxis_transform()
    bb = dict(fc=SURF, ec="none", pad=0.5)
    for gi, row in enumerate(rows):
        if row[0] == "hdr":
            y += 0.12 if gi else 0.0
            axr.text(0.0, y + 0.12, row[1], transform=tr, size=FS_P, color=INK, weight="bold", va="center",
                     ha="left")
            y += HDR
            continue
        if row[0] == "pt":  # a post hoc point with no committed interval
            axr.text(0.0, y + 0.1, row[2], transform=tr, size=FS_N, color=INK2, va="center", ha="left", bbox=bb,
                     zorder=5)
            axr.plot([(row[1] - 1) * 100], [y + 0.27], marker="s", ms=6.2, mfc=SURF, mec=BLUE, mew=1.5, ls="none",
                     zorder=4)
            y += PT
            continue
        r, lab, tag, src, col, mk, hollow, spec, ci_spec = row
        s, est, lo, hi = pct(F, r, src, spec, ci_spec)
        axr.text(0.0, y + 0.1, lab, transform=tr, size=FS, color=INK, va="center", ha="left", bbox=bb, zorder=5)
        axr.text(0.0, y + 0.27, f"{s} · {tag}", transform=tr, size=FS_N, color=INK2, va="center", ha="left",
                 bbox=bb, zorder=5)
        interval(axr, est, lo, hi, y + 0.43, col, mk, lw=1.5 if hollow else 2.4, ms=6.2 if hollow else MS,
                 hollow=hollow)
        y += PITCH
    axr.set_ylim(H_r, 0)
    axr.set_xticks([-80, -60, -40, -20, 0])
    axr.set_xticklabels(["−80%", "−60%", "−40%", "−20%", "0"])
    axr.axvline(0, color=INK2, lw=1.0, zorder=1)
    axr.grid(axis="x", color=GRID, lw=0.8)
    axr.set_axisbelow(True)
    axr.set_yticks([])
    axr.spines["left"].set_visible(False)
    axr.set_xlabel("Change in squared error (left = better)", labelpad=3)
    alt = minus(
        f"Left: squared forecast error for 2025 rookies after 0 to 4 qualifying games. Using only his own games is "
        f"far worse early (about {g1['est']:.1f} after one game vs about {curves['1']['se_blend_C2']['est']:.1f} for "
        f"the rule), and the rules with the {PA} and with the {FC} as prior overlap. Right: percent change in error "
        f"on one axis. The preregistered rule cuts error {1 - H1['ratio']:.0%} after one game and "
        f"{1 - H2['ratio']:.0%} after two ({h1p_txt}% after one game on the nine priced traits, post hoc); a {PA} "
        f"rule does nearly the same. The {FC} beats the {PA} by "
        f"{(1 - k0['ratio']) * 100:.0f}% before the first game (post hoc, not robust to multiplicity), "
        f"{(1 - k0id['ratio']) * 100:.1f}% on the priced traits, and {(1 - k1['ratio']) * 100:.1f}% after one game; "
        f"adding Combine tracking is {(H3['ratio'] - 1) * 100:.1f}% worse. Bottom right: the rule's 80% intervals "
        f"held {c_lo} to {c_hi}% of outcomes at every k from 0 to 4.")
    save(fig, "fig3_rookie_test.png", alt)


# =============================================================================================================
# Figure 4: transfer map against two baselines
# =============================================================================================================
# Plain-English link labels (E009.json transfer_map[*].label is the technical label). Grouped by position.
LINKS = [
    ("WR", [("E006-P1", "40 top speed (fitted) → game top-speed capacity †"),
            ("E001-WR", "40 peak speed → game top speed (same signal)"),
            ("E002-P1", "Route-drill speed kept → OUT/IN speed kept"),
            ("E002-P2", "Route-drill speed kept → separation at the throw"),
            ("E002-P3", "Stop-route drill braking → HITCH braking")]),
    ("DL", [("E003-P1", "40 first-second burst → first-second rush distance"),
            ("E003-P2", "40 first-second burst → NGS get-off"),
            ("E003-P3", "40 first-second burst → quick pressures"),
            ("E001-DL", "40 peak speed → game top speed")]),
    ("OL", [("E004-P1", "Mirror-drill reversals → pass-set acceleration"),
            ("E004-P2", "Mirror-drill reversals → pressure rate allowed"),
            ("E004-P3", "Mirror-drill reversals → pressures allowed"),
            ("E001-OL", "40 peak speed → game top speed")]),
    ("DB", [("E005-P1", "Drill backpedal speed → pre-throw retreat speed"),
            ("E005-P2", "Drill plant-and-drive → post-throw redirection"),
            ("E001-DB", "40 peak speed → game top speed")]),
    ("Pooled", [("E006-P2", "40 acceleration (fitted) → game max acceleration"),
                ("E006-P3", "40 top speed (fitted) → yards after catch")]),
]


def fig4() -> None:
    F = "fig4"
    tm = {r["id"]: r for r in E9["transfer_map"]}
    assert len(tm) == E8["summary"]["transfer"]["n_links"] == 18
    assert {i for _, g in LINKS for i, _ in g} == set(tm)
    n_null_b2, n_null_b1 = E8["summary"]["transfer"]["n_nulls"], E9["summary"]["b1_n_nulls"]
    assert n_null_b2 == n_null_b1 == 16
    med = num(F, E9["summary"]["draft_delta_abs_median"], ".2f", "E009.summary.draft_delta_abs_median")
    same(med, "0.01", "median draft-slot change")
    mult = E8["summary"]["multiplicity"]["E006-P1"]
    qtxt = num(F, mult["bh_q_all_round1_tests"], ".2f", "E008.summary.multiplicity.E006-P1.bh_q_all_round1_tests")
    same(qtxt, "0.06", "E006-P1 q across round 1")
    m_tests = mult["m_round1_tests"]
    w = tm["E006-P1"]
    assert w["b2_passes_1to6"] and not any(tm[i]["b2_passes_1to6"] for i in tm if i != "E006-P1")
    same(f"{w['b2_partial_r']:+.2f}", "+0.48", "E006-P1 beyond B2")
    same(f"{w['b1_partial_r']:+.2f}", "+0.42", "E006-P1 beyond B1")
    same(f"{w['b2_drop3_partial_r']:+.2f}", "+0.31", "E006-P1 without 3 WRs")

    rows, spans, y = [], [], 0.0  # rows: (link id, label, y); spans: (position, first y, last y)
    for gi, (pos, links) in enumerate(LINKS):
        if gi:
            y += 0.5
        y_first = y
        for lid, lab in links:
            rows.append((lid, lab, y))
            y += 1.0
            if lid == "E006-P1":
                rows.append(("drop3", "without its 3 most supportive receivers", y))
                y += 1.0
        spans.append((pos, y_first, y - 1.0))
    ymax = y - 0.5

    fig = new_fig(9.0)
    PRINTED.append((F, f"{n_null_b2} of {len(tm)}", "E008.summary.transfer.n_nulls (= E009.summary.b1_n_nulls) "
                                                    "of n_links"))
    top = header(fig, f"{n_null_b2} of {len(tm)} Combine-tracking links show nothing detectable, with or "
                      "without the draft slot",
                 "Partial correlation of a Combine tracking signature with the same movement, or an outcome, in "
                 "regular-season games, beyond the stopwatch (traditional Combine results) and beyond stopwatch + "
                 f"draft slot; 95% CIs. Adding the draft slot moves a link by a median of {med}.")
    ky = top - 0.2

    def tick_glyph(kax, xc):
        kax.plot([xc, xc], [-0.7, 0.7], color=INK2, lw=1.3)
    ef = set(E8["summary"]["transfer"]["corrected_beyond_b2_ge_05"])
    assert len(ef) == 6 and ef <= set(tm)  # draft: "effects of 0.5 or more remain possible for six links"
    PRINTED.append((F, "○ x 6", "E008.summary.transfer.corrected_beyond_b2_ge_05"))
    ky = key_row(fig, ky, [(k_point("o", NAVY, ms=5.6, line="-", lw=1.2), "beyond the stopwatch"),
                      (k_point("D", NAVY, ms=6.2, line="-", lw=2.4), "beyond stopwatch + draft slot"),
                           (tick_glyph, "smallest effect detectable (80% power)"),
                           (k_point("o", INK2, ms=6.0, hollow=True), "0.5 or more possible if error-free (post hoc)")],
                 gap=0.25)

    fb = footer(fig, f"† Passes our go criteria beyond the draft slot, but fragile: it fails false-discovery "
                     f"control across ~{m_tests} first-round tests (q {qtxt}). Nulls are bounds, not proof of "
                     "no effect: an interval shows the effects a link rules out. Outcome targets are over expected; "
                     "pooled rows mix positions.")
    lab_w = max(text_w(fig, lab, FS) for _, g in LINKS for _, lab in g)
    ax_x = 0.45 + lab_w + 0.12
    ax_w = W - 0.75 - ax_x
    ax_y = fb + 0.68
    ax_top = ky - 0.45
    ax = axes_in(fig, ax_x, ax_y, ax_w, ax_top - ax_y)
    ax.set_xlim(-1, 1)
    ax.set_ylim(ymax, -0.6)
    ax.axvline(0, color=INK2, lw=1.0, zorder=1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(["−1", "−0.5", "0", "+0.5", "+1"])
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel("Partial correlation (right = the direction we expected)")
    tr = ax.get_yaxis_transform()
    ax.text(1.09, -0.6, "n", transform=tr, size=FS, color=INK2, va="bottom", ha="left")
    lab_ax = axes_in(fig, 0.05, ax_y, 0.32, ax_top - ax_y, sharey=ax)
    lab_ax.set_axis_off()
    lab_ax.set_xlim(0, 1)
    for pos, y_a, y_b in spans:
        lab_ax.plot([0.88, 0.88], [y_a - 0.35, y_b + 0.35], color=BASE, lw=1.0)
        lab_ax.text(0.45, (y_a + y_b) / 2, pos, rotation=90, size=FS_P, weight="bold", color=INK, ha="center",
                    va="center")
    off = 0.17
    for lid, lab, yy in rows:
        if lid == "drop3":
            ax.text(-0.025, yy, lab, transform=tr, size=FS, color=INK2, va="center", ha="right", style="italic")
            interval(ax, w["b2_drop3_partial_r"], w["b2_drop3_ci_lo"], w["b2_drop3_ci_hi"], yy, INK2, "D",
                     lw=1.6, ms=5.6)
            PRINTED.append((F, "drawn", "E009.transfer_map[E006-P1].b2_drop3_partial_r, b2_drop3_ci_lo/hi"))
            ax.text(1.09, yy, num(F, w["b2_drop3_n"], "d", "E009.transfer_map[E006-P1].b2_drop3_n"), transform=tr,
                    size=FS, color=INK2, va="center", ha="left")
            continue
        r = tm[lid]
        sgn = r["expected_sign"]  # stored partial r are raw; sign x r puts the expected direction on the right
        ax.text(-0.025, yy, lab, transform=tr, size=FS, color=INK, va="center", ha="right")
        ax.plot([r["b2_mde80_one_sided"]] * 2, [yy - 0.4, yy + 0.4], color=INK2, lw=1.3, zorder=1.5)
        b1lo, b1hi = sorted((sgn * r["b1_ci_lo"], sgn * r["b1_ci_hi"]))
        b2lo, b2hi = sorted((sgn * r["b2_ci_lo"], sgn * r["b2_ci_hi"]))
        interval(ax, sgn * r["b1_partial_r"], b1lo, b1hi, yy - off, NAVY, "o", lw=1.2, ms=5.6)
        interval(ax, sgn * r["b2_partial_r"], b2lo, b2hi, yy + off, NAVY, "D", lw=2.4, ms=6.2)
        PRINTED.append((F, "drawn", f"E009.transfer_map[{lid}].b1_/b2_partial_r, ci, b2_mde80_one_sided"))
        if lid in ef:
            ax.plot([1.045], [yy], marker="o", ms=6.0, mfc=SURF, mec=INK2, mew=1.4, ls="none", transform=tr,
                    clip_on=False)
        ax.text(1.09, yy, num(F, r["n"], "d", f"E009.transfer_map[{lid}].n"), transform=tr, size=FS, color=INK2,
                va="center", ha="left")
    b1n = E9["summary"]["b1_nulls_partial_r_expected_direction"]
    b2n = E9["summary"]["b2_nulls_partial_r_expected_direction"]
    alt = minus(
        f"Forest plot of {len(tm)} links from Combine tracking signatures to game traits, each estimated against "
        f"two baselines. Sixteen links have intervals centred near zero ({b1n[0]:+.2f} to {b1n[1]:+.2f} beyond "
        f"the stopwatch, {b2n[0]:+.2f} to {b2n[1]:+.2f} beyond stopwatch and draft slot), and the two baselines "
        f"nearly coincide on every row. Two WR top-speed links, one signal, sit at about "
        f"{tm['E001-WR']['b2_partial_r']:+.1f} and {w['b1_partial_r']:+.2f} to {w['b2_partial_r']:+.2f}; the "
        f"stronger is marked fragile, falling to {w['b2_drop3_partial_r']:+.2f} without its three most "
        f"supportive receivers. A tick on each row marks the smallest effect detectable with 80% power; a hollow "
        f"circle marks the six links where effects of 0.5 or more remain possible if both measures were error-free "
        f"(post hoc).")
    save(fig, "fig4_transfer_map.png", alt)


# =============================================================================================================
# Figure 5: drill reliability by measurement type
# =============================================================================================================
def profile_groups() -> list[dict]:
    """E008 drill_profile rows with their comparison group, as E008_transfer_map._profile_groups defines them:
    defined rows only; tracked 40 (probe '40...' or scope '40-yd dash attempts'); DL composites that include the 40
    in neither group; everything else a position drill."""
    out = []
    for r in E8["drill_profile"]:
        rel, sh = r["rel_round1_def"], r["share_session"]
        if rel is None or sh is None or not (np.isfinite(rel) and np.isfinite(sh)):
            continue
        if r["probe"].startswith("40") or r["scope"] == "40-yd dash attempts":
            g = "40-yd dash"
        elif r["kind"] == "cross-drill incl. 40":
            g = "drill composite incl. 40"
        else:
            g = "position drill"
        out.append({**r, "group": g})
    return out


def swarm(x: np.ndarray, d_x: float, d_y: float, max_k: int = 12) -> np.ndarray:
    """Deterministic beeswarm offsets (row units): each point, in x order, takes the smallest |offset| (0, +1, -1,
    +2, ...) x d_y at which it is at least one marker diameter from every point already placed."""
    order = np.argsort(x, kind="stable")
    placed, off = [], np.zeros(len(x))
    for i in order:
        for k in [0] + [s * j for j in range(1, max_k) for s in (1, -1)]:
            oy = k * d_y
            if all(((x[i] - px) / d_x) ** 2 + ((oy - py) / d_y) ** 2 >= 1.0 for px, py in placed):
                break
        placed.append((x[i], oy))
        off[i] = oy
    return off


# The like-for-like comparison: the same DL first-step metrics (src/bdb27/probes/b_dl_burst.py), measured on the
# same players in the 40 and in the DL pass-rush drill (E008 drill_profile, probe "B DL burst (E003)").
BURST = [("d05", "yards in the first 0.5 s"), ("d10", "yards in the first 1.0 s"), ("s05", "speed at 0.5 s"),
         ("s10", "speed at 1.0 s"), ("amax06", "peak acceleration, first 0.6 s"), ("t3", "time to 3 yd/s")]


def fig5() -> None:
    F = "fig5"
    prof = profile_groups()
    bg = E8["summary"]["drill_profile"]["by_group_kind"]
    rows = [("40-yd dash", "repeat attempts", "Tracked 40", "repeat reps"),
            ("position drill", "repeat attempts", "Position drills", "repeat reps"),
            ("position drill", "cross-drill", "Position drills", "agreement across drills")]
    data = {}
    for g, k, _, _ in rows:
        sel = [r for r in prof if r["group"] == g and r["kind"] == k]
        n, n05 = len(sel), sum(r["rel_round1_def"] >= 0.5 for r in sel)
        key = bg[f"{g} | {k}"]
        assert (n, n05) == (key["n"], key["n_ge_05"]), (g, k, n, n05, key)
        data[(g, k)] = sel
    same(f"{bg['40-yd dash | repeat attempts']['n_ge_05']} of {bg['40-yd dash | repeat attempts']['n']}", "28 of 28",
         "tracked-40 metrics >= 0.5")
    same(f"{bg['position drill | repeat attempts']['n_ge_05']} of {bg['position drill | repeat attempts']['n']}",
         "2 of 22", "repeat position-drill metrics >= 0.5")
    same(f"{bg['position drill | cross-drill']['n_ge_05']} of {bg['position drill | cross-drill']['n']}", "10 of 68",
         "cross-drill composites >= 0.5")
    retest = E8["summary"]["drill_profile"]["wr_route_same_drill_retest"]["retest_icc_median"]
    same(f"{retest:.2f}", "0.70", "WR same-drill retest median")
    e10 = E10["combine"]["stop_rule"]["reliability"]
    same(f"{e10:.2f}", "0.74", "drill intensity reliability")
    e11 = {m: v["cross_drill"]["reliability"] for m, v in E11["reliability"].items()
           if isinstance(v, dict) and isinstance(v.get("cross_drill"), dict) and v["cross_drill"].get("reliability")
           is not None}
    assert max(e11, key=e11.get) == "catch_offset"
    e11v = e11["catch_offset"]
    same(f"{e11v:.2f}", "0.42", "best catch mechanic reliability")
    wave = {r["metric"].split()[1]: r["share_session"] for r in prof
            if r["probe"].startswith("C OL") and r["metric"].split()[0] == "wave"
            and r["metric"].split()[1] in ("rev_accel", "peak_decel", "peak_reaccel", "exit_speed")}
    assert len(wave) == 4
    same(f"{min(wave.values()):.0%}–{max(wave.values()):.0%}", "55%–63%", "OL wave session share range")
    # like-for-like pairs: same metric, same players, 40 vs pass-rush drill
    dl = {(r["scope"], r["metric"]): r["rel_round1_def"] for r in E8["drill_profile"]
          if r["probe"] == "B DL burst (E003)" and r["kind"] == "repeat attempts"}
    pairs = [(m, lab, dl[("40-yd dash attempts", m)], dl[("pass-rush drill attempts", m)]) for m, lab in BURST]
    r40 = [p[2] for p in pairs]
    rpr = [p[3] for p in pairs]
    same(f"{min(r40):.2f}–{max(r40):.2f}", "0.66–0.79", "first-step metrics in the 40")
    same(f"{min(rpr):.2f}–{max(rpr):.2f}", "0.23–0.51", "first-step metrics in the DL pass-rush drill")
    pr_ids = {("pass-rush drill attempts", m) for m, _ in BURST}
    dln = {(r["scope"], r["metric"]): r["n_players"] for r in E8["drill_profile"]
           if r["probe"] == "B DL burst (E003)" and r["kind"] == "repeat attempts"}
    n40s = {dln[("40-yd dash attempts", m)] for m, _ in BURST}
    nprs = {dln[("pass-rush drill attempts", m)] for m, _ in BURST}
    assert len(n40s) == len(nprs) == 1  # one n per drill across the six metrics
    n40 = num(F, n40s.pop(), "d", "E008.drill_profile[B DL burst, 40-yd dash attempts].n_players")
    npr = num(F, nprs.pop(), "d", "E008.drill_profile[B DL burst, pass-rush drill attempts].n_players")
    ge = [m for m, _, _, a in pairs if a >= 0.5]
    assert ge == ["s10"]  # the 1 of 6 pass-rush metrics at 0.5 (draft: "mostly not")
    nrange = {}
    for g, k in (("40-yd dash", "repeat attempts"), ("position drill", "repeat attempts"),
                 ("position drill", "cross-drill")):
        ns = [r["n_players"] for r in prof if r["group"] == g and r["kind"] == k]
        nrange[(g, k)] = (f"{num(F, min(ns), 'd', f'E008.drill_profile[{g} | {k}].n_players min')}–"
                          f"{num(F, max(ns), 'd', f'E008.drill_profile[{g} | {k}].n_players max')}")

    fig = new_fig(8.2)
    top = header(fig, "The tracked 40 is a trait; position-drill signatures mostly are not",
                 "Reliability of each Combine tracking metric (0 = noise, 1 = perfectly repeatable). Each row is a "
                 "different kind of reliability, so compare within a row. Most position drills are run once per "
                 "player, so their signatures can only be checked across different drills.")
    fb = footer(fig, "Reliability as used in the round-1 tests (Spearman–Brown; 0 when negative); n = players per "
                     "metric. No per-metric intervals were stored except the catch mechanic’s (post hoc bootstrap, "
                     "95%), so counts against 0.5 are approximate. Not shown: two DL composites that include the 40, "
                     "and OL within-one-rep metrics, which overstate a trait (notebook). Same-session retests are "
                     "inflated by session effects.")
    ky = top - 0.2
    pre = "Position-drill repeat reps:"
    ftext(fig, 0.1, ky, pre, size=FS, color=INK, va="center")
    ky = key_row(fig, ky, [(k_point("o", NAVY, ms=5.6), "DL pass-rush (the same movement each rep)"),
                           (k_point("o", GREY, ms=5.6), "DB reaction or OL opposite side")], gap=0.2,
                 x0=0.1 + text_w(fig, pre, FS) + 0.05)
    # ---- left: strip chart
    ax_x, ax_w = 2.2, 2.75
    ax_y = fb + 0.6
    ax_top = ky - 0.55
    ax = axes_in(fig, ax_x, ax_y, ax_w, ax_top - ax_y)
    yc = {0: 0.0, 1: 1.45, 2: 2.95, "e10": 4.15, "e11": 4.95, "wr": 5.9}
    ymax = 6.35
    ax.set_xlim(-0.04, 1.02)
    ax.set_ylim(ymax, -0.85)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("Reliability")
    ax.axvline(0.5, color=INK2, lw=1.1, zorder=1)
    ax.text(0.5, -0.82, "trait threshold", ha="center", va="bottom", size=FS_N, color=INK2)
    ms = 5.6
    d_x = (ms / 72) / ax_w * 1.06 * 1.05
    d_y = (ms / 72) / (ax_top - ax_y) * (ymax + 0.85) * 1.05
    tr = ax.get_yaxis_transform()

    def row_label(yy, l1, l2, bold=True, l3=None):
        ax.text(-0.04, yy - 0.18, l1, transform=tr, ha="right", va="center", size=FS, color=INK,
                weight="bold" if bold else "normal")
        ax.text(-0.04, yy + 0.2, l2, transform=tr, ha="right", va="center", size=FS_N, color=INK2)
        if l3:
            ax.text(-0.04, yy + 0.55, l3, transform=tr, ha="right", va="center", size=FS_N, color=INK2)

    for i, (g, k, lab1, lab2) in enumerate(rows):
        sel = data[(g, k)]
        x = np.array([r["rel_round1_def"] for r in sel])
        oy = swarm(x, d_x, d_y)
        same_move = np.array([(r["scope"], r["metric"]) in pr_ids for r in sel]) if i == 1 else np.ones(len(sel), bool)
        if i == 1:
            assert same_move.sum() == 6 and (~same_move).sum() == 16
            ax.plot(x[~same_move], yc[i] + oy[~same_move], ls="none", marker="o", ms=ms, mfc=GREY, mec=SURF,
                    mew=0.6, zorder=3)
        ax.plot(x[same_move], yc[i] + oy[same_move], ls="none", marker="o", ms=ms, mfc=NAVY, mec=SURF, mew=0.6,
                zorder=3)
        n05 = int((x >= 0.5).sum())
        row_label(yc[i], lab1, lab2, l3=f"n {nrange[(g, k)]}")
        ax.text(1.04, yc[i], f"{n05} / {len(sel)}", transform=tr, ha="left", va="center", size=FS, color=INK)
        PRINTED.append((F, f"{n05} / {len(sel)}", f"E008.summary.drill_profile.by_group_kind[{g} | {k}]"))
    ax.text(1.04, -0.82, "≥ 0.5", transform=tr, ha="left", va="bottom", size=FS_N, color=INK2)
    # single values: two later tests (not in the counts) and the WR same-drill retest median
    singles = [("e10", e10, "E010.combine.stop_rule.reliability", "Drill intensity", "later test; no game link"),
               ("e11", e11v, "E011.reliability.catch_offset.cross_drill.reliability", "Best catch mechanic",
                "later test; 2025 balls"),
               ("wr", retest, "E008.summary.drill_profile.wr_route_same_drill_retest.retest_icc_median",
                "WR route drills", "same drill and session")]
    ci11 = E11["reliability"]["catch_offset"]["cross_drill_boot_ci95"]  # post hoc bootstrap (E011 revision)
    ax.plot(ci11, [yc["e11"]] * 2, color=NAVY, lw=1.3, zorder=3, solid_capstyle="butt")
    PRINTED.append((F, "drawn", "E011.reliability.catch_offset.cross_drill_boot_ci95"))
    for key, v, src, l1, l2 in singles:
        ax.plot([v], [yc[key]], marker="o", ms=6.4, mfc=NAVY, mec=SURF, mew=0.8, ls="none", zorder=4)
        xl = ci11[1] if key == "e11" else v
        ax.text(xl + 0.035, yc[key], ("median " if key == "wr" else "") + num(F, v, ".2f", src), ha="left",
                va="center", size=FS, color=INK)
        row_label(yc[key], l1, l2, bold=(key == "wr"))

    # ---- right top: like-for-like first-step metrics
    rx = 5.75
    rw = W - 0.2 - rx
    y_t = ky - 0.3
    ftext(fig, rx, y_t, "Same first-step metrics:\n40 vs DL pass-rush drill", size=FS_P, color=INK,
          weight="bold", va="top", linespacing=1.2)
    kx = axes_in(fig, rx, y_t - 0.68, rw, 0.2)
    kx.set_axis_off()
    kx.set_xlim(0, rw)
    kx.set_ylim(-1, 1)
    kx.plot([0.07], [0], marker="s", ms=6.2, color=NAVY, mec=SURF, ls="none")
    kx.text(0.17, 0, f"in the 40 (n {n40})", size=FS_N, color=INK2, va="center")
    x2 = 0.17 + text_w(fig, f"in the 40 (n {n40})", FS_N) + 0.2
    kx.plot([x2], [0], marker="o", ms=6.2, color=NAVY, mec=SURF, ls="none")
    kx.text(x2 + 0.1, 0, f"drill (n {npr})", size=FS_N, color=INK2, va="center")
    # right column, anchored from the bottom: session-effect bars just above the footer, the dumbbell above them
    s_h = 1.0
    s_bot = fb + 0.2
    s_top = s_bot + s_h + 0.05  # top of the bars; their title and subtitle sit above
    d_top = y_t - 0.85
    d_bot = s_top + 0.62 + 0.55  # session title top + room for the dumbbell's tick labels
    d_h = d_top - d_bot
    assert d_h > 1.9, d_h
    dax = axes_in(fig, rx, d_bot, rw, d_h)
    dax.set_xlim(0, 1)
    dax.set_ylim(len(pairs) - 0.4, -0.75)
    dax.set_yticks([])
    dax.spines["left"].set_visible(False)
    dax.set_xticks([0, 0.5, 1])
    dax.set_xticklabels(["0", "0.5", "1"])
    dax.grid(axis="x", color=GRID, lw=0.8)
    dax.set_axisbelow(True)
    dax.axvline(0.5, color=INK2, lw=1.1, zorder=1)
    for j, (m, lab, a40, apr) in enumerate(pairs):
        dax.text(0.0, j - 0.45, lab, size=FS_N, color=INK2, ha="left", va="center", zorder=5,
                 bbox=dict(fc=SURF, ec="none", pad=0.4))
        dax.plot([apr, a40], [j, j], color=NAVY, lw=1.2, zorder=2)
        dax.plot([a40], [j], marker="s", ms=6.2, color=NAVY, mec=SURF, mew=0.8, ls="none", zorder=3)
        dax.plot([apr], [j], marker="o", ms=6.4, color=NAVY, mec=SURF, mew=0.8, ls="none", zorder=3)
        if m in ge:
            dax.text(apr - 0.05, j, "1 of 6 at 0.5", ha="right", va="center", size=FS_N, color=INK, zorder=5,
                     bbox=dict(fc=SURF, ec="none", pad=0.3))
        PRINTED.append((F, "drawn", f"E008.drill_profile[B DL burst (E003), {m}, 40-yd dash / pass-rush drill]"
                                    ".rel_round1_def"))

    # ---- right bottom: session effect (OL wave drill)
    ftext(fig, rx, s_top + 0.62, "Session effect", size=FS_P, color=INK, weight="bold", va="top")
    ftext(fig, rx, s_top + 0.4, "OL wave drill: share of variance\nbetween Combine sessions", size=FS_N,
          color=INK2, va="top", linespacing=1.25)
    iax = axes_in(fig, rx + 0.95, s_bot, rw - 1.35, s_h)
    names = [("rev_accel", "reversal accel."), ("peak_decel", "peak braking"), ("peak_reaccel", "re-acceleration"),
             ("exit_speed", "exit speed")]
    for j, (m, nm) in enumerate(names):
        v = wave[m]
        iax.barh(j, v, height=0.62, color=INK2, lw=0)
        iax.text(v + 0.04, j, num(F, v, ".0%", f"E008.drill_profile[OL wave {m}].share_session"), va="center",
                 ha="left", size=FS_N, color=INK)
        iax.text(-0.05, j, nm, va="center", ha="right", size=FS_N, color=INK2)
    iax.set_xlim(0, 1)
    iax.set_ylim(3.5, -0.5)
    iax.set_yticks([])
    iax.set_xticks([])
    iax.spines["left"].set_visible(False)
    iax.spines["bottom"].set_visible(False)
    iax.axvline(0, color=BASE, lw=0.8)
    f40 = E8["summary"]["drill_profile"]["forty"]["top_speed"]["rel"]
    alt = (f"Strip chart of reliability for Combine tracking metrics by measurement type. All 28 tracked-40 metrics "
           f"sit above 0.5, top speed near {f40[0]:.2f} to {f40[1]:.2f}. Only 2 of 22 position-drill metrics with "
           f"repeat reps and 10 of 68 cross-drill composites reach 0.5; 16 of the 22 repeat reps are reaction or "
           f"opposite-side reps, different movements. The same six DL first-step metrics reach {min(r40):.2f} to "
           f"{max(r40):.2f} in the 40 but {min(rpr):.2f} to {max(rpr):.2f} in the pass-rush drill. Drill intensity "
           f"reaches {e10:.2f}; the best ball-tracking catch mechanic reaches {e11v:.2f}; WR route drills repeat at a "
           f"median {retest:.2f} within one session. Inset: {min(wave.values()):.0%} to {max(wave.values()):.0%} of "
           f"OL wave-drill variance lies between Combine sessions.")
    save(fig, "fig5_drill_reliability.png", alt)


FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()
    for name, fn in FIGS.items():
        if args.only and name not in args.only:
            continue
        fn(**({"gif": not args.no_gif} if name == "fig1" else {}))
    print("\nPrinted numbers (figure, text, source key):")
    for f, s, src in PRINTED:
        print(f"  {f}  {s:>12}  {src}")


if __name__ == "__main__":
    sys.exit(main())
