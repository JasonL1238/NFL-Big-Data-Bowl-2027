"""Sprint speed: the tracked Combine 40-yd dash and the in-game acceleration-speed (A-S) profile.

Conventions: speeds yd/s, times s, 10 Hz. Speed and a_tan are Savitzky-Golay from x/y (bdb27.kinematics);
never use `dis` (wrong in Combine tracking).

Combine 40 (one row per attempt; a player's value is the mean over his QC-passing attempts):
  - The raw Combine frame is already the track (verified 2026-10-14 on all 793 attempts): the dash runs along
    +x from the start line at x = 10 to the finish at x = 50. Tracking starts with the player already moving
    (~0.6 yd/s, ~0.7 yd past the start line) and stops 0.1-2.8 yd past the finish. So along-track distance is
    x - START_X and the split lines are the official ones. Measured from the onset position instead, the 40-yd
    mark lies beyond the end of tracking in 28% of attempts (42% with segments.canonicalize x).
  - Splits: crossing times of the 10/20/30/40-yd lines, linearly interpolated between frames; split times and
    average speeds for 0-10, 10-20, 20-30, 30-40 and the flying 20 (20-40). The 0-10 split starts at motion
    onset (SG speed first reaches 1 yd/s, interpolated; NaN when the first frame is already faster) and covers
    the yards from the onset position to the 10-yd line. Because tracking starts after motion, 0-10 is a weak
    measure; the flying splits do not depend on the start.
  - smax: peak SG speed.
  - Mono-exponential v(t) = vmax (1 - exp(-(t - t0) / tau)), fitted from the frame of peak dv/dt (first 2 s;
    dv/dt rises for 0.3-0.8 s at the start, which a mono-exponential cannot fit) to the frame of peak speed.
    a0 = vmax / tau. Horizontal force-velocity profile of Samozino et al. (2016) with air resistance.
  - QC (E006): path 38-43 yd, >= 15 fitted frames, tau 0.3-3 s, speed drop after the peak <= 1 yd/s, fit
    RMSE <= 0.15 yd/s. A larger RMSE flags `two_phase`: a plateau, then a second rise late in the run.

Game (regular season, post-snap frames): the in-situ A-S profile (Morin et al. 2021), made exposure-invariant.
Per player, R draws of exactly K plays; per draw, the q95 of a_tan in 0.5 yd/s speed bins (3-10.5 yd/s) and a
linear envelope A = A0 - (A0 / S0) S through them. A0 is the mean over draws, so its expectation does not
depend on the number of plays. S0 is the median over draws (a heavy-tailed ratio), which makes it only nearly
invariant: with few plays the draws overlap, so it tends to the per-draw mean, with many plays to the median
(WRs, 2026-10-14: pools of exactly K plays give S0 0.03 +- 0.01 yd/s higher than pools of >= 4K, ~0.1 SD).
"""

import warnings

import numpy as np
import polars as pl
from scipy.optimize import curve_fit

from bdb27 import features, io, kinematics, reliability, segments

YD, LB, INCH = 0.9144, 0.45359237, 0.0254  # m per yd, kg per lb, m per inch

# ---------------------------------------------------------------- Combine 40

START_X = 10.0  # start line in the Combine frame; the finish line is at START_X + 40
MARKS = (10, 20, 30, 40)  # yd past the start line
SPLITS = ((0, 10), (10, 20), (20, 30), (30, 40), (20, 40))
ONSET_SPEED = 1.0
QC = dict(min_path=38.0, max_path=43.0, max_rmse=0.15, min_n_fit=15, tau_range=(0.3, 3.0), max_end_drop=1.0)
METRICS = ["smax", *(f"{p}_{a}_{b}" for p in ("split", "v") for a, b in SPLITS),
           "vmax", "tau", "a0", "F0", "V0", "Pmax", "Sfv", "F0_rel", "Pmax_rel"]


def forty_frames() -> pl.DataFrame:
    """40-yd dash frames: `traj` per attempt, SG kinematics, `t` (s since the attempt's first frame) and
    `path` (cumulative x/y path length, yd)."""
    d = segments.attempts(drill_types=["FORTY_YARD_DASH"])
    d = kinematics.add_kinematics(d, segments.ATTEMPT_KEYS).sort("traj", "time")
    return d.with_columns(
        t=(pl.col("time") - pl.col("time").min().over("traj")).dt.total_milliseconds() / 1000,
        path=(pl.col("x").diff() ** 2 + pl.col("y").diff() ** 2).sqrt().fill_null(0).cum_sum().over("traj"),
    )


def crossing_times(t: np.ndarray, d: np.ndarray, marks) -> np.ndarray:
    """First time d reaches each mark, linearly interpolated between frames (NaN if never, or already at frame 0)."""
    out = np.full(len(marks), np.nan)
    for k, m in enumerate(marks):
        i = np.flatnonzero(d >= m)
        if len(i) and i[0] > 0:
            j = i[0]
            out[k] = t[j - 1] + (m - d[j - 1]) / (d[j] - d[j - 1]) * (t[j] - t[j - 1])
    return out


def onset_time(t: np.ndarray, v: np.ndarray) -> float:
    """Interpolated time v first reaches ONSET_SPEED for 3 frames (NaN if already there at frame 0, or never)."""
    i = segments.onset_index(v, ONSET_SPEED)
    if i == 0:
        return np.nan
    return float(t[i - 1] + (ONSET_SPEED - v[i - 1]) / (v[i] - v[i - 1]) * (t[i] - t[i - 1]))


def forty_splits(t: np.ndarray, x: np.ndarray, v: np.ndarray) -> dict:
    """t_onset, split_<a>_<b> (s) and v_<a>_<b> (average yd/s) for SPLITS. x is the raw Combine x."""
    t_on = onset_time(t, v)
    times = {0: t_on, **dict(zip(MARKS, crossing_times(t, x - START_X, MARKS)))}
    d_on = np.interp(t_on, t, x) - START_X  # yd past the start line at onset (NaN if no onset)
    out = {"t_onset": t_on}
    for a, b in SPLITS:
        out[f"split_{a}_{b}"] = times[b] - times[a]
        out[f"v_{a}_{b}"] = (b - (d_on if a == 0 else a)) / out[f"split_{a}_{b}"]
    return out


def mono_exp(t, vmax, tau, t0):
    return np.where(t > t0, vmax * (1 - np.exp(-(t - t0) / tau)), 0.0)


def fit_window(v: np.ndarray, search_s: float = 2.0) -> tuple[int, int]:
    """[start, end) of the fitted part: from peak dv/dt (first `search_s`) to peak speed (inclusive)."""
    m = int(round(search_s / kinematics.DT))
    i0 = int(np.argmax(np.gradient(v, kinematics.DT)[:m]))
    i1 = int(np.argmax(v)) + 1
    return i0, max(i1, i0 + 5)


def fit_mono_exp(t: np.ndarray, v: np.ndarray) -> dict:
    """vmax, tau, t0 (same clock as t), a0 = vmax / tau and rmse over fit_window (NaN if the fit fails), n_fit."""
    i0, i1 = fit_window(v)
    tt, vv = t[i0:i1], v[i0:i1]
    try:
        p, _ = curve_fit(mono_exp, tt, vv, p0=[vv.max(), 1.0, tt[0] - 0.3],
                         bounds=([3, 0.1, tt[0] - 3], [15, 5, tt[0] + 1]), maxfev=20000)
    except (RuntimeError, ValueError):
        return dict(n_fit=len(tt), vmax=np.nan, tau=np.nan, t0=np.nan, a0=np.nan, rmse=np.nan)
    vmax, tau, t0 = (float(x) for x in p)
    rmse = float(np.sqrt(np.mean((mono_exp(tt, *p) - vv) ** 2)))
    return dict(n_fit=len(tt), vmax=vmax, tau=tau, t0=t0, a0=vmax / tau, rmse=rmse)


def air_k(mass_kg: float, height_m: float, temp_c: float = 20.0, p_mmhg: float = 760.0, cd: float = 0.9) -> float:
    """Aerodynamic friction coefficient k (kg/m), Samozino et al. (2016) / Arsac & Locatelli (2002). Indoors."""
    rho = 1.293 * (p_mmhg / 760) * (273 / (273 + temp_c))
    af = 0.2025 * height_m**0.725 * mass_kg**0.425 * 0.266
    return 0.5 * rho * af * cd


def fv_profile(vmax_yd: float, tau: float, mass_kg: float, height_m: float, duration: float) -> dict:
    """Horizontal F-V profile of the fitted curve over `duration` s from t0: F_H = m a + k v^2, linear in v.
    F0 (N), V0 (m/s), Pmax = F0 V0 / 4 (W), Sfv (N s/m), F0_rel (N/kg), Pmax_rel (W/kg). NaN if undefined."""
    nan = dict.fromkeys(("F0", "V0", "Pmax", "Sfv", "F0_rel", "Pmax_rel"), np.nan)
    if not np.isfinite([vmax_yd, tau, mass_kg, height_m, duration]).all() or duration <= 0:
        return nan
    vmax = vmax_yd * YD
    t = np.arange(0, duration + 1e-9, 0.01)
    v = vmax * (1 - np.exp(-t / tau))
    fh = mass_kg * (vmax / tau) * np.exp(-t / tau) + air_k(mass_kg, height_m) * v**2
    slope, f0 = np.polyfit(v, fh, 1)
    if slope >= 0:
        return nan
    v0 = -f0 / slope
    return dict(F0=float(f0), V0=float(v0), Pmax=float(f0 * v0 / 4), Sfv=float(slope),
                F0_rel=float(f0 / mass_kg), Pmax_rel=float(f0 * v0 / 4 / mass_kg))


def forty_attempts(frames: pl.DataFrame | None = None) -> pl.DataFrame:
    """One row per 40-yd attempt: ids, position, path, smax, splits, mono-exponential fit, F-V profile,
    `two_phase` and `qc_ok`."""
    d = forty_frames() if frames is None else frames
    rows = []
    for (traj,), g in d.group_by(["traj"], maintain_order=True):
        t, x, v = (g[c].to_numpy() for c in ("t", "x", "speed"))
        rows.append(dict(traj=traj, path=float(g["path"][-1]), dur=float(t[-1]), smax=float(v.max()),
                         end_drop=float(v.max() - v[-1]), **forty_splits(t, x, v), **fit_mono_exp(t, v)))
    ids = d.select("traj", "nfl_id", "event_id", "attempt", "draft_year").unique("traj")
    body = io.load("combine_results").select("nfl_id", position="combine_position",
                                             mass_kg=pl.col("combine_weight") * LB,
                                             height_m=pl.col("combine_height") * INCH)
    a = pl.DataFrame(rows).join(ids, on="traj").join(body, on="nfl_id", how="left")
    fv = pl.DataFrame([fv_profile(r["vmax"], r["tau"], r["mass_kg"], r["height_m"], r["dur"] - r["t0"])
                       for r in a.iter_rows(named=True)])
    lo, hi = QC["tau_range"]
    return pl.concat([a, fv], how="horizontal").with_columns(
        two_phase=pl.col("rmse").is_finite() & (pl.col("rmse") > QC["max_rmse"]),
        qc_ok=(pl.col("path").is_between(QC["min_path"], QC["max_path"]) & (pl.col("rmse") <= QC["max_rmse"])
               & (pl.col("n_fit") >= QC["min_n_fit"]) & pl.col("tau").is_between(lo, hi)
               & (pl.col("end_drop") <= QC["max_end_drop"])).fill_null(False),
    ).sort("nfl_id", "event_id", "attempt")


def combine_speed_table(attempts: pl.DataFrame | None = None) -> pl.DataFrame:
    """One row per nfl_id with >= 1 QC-passing 40 attempt: c40_<metric> = mean over those attempts
    (non-finite values skipped; null if none), and c40_n_attempts."""
    a = (forty_attempts() if attempts is None else attempts).filter(pl.col("qc_ok"))
    return (a.group_by("nfl_id")
            .agg(*(pl.col(m).fill_nan(None).mean().alias(f"c40_{m}") for m in METRICS), c40_n_attempts=pl.len())
            .sort("nfl_id"))


def attempt_icc(attempts: pl.DataFrame) -> pl.DataFrame:
    """reliability.icc1 of each metric across QC-passing attempts, after centering within Combine position x
    draft year (position differences and session/year artifacts are not player reliability)."""
    a = attempts.filter(pl.col("qc_ok"))
    rows = []
    for m in METRICS:
        v = pl.col(m).fill_nan(None)
        c = a.select("nfl_id", _c=v - v.mean().over("position", "draft_year"))
        rows.append({"metric": m, **reliability.icc1(c["_c"].to_numpy(), c["nfl_id"].to_numpy())})
    return pl.DataFrame(rows)


# ---------------------------------------------------------------- Game A-S profile

GAME_FRAMES = io.PROCESSED / "game_frames_post_snap.parquet"
MAX_A = 15.0  # yd/s^2; |a_tan| above this is a tracking glitch
AS_BINS = np.arange(3.0, 10.51, 0.5)  # yd/s bin edges; 3 yd/s ~ Morin's 3 m/s lower bound
AS_QUANTILE = 0.95
AS_MIN_FRAMES = 20  # frames per bin for the bin to enter the envelope fit
AS_MIN_BINS = 4
AS_K = 50  # plays per draw
AS_R = 20  # draws per player


def build_game_frames(overwrite: bool = False) -> pl.DataFrame:
    """Post-snap REG frames of every player with SG speed and a_tan (kinematics use the whole play, so the
    snap is not a filter edge). Plays whose max `s` exceeds features.MAX_PLAUSIBLE_SPEED are dropped.
    Cached to GAME_FRAMES."""
    if GAME_FRAMES.exists() and not overwrite:
        return pl.read_parquet(GAME_FRAMES)
    out = []
    for season in io.SEASONS:
        lf = (features.reg_season_tracking().filter(pl.col("season") == season)
              .select("game_id", "play_id", "nfl_id", "time", "x", "y", "s", "event"))
        df = segments.snap_aligned(lf).collect()
        bad = (df.group_by(segments.PLAY_KEYS).agg(smax=pl.col("s").max())
               .filter(pl.col("smax") > features.MAX_PLAUSIBLE_SPEED))
        df = df.join(bad.select(segments.PLAY_KEYS), on=segments.PLAY_KEYS, how="anti")
        df = kinematics.add_kinematics(df.select("game_id", "play_id", "nfl_id", "time", "x", "y", "s", "t_rel"),
                                       segments.PLAY_KEYS)
        out.append(df.filter(pl.col("t_rel") >= 0)
                   .select("game_id", "play_id", "nfl_id", "t_rel", "s", "speed", "a_tan",
                           season=pl.lit(season, pl.Int32)))
    frames = pl.concat(out)
    frames.write_parquet(GAME_FRAMES)
    return frames


def binned_quantiles(speed: np.ndarray, a: np.ndarray, q: float = AS_QUANTILE, bins: np.ndarray = AS_BINS,
                     min_frames: int = AS_MIN_FRAMES) -> np.ndarray:
    """q-quantile of a per speed bin (NaN for bins with < min_frames frames)."""
    idx = np.digitize(speed, bins) - 1
    out = np.full(len(bins) - 1, np.nan)
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() >= min_frames:
            out[b] = np.quantile(a[m], q)
    return out


def envelope(qa: np.ndarray, bins: np.ndarray = AS_BINS, min_bins: int = AS_MIN_BINS) -> tuple[float, float, float]:
    """Linear A-S envelope through binned quantiles at bin centers: (A0, S0, slope). S0 is NaN unless slope < 0."""
    centers = (bins[:-1] + bins[1:]) / 2
    ok = np.isfinite(qa)
    if ok.sum() < min_bins:
        return np.nan, np.nan, np.nan
    slope, a0 = np.polyfit(centers[ok], qa[ok], 1)
    return float(a0), float(-a0 / slope) if slope < 0 else np.nan, float(slope)


def as_profile(fr: pl.DataFrame, k: int = AS_K, r: int = AS_R, q: float = AS_QUANTILE, bins: np.ndarray = AS_BINS,
               min_frames: int = AS_MIN_FRAMES, seed: int = 0) -> dict:
    """A-S profile of one player's frames (game_id, play_id, speed, a_tan) from r draws of exactly k plays:
    A0, slope and binned quantiles qa = mean over draws, S0 = median over draws. NaN if < k plays."""
    plays = fr.select("game_id", "play_id").unique().sort("game_id", "play_id")
    if plays.height < k:
        return dict(A0=np.nan, S0=np.nan, slope=np.nan, qa=np.full(len(bins) - 1, np.nan), n_plays=plays.height)
    key = fr["game_id"].to_numpy() * 100000 + fr["play_id"].to_numpy()
    pkey = plays["game_id"].to_numpy() * 100000 + plays["play_id"].to_numpy()
    order = np.argsort(key, kind="stable")
    key_s, spd, acc = key[order], fr["speed"].to_numpy()[order], fr["a_tan"].to_numpy()[order]
    starts, ends = np.searchsorted(key_s, pkey, "left"), np.searchsorted(key_s, pkey, "right")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(r):
        sel = np.concatenate([np.arange(starts[i], ends[i]) for i in rng.choice(len(pkey), k, replace=False)])
        qa = binned_quantiles(spd[sel], acc[sel], q, bins, min_frames)
        draws.append((*envelope(qa, bins), qa))
    a0, s0, slope, qa = (np.array(x) for x in zip(*draws))
    with warnings.catch_warnings():  # all-NaN (no usable draw or bin) -> NaN, as intended
        warnings.simplefilter("ignore", RuntimeWarning)
        return dict(A0=float(np.nanmean(a0)), S0=float(np.nanmedian(s0)), slope=float(np.nanmean(slope)),
                    qa=np.nanmean(qa, axis=0), n_plays=plays.height)


def game_as_profiles(frames: pl.DataFrame, seed: int = 0, **kw) -> pl.DataFrame:
    """Per-player A-S profile (A0, S0, slope, n_plays, qa_<bin lower edge>) from build_game_frames output.
    Frames with implausible speed or |a_tan| > MAX_A are dropped first. kw goes to as_profile."""
    fr = frames.filter(pl.col("speed").is_between(0, features.MAX_PLAUSIBLE_SPEED) & (pl.col("a_tan").abs() <= MAX_A))
    bins = kw.get("bins", AS_BINS)
    rows = []
    for (pid,), g in fr.group_by(["nfl_id"], maintain_order=True):
        p = as_profile(g, seed=seed + int(pid), **kw)
        rows.append(dict(nfl_id=int(pid), A0=p["A0"], S0=p["S0"], slope=p["slope"], n_plays=p["n_plays"],
                         **{f"qa_{b:.1f}": float(v) for b, v in zip(bins[:-1], p["qa"])}))
    return pl.DataFrame(rows).sort("nfl_id")


def game_as_split_half(frames: pl.DataFrame, seed: int = 0, **kw) -> pl.DataFrame:
    """game_as_profiles on odd and even games (dense rank of game_id within player) side by side:
    <col>_h0, <col>_h1. Players need k plays in each half."""
    fr = frames.with_columns(half=(pl.col("game_id").rank("dense").over("nfl_id") % 2).cast(pl.Int8))
    out = []
    for h in (0, 1):
        p = game_as_profiles(fr.filter(pl.col("half") == h), seed=seed + 7919 * h, **kw)
        out.append(p.rename({c: f"{c}_h{h}" for c in p.columns if c != "nfl_id"}))
    return out[0].join(out[1], on="nfl_id", how="inner")


def split_half_reliability(sh: pl.DataFrame, col: str) -> dict:
    """Spearman-Brown corrected split-half reliability of `col` from game_as_split_half output."""
    m = sh.select(f"{col}_h0", f"{col}_h1").fill_nan(None).drop_nulls()
    if m.height < 10:
        return {"r_half": np.nan, "reliability": np.nan, "n_players": m.height}
    r = float(np.corrcoef(m[f"{col}_h0"].to_numpy(), m[f"{col}_h1"].to_numpy())[0, 1])
    return {"r_half": r, "reliability": 2 * r / (1 + r), "n_players": m.height}
