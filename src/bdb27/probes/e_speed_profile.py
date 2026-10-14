"""Probe E — sprint mechanics profile: Combine 40-yd velocity curve vs the in-game acceleration-speed profile.

Combine (per 40-yd attempt):
  Mono-exponential v(t) = vmax * (1 - exp(-(t - t0) / tau)) fitted to the Savitzky-Golay speed (the same
  smoothing the game side uses). The tracked start is not mono-exponential: dv/dt RISES for the first
  0.3-0.8 s (sensor + filter onset from a 3-point stance), so the fit runs from the frame of peak dv/dt
  (searched in the first 2 s) to the frame of peak speed (some players ease off before the line).
  Derived: a0 = vmax / tau (yd/s^2), and the horizontal force-velocity profile of Samozino et al. (2016)
  with air resistance (indoor: no wind, 20 C, 760 mmHg): F_H(t) = m a(t) + k v(t)^2, linear F_H-v fit ->
  F0 (N), V0 (m/s), Pmax = F0 V0 / 4 (W), S_FV (N s/m), and per-kg versions. Body mass from combine_weight.
  Split residual: model-predicted minus tracked time to cover the first 10 yd of the track (> 0 = the
  real start was quicker than the fitted curve implies).
  Player value = mean over QC-passing attempts.

Game (regular season, post-snap frames):
  In-situ acceleration-speed (A-S) profile (Morin et al. 2021) made exposure-invariant: for each player,
  R random draws of exactly K plays; in each draw, bin speed (yd/s) and take an upper quantile of a_tan per
  bin (not a max), fit the linear envelope A = A0 - (A0 / S0) S over the bins; A0, the slope and the binned
  quantiles are averaged over draws, S0 is the median (a ratio, heavy-tailed). Every estimate uses the same
  number of plays, so the expected value does not depend on snap count; more plays only make it less noisy.

Outcome (WR/TE): yards after catch over expected per catch (winsorized +-10 yd), empirical-Bayes shrunk to
the position mean (DerSimonian-Laird between-player variance).
"""

import warnings

import numpy as np
import polars as pl
from scipy.optimize import curve_fit

from bdb27 import features, io, kinematics, segments

YD = 0.9144  # m per yard
LB = 0.45359237  # kg per lb
INCH = 0.0254  # m per inch
G = 9.81

# ---------------------------------------------------------------- Combine 40


def forty_frames() -> pl.DataFrame:
    """40-yd dash frames with traj (one per attempt), SG kinematics and t (s since the first frame)."""
    d = segments.attempts(drill_types=["FORTY_YARD_DASH"])
    d = kinematics.add_kinematics(d, segments.ATTEMPT_KEYS).sort("traj", "time")
    return d.with_columns(
        t=(pl.col("time") - pl.col("time").min().over("traj")).dt.total_milliseconds() / 1000,
        dist=(((pl.col("x").diff().over("traj")) ** 2 + (pl.col("y").diff().over("traj")) ** 2).sqrt()
              .fill_null(0).cum_sum().over("traj")),
    )


def mono_exp(t, vmax, tau, t0):
    return np.where(t > t0, vmax * (1 - np.exp(-(t - t0) / tau)), 0.0)


def mono_exp_dist(t, vmax, tau, t0):
    """Distance covered since t0 under the mono-exponential."""
    u = np.maximum(t - t0, 0.0)
    return vmax * (u + tau * np.exp(-u / tau) - tau)


def fit_window(v: np.ndarray, search_s: float = 2.0) -> tuple[int, int]:
    """[start, end) of the fitted part: from peak dv/dt (first `search_s`) to peak speed (inclusive)."""
    m = int(round(search_s / kinematics.DT))
    i0 = int(np.argmax(np.gradient(v, kinematics.DT)[:m]))
    i1 = int(np.argmax(v)) + 1
    return i0, max(i1, i0 + 5)


def fit_attempt(t: np.ndarray, v: np.ndarray, dist: np.ndarray) -> dict:
    """Mono-exponential fit on one attempt's speed trace (yd/s). Returns NaNs if the fit fails."""
    i0, i1 = fit_window(v)
    tt, vv = t[i0:i1], v[i0:i1]
    out = dict(i_fit0=i0, i_fit1=i1, n_fit=len(tt), t_fit0=float(t[i0]), v_fit0=float(v[i0]),
               smax=float(v.max()), end_drop=float(v.max() - v[-1]), path=float(dist[-1]), dur=float(t[-1]))
    try:
        p, _ = curve_fit(mono_exp, tt, vv, p0=[vv.max(), 1.0, tt[0] - 0.3],
                         bounds=([3, 0.1, tt[0] - 3], [15, 5, tt[0] + 1]), maxfev=20000)
    except (RuntimeError, ValueError):
        return {**out, **{k: np.nan for k in ("vmax", "tau", "t0", "rmse", "split10_resid")}}
    vmax, tau, t0 = (float(x) for x in p)
    rmse = float(np.sqrt(np.mean((mono_exp(tt, *p) - vv) ** 2)))
    # Split residual: the model's time to cover the track's first 10 yd (from the first frame, with the
    # model's distance anchored at the first fitted frame) minus the tracked time.
    d_model = dist[i0] + mono_exp_dist(t, *p) - mono_exp_dist(t[i0], *p)
    t10_trk = float(np.interp(10.0, dist, t))
    t10_mod = float(np.interp(10.0, d_model, t)) if d_model.max() > 10 else np.nan
    return {**out, "vmax": vmax, "tau": tau, "t0": t0, "rmse": rmse, "split10_resid": t10_mod - t10_trk,
            "t10_trk": t10_trk}


def air_k(mass_kg: float, height_m: float, temp_c: float = 20.0, p_mmhg: float = 760.0, cd: float = 0.9) -> float:
    """Aerodynamic friction coefficient k (kg/m), Samozino et al. (2016) / Arsac & Locatelli (2002)."""
    rho = 1.293 * (p_mmhg / 760) * (273 / (273 + temp_c))
    af = 0.2025 * height_m**0.725 * mass_kg**0.425 * 0.266
    return 0.5 * rho * af * cd


def fv_profile(vmax_yd: float, tau: float, mass_kg: float, height_m: float, t_end: float, t0: float = 0.0) -> dict:
    """Horizontal force-velocity profile from the fitted speed curve (Samozino et al. 2016, with air)."""
    nan = dict(F0=np.nan, V0=np.nan, Pmax=np.nan, Sfv=np.nan, F0_rel=np.nan, Pmax_rel=np.nan)
    if not all(np.isfinite([vmax_yd, tau, mass_kg, height_m, t_end])) or t_end <= t0:
        return nan
    vmax = vmax_yd * YD
    t = np.arange(t0, t_end + 1e-9, 0.01)
    v = vmax * (1 - np.exp(-(t - t0) / tau))
    a = (vmax / tau) * np.exp(-(t - t0) / tau)
    fh = mass_kg * a + air_k(mass_kg, height_m) * v**2
    slope, f0 = np.polyfit(v, fh, 1)
    if slope >= 0:
        return nan
    v0 = -f0 / slope
    pmax = f0 * v0 / 4
    return dict(F0=float(f0), V0=float(v0), Pmax=float(pmax), Sfv=float(slope),
                F0_rel=float(f0 / mass_kg), Pmax_rel=float(pmax / mass_kg))


# QC limits for an attempt (set from the attempt distributions, before any linkage).
QC = dict(min_path=38.0, max_path=43.0, max_rmse=0.15, min_n_fit=15, tau_range=(0.3, 3.0), max_end_drop=1.0)


def combine_forty_fits(frames: pl.DataFrame | None = None) -> pl.DataFrame:
    """One row per 40-yd attempt: fit parameters, F-V profile, QC flag."""
    d = forty_frames() if frames is None else frames
    body = io.load("combine_results").select(
        "nfl_id", mass_kg=pl.col("combine_weight") * LB, height_m=pl.col("combine_height") * INCH)
    ids = d.select("traj", "nfl_id", "event_id", "attempt", "draft_year").unique("traj")
    rows = []
    for (traj,), g in d.group_by(["traj"], maintain_order=True):
        rows.append({"traj": traj, **fit_attempt(g["t"].to_numpy(), g["speed"].to_numpy(), g["dist"].to_numpy())})
    fits = pl.DataFrame(rows).join(ids, on="traj").join(body, on="nfl_id", how="left")
    fv = [fv_profile(r["vmax"], r["tau"], r["mass_kg"], r["height_m"], r["dur"] - r["t0"]) if np.isfinite(r["t0"])
          else fv_profile(np.nan, np.nan, np.nan, np.nan, np.nan) for r in fits.iter_rows(named=True)]
    # fv_profile measures time from t0; pass the span t0 -> end of track.
    fits = pl.concat([fits, pl.DataFrame(fv)], how="horizontal")
    lo, hi = QC["tau_range"]
    fits = fits.with_columns(
        a0=pl.col("vmax") / pl.col("tau"),
        qc_ok=(pl.col("path").is_between(QC["min_path"], QC["max_path"]) & (pl.col("rmse") <= QC["max_rmse"])
               & (pl.col("n_fit") >= QC["min_n_fit"]) & pl.col("tau").is_between(lo, hi)
               & (pl.col("end_drop") <= QC["max_end_drop"])).fill_null(False),
    )
    return fits.sort("nfl_id", "event_id", "attempt")


COMBINE_PARAMS = ["vmax", "tau", "a0", "t0", "F0", "F0_rel", "V0", "Pmax", "Pmax_rel", "Sfv", "split10_resid", "smax"]


def combine_profile(fits: pl.DataFrame) -> pl.DataFrame:
    """Player-level Combine sprint profile: mean of each parameter over QC-passing attempts."""
    ok = fits.filter(pl.col("qc_ok"))
    return (ok.group_by("nfl_id")
            .agg(pl.col(c).mean().alias(f"c40_{c}") for c in COMBINE_PARAMS)
            .join(ok.group_by("nfl_id").agg(c40_n_attempts=pl.len()), on="nfl_id")
            .sort("nfl_id"))


# ---------------------------------------------------------------- Game A-S profile

GAME_FRAMES = io.PROCESSED / "probe_e_game_frames.parquet"
MAX_A = 15.0  # yd/s^2; |a_tan| above this is a tracking glitch (checked on the distribution)


def build_game_frames(overwrite: bool = False) -> pl.DataFrame:
    """Post-snap REG frames of every player with SG speed and a_tan. Plays whose max `s` exceeds
    features.MAX_PLAUSIBLE_SPEED are dropped (glitches). Cached to data/processed/probe_e_game_frames.parquet."""
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
        del df
    frames = pl.concat(out)
    frames.write_parquet(GAME_FRAMES)
    return frames


# Profile settings (fixed from game-side reliability and exposure checks, before any linkage).
AS_BINS = np.arange(3.0, 10.51, 0.5)  # yd/s bin edges; 3 yd/s ~ Morin's 3 m/s lower bound
AS_QUANTILE = 0.95
AS_MIN_FRAMES = 20  # frames per bin for the bin to enter the envelope fit
AS_MIN_BINS = 4
AS_K = 50  # plays per draw
AS_R = 20  # draws per player


def _binned_quantiles(speed: np.ndarray, a: np.ndarray, q: float, bins: np.ndarray, min_frames: int) -> np.ndarray:
    idx = np.digitize(speed, bins) - 1
    out = np.full(len(bins) - 1, np.nan)
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() >= min_frames:
            out[b] = np.quantile(a[m], q)
    return out


def envelope(qa: np.ndarray, bins: np.ndarray, min_bins: int = AS_MIN_BINS) -> tuple[float, float, float]:
    """Linear A-S envelope through binned quantiles: (A0, S0, slope). S0 is NaN unless slope < 0."""
    centers = (bins[:-1] + bins[1:]) / 2
    ok = np.isfinite(qa)
    if ok.sum() < min_bins:
        return np.nan, np.nan, np.nan
    slope, a0 = np.polyfit(centers[ok], qa[ok], 1)
    s0 = -a0 / slope if slope < 0 else np.nan
    return float(a0), float(s0), float(slope)


def as_profile_draws(fr: pl.DataFrame, k: int = AS_K, r: int = AS_R, q: float = AS_QUANTILE,
                     bins: np.ndarray = AS_BINS, min_frames: int = AS_MIN_FRAMES, seed: int = 0) -> dict:
    """A-S profile of one player's frames averaged over r draws of exactly k plays (NaN if < k plays)."""
    plays = fr.select("game_id", "play_id").unique().sort("game_id", "play_id")
    nb = len(bins) - 1
    if plays.height < k:
        return dict(A0=np.nan, S0=np.nan, slope=np.nan, qa=np.full(nb, np.nan), n_plays=plays.height)
    key = fr["game_id"].to_numpy() * 100000 + fr["play_id"].to_numpy()
    pkey = plays["game_id"].to_numpy() * 100000 + plays["play_id"].to_numpy()
    order = np.argsort(key, kind="stable")
    key_s, spd, acc = key[order], fr["speed"].to_numpy()[order], fr["a_tan"].to_numpy()[order]
    starts = np.searchsorted(key_s, pkey, "left")
    ends = np.searchsorted(key_s, pkey, "right")
    rng = np.random.default_rng(seed)
    a0s, s0s, sls, qas = [], [], [], []
    for _ in range(r):
        pick = rng.choice(len(pkey), k, replace=False)
        sel = np.concatenate([np.arange(starts[i], ends[i]) for i in pick])
        qa = _binned_quantiles(spd[sel], acc[sel], q, bins, min_frames)
        a0, s0, sl = envelope(qa, bins)
        a0s.append(a0), s0s.append(s0), sls.append(sl), qas.append(qa)
    with warnings.catch_warnings():  # all-NaN bins in every draw -> NaN, which is intended
        warnings.simplefilter("ignore", RuntimeWarning)
        return dict(A0=float(np.nanmean(a0s)) if np.isfinite(a0s).any() else np.nan,
                    S0=float(np.nanmedian(s0s)) if np.isfinite(s0s).any() else np.nan,
                    slope=float(np.nanmean(sls)) if np.isfinite(sls).any() else np.nan,
                    qa=np.nanmean(np.array(qas), axis=0), n_plays=plays.height)


def game_profiles(frames: pl.DataFrame, players: list[int] | None = None, seed: int = 0, **kw) -> pl.DataFrame:
    """Per-player in-situ A-S profile (A0, S0, slope, binned quantiles) from fixed-size play draws."""
    fr = frames.filter(pl.col("speed").is_between(0, features.MAX_PLAUSIBLE_SPEED) & (pl.col("a_tan").abs() <= MAX_A))
    if players is not None:
        fr = fr.filter(pl.col("nfl_id").is_in(players))
    bins = kw.get("bins", AS_BINS)
    rows = []
    for (pid,), g in fr.group_by(["nfl_id"], maintain_order=True):
        p = as_profile_draws(g, seed=seed + int(pid), **kw)
        row = dict(nfl_id=int(pid), A0=p["A0"], S0=p["S0"], slope=p["slope"], n_plays=p["n_plays"])
        for b, v in zip(bins[:-1], p["qa"]):
            row[f"qa_{b:.1f}"] = float(v)
        rows.append(row)
    return pl.DataFrame(rows).sort("nfl_id")


def game_profiles_split_half(frames: pl.DataFrame, seed: int = 0, **kw) -> pl.DataFrame:
    """Profiles estimated separately on odd and even games (dense rank of game_id within player)."""
    fr = frames.with_columns(half=(pl.col("game_id").rank("dense").over("nfl_id") % 2).cast(pl.Int8))
    out = []
    for h in (0, 1):
        p = game_profiles(fr.filter(pl.col("half") == h), seed=seed + 7919 * h, **kw)
        out.append(p.rename({c: f"{c}_h{h}" for c in p.columns if c != "nfl_id"}))
    return out[0].join(out[1], on="nfl_id", how="inner")


def sb_split_half(sh: pl.DataFrame, col: str) -> dict:
    """Spearman-Brown corrected split-half reliability of a profile trait (from game_profiles_split_half)."""
    m = sh.select(f"{col}_h0", f"{col}_h1").drop_nans().drop_nulls()
    if m.height < 10:
        return {"r_half": np.nan, "reliability": np.nan, "n_players": m.height}
    r = float(np.corrcoef(m[f"{col}_h0"].to_numpy(), m[f"{col}_h1"].to_numpy())[0, 1])
    return {"r_half": r, "reliability": 2 * r / (1 + r), "n_players": m.height}


# ---------------------------------------------------------------- Outcome: YAC over expected


def catches() -> pl.DataFrame:
    """REG-season catches with YAC over expected (yards_after_catch - expected_yards_after_catch)."""
    reg = io.load("games").filter(pl.col("season_type") == "REG").select("game_id")
    pp = io.load("player_play").join(reg, on="game_id", how="semi")
    return (pp.filter((pl.col("pass_result") == "C") & pl.col("yards_after_catch").is_not_null()
                      & pl.col("expected_yards_after_catch").is_not_null()
                      & (pl.col("play_nullified_by_penalty").cast(pl.String) != "Y").fill_null(True))
            .select("game_id", "play_id", "nfl_id", "yards_after_catch", "expected_yards_after_catch",
                    yacoe=pl.col("yards_after_catch") - pl.col("expected_yards_after_catch")))


def eb_shrink(df: pl.DataFrame, value: str, group: str, min_n: int = 1) -> pl.DataFrame:
    """Normal-normal empirical Bayes per player: shrink the per-player mean toward the group's
    precision-weighted mean mu with weight w = tau^2 / (tau^2 + sigma^2 / n).
    sigma^2 = pooled within-player variance (per group); tau^2 = DerSimonian-Laird between-player
    variance (precision-weighted method of moments, floored at 0). The unweighted moment estimator is
    dominated by low-n players and returned tau^2 ~ 0 for WR YACOE; DL weights players by n."""
    per = (df.group_by("nfl_id", group).agg(m=pl.col(value).mean(), v=pl.col(value).var(), n=pl.len())
           .filter(pl.col("n") >= min_n))
    out = []
    for (g,), p in per.group_by([group]):
        n, m = p["n"].to_numpy().astype(float), p["m"].to_numpy()
        v = p["v"].to_numpy()
        w_ok = np.isfinite(v) & (n > 1)
        sigma2 = float(np.sum((n[w_ok] - 1) * v[w_ok]) / np.sum(n[w_ok] - 1))
        prec = n / sigma2
        mu = float(np.sum(prec * m) / np.sum(prec))
        q_stat = float(np.sum(prec * (m - mu) ** 2))
        tau2 = max(0.0, (q_stat - (len(m) - 1)) / (np.sum(prec) - np.sum(prec**2) / np.sum(prec)))
        w = tau2 / (tau2 + sigma2 / n)
        out.append(p.with_columns(shrunk=pl.Series(mu + w * (m - mu)), weight=pl.Series(w),
                                  tau2=pl.lit(tau2), sigma2=pl.lit(sigma2), group_mean=pl.lit(mu)))
    return pl.concat(out).sort("nfl_id")


YACOE_WINSOR = 10.0  # yd; per-catch YACOE clipped to +-10 (breakaway tails), fixed on reliability before linkage
YACOE_MIN_CATCHES = 10


def yacoe_outcome(positions: pl.DataFrame) -> pl.DataFrame:
    """Per WR/TE: EB-shrunk mean winsorized YACOE (yacoe_eb), raw mean, n catches, EB weight.
    `positions` has nfl_id, position. EB is fitted on all catchers; players with < YACOE_MIN_CATCHES dropped."""
    c = catches().join(positions, on="nfl_id").filter(pl.col("position").is_in(["WR", "TE"]))
    c = c.with_columns(y=pl.col("yacoe").clip(-YACOE_WINSOR, YACOE_WINSOR))
    eb = eb_shrink(c, "y", "position")
    return (eb.filter(pl.col("n") >= YACOE_MIN_CATCHES)
            .select("nfl_id", yacoe_eb="shrunk", yacoe_raw="m", yacoe_catches="n", yacoe_eb_weight="weight"))
