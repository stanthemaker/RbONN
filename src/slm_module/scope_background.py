"""780 scatter / dark-current background calibration via the scope (BG sweep).

This is the automated form of the TPA-encoder "send a pattern -> record the
scope mean" operation. It measures the incoherent background that reaches the
PMT so it can be subtracted from the real 420 nm TPA signal later:

  1. Dark / baseline: display the all-off SLM pattern and record the steady
     scope mean (dark current + whatever the all-off state scatters).
  2. Per-channel 780 scatter: for every channel on the encoder grid, turn that
     channel on in isolation and sweep its level 0->1, recording the scope mean
     at each level. Scatter is linear in intensity, so each channel yields a
     line (slope + intercept) plus the raw points.

Assuming the cold-cell 780 scatter is a good-enough approximation of the hot
scatter, the resulting per-channel lines let you predict and subtract the
background for any commanded pattern:

    B(pattern) = dark + Σ_i slope_i · level_i

Everything is in commanded-level space (0..1) and scope volts; absolute
per-channel intensity is never needed.

The result is saved as an NPZ (raw arrays), a JSON summary, and a CSV table so
it can be reloaded and used by downstream code (see ``load_background_result``
and ``predict_background``).
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scope_module.controller import MonitorSettings, ScopeController

from .controller import SLMController
from .encoding import ChannelLayout, encode_to_pattern


class BackgroundAborted(Exception):
    """Raised when a stop_event interrupts a background sweep."""


@dataclass
class BackgroundProgress:
    step: int
    total: int
    message: str
    # optional live-plot point (channel wavelength, measured scatter above dark)
    wl: float | None = None
    metric: float | None = None


ProgressCallback = Callable[["BackgroundProgress"], None]


@dataclass
class ChannelScatter:
    """One channel's 780-scatter response vs commanded level."""

    index: int
    side: str                       # 'x' or 'w'
    x_center: int
    nominal_wl_nm: float
    levels: np.ndarray = field(repr=False)
    means_v: np.ndarray = field(repr=False)     # scope mean (V) at each level
    std_v: np.ndarray = field(repr=False)        # per-level std (V), 0 if repeats=1
    slope_v: float = 0.0             # V per unit level (robust linear fit)
    intercept_v: float = 0.0         # V at level 0 (≈ dark/baseline)
    # True where the level point was kept for the fit (big jumps -> False)
    inlier_mask: np.ndarray | None = field(default=None, repr=False)


@dataclass
class BackgroundResult:
    dark_mean_v: float
    dark_std_v: float
    dark_n: int
    levels: np.ndarray
    channels: list[ChannelScatter]
    center_wl: float
    channel_width_px: int
    pitch_px: int
    nm_per_px: float
    raw_npz_path: str | None = None

    # ---- convenience for downstream use -----------------------------------
    def scatter_by_key(self) -> dict[str, ChannelScatter]:
        """Map ``"<side><index>"`` (e.g. ``"x3"``) -> its ChannelScatter."""
        return {f"{c.side}{c.index}": c for c in self.channels}


def robust_linear_fit(
    levels: np.ndarray,
    means: np.ndarray,
    *,
    floor_v: float = 5e-4,
    mad_k: float = 5.0,
    min_points: int = 2,
) -> tuple[float, float, np.ndarray]:
    """Fit means vs levels after eliminating big-jump outlier points.

    A Theil-Sen line (median of pairwise slopes) gives an outlier-resistant
    first pass; points whose residual exceeds ``mad_k`` robust sigmas (floored
    at ``floor_v`` volts so clean data is never trimmed) are dropped, and an
    ordinary least-squares line is fit to the survivors. Returns
    ``(slope, intercept, inlier_mask)``.
    """
    levels = np.asarray(levels, dtype=float)
    means = np.asarray(means, dtype=float)
    n = levels.size
    if n < 2:
        return 0.0, (float(means[0]) if n else 0.0), np.ones(n, dtype=bool)

    # Theil-Sen: median of all pairwise slopes (robust to a few bad points)
    pair_slopes = [
        (means[j] - means[i]) / (levels[j] - levels[i])
        for i in range(n) for j in range(i + 1, n)
        if levels[j] != levels[i]
    ]
    ts_slope = float(np.median(pair_slopes)) if pair_slopes else 0.0
    ts_int = float(np.median(means - ts_slope * levels))

    resid = means - (ts_slope * levels + ts_int)
    mad = float(np.median(np.abs(resid - np.median(resid))))
    thresh = max(mad_k * 1.4826 * mad, floor_v)
    mask = np.abs(resid) <= thresh
    if mask.sum() < min_points:
        # keep the min_points closest to the robust line
        order = np.argsort(np.abs(resid))
        mask = np.zeros(n, dtype=bool)
        mask[order[:min_points]] = True

    if mask.sum() >= 2:
        slope, intercept = np.polyfit(levels[mask], means[mask], 1)
        return float(slope), float(intercept), mask
    return ts_slope, ts_int, mask


def recompute_fits(result: "BackgroundResult") -> "BackgroundResult":
    """(Re)compute each channel's robust slope/intercept/inlier_mask from raw means.

    Applied after a sweep and after loading so the big-jump elimination and
    linear fit are deterministic and always run on the raw data.
    """
    for c in result.channels:
        slope, intercept, mask = robust_linear_fit(c.levels, c.means_v)
        c.slope_v = slope
        c.intercept_v = intercept
        c.inlier_mask = mask
    return result


def measure_scatter_background(
    scope: ScopeController,
    slm: SLMController,
    layout: ChannelLayout,
    monitor_settings: MonitorSettings,
    *,
    levels: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    dark_samples: int = 5,
    repeats: int = 1,
    stride: int = 1,
    sides: Sequence[str] = ("x", "w"),
    settle: float | None = None,
    capture_dir: str | Path | None = None,
    stop_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> BackgroundResult:
    """Automate dark + per-channel 780-scatter measurement on the scope.

    The scope is configured once (free-run AUTO mean readout via
    ``configure_monitor``); each measurement point displays an SLM pattern,
    waits ``settle`` seconds, and records the scope-computed mean. ``stride``
    measures only every Nth channel index per side; ``repeats`` averages several
    readings per point. When ``capture_dir`` is given, each channel's curve is
    written there incrementally (crash-safe) and a consolidated ``background.npz``
    is built once the sweep finishes.
    """
    levels_arr = np.asarray(list(levels), dtype=float)
    settle_s = float(monitor_settings.hold if settle is None else settle)
    read_timeout = max(30.0, monitor_settings.duration * 3.0 + 10.0)

    capture_path = Path(capture_dir) if capture_dir is not None else None
    if capture_path is not None:
        capture_path.mkdir(parents=True, exist_ok=True)

    slm_width, slm_height = slm.get_slm_info()
    n = layout.n_channels
    zeros = np.zeros(n)

    def _check_stop() -> None:
        if stop_event is not None and stop_event.is_set():
            raise BackgroundAborted("background sweep stopped by request")

    def _read_mean() -> float:
        """One settled scope-mean reading (volts); aborts on stop_event."""
        vals = []
        for _ in range(max(1, repeats)):
            _check_stop()
            sample = scope.monitor_cycle(
                index=0, timeout=read_timeout, stop_event=stop_event
            )
            if sample is None:
                raise BackgroundAborted("scope read aborted")
            vals.append(sample.value)
        return float(np.mean(vals))

    def _display_and_read(pattern: np.ndarray) -> tuple[float, float]:
        slm.display_array(pattern)
        time.sleep(settle_s)                 # settle after the SLM pattern change
        vals = []
        for _ in range(max(1, repeats)):
            _check_stop()
            sample = scope.monitor_cycle(
                index=0, timeout=read_timeout, stop_event=stop_event
            )
            if sample is None:
                raise BackgroundAborted("scope read aborted")
            vals.append(sample.value)
        return float(np.mean(vals)), float(np.std(vals))

    # one-time scope setup for the whole sweep
    scope.configure_monitor(monitor_settings)

    # ---- dark / baseline: all channels off --------------------------------
    _check_stop()
    bg_pattern = encode_to_pattern(zeros, zeros, layout, slm_width, slm_height)
    slm.display_array(bg_pattern)
    time.sleep(settle_s)
    dark_vals = np.array([_read_mean() for _ in range(max(1, dark_samples))])
    dark_mean = float(dark_vals.mean())
    dark_std = float(dark_vals.std())
    if progress_callback is not None:
        progress_callback(
            BackgroundProgress(
                step=0, total=1,
                message=f"dark/baseline = {dark_mean*1000:.4f} mV "
                        f"(±{dark_std*1000:.4f}, n={dark_vals.size})",
            )
        )

    # ---- per-channel scatter sweep ----------------------------------------
    indices = list(range(0, n, max(1, stride)))
    targets: list[tuple[int, str]] = []
    for side in sides:
        targets += [(i, side) for i in indices]
    # +1 for the dark step already reported
    total = len(targets) * len(levels_arr) + 1
    step = 1

    channels: list[ChannelScatter] = []
    for (i, side) in targets:
        _check_stop()
        channel = (layout.x_channels if side == "x" else layout.w_channels)[i]
        means = np.zeros(levels_arr.size)
        stds = np.zeros(levels_arr.size)
        for k, level in enumerate(levels_arr):
            x_vals = zeros.copy()
            w_vals = zeros.copy()
            if side == "x":
                x_vals[i] = float(level)
            else:
                w_vals[i] = float(level)
            pattern = encode_to_pattern(x_vals, w_vals, layout, slm_width, slm_height)
            means[k], stds[k] = _display_and_read(pattern)
            if progress_callback is not None:
                progress_callback(
                    BackgroundProgress(
                        step=step, total=total,
                        message=(
                            f"{side}[{i}] @ {channel.wavelength_nm:.3f} nm  "
                            f"level {level:.2f} -> {means[k]*1000:.4f} mV"
                        ),
                        wl=channel.wavelength_nm,
                        metric=(means[k] - dark_mean),
                    )
                )
            step += 1

        slope, intercept, mask = robust_linear_fit(levels_arr, means)
        scatter = ChannelScatter(
            index=i, side=side, x_center=channel.x_center,
            nominal_wl_nm=channel.wavelength_nm,
            levels=levels_arr.copy(), means_v=means, std_v=stds,
            slope_v=slope, intercept_v=intercept, inlier_mask=mask,
        )
        channels.append(scatter)

        if capture_path is not None:
            np.savez(
                capture_path / f"{side}{i:03d}.npz",
                levels=levels_arr, means_v=means, std_v=stds,
                slope_v=slope, intercept_v=intercept, inlier_mask=mask,
                nominal_wl_nm=channel.wavelength_nm, x_center=channel.x_center,
            )

    result = BackgroundResult(
        dark_mean_v=dark_mean, dark_std_v=dark_std, dark_n=int(dark_vals.size),
        levels=levels_arr, channels=channels,
        center_wl=layout.center_wl, channel_width_px=layout.channel_width_px,
        pitch_px=layout.pitch_px, nm_per_px=layout.nm_per_px,
    )
    if capture_path is not None and channels:
        result.raw_npz_path = save_background_npz(result, capture_path / "background.npz")
    return result


# ======================================================================
# persistence (reusable form for downstream code)
# ======================================================================

def save_background_npz(result: BackgroundResult, path: str | Path) -> str:
    """Consolidate the whole background result into one self-describing NPZ."""
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    n_ch = len(result.channels)
    n_lv = result.levels.size
    means = np.zeros((n_ch, n_lv))
    stds = np.zeros((n_ch, n_lv))
    inliers = np.ones((n_ch, n_lv), dtype=bool)
    for r, c in enumerate(result.channels):
        means[r] = c.means_v
        stds[r] = c.std_v
        if c.inlier_mask is not None:
            inliers[r] = c.inlier_mask
    np.savez_compressed(
        out,
        dark_mean_v=np.array(result.dark_mean_v),
        dark_std_v=np.array(result.dark_std_v),
        dark_n=np.array(result.dark_n),
        levels=result.levels,
        means_v=means,
        std_v=stds,
        inlier_mask=inliers,
        slope_v=np.array([c.slope_v for c in result.channels]),
        intercept_v=np.array([c.intercept_v for c in result.channels]),
        side=np.array([c.side for c in result.channels]),
        index=np.array([c.index for c in result.channels]),
        x_center=np.array([c.x_center for c in result.channels]),
        nominal_wl_nm=np.array([c.nominal_wl_nm for c in result.channels]),
        center_wl=np.array(result.center_wl),
        channel_width_px=np.array(result.channel_width_px),
        pitch_px=np.array(result.pitch_px),
        nm_per_px=np.array(result.nm_per_px),
    )
    return str(out)


def save_background_json(result: BackgroundResult, path: str | Path) -> str:
    """Human-readable JSON summary (per-channel fit + dark)."""
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dark_mean_v": result.dark_mean_v,
        "dark_std_v": result.dark_std_v,
        "dark_n": result.dark_n,
        "levels": result.levels.tolist(),
        "center_wl": result.center_wl,
        "channel_width_px": result.channel_width_px,
        "pitch_px": result.pitch_px,
        "nm_per_px": result.nm_per_px,
        "channels": [
            {
                "side": c.side,
                "index": c.index,
                "x_center": c.x_center,
                "nominal_wl_nm": c.nominal_wl_nm,
                "slope_v": c.slope_v,
                "intercept_v": c.intercept_v,
                "means_v": c.means_v.tolist(),
            }
            for c in result.channels
        ],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(out)


def write_background_csv(result: BackgroundResult, path: str | Path) -> str:
    """Per-channel table: fit + one column per swept level (volts)."""
    import csv

    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        level_cols = [f"mean_L{lv:.3f}" for lv in result.levels]
        writer.writerow(
            ["side", "index", "x_center", "nominal_wl_nm",
             "slope_v_per_level", "intercept_v"] + level_cols
        )
        for c in result.channels:
            writer.writerow(
                [c.side, c.index, c.x_center, f"{c.nominal_wl_nm:.5f}",
                 f"{c.slope_v:.6e}", f"{c.intercept_v:.6e}"]
                + [f"{v:.6e}" for v in c.means_v]
            )
    return str(out)


def load_background_result(path: str | Path) -> BackgroundResult:
    """Load a consolidated background NPZ back into a BackgroundResult."""
    data = np.load(Path(path), allow_pickle=False)
    levels = data["levels"]
    means = data["means_v"]
    stds = data["std_v"]
    slopes = data["slope_v"]
    intercepts = data["intercept_v"]
    sides = data["side"]
    indices = data["index"]
    x_centers = data["x_center"]
    wls = data["nominal_wl_nm"]
    channels = [
        ChannelScatter(
            index=int(indices[r]), side=str(sides[r]), x_center=int(x_centers[r]),
            nominal_wl_nm=float(wls[r]), levels=levels.copy(),
            means_v=means[r], std_v=stds[r],
            slope_v=float(slopes[r]), intercept_v=float(intercepts[r]),
        )
        for r in range(len(indices))
    ]
    result = BackgroundResult(
        dark_mean_v=float(data["dark_mean_v"]),
        dark_std_v=float(data["dark_std_v"]),
        dark_n=int(data["dark_n"]),
        levels=levels, channels=channels,
        center_wl=float(data["center_wl"]),
        channel_width_px=int(data["channel_width_px"]),
        pitch_px=int(data["pitch_px"]),
        nm_per_px=float(data["nm_per_px"]),
        raw_npz_path=str(Path(path).resolve()),
    )
    # re-run the big-jump elimination + linear fit on the raw means so a loaded
    # result is processed exactly like a fresh sweep
    recompute_fits(result)
    return result


def predict_background(
    result: BackgroundResult,
    x_vals: Sequence[float],
    w_vals: Sequence[float],
) -> float:
    """Predict the scope background (volts) for a commanded pattern.

    Uses the linear per-channel fit: ``B = dark + Σ_i slope_i · level_i`` over
    both sides, assuming incoherent additive 780 scatter on top of the dark
    baseline.
    """
    by_key = result.scatter_by_key()
    total = result.dark_mean_v
    for side, vals in (("x", x_vals), ("w", w_vals)):
        for i, level in enumerate(vals):
            ch = by_key.get(f"{side}{i}")
            if ch is not None:
                total += ch.slope_v * float(level)
    return float(total)
