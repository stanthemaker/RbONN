"""Per-channel-pair 420 nm TPA efficiency calibration via the scope.

This is the quadratic sibling of :mod:`scope_background`. Where the background
sweep turns each channel on in isolation to measure the *linear* 780 scatter,
this sweep turns on each channel *pair* (x[i] together with w[i]) and measures
the *quadratic* 420 nm two-photon-absorption (TPA) signal that survives once the
780 scatter + PMT dark background is removed.

Model (commanded-level space, 0..1, scope volts)::

    S(pattern) = B(pattern) + Σ_i a_i · (x_i · w_i)²
               └ dark + 780 scatter ┘   └ 420 TPA (quadratic) ┘

Procedure, for every channel pair i on the encoder grid:

  1. Sweep the commanded product ``u = x_i · w_i`` from 0→1 by driving both
     sides of the pair equally (``x_i = w_i = √u``) so the product lands on the
     requested value, and record the scope mean at each step.
  2. Subtract the 780 scatter + dark background predicted from a previously
     measured :class:`~slm_module.scope_background.BackgroundResult`
     (``B = dark + slope_x·x_i + slope_w·w_i``).
  3. Fit the background-removed signal to ``a_i · u²`` (quadratic through the
     origin, big-jump outliers rejected) to get the per-channel TPA efficiency
     coefficient ``a_i``.

The result is saved as an NPZ (raw arrays), a JSON summary, and a CSV table so
it can be reloaded and re-fit downstream (see ``load_tpa_result``).
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
from .scope_background import BackgroundResult, predict_background


class TPAAborted(Exception):
    """Raised when a stop_event interrupts a TPA sweep."""


@dataclass
class TPAProgress:
    step: int
    total: int
    message: str
    # optional live-plot point (pair wavelength, background-removed TPA signal)
    wl: float | None = None
    metric: float | None = None


ProgressCallback = Callable[["TPAProgress"], None]


@dataclass
class ChannelTPA:
    """One channel pair's 420 TPA response vs commanded product x·w."""

    index: int
    x_center_x: int                 # x-channel column centre
    x_center_w: int                 # w-channel column centre
    wl_x_nm: float
    wl_w_nm: float
    nominal_wl_nm: float            # mean of the pair, for plotting
    products: np.ndarray = field(repr=False)     # commanded u = x·w (0..1)
    x_levels: np.ndarray = field(repr=False)     # x_i commanded per point (√u)
    w_levels: np.ndarray = field(repr=False)     # w_i commanded per point (√u)
    means_v: np.ndarray = field(repr=False)      # raw scope mean (V) at each u
    std_v: np.ndarray = field(repr=False)        # per-point std (V), 0 if repeats=1
    bg_v: np.ndarray = field(repr=False)         # predicted 780+dark background (V)
    signal_v: np.ndarray = field(repr=False)     # means_v - bg_v (420 TPA, V)
    coeff_a: float = 0.0            # V per unit product² (quadratic fit)
    # True where the point was kept for the fit (big jumps -> False)
    inlier_mask: np.ndarray | None = field(default=None, repr=False)


@dataclass
class TPAResult:
    levels: np.ndarray              # the commanded product levels swept (0..1)
    channels: list[ChannelTPA]
    center_wl: float
    channel_width_px: int
    pitch_px: int
    nm_per_px: float
    dark_mean_v: float = 0.0        # carried from the background reference
    background_path: str | None = None
    raw_npz_path: str | None = None

    def tpa_by_index(self) -> dict[int, ChannelTPA]:
        return {c.index: c for c in self.channels}


def robust_quadratic_fit(
    products: np.ndarray,
    signals: np.ndarray,
    *,
    floor_v: float = 5e-4,
    mad_k: float = 5.0,
    min_points: int = 2,
) -> tuple[float, np.ndarray]:
    """Fit ``signal = a · u²`` (through origin) after removing big-jump points.

    A robust median of the per-point ``signal / u²`` estimates gives an
    outlier-resistant first pass; points whose residual exceeds ``mad_k`` robust
    sigmas (floored at ``floor_v`` volts so clean data is never trimmed) are
    dropped, and a least-squares-through-origin coefficient is fit to the
    survivors. Returns ``(a, inlier_mask)``.
    """
    u = np.asarray(products, dtype=float)
    s = np.asarray(signals, dtype=float)
    n = u.size
    t = u * u                       # quadratic regressor
    if n == 0:
        return 0.0, np.ones(0, dtype=bool)

    nz = t > 1e-12
    if not nz.any():
        return 0.0, np.ones(n, dtype=bool)

    # robust first pass: median of the per-point slope estimates s/t
    a_est = float(np.median(s[nz] / t[nz]))
    resid = s - a_est * t
    mad = float(np.median(np.abs(resid - np.median(resid))))
    thresh = max(mad_k * 1.4826 * mad, floor_v)
    mask = np.abs(resid) <= thresh
    if mask.sum() < min_points:
        order = np.argsort(np.abs(resid))
        mask = np.zeros(n, dtype=bool)
        mask[order[:min_points]] = True

    tt = t[mask]
    ss = s[mask]
    denom = float(np.sum(tt * tt))
    a = float(np.sum(tt * ss) / denom) if denom > 0 else a_est
    return a, mask


def recompute_fits(result: "TPAResult") -> "TPAResult":
    """(Re)compute each pair's quadratic coefficient + inlier mask from raw signal."""
    for c in result.channels:
        a, mask = robust_quadratic_fit(c.products, c.signal_v)
        c.coeff_a = a
        c.inlier_mask = mask
    return result


def measure_tpa_efficiency(
    scope: ScopeController,
    slm: SLMController,
    layout: ChannelLayout,
    monitor_settings: MonitorSettings,
    background: BackgroundResult | None,
    *,
    products: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    repeats: int = 1,
    stride: int = 1,
    settle: float | None = None,
    capture_dir: str | Path | None = None,
    col_ratio: np.ndarray | None = None,
    stop_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TPAResult:
    """Automate the per-channel-pair 420 TPA efficiency sweep on the scope.

    For each pair ``i`` the commanded product ``u = x_i·w_i`` is swept over
    ``products`` by driving both sides equally (``x_i = w_i = √u``). The 780
    scatter + dark background from ``background`` is subtracted from each reading
    and the survivor points are fit to ``a_i·u²``. ``stride`` measures only every
    Nth pair; ``repeats`` averages several readings per point. ``col_ratio`` is
    the per-column encoding shape forwarded to :func:`encode_to_pattern` so the
    efficiency is measured with the deployed channel taper (``None`` = flat band).
    When ``capture_dir`` is given each pair's curve is written there incrementally
    (crash-safe) plus a consolidated ``tpa.npz`` once the sweep finishes.
    """
    products_arr = np.asarray(list(products), dtype=float)
    settle_s = float(monitor_settings.hold if settle is None else settle)
    read_timeout = max(30.0, monitor_settings.duration * 3.0 + 10.0)

    capture_path = Path(capture_dir) if capture_dir is not None else None
    if capture_path is not None:
        capture_path.mkdir(parents=True, exist_ok=True)

    slm_width, slm_height = slm.get_slm_info()
    n = layout.n_channels
    zeros = np.zeros(n)
    dark_v = float(background.dark_mean_v) if background is not None else 0.0

    def _check_stop() -> None:
        if stop_event is not None and stop_event.is_set():
            raise TPAAborted("TPA sweep stopped by request")

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
                raise TPAAborted("scope read aborted")
            vals.append(sample.value)
        return float(np.mean(vals)), float(np.std(vals))

    # one-time scope setup for the whole sweep
    scope.configure_monitor(monitor_settings)

    indices = list(range(0, n, max(1, stride)))
    total = len(indices) * products_arr.size
    step = 0

    channels: list[ChannelTPA] = []
    for i in indices:
        _check_stop()
        x_ch = layout.x_channels[i]
        w_ch = layout.w_channels[i]
        wl_pair = 0.5 * (x_ch.wavelength_nm + w_ch.wavelength_nm)

        means = np.zeros(products_arr.size)
        stds = np.zeros(products_arr.size)
        bgs = np.zeros(products_arr.size)
        x_lv = np.sqrt(np.clip(products_arr, 0.0, 1.0))   # x_i = w_i = √u
        w_lv = x_lv.copy()
        for k, u in enumerate(products_arr):
            x_vals = zeros.copy()
            w_vals = zeros.copy()
            x_vals[i] = float(x_lv[k])
            w_vals[i] = float(w_lv[k])
            pattern = encode_to_pattern(x_vals, w_vals, layout, slm_width,
                                        slm_height, col_ratio=col_ratio)
            means[k], stds[k] = _display_and_read(pattern)
            bgs[k] = (
                predict_background(background, x_vals, w_vals)
                if background is not None else 0.0
            )
            signal_k = means[k] - bgs[k]
            step += 1
            if progress_callback is not None:
                progress_callback(
                    TPAProgress(
                        step=step, total=total,
                        message=(
                            f"pair[{i}] @ {wl_pair:.3f} nm  x·w={u:.2f} "
                            f"-> {means[k]*1000:.4f} mV (TPA {signal_k*1000:.4f} mV)"
                        ),
                        wl=wl_pair, metric=signal_k,
                    )
                )

        signal = means - bgs
        coeff_a, mask = robust_quadratic_fit(products_arr, signal)
        pair = ChannelTPA(
            index=i, x_center_x=x_ch.x_center, x_center_w=w_ch.x_center,
            wl_x_nm=x_ch.wavelength_nm, wl_w_nm=w_ch.wavelength_nm,
            nominal_wl_nm=wl_pair, products=products_arr.copy(),
            x_levels=x_lv, w_levels=w_lv, means_v=means, std_v=stds,
            bg_v=bgs, signal_v=signal, coeff_a=coeff_a, inlier_mask=mask,
        )
        channels.append(pair)

        if capture_path is not None:
            np.savez(
                capture_path / f"pair{i:03d}.npz",
                products=products_arr, means_v=means, std_v=stds, bg_v=bgs,
                signal_v=signal, coeff_a=coeff_a, inlier_mask=mask,
                nominal_wl_nm=wl_pair,
            )

    result = TPAResult(
        levels=products_arr, channels=channels,
        center_wl=layout.center_wl, channel_width_px=layout.channel_width_px,
        pitch_px=layout.pitch_px, nm_per_px=layout.nm_per_px,
        dark_mean_v=dark_v,
        background_path=(background.raw_npz_path if background is not None else None),
    )
    if capture_path is not None and channels:
        result.raw_npz_path = save_tpa_npz(result, capture_path / "tpa.npz")
    return result


# ======================================================================
# persistence (reusable form for downstream code)
# ======================================================================

def save_tpa_npz(result: TPAResult, path: str | Path) -> str:
    """Consolidate the whole TPA result into one self-describing NPZ."""
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    n_ch = len(result.channels)
    n_lv = result.levels.size
    means = np.zeros((n_ch, n_lv))
    stds = np.zeros((n_ch, n_lv))
    bgs = np.zeros((n_ch, n_lv))
    signal = np.zeros((n_ch, n_lv))
    inliers = np.ones((n_ch, n_lv), dtype=bool)
    for r, c in enumerate(result.channels):
        means[r] = c.means_v
        stds[r] = c.std_v
        bgs[r] = c.bg_v
        signal[r] = c.signal_v
        if c.inlier_mask is not None:
            inliers[r] = c.inlier_mask
    np.savez_compressed(
        out,
        levels=result.levels,
        means_v=means,
        std_v=stds,
        bg_v=bgs,
        signal_v=signal,
        inlier_mask=inliers,
        coeff_a=np.array([c.coeff_a for c in result.channels]),
        index=np.array([c.index for c in result.channels]),
        x_center_x=np.array([c.x_center_x for c in result.channels]),
        x_center_w=np.array([c.x_center_w for c in result.channels]),
        wl_x_nm=np.array([c.wl_x_nm for c in result.channels]),
        wl_w_nm=np.array([c.wl_w_nm for c in result.channels]),
        nominal_wl_nm=np.array([c.nominal_wl_nm for c in result.channels]),
        dark_mean_v=np.array(result.dark_mean_v),
        center_wl=np.array(result.center_wl),
        channel_width_px=np.array(result.channel_width_px),
        pitch_px=np.array(result.pitch_px),
        nm_per_px=np.array(result.nm_per_px),
        background_path=np.array(result.background_path or ""),
    )
    return str(out)


def save_tpa_json(result: TPAResult, path: str | Path) -> str:
    """Human-readable JSON summary (per-pair quadratic coefficient)."""
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "levels": result.levels.tolist(),
        "dark_mean_v": result.dark_mean_v,
        "center_wl": result.center_wl,
        "channel_width_px": result.channel_width_px,
        "pitch_px": result.pitch_px,
        "nm_per_px": result.nm_per_px,
        "background_path": result.background_path,
        "channels": [
            {
                "index": c.index,
                "x_center_x": c.x_center_x,
                "x_center_w": c.x_center_w,
                "wl_x_nm": c.wl_x_nm,
                "wl_w_nm": c.wl_w_nm,
                "nominal_wl_nm": c.nominal_wl_nm,
                "coeff_a": c.coeff_a,
                "products": c.products.tolist(),
                "means_v": c.means_v.tolist(),
                "bg_v": c.bg_v.tolist(),
                "signal_v": c.signal_v.tolist(),
            }
            for c in result.channels
        ],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(out)


def write_tpa_csv(result: TPAResult, path: str | Path) -> str:
    """Per-pair table: quadratic coefficient + one signal column per product."""
    import csv

    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        prod_cols = [f"signal_u{u:.3f}" for u in result.levels]
        writer.writerow(
            ["index", "wl_x_nm", "wl_w_nm", "nominal_wl_nm",
             "coeff_a_v_per_u2"] + prod_cols
        )
        for c in result.channels:
            writer.writerow(
                [c.index, f"{c.wl_x_nm:.5f}", f"{c.wl_w_nm:.5f}",
                 f"{c.nominal_wl_nm:.5f}", f"{c.coeff_a:.6e}"]
                + [f"{v:.6e}" for v in c.signal_v]
            )
    return str(out)


def load_tpa_result(path: str | Path) -> TPAResult:
    """Load a consolidated TPA NPZ back into a TPAResult (re-fit on load)."""
    data = np.load(Path(path), allow_pickle=False)
    levels = data["levels"]
    means = data["means_v"]
    stds = data["std_v"]
    bgs = data["bg_v"]
    signal = data["signal_v"]
    coeffs = data["coeff_a"]
    indices = data["index"]
    xc_x = data["x_center_x"]
    xc_w = data["x_center_w"]
    wl_x = data["wl_x_nm"]
    wl_w = data["wl_w_nm"]
    wls = data["nominal_wl_nm"]
    x_lv = np.sqrt(np.clip(levels, 0.0, 1.0))
    channels = [
        ChannelTPA(
            index=int(indices[r]), x_center_x=int(xc_x[r]), x_center_w=int(xc_w[r]),
            wl_x_nm=float(wl_x[r]), wl_w_nm=float(wl_w[r]),
            nominal_wl_nm=float(wls[r]), products=levels.copy(),
            x_levels=x_lv.copy(), w_levels=x_lv.copy(),
            means_v=means[r], std_v=stds[r], bg_v=bgs[r], signal_v=signal[r],
            coeff_a=float(coeffs[r]),
        )
        for r in range(len(indices))
    ]
    result = TPAResult(
        levels=levels, channels=channels,
        center_wl=float(data["center_wl"]),
        channel_width_px=int(data["channel_width_px"]),
        pitch_px=int(data["pitch_px"]),
        nm_per_px=float(data["nm_per_px"]),
        dark_mean_v=float(data["dark_mean_v"]),
        background_path=(str(data["background_path"]) or None)
        if "background_path" in data else None,
        raw_npz_path=str(Path(path).resolve()),
    )
    # re-run the big-jump elimination + quadratic fit on the raw signal
    recompute_fits(result)
    return result


def predict_tpa(result: TPAResult, x_vals: Sequence[float], w_vals: Sequence[float]) -> float:
    """Predict the 420 TPA signal (volts) for a commanded pattern.

    Uses the per-pair quadratic fit: ``Σ_i a_i · (x_i·w_i)²``. This is the
    background-removed TPA contribution only (add the 780/dark background from
    the paired BackgroundResult for the full expected scope mean).
    """
    by_index = result.tpa_by_index()
    total = 0.0
    n = min(len(x_vals), len(w_vals))
    for i in range(n):
        ch = by_index.get(i)
        if ch is not None:
            u = float(x_vals[i]) * float(w_vals[i])
            total += ch.coeff_a * u * u
    return float(total)
