"""Step-6 pair-grid sweep driver (SLM + monitor).

LEGACY -- kept only so the current GUI page and pipeline stage keep running.

This is the in-app sweep driver: it talks to BOTH the SLM and the monitor, so
it is orchestration, not physics.  The offline step scripts under ``calibration_module/steps`` do
not use it -- they compose the SLM and DAQ calls themselves, which is the
pattern the rebuilt calibration pipeline should follow.  Nothing new should
import from here; delete this module once GUI Step 6 / ``pipeline._run_pair_eta`` is rebuilt.

The physics lives in :mod:`calibration_module.pair`.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .pair import (
    ChannelPairGrid,
    TPAPairResult,
    build_pair_points,
    fit_grid,
)


class TPAPairAborted(Exception):
    """Raised when a stop_event interrupts a pair-grid sweep."""


@dataclass
class TPAPairProgress:
    step: int
    total: int
    message: str
    pair_index: int | None = None
    eta: float | None = None          # filled in once a pair's grid is fit


ProgressCallback = Callable[["TPAPairProgress"], None]


def _read_mean_std(
    monitor, timeout: float = 30.0, single: bool = False
) -> tuple[float, float]:
    """One averaged reading and its trace std -> ``(mean, std)``.

    ``std`` is the (low-passed) trace spread the monitor reports and is what the
    fit weights by; monitors that don't report one fall back to the raw-waveform
    std on ``last_values``.  ``single`` marks a weak point -- at most one beam on
    (``x == 0 or w == 0``, including the all-off dark): the DAQ reads it over its
    longer T_single window (``single_duration``); the scope ignores the flag.
    """
    sample = monitor.monitor_cycle(timeout=timeout, single=single)
    if sample is None:
        raise TPAPairAborted("monitor read aborted")
    mean_v = float(sample.value)
    std = getattr(sample, "std", None)
    waveform = getattr(monitor, "last_values", None)
    std_v = (
        float(std) if std is not None and np.isfinite(std)
        else (float(np.std(waveform)) if waveform is not None and np.size(waveform) > 1
              else 0.0)
    )
    return mean_v, std_v


def measure_pair_grids(
    monitor,
    slm,
    layout,
    *,
    pair_indices: Sequence[int],
    sweep: Sequence[float],
    points: Sequence[tuple[float, float]] | None = None,
    settle: float = 0.15,
    read_timeout: float = 30.0,
    col_ratio: np.ndarray | None = None,
    stop_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TPAPairResult:
    """Sweep each requested pair's (x, w) points once and fit eta for each.

    ``monitor`` must already be configured (the caller runs
    ``configure_monitor``); this only calls ``monitor_cycle``.  Each ``(x, w)``
    point is measured once (all other channels held off) -- the DAQ's long
    fixed windows already average enough per point -- then the pair's data is
    fit to the TPA model.  Points with at most one beam on (``x == 0 or
    w == 0``, including the all-off dark) are read over the DAQ's longer
    T_single window; both-beams points use T_both.  By default the points
    are the full outer-product grid of ``sweep`` x ``sweep``; pass ``points`` to
    measure an explicit list instead (e.g. the reduced 1-D curves from
    :func:`build_pair_points`), in which case ``sweep`` is only recorded as the
    ramp on the result.  ``settle`` seconds are waited after every pattern change
    before reading.  ``col_ratio`` is the per-column encoding shape forwarded to
    :func:`encode_to_pattern` so the calibration is measured with the same
    channel taper that will be deployed (``None`` = flat band).  Raises
    :class:`TPAPairAborted` if ``stop_event`` is set.
    """
    sweep_arr = np.asarray(list(sweep), dtype=float)
    if points is not None:
        grid_pts = [(float(x), float(w)) for x, w in points]
    else:
        grid_pts = [(float(x), float(w)) for x in sweep_arr for w in sweep_arr]
    indices = list(pair_indices)
    n = layout.n_channels
    zeros = np.zeros(n)

    slm_width, slm_height = slm.get_slm_info()

    from slm_module.encoding import encode_to_pattern

    def _check_stop() -> None:
        if stop_event is not None and stop_event.is_set():
            raise TPAPairAborted("pair-grid sweep stopped by request")

    # accumulate raw rows per pair across all trials
    rows: dict[int, list[tuple[int, float, float, float, float, float]]] = {
        i: [] for i in indices
    }

    total = max(len(indices) * len(grid_pts), 1)
    step = 0
    for i in indices:
        _check_stop()
        x_ch = layout.x_channels[i]
        w_ch = layout.w_channels[i]
        wl_pair = 0.5 * (x_ch.wavelength_nm + w_ch.wavelength_nm)
        for x_val, w_val in grid_pts:
            _check_stop()
            x_vals = zeros.copy()
            w_vals = zeros.copy()
            x_vals[i] = x_val
            w_vals[i] = w_val
            pattern = encode_to_pattern(x_vals, w_vals, layout, slm_width,
                                        slm_height, col_ratio=col_ratio)
            slm.display_array(pattern)
            if settle:
                time.sleep(settle)
            single = x_val == 0.0 or w_val == 0.0       # at most one beam on
            mean_v, std_v = _read_mean_std(monitor, read_timeout, single=single)
            rows[i].append((0, x_val, w_val, mean_v, std_v))
            step += 1
            if progress_callback is not None:
                progress_callback(
                    TPAPairProgress(
                        step=step, total=total,
                        message=(
                            f"pair[{i}] @ {wl_pair:.2f} nm "
                            f"x={x_val:.2f} w={w_val:.2f} "
                            f"-> {mean_v*1000:.4f} mV"
                        ),
                        pair_index=i,
                    )
                )

    channels: list[ChannelPairGrid] = []
    for i in indices:
        x_ch = layout.x_channels[i]
        w_ch = layout.w_channels[i]
        data = rows[i]
        grid = ChannelPairGrid(
            index=i,
            wl_x_nm=float(x_ch.wavelength_nm),
            wl_w_nm=float(w_ch.wavelength_nm),
            nominal_wl_nm=0.5 * (x_ch.wavelength_nm + w_ch.wavelength_nm),
            x_center_x=int(x_ch.x_center),
            x_center_w=int(w_ch.x_center),
            trial=np.array([r[0] for r in data], dtype=int),
            x=np.array([r[1] for r in data], dtype=float),
            w=np.array([r[2] for r in data], dtype=float),
            voltage_mean_v=np.array([r[3] for r in data], dtype=float),
            voltage_std_v=np.array([r[4] for r in data], dtype=float),
        )
        fit_grid(grid)
        channels.append(grid)
        if progress_callback is not None and grid.fit is not None:
            progress_callback(
                TPAPairProgress(
                    step=total, total=total,
                    message=f"pair[{i}] fit: eta = {grid.fit.eta:.4g}",
                    pair_index=i, eta=grid.fit.eta,
                )
            )

    return TPAPairResult(
        sweep=sweep_arr, n_trials=1, channels=channels,
        center_wl=float(getattr(layout, "center_wl", 0.0)),
    )


__all__ = [
    "TPAPairAborted",
    "TPAPairProgress",
    "ProgressCallback",
    "measure_pair_grids",
]
