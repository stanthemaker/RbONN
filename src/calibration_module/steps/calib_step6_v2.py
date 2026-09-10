"""Step 6 v2 -- per-pair TPA efficiency eta from the DIFFERENCE estimator.

Needs real hardware for the sweep; the refit path is offline::

    python src/calibration_module/steps/calib_step6_v2.py            # sweep, fit, plot
    python src/calibration_module/steps/calib_step6_v2.py --meas     # raw CSV only, no fit
    python src/calibration_module/steps/calib_step6_v2.py some.csv   # re-fit a CSV offline
    python src/calibration_module/steps/calib_step6_v2.py some.csv --anchor   # + D(0) in the fit

The underlying physics is unchanged from v1::

    Y(x, w) = eta^2*(x*w) + a_x*x + q_x*x^2 + a_w*w + q_w*w^2 + d

What changes is how eta is *extracted*.

The estimator itself -- how the difference is formed, the GLS covariance that
ties the two fits together, the four verification checks and where the
uncertainties come from -- lives with the code it describes, in
:mod:`calibration_module.fit.pair_v2`.  This file is the runner: it loads the
Step-3 calibration, drives the schedule through
:mod:`calibration_module.measure.pair_v2`, then fits, reports, plots and saves.
Everything tunable is the block of constants below -- the fit's half in
``CONFIG``, the acquisition's in ``ACQ``.  Both are plain values the GUI builds
too, so the two front ends run the same estimator over the same schedule.

Acquisition order
-----------------
Repeats are interleaved: the schedule runs round-robin passes over the grid
rather than n back-to-back reads of one level, so a slow drift cannot
masquerade as a slope.  Within each pass levels run brightest first, so
``(1, 1.0)`` is the very first acquisition and a dead or blocked beam shows
immediately.

The CSV keeps v1's column layout, so the same data can be re-fit with v1's
joint 6-parameter estimator for comparison::

    python src/calibration_module/steps/calib_step6_v1.py <a_v2_meas.csv>

That is the cross-check that justifies the change: same rows, two estimators.

Output is one ``calib_step6v2_result_MMDD_HHMM.json`` embedding the input
Step-3 calibration, and carrying ``a_x, q_x, a_w, q_w, d`` and ``eta`` per pair
(a q that was not fitted is written out as an exact zero)
in the schema step 7 already reads (``PairModel.from_json_channel``).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration_module.fit.pair_v2 import (  # noqa: E402
    PairV2Config,
    PairV2Fit,
    average_levels,
    fit_pair,
    load_meas_csv,
    report,
    save_combined_json,
    save_plot,
    write_meas_csv,
)
from calibration_module.fit.sigma import STD_FLOOR_V, floor_std  # noqa: E402,F401
from calibration_module.measure.bench import connect_daq, connect_slm  # noqa: E402
from calibration_module.measure.pair_v2 import (  # noqa: E402
    PairV2Acq,
    build_schedule,
    measure_pair,
    run_seconds,
)
from slm_module.calibration.calibration_new import load_calibration_result  # noqa: E402
from slm_module.encoding import channel_layout_from_calibration  # noqa: E402

# ---- Edit these to match your setup ----
CALIB_PATH = REPO_ROOT / "src/calib_data"   # data directory: inputs + outputs live here

PAIR_INDICES = [2, 3, 4, 5, 6]              # pair labels to calibrate
IN_STEP3 = CALIB_PATH / "calib_step3c_0907_1358_pad10.json"   # Step 3 calib

# The estimator's setup.  Every field has the validated default (see
# PairV2Config); name one here only to change it.  Keep pair_index_base in step
# with calib_step7_v2.PAIR_INDEX_BASE -- step 7 reads this script's pair labels
# straight out of the step-6 JSON.
CONFIG = PairV2Config(
    pair_index_base=1,                      # pairs are numbered 1..N
    encoding_method="fit",
    fit_q=False,
    verify_enabled=True,
)

# Acquisition timing and input range.  See PairV2Acq for what each one costs.
ACQ = PairV2Acq(
    t_single_s=10.0,
    t_both_s=8.0,
    settle_s=0.25,
    range_v=0.1,
    range_wide_v=0.2,
    autorange=True,
    invert=True,
)

SLM_DISPLAY_NO = None            # None -> auto-detect the LCOS-SLM display
USB_SLM_NO = 1                   # SLM_Ctrl_* device index for the DVI-mode switch

DAQ_DEVICE = "Dev1"
DAQ_CHANNEL = "ai0"


# ======================================================================
# inputs
# ======================================================================

def _load_layout():
    """Load the Step-3 calibration -> channel layout, validating PAIR_INDICES.

    The Step-3b/3c rows ARE the channels, so the layout is loaded verbatim (the
    same ``channel_layout_from_calibration`` the GUI encoding page uses) -- no
    re-tiling, so pair indices here mean the same thing as in the UI.

    Step 3 records only a wavelength per channel, never an index -- a channel's
    identity IS its position in that list -- so the pair label is positional
    too: pair ``i`` is the ``CONFIG.slot(i)``-th channel of the Step-3
    calibration.
    """
    if not IN_STEP3.is_file():
        raise FileNotFoundError(
            f"Step-3 calibration not found: {IN_STEP3}\n"
            f"(CALIB_PATH is the calib_data directory; IN_STEP3 is the JSON in it.)"
        )
    layout = channel_layout_from_calibration(
        load_calibration_result(IN_STEP3), method=CONFIG.encoding_method
    )
    for pi in PAIR_INDICES:
        if not (0 <= CONFIG.slot(pi) < layout.n_channels):
            raise ValueError(
                f"pair index {pi} out of range (layout has {layout.n_channels} "
                f"pairs, numbered from {CONFIG.pair_index_base})"
            )
    return layout


def _pair_wavelengths(layout, index: int) -> tuple[float, float, float]:
    """Wavelengths of a pair's two channels, or NaN if the layout lacks that slot.

    ``index`` is a pair LABEL; ``CONFIG.slot`` maps it into the Step-3 layout.
    A CSV may carry pair labels the loaded layout does not cover (a synthetic
    file, or a layout from a different run); the wavelengths are cosmetic --
    only the JSON/plot labels use them -- so fall back to NaN rather than
    killing the fit.
    """
    nan = float("nan")
    slot = CONFIG.slot(index)
    if not (0 <= slot < min(len(layout.x_channels), len(layout.w_channels))):
        return nan, nan, nan
    x_ch = layout.x_channels[slot]
    w_ch = layout.w_channels[slot]
    return (float(x_ch.wavelength_nm), float(w_ch.wavelength_nm),
            0.5 * (float(x_ch.wavelength_nm) + float(w_ch.wavelength_nm)))


# ======================================================================
# the run
# ======================================================================

def _run_sweep(fit_after: bool, *, include_anchor: bool = False) -> None:
    """Drive every pair's interleaved schedule; optionally fit, plot and save."""
    layout = _load_layout()
    schedule = build_schedule(CONFIG)
    secs = run_seconds(schedule, ACQ)
    n_verify = len(CONFIG.verify_grid) if CONFIG.verify_enabled else 0
    print(f"Step 6 v2: {len(schedule)} acquisitions/pair over "
          f"{len(CONFIG.full_grid())} levels ({len(CONFIG.grid)} estimator + "
          f"{n_verify} verification), interleaved, brightest first "
          f"(~{secs/60:.1f} min/pair)")
    print(f"Pairs: {list(PAIR_INDICES)}  "
          f"(~{secs * len(PAIR_INDICES) / 60:.0f} min total)")
    # The aperture is not a knob here -- it comes from the Step-3 file, which
    # records the window it swept -- but it belongs in run.log all the same, so
    # a result can be read back knowing what was lit.
    print(f"Layout: {layout.channel_width_px} px window + "
          f"{layout.pitch_px - layout.channel_width_px} px pad "
          f"(pitch {layout.pitch_px} px), encoding {CONFIG.encoding_method!r}")

    slm = connect_slm(SLM_DISPLAY_NO, USB_SLM_NO)
    daq = connect_daq(device=DAQ_DEVICE, channel=DAQ_CHANNEL,
                      t_both=ACQ.t_both_s, t_single=ACQ.t_single_s,
                      min_val=-ACQ.range_v, max_val=ACQ.range_v)
    rows_by_pair: dict[int, list] = {}
    total = len(schedule) * len(PAIR_INDICES)
    try:
        for n, index in enumerate(PAIR_INDICES):
            print(f"\n=== Sweep: pair {index} ===")
            rows_by_pair[index] = measure_pair(
                daq, slm, layout, index, schedule,
                cfg=CONFIG, acq=ACQ,
                step0=n * len(schedule), total=total,
                progress_callback=lambda p: print(p.line()),
                log=print,
            )
    finally:
        slm.close_slm()
        daq.disconnect()

    stamp = time.strftime("%m%d_%H%M")
    csv_path = write_meas_csv(rows_by_pair,
                              CALIB_PATH / f"calib_step6v2_meas_{stamp}.csv")
    n_rows = sum(len(v) for v in rows_by_pair.values())
    print(f"\nSaved {n_rows} rows to {csv_path}")  # raw rows on disk BEFORE fitting
    if not fit_after:
        return
    _fit_and_save(rows_by_pair, stamp, layout=layout, include_anchor=include_anchor)


def _fit_and_save(rows_by_pair: dict[int, list], stamp: str, *, layout=None,
                  include_anchor: bool = False) -> None:
    """Fit every pair, print the report, write the combined JSON and the plots."""
    fits: list[PairV2Fit] = []
    for index in sorted(rows_by_pair):
        levels = average_levels(rows_by_pair[index])
        try:
            fit = fit_pair(index, levels, CONFIG, include_anchor=include_anchor)
        except (ValueError, np.linalg.LinAlgError) as exc:
            print(f"\n=== pair {index} ===\n  fit FAILED: {exc}")
            continue
        if layout is not None:
            fit.wl_x_nm, fit.wl_w_nm, fit.nominal_wl_nm = _pair_wavelengths(layout, index)
        fits.append(fit)
        print(f"\n=== pair {index} ===")
        report(fit)

    if not fits:
        print("\nNo pair fitted -- nothing saved.")
        return

    center_wl = float(getattr(layout, "center_wl", 0.0)) if layout is not None else 0.0
    json_path = CALIB_PATH / f"calib_step6v2_result_{stamp}.json"
    save_combined_json(fits, json_path, step3=IN_STEP3, center_wl=center_wl)
    print(f"\nSaved Step-3 calib + Step-6 v2 fits -> {json_path}")
    for fit in fits:
        plot_path = json_path.with_name(f"calib_step6v2_pair{fit.index}_{stamp}.png")
        save_plot(fit, plot_path)
        print(f"Plot saved to {plot_path}")


def fit_csv(path: str | Path, *, include_anchor: bool = False) -> None:
    """Re-fit an already-recorded v2 CSV offline (no hardware).

    Writes the same combined JSON and per-pair PNGs as the hardware run, under
    a fresh timestamp so a refit never clobbers an earlier result.  The Step-3
    layout is loaded when available (for the wavelength columns) but is not
    required -- the fit itself needs only the CSV.
    """
    rows_by_pair = load_meas_csv(path)
    n = sum(len(v) for v in rows_by_pair.values())
    print(f"Loaded {path}: {len(rows_by_pair)} pair(s), {n} acquisitions")
    if include_anchor:
        print("Anchor: D(0) included as a fitted point (intercept pinned on data).")
    try:
        layout = _load_layout()
    except (FileNotFoundError, ValueError) as exc:
        print(f"(layout unavailable, wavelengths left as NaN: {exc})")
        layout = None
    _fit_and_save(rows_by_pair, time.strftime("%m%d_%H%M"),
                  layout=layout, include_anchor=include_anchor)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    flags = {"--meas", "-m", "--anchor"}
    include_anchor = "--anchor" in argv
    positional = [a for a in argv if a not in flags]
    if positional:                      # a CSV path -> offline re-fit, no hardware
        fit_csv(positional[0], include_anchor=include_anchor)
        return 0
    _run_sweep(fit_after=not any(a in ("--meas", "-m") for a in argv),
               include_anchor=include_anchor)
    return 0


if __name__ == "__main__":
    sys.exit(main())
