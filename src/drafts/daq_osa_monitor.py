"""Long-term monitor: DAQ filtered mean + OSA spectrum every 15 minutes.

Each cycle takes the same DAQ reading as ``python src/drafts/daq_read_waveform.py``
(via :func:`daq_read_waveform.measure`: sign inversion, digital low-pass,
settle-guard drop, ``n_eff`` SEM) and one OSA sweep with the live-viewer
parameters (778 nm center, 8 nm span, HIGH2, 1001 points, 10 uW ref, linear W).

Per cycle it appends one row to a master CSV under ``src/calib_data``::

    timestamp, elapsed_min, daq_mean_mV, daq_sem_mV,
    osa_peak_wl_nm, osa_peak_uW, spectrum_csv

and saves the full OSA spectrum to its own CSV inside a run folder next to the
master file.  DAQ and OSA are measured independently: if one fails that cycle
its columns are left blank and the loop keeps going.  The OSA connection is
opened fresh every cycle so a dropped TCP link never kills an overnight run.

Needs real hardware.  Runs until Ctrl+C::

    python src/drafts/daq_osa_monitor.py
"""
from __future__ import annotations

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[1]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from osa_module.controller import MeasurementSettings, OSAController, TraceData  # noqa: E402

from daq_read_waveform import measure as daq_measure  # noqa: E402

INTERVAL_S = 15 * 60

OSA_HOST = "192.168.1.11"
OSA_PORT = 10001
OSA_AVERAGES = 1
# Mirror the GUI live-viewer sweep parameters (osa_monitor page defaults).
OSA_SETTINGS = MeasurementSettings(
    center_wl="778nm",
    span="8nm",
    sensitivity="HIGH3",
    sampling_points="1001",
    y_unit="LINear",
    reference_level="8uW",
)

OUT_DIR = Path(__file__).resolve().parents[1] / "calib_data"


def read_osa(spectra_dir: Path) -> tuple[float, float, Path]:
    """One sweep -> (peak wavelength nm, peak power uW, saved spectrum path)."""
    with OSAController(host=OSA_HOST, port=OSA_PORT) as osa:
        trace: TraceData = osa.measure(OSA_SETTINGS, averages=OSA_AVERAGES)
    peak = int(np.argmax(trace.powers))
    peak_wl_nm = float(trace.wavelengths_nm[peak])
    peak_uw = float(trace.powers[peak]) * 1e6  # linear W -> uW
    spec_csv = spectra_dir / f"spec_{datetime.now():%m%d_%H%M%S}.csv"
    trace.to_csv(spec_csv)
    return peak_wl_nm, peak_uw, spec_csv


def main() -> None:
    stamp = datetime.now().strftime("%m%d_%H%M")
    out_csv = OUT_DIR / f"daq_osa_monitor_{stamp}.csv"
    spectra_dir = OUT_DIR / f"daq_osa_monitor_{stamp}"
    spectra_dir.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="") as f:
        csv.writer(f).writerow(
            [
                "timestamp",
                "elapsed_min",
                "daq_mean_mV",
                "daq_sem_mV",
                "osa_peak_wl_nm",
                "osa_peak_uW",
                "spectrum_csv",
            ]
        )

    print(f"measuring every {INTERVAL_S / 60:g} min until Ctrl+C")
    print(f"log      -> {out_csv}")
    print(f"spectra  -> {spectra_dir}")

    t0 = time.monotonic()
    cycle = 0
    try:
        while True:
            now = datetime.now()
            elapsed_min = (time.monotonic() - t0) / 60.0

            daq_mean_mv: float | str = ""
            daq_sem_mv: float | str = ""
            try:
                mean_v, sem_v = daq_measure()
                daq_mean_mv = round(abs(mean_v) * 1000.0, 6)
                daq_sem_mv = round(sem_v * 1000.0, 6)
                daq_txt = f"daq mean={daq_mean_mv:.4f} mV sem={daq_sem_mv:.4f} mV"
            except Exception as exc:
                daq_txt = f"daq FAILED ({exc})"

            peak_wl: float | str = ""
            peak_uw: float | str = ""
            spec_name = ""
            try:
                peak_wl, peak_uw, spec_csv = read_osa(spectra_dir)
                spec_name = spec_csv.name
                osa_txt = f"osa peak={peak_uw:.1f} uW @ {peak_wl:.4f} nm"
            except Exception as exc:
                osa_txt = f"osa FAILED ({exc})"

            with out_csv.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        now.isoformat(timespec="seconds"),
                        f"{elapsed_min:.2f}",
                        daq_mean_mv,
                        daq_sem_mv,
                        peak_wl,
                        peak_uw,
                        spec_name,
                    ]
                )

            cycle += 1
            print(f"[{now:%m-%d %H:%M:%S}] cycle {cycle:3d}  {daq_txt}  |  {osa_txt}")

            # sleep to the next slot on the 15-min grid (never drifts with
            # measurement duration; if a cycle overran, fire immediately)
            next_due = t0 + cycle * INTERVAL_S
            time.sleep(max(0.0, next_due - time.monotonic()))
    except KeyboardInterrupt:
        print(f"\nstopped after {cycle} cycles; log -> {out_csv}")


if __name__ == "__main__":
    main()
