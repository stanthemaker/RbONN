"""Long-term monitor: DAQ filtered mean + 2400 resistance + OSA spectrum every 15 minutes.

Each cycle takes the same DAQ reading as ``python src/drafts/daq_read_waveform.py``
(sign inversion, digital low-pass, settle-guard drop, trace std), one
resistance reading from the Keithley 2400 over GPIB, and one OSA sweep with the
live-viewer parameters (778 nm center, 8 nm span, HIGH3, 1001 points, 8 uW ref,
linear W).

Per cycle it appends one row to a master CSV under ``src/calib_data``::

    timestamp, elapsed_min, daq_mean_mV, daq_std_mV, ohm_Ohm,
    osa_peak_wl_nm, osa_peak_uW, spectrum_csv

and saves the full OSA spectrum to its own CSV inside a run folder next to the
master file.  Read the run back with ``plot_osa_monitor.py``.

The cycle loop, CSV logging and per-instrument failure isolation live in
:mod:`monitor_lib`: DAQ, 2400 and OSA are measured independently, so if one
fails that cycle its columns are left blank and the loop keeps going.  The OSA
and 2400 connections are opened fresh every cycle so a dropped link never kills
an overnight run, and the 2400 output is ON only for the ~1 s of its reading, so
the DUT is not driven (or self-heated) between cycles.  Set ``K2400_RESOURCE``
to ``None`` to run without the 2400.

Needs real hardware.  Runs until Ctrl+C::

    python src/drafts/daq_osa_monitor.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[1]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from osa_module.controller import MeasurementSettings, OSAController, TraceData  # noqa: E402

from keithley_2400_read_ohm import fmt_ohm  # noqa: E402
from monitor_lib import (  # noqa: E402
    OUT_DIR,
    Probe,
    RunLog,
    columns_for,
    daq_measure,
    read_ohm_2400,
    run_monitor,
)

PREFIX = "daq_osa_monitor"
INTERVAL_S = 15 * 60

# DAQ: analog input carrying the TIA signal.  Every other acquisition knob
# (rate, window, range, cutoffs, inversion) comes from daq_read_waveform.py.
DAQ_CHANNEL = "ai0"

# Keithley 2400 (see keithley_2400_read_ohm.py for the knobs); None skips it.
K2400_RESOURCE: str | None = "GPIB0::21::INSTR"
K2400_FOUR_WIRE = False
K2400_NPLC = 10.0                    # slowest integration = quietest reading (~0.2 s)
K2400_CURRENT: float | None = None   # None -> auto ohms; else source this many amps
K2400_COMPLIANCE = 21.0              # manual ohms only: voltage limit (V)
K2400_TIMEOUT_S = 10.0

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


def read_daq() -> tuple[list, str]:
    """Filtered mean and its trace std on ``DAQ_CHANNEL``, in mV."""
    (mean_v, std_v), = daq_measure([DAQ_CHANNEL])
    mean_mv = round(abs(mean_v) * 1000.0, 6)
    std_mv = round(std_v * 1000.0, 6)
    return [mean_mv, std_mv], f"daq mean={mean_mv:.4f} mV std={std_mv:.4f} mV"


def read_ohm() -> tuple[list, str]:
    """One 2400 resistance reading, or a blank cell when the 2400 is disabled."""
    if K2400_RESOURCE is None:
        return [""], ""
    ohm = read_ohm_2400(
        K2400_RESOURCE,
        four_wire=K2400_FOUR_WIRE,
        nplc=K2400_NPLC,
        current=K2400_CURRENT,
        compliance=K2400_COMPLIANCE,
        timeout_s=K2400_TIMEOUT_S,
    )
    return [ohm], f"R={fmt_ohm(ohm)}"


def read_osa(spectra_dir: Path) -> tuple[list, str]:
    """One sweep: peak wavelength / peak power, and the full trace to its own CSV."""
    with OSAController(host=OSA_HOST, port=OSA_PORT) as osa:
        trace: TraceData = osa.measure(OSA_SETTINGS, averages=OSA_AVERAGES)
    peak = int(np.argmax(trace.powers))
    peak_wl_nm = float(trace.wavelengths_nm[peak])
    peak_uw = float(trace.powers[peak]) * 1e6  # linear W -> uW
    stamp = datetime.now().strftime("%m%d_%H%M%S")
    spec_csv = spectra_dir / f"spec_{stamp}.csv"
    dup = 1
    while spec_csv.exists():  # the name is second-granular: never clobber a sweep
        spec_csv = spectra_dir / f"spec_{stamp}_{dup}.csv"
        dup += 1
    trace.to_csv(spec_csv)
    return (
        [peak_wl_nm, peak_uw, spec_csv.name],
        f"osa peak={peak_uw:.1f} uW @ {peak_wl_nm:.4f} nm",
    )


def main() -> None:
    ohm_src = "no 2400" if K2400_RESOURCE is None else f"2400 @ {K2400_RESOURCE}"
    probes = [
        Probe(f"DAQ {DAQ_CHANNEL}", ("daq_mean_mV", "daq_std_mV"), read_daq),
        Probe(ohm_src, ("ohm_Ohm",), read_ohm),
        # bound late: the run folder only exists once RunLog has made it
        Probe("OSA", ("osa_peak_wl_nm", "osa_peak_uW", "spectrum_csv"),
              lambda: read_osa(log.run_dir)),
    ]
    log = RunLog.create(PREFIX, columns_for(probes), out_dir=OUT_DIR, with_run_dir=True)
    run_monitor(probes, interval_s=INTERVAL_S, log=log,
                extra_banner=[f"spectra  -> {log.run_dir}"])


if __name__ == "__main__":
    main()
