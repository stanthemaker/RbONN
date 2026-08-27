"""Long-term monitor: signal + laser power (one DAQ acquisition) + PMT temperature.

Sibling of ``daq_osa_monitor.py`` with the OSA dropped and a second analog
input added, to separate *our* drift from the laser's:

* ``ai0`` -- the signal (TIA output, as always)
* ``ai1`` -- the laser power monitor (second TIA, same range and inversion)
* Keithley 2400 ohms -- the PMT's thermistor, i.e. PMT temperature

Both analog inputs are read in a **single** finite acquisition, so they share a
window and their per-cycle ratio ``signal / laser`` is meaningful: anything
left in that ratio is not the laser.  (The board multiplexes, so the two
channels are microseconds apart -- nothing at a 20 Hz cutoff.)  Every other DAQ
knob -- 1 kS/s, 10 s window, +/-0.1 V, sign inversion, 20 Hz digital low-pass,
settle-guard drop, ``n_eff`` SEM -- is inherited from ``daq_read_waveform.py``,
so this reads exactly what that diagnostic plots.

Per cycle it appends one row to a master CSV under ``src/calib_data``::

    timestamp, elapsed_min, sig_mean_mV, sig_sem_mV,
    laser_mean_mV, laser_sem_mV, pmt_ohm_Ohm

The thermistor is logged in raw ohms, not degrees: convert in analysis once the
part is known (``th10k_celsius()`` in ``plot_osa_monitor.py`` is the TH10K
curve, and is *wrong* for any other thermistor).

DAQ and 2400 are measured independently (see :mod:`monitor_lib`): if one fails
that cycle its columns are left blank and the loop keeps going.  The 2400
session is opened fresh each cycle and its output is ON only for the ~1 s of
the reading, so the thermistor is not self-heated between cycles.  Set
``K2400_RESOURCE`` to ``None`` to run without it.

Wiring note: the DAQ reads DIFF, so each input is its ``+8`` sibling's
difference -- ai0 against ai8, ai1 against ai9.  A laser-power PD landed
single-ended on AI1 vs AI GND will read noise here.

Needs real hardware.  Runs until Ctrl+C::

    python src/drafts/daq_pmt_monitor.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

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

PREFIX = "daq_pmt_monitor"
INTERVAL_S = 15 * 60

# DAQ: both channels are read in one acquisition, in this order.  Every other
# acquisition knob (rate, window, range, cutoffs, inversion) comes from
# daq_read_waveform.py -- change it there, not here.
SIGNAL_CHANNEL = "ai0"
LASER_CHANNEL = "ai1"

# Keithley 2400 on the PMT thermistor (see keithley_2400_read_ohm.py); None skips it.
K2400_RESOURCE: str | None = "GPIB0::21::INSTR"
K2400_FOUR_WIRE = False              # True if the thermistor is 4-wire (lead R matters)
K2400_NPLC = 10.0                    # slowest integration = quietest reading (~0.2 s)
K2400_CURRENT: float | None = None   # None -> auto ohms; else source this many amps
K2400_COMPLIANCE = 21.0              # manual ohms only: voltage limit (V)
K2400_TIMEOUT_S = 10.0


def read_daq() -> tuple[list, str]:
    """Signal and laser power from one acquisition, as mean +/- SEM in mV."""
    (sig_v, sig_sem_v), (las_v, las_sem_v) = daq_measure([SIGNAL_CHANNEL, LASER_CHANNEL])
    sig_mv = round(abs(sig_v) * 1000.0, 6)
    sig_sem_mv = round(sig_sem_v * 1000.0, 6)
    las_mv = round(abs(las_v) * 1000.0, 6)
    las_sem_mv = round(las_sem_v * 1000.0, 6)
    # Ratio is for the console only -- the CSV keeps raw measurements, so the
    # normalisation stays a choice made in analysis.
    ratio = f"{sig_mv / las_mv:.4f}" if las_mv else "n/a"
    return (
        [sig_mv, sig_sem_mv, las_mv, las_sem_mv],
        f"sig={sig_mv:.4f}+-{sig_sem_mv:.4f} mV  laser={las_mv:.4f}+-{las_sem_mv:.4f} mV  "
        f"sig/laser={ratio}",
    )


def read_pmt_ohm() -> tuple[list, str]:
    """One 2400 reading of the PMT thermistor, or a blank cell when disabled."""
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
    return [ohm], f"PMT R={fmt_ohm(ohm)}"


def main() -> None:
    ohm_src = "no 2400" if K2400_RESOURCE is None else f"PMT 2400 @ {K2400_RESOURCE}"
    probes = [
        Probe(
            f"DAQ {SIGNAL_CHANNEL}+{LASER_CHANNEL}",
            ("sig_mean_mV", "sig_sem_mV", "laser_mean_mV", "laser_sem_mV"),
            read_daq,
        ),
        Probe(ohm_src, ("pmt_ohm_Ohm",), read_pmt_ohm),
    ]
    log = RunLog.create(PREFIX, columns_for(probes), out_dir=OUT_DIR)
    run_monitor(probes, interval_s=INTERVAL_S, log=log)


if __name__ == "__main__":
    main()
