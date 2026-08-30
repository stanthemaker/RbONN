"""Shared plumbing for the long-term hardware monitors in this folder.

Every monitor here does the same thing: once per interval, read a few
*independent* instruments, append one row to a master CSV, print a one-line
summary, and sleep to the next slot on a fixed grid.  Only the instruments
differ.  That common part lives here, so a monitor script is little more than
its constants plus a list of :class:`Probe`.

The two guarantees the loop makes, which are the whole point of running it
overnight unattended:

* **Failure isolation.**  Each probe is polled inside its own ``try``; a probe
  that raises leaves its CSV cells blank, prints why, and the cycle continues.
  One dead instrument never ends a run.
* **No cadence drift.**  The loop sleeps until ``t0 + cycle * interval``, not
  for ``interval`` after the work, so measurement duration never accumulates
  into the timebase.  A cycle that overran simply fires the next one at once.

Not runnable on its own; see ``daq_osa_monitor.py`` and ``daq_pmt_monitor.py``.
"""
from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[1]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from daq_module import NIDAQDriver, lowpass  # noqa: E402

# The DAQ acquisition parameters are *not* redefined here: they come from the
# diagnostic draft, so a monitor can never drift from the trace you eyeball
# with `python src/drafts/daq_read_waveform.py`.
from daq_read_waveform import (  # noqa: E402
    DEVICE,
    DURATION_S,
    F_CUT_DIG,
    F_CUT_HW,
    FILTER_ORDER,
    INVERT,
    MAX_VAL_V,
    MIN_VAL_V,
    SAMPLE_RATE_HZ,
    effective_f_cut,
    settle_samples,
    stats,
)
from keithley_2400_read_ohm import configure_ohms, open_2400, read_ohm  # noqa: E402

OUT_DIR = REPO_SRC / "calib_data"


# --------------------------------------------------------------------------
# instrument reads
# --------------------------------------------------------------------------
def _channel_spec(device: str, channels: Sequence[str]) -> str:
    """NI-DAQmx physical-channel list for ``NIDAQDriver.read_waveform``.

    DAQmx wants every entry in a channel list fully qualified
    (``Dev1/ai0,Dev1/ai1``), but the driver prefixes the device onto whatever
    it is given -- so only the entries after the first need qualifying here.
    """
    first, *rest = channels
    return ",".join([first, *(f"{device}/{c}" for c in rest)])


def daq_measure(
    channels: Sequence[str],
    *,
    device: str = DEVICE,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    duration_s: float = DURATION_S,
    min_val_v: float = MIN_VAL_V,
    max_val_v: float = MAX_VAL_V,
    invert: bool = INVERT,
    f_cut_hw: float = F_CUT_HW,
    f_cut_dig: float = F_CUT_DIG,
    filter_order: int = FILTER_ORDER,
) -> list[tuple[float, float]]:
    """Read one or more AI channels in a SINGLE acquisition -> ``(mean_v, std_v)`` each.

    Same processing chain as :func:`daq_read_waveform.measure` -- DIFF input,
    sign inversion, digital Butterworth low-pass, settle-guard drop, then the
    std of the retained trace -- applied per channel, but with every channel
    acquired in one finite task so they share a window.  The board
    multiplexes rather than sampling simultaneously, which puts microseconds of
    skew between channels: nothing at a 20 Hz cutoff, so a per-cycle ratio of
    two channels is meaningful.

    ``channels`` order is the order of the returned list.  DIFF pairs the
    channel with its ``+8`` sibling (ai0 with ai8, ai1 with ai9), so the
    negative leg has to be wired -- a channel left single-ended against AI GND
    reads noise here.
    """
    driver = NIDAQDriver(device=device)
    driver.connect()
    raw = driver.read_waveform(
        channel=_channel_spec(device, channels),
        sample_rate=sample_rate_hz,
        duration=duration_s,
        min_val=min_val_v,
        max_val=max_val_v,
        timeout=duration_s + 10.0,
    )
    # one channel -> (n,), several -> (n_channels, n); normalise to 2-D
    raw = np.atleast_2d(np.asarray(raw, dtype=float))
    if invert:
        raw = -raw  # TIA: -volts -> +light

    f_eff = effective_f_cut(f_cut_hw, f_cut_dig)
    n_settle = settle_samples(sample_rate_hz, f_eff)
    out: list[tuple[float, float]] = []
    for trace in raw:
        filtered = lowpass(trace, sample_rate_hz, f_cut_dig, filter_order)
        kept = filtered[n_settle:] if filtered.size > n_settle else filtered
        out.append(stats(kept))
    return out


def read_ohm_2400(
    resource: str,
    *,
    four_wire: bool = False,
    nplc: float = 10.0,
    current: float | None = None,
    compliance: float = 21.0,
    timeout_s: float = 10.0,
) -> float:
    """One Keithley 2400 resistance reading in ohms (``inf`` when over-range).

    Opens the GPIB session fresh every call, drives the output only for the
    reading, and always switches it OFF and closes again -- also when the read
    fails.  So a dropped GPIB link costs one cycle rather than the run, and the
    DUT is neither driven nor self-heated between cycles.
    """
    rm, inst, _ = open_2400(resource, timeout_s=timeout_s)
    try:
        configure_ohms(
            inst,
            four_wire=four_wire,
            nplc=nplc,
            current=current,
            compliance=compliance,
        )
        inst.write(":OUTP ON")
        try:
            return read_ohm(inst)
        finally:
            inst.write(":OUTP OFF")
    finally:
        inst.close()
        rm.close()


# --------------------------------------------------------------------------
# logging + loop
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Probe:
    """One independent instrument reading inside a monitor cycle.

    ``read`` returns ``(values, text)``: one CSV cell per entry in ``columns``,
    plus a short console summary.  Raising is fine and expected -- see
    :meth:`poll`.
    """

    name: str
    columns: tuple[str, ...]
    read: Callable[[], tuple[Sequence, str]]

    def poll(self) -> tuple[list, str]:
        """Read once, never raising: on failure return blank cells and why."""
        try:
            values, text = self.read()
            values = list(values)
            if len(values) != len(self.columns):
                raise ValueError(
                    f"returned {len(values)} values for {len(self.columns)} columns"
                )
        except Exception as exc:
            return [""] * len(self.columns), f"{self.name} FAILED ({exc})"
        return values, text


@dataclass(frozen=True)
class RunLog:
    """The master CSV for one run, plus an optional folder for bulk per-cycle files."""

    csv_path: Path
    run_dir: Path | None = None

    @classmethod
    def create(
        cls,
        prefix: str,
        columns: Sequence[str],
        *,
        out_dir: Path = OUT_DIR,
        with_run_dir: bool = False,
    ) -> "RunLog":
        """Start ``<out_dir>/<prefix>_<MMDD_HHMM>.csv`` with its header row."""
        stamp = datetime.now().strftime("%m%d_%H%M")
        csv_path = out_dir / f"{prefix}_{stamp}.csv"
        run_dir = out_dir / f"{prefix}_{stamp}" if with_run_dir else None
        (run_dir or out_dir).mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            csv.writer(f).writerow(list(columns))
        return cls(csv_path, run_dir)

    def append(self, row: Sequence) -> None:
        """Append one row, closing the file again.

        Reopened per row on purpose: the log is complete on disk after every
        cycle, so a crash or a power cut costs at most the cycle in flight and
        the file can be plotted while the run is still going.
        """
        with self.csv_path.open("a", newline="") as f:
            csv.writer(f).writerow(list(row))


def columns_for(probes: Sequence[Probe]) -> list[str]:
    """Full header row: the common timebase, then every probe's columns in order."""
    return ["timestamp", "elapsed_min", *(c for p in probes for c in p.columns)]


def run_monitor(
    probes: Sequence[Probe],
    *,
    interval_s: float,
    log: RunLog,
    extra_banner: Sequence[str] = (),
) -> None:
    """Poll ``probes`` on a fixed grid until Ctrl+C, one CSV row per cycle."""
    print(
        f"measuring every {interval_s / 60:g} min until Ctrl+C  "
        f"({' + '.join(p.name for p in probes)})"
    )
    print(f"log      -> {log.csv_path}")
    for line in extra_banner:
        print(line)

    t0 = time.monotonic()
    cycle = 0
    try:
        while True:
            now = datetime.now()
            elapsed_min = (time.monotonic() - t0) / 60.0

            cells: list = []
            texts: list[str] = []
            for probe in probes:
                values, text = probe.poll()
                cells.extend(values)
                if text:
                    texts.append(text)

            log.append([now.isoformat(timespec="seconds"), f"{elapsed_min:.2f}", *cells])

            cycle += 1
            print(f"[{now:%m-%d %H:%M:%S}] cycle {cycle:3d}  " + "  |  ".join(texts))

            # Sleep to the next slot on the interval grid, so the cadence never
            # drifts with measurement duration; a cycle that overran fires the
            # next one immediately.
            next_due = t0 + cycle * interval_s
            time.sleep(max(0.0, next_due - time.monotonic()))
    except KeyboardInterrupt:
        print(f"\nstopped after {cycle} cycles; log -> {log.csv_path}")


__all__ = [
    "OUT_DIR",
    "Probe",
    "RunLog",
    "columns_for",
    "daq_measure",
    "read_ohm_2400",
    "run_monitor",
]
