"""Repeat the daq_read_waveform filtered measurement N times and record the means.

Runs the exact same acquisition + processing chain as
``python src/drafts/daq_read_waveform.py`` (same module-level constants: sign
inversion, digital low-pass at ``F_CUT_DIG``, settle-guard drop, ``n_eff``
statistics) via :func:`daq_read_waveform.measure`, but repeats it ``N_RUNS``
times back to back.  Each run prints its ``filtered: mean=... mV`` line; all
runs are saved to a timestamped CSV under ``src/calib_data`` and summarized
(mean / std / spread across runs) at the end.

Needs real hardware, like the parent draft::

    python src/drafts/daq_repeat_measure.py
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from daq_read_waveform import DURATION_S, F_CUT_DIG, SAMPLE_RATE_HZ, measure

N_RUNS = 20
OUT_DIR = Path(__file__).resolve().parents[1] / "calib_data"


def main() -> None:
    stamp = datetime.now().strftime("%m%d_%H%M")
    out_csv = OUT_DIR / f"daq_repeat_measure_{stamp}.csv"

    print(
        f"{N_RUNS} runs of measure() "
        f"({DURATION_S:g} s @ {SAMPLE_RATE_HZ:g} S/s, digital {F_CUT_DIG:g} Hz)"
    )
    means_mv: list[float] = []
    rows: list[tuple[int, float, float]] = []
    for i in range(1, N_RUNS + 1):
        mean_v, sem_v = measure()
        mean_mv, sem_mv = abs(mean_v) * 1000.0, sem_v * 1000.0
        means_mv.append(mean_mv)
        rows.append((i, mean_mv, sem_mv))
        print(f"run {i:2d}/{N_RUNS}  filtered: mean={mean_mv:.4f} mV, sem={sem_mv:.4f} mV")

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "filtered_mean_mV", "filtered_sem_mV"])
        writer.writerows(rows)

    m = np.asarray(means_mv)
    print(
        f"\nsummary over {N_RUNS} runs: mean={m.mean():.4f} mV, "
        f"std={m.std(ddof=1):.4f} mV, spread={m.max() - m.min():.4f} mV "
        f"(min {m.min():.4f} / max {m.max():.4f})"
    )
    print(f"saved -> {out_csv}")


if __name__ == "__main__":
    main()
