"""Read one waveform from an NI-DAQ analog input -- diagnostic script *and* module.

Needs real hardware (no mocks), so it is not a pytest test.  The acquisition
itself lives in :mod:`daq_module` (``NIDAQDriver`` -> ``DAQController``): DIFF
input, one untriggered finite window, digital Butterworth low-pass, mean and
std of the filtered trace.  This draft drives that same path directly to
eyeball a raw trace and its amplitude spectrum -- the bring-up view the
production monitor doesn't plot.

This draft distinguishes TWO cutoffs where the module has one: ``F_CUT_HW`` is
the hardware low-pass (the detector/TIA 3 dB bandwidth -- physics, already
baked into the signal), ``F_CUT_DIG`` is the digital Butterworth applied in
software and may differ (e.g. cut below the hardware bandwidth to reject more
noise).  The narrowest cutoff in the chain (``effective_f_cut`` =
``min(F_CUT_HW, F_CUT_DIG)``) is the slowest filter, so it sets the settle
guard dropped before any statistics.

Two draft-local corrections sit on top of that shared path (neither touches
``daq_module``): every read is sign-inverted -- our TIA outputs negative volts
for positive light -- and the first ``SETTLE_CYCLES / f_cut`` seconds are
discarded after filtering as a turn-on transient before any statistics.

* Run it directly -- read once with the module defaults, print stats, plot
  time + frequency domain::

      python src/drafts/daq_read_waveform.py

* Import it from another draft and grab one read, overriding any acquisition
  parameter per call::

      from daq_read_waveform import measure, read_waveform
      mean_v, std_v = measure(duration_s=1.0)         # low-passed mean + its std
      times, voltages = read_waveform()               # raw trace, if you want it

The module-level constants mirror the :class:`daq_module.DAQMonitorSettings`
defaults (the values the step-6/7 calibration drafts validated), so this
diagnostic can never drift from what the calibrations actually use.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from daq_module import DAQMonitorSettings, NIDAQDriver, lowpass  # noqa: E402

# Defaults come from daq_module (single source of truth); DEVICE names the
# board (see NI-MAX).
_DEFAULTS = DAQMonitorSettings()
DEVICE = "Dev1"
CHANNEL = "ai0"
# CHANNEL = _DEFAULTS.channel
# Hardware vs digital cutoff: F_CUT_HW is the analog low-pass already in the
# signal chain (detector/TIA 3 dB bandwidth) -- it sets the true noise
# bandwidth of what reaches the DAQ.  F_CUT_DIG is the software Butterworth
# applied after acquisition and may be set independently (typically <= HW to
# reject more noise; > HW does nothing physical).
F_CUT_HW = _DEFAULTS.f_cut               # hardware 3 dB bandwidth (Hz)
F_CUT_HW = 150
F_CUT_DIG = 20                     # digital low-pass cutoff (Hz); override to differ
FILTER_ORDER = _DEFAULTS.filter_order    # digital Butterworth low-pass order
SAMPLE_RATE_HZ = _DEFAULTS.sample_rate
# SAMPLE_RATE_HZ = 1_000
# DURATION_S = _DEFAULTS.duration
DURATION_S = 10
# Input range is quantized: the board only offers +/-0.1, 0.2, 0.5, 1, 2, 5, 10 V
# and rounds any request UP to the next one -- +/-0.1 V is the most sensitive.
# MIN_VAL_V = _DEFAULTS.min_val
# MAX_VAL_V = _DEFAULTS.max_val
MAX_VAL_V = 0.1
MIN_VAL_V = -0.1
# Our transimpedance amplifier outputs a NEGATIVE voltage for positive light, so
# every read is inverted to recover a positive light signal (more light -> more
# positive volts).
INVERT = True
# Leading guard discarded after filtering: the raw acquisition settles and the
# zero-phase low-pass anchors its output to the first sample, so the first
# ``SETTLE_CYCLES / f_cut`` seconds are a turn-on transient, not steady state.
# The slowest (lowest-cutoff) filter dominates, so the guard is computed from
# ``effective_f_cut`` = min(hardware, digital).
SETTLE_CYCLES = 3.0


def read_waveform(
    *,
    device: str = DEVICE,
    channel: str = CHANNEL,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    duration_s: float = DURATION_S,
    min_val_v: float = MIN_VAL_V,
    max_val_v: float = MAX_VAL_V,
    invert: bool = INVERT,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """One untriggered finite acquisition via NIDAQDriver; return ``(times, voltages)``.

    Every parameter defaults to the module-level constant, so a direct run needs
    no arguments while an importer can override any of them per call.  With
    ``invert`` (default ``INVERT``) the trace is negated so the TIA's
    negative-for-light output reads as a positive light signal.
    """
    driver = NIDAQDriver(device=device)
    driver.connect()
    voltages = driver.read_waveform(
        channel=channel, sample_rate=sample_rate_hz, duration=duration_s,
        min_val=min_val_v, max_val=max_val_v, timeout=duration_s + 10.0,
    )
    if invert:
        voltages = -np.asarray(voltages, dtype=float)   # TIA: -volts -> +light
    if verbose:
        print(f"read {voltages.size} samples ({duration_s:g} s @ {sample_rate_hz:g} S/s)")
    times = np.arange(voltages.size) / sample_rate_hz
    return times, voltages


def amplitude_spectrum(v: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum ``(freqs_hz, |V|_volts)``.

    A Hann window (with coherent-gain correction) tames spectral leakage from
    the finite record; the DC bin is dropped so the mean doesn't swamp the plot
    and the noise floor / low-pass roll-off are what you see.
    """
    n = v.size
    win = np.hanning(n)
    scale = 2.0 / np.sum(win)  # coherent gain -> single-sided amplitude in volts
    spec = np.abs(np.fft.rfft(v * win)) * scale
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs[1:], spec[1:]  # drop DC bin


def stats(v: np.ndarray) -> tuple[float, float]:
    """Return ``(mean, std)`` for a trace -- the spread as measured, undivided.

    No standard error of the mean is derived: the low-passed samples are
    correlated (drift and 1/f wander dominate over a multi-second window), so
    dividing the spread by any assumed independent-sample count would report an
    uncertainty far smaller than the scatter actually seen between repeats.
    """
    return float(v.mean()), float(v.std())


def report(label: str, v: np.ndarray) -> tuple[float, float]:
    """Print mean, std and the std ratio (std/|mean|); return ``(mean, std)``."""
    mean, std = stats(v)
    print(
        f"{label:>8}: mean={abs(mean)*1000:.4f} mV, std={std*1000:.4f} mV, "
        f"std ratio={abs(std/mean)*100:.4f}%"
    )
    return mean, std


def effective_f_cut(f_cut_hw: float = F_CUT_HW, f_cut_dig: float = F_CUT_DIG) -> float:
    """Narrowest cutoff in the chain -- the bandwidth that governs statistics.

    The hardware low-pass is already in the signal; the digital one is applied
    on top.  Whichever cuts lower sets the noise bandwidth of the filtered
    trace, hence the settle guard.
    """
    return min(f_cut_hw, f_cut_dig)


def settle_samples(fs: float, f_cut: float, cycles: float = SETTLE_CYCLES) -> int:
    """Leading samples to discard as filter/detector turn-on transient.

    The raw acquisition settles over the first few detector cycles and the
    zero-phase low-pass anchors its output to the first sample, so the leading
    ``cycles / f_cut`` seconds are not steady-state signal.  Returns 0 when the
    sample rate or cutoff is non-positive.
    """
    if fs <= 0.0 or f_cut <= 0.0:
        return 0
    return int(round(cycles / f_cut * fs))


def measure(
    *,
    f_cut_hw: float = F_CUT_HW,
    f_cut_dig: float = F_CUT_DIG,
    filter_order: int = FILTER_ORDER,
    device: str = DEVICE,
    channel: str = CHANNEL,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    duration_s: float = DURATION_S,
    min_val_v: float = MIN_VAL_V,
    max_val_v: float = MAX_VAL_V,
    invert: bool = INVERT,
    verbose: bool = False,
) -> tuple[float, float]:
    """Read one waveform and return ``(mean_v, std_v)`` of the low-passed trace.

    Like ``DAQController.monitor_cycle`` -- band-limit and report the mean plus
    the trace spread -- but with the two cutoffs split: the digital Butterworth
    runs at ``f_cut_dig`` while ``f_cut_hw`` states the analog bandwidth already
    in the signal, and the narrower of the two (``f_eff``) sets the settle
    guard.  Two draft-local corrections on top: the read is sign-inverted
    (``invert``) and the first ``SETTLE_CYCLES / f_eff`` s are dropped after
    filtering as a turn-on transient, so the mean and std see only steady-state
    signal.
    """
    _, voltages = read_waveform(
        device=device, channel=channel, sample_rate_hz=sample_rate_hz,
        duration_s=duration_s, min_val_v=min_val_v, max_val_v=max_val_v,
        invert=invert, verbose=verbose,
    )
    filtered = lowpass(voltages, sample_rate_hz, f_cut_dig, filter_order)
    f_eff = effective_f_cut(f_cut_hw, f_cut_dig)
    n_settle = settle_samples(sample_rate_hz, f_eff)
    kept = filtered[n_settle:] if filtered.size > n_settle else filtered
    return stats(kept)


__all__ = [
    "read_waveform",
    "lowpass",
    "amplitude_spectrum",
    "stats",
    "effective_f_cut",
    "settle_samples",
    "measure",
    "report",
]


def main() -> None:
    import matplotlib.pyplot as plt

    times, voltages = read_waveform(verbose=True)   # already sign-inverted
    filtered = lowpass(voltages, SAMPLE_RATE_HZ, F_CUT_DIG, FILTER_ORDER)

    # Drop the leading turn-on transient before any statistics / spectrum.
    # The raw trace is band-limited by the hardware alone; the filtered trace
    # by the narrower of hardware and digital cutoffs.
    f_eff = effective_f_cut()
    n_settle = settle_samples(SAMPLE_RATE_HZ, f_eff)
    settle_s = n_settle / SAMPLE_RATE_HZ
    v_kept, f_kept = voltages[n_settle:], filtered[n_settle:]
    t_kept = v_kept.size / SAMPLE_RATE_HZ
    print(
        f"Read {voltages.size} samples over {times[-1]:.3f} s; "
        f"dropped first {settle_s*1000:.0f} ms warmup ({n_settle} samples); "
        f"kept {t_kept:.2f} s (raw bandwidth {F_CUT_HW:g} Hz, "
        f"filtered {f_eff:g} Hz)"
    )
    report("raw", v_kept)
    report("filtered", f_kept)

    fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(10, 8))

    # --- time domain (full trace shown; shaded span = discarded warmup) ---
    ax_t.plot(times, voltages * 1000.0, linewidth=0.8, alpha=0.5,
              label=f"raw (hw {F_CUT_HW:.1f} Hz)")
    ax_t.plot(times, filtered * 1000.0, linewidth=1.5,
              label=f"digital low-pass {F_CUT_DIG:.1f} Hz")
    if n_settle > 0:
        ax_t.axvspan(0.0, settle_s, color="gray", alpha=0.15,
                     label=f"discarded warmup ({settle_s*1000:.0f} ms)")
    ax_t.set_title(f"{DEVICE}/{CHANNEL}  ({SAMPLE_RATE_HZ/1000:.0f} kS/s, {DURATION_S:.1f} s)")
    ax_t.set_xlabel("Time (s)")
    ax_t.set_ylabel("Voltage (mV)")
    ax_t.legend()
    ax_t.grid(True)

    # --- frequency domain (single-sided amplitude spectrum, steady-state only) ---
    f_raw, s_raw = amplitude_spectrum(v_kept, SAMPLE_RATE_HZ)
    f_filt, s_filt = amplitude_spectrum(f_kept, SAMPLE_RATE_HZ)
    ax_f.loglog(f_raw, s_raw * 1e6, linewidth=0.8, alpha=0.5, label="raw")
    ax_f.loglog(f_filt, s_filt * 1e6, linewidth=1.5,
                label=f"digital low-pass {F_CUT_DIG:.1f} Hz")
    ax_f.axvline(F_CUT_HW, color="k", linestyle="--", linewidth=1.0, alpha=0.6,
                 label=f"hardware {F_CUT_HW:.1f} Hz")
    if F_CUT_DIG != F_CUT_HW:
        ax_f.axvline(F_CUT_DIG, color="tab:red", linestyle=":", linewidth=1.2, alpha=0.8,
                     label=f"digital {F_CUT_DIG:.1f} Hz")
    ax_f.set_title("Amplitude spectrum")
    ax_f.set_xlabel("Frequency (Hz)")
    ax_f.set_ylabel("Amplitude (uV)")
    ax_f.legend()
    ax_f.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
