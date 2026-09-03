"""Manual smoke test (v2): comb phase dPhi_comb per pair, ONE free parameter + a
beta2 dispersion check.

Not a pytest test (no mocks, needs real hardware) -- run it directly.  Two
invocations, exactly as v1:

    python src/calibration_module/steps/calib_step7_v2.py            # COLLECT: sweep each target
                                                     #   pair, write a raw CSV, then
                                                     #   fit + verify straight away
    python src/calibration_module/steps/calib_step7_v2.py some.csv   # REFIT:   fit dPhi_comb from an
                                                     #   existing CSV, offline (no hw)

A COLLECT fits the CSV it just wrote (add ``--no-fit`` to stop after the CSV).
Either way the fit writes a combined ``calib_step7_result_*.json`` (the step-3 +
step-6 payloads carried over from ``IN_STEP6``, the fitted ``{Phi_k}`` spectrum,
and the beta2 verification) -- the single input downstream consumers read.

What changed vs :mod:`calib_step7_v1`
-------------------------------------
1. **Sign convention.**  The fringe is ``cos(dPhi_comb - dPhi_SLM)``: the panel
   phase SUBTRACTS from the comb phase (it compensates the comb's dispersion,
   so the fringe peaks where the two match), where v1 had
   ``cos(dPhi_SLM + dPhi_comb)``.  Since ``cos`` is even the two differ only in
   the SIGN of the recovered ``dPhi_comb`` -- a v2 ``Phi_k`` is the negative of
   a v1 one, so do NOT feed a v2 JSON to a consumer written against v1's
   ``E = sum_k eta_k sqrt(x_k w_k) exp(i[phi_half + phi_half + Phi_k])``
   forward model (e.g. ``calib_step8_v1.py``) without flipping the sign.
2. **Nothing floats but the phase.**  ``a = eta_ref``, ``b = eta_tgt``, the
   single-beam response and the dark are ALL taken from step 6 / the measured
   all-off read, so ``dPhi_comb`` is the ONLY free parameter (v1 also floated a
   shared amplitude scale ``s`` and a residual dark ``d``).  A step-6 error can
   no longer be absorbed by a nuisance parameter -- it shows up as a bad
   ``R^2`` / a non-zero mean residual instead.  There is consequently only one
   fit method, so v1's ``--bounded`` / ``--fix`` flags are gone.
3. **Verification.**  The recovered spectrum is compared against pure
   second-order dispersion: pair ``i`` at comb detuning ``Omega_i`` should carry
   ``dPhi_comb,i = beta2 (Omega_ref^2 - Omega_i^2)`` relative to the reference.
   ``beta2`` is an INPUT here (:data:`BETA2`), not something this script fits --
   the only fitted quantities are the per-pair ``dPhi_comb``.  The check just
   evaluates the model at each pair's real ``Omega`` and PRINTS the comparison
   pair by pair; the plot shows only the measured unwrapped trace, because the
   quoted phase errors do not yet carry the step-6 ``eta`` uncertainty and a
   drawn pull would overstate the disagreement.  ``Omega`` must be the
   REAL comb geometry (``PAIR_OMEGA``, or ``OMEGA_ZERO_INDEX``/``OMEGA_STEP``):
   the regressor is ``Omega^2``, so a pair sitting 10000 comb lines out is not
   the same as one at the centre.  ``--no-verify`` skips it.
4. The measurement grid is UNCHANGED from v1.

What it measures.  Each target pair carries a fixed comb-phase offset
``dPhi_comb`` relative to a common reference pair (the reference defines
``Phi = 0``).  Driving the two pairs at once makes them interfere; the fringe
encodes ``dPhi_comb``.  Running every target builds the phase spectrum
``{Phi_k}``.

The drive (one geometry, same as v1).  The reference pair is held fully on
(``x_r = w_r = 1``); the target's TWO channels are swept TOGETHER
(``x_t = w_t = v``) over the ramp ``SWEEP_MIN..SWEEP_MAX``.  A channel at
intensity ``v`` sits at panel phase ``theta = 2*asin(sqrt(v))`` with field
``sqrt(v)*exp(i theta/2)``, so the target field amplitude is
``g = sqrt(x_t w_t) = sin^2(theta/2)`` and ``dPhi_SLM = theta - pi``::

    Y = a^2 + b^2 g^2 + 2 a b g cos(dPhi_comb - dPhi_SLM)
        + step-6 single-beam background
      = a^2 + |b sin^2(theta/2)|^2
        + 2 a b sin^2(theta/2) cos(dPhi_comb + pi - theta) + background

with ``a`` the fully-on reference amplitude ``eta_ref``.  Sweeping ``v``
0.1 -> 1.0 sweeps ``theta`` over ~37..180 deg, tracing the half fringe.

The fit (in :mod:`calibration_module.phase`).  Every point is reduced to
``(g, dPhi_SLM)`` from its commanded intensities and handed to
:func:`~calibration_module.phase.fit_phase_fixed`: weighted (1/std) nonlinear
least squares in the single parameter ``dPhi_comb``, errors from the weighted
Jacobian as-is (no chi2/dof, no Birge rescaling).

Add ``--flip`` when the photodiode/DAQ reads inverted (more light -> more
negative volts): it negates the raw ``voltage_mean_v`` and its per-row
``dark_v`` (the SAME channel) so the fit's ``y = mean - dark`` becomes the
positive light signal.  On a REFIT it writes a sibling ``*_flipped.csv`` and
fits that; on a COLLECT the values are negated as they are read, before the CSV
is written -- so the automatic fit that follows does NOT negate them again.
The spread (``voltage_std_v``) and ``std_ratio`` are sign-independent.

Each point is one fixed-duration ``daq_module`` acquisition like step 6 (the
same ``DAQController.monitor_cycle`` read the GUI pipeline uses): ``T_SINGLE_S``
for the all-off dark (near-zero signal needs the averaging) and ``T_BOTH_S``
for the sweep points (the reference is fully on, so they are bright), low-passed
at the ``DAQMonitorSettings`` bandwidth.  Every CSV row records the mean, its
trace std and the std ratio (std/|mean|).

Prereq: ONE combined step-6 result JSON (``calib_step6_test.save_combined_json``)
is the only input -- it embeds the raw Step-3 calibration under ``"step3"``
(-> channel layout) and every fitted pair under ``"step6"`` (-> eta + single-beam
/ dark background), so the reference and all targets come from a single file.
Point ``IN_STEP6`` at the latest step-6 run.

All model / background removal / weighted fit / verification / persistence live
in :mod:`calibration_module.phase`; this file only wires up hardware and
prints/plots.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for draft_hw

from draft_hw import connect_daq, connect_slm, read_point  # noqa: E402
from slm_module.calibration.calibration_new import calibration_result_from_dict  # noqa: E402
from slm_module.encoding import channel_layout_from_calibration  # noqa: E402
from calibration_module.phase import (  # noqa: E402
    Beta2Fit,
    PhaseFit,
    PhaseResult,
    beta2_payload,
    fit_beta2,
    fringe_arg,
    load_pair_models,
    load_phase_csv,
    phi_half,
    save_comb_phase_json,
)

# ---- Edit these to match your setup ----
CALIB_PATH = REPO_ROOT / "src/calib_data"          # data directory: inputs + outputs live here
# Pair numbering.  Pairs are labelled from PAIR_INDEX_BASE, and that label is
# what every CSV / JSON records; the SLM layout arrays are always 0-based, so
# _slot() converts.  Set PAIR_INDEX_BASE = 0 for the old 0-based convention.
PAIR_INDEX_BASE = 1                                # pairs are numbered 1..N
REF_INDEX = 1                                      # common reference pair (Phi = 0)
TGT_INDICES = [2, 3, 4, 5]                         # target pairs measured vs the reference

# The ONE input: a combined step-6 result JSON (save_combined_json).  It embeds
# the raw Step-3 calibration under "step3" (-> channel layout) and every fitted
# pair under "step6" (-> eta + single-beam background), so the reference + all
# targets come from this single file -- no separate step-3 import.
IN_STEP6 = CALIB_PATH / "calib_step6_result_0829_0033.json"  # pairs 1 (ref) + 3,4,5 (targets)

# ---- The target sweep (IDENTICAL to v1 -- do not change without redoing v1) ----
# Reference fully on (x_r = w_r = 1); the target's two channels swept TOGETHER
# (x_t = w_t) over this ramp.
SWEEP_MIN = 0.1                  # min per-side target intensity in the ramp (0..1)
SWEEP_MAX = 1.0                  # max per-side target intensity in the ramp (0..1)
N_SWEEP_POINTS = 10              # points in the ramp

OUT_DIR = CALIB_PATH             # all step-7 outputs live in the data directory

SLM_DISPLAY_NO = None            # None -> auto-detect the LCOS-SLM display (like the GUI's Detect)
USB_SLM_NO = 1                   # SLM_Ctrl_* device index for the DVI-mode switch (USB link)

DAQ_DEVICE = "Dev1"
DAQ_CHANNEL = "ai0"

# ---- Fixed per-point acquisition (daq_module) ----
# Sample rate / range / low-pass bandwidth are the DAQMonitorSettings defaults
# (1 kS/s, +/-0.1 V DIFF, 20 Hz).  The all-off dark sits at zero signal, so it
# gets the longer T_single window; sweep points always have the reference
# fully on (bright, both pairs driven) and read T_both.  Every CSV row records
# the per-point std (voltage_std_v) and std_ratio.
T_SINGLE_S = 10.0                 # all-off dark (at most one beam on) (s)
T_BOTH_S = 10.0                   # sweep points: reference + target on (s)

SETTLE_S = 0.25                  # wait after each SLM pattern change, before reading

# Fold in the step-6 single-beam response as a FIXED background.  The reference is
# held fully on -> its single-beam is a constant; only the swept target ramps.
# Keeps the fringe from having to absorb the single-beam ramp.  Together with the
# pinned etas and the measured per-row dark this leaves dPhi_comb as the sole
# free parameter (v2's whole point) -- turn it off only to diagnose step 6.
SINGLE_BEAM_BG = True

# Method label stored in the output JSON (v2 has exactly one fit method).
METHOD = "fixed_comb_only"

# ---- Verification: dPhi_comb,i = beta2 (Omega_ref^2 - Omega_i^2) ----
# Comb detuning Omega of each pair.  This needs the REAL comb geometry, not the
# pair index: the regressor is Omega^2, and pair 1 sitting 1100 comb lines out
# from the two-photon centre is not the same as it sitting at 0.  Writing
# Omega_i = (i - OMEGA_ZERO_INDEX) * OMEGA_STEP keeps that offset explicit --
# for pairs 1..5 on comb lines 1100, 3100, ... 9100 with f_rep = 80 MHz that
# is OMEGA_ZERO_INDEX = 1 - 1100/2000 = 0.45 and OMEGA_STEP = 2000 * 2*pi*f_rep.  Setting
# OMEGA_ZERO_INDEX = PAIR_INDEX_BASE (i.e. "Omega ~ pair index") drops the
# linear-in-index part of the phase and gets beta2 wrong by several fold.
# Pick a unit and stay in it: beta2 comes out in rad per that unit squared.
# calib_synth_v1.py prints a ready-made block for whatever comb you set there.
PAIR_OMEGA: dict[int, float] | None = None   # explicit {pair index: Omega}; None -> uniform comb
OMEGA_ZERO_INDEX = 0.45          # pair index sitting at Omega = 0 (the comb centre)
OMEGA_STEP = 1.005310e12         # Omega per unit pair index (2000 lines * 2 pi f_rep)
OMEGA_UNIT = "rad/s"             # label only, for printing / the JSON

# The dispersion slope itself, in rad per OMEGA_UNIT^2.  beta2 is an INPUT to
# the check, not something this script fits: the only fitted quantities are the
# per-pair dPhi_comb.  The verification just evaluates beta2 (Omega_ref^2 -
# Omega_i^2) at each pair's real Omega and plots it against the measured
# dPhi_comb,i.  For the symmetric-pair geometry (the pair's two comb lines at
# -Omega_i and +Omega_i) this slope is -2 * GDD, so 1e4 fs^2 of GDD is -2e-26.
# Set to None only if the dispersion is unknown and you want it fitted instead.
# Propagate step 6's eta uncertainty into dPhi_comb before the beta2 check.
#
# fit_phase_fixed PINS a = eta_ref and b = eta_tgt, so dphi_comb_err is the
# fringe's photon noise ALONE -- the etas' own error is invisible there, not
# absent.  The lever is d(dPhi)/d(eta) ~ 15 rad per unit eta, so step 6's
# ~0.1% eta puts a couple of tenths of a degree on the phase: several times
# the fringe noise, and the reason this check used to read >3 sigma on data
# that is fine.  Off (False) reproduces the old, too-small error bars.
PROPAGATE_ETA = True

BETA2: float | None = -2.0e-26

# A fringe fit only recovers dPhi_comb modulo 2 pi, so the verification searches
# an integer 2 pi shift per target (+/- this many) and keeps the set that fits
# the quadratic best.  0 disables unwrapping (use the wrapped phases as-is).
MAX_BRANCH = 2


# ======================================================================
# input loading  (layout + step-6 models from the combined JSON)
# ======================================================================

def _slot(pair: int) -> int:
    """Pair label -> its 0-based slot in the SLM layout / drive arrays."""
    return pair - PAIR_INDEX_BASE


def load_layout():
    """Channel layout from the Step-3 calibration EMBEDDED in the step-6 JSON.

    ``save_combined_json`` stores the raw step-3 payload under ``"step3"``, so
    the layout the hardware run drives is guaranteed to be the one the step-6
    etas were calibrated under.
    """
    payload = json.loads(IN_STEP6.read_text(encoding="utf-8"))
    step3 = payload.get("step3")
    if step3 is None:
        raise ValueError(
            f"{IN_STEP6} has no embedded 'step3' calibration; point IN_STEP6 at "
            f"a combined step-6 result (calib_step6_test.save_combined_json)"
        )
    layout = channel_layout_from_calibration(calibration_result_from_dict(step3))
    for name, idx in [("REF_INDEX", REF_INDEX)] + [("TGT_INDICES", k) for k in TGT_INDICES]:
        if not (0 <= _slot(idx) < layout.n_channels):
            raise ValueError(
                f"{name} entry {idx} out of range (layout has {layout.n_channels} "
                f"pairs, numbered from {PAIR_INDEX_BASE})"
            )
    return layout


def load_models():
    """Load the step-6 pair models; require REF_INDEX and every TGT_INDICES entry."""
    models = load_pair_models([IN_STEP6])
    needed = [("reference", REF_INDEX)] + [("target", k) for k in TGT_INDICES]
    for role, idx in needed:
        if idx not in models:
            raise ValueError(
                f"no step-6 model for {role} pair index {idx}; found "
                f"{sorted(models)} in {IN_STEP6}"
            )
    print(f"Step 6: eta[ref {REF_INDEX}] = {models[REF_INDEX].eta:.4g} ; "
          + " ".join(f"eta[{k}]={models[k].eta:.4g}" for k in TGT_INDICES))
    return models


def pair_omega(index: int) -> float:
    """Comb detuning ``Omega`` of a pair index (see the PAIR_OMEGA config block).

    Explicit ``PAIR_OMEGA`` wins; otherwise a uniform comb,
    ``Omega_i = (i - OMEGA_ZERO_INDEX) * OMEGA_STEP``.  ``OMEGA_ZERO_INDEX`` is
    the pair index that would sit at the two-photon centre -- generally NOT the
    first pair, and getting it wrong biases beta2 badly (see the config block).
    """
    if PAIR_OMEGA is not None:
        if index not in PAIR_OMEGA:
            raise ValueError(f"PAIR_OMEGA has no entry for pair index {index}")
        return float(PAIR_OMEGA[index])
    return (float(index) - OMEGA_ZERO_INDEX) * OMEGA_STEP


# ======================================================================
# report + plot
# ======================================================================

def _sigma(value: float, err: float) -> float:
    return abs(value) / err if err else float("nan")


def report(fit: PhaseFit, tgt: int, ref: int) -> None:
    """Print dPhi_comb (rad + deg) and the fit quality of the one-parameter fit."""
    print("Model:  Y = a^2 + b^2 sin^4(theta/2) "
          "+ 2ab sin^2(theta/2) cos(dPhi_comb + pi - theta) + step6 single-beam")
    print("            [both target channels swept together; theta the shared panel phase]")
    print("            a = eta_ref, b = eta_tgt, background and dark ALL fixed from step 6")
    print("            -> dPhi_comb is the ONLY free parameter")
    print(f"Pair {tgt} vs reference {ref}  (value +/- error):")
    print(f"  dPhi_comb = {fit.dphi_comb:+.4f} +/- {fit.dphi_comb_err:.4f} rad"
          f"   ( {fit.dphi_comb_deg:+.2f} +/- {np.degrees(fit.dphi_comb_err):.2f} deg )")
    print(f"  a (ref R_1)      = {fit.a*1e3:.4f} mV^0.5   (pinned to step-6 eta)")
    print(f"  b (tgt eta CxCw) = {fit.b*1e3:.4f} mV^0.5   (pinned to step-6 eta)")
    print(f"  fringe amp 2ab   = {fit.amp*1e3:.4f} mV   (pinned)")
    # Not fitted here (v1 floated it as `d`), so it is a pure check on step 6:
    # a big mean residual means the fixed amplitudes/background are off.
    pulls = fit.residuals / fit.std
    print(f"  mean residual    = {float(np.mean(fit.residuals))*1e3:+.4f} mV   "
          f"(NOT fitted -- should be ~0 if step 6 is right)")
    print(f"  max |pull|       = {float(np.max(np.abs(pulls))):.2f}")
    print(f"  R^2 = {fit.r2:.4f}")


def make_plot(fit: PhaseFit, tgt: int, path) -> None:
    """Measured Y(dPhi_SLM) with the fitted one-parameter model curve + pulls, PNG."""
    import matplotlib

    matplotlib.use("Agg")  # headless: write a PNG rather than open a window
    import matplotlib.pyplot as plt

    dphi = np.degrees(fit.dphi_slm)             # dPhi_SLM at the measured points
    pulls = fit.residuals / fit.std

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Smooth model over the swept geometry.  Rebuild g/dPhi_SLM the same way the
    # fit did (from the per-point commanded intensities) so the curve tracks the
    # data: sweep the shared phase over the full 0..180 deg half turn
    # (x_t = w_t = sin^2(theta/2)).
    wt_s = xt_s = np.sin(np.radians(np.linspace(0.0, 180.0, 400)) / 2.0) ** 2
    xr_c = float(np.median(fit.x_r)) if fit.x_r is not None else 1.0
    wr_c = float(np.median(fit.w_r)) if fit.w_r is not None else 1.0
    g_s = np.sqrt(np.clip(xt_s * wt_s, 0.0, None))
    dslm = phi_half(xt_s) + phi_half(wt_s) - phi_half(xr_c) - phi_half(wr_c)

    # Fixed step-6 single-beam background = fit.known - a^2 - b^2 g^2 at each
    # fitted point; interpolate it onto the smooth grid (exact at the points, and
    # geometry-agnostic, so no assumption about how the background splits in g).
    bg_pts = fit.known - fit.a**2 - fit.b**2 * fit.g**2
    order = np.argsort(fit.dphi_slm)
    bg_s = np.interp(dslm, fit.dphi_slm[order], bg_pts[order])

    # fringe_arg honours the fit's sign convention ("comb-slm" for v2)
    model = (fit.a**2 + fit.b**2 * g_s**2
             + 2.0 * fit.a * fit.b * g_s
             * np.cos(fringe_arg(dslm, fit.dphi_comb, fit.convention))
             + bg_s + fit.offset)
    label = r"fit: $a^2+b^2\sin^4+2ab\sin^2\cos(\Delta\Phi_{comb}-\Delta\Phi_{SLM})$"
    ax1.plot(np.degrees(dslm), model * 1e3, "-", color="tab:blue", lw=1.6, label=label)
    ax1.errorbar(dphi, fit.y * 1e3, yerr=fit.std * 1e3, fmt="o", ms=5, color="tab:orange",
                 ecolor="lightgray", elinewidth=1, capsize=2, zorder=3,
                 label="measured (dark-subtracted)")
    ax1.set_xlabel(r"$\Delta\Phi_{SLM}$  (deg)")
    ax1.set_ylabel(r"$Y$, dark-subtracted  (mV)")
    ax1.set_title(f"Pair {tgt} interference  (both channels, half fringe)")
    ax1.legend(loc="best", fontsize=8)

    ax2.axhspan(-1, 1, color="tab:blue", alpha=0.12, label=r"$\pm1\sigma$")
    ax2.axhline(0, color="gray", ls="--", lw=1)
    ax2.scatter(dphi, pulls, c="tab:red", s=40, edgecolor="k", lw=0.4)
    ax2.set_xlabel(r"$\Delta\Phi_{SLM}$  (deg)")
    ax2.set_ylabel("Pull = residual / std")
    ax2.set_title("Pulls")
    ax2.legend(loc="upper right", fontsize=8)

    txt = (
        f"$\\Delta\\Phi_{{comb}}$ = {fit.dphi_comb_deg:+.2f} $\\pm$ "
        f"{np.degrees(fit.dphi_comb_err):.2f} deg  "
        f"({_sigma(fit.dphi_comb, fit.dphi_comb_err):.0f}$\\sigma$)\n"
        f"a = {fit.a*1e3:.3f}, b = {fit.b*1e3:.3f} mV$^{{1/2}}$ (both pinned to step 6)\n"
        f"mean resid = {float(np.mean(fit.residuals))*1e3:+.3f} mV (not fitted)\n"
        f"R$^2$ = {fit.r2:.4f}  [1 free param: $\\Delta\\Phi_{{comb}}$]"
    )
    ax1.text(0.05, 0.95, txt, transform=ax1.transAxes, va="top",
             bbox=dict(boxstyle="round", fc="white", alpha=0.85), fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)


# ======================================================================
# verification  (Phi_i = beta2 (Omega_0^2 - Omega_i^2))
# ======================================================================

def report_dispersion(bfit: Beta2Fit) -> None:
    """Print the per-pair table comparing dPhi_comb,i against beta2 (Om_r^2 - Om_i^2)."""
    print(f"\n=== Verification: dPhi_comb,i vs beta2 (Omega_{REF_INDEX}^2 - Omega_i^2) ===")
    print(f"Omega unit: {OMEGA_UNIT}   reference pair {REF_INDEX} at "
          f"Omega_{REF_INDEX} = {bfit.omega_ref:g}")
    if PAIR_OMEGA is None:
        print(f"Omega_i = (i - {OMEGA_ZERO_INDEX:g}) * {OMEGA_STEP:g}  "
              f"(uniform comb; set PAIR_OMEGA for the real geometry)")
    if bfit.beta2_fixed:
        print(f"beta2 = {bfit.beta2:+.6g} rad/({OMEGA_UNIT})^2 -- GIVEN, not fitted; "
              f"only dPhi_comb was fitted")
    else:
        print("beta2 was FITTED (BETA2 = None): one free slope over the targets")
    print(f"dPhi_comb is only defined mod 2pi -> branch searched over "
          f"+/-{bfit.max_branch}")
    prop = bfit.cov is not None
    if prop:
        print("sigma includes the PINNED step-6 eta (PROPAGATE_ETA): eta_ref is "
              "shared,\n  so 'pull*' is the covariance-whitened residual -- judge on that")
    else:
        print("sigma is the FRINGE noise only (PROPAGATE_ETA off): the pinned "
              "step-6 eta\n  error is not in it, so these pulls read too large")
    print(f"{'pair':>4} {'Omega':>9} {f'u=O{REF_INDEX}^2-Oi^2':>12} "
          f"{'dPhi meas':>12} {'2pi':>4} {'dPhi used':>10} {'model':>10} "
          f"{'resid':>9} {'sigma':>8} {'pull':>7} {'pull*':>7}")
    for k, om, u, pm, pe, br, pu, mo, rr, pl, pw in zip(
        bfit.indices, bfit.omega, bfit.u, bfit.dphi_comb, bfit.dphi_comb_err,
        bfit.branch, bfit.phi_used, bfit.model, bfit.residuals, bfit.pulls,
        bfit.pulls_white,
    ):
        print(f"{k:>4} {om:>9.4g} {u:>12.4g} "
              f"{np.degrees(pm):>+9.2f}deg {br:>+4d} {np.degrees(pu):>+9.2f} "
              f"{np.degrees(mo):>+10.2f} {np.degrees(rr):>+9.2f} "
              f"{np.degrees(pe):>8.3f} {pl:>+7.2f} {pw:>+7.2f}")
    rms = float(np.sqrt(np.mean(np.degrees(bfit.residuals) ** 2)))
    print(f"residual: rms {rms:.3f} deg,  max {np.max(np.abs(np.degrees(bfit.residuals))):.3f} deg,"
          f"  max |pull| {bfit.max_abs_pull:.2f},  max |pull*| "
          f"{bfit.max_abs_pull_white:.2f},  chi2 = r^T C^-1 r = {bfit.chi2_gls:.2f} "
          f"over {bfit.n_targets} targets")
    # What a free slope WOULD be, on the same branches -- says whether a
    # mismatch is the overall dispersion being off (scale) or the shape.
    dev = (bfit.beta2_free / bfit.beta2 - 1.0) * 100.0 if bfit.beta2 else float("nan")
    print(f"  free-slope fit on these points: {bfit.beta2_free:+.6g} "
          f"+/- {bfit.beta2_free_err:.3g}  ({dev:+.2f}% vs the beta2 used)")
    if OMEGA_UNIT == "rad/s":
        # dPhi_comb,i = 2 GDD (Omega_i^2 - Omega_ref^2) = -2 GDD * u, so the
        # slope is -2x the GDD -- valid for the symmetric-pair geometry (the
        # pair's two comb lines at -Omega_i and +Omega_i).
        print(f"  in GDD terms: beta2 used = {(-bfit.beta2/2.0)/(1e-15)**2:+.2f} fs^2, "
              f"free-slope = {(-bfit.beta2_free/2.0)/(1e-15)**2:+.2f} "
              f"+/- {(bfit.beta2_free_err/2.0)/(1e-15)**2:.2f} fs^2")
    # Judge on the whitened pull: with eta propagated the phases are correlated,
    # and a single common-mode eta_ref shift would otherwise be counted once per
    # target and read as several independent outliers.
    worst = bfit.max_abs_pull_white
    if worst > 3.0:
        print(f"  ** CHECK: a pair sits >3 sigma off the model (max |pull*| = "
              f"{worst:.2f}) -- wrong beta2 or Omega mapping, a mis-fit fringe, "
              f"or genuine higher-order dispersion **")
    else:
        print(f"  OK: every pair within 3 sigma of the model "
              f"(max |pull*| = {worst:.2f})")
        if bfit.max_abs_pull > 3.0:
            print("     (the raw per-point pull reaches "
                  f"{bfit.max_abs_pull:.2f}; that excess is the SHARED eta_ref "
                  "shift, counted once per target)")


def make_spectrum_plot(bfit: Beta2Fit, path) -> None:
    """The recovered comb-phase spectrum against PAIR INDEX -- measured only.

    One trace: the fitted ``dPhi_comb,i`` after 2pi unwrapping, with the
    reference pair sitting at ``dPhi = 0`` by definition.  Each point is
    labelled with its value in degrees and with the pair's REAL comb detuning
    ``Omega`` (from the comb lines, not the index).

    The ``beta2 (Omega_ref^2 - Omega_i^2)`` trace and the residual/pull panel
    are deliberately not drawn -- there is no beta2 anywhere in this figure.
    :func:`report_dispersion` prints both, and that is the honest place for
    them, because a pull is only meaningful once the pinned step-6 ``eta`` is
    in the error and its shared ``eta_ref`` part has been decorrelated.

    The error bars DO include the step-6 ``eta`` once ``PROPAGATE_ETA`` is on
    (``bfit.dphi_comb_err`` is then ``sqrt(diag(cov))``).  They are still not a
    pull: the ``eta_ref`` part is common to every point, so the bars slide
    together rather than independently -- the whitened pulls in
    :func:`report_dispersion` are the ones to read.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # reference included: it anchors the trace at zero by construction
    idx = [REF_INDEX] + list(bfit.indices)
    meas = np.degrees(np.concatenate([[0.0], bfit.phi_used]))
    merr = np.degrees(np.concatenate([[0.0], bfit.dphi_comb_err]))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.errorbar(idx, meas, yerr=merr, fmt="o-", ms=7, color="tab:orange",
                ecolor="gray", elinewidth=1, capsize=3, lw=1.4, zorder=3,
                label=r"measured $\Delta\Phi_{comb,i}$ (2$\pi$ unwrapped)"
                      + (r", $\sigma$ incl. step-6 $\eta$" if bfit.cov is not None
                         else r", $\sigma$ = fringe noise only"))
    ax.scatter([REF_INDEX], [0.0], marker="*", s=200, c="tab:green", zorder=4,
               label=f"reference pair {REF_INDEX} ($\\Delta\\Phi\\equiv0$)")

    for i, ph, er in zip(idx, meas, merr):
        ax.annotate(f"{ph:+.2f} $\\pm$ {er:.2f}$^\\circ$", (i, ph),
                    textcoords="offset points", xytext=(0, 13), fontsize=8,
                    ha="center")
        # off to the right rather than under the marker: the trace rises
        # steeply, so a centred label lands on the line itself
        ax.annotate(f"$\\Omega$ = {pair_omega(i):.3g}", (i, ph),
                    textcoords="offset points", xytext=(10, -12), fontsize=7,
                    ha="left", color="dimgray")

    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(idx)
    # asymmetric: the Omega labels run to the RIGHT of their marker, so the last
    # pair needs more room than the first
    ax.set_xlim(min(idx) - 0.5, max(idx) + 0.9)
    ax.margins(y=0.20)                       # room for the value labels
    ax.set_xlabel("pair index $i$")
    ax.set_ylabel(r"$\Delta\Phi_{comb,i}$  (deg)")
    ax.set_title("Recovered comb phase per pair (measured, unwrapped)"
                 "\n" r"$\beta_2$ comparison is in the printed table, not here")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)


def phase_covariance(fits: dict[int, PhaseFit], pairs: list[int]) -> np.ndarray:
    """Covariance of the fitted dPhi_comb across targets, step-6 eta included.

    Three contributions, for targets ``i``, ``j``::

        fringe noise    diag(dphi_comb_err_i^2)                    independent
        eta_tgt,i       diag((dphi_deta_tgt,i * eta_tgt_err_i)^2)  independent
        eta_ref         dphi_deta_ref_i * dphi_deta_ref_j * s^2    COMMON

    The last term is rank 1 and it is the whole point: every target is pinned
    to the SAME reference amplitude, so an error in eta_ref slides the entire
    spectrum together instead of scattering it.  Four independent sigmas would
    charge that one common shift four separate times.

    The sensitivities come from :func:`~calibration_module.phase.fit_phase_fixed`
    (Gauss-Newton, no extra fit); ``eta_ref_err`` is one number shared by every
    fit, so it is read off the first.
    """
    diag = np.array(
        [fits[k].dphi_comb_err**2 + (fits[k].dphi_deta_tgt * fits[k].eta_tgt_err) ** 2
         for k in pairs], dtype=float)
    c_ref = np.array([fits[k].dphi_deta_ref for k in pairs], dtype=float)
    s_ref = float(fits[pairs[0]].eta_ref_err) if pairs else 0.0
    return np.diag(diag) + np.outer(c_ref, c_ref) * s_ref**2


def verify_dispersion(fits: dict[int, PhaseFit]) -> Beta2Fit:
    """Run + report the beta2 dispersion check over every fitted target pair."""
    pairs = sorted(fits)
    cov = phase_covariance(fits, pairs) if PROPAGATE_ETA else None
    bfit = fit_beta2(
        pairs,
        [fits[k].dphi_comb for k in pairs],
        [fits[k].dphi_comb_err for k in pairs],
        [pair_omega(k) for k in pairs],
        omega_ref=pair_omega(REF_INDEX),
        max_branch=MAX_BRANCH,
        beta2=BETA2,                 # given, not fitted (None -> fit the slope)
        cov=cov,                     # None -> fringe noise only (PROPAGATE_ETA off)
    )
    report_dispersion(bfit)
    plot_path = OUT_DIR / "calib_step7_v2_spectrum.png"
    make_spectrum_plot(bfit, plot_path)
    print(f"Plot saved to {plot_path}")
    return bfit


# ======================================================================
# offline refit  (python calib_step7_v2.py some.csv)
# ======================================================================

def _targets_in_csv(path, default) -> list[int]:
    """Distinct target-pair indices recorded in a CSV (sorted).

    A collected CSV (:func:`measure_only`) stacks every TGT_INDICES entry vs the
    shared reference in one file, so its ``tgt_index`` column lists several pairs.
    Falls back to ``default`` if the column is missing (an old single-target CSV).
    """
    seen: list[int] = []
    with open(Path(path), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(line for line in f if not line.startswith("#")):
            t = row.get("tgt_index")
            if t in (None, ""):
                continue
            k = int(float(t))
            if k not in seen:
                seen.append(k)
    return sorted(seen) if seen else list(default)


def _flip_meas_csv(path) -> Path:
    """Write a sign-flipped copy of a raw step-7 CSV and return its path.

    The photodiode/DAQ reads inverted (more light -> more negative volts), so the
    raw ``voltage_mean_v`` and its per-row ``dark_v`` are negated -- both are the
    same channel, so the fit's ``y = mean - dark`` then yields the positive light
    signal (= dark - |mean|) with the residual dark still near zero.  Every other
    column is copied through unchanged: the spread (``voltage_std_v``) and
    ``std_ratio`` (= |std/mean|) are sign-independent.
    Output lands next to the source as ``<stem>_flipped.csv``.
    """
    src = Path(path)
    with open(src, newline="", encoding="utf-8") as f:
        lines = f.readlines()
    comments = [ln for ln in lines if ln.lstrip().startswith("#")]
    data_lines = [ln for ln in lines if not ln.lstrip().startswith("#")]

    reader = csv.DictReader(data_lines)
    fields = reader.fieldnames or []
    rows = []
    for row in reader:
        for col in ("voltage_mean_v", "dark_v"):
            val = row.get(col)
            if val not in (None, ""):
                row[col] = f"{-float(val):.9g}"
        rows.append(row)

    dst = src.with_name(f"{src.stem}_flipped.csv")
    with open(dst, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        for ln in comments:                              # carry trailing comments over
            parts = ln.lstrip("#").strip().split(",")    # negate a dark_mean_v comment too
            if len(parts) == 2 and parts[0].strip() == "dark_mean_v":
                f.write(f"# dark_mean_v,{-float(parts[1]):.9g}\n")
            else:
                f.write(ln if ln.endswith("\n") else ln + "\n")
    return dst


def fit_csv(path, *, flip: bool = False, verify: bool = True) -> None:
    """Re-fit an already-recorded CSV offline (no hardware), then verify.

    Needs two inputs: this CSV plus the combined step-6 JSON (``IN_STEP6``) for
    each pair's eta + single-beam background.  The CSV may carry several target
    pairs -- a collected file records every TGT_INDICES entry vs the shared
    REF_INDEX -- so every target present (that has a step-6 model) is fit
    separately against the reference and gets its own refit PNG.

    Unlike v1 there is only ONE fit: amplitudes/background/dark all pinned to
    step 6, ``dPhi_comb`` the sole free parameter
    (:func:`~calibration_module.phase.fit_phase_fixed`, ``comb_only=True``).

    ``flip`` handles an inverted photodiode/DAQ read: it writes a sign-flipped
    sibling CSV (:func:`_flip_meas_csv`, negating ``voltage_mean_v`` + ``dark_v``)
    and re-fits that instead, so the fitted fringe is the positive light signal.

    ``verify`` runs the beta2 dispersion check (:func:`verify_dispersion`) over the
    fitted spectrum and stores it in the output JSON.

    Everything is persisted into ONE combined ``calib_step7_result_*.json``
    (:func:`~calibration_module.phase.save_comb_phase_json`).
    """
    if flip:
        path = _flip_meas_csv(path)
        print(f"Flip: negated voltage_mean_v + dark_v -> re-fitting {path}")
    models = load_models()
    targets = _targets_in_csv(path, TGT_INDICES)
    fittable = [k for k in targets if k in models and k != REF_INDEX]
    if not fittable:
        raise ValueError(
            f"no fittable target in {path}: found targets {targets}, but have "
            f"step-6 models only for {sorted(models)} (reference is pair {REF_INDEX})"
        )
    print(f"Fitting pair(s) {fittable} vs reference {REF_INDEX} from {path} "
          f"(one free parameter: dPhi_comb)")
    fits: dict[int, PhaseFit] = {}
    for k in fittable:
        print(f"\n=== Re-fit: pair {k} vs reference {REF_INDEX} ===")
        result = load_phase_csv(path, models[k], models[REF_INDEX],
                                comb_only=True, single_beam_bg=SINGLE_BEAM_BG,
                                only_tgt=k)
        dts = result.per_trial_darks()
        drift = f" +/- {dts.std(ddof=1)*1e3:.4f} drift" if dts.size > 1 else ""
        print(f"Loaded {result.trial.size} rows, "
              f"dark = {result.dark*1e3:.4f}{drift} mV")
        report(result.fit, result.tgt_index, result.ref_index)
        plot_path = OUT_DIR / f"calib_step7_v2_pair{k}_refit.png"
        make_plot(result.fit, result.tgt_index, plot_path)
        print(f"Plot saved to {plot_path}")
        fits[k] = result.fit

    extra = None
    if verify:
        try:
            extra = {"verification": beta2_payload(verify_dispersion(fits),
                                                   omega_unit=OMEGA_UNIT)}
        except Exception as exc:            # never lose the fits over the check
            print(f"\nVerification FAILED ({type(exc).__name__}: {exc})")

    # Persist the fitted spectrum {Phi_k} as ONE combined JSON (step3 + step6
    # carried over verbatim from IN_STEP6) -- the single input for downstream
    # consumers.  NOTE the "comb-slm" convention recorded per fit: these phases
    # are the NEGATIVE of v1's (see the module docstring).
    out_json = OUT_DIR / f"calib_step7_result_{time.strftime('%m%d_%H%M')}.json"
    save_comb_phase_json({(k, METHOD): f for k, f in fits.items()},
                         IN_STEP6, out_json, ref_index=REF_INDEX,
                         csv_path=str(Path(path).resolve()),
                         single_beam_bg=SINGLE_BEAM_BG, extra=extra)
    print(f"\nCombined step-7 result (step3 + step6 + step7) saved to {out_json}")


# ======================================================================
# collect  (python calib_step7_v2.py  ->  drive SLM, record raw CSV, fit)
# ======================================================================

def build_xw_sweep() -> list[tuple[float, float, float, float]]:
    """Drive tuples: reference fully on, the target's two channels swept together.

    Returns target-first commanded-intensity tuples ``(x_t, w_t, x_r, w_r)`` with
    ``x_r = w_r = 1`` and ``x_t = w_t`` stepping over the
    ``SWEEP_MIN..SWEEP_MAX`` ramp.  Unchanged from v1.
    """
    values = np.round(np.linspace(SWEEP_MIN, SWEEP_MAX, N_SWEEP_POINTS), 6)
    return [(float(v), float(v), 1.0, 1.0) for v in values]


_MEAS_CSV_HEADER = [
    "trial", "tgt_index", "ref_index",
    "phi_xt_deg", "phi_wt_deg", "x_t", "w_t", "x_r", "w_r",
    "dark_v", "voltage_mean_v", "voltage_std_v", "std_ratio",
]


def write_meas_csv(results, path) -> str:
    """Write raw rows for one or more target pairs into a single CSV.

    Same column layout as :func:`calibration_module.phase.write_phase_csv` plus a
    trailing ``std_ratio`` (std/|mean|) column, and concatenates several
    :class:`PhaseResult` objects so every row carries its own ``tgt_index`` (and
    the shared ``ref_index``).  Round-trips via
    :func:`calibration_module.phase.load_phase_csv` (used by the offline refit),
    and is byte-compatible with a v1 CSV -- either version can refit either file.
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_MEAS_CSV_HEADER)
        for result in results:
            for t, x_t, w_t, x_r, w_r, dark_v, mean_v, std_v in zip(
                result.trial, result.x_t, result.w_t, result.x_r, result.w_r,
                result.dark_v, result.voltage_mean_v, result.voltage_std_v,
            ):
                phi_xt = np.degrees(2.0 * float(phi_half(x_t)))
                phi_wt = np.degrees(2.0 * float(phi_half(w_t)))
                ratio = abs(std_v / mean_v) if mean_v else float("inf")
                writer.writerow(
                    [int(t), result.tgt_index, result.ref_index,
                     f"{phi_xt:.4g}", f"{phi_wt:.4g}",
                     f"{x_t:.6g}", f"{w_t:.6g}", f"{x_r:.6g}", f"{w_r:.6g}",
                     f"{dark_v:.9g}", f"{mean_v:.9g}", f"{std_v:.9g}",
                     f"{ratio:.6g}"]
                )
    return str(out)


def _read_point(daq, x_t: float, w_t: float, x_r: float, w_r: float) -> tuple[float, float, float]:
    """One fixed-duration DAQ read for a drive point; return ``(mean, std, duration)``.

    Any channel on reads ``T_BOTH_S`` (sweep points are bright -- the reference
    is fully on); the all-off dark reads the DAQ's configured T_single window
    (``T_SINGLE_S``).  Filtering happens inside ``DAQController.monitor_cycle``
    -- the same read the GUI pipeline uses -- and ``std`` is the spread of that
    low-passed trace, recorded verbatim in the CSV as the per-point sigma the
    fit weights by.
    """
    single = not any(v > 0.0 for v in (x_t, w_t, x_r, w_r))
    mean_v, std_v = read_point(daq, single=single)
    return mean_v, std_v, (T_SINGLE_S if single else T_BOTH_S)


def _measure_target(slm, daq, layout, k: int, drive, *, flip: bool = False) -> PhaseResult:
    """Drive pair ``k`` (vs REF_INDEX) over ``drive`` and read Y; PhaseResult, no fit.

    Only channels ``k`` and ``REF_INDEX`` are driven; all others held off.  An
    all-off dark is read once at the start (T_SINGLE_S window) and stored per
    row for per-row subtraction.  Needs no step-6 model -- raw data only.

    ``flip`` negates the raw mean and dark reads (inverted DAQ sign convention:
    more light -> more negative volts), so the CSV this writes already carries the
    positive light signal; the std is a magnitude and stays as read.
    """
    from slm_module.encoding import encode_to_pattern

    n = layout.n_channels
    zeros = np.zeros(n)
    slm_width, slm_height = slm.get_slm_info()

    def _display(x_t, w_t, x_r, w_r) -> None:
        x_vals = zeros.copy()
        w_vals = zeros.copy()
        x_vals[_slot(k)], w_vals[_slot(k)] = x_t, w_t
        x_vals[_slot(REF_INDEX)], w_vals[_slot(REF_INDEX)] = x_r, w_r
        slm.display_array(encode_to_pattern(x_vals, w_vals, layout, slm_width, slm_height))
        if SETTLE_S:
            time.sleep(SETTLE_S)

    total = len(drive) + 1
    step = 0
    rows: list[tuple] = []
    _display(0.0, 0.0, 0.0, 0.0)                     # all-off dark, once
    dark_v, _, dur = _read_point(daq, 0.0, 0.0, 0.0, 0.0)
    if flip:
        dark_v = -dark_v                             # inverted DAQ sign (same channel)
    step += 1
    print(f"[{step}/{total}] pair {k} dark (all off, {dur:.0f}s) "
          f"= {dark_v*1000:.4f} mV")
    for x_t, w_t, x_r, w_r in drive:
        _display(x_t, w_t, x_r, w_r)
        mean_v, std_v, dur = _read_point(daq, x_t, w_t, x_r, w_r)
        if flip:
            mean_v = -mean_v
        rows.append((0, x_t, w_t, x_r, w_r, mean_v, std_v, dark_v))
        step += 1
        ratio = abs(std_v / mean_v) if mean_v else float("inf")
        print(f"[{step}/{total}] pair {k} x=w={x_t:.3f} ({dur:.0f}s) "
              f"-> {mean_v*1000:.4f} mV std ratio {ratio*100:.2f}%")

    return PhaseResult(
        tgt_index=k, ref_index=REF_INDEX,
        trial=np.array([r[0] for r in rows], dtype=int),
        x_t=np.array([r[1] for r in rows], dtype=float),
        w_t=np.array([r[2] for r in rows], dtype=float),
        x_r=np.array([r[3] for r in rows], dtype=float),
        w_r=np.array([r[4] for r in rows], dtype=float),
        voltage_mean_v=np.array([r[5] for r in rows], dtype=float),
        voltage_std_v=np.array([r[6] for r in rows], dtype=float),
        dark_v=np.array([r[7] for r in rows], dtype=float),
        n_trials=1,
    )


def measure_only(*, flip: bool = False) -> Path:
    """Sweep every target pair vs the shared reference; write one raw CSV.

    Loops over TGT_INDICES (each vs REF_INDEX), holding the reference fully on
    (x_ref = w_ref = 1) and sweeping the target's two channels together
    (x_tgt = w_tgt) over the SWEEP_MIN..SWEEP_MAX ramp -- the SAME grid as v1.
    All rows go into a single timestamped CSV, tagged per row with ``tgt_index``
    and ``ref_index``.  Raw data only: no step-6 models, no fit.  Refit later
    with ``python calib_step7_v2.py <that csv>``.

    Returns the written CSV path; :func:`main` fits it straight away (unless
    ``--no-fit``), so a normal run needs no second command.

    ``flip`` negates each raw mean/dark read (inverted DAQ sign) so the written
    CSV already holds the positive light signal -- the fit that follows (and any
    later refit) runs WITHOUT --flip.
    """
    layout = load_layout()
    if flip:
        print("Flip: negating voltage_mean_v + dark_v as read (inverted DAQ sign).")

    drive = build_xw_sweep()
    values = [x_t for x_t, _, _, _ in drive]
    slm = connect_slm(SLM_DISPLAY_NO, USB_SLM_NO)
    daq = connect_daq(device=DAQ_DEVICE, channel=DAQ_CHANNEL,
                      t_both=T_BOTH_S, t_single=T_SINGLE_S)
    results = []
    try:
        for k in TGT_INDICES:
            print(f"\n=== Sweep: pair {k} vs reference {REF_INDEX}  "
                  f"(x{REF_INDEX}=w{REF_INDEX}=1, sweep x{k}=w{k} over "
                  f"{values}) ===")
            results.append(_measure_target(slm, daq, layout, k, drive, flip=flip))
    finally:
        slm.close_slm()
        daq.disconnect()

    csv_path = OUT_DIR / f"calib_step7_meas_{time.strftime('%m%d_%H%M')}.csv"
    write_meas_csv(results, csv_path)
    print(f"\nCSV (ref {REF_INDEX}, targets {TGT_INDICES}) written to {csv_path}")
    return Path(csv_path)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    flip = "--flip" in argv          # inverted DAQ read -> negate voltage_mean_v + dark_v
    verify = "--no-verify" not in argv   # beta2 dispersion check on the fitted spectrum
    positional = [a for a in argv if not a.startswith("-")]
    if positional:                   # a CSV path -> offline refit, no hardware
        fit_csv(positional[0], flip=flip, verify=verify)
        return 0

    csv_path = measure_only(flip=flip)   # no arg -> fresh sweep (drives the SLM/DAQ)
    if "--no-fit" in argv:               # raw data only; fit by hand later
        print(f"Fit with:  python {Path(__file__).name} {csv_path}")
        return 0
    # Fit what was just collected, so no second command is needed.  flip=False:
    # a --flip COLLECT already negated the reads before writing the CSV.
    try:
        fit_csv(csv_path, flip=False, verify=verify)
    except Exception as exc:             # the raw CSV is on disk either way
        print(f"\nAuto-fit FAILED ({type(exc).__name__}: {exc})")
        print(f"Data is safe -- refit with:  python {Path(__file__).name} {csv_path}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
