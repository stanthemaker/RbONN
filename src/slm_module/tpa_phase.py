"""Comb-phase (dPhi_comb) calibration of each pair relative to a common reference.

Step 6 (:mod:`slm_module.tpa_pair`) calibrates each pair's two-photon efficiency
``eta`` *in isolation* -- one pair on at a time, so absolute optical phase never
enters.  This step drives **two pairs at once** and uses their coherent TPA
interference to recover the fixed comb phase offset ``dPhi_comb`` that a target
pair carries relative to a reference pair (pair 0 by convention).

Geometry.  A channel commanded at normalised INTENSITY ``x`` in [0, 1] (the
diffraction efficiency) sits at panel phase ``phi = 2*asin(sqrt(x))`` and has
field ``sqrt(x)*exp(i*phi/2)``.  The measured Step-3 transfer curve is monotonic
over the calibrated level range, so ``x`` in [0, 1] reaches ``phi`` in [0, pi]
only (``phi = pi`` is exactly ``x = 1``, fully on).  That is a *half* phase turn,
which is enough: with the reference fixed at ``phi = pi`` and the target swept
over ``phi in [0, pi]`` the relative SLM phase spans a full half fringe.

Sweep (Table 1).  Reference pair 1: ``x_1 = w_1 = 1`` (both channels fully on,
``phi^x_1 = phi^w_1 = pi``), all other pairs off.  Target pair 2 swept
symmetrically with per-channel field amplitude ``x_2 = w_2 = sin(theta2/2)``
(commanded INTENSITY ``sin(theta2/2)^2``) as ``theta2`` runs over ``[0, pi]``.
Writing ``a := R_1`` (the fixed reference amplitude) and ``b := eta_2 Cx_2 Cw_2``
(the target amplitude scale), the measured signal is::

    Y = a^2                                   (reference self term)
      + b^2 sin(theta2/2)^4                    (target self term R_2^2)
      + 2 a b sin(theta2/2)^2 cos(dPhi_comb - pi + theta2)   (interference)
      + d                                      (dark)

With ``g := sin(theta2/2)^2 = sqrt(x_2 w_2)`` the target-pair field amplitude and
``dPhi_SLM := theta2 - pi`` the SLM phase difference, the fringe argument
``dPhi_comb - pi + theta2`` is exactly ``dPhi_SLM + dPhi_comb``.  Because the
target is calibrated *against* pair 1 and pair 1 defines ``Phi_1 == 0``, the
fitted ``dPhi_comb`` IS the target pair's phase in the spectrum; running Table 1
for every target builds ``{Phi_k}``.

The fit floats ``a``, ``b`` and ``dPhi_comb`` (and a residual DC ``d``) directly
-- it does NOT take the amplitude from the step-6 ``eta``.  The model is LINEAR
in the four coefficients::

    Y = c0*1 + c1*g^2 + c2*[g cos(dPhi_SLM)] + c3*[g sin(dPhi_SLM)]
    c0 = a^2 + d      c1 = b^2
    c2 = 2 a b cos(dPhi_comb)   c3 = -2 a b sin(dPhi_comb)

solved by weighted least squares over ``[1, g^2, g cos, g sin]``; the physical
parameters follow in closed form::

    b = sqrt(c1)                 dPhi_comb = atan2(-c3, c2)
    a = sqrt(c2^2 + c3^2)/(2 b)  d = c0 - a^2

The reference amplitude ``a`` is fixed by the interference (``2ab``) and target
self term (``b^2``), NOT by the flat baseline, so it is separable from the
residual dark ``d``.  Comparing the fitted ``a``/``b`` against the step-6 etas
(``ref_eta``/``tgt_eta``) flags an amplitude/coherence mismatch (the old
eta-fixed fit could only report this as a fringe *visibility* far from 1).  The
per-row measured dark is removed before the fit, so ``d`` should sit near 0.

A second, one-time diagnostic (Table 2, :func:`build_symmetry_grid`) sweeps the
target's two channel phases *independently* on a 3x3 grid to check that phase
depends only on the sum ``phi^x + phi^w`` and amplitude only on the product
(swap invariance); see :func:`swap_invariance`.

The measurement is instrument-agnostic exactly like step 6: it drives an SLM
(``get_slm_info`` + ``display_array``) and reads whatever monitor exposes the
``ScopeController`` / ``DAQController`` shape.  Raw rows are persisted as a CSV
(one row per trial x point) so a run can be reloaded and re-fit offline.
"""
from __future__ import annotations

import csv
import json
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Fitted-parameter vector for the linear interference fit.
PARAMS: tuple[str, ...] = ("A", "B", "c")


class TPAPhaseAborted(Exception):
    """Raised when a stop_event interrupts a phase sweep."""


# ======================================================================
# per-pair step-6 model  (background + eta, used to isolate the fringe)
# ======================================================================

@dataclass(frozen=True)
class PairModel:
    """One pair's step-6 fit: eta plus the single-beam / dark background terms.

    ``single_beam`` is the linear + quadratic single-channel response WITHOUT the
    dark offset (dark is shared between pairs and handled once, per run).
    """

    index: int
    eta: float
    a_x: float
    q_x: float
    a_w: float
    q_w: float
    d: float
    eta_err: float = 0.0

    def amplitude(self, x, w):
        """Field amplitude R = eta * sqrt(x*w) (= eta * sin(phi^x/2) sin(phi^w/2))."""
        x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
        w = np.clip(np.asarray(w, dtype=float), 0.0, 1.0)
        return self.eta * np.sqrt(x * w)

    def self_tpa(self, x, w):
        """Own two-photon pedestal R^2 = eta^2 * x * w."""
        return self.amplitude(x, w) ** 2

    def single_beam(self, x, w):
        """Linear + quadratic single-beam response a_x*x + q_x*x^2 + a_w*w + q_w*w^2."""
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        return self.a_x * x + self.q_x * x**2 + self.a_w * w + self.q_w * w**2

    @classmethod
    def from_fit(cls, index: int, fit) -> "PairModel":
        """Build from a :class:`slm_module.tpa_pair.PairFit`."""
        p = fit.params
        return cls(
            index=index, eta=fit.eta, eta_err=fit.eta_err,
            a_x=p["a_x"][0], q_x=p["q_x"][0],
            a_w=p["a_w"][0], q_w=p["q_w"][0], d=p["d"][0],
        )

    @classmethod
    def from_json_channel(cls, ch: dict) -> "PairModel":
        """Build from one ``channels[]`` entry of a step-6 ``save_tpa_pair_json``."""
        fit = ch["fit"]
        p = fit["params"]
        return cls(
            index=int(ch["index"]), eta=float(fit["eta"]),
            eta_err=float(fit.get("eta_err", 0.0)),
            a_x=float(p["a_x"]["value"]), q_x=float(p["q_x"]["value"]),
            a_w=float(p["a_w"]["value"]), q_w=float(p["q_w"]["value"]),
            d=float(p["d"]["value"]),
        )


def load_pair_models(paths, *, layout=None) -> dict[int, PairModel]:
    """Load per-pair step-6 models from JSON summaries and/or raw CSVs.

    ``paths`` is one path or a sequence of paths.  ``.json`` files are read as
    step-6 ``save_tpa_pair_json`` output; any other extension is treated as a
    raw step-6 CSV and re-fit through :mod:`slm_module.tpa_pair` (so the fit is
    byte-identical to step 6).  ``layout`` is only needed for CSVs.  Later paths
    win on index collisions.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    models: dict[int, PairModel] = {}
    for path in paths:
        path = Path(path)
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            for ch in payload.get("channels", []):
                if ch.get("fit"):
                    m = PairModel.from_json_channel(ch)
                    models[m.index] = m
        else:
            from .tpa_pair import load_tpa_pair_csv
            result = load_tpa_pair_csv(path, layout=layout)
            for grid in result.channels:
                if grid.fit is not None:
                    models[grid.index] = PairModel.from_fit(grid.index, grid.fit)
    return models


# ======================================================================
# phase geometry  (intensity command <-> panel phase)
# ======================================================================

def phi_half(intensity) -> np.ndarray:
    """Half the panel phase depth of a channel, phi/2 = asin(sqrt(x)).

    ``x`` is the commanded normalised intensity (diffraction efficiency) in
    [0, 1]; the channel's field phase is exactly this value.
    """
    x = np.clip(np.asarray(intensity, dtype=float), 0.0, 1.0)
    return np.arcsin(np.sqrt(x))


def intensity_for_phase(phi_rad) -> np.ndarray:
    """Commanded intensity x = sin(phi/2)^2 for a target panel phase in [0, pi].

    Inverse of :func:`phi_half` on the reachable branch: ``phi = pi`` -> ``x = 1``.
    """
    phi = np.asarray(phi_rad, dtype=float)
    return np.sin(phi / 2.0) ** 2


def slm_phase_diff(x_t, w_t, x_r, w_r) -> np.ndarray:
    """dPhi_SLM = 1/2[(phi^x_t+phi^w_t) - (phi^x_r+phi^w_r)] from commanded intensities.

    Target (subscript t) minus reference (subscript r): for a symmetric target
    sweep against a fully-on reference this is ``phi - pi``.
    """
    return phi_half(x_t) + phi_half(w_t) - phi_half(x_r) - phi_half(w_r)


# ======================================================================
# fit  (linear least squares in A = cos dPhi_comb, B = -sin dPhi_comb, [c])
# ======================================================================

@dataclass
class PhaseFit:
    """Weighted-least-squares recovery of a, b and dPhi_comb from Y(theta2)."""

    dphi_comb: float           # radians, wrapped to (-pi, pi]
    dphi_comb_err: float
    a: float                   # reference amplitude R_1 (x_1 = w_1 = 1)
    a_err: float
    b: float                   # target amplitude scale eta_2 Cx_2 Cw_2
    b_err: float
    offset: float              # residual dark d = c0 - a^2 (should be ~0)
    offset_err: float
    chi2_red: float
    dof: int
    birge: float
    r2: float
    # point arrays the fit ran on (kept for plotting)
    dphi_slm: np.ndarray = field(repr=False)     # theta2 - pi
    g: np.ndarray = field(repr=False)            # sin(theta2/2)^2 = sqrt(x_t w_t)
    y: np.ndarray = field(repr=False)            # dark-subtracted measured Y
    sem: np.ndarray = field(repr=False)
    y_pred: np.ndarray = field(repr=False)       # full model prediction
    residuals: np.ndarray = field(repr=False)

    @property
    def dphi_comb_deg(self) -> float:
        return float(np.degrees(self.dphi_comb))


def fit_phase(
    dphi_slm: np.ndarray,
    g: np.ndarray,
    y: np.ndarray,
    sem: np.ndarray,
) -> PhaseFit:
    """Weighted LS fit of ``Y = a^2 + b^2 g^2 + 2 a b g cos(dPhi_SLM + dPhi_comb) + d``.

    ``g = sin(theta2/2)^2 = sqrt(x_t w_t)`` is the target pair-field amplitude and
    ``dPhi_SLM = theta2 - pi``.  The model is linear in the four coefficients of
    ``[1, g^2, g cos(dPhi_SLM), g sin(dPhi_SLM)]``::

        c0 = a^2 + d   c1 = b^2   c2 = 2ab cos(dPhi_comb)   c3 = -2ab sin(dPhi_comb)

    and the physical (a, b, dPhi_comb, d) follow in closed form (see module
    docstring).  Errors are covariance-propagated and Birge-scaled by
    ``sqrt(chi2/dof)`` when chi2/dof > 1.  The amplitude ``a`` is fixed by the
    interference + target self term, so it separates from the flat baseline (and
    hence from the residual dark ``d``).
    """
    dphi_slm = np.asarray(dphi_slm, dtype=float)
    g = np.asarray(g, dtype=float)
    y = np.asarray(y, dtype=float)
    sem = np.asarray(sem, dtype=float)

    cols = [np.ones_like(g), g**2, g * np.cos(dphi_slm), g * np.sin(dphi_slm)]
    A = np.column_stack(cols)

    Aw = A / sem[:, None]
    coeffs, *_ = np.linalg.lstsq(Aw, y / sem, rcond=None)
    cov = np.linalg.inv(Aw.T @ Aw)

    y_pred = A @ coeffs
    residuals = y - y_pred
    dof = max(len(y) - A.shape[1], 1)
    chi2_red = float(np.sum((residuals / sem) ** 2) / dof)
    birge = max(1.0, np.sqrt(chi2_red))
    cov = cov * birge**2

    c0, c1, c2, c3 = (float(coeffs[i]) for i in range(4))
    amp = float(np.hypot(c2, c3))                       # 2 a b
    b = float(np.sqrt(c1)) if c1 > 0 else 0.0
    a = amp / (2.0 * b) if b > 0 else float("nan")
    dphi = float(np.arctan2(-c3, c2))
    d = c0 - a**2 if np.isfinite(a) else float("nan")

    def _err(grad) -> float:
        gvec = np.asarray(grad, dtype=float)
        return float(np.sqrt(max(gvec @ cov @ gvec, 0.0)))

    if b > 0 and amp > 0:
        # gradients wrt (c0, c1, c2, c3)
        grad_b = [0.0, 1.0 / (2 * b), 0.0, 0.0]
        grad_a = [0.0, -a / (2 * c1), c2 / (2 * b * amp), c3 / (2 * b * amp)]
        grad_phi = [0.0, 0.0, c3 / amp**2, -c2 / amp**2]
        grad_d = [1.0, -2 * a * grad_a[1], -2 * a * grad_a[2], -2 * a * grad_a[3]]
        a_err, b_err = _err(grad_a), _err(grad_b)
        dphi_err, offset_err = _err(grad_phi), _err(grad_d)
    else:
        a_err = b_err = dphi_err = offset_err = float("nan")

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return PhaseFit(
        dphi_comb=dphi, dphi_comb_err=dphi_err,
        a=a, a_err=a_err, b=b, b_err=b_err,
        offset=d, offset_err=offset_err,
        chi2_red=chi2_red, dof=dof, birge=birge, r2=r2,
        dphi_slm=dphi_slm, g=g, y=y, sem=sem,
        y_pred=y_pred, residuals=residuals,
    )


# ======================================================================
# result container + fit driver
# ======================================================================

@dataclass
class PhaseResult:
    """One target pair's phase sweep against the reference, plus its fit.

    Intensities are the canonical commanded values (``x = sin(phi/2)^2``); the
    ``_t`` columns are the swept target pair, the ``_r`` columns the fixed
    reference pair.
    """

    tgt_index: int
    ref_index: int
    # raw rows, one entry per (trial, point); kept for save + re-fit
    trial: np.ndarray = field(repr=False)
    x_t: np.ndarray = field(repr=False)
    w_t: np.ndarray = field(repr=False)
    x_r: np.ndarray = field(repr=False)
    w_r: np.ndarray = field(repr=False)
    voltage_mean_v: np.ndarray = field(repr=False)
    voltage_std_v: np.ndarray = field(repr=False)
    # per-row dark measured at that row's trial start; subtracted per row before
    # averaging so per-trial dark drift is removed row-by-row (not as a constant)
    dark_v: np.ndarray = field(repr=False)
    tgt_model: PairModel | None = None
    ref_model: PairModel | None = None
    n_trials: int = 1
    fit: PhaseFit | None = None
    csv_path: str | None = None

    @property
    def dark(self) -> float:
        """Mean dark over all rows (for reporting / back-compat)."""
        return float(np.mean(self.dark_v)) if np.size(self.dark_v) else 0.0

    def per_trial_darks(self) -> np.ndarray:
        """The one dark value used for each trial (constant within a trial)."""
        out = []
        dark_v = np.asarray(self.dark_v)
        trial = np.asarray(self.trial)
        for t in range(self.n_trials):
            mask = trial == t
            if np.any(mask):
                out.append(float(dark_v[mask][0]))
        return np.asarray(out, dtype=float)


def _average_points(result: PhaseResult, dark_override: float | None = None):
    """Per-row dark-subtract, then average repeated trials per cell -> arrays + SEM.

    Each row's dark (measured at its trial's start) is removed BEFORE averaging,
    so per-trial dark drift is taken out row-by-row rather than as a single
    constant.  ``dark_override`` (a scalar) replaces the per-row dark uniformly.
    Cells seen once inherit the median positive SEM so weighting stays finite.
    The returned ``y`` is therefore already dark-subtracted.
    """
    y_raw = np.asarray(result.voltage_mean_v, dtype=float)
    if dark_override is None:
        dark_row = np.asarray(result.dark_v, dtype=float)
    else:
        dark_row = np.full(y_raw.shape, float(dark_override))
    y_sub = y_raw - dark_row

    cells: dict[tuple, list[float]] = defaultdict(list)
    key = np.column_stack([result.x_t, result.w_t, result.x_r, result.w_r])
    for row, y in zip(key, y_sub):
        cells[tuple(np.round(row, 9))].append(float(y))

    keys, ys, sem = [], [], []
    for k, vals in sorted(cells.items()):
        arr = np.asarray(vals, dtype=float)
        keys.append(k)
        ys.append(arr.mean())
        sem.append(arr.std(ddof=1) / np.sqrt(arr.size) if arr.size > 1 else np.nan)

    keys = np.asarray(keys, dtype=float)
    ys = np.asarray(ys, dtype=float)
    sem = np.asarray(sem, dtype=float)
    finite = sem[np.isfinite(sem) & (sem > 0)]
    floor = float(np.median(finite)) if finite.size else 1.0
    sem = np.where(np.isfinite(sem) & (sem > 0), sem, floor)
    return keys[:, 0], keys[:, 1], keys[:, 2], keys[:, 3], ys, sem


def fit_result(
    result: PhaseResult,
    tgt_model: PairModel,
    ref_model: PairModel,
    *,
    dark: float | None = None,
) -> PhaseFit:
    """Fit ``a``, ``b`` and ``dPhi_comb`` to the dark-subtracted Y(theta2).

    Per-row dark-subtracts and averages repeated trials per point (see
    :func:`_average_points`), then floats the full model ``Y = a^2 + b^2 g^2 +
    2ab g cos(dPhi_SLM + dPhi_comb) + d`` -- the reference/target self terms are
    fit parameters (``a^2``, ``b^2 g^2``), NOT subtracted from step-6 etas.
    ``dark`` (scalar) overrides the per-row dark uniformly.  The step-6 models are
    kept on the result only for reference (fitted ``a``/``b`` vs their etas).
    """
    x_t, w_t, x_r, w_r, y, sem = _average_points(result, dark_override=dark)

    g = np.sqrt(np.clip(x_t * w_t, 0.0, None))         # sin(theta2/2)^2, target field
    dphi_slm = slm_phase_diff(x_t, w_t, x_r, w_r)       # theta2 - pi

    result.tgt_model = tgt_model
    result.ref_model = ref_model
    result.fit = fit_phase(dphi_slm, g, y, sem)
    return result.fit


def swap_invariance(result: PhaseResult):
    """Table-2 diagnostic: |Z(x=a,w=b) - Z(x=b,w=a)| for each swap pair.

    The test runs on the CLEAN interference term, not raw Y, so the fitted self
    terms are removed first::

        Z(x,w) = Y(x,w) - a^2 - b^2 (x w) - d
               = 2 a b sqrt(x w) cos(dPhi_SLM + dPhi_comb)

    Under the bilinear model the target amplitude ``sqrt(x w)`` and ``dPhi_SLM``
    (a channel *sum*) are swap-symmetric, so ``Z`` must be too; a residual well
    above the combined SEM flags a genuine channel asymmetry (unequal per-channel
    phase/amplitude law or crosstalk).  Returns ``(x_t, w_t, z, z_swapped,
    abs_diff, sem)`` for the off-diagonal cells.  Falls back to raw Y only if the
    fit is not attached.
    """
    x_t, w_t, x_r, w_r, y, sem = _average_points(result)   # y already dark-subtracted
    fit = result.fit
    if fit is not None and np.isfinite(fit.a) and np.isfinite(fit.b):
        # clean interference: strip the fitted reference/target self terms + d
        sig = y - fit.a**2 - fit.b**2 * (x_t * w_t) - fit.offset
    else:
        sig = y

    lut = {(round(a, 9), round(b, 9)): (zz, ss)
           for a, b, zz, ss in zip(x_t, w_t, sig, sem)}
    out = []
    for a, b, zz, ss in zip(x_t, w_t, sig, sem):
        if round(a, 9) == round(b, 9):
            continue
        swapped = lut.get((round(b, 9), round(a, 9)))
        if swapped is None:
            continue
        z_sw, s_sw = swapped
        out.append((float(a), float(b), float(zz), float(z_sw),
                    abs(float(zz) - float(z_sw)), float(np.hypot(ss, s_sw))))
    return out


# ======================================================================
# drive builders
# ======================================================================

def build_phase_sweep(
    *,
    n_points: int = 15,
    phi_start_deg: float = 0.0,
    phi_stop_deg: float = 180.0,
    ref_phase_deg: float = 180.0,
) -> list[tuple[float, float, float, float]]:
    """Table 1: symmetric target phase sweep vs a fixed reference (half fringe).

    The target pair is driven symmetrically ``phi^x = phi^w = phi`` over
    ``[phi_start_deg, phi_stop_deg]`` (default 0..180 deg -- the full reachable
    half turn), the reference pair fixed at ``ref_phase_deg`` on both channels
    (default 180 deg == intensity 1, fully on).  Returns target-first commanded
    intensity tuples ``(x_t, w_t, x_r, w_r)`` with ``x = sin(phi/2)^2``, so
    ``dPhi_SLM = phi - ref_phase`` sweeps the fringe.
    """
    phis = np.radians(np.linspace(phi_start_deg, phi_stop_deg, int(n_points)))
    x_r = float(intensity_for_phase(np.radians(ref_phase_deg)))
    x_t = intensity_for_phase(phis)
    return [(float(v), float(v), x_r, x_r) for v in x_t]


def build_symmetry_grid(
    *,
    phi_values_deg: Sequence[float] = (90.0, 135.0, 180.0),
    ref_phase_deg: float = 180.0,
) -> list[tuple[float, float, float, float]]:
    """Table 2: 3x3 grid on the target's individual channel phases (symmetry check).

    Sweeps ``phi^x`` and ``phi^w`` of the target *independently* over
    ``phi_values_deg`` with the reference fixed, so swapped cells and equal-sum
    cells can be compared (see :func:`swap_invariance`).  Returns target-first
    commanded intensity tuples.
    """
    x_r = float(intensity_for_phase(np.radians(ref_phase_deg)))
    out: list[tuple[float, float, float, float]] = []
    for px in phi_values_deg:
        xt = float(intensity_for_phase(np.radians(px)))
        for pw in phi_values_deg:
            wt = float(intensity_for_phase(np.radians(pw)))
            out.append((xt, wt, x_r, x_r))
    return out


# ======================================================================
# measurement  (instrument-agnostic two-pair sweep)
# ======================================================================

@dataclass
class TPAPhaseProgress:
    step: int
    total: int
    message: str
    dphi_comb: float | None = None


ProgressCallback = Callable[["TPAPhaseProgress"], None]


def _read_mean_std(monitor, repeats: int, timeout: float) -> tuple[float, float]:
    """Averaged reading + the noise of the recorded waveform behind it."""
    means: list[float] = []
    variances: list[float] = []
    for _ in range(max(1, repeats)):
        sample = monitor.monitor_cycle(timeout=timeout)
        if sample is None:
            raise TPAPhaseAborted("monitor read aborted")
        means.append(float(sample.value))
        waveform = getattr(monitor, "last_values", None)
        if waveform is not None and np.size(waveform) > 1:
            variances.append(float(np.var(waveform)))
    mean_v = float(np.mean(means))
    std_v = float(np.sqrt(np.mean(variances))) if variances else 0.0
    return mean_v, std_v


def measure_phase_sweep(
    monitor,
    slm,
    layout,
    *,
    tgt_index: int,
    ref_index: int,
    drive: Sequence[tuple[float, float, float, float]],
    tgt_model: PairModel,
    ref_model: PairModel,
    n_trials: int = 1,
    repeats: int = 1,
    settle: float = 0.15,
    read_timeout: float = 30.0,
    measure_dark: bool = True,
    dark_per_trial: bool = True,
    col_ratio: np.ndarray | None = None,
    stop_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PhaseResult:
    """Drive target + reference over ``drive``, read Y at each point, fit dPhi_comb.

    ``monitor`` must already be configured (caller runs ``configure_monitor``);
    this only calls ``monitor_cycle``.  Only channels ``tgt_index`` and
    ``ref_index`` are driven; all others held off.  ``drive`` tuples are
    target-first ``(x_t, w_t, x_r, w_r)`` intensities.

    Dark handling: with ``measure_dark`` an all-off reading is taken and stored
    per row for per-row subtraction (drift removal).  ``dark_per_trial`` (default)
    takes a fresh all-off reading at the START OF EACH TRIAL, so slow dark drift
    over the run is tracked; set it False to take a single all-off reading once at
    the start.  If ``measure_dark`` is False the mean of the two step-6 darks is
    used for every row.  Raises :class:`TPAPhaseAborted` if ``stop_event`` is set.
    """
    n = layout.n_channels
    for name, idx in (("tgt_index", tgt_index), ("ref_index", ref_index)):
        if not (0 <= idx < n):
            raise ValueError(f"{name}={idx} out of range (layout has {n} pairs)")
    if tgt_index == ref_index:
        raise ValueError("tgt_index and ref_index must differ")

    zeros = np.zeros(n)
    slm_width, slm_height = slm.get_slm_info()
    from .encoding import encode_to_pattern

    def _check_stop() -> None:
        if stop_event is not None and stop_event.is_set():
            raise TPAPhaseAborted("phase sweep stopped by request")

    def _display(x_t, w_t, x_r, w_r) -> None:
        x_vals = zeros.copy()
        w_vals = zeros.copy()
        x_vals[tgt_index], w_vals[tgt_index] = x_t, w_t
        x_vals[ref_index], w_vals[ref_index] = x_r, w_r
        pattern = encode_to_pattern(x_vals, w_vals, layout, slm_width, slm_height,
                                    col_ratio=col_ratio)
        slm.display_array(pattern)
        if settle:
            time.sleep(settle)

    # dark handling: step-6 mean is the fallback; a measured all-off reading
    # overrides it (once, or per trial for drift tracking)
    fallback_dark = 0.5 * (tgt_model.d + ref_model.d)

    def _read_dark(trial: int, step: int, total: int) -> float:
        _check_stop()
        _display(0.0, 0.0, 0.0, 0.0)
        d, _ = _read_mean_std(monitor, repeats, read_timeout)
        if progress_callback is not None:
            progress_callback(TPAPhaseProgress(
                step=step, total=total,
                message=f"trial {trial} dark (all off) = {d*1000:.4f} mV"))
        return d

    drive = list(drive)
    reads_per_trial = len(drive) + (1 if measure_dark and dark_per_trial else 0)
    total = max(n_trials * reads_per_trial + (1 if measure_dark and not dark_per_trial else 0), 1)

    start_dark = fallback_dark
    step = 0
    if measure_dark and not dark_per_trial:
        step += 1
        start_dark = _read_dark(0, step, total)

    rows: list[tuple[int, float, float, float, float, float, float, float]] = []
    for trial in range(n_trials):
        if measure_dark and dark_per_trial:
            step += 1
            trial_dark = _read_dark(trial, step, total)
        elif measure_dark:
            trial_dark = start_dark
        else:
            trial_dark = fallback_dark
        for x_t, w_t, x_r, w_r in drive:
            _check_stop()
            _display(x_t, w_t, x_r, w_r)
            mean_v, std_v = _read_mean_std(monitor, repeats, read_timeout)
            rows.append((trial, x_t, w_t, x_r, w_r, mean_v, std_v, trial_dark))
            step += 1
            if progress_callback is not None:
                dphi_slm = float(slm_phase_diff(x_t, w_t, x_r, w_r))
                phi_t = float(np.degrees(2.0 * phi_half(x_t)))
                progress_callback(
                    TPAPhaseProgress(
                        step=step, total=total,
                        message=(
                            f"trial {trial} phi_t={phi_t:.1f}deg "
                            f"dPhi_SLM={np.degrees(dphi_slm):+.1f}deg "
                            f"-> {mean_v*1000:.4f} mV (dark {trial_dark*1000:.4f})"
                        ),
                    )
                )

    result = PhaseResult(
        tgt_index=tgt_index, ref_index=ref_index,
        trial=np.array([r[0] for r in rows], dtype=int),
        x_t=np.array([r[1] for r in rows], dtype=float),
        w_t=np.array([r[2] for r in rows], dtype=float),
        x_r=np.array([r[3] for r in rows], dtype=float),
        w_r=np.array([r[4] for r in rows], dtype=float),
        voltage_mean_v=np.array([r[5] for r in rows], dtype=float),
        voltage_std_v=np.array([r[6] for r in rows], dtype=float),
        dark_v=np.array([r[7] for r in rows], dtype=float),
        n_trials=n_trials,
    )
    fit_result(result, tgt_model, ref_model)
    if progress_callback is not None and result.fit is not None:
        progress_callback(
            TPAPhaseProgress(
                step=total, total=total,
                message=(
                    f"fit: dPhi_comb = {np.degrees(result.fit.dphi_comb):+.2f} deg "
                    f"(a = {result.fit.a:.4g}, b = {result.fit.b:.4g})"
                ),
                dphi_comb=result.fit.dphi_comb,
            )
        )
    return result


# ======================================================================
# persistence
# ======================================================================

_CSV_HEADER = [
    "trial", "tgt_index", "ref_index",
    "phi_xt_deg", "phi_wt_deg", "x_t", "w_t", "x_r", "w_r",
    "dark_v", "voltage_mean_v", "voltage_std_v",
]


def write_phase_csv(result: PhaseResult, path: str | Path) -> str:
    """Raw rows: one line per (trial, point).  Round-trips via load.

    ``phi_xt_deg`` / ``phi_wt_deg`` are the target channel phases (for readable
    comparison with the sweep tables); the fit reloads from the canonical
    intensities.  ``dark_v`` is the per-row dark (that row's trial start) used for
    per-row subtraction; the run's mean dark is also stashed as a trailing comment.
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_HEADER)
        for t, x_t, w_t, x_r, w_r, dark_v, mean_v, std_v in zip(
            result.trial, result.x_t, result.w_t, result.x_r, result.w_r,
            result.dark_v, result.voltage_mean_v, result.voltage_std_v,
        ):
            phi_xt = np.degrees(2.0 * float(phi_half(x_t)))
            phi_wt = np.degrees(2.0 * float(phi_half(w_t)))
            writer.writerow(
                [int(t), result.tgt_index, result.ref_index,
                 f"{phi_xt:.4g}", f"{phi_wt:.4g}",
                 f"{x_t:.6g}", f"{w_t:.6g}", f"{x_r:.6g}", f"{w_r:.6g}",
                 f"{dark_v:.9g}", f"{mean_v:.9g}", f"{std_v:.9g}"]
            )
    with open(out, "a", newline="", encoding="utf-8") as f:
        f.write(f"# dark_mean_v,{result.dark:.9g}\n")
    result.csv_path = str(out)
    return str(out)


def load_phase_csv(
    path: str | Path,
    tgt_model: PairModel,
    ref_model: PairModel,
    *,
    dark: float | None = None,
) -> PhaseResult:
    """Load a raw phase-sweep CSV and re-fit dPhi_comb with the given step-6 models.

    The per-row ``dark_v`` column is used when present; otherwise the scalar
    ``# dark_mean_v`` (or legacy ``# dark_v``) comment, then the step-6 mean, is
    filled for every row.  ``dark`` (scalar) overrides all of them uniformly.
    """
    file_dark: float | None = None
    with open(Path(path), newline="", encoding="utf-8") as f:
        for raw in f:
            if raw.startswith("#"):
                parts = raw.lstrip("#").strip().split(",")
                if len(parts) == 2 and parts[0].strip() in ("dark_mean_v", "dark_v"):
                    file_dark = float(parts[1])

    rows: list[tuple[int, float, float, float, float, float, float, float | None]] = []
    tgt_index, ref_index = tgt_model.index, ref_model.index
    with open(Path(path), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(line for line in f if not line.startswith("#")):
            dv = row.get("dark_v")
            rows.append((
                int(float(row.get("trial", 0))),
                float(row["x_t"]), float(row["w_t"]),
                float(row["x_r"]), float(row["w_r"]),
                float(row["voltage_mean_v"]),
                float(row.get("voltage_std_v", "nan") or "nan"),
                float(dv) if dv not in (None, "") else None,
            ))
            tgt_index = int(float(row.get("tgt_index", tgt_index)))
            ref_index = int(float(row.get("ref_index", ref_index)))

    trials = np.array([r[0] for r in rows], dtype=int)
    scalar_dark = (
        dark if dark is not None
        else file_dark if file_dark is not None
        else 0.5 * (tgt_model.d + ref_model.d)
    )
    # per-row dark: CSV column if present (and not overridden), else the scalar
    if dark is None and all(r[7] is not None for r in rows) and rows:
        dark_v = np.array([r[7] for r in rows], dtype=float)
    else:
        dark_v = np.full(len(rows), float(scalar_dark), dtype=float)

    result = PhaseResult(
        tgt_index=tgt_index, ref_index=ref_index,
        trial=trials,
        x_t=np.array([r[1] for r in rows], dtype=float),
        w_t=np.array([r[2] for r in rows], dtype=float),
        x_r=np.array([r[3] for r in rows], dtype=float),
        w_r=np.array([r[4] for r in rows], dtype=float),
        voltage_mean_v=np.array([r[5] for r in rows], dtype=float),
        voltage_std_v=np.array([r[6] for r in rows], dtype=float),
        dark_v=dark_v,
        n_trials=int(trials.max()) + 1 if trials.size else 1,
        csv_path=str(Path(path).resolve()),
    )
    fit_result(result, tgt_model, ref_model)
    return result


def save_phase_json(result: PhaseResult, path: str | Path) -> str:
    """Human-readable dPhi_comb summary (radians + degrees) and fit quality."""
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fit = result.fit
    per_trial = result.per_trial_darks()
    payload = {
        "tgt_index": result.tgt_index,
        "ref_index": result.ref_index,
        "dark_mean_v": result.dark,
        "dark_drift_std_v": float(per_trial.std(ddof=1)) if per_trial.size > 1 else 0.0,
        "n_trials": result.n_trials,
        "tgt_eta": result.tgt_model.eta if result.tgt_model else None,
        "ref_eta": result.ref_model.eta if result.ref_model else None,
        "fit": None if fit is None else {
            "dphi_comb_rad": fit.dphi_comb,
            "dphi_comb_deg": fit.dphi_comb_deg,
            "dphi_comb_err_rad": fit.dphi_comb_err,
            "dphi_comb_err_deg": float(np.degrees(fit.dphi_comb_err)),
            "a": fit.a,                 # reference amplitude R_1
            "a_err": fit.a_err,
            "b": fit.b,                 # target amplitude scale eta_2 Cx_2 Cw_2
            "b_err": fit.b_err,
            "dark_resid_v": fit.offset,  # d = c0 - a^2 (residual after dark subtraction)
            "dark_resid_err_v": fit.offset_err,
            "chi2_red": fit.chi2_red,
            "dof": fit.dof,
            "birge": fit.birge,
            "r2": fit.r2,
        },
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(out)


__all__ = [
    "PARAMS",
    "TPAPhaseAborted",
    "TPAPhaseProgress",
    "PairModel",
    "PhaseFit",
    "PhaseResult",
    "load_pair_models",
    "phi_half",
    "intensity_for_phase",
    "slm_phase_diff",
    "fit_phase",
    "fit_result",
    "swap_invariance",
    "build_phase_sweep",
    "build_symmetry_grid",
    "measure_phase_sweep",
    "write_phase_csv",
    "load_phase_csv",
    "save_phase_json",
]
