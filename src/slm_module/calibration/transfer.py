"""The fitted sin^2 transfer curve -- the encoder's model of one channel.

A grayscale level ``L`` sets an LC retardance ``Delta(L)``, and an amplitude
modulator between polarizers passes::

    I(L) = floor + contrast * sin(Delta(L) / 2)^2
    Delta(L) = phase_slope * L + phase_offset

Four parameters per channel.  ``floor`` is the extinction leak the panel cannot
close, ``contrast`` the swing above it; ``Delta`` is taken linear in level,
which the data supports (see "Why linear" below).

Why fit at all
--------------
:meth:`slm_module.encoding.EncodingChannel.level_for` used to invert the
*measured* Step-3b curve point by point.  Three problems, all visible in the
0903 calibration:

1. **``v = 1`` was defined by ``argmax``.**  One noisy sample redefines full
   scale.  Between the 0828 and 0903 calibrations of the same ten channels the
   argmax peak moved by 45 levels rms -- and by 120 levels on one channel, where
   a single +6% spike at level 770 sat 9 swept levels below the real peak.  The
   fitted ``Delta = pi`` point moved 17.7 levels rms, and 10 on that channel.
2. **Every point carried the sweep's single-shot noise.**  The 4-parameter fit
   pools 36 points, so a level comes from the whole curve rather than from the
   two samples that happen to bracket the target.
3. **The delivered phase disagreed with the phase the fits assume.**
   :func:`calibration_module.phase.phi_half` takes ``phi/2 = asin(sqrt(v))``,
   i.e. that the commanded ``v`` *is* ``sin^2(Delta/2)``.  Interpolating raw
   samples between an argmin and an argmax does not make that true: on the 0903
   channels the retardance actually delivered ran up to 15 deg from the assumed
   value, against the +/-1 deg step 7 quotes for ``dPhi_comb``.

Inverting this model instead makes point 3 an identity.  ``level_for(v)`` solves
``sin^2(Delta/2) = v`` for ``Delta = 2 asin(sqrt(v))`` -- which is exactly
``2 * phi_half(v)`` -- and then inverts the linear ``Delta(L)``.  The amplitude
the encoder delivers and the phase steps 6/7/8 assume now come from one fit.

Why linear in level
-------------------
Fitted over the ten 0903 channels, the residuals of this 4-parameter model are
white: median lag-1 autocorrelation 0.095, sign-run counts on the white-noise
expectation (17 vs 19 expected).  Letting ``Delta`` take cubic terms buys ~10%
in rms and no structure.  The 2-3% rms that remains is the single-shot noise of
the Step-3b sweep, not model error -- so more parameters would fit noise.  If a
finer or averaged sweep ever resolves a real nonlinearity, this is the place to
add it, and :func:`fit_transfer_curve` is where the residual diagnostics live to
notice.

Branch convention
-----------------
``sin^2`` is periodic, so the raw fit leaves ``phase_offset`` on an arbitrary
branch.  :func:`fit_transfer_curve` normalizes it: ``phase_slope > 0`` always
(the model is invariant under ``(a, b) -> (-a, -b)``), and ``phase_offset`` is
folded so that ``Delta = 0`` lands at the darkest swept level.  The encoding
window is then always ``Delta in [0, pi]``, i.e. levels ``off_level..on_level``,
and the stored four numbers are self-contained.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Panel grayscale is 10-bit over DVI; see :mod:`slm_module.controller`.
MAX_LEVEL = 1023

__all__ = [
    "MAX_LEVEL",
    "TransferFit",
    "TransferFitError",
    "fit_transfer_curve",
    "fit_transfer_curves",
]


class TransferFitError(ValueError):
    """The sin^2 transfer model could not be fitted to a channel's curve."""


@dataclass(frozen=True)
class TransferFit:
    """``I(L) = floor + contrast * sin((phase_slope L + phase_offset)/2)^2``.

    The four fitted parameters, plus the diagnostics needed to judge them.  Only
    the four are model; everything else records what the fit was built from, so
    a stored fit can be sanity-checked without the original curve.
    """

    floor: float          # normalized power at Delta = 0 (extinction leak)
    contrast: float       # peak-to-floor swing
    phase_slope: float    # radians of retardance per grayscale level (> 0)
    phase_offset: float   # retardance at level 0, folded onto the [0, pi] branch

    # --- diagnostics (not model) ---
    level_min: int = 0    # lowest level actually swept
    level_max: int = 0    # highest level actually swept
    rms: float = 0.0      # rms residual, normalized power units
    max_abs: float = 0.0  # largest |residual|
    n_used: int = 0       # finite points the fit saw

    # ---------------------------------------------------------------- model

    def retardance(self, level) -> np.ndarray:
        """Delta(L) in radians."""
        return self.phase_slope * np.asarray(level, dtype=float) + self.phase_offset

    def intensity(self, level) -> np.ndarray:
        """Predicted normalized output power at a grayscale level."""
        return self.floor + self.contrast * np.sin(self.retardance(level) / 2.0) ** 2

    @property
    def off_level_exact(self) -> float:
        """Level where Delta = 0 -- the model's extinction point, unrounded."""
        return -self.phase_offset / self.phase_slope

    @property
    def on_level_exact(self) -> float:
        """Level where Delta = pi -- the model's full scale, unrounded."""
        return (math.pi - self.phase_offset) / self.phase_slope

    @property
    def off_level(self) -> int:
        return int(round(min(max(self.off_level_exact, 0.0), float(MAX_LEVEL))))

    @property
    def on_level(self) -> int:
        return int(round(min(max(self.on_level_exact, 0.0), float(MAX_LEVEL))))

    @property
    def extrapolated(self) -> bool:
        """True if the encoding window runs outside the levels actually swept.

        Not an error -- a channel whose peak sits past the top of the sweep is
        still encodable -- but its ``v = 1`` is an extrapolation, so it is worth
        knowing before a run rather than after.  Widening ``level_range`` in
        Step 3b is the fix.
        """
        lo = min(self.off_level_exact, self.on_level_exact)
        hi = max(self.off_level_exact, self.on_level_exact)
        return bool(lo < self.level_min - 0.5 or hi > self.level_max + 0.5)

    @property
    def clipped(self) -> bool:
        """True if the encoding window runs outside the panel's 0..1023."""
        return bool(
            self.off_level_exact < 0.0
            or self.on_level_exact > float(MAX_LEVEL)
        )

    # -------------------------------------------------------------- encoding

    def level_for(self, val: float) -> int:
        """Grayscale level delivering normalized output ``val`` in [0, 1].

        ``val`` is the fraction of this channel's fitted contrast above its
        fitted floor, so ``sin^2(Delta/2) = val`` exactly and

            Delta = 2 asin(sqrt(val)) = 2 * phase.phi_half(val)

        which is the retardance the downstream step-6/7/8 model assumes for a
        channel commanded at ``val``.  Inverting the linear ``Delta(L)`` gives
        the level; the result is rounded to an integer grayscale and clamped to
        the panel's range, so it is not restricted to the swept levels.
        """
        v = float(np.clip(val, 0.0, 1.0))
        theta = 2.0 * math.asin(math.sqrt(v))          # in [0, pi]
        level = (theta - self.phase_offset) / self.phase_slope
        return int(round(min(max(level, 0.0), float(MAX_LEVEL))))

    # ----------------------------------------------------------------- io

    def to_dict(self) -> dict:
        return {
            "floor": float(self.floor),
            "contrast": float(self.contrast),
            "phase_slope": float(self.phase_slope),
            "phase_offset": float(self.phase_offset),
            "level_min": int(self.level_min),
            "level_max": int(self.level_max),
            "rms": float(self.rms),
            "max_abs": float(self.max_abs),
            "n_used": int(self.n_used),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TransferFit":
        try:
            return cls(
                floor=float(payload["floor"]),
                contrast=float(payload["contrast"]),
                phase_slope=float(payload["phase_slope"]),
                phase_offset=float(payload["phase_offset"]),
                level_min=int(payload.get("level_min", 0)),
                level_max=int(payload.get("level_max", 0)),
                rms=float(payload.get("rms", 0.0)),
                max_abs=float(payload.get("max_abs", 0.0)),
                n_used=int(payload.get("n_used", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TransferFitError(f"malformed transfer fit payload: {exc}") from exc


def _model(level, floor, contrast, slope, offset):
    return floor + contrast * np.sin((slope * level + offset) / 2.0) ** 2


def fit_transfer_curve(
    levels: np.ndarray,
    values: np.ndarray,
    *,
    robust: bool = True,
) -> TransferFit:
    """Fit one channel's measured curve to the 4-parameter sin^2 model.

    ``levels`` are the swept grayscale levels and ``values`` the normalized
    output power at each (``CalibrationResult.intensity_levels`` rows).  NaNs
    are dropped.

    Constrained, not free-running: ``floor >= 0`` because a normalized power
    cannot be negative (unconstrained, all ten 0903 channels want a slightly
    negative floor -- the reference normalization over-subtracts by ~1%), and
    ``contrast >= 0``.  Pinning the floor at zero where the fit wants to go
    below it is also what keeps ``v -> Delta`` honest for the downstream phase
    model, which has no floor term to absorb it.

    ``robust`` uses a soft-L1 loss so one spiked cell cannot drag the curve --
    the failure mode this whole module exists to fix.  Turn it off for a plain
    least-squares fit.

    Raises :class:`TransferFitError` if there is nothing to fit or the optimizer
    will not converge.  There is deliberately no polynomial fallback: a channel
    whose transfer curve is not sin^2-shaped is a channel to investigate, not to
    silently encode against a shape with no phase meaning.
    """
    levels = np.asarray(levels, dtype=float)
    values = np.asarray(values, dtype=float)
    if levels.shape != values.shape or levels.ndim != 1:
        raise TransferFitError("levels and values must be matching 1-D arrays")

    finite = np.isfinite(levels) & np.isfinite(values)
    if int(finite.sum()) < 5:
        raise TransferFitError(
            f"need at least 5 finite points to fit 4 parameters, got {int(finite.sum())}"
        )
    lv = levels[finite]
    val = values[finite]

    span = float(np.ptp(lv))
    swing = float(np.ptp(val))
    if span <= 0.0 or swing <= 0.0:
        raise TransferFitError("curve is flat in level or in value; nothing to fit")

    # seed: the darkest swept level is Delta ~ 0, the brightest Delta ~ pi
    lo = float(lv[int(np.argmin(val))])
    hi = float(lv[int(np.argmax(val))])
    slope0 = math.pi / (hi - lo) if hi != lo else math.pi / span
    seed = (max(float(np.min(val)), 0.0), swing, slope0, -slope0 * lo)
    bounds = (
        (0.0, 0.0, -np.inf, -np.inf),
        (max(float(np.max(val)), 1e-9), 5.0 * swing, np.inf, np.inf),
    )

    try:
        from scipy.optimize import curve_fit

        kwargs = {}
        if robust:
            # f_scale ~ the noise we expect; residuals past it are down-weighted
            kwargs = {"loss": "soft_l1", "f_scale": max(1e-4, 0.05 * swing)}
        popt, _ = curve_fit(
            _model, lv, val, p0=seed, bounds=bounds, method="trf",
            maxfev=40000, **kwargs,
        )
    except TransferFitError:
        raise
    except Exception as exc:                       # convergence, import, ...
        raise TransferFitError(f"sin^2 transfer fit did not converge: {exc}") from exc

    floor, contrast, slope, offset = (float(v) for v in popt)
    if not all(math.isfinite(p) for p in (floor, contrast, slope, offset)):
        raise TransferFitError("sin^2 transfer fit returned non-finite parameters")
    if slope == 0.0 or contrast <= 0.0:
        raise TransferFitError(
            f"degenerate transfer fit (slope {slope:.3g}, contrast {contrast:.3g})"
        )

    # --- normalize the branch (see the module docstring) ---
    # The model is invariant under (slope, offset) -> (-slope, -offset), so take
    # slope > 0; then fold offset by whole periods so Delta = 0 sits at the
    # darkest swept level, putting the encoding window on Delta in [0, pi].
    if slope < 0.0:
        slope, offset = -slope, -offset
    k = round((slope * lo + offset) / (2.0 * math.pi))
    offset -= 2.0 * math.pi * k

    resid = val - _model(lv, floor, contrast, slope, offset)
    return TransferFit(
        floor=floor,
        contrast=contrast,
        phase_slope=slope,
        phase_offset=offset,
        level_min=int(round(float(np.min(lv)))),
        level_max=int(round(float(np.max(lv)))),
        rms=float(np.sqrt(np.mean(resid ** 2))),
        max_abs=float(np.max(np.abs(resid))),
        n_used=int(lv.size),
    )


def fit_transfer_curves(
    levels: np.ndarray,
    curves: np.ndarray,
    *,
    robust: bool = True,
) -> list[TransferFit]:
    """Fit every row of an ``intensity_levels`` array; one TransferFit per channel.

    Raises :class:`TransferFitError` naming the row that failed, so a bad
    channel is identifiable without re-running the fits one at a time.
    """
    curves = np.asarray(curves, dtype=float)
    if curves.ndim != 2:
        raise TransferFitError("curves must be a 2-D (channel, level) array")
    out: list[TransferFit] = []
    for row in range(curves.shape[0]):
        try:
            out.append(fit_transfer_curve(levels, curves[row], robust=robust))
        except TransferFitError as exc:
            raise TransferFitError(f"channel row {row}: {exc}") from exc
    return out
