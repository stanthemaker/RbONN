"""The per-point sigma floor -- one number, one owner.

Every offline step weights its fit by ``1/sigma^2``, and the DAQ's reported
trace spread follows a shot-like law.  Fitted across the 175 usable points of
the 0903 step-6/7/8 run (two clipped points dropped)::

    sigma_trace(y) = sqrt(sigma0^2 + kappa^2 y)
    sigma0 = 90 uV        kappa = 0.264 mV^0.5      y in mV

That law is real -- a point reading near zero genuinely fluctuates less, and
the 29 all-off reads in that run sit at a median 71 uV.  But it means
``1/sigma^2`` hands the fit to whichever points sit near zero.  Across one
step-6 sweep the bright-to-dark weight ratio is 355:1, and pair 1's all-off
level alone took 45.8% of its background block.  Those points carry almost no
information about ``eta`` or ``dPhi_comb``, so a fit that is mostly about them
is a fit about nothing.

The cure is a systematic floor added in quadrature -- no point is treated as
better known than :data:`STD_FLOOR_V`, however quiet its trace was::

    sigma^2 = sigma_trace^2 + STD_FLOOR_V^2

Why 2.0e-4
----------
Sized from measurement, not taste.  Step 8 is the clean probe: it fits nothing,
so ``meas - pred`` there IS the whole error budget.  Over its eight points
below 1 mV the residual runs 12..553 uV, median ~160 uV, against a median
quoted std of ~165 uV -- and fitting ``resid^2 = f^2 + (c y)^2`` on the
one-pair blocks gives ``f ~ 0`` with ``c = 6.1%``.  So the near-zero std is
optimistic by roughly 1.5x, not by 10x, and the floor must be small enough to
say only that.

200 uV is 2.2x the fitted ``sigma0`` and ~2.8x the median dark trace std, and
it lands on the measured near-zero residual scale.  What it does to the 0903
run:

* step-6 sweep weight ratio 355x -> 60x; the all-off level's share of the
  background block 45.8% -> 23.0%, i.e. the five background levels end up
  nearly equal rather than one of them owning half the block
* step-6 ``eta`` moves <= 0.09% (pair 1: 0.207645 -> 0.207464) and ``eta_err``
  goes 1.20% -> 1.32%
* step-7 ``dPhi_comb`` moves -0.22 deg (pair 3) and -0.32 deg (pair 5), both
  well inside their ~0.6 deg statistical error
* at the bright end it costs nothing: x1.01 at 20-41 mV, x1.05 at 6 mV

So it rebalances the weights without moving any answer -- which is the whole
point.  3e-4 is still defensible (weight ratio 30x, eta_err 1.46%).  5e-4 is
not: inflating eta_err by 53% and dPhi by 1.2 deg on evidence that the sigma
was only ~1.5x optimistic is Birge rescaling under another name, and this
project retired that deliberately.

What this is NOT for
--------------------
A floor cannot rescue a *clipped* point.  A railed trace has an artificially
collapsed std -- step 7's pair-3 sweep read 103.6 mV on a +/-0.1 V range and
reported 0.956 mV where the law predicts 2.8 mV, which is how one point came to
carry 69.9% of the fit's Fisher information.  A 200 uV floor moves that to
69.1%; the actual cure is the input range (``MAX_VAL_V``) and the drive cap.
Do not reach for a bigger floor to paper over saturation.
"""

from __future__ import annotations

import numpy as np

__all__ = ["STD_FLOOR_V", "floor_std"]


#: Systematic per-point sigma floor, volts.  Added in quadrature to the trace
#: spread wherever a fit weight or a residual pull is formed.  See the module
#: docstring for how the value was measured; 0.0 disables the floor entirely
#: and restores the raw ``1/trace_std^2`` weighting.
STD_FLOOR_V = 2.0e-4


def floor_std(std, floor: float | None = None) -> np.ndarray:
    """``hypot(std, STD_FLOOR_V)`` -- the sigma a fit should actually weight by.

    ``floor`` overrides :data:`STD_FLOOR_V` for one call (tests, floor scans).
    Shape-preserving, so it drops in wherever the raw std was used; pass a
    scalar and get a 0-d array back, which floats fine.
    """
    f = STD_FLOOR_V if floor is None else float(floor)
    return np.hypot(np.asarray(std, dtype=float), f)
