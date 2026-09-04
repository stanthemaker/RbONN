"""Step 6 v2 -- per-pair TPA efficiency eta from the DIFFERENCE estimator.

Needs real hardware for the sweep; the refit path is offline::

    python src/calibration_module/steps/calib_step6_v2.py            # sweep, fit, plot
    python src/calibration_module/steps/calib_step6_v2.py --meas     # raw CSV only, no fit
    python src/calibration_module/steps/calib_step6_v2.py some.csv   # re-fit a CSV offline
    python src/calibration_module/steps/calib_step6_v2.py some.csv --anchor   # + D(0) in the fit

The underlying physics is unchanged from v1::

    Y(x, w) = eta^2*(x*w) + a_x*x + q_x*x^2 + a_w*w + q_w*w^2 + d

What changes is how eta is *extracted*.

Estimator
---------
Along the cross line x = 1, subtract the w-only background::

    D(w) = Y(1, w) - Y(0, w) = eta^2*w + (a_x + q_x)

``a_w``, ``q_w`` and ``d`` cancel identically -- they play no part in eta.
``a_x`` and ``q_x`` survive only as their sum, the intercept ``beta0``.  eta is
then the slope of a two-parameter straight line, so no assumption is needed
about the shape of the single-beam background and the q-term debate disappears.

The q columns are nonetheless OFF by default (:data:`FIT_Q`).  With two drive
levels per arm plus dark, fitting them saturates the background block, and a
saturated block splits one measured curve between ``a`` and ``q`` arbitrarily --
which is why it kept returning a negative ``a`` or ``q``, neither of which a
single-beam response can be.  Dropping them leaves ``a_x*x + a_w*w + d`` with
2 dof, whose residual is printed with every fit.

Why this replaces v1's joint 6-parameter fit: there ``d`` is shared across all
three lines, so misfit on the x-only line leaks into ``d`` and from there into
the cross line's intercept/slope split.  Measured leak on pair 0 -- dropping
``q_x`` moved eta by +0.55% / +0.61%.  The difference estimator is immune by
construction.

How the difference is formed
----------------------------
The spec allows either a point-by-point difference or a subtraction against the
fitted background curve.  **This grid only supports the fitted curve**: the
w-only block sits at w in {0.5, 1.0} while the cross block sits at
w in {0.2, 0.45, 0.7, 0.9}, so ``Y(0, w)`` is never measured at a cross w and
there is no point-by-point partner to subtract.  So::

    D(w) = Y(1, w) - Bhat(w),   Bhat(w) = a_w*w + d   [+ q_w*w^2 with FIT_Q]

``Bhat`` is fitted, which means every D shares its parameters and the D points
are *correlated*.  That correlation is carried properly: the background
covariance is propagated into a full covariance for D and the slope comes from
a generalized least squares fit, not an ordinary one.  Getting this wrong would
re-introduce exactly the slope/intercept leak the estimator exists to remove.

Fit procedure
-------------
1. Background block (dark + x-only + w-only, 5 levels) gives the
   :data:`PARAMS_BG` parameters with their covariance -- 3 of them and 2 dof by
   default, 5 and saturated with :data:`FIT_Q`.  These are needed only for the
   step-7/8 forward model, never for eta.
2. Form ``D(w)`` on the cross levels, propagating the background covariance.
3. Restrict to ``w`` in ``FIT_W_RANGE`` and fit ``D(w) = eta^2*w + beta0`` by
   GLS; ``eta = sqrt(eta^2)``.
4. Report the fit range with eta -- the cross line is not exactly straight, so
   eta is a *local* slope, not a global constant.

Verification
-----------
Four checks are printed with every fit.  **None of them feeds the fit** -- they
say whether the model eta is defined within still holds on this pair.

Two come free from the existing grid:

* **anchor** -- ``D(0)`` is measured directly as ``Y(1,0) - Y(0,0)`` and must
  agree with the fitted intercept ``beta0``.
* **top-drive compression** -- the ``(1, 1.0)`` level is excluded from the slope
  fit and compared against the extrapolated line, which is where v1's largest
  residual lived (pair 0: -3.44 mV on exactly this point).

Two are the explicit verification section:

1. **Intercept identity** -- ``D(w) = eta^2 w + (a_x + q_x)`` says the cross
   line extrapolated to ``w = 0`` must land on the x-side response measured
   independently in the x-only block (``a_x`` alone without the q columns).
   Curvature over the fit window shows up here first, because extrapolating to
   zero amplifies it.  With :data:`FIT_Q` the background block is saturated, so
   that sum equals the raw ``Y(1,0) - Y(0,0)`` exactly and the check only
   restates the anchor through the fitted parameters; without it the block
   carries 2 dof and the two genuinely differ, so the check has teeth.

2. **Product-only dependence** -- the model says ``Y`` depends on the two drives
   only through ``x*w``.  ``(1, 0.25)`` and ``(0.5, 0.5)`` share that product but
   split the drive very differently, so after subtracting the full single-beam
   background their TPA residues must agree.  This costs the six extra
   acquisitions in :data:`VERIFY_GRID` (31 -> 37, about +1 min/pair), and it is
   worth them: a split here does not mean this estimator is wrong, it means the
   pair has **no single eta at all**, which would invalidate the
   one-number-per-pair model steps 7 and 8 are built on.

   The slope fit alone will not reliably catch that.  On synthetic data with a
   10% drive-split tilt injected, R^2 still came out 0.9995 and eta moved only
   0.26% -- the tilt is nearly linear in w along ``x = 1``, so it is absorbed
   into the slope and intercept.  The residual pulls did rise (max 7.2), but
   only the product check names the cause, at +6.8% split / 9.1 sigma.

Set ``VERIFY_ENABLED = False`` for the bare 31-acquisition grid; the product
check then reports itself as not measured rather than silently passing.

Uncertainties
-------------
Every level is repeated, and the SLM is rewritten between repeats, so the
scatter across repeats measures *encoding* repeatability and not just detector
jitter.  A level's sigma is therefore built from that scatter::

    sigma = hypot(max(rep_std, trace_std) / sqrt(n_repeats), STD_FLOOR_V)

``trace_std`` (the low-passed trace spread that v1 weights by) is kept only as a
floor, so two repeats that happen to agree cannot manufacture an infinite
weight.  :data:`~calibration_module.sigma.STD_FLOOR_V` is the systematic floor
on top of that: repeats average down statistical noise but not a systematic, and
without it the all-off level -- the quietest read in the grid, since the trace
spread scales as sqrt(signal) -- takes ~46% of the background block on its own.

The ``/sqrt(n)`` is *not* the retired trace-SEM: repeats are genuinely independent
acquisitions with a panel rewrite in between, so dividing by sqrt(n) is
legitimate -- and it is what the grid intends, since the highest-leverage cross
level carries n = 6.  Every fit prints rep_std beside trace_std per level so
the two can be compared directly.

Acquisition order
-----------------
Repeats are interleaved: the schedule runs round-robin passes over the grid
rather than n back-to-back reads of one level, so a slow drift cannot
masquerade as a slope.  Within each pass levels run brightest first, so
``(1, 1.0)`` is the very first acquisition and a dead or blocked beam shows
immediately.

The CSV keeps v1's column layout, so the same data can be re-fit with v1's
joint 6-parameter estimator for comparison::

    python src/calibration_module/steps/calib_step6_v1.py <a_v2_meas.csv>

That is the cross-check that justifies the change: same rows, two estimators.

Output is one ``calib_step6v2_result_MMDD_HHMM.json`` embedding the input
Step-3 calibration, and carrying ``a_x, q_x, a_w, q_w, d`` and ``eta`` per pair
(a q that was not fitted is written out as an exact zero)
in the schema step 7 already reads (``PairModel.from_json_channel``).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for draft_hw

from calibration_module.sigma import STD_FLOOR_V, floor_std  # noqa: E402
from draft_hw import connect_daq, connect_slm, read_point  # noqa: E402
from slm_module.calibration.calibration_new import load_calibration_result  # noqa: E402
from slm_module.encoding import channel_layout_from_calibration  # noqa: E402

# ---- Edit these to match your setup ----
CALIB_PATH = REPO_ROOT / "src/calib_data"   # data directory: inputs + outputs live here

# Pair numbering.  Pairs are labelled from PAIR_INDEX_BASE and that label is what
# every CSV / JSON records; the Step-3 layout arrays are always 0-based, so
# _slot() converts.  Set PAIR_INDEX_BASE = 0 for the old 0-based convention.
# Keep it in step with calib_step7_v2.PAIR_INDEX_BASE -- step 7 reads this
# script's pair labels straight out of the step-6 JSON.
PAIR_INDEX_BASE = 1                         # pairs are numbered 1..N
PAIR_INDICES = [4]                    # pair labels to calibrate
IN_STEP3 = CALIB_PATH / "run_0903" / "calib_step3b_0903_1517.json"   # Step 3 calib

SLM_DISPLAY_NO = None            # None -> auto-detect the LCOS-SLM display
USB_SLM_NO = 1                   # SLM_Ctrl_* device index for the DVI-mode switch

DAQ_DEVICE = "Dev1"
DAQ_CHANNEL = "ai0"

# ---- Fixed per-point acquisition (daq_module) ----
# Sample rate / range / low-pass are the DAQMonitorSettings defaults (1 kS/s,
# +/-0.1 V DIFF, 20 Hz).  T_SINGLE_S when at most one beam is on, else T_BOTH_S.
T_SINGLE_S = 10.0
T_BOTH_S = 10.0
SETTLE_S = 0.25                  # wait after each SLM pattern change, before reading

# ---- The measurement grid: (x, w, n_repeats) ----
# 31 acquisitions for the estimator; the verification block below adds 6 more.
# Every level is repeated (n >= 2) with an SLM rewrite between repeats, so the
# repeat scatter measures encoding repeatability rather than detector jitter.
GRID: tuple[tuple[float, float, int], ...] = (
    (0.0, 0.00, 2),   # dark    -> d
    (0.5, 0.00, 2),   # x-only  -> a_x, q_x
    (1.0, 0.00, 3),   # x-only  -> a_x, q_x ; also the direct D(0) anchor
    (0.0, 0.50, 2),   # w-only  -> a_w, q_w
    (0.0, 1.00, 2),   # w-only  -> a_w, q_w
    (1.0, 0.20, 4),   # cross   -> eta^2
    (1.0, 0.45, 4),   # cross
    (1.0, 0.70, 4),   # cross
    (1.0, 0.90, 6),   # cross   (highest leverage -> most repeats)
    (1.0, 1.00, 2),   # cross, EXCLUDED from the slope fit: top-drive compression
)

# Slope-fit window in w.  The cross line is not exactly straight, so eta is a
# local slope over this range -- it is reported alongside eta, and the levels
# outside it become the compression diagnostic instead of being dropped.
FIT_W_RANGE = (0.2, 0.9)

# ---- Verification block ----
# Extra levels measured purely to test the model's assumptions, never to fit
# eta.  They cost 6 more acquisitions (31 -> 37, roughly +1 min/pair).
#
# PRODUCT_CHECKS lists groups of levels that share the same product x*w.  The
# model says Y depends on the two drives ONLY through that product, so once the
# single-beam background is subtracted the groups' members must agree.  They sit
# at different x, which is exactly the point: (1, 0.25) drives the x side hard
# and the w side softly, (0.5, 0.5) splits the drive evenly.  If those disagree,
# the pair's response is not a function of x*w alone and no single eta describes
# it -- which would undermine the whole one-number-per-pair model, not just this
# estimator.
PRODUCT_CHECKS: tuple[tuple[tuple[float, float], ...], ...] = (
    ((1.0, 0.25), (0.5, 0.50)),      # both x*w = 0.25
)
VERIFY_GRID: tuple[tuple[float, float, int], ...] = (
    (1.0, 0.25, 3),
    (0.5, 0.50, 3),
)
VERIFY_ENABLED = True            # False -> the bare 31-acquisition grid

# The verification levels stay OUT of the slope fit by default, so the estimator
# sees exactly the levels it was specified with and the checks stay independent
# of what they are checking.  (1, 0.25) is a legitimate cross point and could
# join the fit; flip this to let it.
VERIFY_IN_SLOPE_FIT = False

# ---- Single-beam background model ----
# FIT_Q keeps the quadratic saturation columns q_x*x^2 and q_w*w^2.
#
# Off by default.  The grid has only TWO drive levels per arm plus the dark
# level, so with the q columns the background block is exactly saturated:
# a_w and q_w are pinned by two points, and one measured curve can be split
# between them any way at all.  Noise then routinely lands on a negative a or a
# negative q -- a single-beam response cannot go DOWN as its drive goes up, so
# that split was never measured, only interpolated.  Dropping the q columns
# fits ``Y_bg = a_x*x + a_w*w + d``: 5 levels, 3 parameters, 2 dof, and the
# background residual becomes visible instead of being absorbed into a q.
#
# What it costs: eta only ever sees ``Bhat(w)``, so this moves eta only if the
# w-only arm is genuinely curved -- and with 2 dof that curvature now shows up
# in the printed background residual instead of hiding in q_w.  If those pulls
# go large, add w levels (so a and q are separately measured) rather than just
# flipping FIT_Q back on, which only returns to the arbitrary split.
FIT_Q = False

# Background parameters, in the order the design matrix columns are built.
PARAMS_BG = ("a_x", "q_x", "a_w", "q_w", "d") if FIT_Q else ("a_x", "a_w", "d")

# The x-side terms -- what the difference's intercept measures, since the
# cross line sits at x = 1:  D(w) = eta^2 w + (a_x + q_x), or + a_x alone.
X_SIDE = tuple(n for n in PARAMS_BG if n in ("a_x", "q_x"))
X_SIDE_LABEL = " + ".join(X_SIDE)
BHAT_LABEL = "a_w w + q_w w^2 + d" if FIT_Q else "a_w w + d"

_CSV_HEADER = [
    "trial", "pair_index", "x", "w", "product",
    "voltage_mean_v", "voltage_std_v", "std_ratio",
]

_BLOCK_ORDER = {"dark": 0, "x-only": 1, "w-only": 2, "cross": 3}


# ======================================================================
# levels  (repeat averaging)
# ======================================================================

def block_of(x: float, w: float) -> str:
    """Which block of the grid a level belongs to."""
    if x == 0.0 and w == 0.0:
        return "dark"
    if w == 0.0:
        return "x-only"
    if x == 0.0:
        return "w-only"
    return "cross"


def full_grid() -> tuple[tuple[float, float, int], ...]:
    """Everything actually measured: the estimator's levels + the checks."""
    return GRID + VERIFY_GRID if VERIFY_ENABLED else GRID


def verification_points() -> set[tuple[float, float]]:
    """Levels measured only to test the model, never to fit eta."""
    if not VERIFY_ENABLED or VERIFY_IN_SLOPE_FIT:
        return set()
    return {(round(x, 6), round(w, 6)) for x, w, _ in VERIFY_GRID}


@dataclass
class Level:
    """One commanded ``(x, w)`` setting, averaged over its repeats.

    ``rep_std`` is the scatter of the repeat means -- encoding repeatability
    plus drift plus detector noise.  ``trace_std`` is the mean within-acquisition
    low-passed trace spread -- detector noise only.  Comparing the two is the
    point of repeating every level; ``sigma`` is what the fits actually weight
    by (see the module docstring).
    """

    x: float
    w: float
    means: np.ndarray          # one entry per repeat, volts
    trace_stds: np.ndarray     # one entry per repeat, volts

    @property
    def n(self) -> int:
        return int(self.means.size)

    @property
    def block(self) -> str:
        return block_of(self.x, self.w)

    @property
    def mean(self) -> float:
        return float(np.mean(self.means))

    @property
    def rep_std(self) -> float:
        """Sample std across repeats (nan for a single acquisition)."""
        if self.n < 2:
            return float("nan")
        return float(np.std(self.means, ddof=1))

    @property
    def trace_std(self) -> float:
        return float(np.mean(self.trace_stds))

    @property
    def sigma(self) -> float:
        """Uncertainty of this level's mean: trace spread, then the systematic.

        The ``/sqrt(n)`` is legitimate (independent acquisitions with a panel
        rewrite between them), but it applies only to what averaging can reduce.
        STD_FLOOR_V goes on AFTER it: a systematic does not average away, and
        no level should read as better known than the floor however many
        repeats it got.
        """
        rep = self.rep_std
        spread = max(rep if np.isfinite(rep) else 0.0, self.trace_std)
        sigma = spread / np.sqrt(self.n)
        sigma = sigma if sigma > 0.0 else 1e-12   # never an infinite weight
        return float(floor_std(sigma))

    @property
    def is_verification(self) -> bool:
        """Measured only to check the model (see :data:`PRODUCT_CHECKS`)."""
        return (round(self.x, 6), round(self.w, 6)) in verification_points()

    def on_cross_line(self) -> bool:
        """On the ``x = 1`` line the difference estimator is derived for.

        ``D(w) = Y(1,w) - Y(0,w)`` assumes ``x = 1``; a cross level at any other
        x is not on this line at all and must never reach the slope fit.
        """
        return self.block == "cross" and self.x == 1.0 and not self.is_verification

    def in_fit_window(self) -> bool:
        """Cross level inside ``FIT_W_RANGE`` -- i.e. one the slope fit uses."""
        lo, hi = FIT_W_RANGE
        return self.on_cross_line() and lo - 1e-9 <= self.w <= hi + 1e-9


def average_levels(rows) -> list[Level]:
    """Group raw ``(rep, x, w, mean_v, std_v)`` rows into repeat-averaged levels."""
    grouped: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for _rep, x, w, mean_v, std_v in rows:
        grouped.setdefault((round(float(x), 6), round(float(w), 6)), []).append(
            (float(mean_v), float(std_v))
        )
    levels = [
        Level(
            x=x, w=w,
            means=np.array([m for m, _ in vals], dtype=float),
            trace_stds=np.array([s for _, s in vals], dtype=float),
        )
        for (x, w), vals in grouped.items()
    ]
    levels.sort(key=lambda L: (_BLOCK_ORDER[L.block], L.x, L.w))
    return levels


# ======================================================================
# the fit
# ======================================================================

def design_bg(x, w) -> np.ndarray:
    """Background design matrix, one column per :data:`PARAMS_BG` entry.

    ``[x, x^2, w, w^2, 1]`` with :data:`FIT_Q`, ``[x, w, 1]`` without.  The
    order always follows ``PARAMS_BG`` so the covariance rows line up with the
    parameter names.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    w = np.atleast_1d(np.asarray(w, dtype=float))
    cols = {"a_x": x, "q_x": x**2, "a_w": w, "q_w": w**2, "d": np.ones_like(x)}
    return np.column_stack([cols[n] for n in PARAMS_BG])


def grad_bg_w(w) -> np.ndarray:
    """d/dparams of the w-only background ``Bhat(w)`` (:data:`BHAT_LABEL`).

    Same parameter order as :data:`PARAMS_BG`; the x-side columns are zero --
    that is exactly why ``a_x``/``q_x`` cancel out of the difference.
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    z = np.zeros_like(w)
    cols = {"a_x": z, "q_x": z, "a_w": w, "q_w": w**2, "d": np.ones_like(w)}
    return np.column_stack([cols[n] for n in PARAMS_BG])


@dataclass
class PairV2Fit:
    """Everything the v2 estimator produces for one pair."""

    index: int
    levels: list[Level]
    # -- background block
    bg: dict[str, tuple[float, float]]      # name -> (value, err)
    bg_cov: np.ndarray
    # -- difference fit (the eta estimator)
    fit_w: np.ndarray                       # cross levels used
    fit_d: np.ndarray                       # measured D(w)
    fit_sigma: np.ndarray                   # sqrt of the diagonal of cov_d
    fit_pred: np.ndarray
    b: float                                # eta^2, the slope
    b_err: float
    beta0: float                            # intercept = a_x + q_x
    beta0_err: float
    eta: float
    eta_err: float
    r2: float
    # -- checks
    anchor: tuple[float, float]             # measured D(0) +/- err
    excluded: list[tuple[float, float, float, float, float]]
    # (w, D_meas, D_meas_err, D_extrapolated, D_extrapolated_err)
    checks: dict = field(default_factory=dict)   # see verify_intercept/_product
    wl_x_nm: float = float("nan")
    wl_w_nm: float = float("nan")
    nominal_wl_nm: float = float("nan")

    @property
    def residuals(self) -> np.ndarray:
        return self.fit_d - self.fit_pred

    @property
    def pulls(self) -> np.ndarray:
        return self.residuals / self.fit_sigma

    def bg_residuals(self) -> tuple[list["Level"], np.ndarray, np.ndarray]:
        """Background-block levels with their fit residuals and pulls.

        Returns empty arrays when the block is saturated (:data:`FIT_Q`): a
        zero-dof fit passes exactly through its points, so the residual is
        identically zero and says nothing.  With the q columns dropped there
        are 2 dof, and these pulls are the evidence that the straight
        single-beam background is good enough -- large ones mean the arm is
        genuinely curved and needs more w levels, not a re-fitted q.
        """
        used = [L for L in self.levels if L.block != "cross"]
        if len(used) <= len(PARAMS_BG):
            return used, np.zeros(0), np.zeros(0)
        p = np.array([self.bg[n][0] for n in PARAMS_BG])
        resid = (np.array([L.mean for L in used])
                 - design_bg([L.x for L in used], [L.w for L in used]) @ p)
        return used, resid, resid / np.array([L.sigma for L in used])

    def background(self, x, w) -> np.ndarray:
        """Full single-beam background at ``(x, w)`` -- for plots and step 7/8."""
        p = {k: v[0] for k, v in self.bg.items()}
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        return (p["a_x"] * x + p.get("q_x", 0.0) * x**2
                + p["a_w"] * w + p.get("q_w", 0.0) * w**2 + p["d"])

    def background_w(self, w) -> np.ndarray:
        """The w-only background ``Bhat(w)`` that the difference subtracts."""
        p = {k: v[0] for k, v in self.bg.items()}
        w = np.asarray(w, dtype=float)
        return p["a_w"] * w + p.get("q_w", 0.0) * w**2 + p["d"]


def fit_background(levels: list[Level]) -> tuple[dict[str, tuple[float, float]], np.ndarray]:
    """WLS the single-beam block -> ``(params, covariance)``.

    Uses the dark, x-only and w-only levels only: five distinct settings for
    the :data:`PARAMS_BG` parameters.  With :data:`FIT_Q` that is five for five
    -- saturated, passing exactly through the level means, and the a/q split on
    each arm is interpolation rather than measurement.  Without it, three
    parameters and 2 dof, so the block is genuinely over-determined and its
    residual is a real check (see :meth:`PairV2Fit.bg_residuals`).

    Either way the covariance comes entirely from the level sigmas, which is
    what makes the propagation into D honest.
    """
    used = [L for L in levels if L.block != "cross"]
    if len(used) < len(PARAMS_BG):
        raise ValueError(
            f"background block has {len(used)} levels, need >= {len(PARAMS_BG)}"
        )
    a = design_bg([L.x for L in used], [L.w for L in used])
    y = np.array([L.mean for L in used], dtype=float)
    sigma = np.array([L.sigma for L in used], dtype=float)

    aw = a / sigma[:, None]
    yw = y / sigma
    params, *_ = np.linalg.lstsq(aw, yw, rcond=None)
    cov = np.linalg.pinv(aw.T @ aw)
    errs = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return {n: (float(params[i]), float(errs[i])) for i, n in enumerate(PARAMS_BG)}, cov


def fit_difference(
    levels: list[Level],
    bg: dict[str, tuple[float, float]],
    bg_cov: np.ndarray,
    *,
    include_anchor: bool = False,
) -> dict:
    """GLS ``D(w) = eta^2*w + beta0`` over the in-window cross levels.

    ``D`` is formed against the *fitted* background, so all its points share the
    background parameters and are correlated.  The covariance is built as::

        Cov(D) = diag(sigma_Y^2) + G Cov(bg) G^T

    with ``G`` the gradient of ``Bhat(w)`` -- and the slope is then a
    generalized least squares solution against that full matrix.  Treating the
    points as independent here would bias the slope/intercept split, which is
    the very thing the difference estimator exists to avoid.

    ``include_anchor`` additionally puts the directly measured
    ``D(0) = Y(1,0) - Y(0,0)`` into the fit at ``w = 0``, pinning the intercept
    on measured data rather than on extrapolation.  Off by default: the spec
    fits the window only and keeps ``D(0)`` as an independent check.
    """
    used = [L for L in levels if L.in_fit_window()]
    if len(used) < 2:
        raise ValueError(f"need >= 2 cross levels in {FIT_W_RANGE}, got {len(used)}")

    w = np.array([L.w for L in used], dtype=float)
    y = np.array([L.mean for L in used], dtype=float)
    y_sig = np.array([L.sigma for L in used], dtype=float)

    g = grad_bg_w(w)
    d = y - (g @ np.array([bg[n][0] for n in PARAMS_BG]))
    cov_d = np.diag(y_sig**2) + g @ bg_cov @ g.T

    if include_anchor:
        dark = _level_at(levels, 0.0, 0.0)
        x_on = _level_at(levels, 1.0, 0.0)
        if dark is not None and x_on is not None:
            # D(0) = Y(1,0) - Y(0,0), measured directly: no background model in
            # it at all, so it is uncorrelated with the other D points.
            w = np.append(w, 0.0)
            d = np.append(d, x_on.mean - dark.mean)
            extra = x_on.sigma**2 + dark.sigma**2
            cov_d = np.pad(cov_d, ((0, 1), (0, 1)))
            cov_d[-1, -1] = extra
            y_sig = np.append(y_sig, np.sqrt(extra))

    a = np.column_stack([w, np.ones_like(w)])
    cov_inv = np.linalg.pinv(cov_d)
    m = np.linalg.pinv(a.T @ cov_inv @ a)
    beta = m @ (a.T @ cov_inv @ d)
    pred = a @ beta

    resid = d - pred
    chi2 = float(resid @ cov_inv @ resid)
    one = np.ones_like(d)
    d_bar = float((one @ cov_inv @ d) / (one @ cov_inv @ one))
    total = float((d - d_bar) @ cov_inv @ (d - d_bar))
    r2 = 1.0 - chi2 / total if total > 0 else float("nan")

    return {
        "w": w, "d": d, "sigma": np.sqrt(np.clip(np.diag(cov_d), 0.0, None)),
        "pred": pred, "beta": beta, "cov": m, "r2": r2, "used": used,
    }


def _level_at(levels: list[Level], x: float, w: float) -> Level | None:
    for L in levels:
        if abs(L.x - x) < 1e-9 and abs(L.w - w) < 1e-9:
            return L
    return None


def fit_pair(index: int, levels: list[Level], *, include_anchor: bool = False) -> PairV2Fit:
    """Background block, then the difference slope; plus both consistency checks."""
    bg, bg_cov = fit_background(levels)
    df = fit_difference(levels, bg, bg_cov, include_anchor=include_anchor)

    b, beta0 = float(df["beta"][0]), float(df["beta"][1])
    b_err = float(np.sqrt(max(df["cov"][0, 0], 0.0)))
    beta0_err = float(np.sqrt(max(df["cov"][1, 1], 0.0)))
    if b > 0:
        eta = float(np.sqrt(b))
        eta_err = b_err / (2.0 * eta)        # sigma_eta = sigma_b / 2 eta
    else:
        eta, eta_err = float("nan"), float("nan")

    # anchor: D(0) straight off the measured levels, no background model in it
    dark = _level_at(levels, 0.0, 0.0)
    x_on = _level_at(levels, 1.0, 0.0)
    if dark is not None and x_on is not None:
        anchor = (x_on.mean - dark.mean,
                  float(np.hypot(x_on.sigma, dark.sigma)))
    else:
        anchor = (float("nan"), float("nan"))

    # top-drive diagnostic: cross-line levels the slope fit deliberately skipped
    # (verification levels are not on this line and are handled separately)
    excluded = []
    for L in levels:
        if not L.on_cross_line() or L.in_fit_window():
            continue
        g = grad_bg_w([L.w])
        d_meas = L.mean - float((g @ np.array([bg[n][0] for n in PARAMS_BG]))[0])
        d_err = float(np.sqrt(L.sigma**2 + float((g @ bg_cov @ g.T)[0, 0])))
        a_row = np.array([L.w, 1.0])
        d_pred = float(a_row @ df["beta"])
        d_pred_err = float(np.sqrt(max(a_row @ df["cov"] @ a_row, 0.0)))
        excluded.append((L.w, d_meas, d_err, d_pred, d_pred_err))

    fit = PairV2Fit(
        index=index, levels=levels, bg=bg, bg_cov=bg_cov,
        fit_w=df["w"], fit_d=df["d"], fit_sigma=df["sigma"], fit_pred=df["pred"],
        b=b, b_err=b_err, beta0=beta0, beta0_err=beta0_err,
        eta=eta, eta_err=eta_err, r2=float(df["r2"]),
        anchor=anchor, excluded=excluded,
    )
    fit.checks = {
        "intercept": verify_intercept(fit),
        "product": verify_product(fit),
    }
    return fit


# ======================================================================
# verification  (checks on the model, not inputs to it)
# ======================================================================

def verify_intercept(fit: "PairV2Fit") -> dict:
    """Check 1 -- the fitted intercept must equal the single-beam x-side sum.

    ``D(w) = eta^2*w + (a_x + q_x)`` -- or ``+ a_x`` alone without the q columns
    -- so the cross line extrapolated to ``w = 0`` has to land on the x-side
    response measured independently in the x-only block.  A disagreement means
    the cross line is not the straight line the estimator assumes; curvature
    over the fit window shows up here first, because extrapolating to zero
    amplifies it.

    With :data:`FIT_Q` the background block is saturated, so the x-side sum
    equals the raw ``Y(1,0) - Y(0,0)`` exactly and this is the same statement as
    the anchor check, just phrased through the fitted parameters.  Without it
    the block carries 2 dof, the x-only levels are no longer reproduced exactly,
    and the two genuinely differ -- so the check has teeth on the standard grid.

    ``beta0`` and the x-side sum are not independent -- the background
    covariance is inside ``beta0``'s error -- so the quoted joint error is an
    overestimate and the pull is conservative.
    """
    idx = [PARAMS_BG.index(n) for n in X_SIDE]
    value = float(sum(fit.bg[n][0] for n in X_SIDE))
    # sum of the full sub-block, i.e. Var(a_x) + 2Cov(a_x,q_x) + Var(q_x)
    var = float(fit.bg_cov[np.ix_(idx, idx)].sum())
    err = float(np.sqrt(max(var, 0.0)))
    diff = fit.beta0 - value
    joint = float(np.hypot(fit.beta0_err, err))
    return {
        "beta0": fit.beta0, "beta0_err": fit.beta0_err,
        "a_x_plus_q_x": value, "a_x_plus_q_x_err": err,
        "diff": diff, "err": joint,
        "pull": diff / joint if joint else float("nan"),
    }


def verify_product(fit: "PairV2Fit") -> list[dict]:
    """Check 2 -- ``Y`` must depend on the drives only through the product ``x*w``.

    For each group in :data:`PRODUCT_CHECKS` every member shares the same
    ``x*w`` but splits the drive differently -- ``(1, 0.25)`` pushes the x side
    hard, ``(0.5, 0.5)`` splits it evenly.  Subtract the full single-beam
    background from each and what remains is the TPA term, which the model says
    is ``eta^2*(x*w)`` and therefore identical across the group.

    A split here is not a problem with this estimator -- it means the pair has
    no single eta at all, because its response is not a function of the product.
    That would invalidate the one-number-per-pair model that steps 7 and 8 are
    built on, so it is worth measuring even though it costs acquisitions.

    Both residues are built from the same fitted background, so they are
    correlated; the difference propagates that properly via
    ``(g_a - g_b) C (g_a - g_b)^T`` rather than adding the two errors in
    quadrature.
    """
    out: list[dict] = []
    p = np.array([fit.bg[n][0] for n in PARAMS_BG])
    for group in PRODUCT_CHECKS:
        pts = []
        for x, w in group:
            level = _level_at(fit.levels, x, w)
            if level is None:
                continue
            g = design_bg([x], [w])          # full background, PARAMS_BG order
            tpa = level.mean - float((g @ p)[0])
            pts.append({
                "x": x, "w": w, "n": level.n, "tpa": tpa,
                "tpa_err": float(np.sqrt(level.sigma**2
                                         + float((g @ fit.bg_cov @ g.T)[0, 0]))),
                "_g": g[0], "_sigma": level.sigma,
            })
        if len(pts) < 2:
            continue
        product = float(group[0][0] * group[0][1])
        rec = {"product": product, "expected": fit.b * product,
               "points": pts, "pairs": []}
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                a, b = pts[i], pts[j]
                dg = a["_g"] - b["_g"]
                var = (a["_sigma"] ** 2 + b["_sigma"] ** 2
                       + float(dg @ fit.bg_cov @ dg))
                err = float(np.sqrt(max(var, 0.0)))
                diff = a["tpa"] - b["tpa"]
                mid = 0.5 * (a["tpa"] + b["tpa"])
                rec["pairs"].append({
                    "a": (a["x"], a["w"]), "b": (b["x"], b["w"]),
                    "diff": diff, "err": err,
                    "pull": diff / err if err else float("nan"),
                    "frac": diff / mid if mid else float("nan"),
                })
        out.append(rec)
    return out


# ======================================================================
# reporting
# ======================================================================

def _sigma_ratio(value: float, err: float) -> float:
    return abs(value) / err if err else float("nan")


def report(fit: PairV2Fit) -> None:
    """Print the level table, both fit stages and the two consistency checks."""
    lo, hi = FIT_W_RANGE
    print("Estimator: difference   "
          f"D(w) = Y(1,w) - [{BHAT_LABEL}] = eta^2 w + ({X_SIDE_LABEL})")
    print(f"Fit window: w in [{lo:.2f}, {hi:.2f}]  ({fit.fit_w.size} levels)")

    print(f"\n  Level means  (sigma = hypot(max(rep_std, trace_std)/sqrt(n), "
          f"{STD_FLOOR_V*1e3:.3f} mV systematic floor)):")
    print("    block     x     w   n   mean(mV)  rep_std(mV)  trace_std(mV)  sigma(mV)")
    for L in fit.levels:
        tag = L.block
        if L.is_verification:
            tag = "verify+"
        elif L.block == "cross" and not L.in_fit_window():
            tag = "cross*"
        rep = f"{L.rep_std*1e3:11.4f}" if np.isfinite(L.rep_std) else f"{'--':>11}"
        print(f"    {tag:<7} {L.x:5.2f} {L.w:5.2f} {L.n:3d} {L.mean*1e3:10.4f} "
              f"{rep}  {L.trace_std*1e3:13.4f} {L.sigma*1e3:10.4f}")
    print("    (* excluded from the slope fit -- top-drive diagnostic; "
          "+ verification only)")

    repeated = [L for L in fit.levels if np.isfinite(L.rep_std)]
    if not repeated:
        print("    no level was repeated -- sigma is the trace spread alone, so these "
              "error bars carry no encoding repeatability")
    else:
        dominated = [L for L in repeated if L.rep_std > L.trace_std]
        print(f"    repeat scatter exceeds trace noise on {len(dominated)}/{len(repeated)} "
              f"repeated levels -- encoding repeatability "
              f"{'dominates' if len(dominated) > len(repeated) / 2 else 'is not dominant'}")

    n_bg = sum(1 for L in fit.levels if L.block != "cross")
    shape = "saturated" if n_bg == len(PARAMS_BG) else f"{n_bg - len(PARAMS_BG)} dof"
    print(f"\n  Background block ({n_bg} levels, {len(PARAMS_BG)} parameters -> {shape}; "
          "used by steps 7/8, not by eta):")
    for name in PARAMS_BG:
        v, e = fit.bg[name]
        scale, unit = (1e3, "mV") if name == "d" else (1.0, "")
        print(f"    {name:<3} = {v*scale:.4e} +/- {e*scale:.3e} {unit}".rstrip())
    if not FIT_Q:
        print("    (q_x/q_w dropped -- FIT_Q = False; with two drive levels per arm")
        print("     they are not separable from a_x/a_w, which is what produced "
              "negative background terms)")

    # With the q columns gone the block has dof, so its residual is a real
    # statement about whether the straight single-beam background holds.  A
    # saturated block returns nothing here: it fits its own points exactly.
    used_bg, bg_resid, bg_pull = fit.bg_residuals()
    if bg_resid.size:
        print("      block     x     w   resid(mV)   pull")
        for L, r, pl in zip(used_bg, bg_resid, bg_pull):
            print(f"      {L.block:<7} {L.x:5.2f} {L.w:5.2f} "
                  f"{r*1e3:10.4f} {pl:6.2f}")
        rms_bg = float(np.sqrt(np.mean(bg_resid**2)))
        mx = float(np.max(np.abs(bg_pull)))
        verdict = "OK" if mx < 3.0 else "** CHECK: arm is curved, add w levels **"
        print(f"      residual RMS = {rms_bg*1e3:.4f} mV   "
              f"max |pull| = {mx:.2f}   {verdict}")

    print("\n  Difference fit (GLS, background covariance propagated):")
    print(f"    eta^2 = {fit.b:.4e} +/- {fit.b_err:.3e}")
    print(f"    eta   = {fit.eta:.4e} +/- {fit.eta_err:.3e}   "
          f"({_sigma_ratio(fit.eta, fit.eta_err):.1f} sigma)   over w in [{lo:.2f}, {hi:.2f}]")
    print(f"    beta0 = {fit.beta0*1e3:.4f} +/- {fit.beta0_err*1e3:.4f} mV   "
          f"[= {X_SIDE_LABEL}]")
    print(f"    R^2   = {fit.r2:.4f}")

    print("      w    D_meas(mV)   D_fit(mV)  resid(mV)   pull")
    for i in np.argsort(fit.fit_w):
        print(f"    {fit.fit_w[i]:5.2f} {fit.fit_d[i]*1e3:11.4f} "
              f"{fit.fit_pred[i]*1e3:11.4f} {fit.residuals[i]*1e3:10.4f} "
              f"{fit.pulls[i]:6.2f}")
    rms = float(np.sqrt(np.mean(fit.residuals**2)))
    print(f"    residual RMS = {rms*1e3:.4f} mV")

    print("\n  Checks:")
    a_val, a_err = fit.anchor
    if np.isfinite(a_val):
        pull = (a_val - fit.beta0) / np.hypot(a_err, fit.beta0_err)
        print(f"    anchor      D(0) measured = {a_val*1e3:8.4f} +/- {a_err*1e3:.4f} mV  "
              f"vs beta0 = {fit.beta0*1e3:8.4f} mV   pull = {pull:+.2f}")
    for w, dm, dme, dp, dpe in fit.excluded:
        frac = (dm / dp - 1.0) * 100.0 if dp else float("nan")
        pull = (dm - dp) / np.hypot(dme, dpe) if np.hypot(dme, dpe) else float("nan")
        print(f"    top-drive   D({w:.2f}) measured = {dm*1e3:8.4f} +/- {dme*1e3:.4f} mV  "
              f"vs line = {dp*1e3:8.4f} mV   {frac:+.2f}%  (pull {pull:+.2f})")

    report_verification(fit)


def report_verification(fit: PairV2Fit) -> None:
    """The two model checks: the intercept identity and product-only dependence.

    Neither feeds the fit -- they say whether the model eta is defined within
    still holds on this pair.
    """
    checks = fit.checks or {}
    print("\n  Verification (checks on the model, not inputs to it):")

    ic = checks.get("intercept")
    if ic:
        verdict = "OK" if abs(ic["pull"]) < 3.0 else "** CHECK **"
        print(f"    1. intercept identity   beta0 = eta^2*0 + ({X_SIDE_LABEL})")
        print(f"       beta0       = {ic['beta0']*1e3:9.4f} +/- {ic['beta0_err']*1e3:.4f} mV "
              f"(cross line extrapolated to w=0)")
        print(f"       {X_SIDE_LABEL:<11} = {ic['a_x_plus_q_x']*1e3:9.4f} +/- "
              f"{ic['a_x_plus_q_x_err']*1e3:.4f} mV (x-only block)")
        print(f"       difference  = {ic['diff']*1e3:+9.4f} +/- {ic['err']*1e3:.4f} mV   "
              f"pull = {ic['pull']:+.2f}   {verdict}")
        n_bg = sum(1 for L in fit.levels if L.block != "cross")
        if n_bg == len(PARAMS_BG):
            print(f"       (background block is saturated, so {X_SIDE_LABEL} is exactly "
                  "the measured Y(1,0) - Y(0,0);")
            print("        this therefore tests the cross line's extrapolation, "
                  "not the arithmetic)")

    prods = checks.get("product") or []
    if not prods:
        print("    2. product-only dependence: NOT MEASURED "
              "(needs the verification levels -- set VERIFY_ENABLED = True and re-run)")
        return
    print("    2. product-only dependence   Y depends on (x, w) only via x*w")
    for rec in prods:
        print(f"       x*w = {rec['product']:.3f}   "
              f"model expects eta^2*(x*w) = {rec['expected']*1e3:.4f} mV")
        for pt in rec["points"]:
            print(f"         ({pt['x']:.2f}, {pt['w']:.2f}) n={pt['n']}  "
                  f"TPA residue = {pt['tpa']*1e3:9.4f} +/- {pt['tpa_err']*1e3:.4f} mV")
        for pr in rec["pairs"]:
            verdict = "OK" if abs(pr["pull"]) < 3.0 else "** CHECK **"
            print(f"         {pr['a']} - {pr['b']} = {pr['diff']*1e3:+.4f} +/- "
                  f"{pr['err']*1e3:.4f} mV  ({pr['frac']*100:+.2f}%)  "
                  f"pull = {pr['pull']:+.2f}   {verdict}")


def make_plot(fit: PairV2Fit, path: str | Path | None = None) -> None:
    """Four panels: background block, the difference line, pulls, repeatability.

    Top-left -- the single-beam block: x-only and w-only level means with the
    saturated background curves through them, plus the shared dark point.

    Top-right -- **the estimator**: ``D(w)`` against ``w`` with the GLS line.
    The fit window is shaded; the excluded top-drive level is drawn as an open
    marker, and the directly measured ``D(0)`` anchor as a square, so the two
    consistency checks are visible rather than buried in the text report.

    Bottom-left -- pulls of the fitted D points against the +/-1 sigma band.

    Bottom-right -- repeatability: each level's repeat scatter beside its trace
    noise.  Bars above the diagonal are levels where rewriting the SLM moves the
    reading more than the detector does, which is what the repeats are for.
    """
    import matplotlib

    if path is not None:
        matplotlib.use("Agg")     # headless: write a PNG rather than open a window
    import matplotlib.pyplot as plt

    lo, hi = FIT_W_RANGE
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 9.5))
    (ax1, ax2, ax5), (ax3, ax4, ax6) = axes

    # --- background block ---------------------------------------------------
    for label, sel, level_of, curve, color in (
        ("x-only (w=0)", lambda L: L.block in ("dark", "x-only"), lambda L: L.x,
         lambda r: fit.background(r, 0.0), "tab:blue"),
        ("w-only (x=0)", lambda L: L.block in ("dark", "w-only"), lambda L: L.w,
         lambda r: fit.background(0.0, r), "tab:orange"),
    ):
        pts = [L for L in fit.levels if sel(L)]
        r = np.array([level_of(L) for L in pts])
        y = np.array([L.mean for L in pts])
        e = np.array([L.sigma for L in pts])
        r_fine = np.linspace(0.0, float(r.max()) if r.size else 1.0, 200)
        ax1.plot(r_fine, curve(r_fine) * 1e3, "-", color=color, lw=1.4, zorder=2)
        order = np.argsort(r)
        ax1.errorbar(r[order], y[order] * 1e3, yerr=e[order] * 1e3, fmt="o",
                     color=color, ms=6, capsize=2, lw=0.8, mec="k", mew=0.3,
                     zorder=3, label=label)
    p = {k: v[0] for k, v in fit.bg.items()}
    txt = "\n".join(f"{n} = {p[n]*(1e3 if n == 'd' else 1.0):.3g}"
                     f"{' mV' if n == 'd' else ''}" for n in PARAMS_BG)
    _, bg_resid, bg_pull = fit.bg_residuals()
    if bg_resid.size:
        txt += (f"\nresid RMS = {float(np.sqrt(np.mean(bg_resid**2)))*1e3:.4f} mV"
                f"\nmax |pull| = {float(np.max(np.abs(bg_pull))):.2f}")
    ax1.text(0.03, 0.97, txt,
             transform=ax1.transAxes, va="top", fontsize=8,
             bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax1.set_xlabel("per-side level r")
    ax1.set_ylabel("Voltage (mV)")
    n_bg = sum(1 for L in fit.levels if L.block != "cross")
    dof = n_bg - len(PARAMS_BG)
    ax1.set_title(f"Single-beam block ({'saturated' if dof == 0 else f'{dof} dof'} fit)"
                  " -- for steps 7/8, not for $\\eta$")
    ax1.legend(loc="lower right", fontsize=8)

    # --- the difference line: this is the estimator --------------------------
    ax2.axvspan(lo, hi, color="tab:green", alpha=0.10, zorder=0,
                label=f"fit window [{lo:g}, {hi:g}]")
    w_fine = np.linspace(0.0, 1.05, 200)
    ax2.plot(w_fine, (fit.b * w_fine + fit.beta0) * 1e3, "-", color="tab:green",
             lw=1.5, zorder=2, label="GLS  $D = \\eta^2 w + \\beta_0$")
    order = np.argsort(fit.fit_w)
    ax2.errorbar(fit.fit_w[order], fit.fit_d[order] * 1e3,
                 yerr=fit.fit_sigma[order] * 1e3, fmt="o", color="tab:green",
                 ms=6, capsize=2, lw=0.8, mec="k", mew=0.3, zorder=4,
                 label="$D(w)$ fitted")
    for w, dm, dme, _dp, _dpe in fit.excluded:
        ax2.errorbar([w], [dm * 1e3], yerr=[dme * 1e3], fmt="o", mfc="none",
                     color="tab:red", ms=8, capsize=2, lw=0.9, mew=1.2, zorder=4,
                     label="excluded (top drive)")
    a_val, a_err = fit.anchor
    if np.isfinite(a_val):
        ax2.errorbar([0.0], [a_val * 1e3], yerr=[a_err * 1e3], fmt="s",
                     color="tab:purple", ms=7, capsize=2, lw=0.9, mec="k",
                     mew=0.3, zorder=4, label="$D(0)$ measured (anchor)")
    ax2.text(0.03, 0.97,
             f"eta = {fit.eta:.4g} $\\pm$ {fit.eta_err:.2g}  "
             f"({_sigma_ratio(fit.eta, fit.eta_err):.0f}$\\sigma$)\n"
             f"$\\eta^2$ = {fit.b:.4g}\n"
             f"$\\beta_0$ = {fit.beta0*1e3:.3f} mV\n"
             f"R$^2$ = {fit.r2:.4f}",
             transform=ax2.transAxes, va="top", fontsize=8,
             bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax2.set_xlabel("w   (x = 1 on this line)")
    ax2.set_ylabel("$D(w) = Y(1,w) - \\hat{B}(w)$   (mV)")
    ax2.set_title("Difference estimator -- $\\eta$ is this slope")
    # de-duplicate the repeated "excluded" label
    h, lb = ax2.get_legend_handles_labels()
    seen: dict[str, object] = {}
    for handle, name in zip(h, lb):
        seen.setdefault(name, handle)
    ax2.legend(seen.values(), seen.keys(), loc="lower right", fontsize=8)

    # --- pulls ---------------------------------------------------------------
    ax3.axhspan(-1, 1, color="tab:blue", alpha=0.12, label="$\\pm1\\sigma$")
    ax3.axhline(0, color="gray", ls="--", lw=1, zorder=1)
    ax3.scatter(fit.fit_w, fit.pulls, c="tab:green", s=55, edgecolor="k", lw=0.4,
                zorder=3, label="fitted $D(w)$")
    rms = float(np.sqrt(np.mean(fit.residuals**2)))
    ax3.text(0.03, 0.97, f"RMS = {rms*1e3:.4f} mV", transform=ax3.transAxes,
             va="top", fontsize=8,
             bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    span = float(np.max(np.abs(fit.pulls))) * 1.25 if fit.pulls.size else 1.5
    ax3.set_ylim(-max(1.5, span), max(1.5, span))    # never clip a pull
    ax3.set_xlabel("w")
    ax3.set_ylabel("Pull = residual / $\\sigma$")
    ax3.set_title("Difference-fit residuals")
    ax3.legend(loc="lower right", fontsize=8)

    # --- repeatability -------------------------------------------------------
    # Ratio rather than a scatter: the only question is whether rewriting the
    # panel moves a level more than the detector noise does, and a bar chart
    # answers that at a glance without overlapping point labels.
    repeated = [L for L in fit.levels if np.isfinite(L.rep_std) and L.trace_std > 0]
    if repeated:
        ratios = [L.rep_std / L.trace_std for L in repeated]
        labels = [f"({L.x:g}, {L.w:g})  n={L.n}" for L in repeated]
        colors = [{"dark": "k", "x-only": "tab:blue", "w-only": "tab:orange",
                   "cross": "tab:green"}[L.block] for L in repeated]
        pos = np.arange(len(repeated))
        ax4.barh(pos, ratios, color=colors, alpha=0.75, edgecolor="k", lw=0.4)
        ax4.axvline(1.0, color="0.4", ls="--", lw=1.2,
                    label="rewrite = detector noise")
        ax4.set_yticks(pos)
        ax4.set_yticklabels(labels, fontsize=7)
        ax4.invert_yaxis()
        ax4.set_xlabel("repeat std / trace std")
        ax4.set_title("Repeatability: past the line, encoding dominates")
        ax4.legend(loc="lower right", fontsize=8)
    else:
        ax4.text(0.5, 0.5, "no level was repeated\n(sigma is the trace spread alone)",
                 transform=ax4.transAxes, ha="center", va="center", fontsize=10,
                 color="0.4")
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.set_title("Repeatability")

    # --- product check: equal x*w must give equal TPA residue ---------------
    prods = (fit.checks or {}).get("product") or []
    if prods:
        for k, rec in enumerate(prods):
            labels = [f"({p['x']:g}, {p['w']:g})" for p in rec["points"]]
            vals = np.array([p["tpa"] for p in rec["points"]]) * 1e3
            errs = np.array([p["tpa_err"] for p in rec["points"]]) * 1e3
            pos = np.arange(len(vals)) + k * (len(vals) + 1)
            ax5.errorbar(pos, vals, yerr=errs, fmt="o", ms=8, capsize=4, lw=1.2,
                         color="tab:purple", mec="k", mew=0.4, zorder=3)
            ax5.axhline(rec["expected"] * 1e3, color="tab:green", ls="--", lw=1.2,
                        zorder=1,
                        label="$\\eta^2(x\\,w)$ from the slope fit" if k == 0 else None)
            ax5.set_xticks(pos)
            ax5.set_xticklabels([f"{lb}\n$x\\,w$={rec['product']:g}" for lb in labels],
                                fontsize=8)
            for n_pr, pr in enumerate(rec["pairs"]):
                ax5.text(0.03, 0.97 - 0.07 * n_pr,
                         f"split = {pr['diff']*1e3:+.4f} $\\pm$ {pr['err']*1e3:.4f} mV "
                         f"({pr['frac']*100:+.2f}%, pull {pr['pull']:+.2f})",
                         transform=ax5.transAxes, ha="left", va="top", fontsize=8,
                         bbox=dict(boxstyle="round", fc="white", alpha=0.85))
        n_slots = sum(len(r["points"]) + 1 for r in prods) - 1
        ax5.set_xlim(-0.6, n_slots - 0.4)      # keep ticks off the axis edges
        ax5.set_ylabel("TPA residue $Y - \\hat{B}(x,w)$   (mV)")
        ax5.set_title("Product check -- same $x\\,w$, different split")
        ax5.legend(loc="upper right", fontsize=8)
    else:
        ax5.text(0.5, 0.5,
                 "product check not measured\n\nneeds the verification levels\n"
                 "(VERIFY_ENABLED = True)",
                 transform=ax5.transAxes, ha="center", va="center", fontsize=10,
                 color="0.4")
        ax5.set_xticks([])
        ax5.set_yticks([])
        ax5.set_title("Product check")

    # --- verification summary text ------------------------------------------
    ax6.axis("off")
    lines = ["Verification", ""]
    ic = (fit.checks or {}).get("intercept")
    if ic:
        lines += [
            f"1. intercept identity   $\\beta_0$ = {X_SIDE_LABEL}",
            f"     $\\beta_0$ = {ic['beta0']*1e3:8.4f} $\\pm$ "
            f"{ic['beta0_err']*1e3:.4f} mV",
            f"     {X_SIDE_LABEL} = {ic['a_x_plus_q_x']*1e3:8.4f} $\\pm$ "
            f"{ic['a_x_plus_q_x_err']*1e3:.4f} mV",
            f"     pull = {ic['pull']:+.2f}   "
            f"{'OK' if abs(ic['pull']) < 3 else 'CHECK'}",
            "",
        ]
    if prods:
        lines.append("2. product-only dependence")
        for rec in prods:
            lines.append(f"     $x\\,w$ = {rec['product']:g}")
            for pr in rec["pairs"]:
                lines.append(
                    f"     {pr['a']} vs {pr['b']}: {pr['frac']*100:+.2f}%   "
                    f"pull = {pr['pull']:+.2f}   "
                    f"{'OK' if abs(pr['pull']) < 3 else 'CHECK'}")
    else:
        lines += ["2. product-only dependence", "     not measured"]
    lines += ["", "Neither check feeds the fit -- they say whether",
              "the model $\\eta$ is defined within still holds."]
    ax6.text(0.02, 0.98, "\n".join(lines), transform=ax6.transAxes, va="top",
             ha="left", fontsize=9, family="monospace",
             bbox=dict(boxstyle="round", fc="0.97", ec="0.7"))

    fig.suptitle(f"Step 6 v2 -- pair {fit.index}   "
                 f"$\\eta$ = {fit.eta:.4g} $\\pm$ {fit.eta_err:.2g}", fontsize=12)
    fig.tight_layout()
    if path is None:
        plt.show()
        return
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ======================================================================
# persistence
# ======================================================================

def build_schedule(grid=None) -> list[tuple[int, float, float]]:
    """Interleaved acquisition order -> ``(repeat, x, w)`` per acquisition.

    Round-robin passes rather than n back-to-back reads of one level, so a slow
    drift shows up as scatter across a level's repeats instead of masquerading
    as a slope.  Within each pass, levels run brightest first, so the very first
    acquisition of a pair is its brightest point and a dead or blocked beam is
    visible immediately.

    Defaults to :func:`full_grid` -- the estimator's levels plus the
    verification block -- resolved at call time so ``VERIFY_ENABLED`` can be
    flipped without reimporting.
    """
    grid = full_grid() if grid is None else grid
    order = sorted(
        range(len(grid)),
        key=lambda i: (-(grid[i][0] * grid[i][1]), -max(grid[i][0], grid[i][1]),
                       -grid[i][0]),
    )
    sched: list[tuple[int, float, float]] = []
    for rep in range(max(g[2] for g in grid)):
        for i in order:
            x, w, n = grid[i]
            if n > rep:
                sched.append((rep, float(x), float(w)))
    return sched


def write_meas_csv(rows_by_pair: dict[int, list], path: str | Path) -> Path:
    """Raw rows, v1's column layout (``trial`` carries the repeat index).

    Keeping v1's columns means the same file can be re-fit with v1's joint
    6-parameter estimator for a direct comparison of the two estimators on
    identical data.
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_HEADER)
        for index in sorted(rows_by_pair):
            for rep, x, w, mean_v, std_v in rows_by_pair[index]:
                ratio = abs(std_v / mean_v) if mean_v else float("inf")
                writer.writerow([int(rep), index, f"{x:.6g}", f"{w:.6g}",
                                 f"{x*w:.6g}", f"{mean_v:.9g}", f"{std_v:.9g}",
                                 f"{ratio:.6g}"])
    return out


def load_meas_csv(path: str | Path) -> dict[int, list]:
    """Read a v2 measurement CSV back into ``{pair_index: rows}``."""
    rows_by_pair: dict[int, list] = {}
    with open(Path(path), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            index = int(float(row["pair_index"]))
            rows_by_pair.setdefault(index, []).append((
                int(float(row.get("trial", 0))),
                float(row["x"]), float(row["w"]),
                float(row["voltage_mean_v"]), float(row["voltage_std_v"]),
            ))
    return rows_by_pair


def _checks_dict(fit: PairV2Fit) -> dict:
    """JSON-safe view of the verification results (drops the working arrays)."""
    checks = fit.checks or {}
    return {
        "intercept": checks.get("intercept"),
        "product": [
            {
                "product": rec["product"],
                "expected": rec["expected"],
                "points": [{k: v for k, v in pt.items() if not k.startswith("_")}
                           for pt in rec["points"]],
                "pairs": rec["pairs"],
            }
            for rec in (checks.get("product") or [])
        ],
    }


def _background_dict(fit: PairV2Fit) -> dict:
    """Background-block shape and residual -- the evidence for the FIT_Q choice."""
    used, resid, pull = fit.bg_residuals()
    return {
        "fit_q": FIT_Q,
        "params": list(PARAMS_BG),
        "n_levels": len(used),
        "dof": len(used) - len(PARAMS_BG),
        "resid_rms_v": float(np.sqrt(np.mean(resid**2))) if resid.size else None,
        "max_abs_pull": float(np.max(np.abs(pull))) if pull.size else None,
        "levels": [{"x": L.x, "w": L.w, "resid_v": float(r), "pull": float(pl)}
                   for L, r, pl in zip(used, resid, pull)],
    }


def save_combined_json(fits: list[PairV2Fit], out_path: str | Path,
                       *, center_wl: float = 0.0) -> Path:
    """Step-3 calibration + every v2 pair fit, in the schema step 7 reads.

    ``channels[].fit.{eta, eta_err, params.{a_x,q_x,a_w,q_w,d}.value}`` is the
    contract :meth:`calibration_module.phase.PairModel.from_json_channel`
    parses, so step 7 consumes a v2 result with no change.  The v2-specific
    numbers (the slope, the fit window, both checks) ride alongside it.
    """
    out_path = Path(out_path)

    def _fit_dict(fit: PairV2Fit) -> dict:
        params = {n: {"value": v, "err": e} for n, (v, e) in fit.bg.items()}
        # PairModel reads a_x/q_x/a_w/q_w/d unconditionally, so a dropped q is
        # written as an exact zero rather than left out of the file.
        for n in ("q_x", "q_w"):
            params.setdefault(n, {"value": 0.0, "err": 0.0})
        params["b"] = {"value": fit.b, "err": fit.b_err}
        params["beta0"] = {"value": fit.beta0, "err": fit.beta0_err}
        return {
            "eta": fit.eta,
            "eta_err": fit.eta_err,
            "params": params,
            "r2": fit.r2,
            "estimator": "difference",
            "fit_w_range": list(FIT_W_RANGE),
            "anchor_d0": {"value": fit.anchor[0], "err": fit.anchor[1]},
            "checks": _checks_dict(fit),
            "background": _background_dict(fit),
            "excluded": [
                {"w": w, "d_meas": dm, "d_meas_err": dme,
                 "d_line": dp, "d_line_err": dpe}
                for w, dm, dme, dp, dpe in fit.excluded
            ],
            "levels": [
                {"x": L.x, "w": L.w, "n": L.n, "mean_v": L.mean,
                 "rep_std_v": None if not np.isfinite(L.rep_std) else L.rep_std,
                 "trace_std_v": L.trace_std, "sigma_v": L.sigma}
                for L in fit.levels
            ],
        }

    step6 = {
        "version": 2,
        "estimator": "difference",
        "fit_q": FIT_Q,
        "params_bg": list(PARAMS_BG),
        "fit_w_range": list(FIT_W_RANGE),
        "grid": [{"x": x, "w": w, "n": n} for x, w, n in GRID],
        "verify_grid": ([{"x": x, "w": w, "n": n} for x, w, n in VERIFY_GRID]
                        if VERIFY_ENABLED else []),
        "product_checks": [[list(pt) for pt in group] for group in PRODUCT_CHECKS],
        "center_wl": center_wl,
        "channels": [
            {
                "index": fit.index,
                "wl_x_nm": fit.wl_x_nm,
                "wl_w_nm": fit.wl_w_nm,
                "nominal_wl_nm": fit.nominal_wl_nm,
                "fit": _fit_dict(fit),
            }
            for fit in fits
        ],
    }
    step3 = json.loads(IN_STEP3.read_text(encoding="utf-8"))
    out_path.write_text(json.dumps({"step3": step3, "step6": step6}, indent=2),
                        encoding="utf-8")
    return out_path


# ======================================================================
# hardware drive
# ======================================================================

def _slot(pair: int) -> int:
    """Pair label -> its 0-based slot in the Step-3 layout / SLM drive arrays."""
    return pair - PAIR_INDEX_BASE


def _load_layout():
    """Load the Step-3 calibration -> channel layout, validating PAIR_INDICES.

    The Step-3b/3c rows ARE the channels, so the layout is loaded verbatim (the
    same ``channel_layout_from_calibration`` the GUI encoding page uses) -- no
    re-tiling, so pair indices here mean the same thing as in the UI.

    Step 3 records only a wavelength per channel, never an index -- a channel's
    identity IS its position in that list -- so the pair label is positional
    too: pair ``i`` is the ``_slot(i)``-th channel of the Step-3 calibration.
    """
    if not IN_STEP3.is_file():
        raise FileNotFoundError(
            f"Step-3 calibration not found: {IN_STEP3}\n"
            f"(CALIB_PATH is the calib_data directory; IN_STEP3 is the JSON in it.)"
        )
    layout = channel_layout_from_calibration(load_calibration_result(IN_STEP3))
    for pi in PAIR_INDICES:
        if not (0 <= _slot(pi) < layout.n_channels):
            raise ValueError(
                f"pair index {pi} out of range (layout has {layout.n_channels} "
                f"pairs, numbered from {PAIR_INDEX_BASE})"
            )
    return layout


def _measure_pair(slm, daq, layout, index: int, schedule) -> list:
    """Run one pair's interleaved schedule -> raw ``(rep, x, w, mean, std)`` rows.

    ``index`` is the pair LABEL; ``_slot`` maps it onto the 0-based drive
    arrays.  Only that pair is on, every other channel off.  The SLM is rewritten
    for **every** acquisition, repeats included -- that rewrite is what makes a
    level's repeat scatter measure encoding repeatability rather than detector
    jitter, so it must not be optimised away for repeated levels.
    """
    from slm_module.encoding import encode_to_pattern

    zeros = np.zeros(layout.n_channels)
    slm_width, slm_height = slm.get_slm_info()

    rows = []
    total = len(schedule)
    for step, (rep, x_val, w_val) in enumerate(schedule, start=1):
        x_vals = zeros.copy()
        w_vals = zeros.copy()
        x_vals[_slot(index)] = x_val
        w_vals[_slot(index)] = w_val
        slm.display_array(
            encode_to_pattern(x_vals, w_vals, layout, slm_width, slm_height)
        )
        if SETTLE_S:
            time.sleep(SETTLE_S)
        single = x_val == 0.0 or w_val == 0.0
        mean_v, std_v = read_point(daq, single=single)
        rows.append((rep, float(x_val), float(w_val), mean_v, std_v))
        ratio = abs(std_v / mean_v) if mean_v else float("inf")
        print(f"[{step}/{total}] pair {index} rep {rep} "
              f"x={x_val:.3f} w={w_val:.3f} "
              f"({T_SINGLE_S if single else T_BOTH_S:.0f}s) -> "
              f"{mean_v*1000:.4f} mV  std ratio {ratio*100:.2f}%")
    return rows


def _pair_wavelengths(layout, index: int) -> tuple[float, float, float]:
    """Wavelengths of a pair's two channels, or NaN if the layout lacks that slot.

    ``index`` is a pair LABEL; ``_slot`` maps it into the Step-3 layout.  A CSV
    may carry pair labels the loaded layout does not cover (a synthetic file, or
    a layout from a different run); the wavelengths are cosmetic -- only the
    JSON/plot labels use them -- so fall back to NaN rather than killing the fit.
    """
    nan = float("nan")
    slot = _slot(index)
    if not (0 <= slot < min(len(layout.x_channels), len(layout.w_channels))):
        return nan, nan, nan
    x_ch = layout.x_channels[slot]
    w_ch = layout.w_channels[slot]
    return (float(x_ch.wavelength_nm), float(w_ch.wavelength_nm),
            0.5 * (float(x_ch.wavelength_nm) + float(w_ch.wavelength_nm)))


def _run_sweep(fit_after: bool, *, include_anchor: bool = False) -> None:
    """Drive every pair's interleaved schedule; optionally fit, plot and save."""
    layout = _load_layout()
    schedule = build_schedule()
    n_single = sum(1 for _, x, w in schedule if x == 0.0 or w == 0.0)
    secs = n_single * T_SINGLE_S + (len(schedule) - n_single) * T_BOTH_S
    secs += len(schedule) * SETTLE_S
    n_verify = len(VERIFY_GRID) if VERIFY_ENABLED else 0
    print(f"Step 6 v2: {len(schedule)} acquisitions/pair over "
          f"{len(full_grid())} levels ({len(GRID)} estimator + {n_verify} "
          f"verification), interleaved, brightest first (~{secs/60:.1f} min/pair)")
    print(f"Pairs: {list(PAIR_INDICES)}")

    slm = connect_slm(SLM_DISPLAY_NO, USB_SLM_NO)
    daq = connect_daq(device=DAQ_DEVICE, channel=DAQ_CHANNEL,
                      t_both=T_BOTH_S, t_single=T_SINGLE_S)
    rows_by_pair: dict[int, list] = {}
    try:
        for index in PAIR_INDICES:
            print(f"\n=== Sweep: pair {index} ===")
            rows_by_pair[index] = _measure_pair(slm, daq, layout, index, schedule)
    finally:
        slm.close_slm()
        daq.disconnect()

    stamp = time.strftime("%m%d_%H%M")
    csv_path = write_meas_csv(rows_by_pair,
                              CALIB_PATH / f"calib_step6v2_meas_{stamp}.csv")
    total = sum(len(v) for v in rows_by_pair.values())
    print(f"\nSaved {total} rows to {csv_path}")   # raw rows on disk BEFORE fitting
    if not fit_after:
        return
    _fit_and_save(rows_by_pair, stamp, layout=layout, include_anchor=include_anchor)


def _fit_and_save(rows_by_pair: dict[int, list], stamp: str, *, layout=None,
                  include_anchor: bool = False) -> None:
    """Fit every pair, print the report, write the combined JSON and the plots."""
    fits: list[PairV2Fit] = []
    for index in sorted(rows_by_pair):
        levels = average_levels(rows_by_pair[index])
        try:
            fit = fit_pair(index, levels, include_anchor=include_anchor)
        except (ValueError, np.linalg.LinAlgError) as exc:
            print(f"\n=== pair {index} ===\n  fit FAILED: {exc}")
            continue
        if layout is not None:
            fit.wl_x_nm, fit.wl_w_nm, fit.nominal_wl_nm = _pair_wavelengths(layout, index)
        fits.append(fit)
        print(f"\n=== pair {index} ===")
        report(fit)

    if not fits:
        print("\nNo pair fitted -- nothing saved.")
        return

    center_wl = float(getattr(layout, "center_wl", 0.0)) if layout is not None else 0.0
    json_path = CALIB_PATH / f"calib_step6v2_result_{stamp}.json"
    save_combined_json(fits, json_path, center_wl=center_wl)
    print(f"\nSaved Step-3 calib + Step-6 v2 fits -> {json_path}")
    for fit in fits:
        plot_path = json_path.with_name(f"calib_step6v2_pair{fit.index}_{stamp}.png")
        make_plot(fit, plot_path)
        print(f"Plot saved to {plot_path}")


def fit_csv(path: str | Path, *, include_anchor: bool = False) -> None:
    """Re-fit an already-recorded v2 CSV offline (no hardware).

    Writes the same combined JSON and per-pair PNGs as the hardware run, under
    a fresh timestamp so a refit never clobbers an earlier result.  The Step-3
    layout is loaded when available (for the wavelength columns) but is not
    required -- the fit itself needs only the CSV.
    """
    rows_by_pair = load_meas_csv(path)
    n = sum(len(v) for v in rows_by_pair.values())
    print(f"Loaded {path}: {len(rows_by_pair)} pair(s), {n} acquisitions")
    if include_anchor:
        print("Anchor: D(0) included as a fitted point (intercept pinned on data).")
    try:
        layout = _load_layout()
    except (FileNotFoundError, ValueError) as exc:
        print(f"(layout unavailable, wavelengths left as NaN: {exc})")
        layout = None
    _fit_and_save(rows_by_pair, time.strftime("%m%d_%H%M"),
                  layout=layout, include_anchor=include_anchor)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    flags = {"--meas", "-m", "--anchor"}
    include_anchor = "--anchor" in argv
    positional = [a for a in argv if a not in flags]
    if positional:                      # a CSV path -> offline re-fit, no hardware
        fit_csv(positional[0], include_anchor=include_anchor)
        return 0
    _run_sweep(fit_after=not any(a in ("--meas", "-m") for a in argv),
               include_anchor=include_anchor)
    return 0


if __name__ == "__main__":
    sys.exit(main())
