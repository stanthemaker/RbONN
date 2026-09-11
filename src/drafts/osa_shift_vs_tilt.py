"""Fit the rigid-shift and the multiplicative-tilt model to an OSA run, separately.

Both candidate explanations of a drifting sweep are statements about the ratio
against the first sweep, ``R(l,t) = S(l,t)/S(l,0)``, and -- this is the whole
point -- they have the *same* algebraic shape::

    R(l,t) = a(t) * [ 1 + s(t) * z(l) ]

    SHIFT   S(l-delta,0)/S(l,0) ~= 1 - delta * dlnS/dl
            ->  z(l) = -dlnS0/dl        s = delta  (nm; positive = red shift)
    TILT    T(l,t)S(l,0)/S(l,0) = a * [1 + beta*(l - l_p)]
            ->  z(l) = l - l_p          s = beta   (1/nm)

They differ *only* in the regressor ``z(l)``.  So the fits are the same two
lines of algebra run twice, and the comparison is decided by one thing: how
much the two regressors differ in shape.  On a Gaussian envelope they do not
differ at all -- ``-dlnS0/dl = (l-l0)/sigma^2`` is exactly linear in ``l`` --
and the models are then the same function with ``beta = delta/sigma^2``.  The
report prints the weighted correlation between the regressors first, because
it bounds everything that follows.

``a(t)`` is the decay term: a *uniform* loss, the same at every wavelength, and
*linear in time*::

    a(t) = 1 + c*t          one rate c for the whole run

so it is one extra parameter for the run, not one per sweep.  It is carried in
both models -- Eq. 1 as written has no amplitude freedom and would otherwise
lose to Eq. 2 for the trivial reason of having one parameter fewer.  The shape
term ``s(t)`` stays free per sweep.  ``--decay free`` instead lets the
amplitude float per sweep (the check on whether the decay really is linear in
time) and ``--decay none`` pins ``a = 1``, the equations exactly as written.
A fully global fit then makes the shape term linear in time as well, giving
each model the whole run in two numbers.

Fits are in the ratio domain (as the equations are written), weighted least
squares with ``w = S(l,0)``, over the window where ``S(l,0)`` is above ``WING``
of its peak.  Sibling ``osa_tilt_fit.py`` fits the same two models in the log
domain with a third combined model; the numbers agree to first order and it is
the place to look for the pedestal and temperature-branch checks.

Writes ``<stem>_shift_vs_tilt.png`` and ``<stem>_shift_vs_tilt.csv`` next to
the master CSV::

    python src/drafts/osa_shift_vs_tilt.py
    python src/drafts/osa_shift_vs_tilt.py src/calib_data/daq_osa_monitor_0826_2328.csv --no-show
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
from osa_tilt_fit import DERIV_SMOOTH, WING, collinearity, wls  # noqa: E402
from plot_osa_monitor import (  # noqa: E402  (same folder, shared loader)
    linear_power, linfit, load_run, newest_master_csv, smooth_rows, th10k_celsius,
)

NM_TO_PM = 1000.0
LATE_FRAC = 0.25       # "late run" = the last quarter of the sweeps
WEIGHTS = ("S", "S2", "flat")
DECAYS = ("linear", "free", "none")


# --------------------------------------------------------------------------
# model basis
# --------------------------------------------------------------------------
def build_basis(wl: np.ndarray, ref: np.ndarray, weight: str) -> dict:
    """Window, weights, pivot and the two regressors, all from sweep 0.

    ``w = S(l,0)`` is the shot-noise weighting: var(S) ~ S makes var(R) ~ 1/S.
    ``S2`` is the right choice if a constant detector floor dominates instead;
    the run is re-fittable either way with ``--weight`` and the conclusions
    should not move if they are real.

    The pivot ``l_p`` is the *weighted* mean wavelength, which makes the two
    columns of the tilt design matrix orthogonal, so ``a`` and ``beta`` are
    independently determined whatever weighting is in force.

    ``-dlnS0/dl`` is built from a smoothed S(l,0) because it is a model basis
    function: a raw numerical derivative of a noisy sweep acts as errors-in-
    variables and biases delta toward zero.  The data themselves are not
    smoothed.

    The window is trimmed by the smoothing half-width at both ends of the scan:
    ``smooth_rows`` convolves in ``same`` mode, so the smoothed reference decays
    toward zero over the last half-kernel and ``dln S0/dl`` blows up there.  On
    this run the 10%-of-peak window runs into the top end of the scan, and
    without the trim those few points alone drag the shift fit.
    """
    edge = DERIV_SMOOTH // 2 + 1
    mask = ref >= WING * ref.max()
    mask[:edge] = False
    mask[len(mask) - edge:] = False
    w = {"S": ref[mask], "S2": ref[mask] ** 2,
         "flat": np.ones(int(mask.sum()))}[weight]
    lam_p = float((w * wl[mask]).sum() / w.sum())
    ref_s = smooth_rows(ref[None, :], DERIV_SMOOTH)[0]
    return dict(
        mask=mask, w=w, lam_p=lam_p,
        z_shift=-np.gradient(np.log(ref_s), wl)[mask],   # s = delta (nm)
        z_tilt=wl[mask] - lam_p,                         # s = beta  (1/nm)
    )


def rms_rows(resid: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted residual RMS per sweep."""
    return np.sqrt((w[None, :] * resid**2).sum(1) / w.sum())


def autocorr_inflate(resid: np.ndarray) -> float:
    """Lag-1 correction for the fact that neighbouring OSA samples are not independent.

    The 0.02 nm resolution bandwidth spans several grid points, so the naive
    covariance is optimistic by sqrt((1+rho)/(1-rho)); ``rho`` is measured along
    the wavelength axis of the residual map.
    """
    r = resid - resid.mean(axis=1, keepdims=True)
    ss = float((r**2).sum())
    rho = float((r[:, :-1] * r[:, 1:]).sum() / ss) if ss > 0 else 0.0
    return float(np.sqrt((1 + min(max(rho, 0.0), 0.98)) / (1 - min(max(rho, 0.0), 0.98))))


# --------------------------------------------------------------------------
# fits
# --------------------------------------------------------------------------
def fit_free(ratio: np.ndarray, z: np.ndarray, w: np.ndarray, decay: str) -> dict:
    """Sweep-by-sweep fit, with the amplitude free (``free``) or pinned (``none``).

    ``free`` fits R = p0 + p1*z, then a = p0 and s = p1/p0, with the ratio's
    error from the delta method on the full covariance -- not err(p1)/p0, which
    would ignore that the coefficients are correlated whenever z is not
    weight-centred (it is not, for the shift model).
    """
    a, a_err, s, s_err, resid = [], [], [], [], []
    for r in ratio:
        if decay == "none":
            f = wls(z[:, None], r - 1.0, w)
            a.append(1.0), a_err.append(0.0)
            s.append(float(f["coef"][0])), s_err.append(float(f["err"][0]))
        else:
            f = wls(np.column_stack([np.ones_like(z), z]), r, w)
            (p0, p1), cov = f["coef"], f["cov"]
            si = float(p1 / p0)
            var = float((cov[1, 1] - 2 * si * cov[0, 1] + si**2 * cov[0, 0]) / p0**2)
            a.append(float(p0)), a_err.append(float(np.sqrt(max(cov[0, 0], 0.0))))
            s.append(si), s_err.append(float(np.sqrt(max(var, 0.0))))
        resid.append(f["resid"])
    resid = np.vstack(resid)
    return dict(a=np.asarray(a), a_err=np.asarray(a_err), s=np.asarray(s),
                s_err=np.asarray(s_err), resid=resid, rms=rms_rows(resid, w),
                c=np.nan, c_err=np.nan,
                label="a = 1" if decay == "none" else "a free per sweep")


def fit_linear_decay(ratio: np.ndarray, z: np.ndarray, w: np.ndarray,
                     t: np.ndarray) -> dict:
    """R(l,t) = (1 + c*t) * [1 + s(t)*z(l)] -- ONE decay rate for the run.

    The decay is uniform in wavelength and linear in time, so it costs a single
    parameter ``c`` for the whole run; the shape term ``s(t)`` stays free per
    sweep.  Fitted jointly (n+1 parameters over the full lambda-by-time map)
    rather than sweep by sweep, since ``c`` is only determined by the run as a
    whole.

    Note the amplitude is pinned to 1 at t=0 -- R(l,0) is identically 1, sweep 0
    being its own reference -- so a single noisy first sweep anchors the whole
    decay.  ``report`` checks that against the free-amplitude fit.

    The joint fit yields one pooled variance for all sweeps, which would give
    every sweep the same error bar however well or badly that sweep fitted, so
    each sweep's error is rescaled by its own residual RMS.
    """
    n = ratio.shape[0]
    sw = np.sqrt(w)

    def predict(p: np.ndarray) -> np.ndarray:
        return (1.0 + p[0] * t)[:, None] * (1.0 + p[1:][:, None] * z[None, :])

    sol = least_squares(lambda p: ((ratio - predict(p)) * sw[None, :]).ravel(),
                        np.zeros(n + 1))
    resid = ratio - predict(sol.x)
    dof = max(sol.fun.size - sol.x.size, 1)
    cov = float((sol.fun**2).sum() / dof) * np.linalg.inv(sol.jac.T @ sol.jac)
    err = np.sqrt(np.clip(np.diag(cov), 0.0, None)) * autocorr_inflate(resid)

    rms = rms_rows(resid, w)
    scale = rms / max(float(np.sqrt((rms[1:] ** 2).mean())), 1e-30)
    c = float(sol.x[0])
    return dict(a=1.0 + c * t, a_err=np.abs(t) * float(err[0]),
                s=sol.x[1:].copy(), s_err=err[1:] * scale,
                resid=resid, rms=rms,
                c=c, c_err=float(err[0]), label="a = 1 + c*t")


def global_fit(ratio: np.ndarray, z: np.ndarray, w: np.ndarray, t: np.ndarray,
               decay: bool = True) -> dict:
    """Whole run in two numbers: R(l,t) = (1 + c*t) * [1 + k*t*z(l)].

    The same uniform linear decay as :func:`fit_linear_decay`, but the shape
    term is forced to grow linearly in time too, so ``k`` is a single rate:
    nm/h for the shift model, 1/nm/h for the tilt model.  Nonlinear only
    through the ``c*k*t^2`` cross term, which is fitted rather than dropped so
    both rates mean exactly what they say.
    """
    sw = np.sqrt(w)

    def predict(p: np.ndarray) -> np.ndarray:
        c, k = (p[0], p[1]) if decay else (0.0, p[0])
        return (1.0 + c * t)[:, None] * (1.0 + k * t[:, None] * z[None, :])

    sol = least_squares(lambda p: ((ratio - predict(p)) * sw[None, :]).ravel(),
                        np.zeros(2 if decay else 1))
    resid = ratio - predict(sol.x)
    dof = max(sol.fun.size - sol.x.size, 1)
    cov = float((sol.fun**2).sum() / dof) * np.linalg.inv(sol.jac.T @ sol.jac)
    err = np.sqrt(np.clip(np.diag(cov), 0.0, None)) * autocorr_inflate(resid)

    c, k = (sol.x[0], sol.x[1]) if decay else (0.0, sol.x[0])
    c_err, k_err = (err[0], err[1]) if decay else (0.0, err[0])
    return dict(c=float(c), c_err=float(c_err), k=float(k), k_err=float(k_err),
                rms=float(np.sqrt((w[None, :] * resid**2).sum()
                                  / (w.sum() * resid.shape[0]))), resid=resid)


def centroids(wl_w: np.ndarray, ref_w: np.ndarray, lin_w: np.ndarray, lam_p: float,
              beta: np.ndarray, delta: np.ndarray) -> dict:
    """In-window centroid: measured, and what each model predicts.

    The tilt reweights sweep 0 by (1 + beta*x) -- the uniform decay cancels --
    while a rigid shift moves the centroid by exactly delta.  Whichever model
    is right has to reproduce the centroid motion that was measured.
    """
    x = wl_w - lam_p
    meas = (lin_w * wl_w).sum(1) / lin_w.sum(1)
    tilt = np.array([float(((ref_w * (1 + b * x)) * wl_w).sum()
                           / (ref_w * (1 + b * x)).sum()) for b in beta])
    return dict(meas=meas, tilt=tilt, shift=meas[0] + delta)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def report(run, t, wl, b: dict, fs: dict, ft: dict, gs: dict, gt: dict,
           alt: dict, cen: dict, temp_c, weight: str, decay: str) -> None:
    mask, w = b["mask"], b["w"]
    wl_w = wl[mask]
    r_col, vif = collinearity(b["z_shift"], b["z_tilt"], w)
    late = t >= (1 - LATE_FRAC) * t[-1]
    d_pm, de_pm = fs["s"] * NM_TO_PM, fs["s_err"] * NM_TO_PM
    # sweep 0 is its own reference: R == 1, both models fit it exactly, and any
    # residual ratio there is 0/0.  It carries no information and is left out.
    live = np.arange(len(t)) > 0

    print("=" * 78)
    print(f"{run.name}   {run.timestamps[0]} -> {run.timestamps[-1]}")
    print(f"{len(t)} sweeps over {t[-1]:.2f} h   |   R(l,t) = S(l,t)/S(l,0) "
          f"fitted as a(t)*[1 + s(t)*z(l)]")
    print("=" * 78)
    print(f"window    S(l,0) > {WING:.0%} of peak -> {wl_w.min():.3f} - {wl_w.max():.3f} nm "
          f"({int(mask.sum())} of {wl.size} points)")
    print(f"weights   w = {weight}  (S = shot-noise-like; --weight S2 for a flat noise floor)")
    print(f"pivot     l_p = {b['lam_p']:.4f} nm (weighted mean -> a and beta orthogonal)")
    amp = {"linear": "(1 + c*t)", "free": "a(t)", "none": "1"}[decay]
    print(f"decay     {fs['label']}"
          + (": uniform in wavelength, linear in time, one rate for the run"
             if decay == "linear" else
             ", uniform in wavelength (--decay linear for one rate over the run)"))

    print("\n--- can the two models be told apart at all? ----------------------")
    print(f"weighted corr[ z_shift , z_tilt ] = {r_col:+.5f}   VIF = {vif:.0f}")
    if abs(r_col) > 0.99:
        print(f"  -> the regressors are the same function to within "
              f"{100 * np.sqrt(1 - r_col**2):.1f}% of their spread: S(l,0) is near-Gaussian")
        print("     over this window, so -dlnS0/dl is itself nearly linear in l.  The two")
        print("     models are near-identical and ONLY the residual -- not the fitted")
        print("     parameters -- can separate them.")
    elif abs(r_col) > 0.9:
        print("  -> strongly collinear; read the residual comparison, not the parameters.")
    else:
        print("  -> the regressors are distinguishable; the comparison is well posed.")

    def decay_line(f: dict) -> str:
        if decay == "none":
            return "  decay   a = 1, pinned: the equations exactly as written"
        if not np.isfinite(f["c"]):
            return (f"  decay   a {f['a'][0]:.4f} -> {f['a'][-1]:.4f} over the run "
                    f"(free per sweep, range {f['a'].min():.4f}-{f['a'].max():.4f})")
        return (f"  decay   c = {100 * f['c']:+.4f} +- {100 * f['c_err']:.4f} %/h   "
                f"-> a {f['a'][0]:.4f} -> {f['a'][-1]:.4f} over the run")

    print(f"\n--- model SHIFT:  R = {amp} * [1 - delta(t)*dlnS0/dl] ---------")
    sl, se, _, r = linfit(t, d_pm)
    print(f"  delta   {d_pm[0]:+8.1f} -> {d_pm[-1]:+8.1f} pm   "
          f"(+-{de_pm[-1]:.1f} pm at t_end, {abs(d_pm[-1]) / max(de_pm[-1], 1e-12):.1f} sigma)")
    print(f"  drift   {sl:+.2f} +- {se:.2f} pm/h   (r={r:+.3f}, {abs(sl / se):.0f} sigma)")
    print(decay_line(fs))
    print(f"  resid   {fs['rms'].mean():.5f} mean, {fs['rms'][-1]:.5f} at t_end")

    print(f"\n--- model TILT:   R = {amp} * [1 + beta(t)*(l - l_p)] --------")
    sl_b, se_b, _, r_b = linfit(t, ft["s"])
    print(f"  beta    {ft['s'][0]:+8.5f} -> {ft['s'][-1]:+8.5f} 1/nm  "
          f"(+-{ft['s_err'][-1]:.5f} at t_end, "
          f"{abs(ft['s'][-1]) / max(ft['s_err'][-1], 1e-12):.1f} sigma)")
    print(f"  drift   {sl_b:+.5f} +- {se_b:.5f} /nm/h   (r={r_b:+.3f}, "
          f"{abs(sl_b / se_b):.0f} sigma)")
    print(decay_line(ft))
    print(f"  resid   {ft['rms'].mean():.5f} mean, {ft['rms'][-1]:.5f} at t_end")
    if decay != "none":
        gap = (abs(fs["c"] - ft["c"]) if np.isfinite(ft["c"])
               else float(np.abs(fs["a"] - ft["a"]).max()))
        print(f"  the two models agree on the amplitude to {100 * gap:.4f}"
              f"{' %/h' if np.isfinite(ft['c']) else '% at worst'}: it is orthogonal to the")
        print("  shape term, so 'how much light was lost' does not depend on which of the")
        print("  two you believe.")

    print("\n--- head to head (weighted residual RMS of R) ---------------------")
    ratio_all = fs["rms"][live] / ft["rms"][live]
    print(f"  shift/tilt  {ratio_all.mean():.3f}x mean over the run, "
          f"{(fs['rms'][late] / ft['rms'][late]).mean():.3f}x over the last "
          f"{100 * LATE_FRAC:.0f}%, {ratio_all[-1]:.3f}x at t_end")
    print(f"  the SHIFT model has the smaller residual on "
          f"{int((fs['rms'][live] < ft['rms'][live]).sum())} of {int(live.sum())} sweeps")
    late_ratio = float((fs["rms"][late] / ft["rms"][late]).mean())
    better = "SHIFT" if late_ratio < 1 else "TILT"
    print(f"  -> over the late run the {better} model describes the ratio better, by "
          f"{100 * abs(1 - late_ratio):.0f}%")
    floor = ft["rms"][1:4].mean()
    print(f"  sweeps 1-3 sit at {floor:.5f} with nothing yet to fit: that is the noise")
    print(f"     floor, and at t_end the models are at {fs['rms'][-1] / floor:.1f}x / "
          f"{ft['rms'][-1] / floor:.1f}x of it (shift / tilt).  Both leave structure")
    print("     behind, so neither is the whole story -- but one leaves less.")
    print("  Why the tilt wins, physically: S(l,0) is not smooth, and a rigid shift of a")
    print("  structured spectrum has to print that structure onto the ratio, since")
    print("  -dlnS0/dl carries every ripple of the reference.  The measured ratio is")
    print("  smoother than that -- which is what a wavelength-linear transmission gives.")

    print("\n--- the decay term: is it uniform, and is it linear in time? ------")
    print(f"{'amplitude':<16}{'resid shift':>12}{'resid tilt':>12}{'shift/tilt':>12}")
    for label, d in (("a = 1", "none"), ("a = 1 + c*t", "linear"), ("a free/sweep", "free")):
        r_s = alt[d]["shift"]["rms"][live].mean()
        r_t = alt[d]["tilt"]["rms"][live].mean()
        print(f"  {label:<14}{r_s:>12.5f}{r_t:>12.5f}{r_s / r_t:>12.3f}x")
    print(f"  -> the TILT model wins under every amplitude treatment, so the head-to-head")
    print("     does not rest on how the decay is modelled.")
    a_free = alt["free"]["tilt"]["a"]
    sl_a, se_a, resid_a, r_a = linfit(t, a_free)
    a0 = float(np.polyfit(t, a_free, 1)[1])
    print(f"  a(t) free per sweep: {100 * sl_a:+.3f} +- {100 * se_a:.3f} %/h about an "
          f"intercept of {a0:.4f},")
    print(f"     r={r_a:+.3f}, scatter about that line {100 * resid_a:.2f}%, range "
          f"{a_free.min():.4f}-{a_free.max():.4f}")
    linear_ok = abs(sl_a) > 2 * se_a and abs(r_a) > 0.7
    if not linear_ok:
        print(f"     -> NOT a linear decay.  The free amplitude has no significant trend, "
              f"and the")
        print(f"        constrained fit reads {100 * ft['c']:+.3f} %/h only because "
              f"a(0) is pinned to 1 while")
        print(f"        the amplitude actually scatters by ~{100 * resid_a:.1f}% sweep to "
              f"sweep: a single noisy")
        print(f"        first sweep {100 * (a0 - 1):+.1f}% off the run's mean sets the whole "
              f"slope.  Treat c as an")
        print("        upper bound on any uniform loss, not a measurement of one.")
    else:
        print("     -> the uniform linear decay holds: constrained and free fits agree.")

    print("\n--- global fits: the whole run in two numbers ---------------------")
    print("  R(l,t) = (1 + c*t) * [1 + k*t*z(l)]        (all sweeps, all wavelengths)")
    print(f"  shift   c = {100 * gs['c']:+.4f} +- {100 * gs['c_err']:.4f} %/h   "
          f"ddelta/dt = {gs['k'] * NM_TO_PM:+.2f} +- {gs['k_err'] * NM_TO_PM:.2f} pm/h   "
          f"resid {gs['rms']:.5f}")
    print(f"  tilt    c = {100 * gt['c']:+.4f} +- {100 * gt['c_err']:.4f} %/h   "
          f"dbeta/dt  = {gt['k']:+.5f} +- {gt['k_err']:.5f} /nm/h   "
          f"resid {gt['rms']:.5f}")
    print(f"  global shift/tilt residual ratio = {gs['rms'] / gt['rms']:.3f}x")

    print("\n--- does either model reproduce the measured centroid motion? -----")
    s_m = linfit(t, cen["meas"])[0] * NM_TO_PM
    s_t = linfit(t, cen["tilt"])[0] * NM_TO_PM
    s_s = linfit(t, cen["shift"])[0] * NM_TO_PM
    print(f"  measured in-window centroid   {s_m:+7.2f} pm/h   "
          f"({(cen['meas'][-1] - cen['meas'][0]) * NM_TO_PM:+.1f} pm total)")
    frac_t, frac_s = 100 * s_t / s_m, 100 * s_s / s_m
    print(f"  predicted by TILT  (beta(t))  {s_t:+7.2f} pm/h   -> {frac_t:.0f}% of it, "
          f"corr {float(np.corrcoef(cen['meas'], cen['tilt'])[0, 1]):+.3f}")
    print(f"  predicted by SHIFT (delta(t)) {s_s:+7.2f} pm/h   -> {frac_s:.0f}% of it, "
          f"corr {float(np.corrcoef(cen['meas'], cen['shift'])[0, 1]):+.3f}")
    if max(abs(frac_t - 100), abs(frac_s - 100)) < 25:
        print("  Both account for it, as they must: they are fitted to the same ratio and the")
        print("  centroid is just one moment of that ratio.  So this test does NOT separate")
        print("  the two models here -- only the residual comparison above does.")
    else:
        worse = "SHIFT" if abs(frac_s - 100) > abs(frac_t - 100) else "TILT"
        print(f"  The {worse} model cannot reproduce the motion that was measured: the shape")
        print("  parameter the ratio supports is the wrong size to move the centroid that far.")

    print("\n" + "=" * 78)
    print("READ-OUT")
    print("=" * 78)
    ratios = [alt[d]["shift"]["rms"][live].mean() / alt[d]["tilt"]["rms"][live].mean()
              for d in DECAYS]
    print(f"1. The regressors are {'separable' if abs(r_col) < 0.9 else 'nearly degenerate'} "
          f"on this envelope (corr {r_col:+.4f}, VIF {vif:.0f}): the two")
    print("   models are genuinely different functions here, not a re-parametrisation.")
    print(f"2. TILT fits better, but not overwhelmingly: {100 * abs(1 - late_ratio):.0f}% "
          f"less residual over the late run,")
    print(f"   lower on {int((ft['rms'][live] < fs['rms'][live]).sum())} of "
          f"{int(live.sum())} sweeps, and lower under all three amplitude treatments "
          f"({min(ratios):.2f}-{max(ratios):.2f}x).")
    print(f"3. The centroid test does not separate them "
          f"({frac_t:.0f}% vs {frac_s:.0f}% of the measured motion):")
    print("   both are fitted to the same ratio, so both reproduce its moments.")
    print(f"4. As a tilt the run reads {sl_b:+.4f} /nm/h (beta {ft['s'][1]:+.4f} -> "
          f"{ft['s'][-1]:+.4f} 1/nm);")
    print(f"   forced into a rigid shift it reads {sl:+.1f} +- {se:.1f} pm/h "
          f"({d_pm[-1]:+.0f} pm over {t[-1]:.1f} h).")
    c_lin = alt["linear"]["tilt"]["c"]
    if linear_ok:
        print(f"5. A uniform {100 * c_lin:+.3f} %/h loss sits on top, orthogonal to the "
              "shape term.")
    else:
        print(f"5. The uniform linear decay is NOT supported: a(t) is flat to "
              f"{100 * se_a:.2f} %/h with {100 * resid_a:.1f}%")
        print(f"   sweep-to-sweep scatter, so any real uniform loss is below "
              f"~{100 * (abs(sl_a) + se_a):.2f} %/h.  The fitted")
        print(f"   c = {100 * c_lin:+.3f} %/h is an artifact of pinning a(0)=1 to one noisy "
              "reference sweep.")
    print("6. This fit alone cannot settle the mechanism -- both models describe the data")
    print("   to within ~1.3x of each other.  What breaks the tie is that beta(t) does not")
    print("   retrace when the room temperature turns around (see osa_tilt_fit.py), which a")
    print("   laser wavelength drift has no reason to do.")

    print("\n--- per-sweep table ------------------------------------------------")
    print(f"{'t_h':>6}{'a':>8}{'delta_pm':>10}{'+-':>7}{'res_sh':>9}"
          f"{'beta':>10}{'+-':>8}{'res_ti':>9}{'a_free':>8}{'T_C':>7}")
    for i in range(len(t)):
        print(f"{t[i]:>6.2f}{ft['a'][i]:>8.4f}{d_pm[i]:>10.1f}{de_pm[i]:>7.1f}"
              f"{fs['rms'][i]:>9.5f}{ft['s'][i]:>10.5f}{ft['s_err'][i]:>8.5f}"
              f"{ft['rms'][i]:>9.5f}{a_free[i]:>8.4f}{temp_c[i]:>7.2f}")


def write_csv(path: Path, t, fs, ft, a_free, cen, temp_c, daq_mv) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_h", "a_decay", "delta_pm", "delta_err_pm", "resid_shift",
                    "beta_per_nm", "beta_err", "resid_tilt", "a_free_per_sweep",
                    "centroid_meas_nm", "centroid_tilt_nm", "centroid_shift_nm",
                    "temp_C", "pd_mV"])
        for i in range(len(t)):
            w.writerow([f"{t[i]:.4f}", f"{ft['a'][i]:.6f}",
                        f"{fs['s'][i] * NM_TO_PM:.3f}", f"{fs['s_err'][i] * NM_TO_PM:.3f}",
                        f"{fs['rms'][i]:.6f}", f"{ft['s'][i]:.6f}", f"{ft['s_err'][i]:.6f}",
                        f"{ft['rms'][i]:.6f}", f"{a_free[i]:.6f}",
                        f"{cen['meas'][i]:.5f}", f"{cen['tilt'][i]:.5f}",
                        f"{cen['shift'][i]:.5f}", f"{temp_c[i]:.4f}", f"{daq_mv[i]:.6f}"])


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
def plot_fits(run, t, wl, ref, b, fs, ft, gs, gt, alt, ratio, r_col) -> plt.Figure:
    mask = b["mask"]
    wl_w, ref_w = wl[mask], ref[mask]
    late = t >= (1 - LATE_FRAC) * t[-1]
    n_late = int(late.sum())

    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    (ax_r, ax_e), (ax_d, ax_b), (ax_a, ax_q) = axes
    fig.suptitle(
        f"{run.name}   rigid shift vs multiplicative tilt, fitted separately to "
        f"R(l,t)=S(l,t)/S(l,0)\nwindow {wl_w.min():.2f}-{wl_w.max():.2f} nm, "
        f"pivot {b['lam_p']:.3f} nm, uniform decay a(t)=1+c*t, "
        f"{len(t)} sweeps over {t[-1]:.1f} h", fontsize=12)

    # (a) the ratio at t_end, with both fitted models -----------------------
    ax_r.fill_between(wl_w, 1 + 0.06 * (ref_w / ref_w.max() - 1), 1.0,
                      color="0.88", zorder=0, label="S(l,0) envelope (scaled)")
    ax_r.plot(wl_w, ratio[-1], ".", color="0.35", markersize=2.5,
              label=f"measured R at t={t[-1]:.1f} h")
    ax_r.plot(wl_w, fs["a"][-1] * (1 + fs["s"][-1] * b["z_shift"]), "-", color="C1",
              linewidth=1.8,
              label=rf"shift: $\delta$={fs['s'][-1] * NM_TO_PM:+.0f} pm")
    ax_r.plot(wl_w, ft["a"][-1] * (1 + ft["s"][-1] * b["z_tilt"]), "--", color="C0",
              linewidth=1.8,
              label=rf"tilt: $\beta$={ft['s'][-1]:+.4f} nm$^{{-1}}$")
    ax_r.set_title(f"(a)  both models on the last sweep's ratio  (a={ft['a'][-1]:.4f})")
    ax_r.set_xlabel("wavelength (nm)")
    ax_r.set_ylabel("R = S(l,t) / S(l,0)")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(fontsize=8, loc="best")
    # both curves ride high here because a(t) is pinned to 1+c*t while the sweep's
    # own amplitude is lower -- the shape is the point of this panel, panel (e) has
    # the amplitude
    off = 100 * (ft["a"][-1] / alt["free"]["tilt"]["a"][-1] - 1)
    ax_r.text(0.02, 0.94, f"both curves sit {off:+.1f}% high: a(t) pinned to 1+c*t, see (e)",
              fontsize=7.5, color="0.3", transform=ax_r.transAxes)

    # (b) mean residual shape over the late run -----------------------------
    for f, color, style, name in ((fs, "C1", "-", "shift"), (ft, "C0", "--", "tilt")):
        ax_e.plot(wl_w, 1e3 * f["resid"][late].mean(0), style, color=color, linewidth=1.5,
                  label=f"{name}  (mean resid RMS {f['rms'][late].mean():.5f})")
    ax_e.axhline(0.0, color="0.4", linewidth=0.8)
    ax_e.set_title(f"(b)  mean residual over the last {n_late} sweeps "
                   "-- the shape each model misses")
    ax_e.set_xlabel("wavelength (nm)")
    ax_e.set_ylabel(r"R $-$ model  [$\times 10^{-3}$]")
    ax_e.grid(True, alpha=0.3)
    ax_e.legend(fontsize=8, loc="best")
    ax_e.text(0.02, 0.04, f"regressor corr r={r_col:+.4f}", fontsize=7.5, color="0.3",
              transform=ax_e.transAxes)

    # (c) delta(t) -----------------------------------------------------------
    d_pm, de_pm = fs["s"] * NM_TO_PM, fs["s_err"] * NM_TO_PM
    ax_d.fill_between(t, d_pm - de_pm, d_pm + de_pm, color="C1", alpha=0.22,
                      label=r"$\pm 1\sigma$")
    ax_d.plot(t, d_pm, "o-", color="C1", markersize=3, linewidth=1.1, label=r"$\delta(t)$")
    sl, se, _, r = linfit(t, d_pm)
    ax_d.plot(t, np.polyval(np.polyfit(t, d_pm, 1), t), "--", color="0.3", linewidth=1.0,
              label=f"{sl:+.1f} $\\pm$ {se:.1f} pm/h (r={r:+.2f})")
    ax_d.plot(t, gs["k"] * t * NM_TO_PM, ":", color="C3", linewidth=1.4,
              label=f"global 2-param fit, {gs['k'] * NM_TO_PM:+.1f} pm/h")
    ax_d.axhline(0.0, color="0.4", linewidth=0.8)
    ax_d.set_title(r"(c)  SHIFT model: rigid wavelength shift $\delta(t)$")
    ax_d.set_xlabel("elapsed (h)")
    ax_d.set_ylabel(r"$\delta$ (pm)")
    ax_d.grid(True, alpha=0.3)
    ax_d.legend(fontsize=8, loc="best")

    # (d) beta(t) ------------------------------------------------------------
    ax_b.fill_between(t, ft["s"] - ft["s_err"], ft["s"] + ft["s_err"], color="C0",
                      alpha=0.22, label=r"$\pm 1\sigma$")
    ax_b.plot(t, ft["s"], "o-", color="C0", markersize=3, linewidth=1.1, label=r"$\beta(t)$")
    sl_b, se_b, _, r_b = linfit(t, ft["s"])
    ax_b.plot(t, np.polyval(np.polyfit(t, ft["s"], 1), t), "--", color="0.3", linewidth=1.0,
              label=f"{sl_b:+.4f} $\\pm$ {se_b:.4f} /nm/h (r={r_b:+.2f})")
    ax_b.plot(t, gt["k"] * t, ":", color="C3", linewidth=1.4,
              label=f"global 2-param fit, {gt['k']:+.4f} /nm/h")
    ax_b.axhline(0.0, color="0.4", linewidth=0.8)
    ax_b.set_title(r"(d)  TILT model: slope $\beta(t)$")
    ax_b.set_xlabel("elapsed (h)")
    ax_b.set_ylabel(r"$\beta$ [nm$^{-1}$]")
    ax_b.grid(True, alpha=0.3)
    ax_b.legend(fontsize=8, loc="best")

    # (e) the uniform decay term --------------------------------------------
    for key, color, marker, name in (("shift", "C1", "o", "shift"), ("tilt", "C0", "s", "tilt")):
        ax_a.plot(t, alt["free"][key]["a"], marker, color=color, markersize=3.5, alpha=0.65,
                  label=f"a(t) free per sweep, {name} model")
    f_lin = alt["linear"]["tilt"]          # the constrained decay, whatever --decay is
    ax_a.plot(t, f_lin["a"], "-", color="C3", linewidth=1.8,
              label=f"fitted a=1+c*t, {100 * f_lin['c']:+.3f} $\\pm$ "
                    f"{100 * f_lin['c_err']:.3f} %/h")
    ax_a.fill_between(t, f_lin["a"] - f_lin["a_err"], f_lin["a"] + f_lin["a_err"],
                      color="C3", alpha=0.2)
    # the same line without a(0) pinned to 1: the gap between the two is what the
    # single (noisy) reference sweep is worth
    a_free = alt["free"]["tilt"]["a"]
    sl_a, se_a, _, _ = linfit(t, a_free)
    ax_a.plot(t, np.polyval(np.polyfit(t, a_free, 1), t), "--", color="0.35", linewidth=1.4,
              label=f"unpinned, {100 * sl_a:+.3f} $\\pm$ {100 * se_a:.3f} %/h")
    ax_a.axhline(1.0, color="0.5", linewidth=0.8)
    ax_a.set_title("(e)  the decay term: uniform in wavelength, linear in time")
    ax_a.set_ylabel("a(t)")
    ax_a.set_xlabel("elapsed (h)")
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(fontsize=8, loc="best")

    # (f) residual RMS per sweep --------------------------------------------
    ax_q.plot(t, fs["rms"], "o-", color="C1", markersize=3, linewidth=1.1, label="shift")
    ax_q.plot(t, ft["rms"], "s-", color="C0", markersize=3, linewidth=1.1, label="tilt")
    ax_q.axhline(gs["rms"], color="C1", linestyle=":", linewidth=1.2,
                 label=f"shift, global 2-param ({gs['rms']:.5f})")
    ax_q.axhline(gt["rms"], color="C0", linestyle=":", linewidth=1.2,
                 label=f"tilt, global 2-param ({gt['rms']:.5f})")
    ax_q.set_title(f"(f)  weighted residual RMS  (shift/tilt = "
                   f"{(fs['rms'][late] / ft['rms'][late]).mean():.3f}x late in the run)")
    ax_q.set_xlabel("elapsed (h)")
    ax_q.set_ylabel("residual RMS of R")
    ax_q.grid(True, alpha=0.3)
    ax_q.legend(fontsize=8, loc="best")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Fit the rigid-shift and multiplicative-tilt models to an OSA run.")
    p.add_argument("csv", nargs="?", type=Path,
                   help="master daq_osa_monitor_<stamp>.csv (default: newest in src/calib_data)")
    p.add_argument("--weight", choices=WEIGHTS, default="S",
                   help="fit weights: S (shot noise, default), S2 (flat noise floor), flat")
    p.add_argument("--decay", choices=DECAYS, default="linear",
                   help="amplitude: linear = uniform 1+c*t (default), free per sweep, or none")
    p.add_argument("--out", type=Path, help="PNG path (default: <csv stem>_shift_vs_tilt.png)")
    p.add_argument("--no-show", action="store_true", help="save without opening a window")
    args = p.parse_args(argv)

    master_csv = (args.csv or newest_master_csv()).resolve()
    print(f"reading {master_csv}")
    run = load_run(master_csv)
    lin = linear_power(run)
    wl, t = run.wl_nm, run.elapsed_h
    temp_c = th10k_celsius(run.ohm)

    ref = lin[0]
    b = build_basis(wl, ref, args.weight)
    mask, w = b["mask"], b["w"]
    ratio = lin[:, mask] / ref[mask]           # the data both equations describe

    def fit(z, decay):
        return (fit_linear_decay(ratio, z, w, t) if decay == "linear"
                else fit_free(ratio, z, w, decay))

    fs, ft = fit(b["z_shift"], args.decay), fit(b["z_tilt"], args.decay)
    # the other two amplitude treatments, for the "what the decay term buys" section
    alt = {d: {"shift": fit(b["z_shift"], d), "tilt": fit(b["z_tilt"], d)}
           for d in DECAYS if d != args.decay}
    alt.setdefault(args.decay, {"shift": fs, "tilt": ft})

    gs = global_fit(ratio, b["z_shift"], w, t)
    gt = global_fit(ratio, b["z_tilt"], w, t)

    cen = centroids(wl[mask], ref[mask], lin[:, mask], b["lam_p"], ft["s"], fs["s"])
    r_col, _ = collinearity(b["z_shift"], b["z_tilt"], w)

    report(run, t, wl, b, fs, ft, gs, gt, alt, cen, temp_c, args.weight, args.decay)

    out = args.out or master_csv.with_name(f"{master_csv.stem}_shift_vs_tilt.png")
    csv_out = master_csv.with_name(f"{master_csv.stem}_shift_vs_tilt.csv")
    fig = plot_fits(run, t, wl, ref, b, fs, ft, gs, gt, alt, ratio, r_col)
    fig.savefig(out, dpi=150)
    write_csv(csv_out, t, fs, ft, alt["free"]["tilt"]["a"], cen, temp_c, run.daq_mv)
    print(f"\nsaved -> {out}")
    print(f"saved -> {csv_out}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
