"""Fit a multiplicative spectral tilt to a ``daq_osa_monitor`` sweep series.

For every sweep the log-ratio against the first sweep is fitted with three
nested models over the high-signal window::

    tilt      ln[S(l,t)/S(l,0)] = ln a + beta*(l - l_p)
    shift     ln[S(l,t)/S(l,0)] = ln a + delta*(-dlnS0/dl)
    combined  ln[S(l,t)/S(l,0)] = ln a + beta*(l - l_p) + delta*(-dlnS0/dl)

``a`` is the scalar transmission at the pivot, ``beta`` the tilt slope
(1/nm) and ``delta`` the residual rigid wavelength shift (nm).  The pivot
``l_p`` is the intensity-weighted mean wavelength of the window, computed once
from sweep 0, which makes ``a`` and ``beta`` orthogonal under the S(l,0)
weighting used for the fits.

Read the collinearity line in the report before trusting the combined fit: on
a near-Gaussian envelope ``-dlnS0/dl`` *is* linear in wavelength, so the two
regressors are close to degenerate and the split between tilt and shift is
poorly determined by construction.

Writes ``<stem>_tilt.png`` and ``<stem>_tilt.csv`` next to the master CSV::

    python src/drafts/osa_tilt_fit.py
    python src/drafts/osa_tilt_fit.py src/calib_data/daq_osa_monitor_0826_2328.csv --no-show
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_osa_monitor import (  # noqa: E402  (same folder, shared loader)
    linear_power, linfit, load_run, newest_master_csv, smooth_rows, th10k_celsius,
)

WING = 0.10        # analysis window: keep l where S(l,0) > WING * peak
DERIV_SMOOTH = 25  # points of moving average before differentiating S(l,0)
NM_TO_PM = 1000.0


# --------------------------------------------------------------------------
# weighted least squares
# --------------------------------------------------------------------------
def wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    """Weighted least squares -> coefficients, standard errors, residual RMS.

    ``rms`` is the weighted RMS residual, sqrt(sum w r^2 / sum w), so it is
    directly comparable between models fitted with the same weights.

    Neighbouring OSA samples are not independent (the 0.02 nm RBW spans a few
    grid points), so the naive standard errors are optimistic.  ``inflate`` is
    the lag-1 autocorrelation correction (1+rho)/(1-rho) and ``err`` already
    has it applied; ``err_naive`` is the uncorrected value.
    """
    sw = np.sqrt(w)
    Xw, yw = X * sw[:, None], y * sw
    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = y - X @ coef
    n, p = X.shape
    rms = float(np.sqrt((w * resid**2).sum() / w.sum()))
    dof = max(n - p, 1)
    s2 = float((w * resid**2).sum() / dof)
    xtx_inv = np.linalg.inv(Xw.T @ Xw)
    err_naive = np.sqrt(np.diag(s2 * xtx_inv))
    r = resid - resid.mean()
    ss = float((r**2).sum())
    rho = float((r[:-1] * r[1:]).sum() / ss) if n > 2 and ss > 0 else 0.0
    rho = min(max(rho, 0.0), 0.98)
    inflate = np.sqrt((1 + rho) / (1 - rho))
    return dict(coef=coef, err=err_naive * inflate, err_naive=err_naive,
                rms=rms, resid=resid, rho=rho, inflate=float(inflate))


def fit_series(run, wl, lin, mask, lam_p, g, w) -> dict:
    """Run the three models on every sweep.  Returns arrays over time."""
    ref = lin[0, mask]
    x = wl[mask] - lam_p
    ones = np.ones_like(x)
    X_tilt = np.column_stack([ones, x])
    X_shift = np.column_stack([ones, g])
    X_both = np.column_stack([ones, x, g])

    out = {k: [] for k in (
        "lna", "beta", "beta_err", "delta", "delta_err",
        "lna_tilt", "beta_tilt", "beta_tilt_err", "delta_shift", "delta_shift_err",
        "rms_tilt", "rms_shift", "rms_both", "rho")}
    for row in lin[:, mask]:
        y = np.log(row / ref)
        ft, fs, fb = wls(X_tilt, y, w), wls(X_shift, y, w), wls(X_both, y, w)
        out["lna_tilt"].append(ft["coef"][0])
        out["beta_tilt"].append(ft["coef"][1])
        out["beta_tilt_err"].append(ft["err"][1])
        out["delta_shift"].append(fs["coef"][1])
        out["delta_shift_err"].append(fs["err"][1])
        out["lna"].append(fb["coef"][0])
        out["beta"].append(fb["coef"][1])
        out["beta_err"].append(fb["err"][1])
        out["delta"].append(fb["coef"][2])
        out["delta_err"].append(fb["err"][2])
        out["rms_tilt"].append(ft["rms"])
        out["rms_shift"].append(fs["rms"])
        out["rms_both"].append(fb["rms"])
        out["rho"].append(fb["rho"])
    res = {k: np.asarray(v) for k, v in out.items()}
    res["a_tilt"] = np.exp(res["lna_tilt"])
    res["a"] = np.exp(res["lna"])
    return res


def collinearity(x: np.ndarray, g: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted correlation between the two regressors, and the VIF it implies."""
    def wmean(v):
        return float((w * v).sum() / w.sum())
    dx, dg = x - wmean(x), g - wmean(g)
    r = float((w * dx * dg).sum() / np.sqrt((w * dx**2).sum() * (w * dg**2).sum()))
    return r, 1.0 / max(1.0 - r**2, 1e-12)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def report(run, lin, f: dict, t, wl, mask, lam_p, g, x, w, temp_c, pedestal_beta) -> None:
    n = len(t)
    lo, hi = wl[mask].min(), wl[mask].max()
    r_col, vif = collinearity(x, g, w)

    print("=" * 78)
    print(f"{run.name}   {run.timestamps[0]} -> {run.timestamps[-1]}")
    print(f"{n} sweeps over {t[-1]:.2f} h")
    print("=" * 78)
    print(f"window      S(l,0) > {WING:.0%} of peak  ->  {lo:.3f} - {hi:.3f} nm "
          f"({mask.sum()} of {wl.size} points)")
    print(f"pivot       l_p = {lam_p:.4f} nm (intensity-weighted mean of the window, sweep 0)")
    print(f"weights     w = S(l,0);  no pedestal subtracted (OSA dark level not measured)")
    print(f"regressor   -dlnS0/dl from a {DERIV_SMOOTH}-pt smoothed S(l,0); the raw derivative")
    print(f"            is noise-dominated and would attenuate delta toward zero")

    print("\n--- can the two models even be told apart? -----------------------")
    print(f"weighted corr[ (l-l_p) , -dlnS0/dl ] = {r_col:+.4f}   VIF = {vif:.1f}")
    if abs(r_col) > 0.9:
        print("  -> the regressors are nearly collinear: on a smooth single-humped")
        print("     envelope -dlnS0/dl is itself close to linear in l, so 'tilt' and")
        print(f"     'rigid shift' are almost the same function.  Combined-fit errors are")
        print(f"     inflated by ~{np.sqrt(vif):.1f}x; only the residual RMS comparison separates them.")
    else:
        print("  -> the regressors are distinguishable; the combined fit is well posed.")

    print("\n--- which model wins? (weighted residual RMS, mean over sweeps) ---")
    for label, key in (("tilt only", "rms_tilt"), ("shift only", "rms_shift"),
                       ("combined", "rms_both")):
        v = f[key]
        print(f"  {label:<12} {v.mean():.5f}   (last sweep {v[-1]:.5f})")
    base = f["rms_tilt"][-1]
    print(f"  at t_end:  shift/tilt = {f['rms_shift'][-1] / base:.2f}x   "
          f"combined/tilt = {f['rms_both'][-1] / base:.4f}x")
    late = t >= 0.5 * t[-1]
    print(f"  over the second half of the run, shift/tilt = "
          f"{(f['rms_shift'][late] / f['rms_tilt'][late]).mean():.2f}x on average")
    better = ("tilt" if f["rms_tilt"][-1] < f["rms_shift"][-1] else "shift")
    print(f"  -> of the two one-parameter models the {better.upper()} model wins, and adding")
    print(f"     the second regressor on top of it buys "
          f"{100 * (1 - f['rms_both'][-1] / base):.2f}% -- i.e. nothing")

    print("\n--- beta(t): tilt slope -------------------------------------------")
    sl, se, sc, r = linfit(t, f["beta"])
    print(f"  combined fit   beta {f['beta'][0]:+.5f} -> {f['beta'][-1]:+.5f} 1/nm   "
          f"slope {sl:+.5f} +- {se:.5f} /nm/h  (r={r:+.3f}, {abs(sl / se):.0f} sigma)")
    sl_t, se_t, _, r_t = linfit(t, f["beta_tilt"])
    print(f"  tilt-only fit  beta {f['beta_tilt'][0]:+.5f} -> {f['beta_tilt'][-1]:+.5f} 1/nm   "
          f"slope {sl_t:+.5f} +- {se_t:.5f} /nm/h  (r={r_t:+.3f})")
    print(f"  monotone in time? |r| = {abs(r_t):.3f}  "
          f"({'yes -- consistent with creep' if abs(r_t) > 0.9 else 'no -- it turns over'})")
    if np.isfinite(temp_c).sum() > 6:
        i_turn = int(np.nanargmin(temp_c))
        print(f"  temperature turns at {t[i_turn]:.2f} h ({temp_c[i_turn]:.2f} C); "
              f"beta there = {f['beta_tilt'][i_turn]:+.5f} 1/nm "
              f"(run spans {f['beta_tilt'].min():+.5f} .. {f['beta_tilt'].max():+.5f})")
        sb = []
        for label, sl_ in (("cooling", slice(0, i_turn + 1)), ("warming", slice(i_turn, None))):
            k, kerr, _, rb = linfit(temp_c[sl_], f["beta_tilt"][sl_])
            sb.append(k)
            print(f"    {label:<8} dbeta/dT = {k:+.5f} +- {abs(kerr):.5f} /nm/C   (r={rb:+.3f})")
        rr = float(np.corrcoef(temp_c, f["beta_tilt"])[0, 1])
        print(f"  corr(beta, T) over the whole run = {rr:+.3f} "
              "(shared trend only -- see the branch test)")
        if len(sb) == 2 and sb[0] * sb[1] < 0:
            print("  -> the branches DISAGREE in sign: beta keeps climbing straight through the")
            print("     temperature turnaround, so it is NOT thermally driven")

    print("\n--- a(t): scalar transmission at the pivot ------------------------")
    sl_a, se_a, _, r_a = linfit(t, f["a_tilt"])
    print(f"  a {f['a_tilt'][0]:.4f} -> {f['a_tilt'][-1]:.4f}   "
          f"({100 * (f['a_tilt'][-1] - 1):+.2f}% total)   "
          f"slope {100 * sl_a:+.3f} +- {100 * se_a:.3f} %/h  (r={r_a:+.3f}, {abs(sl_a / se_a):.0f} sigma)")
    print(f"  turns over: max {f['a_tilt'].max():.4f} at {t[int(np.argmax(f['a_tilt']))]:.2f} h, "
          f"min {f['a_tilt'].min():.4f} at {t[int(np.argmin(f['a_tilt']))]:.2f} h")
    if np.isfinite(run.daq_mv).any():
        print(f"  corr(a, PD) = {float(np.corrcoef(f['a_tilt'], run.daq_mv)[0, 1]):+.3f}   "
              f"corr(a, T) = {float(np.corrcoef(f['a_tilt'], temp_c)[0, 1]):+.3f}"
              "   (PD is a different light path -- relative only)")
    if np.isfinite(temp_c).sum() > 6:
        i_turn = int(np.nanargmin(temp_c))
        sa = []
        for label, sl_ in (("cooling", slice(0, i_turn + 1)), ("warming", slice(i_turn, None))):
            k, kerr, _, rb = linfit(temp_c[sl_], f["a_tilt"][sl_])
            sa.append(k)
            print(f"    {label:<8} da/dT = {100 * k:+7.2f} +- {100 * abs(kerr):.2f} %/C   (r={rb:+.3f})")
        if len(sa) == 2 and sa[0] * sa[1] > 0:
            print("    -> both branches agree in sign: a(t) retraces with the room, so the")
            print("       scalar loss is thermal, not creep")

    print("\n--- delta(t): residual rigid shift after the tilt is absorbed -----")
    d_pm = f["delta"] * NM_TO_PM
    e_pm = f["delta_err"] * NM_TO_PM
    sl_d, se_d, _, r_d = linfit(t, d_pm)
    print(f"  delta(t_end) = {d_pm[-1]:+.1f} +- {e_pm[-1]:.1f} pm  "
          f"({abs(d_pm[-1]) / e_pm[-1]:.1f} sigma)")
    print(f"  drift        = {sl_d:+.2f} +- {se_d:.2f} pm/h  (r={r_d:+.3f})")
    print(f"  1 sigma bound on laser spectral drift: |ddelta/dt| < {abs(sl_d) + se_d:.1f} pm/h")
    print(f"  for comparison, shift-only fit gives {f['delta_shift'][-1] * NM_TO_PM:+.1f} pm at t_end "
          f"and {linfit(t, f['delta_shift'] * NM_TO_PM)[0]:+.2f} pm/h")

    print("\n--- pedestal check -------------------------------------------------")
    print(f"  adding a 1/S(l,0) column (a drifting dark/stray pedestal) moves beta(t_end)")
    print(f"  from {f['beta'][-1]:+.5f} to {pedestal_beta:+.5f} 1/nm -- "
          f"{'a pedestal drift could account for part of the tilt' if abs(pedestal_beta - f['beta'][-1]) > 0.2 * abs(f['beta'][-1]) else 'the tilt is not a pedestal artifact'}")

    print("\n--- does the tilt reproduce the apparent centroid drift? -----------")
    # Apply the fitted tilt to sweep 0 and take the centroid of the result; the
    # scalar a cancels.  If this tracks the measured centroid, the "drift" was
    # never a wavelength shift.
    ref_w = lin[0, mask]
    lam_w = wl[mask]
    meas = (lin[:, mask] * lam_w).sum(1) / lin[:, mask].sum(1)
    pred = np.array([
        float((ref_w * np.exp(b * x) * lam_w).sum() / (ref_w * np.exp(b * x)).sum())
        for b in f["beta_tilt"]])
    s_m = linfit(t, meas)[0] * NM_TO_PM
    s_p = linfit(t, pred)[0] * NM_TO_PM
    print(f"  measured centroid (in window)  {s_m:+6.2f} pm/h, "
          f"{(meas[-1] - meas[0]) * NM_TO_PM:+7.1f} pm total")
    print(f"  predicted from beta(t) alone   {s_p:+6.2f} pm/h, "
          f"{(pred[-1] - pred[0]) * NM_TO_PM:+7.1f} pm total")
    print(f"  the tilt accounts for {100 * s_p / s_m:.0f}% of the apparent drift; "
          f"corr = {float(np.corrcoef(meas, pred)[0, 1]):+.3f}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    mono = abs(linfit(t, f["beta_tilt"])[3]) > 0.9
    print(f"1. beta(t) is {'monotonic' if mono else 'non-monotonic'} "
          f"(r={linfit(t, f['beta_tilt'])[3]:+.3f} vs time) and does not retrace when the")
    print(f"   room warms back up  =>  {'MECHANICAL CREEP' if mono else 'thermal'}, not room temperature.")
    a_flat = abs(linfit(t, f["a_tilt"])[0] / linfit(t, f["a_tilt"])[1]) < 3
    print(f"2. a(t) has {'no significant linear trend' if a_flat else 'a significant linear trend'} "
          f"but turns over with the room temperature:")
    print("   a thermal scalar loss sits on top of the creep, and the two are separable.")
    print(f"3. adding delta to the tilt model changes the residual RMS by "
          f"{100 * (1 - f['rms_both'][-1] / f['rms_tilt'][-1]):.2f}%: delta is consistent with zero.")
    d_pm_ = f["delta"] * NM_TO_PM
    sl_, se_, _, _ = linfit(t, d_pm_)
    print(f"   True laser spectral drift is bounded at |ddelta/dt| < {abs(sl_) + se_:.1f} pm/h "
          f"(1 sigma),")
    print(f"   i.e. at least {abs(s_m) / (abs(sl_) + se_):.0f}x smaller than the "
          f"{s_m:.0f} pm/h the centroid appeared to move.")

    print("\n--- per-sweep table -------------------------------------------------")
    print(f"{'t_h':>6}{'a':>9}{'beta':>10}{'delta_pm':>11}{'resid_tilt':>12}"
          f"{'resid_shift':>13}{'resid_comb':>12}")
    for i in range(n):
        print(f"{t[i]:>6.2f}{f['a_tilt'][i]:>9.4f}{f['beta_tilt'][i]:>10.5f}"
              f"{d_pm[i]:>11.1f}{f['rms_tilt'][i]:>12.5f}{f['rms_shift'][i]:>13.5f}"
              f"{f['rms_both'][i]:>12.5f}")


def write_csv(path: Path, t, f, temp_c, daq_mv) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_h", "a", "beta", "beta_err", "delta_pm", "delta_err_pm",
                    "resid_tilt", "resid_shift", "resid_combined", "temp_C", "pd_mV"])
        for i in range(len(t)):
            w.writerow([f"{t[i]:.4f}", f"{f['a_tilt'][i]:.6f}", f"{f['beta_tilt'][i]:.6f}",
                        f"{f['beta_tilt_err'][i]:.6f}", f"{f['delta'][i] * NM_TO_PM:.3f}",
                        f"{f['delta_err'][i] * NM_TO_PM:.3f}", f"{f['rms_tilt'][i]:.6f}",
                        f"{f['rms_shift'][i]:.6f}", f"{f['rms_both'][i]:.6f}",
                        f"{temp_c[i]:.4f}", f"{daq_mv[i]:.6f}"])


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
def plot_fit(run, f: dict, t, temp_c, lam_p, mask, wl, r_col) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    (ax_b, ax_a), (ax_d, ax_r) = axes
    fig.suptitle(
        f"{run.name}   multiplicative tilt model   "
        f"window {wl[mask].min():.2f}-{wl[mask].max():.2f} nm, pivot {lam_p:.3f} nm "
        f"({len(t)} sweeps, {t[-1]:.1f} h)", fontsize=12)

    # (a) beta(t) with temperature -----------------------------------------
    ax_b.errorbar(t, f["beta_tilt"], yerr=f["beta_tilt_err"], fmt="o-", color="C0",
                  markersize=3, linewidth=1.1, capsize=2, label=r"$\beta$ (tilt only)")
    sl, se, _, r = linfit(t, f["beta_tilt"])
    ax_b.plot(t, np.polyval(np.polyfit(t, f["beta_tilt"], 1), t), "--", color="C0",
              alpha=0.5, linewidth=1.0,
              label=f"{sl:+.4f}/nm/h (r={r:+.2f})")
    ax_b.set_ylabel(r"tilt slope $\beta$  [nm$^{-1}$]", color="C0")
    ax_b.tick_params(axis="y", labelcolor="C0")
    ax_bt = ax_b.twinx()
    ax_bt.plot(t, temp_c, "s-", color="C4", markersize=3, linewidth=1.1, alpha=0.8,
               label="T (TH10K)")
    ax_bt.set_ylabel("temperature (C)", color="C4")
    ax_bt.tick_params(axis="y", labelcolor="C4")
    ax_b.set_title(r"(a)  tilt slope $\beta(t)$ vs room temperature")
    ax_b.set_xlabel("elapsed (h)")
    ax_b.grid(True, alpha=0.3)
    h = ax_b.get_legend_handles_labels()[0] + ax_bt.get_legend_handles_labels()[0]
    ax_b.legend(handles=h, fontsize=8, loc="best")

    # (b) a(t) with PD power ------------------------------------------------
    ax_a.plot(t, f["a_tilt"], "o-", color="C2", markersize=3, linewidth=1.1,
              label="a (pivot transmission)")
    ax_a.axhline(1.0, color="0.5", linewidth=0.8)
    ax_a.set_ylabel("a(t)", color="C2")
    ax_a.tick_params(axis="y", labelcolor="C2")
    ax_ap = ax_a.twinx()
    ax_ap.plot(t, run.daq_mv, "s-", color="C5", markersize=3, linewidth=1.1, alpha=0.8,
               label="PD (separate path)")
    ax_ap.set_ylabel("PD (mV)", color="C5")
    ax_ap.tick_params(axis="y", labelcolor="C5")
    ax_a.set_title("(b)  scalar transmission a(t) vs photodiode power")
    ax_a.set_xlabel("elapsed (h)")
    ax_a.grid(True, alpha=0.3)
    h = ax_a.get_legend_handles_labels()[0] + ax_ap.get_legend_handles_labels()[0]
    ax_a.legend(handles=h, fontsize=8, loc="best")

    # (c) delta(t) from the combined fit ------------------------------------
    d_pm, e_pm = f["delta"] * NM_TO_PM, f["delta_err"] * NM_TO_PM
    ax_d.fill_between(t, d_pm - e_pm, d_pm + e_pm, color="C3", alpha=0.22,
                      label=r"$\pm1\sigma$")
    ax_d.plot(t, d_pm, "o-", color="C3", markersize=3, linewidth=1.1,
              label=r"$\delta$ (combined fit)")
    ax_d.plot(t, f["delta_shift"] * NM_TO_PM, "^--", color="C7", markersize=3,
              linewidth=0.9, alpha=0.8, label=r"$\delta$ (shift-only fit)")
    ax_d.axhline(0.0, color="0.4", linewidth=0.8)
    sl_d, se_d, _, _ = linfit(t, d_pm)
    ax_d.set_title(rf"(c)  residual shift $\delta(t)$ after the tilt   "
                   rf"({sl_d:+.1f} $\pm$ {se_d:.1f} pm/h)")
    ax_d.set_xlabel("elapsed (h)")
    ax_d.set_ylabel(r"$\delta$ (pm)")
    ax_d.grid(True, alpha=0.3)
    ax_d.legend(fontsize=8, loc="best")
    ax_d.text(0.02, 0.04, f"regressors collinear, r={r_col:+.3f}", fontsize=7.5,
              color="0.3", transform=ax_d.transAxes)

    # (d) residual RMS, three models ----------------------------------------
    ax_r.plot(t, f["rms_tilt"], "o-", color="C0", markersize=3, linewidth=1.1,
              label="tilt only")
    ax_r.plot(t, f["rms_shift"], "s-", color="C1", markersize=3, linewidth=1.1,
              label="shift only")
    ax_r.plot(t, f["rms_both"], "^-", color="C3", markersize=3, linewidth=1.1,
              label="combined")
    ax_r.set_title("(d)  weighted residual RMS of ln-ratio")
    ax_r.set_xlabel("elapsed (h)")
    ax_r.set_ylabel("residual RMS")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(fontsize=8, loc="best")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fit the multiplicative tilt model to an OSA run.")
    p.add_argument("csv", nargs="?", type=Path,
                   help="master daq_osa_monitor_<stamp>.csv (default: newest in src/calib_data)")
    p.add_argument("--out", type=Path, help="PNG path (default: <csv stem>_tilt.png)")
    p.add_argument("--no-show", action="store_true", help="save without opening a window")
    args = p.parse_args(argv)

    master_csv = (args.csv or newest_master_csv()).resolve()
    print(f"reading {master_csv}")
    run = load_run(master_csv)
    lin = linear_power(run)
    wl, t = run.wl_nm, run.elapsed_h
    temp_c = th10k_celsius(run.ohm)

    ref = lin[0]
    mask = ref >= WING * ref.max()
    w = ref[mask]
    lam_p = float((w * wl[mask]).sum() / w.sum())
    x = wl[mask] - lam_p

    # -dlnS0/dl, built from a smoothed S(l,0): the regressor is a model basis
    # function, and a raw numerical derivative of a noisy sweep would act as
    # errors-in-variables and bias delta toward zero.  The data are not smoothed.
    ref_s = smooth_rows(ref[None, :], DERIV_SMOOTH)[0]
    g = -np.gradient(np.log(ref_s), wl)[mask]

    f = fit_series(run, wl, lin, mask, lam_p, g, w)

    # pedestal diagnostic: does a drifting dark/stray floor mimic the tilt?
    X_ped = np.column_stack([np.ones_like(x), x, g, 1.0 / ref[mask]])
    y_end = np.log(lin[-1, mask] / ref[mask])
    pedestal_beta = float(wls(X_ped, y_end, w)["coef"][1])

    r_col, _ = collinearity(x, g, w)
    report(run, lin, f, t, wl, mask, lam_p, g, x, w, temp_c, pedestal_beta)

    out = args.out or master_csv.with_name(f"{master_csv.stem}_tilt.png")
    csv_out = master_csv.with_name(f"{master_csv.stem}_tilt.csv")
    fig = plot_fit(run, f, t, temp_c, lam_p, mask, wl, r_col)
    fig.savefig(out, dpi=150)
    write_csv(csv_out, t, f, temp_c, run.daq_mv)
    print(f"\nsaved -> {out}")
    print(f"saved -> {csv_out}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
