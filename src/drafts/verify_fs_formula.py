"""
Check the polygon / full-scale formula against the RbONN model, for any
number of pairs n (the sweep is 2n-dimensional).

The formula in question assumes n phasors of FIXED magnitudes R_k and FREE
phases.  With S = sum_k R_k:

    Y_max = S^2                                        (all phases aligned)
    Y_min = (2 R_max - S)^2  if R_max >= S/2           (one pair dominates)
          = 0                if R_max <  S/2           (polygon closes)
    FS    = Y_max - Y_min
          = 4 R_max sum_{k!=max} R_k  if R_max >= S/2  (split form)
          = S^2                       if R_max <  S/2

Part A verifies that formula numerically in its own model (free phases).

Part B applies it to RbONN with R_k = G_k, the largest magnitude a pair can
reach, and compares with the true optimum.  It does NOT match, because in
RbONN phase and magnitude are not independent:

    z_k = G_k sin(u_k) sin(v_k) exp( j (Phi_k - u_k - v_k) )

A pair only reaches |z_k| = G_k at u_k = v_k = pi/2, and there its phase is
pinned to Phi_k - pi.  Buying phase costs amplitude, so (sum G_k)^2 is an
upper bound that is attained only if all Phi_k are equal.

Part C gives the cheap exact replacement for the 2n-D sweep.  Because the
pairs are controlled independently, the reachable set of the sum is the
Minkowski sum of the per-pair sets, and its support function is the SUM of
the per-pair support functions -- so for a fixed projection axis psi each
pair is optimized on its own:

    h_k(psi) = max_{u,v} Re( z_k e^{-j psi} )
             = max_{t in [0,pi]} G_k (1 - cos t)/2 cos(a_k - t) ,  a_k = Phi_k - psi

  (for fixed s = u+v the phase is fixed and sin u sin v peaks at u = v = s/2,
   so u = v = t/2 is exactly optimal), and d/dt = 0 gives the closed form

    t* = (pi + 2 a_k)/3  (+ 2 pi n/3, branches in [0, pi]; endpoints too)

    Y_max = max_psi ( sum_k h_k(psi) )^2

The psi scan stays 1-D for every n; only the per-psi sum grows, so the cost
is O(n) while a grid sweep is O(m^(2n)).

Part D runs that comparison for n = 2..8 pairs (up to a 16-D sweep).

Usage
    python verify_fs_formula.py calib_step7_result_0910_2034.json
    python verify_fs_formula.py calib.json --pairs 1 2 3 4 5 6
    python verify_fs_formula.py calib.json --pairs 1 5 6 --max-pairs 8
"""
import argparse
import time

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from rbonn_3pair_sweep import (AXIS, GRID, INK, INK2, MUTED, SURFACE, HALF_PI,
                               Z, exhaustive, load, refine, support_scan)

# 8-slot categorical palette (see dataviz reference palette)
SERIES8 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


# ------------------------------------------------- A. the formula itself ----
def formula(R):
    """Free-phase prediction for fixed magnitudes R."""
    R = np.asarray(R, float)
    S, Rmax = R.sum(), R.max()
    dominates = 2.0 * Rmax >= S
    ymax = S ** 2
    ymin = (2.0 * Rmax - S) ** 2 if dominates else 0.0
    fs_split = 4.0 * Rmax * (S - Rmax) if dominates else S ** 2
    return ymax, ymin, fs_split, dominates


def brute_free_phase(R, restarts=200, seed=0):
    """max/min |sum R_k e^{j phi_k}|^2 over free phases, by multistart L-BFGS-B
    on the relative phases (phi_0 = 0)."""
    R = np.asarray(R, float)
    rng = np.random.default_rng(seed)
    out = []
    for sign in (+1.0, -1.0):
        best = -np.inf
        for s in [rng.uniform(0.0, 2.0 * np.pi, len(R) - 1) for _ in range(restarts)]:
            r = minimize(lambda p: -sign * abs(R[0] + (R[1:] * np.exp(1j * p)).sum()) ** 2,
                         s, method="L-BFGS-B", options=dict(maxiter=500, ftol=1e-18))
            best = max(best, -r.fun)
        out.append(sign * best)
    return out[0], out[1]


def part_a():
    print("=" * 78)
    print("A. formula vs brute force, in its own model (fixed |z_k|, free phases)")
    print("=" * 78)
    cases = [("equal", [1.0, 1.0, 1.0]),
             ("closes", [0.55, 0.5, 0.45]),
             ("dominant", [1.0, 0.2, 0.15]),
             ("marginal", [0.5, 0.3, 0.2]),
             ("6 phasors", [0.9, 0.5, 0.4, 0.3, 0.2, 0.1]),
             ("8 phasors", [0.9, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.02])]
    rng = np.random.default_rng(1)
    cases += [(f"random {i+1}", rng.uniform(0.05, 1.0, 3 + i).round(4)) for i in range(3)]

    print(f"  {'case':<11} {'n':>2} {'regime':<9} "
          f"{'Y_max err':>10} {'Y_min err':>10} {'FS split err':>13}")
    worst = 0.0
    for name, R in cases:
        ymax, ymin, fs_split, dom = formula(R)
        bmax, bmin = brute_free_phase(R)
        e_max, e_min = abs(ymax - bmax), abs(ymin - bmin)
        e_fs = abs(fs_split - (ymax - ymin))          # split form vs Y_max - Y_min
        worst = max(worst, e_max, e_min, e_fs)
        print(f"  {name:<11} {len(R):>2} {'dominant' if dom else 'closes':<9} "
              f"{e_max:10.2e} {e_min:10.2e} {e_fs:13.2e}")
    print(f"\n  worst deviation {worst:.2e}  (multistart optimizer tolerance)")
    print("  -> the formula is exact FOR FREE PHASES, any n; the split form is the")
    print("     identity S^2 - (2R_max - S)^2 = 4 R_max (S - R_max).")


# ------------------------------------------------ C. closed-form support ----
def h_closed(psi, G, Phi):
    """h_k(psi) = max_{u,v} Re(z_k e^{-j psi}), and the t = u+v that attains it."""
    a = Phi - np.atleast_1d(psi)                                  # (P,)
    # stationary points repeat every 2 pi / 3 in t; a spans (-2 pi, 2 pi], so the
    # base (pi + 2a)/3 spans (-pi, 5 pi/3] and needs n = -3..3 to be sure every
    # branch landing in [0, pi] is tried (too narrow a window silently undershoots).
    n = np.arange(-3.0, 4.0)[:, None]
    t = np.concatenate([(np.pi + 2.0 * a) / 3.0 + 2.0 * np.pi * n / 3.0,
                        np.zeros_like(a)[None, :], np.full_like(a, np.pi)[None, :]])
    ok = (t >= -1e-12) & (t <= np.pi + 1e-12)
    t = np.clip(t, 0.0, np.pi)
    f = np.where(ok, G * 0.5 * (1.0 - np.cos(t)) * np.cos(a - t), -np.inf)
    i = f.argmax(axis=0)
    j = np.arange(f.shape[1])
    return f[i, j], t[i, j]


def closed_form_max(G, Phi, n_psi=200001):
    """Y_max = max_psi (sum_k h_k(psi))^2 -- a 1-D scan for any n, then a refine."""
    def total(psi):
        return sum(h_closed(psi, G[k], Phi[k])[0] for k in range(len(G)))

    psi = np.linspace(-np.pi, np.pi, n_psi)
    s = total(psi)
    i = int(s.argmax())
    d = psi[1] - psi[0]
    r = minimize_scalar(lambda p: -total(np.array([p]))[0],
                        bracket=(psi[i] - d, psi[i], psi[i] + d))
    psi_star = float(r.x)
    t = np.array([h_closed(np.array([psi_star]), G[k], Phi[k])[1][0] for k in range(len(G))])
    u = v = t / 2.0                        # u = v = (u+v)/2 is exactly optimal
    return total(np.array([psi_star]))[0] ** 2, u, v, psi_star


def part_b_c(path, pairs, grid, fine, restarts):
    G, Phi, _ = load(path, pairs)
    n = len(G)
    S, Rmax = G.sum(), G.max()

    print("\n" + "=" * 78)
    print(f"B. the formula applied to RbONN, {n} pairs, R_k = G_k (max reachable |z_k|)")
    print("=" * 78)
    ymax_f, ymin_f, fs_f, dom = formula(G)
    print(f"  G_k         {np.round(G, 5)}")
    print(f"  Phi_k [deg] {np.round(np.degrees(Phi), 2)}")
    print(f"  S = sum G_k {S:.6f}    R_max {Rmax:.6f}    S/2 {S/2:.6f}"
          f"   -> {'one pair dominates' if dom else 'polygon closes'}")
    print(f"  formula:  Y_max = S^2 = {ymax_f:.6f}    Y_min = {ymin_f:.6f}    FS = {fs_f:.6f}")

    t0 = time.perf_counter()
    ycf, u, v, psi_star = closed_form_max(G, Phi)
    t_cf = time.perf_counter() - t0
    ysup, usup, vsup, _ = support_scan(G, Phi, fine)
    t0 = time.perf_counter()
    yref, zref = refine(G, Phi, restarts, x0=np.concatenate([usup, vsup]))
    t_ref = time.perf_counter() - t0

    print(f"  truth:    Y_max = {yref:.6f}  ({2*n}-D multistart refinement)")
    print(f"  deficit:  {ymax_f - yref:.6f} V = {100*(ymax_f - yref)/ymax_f:.2f} % of S^2")
    print("  -> the formula OVER-predicts: it assumes each pair can hold |z_k| = G_k")
    print("     at any phase, but |z_k| = G_k forces arg z_k = Phi_k - pi.")
    print(f"     Phi spread here is {np.degrees(Phi.max()-Phi.min()):.1f} deg, so the"
          f" phasors cannot all align.")
    print("  -> Y_min = 0 does match, but trivially: RbONN can zero an amplitude")
    print("     (x_k = 0), so it never needs the polygon to close.")

    print("\n" + "=" * 78)
    print(f"C. the cheap exact route: 1-D scan over psi, {n} pairs ({2*n}-D sweep)")
    print("=" * 78)
    print(f"  closed form   Y_max = {ycf:.9f}   ({t_cf*1e3:.0f} ms)")
    print(f"  {2*n}-D refine   Y_max = {yref:.9f}   ({t_ref*1e3:.0f} ms, "
          f"restarts {restarts})")
    if n == 3:
        t0 = time.perf_counter()
        ygrid, *_ = exhaustive(G, Phi, grid)
        print(f"  {grid}^6 grid   Y_max = {ygrid:.9f}   "
              f"({(time.perf_counter()-t0)*1e3:.0f} ms, {grid**6:,} points)")
    else:
        print(f"  {grid}^{2*n} grid  skipped -- {grid**(2*n):.3e} points")
    print(f"  |closed form - refine| = {abs(ycf - yref):.2e}")
    print(f"  psi* = {np.degrees(psi_star):.4f} deg")
    print(f"  {'pair':>4}  {'x_k':>7}  {'w_k':>7}  {'theta^x':>8}  {'theta^w':>8}"
          f"  {'refine x':>9}  {'refine w':>9}")
    for k, p in enumerate(pairs):
        print(f"  {p:>4}  {np.sin(u[k]):7.4f}  {np.sin(v[k]):7.4f}"
              f"  {np.degrees(2*u[k]):7.2f}°  {np.degrees(2*v[k]):7.2f}°"
              f"  {np.sin(zref[k]):9.4f}  {np.sin(zref[n+k]):9.4f}")
    return G, Phi, u, v, psi_star, ycf


# ------------------------------------------------------ D. scaling in n ----
def part_d(path, max_pairs, fine, restarts):
    print("\n" + "=" * 78)
    print(f"D. scaling: closed form vs {2}n-D multistart, n = 2..{max_pairs} pairs")
    print("=" * 78)
    print(f"  {'n':>2} {'dims':>5} {'closed form Y_max':>18} {'multistart Y_max':>18}"
          f" {'|diff|':>9} {'t_cf':>8} {'t_ms':>8}")
    rows = []
    for n in range(2, max_pairs + 1):
        pairs = list(range(1, n + 1))
        G, Phi, _ = load(path, pairs)
        t0 = time.perf_counter()
        ycf, *_ = closed_form_max(G, Phi)
        t_cf = time.perf_counter() - t0
        ysup, usup, vsup, _ = support_scan(G, Phi, fine)
        t0 = time.perf_counter()
        yref, _ = refine(G, Phi, restarts, x0=np.concatenate([usup, vsup]))
        t_ms = time.perf_counter() - t0
        print(f"  {n:>2} {2*n:>5} {ycf:18.9f} {yref:18.9f} {abs(ycf-yref):9.1e}"
              f" {t_cf*1e3:7.0f}m {t_ms*1e3:7.0f}m")
        rows.append(dict(n=n, ycf=ycf, yref=yref, t_cf=t_cf, t_ms=t_ms,
                         bound=G.sum() ** 2))
    print("  -> the closed form stays 1-D in psi for every n; only the per-psi sum")
    print(f"     grows (O(n)).  A grid sweep would be O(m^(2n)).")
    return rows


# --------------------------------------------------------------- plots ----
def plot(G, Phi, u, v, psi_star, ymax, pairs, rows, path, show_window):
    import matplotlib
    if not show_window:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.spatial import ConvexHull

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.labelsize": 10, "axes.spines.top": False,
        "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "font.size": 9,
        "legend.frameon": False,
    })
    n = len(G)
    col = [SERIES8[k % 8] for k in range(n)]
    z_opt = Z(u, v, G, Phi)
    fig, axs = plt.subplots(1, 4, figsize=(22, 5.6), layout="constrained")

    # --- 1. reachable set per pair (<=3 pairs) or the phasor chain at Y_max
    ax = axs[0]
    if n <= 3:
        g = np.linspace(0.0, HALF_PI, 161)
        U, V = np.meshgrid(g, g)
        ang = np.linspace(0, 2 * np.pi, 361)
        for k in range(n):
            z = Z(U, V, G[k], Phi[k]).ravel()
            pts = np.column_stack([z.real, z.imag])
            hull = ConvexHull(pts).vertices
            ax.fill(pts[hull, 0], pts[hull, 1], color=col[k], alpha=0.12, lw=0)
            ax.plot(np.r_[pts[hull, 0], pts[hull, 0][0]], np.r_[pts[hull, 1], pts[hull, 1][0]],
                    color=col[k], lw=2)
            ax.plot(G[k] * np.cos(ang), G[k] * np.sin(ang), color=AXIS, lw=1.2, ls="--")
            ax.plot(z_opt[k].real, z_opt[k].imag, "o", ms=9, mfc="white", mec=col[k],
                    mew=2, zorder=5)
        handles = [Line2D([], [], color=col[k], lw=2, label=f"pair {pairs[k]} reachable set")
                   for k in range(n)]
        handles += [Line2D([], [], color=AXIS, lw=1.2, ls="--",
                           label="free-phase circle |z| = G_k"),
                    Line2D([], [], color=MUTED, lw=0, marker="o", mfc="white", mec=INK,
                           mew=1.5, label="z_k at Y_max")]
        title = "what one pair can reach vs what the formula assumes"
    else:
        tail, tips = 0j, [0j]
        for k in range(n):
            tip = tail + z_opt[k]
            ax.annotate("", xy=(tip.real, tip.imag), xytext=(tail.real, tail.imag),
                        arrowprops=dict(arrowstyle="-|>", color=col[k], lw=2,
                                        shrinkA=0, shrinkB=0, mutation_scale=12))
            tips.append(tip)
            tail = tip
        ax.plot([0, tail.real], [0, tail.imag], ls="--", lw=1.5, color=INK)
        P = np.array(tips)
        pad = 0.1 * max(np.ptp(P.real), np.ptp(P.imag))
        ax.set_xlim(P.real.min() - pad, P.real.max() + pad)
        ax.set_ylim(P.imag.min() - pad, P.imag.max() + pad)
        handles = [Line2D([], [], color=col[k], lw=2, label=f"pair {pairs[k]}")
                   for k in range(n)]
        handles += [Line2D([], [], color=INK, lw=1.5, ls="--", label="resultant")]
        title = f"phasor chain at Y_max ({n} pairs)"
    ax.axhline(0, color=GRID, lw=0.8)
    ax.axvline(0, color=GRID, lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("Re z")
    ax.set_ylabel("Im z")
    ax.set_title(title, loc="left")
    ax.legend(handles=handles, loc="lower left", labelcolor=INK2, fontsize=8)

    # --- 2. support curves: sum_k h_k(psi) and the bound sum_k G_k
    ax = axs[1]
    psi = np.linspace(-np.pi, np.pi, 3601)
    hs = np.array([h_closed(psi, G[k], Phi[k])[0] for k in range(n)])
    tot = hs.sum(axis=0)
    d = np.degrees(psi)
    for k in range(n):
        ax.plot(d, hs[k], color=col[k], lw=2)
    ax.plot(d, tot, color=INK, lw=2)
    ax.axhline(G.sum(), color=AXIS, lw=1.5, ls="--")
    ax.annotate(f"free-phase bound  sum G_k = {G.sum():.4f}", (d[0], G.sum()),
                xytext=(4, 5), textcoords="offset points", color=INK2, va="bottom")
    ax.plot(np.degrees(psi_star), np.sqrt(ymax), "o", ms=9, mfc="white", mec=INK,
            mew=1.5, zorder=5)
    ax.annotate(f"max at psi* = {np.degrees(psi_star):.1f}°\n"
                f"sum h_k = {np.sqrt(ymax):.4f}   Y = {ymax:.5f}",
                (np.degrees(psi_star), np.sqrt(ymax)), xytext=(0.30, 0.80),
                textcoords="axes fraction", color=INK, va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.set_xlim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_xlabel("projection axis psi [deg]")
    ax.set_ylabel("support h(psi)")
    ax.set_title(f"the 1-D problem that replaces the {2*n}-D sweep", loc="left")
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], color=col[k], lw=2, label=f"pair {pairs[k]}")
                       for k in range(n)]
              + [Line2D([], [], color=INK, lw=2, label="sum of pairs")],
              loc="center left", bbox_to_anchor=(0.62, 0.55), labelcolor=INK2, fontsize=8)

    # --- 3. the levels side by side
    ax = axs[2]
    names = ["formula\n(sum G_k)²", "true Y_max\n(this model)", "all θ = 180°",
             "incoherent\nsum G_k²"]
    vals = [G.sum() ** 2, ymax, abs((G * np.exp(1j * Phi)).sum()) ** 2, (G ** 2).sum()]
    bars = ax.bar(names, vals, color=[AXIS, SERIES8[0], SERIES8[1], SERIES8[2]], width=0.62)
    for b, val in zip(bars, vals):
        ax.annotate(f"{val:.5f}", (b.get_x() + b.get_width() / 2, val), xytext=(0, 4),
                    textcoords="offset points", ha="center", color=INK)
    ax.set_ylabel("Y [V]")
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_title("the formula over-predicts Y_max by "
                 f"{100*(G.sum()**2 - ymax)/G.sum()**2:.1f} %", loc="left")
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)

    # --- 4. scaling in the number of pairs
    ax = axs[3]
    ns = [r["n"] for r in rows]
    for key, c, lab in ((("t_cf"), SERIES8[0], "closed form (1-D scan)"),
                        (("t_ms"), SERIES8[1], "multistart, 2n-D")):
        y = [r[key] * 1e3 for r in rows]
        ax.plot(ns, y, color=c, lw=2, marker="o", ms=8, mfc=SURFACE, mew=2)
        ax.annotate(lab, (ns[-1], y[-1]), xytext=(-6, 10), textcoords="offset points",
                    color=c, ha="right")
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xlabel("number of pairs n   (sweep dimension 2n)")
    ax.set_ylabel("time [ms]")
    ax.set_title("cost vs number of pairs (same Y_max to ~1e-9)", loc="left")
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    fig.suptitle(f"RbONN pairs {pairs}: polygon / full-scale formula check",
                 color=INK, fontsize=13, x=0.005, ha="left")
    fig.savefig(path, dpi=150)
    print(f"\nplot saved -> {path}")
    if show_window:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--pairs", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--max-pairs", type=int, default=8, help="largest n for part D")
    ap.add_argument("--grid", type=int, default=25)
    ap.add_argument("--fine", type=int, default=201)
    ap.add_argument("--restarts", type=int, default=400)
    ap.add_argument("--plot", default=None)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    part_a()
    G, Phi, u, v, psi_star, ycf = part_b_c(a.json, a.pairs, a.grid, a.fine, a.restarts)
    rows = part_d(a.json, a.max_pairs, a.fine, a.restarts)
    path = a.plot or f"fs_formula_check_{''.join(map(str, a.pairs))}.png"
    plot(G, Phi, u, v, psi_star, ycf, a.pairs, rows, path, a.show)


if __name__ == "__main__":
    main()
