"""
Round-trip check for 6-pair combinations drawn from the 8 calibrated pairs.

For each combination:
  1. find Y_max with the 1-D psi scan (closed form per pair), and record the
     12 settings x_k = sin(u_k), w_k = sin(v_k) that attain it;
  2. feed those 12 recorded numbers back through the original model

        Y = | sum_k G_k sin(u_k) sin(v_k) exp( j (Phi_k - u_k - v_k) ) |^2
        with u_k = arcsin(x_k), v_k = arcsin(w_k)

     -- no shortcuts, the prediction is not reused;
  3. compare Y_max_pred with Y_max_real, and also with what you get if the
     recorded x, w are rounded to 4 decimals before being applied.

Usage
    python report_6pair_combos.py calib_step7_result_0910_2034.json
    python report_6pair_combos.py calib.json --combos 1,2,3,4,5,6 2,3,4,5,6,7
"""
import argparse

import numpy as np

from rbonn_3pair_sweep import AXIS, GRID, INK, INK2, MUTED, SURFACE, load
from verify_fs_formula import SERIES8, closed_form_max

DEFAULT_COMBOS = ["1,2,3,4,5,6", "1,2,3,4,5,7", "1,2,3,4,7,8",
                  "2,3,5,6,7,8", "3,4,5,6,7,8"]


def Y_from_xw(x, w, G, Phi):
    """The original model, evaluated from the recorded x, w only."""
    u = np.arcsin(np.clip(np.asarray(x, float), 0.0, 1.0))
    v = np.arcsin(np.clip(np.asarray(w, float), 0.0, 1.0))
    return abs((G * np.sin(u) * np.sin(v) * np.exp(1j * (Phi - u - v))).sum()) ** 2


def run(path, combos, dp):
    rows = []
    for combo in combos:
        pairs = [int(s) for s in combo.split(",")]
        G, Phi, _ = load(path, pairs)
        ypred, u, v, psi = closed_form_max(G, Phi)
        x, w = np.sin(u), np.sin(v)                       # the 12 recorded settings
        yreal = Y_from_xw(x, w, G, Phi)                   # recomputed from scratch
        yround = Y_from_xw(np.round(x, dp), np.round(w, dp), G, Phi)
        rows.append(dict(pairs=pairs, x=x, w=w, psi=psi, ypred=ypred,
                         yreal=yreal, yround=yround))
    return rows


def report(rows, dp):
    print("=" * 96)
    print("6-pair combinations: 1-D scan prediction vs the original formula re-evaluated")
    print("=" * 96)
    for r in rows:
        print(f"\n  pairs {r['pairs']}    psi* = {np.degrees(r['psi']):8.3f} deg")
        print(f"    {'pair':>5} {'x_k':>9} {'w_k':>9} {'theta^x':>9} {'theta^w':>9}")
        for k, p in enumerate(r["pairs"]):
            print(f"    {p:>5} {r['x'][k]:9.6f} {r['w'][k]:9.6f}"
                  f" {np.degrees(2*np.arcsin(r['x'][k])):8.3f}°"
                  f" {np.degrees(2*np.arcsin(r['w'][k])):8.3f}°")
        print(f"    Y_max_pred {r['ypred']:.12f}")
        print(f"    Y_max_real {r['yreal']:.12f}   (from the 12 x, w above)")
        print(f"    difference {abs(r['yreal']-r['ypred']):.3e}"
              f"   rel {abs(r['yreal']-r['ypred'])/r['ypred']:.3e}")
        print(f"    Y at x, w rounded to {dp} dp: {r['yround']:.12f}"
              f"   ({100*(r['yround']-r['yreal'])/r['yreal']:+.4f} %)")

    print("\n" + "=" * 96)
    print(f"  {'pairs':<22} {'Y_max_pred':>14} {'Y_max_real':>14} {'|diff|':>10}"
          f" {'rel':>10} {f'Y @ {dp}dp':>14}")
    for r in rows:
        d = abs(r["yreal"] - r["ypred"])
        print(f"  {str(r['pairs']):<22} {r['ypred']:14.9f} {r['yreal']:14.9f}"
              f" {d:10.1e} {d/r['ypred']:10.1e} {r['yround']:14.9f}")
    worst = max(abs(r["yreal"] - r["ypred"]) / r["ypred"] for r in rows)
    print(f"\n  worst relative difference {worst:.2e}"
          f"  -> the recorded x, w reproduce the predicted Y_max")


def plot(rows, dp, path, show_window):
    import matplotlib
    if not show_window:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.labelsize": 10, "axes.spines.top": False,
        "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "font.size": 9,
        "legend.frameon": False,
    })
    labels = ["\n".join([",".join(map(str, r["pairs"][:3])),
                         ",".join(map(str, r["pairs"][3:]))]) for r in rows]
    pred = np.array([r["ypred"] for r in rows])
    real = np.array([r["yreal"] for r in rows])
    rel = np.abs(real - pred) / pred

    fig, axd = plt.subplot_mosaic([["bars", "resid"], ["table", "table"]],
                                  figsize=(16, 9.5), layout="constrained",
                                  gridspec_kw=dict(height_ratios=[1.0, 1.25]))
    fig.suptitle("6-pair combinations: predicted Y_max vs the original formula "
                 "re-evaluated from the recorded x, w",
                 color=INK, fontsize=13, x=0.005, ha="left")

    # --- predicted vs recomputed, side by side
    ax = axd["bars"]
    i = np.arange(len(rows))
    ax.bar(i - 0.19, pred, width=0.36, color=SERIES8[0], label="Y_max_pred (1-D scan)")
    ax.bar(i + 0.19, real, width=0.36, color=SERIES8[1],
           label="Y_max_real (original formula)")
    for k in i:
        ax.annotate(f"{real[k]:.6f}", (k, real[k]), xytext=(0, 5),
                    textcoords="offset points", ha="center", color=INK)
    ax.set_xticks(i)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, pred.max() * 1.18)
    ax.set_ylabel("Y [V]")
    ax.set_xlabel("pairs in the combination")
    ax.set_title("the two agree to every digit shown", loc="left")
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper right", labelcolor=INK2)

    # --- residuals
    ax = axd["resid"]
    ax.bar(i, np.maximum(rel, 1e-18), width=0.5, color=SERIES8[0])
    ax.axhline(2.2e-16, color=AXIS, lw=1.5, ls="--")
    ax.annotate("double-precision floor  ~2e-16", (len(rows) - 0.5, 2.2e-16),
                xytext=(0, 6), textcoords="offset points", ha="right",
                color=INK2, va="bottom")
    for k in i:
        ax.annotate(f"{rel[k]:.0e}", (k, max(rel[k], 1e-18)), xytext=(0, 4),
                    textcoords="offset points", ha="center", color=INK)
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 1e-10)
    ax.set_xticks(i)
    ax.set_xticklabels(labels)
    ax.set_ylabel("|Y_real - Y_pred| / Y_pred")
    ax.set_xlabel("pairs in the combination")
    ax.set_title("relative difference: round-trip error only", loc="left")
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)

    # --- the recorded settings
    ax = axd["table"]
    ax.axis("off")
    cols = ["pairs"] + [f"pair {p}" for p in range(1, 9)] + \
           ["Y_max_pred", "Y_max_real", f"Y @ {dp} dp"]
    cells = []
    for r in rows:
        row = [",".join(map(str, r["pairs"]))]
        for p in range(1, 9):
            if p in r["pairs"]:
                k = r["pairs"].index(p)
                row.append(f"{r['x'][k]:.4f}\n{r['w'][k]:.4f}")
            else:
                row.append("-")
        row += [f"{r['ypred']:.9f}", f"{r['yreal']:.9f}", f"{r['yround']:.9f}"]
        cells.append(row)
    tb = ax.table(cellText=cells, colLabels=cols, cellLoc="center", loc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.5)
    tb.scale(1, 3.0)
    for (row, col), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor(SURFACE)
        cell.get_text().set_color(INK if row else INK2)
        if row == 0:
            cell.get_text().set_fontweight("bold")
        elif col == 0:
            cell.get_text().set_color(INK2)
        elif 1 <= col <= 8 and cells[row - 1][col] != "-":
            cell.get_text().set_color(SERIES8[(col - 1) % 8])
    ax.set_title("the 12 recorded settings per combination "
                 "(x on the upper line, w on the lower) and the resulting Y",
                 loc="left", color=INK)

    fig.savefig(path, dpi=150)
    print(f"\nplot saved -> {path}")
    if show_window:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--combos", nargs="+", default=DEFAULT_COMBOS,
                    help="comma-separated pair indices, e.g. 1,2,3,4,5,6")
    ap.add_argument("--dp", type=int, default=4, help="rounding test on x, w")
    ap.add_argument("--plot", default="report_6pair_combos.png")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    rows = run(a.json, a.combos, a.dp)
    report(rows, a.dp)
    plot(rows, a.dp, a.plot, a.show)


if __name__ == "__main__":
    main()
