"""Manual smoke test: predict vs measure, with one pair and with two pairs.

Steps 3-7 calibrated every parameter the forward model needs, and a single
combined step-7 result JSON carries all of it: the channel layout (``step3``),
each pair's TPA efficiency ``eta`` + single-beam background (``step6``, embedded
by the step-7 REFIT) and each pair's comb phase ``Phi_k`` vs the reference pair
(``step7``).  So for any drive ``(x_k, w_k)`` the model predicts the detector
output with no free parameters::

    E      = sum_k  eta_k sqrt(x_k w_k) exp(i [phi_half(x_k) + phi_half(w_k) + Phi_k])
    Y_pred = |E|^2 + sum_k single_beam_k(x_k, w_k)

with ``phi_half(x) = asin(sqrt(x))`` and ``Phi_ref = 0``.  This script drives
the two simplest cases, measures, and prints the difference:

* **one pair**  -- pair k alone, swept over ``SWEEP_POINTS`` levels with
  ``x_k = w_k = v``.
* **two pairs** -- pairs j and k driven together, both at ``x = w = v``.

Nothing is fitted to the new data; ``diff = meas - pred`` and
``pull = diff / std`` are the whole result -- ``std`` being the spread of the
low-passed trace, the one uncertainty these drafts carry.

Not a pytest test (no mocks, needs real hardware) -- run it directly::

    python src/drafts/calib_step8_simple.py            # COLLECT: drive, write raw
                                                       #   CSV, compare in place
    python src/drafts/calib_step8_simple.py some.csv   # COMPARE an existing CSV
                                                       #   offline, no hardware

``--bounded`` / ``--fix`` pick WHICH stored step-7 spectrum to predict from (a
combined JSON may carry both; no flag defaults to ``PHASE_METHOD``).  Pass both
to compare them back to back.  ``--flip`` mirrors the step-6/7 inverted-DAQ
convention on a COLLECT: raw mean + dark are negated as they are read, so the
CSV already stores the positive light signal (re-analyze it WITHOUT --flip).
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "drafts"))  # for draft_hw

from draft_hw import connect_daq, connect_slm, read_point  # noqa: E402
from slm_module.calibration.calibration_new import calibration_result_from_dict  # noqa: E402
from slm_module.encoding import channel_layout_from_calibration  # noqa: E402
from calibration_module.phase import (  # noqa: E402
    load_comb_phase_json,
    load_pair_models,
    phi_half,
)

# ---- Edit these to match your setup ----
CALIB_PATH = REPO_ROOT / "src/calib_data"          # data directory: inputs + outputs live here

# The ONE input: a combined step-7 result JSON (calib_step7_test.py REFIT output),
# which embeds step 3 (-> layout) and step 6 (-> eta + single-beam per pair).
IN_STEP7 = CALIB_PATH / "calib_step7_result_0828_2333.json"   # ref pair 0, targets 2 + 4

PAIRS = [0, 2, 4]           # pairs to drive alone; each needs a step-6 model, and a
                            # step-7 phase unless it IS the reference pair (Phi = 0)
COMBOS = None               # two-pair blocks; None -> every unordered pair of PAIRS,
                            # e.g. [(0, 2), (0, 4), (2, 4)].  Set explicitly to trim.
PHASE_METHOD = "bounded"    # default stored step-7 fit to predict from (--bounded/--fix)

# ---- The drive ----
# Every point sets x = w = v on each driven pair (both beams of a pair share the
# level, as in step 7's target sweep); everything else is off.  A block is one
# driven set swept over these levels, so the run is
# (len(PAIRS) + len(COMBOS)) * SWEEP_POINTS points plus one dark.
SWEEP_MIN, SWEEP_MAX = 0.1, 1.0
SWEEP_POINTS = 6

OUT_DIR = CALIB_PATH        # all step-8 outputs live in the data directory

SLM_DISPLAY_NO = None       # None -> auto-detect the LCOS-SLM display
USB_SLM_NO = 1              # SLM_Ctrl_* device index for the DVI-mode switch

DAQ_DEVICE = "Dev1"
DAQ_CHANNEL = "ai0"

# Fixed per-point acquisition (daq_module), same windows as steps 6/7: the
# all-off dark reads the longer T_single window; driven points read T_both.
T_SINGLE_S = 5.0
T_BOTH_S = 3.0

SETTLE_S = 0.25             # wait after each SLM pattern change, before reading


# ======================================================================
# input loading  (layout + step-6 models + step-7 phases, all from IN_STEP7)
# ======================================================================

def load_inputs(method: str, pairs):
    """Layout, per-pair step-6 models and step-7 comb phases from ``IN_STEP7``.

    Returns ``(layout, models, phases)`` with ``phases[k]`` in radians for every
    ``pairs`` entry.  The reference pair defines ``Phi = 0`` and needs no stored
    step-7 fit; every other pair must carry a ``method`` fit.
    """
    payload = json.loads(IN_STEP7.read_text(encoding="utf-8"))
    step3 = payload.get("step3")
    if step3 is None:
        raise ValueError(
            f"{IN_STEP7} has no embedded 'step3' calibration; point IN_STEP7 at "
            f"a combined step-7 result (calib_step7_test.py REFIT output)"
        )
    layout = channel_layout_from_calibration(calibration_result_from_dict(step3))
    models = load_pair_models([IN_STEP7])              # reads the embedded "step6"
    ref_index, entries = load_comb_phase_json(IN_STEP7, method=method)
    phases = {ref_index: 0.0}
    phases.update({k: float(e["fit"]["dphi_comb_rad"]) for k, e in entries.items()})
    for k in pairs:
        if not (0 <= k < layout.n_channels):
            raise ValueError(
                f"pair {k} out of range (layout has {layout.n_channels} pairs)"
            )
        if k not in models:
            raise ValueError(f"no step-6 model for pair {k} in {IN_STEP7}")
        if k not in phases:
            raise ValueError(
                f"no step-7 '{method}' phase for pair {k} in {IN_STEP7} "
                f"(reference is pair {ref_index})"
            )
    print(f"Step 7 [{method}] vs ref {ref_index}:  "
          + "  ".join(f"Phi[{k}] = {np.degrees(phases[k]):+7.2f} deg" for k in pairs))
    print("Step 6:  " + "  ".join(f"eta[{k}] = {models[k].eta:.4g}" for k in pairs))
    return layout, models, phases


def build_blocks(pairs, combos):
    """``(blocks, levels)``: the driven sets, singles first, then the two-pair ones."""
    levels = np.round(np.linspace(SWEEP_MIN, SWEEP_MAX, SWEEP_POINTS), 6)
    singles = [(int(k),) for k in pairs]
    if combos is None:
        combos = itertools.combinations(pairs, 2)
    doubles = [tuple(int(k) for k in c) for c in combos]
    return singles + doubles, levels


# ======================================================================
# the forward model  (prediction is dark-free, like the analyzed y)
# ======================================================================

def predict(models, phases, driven, v) -> float:
    """One point's dark-free prediction ``|E|^2 + background``, all parameters fixed.

    With ``x = w = v`` the field of pair k is
    ``eta_k v exp(i[2 asin(sqrt(v)) + Phi_k])``.  The single-beam response is
    detector background, not TPA field, so it adds OUTSIDE the coherent sum.
    """
    ph_slm = 2.0 * float(phi_half(v))          # phi_half(x) + phi_half(w), x = w = v
    field = 0.0 + 0.0j
    bg = 0.0
    for k in driven:
        m = models[k]
        field += m.eta * float(v) * np.exp(1j * (ph_slm + phases[k]))
        bg += float(m.single_beam(v, v))
    return float(np.abs(field) ** 2) + bg


def predict_curve(models, phases, driven, levels) -> np.ndarray:
    """:func:`predict` over an array of levels."""
    return np.array([predict(models, phases, driven, v) for v in levels])


# ======================================================================
# persistence  (raw rows; the driven set is named in the `pairs` column)
# ======================================================================

def _csv_header(pairs) -> list[str]:
    cols = ["block", "pairs", "v"]
    for k in pairs:
        cols += [f"x_{k}", f"w_{k}"]
    return cols + ["dark_v", "voltage_mean_v", "voltage_std_v", "std_ratio",
                   "pred_v"]


def _pairs_tag(driven) -> str:
    """``(2, 4)`` -> ``"2+4"`` -- the self-describing driven-set column."""
    return "+".join(str(k) for k in driven)


def write_csv(path, pairs, rows, *, method: str) -> str:
    """One row per drive point.  ``rows`` are the dicts built by :func:`measure`.

    ``pred_v`` is the collect-time prediction, kept for eyeballing; a COMPARE
    recomputes it from ``IN_STEP7``.
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_csv_header(pairs))
        for r in rows:
            driven = r["driven"]
            line = [r["block"], _pairs_tag(driven), f"{r['v']:.6g}"]
            for k in pairs:
                on = float(r["v"]) if k in driven else 0.0
                line += [f"{on:.6g}", f"{on:.6g}"]
            mean_v = r["mean_v"]
            ratio = abs(r["std_v"] / mean_v) if mean_v else float("inf")
            line += [f"{r['dark_v']:.9g}", f"{mean_v:.9g}", f"{r['std_v']:.9g}",
                     f"{ratio:.6g}", f"{r['pred_v']:.9g}"]
            writer.writerow(line)
        f.write(f"# step7_json,{IN_STEP7}\n")
        f.write(f"# pred_method,{method}\n")
        f.write(f"# sweep,min={SWEEP_MIN},max={SWEEP_MAX},points={SWEEP_POINTS},x_eq_w\n")
    return str(out)


def load_csv(path):
    """Reload a step-8-simple CSV -> ``(pairs, blocks)``.

    ``blocks`` is a list of ``(driven, levels, dark, mean, std)`` in file
    order; the driven set of each block comes from the ``pairs`` column, so the
    file is self-describing and PAIRS may have changed since it was collected.

    ``std`` is read from ``voltage_std_v``, present in every generation of this
    CSV; the retired ``voltage_sem_v``/``sem_ratio`` columns of older files are
    ignored, so they still load.
    """
    with open(Path(path), newline="", encoding="utf-8") as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no data rows (not a step-8-simple CSV)")
    if "pairs" not in rows[0]:
        raise ValueError(
            f"{path} has no 'pairs' column, so it is not a step-8-simple CSV "
            f"(a calib_step8_test.py grid CSV carries x_<k>/w_<k> only -- "
            f"analyze that one with the grid script)"
        )
    grouped: dict[str, list[dict]] = {}
    for r in rows:                              # dicts keep insertion order
        grouped.setdefault(r["pairs"], []).append(r)
    blocks = []
    seen: list[int] = []
    for tag, group in grouped.items():
        driven = tuple(int(t) for t in tag.split("+"))
        seen += [k for k in driven if k not in seen]
        blocks.append((
            driven,
            np.array([float(r["v"]) for r in group]),
            np.array([float(r["dark_v"]) for r in group]),
            np.array([float(r["voltage_mean_v"]) for r in group]),
            np.array([float(r.get("voltage_std_v", "nan") or "nan") for r in group]),
        ))
    return sorted(seen), blocks


# ======================================================================
# comparison  (no fit -- just meas vs pred, block by block)
# ======================================================================

def compare(blocks, models, phases, *, method: str, png_path) -> None:
    """Print ``meas`` vs ``pred`` for every block, then the summary; write the PNG."""
    summary, plot_rows = [], []

    for driven, v, dark, mean, std in blocks:
        y = mean - dark
        pred = predict_curve(models, phases, driven, v)
        resid = y - pred
        pull = resid / std

        kind = "One pair" if len(driven) == 1 else "Two pairs"
        print(f"\n=== {kind}: {_pairs_tag(driven)}  [{method}] "
              f"(x = w = v, dark-subtracted, mV) ===")
        print("      v     meas     pred     diff    pull")
        for i in range(y.size):
            print(f"  {v[i]:5.3f}  {y[i]*1e3:7.4f}  {pred[i]*1e3:7.4f}  "
                  f"{resid[i]*1e3:+7.4f}  {pull[i]:+6.2f}")
        rms = float(np.sqrt(np.mean(resid**2)))
        rms_std = float(np.sqrt(np.mean(std**2)))
        print(f"  rms(meas - pred) = {rms*1e3:.4f} mV   "
              f"max |diff| = {float(np.max(np.abs(resid)))*1e3:.4f} mV   "
              f"rms(std) = {rms_std*1e3:.4f} mV")
        summary.append((driven, rms, float(np.max(np.abs(pull))), rms_std))
        plot_rows.append((driven, v, y, std, pred, pull))

    print(f"\n=== Summary [{method}] ===")
    # rms(std) is the measurement noise the residual should be judged
    # against: rms(diff) at or below it means the forward model is as good as
    # the data.
    print("  block        rms (mV)   max |pull|   rms(std) (mV)")
    for driven, rms, max_pull, rms_std in summary:
        print(f"    {_pairs_tag(driven):<8s}   {rms*1e3:7.4f}     {max_pull:7.2f}    "
              f"{rms_std*1e3:11.4f}")

    make_plot(plot_rows, method=method, path=png_path)
    print(f"\n  Plot saved to {png_path}")


def make_plot(plot_rows, *, method: str, path) -> None:
    """Two panels: the measured points on their model curves, and the pulls."""
    import matplotlib

    matplotlib.use("Agg")  # headless: write a PNG rather than open a window
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (driven, v, y, std, pred, pull) in enumerate(plot_rows):
        col = colors[i % len(colors)]
        tag = _pairs_tag(driven)
        solo = len(driven) == 1
        ax1.plot(v, pred * 1e3, "-", color=col, lw=1.5, alpha=0.9,
                 label=f"{'pair' if solo else 'pairs'} {tag} pred")
        ax1.errorbar(v, y * 1e3, yerr=std * 1e3, fmt="o" if solo else "s", ms=4.5,
                     color=col, mfc="none" if solo else col, ecolor="lightgray",
                     elinewidth=0.8, capsize=1.5, ls="none", zorder=3)
        ax2.plot(v, pull, "o" if solo else "s", color=col, ms=5,
                 mfc="none" if solo else col, label=tag)

    ax1.set_xlabel("level  v  (x = w)")
    ax1.set_ylabel("Y, dark-subtracted  (mV)")
    ax1.set_title(f"Predicted (line) vs measured (points)  [{method}]\n"
                  "open circles = one pair, filled squares = two pairs")
    ax1.legend(fontsize=7, loc="upper left")

    ax2.axhspan(-1, 1, color="tab:blue", alpha=0.12, label=r"$\pm1\sigma$")
    ax2.axhline(0, color="gray", ls="--", lw=1)
    ax2.set_xlabel("level  v")
    ax2.set_ylabel("Pull = (meas - pred) / std")
    ax2.set_title("Pulls")
    ax2.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(path, dpi=150)


def compare_csv(path, *, methods: tuple[str, ...]) -> None:
    """Recompute the prediction(s) for an existing CSV and compare (no hardware)."""
    pairs, blocks = load_csv(path)
    n = sum(b[1].size for b in blocks)
    print(f"Loaded {n} rows in {len(blocks)} blocks (pairs {pairs}) from {path}")
    for method in methods:
        _, models, phases = load_inputs(method, pairs)
        png = OUT_DIR / f"{Path(path).stem}_compare_{method}.png"
        compare(blocks, models, phases, method=method, png_path=png)


# ======================================================================
# collect  (python calib_step8_simple.py  ->  drive SLM, record CSV, compare)
# ======================================================================

def measure(*, flip: bool = False, methods: tuple[str, ...]) -> None:
    """Drive the singles then the combos; record, then compare.

    An all-off dark is read once at the start (T_SINGLE_S window) and stored per
    row.  The live printout compares each read against the ``methods[0]``
    prediction; the CSV keeps only raw data (+ that prediction as ``pred_v``),
    so it can be re-compared offline against any stored step-7 spectrum.

    ``flip`` negates the raw mean/dark as read (inverted DAQ sign) so the CSV
    already holds the positive light signal -- re-analyze it WITHOUT --flip.
    """
    from slm_module.encoding import encode_to_pattern

    method = methods[0]
    layout, models, phases = load_inputs(method, PAIRS)
    blocks, levels = build_blocks(PAIRS, COMBOS)
    n_points = len(blocks) * levels.size
    n_solo = sum(1 for b in blocks if len(b) == 1)
    print(f"Blocks: {n_solo} one-pair + {len(blocks) - n_solo} two-pair, "
          f"{levels.size} levels in [{SWEEP_MIN}, {SWEEP_MAX}] (x = w) "
          f"-> {n_points} points + 1 dark")
    if flip:
        print("Flip: negating voltage_mean_v + dark_v as read (inverted DAQ sign).")

    slm = connect_slm(SLM_DISPLAY_NO, USB_SLM_NO)
    daq = connect_daq(device=DAQ_DEVICE, channel=DAQ_CHANNEL,
                      t_both=T_BOTH_S, t_single=T_SINGLE_S)
    zeros = np.zeros(layout.n_channels)
    slm_width, slm_height = slm.get_slm_info()

    def _display(driven, v) -> None:
        x_vals = zeros.copy()
        w_vals = zeros.copy()
        for k in driven:
            x_vals[k] = w_vals[k] = float(v)
        slm.display_array(encode_to_pattern(x_vals, w_vals, layout,
                                            slm_width, slm_height))
        if SETTLE_S:
            time.sleep(SETTLE_S)

    rows = []
    try:
        _display((), 0.0)                                  # all off
        dark, _ = read_point(daq, single=True)
        if flip:
            dark = -dark
        print(f"[0/{n_points}] dark (all off, {T_SINGLE_S:.0f}s) = {dark*1000:.4f} mV")
        i = 0
        for b, driven in enumerate(blocks):
            print(f"\n--- Block {b}: "
                  f"{'pair' if len(driven) == 1 else 'pairs'} {_pairs_tag(driven)} ---")
            for v in levels:
                _display(driven, v)
                mean_v, std_v = read_point(daq)
                if flip:
                    mean_v = -mean_v
                pred = predict(models, phases, driven, v)
                i += 1
                rows.append({"block": b, "driven": driven, "v": float(v),
                             "dark_v": dark, "mean_v": mean_v, "std_v": std_v,
                             "pred_v": pred})
                print(f"[{i}/{n_points}] v={v:.3f} -> {(mean_v-dark)*1000:.4f} mV  "
                      f"(pred {pred*1000:.4f}, diff {(mean_v-dark-pred)*1000:+.4f})")
    finally:
        slm.close_slm()
        daq.disconnect()

    csv_path = OUT_DIR / f"calib_step8_simple_{time.strftime('%m%d_%H%M')}.csv"
    write_csv(csv_path, PAIRS, rows, method=method)
    print(f"\nCSV ({len(blocks)} blocks, {n_points} points) written to {csv_path}")
    print(f"Re-compare with:  python {Path(__file__).name} {csv_path}")
    compare_csv(csv_path, methods=methods)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    flip = "--flip" in argv     # inverted DAQ read (COLLECT only; CSVs are stored positive)
    methods = tuple(m for m in ("bounded", "fix") if f"--{m}" in argv) or (PHASE_METHOD,)
    positional = [a for a in argv if not a.startswith("-")]
    if positional:              # a CSV path -> offline comparison, no hardware
        if flip:
            print("Note: --flip only affects a COLLECT (CSVs already store the "
                  "positive signal); ignoring it.")
        compare_csv(positional[0], methods=methods)
    else:                       # no arg -> collect a fresh run (drives SLM/DAQ)
        measure(flip=flip, methods=methods)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
