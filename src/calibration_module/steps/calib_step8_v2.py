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
low-passed trace with ``calibration_module.sigma.STD_FLOOR_V`` in quadrature,
the one uncertainty these drafts carry.  The floor matters most here: a
two-pair block driven to a null reads near zero, where the trace spread alone
collapses and an ordinary 0.2 mV miss would print as a 3-sigma failure.

Not a pytest test (no mocks, needs real hardware) -- run it directly::

    python src/calibration_module/steps/calib_step8_v2.py            # COLLECT: drive, write raw
                                                       #   CSV, compare in place
    python src/calibration_module/steps/calib_step8_v2.py some.csv   # COMPARE an existing CSV
                                                       #   offline, no hardware

``--bounded`` / ``--fix`` pick WHICH stored step-7 spectrum to predict from when
a legacy v1 JSON carries both; with no flag the single stored fit is used, which
is all a step-7 **v2** JSON ever has (one method, ``fixed_comb_only``).  A v2 fit
is stored in the ``comb-slm`` convention -- the NEGATIVE of the ``slm+comb``
phase this forward model wants -- so :func:`load_inputs` reads each fit's
``convention`` field and flips the sign, and v1 and v2 JSONs both predict
correctly.

Sign.  There is no ``--flip`` here, and there must not be.  The transimpedance
amplifier puts out a NEGATIVE voltage for light, and that is undone exactly once
for the whole calibration, in ``draft_hw.read_point`` (``INVERT = True``), which
every step reads through -- so ``read_point`` already hands this script a
positive light signal.  The old ``--flip`` negated it a SECOND time, writing a
CSV of negative means against a positive dark; that is what happened on the 0903
run, and why it has a hand-made ``*_flipped.csv`` sibling undoing it.  A COLLECT
now checks the sign of its first driven read and aborts if it comes out
negative, and :func:`load_csv` refuses a CSV whose dark-subtracted signal is
negative, rather than silently comparing an inverted measurement against the
forward model.  If the rig's amplifier polarity ever really does change, flip
``draft_hw.INVERT`` -- the one place that owns it.
"""
from __future__ import annotations

import csv
import itertools
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
from calibration_module.sigma import STD_FLOOR_V, floor_std  # noqa: E402
from calibration_module.phase import (  # noqa: E402
    load_comb_phase_json,
    load_pair_models,
    phi_half,
)

# ---- Edit these to match your setup ----
CALIB_PATH = REPO_ROOT / "src/calib_data"          # data directory: inputs + outputs live here

# The ONE input: a combined step-7 result JSON (calib_step7_test.py REFIT output),
# which embeds step 3 (-> layout) and step 6 (-> eta + single-beam per pair).
IN_STEP7 = CALIB_PATH / "run_0903" / "calib_step7_result_0903_1704.json"   # ref pair 1, targets 3 + 5

# Pair numbering.  Pairs are labelled from PAIR_INDEX_BASE and that label is what
# the step-6/7 JSONs record; the Step-3 layout arrays are always 0-based, so
# _slot() converts.  Keep it in step with calib_step6_v2/calib_step7_v2, whose
# pair labels this script reads straight out of IN_STEP7.
PAIR_INDEX_BASE = 1         # pairs are numbered 1..N
PAIRS = [1, 3, 5]           # pairs to drive alone; each needs a step-6 model, and a
                            # step-7 phase unless it IS the reference pair (Phi = 0)
COMBOS = None               # two-pair blocks; None -> every unordered pair of PAIRS,
                            # e.g. [(1, 3), (1, 5), (3, 5)].  Set explicitly to trim.
PHASE_METHOD = None         # stored step-7 fit to predict from; None -> the single
                            # fit a v2 JSON carries.  --bounded/--fix override it
                            # for a legacy v1 JSON that stores several.

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

def _slot(pair: int) -> int:
    """Pair label -> its 0-based slot in the Step-3 layout / SLM drive arrays."""
    return pair - PAIR_INDEX_BASE


def _method_tag(method) -> str:
    """Print/filename label for the selected fit (``None`` -> the single stored one)."""
    return method or "auto"


def load_inputs(method, pairs):
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
    for k, e in entries.items():
        fit = e["fit"]
        # Step-7 v2 fits cos(dPhi_comb - dPhi_SLM); this model is
        # exp(i[phi_half(x) + phi_half(w) + Phi]), so its Phi is the negative.
        # A v1 JSON stores no "convention" and is already slm+comb.
        conv = str(fit.get("convention") or "slm+comb")
        sign = -1.0 if conv == "comb-slm" else 1.0
        phases[k] = sign * float(fit["dphi_comb_rad"])
    for k in pairs:
        if not (0 <= _slot(k) < layout.n_channels):
            raise ValueError(
                f"pair {k} out of range (layout has {layout.n_channels} pairs, "
                f"numbered from {PAIR_INDEX_BASE})"
            )
        if k not in models:
            raise ValueError(f"no step-6 model for pair {k} in {IN_STEP7}")
        if k not in phases:
            raise ValueError(
                f"no step-7 '{method}' phase for pair {k} in {IN_STEP7} "
                f"(reference is pair {ref_index})"
            )
    stored = sorted({str(e.get("method")) for e in entries.values()})
    print(f"Step 7 [{_method_tag(method)} -> {'/'.join(stored)}] vs ref {ref_index}:  "
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


def full_scale(models, driven, pred) -> float:
    """The block's full scale: the peak-to-peak swing its Y can span.

    For a two-pair block the phase rides a fringe on top of a fixed pedestal,

        Y = R_j^2 + R_k^2 + 2 R_j R_k cos(dPhi) + background,

    so Y_max = (R_j + R_k)^2 + bg, Y_min = (R_j - R_k)^2 + bg, and the swing is

        FS = 4 R_j R_k                    (at top drive, v = 1)

    -- the range the phase actually commands.  This is the reason to normalize
    here rather than by the measured or predicted value: FS does NOT collapse on
    a block that nulls.  1+5 reads near zero because its phases cancel, not
    because its amplitudes are small, so it gets the LARGEST FS of the three and
    its sub-mV residual reads as the fraction of a percent it really is.

    A one-pair block has no fringe and so no FS in that sense; it falls back to
    its own predicted span.  That is a ~4x smaller ruler than a two-pair block's,
    but its noise floor is divided by the same number, so a block is still only
    ever judged against its own cap -- never across the two kinds.
    """
    if len(driven) == 2:
        j, k = driven
        return 4.0 * float(models[j].amplitude(1.0, 1.0)
                           * models[k].amplitude(1.0, 1.0))
    return float(np.max(pred) - np.min(pred))


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
    _check_sign(path, blocks)
    return sorted(seen), blocks


def _check_sign(path, blocks) -> None:
    """Refuse a CSV whose dark-subtracted signal is negative (double-negated).

    ``draft_hw.read_point`` already inverts the TIA's negative-for-light output,
    so a well-collected CSV has ``mean - dark > 0`` everywhere.  Taken as a
    median over all points, so one noisy near-dark row cannot trip it.  The
    forward model predicts a positive intensity; compared against a negated
    measurement it yields a plausible-looking table of large residuals rather
    than an error, which is exactly the failure this catches.
    """
    y = np.concatenate([mean - dark for _, _, dark, mean, _ in blocks])
    med = float(np.median(y))
    if med >= 0.0:
        return
    raise ValueError(
        f"{path}: dark-subtracted signal is NEGATIVE (median {med * 1e3:.4f} "
        f"mV), so this CSV was collected with the sign inverted twice -- once "
        f"in draft_hw.read_point (INVERT = True, which is correct) and once "
        f"more by the retired --flip. Re-collect it, or negate voltage_mean_v "
        f"and dark_v to undo the second inversion."
    )


# ======================================================================
# comparison  (no fit -- just meas vs pred, block by block)
# ======================================================================

def compare(blocks, models, phases, *, method: str, png_path) -> None:
    """Print ``meas`` vs ``pred`` for every block, then the summary; write the PNG."""
    summary, plot_rows = [], []

    for driven, v, dark, mean, std in blocks:
        y = mean - dark
        # Same sigma the fits upstream weight by: the trace spread with the
        # systematic floor in quadrature.  Nothing is fitted here, but the
        # pulls and the noise floor below are read as sigmas, and a nulled
        # block reads near zero -- where the raw trace spread alone would make
        # a 0.2 mV miss look like a 3-sigma failure.
        std = floor_std(std)
        pred = predict_curve(models, phases, driven, v)
        resid = y - pred
        pull = resid / std

        kind = "One pair" if len(driven) == 1 else "Two pairs"
        print(f"\n=== {kind}: {_pairs_tag(driven)}  [{method}] "
              f"(x = w = v, dark-subtracted, mV) ===")
        # Normalized to the block's own full scale -- one constant per block, so
        # the percentages are just the mV errors on a common ruler and nothing
        # divides by a vanishing signal.  See full_scale().
        fs = full_scale(models, driven, pred)
        pct = 100.0 * np.abs(resid) / fs        # per-point, % of full scale
        print("      v     meas     pred     diff    pull    %FS")
        for i in range(y.size):
            print(f"  {v[i]:5.3f}  {y[i]*1e3:7.4f}  {pred[i]*1e3:7.4f}  "
                  f"{resid[i]*1e3:+7.4f}  {pull[i]:+6.2f}  {pct[i]:6.2f}")
        rms = float(np.sqrt(np.mean(resid**2)))
        rms_std = float(np.sqrt(np.mean(std**2)))
        nrmse = 100.0 * rms / fs                # rms(meas - pred) as % of full scale
        nrmse_std = 100.0 * rms_std / fs        # the noise floor it is judged against
        print(f"  rms(meas - pred) = {rms*1e3:.4f} mV   "
              f"NRMSE = {nrmse:.3f} % FS   "
              f"max |diff| = {float(np.max(np.abs(resid)))*1e3:.4f} mV   "
              f"FS = {fs*1e3:.2f} mV   "
              f"rms(std) = {rms_std*1e3:.4f} mV  ({nrmse_std:.3f} % FS)")
        if len(driven) == 2:
            # A pure phase error d shifts the fringe by dY = -2 R_j R_k sin(dPhi) d,
            # i.e. by (d/2) sin(dPhi) of FS -- so NRMSE converts straight into the
            # phase error that would account for it.  Rough (it charges the whole
            # residual to the phase) and blind near a fringe extremum, where
            # sin(dPhi) -> 0 and the block simply cannot see phase.
            j, k = driven
            dphi = float(phases[k] - phases[j])
            sin_dphi = abs(np.sin(dphi))
            if sin_dphi > 0.05:
                d_deg = np.degrees(2.0 * (rms / fs) / sin_dphi)
                print(f"  dPhi = {np.degrees(dphi):+.2f} deg   "
                      f"|sin dPhi| = {sin_dphi:.3f}   "
                      f"-> a phase error of ~{d_deg:.2f} deg would explain it")
        summary.append((driven, rms, nrmse, float(np.max(np.abs(pull))),
                        nrmse_std, fs))
        plot_rows.append((driven, v, y, std, pred, pull, nrmse, nrmse_std))

    print(f"\n=== Summary [{method}] ===")
    # rms(std) is the measurement noise the residual should be judged
    # against: rms(diff) at or below it means the forward model is as good as
    # the data.
    print("  block      FS (mV)   rms (mV)   NRMSE (%FS)   noise floor (%FS)   "
          "max |pull|")
    for driven, rms, nrmse, max_pull, nrmse_std, fs in summary:
        print(f"    {_pairs_tag(driven):<8s} {fs*1e3:8.2f}   {rms*1e3:7.4f}   "
              f"{nrmse:10.3f}   {nrmse_std:16.3f}   {max_pull:10.2f}")
    for n, kind in ((1, "one pair "), (2, "two pairs")):
        grp = [(row[2], row[4]) for row in summary if len(row[0]) == n]
        if grp:
            print(f"    {kind} mean NRMSE over {len(grp)} block(s) = "
                  f"{float(np.mean([a for a, _ in grp])):.3f} % FS   "
                  f"(floor {float(np.mean([b for _, b in grp])):.3f} %)")
    # The two kinds of block are on different rulers -- read a block against the
    # floor beside it, and compare across blocks only within one kind.
    print("  (FS = 4*R_j*R_k at top drive, the fringe swing the phase commands;")
    print("   a one-pair block has no fringe, so its FS is its own predicted span,")
    print("   a ~4x smaller ruler that its noise floor shares.)")
    print(f"  (sigma = hypot(trace std, {STD_FLOOR_V*1e3:.3f} mV systematic floor) "
          f"-- the pull denominator and the noise floor both use it.)")

    make_plot(plot_rows, method=method, path=png_path)
    print(f"\n  Plot saved to {png_path}")


def make_plot(plot_rows, *, method: str, path) -> None:
    """Three panels: measured points on their model curves, the pulls, the NRMSEs."""
    import matplotlib

    matplotlib.use("Agg")  # headless: write a PNG rather than open a window
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 4.8))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (driven, v, y, std, pred, pull, nrmse, nrmse_std) in enumerate(plot_rows):
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

    # --- panel 3: NRMSE = rms(meas - pred) / FS per block, singles first ---
    # The bar is the model error as a fraction of the swing that block can span
    # (full_scale(): the fringe 4*R_j*R_k for a two-pair block, the predicted span
    # for a one-pair one); the grey cap on it is rms(std) over the SAME FS.  A bar
    # at or under its cap is as good as the data.  Since FS is one constant per
    # block, the bars are the mV errors rescaled -- so the ordering is the
    # absolute-error ordering and a linear y axis reads honestly.  Singles and
    # doubles use different rulers: compare a bar to its own cap, not across.
    order = sorted(range(len(plot_rows)), key=lambda i: len(plot_rows[i][0]))
    xs = np.arange(len(order))
    for pos, i in enumerate(order):
        driven, _, _, _, _, _, nrmse, nrmse_std = plot_rows[i]
        solo = len(driven) == 1
        ax3.bar(pos, nrmse, width=0.62, color=colors[i % len(colors)],
                alpha=0.45 if solo else 0.85,
                edgecolor=colors[i % len(colors)], lw=1.4,
                hatch="//" if solo else "")
        ax3.plot([pos - 0.31, pos + 0.31], [nrmse_std] * 2, "-",
                 color="0.35", lw=1.6, zorder=4)
        ax3.text(pos, nrmse, f"{nrmse:.2f}%", ha="center", va="bottom", fontsize=8)
    # group means, drawn only across the blocks they average over
    for n, style in ((1, ":"), (2, "--")):
        grp = [pos for pos, i in enumerate(order) if len(plot_rows[i][0]) == n]
        if not grp:
            continue
        gm = float(np.mean([plot_rows[i][6] for i in order
                            if len(plot_rows[i][0]) == n]))
        ax3.plot([min(grp) - 0.45, max(grp) + 0.45], [gm] * 2, style,
                 color="crimson", lw=1.5,
                 label=f"{'one pair' if n == 1 else 'two pairs'} mean = {gm:.2f} %")
    ax3.plot([], [], "-", color="0.35", lw=1.6,
             label="rms(std) / FS, noise floor")
    ax3.set_xticks(xs)
    ax3.set_xticklabels([_pairs_tag(plot_rows[i][0]) for i in order])
    ax3.set_xlabel("driven block  (hatched = one pair, solid = two pairs)")
    ax3.set_ylabel("NRMSE = rms(meas - pred) / FS  (% of full scale)")
    ax3.set_title(f"Model error per block, % of full scale  [{method}]\n"
                  r"FS = $4R_jR_k$ (two pairs), own span (one pair)")
    ax3.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=150)


def compare_csv(path, *, methods: tuple[str, ...]) -> None:
    """Recompute the prediction(s) for an existing CSV and compare (no hardware)."""
    pairs, blocks = load_csv(path)
    n = sum(b[1].size for b in blocks)
    print(f"Loaded {n} rows in {len(blocks)} blocks (pairs {pairs}) from {path}")
    for method in methods:
        _, models, phases = load_inputs(method, pairs)
        tag = _method_tag(method)
        png = OUT_DIR / f"{Path(path).stem}_compare_{tag}.png"
        compare(blocks, models, phases, method=tag, png_path=png)


# ======================================================================
# collect  (python calib_step8_simple.py  ->  drive SLM, record CSV, compare)
# ======================================================================

def measure(*, methods: tuple[str, ...]) -> None:
    """Drive the singles then the combos; record, then compare.

    An all-off dark is read once at the start (T_SINGLE_S window) and stored per
    row.  The live printout compares each read against the ``methods[0]``
    prediction; the CSV keeps only raw data (+ that prediction as ``pred_v``),
    so it can be re-compared offline against any stored step-7 spectrum.

    No sign handling of its own: ``draft_hw.read_point`` already inverts the
    TIA's negative-for-light output, so what arrives here is the positive light
    signal.  The first driven read is checked against that and the run aborts if
    it comes in below the dark -- an hour of driving is not worth spending on
    data the forward model cannot be compared against.
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
    slm = connect_slm(SLM_DISPLAY_NO, USB_SLM_NO)
    daq = connect_daq(device=DAQ_DEVICE, channel=DAQ_CHANNEL,
                      t_both=T_BOTH_S, t_single=T_SINGLE_S)
    zeros = np.zeros(layout.n_channels)
    slm_width, slm_height = slm.get_slm_info()

    def _display(driven, v) -> None:
        x_vals = zeros.copy()
        w_vals = zeros.copy()
        for k in driven:
            x_vals[_slot(k)] = w_vals[_slot(k)] = float(v)
        slm.display_array(encode_to_pattern(x_vals, w_vals, layout,
                                            slm_width, slm_height))
        if SETTLE_S:
            time.sleep(SETTLE_S)

    rows = []
    try:
        _display((), 0.0)                                  # all off
        dark, _ = read_point(daq, single=True)
        print(f"[0/{n_points}] dark (all off, {T_SINGLE_S:.0f}s) = {dark*1000:.4f} mV")
        i = 0
        for b, driven in enumerate(blocks):
            print(f"\n--- Block {b}: "
                  f"{'pair' if len(driven) == 1 else 'pairs'} {_pairs_tag(driven)} ---")
            for v in levels:
                _display(driven, v)
                mean_v, std_v = read_point(daq)
                if i == 0 and mean_v - dark < 0.0:
                    # read_point already un-inverts the TIA, so a driven point
                    # has to read ABOVE the dark.  Bail on the first one rather
                    # than collect a run the forward model cannot be compared
                    # against.
                    raise RuntimeError(
                        f"first driven read is BELOW the dark "
                        f"({(mean_v - dark) * 1e3:+.4f} mV): the light signal "
                        f"is arriving negative. draft_hw.read_point already "
                        f"inverts the TIA (INVERT = True), so check the "
                        f"amplifier polarity and that the beam is on -- do not "
                        f"negate it a second time here."
                    )
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
    write_csv(csv_path, PAIRS, rows, method=_method_tag(method))
    print(f"\nCSV ({len(blocks)} blocks, {n_points} points) written to {csv_path}")
    print(f"Re-compare with:  python {Path(__file__).name} {csv_path}")
    compare_csv(csv_path, methods=methods)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--flip" in argv:
        # Retired, and refused rather than ignored: it double-negated a signal
        # read_point had already un-inverted, and a run collected with it looks
        # fine until the residuals come back huge.
        print("ERROR: --flip is gone. draft_hw.read_point already inverts the "
              "TIA's negative-for-light output (INVERT = True), so --flip "
              "negated it a second time and wrote a CSV of negative means. If "
              "the amplifier polarity really did change, flip draft_hw.INVERT.")
        return 2
    methods = tuple(m for m in ("bounded", "fix") if f"--{m}" in argv) or (PHASE_METHOD,)
    positional = [a for a in argv if not a.startswith("-")]
    if positional:              # a CSV path -> offline comparison, no hardware
        compare_csv(positional[0], methods=methods)
    else:                       # no arg -> collect a fresh run (drives SLM/DAQ)
        measure(methods=methods)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
