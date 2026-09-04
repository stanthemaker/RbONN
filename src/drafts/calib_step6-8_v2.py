"""Draft: run calibration steps 6 -> 7 -> 8 back to back into ONE run folder.

    python src/drafts/calib_step6-8_v2.py                  # full 6 -> 7 -> 8 run
    python src/drafts/calib_step6-8_v2.py --dry-run        # resolve inputs, touch no hardware
    python src/drafts/calib_step6-8_v2.py --pairs 1,3,5    # override the default pair list
    python src/drafts/calib_step6-8_v2.py --pairs 1,3,5 --ref 1
    python src/drafts/calib_step6-8_v2.py --start-at 7 --run-dir src/calib_data/calib_run_0904_1530

Every run gets a fresh directory ``src/calib_data/calib_run_<MMDD_HHMM>/`` and
*all* of the three steps' outputs -- raw CSVs, result JSONs, PNGs -- land in it.
Nothing goes to the flat ``src/calib_data`` any more, so one run is one
self-contained folder that can be zipped, moved or thrown away whole.

Wiring
------
The step scripts are configured by module-level constants, not by CLI flags, so
this runner imports each one by path and rebinds those constants before calling
its ``main([])``.  Every constant below is read at *call* time inside the step,
so rebinding after import is enough:

===========  =============================  ==================
step         input rebound                  output dir rebound
===========  =============================  ==================
6 (v2)       ``IN_STEP3``  <- newest         ``CALIB_PATH``
             ``calib_step3b_*.json``
7 (v2)       ``IN_STEP6``  <- step 6's       ``OUT_DIR``
             ``calib_step6v2_result_*``
8 (v2)       ``IN_STEP7``  <- step 7's       ``OUT_DIR``
             ``calib_step7_result_*``
===========  =============================  ==================

Only step 6 reaches outside the run folder, for the step-3 calibration; steps 7
and 8 are chained to the file the previous step just wrote, found by globbing
the (initially empty) run folder for that step's result pattern.

Note that step 6's ``CALIB_PATH`` is BOTH its output directory and the directory
``IN_STEP3`` was built from at import; rebinding ``CALIB_PATH`` alone would
silently leave ``IN_STEP3`` pointing into the flat directory (it is already a
fully-resolved Path by then), which is why both are set explicitly.

``--flip`` is never passed to any step: the DAQ sign on this rig is positive.

Failure
-------
A step that returns non-zero or raises stops the sequence -- the run folder
keeps whatever it produced, ``sequence.json`` records how far it got, and the
exact ``--start-at`` command to resume is printed.  Console output is teed to
``run.log`` in the run folder, so the diagnostics the steps print survive.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STEPS_DIR = REPO_ROOT / "src/calibration_module/steps"
CALIB_PATH = REPO_ROOT / "src/calib_data"       # where run folders are created

# ---- Edit these to match your setup ----
RUN_PREFIX = "run"                   # run folder is <RUN_PREFIX>_<MMDD_HHMM>
STEP3_GLOB = "calib_step3b_*.json"   # step 6's input; newest by mtime wins
STEP3_SKIP_DIRS = ("archive",)       # subtrees the step-3 search ignores

# One pair list for all three steps (step 6 PAIR_INDICES, step 7 TGT_INDICES,
# step 8 PAIRS), so the three cannot disagree -- a pair measured in 6 but missing
# from 8 is a silent hole.  --pairs overrides it; None falls back to each step's
# own hard-coded list.
#
# REF_INDEX must be IN the pair list: step 7 fits every target against it and
# needs its step-6 eta, and step 8 predicts from phases defined relative to it.
# Leaving it in TGT_INDICES too is deliberate -- step 7 then measures the
# reference against itself, which is not fitted (fit_csv drops k == REF_INDEX)
# but gives a free reference-only baseline to check step 6's eta against.
PAIRS = [2, 4, 5]               # pairs calibrated end to end
REF_INDEX = 2                   # step 7's reference pair (Phi = 0); must be in PAIRS

# Per step: module stem, the glob that finds the result it hands downstream, and
# the constants to rebind.  ``result_glob = None`` marks the terminal step.
STEPS = {
    6: {"module": "calib_step6_v2", "out_attr": "CALIB_PATH",
        "in_attr": "IN_STEP3", "pairs_attr": "PAIR_INDICES",
        "result_glob": "calib_step6v2_result_*.json"},
    7: {"module": "calib_step7_v2", "out_attr": "OUT_DIR",
        "in_attr": "IN_STEP6", "pairs_attr": "TGT_INDICES",
        "result_glob": "calib_step7_result_*.json"},
    8: {"module": "calib_step8_v2", "out_attr": "OUT_DIR",
        "in_attr": "IN_STEP7", "pairs_attr": "PAIRS",
        "result_glob": None},
}


# ======================================================================
# plumbing
# ======================================================================

class _Tee:
    """Mirror a text stream into a file, so run.log holds the full console."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, s):
        self._stream.write(s)
        self._stream.flush()        # unbuffered: a crash must not eat the tail
        self._fh.write(s)
        self._fh.flush()
        return len(s)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def __getattr__(self, name):    # isatty, encoding, ... -> the real stream
        return getattr(object.__getattribute__(self, "_stream"), name)


def _load_step(stem: str):
    """Import a step script by path (they live outside any importable package)."""
    if stem in sys.modules:
        return sys.modules[stem]
    path = STEPS_DIR / f"{stem}.py"
    if not path.is_file():
        raise FileNotFoundError(f"step script not found: {path}")
    spec = importlib.util.spec_from_file_location(stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[stem] = mod         # register before exec, as import does
    spec.loader.exec_module(mod)
    return mod


def _newest(directory: Path, pattern: str) -> Path | None:
    """Newest file matching ``pattern`` directly in ``directory`` (not below)."""
    hits = [p for p in directory.glob(pattern) if p.is_file()]
    if not hits:
        return None
    return max(hits, key=lambda p: (p.stat().st_mtime, p.name))


def resolve_step3(explicit: str | None) -> Path:
    """The step-3b calibration step 6 starts from: the newest one in calib_data.

    Searched RECURSIVELY: calib_data holds both loose files and per-run folders
    (this script's own output, ``run_0903/`` and friends), and the newest step 3b
    normally sits inside the most recent of those rather than at the top level.
    ``STEP3_SKIP_DIRS`` keeps the search out of the archive.

    Ordered by mtime rather than by the MMDD_HHMM in the name -- the name has no
    year, so name order breaks across a new year while mtime does not.
    """
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--step3 file not found: {path}")
        return path
    hits = [p for p in CALIB_PATH.rglob(STEP3_GLOB)
            if p.is_file() and not any(d in p.parts for d in STEP3_SKIP_DIRS)]
    if not hits:
        raise FileNotFoundError(
            f"no {STEP3_GLOB} under {CALIB_PATH} -- run step 3b first, or pass "
            f"--step3 PATH"
        )
    return max(hits, key=lambda p: (p.stat().st_mtime, p.name))


def make_run_dir(explicit: str | None, *, create: bool = True) -> Path:
    """Create (or reuse, with --run-dir) the folder every output goes into.

    ``create=False`` only resolves the name -- a --dry-run must not litter
    calib_data with empty run folders.
    """
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path
    stamp = time.strftime("%m%d_%H%M")
    path = CALIB_PATH / f"{RUN_PREFIX}_{stamp}"
    suffix = 0
    while path.exists():            # two runs inside the same minute
        suffix += 1
        path = CALIB_PATH / f"{RUN_PREFIX}_{stamp}_{suffix}"
    if create:
        path.mkdir(parents=True)
    return path


def _rel(path: Path) -> str:
    """Repo-relative display path, falling back to absolute for off-tree files."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


# ======================================================================
# the sequence
# ======================================================================

def run_step(n: int, run_dir: Path, in_path: Path, *, pairs=None,
             ref_index=None, dry_run: bool = False) -> Path | None:
    """Point step ``n`` at this run folder, run it, return its result JSON.

    ``in_path`` is what the step reads (step 3 for step 6, the previous step's
    combined JSON for 7 and 8).  The result is found by globbing the run folder
    afterwards: the steps stamp their own filenames and return nothing.

    Under --dry-run ``in_path`` may be a plain string standing in for a file the
    previous step has not written yet; it is printed but never bound.
    """
    spec = STEPS[n]
    pending = isinstance(in_path, str)          # dry-run placeholder, not a path
    mod = _load_step(spec["module"])
    setattr(mod, spec["out_attr"], run_dir)     # all outputs -> the run folder
    if not pending:
        setattr(mod, spec["in_attr"], in_path)
    if pairs is not None:
        setattr(mod, spec["pairs_attr"], list(pairs))
    if ref_index is not None and hasattr(mod, "REF_INDEX"):
        mod.REF_INDEX = int(ref_index)

    head = (f"\n{'=' * 70}\n== STEP {n}  ({spec['module']})\n"
            f"==   in   : {in_path if pending else _rel(in_path)}\n"
            f"==   out  : {_rel(run_dir)}\n"
            f"==   pairs: {getattr(mod, spec['pairs_attr'])}")
    if hasattr(mod, "REF_INDEX"):
        head += f"   ref: {mod.REF_INDEX}"
    print(f"{head}\n{'=' * 70}", flush=True)
    if dry_run:
        print("   (dry run -- not executed)")
        return None

    t0 = time.time()
    rc = mod.main([])               # [] not None: the step must not see OUR argv
    if rc:
        raise RuntimeError(f"step {n} ({spec['module']}) returned {rc}")
    print(f"\n-- step {n} finished in {(time.time() - t0) / 60:.1f} min")

    if spec["result_glob"] is None:
        return None
    result = _newest(run_dir, spec["result_glob"])
    if result is None:
        raise RuntimeError(
            f"step {n} wrote no {spec['result_glob']} into {run_dir} -- the "
            f"sweep may have run while the fit failed, so there is nothing to "
            f"hand to the next step"
        )
    print(f"-- step {n} result: {_rel(result)}")
    return result


def _write_manifest(run_dir: Path, manifest: dict) -> None:
    """Rewrite sequence.json after each step, so a crash still leaves a record."""
    manifest["outputs"] = sorted(
        p.name for p in run_dir.iterdir()
        if p.is_file() and p.name not in ("sequence.json", "run.log"))
    (run_dir / "sequence.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")


def run_sequence(*, run_dir: Path, step3: Path, start_at: int = 6,
                 pairs=None, ref_index=None, dry_run: bool = False) -> int:
    order = [n for n in (6, 7, 8) if n >= start_at]
    manifest = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": _rel(run_dir),
        "steps_run": order,
        "pairs": list(pairs) if pairs else None,
        "ref_index": ref_index,
        "inputs": {"step3": _rel(step3)},
        "status": "running",
        "outputs": [],
    }

    # Resuming mid-sequence: the skipped steps' results have to be in the run
    # folder already, since that is the only place the chain looks for them.
    carried = step3
    for n in (6, 7):
        if n >= start_at:
            break
        got = _newest(run_dir, STEPS[n]["result_glob"])
        if got is None:
            print(f"ERROR: --start-at {start_at} needs step {n}'s "
                  f"{STEPS[n]['result_glob']} in {_rel(run_dir)}, found none")
            return 2
        carried = got
        manifest["inputs"][f"step{n}_result"] = _rel(got)
        print(f"Reusing step {n} result: {_rel(got)}")

    if not dry_run:
        _write_manifest(run_dir, manifest)

    for n in order:
        try:
            result = run_step(n, run_dir, carried, pairs=pairs,
                              ref_index=ref_index, dry_run=dry_run)
        except Exception as exc:
            print(f"\n!! STEP {n} FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            manifest["status"] = f"failed at step {n}: {type(exc).__name__}: {exc}"
            manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
            if not dry_run:
                _write_manifest(run_dir, manifest)
            print(f"\nRun folder kept: {_rel(run_dir)}")
            print(f"Resume with:  python {_rel(Path(__file__))} "
                  f"--start-at {n} --run-dir {_rel(run_dir)}")
            return 1
        if dry_run and STEPS[n]["result_glob"]:
            carried = f"<step {n} output: {STEPS[n]['result_glob']}>"
        if result is not None:
            carried = result
            manifest["inputs"][f"step{n}_result"] = _rel(result)
        if not dry_run:
            _write_manifest(run_dir, manifest)

    manifest["status"] = "ok"
    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if dry_run:
        print(f"\nDry run complete -- would write into {_rel(run_dir)}")
        return 0
    _write_manifest(run_dir, manifest)
    print(f"\n{'=' * 70}\nSequence complete -- everything in {_rel(run_dir)}")
    for name in manifest["outputs"]:
        print(f"   {name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run calibration steps 6 -> 7 -> 8 into one run folder.")
    ap.add_argument("--step3", metavar="PATH", default=None,
                    help="step-3b JSON for step 6 (default: newest in calib_data)")
    ap.add_argument("--run-dir", metavar="PATH", default=None,
                    help="reuse this folder instead of creating a new one "
                         "(required with --start-at 7/8)")
    ap.add_argument("--start-at", type=int, choices=(6, 7, 8), default=6,
                    help="skip earlier steps, reusing their results in --run-dir")
    ap.add_argument("--pairs", metavar="1,3,5", default=None,
                    help="pair list for all three steps (default: each step's own)")
    ap.add_argument("--ref", type=int, default=None,
                    help="step-7 reference pair (default: the step's own)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and print the wiring; touch no hardware")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    if args.start_at > 6 and not args.run_dir:
        ap.error("--start-at 7/8 needs --run-dir (the earlier steps' outputs)")

    pairs = PAIRS
    if args.pairs:
        pairs = [int(tok) for tok in args.pairs.replace(" ", "").split(",") if tok]
    ref_index = REF_INDEX if args.ref is None else args.ref
    # Caught here rather than an hour into step 7: the reference has to be one
    # of the pairs step 6 measured, or load_models() has no eta for it.
    if pairs and ref_index is not None and ref_index not in pairs:
        ap.error(f"--ref {ref_index} is not in --pairs {pairs}: step 7 fits every "
                 f"target against the reference and needs its step-6 eta")

    step3 = resolve_step3(args.step3)
    run_dir = make_run_dir(args.run_dir, create=not args.dry_run)

    if args.dry_run:
        print(f"Run folder : {_rel(run_dir)}")
        print(f"Step 3 in  : {_rel(step3)}")
        return run_sequence(run_dir=run_dir, step3=step3, start_at=args.start_at,
                            pairs=pairs, ref_index=ref_index, dry_run=True)

    log_path = run_dir / "run.log"
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"start-at {args.start_at} ====\n")
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(stdout, fh)
        sys.stderr = _Tee(stderr, fh)
        try:
            print(f"Run folder : {_rel(run_dir)}")
            print(f"Step 3 in  : {_rel(step3)}")
            print(f"Log        : {_rel(log_path)}")
            return run_sequence(run_dir=run_dir, step3=step3,
                                start_at=args.start_at, pairs=pairs,
                                ref_index=ref_index)
        finally:
            sys.stdout, sys.stderr = stdout, stderr


if __name__ == "__main__":
    sys.exit(main())
