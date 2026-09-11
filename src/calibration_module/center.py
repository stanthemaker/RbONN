"""TPA centre calibration by scanning the layout centre wavelength.

The TPA pair-grid sweep in :mod:`slm_module.tpa_pair` assumes the encoder
layout is already centred on the true two-photon resonance. This module adds a
lighter-weight 1-D scan: rebuild the symmetric x/w layout at a list of centre
wavelengths, turn on one pair at a fixed drive level, read the fluorescence
brightness from the active monitor, then fit the resulting peak with a weighted
quadratic.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class TPACenterFit:
    center_wl_nm: float
    center_wl_err_nm: float
    peak_signal_v: float
    peak_signal_err_v: float
    coeffs: tuple[float, float, float]
    coeff_errs: tuple[float, float, float]
    chi2_red: float
    dof: int
    birge: float
    best_sample_center_wl_nm: float
    best_sample_signal_v: float
    valid: bool
    message: str
    center_wl: np.ndarray = field(repr=False)
    signal_v: np.ndarray = field(repr=False)
    sem_v: np.ndarray = field(repr=False)
    signal_pred_v: np.ndarray = field(repr=False)
    residuals_v: np.ndarray = field(repr=False)


@dataclass
class TPACenterResult:
    center_wl_nm: np.ndarray = field(repr=False)
    center_x_px: np.ndarray = field(repr=False)
    trial: np.ndarray = field(repr=False)
    signal_v: np.ndarray = field(repr=False)
    signal_std_v: np.ndarray = field(repr=False)
    background_v: np.ndarray = field(repr=False)
    background_std_v: np.ndarray = field(repr=False)
    net_signal_v: np.ndarray = field(repr=False)
    fit: TPACenterFit | None = None
    pair_index: int = 0
    drive_level: float = 1.0
    n_trials: int = 1
    repeats: int = 1
    subtract_background: bool = False


def average_trace_points(
    center_wl_nm: np.ndarray,
    signal_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average repeated readings at each scanned centre wavelength."""
    grouped: dict[float, list[float]] = defaultdict(list)
    for wl, signal in zip(np.asarray(center_wl_nm, dtype=float), np.asarray(signal_v, dtype=float)):
        grouped[float(wl)].append(float(signal))

    wl_out: list[float] = []
    mean_out: list[float] = []
    sem_out: list[float] = []
    for wl in sorted(grouped):
        arr = np.asarray(grouped[wl], dtype=float)
        wl_out.append(wl)
        mean_out.append(float(arr.mean()))
        sem_out.append(
            float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else float("nan")
        )

    wl_arr = np.asarray(wl_out, dtype=float)
    mean_arr = np.asarray(mean_out, dtype=float)
    sem_arr = np.asarray(sem_out, dtype=float)
    finite = sem_arr[np.isfinite(sem_arr) & (sem_arr > 0.0)]
    if finite.size:
        floor = float(np.median(finite))
    else:
        floor = float(np.std(mean_arr, ddof=1)) if mean_arr.size > 1 else abs(float(mean_arr[0]))
        floor = max(floor, 1e-12)
    sem_arr = np.where(np.isfinite(sem_arr) & (sem_arr > 0.0), sem_arr, floor)
    return wl_arr, mean_arr, sem_arr


def fit_center_trace(
    center_wl_nm: np.ndarray,
    signal_v: np.ndarray,
    sem_v: np.ndarray,
) -> TPACenterFit:
    """Weighted quadratic fit of brightness vs centre wavelength."""
    wl = np.asarray(center_wl_nm, dtype=float)
    signal = np.asarray(signal_v, dtype=float)
    sem = np.asarray(sem_v, dtype=float)
    if wl.ndim != 1 or signal.ndim != 1 or sem.ndim != 1:
        raise ValueError("centre-trace arrays must be 1-D")
    if wl.size != signal.size or wl.size != sem.size:
        raise ValueError("centre-trace arrays must have matching lengths")
    if wl.size < 3:
        raise ValueError("need at least three centre points for a quadratic fit")

    A = np.column_stack([wl**2, wl, np.ones_like(wl)])
    Aw = A / sem[:, None]
    coeffs, *_ = np.linalg.lstsq(Aw, signal / sem, rcond=None)
    cov = np.linalg.pinv(Aw.T @ Aw)

    pred = A @ coeffs
    residuals = signal - pred
    dof = max(len(signal) - 3, 1)
    chi2_red = float(np.sum((residuals / sem) ** 2) / dof)
    birge = max(1.0, float(np.sqrt(chi2_red)))
    cov_scaled = cov * (birge**2)
    coeff_errs = tuple(float(v) for v in np.sqrt(np.diag(cov_scaled)))

    a, b, c = (float(v) for v in coeffs)
    best_idx = int(np.nanargmax(signal))
    best_sample_center = float(wl[best_idx])
    best_sample_signal = float(signal[best_idx])

    center = float("nan")
    center_err = float("nan")
    peak = float("nan")
    peak_err = float("nan")
    valid = False
    message = "quadratic fit is invalid"
    if np.isfinite(a) and np.isfinite(b) and np.isfinite(c) and abs(a) > 0.0:
        center = float(-b / (2.0 * a))
        peak = float(c - (b * b) / (4.0 * a))
        grad_center = np.array([b / (2.0 * a * a), -1.0 / (2.0 * a), 0.0], dtype=float)
        grad_peak = np.array([b * b / (4.0 * a * a), -b / (2.0 * a), 1.0], dtype=float)
        center_var = float(grad_center @ cov_scaled @ grad_center)
        peak_var = float(grad_peak @ cov_scaled @ grad_peak)
        center_err = float(np.sqrt(center_var)) if center_var >= 0.0 else float("nan")
        peak_err = float(np.sqrt(peak_var)) if peak_var >= 0.0 else float("nan")
        if a >= 0.0:
            message = "fit is convex; no local maximum in the scanned window"
        elif center < float(np.min(wl)) or center > float(np.max(wl)):
            message = "fit peak lies outside the scanned wavelength range"
        else:
            valid = True
            message = "ok"

    return TPACenterFit(
        center_wl_nm=center,
        center_wl_err_nm=center_err,
        peak_signal_v=peak,
        peak_signal_err_v=peak_err,
        coeffs=(a, b, c),
        coeff_errs=coeff_errs,
        chi2_red=chi2_red,
        dof=dof,
        birge=birge,
        best_sample_center_wl_nm=best_sample_center,
        best_sample_signal_v=best_sample_signal,
        valid=valid,
        message=message,
        center_wl=wl,
        signal_v=signal,
        sem_v=sem,
        signal_pred_v=pred,
        residuals_v=residuals,
    )


_SCHEMA = "tpa_center_result_v1"


def save_tpa_center_json(result: TPACenterResult, path: str | Path) -> str:
    """Persist a centre scan (raw rows + fit + scan config) as JSON."""
    fit = result.fit
    fit_payload = None
    if fit is not None:
        fit_payload = {
            "center_wl_nm": fit.center_wl_nm,
            "center_wl_err_nm": fit.center_wl_err_nm,
            "peak_signal_v": fit.peak_signal_v,
            "peak_signal_err_v": fit.peak_signal_err_v,
            "coeffs": list(fit.coeffs),
            "coeff_errs": list(fit.coeff_errs),
            "chi2_red": fit.chi2_red,
            "dof": fit.dof,
            "birge": fit.birge,
            "best_sample_center_wl_nm": fit.best_sample_center_wl_nm,
            "best_sample_signal_v": fit.best_sample_signal_v,
            "valid": fit.valid,
            "message": fit.message,
            "center_wl": fit.center_wl.tolist(),
            "signal_v": fit.signal_v.tolist(),
            "sem_v": fit.sem_v.tolist(),
            "signal_pred_v": fit.signal_pred_v.tolist(),
            "residuals_v": fit.residuals_v.tolist(),
        }
    payload = {
        "schema": _SCHEMA,
        "pair_index": result.pair_index,
        "drive_level": result.drive_level,
        "n_trials": result.n_trials,
        "repeats": result.repeats,
        "subtract_background": result.subtract_background,
        "center_wl_nm": result.center_wl_nm.tolist(),
        "center_x_px": result.center_x_px.tolist(),
        "trial": result.trial.tolist(),
        "signal_v": result.signal_v.tolist(),
        "signal_std_v": result.signal_std_v.tolist(),
        "background_v": result.background_v.tolist(),
        "background_std_v": result.background_std_v.tolist(),
        "net_signal_v": result.net_signal_v.tolist(),
        "fit": fit_payload,
    }
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(out)


def load_tpa_center_json(path: str | Path) -> TPACenterResult:
    """Rebuild a :class:`TPACenterResult` saved by :func:`save_tpa_center_json`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != _SCHEMA:
        raise ValueError(
            f"{path}: expected schema {_SCHEMA!r}, got {payload.get('schema')!r}"
        )
    fit = None
    fp = payload.get("fit")
    if fp is not None:
        fit = TPACenterFit(
            center_wl_nm=float(fp["center_wl_nm"]),
            center_wl_err_nm=float(fp["center_wl_err_nm"]),
            peak_signal_v=float(fp["peak_signal_v"]),
            peak_signal_err_v=float(fp["peak_signal_err_v"]),
            coeffs=tuple(float(v) for v in fp["coeffs"]),
            coeff_errs=tuple(float(v) for v in fp["coeff_errs"]),
            chi2_red=float(fp["chi2_red"]),
            dof=int(fp["dof"]),
            birge=float(fp["birge"]),
            best_sample_center_wl_nm=float(fp["best_sample_center_wl_nm"]),
            best_sample_signal_v=float(fp["best_sample_signal_v"]),
            valid=bool(fp["valid"]),
            message=str(fp["message"]),
            center_wl=np.asarray(fp["center_wl"], dtype=float),
            signal_v=np.asarray(fp["signal_v"], dtype=float),
            sem_v=np.asarray(fp["sem_v"], dtype=float),
            signal_pred_v=np.asarray(fp["signal_pred_v"], dtype=float),
            residuals_v=np.asarray(fp["residuals_v"], dtype=float),
        )
    return TPACenterResult(
        center_wl_nm=np.asarray(payload["center_wl_nm"], dtype=float),
        center_x_px=np.asarray(payload["center_x_px"], dtype=float),
        trial=np.asarray(payload["trial"], dtype=int),
        signal_v=np.asarray(payload["signal_v"], dtype=float),
        signal_std_v=np.asarray(payload["signal_std_v"], dtype=float),
        background_v=np.asarray(payload["background_v"], dtype=float),
        background_std_v=np.asarray(payload["background_std_v"], dtype=float),
        net_signal_v=np.asarray(payload["net_signal_v"], dtype=float),
        fit=fit,
        pair_index=int(payload["pair_index"]),
        drive_level=float(payload["drive_level"]),
        n_trials=int(payload["n_trials"]),
        repeats=int(payload["repeats"]),
        subtract_background=bool(payload["subtract_background"]),
    )


__all__ = [
    "TPACenterFit",
    "TPACenterResult",
    "average_trace_points",
    "fit_center_trace",
    "load_tpa_center_json",
    "save_tpa_center_json",
]
