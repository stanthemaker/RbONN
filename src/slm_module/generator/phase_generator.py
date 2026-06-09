from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


MIN_LEVEL = 0
MAX_LEVEL = 1023


@dataclass(frozen=True)
class PhasePattern:
    x_start: int
    x_end: int
    data: np.ndarray


def make_vertical_window(
    width: int,
    height: int,
    x_start: int,
    level: int,
    window_px: int = 5,
) -> np.ndarray:
    width = _positive_int(width, "width")
    height = _positive_int(height, "height")
    x_start = _bounded_int(x_start, "x_start", 0, width - 1)
    window_px = _positive_int(window_px, "window_px")
    level = _bounded_int(level, "level", MIN_LEVEL, MAX_LEVEL)

    data = np.zeros((height, width), dtype=np.uint16)
    x_end = min(width, x_start + window_px)
    data[:, x_start:x_end] = level
    return data


def iter_center_scan_positions(
    width: int,
    *,
    window_px: int = 5,
    step_px: int = 5,
    start_x: int = 0,
    end_x: int | None = None,
) -> Iterator[int]:
    width = _positive_int(width, "width")
    window_px = _positive_int(window_px, "window_px")
    step_px = _positive_int(step_px, "step_px")
    start_x = _bounded_int(start_x, "start_x", 0, width - 1)
    if end_x is None:
        end_x = width - 1
    end_x = _bounded_int(end_x, "end_x", 0, width - 1)
    if end_x < start_x:
        raise ValueError("end_x must be greater than or equal to start_x")

    position = start_x
    while position <= end_x:
        yield position
        position += step_px


def generate_center_scan(
    width: int,
    height: int,
    level: int,
    *,
    window_px: int = 5,
    step_px: int = 5,
    start_x: int = 0,
    end_x: int | None = None,
) -> Iterator[PhasePattern]:
    for x_start in iter_center_scan_positions(
        width,
        window_px=window_px,
        step_px=step_px,
        start_x=start_x,
        end_x=end_x,
    ):
        data = make_vertical_window(width, height, x_start, level, window_px)
        yield PhasePattern(x_start=x_start, x_end=min(width, x_start + window_px), data=data)


def write_santec_csv(data: np.ndarray, csv_path: str | Path) -> Path:
    data_uint16 = _validate_mask_array(data)
    path = Path(csv_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    height, width = data_uint16.shape
    # plain ASCII without BOM: the DLL's CSV reader may not skip a UTF-8 BOM
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["y/x", *range(width)])
        for y in range(height):
            writer.writerow([y, *data_uint16[y].tolist()])

    return path


def export_center_scan_sequence(
    output_dir: str | Path,
    width: int,
    height: int,
    level: int,
    *,
    window_px: int = 5,
    step_px: int = 5,
    start_x: int = 0,
    end_x: int | None = None,
    prefix: str = "center_scan",
) -> list[Path]:
    output_path = Path(output_dir).resolve()
    paths: list[Path] = []
    for pattern in generate_center_scan(
        width,
        height,
        level,
        window_px=window_px,
        step_px=step_px,
        start_x=start_x,
        end_x=end_x,
    ):
        csv_path = output_path / f"{prefix}_x{pattern.x_start:04d}.csv"
        paths.append(write_santec_csv(pattern.data, csv_path))
    return paths


def _validate_mask_array(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data)
    if array.ndim != 2:
        raise ValueError("SLM mask data must be a 2D array")
    if array.size == 0:
        raise ValueError("SLM mask data cannot be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("SLM mask data must be finite")
    if np.any(array < MIN_LEVEL) or np.any(array > MAX_LEVEL):
        raise ValueError("SLM mask data must be in 0..1023")
    rounded = np.rint(array)
    if not np.array_equal(array, rounded):
        raise ValueError("SLM mask data must contain integer levels")
    return rounded.astype(np.uint16, copy=False)


def _positive_int(value: int, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _bounded_int(value: int, name: str, minimum: int, maximum: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum or result > maximum:
        raise ValueError(f"{name} must be in {minimum}..{maximum}")
    return result
