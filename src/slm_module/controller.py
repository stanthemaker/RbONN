from __future__ import annotations

import csv
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .driver import MODE_DVI, MODE_MEMORY, SLM_DVI_Driver as SLMDriver
from .generator import generate_center_scan, make_vertical_window, write_santec_csv


def validate_slm_csv(
    csv_path: str | Path,
    expected_width: int | None = None,
    expected_height: int | None = None,
    check_header_index: bool = True,
) -> tuple[int, int]:
    """
    Validate Santec SLM CSV format.

    Expected format:
        A1: y/x or similar label
        row 1, columns B...: x indices, 0, 1, 2, ...
        column A, rows 2...: y indices, 0, 1, 2, ...
        data area: integer grayscale values in [0, 1023]

    Returns:
        (width, height)
    """
    csv_path = Path(csv_path).resolve()

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file)
            rows = list(reader)
    except UnicodeDecodeError as exc:
        raise ValueError(f"CSV file is not valid UTF-8/UTF-8-BOM: {csv_path}") from exc

    if not rows:
        raise ValueError(f"CSV file is empty: {csv_path}")

    rows = [row[:-1] if row and row[-1] == "" else row for row in rows]
    header = rows[0]
    x_labels = header[1:]
    width = len(x_labels)
    height = len(rows) - 1

    if expected_width is not None and width != expected_width:
        raise ValueError(f"CSV width mismatch: got {width}, expected {expected_width}")

    if expected_height is not None and height != expected_height:
        raise ValueError(f"CSV height mismatch: got {height}, expected {expected_height}")

    expected_cols = width + 1
    for row_idx, row in enumerate(rows):
        if len(row) != expected_cols:
            raise ValueError(
                f"CSV row {row_idx + 1} has {len(row)} columns, "
                f"expected {expected_cols}"
            )

    if check_header_index:
        for x, label in enumerate(x_labels):
            try:
                value = int(label)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid x header at row 1, column {x + 2}: {label!r}"
                ) from exc

            if value != x:
                raise ValueError(
                    f"Wrong x index at row 1, column {x + 2}: "
                    f"got {value}, expected {x}"
                )

    for y, row in enumerate(rows[1:]):
        y_label = row[0]

        if check_header_index:
            try:
                y_value = int(y_label)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid y header at row {y + 2}, column 1: {y_label!r}"
                ) from exc

            if y_value != y:
                raise ValueError(
                    f"Wrong y index at row {y + 2}, column 1: "
                    f"got {y_value}, expected {y}"
                )

        for x, cell in enumerate(row[1:]):
            try:
                value = int(cell)
            except ValueError as exc:
                raise ValueError(f"Invalid grayscale at x={x}, y={y}: {cell!r}") from exc

            if not (0 <= value <= 1023):
                raise ValueError(
                    f"Grayscale out of range at x={x}, y={y}: "
                    f"{value}, expected 0~1023"
                )
    return width, height


class SLMController:
    def __init__(
        self,
        display_no: int = 1,
        driver: Any | None = None,
        rate120: bool = False,
    ):
        self.display_no = int(display_no)
        if driver is not None:
            self.driver = driver
        else:
            self.driver = SLMDriver(display_no, rate120=rate120)
        self._opened = False

    @property
    def is_open(self) -> bool:
        return self._opened

    def get_slm_info(self) -> tuple[int, int]:
        return self.driver.slm_info()

    def detect_displays(self) -> list[tuple[int, int, int, str]]:
        """Probe display numbers and return (no, width, height, name) tuples.

        The SLM reports a name starting with "LCOS-SLM" (Guide 2.4.2).
        """
        if not hasattr(self.driver, "search_displays"):
            raise RuntimeError("driver does not support display search")
        return self.driver.search_displays()

    def open_slm(self) -> None:
        self.driver.open_slm()
        self._opened = True

    def close_slm(self) -> None:
        try:
            self.driver.close_slm()
        finally:
            self._opened = False

    def _ensure_open(self) -> None:
        # Guide 1.3.2: SLM_Disp_Open must precede the display functions.
        if not self._opened:
            self.open_slm()

    def set_dvi_mode(self, slm_number: int = 1) -> None:
        self.driver.set_video_mode(MODE_DVI, slm_number=slm_number)

    def set_memory_mode(self, slm_number: int = 1) -> None:
        self.driver.set_video_mode(MODE_MEMORY, slm_number=slm_number)

    def display_grayscale(self, grayscale_value: int, interval: float = 0.2) -> None:
        grayscale_value = _validate_level(grayscale_value)
        self._ensure_open()
        self.driver.load_grayscale(grayscale_value, interval)

    def display_csv(self, csv_path: str | Path, interval: float = 0.2) -> None:
        slm_width, slm_height = self.get_slm_info()
        validate_slm_csv(csv_path, expected_width=slm_width, expected_height=slm_height)
        self._ensure_open()
        self.driver.load_csv(str(Path(csv_path).resolve()), interval)

    def display_mask_csv(
        self,
        data,
        csv_path: str | Path | None = None,
        interval: float = 0.2,
    ) -> Path:
        slm_width, slm_height = self.get_slm_info()
        if csv_path is None:
            csv_path = _temporary_csv_path("slm_mask_")
        csv_path = write_santec_csv(data, csv_path)
        validate_slm_csv(csv_path, expected_width=slm_width, expected_height=slm_height)
        self._ensure_open()
        self.driver.load_csv(str(csv_path), interval)
        return csv_path

    def display_vertical_window(
        self,
        x_start: int,
        level: int,
        window_px: int = 5,
        csv_path: str | Path | None = None,
        interval: float = 0.2,
    ) -> Path:
        slm_width, slm_height = self.get_slm_info()
        data = make_vertical_window(slm_width, slm_height, x_start, level, window_px)
        return self.display_mask_csv(data, csv_path=csv_path, interval=interval)

    def display_center_scan(
        self,
        level: int,
        *,
        window_px: int = 5,
        step_px: int = 5,
        start_x: int = 0,
        end_x: int | None = None,
        dwell_seconds: float = 0.2,
        output_dir: str | Path | None = None,
        stop_event: threading.Event | None = None,
        progress_callback: Callable[[int, Path], None] | None = None,
    ) -> list[Path]:
        slm_width, slm_height = self.get_slm_info()
        if output_dir is None:
            output_path = Path(tempfile.mkdtemp(prefix="santec_center_scan_"))
        else:
            output_path = Path(output_dir).resolve()
            output_path.mkdir(parents=True, exist_ok=True)

        self._ensure_open()
        paths: list[Path] = []
        for index, pattern in enumerate(
            generate_center_scan(
                slm_width,
                slm_height,
                level,
                window_px=window_px,
                step_px=step_px,
                start_x=start_x,
                end_x=end_x,
            )
        ):
            if stop_event is not None and stop_event.is_set():
                break

            csv_path = output_path / f"center_scan_x{pattern.x_start:04d}.csv"
            write_santec_csv(pattern.data, csv_path)
            validate_slm_csv(csv_path, expected_width=slm_width, expected_height=slm_height)
            self.driver.load_csv(str(csv_path), dwell_seconds)
            paths.append(csv_path)

            if progress_callback is not None:
                progress_callback(index, csv_path)
        return paths


def _validate_level(level: int) -> int:
    try:
        value = int(level)
    except (TypeError, ValueError) as exc:
        raise ValueError("grayscale level must be an integer") from exc
    if not 0 <= value <= 1023:
        raise ValueError("grayscale level must be in 0..1023")
    return value


def _temporary_csv_path(prefix: str) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix=prefix, delete=False
    )
    handle.close()
    return Path(handle.name)
