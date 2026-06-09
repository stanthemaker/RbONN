from __future__ import annotations

import json
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets

from ..calibration import CalibrationFit, fit_calibration, load_calibration_csv
from ..controller import SLMController
from ..generator import MAX_LEVEL, iter_center_scan_positions, make_vertical_window


class WorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)


class FunctionWorker(QtCore.QRunnable):
    def __init__(self, func: Callable[[], Any]):
        super().__init__()
        self.func = func
        self.signals = WorkerSignals()

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            self.signals.finished.emit(self.func())
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class MainWindow(QtWidgets.QMainWindow):
    scan_progress = QtCore.pyqtSignal(int, str)
    scan_total = QtCore.pyqtSignal(int, int, int)

    def __init__(
        self,
        controller_factory: Callable[..., SLMController] = SLMController,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.controller_factory = controller_factory
        self.controller: SLMController | None = None
        self.controller_display_no: int | None = None
        self.thread_pool = QtCore.QThreadPool.globalInstance()
        self._workers: set[FunctionWorker] = set()
        self.slm_size = (1920, 1200)
        self.calibration_fits: dict[float, CalibrationFit] = {}
        self.scan_stop_event: threading.Event | None = None

        self.setWindowTitle("Santec SLM Control")
        self.resize(1260, 820)
        self._build_ui()
        self._apply_style()
        self.scan_progress.connect(self._on_scan_progress)
        self.scan_total.connect(self._on_scan_total)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.nav = QtWidgets.QListWidget()
        self.nav.setObjectName("Navigation")
        self.nav.setFixedWidth(210)
        for label in ("SLM Control", "Calibration", "Center Scan", "TPA Encoding"):
            item = QtWidgets.QListWidgetItem(label)
            item.setSizeHint(QtCore.QSize(180, 52))
            self.nav.addItem(item)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self._build_control_page())
        self.stack.addWidget(self._build_calibration_page())
        self.stack.addWidget(self._build_scan_page())
        self.stack.addWidget(self._build_tpa_page())

        layout.addWidget(self.nav)
        layout.addWidget(self.stack, 1)
        self.setCentralWidget(central)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)

    def _build_control_page(self) -> QtWidgets.QWidget:
        page = self._page_shell("SLM Control")

        connection = self._panel("Connection")
        connection_layout = QtWidgets.QGridLayout(connection)
        self.display_no_spin = QtWidgets.QSpinBox()
        self.display_no_spin.setRange(1, 8)
        self.display_no_spin.setValue(1)
        self.display_no_spin.valueChanged.connect(self._reset_controller)
        self.rate120_check = QtWidgets.QCheckBox("120 Hz model")
        self.rate120_check.toggled.connect(self._reset_controller)
        self.conn_status_label = QtWidgets.QLabel("Status: closed")
        self.info_label = QtWidgets.QLabel("Size: unknown")

        detect_button = QtWidgets.QPushButton("Detect SLM")
        open_button = QtWidgets.QPushButton("Open")
        close_button = QtWidgets.QPushButton("Close")
        info_button = QtWidgets.QPushButton("Read Info")
        detect_button.clicked.connect(self._detect_slm)
        open_button.clicked.connect(self._open_slm)
        close_button.clicked.connect(self._close_slm)
        info_button.clicked.connect(
            lambda: self._run_task(
                "Read SLM info",
                lambda: self._controller().get_slm_info(),
                self._on_info_read,
            )
        )

        self.usb_slm_no_spin = QtWidgets.QSpinBox()
        self.usb_slm_no_spin.setRange(1, 8)
        self.usb_slm_no_spin.setValue(1)
        dvi_mode_button = QtWidgets.QPushButton("Switch to DVI Mode")
        dvi_mode_button.setToolTip(
            "Set the SLM video interface to DVI over USB "
            "(required before using the display functions)"
        )
        dvi_mode_button.clicked.connect(self._switch_to_dvi_mode)

        connection_layout.addWidget(QtWidgets.QLabel("Display"), 0, 0)
        connection_layout.addWidget(self.display_no_spin, 0, 1)
        connection_layout.addWidget(detect_button, 0, 2)
        connection_layout.addWidget(open_button, 0, 3)
        connection_layout.addWidget(close_button, 0, 4)
        connection_layout.addWidget(info_button, 0, 5)
        connection_layout.addWidget(QtWidgets.QLabel("USB SLM"), 1, 0)
        connection_layout.addWidget(self.usb_slm_no_spin, 1, 1)
        connection_layout.addWidget(dvi_mode_button, 1, 2)
        connection_layout.addWidget(self.rate120_check, 1, 3, 1, 2)
        connection_layout.addWidget(self.conn_status_label, 2, 0, 1, 3)
        connection_layout.addWidget(self.info_label, 2, 3, 1, 3)

        grayscale = self._panel("Grayscale")
        grayscale_layout = QtWidgets.QGridLayout(grayscale)
        self.gray_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gray_slider.setRange(0, 1023)
        self.gray_slider.setValue(0)
        self.gray_spin = QtWidgets.QSpinBox()
        self.gray_spin.setRange(0, 1023)
        self.gray_slider.valueChanged.connect(self.gray_spin.setValue)
        self.gray_spin.valueChanged.connect(self.gray_slider.setValue)
        gray_button = QtWidgets.QPushButton("Display Level")
        gray_button.clicked.connect(self._display_grayscale)

        grayscale_layout.addWidget(self.gray_slider, 0, 0)
        grayscale_layout.addWidget(self.gray_spin, 0, 1)
        grayscale_layout.addWidget(gray_button, 0, 2)

        csv_panel = self._panel("CSV Display")
        csv_layout = QtWidgets.QGridLayout(csv_panel)
        self.csv_path_edit = QtWidgets.QLineEdit()
        csv_browse = QtWidgets.QPushButton("Browse")
        csv_display = QtWidgets.QPushButton("Display CSV")
        csv_browse.clicked.connect(self._browse_display_csv)
        csv_display.clicked.connect(self._display_csv)
        csv_layout.addWidget(self.csv_path_edit, 0, 0)
        csv_layout.addWidget(csv_browse, 0, 1)
        csv_layout.addWidget(csv_display, 0, 2)

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setObjectName("LogBox")

        page.layout().addWidget(connection)
        page.layout().addWidget(grayscale)
        page.layout().addWidget(csv_panel)
        page.layout().addWidget(self._panel_with_widget("Status", self.log_box), 1)
        return page

    def _build_calibration_page(self) -> QtWidgets.QWidget:
        page = self._page_shell("Calibration")

        controls = self._panel("Measurements")
        controls_layout = QtWidgets.QGridLayout(controls)
        self.calibration_path_edit = QtWidgets.QLineEdit()
        browse_button = QtWidgets.QPushButton("Browse")
        fit_button = QtWidgets.QPushButton("Run Fit")
        self.save_fit_button = QtWidgets.QPushButton("Save Result")
        self.save_fit_button.setEnabled(False)
        browse_button.clicked.connect(self._browse_calibration_csv)
        fit_button.clicked.connect(self._run_calibration_fit)
        self.save_fit_button.clicked.connect(self._save_calibration_result)

        self.wavelength_combo = QtWidgets.QComboBox()
        self.wavelength_combo.currentIndexChanged.connect(self._update_calibration_view)

        controls_layout.addWidget(self.calibration_path_edit, 0, 0)
        controls_layout.addWidget(browse_button, 0, 1)
        controls_layout.addWidget(fit_button, 0, 2)
        controls_layout.addWidget(self.save_fit_button, 0, 3)
        controls_layout.addWidget(QtWidgets.QLabel("Wavelength"), 1, 0)
        controls_layout.addWidget(self.wavelength_combo, 1, 1, 1, 3)

        self.fit_table = QtWidgets.QTableWidget(0, 2)
        self.fit_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.fit_table.horizontalHeader().setStretchLastSection(True)
        self.fit_table.verticalHeader().setVisible(False)

        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        plot_panel = self._panel("Fit Curve")
        plot_layout = QtWidgets.QVBoxLayout(plot_panel)
        plot_layout.addWidget(self.canvas)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self._panel_with_widget("Fit Parameters", self.fit_table))
        split.addWidget(plot_panel)
        split.setSizes([360, 720])

        page.layout().addWidget(controls)
        page.layout().addWidget(split, 1)
        return page

    def _build_scan_page(self) -> QtWidgets.QWidget:
        page = self._page_shell("Center Scan")

        controls = self._panel("Pattern")
        form = QtWidgets.QGridLayout(controls)
        self.scan_level_spin = self._spin(0, 1023, 512)
        self.window_px_spin = self._spin(1, 256, 5)
        self.step_px_spin = self._spin(1, 1024, 5)
        self.start_x_spin = self._spin(0, 8191, 0)
        self.end_x_spin = self._spin(0, 8191, 1919)
        self.dwell_spin = QtWidgets.QDoubleSpinBox()
        self.dwell_spin.setRange(0.01, 60.0)
        self.dwell_spin.setSingleStep(0.05)
        self.dwell_spin.setValue(0.2)
        self.dwell_spin.setSuffix(" s")

        fields = [
            ("Level", self.scan_level_spin),
            ("Window", self.window_px_spin),
            ("Step", self.step_px_spin),
            ("Start x", self.start_x_spin),
            ("End x", self.end_x_spin),
            ("Dwell", self.dwell_spin),
        ]
        for index, (label, widget) in enumerate(fields):
            row = index // 3
            col = (index % 3) * 2
            form.addWidget(QtWidgets.QLabel(label), row, col)
            form.addWidget(widget, row, col + 1)

        for widget in (
            self.scan_level_spin,
            self.window_px_spin,
            self.step_px_spin,
            self.start_x_spin,
            self.end_x_spin,
        ):
            widget.valueChanged.connect(self._update_scan_preview)

        output = self._panel("Output")
        output_layout = QtWidgets.QGridLayout(output)
        self.scan_output_edit = QtWidgets.QLineEdit()
        output_browse = QtWidgets.QPushButton("Browse")
        self.start_scan_button = QtWidgets.QPushButton("Start Scan")
        self.stop_scan_button = QtWidgets.QPushButton("Stop")
        self.stop_scan_button.setEnabled(False)
        output_browse.clicked.connect(self._browse_scan_output)
        self.start_scan_button.clicked.connect(self._start_center_scan)
        self.stop_scan_button.clicked.connect(self._stop_center_scan)
        output_layout.addWidget(self.scan_output_edit, 0, 0)
        output_layout.addWidget(output_browse, 0, 1)
        output_layout.addWidget(self.start_scan_button, 0, 2)
        output_layout.addWidget(self.stop_scan_button, 0, 3)

        self.scan_size_label = QtWidgets.QLabel("Using preview size 1920 x 1200")
        self.scan_progress_bar = QtWidgets.QProgressBar()
        self.scan_progress_bar.setValue(0)
        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setMinimumHeight(280)
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setObjectName("Preview")

        page.layout().addWidget(controls)
        page.layout().addWidget(output)
        page.layout().addWidget(self.scan_size_label)
        page.layout().addWidget(self.scan_progress_bar)
        page.layout().addWidget(self.preview_label, 1)
        self._update_scan_preview()
        return page

    def _build_tpa_page(self) -> QtWidgets.QWidget:
        page = self._page_shell("TPA Encoding")
        panel = self._panel("Encoding")
        panel.setEnabled(False)
        layout = QtWidgets.QGridLayout(panel)
        layout.addWidget(QtWidgets.QLabel("Strategy"), 0, 0)
        layout.addWidget(QtWidgets.QLineEdit("TPA Multiplication"), 0, 1)
        layout.addWidget(QtWidgets.QPushButton("Encode"), 1, 1)
        page.layout().addWidget(panel)
        page.layout().addStretch(1)
        return page

    def _page_shell(self, title: str) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(26, 24, 26, 24)
        layout.setSpacing(18)
        heading = QtWidgets.QLabel(title)
        heading.setObjectName("PageTitle")
        layout.addWidget(heading)
        return page

    def _panel(self, title: str) -> QtWidgets.QGroupBox:
        panel = QtWidgets.QGroupBox(title)
        panel.setObjectName("Panel")
        return panel

    def _panel_with_widget(self, title: str, widget: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        panel = self._panel(title)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.addWidget(widget)
        return panel

    def _spin(self, minimum: int, maximum: int, value: int) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _controller(self) -> SLMController:
        display_no = self.display_no_spin.value()
        rate120 = self.rate120_check.isChecked()
        if self.controller is None or self.controller_display_no != display_no:
            try:
                controller = self.controller_factory(display_no, rate120=rate120)
            except TypeError:
                controller = self.controller_factory(display_no)
            self.controller = controller
            self.controller_display_no = display_no
        return self.controller

    def _reset_controller(self) -> None:
        old_controller = self.controller
        self.controller = None
        self.controller_display_no = None
        self.conn_status_label.setText("Status: closed")
        if old_controller is not None and getattr(old_controller, "is_open", False):
            self._run_task("Close previous SLM", old_controller.close_slm)

    def _run_task(
        self,
        label: str,
        func: Callable[[], Any],
        on_success: Callable[[Any], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> FunctionWorker:
        self._log(f"{label} started")
        worker = FunctionWorker(func)
        self._workers.add(worker)

        def finish(result: Any) -> None:
            self._workers.discard(worker)
            self._finish_task(label, result, on_success)

        def fail(error: str) -> None:
            self._workers.discard(worker)
            self._fail_task(label, error, on_error)

        worker.signals.finished.connect(finish)
        worker.signals.error.connect(fail)
        self.thread_pool.start(worker)
        return worker

    def _finish_task(
        self,
        label: str,
        result: Any,
        on_success: Callable[[Any], None] | None,
    ) -> None:
        self._log(f"{label} complete")
        if on_success is not None:
            on_success(result)
        self._refresh_conn_status()

    def _refresh_conn_status(self) -> None:
        is_open = self.controller is not None and getattr(self.controller, "is_open", False)
        self.conn_status_label.setText("Status: open" if is_open else "Status: closed")

    def _fail_task(
        self,
        label: str,
        error: str,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self._log(f"{label} failed")
        self._log(error)
        if on_error is not None:
            on_error(error)
        QtWidgets.QMessageBox.critical(self, label, error)

    def _log(self, message: str) -> None:
        if hasattr(self, "log_box"):
            self.log_box.appendPlainText(message.rstrip())
        self.statusBar().showMessage(message.splitlines()[0], 6000)

    def _open_slm(self) -> None:
        self._run_task("Open SLM", lambda: self._controller().open_slm())

    def _close_slm(self) -> None:
        self._run_task("Close SLM", lambda: self._controller().close_slm())

    def _detect_slm(self) -> None:
        self._run_task(
            "Detect SLM",
            lambda: self._controller().detect_displays(),
            self._on_detect,
        )

    def _on_detect(self, displays: list[tuple[int, int, int, str]]) -> None:
        if not displays:
            self._log("No displays found")
            return
        slm_no = None
        for display_no, width, height, name in displays:
            self._log(f"Display {display_no}: {width} x {height} ({name})")
            if slm_no is None and name.startswith("LCOS-SLM"):
                slm_no = display_no
        if slm_no is None:
            self._log("No LCOS-SLM display found; check connection and mode")
            return
        self._log(f"LCOS-SLM found on display {slm_no}")
        self.display_no_spin.setValue(slm_no)

    def _switch_to_dvi_mode(self) -> None:
        slm_number = self.usb_slm_no_spin.value()
        self._run_task(
            "Switch to DVI mode",
            lambda: self._controller().set_dvi_mode(slm_number),
        )

    def _on_info_read(self, result: tuple[int, int]) -> None:
        width, height = result
        self.slm_size = (int(width), int(height))
        self.info_label.setText(f"Size: {width} x {height}")
        self.scan_size_label.setText(f"Using SLM size {width} x {height}")
        self.start_x_spin.setMaximum(width - 1)
        self.end_x_spin.setMaximum(width - 1)
        self.end_x_spin.setValue(width - 1)
        self._update_scan_preview()

    def _display_grayscale(self) -> None:
        value = self.gray_spin.value()
        self._run_task(
            "Display grayscale",
            lambda: self._controller().display_grayscale(value),
        )

    def _browse_display_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select SLM CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self.csv_path_edit.setText(path)

    def _display_csv(self) -> None:
        path = self.csv_path_edit.text().strip()
        if not path:
            self._log("Select a CSV file first")
            return
        self._run_task("Display CSV", lambda: self._controller().display_csv(path))

    def _browse_calibration_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Calibration CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self.calibration_path_edit.setText(path)

    def _run_calibration_fit(self) -> None:
        path = self.calibration_path_edit.text().strip()
        if not path:
            self._log("Select a calibration CSV first")
            return

        def fit_file() -> dict[float, CalibrationFit]:
            points = load_calibration_csv(path)
            return fit_calibration(points)

        self._run_task("Calibration fit", fit_file, self._on_calibration_fit)

    def _on_calibration_fit(self, fits: dict[float, CalibrationFit]) -> None:
        self.calibration_fits = fits
        self.wavelength_combo.blockSignals(True)
        self.wavelength_combo.clear()
        for wavelength in fits:
            self.wavelength_combo.addItem(f"{wavelength:g} nm", wavelength)
        self.wavelength_combo.blockSignals(False)
        self.save_fit_button.setEnabled(True)
        self._update_calibration_view()

    def _update_calibration_view(self) -> None:
        if not self.calibration_fits or self.wavelength_combo.count() == 0:
            return
        wavelength = float(self.wavelength_combo.currentData())
        fit = self.calibration_fits[wavelength]

        rows = [
            ("wavelength_nm", fit.wavelength_nm),
            ("I0", fit.i0),
            ("phase_slope", fit.phase_slope),
            ("phase_offset", fit.phase_offset),
            ("RMSE", fit.rmse),
            ("R2", fit.r_squared),
        ]
        self.fit_table.setRowCount(len(rows))
        for row, (name, value) in enumerate(rows):
            self.fit_table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            self.fit_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{value:.8g}"))
        self.fit_table.resizeColumnsToContents()

        self.figure.clear()
        axes = self.figure.add_subplot(111)
        axes.set_facecolor("#101820")
        axes.scatter(fit.levels, fit.intensities, color="#47b8e0", label="Measured", s=32)
        axes.plot(fit.levels, fit.fitted_intensities, color="#f5c542", label="Fit", linewidth=2)
        axes.set_xlabel("Level")
        axes.set_ylabel("Intensity")
        axes.grid(True, color="#2b3a42", linewidth=0.7)
        axes.legend()
        self.figure.patch.set_facecolor("#101820")
        axes.tick_params(colors="#d8dee9")
        axes.xaxis.label.set_color("#d8dee9")
        axes.yaxis.label.set_color("#d8dee9")
        for spine in axes.spines.values():
            spine.set_color("#41515c")
        self.canvas.draw_idle()

    def _save_calibration_result(self) -> None:
        if not self.calibration_fits:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Calibration Result", "calibration_fit.json", "JSON Files (*.json)"
        )
        if not path:
            return
        payload = {
            f"{wavelength:g}": fit.to_dict()
            for wavelength, fit in self.calibration_fits.items()
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
        self._log(f"Saved calibration result: {path}")

    def _browse_scan_output(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.scan_output_edit.setText(path)

    def _start_center_scan(self) -> None:
        level = self.scan_level_spin.value()
        window_px = self.window_px_spin.value()
        step_px = self.step_px_spin.value()
        start_x = self.start_x_spin.value()
        end_x = self.end_x_spin.value()
        dwell_seconds = self.dwell_spin.value()
        output_dir = self.scan_output_edit.text().strip() or None

        self.scan_progress_bar.setValue(0)
        self.scan_stop_event = threading.Event()
        self.start_scan_button.setEnabled(False)
        self.stop_scan_button.setEnabled(True)

        def run_scan() -> list[Path]:
            controller = self._controller()
            width, height = controller.get_slm_info()
            clamped_start = min(start_x, width - 1)
            clamped_end = min(end_x, width - 1)
            positions = list(
                iter_center_scan_positions(
                    width,
                    window_px=window_px,
                    step_px=step_px,
                    start_x=clamped_start,
                    end_x=clamped_end,
                )
            )
            self.scan_total.emit(len(positions), width, height)
            return controller.display_center_scan(
                level,
                window_px=window_px,
                step_px=step_px,
                start_x=clamped_start,
                end_x=clamped_end,
                dwell_seconds=dwell_seconds,
                output_dir=output_dir,
                stop_event=self.scan_stop_event,
                progress_callback=lambda index, path: self.scan_progress.emit(
                    index + 1, str(path)
                ),
            )

        self._run_task("Center scan", run_scan, self._on_scan_finished, self._on_scan_error)

    def _stop_center_scan(self) -> None:
        if self.scan_stop_event is not None:
            self.scan_stop_event.set()
            self._log("Center scan stop requested")

    def _on_scan_total(self, total: int, width: int, height: int) -> None:
        self.scan_progress_bar.setMaximum(max(total, 1))
        self.slm_size = (width, height)
        self.scan_size_label.setText(f"Using SLM size {width} x {height}")

    def _on_scan_progress(self, count: int, path: str) -> None:
        self.scan_progress_bar.setValue(count)
        self._log(f"Displayed {Path(path).name}")

    def _on_scan_finished(self, paths: list[Path]) -> None:
        self.start_scan_button.setEnabled(True)
        self.stop_scan_button.setEnabled(False)
        self.scan_stop_event = None
        self._log(f"Center scan frames displayed: {len(paths)}")

    def _on_scan_error(self, _error: str) -> None:
        self.start_scan_button.setEnabled(True)
        self.stop_scan_button.setEnabled(False)
        self.scan_stop_event = None

    def _update_scan_preview(self) -> None:
        width, height = self.slm_size
        try:
            data = make_vertical_window(
                width,
                height,
                min(self.start_x_spin.value(), width - 1),
                self.scan_level_spin.value(),
                self.window_px_spin.value(),
            )
        except ValueError as exc:
            self.preview_label.setText(str(exc))
            return

        # render the real grayscale levels (0..1023) as display brightness
        preview = (data.astype(np.float32) / MAX_LEVEL * 217.0 + 18.0).astype(np.uint8)
        image = QtGui.QImage(
            preview.data,
            preview.shape[1],
            preview.shape[0],
            preview.shape[1],
            QtGui.QImage.Format_Grayscale8,
        ).copy()
        pixmap = QtGui.QPixmap.fromImage(image).scaled(
            self.preview_label.size().expandedTo(QtCore.QSize(760, 280)),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(pixmap)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if hasattr(self, "preview_label"):
            self._update_scan_preview()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.scan_stop_event is not None:
            self.scan_stop_event.set()
        self.thread_pool.waitForDone(3000)
        if self.controller is not None and getattr(self.controller, "is_open", False):
            try:
                self.controller.close_slm()
            except Exception:
                pass
        super().closeEvent(event)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #0f1419;
                color: #d8dee9;
                font-family: Segoe UI, Arial, sans-serif;
                font-size: 10.5pt;
            }
            #Navigation {
                background: #141b22;
                border: none;
                padding: 12px;
            }
            #Navigation::item {
                border-radius: 6px;
                padding: 14px 12px;
                margin: 3px 0;
            }
            #Navigation::item:selected {
                background: #1f6f78;
                color: white;
            }
            #PageTitle {
                font-size: 24pt;
                font-weight: 650;
                color: #f4f7f8;
            }
            QGroupBox#Panel {
                border: 1px solid #2c3a43;
                border-radius: 8px;
                margin-top: 16px;
                padding: 18px 12px 12px 12px;
                background: #111a20;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #9fc9cf;
                font-weight: 600;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTableWidget {
                background: #0b1116;
                border: 1px solid #344650;
                border-radius: 6px;
                padding: 7px;
                selection-background-color: #1f6f78;
            }
            QPushButton {
                background: #236d77;
                border: 1px solid #2f8995;
                border-radius: 6px;
                color: white;
                padding: 8px 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #2b7f8a;
            }
            QPushButton:disabled {
                background: #26313a;
                border-color: #2d3942;
                color: #6d7b84;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #273640;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #f5c542;
                width: 18px;
                margin: -7px 0;
                border-radius: 9px;
            }
            QProgressBar {
                background: #0b1116;
                border: 1px solid #344650;
                border-radius: 6px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background: #f5c542;
                border-radius: 5px;
            }
            #Preview {
                border: 1px solid #2c3a43;
                border-radius: 8px;
                background: #0b1116;
            }
            #LogBox {
                font-family: Consolas, monospace;
                font-size: 9.5pt;
            }
            """
        )


def main(argv: list[str] | None = None) -> int:
    app = QtWidgets.QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Santec SLM Control")
    window = MainWindow()
    window.show()
    return app.exec_()
