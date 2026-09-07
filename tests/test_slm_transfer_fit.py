"""The fitted sin^2 transfer curve and the encoding built on it.

Covers :mod:`slm_module.calibration.transfer` and the ``method="fit"`` path
through :class:`slm_module.encoding.EncodingChannel`.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slm_module.calibration.calibration_new import (  # noqa: E402
    CalibrationResult,
    load_calibration_result,
    save_calibration_result,
)
from slm_module.calibration.transfer import (  # noqa: E402
    MAX_LEVEL,
    TransferFit,
    TransferFitError,
    fit_transfer_curve,
    fit_transfer_curves,
)
from slm_module.encoding import (  # noqa: E402
    EncodingChannel,
    build_channel_layout,
    build_single_anchor_layout,
    channel_layout_from_calibration,
)

# a realistic sweep: 35 levels over roughly the range Step 3b uses
LEVELS = np.arange(380, 891, 15, dtype=int)
OFF_TRUE, ON_TRUE = 390.0, 860.0


def curve(levels=LEVELS, *, floor=0.0, contrast=1.0,
          off_level=OFF_TRUE, on_level=ON_TRUE) -> np.ndarray:
    """Exact sin^2 curve with retardance 0 at ``off_level`` and pi at ``on_level``."""
    slope = math.pi / (on_level - off_level)
    theta = slope * (np.asarray(levels, dtype=float) - off_level)
    return floor + contrast * np.sin(theta / 2.0) ** 2


class FitTransferCurveTests(unittest.TestCase):
    def test_recovers_known_parameters(self) -> None:
        fit = fit_transfer_curve(LEVELS, curve(contrast=0.97))
        self.assertAlmostEqual(fit.contrast, 0.97, places=4)
        self.assertAlmostEqual(fit.floor, 0.0, places=4)
        self.assertAlmostEqual(fit.off_level_exact, OFF_TRUE, places=2)
        self.assertAlmostEqual(fit.on_level_exact, ON_TRUE, places=2)
        self.assertLess(fit.rms, 1e-5)
        self.assertEqual(fit.n_used, LEVELS.size)
        self.assertEqual((fit.level_min, fit.level_max), (380, 890))

    def test_floor_is_clamped_non_negative(self) -> None:
        """A normalized power cannot be negative, so the floor is bounded at 0.

        Real curves want a slightly negative floor -- the reference
        normalization over-subtracts by ~1% -- and letting them have it would
        break the v -> retardance identity the phase model relies on.
        """
        dipped = curve() - 0.02
        fit = fit_transfer_curve(LEVELS, dipped)
        self.assertGreaterEqual(fit.floor, 0.0)

    def test_slope_normalized_positive_and_offset_on_first_branch(self) -> None:
        """Whatever branch the optimizer lands on, the stored fit is canonical."""
        fit = fit_transfer_curve(LEVELS, curve())
        self.assertGreater(fit.phase_slope, 0.0)
        # retardance ~ 0 at the darkest swept level, pi at full scale
        self.assertAlmostEqual(float(fit.retardance(OFF_TRUE)), 0.0, places=3)
        self.assertAlmostEqual(float(fit.retardance(ON_TRUE)), math.pi, places=3)

    def test_rejects_curves_it_cannot_fit(self) -> None:
        with self.assertRaisesRegex(TransferFitError, "at least 5"):
            fit_transfer_curve(np.array([0, 1, 2]), np.array([0.0, 0.5, 1.0]))
        with self.assertRaisesRegex(TransferFitError, "flat"):
            fit_transfer_curve(LEVELS, np.zeros(LEVELS.size))
        with self.assertRaisesRegex(TransferFitError, "matching 1-D"):
            fit_transfer_curve(LEVELS, np.zeros((2, LEVELS.size)))

    def test_fit_transfer_curves_names_the_bad_row(self) -> None:
        rows = np.stack([curve(), np.zeros(LEVELS.size)])
        with self.assertRaisesRegex(TransferFitError, "channel row 1"):
            fit_transfer_curves(LEVELS, rows)

    def test_extrapolated_flags_a_peak_past_the_sweep(self) -> None:
        inside = fit_transfer_curve(LEVELS, curve())
        self.assertFalse(inside.extrapolated)
        # peak at 1000, well past the 890 the sweep reached
        beyond = fit_transfer_curve(LEVELS, curve(on_level=1000.0))
        self.assertTrue(beyond.extrapolated)
        self.assertFalse(beyond.clipped)          # still inside the panel's range

    def test_dict_round_trip(self) -> None:
        fit = fit_transfer_curve(LEVELS, curve(contrast=0.9))
        back = TransferFit.from_dict(json.loads(json.dumps(fit.to_dict())))
        self.assertEqual(back, fit)
        with self.assertRaises(TransferFitError):
            TransferFit.from_dict({"floor": 0.0})     # missing parameters


class SpikeRejectionTests(unittest.TestCase):
    """The failure this module exists to fix: argmax chasing one noisy sample.

    In the 0903 calibration one channel had a spike at level 770 -- 12% above
    its immediate neighbour, nine swept levels below its real peak.  ``argmax``
    took it for full scale, so ``v = 1`` delivered ~94% of the channel's actual
    maximum and every other ``v`` over-delivered relative to it.
    """

    def setUp(self) -> None:
        self.clean = curve()
        self.spiked = self.clean.copy()
        self.spike_idx = int(np.argmin(np.abs(LEVELS - 770)))
        self.spiked[self.spike_idx] *= 1.12

    def test_argmax_takes_the_spike_but_the_fit_does_not(self) -> None:
        self.assertEqual(int(LEVELS[int(np.argmax(self.spiked))]),
                         int(LEVELS[self.spike_idx]))
        fit = fit_transfer_curve(LEVELS, self.spiked)
        # the fit stays within a few levels of the true peak
        self.assertAlmostEqual(fit.on_level_exact, ON_TRUE, delta=8.0)

    def test_robust_loss_limits_the_spike_pull(self) -> None:
        robust = fit_transfer_curve(LEVELS, self.spiked, robust=True)
        plain = fit_transfer_curve(LEVELS, self.spiked, robust=False)
        self.assertLessEqual(
            abs(robust.on_level_exact - ON_TRUE),
            abs(plain.on_level_exact - ON_TRUE) + 1e-9,
        )

    def test_encoding_channel_full_scale_ignores_the_spike(self) -> None:
        ch = self._channel(self.spiked, method="fit")
        interp = self._channel(self.spiked, method="interp")
        self.assertEqual(interp.on_level, int(LEVELS[self.spike_idx]))   # 770
        self.assertGreater(ch.on_level, interp.on_level + 50)            # ~860

    @staticmethod
    def _channel(values, *, method):
        return EncodingChannel(
            index=0, side="x", x_center=100, x_start=93, x_end=108,
            wavelength_nm=780.0, levels=LEVELS.copy(),
            intensity_curve=np.asarray(values, dtype=float), method=method,
        )


class FitEncodingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ch = EncodingChannel(
            index=0, side="x", x_center=100, x_start=93, x_end=108,
            wavelength_nm=780.0, levels=LEVELS.copy(),
            intensity_curve=curve(), method="fit",
        )

    def test_endpoints_are_the_fitted_window(self) -> None:
        self.assertEqual(self.ch.level_for(0.0), self.ch.off_level)
        self.assertEqual(self.ch.level_for(1.0), self.ch.on_level)
        self.assertAlmostEqual(self.ch.off_level, round(OFF_TRUE), delta=1)
        self.assertAlmostEqual(self.ch.on_level, round(ON_TRUE), delta=1)

    def test_out_of_range_values_clip(self) -> None:
        self.assertEqual(self.ch.level_for(-0.5), self.ch.off_level)
        self.assertEqual(self.ch.level_for(1.5), self.ch.on_level)

    def test_monotone_in_v(self) -> None:
        out = [self.ch.level_for(v) for v in np.linspace(0.0, 1.0, 201)]
        self.assertTrue(all(b >= a for a, b in zip(out, out[1:])))

    def test_levels_stay_inside_the_panel_range(self) -> None:
        far = EncodingChannel(
            index=0, side="x", x_center=100, x_start=93, x_end=108,
            wavelength_nm=780.0, levels=LEVELS.copy(),
            intensity_curve=curve(on_level=1400.0), method="fit",
        )
        for v in (0.0, 0.5, 1.0):
            self.assertTrue(0 <= far.level_for(v) <= MAX_LEVEL)

    def test_delivered_power_matches_the_request(self) -> None:
        """v is the fraction of the fitted contrast above the fitted floor."""
        fit = self.ch.transfer_fit
        for v in (0.1, 0.25, 0.5, 0.75, 0.9):
            got = float(fit.intensity(self.ch.level_for(v)))
            want = fit.floor + fit.contrast * v
            # the residual is the integer-grayscale quantum: rounding the level
            # by up to half a step moves the retardance by slope/2 rad
            self.assertAlmostEqual(got, want, delta=2e-3)

    def test_v_maps_to_the_retardance_the_phase_model_assumes(self) -> None:
        """The identity that interpolation could not promise.

        ``calibration_module.phase.phi_half`` takes phi/2 = asin(sqrt(v)) for a
        channel commanded at v.  Inverting the fit makes that exact, so the
        amplitude the encoder writes and the phase steps 6/7/8 assume come from
        one model.
        """
        from calibration_module.phase import phi_half

        fit = self.ch.transfer_fit
        for v in (0.05, 0.2, 0.5, 0.8, 1.0):
            delivered = float(fit.retardance(fit.level_for(v)))
            self.assertAlmostEqual(delivered, 2.0 * float(phi_half(v)), places=2)

    def test_interp_still_available_on_a_fitted_channel(self) -> None:
        """Both mappings live on every channel, so they can be compared."""
        self.assertEqual(self.ch.method, "fit")
        self.assertIsNotNone(self.ch.transfer_fit)
        # on a clean curve the two agree closely; the point is that both run
        for v in (0.25, 0.5, 0.75):
            self.assertLess(abs(self.ch.level_for_fit(v) - self.ch.level_for_interp(v)), 12)

    def test_unknown_method_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown encoding method"):
            EncodingChannel(
                index=0, side="x", x_center=100, x_start=93, x_end=108,
                wavelength_nm=780.0, levels=LEVELS.copy(),
                intensity_curve=curve(), method="spline",
            )


class CalibrationPlumbingTests(unittest.TestCase):
    """Fits are written with the calibration and picked up by the loader."""

    GRID = np.array([420.0, 440.0, 480.0, 500.0, 520.0, 540.0, 580.0, 600.0])

    def _calib(self) -> CalibrationResult:
        rows = np.stack([curve(contrast=1.0 + 0.01 * i) for i in range(self.GRID.size)])
        return CalibrationResult(
            wavelength=-0.005 * self.GRID + 781.0,
            coordinates=self.GRID.copy(),
            max_level=860,
            min_level=390,
            level_range=LEVELS.copy(),
            intensity_levels=rows,
        )

    def test_save_fits_and_load_restores_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = save_calibration_result(self._calib(), Path(tmp) / "step3b.json")
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "calibration_result_v2")
            self.assertEqual(len(payload["transfer_fits"]), self.GRID.size)
            # the raw sweep stays in the file, so a curve can always be refitted
            self.assertIsNotNone(payload["intensity_levels"])

            back = load_calibration_result(out)
            self.assertEqual(len(back.transfer_fits), self.GRID.size)
            self.assertAlmostEqual(back.transfer_fits[0].on_level_exact, ON_TRUE,
                                   places=2)

    def test_save_without_fitting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = save_calibration_result(self._calib(), Path(tmp) / "raw.json",
                                          fit_transfer=False)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertIsNone(payload["transfer_fits"])
            self.assertIsNone(load_calibration_result(out).transfer_fits)

    def test_unfittable_curves_still_save(self) -> None:
        """A fit failure must never cost the measurement."""
        calib = self._calib()
        calib.intensity_levels = np.zeros_like(calib.intensity_levels)
        with tempfile.TemporaryDirectory() as tmp:
            out = save_calibration_result(calib, Path(tmp) / "flat.json")
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertIsNone(payload["transfer_fits"])
            self.assertIsNotNone(payload["intensity_levels"])

    def test_layout_prefers_stored_fits_over_refitting(self) -> None:
        calib = self._calib()
        stored = fit_transfer_curves(calib.level_range, calib.intensity_levels)
        # a recognisable marker: shift one channel's window by 40 levels
        row = 0
        moved = TransferFit(
            floor=stored[row].floor, contrast=stored[row].contrast,
            phase_slope=stored[row].phase_slope,
            phase_offset=stored[row].phase_offset - 40.0 * stored[row].phase_slope,
            level_min=stored[row].level_min, level_max=stored[row].level_max,
        )
        calib.transfer_fits = [moved] + stored[1:]
        layout = channel_layout_from_calibration(calib, warn=False)

        # row 0 is coordinate 420 px -- the outermost x channel
        ch = next(c for c in layout.all_channels if c.x_center == 420)
        self.assertAlmostEqual(ch.transfer_fit.on_level_exact, ON_TRUE + 40.0, places=2)
        # everything else refits to the true window
        other = next(c for c in layout.all_channels if c.x_center == 500)
        self.assertAlmostEqual(other.transfer_fit.on_level_exact, ON_TRUE, places=2)

    def test_layout_fits_when_the_file_has_none(self) -> None:
        layout = channel_layout_from_calibration(self._calib(), warn=False)
        self.assertTrue(all(c.method == "fit" for c in layout.all_channels))
        self.assertTrue(all(c.transfer_fit is not None for c in layout.all_channels))
        # background columns sit at the modelled extinction, not the swept argmin
        self.assertTrue(
            np.allclose(layout.calib_off_levels, round(OFF_TRUE), atol=1)
        )

    def test_interp_method_still_selectable(self) -> None:
        layout = channel_layout_from_calibration(self._calib(), method="interp")
        self.assertTrue(all(c.method == "interp" for c in layout.all_channels))
        self.assertTrue(all(c.transfer_fit is None for c in layout.all_channels))
        # interp full scale is the swept argmax
        self.assertEqual(layout.x_channels[0].on_level, int(LEVELS[np.argmax(curve())]))

    def test_unknown_layout_method_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown encoding method"):
            channel_layout_from_calibration(self._calib(), method="spline")


class AlignmentBuilderTests(unittest.TestCase):
    """The tiling and anchor builders default to interp but accept "fit".

    They give several channels the SAME measured curve -- one snapped from the
    nearest calibration coordinate, one shared anchor -- so a fitted per-channel
    full scale would claim precision the data does not carry.  The option is
    there for a grid that really is one curve per channel.
    """

    def test_build_channel_layout_defaults_to_interp_and_accepts_fit(self) -> None:
        coords = np.linspace(100.0, 900.0, 9)
        calib = CalibrationResult(
            wavelength=-0.005 * coords + 781.0,
            coordinates=coords,
            max_level=860,
            min_level=390,
            level_range=LEVELS.copy(),
            intensity_levels=np.stack([curve() for _ in coords]),
        )
        default = build_channel_layout(calib, n_channels=4)
        self.assertTrue(all(c.method == "interp" for c in default.all_channels))

        fitted = build_channel_layout(calib, n_channels=4, method="fit")
        self.assertTrue(all(c.method == "fit" for c in fitted.all_channels))
        for ch in fitted.all_channels:
            self.assertAlmostEqual(ch.on_level, round(ON_TRUE), delta=1)

    def test_single_anchor_layout_defaults_to_interp_and_accepts_fit(self) -> None:
        step2 = CalibrationResult(
            wavelength=np.asarray([780.0, 778.0, 776.0]),
            coordinates=np.asarray([0.0, 100.0, 200.0]),
            max_level=860, min_level=390, level_range=LEVELS.copy(),
        )
        intensity = CalibrationResult(
            wavelength=np.asarray([778.0]),
            coordinates=np.asarray([100.0]),
            max_level=860, min_level=390, level_range=LEVELS.copy(),
            intensity_levels=curve()[None, :],
        )
        default, _ = build_single_anchor_layout(step2, intensity,
                                                target_wavelength_nm=778.0)
        self.assertTrue(all(c.method == "interp" for c in default.all_channels))

        fitted, _ = build_single_anchor_layout(step2, intensity,
                                               target_wavelength_nm=778.0,
                                               method="fit")
        self.assertTrue(all(c.method == "fit" for c in fitted.all_channels))
        # every channel shares the anchor's fit, hence its window
        self.assertEqual({c.on_level for c in fitted.all_channels},
                         {round(ON_TRUE)})


if __name__ == "__main__":
    unittest.main()
