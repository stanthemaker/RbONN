"""Phase-fit rendering and comb-phase JSON round trip (no hardware).

Both suites moved here verbatim from ``tests/test_pipeline.py`` when the unified
pipeline was deleted: neither ever touched the pipeline -- they cover
:mod:`calibration_module.fit.report` and :mod:`calibration_module.fit.phase`, which are
very much alive, and would have been lost with the file that happened to host
them.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class PhaseReportRenderTests(unittest.TestCase):
    def test_plot_fringe_renders_on_agg(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        from calibration_module.fit.phase import PhaseFit
        from calibration_module.fit.report import plot_fringe

        n = 15
        theta = np.linspace(0.0, np.pi, n)
        g = np.sin(theta / 2.0) ** 2
        dphi_slm = theta - np.pi
        a, b, dphi_comb = 0.03, 0.02, 0.4
        y = a**2 + b**2 * g**2 + 2 * a * b * g * np.cos(dphi_slm + dphi_comb)
        std = np.full(n, 1e-5)
        fit = PhaseFit(
            dphi_comb=dphi_comb, dphi_comb_err=0.02,
            a=a, a_err=1e-3, b=b, b_err=1e-3,
            amp=2 * a * b, amp_err=1e-4,
            offset=0.0, offset_err=1e-5,
            r2=0.99,
            eta_ref=a, eta_tgt=b, bound_frac=1.0,
            a_at_bound=False, b_at_bound=False,
            bg0=0.0, bg1=0.0, bg2=0.0,
            dphi_slm=dphi_slm, g=g, y=y, std=std,
            known=a**2 + b**2 * g**2, y_pred=y, residuals=np.zeros(n),
        )
        fig = Figure(figsize=(8, 4))
        plot_fringe(fig, fit, tgt=3)      # must render without raising
        self.assertEqual(len(fig.axes), 2)


class CombPhaseJsonTests(unittest.TestCase):
    """save_comb_phase_json / load_comb_phase_json round trip (no hardware)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    @staticmethod
    def _fit(dphi: float, frac: float):
        from calibration_module.fit.phase import PhaseFit

        n = 5
        arr = np.zeros(n)
        return PhaseFit(
            dphi_comb=dphi, dphi_comb_err=0.02,
            a=0.05, a_err=1e-3, b=0.048, b_err=1e-3,
            amp=2 * 0.05 * 0.048, amp_err=1e-4,
            offset=0.0, offset_err=1e-5,
            r2=0.99,
            eta_ref=0.05, eta_tgt=0.048, bound_frac=frac,
            a_at_bound=False, b_at_bound=False,
            bg0=0.0, bg1=0.0, bg2=0.0,
            dphi_slm=arr, g=arr, y=arr, std=np.ones(n),
            known=arr, y_pred=arr, residuals=arr,
        )

    def test_round_trip_and_method_selection(self) -> None:
        import json

        from calibration_module.fit.phase import load_comb_phase_json, save_comb_phase_json

        step6 = self.out / "step6.json"
        step6.write_text(
            json.dumps({"step3": {"probe": 1}, "step6": {"channels": []}}),
            encoding="utf-8",
        )
        fits = {
            (3, "bounded"): self._fit(0.30, 1.0),
            (3, "fix"): self._fit(0.29, 0.0),
            (5, "bounded"): self._fit(-0.23, 1.0),
            (5, "fix"): self._fit(-0.51, 0.0),
        }
        out = self.out / "step7.json"
        save_comb_phase_json(fits, step6, out, ref_index=1,
                             csv_path="meas.csv", single_beam_bg=True)

        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["step3"], {"probe": 1})   # carried over verbatim
        self.assertEqual(payload["step6"], {"channels": []})
        self.assertEqual(payload["step7"]["ref_index"], 1)

        ref, entries = load_comb_phase_json(out, method="bounded")
        self.assertEqual(ref, 1)
        self.assertEqual(sorted(entries), [3, 5])
        self.assertAlmostEqual(entries[3]["fit"]["dphi_comb_rad"], 0.30)
        self.assertAlmostEqual(entries[5]["fit"]["dphi_comb_rad"], -0.23)

        _, fixed = load_comb_phase_json(out, method="fix")
        self.assertAlmostEqual(fixed[5]["fit"]["dphi_comb_rad"], -0.51)
        self.assertEqual(fixed[3]["fit"]["bound_frac"], 0.0)

        with self.assertRaisesRegex(ValueError, "several stored fits"):
            load_comb_phase_json(out)                      # ambiguous without method
        with self.assertRaisesRegex(ValueError, "no 'other' fit"):
            load_comb_phase_json(out, method="other")

if __name__ == "__main__":
    unittest.main()
