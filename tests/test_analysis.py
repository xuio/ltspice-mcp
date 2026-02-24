from __future__ import annotations

import unittest

from ltspice_mcp.analysis import (
    compute_bandwidth,
    compute_gain_phase_margin,
    compute_rise_fall_time,
    compute_settling_time,
    find_local_extrema,
    interpolate_series,
    sample_indices,
)


class TestAnalysis(unittest.TestCase):
    def test_sample_indices_keep_bounds(self) -> None:
        indices = sample_indices(total_points=100, max_points=9)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 99)
        self.assertLessEqual(len(indices), 9)

    def test_interpolate_series(self) -> None:
        scale = [0.0, 1.0, 2.0]
        series = [complex(0.0, 0.0), complex(1.0, 1.0), complex(2.0, 2.0)]
        sampled = interpolate_series(scale, series, [0.5, 1.5])
        self.assertAlmostEqual(sampled[0].real, 0.5, places=8)
        self.assertAlmostEqual(sampled[0].imag, 0.5, places=8)
        self.assertAlmostEqual(sampled[1].real, 1.5, places=8)

    def test_find_local_extrema(self) -> None:
        scale = [0, 1, 2, 3, 4, 5]
        series = [complex(0), complex(2), complex(1), complex(3), complex(0), complex(2)]
        extrema = find_local_extrema(
            scale=scale,
            series=series,
            include_minima=True,
            include_maxima=True,
            threshold=0.1,
        )
        labels = {(item["type"], item["index"]) for item in extrema}
        self.assertIn(("max", 1), labels)
        self.assertIn(("min", 2), labels)
        self.assertIn(("max", 3), labels)
        self.assertIn(("min", 4), labels)

    def test_compute_bandwidth(self) -> None:
        freq = [1.0, 10.0, 100.0, 1000.0]
        # 1st-order low-pass magnitude approximation.
        response = [complex(1.0), complex(0.995), complex(0.707), complex(0.1)]
        result = compute_bandwidth(freq, response, reference="first", drop_db=3.0)
        self.assertIsNotNone(result["lowpass_bandwidth_hz"])
        self.assertAlmostEqual(result["lowpass_bandwidth_hz"], 100.0, delta=5.0)

    def test_compute_gain_phase_margin(self) -> None:
        freq = [1.0, 10.0, 100.0, 1000.0]
        # Magnitude crosses 0 dB between 10 and 100, phase crosses -180 between 100 and 1000.
        response = [
            complex(10.0, 0.0),  # +20 dB, 0 deg
            complex(2.0, -2.0),  # +9 dB, -45 deg
            complex(-0.5, -0.5),  # -3 dB, -135 deg
            complex(-0.1, 0.0),  # -20 dB, -180 deg
        ]
        result = compute_gain_phase_margin(freq, response)
        self.assertIsNotNone(result["gain_crossover_hz"])
        self.assertIsNotNone(result["phase_margin_deg"])

    def test_compute_rise_fall_time(self) -> None:
        t = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [complex(0.0), complex(0.2), complex(0.8), complex(1.0), complex(1.0)]
        result = compute_rise_fall_time(t, y, low_threshold_pct=10.0, high_threshold_pct=90.0)
        self.assertIsNotNone(result["rise_time_s"])
        self.assertGreater(result["rise_time_s"], 0.0)

    def test_compute_settling_time(self) -> None:
        t = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [complex(0.0), complex(1.5), complex(1.1), complex(1.02), complex(1.0)]
        result = compute_settling_time(t, y, tolerance_percent=2.0, target_value=1.0)
        self.assertIsNotNone(result["settling_time_s"])
        self.assertGreaterEqual(result["settling_time_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
