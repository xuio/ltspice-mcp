from __future__ import annotations

import unittest

from ltspice_mcp.analysis import find_local_extrema, interpolate_series, sample_indices


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


if __name__ == "__main__":
    unittest.main()
