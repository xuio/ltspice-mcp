from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ltspice_mcp import server
from ltspice_mcp.models import RawDataset, RawVariable


class TestPlotRenderBackendSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_plot_backend_test_"))
        server._configure_runner(workdir=self.temp_dir, ltspice_binary=None, timeout=10)

    def _dataset(self) -> RawDataset:
        raw_path = self.temp_dir / "sample.raw"
        raw_path.write_text("raw placeholder\n", encoding="utf-8")
        return RawDataset(
            path=raw_path.resolve(),
            plot_name="Transient Analysis",
            flags={"real"},
            metadata={},
            variables=[
                RawVariable(index=0, name="time", kind="time"),
                RawVariable(index=1, name="V(out)", kind="voltage"),
            ],
            values=[
                [0.0 + 0.0j, 1e-3 + 0.0j, 2e-3 + 0.0j],
                [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            ],
            steps=[],
        )

    def test_auto_backend_uses_ltspice_with_plt_settings(self) -> None:
        dataset = self._dataset()
        png_path = self.temp_dir / "ltspice_capture.png"

        def _fake_capture(**kwargs: object) -> dict[str, object]:
            output_path = Path(str(kwargs["output_path"]))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"\x89PNG\r\n\x1a\n")
            return {
                "image_path": str(output_path),
                "format": "png",
                "capture_backend": "screencapturekit",
                "plot_selection": {
                    "selected_count": 1,
                    "selected_vectors": ["V(out)"],
                },
            }

        with (
            patch("ltspice_mcp.server._resolve_dataset", return_value=dataset),
            patch("ltspice_mcp.server.capture_ltspice_window_screenshot", side_effect=_fake_capture) as capture_mock,
            patch("ltspice_mcp.server.render_plot_svg") as svg_mock,
        ):
            result = server.renderLtspicePlotImage(
                vectors=["V(out)"],
                backend="auto",
                output_path=str(png_path),
                settle_seconds=0.0,
            )

        payload = result.structuredContent
        self.assertEqual(payload["backend_used"], "ltspice")
        self.assertTrue(str(payload["image_path"]).endswith(".png"))
        self.assertIn("plot_settings", payload)
        self.assertTrue(Path(str(payload["plot_settings"]["plt_path"])).exists())
        capture_kwargs = capture_mock.call_args.kwargs
        self.assertEqual(Path(str(capture_kwargs["open_path"])), dataset.path)
        svg_mock.assert_not_called()

    def test_svg_backend_uses_svg_renderer(self) -> None:
        dataset = self._dataset()
        svg_path = self.temp_dir / "plot.svg"

        def _fake_svg_render(**_: object) -> dict[str, object]:
            svg_path.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="60"></svg>\n',
                encoding="utf-8",
            )
            return {
                "image_path": str(svg_path),
                "format": "svg",
                "width": 640,
                "height": 360,
                "points_total": 3,
                "points_rendered": 3,
                "x_log": False,
                "y_mode": "real",
                "y_range": [0.0, 1.0],
                "x_range": [0.0, 0.002],
            }

        with (
            patch("ltspice_mcp.server._resolve_dataset", return_value=dataset),
            patch("ltspice_mcp.server.capture_ltspice_window_screenshot") as capture_mock,
            patch("ltspice_mcp.server.render_plot_svg", side_effect=_fake_svg_render),
        ):
            result = server.renderLtspicePlotImage(
                vectors=["V(out)"],
                backend="svg",
            )

        payload = result.structuredContent
        self.assertEqual(payload["backend_used"], "svg")
        self.assertTrue(str(payload["image_path"]).endswith(".svg"))
        capture_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
