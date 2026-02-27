from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ltspice_mcp import server
from ltspice_mcp.models import RawDataset, RawVariable


class TestRenderSessions(unittest.TestCase):
    def setUp(self) -> None:
        server._render_sessions.clear()

    def test_render_session_reuses_window_and_closes_on_end(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_render_session_test_"))
        asc_path = temp_dir / "session.asc"
        asc_path.write_text("Version 4\nSHEET 1 100 100\n", encoding="utf-8")

        def _capture_side_effect(**kwargs: object) -> dict[str, object]:
            output_path = Path(str(kwargs["output_path"]))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"\x89PNG\r\n\x1a\n")
            return {
                "image_path": str(output_path),
                "capture_backend": "screencapturekit",
                "capture_window_info": {"capture_mode": "screencapturekit_window"},
            }

        with (
            patch("ltspice_mcp.server.open_in_ltspice_ui") as open_mock,
            patch("ltspice_mcp.server.capture_ltspice_window_screenshot", side_effect=_capture_side_effect) as capture_mock,
            patch("ltspice_mcp.server.close_ltspice_window") as close_mock,
        ):
            open_mock.return_value = {"opened": True}
            close_mock.return_value = {"closed": True}

            session = server.startLtspiceRenderSession(path=str(asc_path))
            session_id = session["render_session_id"]

            result = server.renderLtspiceSchematicImage(
                asc_path=str(asc_path),
                render_session_id=session_id,
            )
            self.assertFalse(result.isError)
            capture_kwargs = capture_mock.call_args.kwargs
            self.assertIsNone(capture_kwargs["open_path"])
            self.assertEqual(capture_kwargs["close_after_capture"], False)

            end = server.endLtspiceRenderSession(render_session_id=session_id)
            self.assertTrue(end["close_event"]["closed"])
            close_mock.assert_called_once()

    def test_close_ltspice_window_tool(self) -> None:
        with patch("ltspice_mcp.server.close_ltspice_window") as close_mock:
            close_mock.return_value = {"closed": True}
            payload = server.closeLtspiceWindow("foo.asc")
        self.assertTrue(payload["closed"])

    def test_plot_render_session_reuses_window(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_plot_render_session_test_"))
        raw_path = temp_dir / "session.raw"
        raw_path.write_text("raw placeholder\n", encoding="utf-8")
        dataset = RawDataset(
            path=raw_path.resolve(),
            plot_name="AC Analysis",
            flags={"complex"},
            metadata={},
            variables=[
                RawVariable(index=0, name="frequency", kind="frequency"),
                RawVariable(index=1, name="V(out)", kind="voltage"),
            ],
            values=[
                [10.0 + 0j, 100.0 + 0j, 1000.0 + 0j],
                [1.0 + 0j, 0.5 - 0.5j, 0.1 - 0.9j],
            ],
            steps=[],
        )

        def _capture_side_effect(**kwargs: object) -> dict[str, object]:
            output_path = Path(str(kwargs["output_path"]))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"\x89PNG\r\n\x1a\n")
            return {
                "image_path": str(output_path),
                "capture_backend": "screencapturekit",
                "capture_window_info": {"capture_mode": "screencapturekit_window"},
            }

        with (
            patch("ltspice_mcp.server.open_in_ltspice_ui") as open_mock,
            patch("ltspice_mcp.server.capture_ltspice_window_screenshot", side_effect=_capture_side_effect) as capture_mock,
            patch("ltspice_mcp.server._resolve_dataset", return_value=dataset),
            patch(
                "ltspice_mcp.server._validate_plot_capture",
                return_value={"valid": True, "trace_pixels": 400, "min_trace_pixels": 120},
            ),
        ):
            open_mock.return_value = {"opened": True}
            session = server.startLtspiceRenderSession(path=str(raw_path))
            session_id = session["render_session_id"]
            result = server.renderLtspicePlotImage(
                vectors=["V(out)"],
                render_session_id=session_id,
            )
            self.assertFalse(result.isError)
            capture_kwargs = capture_mock.call_args.kwargs
            self.assertIsNone(capture_kwargs["open_path"])
            self.assertEqual(capture_kwargs["close_after_capture"], False)

    def test_start_render_session_fails_when_ui_open_fails(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_render_session_fail_test_"))
        asc_path = temp_dir / "session_fail.asc"
        asc_path.write_text("Version 4\nSHEET 1 100 100\n", encoding="utf-8")

        with patch("ltspice_mcp.server.open_in_ltspice_ui") as open_mock:
            open_mock.return_value = {"opened": False, "return_code": 1, "stderr": "launch failed"}
            with self.assertRaisesRegex(RuntimeError, "Failed to open LTspice UI target"):
                server.startLtspiceRenderSession(path=str(asc_path))

        self.assertEqual(server._render_sessions, {})


if __name__ == "__main__":
    unittest.main()
