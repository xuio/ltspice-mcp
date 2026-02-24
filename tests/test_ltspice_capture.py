from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from ltspice_mcp.ltspice import (
    _capture_ltspice_window_with_screencapturekit,
    capture_ltspice_window_screenshot,
    open_in_ltspice_ui,
)


class TestOpenInLtspiceUi(unittest.TestCase):
    def test_open_in_ltspice_ui_uses_background_flag(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_open_ui_test_"))
        target = temp_dir / "test.asc"
        target.write_text("Version 4\nSHEET 1 100 100\n", encoding="utf-8")

        with (
            patch("ltspice_mcp.ltspice.platform.system", return_value="Darwin"),
            patch("ltspice_mcp.ltspice.subprocess.run") as run_mock,
        ):
            run_mock.return_value = CompletedProcess(
                args=["open"],
                returncode=0,
                stdout="",
                stderr="",
            )
            payload = open_in_ltspice_ui(target, background=True)

        self.assertTrue(payload["opened"])
        self.assertTrue(payload["background"])
        self.assertEqual(payload["command"][:2], ["open", "-g"])
        self.assertIn("-j", payload["command"])
        self.assertIn(str(target.resolve()), payload["command"])


class TestScreenCaptureKitPath(unittest.TestCase):
    def test_screencapturekit_helper_invokes_xcrun_and_parses_json(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_sck_helper_test_"))
        output_path = temp_dir / "capture.png"

        def _run_side_effect(cmd: list[str], **_: object) -> CompletedProcess[str]:
            output_path.write_bytes(b"\x89PNG\r\n\x1a\n")
            return CompletedProcess(
                args=cmd,
                returncode=0,
                stdout='noise\n{"window_id": 42, "capture_mode": "screencapturekit_window", "window_title": "test"}\n',
                stderr="",
            )

        with (
            patch("ltspice_mcp.ltspice.platform.system", return_value="Darwin"),
            patch("ltspice_mcp.ltspice.shutil.which", return_value="/usr/bin/xcrun"),
            patch("ltspice_mcp.ltspice.subprocess.run", side_effect=_run_side_effect) as run_mock,
        ):
            payload = _capture_ltspice_window_with_screencapturekit(
                output_path=output_path,
                title_hint="foo.asc",
                timeout_seconds=5.0,
            )

        self.assertEqual(payload["window_id"], 42)
        self.assertEqual(payload["capture_mode"], "screencapturekit_window")
        self.assertTrue(output_path.exists())
        cmd = run_mock.call_args[0][0]
        self.assertEqual(cmd[:2], ["xcrun", "swift"])
        self.assertEqual(cmd[-1], "foo.asc")

    def test_capture_ltspice_window_screenshot_uses_sck_and_background_open(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_sck_path_test_"))
        open_path = temp_dir / "source.asc"
        open_path.write_text("Version 4\nSHEET 1 100 100\n", encoding="utf-8")
        output_path = temp_dir / "shot.png"

        def _sck_side_effect(**kwargs: object) -> dict[str, object]:
            output_path.write_bytes(b"\x89PNG\r\n\x1a\n")
            return {
                "window_id": 777,
                "window_title": "source.asc",
                "capture_mode": "screencapturekit_window",
                "window_frame": {"x": 1, "y": 2, "width": 300, "height": 200},
            }

        with (
            patch("ltspice_mcp.ltspice.open_in_ltspice_ui") as open_mock,
            patch(
                "ltspice_mcp.ltspice._capture_ltspice_window_with_screencapturekit",
                side_effect=_sck_side_effect,
            ) as sck_mock,
            patch("ltspice_mcp.ltspice._downscale_image_file", return_value={"downscaled": True}),
            patch("ltspice_mcp.ltspice._probe_image_dimensions", return_value=(300, 200)),
        ):
            open_mock.return_value = {"opened": True, "path": str(open_path)}
            payload = capture_ltspice_window_screenshot(
                output_path=output_path,
                open_path=open_path,
                settle_seconds=0.0,
                downscale_factor=0.5,
                prefer_screencapturekit=True,
                avoid_space_switch=True,
            )

        open_mock.assert_called_once_with(open_path, background=True)
        self.assertEqual(sck_mock.call_count, 1)
        self.assertEqual(payload["capture_backend"], "screencapturekit")
        self.assertEqual(payload["window_id"], 777)
        self.assertEqual(payload["capture_window_info"]["capture_mode"], "screencapturekit_window")
        self.assertEqual(payload["capture_command"], None)
        self.assertEqual(payload["width"], 300)
        self.assertEqual(payload["height"], 200)

    def test_capture_ltspice_window_screenshot_raises_on_sck_failure_when_preferred(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_sck_fail_test_"))
        open_path = temp_dir / "source.asc"
        open_path.write_text("Version 4\nSHEET 1 100 100\n", encoding="utf-8")

        with (
            patch("ltspice_mcp.ltspice.open_in_ltspice_ui") as open_mock,
            patch(
                "ltspice_mcp.ltspice._capture_ltspice_window_with_screencapturekit",
                side_effect=RuntimeError("SCK failed"),
            ),
        ):
            open_mock.return_value = {"opened": True, "path": str(open_path)}
            with self.assertRaises(RuntimeError):
                capture_ltspice_window_screenshot(
                    output_path=temp_dir / "shot.png",
                    open_path=open_path,
                    settle_seconds=0.0,
                    prefer_screencapturekit=True,
                )

    def test_capture_ltspice_window_screenshot_falls_back_when_sck_disabled(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_sck_disabled_test_"))
        open_path = temp_dir / "source.asc"
        open_path.write_text("Version 4\nSHEET 1 100 100\n", encoding="utf-8")
        output_path = temp_dir / "shot.png"

        def _run_side_effect(cmd: list[str], **_: object) -> CompletedProcess[str]:
            output_path.write_bytes(b"\x89PNG\r\n\x1a\n")
            return CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        with (
            patch("ltspice_mcp.ltspice.open_in_ltspice_ui") as open_mock,
            patch("ltspice_mcp.ltspice.subprocess.run", side_effect=_run_side_effect),
            patch("ltspice_mcp.ltspice._downscale_image_file", return_value={"downscaled": False}),
            patch("ltspice_mcp.ltspice._probe_image_dimensions", return_value=(640, 480)),
        ):
            open_mock.return_value = {"opened": True, "path": str(open_path)}
            payload = capture_ltspice_window_screenshot(
                output_path=output_path,
                open_path=open_path,
                settle_seconds=0.0,
                prefer_screencapturekit=False,
            )

        self.assertEqual(payload["capture_backend"], "screencapture")
        self.assertIsInstance(payload["capture_command"], list)
        self.assertEqual(payload["capture_command"][:2], ["screencapture", "-x"])


if __name__ == "__main__":
    unittest.main()
