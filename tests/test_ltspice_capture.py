from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired
from unittest.mock import patch

from ltspice_mcp.ltspice import (
    _capture_ltspice_window_with_screencapturekit,
    capture_ltspice_window_screenshot,
    close_ltspice_window,
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
        self.assertEqual(payload["background_requested"], True)
        self.assertEqual(payload["command"][:2], ["open", "-g"])
        self.assertNotIn("-j", payload["command"])
        self.assertIn(str(target.resolve()), payload["command"])

    def test_open_in_ltspice_ui_can_launch_hidden_when_enabled(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_open_ui_hidden_test_"))
        target = temp_dir / "test.asc"
        target.write_text("Version 4\nSHEET 1 100 100\n", encoding="utf-8")

        with (
            patch("ltspice_mcp.ltspice.platform.system", return_value="Darwin"),
            patch("ltspice_mcp.ltspice.os.getenv", return_value="1"),
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
        self.assertIn("-j", payload["command"])


class TestScreenCaptureKitPath(unittest.TestCase):
    def test_screencapturekit_helper_invokes_persistent_helper_and_parses_json(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_sck_helper_test_"))
        output_path = temp_dir / "capture.png"
        helper_path = temp_dir / "ltspice-sck-helper"

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
            patch(
                "ltspice_mcp.ltspice._ensure_screencapturekit_helper",
                return_value=(
                    helper_path,
                    {"helper_path": str(helper_path), "helper_source": "compiled_cache", "compiled": False},
                ),
            ),
            patch("ltspice_mcp.ltspice.subprocess.run", side_effect=_run_side_effect) as run_mock,
        ):
            payload = _capture_ltspice_window_with_screencapturekit(
                output_path=output_path,
                title_hint="foo.asc",
                timeout_seconds=5.0,
            )

        self.assertEqual(payload["window_id"], 42)
        self.assertEqual(payload["capture_mode"], "screencapturekit_window")
        self.assertIn("capture_diagnostics", payload)
        self.assertTrue(output_path.exists())
        cmd = run_mock.call_args[0][0]
        self.assertEqual(cmd[0], str(helper_path))
        self.assertEqual(cmd[2], "foo.asc")

    def test_screencapturekit_helper_retries_after_timeout(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_sck_retry_test_"))
        output_path = temp_dir / "capture.png"
        helper_path = temp_dir / "ltspice-sck-helper"
        timeout_exc = TimeoutExpired(cmd=[str(helper_path)], timeout=3.0)

        def _run_side_effect(cmd: list[str], **_: object) -> CompletedProcess[str]:
            if _run_side_effect.calls == 0:
                _run_side_effect.calls += 1
                raise timeout_exc
            output_path.write_bytes(b"\x89PNG\r\n\x1a\n")
            return CompletedProcess(
                args=cmd,
                returncode=0,
                stdout='{"window_id": 101, "capture_mode": "screencapturekit_window"}\n',
                stderr="",
            )
        _run_side_effect.calls = 0  # type: ignore[attr-defined]

        with (
            patch("ltspice_mcp.ltspice.platform.system", return_value="Darwin"),
            patch(
                "ltspice_mcp.ltspice._ensure_screencapturekit_helper",
                return_value=(
                    helper_path,
                    {"helper_path": str(helper_path), "helper_source": "compiled_cache", "compiled": False},
                ),
            ),
            patch("ltspice_mcp.ltspice.subprocess.run", side_effect=_run_side_effect) as run_mock,
        ):
            payload = _capture_ltspice_window_with_screencapturekit(
                output_path=output_path,
                title_hint="foo.asc",
                timeout_seconds=6.0,
                attempts=2,
                retry_delay=0.0,
            )

        self.assertEqual(payload["window_id"], 101)
        self.assertEqual(payload["capture_diagnostics"]["attempt_count"], 2)
        self.assertEqual(run_mock.call_count, 2)

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
            patch("ltspice_mcp.ltspice.close_ltspice_window") as close_mock,
            patch("ltspice_mcp.ltspice._downscale_image_file", return_value={"downscaled": True}),
            patch("ltspice_mcp.ltspice._probe_image_dimensions", return_value=(300, 200)),
        ):
            open_mock.return_value = {"opened": True, "path": str(open_path)}
            close_mock.return_value = {"closed": True, "title_hint": "source.asc"}
            payload = capture_ltspice_window_screenshot(
                output_path=output_path,
                open_path=open_path,
                settle_seconds=0.0,
                downscale_factor=0.5,
                prefer_screencapturekit=True,
                avoid_space_switch=True,
            )

        open_mock.assert_called_once_with(open_path, background=True)
        close_mock.assert_called_once_with(
            "source.asc",
            window_id=777,
            exact_title="source.asc",
            attempts=5,
            retry_delay=0.2,
        )
        self.assertEqual(sck_mock.call_count, 1)
        self.assertEqual(payload["capture_backend"], "screencapturekit")
        self.assertEqual(payload["window_id"], 777)
        self.assertEqual(payload["capture_window_info"]["capture_mode"], "screencapturekit_window")
        self.assertEqual(payload["capture_command"], None)
        self.assertEqual(payload["width"], 300)
        self.assertEqual(payload["height"], 200)
        self.assertIsInstance(payload["capture_id"], str)
        self.assertEqual(payload["capture_diagnostics"]["capture_id"], payload["capture_id"])
        self.assertEqual(
            sck_mock.call_args.kwargs.get("capture_id"),
            payload["capture_id"],
        )
        self.assertTrue(payload["close_event"]["closed"])
        self.assertIn("capture_diagnostics", payload)

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
            patch("ltspice_mcp.ltspice.close_ltspice_window") as close_mock,
        ):
            open_mock.return_value = {"opened": True, "path": str(open_path)}
            close_mock.return_value = {"closed": True, "title_hint": "source.asc"}
            with self.assertRaises(RuntimeError):
                capture_ltspice_window_screenshot(
                    output_path=temp_dir / "shot.png",
                    open_path=open_path,
                    settle_seconds=0.0,
                    prefer_screencapturekit=True,
                )
            close_mock.assert_called_once_with(
                "source.asc",
                attempts=5,
                retry_delay=0.2,
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
            patch("ltspice_mcp.ltspice.close_ltspice_window") as close_mock,
            patch("ltspice_mcp.ltspice._downscale_image_file", return_value={"downscaled": False}),
            patch("ltspice_mcp.ltspice._probe_image_dimensions", return_value=(640, 480)),
        ):
            open_mock.return_value = {"opened": True, "path": str(open_path)}
            close_mock.return_value = {"closed": True, "title_hint": "source.asc"}
            payload = capture_ltspice_window_screenshot(
                output_path=output_path,
                open_path=open_path,
                settle_seconds=0.0,
                prefer_screencapturekit=False,
            )

        self.assertEqual(payload["capture_backend"], "screencapture")
        self.assertIsInstance(payload["capture_command"], list)
        self.assertEqual(payload["capture_command"][:2], ["screencapture", "-x"])
        close_mock.assert_called_once_with(
            "source.asc",
            attempts=5,
            retry_delay=0.2,
        )
        self.assertTrue(payload["close_event"]["closed"])


class TestCloseLtspiceWindow(unittest.TestCase):
    def test_close_ltspice_window_requires_any_selector(self) -> None:
        with patch("ltspice_mcp.ltspice.platform.system", return_value="Darwin"):
            payload = close_ltspice_window("  ")
        self.assertFalse(payload["closed"])
        self.assertIn("selector", payload["error"])

    def test_close_ltspice_window_uses_ax_helper_first(self) -> None:
        with (
            patch("ltspice_mcp.ltspice.platform.system", return_value="Darwin"),
            patch(
                "ltspice_mcp.ltspice._close_ltspice_window_with_ax_helper",
                return_value={
                    "closed": True,
                    "partially_closed": False,
                    "matched_windows": 1,
                    "closed_windows": 1,
                    "close_strategy": "ax_helper",
                    "status": "OK",
                    "return_code": 0,
                },
            ) as helper_mock,
            patch("ltspice_mcp.ltspice.subprocess.run") as run_mock,
        ):
            payload = close_ltspice_window("foo.asc")
        self.assertTrue(payload["closed"])
        self.assertEqual(payload["close_strategy"], "ax_helper")
        helper_mock.assert_called_once()
        run_mock.assert_not_called()

    def test_close_ltspice_window_returns_ax_helper_no_match_without_fallback(self) -> None:
        with (
            patch("ltspice_mcp.ltspice.platform.system", return_value="Darwin"),
            patch(
                "ltspice_mcp.ltspice._close_ltspice_window_with_ax_helper",
                return_value={
                    "closed": False,
                    "partially_closed": False,
                    "matched_windows": 0,
                    "closed_windows": 0,
                    "close_strategy": "ax_helper",
                    "status": "OK",
                    "return_code": 0,
                },
            ),
            patch("ltspice_mcp.ltspice.subprocess.run") as run_mock,
        ):
            payload = close_ltspice_window("foo.asc")
        self.assertFalse(payload["closed"])
        self.assertEqual(payload["status"], "OK")
        run_mock.assert_not_called()

    def test_close_ltspice_window_reports_no_match_as_not_closed(self) -> None:
        with (
            patch("ltspice_mcp.ltspice.platform.system", return_value="Darwin"),
            patch("ltspice_mcp.ltspice._close_ltspice_window_with_ax_helper", return_value=None),
            patch("ltspice_mcp.ltspice.subprocess.run") as run_mock,
        ):
            run_mock.return_value = CompletedProcess(
                args=["osascript"],
                returncode=0,
                stdout="OK|0|0\n",
                stderr="",
            )
            payload = close_ltspice_window("foo.asc")
        self.assertFalse(payload["closed"])
        self.assertEqual(payload["matched_windows"], 0)
        self.assertEqual(payload["closed_windows"], 0)
        self.assertEqual(payload["status"], "OK")

    def test_close_ltspice_window_parses_close_strategy_suffix(self) -> None:
        with (
            patch("ltspice_mcp.ltspice.platform.system", return_value="Darwin"),
            patch("ltspice_mcp.ltspice._close_ltspice_window_with_ax_helper", return_value=None),
            patch("ltspice_mcp.ltspice.subprocess.run") as run_mock,
        ):
            run_mock.return_value = CompletedProcess(
                args=["osascript"],
                returncode=0,
                stdout="OK|1|1|ax\n",
                stderr="",
            )
            payload = close_ltspice_window("foo.asc")
        self.assertTrue(payload["closed"])
        self.assertEqual(payload["close_strategy"], "ax")

    def test_close_ltspice_window_retries_until_closed(self) -> None:
        with (
            patch("ltspice_mcp.ltspice.platform.system", return_value="Darwin"),
            patch("ltspice_mcp.ltspice._close_ltspice_window_with_ax_helper", return_value=None),
            patch("ltspice_mcp.ltspice.subprocess.run") as run_mock,
        ):
            run_mock.side_effect = [
                CompletedProcess(
                    args=["osascript"],
                    returncode=0,
                    stdout="OK|0|0\n",
                    stderr="",
                ),
                CompletedProcess(
                    args=["osascript"],
                    returncode=0,
                    stdout="OK|1|1|ax\n",
                    stderr="",
                ),
            ]
            payload = close_ltspice_window("foo.asc", attempts=3, retry_delay=0.0)
        self.assertTrue(payload["closed"])
        self.assertEqual(payload["attempt_count"], 2)
        self.assertEqual(run_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
