from __future__ import annotations

import os
import platform
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from ltspice_mcp.ltspice import (
    _capture_ltspice_window_with_screencapturekit,
    capture_ltspice_window_screenshot,
    find_ltspice_executable,
    open_in_ltspice_ui,
)


def _real_sck_tests_enabled() -> bool:
    value = os.getenv("LTSPICE_MCP_RUN_REAL_SCK", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


@unittest.skipUnless(platform.system() == "Darwin", "ScreenCaptureKit tests require macOS")
class TestScreenCaptureKitIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not _real_sck_tests_enabled():
            raise unittest.SkipTest(
                "Set LTSPICE_MCP_RUN_REAL_SCK=1 to run real ScreenCaptureKit integration tests."
            )
        if shutil.which("xcrun") is None:
            raise unittest.SkipTest("xcrun is not available; ScreenCaptureKit backend cannot run.")
        if find_ltspice_executable() is None:
            raise unittest.SkipTest("LTspice executable was not found on this system.")

        cls.temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_sck_integration_"))
        cls.asc_path = cls.temp_dir / "sck_probe.asc"
        cls.asc_path.write_text(
            "Version 4\n"
            "SHEET 1 880 680\n"
            "WIRE 128 128 272 128\n"
            "SYMBOL voltage 128 128 R0\n"
            "SYMATTR InstName V1\n"
            "SYMBOL res 272 128 R0\n"
            "SYMATTR InstName R1\n"
            "SYMATTR Value 1k\n"
            "FLAG 128 176 0\n"
            "FLAG 272 176 out\n"
            "TEXT 64 224 Left 2 !.op\n",
            encoding="utf-8",
        )
        warmup_event = open_in_ltspice_ui(cls.asc_path, background=True)
        if not warmup_event.get("opened"):
            raise unittest.SkipTest(f"Failed to warm up LTspice UI: {warmup_event}")
        time.sleep(1.5)

    def test_real_screencapturekit_helper(self) -> None:
        open_event = open_in_ltspice_ui(self.asc_path, background=True)
        self.assertTrue(open_event.get("opened"), f"LTspice open failed: {open_event}")

        time.sleep(1.0)
        output_path = self.temp_dir / "helper_capture.png"
        payload = _capture_ltspice_window_with_screencapturekit(
            output_path=output_path,
            title_hint=self.asc_path.name,
            timeout_seconds=30.0,
        )

        self.assertTrue(output_path.exists(), "ScreenCaptureKit helper did not write an image.")
        self.assertGreater(output_path.stat().st_size, 0, "Captured image is empty.")
        self.assertEqual(payload.get("capture_mode"), "screencapturekit_window")
        self.assertIsInstance(payload.get("window_id"), int)

    def test_real_capture_ltspice_window_screenshot(self) -> None:
        output_path = self.temp_dir / "capture_wrapper.png"
        payload = capture_ltspice_window_screenshot(
            output_path=output_path,
            open_path=self.asc_path,
            title_hint=self.asc_path.name,
            settle_seconds=1.5,
            downscale_factor=0.6,
            avoid_space_switch=True,
            prefer_screencapturekit=True,
        )

        self.assertTrue(output_path.exists(), "capture_ltspice_window_screenshot did not write an image.")
        self.assertGreater(output_path.stat().st_size, 0, "Captured image is empty.")
        self.assertEqual(payload.get("capture_backend"), "screencapturekit")
        self.assertTrue(payload.get("avoid_space_switch"))
        self.assertEqual(payload.get("capture_window_info", {}).get("capture_mode"), "screencapturekit_window")
        self.assertTrue(payload.get("open_event", {}).get("opened"))
        self.assertTrue(payload.get("open_event", {}).get("background"))
        self.assertEqual(payload.get("downscale_factor"), 0.6)
        self.assertGreater(int(payload.get("width") or 0), 0)
        self.assertGreater(int(payload.get("height") or 0), 0)
        self.assertTrue(payload.get("close_event", {}).get("closed"))


if __name__ == "__main__":
    unittest.main()
