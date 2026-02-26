from __future__ import annotations

import plistlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ltspice_mcp.ltspice import (
    LTspiceRunner,
    _write_utf8_log_sidecar,
    analyze_log,
    get_ltspice_version,
    read_ltspice_window_text,
)


class TestLtspiceLogDiagnostics(unittest.TestCase):
    def test_structured_diagnostics_from_utf16_log(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_log_test_"))
        log_path = temp_dir / "sample.log"
        log_path.write_text(
            "\n".join(
                [
                    "Warning: timestep was adjusted",
                    "Singular matrix: node n001 is floating",
                    "Time step too small; convergence failed",
                    "Unknown subcircuit: XU1",
                ]
            )
            + "\n",
            encoding="utf-16le",
        )

        issues, warnings, diagnostics = analyze_log(log_path)

        self.assertGreaterEqual(len(issues), 2)
        self.assertGreaterEqual(len(warnings), 1)
        categories = {item.category for item in diagnostics}
        self.assertIn("floating_node", categories)
        self.assertIn("convergence", categories)
        self.assertIn("model_missing", categories)
        self.assertIn("warning", categories)

    def test_write_utf8_log_sidecar_from_utf16_source(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_log_sidecar_test_"))
        log_path = temp_dir / "sidecar.log"
        log_path.write_text("Warning: timestep was adjusted\n", encoding="utf-16le")

        sidecar = _write_utf8_log_sidecar(log_path)

        self.assertIsNotNone(sidecar)
        assert sidecar is not None
        self.assertTrue(sidecar.exists())
        self.assertTrue(str(sidecar).endswith(".log.utf8.txt"))
        self.assertEqual(sidecar.read_text(encoding="utf-8"), "Warning: timestep was adjusted\n")

    def test_get_ltspice_version_prefers_info_plist_without_subprocess(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_version_test_"))
        app_dir = temp_dir / "LTspice.app" / "Contents"
        macos_dir = app_dir / "MacOS"
        macos_dir.mkdir(parents=True, exist_ok=True)
        executable = macos_dir / "LTspice"
        executable.write_text("", encoding="utf-8")
        plist_path = app_dir / "Info.plist"
        plist_path.write_bytes(
            plistlib.dumps(
                {
                    "CFBundleShortVersionString": "17.2.4",
                    "CFBundleVersion": "1234",
                }
            )
        )
        get_ltspice_version.cache_clear()
        with patch("ltspice_mcp.ltspice.subprocess.run") as run_mock:
            version = get_ltspice_version(executable)
        self.assertEqual(version, "17.2.4 (1234)")
        run_mock.assert_not_called()

    def test_run_file_adds_ascii_mode_hint_without_log(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_ascii_mode_test_"))
        executable = temp_dir / "LTspice"
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
        netlist = temp_dir / "case.cir"
        netlist.write_text("* test\n.end\n", encoding="utf-8")

        runner = LTspiceRunner(workdir=temp_dir, executable=executable, default_timeout_seconds=5)
        with patch(
            "ltspice_mcp.ltspice.subprocess.run",
            return_value=SimpleNamespace(returncode=255, stdout="", stderr=""),
        ) as run_mock:
            run = runner.run_file(netlist, ascii_raw=True)

        invoked_command = run_mock.call_args.args[0]
        self.assertEqual(
            invoked_command,
            [str(executable.resolve()), "-b", str(netlist.resolve()), "-ascii"],
        )
        self.assertIn("LTspice exited with return code 255.", run.issues)
        self.assertIn(
            "No .log file was generated in -ascii mode; retry with ascii_raw=false to obtain diagnostics.",
            run.issues,
        )
        categories = {item.category for item in run.diagnostics}
        self.assertIn("ascii_raw_mode", categories)

    def test_run_file_purges_stale_outputs_and_flags_missing_regeneration(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_stale_output_test_"))
        executable = temp_dir / "LTspice"
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
        netlist = temp_dir / "stale_case.cir"
        netlist.write_text("* test\n.end\n", encoding="utf-8")
        stale_log = temp_dir / "stale_case.log"
        stale_raw = temp_dir / "stale_case.raw"
        stale_log.write_text("vmax: MAX(v(out))=2\n", encoding="utf-8")
        stale_raw.write_text("stale", encoding="utf-8")

        runner = LTspiceRunner(workdir=temp_dir, executable=executable, default_timeout_seconds=5)
        with patch(
            "ltspice_mcp.ltspice.subprocess.run",
            return_value=SimpleNamespace(returncode=255, stdout="", stderr=""),
        ):
            run = runner.run_file(netlist, ascii_raw=False)

        self.assertFalse(stale_log.exists())
        self.assertFalse(stale_raw.exists())
        self.assertIsNone(run.log_path)
        self.assertEqual(run.raw_files, [])
        self.assertIn(
            "Simulation artifacts were not regenerated for this run; refusing to reuse stale .log/.raw files.",
            run.issues,
        )
        categories = {item.category for item in run.diagnostics}
        self.assertIn("artifact_stale_or_missing", categories)

    def test_read_ltspice_window_text_parses_helper_payload(self) -> None:
        with (
            patch("ltspice_mcp.ltspice.platform.system", return_value="Darwin"),
            patch(
                "ltspice_mcp.ltspice._ensure_ax_text_helper",
                return_value=(Path("/tmp/ltspice-ax-text-helper"), {"helper_source": "test"}),
            ),
            patch(
                "ltspice_mcp.ltspice.subprocess.run",
                return_value=SimpleNamespace(
                    returncode=0,
                    stdout='{"status":"OK","text":"Measurement: v1\\nv1: MAX(v(out))=1","matched_windows":1,"window_title":"demo.log"}\n',
                    stderr="",
                ),
            ) as run_mock,
        ):
            payload = read_ltspice_window_text(title_hint="demo.log", max_chars=2000)

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["status"], "OK")
        self.assertIn("Measurement: v1", payload["text"])
        self.assertEqual(payload["matched_windows"], 1)
        self.assertEqual(payload["window_title"], "demo.log")
        self.assertEqual(
            run_mock.call_args.args[0],
            ["/tmp/ltspice-ax-text-helper", "demo.log", "", "", "2000"],
        )

    def test_read_ltspice_window_text_requires_selector(self) -> None:
        payload = read_ltspice_window_text(title_hint="", exact_title=None, window_id=None)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["status"], "INVALID_SELECTORS")


if __name__ == "__main__":
    unittest.main()
