from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ltspice_mcp import server
from ltspice_mcp.models import SimulationRun


class TestSchematicRuntimeTools(unittest.TestCase):
    def test_validate_schematic_reports_missing_directive(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_validate_schematic_test_"))
        asc_path = temp_dir / "invalid.asc"
        asc_path.write_text(
            "Version 4\n"
            "SHEET 1 880 680\n"
            "SYMBOL res 120 120 R0\n"
            "SYMATTR InstName R1\n"
            "SYMATTR Value 1k\n"
            "FLAG 120 216 0\n",
            encoding="utf-8",
        )

        payload = server.validateSchematic(str(asc_path))
        self.assertEqual(payload["asc_path"], str(asc_path.resolve()))
        self.assertFalse(payload["valid"])
        self.assertTrue(any("simulation directive" in issue.lower() for issue in payload["issues"]))

    def test_simulate_schematic_file_preflight_and_run(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_simulate_schematic_test_"))
        asc_path = temp_dir / "valid.asc"
        sidecar = asc_path.with_suffix(".cir")
        sidecar.write_text("* sidecar\n.op\n.end\n", encoding="utf-8")
        asc_path.write_text(
            "Version 4\n"
            "SHEET 1 880 680\n"
            "SYMBOL voltage 120 120 R0\n"
            "SYMATTR InstName V1\n"
            "SYMATTR Value DC 1\n"
            "FLAG 120 216 0\n"
            "TEXT 48 560 Left 2 !.op\n",
            encoding="utf-8",
        )

        fake_run = SimulationRun(
            run_id="run-schematic-1",
            netlist_path=sidecar.resolve(),
            command=["LTspice", "-b", str(sidecar.resolve())],
            ltspice_executable=Path("/Applications/LTspice.app/Contents/MacOS/LTspice"),
            started_at="2026-02-25T00:00:00+00:00",
            duration_seconds=0.1,
            return_code=0,
            stdout="",
            stderr="",
            log_path=sidecar.with_suffix(".log"),
            raw_files=[sidecar.with_suffix(".raw")],
            artifacts=[sidecar.resolve()],
            issues=[],
            warnings=[],
            diagnostics=[],
        )

        server._configure_runner(workdir=temp_dir, ltspice_binary=None, timeout=10)

        with patch("ltspice_mcp.server._run_simulation_with_ui") as run_mock:
            run_mock.return_value = (fake_run, [], False)
            payload = server.simulateSchematicFile(
                asc_path=str(asc_path),
                validate_first=True,
                abort_on_validation_error=False,
            )

        self.assertEqual(payload["run_id"], "run-schematic-1")
        self.assertEqual(payload["schematic_path"], str(asc_path.resolve()))
        self.assertEqual(payload["run_target_path"], str(sidecar.resolve()))
        self.assertTrue(payload["used_sidecar_netlist"])
        self.assertIn("schematic_validation", payload)
        self.assertTrue(payload["schematic_validation"]["valid"])
        run_mock.assert_called_once()
        self.assertEqual(run_mock.call_args.kwargs["netlist_path"], sidecar.resolve())

    def test_simulate_schematic_file_can_abort_on_invalid_preflight(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_simulate_schematic_abort_test_"))
        asc_path = temp_dir / "invalid.asc"
        asc_path.write_text(
            "Version 4\n"
            "SHEET 1 880 680\n"
            "SYMBOL res 120 120 R0\n"
            "SYMATTR InstName R1\n"
            "SYMATTR Value 1k\n",
            encoding="utf-8",
        )

        server._configure_runner(workdir=temp_dir, ltspice_binary=None, timeout=10)
        with self.assertRaises(ValueError):
            server.simulateSchematicFile(
                asc_path=str(asc_path),
                validate_first=True,
                abort_on_validation_error=True,
            )

    def test_simulate_schematic_file_requires_sidecar_on_macos(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_simulate_schematic_macos_test_"))
        asc_path = temp_dir / "no_sidecar.asc"
        asc_path.write_text(
            "Version 4\n"
            "SHEET 1 880 680\n"
            "SYMBOL voltage 120 120 R0\n"
            "SYMATTR InstName V1\n"
            "SYMATTR Value DC 1\n"
            "FLAG 120 216 0\n"
            "TEXT 48 560 Left 2 !.op\n",
            encoding="utf-8",
        )

        server._configure_runner(workdir=temp_dir, ltspice_binary=None, timeout=10)
        with (
            patch("ltspice_mcp.server.platform.system", return_value="Darwin"),
            patch("ltspice_mcp.server._run_simulation_with_ui") as run_mock,
        ):
            with self.assertRaisesRegex(ValueError, "does not support LTspice batch simulation directly from \\.asc"):
                server.simulateSchematicFile(
                    asc_path=str(asc_path),
                    validate_first=True,
                    abort_on_validation_error=False,
                )
            run_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
