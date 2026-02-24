from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ltspice_mcp import server
from ltspice_mcp.models import SimulationRun


class TestServerStatePersistence(unittest.TestCase):
    def test_run_state_persisted_and_reloaded(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_state_test_"))
        netlist = temp_dir / "a.cir"
        netlist.write_text("* test\n.end\n")

        server._configure_runner(workdir=temp_dir, ltspice_binary=None, timeout=10)
        self.assertEqual(len(server._run_order), 0)

        run = SimulationRun(
            run_id="run-1",
            netlist_path=netlist,
            command=["LTspice", "-b", str(netlist)],
            ltspice_executable=Path("/Applications/LTspice.app/Contents/MacOS/LTspice"),
            started_at="2026-02-24T12:00:00+00:00",
            duration_seconds=0.1,
            return_code=0,
            stdout="",
            stderr="",
            log_path=netlist.with_suffix(".log"),
            raw_files=[netlist.with_suffix(".raw")],
            artifacts=[netlist],
            issues=[],
            warnings=[],
            diagnostics=[],
        )
        server._register_run(run)
        self.assertTrue((temp_dir / ".ltspice_mcp_runs.json").exists())

        server._configure_runner(workdir=temp_dir, ltspice_binary=None, timeout=10)
        self.assertIn("run-1", server._runs)
        self.assertIn("run-1", server._run_order)

    def test_open_schematic_ui_routes_to_single_live_path(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_schematic_ui_test_"))
        source_asc = temp_dir / "source.asc"
        source_asc.write_text("Version 4\nSHEET 1 100 100\n", encoding="utf-8")

        server._configure_runner(workdir=temp_dir, ltspice_binary=None, timeout=10)
        server._schematic_single_window_enabled = True

        original_open = server.open_in_ltspice_ui
        try:
            server.open_in_ltspice_ui = lambda path: {"opened": True, "path": str(Path(path).resolve())}
            event = server._open_schematic_ui(source_asc)
        finally:
            server.open_in_ltspice_ui = original_open

        self.assertTrue(event["opened"])
        self.assertTrue(event["single_window_mode"])
        self.assertTrue(event["routed_to_single_window"])
        self.assertEqual(event["requested_path"], str(source_asc.resolve()))
        self.assertEqual(event["ui_path"], str(server._schematic_live_path))
        self.assertTrue(server._schematic_live_path.exists())
        self.assertEqual(
            server._schematic_live_path.read_text(encoding="utf-8"),
            source_asc.read_text(encoding="utf-8"),
        )

    def test_open_schematic_ui_direct_path_when_single_window_disabled(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_schematic_ui_test_"))
        source_asc = temp_dir / "direct.asc"
        source_asc.write_text("Version 4\nSHEET 1 120 120\n", encoding="utf-8")

        server._configure_runner(workdir=temp_dir, ltspice_binary=None, timeout=10)
        server._schematic_single_window_enabled = False

        original_open = server.open_in_ltspice_ui
        try:
            server.open_in_ltspice_ui = lambda path: {"opened": True, "path": str(Path(path).resolve())}
            event = server._open_schematic_ui(source_asc)
        finally:
            server.open_in_ltspice_ui = original_open

        self.assertTrue(event["opened"])
        self.assertFalse(event["single_window_mode"])
        self.assertFalse(event["routed_to_single_window"])
        self.assertEqual(event["requested_path"], str(source_asc.resolve()))
        self.assertEqual(event["ui_path"], str(source_asc.resolve()))

    def test_load_circuit_generates_schematic(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_load_circuit_test_"))
        server._configure_runner(workdir=temp_dir, ltspice_binary=None, timeout=10)

        result = server.loadCircuit(
            netlist=(
                "* RC low-pass\n"
                "V1 in 0 AC 1\n"
                "R1 in out 1k\n"
                "C1 out 0 100n\n"
                ".ac dec 20 10 1e6\n"
                ".end\n"
            ),
            circuit_name="rc_load_case",
        )

        self.assertTrue(result["loaded"])
        self.assertIn("asc_path", result)
        asc_path = Path(result["asc_path"])
        self.assertTrue(asc_path.exists())
        text = asc_path.read_text(encoding="utf-8")
        self.assertIn("SYMBOL res", text)
        self.assertIn("SYMBOL cap", text)


if __name__ == "__main__":
    unittest.main()
