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


if __name__ == "__main__":
    unittest.main()
