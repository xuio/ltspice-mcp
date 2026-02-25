from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ltspice_mcp.ltspice import _write_utf8_log_sidecar, analyze_log


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


if __name__ == "__main__":
    unittest.main()
