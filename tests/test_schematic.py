from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ltspice_mcp.schematic import (
    PinDef,
    SymbolDef,
    build_schematic_from_netlist,
    build_schematic_from_spec,
)


class _StubLibrary:
    def get(self, symbol: str) -> SymbolDef:
        # Two-pin vertical primitive for deterministic pin positions in tests.
        return SymbolDef(
            symbol=symbol,
            zip_entry=f"{symbol}.asy",
            pins=[
                PinDef(x=0, y=0, spice_order=1, name="A"),
                PinDef(x=0, y=80, spice_order=2, name="B"),
            ],
        )


class TestSchematicBuilders(unittest.TestCase):
    def test_build_schematic_from_spec(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_schematic_spec_"))
        result = build_schematic_from_spec(
            workdir=temp_dir,
            components=[
                {
                    "symbol": "res",
                    "reference": "R1",
                    "x": 120,
                    "y": 160,
                    "value": "1k",
                    "pin_nets": {"1": "in", "2": "out"},
                }
            ],
            directives=[".op"],
            circuit_name="spec_case",
            library=_StubLibrary(),
        )
        asc = Path(result["asc_path"])
        self.assertTrue(asc.exists())
        text = asc.read_text(encoding="utf-8")
        self.assertIn("SYMBOL res 120 160 R0", text)
        self.assertIn("SYMATTR InstName R1", text)
        self.assertIn("FLAG 120 160 in", text)
        self.assertIn("FLAG 120 240 out", text)
        self.assertIn("TEXT", text)

    def test_build_schematic_from_netlist(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_schematic_netlist_"))
        result = build_schematic_from_netlist(
            workdir=temp_dir,
            netlist_content=(
                "* RC netlist\n"
                "V1 in 0 AC 1\n"
                "R1 in out 1k\n"
                "C1 out 0 1u\n"
                ".ac dec 10 10 10k\n"
                ".end\n"
            ),
            circuit_name="netlist_case",
            library=_StubLibrary(),
        )
        asc = Path(result["asc_path"])
        self.assertTrue(asc.exists())
        text = asc.read_text(encoding="utf-8")
        self.assertIn("SYMBOL voltage", text)
        self.assertIn("SYMBOL res", text)
        self.assertIn("SYMBOL cap", text)
        self.assertIn("FLAG", text)
        self.assertIn("WIRE", text)
        self.assertIn("!.ac dec 10 10 10k", text)


if __name__ == "__main__":
    unittest.main()
