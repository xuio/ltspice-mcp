from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from ltspice_mcp.schematic import SymbolLibrary


OPAMP2_ASY = """Version 4
SymbolType CELL
SYMATTR Value opamp2
PIN -32 80 NONE 0
PINATTR PinName In+
PINATTR SpiceOrder 1
PIN -32 48 NONE 0
PINATTR PinName In-
PINATTR SpiceOrder 2
PIN 0 32 NONE 0
PINATTR PinName V+
PINATTR SpiceOrder 3
PIN 0 96 NONE 0
PINATTR PinName V-
PINATTR SpiceOrder 4
PIN 32 64 NONE 0
PINATTR PinName OUT
PINATTR SpiceOrder 5
"""

RES_ASY = """Version 4
SymbolType CELL
SYMATTR Value res
PIN 0 0 NONE 0
PINATTR PinName A
PINATTR SpiceOrder 1
PIN 0 80 NONE 0
PINATTR PinName B
PINATTR SpiceOrder 2
"""


class TestSymbolLibrary(unittest.TestCase):
    def setUp(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_symbol_lib_test_"))
        self.zip_path = temp_dir / "lib.zip"
        with ZipFile(self.zip_path, "w") as archive:
            archive.writestr("lib/sym/OpAmps/opamp2.asy", OPAMP2_ASY)
            archive.writestr("lib/sym/Passives/res.asy", RES_ASY)
            archive.writestr("README.txt", "not a symbol")

    def test_list_entries_and_symbols(self) -> None:
        library = SymbolLibrary(self.zip_path)
        entries = library.list_entries(limit=10)
        self.assertEqual(len(entries), 2)
        self.assertIn("lib/sym/OpAmps/opamp2.asy", entries)

        filtered_entries = library.list_entries(query="opamp2", limit=10)
        self.assertEqual(filtered_entries, ["lib/sym/OpAmps/opamp2.asy"])

        symbols = library.list_symbols(query="opamp", limit=10)
        self.assertEqual(len(symbols), 1)
        self.assertEqual(symbols[0]["symbol"], "opamp2")
        self.assertEqual(symbols[0]["category"], "OpAmps")

    def test_symbol_info_and_source(self) -> None:
        library = SymbolLibrary(self.zip_path)
        info = library.symbol_info("opamp2")
        self.assertEqual(info["zip_entry"], "lib/sym/OpAmps/opamp2.asy")
        self.assertEqual(info["pin_count"], 5)
        self.assertEqual(info["pins"][0]["spice_order"], 1)
        self.assertEqual(info["pins"][0]["name"], "In+")

        source = library.read_symbol_source("opamp2")
        self.assertIn("SYMATTR Value opamp2", source)


if __name__ == "__main__":
    unittest.main()
