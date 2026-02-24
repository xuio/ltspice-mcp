from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from ltspice_mcp.models import RawDataset, RawVariable
from ltspice_mcp.schematic import SymbolLibrary
from ltspice_mcp.visualization import render_plot_svg, render_schematic_svg, render_symbol_svg


OPAMP2_ASY = """Version 4
SymbolType CELL
LINE Normal -32 32 32 64
LINE Normal -32 96 32 64
LINE Normal -32 32 -32 96
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


class TestVisualization(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_visual_test_"))
        self.zip_path = self.temp_dir / "lib.zip"
        with ZipFile(self.zip_path, "w") as archive:
            archive.writestr("lib/sym/OpAmps/opamp2.asy", OPAMP2_ASY)

    def test_render_symbol_svg(self) -> None:
        library = SymbolLibrary(self.zip_path)
        try:
            result = render_symbol_svg(
                workdir=self.temp_dir,
                symbol="opamp2",
                width=640,
                height=420,
                downscale_factor=0.5,
                library=library,
            )
        finally:
            library.close()

        image_path = Path(result["image_path"])
        self.assertTrue(image_path.exists())
        self.assertEqual(result["format"], "svg")
        self.assertEqual(result["width"], 320)
        self.assertEqual(result["height"], 210)
        text = image_path.read_text(encoding="utf-8")
        self.assertIn("<svg", text)
        self.assertIn("symbol: opamp2", text)

    def test_render_schematic_svg(self) -> None:
        asc_path = self.temp_dir / "amp.asc"
        asc_path.write_text(
            "Version 4\n"
            "SHEET 1 880 680\n"
            "WIRE 168 240 168 120\n"
            "FLAG 168 240 0\n"
            "SYMBOL opamp2 240 120 R0\n"
            "SYMATTR InstName U1\n"
            "SYMATTR Value opamp2\n"
            "TEXT 48 560 Left 2 !.op\n",
            encoding="utf-8",
        )

        library = SymbolLibrary(self.zip_path)
        try:
            result = render_schematic_svg(
                workdir=self.temp_dir,
                asc_path=asc_path,
                width=1200,
                height=900,
                downscale_factor=0.5,
                library=library,
            )
        finally:
            library.close()

        image_path = Path(result["image_path"])
        self.assertTrue(image_path.exists())
        self.assertEqual(result["width"], 600)
        self.assertEqual(result["height"], 450)
        text = image_path.read_text(encoding="utf-8")
        self.assertIn("<svg", text)
        self.assertIn("U1", text)

    def test_render_plot_svg(self) -> None:
        dataset = RawDataset(
            path=self.temp_dir / "example.raw",
            plot_name="Transient Analysis",
            flags=set(),
            metadata={},
            variables=[
                RawVariable(index=0, name="time", kind="time"),
                RawVariable(index=1, name="V(out)", kind="voltage"),
            ],
            values=[
                [complex(0.0), complex(1.0), complex(2.0), complex(3.0)],
                [complex(0.0), complex(1.0), complex(0.5), complex(1.5)],
            ],
        )

        result = render_plot_svg(
            workdir=self.temp_dir,
            dataset=dataset,
            vectors=["V(out)"],
            width=1000,
            height=600,
            downscale_factor=0.5,
            y_mode="real",
        )
        image_path = Path(result["image_path"])
        self.assertTrue(image_path.exists())
        self.assertEqual(result["width"], 500)
        self.assertEqual(result["height"], 300)
        text = image_path.read_text(encoding="utf-8")
        self.assertIn("<polyline", text)


if __name__ == "__main__":
    unittest.main()
