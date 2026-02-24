from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ltspice_mcp.schematic import (
    PinDef,
    SymbolLibrary,
    SymbolDef,
    _transform_point,
    build_schematic_from_template,
    build_schematic_from_netlist,
    build_schematic_from_spec,
    list_schematic_templates,
    sync_schematic_from_netlist_file,
    watch_schematic_from_netlist_file,
)


class _StubLibrary:
    def __init__(self) -> None:
        self._symbol = SymbolDef(
            symbol="stub",
            zip_entry="stub.asy",
            pins=[
                PinDef(x=0, y=0, spice_order=1, name="A"),
                PinDef(x=0, y=80, spice_order=2, name="B"),
            ],
        )

    def get(self, symbol: str) -> SymbolDef:
        return SymbolDef(
            symbol=symbol,
            zip_entry=f"{symbol}.asy",
            pins=list(self._symbol.pins),
        )

    def pin_offset(self, symbol: str, orientation: str, spice_order: int) -> tuple[int, int]:
        _ = symbol
        pin = self.get("stub").pin_for_order(spice_order)
        if pin is None:
            raise ValueError(f"Unsupported spice_order {spice_order}")
        return _transform_point(pin.x, pin.y, orientation)


class TestSchematicBuilders(unittest.TestCase):
    @staticmethod
    def _fixtures_dir() -> Path:
        return (Path(__file__).resolve().parent / "fixtures" / "schematic").resolve()

    @classmethod
    def _fixture_text(cls, name: str) -> str:
        return (cls._fixtures_dir() / name).read_text(encoding="utf-8")

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_schematic_test_"))
        self.stub = _StubLibrary()

    def test_build_schematic_from_spec(self) -> None:
        output_path = self.temp_dir / "spec_case.asc"
        result = build_schematic_from_spec(
            workdir=self.temp_dir,
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
            output_path=str(output_path),
            library=self.stub,
        )
        asc = Path(result["asc_path"])
        self.assertTrue(asc.exists())
        text = asc.read_text(encoding="utf-8")
        self.assertEqual(text, self._fixture_text("spec_case.asc"))

    def test_build_schematic_from_netlist(self) -> None:
        output_path = self.temp_dir / "netlist_case.asc"
        result = build_schematic_from_netlist(
            workdir=self.temp_dir,
            netlist_content=(
                "* RC netlist\n"
                "V1 in 0 AC 1\n"
                "R1 in out 1k\n"
                "C1 out 0 1u\n"
                ".ac dec 10 10 10k\n"
                ".end\n"
            ),
            circuit_name="netlist_case",
            output_path=str(output_path),
            library=self.stub,
        )
        asc = Path(result["asc_path"])
        self.assertTrue(asc.exists())
        text = asc.read_text(encoding="utf-8")
        self.assertEqual(text, self._fixture_text("netlist_case.asc"))

    def test_template_listing_and_rendering(self) -> None:
        listing = list_schematic_templates()
        names = {entry["name"] for entry in listing["templates"]}
        self.assertIn("rc_lowpass_ac", names)
        self.assertIn("non_inverting_opamp_spec", names)

        output_path = self.temp_dir / "template_case.asc"
        result = build_schematic_from_template(
            workdir=self.temp_dir,
            template_name="rc_lowpass_ac",
            parameters={
                "vin_ac": "2",
                "r_value": "2k",
                "c_value": "2u",
                "ac_points": "10",
                "f_start": "20",
                "f_stop": "20k",
            },
            circuit_name="template_case",
            output_path=str(output_path),
            library=self.stub,
        )
        self.assertEqual(result["template_name"], "rc_lowpass_ac")
        text = Path(result["asc_path"]).read_text(encoding="utf-8")
        self.assertIn("SYMATTR Value AC 2", text)
        self.assertIn("SYMATTR Value 2k", text)
        self.assertIn("SYMATTR Value 2u", text)

    def test_non_inverting_opamp_template_rendering(self) -> None:
        output_path = self.temp_dir / "non_inverting_template.asc"
        result = build_schematic_from_template(
            workdir=self.temp_dir,
            template_name="non_inverting_opamp_spec",
            parameters={
                "vin_signal": "SINE(0 0.05 2k) AC 1",
                "rf_value": "22k",
                "rg_value": "2k",
                "vplus": "12",
                "vminus": "-12",
            },
            circuit_name="non_inverting_template",
            output_path=str(output_path),
        )
        text = Path(result["asc_path"]).read_text(encoding="utf-8")
        self.assertIn("SYMBOL UniversalOpAmp2", text)
        self.assertIn("SYMATTR Value 22k", text)
        self.assertIn("SYMATTR Value 2k", text)
        self.assertIn("SYMATTR Value 12", text)
        self.assertIn("SYMATTR Value -12", text)

    def test_sync_regenerates_only_on_change(self) -> None:
        netlist = self.temp_dir / "sync_case.cir"
        netlist.write_text(
            "* Sync test\n"
            "V1 in 0 AC 1\n"
            "R1 in out 1k\n"
            "C1 out 0 1u\n"
            ".ac dec 10 10 10k\n"
            ".end\n",
            encoding="utf-8",
        )

        first = sync_schematic_from_netlist_file(
            workdir=self.temp_dir,
            netlist_path=netlist,
            circuit_name="sync_case",
            library=self.stub,
        )
        self.assertTrue(first["updated"])
        self.assertIn(first["reason"], {"missing_output", "source_changed", "forced"})

        second = sync_schematic_from_netlist_file(
            workdir=self.temp_dir,
            netlist_path=netlist,
            circuit_name="sync_case",
            library=self.stub,
        )
        self.assertFalse(second["updated"])
        self.assertEqual(second["reason"], "unchanged")

        netlist.write_text(netlist.read_text(encoding="utf-8") + "\n* changed\n", encoding="utf-8")
        third = sync_schematic_from_netlist_file(
            workdir=self.temp_dir,
            netlist_path=netlist,
            circuit_name="sync_case",
            library=self.stub,
        )
        self.assertTrue(third["updated"])
        self.assertEqual(third["reason"], "source_changed")

    def test_watch_netlist_file(self) -> None:
        netlist = self.temp_dir / "watch_case.cir"
        netlist.write_text(
            "* Watch test\n"
            "V1 in 0 AC 1\n"
            "R1 in out 1k\n"
            ".ac dec 10 10 10k\n"
            ".end\n",
            encoding="utf-8",
        )
        result = watch_schematic_from_netlist_file(
            workdir=self.temp_dir,
            netlist_path=netlist,
            circuit_name="watch_case",
            duration_seconds=0.0,
            poll_interval_seconds=0.01,
            max_updates=2,
            force_initial_refresh=True,
            library=self.stub,
        )
        self.assertEqual(result["poll_count"], 1)
        self.assertEqual(result["updates_count"], 1)
        self.assertTrue(result["updates"][0]["updated"])

    def test_real_symbol_rc_lowpass_capacitor_connectivity(self) -> None:
        try:
            lib = SymbolLibrary()
        except FileNotFoundError:
            self.skipTest("LTspice lib.zip not available on this machine.")

        output_path = self.temp_dir / "real_symbol_rc.asc"
        result = build_schematic_from_netlist(
            workdir=self.temp_dir,
            netlist_content=(
                "* RC low-pass\n"
                "V1 in 0 AC 1\n"
                "R1 in out 1k\n"
                "C1 out 0 100n\n"
                ".ac dec 30 10 1e6\n"
                ".op\n"
                ".end\n"
            ),
            circuit_name="real_symbol_rc",
            output_path=str(output_path),
            library=lib,
        )
        asc = Path(result["asc_path"])
        text = asc.read_text(encoding="utf-8")

        cap_x = cap_y = None
        cap_orientation = None
        wires: list[tuple[int, int, int, int]] = []
        flags: list[tuple[int, int, str]] = []

        for raw in text.splitlines():
            parts = raw.split()
            if not parts:
                continue
            if parts[0] == "SYMBOL" and len(parts) >= 5 and parts[1].lower() == "cap":
                cap_x = int(parts[2])
                cap_y = int(parts[3])
                cap_orientation = parts[4]
            elif parts[0] == "WIRE" and len(parts) >= 5:
                wires.append((int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])))
            elif parts[0] == "FLAG" and len(parts) >= 4:
                flags.append((int(parts[1]), int(parts[2]), parts[3]))

        self.assertIsNotNone(cap_x, "Capacitor symbol not found in generated schematic.")
        self.assertIsNotNone(cap_y, "Capacitor symbol not found in generated schematic.")
        self.assertEqual(cap_orientation, "R0", "RC low-pass capacitor should be vertical to ground.")

        pin1_dx, pin1_dy = lib.pin_offset("cap", orientation=cap_orientation, spice_order=1)
        pin2_dx, pin2_dy = lib.pin_offset("cap", orientation=cap_orientation, spice_order=2)
        cap_pin1 = (cap_x + pin1_dx, cap_y + pin1_dy)
        cap_pin2 = (cap_x + pin2_dx, cap_y + pin2_dy)

        self.assertTrue(
            any((x1, y1) == cap_pin1 or (x2, y2) == cap_pin1 for x1, y1, x2, y2 in wires),
            "Capacitor signal pin is not connected to any wire.",
        )
        self.assertIn(
            (cap_pin2[0], cap_pin2[1], "0"),
            flags,
            "Capacitor ground pin is not tied to net 0.",
        )


if __name__ == "__main__":
    unittest.main()
