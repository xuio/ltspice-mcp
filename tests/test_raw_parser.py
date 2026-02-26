from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from ltspice_mcp.raw_parser import parse_raw_file


def _build_header(
    *,
    plot: str,
    flags: str,
    variables: list[tuple[str, str]],
    points: int,
    encoding: str,
) -> bytes:
    lines = [
        "Title: test.cir",
        "Date: Tue Feb 24 10:00:00 2026",
        f"Plotname: {plot}",
        f"Flags: {flags}",
        f"No. Variables: {len(variables)}",
        f"No. Points: {points}",
        "Offset: 0.0",
        "Command: LTspice",
        "Variables:",
    ]
    for idx, (name, kind) in enumerate(variables):
        lines.append(f"\t{idx}\t{name}\t{kind}")
    text = "\n".join(lines) + "\n"
    return text.encode(encoding)


def _write_blob(blob: bytes) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_raw_test_"))
    path = temp_dir / "test.raw"
    path.write_bytes(blob)
    return path


class TestRawParser(unittest.TestCase):
    def test_parse_binary_real(self) -> None:
        variables = [("time", "time"), ("V(out)", "voltage"), ("I(R1)", "device_current")]
        points = [
            (0.0, 1.0, -1.0),
            (1.0, 1.5, -0.5),
        ]
        payload = bytearray()
        for row in points:
            payload.extend(struct.pack("<d", row[0]))
            payload.extend(struct.pack("<f", row[1]))
            payload.extend(struct.pack("<f", row[2]))

        blob = (
            _build_header(
                plot="Transient Analysis",
                flags="real forward",
                variables=variables,
                points=len(points),
                encoding="utf-16le",
            )
            + "Binary:\n".encode("utf-16le")
            + bytes(payload)
        )
        path = _write_blob(blob)
        data = parse_raw_file(path)

        self.assertEqual(data.plot_name, "Transient Analysis")
        self.assertEqual(len(data.variables), 3)
        self.assertAlmostEqual(data.get_vector("time")[1].real, 1.0)
        self.assertAlmostEqual(data.get_vector("V(out)")[1].real, 1.5, places=6)
        self.assertAlmostEqual(data.get_vector("I(R1)")[0].real, -1.0, places=6)
        self.assertAlmostEqual(data.get_vector("v(out)")[1].real, 1.5, places=6)
        self.assertAlmostEqual(data.get_vector(" i(r1) ")[0].real, -1.0, places=6)

    def test_parse_binary_real_normalizes_sign_encoded_time_axis(self) -> None:
        variables = [("time", "time"), ("V(out)", "voltage")]
        points = [
            (0.0, 0.0),
            (1e-9, 0.1),
            (-0.001, 0.2),
            (0.002, 0.3),
            (-0.003, 0.4),
            (0.004, 0.5),
        ]
        payload = bytearray()
        for row in points:
            payload.extend(struct.pack("<d", row[0]))
            payload.extend(struct.pack("<f", row[1]))

        blob = (
            _build_header(
                plot="Transient Analysis",
                flags="real forward",
                variables=variables,
                points=len(points),
                encoding="utf-16le",
            )
            + "Binary:\n".encode("utf-16le")
            + bytes(payload)
        )
        path = _write_blob(blob)
        data = parse_raw_file(path)

        time_values = [value.real for value in data.get_vector("time")]
        self.assertEqual(time_values, [0.0, 1e-9, 0.001, 0.002, 0.003, 0.004])
        self.assertTrue(all(time_values[idx] >= time_values[idx - 1] for idx in range(1, len(time_values))))

    def test_parse_binary_complex(self) -> None:
        variables = [("frequency", "frequency"), ("V(out)", "voltage")]
        points = [
            (complex(10.0, 0.0), complex(0.0, -1.0)),
            (complex(20.0, 0.0), complex(0.5, -0.25)),
        ]
        payload = bytearray()
        for row in points:
            for value in row:
                payload.extend(struct.pack("<dd", value.real, value.imag))

        blob = (
            _build_header(
                plot="AC Analysis",
                flags="complex forward log",
                variables=variables,
                points=len(points),
                encoding="utf-16le",
            )
            + "Binary:\n".encode("utf-16le")
            + bytes(payload)
        )
        path = _write_blob(blob)
        data = parse_raw_file(path)

        self.assertIn("complex", data.flags)
        self.assertAlmostEqual(data.get_vector("frequency")[1].real, 20.0, places=8)
        self.assertAlmostEqual(data.get_vector("V(out)")[0].imag, -1.0, places=8)

    def test_parse_binary_fastaccess_real(self) -> None:
        variables = [("time", "time"), ("V(a)", "voltage"), ("V(b)", "voltage")]
        time_col = [0.0, 1.0, 2.0]
        va_col = [0.0, 0.5, 1.0]
        vb_col = [5.0, 4.5, 4.0]
        payload = bytearray()
        for value in time_col:
            payload.extend(struct.pack("<d", value))
        for value in va_col:
            payload.extend(struct.pack("<f", value))
        for value in vb_col:
            payload.extend(struct.pack("<f", value))

        blob = (
            _build_header(
                plot="Transient Analysis",
                flags="real forward fastaccess",
                variables=variables,
                points=3,
                encoding="utf-16le",
            )
            + "Binary:\n".encode("utf-16le")
            + bytes(payload)
        )
        path = _write_blob(blob)
        data = parse_raw_file(path)

        self.assertIn("fastaccess", data.flags)
        self.assertEqual(data.points, 3)
        self.assertAlmostEqual(data.get_vector("V(a)")[2].real, 1.0, places=6)
        self.assertAlmostEqual(data.get_vector("V(b)")[0].real, 5.0, places=6)

    def test_parse_ascii_values(self) -> None:
        variables = [("time", "time"), ("V(out)", "voltage")]
        values = "\n".join(
            [
                "0\t0.0",
                "\t1.0",
                "1\t1.0",
                "\t2.0",
            ]
        )
        blob = (
            _build_header(
                plot="Transient Analysis",
                flags="real",
                variables=variables,
                points=2,
                encoding="utf-8",
            )
            + "Values:\n".encode("utf-8")
            + values.encode("utf-8")
        )
        path = _write_blob(blob)
        data = parse_raw_file(path)
        self.assertEqual(data.points, 2)
        self.assertAlmostEqual(data.get_vector("time")[1].real, 1.0, places=6)
        self.assertAlmostEqual(data.get_vector("V(out)")[1].real, 2.0, places=6)

    def test_parse_stepped_data_and_labels(self) -> None:
        variables = [("time", "time"), ("V(out)", "voltage")]
        points = [
            (0.0, 0.0),
            (1.0, 1.0),
            (2.0, 2.0),
            (0.0, 10.0),
            (1.0, 11.0),
            (2.0, 12.0),
        ]
        payload = bytearray()
        for row in points:
            payload.extend(struct.pack("<d", row[0]))
            payload.extend(struct.pack("<f", row[1]))

        blob = (
            _build_header(
                plot="Transient Analysis",
                flags="real forward stepped",
                variables=variables,
                points=len(points),
                encoding="utf-16le",
            )
            + "Binary:\n".encode("utf-16le")
            + bytes(payload)
        )
        path = _write_blob(blob)
        path.with_suffix(".log").write_text(
            "LTspice log\n.step rval=1000\n.step rval=2000\n",
            encoding="utf-16le",
        )
        data = parse_raw_file(path)

        self.assertEqual(data.step_count, 2)
        self.assertEqual(data.steps[0].label, "rval=1000")
        self.assertEqual(data.steps[1].label, "rval=2000")
        self.assertEqual(len(data.scale_values(step_index=0)), 3)
        self.assertAlmostEqual(data.get_vector("V(out)", step_index=1)[0].real, 10.0, places=6)


if __name__ == "__main__":
    unittest.main()
