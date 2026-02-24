from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

from .models import RawDataset, RawVariable


class RawParseError(ValueError):
    pass


@dataclass(slots=True)
class _SectionMarker:
    index: int
    kind: str
    encoding: str
    token_len: int


_INT_RE = re.compile(r"-?\d+")
_SCALAR_COMPLEX_RE = re.compile(r"^\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)$")


def _find_section_marker(blob: bytes) -> _SectionMarker:
    candidates: list[_SectionMarker] = []
    for label, kind in (("Binary", "binary"), ("Values", "values")):
        for encoding in ("utf-16le", "utf-8"):
            for suffix in ("\r\n", "\n"):
                token = f"{label}:{suffix}".encode(encoding)
                index = blob.find(token)
                if index != -1:
                    candidates.append(
                        _SectionMarker(
                            index=index,
                            kind=kind,
                            encoding=encoding,
                            token_len=len(token),
                        )
                    )
    if not candidates:
        raise RawParseError("Could not find Binary:/Values: section marker in .raw file.")
    return min(candidates, key=lambda marker: marker.index)


def _parse_header(header: str) -> tuple[dict[str, str], list[RawVariable]]:
    metadata: dict[str, str] = {}
    variables: list[RawVariable] = []
    in_variables = False

    for raw_line in header.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower() == "variables:":
            in_variables = True
            continue
        if in_variables:
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                index = int(parts[0])
                name = parts[1]
                kind = " ".join(parts[2:])
                variables.append(RawVariable(index=index, name=name, kind=kind))
                continue
        if ":" in raw_line:
            key, value = raw_line.split(":", 1)
            metadata[key.strip()] = value.strip()

    return metadata, variables


def _read_required_int(metadata: dict[str, str], key: str) -> int:
    value = metadata.get(key)
    if value is None:
        raise RawParseError(f"Missing '{key}' field in RAW header.")
    match = _INT_RE.search(value.replace(",", ""))
    if not match:
        raise RawParseError(f"Could not parse integer from '{key}: {value}'.")
    return int(match.group(0))


def _parse_scalar(token: str) -> complex:
    cleaned = token.strip().rstrip(",")
    complex_match = _SCALAR_COMPLEX_RE.match(cleaned)
    if complex_match:
        return complex(float(complex_match.group(1)), float(complex_match.group(2)))
    return complex(float(cleaned), 0.0)


def _expected_binary_bytes(num_vars: int, num_points: int, complex_mode: bool) -> int:
    if num_vars <= 0 or num_points <= 0:
        return 0
    if complex_mode:
        return num_vars * num_points * 16
    return num_points * (8 + max(num_vars - 1, 0) * 4)


def _parse_binary_values(
    payload: bytes,
    num_vars: int,
    num_points: int,
    complex_mode: bool,
    fast_access: bool,
) -> list[list[complex]]:
    values: list[list[complex]] = [[0j for _ in range(num_points)] for _ in range(num_vars)]
    expected = _expected_binary_bytes(num_vars, num_points, complex_mode)
    if len(payload) < expected:
        raise RawParseError(
            f"Binary payload too short ({len(payload)} bytes, expected at least {expected} bytes)."
        )

    offset = 0
    if complex_mode:
        if fast_access:
            for var_idx in range(num_vars):
                for point_idx in range(num_points):
                    real, imag = struct.unpack_from("<dd", payload, offset)
                    values[var_idx][point_idx] = complex(real, imag)
                    offset += 16
        else:
            for point_idx in range(num_points):
                for var_idx in range(num_vars):
                    real, imag = struct.unpack_from("<dd", payload, offset)
                    values[var_idx][point_idx] = complex(real, imag)
                    offset += 16
        return values

    if fast_access:
        for point_idx in range(num_points):
            real = struct.unpack_from("<d", payload, offset)[0]
            values[0][point_idx] = complex(real, 0.0)
            offset += 8
        for var_idx in range(1, num_vars):
            for point_idx in range(num_points):
                real = struct.unpack_from("<f", payload, offset)[0]
                values[var_idx][point_idx] = complex(real, 0.0)
                offset += 4
        return values

    for point_idx in range(num_points):
        real = struct.unpack_from("<d", payload, offset)[0]
        values[0][point_idx] = complex(real, 0.0)
        offset += 8
        for var_idx in range(1, num_vars):
            real = struct.unpack_from("<f", payload, offset)[0]
            values[var_idx][point_idx] = complex(real, 0.0)
            offset += 4
    return values


def _parse_values_section(section_text: str, num_vars: int, num_points: int) -> list[list[complex]]:
    values: list[list[complex]] = [[0j for _ in range(num_points)] for _ in range(num_vars)]
    raw_lines = [line.strip() for line in section_text.splitlines() if line.strip()]

    cursor = 0
    for point_idx in range(num_points):
        if cursor >= len(raw_lines):
            raise RawParseError("ASCII RAW has fewer rows than expected.")
        head_parts = raw_lines[cursor].split()
        if not head_parts:
            raise RawParseError("Malformed ASCII RAW row.")
        if len(head_parts) == 1:
            first_token = head_parts[0]
        else:
            try:
                int(head_parts[0])
                first_token = head_parts[1]
            except ValueError:
                first_token = head_parts[0]
        values[0][point_idx] = _parse_scalar(first_token)
        cursor += 1

        for var_idx in range(1, num_vars):
            if cursor >= len(raw_lines):
                raise RawParseError("ASCII RAW has fewer vector entries than expected.")
            token = raw_lines[cursor].split()[0]
            values[var_idx][point_idx] = _parse_scalar(token)
            cursor += 1

    return values


def parse_raw_file(path: str | Path) -> RawDataset:
    raw_path = Path(path).expanduser().resolve()
    blob = raw_path.read_bytes()

    marker = _find_section_marker(blob)
    header_text = blob[: marker.index].decode(marker.encoding, errors="replace")
    metadata, variables = _parse_header(header_text)

    num_vars = _read_required_int(metadata, "No. Variables")
    num_points = _read_required_int(metadata, "No. Points")
    if num_vars < 0 or num_points < 0:
        raise RawParseError("RAW header has negative vector or point counts.")

    if len(variables) != num_vars:
        # Fallback if formatting differed and variable parsing missed lines.
        variables = [
            RawVariable(index=index, name=f"var{index}", kind="unknown")
            for index in range(num_vars)
        ]

    flags = {part.lower() for part in metadata.get("Flags", "").split()}
    payload = blob[marker.index + marker.token_len :]

    if marker.kind == "binary":
        values = _parse_binary_values(
            payload=payload,
            num_vars=num_vars,
            num_points=num_points,
            complex_mode="complex" in flags,
            fast_access="fastaccess" in flags,
        )
    else:
        section_text = payload.decode(marker.encoding, errors="replace")
        values = _parse_values_section(section_text, num_vars=num_vars, num_points=num_points)

    plot_name = metadata.get("Plotname", "Unknown Plot")
    return RawDataset(
        path=raw_path,
        plot_name=plot_name,
        flags=flags,
        metadata=metadata,
        variables=variables,
        values=values,
    )
