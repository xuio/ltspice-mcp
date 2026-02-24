from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

from .models import RawDataset, RawStep, RawVariable
from .textio import read_text_auto


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
_STEP_LINE_RE = re.compile(r"^\s*\.step\s+(.+?)\s*$", re.IGNORECASE)


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


def _parse_step_labels_from_log(raw_path: Path) -> list[str]:
    log_path = raw_path.with_suffix(".log")
    if not log_path.exists():
        return []

    labels: list[str] = []
    for raw_line in read_text_auto(log_path).splitlines():
        match = _STEP_LINE_RE.match(raw_line)
        if not match:
            continue
        labels.append(match.group(1).strip())
    return labels


def _detect_step_starts(scale: list[float]) -> list[int]:
    if len(scale) < 2:
        return [0]
    starts = [0]
    for idx in range(1, len(scale)):
        prev = scale[idx - 1]
        curr = scale[idx]
        tol = max(1e-30, abs(prev) * 1e-12, abs(curr) * 1e-12)
        if curr < prev - tol:
            starts.append(idx)
    return starts


def _build_steps(
    *,
    raw_path: Path,
    flags: set[str],
    values: list[list[complex]],
) -> list[RawStep]:
    total_points = len(values[0]) if values else 0
    if total_points <= 0:
        return [RawStep(index=0, start=0, end=0)]

    labels = _parse_step_labels_from_log(raw_path)
    if "stepped" not in flags:
        return [RawStep(index=0, start=0, end=total_points, label=labels[0] if labels else None)]

    starts = _detect_step_starts([value.real for value in values[0]])
    if len(starts) == 1 and labels:
        if total_points == len(labels):
            starts = list(range(total_points))
        elif total_points % len(labels) == 0:
            points_per_step = total_points // len(labels)
            starts = [step_idx * points_per_step for step_idx in range(len(labels))]

    starts = sorted(set(starts))
    if starts[0] != 0:
        starts.insert(0, 0)
    if starts[-1] != total_points:
        starts.append(total_points)
    else:
        starts = starts + [total_points]

    steps: list[RawStep] = []
    for step_idx in range(len(starts) - 1):
        start = starts[step_idx]
        end = starts[step_idx + 1]
        if end <= start:
            continue
        label = labels[step_idx] if step_idx < len(labels) else None
        steps.append(RawStep(index=step_idx, start=start, end=end, label=label))

    if not steps:
        steps = [RawStep(index=0, start=0, end=total_points, label=labels[0] if labels else None)]
    return steps


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
    steps = _build_steps(
        raw_path=raw_path,
        flags=flags,
        values=values,
    )
    return RawDataset(
        path=raw_path,
        plot_name=plot_name,
        flags=flags,
        metadata=metadata,
        variables=variables,
        values=values,
        steps=steps,
    )
