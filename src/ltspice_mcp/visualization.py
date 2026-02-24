from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .analysis import sample_indices
from .models import RawDataset
from .schematic import SymbolLibrary, _normalize_orientation, _sanitize_name, _transform_point


@dataclass(slots=True)
class _Primitive:
    kind: str
    coords: tuple[int, ...]


@dataclass(slots=True)
class _Pin:
    x: int
    y: int
    name: str | None = None
    spice_order: int | None = None


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _to_int(value: str) -> int:
    return int(float(value))


def _resolve_image_output_path(
    *,
    workdir: Path,
    kind: str,
    name: str,
    output_path: str | None,
) -> Path:
    if output_path:
        return Path(output_path).expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = _sanitize_name(name)
    return (workdir / "images" / kind / f"{stamp}_{safe}.svg").resolve()


def _apply_downscale(width: int, height: int, downscale_factor: float) -> tuple[int, int]:
    if downscale_factor <= 0:
        raise ValueError("downscale_factor must be > 0")
    if downscale_factor > 1.0:
        downscale_factor = 1.0
    return (
        max(100, int(round(width * downscale_factor))),
        max(100, int(round(height * downscale_factor))),
    )


def _parse_symbol_source(source: str) -> tuple[list[_Primitive], list[_Pin]]:
    primitives: list[_Primitive] = []
    pins: list[_Pin] = []
    current_pin: _Pin | None = None

    for raw in source.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        tag = parts[0].upper()

        if tag == "LINE" and len(parts) >= 6:
            primitives.append(
                _Primitive(
                    kind="line",
                    coords=(
                        _to_int(parts[-4]),
                        _to_int(parts[-3]),
                        _to_int(parts[-2]),
                        _to_int(parts[-1]),
                    ),
                )
            )
            continue

        if tag == "RECTANGLE" and len(parts) >= 6:
            primitives.append(
                _Primitive(
                    kind="rect",
                    coords=(
                        _to_int(parts[-4]),
                        _to_int(parts[-3]),
                        _to_int(parts[-2]),
                        _to_int(parts[-1]),
                    ),
                )
            )
            continue

        if tag == "CIRCLE" and len(parts) >= 6:
            primitives.append(
                _Primitive(
                    kind="circle",
                    coords=(
                        _to_int(parts[-4]),
                        _to_int(parts[-3]),
                        _to_int(parts[-2]),
                        _to_int(parts[-1]),
                    ),
                )
            )
            continue

        if tag == "ARC" and len(parts) >= 10:
            primitives.append(
                _Primitive(
                    kind="arc",
                    coords=(
                        _to_int(parts[-8]),
                        _to_int(parts[-7]),
                        _to_int(parts[-6]),
                        _to_int(parts[-5]),
                        _to_int(parts[-4]),
                        _to_int(parts[-3]),
                        _to_int(parts[-2]),
                        _to_int(parts[-1]),
                    ),
                )
            )
            continue

        if tag == "PIN" and len(parts) >= 3:
            current_pin = _Pin(x=_to_int(parts[1]), y=_to_int(parts[2]))
            pins.append(current_pin)
            continue

        if tag == "PINATTR" and current_pin is not None and len(parts) >= 3:
            attr = parts[1].lower()
            value = " ".join(parts[2:])
            if attr == "pinname":
                current_pin.name = value
            elif attr == "spiceorder":
                try:
                    current_pin.spice_order = int(value)
                except ValueError:
                    current_pin.spice_order = None

    return primitives, pins


def _primitive_points(primitives: list[_Primitive]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for primitive in primitives:
        c = primitive.coords
        if primitive.kind in {"line", "rect", "circle"}:
            points.extend([(c[0], c[1]), (c[2], c[3])])
        elif primitive.kind == "arc":
            points.extend([(c[0], c[1]), (c[2], c[3]), (c[4], c[5]), (c[6], c[7])])
    return points


def _make_mapper(
    *,
    points: list[tuple[float, float]],
    width: int,
    height: int,
    padding: int = 24,
) -> tuple[Any, dict[str, float]]:
    if not points:
        points = [(0.0, 0.0), (1.0, 1.0)]

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)
    scale = min((width - 2 * padding) / span_x, (height - 2 * padding) / span_y)

    def mapper(x: float, y: float) -> tuple[float, float]:
        return (
            padding + (x - min_x) * scale,
            padding + (y - min_y) * scale,
        )

    meta = {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "span_x": span_x,
        "span_y": span_y,
        "scale": scale,
        "padding": float(padding),
    }
    return mapper, meta


def render_symbol_svg(
    *,
    workdir: Path,
    symbol: str,
    output_path: str | None = None,
    width: int = 640,
    height: int = 420,
    downscale_factor: float = 1.0,
    include_pins: bool = True,
    include_pin_labels: bool = True,
    library: SymbolLibrary | None = None,
) -> dict[str, Any]:
    if width < 100 or height < 100:
        raise ValueError("width and height must be at least 100")
    width, height = _apply_downscale(width, height, downscale_factor)

    owns_library = library is None
    lib = library or SymbolLibrary()
    try:
        source = lib.read_symbol_source(symbol)
        entry = lib.resolve_entry(symbol)
        primitives, pins = _parse_symbol_source(source)

        points = _primitive_points(primitives)
        if include_pins:
            points.extend((pin.x, pin.y) for pin in pins)

        mapper, meta = _make_mapper(points=points, width=width, height=height)

        lines: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fafafa"/>',
            '<g stroke="#1f2937" stroke-width="2" fill="none">',
        ]

        for primitive in primitives:
            c = primitive.coords
            if primitive.kind == "line":
                x1, y1 = mapper(c[0], c[1])
                x2, y2 = mapper(c[2], c[3])
                lines.append(
                    f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}"/>'
                )
            elif primitive.kind == "rect":
                p1 = mapper(c[0], c[1])
                p2 = mapper(c[2], c[3])
                x = min(p1[0], p2[0])
                y = min(p1[1], p2[1])
                w = abs(p2[0] - p1[0])
                h = abs(p2[1] - p1[1])
                lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}"/>')
            elif primitive.kind == "circle":
                p1 = mapper(c[0], c[1])
                p2 = mapper(c[2], c[3])
                cx = (p1[0] + p2[0]) / 2.0
                cy = (p1[1] + p2[1]) / 2.0
                rx = abs(p2[0] - p1[0]) / 2.0
                ry = abs(p2[1] - p1[1]) / 2.0
                r = (rx + ry) / 2.0
                lines.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}"/>')
            elif primitive.kind == "arc":
                # Simple approximation: draw the chord between arc endpoints.
                x1, y1 = mapper(c[4], c[5])
                x2, y2 = mapper(c[6], c[7])
                lines.append(
                    f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke-dasharray="4,4"/>'
                )

        lines.append("</g>")

        if include_pins and pins:
            lines.append('<g stroke="#0f766e" fill="#14b8a6">')
            for pin in pins:
                x, y = mapper(pin.x, pin.y)
                lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2"/>')
            lines.append("</g>")

            if include_pin_labels:
                lines.append('<g font-family="Menlo,Monaco,monospace" font-size="11" fill="#0f172a">')
                sorted_pins = sorted(
                    pins,
                    key=lambda item: (
                        item.spice_order if item.spice_order is not None else 9999,
                        item.name or "",
                    ),
                )
                for pin in sorted_pins:
                    x, y = mapper(pin.x, pin.y)
                    label = pin.name or "pin"
                    if pin.spice_order is not None:
                        label = f"{pin.spice_order}:{label}"
                    lines.append(f'<text x="{x + 6:.2f}" y="{y - 6:.2f}">{_svg_escape(label)}</text>')
                lines.append("</g>")

        lines.append('<g font-family="Menlo,Monaco,monospace" font-size="12" fill="#111827">')
        lines.append(f'<text x="12" y="18">symbol: {_svg_escape(symbol)}</text>')
        lines.append(f'<text x="12" y="34">entry: {_svg_escape(entry)}</text>')
        lines.append("</g>")
        lines.append("</svg>")

        target = _resolve_image_output_path(
            workdir=workdir,
            kind="symbols",
            name=symbol,
            output_path=output_path,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")

        return {
            "image_path": str(target),
            "format": "svg",
            "symbol": symbol,
            "zip_entry": entry,
            "zip_path": str(lib.zip_path),
            "width": width,
            "height": height,
            "downscale_factor": min(1.0, float(downscale_factor)),
            "pin_count": len(pins),
            "primitive_count": len(primitives),
            "bounds": meta,
        }
    finally:
        if owns_library:
            lib.close()


def _parse_asc_source(source: str) -> dict[str, Any]:
    wires: list[tuple[int, int, int, int]] = []
    flags: list[tuple[int, int, str]] = []
    directives: list[tuple[int, int, str]] = []
    symbols: list[dict[str, Any]] = []
    current_symbol: dict[str, Any] | None = None

    for raw in source.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        tag = parts[0].upper()

        if tag == "WIRE" and len(parts) >= 5:
            wires.append((_to_int(parts[1]), _to_int(parts[2]), _to_int(parts[3]), _to_int(parts[4])))
            current_symbol = None
            continue

        if tag == "FLAG" and len(parts) >= 4:
            flags.append((_to_int(parts[1]), _to_int(parts[2]), " ".join(parts[3:])))
            current_symbol = None
            continue

        if tag == "TEXT" and len(parts) >= 6:
            directives.append((_to_int(parts[1]), _to_int(parts[2]), " ".join(parts[5:])))
            current_symbol = None
            continue

        if tag == "SYMBOL" and len(parts) >= 5:
            current_symbol = {
                "symbol": parts[1],
                "x": _to_int(parts[2]),
                "y": _to_int(parts[3]),
                "orientation": parts[4],
                "attrs": {},
            }
            symbols.append(current_symbol)
            continue

        if tag == "SYMATTR" and current_symbol is not None and len(parts) >= 3:
            key = parts[1]
            value = " ".join(parts[2:])
            current_symbol["attrs"][key] = value

    return {
        "wires": wires,
        "flags": flags,
        "directives": directives,
        "symbols": symbols,
    }


def _transform_symbol_primitive(
    primitive: _Primitive,
    origin_x: int,
    origin_y: int,
    orientation: str,
) -> _Primitive:
    c = primitive.coords

    def point(x: int, y: int) -> tuple[int, int]:
        tx, ty = _transform_point(x, y, orientation)
        return origin_x + tx, origin_y + ty

    if primitive.kind in {"line", "rect", "circle"}:
        p1 = point(c[0], c[1])
        p2 = point(c[2], c[3])
        if primitive.kind == "line":
            return _Primitive("line", (p1[0], p1[1], p2[0], p2[1]))
        # Rectangles and circles are rendered axis-aligned after transform.
        return _Primitive(primitive.kind, (p1[0], p1[1], p2[0], p2[1]))

    if primitive.kind == "arc":
        p1 = point(c[0], c[1])
        p2 = point(c[2], c[3])
        p3 = point(c[4], c[5])
        p4 = point(c[6], c[7])
        return _Primitive("arc", (p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1]))

    return primitive


def render_schematic_svg(
    *,
    workdir: Path,
    asc_path: str | Path,
    output_path: str | None = None,
    width: int = 1400,
    height: int = 900,
    downscale_factor: float = 1.0,
    include_symbol_graphics: bool = True,
    library: SymbolLibrary | None = None,
) -> dict[str, Any]:
    if width < 200 or height < 200:
        raise ValueError("width and height must be at least 200")
    width, height = _apply_downscale(width, height, downscale_factor)

    asc = Path(asc_path).expanduser().resolve()
    if not asc.exists():
        raise FileNotFoundError(f"Schematic file not found: {asc}")

    data = _parse_asc_source(asc.read_text(encoding="utf-8", errors="ignore"))

    owns_library = library is None and include_symbol_graphics
    lib = library or (SymbolLibrary() if include_symbol_graphics else None)

    symbol_primitives: list[_Primitive] = []
    symbol_labels: list[tuple[int, int, str]] = []
    symbol_warnings: list[str] = []

    try:
        for symbol in data["symbols"]:
            origin_x = int(symbol["x"])
            origin_y = int(symbol["y"])
            orientation = _normalize_orientation(symbol.get("orientation", "R0"))
            name = str(symbol["symbol"])
            attrs = symbol.get("attrs", {})
            inst_name = attrs.get("InstName", "")
            value = attrs.get("Value", "")
            if inst_name:
                symbol_labels.append((origin_x + 12, origin_y - 12, inst_name))
            if value:
                symbol_labels.append((origin_x + 12, origin_y + 8, value))

            if not include_symbol_graphics:
                symbol_primitives.append(_Primitive("rect", (origin_x - 28, origin_y - 20, origin_x + 28, origin_y + 20)))
                continue

            try:
                source = lib.read_symbol_source(name) if lib is not None else ""
                primitives, _ = _parse_symbol_source(source)
                if not primitives:
                    symbol_primitives.append(
                        _Primitive("rect", (origin_x - 28, origin_y - 20, origin_x + 28, origin_y + 20))
                    )
                for primitive in primitives:
                    symbol_primitives.append(
                        _transform_symbol_primitive(
                            primitive,
                            origin_x=origin_x,
                            origin_y=origin_y,
                            orientation=orientation,
                        )
                    )
            except Exception as exc:
                symbol_warnings.append(f"Failed symbol render for {name}: {exc}")
                symbol_primitives.append(_Primitive("rect", (origin_x - 28, origin_y - 20, origin_x + 28, origin_y + 20)))

        points: list[tuple[float, float]] = []
        for x1, y1, x2, y2 in data["wires"]:
            points.extend([(x1, y1), (x2, y2)])
        for x, y, _ in data["flags"]:
            points.append((x, y))
        points.extend(_primitive_points(symbol_primitives))
        for x, y, _ in data["directives"]:
            points.append((x, y))
        for x, y, _ in symbol_labels:
            points.append((x, y))

        mapper, meta = _make_mapper(points=points, width=width, height=height)

        lines: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
            '<g stroke="#0f172a" stroke-width="1.8" fill="none">',
        ]

        for x1, y1, x2, y2 in data["wires"]:
            p1 = mapper(x1, y1)
            p2 = mapper(x2, y2)
            lines.append(f'<line x1="{p1[0]:.2f}" y1="{p1[1]:.2f}" x2="{p2[0]:.2f}" y2="{p2[1]:.2f}"/>')

        for primitive in symbol_primitives:
            c = primitive.coords
            if primitive.kind == "line":
                p1 = mapper(c[0], c[1])
                p2 = mapper(c[2], c[3])
                lines.append(f'<line x1="{p1[0]:.2f}" y1="{p1[1]:.2f}" x2="{p2[0]:.2f}" y2="{p2[1]:.2f}"/>')
            elif primitive.kind == "rect":
                p1 = mapper(c[0], c[1])
                p2 = mapper(c[2], c[3])
                x = min(p1[0], p2[0])
                y = min(p1[1], p2[1])
                w = abs(p2[0] - p1[0])
                h = abs(p2[1] - p1[1])
                lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}"/>')
            elif primitive.kind == "circle":
                p1 = mapper(c[0], c[1])
                p2 = mapper(c[2], c[3])
                cx = (p1[0] + p2[0]) / 2.0
                cy = (p1[1] + p2[1]) / 2.0
                r = (abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])) / 4.0
                lines.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}"/>')
            elif primitive.kind == "arc":
                p1 = mapper(c[4], c[5])
                p2 = mapper(c[6], c[7])
                lines.append(
                    f'<line x1="{p1[0]:.2f}" y1="{p1[1]:.2f}" x2="{p2[0]:.2f}" y2="{p2[1]:.2f}" stroke-dasharray="3,3"/>'
                )

        lines.append("</g>")

        if data["flags"]:
            lines.append('<g fill="#b91c1c" font-family="Menlo,Monaco,monospace" font-size="11">')
            for x, y, name in data["flags"]:
                p = mapper(x, y)
                lines.append(f'<circle cx="{p[0]:.2f}" cy="{p[1]:.2f}" r="2.8"/>')
                lines.append(f'<text x="{p[0] + 5:.2f}" y="{p[1] - 4:.2f}">{_svg_escape(name)}</text>')
            lines.append("</g>")

        if symbol_labels:
            lines.append('<g fill="#111827" font-family="Menlo,Monaco,monospace" font-size="11">')
            for x, y, text in symbol_labels:
                p = mapper(x, y)
                lines.append(f'<text x="{p[0]:.2f}" y="{p[1]:.2f}">{_svg_escape(text)}</text>')
            lines.append("</g>")

        if data["directives"]:
            lines.append('<g fill="#0f766e" font-family="Menlo,Monaco,monospace" font-size="11">')
            for x, y, text in data["directives"]:
                p = mapper(x, y)
                lines.append(f'<text x="{p[0]:.2f}" y="{p[1]:.2f}">{_svg_escape(text)}</text>')
            lines.append("</g>")

        lines.append("</svg>")

        target = _resolve_image_output_path(
            workdir=workdir,
            kind="schematics",
            name=asc.stem,
            output_path=output_path,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")

        return {
            "image_path": str(target),
            "format": "svg",
            "asc_path": str(asc),
            "width": width,
            "height": height,
            "downscale_factor": min(1.0, float(downscale_factor)),
            "wires": len(data["wires"]),
            "flags": len(data["flags"]),
            "symbols": len(data["symbols"]),
            "directives": len(data["directives"]),
            "warnings": symbol_warnings,
            "bounds": meta,
        }
    finally:
        if owns_library and lib is not None:
            lib.close()


def _series_to_real(series: list[complex], mode: str) -> list[float]:
    if mode == "real":
        return [value.real for value in series]
    if mode == "imag":
        return [value.imag for value in series]
    if mode == "magnitude":
        return [abs(value) for value in series]
    if mode == "phase_deg":
        return [math.degrees(math.atan2(value.imag, value.real)) for value in series]
    raise ValueError("y_mode must be one of: real, imag, magnitude, phase_deg")


def render_plot_svg(
    *,
    workdir: Path,
    dataset: RawDataset,
    vectors: list[str],
    step_index: int | None = None,
    output_path: str | None = None,
    width: int = 1280,
    height: int = 720,
    downscale_factor: float = 1.0,
    max_points: int = 2000,
    y_mode: str = "magnitude",
    x_log: bool | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    if width < 300 or height < 200:
        raise ValueError("width must be >= 300 and height must be >= 200")
    width, height = _apply_downscale(width, height, downscale_factor)
    if max_points <= 1:
        raise ValueError("max_points must be > 1")
    if not vectors:
        raise ValueError("vectors must contain at least one name")

    x_values = dataset.scale_values(step_index=step_index)
    if not x_values:
        raise ValueError("dataset has no points")

    use_x_log = dataset.scale_name.lower() == "frequency" if x_log is None else bool(x_log)
    if use_x_log and any(value <= 0 for value in x_values):
        use_x_log = False

    total_points = len(x_values)
    sampled = sample_indices(total_points, min(max_points, total_points))

    series_map: dict[str, list[float]] = {}
    for vector in vectors:
        values = dataset.get_vector(vector, step_index=step_index)
        if len(values) != total_points:
            raise ValueError(f"Vector '{vector}' length mismatch")
        series_map[vector] = _series_to_real(values, y_mode)

    sampled_x: list[float] = []
    sampled_y: dict[str, list[float]] = {vector: [] for vector in vectors}

    for idx in sampled:
        x_value = x_values[idx]
        if use_x_log and x_value <= 0:
            continue
        y_values = [series_map[vector][idx] for vector in vectors]
        if any(not math.isfinite(value) for value in y_values):
            continue
        sampled_x.append(x_value)
        for vector in vectors:
            sampled_y[vector].append(series_map[vector][idx])

    if len(sampled_x) < 2:
        raise ValueError("Not enough valid plot points after sampling")

    x_draw = [math.log10(value) if use_x_log else value for value in sampled_x]
    x_min = min(x_draw)
    x_max = max(x_draw)
    if x_max == x_min:
        x_max = x_min + 1.0

    all_y_values = [value for values in sampled_y.values() for value in values]
    y_min = min(all_y_values)
    y_max = max(all_y_values)
    if y_max == y_min:
        y_max = y_min + 1.0

    margin_left = 80
    margin_right = 24
    margin_top = 30
    margin_bottom = 60
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def map_x(value: float) -> float:
        return margin_left + ((value - x_min) / (x_max - x_min)) * plot_w

    def map_y(value: float) -> float:
        return margin_top + (1.0 - (value - y_min) / (y_max - y_min)) * plot_h

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#cbd5e1"/>',
    ]

    # Grid and axis ticks
    lines.append('<g stroke="#e2e8f0" stroke-width="1">')
    for idx in range(6):
        fx = idx / 5
        x = margin_left + fx * plot_w
        y = margin_top + fx * plot_h
        lines.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + plot_h}"/>')
        lines.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_w}" y2="{y:.2f}"/>')
    lines.append("</g>")

    palette = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0f766e"]

    lines.append('<g fill="none" stroke-width="2">')
    for index, vector in enumerate(vectors):
        color = palette[index % len(palette)]
        points = [
            f"{map_x(x):.2f},{map_y(y):.2f}"
            for x, y in zip(x_draw, sampled_y[vector], strict=False)
        ]
        lines.append(
            f'<polyline stroke="{color}" points="{" ".join(points)}"/>'
        )
    lines.append("</g>")

    lines.append('<g font-family="Menlo,Monaco,monospace" font-size="12" fill="#0f172a">')
    x_label = f"{dataset.scale_name}{' (log10)' if use_x_log else ''}"
    lines.append(f'<text x="{margin_left}" y="{height - 18}">{_svg_escape(x_label)}</text>')
    lines.append(f'<text x="16" y="{margin_top + 12}">{_svg_escape(y_mode)}</text>')

    if title is None:
        vector_label = ", ".join(vectors)
        title = f"{dataset.plot_name}: {vector_label}"
    lines.append(f'<text x="{margin_left}" y="20">{_svg_escape(title)}</text>')

    # Simple legend
    legend_x = margin_left + 8
    legend_y = margin_top + 18
    for index, vector in enumerate(vectors):
        color = palette[index % len(palette)]
        y = legend_y + index * 16
        lines.append(f'<rect x="{legend_x}" y="{y - 9}" width="10" height="10" fill="{color}"/>')
        lines.append(f'<text x="{legend_x + 14}" y="{y}">{_svg_escape(vector)}</text>')

    # Axis labels
    for idx in range(6):
        fx = idx / 5
        x_pos = margin_left + fx * plot_w
        y_pos = margin_top + plot_h + 16
        draw_x = x_min + fx * (x_max - x_min)
        label_x_value = 10 ** draw_x if use_x_log else draw_x
        lines.append(f'<text x="{x_pos:.2f}" y="{y_pos:.2f}" text-anchor="middle">{label_x_value:.4g}</text>')

        y_pos_tick = margin_top + plot_h - fx * plot_h
        label_y = y_min + fx * (y_max - y_min)
        lines.append(
            f'<text x="{margin_left - 8}" y="{y_pos_tick + 4:.2f}" text-anchor="end">{label_y:.4g}</text>'
        )

    lines.append("</g>")
    lines.append("</svg>")

    name = vectors[0] if len(vectors) == 1 else "multi_vector"
    target = _resolve_image_output_path(
        workdir=workdir,
        kind="plots",
        name=f"{dataset.path.stem}_{name}",
        output_path=output_path,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "image_path": str(target),
        "format": "svg",
        "raw_path": str(dataset.path),
        "plot_name": dataset.plot_name,
        "scale_name": dataset.scale_name,
        "vectors": vectors,
        "width": width,
        "height": height,
        "downscale_factor": min(1.0, float(downscale_factor)),
        "points_total": total_points,
        "points_rendered": len(sampled_x),
        "x_log": use_x_log,
        "y_mode": y_mode,
        "y_range": [y_min, y_max],
        "x_range": [sampled_x[0], sampled_x[-1]],
    }
