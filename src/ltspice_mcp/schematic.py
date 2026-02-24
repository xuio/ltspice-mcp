from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zipfile import ZipFile


DEFAULT_LTSPICE_LIB_ZIP = Path("/Applications/LTspice.app/Contents/Resources/lib.zip")
DEFAULT_TEMPLATES_JSON = Path(__file__).with_name("schematic_templates.json")


def _sanitize_name(name: str) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in name)
    return cleaned.strip("_") or "schematic"


def _normalize_orientation(raw: Any) -> str:
    if isinstance(raw, int):
        value = raw % 360
        if value not in {0, 90, 180, 270}:
            raise ValueError("rotation must be one of: 0, 90, 180, 270")
        return f"R{value}"
    if isinstance(raw, str):
        text = raw.strip().upper()
        if text.startswith("R"):
            value = int(text[1:])
            if value not in {0, 90, 180, 270}:
                raise ValueError("orientation must be R0/R90/R180/R270")
            return text
        if text.startswith("M"):
            value = int(text[1:])
            if value not in {0, 90, 180, 270}:
                raise ValueError("orientation must be M0/M90/M180/M270")
            return text
    raise ValueError("orientation must be int rotation or string (R0/R90/R180/R270/M*)")


def _transform_point(x: int, y: int, orientation: str) -> tuple[int, int]:
    mode = _normalize_orientation(orientation)
    if mode == "R0":
        return x, y
    if mode == "R90":
        return -y, x
    if mode == "R180":
        return -x, -y
    if mode == "R270":
        return y, -x
    if mode == "M0":
        return -x, y
    if mode == "M90":
        return y, x
    if mode == "M180":
        return x, -y
    if mode == "M270":
        return -y, -x
    return x, y


@dataclass(slots=True)
class PinDef:
    x: int
    y: int
    spice_order: int | None
    name: str | None


@dataclass(slots=True)
class SymbolDef:
    symbol: str
    zip_entry: str
    pins: list[PinDef]

    def pin_for_order(self, spice_order: int) -> PinDef | None:
        for pin in self.pins:
            if pin.spice_order == spice_order:
                return pin
        if 1 <= spice_order <= len(self.pins):
            return sorted(self.pins, key=lambda item: (item.spice_order or 9999, item.name or ""))[spice_order - 1]
        return None


class SymbolLibrary:
    _INDEX_CACHE: dict[Path, dict[str, str]] = {}
    _ENTRY_CACHE: dict[Path, list[str]] = {}
    _SYMBOL_LIST_CACHE: dict[Path, list[dict[str, str]]] = {}
    _SYMBOL_CACHE: dict[tuple[Path, str], SymbolDef] = {}
    _SOURCE_CACHE: dict[tuple[Path, str], str] = {}

    def __init__(self, zip_path: Path = DEFAULT_LTSPICE_LIB_ZIP) -> None:
        self.zip_path = zip_path.expanduser().resolve()
        if not self.zip_path.exists():
            raise FileNotFoundError(f"LTspice symbol library not found at {self.zip_path}")
        self._zip = ZipFile(self.zip_path)
        self._index = self._build_index()
        self._entries = self._build_entries()
        self._pin_offset_cache: dict[tuple[str, str, int], tuple[int, int]] = {}

    def _build_entries(self) -> list[str]:
        cached = self._ENTRY_CACHE.get(self.zip_path)
        if cached is not None:
            return cached

        entries = sorted(
            Path(name).as_posix()
            for name in self._zip.namelist()
            if name.lower().endswith(".asy")
        )
        self._ENTRY_CACHE[self.zip_path] = entries
        return entries

    def _build_index(self) -> dict[str, str]:
        cached = self._INDEX_CACHE.get(self.zip_path)
        if cached is not None:
            return cached

        index: dict[str, str] = {}
        for name in self._zip.namelist():
            if not name.lower().endswith(".asy"):
                continue
            normalized = Path(name).as_posix().lower()
            base = Path(name).stem.lower()
            index.setdefault(base, name)
            index.setdefault(normalized, name)
            index.setdefault(Path(normalized).name, name)
        self._INDEX_CACHE[self.zip_path] = index
        return index

    def _resolve_entry(self, symbol: str) -> str:
        key = symbol.strip().lower()
        if key in self._index:
            return self._index[key]
        if key.endswith(".asy"):
            short = Path(key).name
            if short in self._index:
                return self._index[short]
        alt = f"lib/sym/{key}.asy"
        if alt in self._index:
            return self._index[alt]
        raise ValueError(f"Symbol '{symbol}' not found in LTspice lib.zip")

    def get(self, symbol: str) -> SymbolDef:
        cache_key = symbol.strip().lower()
        cached = self._SYMBOL_CACHE.get((self.zip_path, cache_key))
        if cached:
            return cached

        entry = self._resolve_entry(symbol)
        text = self._zip.read(entry).decode("utf-8", errors="ignore")
        pins = _parse_symbol_pins(text)
        symbol_def = SymbolDef(symbol=symbol, zip_entry=entry, pins=pins)
        self._SYMBOL_CACHE[(self.zip_path, cache_key)] = symbol_def
        return symbol_def

    def resolve_entry(self, symbol: str) -> str:
        return self._resolve_entry(symbol)

    def read_symbol_source(self, symbol: str) -> str:
        entry = self._resolve_entry(symbol)
        cache_key = (self.zip_path, entry)
        cached = self._SOURCE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        text = self._zip.read(entry).decode("utf-8", errors="ignore")
        self._SOURCE_CACHE[cache_key] = text
        return text

    def list_entries(self, *, query: str | None = None, limit: int = 500) -> list[str]:
        if limit <= 0:
            return []
        entries = self._entries
        if query:
            needle = query.strip().lower()
            entries = [entry for entry in entries if needle in entry.lower()]
        return entries[:limit]

    def list_symbols(
        self,
        *,
        query: str | None = None,
        library: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, str]]:
        if limit <= 0:
            return []

        cached = self._SYMBOL_LIST_CACHE.get(self.zip_path)
        if cached is None:
            seen: set[str] = set()
            rows: list[dict[str, str]] = []
            for entry in self._entries:
                symbol = Path(entry).stem
                key = symbol.lower()
                if key in seen:
                    continue
                seen.add(key)
                parts = Path(entry).parts
                category = ""
                if "sym" in parts:
                    idx = parts.index("sym")
                    if idx + 1 < len(parts) - 1:
                        category = parts[idx + 1]
                rows.append(
                    {
                        "symbol": symbol,
                        "entry": entry,
                        "category": category,
                    }
                )
            cached = sorted(rows, key=lambda row: row["symbol"].lower())
            self._SYMBOL_LIST_CACHE[self.zip_path] = cached

        rows = cached
        if library:
            library_needle = library.strip().lower()
            rows = [row for row in rows if library_needle in row["category"].lower()]
        if query:
            needle = query.strip().lower()
            rows = [
                row
                for row in rows
                if needle in row["symbol"].lower() or needle in row["entry"].lower()
            ]
        return rows[:limit]

    def symbol_info(self, symbol: str) -> dict[str, Any]:
        symbol_def = self.get(symbol)
        pins_out = [
            {
                "spice_order": pin.spice_order,
                "name": pin.name,
                "x": pin.x,
                "y": pin.y,
            }
            for pin in symbol_def.pins
        ]
        pins_out = sorted(
            pins_out,
            key=lambda item: (
                item["spice_order"] if item["spice_order"] is not None else 9999,
                item["name"] or "",
            ),
        )
        return {
            "symbol": symbol.strip(),
            "zip_path": str(self.zip_path),
            "zip_entry": symbol_def.zip_entry,
            "pin_count": len(symbol_def.pins),
            "pins": pins_out,
        }

    def pin_offset(self, symbol: str, orientation: str, spice_order: int) -> tuple[int, int]:
        normalized_orientation = _normalize_orientation(orientation)
        cache_key = (symbol.strip().lower(), normalized_orientation, int(spice_order))
        cached = self._pin_offset_cache.get(cache_key)
        if cached is not None:
            return cached

        symbol_def = self.get(symbol)
        pin = symbol_def.pin_for_order(spice_order)
        if pin is None:
            raise ValueError(
                f"Symbol '{symbol}' does not expose SpiceOrder {spice_order}"
            )
        offset = _transform_point(pin.x, pin.y, normalized_orientation)
        self._pin_offset_cache[cache_key] = offset
        return offset

    def close(self) -> None:
        self._zip.close()


def _parse_symbol_pins(text: str) -> list[PinDef]:
    pins: list[PinDef] = []
    current: PinDef | None = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if parts[0].upper() == "PIN" and len(parts) >= 3:
            current = PinDef(x=int(parts[1]), y=int(parts[2]), spice_order=None, name=None)
            pins.append(current)
            continue
        if parts[0].upper() == "PINATTR" and current is not None and len(parts) >= 3:
            attr_name = parts[1].lower()
            attr_value = " ".join(parts[2:])
            if attr_name == "spiceorder":
                try:
                    current.spice_order = int(attr_value)
                except ValueError:
                    current.spice_order = None
            elif attr_name == "pinname":
                current.name = attr_value
    return pins


@dataclass(slots=True)
class ComponentPlacement:
    symbol: str
    reference: str
    x: int
    y: int
    orientation: str = "R0"
    value: str | None = None
    attributes: dict[str, str] | None = None


class SchematicBuilder:
    def __init__(self, *, sheet_width: int = 880, sheet_height: int = 680) -> None:
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.wires: list[tuple[int, int, int, int]] = []
        self.flags: list[tuple[int, int, str]] = []
        self.components: list[ComponentPlacement] = []
        self.directives: list[tuple[int, int, str]] = []

    def add_wire(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self.wires.append((int(x1), int(y1), int(x2), int(y2)))

    def add_flag(self, x: int, y: int, name: str) -> None:
        self.flags.append((int(x), int(y), str(name)))

    def add_component(self, component: ComponentPlacement) -> None:
        self.components.append(component)

    def add_directive(self, x: int, y: int, text: str) -> None:
        directive = text.strip()
        if not directive:
            return
        if not directive.startswith("!"):
            directive = "!" + directive
        self.directives.append((int(x), int(y), directive.replace("\n", "\\n")))

    def render(self) -> str:
        lines = ["Version 4", f"SHEET 1 {self.sheet_width} {self.sheet_height}"]
        for x1, y1, x2, y2 in self.wires:
            lines.append(f"WIRE {x1} {y1} {x2} {y2}")
        for x, y, name in self.flags:
            lines.append(f"FLAG {x} {y} {name}")
        for component in self.components:
            orientation = _normalize_orientation(component.orientation)
            lines.append(
                f"SYMBOL {component.symbol} {int(component.x)} {int(component.y)} {orientation}"
            )
            lines.append(f"SYMATTR InstName {component.reference}")
            if component.value is not None:
                lines.append(f"SYMATTR Value {component.value}")
            for key, value in (component.attributes or {}).items():
                lines.append(f"SYMATTR {key} {value}")
        for x, y, text in self.directives:
            lines.append(f"TEXT {x} {y} Left 2 {text}")
        return "\n".join(lines) + "\n"

    def write(self, path: Path) -> Path:
        target = path.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.render(), encoding="utf-8")
        return target


def _component_pin_position(
    library: SymbolLibrary,
    placement: ComponentPlacement,
    spice_order: int,
) -> tuple[int, int]:
    dx, dy = library.pin_offset(
        placement.symbol,
        orientation=placement.orientation,
        spice_order=spice_order,
    )
    return placement.x + dx, placement.y + dy


def _resolve_output_path(
    *,
    workdir: Path,
    circuit_name: str,
    output_path: str | None,
) -> Path:
    if output_path:
        return Path(output_path).expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = _sanitize_name(circuit_name)
    return (workdir / "schematics" / f"{stamp}_{safe}" / f"{safe}.asc").resolve()


def _resolve_stable_output_path(
    *,
    workdir: Path,
    circuit_name: str,
    output_path: str | None,
) -> Path:
    if output_path:
        return Path(output_path).expanduser().resolve()
    safe = _sanitize_name(circuit_name)
    return (workdir / "schematics" / f"{safe}.asc").resolve()


def _sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _safe_format(template: str, parameters: dict[str, Any]) -> str:
    class _MissingDict(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return template.format_map(_MissingDict({key: str(value) for key, value in parameters.items()}))


def _format_recursive(value: Any, parameters: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return _safe_format(value, parameters)
    if isinstance(value, list):
        return [_format_recursive(item, parameters) for item in value]
    if isinstance(value, dict):
        return {str(key): _format_recursive(item, parameters) for key, item in value.items()}
    return value


def _load_template_payload(template_path: str | Path | None = None) -> tuple[Path, dict[str, Any]]:
    source_path = (
        Path(template_path).expanduser().resolve()
        if template_path is not None
        else DEFAULT_TEMPLATES_JSON.resolve()
    )
    if not source_path.exists():
        raise FileNotFoundError(f"Template JSON not found: {source_path}")

    payload = _read_json(source_path)
    templates = payload.get("templates")
    if not isinstance(templates, list):
        raise ValueError("Template JSON must contain a 'templates' array")
    return source_path, payload


def list_schematic_templates(template_path: str | Path | None = None) -> dict[str, Any]:
    source_path, payload = _load_template_payload(template_path)
    templates_out: list[dict[str, Any]] = []
    for raw in payload.get("templates", []):
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", "")).strip()
        if not name:
            continue
        templates_out.append(
            {
                "name": name,
                "description": str(raw.get("description", "")),
                "type": str(raw.get("type", "netlist")),
                "circuit_name": str(raw.get("circuit_name") or name),
            }
        )
    return {
        "template_path": str(source_path),
        "version": int(payload.get("version", 1)),
        "templates": templates_out,
    }


def build_schematic_from_spec(
    *,
    workdir: Path,
    components: list[dict[str, Any]],
    wires: list[dict[str, Any]] | None = None,
    directives: list[dict[str, Any] | str] | None = None,
    labels: list[dict[str, Any]] | None = None,
    circuit_name: str = "schematic",
    output_path: str | None = None,
    sheet_width: int = 880,
    sheet_height: int = 680,
    library: SymbolLibrary | None = None,
) -> dict[str, Any]:
    lib = library
    builder = SchematicBuilder(sheet_width=sheet_width, sheet_height=sheet_height)

    component_objs: list[ComponentPlacement] = []
    for raw in components:
        try:
            component = ComponentPlacement(
                symbol=str(raw["symbol"]),
                reference=str(raw["reference"]),
                x=int(raw["x"]),
                y=int(raw["y"]),
                orientation=_normalize_orientation(raw.get("orientation", raw.get("rotation", "R0"))),
                value=str(raw["value"]) if raw.get("value") is not None else None,
                attributes={
                    str(key): str(value)
                    for key, value in (raw.get("attributes") or {}).items()
                },
            )
        except KeyError as exc:
            raise ValueError(f"Missing required component field: {exc}") from exc
        builder.add_component(component)
        component_objs.append(component)

    for wire in wires or []:
        builder.add_wire(int(wire["x1"]), int(wire["y1"]), int(wire["x2"]), int(wire["y2"]))

    for label in labels or []:
        builder.add_flag(int(label["x"]), int(label["y"]), str(label["name"]))

    # Optional pin-level net labels directly from component definition.
    for raw, component in zip(components, component_objs, strict=False):
        pin_nets = raw.get("pin_nets") or {}
        if pin_nets and lib is None:
            lib = SymbolLibrary()
        for raw_pin, raw_net in pin_nets.items():
            net_name = str(raw_net)
            try:
                spice_order = int(raw_pin)
            except ValueError as exc:
                raise ValueError(
                    f"pin_nets keys must be SpiceOrder integers. Invalid key '{raw_pin}'"
                ) from exc
            x, y = _component_pin_position(lib, component, spice_order=spice_order)
            builder.add_flag(x, y, net_name)

    for index, directive in enumerate(directives or []):
        if isinstance(directive, str):
            builder.add_directive(48, sheet_height - 120 + index * 24, directive)
            continue
        text = str(directive.get("text", "")).strip()
        if not text:
            continue
        x = int(directive.get("x", 48))
        y = int(directive.get("y", sheet_height - 120 + index * 24))
        builder.add_directive(x, y, text)

    out_path = _resolve_output_path(workdir=workdir, circuit_name=circuit_name, output_path=output_path)
    builder.write(out_path)
    return {
        "asc_path": str(out_path),
        "components": len(component_objs),
        "wires": len(builder.wires),
        "flags": len(builder.flags),
        "directives": len(builder.directives),
    }


def _strip_inline_comment(line: str) -> str:
    # ';' is a common inline comment marker in LTspice netlists.
    if ";" in line:
        return line.split(";", 1)[0].rstrip()
    return line


def _split_netlist_tokens(line: str) -> list[str]:
    cleaned = _strip_inline_comment(line).strip()
    if not cleaned:
        return []
    return cleaned.split()


def _element_to_symbol(name: str) -> str | None:
    lead = name[0].upper()
    mapping = {
        "R": "res",
        "C": "cap",
        "L": "ind",
        "D": "diode",
        "V": "voltage",
        "I": "current",
    }
    return mapping.get(lead)


def _parse_two_pin_nodes(tokens: list[str]) -> list[str] | None:
    if len(tokens) < 4:
        return None
    return [tokens[1], tokens[2]]


def _canonical_net(name: str) -> str:
    lowered = name.strip().lower()
    if lowered in {"0", "gnd", "ground"}:
        return "0"
    return name.strip()


def _choose_two_pin_orientation(symbol: str, nodes: list[str]) -> str:
    if len(nodes) < 2:
        return "R0"
    n1 = _canonical_net(nodes[0])
    n2 = _canonical_net(nodes[1])
    symbol_key = symbol.strip().lower()

    if symbol_key in {"res", "cap", "ind", "diode"}:
        if n1 == "0" and n2 != "0":
            return "R180"
        if n2 == "0" and n1 != "0":
            return "R0"
        # Horizontal by default when both pins are signal nets.
        return "M90"

    if symbol_key in {"voltage", "current"}:
        if n1 == "0" and n2 != "0":
            return "R180"
        return "R0"

    return "R0"


def _route_two_point_net(
    builder: SchematicBuilder,
    point_a: tuple[int, int],
    point_b: tuple[int, int],
) -> None:
    x1, y1 = point_a
    x2, y2 = point_b
    if (x1, y1) == (x2, y2):
        return
    if x1 == x2 or y1 == y2:
        builder.add_wire(x1, y1, x2, y2)
        return
    # Manhattan route with one bend keeps LTspice schematics legible.
    builder.add_wire(x1, y1, x2, y1)
    builder.add_wire(x2, y1, x2, y2)


def build_schematic_from_netlist(
    *,
    workdir: Path,
    netlist_content: str,
    circuit_name: str = "schematic_from_netlist",
    output_path: str | None = None,
    sheet_width: int = 1200,
    sheet_height: int = 900,
    library: SymbolLibrary | None = None,
) -> dict[str, Any]:
    if not netlist_content.strip():
        raise ValueError("netlist_content cannot be empty")

    lib = library or SymbolLibrary()
    builder = SchematicBuilder(sheet_width=sheet_width, sheet_height=sheet_height)
    warnings: list[str] = []

    components: list[tuple[ComponentPlacement, list[str]]] = []
    directives: list[str] = []

    for raw_line in netlist_content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("*"):
            continue
        tokens = _split_netlist_tokens(line)
        if not tokens:
            continue

        if tokens[0].startswith("."):
            directive = " ".join(tokens)
            if directive.lower() != ".end":
                directives.append(directive)
            continue

        ref = tokens[0]
        symbol = _element_to_symbol(ref)
        if symbol is None:
            warnings.append(f"Unsupported element '{ref}' was skipped.")
            continue

        nodes = _parse_two_pin_nodes(tokens)
        if nodes is None:
            warnings.append(f"Could not parse nodes for element '{ref}'.")
            continue

        canonical_nodes = [_canonical_net(nodes[0]), _canonical_net(nodes[1])]
        value = " ".join(tokens[3:]) if len(tokens) > 3 else None
        index = len(components)
        columns = max(1, (sheet_width - 160) // 160)
        x = 120 + (index % columns) * 160
        row = index // columns
        base_row_y = 80 + row * 200
        signal_row_y = 96 + row * 200
        orientation = _choose_two_pin_orientation(symbol, canonical_nodes)
        y = base_row_y

        preferred_pin_order = 1
        if canonical_nodes[0] == "0" and canonical_nodes[1] != "0":
            preferred_pin_order = 2
        try:
            _, pin_y = lib.pin_offset(symbol, orientation=orientation, spice_order=preferred_pin_order)
            y = signal_row_y - pin_y
        except Exception:
            y = base_row_y

        placement = ComponentPlacement(
            symbol=symbol,
            reference=ref,
            x=x,
            y=y,
            orientation=orientation,
            value=value,
        )
        components.append((placement, canonical_nodes))
        builder.add_component(placement)

    if not components:
        raise ValueError("No supported components were parsed from netlist.")

    net_to_points: dict[str, list[tuple[int, int]]] = {}
    for placement, nodes in components:
        for spice_order, net_name in enumerate(nodes, start=1):
            try:
                px, py = _component_pin_position(lib, placement, spice_order=spice_order)
            except Exception as exc:
                warnings.append(f"Failed pin mapping for '{placement.reference}' order {spice_order}: {exc}")
                continue
            net_to_points.setdefault(net_name, []).append((px, py))

    multi_net_index = 0
    for net_name, points in net_to_points.items():
        unique_points = sorted(set(points))
        if not unique_points:
            continue
        if net_name == "0":
            for x, y in unique_points:
                builder.add_flag(x, y, "0")
            continue
        if len(unique_points) == 1:
            x, y = unique_points[0]
            builder.add_flag(x, y, net_name)
            continue

        if len(unique_points) == 2:
            point_a, point_b = unique_points
            _route_two_point_net(builder, point_a, point_b)
            label_point = min(unique_points, key=lambda point: (point[0], point[1]))
            builder.add_flag(label_point[0], label_point[1], net_name)
            continue

        trunk_x = min(point[0] for point in unique_points) - 40
        y_min = min(point[1] for point in unique_points)
        lane_y = max(32, y_min - 48 - (multi_net_index * 32))
        multi_net_index += 1
        builder.add_wire(trunk_x, lane_y, max(point[0] for point in unique_points), lane_y)
        for x, y in unique_points:
            if y != lane_y:
                builder.add_wire(x, y, x, lane_y)
        builder.add_flag(trunk_x, lane_y, net_name)

    for idx, directive in enumerate(directives):
        builder.add_directive(48, sheet_height - 140 + idx * 24, directive)

    out_path = _resolve_output_path(workdir=workdir, circuit_name=circuit_name, output_path=output_path)
    builder.write(out_path)
    return {
        "asc_path": str(out_path),
        "components": len(components),
        "nets": len(net_to_points),
        "wires": len(builder.wires),
        "flags": len(builder.flags),
        "directives": len(builder.directives),
        "warnings": warnings,
    }


def build_schematic_from_template(
    *,
    workdir: Path,
    template_name: str,
    parameters: dict[str, Any] | None = None,
    circuit_name: str | None = None,
    output_path: str | None = None,
    sheet_width: int | None = None,
    sheet_height: int | None = None,
    template_path: str | Path | None = None,
    library: SymbolLibrary | None = None,
) -> dict[str, Any]:
    source_path, payload = _load_template_payload(template_path)
    templates = payload.get("templates", [])
    matched: dict[str, Any] | None = None
    for raw in templates:
        if not isinstance(raw, dict):
            continue
        if str(raw.get("name", "")).strip().lower() == template_name.strip().lower():
            matched = raw
            break
    if matched is None:
        raise ValueError(
            f"Template '{template_name}' was not found in {source_path}"
        )

    params = {str(key): value for key, value in (parameters or {}).items()}
    rendered = _format_recursive(matched, params)
    tpl_type = str(rendered.get("type", "netlist")).strip().lower()
    resolved_name = circuit_name or str(rendered.get("circuit_name") or template_name)
    resolved_width = int(
        sheet_width
        if sheet_width is not None
        else rendered.get("sheet_width", 1200 if tpl_type == "netlist" else 880)
    )
    resolved_height = int(
        sheet_height
        if sheet_height is not None
        else rendered.get("sheet_height", 900 if tpl_type == "netlist" else 680)
    )

    if tpl_type == "netlist":
        netlist_content = str(rendered.get("netlist_content", "")).strip()
        if not netlist_content:
            raise ValueError(f"Template '{template_name}' is missing netlist_content")
        result = build_schematic_from_netlist(
            workdir=workdir,
            netlist_content=netlist_content,
            circuit_name=resolved_name,
            output_path=output_path,
            sheet_width=resolved_width,
            sheet_height=resolved_height,
            library=library,
        )
    elif tpl_type == "spec":
        components = rendered.get("components")
        if not isinstance(components, list) or not components:
            raise ValueError(f"Template '{template_name}' is missing components")
        result = build_schematic_from_spec(
            workdir=workdir,
            components=components,
            wires=rendered.get("wires"),
            directives=rendered.get("directives"),
            labels=rendered.get("labels"),
            circuit_name=resolved_name,
            output_path=output_path,
            sheet_width=resolved_width,
            sheet_height=resolved_height,
            library=library,
        )
    else:
        raise ValueError(
            f"Template '{template_name}' has unsupported type '{tpl_type}'. Use netlist or spec."
        )

    result["template_name"] = str(matched.get("name", template_name))
    result["template_type"] = tpl_type
    result["template_path"] = str(source_path)
    result["parameters"] = {key: str(value) for key, value in params.items()}
    return result


def sync_schematic_from_netlist_file(
    *,
    workdir: Path,
    netlist_path: str | Path,
    circuit_name: str | None = None,
    output_path: str | None = None,
    state_path: str | None = None,
    sheet_width: int = 1200,
    sheet_height: int = 900,
    force: bool = False,
    library: SymbolLibrary | None = None,
) -> dict[str, Any]:
    source = Path(netlist_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Netlist file not found: {source}")
    if not source.is_file():
        raise ValueError(f"Netlist path must be a file: {source}")

    resolved_name = circuit_name or source.stem
    asc_path = _resolve_stable_output_path(
        workdir=workdir,
        circuit_name=resolved_name,
        output_path=output_path,
    )
    resolved_state_path = (
        Path(state_path).expanduser().resolve()
        if state_path
        else asc_path.with_suffix(".sync.json")
    )

    netlist_content = source.read_text(encoding="utf-8", errors="ignore")
    source_digest = _sha256_text(netlist_content)
    source_stat = source.stat()
    previous = _read_json(resolved_state_path)

    previous_digest = str(previous.get("source_sha256", ""))
    previous_output = str(previous.get("asc_path", ""))
    output_exists = asc_path.exists()
    changed = (
        force
        or not output_exists
        or previous_digest != source_digest
        or (previous_output and Path(previous_output).expanduser().resolve() != asc_path)
    )
    reason = "unchanged"
    if force:
        reason = "forced"
    elif not output_exists:
        reason = "missing_output"
    elif previous_digest != source_digest:
        reason = "source_changed"
    elif previous_output and Path(previous_output).expanduser().resolve() != asc_path:
        reason = "output_path_changed"

    if changed:
        built = build_schematic_from_netlist(
            workdir=workdir,
            netlist_content=netlist_content,
            circuit_name=resolved_name,
            output_path=str(asc_path),
            sheet_width=sheet_width,
            sheet_height=sheet_height,
            library=library,
        )
        payload = {
            "version": 1,
            "updated_at_epoch_s": time.time(),
            "source_path": str(source),
            "source_sha256": source_digest,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "source_size_bytes": source_stat.st_size,
            "asc_path": built["asc_path"],
            "circuit_name": resolved_name,
            "sheet_width": int(sheet_width),
            "sheet_height": int(sheet_height),
            "components": int(built.get("components", 0)),
            "nets": int(built.get("nets", 0)),
            "wires": int(built.get("wires", 0)),
            "flags": int(built.get("flags", 0)),
            "directives": int(built.get("directives", 0)),
            "warnings": built.get("warnings", []),
        }
        _write_json(resolved_state_path, payload)
        return {
            **built,
            "updated": True,
            "reason": reason,
            "state_path": str(resolved_state_path),
            "source_path": str(source),
            "source_sha256": source_digest,
        }

    return {
        "asc_path": str(asc_path),
        "components": int(previous.get("components", 0)),
        "nets": int(previous.get("nets", 0)),
        "wires": int(previous.get("wires", 0)),
        "flags": int(previous.get("flags", 0)),
        "directives": int(previous.get("directives", 0)),
        "warnings": list(previous.get("warnings", [])),
        "updated": False,
        "reason": reason,
        "state_path": str(resolved_state_path),
        "source_path": str(source),
        "source_sha256": source_digest,
    }


def watch_schematic_from_netlist_file(
    *,
    workdir: Path,
    netlist_path: str | Path,
    circuit_name: str | None = None,
    output_path: str | None = None,
    state_path: str | None = None,
    sheet_width: int = 1200,
    sheet_height: int = 900,
    duration_seconds: float = 10.0,
    poll_interval_seconds: float = 0.5,
    max_updates: int = 20,
    force_initial_refresh: bool = False,
    library: SymbolLibrary | None = None,
) -> dict[str, Any]:
    if duration_seconds < 0:
        raise ValueError("duration_seconds must be >= 0")
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be > 0")
    if max_updates <= 0:
        raise ValueError("max_updates must be > 0")

    started = time.monotonic()
    updates: list[dict[str, Any]] = []
    polls = 0
    last_result: dict[str, Any] | None = None

    while True:
        polls += 1
        result = sync_schematic_from_netlist_file(
            workdir=workdir,
            netlist_path=netlist_path,
            circuit_name=circuit_name,
            output_path=output_path,
            state_path=state_path,
            sheet_width=sheet_width,
            sheet_height=sheet_height,
            force=(force_initial_refresh and polls == 1),
            library=library,
        )
        last_result = result
        if result.get("updated"):
            update_event = dict(result)
            update_event["poll_index"] = polls
            update_event["elapsed_s"] = round(time.monotonic() - started, 6)
            updates.append(update_event)
            if len(updates) >= max_updates:
                break

        elapsed = time.monotonic() - started
        if elapsed >= duration_seconds:
            break
        time.sleep(min(poll_interval_seconds, duration_seconds - elapsed))

    return {
        "source_path": str(Path(netlist_path).expanduser().resolve()),
        "watch_duration_seconds": round(time.monotonic() - started, 6),
        "poll_interval_seconds": float(poll_interval_seconds),
        "poll_count": polls,
        "updates_count": len(updates),
        "updates": updates,
        "last_result": last_result,
    }
