from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zipfile import ZipFile


DEFAULT_LTSPICE_LIB_ZIP = Path("/Applications/LTspice.app/Contents/Resources/lib.zip")


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
    def __init__(self, zip_path: Path = DEFAULT_LTSPICE_LIB_ZIP) -> None:
        self.zip_path = zip_path
        if not self.zip_path.exists():
            raise FileNotFoundError(f"LTspice symbol library not found at {self.zip_path}")
        self._zip = ZipFile(self.zip_path)
        self._index = self._build_index()
        self._cache: dict[str, SymbolDef] = {}

    def _build_index(self) -> dict[str, str]:
        index: dict[str, str] = {}
        for name in self._zip.namelist():
            if not name.lower().endswith(".asy"):
                continue
            base = Path(name).stem.lower()
            index.setdefault(base, name)
            index.setdefault(name.lower(), name)
            index.setdefault(Path(name).as_posix().lower(), name)
        return index

    def _resolve_entry(self, symbol: str) -> str:
        key = symbol.strip().lower()
        if key in self._index:
            return self._index[key]
        if key.endswith(".asy") and key in self._index:
            return self._index[key]
        alt = f"lib/sym/{key}.asy"
        if alt in self._index:
            return self._index[alt]
        raise ValueError(f"Symbol '{symbol}' not found in LTspice lib.zip")

    def get(self, symbol: str) -> SymbolDef:
        cache_key = symbol.strip().lower()
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        entry = self._resolve_entry(symbol)
        text = self._zip.read(entry).decode("utf-8", errors="ignore")
        pins = _parse_symbol_pins(text)
        symbol_def = SymbolDef(symbol=symbol, zip_entry=entry, pins=pins)
        self._cache[cache_key] = symbol_def
        return symbol_def


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
    symbol = library.get(placement.symbol)
    pin = symbol.pin_for_order(spice_order)
    if pin is None:
        raise ValueError(
            f"Symbol '{placement.symbol}' does not expose SpiceOrder {spice_order}"
        )
    dx, dy = _transform_point(pin.x, pin.y, _normalize_orientation(placement.orientation))
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

        value = " ".join(tokens[3:]) if len(tokens) > 3 else None
        index = len(components)
        columns = max(1, (sheet_width - 160) // 160)
        x = 120 + (index % columns) * 160
        y = 80 + (index // columns) * 200
        placement = ComponentPlacement(
            symbol=symbol,
            reference=ref,
            x=x,
            y=y,
            orientation="R0",
            value=value,
        )
        components.append((placement, [_canonical_net(nodes[0]), _canonical_net(nodes[1])]))
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

        trunk_x = min(point[0] for point in unique_points) - 40
        y_min = min(point[1] for point in unique_points)
        y_max = max(point[1] for point in unique_points)
        if y_min != y_max:
            builder.add_wire(trunk_x, y_min, trunk_x, y_max)
        for x, y in unique_points:
            if x != trunk_x:
                builder.add_wire(x, y, trunk_x, y)
        builder.add_flag(trunk_x, y_min, net_name)

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
