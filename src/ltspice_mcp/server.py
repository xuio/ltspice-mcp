from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import mcp.types as types
from mcp.server.fastmcp import FastMCP

from .analysis import (
    compute_bandwidth,
    compute_gain_phase_margin,
    compute_rise_fall_time,
    compute_settling_time,
    find_local_extrema,
    format_series,
    interpolate_series,
    sample_indices,
)
from .ltspice import (
    LTspiceRunner,
    capture_ltspice_window_screenshot,
    find_ltspice_executable,
    get_ltspice_version,
    is_ltspice_ui_running,
    open_in_ltspice_ui,
    tail_text_file,
)
from .models import RawDataset, SimulationRun
from .raw_parser import RawParseError, parse_raw_file
from .schematic import (
    DEFAULT_LTSPICE_LIB_ZIP,
    SymbolLibrary,
    build_schematic_from_netlist,
    build_schematic_from_spec,
    build_schematic_from_template,
    list_schematic_templates,
    sync_schematic_from_netlist_file,
    watch_schematic_from_netlist_file,
)
from .visualization import render_plot_svg, render_schematic_svg, render_symbol_svg


mcp = FastMCP("ltspice-mcp-macos")


def _read_env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


_DEFAULT_WORKDIR = Path(os.getenv("LTSPICE_MCP_WORKDIR", os.getcwd()))
_DEFAULT_TIMEOUT = int(os.getenv("LTSPICE_MCP_TIMEOUT", "120"))
_DEFAULT_BINARY = os.getenv("LTSPICE_BINARY")
_DEFAULT_UI_ENABLED = _read_env_bool("LTSPICE_MCP_UI_ENABLED", default=False)
_DEFAULT_SCHEMATIC_SINGLE_WINDOW = _read_env_bool(
    "LTSPICE_MCP_SCHEMATIC_SINGLE_WINDOW",
    default=True,
)
_DEFAULT_SCHEMATIC_LIVE_PATH = os.getenv("LTSPICE_MCP_SCHEMATIC_LIVE_PATH")

_runner = LTspiceRunner(
    workdir=_DEFAULT_WORKDIR,
    executable=_DEFAULT_BINARY,
    default_timeout_seconds=_DEFAULT_TIMEOUT,
)
_runs: dict[str, SimulationRun] = {}
_run_order: list[str] = []
_loaded_netlist: Path | None = None
_raw_cache: dict[Path, tuple[float, RawDataset]] = {}
_state_path: Path = _DEFAULT_WORKDIR / ".ltspice_mcp_runs.json"
_ui_enabled: bool = _DEFAULT_UI_ENABLED
_schematic_single_window_enabled: bool = _DEFAULT_SCHEMATIC_SINGLE_WINDOW
_schematic_live_path: Path = (
    Path(_DEFAULT_SCHEMATIC_LIVE_PATH).expanduser().resolve()
    if _DEFAULT_SCHEMATIC_LIVE_PATH
    else (_DEFAULT_WORKDIR / ".ui" / "live_schematic.asc").resolve()
)
_symbol_library: SymbolLibrary | None = None
_symbol_library_zip_path: Path | None = None


def _save_run_state() -> None:
    _state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "runs": [_runs[run_id].to_storage_dict() for run_id in _run_order if run_id in _runs],
    }
    tmp_path = _state_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(_state_path)


def _load_run_state() -> None:
    _runs.clear()
    _run_order.clear()
    if not _state_path.exists():
        return

    try:
        payload = json.loads(_state_path.read_text(encoding="utf-8"))
    except Exception:
        return

    entries = payload.get("runs", [])
    if not isinstance(entries, list):
        return

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            run = SimulationRun.from_storage_dict(entry)
        except Exception:
            continue
        _runs[run.run_id] = run
        _run_order.append(run.run_id)


def _register_run(run: SimulationRun) -> SimulationRun:
    _runs[run.run_id] = run
    _run_order.append(run.run_id)
    _save_run_state()
    return run


def _resolve_run(run_id: str | None = None) -> SimulationRun:
    if run_id:
        run = _runs.get(run_id)
        if not run:
            raise ValueError(f"Unknown run_id '{run_id}'")
        return run
    if not _run_order:
        raise ValueError("No simulation has been run yet.")
    return _runs[_run_order[-1]]


def _select_primary_raw(run: SimulationRun) -> Path | None:
    if not run.raw_files:
        return None
    preferred = run.netlist_path.with_suffix(".raw")
    if preferred.exists():
        return preferred
    non_op = [path for path in run.raw_files if not path.name.endswith(".op.raw")]
    if non_op:
        return max(non_op, key=lambda path: path.stat().st_size)
    return max(run.raw_files, key=lambda path: path.stat().st_size)


def _target_path_from_run(run: SimulationRun, target: str) -> Path:
    if target == "netlist":
        return run.netlist_path
    if target == "raw":
        selected = _select_primary_raw(run)
        if not selected:
            raise ValueError(f"Run '{run.run_id}' does not have a RAW file to open.")
        return selected
    if target == "log":
        if run.log_path is None:
            raise ValueError(f"Run '{run.run_id}' does not have a log file.")
        return run.log_path
    raise ValueError("target must be one of: netlist, raw, log")


def _open_ui_target(
    *,
    run: SimulationRun | None = None,
    path: Path | None = None,
    target: str = "netlist",
) -> dict[str, Any]:
    if path is not None:
        return open_in_ltspice_ui(path)
    if run is None:
        raise ValueError("Either run or path must be provided for UI open.")
    return open_in_ltspice_ui(_target_path_from_run(run, target))


def _effective_open_ui(open_ui: bool | None) -> bool:
    return _ui_enabled if open_ui is None else bool(open_ui)


def _guess_image_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".svg":
        return "image/svg+xml"
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def _safe_name(name: str) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in name)
    return cleaned.strip("_") or "image"


def _resolve_png_output_path(
    *,
    kind: str,
    name: str,
    output_path: str | None,
) -> Path:
    if output_path:
        target = Path(output_path).expanduser().resolve()
        if target.suffix.lower() in {".png", ".jpg", ".jpeg", ".tiff"}:
            return target
        return target.with_suffix(".png")
    return (
        _runner.workdir
        / "images"
        / kind
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_safe_name(name)}.png"
    ).resolve()


def _normalize_image_backend(value: str) -> str:
    backend = value.strip().lower()
    if backend not in {"auto", "ltspice", "svg"}:
        raise ValueError("backend must be one of: auto, ltspice, svg")
    return backend


def _image_tool_result(payload: dict[str, Any]) -> types.CallToolResult:
    image_path = payload.get("image_path")
    if not image_path:
        raise ValueError("image payload missing image_path")
    path = Path(str(image_path)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"image_path does not exist: {path}")

    data_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    mime = _guess_image_mime(path)
    content: list[types.TextContent | types.ImageContent] = [
        types.ImageContent(type="image", mimeType=mime, data=data_b64),
        types.TextContent(type="text", text=json.dumps(payload, indent=2)),
    ]
    return types.CallToolResult(content=content, structuredContent=payload, isError=False)


def _resolve_symbol_library(lib_zip_path: str | None = None) -> SymbolLibrary:
    global _symbol_library, _symbol_library_zip_path
    target_path = (
        Path(lib_zip_path).expanduser().resolve()
        if lib_zip_path
        else DEFAULT_LTSPICE_LIB_ZIP.expanduser().resolve()
    )
    if _symbol_library is not None and _symbol_library_zip_path == target_path:
        return _symbol_library
    if _symbol_library is not None:
        _symbol_library.close()
    _symbol_library = SymbolLibrary(target_path)
    _symbol_library_zip_path = target_path
    return _symbol_library


def _sync_schematic_live_file(source_path: Path) -> Path:
    source = source_path.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Schematic file not found: {source}")
    target = _schematic_live_path.expanduser().resolve()
    if target == source:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_suffix(".tmp")
    shutil.copyfile(source, tmp_target)
    tmp_target.replace(target)
    return target


def _open_schematic_ui(path: Path) -> dict[str, Any]:
    requested_path = path.expanduser().resolve()
    if _schematic_single_window_enabled:
        ui_path = _sync_schematic_live_file(requested_path)
        routed = ui_path != requested_path
    else:
        ui_path = requested_path
        routed = False
    event = _open_ui_target(path=ui_path)
    event["single_window_mode"] = _schematic_single_window_enabled
    event["requested_path"] = str(requested_path)
    event["ui_path"] = str(ui_path)
    event["routed_to_single_window"] = routed
    return event


def _resolve_raw_path(raw_path: str | None, run_id: str | None) -> Path:
    if raw_path:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"RAW file not found: {path}")
        return path

    run = _resolve_run(run_id)
    selected = _select_primary_raw(run)
    if not selected:
        raise ValueError(
            f"Run '{run.run_id}' has no RAW files. Check the log or LTspice output."
        )
    return selected


def _load_dataset(path: Path) -> RawDataset:
    cache_key = path.resolve()
    stat = cache_key.stat()
    cached = _raw_cache.get(cache_key)
    if cached and cached[0] == stat.st_mtime:
        return cached[1]

    dataset = parse_raw_file(cache_key)
    _raw_cache[cache_key] = (stat.st_mtime, dataset)
    return dataset


def _resolve_dataset(plot: str | None, run_id: str | None, raw_path: str | None) -> RawDataset:
    if raw_path:
        dataset = _load_dataset(_resolve_raw_path(raw_path, run_id))
        if plot and dataset.plot_name != plot:
            raise ValueError(f"RAW plot '{dataset.plot_name}' does not match requested plot '{plot}'.")
        return dataset

    run = _resolve_run(run_id)
    if plot:
        for candidate in run.raw_files:
            try:
                dataset = _load_dataset(candidate)
            except RawParseError:
                continue
            if dataset.plot_name == plot:
                return dataset
        raise ValueError(
            f"Could not find plot '{plot}' in run '{run.run_id}'. Use getPlotNames first."
        )

    primary = _select_primary_raw(run)
    if not primary:
        raise ValueError(
            f"Run '{run.run_id}' has no RAW files. Check the log or LTspice output."
        )
    return _load_dataset(primary)


def _resolve_step_index(dataset: RawDataset, step_index: int | None) -> int | None:
    if not dataset.steps:
        return None
    if step_index is None:
        return 0
    if step_index < 0 or step_index >= len(dataset.steps):
        raise ValueError(f"step_index must be in range [0, {len(dataset.steps) - 1}]")
    return step_index


def _step_payload(dataset: RawDataset, step_index: int | None) -> dict[str, Any]:
    if not dataset.steps:
        return {
            "step_count": 1,
            "selected_step": 0 if step_index is None else step_index,
            "steps": [{"step_index": 0, "start_index": 0, "end_index": dataset.points, "points": dataset.points}],
        }
    selected = _resolve_step_index(dataset, step_index)
    return {
        "step_count": dataset.step_count,
        "selected_step": selected,
        "steps": [step.as_dict() for step in dataset.steps],
    }


def _run_payload(run: SimulationRun, *, include_output: bool, log_tail_lines: int) -> dict[str, Any]:
    payload = run.as_dict(include_output=include_output)
    payload["log_tail"] = tail_text_file(run.log_path, max_lines=log_tail_lines)
    return payload


def _run_simulation_with_ui(
    *,
    netlist_path: Path,
    ascii_raw: bool,
    timeout_seconds: int | None,
    show_ui: bool | None,
    open_raw_after_run: bool,
) -> tuple[SimulationRun, list[dict[str, Any]], bool]:
    effective_ui = _ui_enabled if show_ui is None else bool(show_ui)
    ui_events: list[dict[str, Any]] = []

    if effective_ui:
        try:
            event = _open_ui_target(path=netlist_path)
            ui_events.append({"stage": "before_run", "target": "netlist", **event})
        except Exception as exc:
            ui_events.append(
                {
                    "stage": "before_run",
                    "target": "netlist",
                    "opened": False,
                    "error": str(exc),
                }
            )

    run = _register_run(
        _runner.run_file(
            netlist_path,
            ascii_raw=ascii_raw,
            timeout_seconds=timeout_seconds,
        )
    )

    if effective_ui and open_raw_after_run:
        try:
            event = _open_ui_target(run=run, target="raw")
            ui_events.append({"stage": "after_run", "target": "raw", **event})
        except Exception as exc:
            ui_events.append(
                {
                    "stage": "after_run",
                    "target": "raw",
                    "opened": False,
                    "error": str(exc),
                }
            )

    return run, ui_events, effective_ui


def _sanitize_representation(value: str) -> str:
    valid = {"auto", "real", "rectangular", "magnitude-phase", "both"}
    if value not in valid:
        raise ValueError(f"representation must be one of: {', '.join(sorted(valid))}")
    return value


_load_run_state()


@mcp.tool()
def getLtspiceStatus() -> dict[str, Any]:
    """Get LTspice executable status and server configuration."""
    executable = _runner.executable or find_ltspice_executable()
    version = get_ltspice_version(executable) if executable else None
    return {
        "ltspice_executable": str(executable) if executable else None,
        "ltspice_version": version,
        "workdir": str(_runner.workdir),
        "default_timeout_seconds": _runner.default_timeout_seconds,
        "runs_recorded": len(_run_order),
        "run_state_path": str(_state_path),
        "ltspice_lib_zip_path": str(DEFAULT_LTSPICE_LIB_ZIP.expanduser().resolve()),
        "ui_enabled_default": _ui_enabled,
        "schematic_single_window_default": _schematic_single_window_enabled,
        "schematic_live_path": str(_schematic_live_path),
    }


@mcp.tool()
def getLtspiceUiStatus() -> dict[str, Any]:
    """Return whether LTspice UI appears to be running on this machine."""
    return {
        "ui_enabled_default": _ui_enabled,
        "schematic_single_window_default": _schematic_single_window_enabled,
        "schematic_live_path": str(_schematic_live_path),
        "ui_running": is_ltspice_ui_running(),
        "runs_recorded": len(_run_order),
        "latest_run_id": _run_order[-1] if _run_order else None,
    }


@mcp.tool()
def getLtspiceLibraryStatus(lib_zip_path: str | None = None) -> dict[str, Any]:
    """Get LTspice symbol library ZIP status and basic counts."""
    zip_path = (
        Path(lib_zip_path).expanduser().resolve()
        if lib_zip_path
        else DEFAULT_LTSPICE_LIB_ZIP.expanduser().resolve()
    )
    exists = zip_path.exists()
    if not exists:
        return {
            "zip_path": str(zip_path),
            "exists": False,
            "entries_count": 0,
            "symbols_count": 0,
        }

    library = _resolve_symbol_library(str(zip_path))
    entries = library.list_entries(limit=1_000_000)
    symbols = library.list_symbols(limit=1_000_000)
    return {
        "zip_path": str(zip_path),
        "exists": True,
        "entries_count": len(entries),
        "symbols_count": len(symbols),
    }


@mcp.tool()
def listLtspiceLibraryEntries(
    query: str | None = None,
    limit: int = 200,
    lib_zip_path: str | None = None,
) -> dict[str, Any]:
    """List LTspice symbol ZIP entries (.asy), optionally filtered by query."""
    if limit <= 0:
        return {"zip_path": str(DEFAULT_LTSPICE_LIB_ZIP), "entries": [], "returned_count": 0}

    safe_limit = min(int(limit), 5000)
    library = _resolve_symbol_library(lib_zip_path)
    entries = library.list_entries(query=query, limit=safe_limit)
    return {
        "zip_path": str(library.zip_path),
        "query": query,
        "limit": safe_limit,
        "returned_count": len(entries),
        "entries": entries,
    }


@mcp.tool()
def listLtspiceSymbols(
    query: str | None = None,
    library: str | None = None,
    limit: int = 200,
    lib_zip_path: str | None = None,
) -> dict[str, Any]:
    """List/search LTspice symbols parsed from the symbol library ZIP."""
    if limit <= 0:
        return {"symbols": [], "returned_count": 0}

    safe_limit = min(int(limit), 5000)
    symbol_lib = _resolve_symbol_library(lib_zip_path)
    symbols = symbol_lib.list_symbols(
        query=query,
        library=library,
        limit=safe_limit,
    )
    return {
        "zip_path": str(symbol_lib.zip_path),
        "query": query,
        "library": library,
        "limit": safe_limit,
        "returned_count": len(symbols),
        "symbols": symbols,
    }


@mcp.tool()
def getLtspiceSymbolInfo(
    symbol: str,
    include_source: bool = False,
    source_max_chars: int = 8000,
    lib_zip_path: str | None = None,
) -> dict[str, Any]:
    """Get pin map and metadata for a symbol in LTspice's lib.zip."""
    if not symbol.strip():
        raise ValueError("symbol must be a non-empty string")

    symbol_lib = _resolve_symbol_library(lib_zip_path)
    payload = symbol_lib.symbol_info(symbol)

    if include_source:
        raw_source = symbol_lib.read_symbol_source(symbol)
        max_chars = max(0, int(source_max_chars))
        if max_chars and len(raw_source) > max_chars:
            payload["source"] = raw_source[:max_chars]
            payload["source_truncated"] = True
        else:
            payload["source"] = raw_source
            payload["source_truncated"] = False
        payload["source_line_count"] = len(raw_source.splitlines())

    return payload


@mcp.tool()
def renderLtspiceSymbolImage(
    symbol: str,
    output_path: str | None = None,
    width: int = 640,
    height: int = 420,
    downscale_factor: float = 1.0,
    backend: str = "auto",
    settle_seconds: float = 1.0,
    include_pins: bool = True,
    include_pin_labels: bool = True,
    lib_zip_path: str | None = None,
) -> types.CallToolResult:
    """
    Render an LTspice symbol to an image and return the image through MCP.

    The response includes both image content and structured metadata (image_path, bounds, etc.).
    """
    if not symbol.strip():
        raise ValueError("symbol must be a non-empty string")

    normalized_backend = _normalize_image_backend(backend)
    warnings: list[str] = []

    if normalized_backend in {"auto", "ltspice"}:
        try:
            preview = build_schematic_from_spec(
                workdir=_runner.workdir,
                components=[
                    {
                        "symbol": symbol,
                        "reference": "X1",
                        "x": 240,
                        "y": 180,
                        "value": symbol,
                    }
                ],
                circuit_name=f"symbol_preview_{symbol}",
                sheet_width=max(600, width),
                sheet_height=max(420, height),
            )
            screenshot_payload = capture_ltspice_window_screenshot(
                output_path=_resolve_png_output_path(
                    kind="symbols",
                    name=symbol,
                    output_path=output_path,
                ),
                open_path=preview["asc_path"],
                title_hint=f"symbol_preview_{symbol}",
                settle_seconds=settle_seconds,
                downscale_factor=downscale_factor,
                avoid_space_switch=True,
                prefer_screencapturekit=True,
            )
            payload = {
                **screenshot_payload,
                "symbol": symbol,
                "preview_asc_path": preview["asc_path"],
                "backend_requested": normalized_backend,
                "backend_used": "ltspice",
            }
            return _image_tool_result(payload)
        except Exception as exc:
            if normalized_backend == "ltspice":
                raise
            warnings.append(f"ltspice backend failed; fell back to svg: {exc}")

    symbol_lib = _resolve_symbol_library(lib_zip_path)
    svg_output_path = output_path
    if output_path and Path(output_path).suffix.lower() not in {".svg"}:
        svg_output_path = str(Path(output_path).with_suffix(".svg"))
        warnings.append("output_path suffix adjusted to .svg for svg backend")
    payload = render_symbol_svg(
        workdir=_runner.workdir,
        symbol=symbol,
        output_path=svg_output_path,
        width=width,
        height=height,
        downscale_factor=downscale_factor,
        include_pins=include_pins,
        include_pin_labels=include_pin_labels,
        library=symbol_lib,
    )
    payload["backend_requested"] = normalized_backend
    payload["backend_used"] = "svg"
    if warnings:
        payload["warnings"] = warnings
    return _image_tool_result(payload)


@mcp.tool()
def renderLtspiceSchematicImage(
    asc_path: str,
    output_path: str | None = None,
    width: int = 1400,
    height: int = 900,
    downscale_factor: float = 1.0,
    backend: str = "auto",
    settle_seconds: float = 1.0,
    include_symbol_graphics: bool = True,
    lib_zip_path: str | None = None,
) -> types.CallToolResult:
    """
    Render an LTspice schematic (.asc) to an image and return it through MCP.

    `downscale_factor` lets clients request smaller rendered images.
    """
    normalized_backend = _normalize_image_backend(backend)
    asc_resolved = Path(asc_path).expanduser().resolve()
    warnings: list[str] = []

    if normalized_backend in {"auto", "ltspice"}:
        try:
            screenshot_payload = capture_ltspice_window_screenshot(
                output_path=_resolve_png_output_path(
                    kind="schematics",
                    name=asc_resolved.stem,
                    output_path=output_path,
                ),
                open_path=asc_resolved,
                title_hint=asc_resolved.name,
                settle_seconds=settle_seconds,
                downscale_factor=downscale_factor,
                avoid_space_switch=True,
                prefer_screencapturekit=True,
            )
            payload = {
                **screenshot_payload,
                "asc_path": str(asc_resolved),
                "backend_requested": normalized_backend,
                "backend_used": "ltspice",
            }
            return _image_tool_result(payload)
        except Exception as exc:
            if normalized_backend == "ltspice":
                raise
            warnings.append(f"ltspice backend failed; fell back to svg: {exc}")

    symbol_lib = _resolve_symbol_library(lib_zip_path) if include_symbol_graphics else None
    svg_output_path = output_path
    if output_path and Path(output_path).suffix.lower() not in {".svg"}:
        svg_output_path = str(Path(output_path).with_suffix(".svg"))
        warnings.append("output_path suffix adjusted to .svg for svg backend")
    payload = render_schematic_svg(
        workdir=_runner.workdir,
        asc_path=asc_resolved,
        output_path=svg_output_path,
        width=width,
        height=height,
        downscale_factor=downscale_factor,
        include_symbol_graphics=include_symbol_graphics,
        library=symbol_lib,
    )
    payload["backend_requested"] = normalized_backend
    payload["backend_used"] = "svg"
    if warnings:
        payload["warnings"] = warnings
    return _image_tool_result(payload)


@mcp.tool()
def renderLtspicePlotImage(
    vectors: list[str],
    plot: str | None = None,
    run_id: str | None = None,
    raw_path: str | None = None,
    step_index: int | None = None,
    output_path: str | None = None,
    width: int = 1280,
    height: int = 720,
    downscale_factor: float = 1.0,
    backend: str = "auto",
    settle_seconds: float = 1.0,
    max_points: int = 2000,
    y_mode: str = "magnitude",
    x_log: bool | None = None,
    title: str | None = None,
) -> types.CallToolResult:
    """
    Render one or more vectors from a RAW dataset to a plot image and return it through MCP.

    Supports run_id/raw_path resolution and optional step filtering for stepped runs.
    """
    if not vectors:
        raise ValueError("vectors must contain at least one vector name")
    dataset = _resolve_dataset(plot=plot, run_id=run_id, raw_path=raw_path)
    selected_step = _resolve_step_index(dataset, step_index)
    normalized_backend = _normalize_image_backend(backend)
    warnings: list[str] = []

    if normalized_backend in {"auto", "ltspice"}:
        try:
            screenshot_payload = capture_ltspice_window_screenshot(
                output_path=_resolve_png_output_path(
                    kind="plots",
                    name=f"{dataset.path.stem}_{vectors[0]}",
                    output_path=output_path,
                ),
                open_path=dataset.path,
                title_hint=dataset.path.name,
                settle_seconds=settle_seconds,
                downscale_factor=downscale_factor,
                avoid_space_switch=True,
                prefer_screencapturekit=True,
            )
            payload = {
                **screenshot_payload,
                "raw_path": str(dataset.path),
                "plot_name": dataset.plot_name,
                "vectors": vectors,
                "backend_requested": normalized_backend,
                "backend_used": "ltspice",
                **_step_payload(dataset, selected_step),
            }
            return _image_tool_result(payload)
        except Exception as exc:
            if normalized_backend == "ltspice":
                raise
            warnings.append(f"ltspice backend failed; fell back to svg: {exc}")

    svg_output_path = output_path
    if output_path and Path(output_path).suffix.lower() not in {".svg"}:
        svg_output_path = str(Path(output_path).with_suffix(".svg"))
        warnings.append("output_path suffix adjusted to .svg for svg backend")
    payload = render_plot_svg(
        workdir=_runner.workdir,
        dataset=dataset,
        vectors=vectors,
        step_index=selected_step,
        output_path=svg_output_path,
        width=width,
        height=height,
        downscale_factor=downscale_factor,
        max_points=max_points,
        y_mode=y_mode,
        x_log=x_log,
        title=title,
    )
    payload["backend_requested"] = normalized_backend
    payload["backend_used"] = "svg"
    if warnings:
        payload["warnings"] = warnings
    payload.update(_step_payload(dataset, selected_step))
    return _image_tool_result(payload)


@mcp.tool()
def setLtspiceUiEnabled(enabled: bool) -> dict[str, Any]:
    """Set default UI behavior for simulation calls where show_ui is omitted."""
    global _ui_enabled
    _ui_enabled = bool(enabled)
    return {"ui_enabled_default": _ui_enabled}


@mcp.tool()
def setSchematicUiSingleWindow(enabled: bool) -> dict[str, Any]:
    """Set whether schematic UI opens reuse a single live schematic path."""
    global _schematic_single_window_enabled
    _schematic_single_window_enabled = bool(enabled)
    return {
        "schematic_single_window_default": _schematic_single_window_enabled,
        "schematic_live_path": str(_schematic_live_path),
    }


@mcp.tool()
def openLtspiceUi(
    run_id: str | None = None,
    path: str | None = None,
    target: str = "netlist",
) -> dict[str, Any]:
    """
    Open LTspice UI on a selected artifact.

    If `path` is provided it is opened directly.
    Otherwise, the path is resolved from `run_id` and `target`:
    - target=netlist|raw|log
    """
    if path:
        return _open_ui_target(path=Path(path).expanduser().resolve())
    run = _resolve_run(run_id)
    return _open_ui_target(run=run, target=target)


@mcp.tool()
def createSchematic(
    components: list[dict[str, Any]],
    wires: list[dict[str, Any]] | None = None,
    directives: list[dict[str, Any] | str] | None = None,
    labels: list[dict[str, Any]] | None = None,
    circuit_name: str = "schematic",
    output_path: str | None = None,
    sheet_width: int = 880,
    sheet_height: int = 680,
    open_ui: bool | None = None,
) -> dict[str, Any]:
    """
    Create an LTspice .asc schematic from structured component/wire/directive data.

    Components must include: symbol, reference, x, y.
    Optional component fields: value, orientation|rotation, attributes, pin_nets.
    """
    if not components:
        raise ValueError("components must contain at least one component")

    result = build_schematic_from_spec(
        workdir=_runner.workdir,
        components=components,
        wires=wires,
        directives=directives,
        labels=labels,
        circuit_name=circuit_name,
        output_path=output_path,
        sheet_width=sheet_width,
        sheet_height=sheet_height,
    )

    should_open = _effective_open_ui(open_ui)
    if should_open:
        try:
            result["ui"] = _open_schematic_ui(Path(result["asc_path"]))
        except Exception as exc:
            result["ui"] = {"opened": False, "error": str(exc)}
    result["ui_enabled"] = should_open
    result["single_window_mode"] = _schematic_single_window_enabled
    return result


@mcp.tool()
def createSchematicFromNetlist(
    netlist_content: str,
    circuit_name: str = "schematic_from_netlist",
    output_path: str | None = None,
    sheet_width: int = 1200,
    sheet_height: int = 900,
    open_ui: bool | None = None,
) -> dict[str, Any]:
    """
    Create an LTspice .asc schematic from a SPICE netlist using simple auto-placement/routing.

    Currently supports common two-pin primitives (R, C, L, D, V, I).
    """
    result = build_schematic_from_netlist(
        workdir=_runner.workdir,
        netlist_content=netlist_content,
        circuit_name=circuit_name,
        output_path=output_path,
        sheet_width=sheet_width,
        sheet_height=sheet_height,
    )

    should_open = _effective_open_ui(open_ui)
    if should_open:
        try:
            result["ui"] = _open_schematic_ui(Path(result["asc_path"]))
        except Exception as exc:
            result["ui"] = {"opened": False, "error": str(exc)}
    result["ui_enabled"] = should_open
    result["single_window_mode"] = _schematic_single_window_enabled
    return result


@mcp.tool()
def listSchematicTemplates(template_path: str | None = None) -> dict[str, Any]:
    """List available schematic templates from the built-in or user-provided JSON file."""
    return list_schematic_templates(template_path=template_path)


@mcp.tool()
def createSchematicFromTemplate(
    template_name: str,
    parameters: dict[str, Any] | None = None,
    circuit_name: str | None = None,
    output_path: str | None = None,
    sheet_width: int | None = None,
    sheet_height: int | None = None,
    template_path: str | None = None,
    open_ui: bool | None = None,
) -> dict[str, Any]:
    """
    Create an LTspice .asc schematic from a JSON template.

    Templates support type=netlist (auto-layout) and type=spec (explicit placement).
    """
    result = build_schematic_from_template(
        workdir=_runner.workdir,
        template_name=template_name,
        parameters=parameters,
        circuit_name=circuit_name,
        output_path=output_path,
        sheet_width=sheet_width,
        sheet_height=sheet_height,
        template_path=template_path,
    )

    should_open = _effective_open_ui(open_ui)
    if should_open:
        try:
            result["ui"] = _open_schematic_ui(Path(result["asc_path"]))
        except Exception as exc:
            result["ui"] = {"opened": False, "error": str(exc)}
    result["ui_enabled"] = should_open
    result["single_window_mode"] = _schematic_single_window_enabled
    return result


@mcp.tool()
def syncSchematicFromNetlistFile(
    netlist_path: str,
    circuit_name: str | None = None,
    output_path: str | None = None,
    state_path: str | None = None,
    sheet_width: int = 1200,
    sheet_height: int = 900,
    force: bool = False,
    open_ui: bool | None = None,
) -> dict[str, Any]:
    """
    Regenerate schematic from a netlist file only when source content changed.

    Stores sync metadata in JSON so repeated calls are fast and deterministic.
    """
    result = sync_schematic_from_netlist_file(
        workdir=_runner.workdir,
        netlist_path=netlist_path,
        circuit_name=circuit_name,
        output_path=output_path,
        state_path=state_path,
        sheet_width=sheet_width,
        sheet_height=sheet_height,
        force=force,
    )

    should_open = _effective_open_ui(open_ui)
    if should_open and result.get("updated"):
        try:
            result["ui"] = _open_schematic_ui(Path(result["asc_path"]))
        except Exception as exc:
            result["ui"] = {"opened": False, "error": str(exc)}
    result["ui_enabled"] = should_open
    result["single_window_mode"] = _schematic_single_window_enabled
    return result


@mcp.tool()
def watchSchematicFromNetlistFile(
    netlist_path: str,
    circuit_name: str | None = None,
    output_path: str | None = None,
    state_path: str | None = None,
    sheet_width: int = 1200,
    sheet_height: int = 900,
    duration_seconds: float = 10.0,
    poll_interval_seconds: float = 0.5,
    max_updates: int = 20,
    force_initial_refresh: bool = False,
    open_ui: bool | None = None,
) -> dict[str, Any]:
    """
    Poll a netlist file and regenerate schematic whenever the netlist changes.

    Returns update events for each rebuild detected during the watch interval.
    """
    result = watch_schematic_from_netlist_file(
        workdir=_runner.workdir,
        netlist_path=netlist_path,
        circuit_name=circuit_name,
        output_path=output_path,
        state_path=state_path,
        sheet_width=sheet_width,
        sheet_height=sheet_height,
        duration_seconds=duration_seconds,
        poll_interval_seconds=poll_interval_seconds,
        max_updates=max_updates,
        force_initial_refresh=force_initial_refresh,
    )

    should_open = _effective_open_ui(open_ui)
    if should_open:
        latest_path = None
        if result.get("updates"):
            latest_path = result["updates"][-1].get("asc_path")
        elif isinstance(result.get("last_result"), dict):
            latest_path = result["last_result"].get("asc_path")
        if latest_path:
            try:
                result["ui"] = _open_schematic_ui(Path(latest_path))
            except Exception as exc:
                result["ui"] = {"opened": False, "error": str(exc)}
    result["ui_enabled"] = should_open
    result["single_window_mode"] = _schematic_single_window_enabled
    return result


@mcp.tool()
def loadCircuit(netlist: str, circuit_name: str = "circuit") -> dict[str, Any]:
    """Create a netlist file in the MCP workdir and mark it as the currently loaded circuit."""
    global _loaded_netlist
    path = _runner.write_netlist(netlist, circuit_name=circuit_name)
    _loaded_netlist = path
    response: dict[str, Any] = {"netlist_path": str(path), "loaded": True}
    try:
        schematic = build_schematic_from_netlist(
            workdir=_runner.workdir,
            netlist_content=netlist,
            circuit_name=f"{circuit_name}_schematic",
            output_path=str(path.with_suffix(".asc")),
            sheet_width=1200,
            sheet_height=900,
        )
        response["asc_path"] = schematic["asc_path"]
        response["schematic"] = schematic
    except Exception as exc:
        response["schematic_error"] = str(exc)
    return response


@mcp.tool()
def loadNetlistFromFile(filepath: str) -> dict[str, Any]:
    """Load an existing .cir/.net/.asc file as the current circuit."""
    global _loaded_netlist
    path = Path(filepath).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Netlist file not found: {path}")
    _loaded_netlist = path
    return {"netlist_path": str(path), "loaded": True}


@mcp.tool()
def runSimulation(
    command: str = "",
    ascii_raw: bool = False,
    timeout_seconds: int | None = None,
    show_ui: bool | None = None,
    open_raw_after_run: bool = False,
) -> dict[str, Any]:
    """
    Run LTspice in batch mode for the currently loaded netlist.

    The ngspice-style `command` parameter is accepted for compatibility, but LTspice ignores it.
    """
    if _loaded_netlist is None:
        raise ValueError("No netlist is loaded. Use loadCircuit or loadNetlistFromFile first.")

    run, ui_events, effective_ui = _run_simulation_with_ui(
        netlist_path=_loaded_netlist,
        ascii_raw=ascii_raw,
        timeout_seconds=timeout_seconds,
        show_ui=show_ui,
        open_raw_after_run=open_raw_after_run,
    )
    notes: list[str] = []
    if command.strip():
        notes.append(
            "LTspice batch mode ignores ngspice commands; use directives (.tran/.ac/.op) in the netlist."
        )
    response = _run_payload(run, include_output=False, log_tail_lines=120)
    if notes:
        response["notes"] = notes
    response["ui_enabled"] = effective_ui
    if ui_events:
        response["ui_events"] = ui_events
    return response


@mcp.tool()
def simulateNetlist(
    netlist_content: str,
    circuit_name: str = "circuit",
    ascii_raw: bool = False,
    timeout_seconds: int | None = None,
    show_ui: bool | None = None,
    open_raw_after_run: bool = False,
) -> dict[str, Any]:
    """Write a netlist and run LTspice batch simulation in one call."""
    global _loaded_netlist
    netlist_path = _runner.write_netlist(netlist_content, circuit_name=circuit_name)
    _loaded_netlist = netlist_path
    run, ui_events, effective_ui = _run_simulation_with_ui(
        netlist_path=netlist_path,
        ascii_raw=ascii_raw,
        timeout_seconds=timeout_seconds,
        show_ui=show_ui,
        open_raw_after_run=open_raw_after_run,
    )
    response = _run_payload(run, include_output=False, log_tail_lines=120)
    response["ui_enabled"] = effective_ui
    if ui_events:
        response["ui_events"] = ui_events
    return response


@mcp.tool()
def simulateNetlistFile(
    netlist_path: str,
    ascii_raw: bool = False,
    timeout_seconds: int | None = None,
    show_ui: bool | None = None,
    open_raw_after_run: bool = False,
) -> dict[str, Any]:
    """Run LTspice batch simulation for an existing netlist path."""
    global _loaded_netlist
    path = Path(netlist_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Netlist file not found: {path}")
    _loaded_netlist = path
    run, ui_events, effective_ui = _run_simulation_with_ui(
        netlist_path=path,
        ascii_raw=ascii_raw,
        timeout_seconds=timeout_seconds,
        show_ui=show_ui,
        open_raw_after_run=open_raw_after_run,
    )
    response = _run_payload(run, include_output=False, log_tail_lines=120)
    response["ui_enabled"] = effective_ui
    if ui_events:
        response["ui_events"] = ui_events
    return response


@mcp.tool()
def listRuns(limit: int = 20) -> list[dict[str, Any]]:
    """List recent simulation runs (newest first)."""
    if limit <= 0:
        return []
    selected_ids = list(reversed(_run_order[-limit:]))
    return [_runs[run_id].as_dict(include_output=False) for run_id in selected_ids]


@mcp.tool()
def getRunDetails(
    run_id: str | None = None,
    include_output: bool = True,
    log_tail_lines: int = 200,
) -> dict[str, Any]:
    """Get full details for a run_id, or the latest run if omitted."""
    run = _resolve_run(run_id)
    return _run_payload(run, include_output=include_output, log_tail_lines=log_tail_lines)


@mcp.tool()
def getPlotNames(run_id: str | None = None, raw_path: str | None = None) -> dict[str, Any]:
    """List LTspice plot names available in one RAW file or all RAW files from a run."""
    if raw_path:
        dataset = _resolve_dataset(plot=None, run_id=run_id, raw_path=raw_path)
        return {
            "run_id": run_id,
            "plots": [
                {
                    "plot_name": dataset.plot_name,
                    "raw_path": str(dataset.path),
                    "points": dataset.points,
                    "step_count": dataset.step_count,
                }
            ],
        }

    run = _resolve_run(run_id)
    plots: list[dict[str, Any]] = []
    for candidate in run.raw_files:
        try:
            dataset = _load_dataset(candidate)
            plots.append(
                {
                    "plot_name": dataset.plot_name,
                    "raw_path": str(candidate),
                    "points": dataset.points,
                    "step_count": dataset.step_count,
                }
            )
        except RawParseError as exc:
            plots.append({"plot_name": "ERROR", "raw_path": str(candidate), "error": str(exc)})
    return {"run_id": run.run_id, "plots": plots}


@mcp.tool()
def getVectorsInfo(
    plot: str | None = None,
    run_id: str | None = None,
    raw_path: str | None = None,
    step_index: int | None = None,
) -> dict[str, Any]:
    """Get detailed information about vectors in a plot."""
    dataset = _resolve_dataset(plot=plot, run_id=run_id, raw_path=raw_path)
    selected_step = _resolve_step_index(dataset, step_index)
    vectors: list[dict[str, Any]] = []
    for variable in dataset.variables:
        series = dataset.get_vector(variable.name, step_index=selected_step)
        vectors.append(
            {
                "index": variable.index,
                "name": variable.name,
                "kind": variable.kind,
                "is_scale": dataset.has_natural_scale() and variable.index == 0,
                "is_complex": any(abs(value.imag) > 0.0 for value in series),
            }
        )
    return {
        "plot_name": dataset.plot_name,
        "raw_path": str(dataset.path),
        "points": len(dataset.scale_values(step_index=selected_step)),
        "total_points": dataset.points,
        "flags": sorted(dataset.flags),
        "vectors": vectors,
        **_step_payload(dataset, selected_step),
    }


@mcp.tool()
def getVectorData(
    vectors: list[str],
    plot: str | None = None,
    run_id: str | None = None,
    raw_path: str | None = None,
    step_index: int | None = None,
    points: list[float] | None = None,
    representation: str = "auto",
    max_points: int = 400,
) -> dict[str, Any]:
    """Get data for vectors, optionally interpolated at explicit scale points."""
    if not vectors:
        raise ValueError("vectors must contain at least one vector name")
    representation = _sanitize_representation(representation)

    dataset = _resolve_dataset(plot=plot, run_id=run_id, raw_path=raw_path)
    selected_step = _resolve_step_index(dataset, step_index)
    scale = dataset.scale_values(step_index=selected_step)

    if points is not None:
        sampled_scale = points
        sampled_vectors: dict[str, list[complex]] = {}
        for vector_name in vectors:
            sampled_vectors[vector_name] = interpolate_series(
                scale=scale,
                series=dataset.get_vector(vector_name, step_index=selected_step),
                points=points,
            )
    else:
        idx = sample_indices(len(scale), max_points=max_points)
        sampled_scale = [scale[i] for i in idx]
        sampled_vectors = {
            vector_name: [dataset.get_vector(vector_name, step_index=selected_step)[i] for i in idx]
            for vector_name in vectors
        }

    payload_vectors: dict[str, Any] = {}
    for vector_name, series in sampled_vectors.items():
        payload_vectors[vector_name] = format_series(
            series,
            representation=representation,
            prefer_real=True,
        )

    return {
        "plot_name": dataset.plot_name,
        "raw_path": str(dataset.path),
        "scale_name": dataset.scale_name,
        "scale_points": sampled_scale,
        "vectors": payload_vectors,
        **_step_payload(dataset, selected_step),
    }


@mcp.tool()
def getLocalExtrema(
    vectors: list[str],
    plot: str | None = None,
    run_id: str | None = None,
    raw_path: str | None = None,
    step_index: int | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get local minima/maxima for vectors."""
    if not vectors:
        raise ValueError("vectors must contain at least one vector name")

    options = options or {}
    include_minima = bool(options.get("minima", True))
    include_maxima = bool(options.get("maxima", True))
    threshold = float(options.get("threshold", 0.0))
    max_results = int(options.get("max_results", 200))

    dataset = _resolve_dataset(plot=plot, run_id=run_id, raw_path=raw_path)
    selected_step = _resolve_step_index(dataset, step_index)
    scale = dataset.scale_values(step_index=selected_step)
    extrema_by_vector: dict[str, Any] = {}

    for vector_name in vectors:
        extrema_by_vector[vector_name] = find_local_extrema(
            scale=scale,
            series=dataset.get_vector(vector_name, step_index=selected_step),
            include_minima=include_minima,
            include_maxima=include_maxima,
            threshold=threshold,
            max_results=max_results,
        )

    return {
        "plot_name": dataset.plot_name,
        "raw_path": str(dataset.path),
        "extrema": extrema_by_vector,
        **_step_payload(dataset, selected_step),
    }


@mcp.tool()
def getBandwidth(
    vector: str,
    plot: str | None = None,
    run_id: str | None = None,
    raw_path: str | None = None,
    step_index: int | None = None,
    reference: str = "first",
    drop_db: float = 3.0,
) -> dict[str, Any]:
    """Compute -drop_db bandwidth for an AC response vector."""
    dataset = _resolve_dataset(plot=plot, run_id=run_id, raw_path=raw_path)
    selected_step = _resolve_step_index(dataset, step_index)
    if dataset.scale_name.lower() != "frequency":
        raise ValueError("Bandwidth requires a frequency-domain plot (AC analysis).")

    result = compute_bandwidth(
        frequency_hz=dataset.scale_values(step_index=selected_step),
        response=dataset.get_vector(vector, step_index=selected_step),
        reference=reference,
        drop_db=drop_db,
    )
    return {
        "plot_name": dataset.plot_name,
        "raw_path": str(dataset.path),
        "vector": vector,
        **result,
        **_step_payload(dataset, selected_step),
    }


@mcp.tool()
def getGainPhaseMargin(
    vector: str,
    plot: str | None = None,
    run_id: str | None = None,
    raw_path: str | None = None,
    step_index: int | None = None,
) -> dict[str, Any]:
    """Compute gain and phase margins from an AC loop-gain vector."""
    dataset = _resolve_dataset(plot=plot, run_id=run_id, raw_path=raw_path)
    selected_step = _resolve_step_index(dataset, step_index)
    if dataset.scale_name.lower() != "frequency":
        raise ValueError("Gain/phase margin requires a frequency-domain plot (AC analysis).")

    result = compute_gain_phase_margin(
        frequency_hz=dataset.scale_values(step_index=selected_step),
        response=dataset.get_vector(vector, step_index=selected_step),
    )
    return {
        "plot_name": dataset.plot_name,
        "raw_path": str(dataset.path),
        "vector": vector,
        **result,
        **_step_payload(dataset, selected_step),
    }


@mcp.tool()
def getRiseFallTime(
    vector: str,
    plot: str | None = None,
    run_id: str | None = None,
    raw_path: str | None = None,
    step_index: int | None = None,
    low_threshold_pct: float = 10.0,
    high_threshold_pct: float = 90.0,
) -> dict[str, Any]:
    """Compute first rise/fall times for a transient response."""
    dataset = _resolve_dataset(plot=plot, run_id=run_id, raw_path=raw_path)
    selected_step = _resolve_step_index(dataset, step_index)
    if dataset.scale_name.lower() != "time":
        raise ValueError("Rise/fall time requires a time-domain plot (transient analysis).")

    result = compute_rise_fall_time(
        time_s=dataset.scale_values(step_index=selected_step),
        signal=dataset.get_vector(vector, step_index=selected_step),
        low_threshold_pct=low_threshold_pct,
        high_threshold_pct=high_threshold_pct,
    )
    return {
        "plot_name": dataset.plot_name,
        "raw_path": str(dataset.path),
        "vector": vector,
        **result,
        **_step_payload(dataset, selected_step),
    }


@mcp.tool()
def getSettlingTime(
    vector: str,
    plot: str | None = None,
    run_id: str | None = None,
    raw_path: str | None = None,
    step_index: int | None = None,
    tolerance_percent: float = 2.0,
    target_value: float | None = None,
) -> dict[str, Any]:
    """Compute settling time to a target within tolerance_percent band."""
    dataset = _resolve_dataset(plot=plot, run_id=run_id, raw_path=raw_path)
    selected_step = _resolve_step_index(dataset, step_index)
    if dataset.scale_name.lower() != "time":
        raise ValueError("Settling time requires a time-domain plot (transient analysis).")

    result = compute_settling_time(
        time_s=dataset.scale_values(step_index=selected_step),
        signal=dataset.get_vector(vector, step_index=selected_step),
        tolerance_percent=tolerance_percent,
        target_value=target_value,
    )
    return {
        "plot_name": dataset.plot_name,
        "raw_path": str(dataset.path),
        "vector": vector,
        **result,
        **_step_payload(dataset, selected_step),
    }


def _configure_runner(
    *,
    workdir: Path,
    ltspice_binary: str | None,
    timeout: int,
    ui_enabled: bool | None = None,
    schematic_single_window: bool | None = None,
    schematic_live_path: str | None = None,
) -> None:
    global _runner, _loaded_netlist, _raw_cache, _state_path, _ui_enabled
    global _schematic_single_window_enabled, _schematic_live_path
    global _symbol_library, _symbol_library_zip_path
    _runner = LTspiceRunner(
        workdir=workdir,
        executable=ltspice_binary,
        default_timeout_seconds=timeout,
    )
    _state_path = workdir / ".ltspice_mcp_runs.json"
    _loaded_netlist = None
    _raw_cache = {}
    if _symbol_library is not None:
        _symbol_library.close()
    _symbol_library = None
    _symbol_library_zip_path = None
    _ui_enabled = _DEFAULT_UI_ENABLED if ui_enabled is None else bool(ui_enabled)
    _schematic_single_window_enabled = (
        _DEFAULT_SCHEMATIC_SINGLE_WINDOW
        if schematic_single_window is None
        else bool(schematic_single_window)
    )
    if schematic_live_path:
        _schematic_live_path = Path(schematic_live_path).expanduser().resolve()
    elif _DEFAULT_SCHEMATIC_LIVE_PATH:
        _schematic_live_path = Path(_DEFAULT_SCHEMATIC_LIVE_PATH).expanduser().resolve()
    else:
        _schematic_live_path = (workdir / ".ui" / "live_schematic.asc").resolve()
    _load_run_state()


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP server for LTspice on macOS")
    parser.add_argument(
        "--workdir",
        default=os.getenv("LTSPICE_MCP_WORKDIR", os.getcwd()),
        help="Directory where netlists/runs are created",
    )
    parser.add_argument(
        "--ltspice-binary",
        default=os.getenv("LTSPICE_BINARY"),
        help="Path to LTspice executable (inside LTspice.app)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("LTSPICE_MCP_TIMEOUT", "120")),
        help="Default simulation timeout in seconds",
    )
    parser.add_argument(
        "--ui-enabled",
        dest="ui_enabled",
        action="store_true",
        help="Enable LTspice UI opening by default in simulation calls",
    )
    parser.add_argument(
        "--ui-disabled",
        dest="ui_enabled",
        action="store_false",
        help="Disable LTspice UI opening by default in simulation calls",
    )
    parser.set_defaults(ui_enabled=None)
    parser.add_argument(
        "--schematic-single-window",
        dest="schematic_single_window",
        action="store_true",
        help="Reuse a single LTspice window path for schematic UI opens (default).",
    )
    parser.add_argument(
        "--schematic-multi-window",
        dest="schematic_single_window",
        action="store_false",
        help="Open generated schematics directly (can create multiple UI windows).",
    )
    parser.set_defaults(schematic_single_window=None)
    parser.add_argument(
        "--schematic-live-path",
        default=os.getenv("LTSPICE_MCP_SCHEMATIC_LIVE_PATH"),
        help="Override path used for single-window schematic UI mode.",
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse"],
        help="MCP transport",
    )
    args = parser.parse_args()

    _configure_runner(
        workdir=Path(args.workdir).expanduser().resolve(),
        ltspice_binary=args.ltspice_binary,
        timeout=args.timeout,
        ui_enabled=args.ui_enabled,
        schematic_single_window=args.schematic_single_window,
        schematic_live_path=args.schematic_live_path,
    )
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
