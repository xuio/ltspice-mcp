from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

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
from .ltspice import LTspiceRunner, find_ltspice_executable, get_ltspice_version, tail_text_file
from .models import RawDataset, SimulationRun
from .raw_parser import RawParseError, parse_raw_file


mcp = FastMCP("ltspice-mcp-macos")

_DEFAULT_WORKDIR = Path(os.getenv("LTSPICE_MCP_WORKDIR", os.getcwd()))
_DEFAULT_TIMEOUT = int(os.getenv("LTSPICE_MCP_TIMEOUT", "120"))
_DEFAULT_BINARY = os.getenv("LTSPICE_BINARY")

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
    }


@mcp.tool()
def loadCircuit(netlist: str, circuit_name: str = "circuit") -> dict[str, Any]:
    """Create a netlist file in the MCP workdir and mark it as the currently loaded circuit."""
    global _loaded_netlist
    path = _runner.write_netlist(netlist, circuit_name=circuit_name)
    _loaded_netlist = path
    return {"netlist_path": str(path), "loaded": True}


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
) -> dict[str, Any]:
    """
    Run LTspice in batch mode for the currently loaded netlist.

    The ngspice-style `command` parameter is accepted for compatibility, but LTspice ignores it.
    """
    if _loaded_netlist is None:
        raise ValueError("No netlist is loaded. Use loadCircuit or loadNetlistFromFile first.")

    run = _register_run(
        _runner.run_file(
            _loaded_netlist,
            ascii_raw=ascii_raw,
            timeout_seconds=timeout_seconds,
        )
    )
    notes: list[str] = []
    if command.strip():
        notes.append(
            "LTspice batch mode ignores ngspice commands; use directives (.tran/.ac/.op) in the netlist."
        )
    response = _run_payload(run, include_output=False, log_tail_lines=120)
    if notes:
        response["notes"] = notes
    return response


@mcp.tool()
def simulateNetlist(
    netlist_content: str,
    circuit_name: str = "circuit",
    ascii_raw: bool = False,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Write a netlist and run LTspice batch simulation in one call."""
    global _loaded_netlist
    netlist_path = _runner.write_netlist(netlist_content, circuit_name=circuit_name)
    _loaded_netlist = netlist_path
    run = _register_run(
        _runner.run_file(
            netlist_path,
            ascii_raw=ascii_raw,
            timeout_seconds=timeout_seconds,
        )
    )
    return _run_payload(run, include_output=False, log_tail_lines=120)


@mcp.tool()
def simulateNetlistFile(
    netlist_path: str,
    ascii_raw: bool = False,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Run LTspice batch simulation for an existing netlist path."""
    global _loaded_netlist
    path = Path(netlist_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Netlist file not found: {path}")
    _loaded_netlist = path
    run = _register_run(
        _runner.run_file(
            path,
            ascii_raw=ascii_raw,
            timeout_seconds=timeout_seconds,
        )
    )
    return _run_payload(run, include_output=False, log_tail_lines=120)


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


def _configure_runner(*, workdir: Path, ltspice_binary: str | None, timeout: int) -> None:
    global _runner, _loaded_netlist, _raw_cache, _state_path
    _runner = LTspiceRunner(
        workdir=workdir,
        executable=ltspice_binary,
        default_timeout_seconds=timeout,
    )
    _state_path = workdir / ".ltspice_mcp_runs.json"
    _loaded_netlist = None
    _raw_cache = {}
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
    )
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
