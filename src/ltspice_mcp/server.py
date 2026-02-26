from __future__ import annotations

import argparse
import base64
import difflib
import functools
import inspect
import json
import logging
import math
import mimetypes
import os
import platform
import queue
import random
import re
import shutil
import struct
import threading
import time
import uuid
import zlib
from collections import deque
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import mcp.types as types
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase as _FastMCPArgModelBase

from . import __version__ as LTSPICE_MCP_VERSION
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
    analyze_log,
    capture_ltspice_window_screenshot,
    close_ltspice_window,
    find_ltspice_executable,
    get_capture_event_history,
    get_capture_health_snapshot,
    get_ltspice_version,
    is_ltspice_ui_running,
    open_in_ltspice_ui,
    tail_text_file,
)
from .models import RawDataset, SimulationDiagnostic, SimulationRun
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
from .textio import read_text_auto


mcp = FastMCP("ltspice-mcp-macos")
CallToolResult = types.CallToolResult
_LOGGER = logging.getLogger(__name__)

# Reject unknown tool arguments instead of silently dropping them.
# This keeps retired parameters (for example `backend`) from being accidentally ignored.
_FastMCPArgModelBase.model_config = {  # type: ignore[assignment]
    **dict(_FastMCPArgModelBase.model_config),
    "extra": "forbid",
}


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


_TOOL_LOGGING_ENABLED = _read_env_bool("LTSPICE_MCP_TOOL_LOGGING", default=True)
_TOOL_LOG_MAX_ITEMS = max(1, int(os.getenv("LTSPICE_MCP_TOOL_LOG_MAX_ITEMS", "16")))
_TOOL_LOG_MAX_CHARS = max(80, int(os.getenv("LTSPICE_MCP_TOOL_LOG_MAX_CHARS", "300")))
_DEFAULT_LOG_LEVEL = os.getenv("LTSPICE_MCP_LOG_LEVEL", "INFO")


_DEFAULT_WORKDIR = Path(os.getenv("LTSPICE_MCP_WORKDIR", os.getcwd()))
_DEFAULT_TIMEOUT = int(os.getenv("LTSPICE_MCP_TIMEOUT", "120"))
_DEFAULT_BINARY = os.getenv("LTSPICE_BINARY")
_DEFAULT_UI_ENABLED = _read_env_bool("LTSPICE_MCP_UI_ENABLED", default=False)
_DEFAULT_JSON_RESPONSE = _read_env_bool("LTSPICE_MCP_JSON_RESPONSE", default=True)
_DEFAULT_STATELESS_HTTP = _read_env_bool("LTSPICE_MCP_STATELESS_HTTP", default=True)
_DEFAULT_SCHEMATIC_SINGLE_WINDOW = _read_env_bool(
    "LTSPICE_MCP_SCHEMATIC_SINGLE_WINDOW",
    default=True,
)
_DEFAULT_SCHEMATIC_LIVE_PATH = os.getenv("LTSPICE_MCP_SCHEMATIC_LIVE_PATH")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_AGENT_README_PATH = _PROJECT_ROOT / "AGENT_README.md"

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
_render_sessions: dict[str, dict[str, Any]] = {}
_TOOL_TELEMETRY_WINDOW = max(10, int(os.getenv("LTSPICE_MCP_TELEMETRY_WINDOW", "200")))
_tool_telemetry: dict[str, dict[str, Any]] = {}
_uvicorn_noise_filters_installed = False
_root_noise_filter_installed = False
_job_lock = threading.RLock()
_jobs: dict[str, dict[str, Any]] = {}
_job_order: list[str] = []
_job_queue: queue.PriorityQueue[tuple[int, int, str]] = queue.PriorityQueue()
_job_worker_thread: threading.Thread | None = None
_job_worker_stop = threading.Event()
_job_state_path: Path = _DEFAULT_WORKDIR / ".ltspice_mcp_jobs.json"
_job_history_path: Path = _DEFAULT_WORKDIR / ".ltspice_mcp_job_history.json"
_job_history: list[dict[str, Any]] = []
_job_history_index: dict[str, int] = {}
_job_history_retention = max(50, int(os.getenv("LTSPICE_MCP_JOB_HISTORY_RETENTION", "1000")))
_job_seq = 0
_JOB_TERMINAL_STATUSES = {"succeeded", "failed", "canceled"}

# Visual style defaults calibrated from manually curated fixture schematics.
_SCHEMATIC_STYLE_PROFILE: dict[str, int] = {
    "grid": 16,
    "anchor_x": 120,
    "anchor_y": 156,
    "directive_x": 48,
    "directive_gap_y": 96,
    "directive_line_step": 24,
}


def _configure_logging(log_level: str | None = None) -> str:
    raw = (log_level or _DEFAULT_LOG_LEVEL).strip().upper()
    level_name = raw if raw in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"} else "INFO"
    level = getattr(logging, level_name, logging.INFO)
    root_logger = logging.getLogger()

    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    else:
        root_logger.setLevel(level)

    logging.getLogger("ltspice_mcp").setLevel(level)
    _install_root_noise_filter()
    return level_name


class _UvicornAccessNoiseFilter(logging.Filter):
    def __init__(self, streamable_http_path: str) -> None:
        super().__init__()
        normalized = streamable_http_path.strip() or "/mcp"
        if not normalized.startswith("/"):
            normalized = f"/{normalized}"
        self._streamable_http_path = normalized

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        match = re.search(r'"([A-Z]+)\s+([^ ]+)\s+HTTP/[0-9.]+"\s+(\d{3})', message)
        if not match:
            return True
        method = match.group(1).upper()
        path = match.group(2)
        status_code = int(match.group(3))
        if (
            method == "GET"
            and status_code == 404
            and (
                path.startswith("/.well-known/oauth-authorization-server")
                or path.startswith("/.well-known/oauth-protected-resource")
                or path.startswith(f"{self._streamable_http_path}/.well-known/oauth-authorization-server")
                or path.startswith(f"{self._streamable_http_path}/.well-known/oauth-protected-resource")
            )
        ):
            return False
        if method == "GET" and status_code == 400 and path == self._streamable_http_path:
            return False
        if method == "DELETE" and status_code == 405 and path == self._streamable_http_path:
            return False
        return True


class _UvicornErrorNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "ASGI callable returned without completing response" in message:
            return False
        return True


class _RootNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage().strip()
        if message == "Terminating session: None":
            return False
        return True


def _install_root_noise_filter() -> None:
    global _root_noise_filter_installed
    if _root_noise_filter_installed:
        return
    logging.getLogger().addFilter(_RootNoiseFilter())
    _root_noise_filter_installed = True


def _install_uvicorn_noise_filters(streamable_http_path: str) -> None:
    global _uvicorn_noise_filters_installed
    if _uvicorn_noise_filters_installed:
        return
    if os.getenv("LTSPICE_MCP_DISABLE_UVICORN_NOISE_FILTERS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        _uvicorn_noise_filters_installed = True
        return
    logging.getLogger("uvicorn.access").addFilter(_UvicornAccessNoiseFilter(streamable_http_path))
    logging.getLogger("uvicorn.error").addFilter(_UvicornErrorNoiseFilter())
    _uvicorn_noise_filters_installed = True


def _record_tool_telemetry(
    *,
    tool_name: str,
    elapsed_ms: float,
    success: bool,
    error_type: str | None = None,
) -> None:
    now = datetime.now().isoformat()
    entry = _tool_telemetry.setdefault(
        tool_name,
        {
            "tool": tool_name,
            "calls_total": 0,
            "calls_ok": 0,
            "calls_error": 0,
            "total_ms": 0.0,
            "min_ms": None,
            "max_ms": None,
            "last_ms": 0.0,
            "last_status": "ok",
            "last_error_type": None,
            "last_started_at": None,
            "last_finished_at": None,
            "recent_ms": deque(maxlen=_TOOL_TELEMETRY_WINDOW),
        },
    )
    recent_ms = entry["recent_ms"]
    if not isinstance(recent_ms, deque):
        recent_ms = deque(maxlen=_TOOL_TELEMETRY_WINDOW)
        entry["recent_ms"] = recent_ms

    entry["calls_total"] = int(entry["calls_total"]) + 1
    if success:
        entry["calls_ok"] = int(entry["calls_ok"]) + 1
        entry["last_status"] = "ok"
        entry["last_error_type"] = None
    else:
        entry["calls_error"] = int(entry["calls_error"]) + 1
        entry["last_status"] = "error"
        entry["last_error_type"] = error_type
    entry["total_ms"] = float(entry["total_ms"]) + float(elapsed_ms)
    min_ms = entry.get("min_ms")
    max_ms = entry.get("max_ms")
    entry["min_ms"] = float(elapsed_ms) if min_ms is None else min(float(min_ms), float(elapsed_ms))
    entry["max_ms"] = float(elapsed_ms) if max_ms is None else max(float(max_ms), float(elapsed_ms))
    entry["last_ms"] = float(elapsed_ms)
    entry["last_finished_at"] = now
    recent_ms.append(float(elapsed_ms))


@contextmanager
def _tool_telemetry_scope(tool_name: str):
    started_perf = time.perf_counter()
    started_wall = datetime.now().isoformat()
    entry = _tool_telemetry.setdefault(
        tool_name,
        {
            "tool": tool_name,
            "calls_total": 0,
            "calls_ok": 0,
            "calls_error": 0,
            "total_ms": 0.0,
            "min_ms": None,
            "max_ms": None,
            "last_ms": 0.0,
            "last_status": "ok",
            "last_error_type": None,
            "last_started_at": None,
            "last_finished_at": None,
            "recent_ms": deque(maxlen=_TOOL_TELEMETRY_WINDOW),
        },
    )
    entry["last_started_at"] = started_wall
    try:
        yield
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started_perf) * 1000.0
        _record_tool_telemetry(
            tool_name=tool_name,
            elapsed_ms=elapsed_ms,
            success=False,
            error_type=type(exc).__name__,
        )
        raise
    elapsed_ms = (time.perf_counter() - started_perf) * 1000.0
    _record_tool_telemetry(
        tool_name=tool_name,
        elapsed_ms=elapsed_ms,
        success=True,
        error_type=None,
    )


def _truncate_for_log(value: str, *, max_chars: int = _TOOL_LOG_MAX_CHARS) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 12] + "...(truncated)"


def _summarize_for_log(value: Any, *, depth: int = 0) -> Any:
    if depth >= 3:
        return {"type": type(value).__name__}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_for_log(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, types.CallToolResult):
        structured = getattr(value, "structuredContent", None)
        summary: dict[str, Any] = {
            "type": "CallToolResult",
            "is_error": bool(getattr(value, "isError", False)),
            "content_items": len(getattr(value, "content", []) or []),
        }
        if isinstance(structured, dict):
            keys = list(structured.keys())
            summary["structured_keys"] = [str(key) for key in keys[:_TOOL_LOG_MAX_ITEMS]]
            if len(keys) > _TOOL_LOG_MAX_ITEMS:
                summary["structured_more_keys"] = len(keys) - _TOOL_LOG_MAX_ITEMS
        elif structured is not None:
            summary["structured_type"] = type(structured).__name__
        return summary
    if isinstance(value, dict):
        items = list(value.items())
        out: dict[str, Any] = {}
        for key, raw_val in items[:_TOOL_LOG_MAX_ITEMS]:
            out[str(key)] = _summarize_for_log(raw_val, depth=depth + 1)
        if len(items) > _TOOL_LOG_MAX_ITEMS:
            out["__more_items__"] = len(items) - _TOOL_LOG_MAX_ITEMS
        return out
    if isinstance(value, (list, tuple, set)):
        sequence = list(value)
        summarized = [
            _summarize_for_log(item, depth=depth + 1)
            for item in sequence[:_TOOL_LOG_MAX_ITEMS]
        ]
        if len(sequence) > _TOOL_LOG_MAX_ITEMS:
            summarized.append(f"... ({len(sequence) - _TOOL_LOG_MAX_ITEMS} more)")
        return summarized
    return {
        "type": type(value).__name__,
        "repr": _truncate_for_log(repr(value)),
    }


def _log_tool_event(event: str, **fields: Any) -> None:
    if not _TOOL_LOGGING_ENABLED:
        return
    payload = {"event": event, **fields}
    try:
        encoded = json.dumps(payload, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001
        encoded = str(payload)
    _LOGGER.info("mcp_tool %s", encoded)


def _telemetry_tool(func):
    if getattr(func, "__ltspice_tool_wrapped__", False):
        return func
    # Evaluate forward-reference annotations before preserving the signature.
    # This keeps FastMCP return-type handling correct for CallToolResult tools.
    try:
        signature = inspect.signature(func, eval_str=True)
    except Exception:  # noqa: BLE001
        signature = inspect.signature(func)
    tool_name = func.__name__

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        call_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        _log_tool_event(
            "tool_call_start",
            call_id=call_id,
            tool=tool_name,
            args=_summarize_for_log(list(args)),
            kwargs=_summarize_for_log(kwargs),
        )
        try:
            with _tool_telemetry_scope(tool_name):
                result = func(*args, **kwargs)
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
            _log_tool_event(
                "tool_call_error",
                call_id=call_id,
                tool=tool_name,
                elapsed_ms=elapsed_ms,
                error_type=type(exc).__name__,
                error=_truncate_for_log(str(exc)),
            )
            raise
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        _log_tool_event(
            "tool_call_success",
            call_id=call_id,
            tool=tool_name,
            elapsed_ms=elapsed_ms,
            result=_summarize_for_log(result),
        )
        return result

    wrapper.__signature__ = signature
    setattr(wrapper, "__ltspice_tool_wrapped__", True)
    return wrapper


_ORIGINAL_MCP_TOOL = mcp.tool


def _mcp_tool_with_logging(*tool_args, **tool_kwargs):
    if tool_args and callable(tool_args[0]) and len(tool_args) == 1 and not tool_kwargs:
        return _ORIGINAL_MCP_TOOL()(_telemetry_tool(tool_args[0]))

    decorator = _ORIGINAL_MCP_TOOL(*tool_args, **tool_kwargs)

    def _apply(func):
        return decorator(_telemetry_tool(func))

    return _apply


mcp.tool = _mcp_tool_with_logging  # type: ignore[assignment]


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pct_clamped = max(0.0, min(1.0, float(pct)))
    idx = int(round((len(ordered) - 1) * pct_clamped))
    return float(ordered[idx])


def _read_agent_readme_text() -> str:
    if not _AGENT_README_PATH.exists():
        raise FileNotFoundError(f"Agent guide not found: {_AGENT_README_PATH}")
    return _AGENT_README_PATH.read_text(encoding="utf-8")


def _parse_markdown_headings(text: str) -> list[dict[str, Any]]:
    headings: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", raw_line)
        if not match:
            continue
        level = len(match.group(1))
        title = match.group(2).strip()
        headings.append(
            {
                "index": len(headings) + 1,
                "line_number": line_no,
                "level": level,
                "title": title,
            }
        )
    return headings


def _resolve_heading_query(
    headings: list[dict[str, Any]],
    section: str,
) -> dict[str, Any] | None:
    query = section.strip()
    if not query:
        return None
    if query.isdigit():
        idx = int(query)
        if idx <= 0:
            return None
        for heading in headings:
            if int(heading["index"]) == idx:
                return heading
    normalized = query.casefold()
    for heading in headings:
        if str(heading["title"]).casefold() == normalized:
            return heading
    for heading in headings:
        if normalized in str(heading["title"]).casefold():
            return heading
    return None


def _extract_heading_section(
    text: str,
    headings: list[dict[str, Any]],
    heading: dict[str, Any],
) -> str:
    lines = text.splitlines()
    start_line = int(heading["line_number"])
    start_idx = max(0, start_line - 1)
    end_idx = len(lines)
    selected_level = int(heading["level"])
    for candidate in headings:
        if int(candidate["line_number"]) <= start_line:
            continue
        if int(candidate["level"]) <= selected_level:
            end_idx = int(candidate["line_number"]) - 1
            break
    section_lines = lines[start_idx:end_idx]
    content = "\n".join(section_lines).strip()
    if content:
        content += "\n"
    return content


def _search_text_lines(
    text: str,
    query: str,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    needle = query.strip().casefold()
    if not needle:
        return []
    matches: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if needle not in line.casefold():
            continue
        matches.append({"line_number": line_no, "line": line})
        if len(matches) >= limit:
            break
    return matches


def _telemetry_payload() -> dict[str, Any]:
    tools_payload: list[dict[str, Any]] = []
    for tool_name, entry in sorted(_tool_telemetry.items()):
        recent = list(entry.get("recent_ms") or [])
        total = int(entry.get("calls_total", 0))
        total_ms = float(entry.get("total_ms", 0.0))
        avg_ms = (total_ms / total) if total > 0 else 0.0
        tools_payload.append(
            {
                "tool": tool_name,
                "calls_total": total,
                "calls_ok": int(entry.get("calls_ok", 0)),
                "calls_error": int(entry.get("calls_error", 0)),
                "total_ms": round(total_ms, 3),
                "avg_ms": round(avg_ms, 3),
                "min_ms": round(float(entry["min_ms"]), 3) if entry.get("min_ms") is not None else None,
                "max_ms": round(float(entry["max_ms"]), 3) if entry.get("max_ms") is not None else None,
                "p50_ms": round(float(_percentile(recent, 0.50)), 3)
                if _percentile(recent, 0.50) is not None
                else None,
                "p95_ms": round(float(_percentile(recent, 0.95)), 3)
                if _percentile(recent, 0.95) is not None
                else None,
                "last_ms": round(float(entry.get("last_ms", 0.0)), 3),
                "last_status": str(entry.get("last_status", "ok")),
                "last_error_type": entry.get("last_error_type"),
                "last_started_at": entry.get("last_started_at"),
                "last_finished_at": entry.get("last_finished_at"),
                "recent_sample_count": len(recent),
            }
        )
    return {
        "window_size": _TOOL_TELEMETRY_WINDOW,
        "tool_count": len(tools_payload),
        "tools": tools_payload,
    }


def _save_run_state() -> None:
    _state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "runs": [_runs[run_id].to_storage_dict() for run_id in _run_order if run_id in _runs],
    }
    tmp_path = _state_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(_state_path)


def _next_job_seq() -> int:
    global _job_seq
    _job_seq += 1
    return _job_seq


def _save_job_state() -> None:
    with _job_lock:
        active_jobs = [
            dict(_jobs[job_id])
            for job_id in _job_order
            if job_id in _jobs and str(_jobs[job_id].get("status", "")).lower() not in _JOB_TERMINAL_STATUSES
        ]
        if not active_jobs:
            if _job_state_path.exists():
                try:
                    _job_state_path.unlink()
                except Exception:
                    pass
            return
        payload = {
            "version": 1,
            "job_seq": int(_job_seq),
            "job_order": [str(job.get("job_id", "")) for job in active_jobs if str(job.get("job_id", "")).strip()],
            "jobs": active_jobs,
        }
    _job_state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _job_state_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(_job_state_path)


def _enqueue_job_locked(job_id: str) -> None:
    job = _jobs.get(job_id)
    if job is None:
        return
    if str(job.get("status", "")).lower() != "queued":
        return
    priority = int(job.get("priority", 50))
    seq = _next_job_seq()
    _job_queue.put((priority, seq, job_id))
    job["queue_seq"] = seq


def _load_job_state() -> None:
    global _job_seq
    _jobs.clear()
    _job_order.clear()
    _job_seq = 0
    if not _job_state_path.exists():
        return
    try:
        payload = json.loads(_job_state_path.read_text(encoding="utf-8"))
    except Exception:
        return
    entries = payload.get("jobs", [])
    if not isinstance(entries, list):
        return

    ordered_ids: list[str] = []
    if isinstance(payload.get("job_order"), list):
        ordered_ids = [str(item) for item in payload["job_order"] if isinstance(item, str)]

    loaded: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        job_id = str(entry.get("job_id", "")).strip()
        if not job_id:
            continue
        job = dict(entry)
        job["job_id"] = job_id
        job["status"] = str(job.get("status", "queued")).lower()
        job["cancel_requested"] = bool(job.get("cancel_requested", False))
        job["priority"] = int(job.get("priority", 50))
        job["max_retries"] = max(0, int(job.get("max_retries", 0)))
        job["retry_count"] = max(0, int(job.get("retry_count", 0)))
        job["created_at"] = str(job.get("created_at") or _now_iso())
        if job["status"] not in _JOB_TERMINAL_STATUSES:
            # Jobs that were queued/running before daemon restart resume from queue.
            job["status"] = "queued"
            job["started_at"] = None
            job["finished_at"] = None
            if not job["cancel_requested"]:
                job["summary"] = "Resumed from persisted queue state."
        loaded[job_id] = job

    for job_id in ordered_ids:
        if job_id in loaded:
            _job_order.append(job_id)
    for job_id in loaded:
        if job_id not in _job_order:
            _job_order.append(job_id)
    _jobs.update(loaded)
    _job_seq = max(int(payload.get("job_seq", 0)), len(_job_order))

    with _job_lock:
        for job_id in _job_order:
            job = _jobs.get(job_id)
            if job is None:
                continue
            if job.get("cancel_requested"):
                continue
            if str(job.get("status", "")).lower() == "queued":
                _enqueue_job_locked(job_id)


def _trim_job_history_locked() -> None:
    global _job_history
    if len(_job_history) <= _job_history_retention:
        _job_history_index.clear()
        _job_history_index.update({str(item.get("job_id", "")): idx for idx, item in enumerate(_job_history)})
        return
    _job_history = _job_history[-_job_history_retention:]
    _job_history_index.clear()
    _job_history_index.update({str(item.get("job_id", "")): idx for idx, item in enumerate(_job_history)})


def _archive_job_locked(job: dict[str, Any]) -> None:
    job_id = str(job.get("job_id", "")).strip()
    if not job_id:
        return
    payload = dict(job)
    payload["archived_at"] = _now_iso()
    existing = _job_history_index.get(job_id)
    if existing is None:
        _job_history.append(payload)
        _job_history_index[job_id] = len(_job_history) - 1
    else:
        _job_history[existing] = payload
    _trim_job_history_locked()


def _save_job_history_state() -> None:
    with _job_lock:
        if not _job_history:
            if _job_history_path.exists():
                try:
                    _job_history_path.unlink()
                except Exception:
                    pass
            return
        payload = {
            "version": 1,
            "retention": int(_job_history_retention),
            "history": list(_job_history),
        }
    _job_history_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _job_history_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(_job_history_path)


def _load_job_history_state() -> None:
    _job_history.clear()
    _job_history_index.clear()
    if not _job_history_path.exists():
        return
    try:
        payload = json.loads(_job_history_path.read_text(encoding="utf-8"))
    except Exception:
        return
    entries = payload.get("history", [])
    if not isinstance(entries, list):
        return
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        job_id = str(entry.get("job_id", "")).strip()
        if not job_id:
            continue
        _job_history.append(dict(entry))
    _trim_job_history_locked()


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


def _new_render_session_id() -> str:
    return uuid.uuid4().hex


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


_PLOT_MODES = {"auto", "db", "phase", "real", "imag"}
_PLOT_Y_MODES = {"magnitude", "phase", "real", "imag", "db"}
_PANE_LAYOUTS = {"single", "split", "per_trace"}


def _format_plt_number(value: float) -> str:
    if not math.isfinite(value):
        return "0"
    return format(value, ".12g")


def _safe_axis_range(values: list[float], *, fallback_span: float) -> tuple[float, float]:
    finite = [item for item in values if math.isfinite(item)]
    if not finite:
        return 0.0, fallback_span
    lo = min(finite)
    hi = max(finite)
    if math.isclose(lo, hi):
        pad = max(abs(lo), abs(hi), fallback_span) * 0.5
        lo -= pad
        hi += pad
    return lo, hi


def _normalize_plot_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in _PLOT_MODES:
        raise ValueError("mode must be one of: auto, db, phase, real, imag")
    return normalized


def _normalize_plot_y_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in _PLOT_Y_MODES:
        raise ValueError("y_mode must be one of: magnitude, phase, real, imag, db")
    return normalized


def _normalize_pane_layout(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in _PANE_LAYOUTS:
        raise ValueError("pane_layout must be one of: single, split, per_trace")
    return normalized


def _infer_plot_type(dataset: RawDataset) -> str:
    plot_name = dataset.plot_name.lower()
    scale_name = dataset.scale_name.lower()
    if "ac analysis" in plot_name or scale_name == "frequency":
        return "ac"
    if "transient analysis" in plot_name or scale_name == "time":
        return "tran"
    if "dc transfer characteristic" in plot_name or "dc" in plot_name:
        return "dc"
    return "generic"


def _series_for_expression(dataset: RawDataset, expression: str, step_index: int | None) -> list[complex] | None:
    try:
        return dataset.get_vector(expression, step_index=step_index)
    except KeyError:
        return None


def _transform_series(series: list[complex], mode: str) -> list[float]:
    if mode == "db":
        return [20.0 * math.log10(max(abs(value), 1e-30)) for value in series]
    if mode == "phase":
        return [math.degrees(math.atan2(value.imag, value.real)) for value in series]
    if mode == "imag":
        return [value.imag for value in series]
    return [value.real for value in series]


def _resolve_axis_range(
    *,
    values: list[float],
    fallback_span: float,
    lower_override: float | None,
    upper_override: float | None,
) -> tuple[float, float, float]:
    lo, hi = _safe_axis_range(values, fallback_span=fallback_span)
    if lower_override is not None:
        lo = float(lower_override)
    if upper_override is not None:
        hi = float(upper_override)
    if hi <= lo:
        raise ValueError("axis max must be greater than axis min")
    step = max((hi - lo) / 10.0, 1e-12)
    return lo, step, hi


def _axis_prefix_for_range(lo: float, hi: float) -> str:
    scale = max(abs(lo), abs(hi))
    if scale >= 1e9:
        return "G"
    if scale >= 1e6:
        return "M"
    if scale >= 1e3:
        return "K"
    if scale >= 1:
        return " "
    if scale >= 1e-3:
        return "m"
    if scale >= 1e-6:
        return "u"
    if scale >= 1e-9:
        return "n"
    return "p"


def _trace_expression_for_mode(expression: str, *, mode: str, plot_type: str, dual_axis: bool) -> str:
    if mode == "db" and dual_axis and plot_type == "ac":
        return expression
    if mode == "db":
        return f"dB({expression})"
    if mode == "phase":
        return f"ph({expression})"
    if mode == "imag":
        return f"im({expression})"
    if mode == "real" and plot_type == "ac":
        return f"re({expression})"
    return expression


def _pane_groups(vectors: list[str], pane_layout: str) -> list[list[str]]:
    if pane_layout == "single" or len(vectors) <= 1:
        return [vectors]
    if pane_layout == "per_trace":
        return [[vector] for vector in vectors]
    split_index = max(1, math.ceil(len(vectors) / 2))
    left = vectors[:split_index]
    right = vectors[split_index:]
    if right:
        return [left, right]
    return [left]


def _ascii_raw_scalar(value: complex, *, complex_mode: bool) -> str:
    if complex_mode:
        return f"({value.real:.16g},{value.imag:.16g})"
    return f"{value.real:.16g}"


def _write_raw_dataset_ascii(dataset: RawDataset, output_path: Path) -> None:
    complex_mode = "complex" in dataset.flags or any(
        abs(value.imag) > 1e-24
        for series in dataset.values
        for value in series
    )
    flags = sorted(
        flag for flag in dataset.flags if flag not in {"stepped", "fastaccess", "compressed"}
    )
    flags = [flag for flag in flags if flag not in {"real", "complex"}]
    flags.insert(0, "complex" if complex_mode else "real")

    header_lines = [
        f"Title: {dataset.metadata.get('Title', 'Generated by ltspice-mcp')}",
        f"Date: {dataset.metadata.get('Date', datetime.now().ctime())}",
        f"Plotname: {dataset.plot_name}",
        f"Flags: {' '.join(flags)}",
        f"No. Variables: {len(dataset.variables)}",
        f"No. Points: {dataset.points}",
        "Offset: 0",
        "Command: generated by ltspice-mcp",
        "Variables:",
    ]
    for variable in dataset.variables:
        header_lines.append(f"\t{variable.index}\t{variable.name}\t{variable.kind}")
    header_lines.append("Values:")

    value_lines: list[str] = []
    for point_index in range(dataset.points):
        value_lines.append(
            f"\t{point_index}\t{_ascii_raw_scalar(dataset.values[0][point_index], complex_mode=complex_mode)}"
        )
        for variable_index in range(1, len(dataset.variables)):
            value_lines.append(
                f"\t\t{_ascii_raw_scalar(dataset.values[variable_index][point_index], complex_mode=complex_mode)}"
            )
    output_path.write_text("\n".join([*header_lines, *value_lines]) + "\n", encoding="utf-8")


def _materialize_plot_step_dataset(
    *,
    dataset: RawDataset,
    selected_step: int | None,
) -> tuple[RawDataset, dict[str, Any]]:
    if not dataset.is_stepped or selected_step is None:
        return dataset, {
            "source_raw_path": str(dataset.path),
            "render_raw_path": str(dataset.path),
            "step_materialized": False,
            "step_index": selected_step,
        }

    step = dataset.steps[selected_step]
    step_values = [series[step.start : step.end] for series in dataset.values]
    step_flags = {flag for flag in dataset.flags if flag not in {"stepped", "fastaccess"}}
    step_raw_path = dataset.path.with_name(f"{dataset.path.stem}__step{selected_step}.raw")
    step_dataset = RawDataset(
        path=step_raw_path,
        plot_name=dataset.plot_name,
        flags=step_flags,
        metadata=dict(dataset.metadata),
        variables=list(dataset.variables),
        values=step_values,
        steps=[],
    )
    _write_raw_dataset_ascii(step_dataset, step_raw_path)
    return step_dataset, {
        "source_raw_path": str(dataset.path),
        "render_raw_path": str(step_raw_path),
        "step_materialized": True,
        "step_index": selected_step,
        "materialized_points": step_dataset.points,
    }


def _build_ltspice_plot_settings_text(
    *,
    dataset: RawDataset,
    vectors: list[str],
    mode: str,
    pane_layout: str,
    dual_axis: bool | None,
    x_log: bool | None,
    x_min: float | None,
    x_max: float | None,
    y_min: float | None,
    y_max: float | None,
) -> dict[str, Any]:
    if not vectors:
        raise ValueError("vectors must contain at least one vector name")

    plot_type = _infer_plot_type(dataset)
    mode_used = mode
    if mode_used == "auto":
        mode_used = "db" if plot_type == "ac" else "real"

    dual_axis_enabled = (
        bool(dual_axis)
        if dual_axis is not None
        else (plot_type == "ac" and mode_used == "db")
    )
    if plot_type != "ac" or mode_used != "db":
        dual_axis_enabled = False

    x_log_enabled = bool(x_log) if x_log is not None else (plot_type == "ac")
    pane_groups = _pane_groups(vectors, pane_layout=pane_layout)
    x_values = dataset.scale_values(step_index=None)
    x_lo, x_step, x_hi = _resolve_axis_range(
        values=x_values,
        fallback_span=1.0,
        lower_override=x_min,
        upper_override=x_max,
    )
    x_prefix = _axis_prefix_for_range(x_lo, x_hi)

    lines: list[str] = [
        f"[{dataset.plot_name}]",
        "{",
        f"   Npanes: {len(pane_groups)}",
    ]
    pane_payload: list[dict[str, Any]] = []

    next_trace_id = 524290 if plot_type == "ac" else 268959746
    for pane_index, pane_vectors in enumerate(pane_groups):
        trace_expressions: list[str] = []
        raw_to_rendered: dict[str, str] = {}
        for raw_expression in pane_vectors:
            rendered = _trace_expression_for_mode(
                raw_expression,
                mode=mode_used,
                plot_type=plot_type,
                dual_axis=dual_axis_enabled,
            )
            trace_expressions.append(rendered)
            raw_to_rendered[raw_expression] = rendered

        first_series = _series_for_expression(dataset, pane_vectors[0], step_index=None)
        y_lines: list[str] = []
        warnings: list[str] = []
        if dual_axis_enabled:
            # LTspice's Bode rendering expects these canonical axis settings with PltMag/PltPhi.
            # Using direct dB/phase ranges here can hide traces in some LTspice builds.
            mag_lo, mag_step, mag_hi = 0.001, 6.0, 1.0
            phase_lo, phase_step, phase_hi = -90.0, 9.0, 0.0
            if y_min is not None or y_max is not None:
                warnings.append(
                    "y_min/y_max are ignored when dual_axis Bode mode is enabled."
                )
            y_lines.extend(
                [
                    f"      Y[0]: (' ',0,{_format_plt_number(mag_lo)},{_format_plt_number(mag_step)},{_format_plt_number(mag_hi)})",
                    f"      Y[1]: (' ',0,{_format_plt_number(phase_lo)},{_format_plt_number(phase_step)},{_format_plt_number(phase_hi)})",
                    f"      Log: {1 if x_log_enabled else 0} 2 0",
                    "      PltMag: 1",
                    "      PltPhi: 1 0",
                ]
            )
        else:
            sampled = _transform_series(first_series or [], mode_used)
            fallback = 180.0 if mode_used == "phase" else (60.0 if mode_used == "db" else 1.0)
            y_lo, y_step, y_hi = _resolve_axis_range(
                values=sampled,
                fallback_span=fallback,
                lower_override=y_min,
                upper_override=y_max,
            )
            y_lines.extend(
                [
                    f"      Y[0]: (' ',0,{_format_plt_number(y_lo)},{_format_plt_number(y_step)},{_format_plt_number(y_hi)})",
                    "      Y[1]: ('_',0,1e+308,0,-1e+308)",
                    f"      Log: {1 if x_log_enabled else 0} 0 0",
                ]
            )
            if plot_type != "ac":
                y_lines.insert(
                    2,
                    f"      Volts: (' ',0,0,0,{_format_plt_number(y_lo)},{_format_plt_number(y_step)},{_format_plt_number(y_hi)})",
                )
            if first_series is None:
                warnings.append(
                    f"Could not resolve raw vector '{pane_vectors[0]}' for axis auto-scaling; used fallback range."
                )

        pane_trace_tokens: list[str] = []
        for rendered_expression in trace_expressions:
            escaped = rendered_expression.replace('"', '\\"')
            pane_trace_tokens.append(f'{{{next_trace_id},0,"{escaped}"}}')
            next_trace_id += 1

        lines.extend(
            [
                "   {",
                f"      traces: {len(pane_trace_tokens)} {' '.join(pane_trace_tokens)}",
                f"      X: ('{x_prefix}',0,{_format_plt_number(x_lo)},{_format_plt_number(x_step)},{_format_plt_number(x_hi)})",
                *y_lines,
                "      NeyeDiagPeriods: 0",
                "   }",
            ]
        )
        pane_payload.append(
            {
                "pane_index": pane_index,
                "trace_count": len(pane_vectors),
                "input_traces": pane_vectors,
                "rendered_traces": trace_expressions,
                "trace_mapping": raw_to_rendered,
                "warnings": warnings,
            }
        )
    lines.append("}")

    return {
        "text": "\n".join(lines) + "\n",
        "plot_name": dataset.plot_name,
        "plot_type": plot_type,
        "mode_used": mode_used,
        "pane_layout": pane_layout,
        "dual_axis": dual_axis_enabled,
        "x_log": x_log_enabled,
        "x_range": [x_lo, x_hi],
        "x_step": x_step,
        "panes": pane_payload,
    }


def _parse_ltspice_plot_settings_text(text: str) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    npanes = None
    pane_trace_counts: list[int] = []
    pane_traces: list[list[str]] = []
    for line in lines:
        if line.lower().startswith("npanes:"):
            try:
                npanes = int(line.split(":", 1)[1].strip())
            except Exception as exc:
                raise ValueError(f"Invalid Npanes line in .plt: {line}") from exc
            continue
        if line.lower().startswith("traces:"):
            tail = line.split(":", 1)[1].strip()
            count_token = tail.split(" ", 1)[0]
            try:
                count = int(count_token)
            except Exception as exc:
                raise ValueError(f"Invalid traces count in .plt: {line}") from exc
            expressions: list[str] = []
            quote_parts = line.split('"')
            for idx in range(1, len(quote_parts), 2):
                expressions.append(quote_parts[idx])
            pane_trace_counts.append(count)
            pane_traces.append(expressions)

    if npanes is None:
        raise ValueError("Could not parse Npanes from .plt content")
    if len(pane_trace_counts) != npanes:
        raise ValueError(
            f".plt pane mismatch: Npanes={npanes}, found {len(pane_trace_counts)} pane trace blocks"
        )
    return {
        "npanes": npanes,
        "pane_trace_counts": pane_trace_counts,
        "pane_traces": pane_traces,
    }


def _write_ltspice_plot_settings_file(
    *,
    dataset: RawDataset,
    vectors: list[str],
    mode: str,
    pane_layout: str,
    dual_axis: bool | None,
    x_log: bool | None,
    x_min: float | None,
    x_max: float | None,
    y_min: float | None,
    y_max: float | None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    settings = _build_ltspice_plot_settings_text(
        dataset=dataset,
        vectors=vectors,
        mode=mode,
        pane_layout=pane_layout,
        dual_axis=dual_axis,
        x_log=x_log,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )
    plt_path = output_path or dataset.path.with_suffix(".plt")
    plt_path.write_text(str(settings["text"]), encoding="utf-16le")
    parsed = _parse_ltspice_plot_settings_text(str(settings["text"]))
    total_declared_traces = sum(int(item) for item in parsed["pane_trace_counts"])
    if total_declared_traces < len(vectors):
        raise ValueError(
            "Generated .plt file declares fewer traces than requested vectors."
        )
    return {
        "plt_path": str(plt_path),
        "trace_count": len(vectors),
        "trace_expressions": vectors,
        "plot_name": settings["plot_name"],
        "plot_type": settings["plot_type"],
        "mode_used": settings["mode_used"],
        "pane_layout": settings["pane_layout"],
        "dual_axis": settings["dual_axis"],
        "x_log": settings["x_log"],
        "x_range": settings["x_range"],
        "x_step": settings["x_step"],
        "panes": settings["panes"],
        "parsed": parsed,
        "text": settings["text"],
    }


def _png_plot_trace_metrics(path: Path) -> dict[str, Any]:
    blob = path.read_bytes()
    if blob[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"plot capture is not a PNG file: {path}")

    cursor = 8
    idat = bytearray()
    width: int | None = None
    height: int | None = None
    color_type: int | None = None
    bit_depth: int | None = None

    while cursor + 8 <= len(blob):
        chunk_len = struct.unpack(">I", blob[cursor : cursor + 4])[0]
        cursor += 4
        chunk_type = blob[cursor : cursor + 4]
        cursor += 4
        chunk = blob[cursor : cursor + chunk_len]
        cursor += chunk_len + 4  # Skip payload + CRC
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", chunk)
        elif chunk_type == b"IDAT":
            idat.extend(chunk)
        elif chunk_type == b"IEND":
            break

    if width is None or height is None or color_type is None or bit_depth is None:
        raise ValueError(f"Missing PNG IHDR in capture: {path}")
    if bit_depth != 8:
        raise ValueError(f"Unsupported PNG bit depth {bit_depth}; expected 8-bit image")

    if color_type == 6:
        bytes_per_pixel = 4
    elif color_type == 2:
        bytes_per_pixel = 3
    elif color_type == 0:
        bytes_per_pixel = 1
    else:
        raise ValueError(f"Unsupported PNG color type {color_type}")

    decoded = zlib.decompress(bytes(idat))
    stride = width * bytes_per_pixel
    expected_len = (stride + 1) * height
    if len(decoded) < expected_len:
        raise ValueError("PNG pixel payload shorter than expected")

    position = 0
    previous = bytearray(stride)
    trace_pixels = 0
    green_pixels = 0
    non_black_pixels = 0

    for _ in range(height):
        filter_type = decoded[position]
        position += 1
        row = bytearray(decoded[position : position + stride])
        position += stride
        recon = bytearray(stride)

        for index in range(stride):
            left = recon[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            up = previous[index]
            up_left = previous[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            if filter_type == 0:
                value = row[index]
            elif filter_type == 1:
                value = (row[index] + left) & 0xFF
            elif filter_type == 2:
                value = (row[index] + up) & 0xFF
            elif filter_type == 3:
                value = (row[index] + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                predictor = left + up - up_left
                pa = abs(predictor - left)
                pb = abs(predictor - up)
                pc = abs(predictor - up_left)
                nearest = left if pa <= pb and pa <= pc else (up if pb <= pc else up_left)
                value = (row[index] + nearest) & 0xFF
            else:
                raise ValueError(f"Unsupported PNG filter type {filter_type}")
            recon[index] = value
        previous = recon

        for idx in range(0, stride, bytes_per_pixel):
            if color_type == 0:
                red = green = blue = recon[idx]
            else:
                red = recon[idx]
                green = recon[idx + 1]
                blue = recon[idx + 2]
            if red > 0 or green > 0 or blue > 0:
                non_black_pixels += 1
            if green > red + 20 and green > blue + 20 and green > 70:
                trace_pixels += 1
            if green > 100 and red < 120 and blue < 120:
                green_pixels += 1

    return {
        "width": width,
        "height": height,
        "trace_pixels": trace_pixels,
        "green_pixels": green_pixels,
        "non_black_pixels": non_black_pixels,
    }


def _validate_plot_capture(path: Path) -> dict[str, Any]:
    metrics = _png_plot_trace_metrics(path)
    area = max(1, int(metrics["width"]) * int(metrics["height"]))
    min_trace_pixels = max(130, int(area * 0.0018))
    min_green_pixels = max(120, int(area * 0.0008))
    valid = (
        int(metrics["trace_pixels"]) >= min_trace_pixels
        or int(metrics["green_pixels"]) >= min_green_pixels
    )
    return {
        **metrics,
        "min_trace_pixels": min_trace_pixels,
        "min_green_pixels": min_green_pixels,
        "valid": valid,
    }


def _image_tool_result(payload: dict[str, Any]) -> CallToolResult:
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
    return CallToolResult(content=content, structuredContent=payload, isError=False)


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
    if run.log_utf8_path is not None:
        payload["log_tail_utf8"] = tail_text_file(run.log_utf8_path, max_lines=log_tail_lines)
    return payload


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _read_netlist_text(path: Path) -> str:
    text = read_text_auto(path)
    return text if text.endswith("\n") else text + "\n"


def _append_unique_lines_before_end(text: str, lines_to_add: list[str]) -> str:
    lines = text.splitlines()
    normalized_existing = {line.strip().lower() for line in lines if line.strip()}
    additions: list[str] = []
    for entry in lines_to_add:
        normalized = entry.strip().lower()
        if not normalized or normalized in normalized_existing:
            continue
        additions.append(entry.rstrip())
        normalized_existing.add(normalized)
    if not additions:
        return text if text.endswith("\n") else text + "\n"
    end_index = next(
        (idx for idx, line in enumerate(lines) if line.strip().lower() == ".end"),
        None,
    )
    if end_index is None:
        lines.extend(additions)
        lines.append(".end")
    else:
        lines[end_index:end_index] = additions
    out = "\n".join(lines).rstrip() + "\n"
    return out


def _resolve_run_target_for_input(
    *,
    run_id: str | None,
    netlist_path: str | None,
    netlist_content: str | None,
    circuit_name: str | None,
    asc_path: str | None,
    ascii_raw: bool,
    timeout_seconds: int | None,
    show_ui: bool | None,
    open_raw_after_run: bool,
) -> dict[str, Any]:
    if run_id:
        run = _resolve_run(run_id)
        return {
            "source": "existing_run",
            "run": run,
            "run_payload": _run_payload(run, include_output=False, log_tail_lines=160),
        }
    if netlist_content is not None:
        run_payload = simulateNetlist(
            netlist_content=netlist_content,
            circuit_name=circuit_name or "verification_plan",
            ascii_raw=ascii_raw,
            timeout_seconds=timeout_seconds,
            show_ui=show_ui,
            open_raw_after_run=open_raw_after_run,
        )
        run = _resolve_run(str(run_payload["run_id"]))
        return {"source": "simulated_netlist_content", "run": run, "run_payload": run_payload}
    if netlist_path is not None:
        run_payload = simulateNetlistFile(
            netlist_path=netlist_path,
            ascii_raw=ascii_raw,
            timeout_seconds=timeout_seconds,
            show_ui=show_ui,
            open_raw_after_run=open_raw_after_run,
        )
        run = _resolve_run(str(run_payload["run_id"]))
        return {"source": "simulated_netlist_path", "run": run, "run_payload": run_payload}
    if asc_path is not None:
        run_payload = simulateSchematicFile(
            asc_path=asc_path,
            ascii_raw=ascii_raw,
            timeout_seconds=timeout_seconds,
            show_ui=show_ui,
            open_raw_after_run=open_raw_after_run,
            validate_first=True,
            abort_on_validation_error=False,
        )
        run = _resolve_run(str(run_payload["run_id"]))
        return {"source": "simulated_schematic_path", "run": run, "run_payload": run_payload}
    run = _resolve_run(None)
    return {
        "source": "latest_run",
        "run": run,
        "run_payload": _run_payload(run, include_output=False, log_tail_lines=160),
    }


_MEAS_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?(?:[A-Za-z]+)?"
_MEAS_VALUE_RE = re.compile(
    rf"^\s*(?P<name>[A-Za-z_][\w.$-]*)\s*[:=]\s*(?P<value>{_MEAS_NUMBER_PATTERN})"
)
_MEAS_COLON_LINE_RE = re.compile(r"^\s*(?P<name>[A-Za-z_][\w.$-]*)\s*:\s*(?P<rhs>.+?)\s*$")
_MEAS_MEASUREMENT_RE = re.compile(
    r"^\s*Measurement:\s*(?P<name>[A-Za-z_][\w.$-]*)",
    re.IGNORECASE,
)
_MEAS_NUMERIC_TOKEN_RE = re.compile(_MEAS_NUMBER_PATTERN)
_MEAS_AT_VALUE_RE = re.compile(rf"\bAT\s+(?P<value>{_MEAS_NUMBER_PATTERN})\b")
_MEAS_HEADER_TOKEN_RE = re.compile(r"^\s*[A-Za-z_][\w()*/+.-]*\s*$")
_MEAS_AUX_HEADER_TOKENS = {
    "step",
    "from",
    "to",
    "at",
    "when",
    "time",
    "freq",
    "frequency",
    "param",
}
_SPICE_SUFFIX_SCALE: dict[str, float] = {
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
    "mil": 25.4e-6,
}


def _parse_spice_number_token(token: str) -> float | None:
    raw = token.strip()
    if not raw:
        return None
    match = re.fullmatch(
        r"(?P<number>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?)(?P<suffix>[A-Za-z]+)?",
        raw,
    )
    if not match:
        return None
    number = match.group("number")
    suffix = (match.group("suffix") or "").strip().lower()
    try:
        value = Decimal(number.replace("D", "e").replace("d", "e"))
    except InvalidOperation:
        return None
    if suffix:
        value *= Decimal(str(_SPICE_SUFFIX_SCALE.get(suffix, 1.0)))
    return float(value)


def _extract_meas_numeric_from_rhs(rhs: str) -> tuple[float, str] | None:
    # For WHEN-style results LTspice emits:
    #   expr=const AT <time_or_freq>
    # Use the AT value (the measured crossing point), not the condition constant.
    at_match = _MEAS_AT_VALUE_RE.search(rhs)
    if at_match:
        at_token = at_match.group("value")
        parsed_at = _parse_spice_number_token(at_token)
        if parsed_at is not None:
            return parsed_at, at_token

    # Try the value segment after '=' first for lines like:
    #   vpp: PP(v(out))=0.731107 FROM 0 TO 0.008
    #   mag_1k: mag(v(out))=(-1.44507dB,0°) at 1000
    candidate_text = rhs
    if "=" in rhs:
        candidate_text = rhs.split("=", 1)[1]
    for token in _MEAS_NUMERIC_TOKEN_RE.findall(candidate_text):
        parsed = _parse_spice_number_token(token)
        if parsed is not None:
            return parsed, token
    return None


def _select_meas_table_columns(
    *,
    pending_name: str,
    headers: list[str],
) -> tuple[int | None, int]:
    lowered = [token.strip().lower() for token in headers]
    step_column = lowered.index("step") if "step" in lowered else None
    target = pending_name.strip().lower()
    if target in lowered:
        return step_column, lowered.index(target)

    candidate_indexes = [
        idx for idx, token in enumerate(lowered) if token not in _MEAS_AUX_HEADER_TOKENS
    ]
    if candidate_indexes:
        return step_column, candidate_indexes[-1]
    return step_column, max(0, len(headers) - 1)


def _parse_meas_results_from_text(text: str) -> dict[str, Any]:
    measurements: dict[str, float] = {}
    measurement_text: dict[str, str] = {}
    measurement_display: dict[str, str] = {}
    measurement_steps: dict[str, list[dict[str, Any]]] = {}
    raw_lines = text.splitlines()
    measurement_order: list[str] = []
    pending_name: str | None = None
    pending_value_column = 0
    pending_step_column: int | None = None
    pending_header_seen = False

    for line in raw_lines:
        match_measure = _MEAS_MEASUREMENT_RE.match(line)
        if match_measure:
            pending_name = match_measure.group("name").strip()
            if pending_name and pending_name not in measurement_order:
                measurement_order.append(pending_name)
            if pending_name:
                measurement_steps.setdefault(pending_name, [])
            pending_value_column = 0
            pending_step_column = None
            pending_header_seen = False
            continue

        match_value = _MEAS_VALUE_RE.match(line)
        if match_value:
            name = match_value.group("name").strip()
            raw_value = match_value.group("value").strip()
            parsed_value = _parse_spice_number_token(raw_value)
            if parsed_value is None:
                continue
            value = parsed_value
            measurements[name] = value
            measurement_text[name] = raw_value
            measurement_display[name] = raw_value
            if name not in measurement_order:
                measurement_order.append(name)
            pending_name = None
            continue

        match_colon_line = _MEAS_COLON_LINE_RE.match(line)
        if match_colon_line:
            name = match_colon_line.group("name").strip()
            rhs = match_colon_line.group("rhs").strip()
            extracted = _extract_meas_numeric_from_rhs(rhs)
            if extracted is not None:
                value, raw_value = extracted
                measurements[name] = value
                measurement_text[name] = raw_value
                measurement_display[name] = rhs
                if name not in measurement_order:
                    measurement_order.append(name)
                pending_name = None
                continue

        if pending_name:
            stripped = line.strip()
            if not stripped:
                continue

            tokens = [token for token in re.split(r"\s+", stripped) if token]
            if (
                not pending_header_seen
                and tokens
                and all(_MEAS_HEADER_TOKEN_RE.match(token) for token in tokens)
            ):
                pending_header_seen = True
                pending_step_column, pending_value_column = _select_meas_table_columns(
                    pending_name=pending_name,
                    headers=tokens,
                )
                continue

            if pending_header_seen:
                step_token = (
                    tokens[pending_step_column]
                    if pending_step_column is not None and len(tokens) > pending_step_column
                    else None
                )
                if step_token is not None and _parse_spice_number_token(step_token) is None:
                    # End of this stepped measurement table; avoid leaking into
                    # unrelated numeric lines such as "Total elapsed time: ...".
                    pending_name = None
                    pending_header_seen = False
                    pending_value_column = 0
                    pending_step_column = None
                    continue

            raw_value = (
                tokens[pending_value_column] if len(tokens) > pending_value_column else None
            )
            parsed_value = _parse_spice_number_token(raw_value or "")
            if parsed_value is None:
                extracted = _extract_meas_numeric_from_rhs(stripped)
                if extracted is not None:
                    parsed_value, raw_value = extracted
            if parsed_value is None or raw_value is None:
                continue
            measurements[pending_name] = parsed_value
            measurement_text[pending_name] = raw_value
            measurement_display[pending_name] = stripped
            entry: dict[str, Any] = {"value": parsed_value, "value_text": raw_value}
            if pending_step_column is not None and len(tokens) > pending_step_column:
                step_value = _parse_spice_number_token(tokens[pending_step_column])
                if step_value is not None:
                    entry["step"] = int(step_value)
            measurement_steps.setdefault(pending_name, []).append(entry)

    items = [
        {
            "name": name,
            "value": measurements[name],
            "value_text": measurement_text.get(name),
            "steps": measurement_steps.get(name, []),
        }
        for name in measurement_order
        if name in measurements
    ]
    return {
        "count": len(items),
        "measurements": measurements,
        "measurements_text": measurement_text,
        "measurements_display": measurement_display,
        "measurement_steps": measurement_steps,
        "items": items,
    }


def _build_meas_statement(entry: dict[str, Any]) -> str:
    if "statement" in entry and str(entry["statement"]).strip():
        raw = str(entry["statement"]).strip()
        return raw if raw.lower().startswith(".meas") else f".meas {raw}"

    analysis = str(entry.get("analysis", "tran")).strip()
    name = str(entry.get("name", "")).strip()
    if not name:
        raise ValueError("measurement entry missing required 'name'")
    operation = str(entry.get("operation", "")).strip()
    target = str(entry.get("target", "")).strip()
    param_expr = str(entry.get("param", "")).strip()

    tokens = [".meas", analysis, name]
    if param_expr:
        tokens.extend(["PARAM", param_expr])
    else:
        if not operation or not target:
            raise ValueError(
                "measurement entry requires either 'statement', or ('operation' and 'target'), or 'param'"
            )
        tokens.extend([operation, target])
        for key in ("from", "to", "at", "when"):
            if key in entry and str(entry[key]).strip():
                tokens.extend([key.upper(), str(entry[key]).strip()])
    return " ".join(tokens)


def _build_meas_statements(measurements: list[dict[str, Any] | str]) -> list[str]:
    statements: list[str] = []
    for item in measurements:
        if isinstance(item, str):
            raw = item.strip()
            if not raw:
                continue
            statements.append(raw if raw.lower().startswith(".meas") else f".meas {raw}")
            continue
        if not isinstance(item, dict):
            raise ValueError("measurement entries must be dicts or strings")
        statements.append(_build_meas_statement(item))
    if not statements:
        raise ValueError("No valid measurement statements were produced.")
    return statements


def _write_meas_netlist(
    *,
    base_netlist_path: Path,
    statements: list[str],
    suffix: str = "meas",
) -> Path:
    base_text = _read_netlist_text(base_netlist_path)
    updated = _append_unique_lines_before_end(base_text, statements)
    output_path = base_netlist_path.with_name(f"{base_netlist_path.stem}_{suffix}{base_netlist_path.suffix}")
    output_path.write_text(updated, encoding="utf-8")
    return output_path


def _aggregate_numeric(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "stddev": None}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values))
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "stddev": math.sqrt(variance),
    }


def _metric_from_series(series: list[complex], statistic: str) -> float:
    if not series:
        raise ValueError("Vector series is empty.")
    stat = statistic.strip().lower()
    real_values = [value.real for value in series]
    abs_values = [abs(value) for value in series]
    if stat == "min":
        return min(real_values)
    if stat == "max":
        return max(real_values)
    if stat == "avg":
        return sum(real_values) / len(real_values)
    if stat == "rms":
        return math.sqrt(sum(value * value for value in real_values) / len(real_values))
    if stat == "pp":
        return max(real_values) - min(real_values)
    if stat == "final":
        return real_values[-1]
    if stat == "abs_max":
        return max(abs_values)
    raise ValueError("statistic must be one of: min, max, avg, rms, pp, final, abs_max")


def _evaluate_limits(
    *,
    value: float | None,
    minimum: float | None,
    maximum: float | None,
) -> tuple[bool, str]:
    if value is None:
        return False, "value is unavailable"
    checks: list[str] = []
    passed = True
    if minimum is not None:
        checks.append(f"value >= {minimum}")
        if value < minimum:
            passed = False
    if maximum is not None:
        checks.append(f"value <= {maximum}")
        if value > maximum:
            passed = False
    if not checks:
        return True, "no bounds provided"
    return passed, " and ".join(checks)


def _evaluate_target_tolerance(
    *,
    value: float | None,
    target: float | None,
    rel_tol_pct: float | None,
    abs_tol: float | None,
) -> tuple[bool, str, dict[str, float] | None]:
    if target is None and rel_tol_pct is None and abs_tol is None:
        return True, "no target/tolerance provided", None
    if target is None:
        return False, "target is required when rel_tol_pct/abs_tol is specified", None
    if value is None:
        return False, "value is unavailable", None

    allowed = 0.0
    rel_allowed = 0.0
    abs_allowed = 0.0
    checks: list[str] = []
    if rel_tol_pct is not None:
        rel_allowed = abs(float(target)) * (abs(float(rel_tol_pct)) / 100.0)
        checks.append(f"abs(value-target) <= {rel_allowed} (rel_tol_pct={rel_tol_pct})")
        allowed = max(allowed, rel_allowed)
    if abs_tol is not None:
        abs_allowed = abs(float(abs_tol))
        checks.append(f"abs(value-target) <= {abs_allowed} (abs_tol)")
        allowed = max(allowed, abs_allowed)
    if not checks:
        checks.append("value == target")

    lower = float(target) - allowed
    upper = float(target) + allowed
    passed = lower <= float(value) <= upper
    details = {
        "target": float(target),
        "allowed_abs_deviation": float(allowed),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "rel_allowed": float(rel_allowed),
        "abs_allowed": float(abs_allowed),
    }
    return passed, " and ".join(checks), details


def _assertion_requires_dataset(assertion: dict[str, Any]) -> bool:
    check_type = str(assertion.get("type", "vector_stat")).strip().lower()
    if check_type in {"all_of", "any_of"}:
        children = assertion.get("assertions") or assertion.get("checks") or []
        if not isinstance(children, list):
            return True
        for item in children:
            if isinstance(item, dict) and _assertion_requires_dataset(item):
                return True
        return False
    return check_type != "meas"


def _evaluate_verification_assertion(
    *,
    assertion: dict[str, Any],
    dataset: RawDataset | None,
    measurement_map: dict[str, float],
    default_id: str,
    fail_fast: bool,
) -> dict[str, Any]:
    check_id = str(assertion.get("id") or assertion.get("name") or default_id)
    check_type = str(assertion.get("type", "vector_stat")).strip().lower()
    if check_type in {"all_of", "any_of"}:
        children = assertion.get("assertions") or assertion.get("checks")
        if not isinstance(children, list) or not children:
            raise ValueError(f"{check_type} assertion requires non-empty `assertions` list")
        child_results: list[dict[str, Any]] = []
        for index, child in enumerate(children, start=1):
            if not isinstance(child, dict):
                child_results.append(
                    {
                        "id": f"{check_id}.{index}",
                        "type": "invalid",
                        "passed": False,
                        "error": "Child assertion must be an object",
                        "details": {"child": child},
                    }
                )
                if fail_fast and check_type == "all_of":
                    break
                continue
            child_result = _evaluate_verification_assertion(
                assertion=child,
                dataset=dataset,
                measurement_map=measurement_map,
                default_id=f"{check_id}.{index}",
                fail_fast=fail_fast,
            )
            child_results.append(child_result)
            if fail_fast:
                if check_type == "all_of" and not bool(child_result.get("passed")):
                    break
                if check_type == "any_of" and bool(child_result.get("passed")):
                    break
        child_passed = [bool(item.get("passed")) for item in child_results]
        passed = all(child_passed) if check_type == "all_of" else any(child_passed)
        return {
            "id": check_id,
            "type": check_type,
            "passed": passed,
            "value": None,
            "condition": "all child assertions must pass"
            if check_type == "all_of"
            else "at least one child assertion must pass",
            "details": {"assertion": assertion},
            "children": child_results,
            "child_count": len(child_results),
            "child_passed_count": sum(1 for item in child_results if bool(item.get("passed"))),
        }

    minimum_raw = assertion.get("min")
    maximum_raw = assertion.get("max")
    target_raw = assertion.get("target")
    rel_tol_raw = assertion.get("rel_tol_pct")
    abs_tol_raw = assertion.get("abs_tol")
    minimum = float(minimum_raw) if minimum_raw is not None else None
    maximum = float(maximum_raw) if maximum_raw is not None else None
    target = float(target_raw) if target_raw is not None else None
    rel_tol_pct = float(rel_tol_raw) if rel_tol_raw is not None else None
    abs_tol = float(abs_tol_raw) if abs_tol_raw is not None else None
    value: float | None = None
    details: dict[str, Any] = {"assertion": assertion}

    if check_type == "meas":
        meas_name = str(assertion.get("name", "")).strip()
        if not meas_name:
            raise ValueError("meas assertion requires `name`")
        value = measurement_map.get(meas_name)
        details["measurement_name"] = meas_name
    else:
        if dataset is None:
            raise ValueError("No RAW dataset available for non-meas assertions")
        step_index_raw = assertion.get("step_index")
        selected_step = _resolve_step_index(dataset, int(step_index_raw) if step_index_raw is not None else None)
        vector = str(assertion.get("vector", "")).strip()
        if not vector:
            raise ValueError(f"{check_type} assertion requires `vector`")
        if check_type == "vector_stat":
            statistic = str(assertion.get("statistic", "final"))
            series = dataset.get_vector(vector, step_index=selected_step)
            value = _metric_from_series(series, statistic)
            details.update({"vector": vector, "statistic": statistic, **_step_payload(dataset, selected_step)})
        elif check_type == "bandwidth":
            metric = str(assertion.get("metric", "lowpass_bandwidth_hz"))
            result = compute_bandwidth(
                frequency_hz=dataset.scale_values(step_index=selected_step),
                response=dataset.get_vector(vector, step_index=selected_step),
                reference=str(assertion.get("reference", "first")),
                drop_db=float(assertion.get("drop_db", 3.0)),
            )
            value = result.get(metric)
            details.update(
                {"vector": vector, "metric": metric, "result": result, **_step_payload(dataset, selected_step)}
            )
        elif check_type == "gain_phase_margin":
            metric = str(assertion.get("metric", "phase_margin_deg"))
            result = compute_gain_phase_margin(
                frequency_hz=dataset.scale_values(step_index=selected_step),
                response=dataset.get_vector(vector, step_index=selected_step),
            )
            value = result.get(metric)
            details.update(
                {"vector": vector, "metric": metric, "result": result, **_step_payload(dataset, selected_step)}
            )
        elif check_type == "rise_fall_time":
            metric = str(assertion.get("metric", "rise_time_s"))
            result = compute_rise_fall_time(
                time_s=dataset.scale_values(step_index=selected_step),
                signal=dataset.get_vector(vector, step_index=selected_step),
                low_threshold_pct=float(assertion.get("low_threshold_pct", 10.0)),
                high_threshold_pct=float(assertion.get("high_threshold_pct", 90.0)),
            )
            value = result.get(metric)
            details.update(
                {"vector": vector, "metric": metric, "result": result, **_step_payload(dataset, selected_step)}
            )
        elif check_type == "settling_time":
            metric = str(assertion.get("metric", "settling_time_s"))
            result = compute_settling_time(
                time_s=dataset.scale_values(step_index=selected_step),
                signal=dataset.get_vector(vector, step_index=selected_step),
                tolerance_percent=float(assertion.get("tolerance_percent", 2.0)),
                target_value=float(assertion["target_value"])
                if "target_value" in assertion and assertion["target_value"] is not None
                else None,
            )
            value = result.get(metric)
            details.update(
                {"vector": vector, "metric": metric, "result": result, **_step_payload(dataset, selected_step)}
            )
        else:
            raise ValueError(f"Unsupported assertion type '{check_type}'")

    passed_limits, condition_limits = _evaluate_limits(value=value, minimum=minimum, maximum=maximum)
    passed_tolerance, condition_tolerance, tolerance_details = _evaluate_target_tolerance(
        value=value,
        target=target,
        rel_tol_pct=rel_tol_pct,
        abs_tol=abs_tol,
    )
    conditions = [condition_limits]
    if condition_tolerance != "no target/tolerance provided":
        conditions.append(condition_tolerance)
    combined_condition = " and ".join(part for part in conditions if part)
    if tolerance_details is not None:
        details["tolerance"] = tolerance_details

    return {
        "id": check_id,
        "type": check_type,
        "passed": bool(passed_limits and passed_tolerance),
        "value": value,
        "min": minimum,
        "max": maximum,
        "target": target,
        "rel_tol_pct": rel_tol_pct,
        "abs_tol": abs_tol,
        "condition": combined_condition,
        "details": details,
    }


def _run_process_simulation_with_cancel(
    *,
    netlist_path: Path,
    ascii_raw: bool,
    timeout_seconds: int | None,
    cancel_requested: callable,
) -> tuple[SimulationRun, bool]:
    executable = _runner.ensure_executable()
    timeout = timeout_seconds or _runner.default_timeout_seconds
    command = [str(executable), "-b"]
    if ascii_raw:
        command.append("-ascii")
    command.append(str(netlist_path))

    started_at = _now_iso()
    start_ts = time.time()
    proc = subprocess.Popen(
        command,
        cwd=netlist_path.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    canceled = False
    timed_out = False

    while proc.poll() is None:
        if cancel_requested():
            canceled = True
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
            break
        if timeout is not None and (time.time() - start_ts) > timeout:
            timed_out = True
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
            break
        time.sleep(0.15)

    stdout, stderr = proc.communicate()
    return_code = proc.returncode if proc.returncode is not None else -1
    duration = time.time() - start_ts

    artifacts = sorted(path for path in netlist_path.parent.glob(f"{netlist_path.stem}*") if path.is_file())
    raw_files = [path for path in artifacts if path.suffix.lower() == ".raw"]
    log_path = netlist_path.with_suffix(".log")
    if not log_path.exists():
        candidates = sorted(netlist_path.parent.glob(f"{netlist_path.stem}*.log"))
        log_path = candidates[0] if candidates else None

    issues, warnings, diagnostics = analyze_log(log_path)
    if canceled:
        issues.append("Simulation was canceled by user request.")
        diagnostics.append(
            SimulationDiagnostic(
                category="canceled",
                severity="error",
                message="Simulation was canceled by user request.",
                suggestion="Restart the job if simulation output is still required.",
            )
        )
    elif timed_out:
        issues.append(f"LTspice timed out after {timeout} seconds.")
        diagnostics.append(
            SimulationDiagnostic(
                category="timeout",
                severity="error",
                message=f"LTspice timed out after {timeout} seconds.",
                suggestion="Increase timeout_seconds or simplify the simulation setup.",
            )
        )
    elif return_code != 0:
        issues.append(f"LTspice exited with return code {return_code}.")
        diagnostics.append(
            SimulationDiagnostic(
                category="process_error",
                severity="error",
                message=f"LTspice exited with return code {return_code}.",
                suggestion="Inspect stdout/stderr and LTspice log details for root cause.",
            )
        )
    if not raw_files and return_code == 0 and not canceled and not timed_out:
        warnings.append("No .raw output file was generated.")

    run = SimulationRun(
        run_id=datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
        netlist_path=netlist_path,
        command=command,
        ltspice_executable=executable,
        started_at=started_at,
        duration_seconds=duration,
        return_code=return_code,
        stdout=stdout or "",
        stderr=stderr or "",
        log_path=log_path,
        raw_files=raw_files,
        artifacts=artifacts,
        issues=issues,
        warnings=warnings,
        diagnostics=diagnostics,
    )
    return run, canceled


def _job_public_payload(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "created_at": job["created_at"],
        "archived_at": job.get("archived_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "cancel_requested": bool(job.get("cancel_requested", False)),
        "kind": job.get("kind"),
        "source_path": job.get("source_path"),
        "priority": int(job.get("priority", 50)),
        "max_retries": int(job.get("max_retries", 0)),
        "retry_count": int(job.get("retry_count", 0)),
        "queue_seq": job.get("queue_seq"),
        "run_id": job.get("run_id"),
        "error": job.get("error"),
        "summary": job.get("summary"),
    }


def _ensure_job_worker() -> None:
    global _job_worker_thread
    with _job_lock:
        if _job_worker_thread is not None and _job_worker_thread.is_alive():
            return
        _job_worker_stop.clear()

        def _worker() -> None:
            while not _job_worker_stop.is_set():
                try:
                    item = _job_queue.get(timeout=0.25)
                except queue.Empty:
                    continue
                if not (isinstance(item, tuple) and len(item) == 3 and isinstance(item[2], str)):
                    _job_queue.task_done()
                    continue
                _priority, _seq, job_id = item
                with _job_lock:
                    job = _jobs.get(job_id)
                if job is None:
                    _job_queue.task_done()
                    continue
                if str(job.get("status", "")).lower() != "queued":
                    _job_queue.task_done()
                    continue
                if job.get("cancel_requested"):
                    with _job_lock:
                        job["status"] = "canceled"
                        job["finished_at"] = _now_iso()
                        job["summary"] = "Canceled before start."
                        _archive_job_locked(job)
                        _save_job_state()
                        _save_job_history_state()
                    _job_queue.task_done()
                    continue
                with _job_lock:
                    job["status"] = "running"
                    job["started_at"] = _now_iso()
                    job["summary"] = "Running simulation."
                    _save_job_state()
                try:
                    run, canceled = _run_process_simulation_with_cancel(
                        netlist_path=Path(str(job["source_path"])).expanduser().resolve(),
                        ascii_raw=bool(job.get("ascii_raw", False)),
                        timeout_seconds=job.get("timeout_seconds"),
                        cancel_requested=lambda: bool(_jobs.get(job_id, {}).get("cancel_requested", False)),
                    )
                    _register_run(run)
                    with _job_lock:
                        job["run_id"] = run.run_id
                        job["error"] = None
                        if canceled:
                            job["status"] = "canceled"
                            job["summary"] = "Canceled while running."
                            job["finished_at"] = _now_iso()
                        elif run.succeeded:
                            job["status"] = "succeeded"
                            job["summary"] = "Simulation completed successfully."
                            job["finished_at"] = _now_iso()
                        else:
                            retry_count = int(job.get("retry_count", 0))
                            max_retries = int(job.get("max_retries", 0))
                            if retry_count < max_retries and not job.get("cancel_requested"):
                                job["retry_count"] = retry_count + 1
                                job["status"] = "queued"
                                job["started_at"] = None
                                job["finished_at"] = None
                                job["summary"] = (
                                    f"Attempt failed; retry {job['retry_count']} of {max_retries} scheduled."
                                )
                                job["error"] = "; ".join(run.issues[:3]) if run.issues else "Simulation failed."
                                _enqueue_job_locked(job_id)
                            else:
                                job["status"] = "failed"
                                job["summary"] = "; ".join(run.issues[:3]) if run.issues else "Simulation failed."
                                job["error"] = "; ".join(run.issues[:3]) if run.issues else "Simulation failed."
                                job["finished_at"] = _now_iso()
                        if str(job.get("status", "")).lower() in _JOB_TERMINAL_STATUSES:
                            _archive_job_locked(job)
                        _save_job_state()
                        _save_job_history_state()
                except Exception as exc:  # noqa: BLE001
                    with _job_lock:
                        retry_count = int(job.get("retry_count", 0))
                        max_retries = int(job.get("max_retries", 0))
                        if retry_count < max_retries and not job.get("cancel_requested"):
                            job["retry_count"] = retry_count + 1
                            job["status"] = "queued"
                            job["started_at"] = None
                            job["finished_at"] = None
                            job["summary"] = (
                                f"Runtime error; retry {job['retry_count']} of {max_retries} scheduled."
                            )
                            job["error"] = str(exc)
                            _enqueue_job_locked(job_id)
                        else:
                            job["status"] = "failed"
                            job["error"] = str(exc)
                            job["summary"] = f"Runtime error: {exc}"
                            job["finished_at"] = _now_iso()
                            _archive_job_locked(job)
                        _save_job_state()
                        _save_job_history_state()
                _job_queue.task_done()

        _job_worker_thread = threading.Thread(
            target=_worker,
            name="ltspice-mcp-job-worker",
            daemon=True,
        )
        _job_worker_thread.start()


def _queue_simulation_job(
    *,
    source_path: Path,
    ascii_raw: bool,
    timeout_seconds: int | None,
    kind: str,
    priority: int,
    max_retries: int,
) -> dict[str, Any]:
    _ensure_job_worker()
    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "status": "queued",
        "created_at": _now_iso(),
        "started_at": None,
        "finished_at": None,
        "cancel_requested": False,
        "ascii_raw": bool(ascii_raw),
        "timeout_seconds": timeout_seconds,
        "kind": kind,
        "source_path": str(source_path),
        "priority": int(priority),
        "max_retries": max(0, int(max_retries)),
        "retry_count": 0,
        "queue_seq": None,
        "run_id": None,
        "error": None,
        "summary": "Queued",
    }
    with _job_lock:
        _jobs[job_id] = job
        _job_order.append(job_id)
        _enqueue_job_locked(job_id)
        _save_job_state()
    return _job_public_payload(job)


def _stop_job_worker() -> None:
    global _job_worker_thread, _job_seq
    thread: threading.Thread | None
    with _job_lock:
        _job_worker_stop.set()
        thread = _job_worker_thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=2)
    with _job_lock:
        _save_job_state()
        _save_job_history_state()
        _job_worker_thread = None
        while True:
            try:
                _job_queue.get_nowait()
                _job_queue.task_done()
            except queue.Empty:
                break
        _jobs.clear()
        _job_order.clear()
        _job_seq = 0


def _parse_schematic_geometry(path: Path) -> dict[str, Any]:
    text = read_text_auto(path)
    components: list[dict[str, Any]] = []
    wires: list[tuple[int, int, int, int]] = []
    flags: list[dict[str, Any]] = []
    texts: list[dict[str, Any]] = []
    current_component: dict[str, Any] | None = None
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        keyword = parts[0].upper()
        if keyword == "SYMBOL" and len(parts) >= 5:
            try:
                component = {
                    "symbol": parts[1],
                    "x": int(parts[2]),
                    "y": int(parts[3]),
                    "orientation": parts[4],
                    "reference": None,
                    "line": line_no,
                }
            except Exception:
                current_component = None
                continue
            components.append(component)
            current_component = component
            continue
        if keyword == "SYMATTR" and len(parts) >= 3 and parts[1].lower() == "instname":
            if current_component is not None:
                current_component["reference"] = " ".join(parts[2:])
            continue
        if keyword == "WIRE" and len(parts) >= 5:
            try:
                wires.append((int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])))
            except Exception:
                pass
            continue
        if keyword == "FLAG" and len(parts) >= 4:
            try:
                flags.append(
                    {
                        "x": int(parts[1]),
                        "y": int(parts[2]),
                        "name": parts[3],
                        "line": line_no,
                    }
                )
            except Exception:
                pass
            continue
        if keyword == "TEXT" and len(parts) >= 5:
            try:
                texts.append(
                    {
                        "x": int(parts[1]),
                        "y": int(parts[2]),
                        "line": line_no,
                        "raw": raw,
                    }
                )
            except Exception:
                pass
    return {"components": components, "wires": wires, "flags": flags, "texts": texts}


def _segment_intersection(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[bool, tuple[int, int] | None]:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    a_horizontal = ay1 == ay2
    b_horizontal = by1 == by2
    if a_horizontal == b_horizontal:
        return False, None
    if a_horizontal:
        y = ay1
        x = bx1
        if min(ax1, ax2) < x < max(ax1, ax2) and min(by1, by2) < y < max(by1, by2):
            return True, (x, y)
        return False, None
    x = ax1
    y = by1
    if min(bx1, bx2) < x < max(bx1, bx2) and min(ay1, ay2) < y < max(ay1, ay2):
        return True, (x, y)
    return False, None


def _analyze_schematic_visual_quality(path: Path) -> dict[str, Any]:
    geometry = _parse_schematic_geometry(path)
    components = geometry["components"]
    wires = geometry["wires"]
    findings: list[dict[str, Any]] = []
    score = 100.0

    for index, left in enumerate(components):
        lx, ly = int(left["x"]), int(left["y"])
        for right in components[index + 1 :]:
            rx, ry = int(right["x"]), int(right["y"])
            dist = math.hypot(lx - rx, ly - ry)
            if dist < 24:
                score -= 10
                findings.append(
                    {
                        "type": "component_overlap",
                        "severity": "error",
                        "components": [left.get("reference"), right.get("reference")],
                        "distance": round(dist, 3),
                        "suggestion": {
                            "action": "move_component",
                            "reference": right.get("reference"),
                            "from": [rx, ry],
                            "to": [rx + 96, ry],
                        },
                    }
                )
            elif dist < 72:
                score -= 3
                findings.append(
                    {
                        "type": "component_crowding",
                        "severity": "warning",
                        "components": [left.get("reference"), right.get("reference")],
                        "distance": round(dist, 3),
                        "suggestion": {
                            "action": "increase_spacing",
                            "reference": right.get("reference"),
                            "from": [rx, ry],
                            "to": [rx + 64, ry],
                        },
                    }
                )

    for idx, left in enumerate(wires):
        for right in wires[idx + 1 :]:
            intersects, point = _segment_intersection(left, right)
            if not intersects:
                continue
            score -= 2
            findings.append(
                {
                    "type": "wire_crossing",
                    "severity": "warning",
                    "point": list(point) if point else None,
                    "wire_a": list(left),
                    "wire_b": list(right),
                    "suggestion": {
                        "action": "reroute_wire",
                        "around": list(point) if point else None,
                    },
                }
            )

    score = max(0.0, min(100.0, score))
    return {
        "asc_path": str(path),
        "score": round(score, 3),
        "component_count": len(components),
        "wire_count": len(wires),
        "finding_count": len(findings),
        "findings": findings,
    }


def _normalize_schematic_grid(
    *,
    source: Path,
    output: Path,
    grid: int,
) -> dict[str, Any]:
    text = read_text_auto(source)
    lines = text.splitlines()
    normalized_lines: list[str] = []
    coord_adjustments = 0

    def _snap(value: int) -> int:
        return int(round(value / grid) * grid)

    entries: list[dict[str, Any]] = []
    layout_points: list[tuple[int, int]] = []

    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            entries.append({"kind": "blank", "raw": ""})
            continue

        text_match = re.match(r"^TEXT\s+(-?\d+)\s+(-?\d+)\s+(.+)$", stripped, re.IGNORECASE)
        if text_match:
            x_old = int(text_match.group(1))
            y_old = int(text_match.group(2))
            x_new = _snap(x_old)
            y_new = _snap(y_old)
            if x_new != x_old:
                coord_adjustments += 1
            if y_new != y_old:
                coord_adjustments += 1
            tail = text_match.group(3)
            is_directive = "!." in tail.casefold()
            entry = {
                "kind": "text",
                "x": x_new,
                "y": y_new,
                "tail": tail,
                "is_directive": is_directive,
            }
            entries.append(entry)
            if not is_directive:
                layout_points.append((x_new, y_new))
            continue

        parts = stripped.split()
        if not parts:
            entries.append({"kind": "blank", "raw": ""})
            continue
        keyword = parts[0].upper()
        mutable = list(parts)
        shift_indices: tuple[int, ...] = ()
        try:
            if keyword == "SYMBOL" and len(mutable) >= 4:
                shift_indices = (2, 3)
            elif keyword == "WIRE" and len(mutable) >= 5:
                shift_indices = (1, 2, 3, 4)
            elif keyword == "FLAG" and len(mutable) >= 3:
                shift_indices = (1, 2)
            elif keyword == "WINDOW" and len(mutable) >= 4:
                # WINDOW coordinates are symbol-local offsets; snap but do not translate globally.
                shift_indices = (2, 3)

            for index in shift_indices:
                old = int(mutable[index])
                new = _snap(old)
                if new != old:
                    coord_adjustments += 1
                mutable[index] = str(new)
        except Exception:
            entries.append({"kind": "raw", "raw": stripped})
            continue

        entries.append({"kind": "tokens", "keyword": keyword, "parts": mutable})
        if keyword == "WIRE" and len(mutable) >= 5:
            layout_points.append((int(mutable[1]), int(mutable[2])))
            layout_points.append((int(mutable[3]), int(mutable[4])))
        elif keyword == "SYMBOL" and len(mutable) >= 4:
            layout_points.append((int(mutable[2]), int(mutable[3])))
        elif keyword == "FLAG" and len(mutable) >= 3:
            layout_points.append((int(mutable[1]), int(mutable[2])))

    style_grid = max(2, int(_SCHEMATIC_STYLE_PROFILE["grid"]))
    style_anchor_x = _SCHEMATIC_STYLE_PROFILE["anchor_x"]
    style_anchor_y = _SCHEMATIC_STYLE_PROFILE["anchor_y"]
    style_directive_x = _SCHEMATIC_STYLE_PROFILE["directive_x"]
    style_directive_gap_y = _SCHEMATIC_STYLE_PROFILE["directive_gap_y"]
    style_directive_step = _SCHEMATIC_STYLE_PROFILE["directive_line_step"]

    min_x = min((point[0] for point in layout_points), default=style_anchor_x)
    min_y = min((point[1] for point in layout_points), default=style_anchor_y)
    dx = _snap(style_anchor_x - min_x)
    dy = _snap(style_anchor_y - min_y)
    if style_grid != grid:
        # Respect the caller grid while keeping profile-aligned placement anchors.
        dx = int(round(dx / grid) * grid)
        dy = int(round(dy / grid) * grid)

    shifted_layout_points: list[tuple[int, int]] = []
    for entry in entries:
        kind = entry.get("kind")
        if kind == "tokens":
            keyword = str(entry.get("keyword", "")).upper()
            parts = list(entry.get("parts") or [])
            try:
                if keyword == "SYMBOL" and len(parts) >= 4:
                    for idx in (2, 3):
                        old = int(parts[idx])
                        new = old + (dx if idx == 2 else dy)
                        if new != old:
                            coord_adjustments += 1
                        parts[idx] = str(new)
                    shifted_layout_points.append((int(parts[2]), int(parts[3])))
                elif keyword == "WIRE" and len(parts) >= 5:
                    for idx in (1, 2, 3, 4):
                        old = int(parts[idx])
                        new = old + (dx if idx in {1, 3} else dy)
                        if new != old:
                            coord_adjustments += 1
                        parts[idx] = str(new)
                    shifted_layout_points.append((int(parts[1]), int(parts[2])))
                    shifted_layout_points.append((int(parts[3]), int(parts[4])))
                elif keyword == "FLAG" and len(parts) >= 3:
                    for idx in (1, 2):
                        old = int(parts[idx])
                        new = old + (dx if idx == 1 else dy)
                        if new != old:
                            coord_adjustments += 1
                        parts[idx] = str(new)
                    shifted_layout_points.append((int(parts[1]), int(parts[2])))
            except Exception:
                pass
            entry["parts"] = parts
        elif kind == "text" and not bool(entry.get("is_directive", False)):
            old_x = int(entry["x"])
            old_y = int(entry["y"])
            new_x = old_x + dx
            new_y = old_y + dy
            if new_x != old_x:
                coord_adjustments += 1
            if new_y != old_y:
                coord_adjustments += 1
            entry["x"] = new_x
            entry["y"] = new_y
            shifted_layout_points.append((new_x, new_y))

    max_y = max((point[1] for point in shifted_layout_points), default=style_anchor_y)
    directive_base_y = _snap(max_y + style_directive_gap_y)
    directive_index = 0
    for entry in entries:
        if entry.get("kind") != "text" or not bool(entry.get("is_directive", False)):
            continue
        old_x = int(entry["x"])
        old_y = int(entry["y"])
        new_x = _snap(style_directive_x)
        new_y = _snap(directive_base_y + directive_index * style_directive_step)
        directive_index += 1
        if new_x != old_x:
            coord_adjustments += 1
        if new_y != old_y:
            coord_adjustments += 1
        entry["x"] = new_x
        entry["y"] = new_y

    for entry in entries:
        kind = entry.get("kind")
        if kind == "blank":
            normalized_lines.append("")
        elif kind == "raw":
            normalized_lines.append(str(entry.get("raw", "")))
        elif kind == "text":
            normalized_lines.append(
                f"TEXT {int(entry['x'])} {int(entry['y'])} {str(entry['tail'])}".rstrip()
            )
        else:
            parts = entry.get("parts") or []
            normalized_lines.append(" ".join(str(item) for item in parts))

    output.write_text("\n".join(normalized_lines).rstrip() + "\n", encoding="utf-8")
    return {
        "source_path": str(source),
        "output_path": str(output),
        "grid": grid,
        "applied_style_profile": dict(_SCHEMATIC_STYLE_PROFILE),
        "translation": {"dx": dx, "dy": dy},
        "coord_adjustments": coord_adjustments,
    }


def _resolve_daemon_log_dir(log_dir: str | None = None) -> Path:
    if log_dir:
        return Path(log_dir).expanduser().resolve()
    env_dir = os.getenv("LTSPICE_MCP_DAEMON_LOG_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return (_runner.workdir / "daemon" / "logs").expanduser().resolve()


def _list_daemon_log_files(log_dir: Path, *, limit: int = 20) -> list[Path]:
    if not log_dir.exists():
        return []
    files = sorted(
        [path for path in log_dir.glob("ltspice-mcp-daemon-*.log") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return files[: max(1, limit)]


def _resolve_daemon_log_path(log_path: str | None = None, log_dir: str | None = None) -> Path:
    if log_path:
        target = Path(log_path).expanduser().resolve()
        if not target.exists():
            raise FileNotFoundError(f"Daemon log file not found: {target}")
        return target
    resolved_dir = _resolve_daemon_log_dir(log_dir)
    files = _list_daemon_log_files(resolved_dir, limit=1)
    if not files:
        raise FileNotFoundError(f"No daemon log files found in {resolved_dir}")
    return files[0]


def _parse_structured_log_payload(line: str, marker: str) -> dict[str, Any] | None:
    if marker not in line:
        return None
    _, suffix = line.split(marker, 1)
    suffix = suffix.strip()
    if not suffix:
        return None
    try:
        payload = json.loads(suffix)
    except Exception:  # noqa: BLE001
        return None
    return payload if isinstance(payload, dict) else None


def _is_recent_error_line(line: str, *, include_warnings: bool) -> tuple[bool, str]:
    lowered = line.lower()
    if " trace" in lowered:
        return False, "trace"
    if " critical " in lowered or lowered.startswith("critical:"):
        return True, "critical"
    if " error " in lowered or lowered.startswith("error:"):
        return True, "error"
    if include_warnings and (" warning " in lowered or lowered.startswith("warning:")):
        return True, "warning"
    if "exception" in lowered or "traceback" in lowered:
        return True, "exception"
    return False, "info"


def _collect_recent_log_entries(
    *,
    limit: int,
    include_warnings: bool,
    log_count: int,
    log_dir: str | None = None,
) -> list[dict[str, Any]]:
    resolved_dir = _resolve_daemon_log_dir(log_dir)
    log_files = _list_daemon_log_files(resolved_dir, limit=max(1, log_count))
    if not log_files:
        return []
    entries: list[dict[str, Any]] = []
    for log_file in log_files:
        lines = read_text_auto(log_file).splitlines()
        for line_no, line in enumerate(lines, start=1):
            include_line, inferred_level = _is_recent_error_line(
                line,
                include_warnings=include_warnings,
            )
            tool_payload = _parse_structured_log_payload(line, "mcp_tool ")
            if tool_payload and str(tool_payload.get("event")) == "tool_call_error":
                include_line = True
                inferred_level = "error"
            capture_payload = _parse_structured_log_payload(line, "ltspice_capture ")
            if capture_payload:
                capture_event = str(capture_payload.get("event", ""))
                if capture_event in {
                    "capture_open_failed",
                    "capture_screencapturekit_failed",
                    "capture_screencapture_failed",
                    "capture_file_missing",
                }:
                    include_line = True
                    inferred_level = "error"
                elif include_warnings and capture_event == "capture_close_incomplete":
                    include_line = True
                    inferred_level = "warning"

            if not include_line:
                continue
            entries.append(
                {
                    "log_path": str(log_file),
                    "line_number": line_no,
                    "level": inferred_level,
                    "message": line,
                    "tool_event": tool_payload,
                    "capture_event": capture_payload,
                }
            )
    return entries[-max(1, limit) :]


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


def _run_has_convergence_issue(run: SimulationRun) -> bool:
    if _has_convergence_issue(run.issues):
        return True
    diagnostic_messages = [
        diagnostic.message
        for diagnostic in run.diagnostics
        if str(diagnostic.severity).lower() == "error"
    ]
    return _has_convergence_issue(diagnostic_messages)


def _prepare_convergence_retry_netlist(netlist_path: Path) -> tuple[Path, list[str]]:
    stem = netlist_path.stem
    suffix = netlist_path.suffix or ".cir"
    retry_path = netlist_path.with_name(f"{stem}_convergence_retry{suffix}")
    if retry_path == netlist_path:
        retry_path = netlist_path.with_name(f"{stem}_convergence_retry_1{suffix}")
    text = read_text_auto(netlist_path)
    retry_path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    applied_lines = _apply_convergence_options(retry_path)
    return retry_path, applied_lines


def _run_simulation_with_auto_convergence_retry(
    *,
    netlist_path: Path,
    ascii_raw: bool,
    timeout_seconds: int | None,
    show_ui: bool | None,
    open_raw_after_run: bool,
) -> tuple[SimulationRun, list[dict[str, Any]], bool, dict[str, Any]]:
    primary_run, ui_events, effective_ui = _run_simulation_with_ui(
        netlist_path=netlist_path,
        ascii_raw=ascii_raw,
        timeout_seconds=timeout_seconds,
        show_ui=show_ui,
        open_raw_after_run=open_raw_after_run,
    )
    attempts: list[dict[str, Any]] = [
        {
            "attempt": 1,
            "kind": "primary",
            "run_id": primary_run.run_id,
            "netlist_path": str(primary_run.netlist_path),
            "succeeded": primary_run.succeeded,
            "issues": list(primary_run.issues),
        }
    ]
    retry_payload: dict[str, Any] = {
        "triggered": False,
        "reason": None,
        "fallback_netlist_path": None,
        "applied_lines": [],
        "attempts": attempts,
    }
    if primary_run.succeeded or not _run_has_convergence_issue(primary_run):
        retry_payload["reason"] = "not_required"
        return primary_run, ui_events, effective_ui, retry_payload

    retry_path, applied_lines = _prepare_convergence_retry_netlist(netlist_path)
    fallback_run, _, _ = _run_simulation_with_ui(
        netlist_path=retry_path,
        ascii_raw=ascii_raw,
        timeout_seconds=timeout_seconds,
        show_ui=False,
        open_raw_after_run=False,
    )
    attempts.append(
        {
            "attempt": 2,
            "kind": "convergence_retry",
            "run_id": fallback_run.run_id,
            "netlist_path": str(fallback_run.netlist_path),
            "succeeded": fallback_run.succeeded,
            "issues": list(fallback_run.issues),
        }
    )
    retry_payload.update(
        {
            "triggered": True,
            "reason": "convergence_detected",
            "fallback_netlist_path": str(retry_path),
            "applied_lines": applied_lines,
            "fallback_run_id": fallback_run.run_id,
        }
    )
    if fallback_run.succeeded and not fallback_run.issues:
        retry_payload["selected_attempt"] = "fallback"
        return fallback_run, ui_events, effective_ui, retry_payload
    retry_payload["selected_attempt"] = "primary"
    return primary_run, ui_events, effective_ui, retry_payload


def _sanitize_representation(value: str) -> str:
    valid = {"auto", "real", "rectangular", "magnitude-phase", "both"}
    if value not in valid:
        raise ValueError(f"representation must be one of: {', '.join(sorted(valid))}")
    return value


_SIM_DIRECTIVE_PREFIXES = (
    ".ac",
    ".dc",
    ".four",
    ".noise",
    ".op",
    ".pz",
    ".tf",
    ".tran",
)

_INTENT_TEMPLATE_MAP: dict[str, str] = {
    "rc_lowpass": "rc_lowpass_ac",
    "rc_highpass": "rc_highpass_ac",
    "non_inverting_amplifier": "non_inverting_opamp_spec",
    "zener_regulator": "zener_regulator_dc",
}

_INTENT_DEFAULT_PARAMETERS: dict[str, dict[str, Any]] = {
    "rc_lowpass": {
        "vin_ac": "1",
        "r_value": "1k",
        "c_value": "100n",
        "ac_points": "30",
        "f_start": "10",
        "f_stop": "1e6",
    },
    "rc_highpass": {
        "vin_ac": "1",
        "r_value": "1k",
        "c_value": "100n",
        "ac_points": "30",
        "f_start": "10",
        "f_stop": "1e6",
    },
    "non_inverting_amplifier": {
        "vin_signal": "SINE(0 0.1 1k) AC 1",
        "rf_value": "10k",
        "rg_value": "1k",
        "vplus": "10",
        "vminus": "-10",
        "tran_stop": "10m",
    },
    "zener_regulator": {
        "vin_dc": "12",
        "r_series": "330",
        "r_load": "1k",
        "zener_voltage": "5.1",
        "dc_start": "6",
        "dc_stop": "18",
        "dc_step": "0.25",
    },
}

_FLOATING_NODE_RE = re.compile(r"(?i)\bnode\s+([a-zA-Z0-9_.$:+-]+)\b.*\bfloating\b")
_MISSING_INCLUDE_RE = re.compile(
    r"(?i)(?:could not open include file|unable to open .*?)(?:\s*[:\"]\s*)([^\"\s]+)"
)
_MISSING_SUBCKT_RE = re.compile(r"(?i)unknown subcircuit(?: called in:)?\s*(.*)")
_MISSING_MODEL_RE = re.compile(
    r"(?i)(?:can't find definition of model|unknown model|could not find model)\s*\"?([a-zA-Z0-9_.:$+-]+)\"?"
)
_SUBCKT_DEF_RE = re.compile(r"(?im)^\s*\.subckt\s+([^\s]+)")
_MODEL_DEF_RE = re.compile(r"(?im)^\s*\.model\s+([^\s]+)")
_CONVERGENCE_RE = re.compile(
    r"(?i)(time step too small|convergence failed|gmin stepping failed|source stepping failed|newton iteration failed)"
)
_MODEL_FILE_SUFFIXES = {".lib", ".sub", ".mod", ".cir", ".spi", ".txt"}
_DEFAULT_MODEL_SEARCH_PATHS = [
    "~/Documents/LTspice/lib/sub",
    "~/Documents/LTspiceXVII/lib/sub",
    "/Applications/LTspice.app/Contents/lib/sub",
    "/Applications/LTspice.app/Contents/lib/cmp",
]


def _normalize_intent(intent: str) -> str:
    normalized = intent.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in _INTENT_TEMPLATE_MAP:
        raise ValueError(
            "intent must be one of: "
            + ", ".join(sorted(_INTENT_TEMPLATE_MAP.keys()))
        )
    return normalized


def _merge_intent_parameters(intent: str, parameters: dict[str, Any] | None) -> dict[str, Any]:
    defaults = dict(_INTENT_DEFAULT_PARAMETERS.get(intent, {}))
    for key, value in (parameters or {}).items():
        defaults[str(key)] = value
    return defaults


def _extract_model_issues_from_text(log_text: str) -> dict[str, Any]:
    missing_includes: set[str] = set()
    missing_subcircuits: set[str] = set()
    missing_models: set[str] = set()
    matched_lines: list[str] = []
    for raw in log_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        matched = False
        include_match = _MISSING_INCLUDE_RE.search(line)
        if include_match:
            missing_includes.add(include_match.group(1).strip().strip('"'))
            matched = True
        subckt_match = _MISSING_SUBCKT_RE.search(line)
        if subckt_match:
            value = subckt_match.group(1).strip()
            if value:
                missing_subcircuits.add(value)
            matched = True
        model_match = _MISSING_MODEL_RE.search(line)
        if model_match:
            missing_models.add(model_match.group(1).strip())
            matched = True
        if matched:
            matched_lines.append(line)
    return {
        "missing_include_files": sorted(missing_includes),
        "missing_subcircuits": sorted(missing_subcircuits),
        "missing_models": sorted(missing_models),
        "matched_lines": matched_lines,
        "has_model_issues": bool(missing_includes or missing_subcircuits or missing_models),
    }


def _discover_model_names(model_text: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for match in _MODEL_DEF_RE.finditer(model_text):
        name = match.group(1).strip()
        lowered = name.lower()
        if not name or lowered in seen:
            continue
        seen.add(lowered)
        names.append(name)
    return names


def _resolve_model_search_paths(
    *,
    run_workdir: Path,
    extra_paths: list[str] | None = None,
    source_path: Path | None = None,
) -> list[Path]:
    candidates: list[Path] = []
    if source_path is not None:
        candidates.append(source_path.expanduser().resolve().parent)
    candidates.append((run_workdir / "models").resolve())
    candidates.append(run_workdir.resolve())
    for default_path in _DEFAULT_MODEL_SEARCH_PATHS:
        candidates.append(Path(default_path).expanduser().resolve())
    for raw in extra_paths or []:
        candidates.append(Path(raw).expanduser().resolve())

    resolved: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.exists() or not candidate.is_dir():
            continue
        resolved.append(candidate)
    return resolved


def _scan_model_inventory(
    *,
    search_paths: list[Path],
    max_scan_files: int,
) -> dict[str, Any]:
    scanned_files: list[Path] = []
    filename_to_paths: dict[str, list[Path]] = {}
    subckt_index: dict[str, list[Path]] = {}
    model_index: dict[str, list[Path]] = {}
    max_files = max(1, int(max_scan_files))
    truncated = False

    for root in search_paths:
        try:
            iterator = root.rglob("*")
        except Exception:
            continue
        for candidate in iterator:
            if len(scanned_files) >= max_files:
                truncated = True
                break
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in _MODEL_FILE_SUFFIXES:
                continue
            scanned_files.append(candidate)
            key = candidate.name.lower()
            filename_to_paths.setdefault(key, []).append(candidate)

            try:
                text = read_text_auto(candidate)
            except Exception:
                continue
            for subckt in _discover_subckt_names(text):
                subckt_index.setdefault(subckt.lower(), []).append(candidate)
            for model in _discover_model_names(text):
                model_index.setdefault(model.lower(), []).append(candidate)
        if truncated:
            break

    return {
        "search_paths": [str(path) for path in search_paths],
        "scanned_file_count": len(scanned_files),
        "scan_truncated": truncated,
        "files": scanned_files,
        "filename_to_paths": filename_to_paths,
        "subckt_index": subckt_index,
        "model_index": model_index,
    }


def _rank_matches(target: str, candidates: list[str], *, limit: int = 5) -> list[dict[str, Any]]:
    needle = target.strip().lower()
    if not needle:
        return []
    normalized = sorted({candidate.strip() for candidate in candidates if candidate.strip()})
    if not normalized:
        return []
    direct_hits = [candidate for candidate in normalized if needle == candidate.lower()]
    contains_hits = [candidate for candidate in normalized if needle in candidate.lower() and candidate not in direct_hits]
    fuzzy_hits = difflib.get_close_matches(needle, [item.lower() for item in normalized], n=limit * 2, cutoff=0.35)
    lookup = {item.lower(): item for item in normalized}
    ordered: list[str] = []
    for entry in [*direct_hits, *contains_hits, *[lookup[item] for item in fuzzy_hits if item in lookup]]:
        if entry in ordered:
            continue
        ordered.append(entry)
        if len(ordered) >= limit:
            break
    ranked: list[dict[str, Any]] = []
    for entry in ordered:
        score = difflib.SequenceMatcher(a=needle, b=entry.lower()).ratio()
        ranked.append({"candidate": entry, "score": round(float(score), 4)})
    return ranked


def _suggest_model_resolutions(
    *,
    issues: dict[str, Any],
    inventory: dict[str, Any],
    limit_per_issue: int = 5,
) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    filename_map = inventory.get("filename_to_paths", {})
    subckt_index = inventory.get("subckt_index", {})
    model_index = inventory.get("model_index", {})

    for missing_include in issues.get("missing_include_files", []):
        filename = Path(str(missing_include)).name.lower()
        direct_paths = [str(path) for path in filename_map.get(filename, [])]
        ranked_names = _rank_matches(
            filename,
            list(filename_map.keys()),
            limit=limit_per_issue,
        )
        ranked_paths: list[dict[str, Any]] = []
        for ranked in ranked_names:
            candidate_name = str(ranked["candidate"]).lower()
            for candidate_path in filename_map.get(candidate_name, [])[:limit_per_issue]:
                ranked_paths.append(
                    {
                        "path": str(candidate_path),
                        "score": ranked["score"],
                    }
                )
                if len(ranked_paths) >= limit_per_issue:
                    break
            if len(ranked_paths) >= limit_per_issue:
                break
        suggestions.append(
            {
                "issue_type": "missing_include_file",
                "missing": missing_include,
                "direct_matches": direct_paths[:limit_per_issue],
                "best_matches": ranked_paths,
            }
        )

    for missing_subckt in issues.get("missing_subcircuits", []):
        direct = [str(path) for path in subckt_index.get(str(missing_subckt).lower(), [])]
        ranked = _rank_matches(
            str(missing_subckt),
            list(subckt_index.keys()),
            limit=limit_per_issue,
        )
        ranked_paths: list[dict[str, Any]] = []
        for item in ranked:
            candidate = str(item["candidate"]).lower()
            for source in subckt_index.get(candidate, [])[:limit_per_issue]:
                ranked_paths.append(
                    {
                        "name": item["candidate"],
                        "path": str(source),
                        "score": item["score"],
                    }
                )
                if len(ranked_paths) >= limit_per_issue:
                    break
            if len(ranked_paths) >= limit_per_issue:
                break
        suggestions.append(
            {
                "issue_type": "missing_subcircuit",
                "missing": missing_subckt,
                "direct_matches": direct[:limit_per_issue],
                "best_matches": ranked_paths,
            }
        )

    for missing_model in issues.get("missing_models", []):
        direct = [str(path) for path in model_index.get(str(missing_model).lower(), [])]
        ranked = _rank_matches(
            str(missing_model),
            list(model_index.keys()),
            limit=limit_per_issue,
        )
        ranked_paths: list[dict[str, Any]] = []
        for item in ranked:
            candidate = str(item["candidate"]).lower()
            for source in model_index.get(candidate, [])[:limit_per_issue]:
                ranked_paths.append(
                    {
                        "name": item["candidate"],
                        "path": str(source),
                        "score": item["score"],
                    }
                )
                if len(ranked_paths) >= limit_per_issue:
                    break
            if len(ranked_paths) >= limit_per_issue:
                break
        suggestions.append(
            {
                "issue_type": "missing_model",
                "missing": missing_model,
                "direct_matches": direct[:limit_per_issue],
                "best_matches": ranked_paths,
            }
        )
    return suggestions


def _extract_floating_nodes(messages: list[str]) -> list[str]:
    nodes: list[str] = []
    seen: set[str] = set()
    for message in messages:
        for match in _FLOATING_NODE_RE.finditer(message):
            node = match.group(1).strip()
            lowered = node.lower()
            if lowered in {"0", "gnd", "ground"}:
                continue
            if lowered in seen:
                continue
            seen.add(lowered)
            nodes.append(node)
    return nodes


def _append_lines_if_missing(path: Path, lines: list[str]) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    raw_lines = text.splitlines()
    existing = {line.strip().lower() for line in raw_lines if line.strip()}
    additions: list[str] = []
    for line in lines:
        normalized = line.strip().lower()
        if not normalized or normalized in existing:
            continue
        additions.append(line)
        existing.add(normalized)
    if not additions:
        return []
    end_index = next(
        (idx for idx, line in enumerate(raw_lines) if line.strip().lower() == ".end"),
        None,
    )
    if end_index is None:
        raw_lines.extend(additions)
        raw_lines.append(".end")
    else:
        raw_lines[end_index:end_index] = additions
    path.write_text("\n".join(raw_lines).rstrip() + "\n", encoding="utf-8")
    return additions


def _resolve_sidecar_netlist_path(asc_path: Path) -> Path | None:
    if asc_path.suffix.lower() != ".asc":
        return asc_path
    for suffix in (".cir", ".net", ".sp", ".spi"):
        candidate = asc_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def _resolve_schematic_simulation_target(
    path: Path,
    *,
    require_sidecar_on_macos: bool = True,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    suffix = resolved.suffix.lower()
    is_macos = platform.system() == "Darwin"
    candidate_paths = [
        str(resolved.with_suffix(ext))
        for ext in (".cir", ".net", ".sp", ".spi")
    ]

    if suffix != ".asc":
        return {
            "input_path": str(resolved),
            "platform": platform.system(),
            "input_suffix": suffix,
            "require_sidecar_on_macos": bool(require_sidecar_on_macos),
            "sidecar_found": False,
            "sidecar_path": None,
            "candidate_sidecar_paths": [],
            "run_target_path": str(resolved),
            "can_batch_simulate": True,
            "reason": "input_is_netlist",
            "suggestions": [],
            "error": None,
        }

    sidecar = _resolve_sidecar_netlist_path(resolved)
    if sidecar is not None:
        return {
            "input_path": str(resolved),
            "platform": platform.system(),
            "input_suffix": suffix,
            "require_sidecar_on_macos": bool(require_sidecar_on_macos),
            "sidecar_found": True,
            "sidecar_path": str(sidecar),
            "candidate_sidecar_paths": candidate_paths,
            "run_target_path": str(sidecar),
            "can_batch_simulate": True,
            "reason": "sidecar_available",
            "suggestions": [],
            "error": None,
        }

    if is_macos and require_sidecar_on_macos:
        error = (
            "simulateSchematicFile on macOS does not support LTspice batch simulation "
            "directly from .asc files. Create a sidecar netlist next to the schematic "
            "(.cir/.net/.sp/.spi) and retry, or use simulateNetlist/simulateNetlistFile."
        )
        return {
            "input_path": str(resolved),
            "platform": platform.system(),
            "input_suffix": suffix,
            "require_sidecar_on_macos": bool(require_sidecar_on_macos),
            "sidecar_found": False,
            "sidecar_path": None,
            "candidate_sidecar_paths": candidate_paths,
            "run_target_path": str(resolved),
            "can_batch_simulate": False,
            "reason": "missing_sidecar_required_on_macos",
            "suggestions": [
                "Create a sidecar netlist with the same basename and a .cir/.net/.sp/.spi extension.",
                "If schematic was generated from netlist/template tools, use the emitted sidecar path.",
                "Use simulateNetlist/simulateNetlistFile when you already have netlist text.",
            ],
            "error": error,
        }

    return {
        "input_path": str(resolved),
        "platform": platform.system(),
        "input_suffix": suffix,
        "require_sidecar_on_macos": bool(require_sidecar_on_macos),
        "sidecar_found": False,
        "sidecar_path": None,
        "candidate_sidecar_paths": candidate_paths,
        "run_target_path": str(resolved),
        "can_batch_simulate": True,
        "reason": "direct_asc_batch_allowed",
        "suggestions": [],
        "error": None,
    }


def _ensure_debug_fix_netlist(path: Path) -> Path:
    netlist_path = _resolve_sidecar_netlist_path(path)
    if netlist_path is None:
        raise ValueError(
            "No sidecar netlist found next to schematic; create one (e.g. .cir) for auto-debug netlist fixes."
        )
    return netlist_path


def _apply_schematic_preflight_fixes(path: Path, validation: dict[str, Any]) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    actions: list[dict[str, Any]] = []
    changed = False
    if not validation.get("has_ground", False):
        text = text.rstrip() + "\nFLAG 48 48 0\n"
        actions.append({"action": "add_ground_flag", "path": str(path), "line": "FLAG 48 48 0"})
        changed = True
    if not validation.get("simulation_directives"):
        text = text.rstrip() + "\nTEXT 48 560 Left 2 !.op\n"
        actions.append(
            {
                "action": "add_simulation_directive",
                "path": str(path),
                "line": "TEXT 48 560 Left 2 !.op",
            }
        )
        changed = True
    if changed:
        path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return actions


def _apply_floating_node_bleeders(netlist_path: Path, nodes: list[str]) -> list[str]:
    if not nodes:
        return []
    text = netlist_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    existing = {line.strip().lower() for line in lines if line.strip()}
    additions: list[str] = []
    used_indices: set[int] = set()
    for entry in existing:
        match = re.match(r"^r__bleed(\d+)\b", entry)
        if match:
            used_indices.add(int(match.group(1)))
    next_index = 1
    while next_index in used_indices:
        next_index += 1
    for node in nodes:
        line = f"R__BLEED{next_index} {node} 0 1G"
        used_indices.add(next_index)
        next_index += 1
        if line.strip().lower() in existing:
            continue
        additions.append(line)
        existing.add(line.strip().lower())
    if not additions:
        return []
    has_end = any(entry.strip().lower() == ".end" for entry in lines)
    if has_end:
        end_index = next(
            idx for idx, entry in enumerate(lines) if entry.strip().lower() == ".end"
        )
        lines[end_index:end_index] = additions
    else:
        lines.extend(additions)
        lines.append(".end")
    netlist_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return additions


def _discover_subckt_names(model_text: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for match in _SUBCKT_DEF_RE.finditer(model_text):
        name = match.group(1).strip()
        lowered = name.lower()
        if not name or lowered in seen:
            continue
        seen.add(lowered)
        names.append(name)
    return names


def _has_convergence_issue(messages: list[str]) -> bool:
    return any(_CONVERGENCE_RE.search(message or "") for message in messages)


def _apply_convergence_options(netlist_path: Path) -> list[str]:
    convergence_lines = [
        ".options reltol=0.01 abstol=1e-9 vntol=1e-6 gmin=1e-12",
        ".options method=gear",
    ]
    return _append_lines_if_missing(netlist_path, convergence_lines)


def _confidence_label(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def _compute_auto_debug_confidence(
    *,
    succeeded: bool,
    final_result: dict[str, Any] | None,
    actions_applied: list[dict[str, Any]],
    iterations_run: int,
) -> dict[str, Any]:
    score = 0.2
    factors: list[dict[str, Any]] = []

    if succeeded:
        score += 0.4
        factors.append({"factor": "run_succeeded", "delta": 0.4})
    else:
        factors.append({"factor": "run_failed", "delta": 0.0})

    if final_result is not None and not final_result.get("issues"):
        score += 0.2
        factors.append({"factor": "no_remaining_issues", "delta": 0.2})
    else:
        remaining = len(final_result.get("issues", [])) if isinstance(final_result, dict) else 1
        penalty = min(0.2, 0.05 * max(1, remaining))
        score -= penalty
        factors.append({"factor": "remaining_issues", "delta": -round(penalty, 4)})

    if actions_applied:
        delta = min(0.15, 0.03 * len(actions_applied))
        score += delta
        factors.append({"factor": "fixes_applied", "delta": round(delta, 4)})

    if iterations_run > 1:
        delta = min(0.1, 0.02 * (iterations_run - 1))
        score += delta
        factors.append({"factor": "iterative_progress", "delta": round(delta, 4)})

    diagnostics = final_result.get("diagnostics", []) if isinstance(final_result, dict) else []
    error_diags = [
        item for item in diagnostics if isinstance(item, dict) and str(item.get("severity", "")).lower() == "error"
    ]
    if error_diags:
        penalty = min(0.2, 0.03 * len(error_diags))
        score -= penalty
        factors.append({"factor": "error_diagnostics", "delta": -round(penalty, 4)})

    clamped = max(0.0, min(1.0, score))
    return {
        "score": round(clamped, 4),
        "label": _confidence_label(clamped),
        "factors": factors,
    }


_PLOT_PRESETS: dict[str, dict[str, Any]] = {
    "bode": {
        "mode": "db",
        "dual_axis": True,
        "x_log": True,
        "pane_layout": "single",
    },
    "transient_startup": {
        "mode": "real",
        "dual_axis": False,
        "x_log": False,
        "pane_layout": "single",
    },
    "noise": {
        "mode": "db",
        "dual_axis": False,
        "x_log": True,
        "pane_layout": "single",
    },
    "step_compare": {
        "mode": "real",
        "dual_axis": False,
        "x_log": False,
        "pane_layout": "per_trace",
    },
}


def _normalize_plot_preset(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in _PLOT_PRESETS:
        raise ValueError(
            "preset must be one of: " + ", ".join(sorted(_PLOT_PRESETS.keys()))
        )
    return normalized


def _default_vectors_for_plot_preset(dataset: RawDataset, preset: str) -> list[str]:
    vector_names = [variable.name for variable in dataset.variables[1:]]
    if not vector_names:
        raise ValueError("RAW dataset does not include any plottable vectors.")
    voltages = [name for name in vector_names if name.upper().startswith("V(")]
    currents = [name for name in vector_names if name.upper().startswith("I(")]
    preferred_out = [name for name in vector_names if name.lower() in {"v(out)", "v(vout)"}]
    if preset in {"bode", "noise"}:
        if preferred_out:
            return [preferred_out[0]]
        if voltages:
            return [voltages[0]]
        return [vector_names[0]]
    if preset == "step_compare":
        if voltages:
            return voltages[: min(4, len(voltages))]
        if currents:
            return currents[: min(4, len(currents))]
        return vector_names[: min(4, len(vector_names))]
    if preferred_out:
        return [preferred_out[0]]
    if voltages:
        return [voltages[0]]
    return [vector_names[0]]


def _validate_schematic_file(path: Path) -> dict[str, Any]:
    text = read_text_auto(path)
    components = 0
    wires = 0
    flags = 0
    has_ground = False
    directives: list[str] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if not parts:
            continue
        keyword = parts[0].upper()
        if keyword == "SYMBOL":
            components += 1
            continue
        if keyword == "WIRE":
            wires += 1
            continue
        if keyword == "FLAG":
            flags += 1
            if len(parts) >= 4 and parts[-1].strip().lower() in {"0", "gnd", "ground"}:
                has_ground = True
            continue
        if keyword == "TEXT" and "!" in line:
            directive = line.split("!", 1)[1].strip()
            if directive:
                directives.append(directive)

    sim_directives = [
        directive
        for directive in directives
        if directive.lower().startswith(_SIM_DIRECTIVE_PREFIXES)
    ]

    issues: list[str] = []
    warnings: list[str] = []
    suggestions: list[str] = []

    if components == 0:
        issues.append("No components were found in the schematic.")
        suggestions.append("Add at least one SYMBOL entry before simulation.")
    if not has_ground:
        issues.append("No ground flag (`FLAG ... 0`) was found in the schematic.")
        suggestions.append("Add at least one ground symbol/net label (`0`).")
    if not sim_directives:
        issues.append("No simulation directive was found in schematic TEXT commands.")
        suggestions.append("Add a directive such as `.op`, `.tran`, `.ac`, or `.dc`.")
    if wires == 0:
        warnings.append("No wire segments were found; component pins may be unconnected.")
    if path.suffix.lower() != ".asc":
        warnings.append("Path does not use `.asc` extension; LTspice usually expects schematic files as `.asc`.")

    return {
        "asc_path": str(path),
        "components": components,
        "wires": wires,
        "flags": flags,
        "has_ground": has_ground,
        "directives": directives,
        "simulation_directives": sim_directives,
        "issues": issues,
        "warnings": warnings,
        "suggestions": suggestions,
        "valid": len(issues) == 0,
    }


def _lint_schematic_file(
    path: Path,
    *,
    library: SymbolLibrary | None,
    strict: bool = False,
) -> dict[str, Any]:
    validation = _validate_schematic_file(path)
    text = read_text_auto(path)

    components: list[dict[str, Any]] = []
    current_component: dict[str, Any] | None = None
    wires: list[tuple[int, int, int, int]] = []
    flags: dict[tuple[int, int], list[str]] = {}
    errors: list[str] = list(validation.get("issues", []))
    warnings: list[str] = list(validation.get("warnings", []))

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        keyword = parts[0].upper()
        if keyword == "SYMBOL" and len(parts) >= 5:
            try:
                component = {
                    "symbol": parts[1],
                    "x": int(parts[2]),
                    "y": int(parts[3]),
                    "orientation": parts[4],
                    "reference": None,
                    "line": line_no,
                }
            except Exception:
                warnings.append(f"Could not parse SYMBOL line {line_no}: {line}")
                current_component = None
                continue
            components.append(component)
            current_component = component
            continue
        if keyword == "SYMATTR" and len(parts) >= 3 and parts[1].lower() == "instname":
            if current_component is not None:
                current_component["reference"] = " ".join(parts[2:])
            continue
        if keyword == "WIRE" and len(parts) >= 5:
            try:
                wires.append((int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])))
            except Exception:
                warnings.append(f"Could not parse WIRE line {line_no}: {line}")
            continue
        if keyword == "FLAG" and len(parts) >= 4:
            try:
                point = (int(parts[1]), int(parts[2]))
            except Exception:
                warnings.append(f"Could not parse FLAG line {line_no}: {line}")
                continue
            flags.setdefault(point, []).append(parts[3])

    refs_seen: dict[str, int] = {}
    for component in components:
        reference = str(component.get("reference") or "").strip()
        if not reference:
            errors.append(f"Component '{component['symbol']}' at line {component['line']} is missing InstName.")
            continue
        lowered = reference.lower()
        if lowered in refs_seen:
            errors.append(
                f"Duplicate InstName '{reference}' at line {component['line']} (already seen at line {refs_seen[lowered]})."
            )
        else:
            refs_seen[lowered] = int(component["line"])

    wire_endpoint_degree: dict[tuple[int, int], int] = {}
    for x1, y1, x2, y2 in wires:
        wire_endpoint_degree[(x1, y1)] = wire_endpoint_degree.get((x1, y1), 0) + 1
        wire_endpoint_degree[(x2, y2)] = wire_endpoint_degree.get((x2, y2), 0) + 1
    wire_endpoints = set(wire_endpoint_degree.keys())

    connected_pin_points: set[tuple[int, int]] = set()
    pin_lint: list[dict[str, Any]] = []
    unresolved_symbol_count = 0
    if library is not None:
        for component in components:
            symbol = str(component["symbol"])
            reference = str(component.get("reference") or f"{symbol}@line{component['line']}")
            try:
                pin_count = len(library.get(symbol).pins)
            except Exception:
                unresolved_symbol_count += 1
                warnings.append(
                    f"Could not resolve symbol '{symbol}' for component '{reference}'; pin-level lint skipped."
                )
                continue
            connected = 0
            unconnected_pins: list[int] = []
            for pin_order in range(1, pin_count + 1):
                try:
                    dx, dy = library.pin_offset(symbol, str(component["orientation"]), pin_order)
                except Exception:
                    continue
                point = (int(component["x"]) + dx, int(component["y"]) + dy)
                if point in wire_endpoints or point in flags:
                    connected += 1
                    connected_pin_points.add(point)
                else:
                    unconnected_pins.append(pin_order)
            if connected == 0:
                errors.append(f"Component '{reference}' has no connected pins.")
            elif unconnected_pins:
                message = f"Component '{reference}' has unconnected pins: {unconnected_pins}."
                if strict:
                    errors.append(message)
                else:
                    warnings.append(message)
            pin_lint.append(
                {
                    "reference": reference,
                    "symbol": symbol,
                    "pin_count": pin_count,
                    "connected_pin_count": connected,
                    "unconnected_pins": unconnected_pins,
                }
            )

    dangling_wire_endpoints: list[tuple[int, int]] = []
    for point, degree in wire_endpoint_degree.items():
        if point in connected_pin_points or point in flags:
            continue
        if degree <= 1:
            dangling_wire_endpoints.append(point)
    if dangling_wire_endpoints:
        message = (
            f"Dangling wire endpoints detected: {dangling_wire_endpoints[:20]}"
            + (" ..." if len(dangling_wire_endpoints) > 20 else "")
        )
        if strict:
            errors.append(message)
        else:
            warnings.append(message)

    for point, names in flags.items():
        unique_names = sorted({name.strip() for name in names if name.strip()})
        if len(unique_names) > 1:
            warnings.append(
                f"Multiple net labels at point {point}: {unique_names}. Verify this is intentional."
            )

    return {
        "asc_path": str(path),
        "valid": len(errors) == 0,
        "strict": bool(strict),
        "errors": errors,
        "warnings": warnings,
        "component_count": len(components),
        "wire_count": len(wires),
        "flag_count": sum(len(values) for values in flags.values()),
        "unresolved_symbol_count": unresolved_symbol_count,
        "dangling_wire_endpoint_count": len(dangling_wire_endpoints),
        "pin_lint": pin_lint,
        "validation": validation,
    }


_load_run_state()
_load_job_state()
_load_job_history_state()
if any(
    str(_jobs[job_id].get("status", "")).lower() == "queued" and not _jobs[job_id].get("cancel_requested")
    for job_id in _job_order
    if job_id in _jobs
):
    _ensure_job_worker()


@mcp.resource(
    "docs://agent-readme",
    name="agent_readme",
    title="LTspice MCP Agent Playbook",
    description="Agent-focused operational guide for this MCP server.",
    mime_type="text/markdown",
)
def agent_readme_resource() -> str:
    return _read_agent_readme_text()


@mcp.tool()
def readAgentGuide(
    section: str | None = None,
    query: str | None = None,
    include_headings: bool = True,
    max_chars: int = 20000,
) -> dict[str, Any]:
    """
    Read AGENT_README.md through MCP for interactive agent guidance.

    - section: heading index (1-based) or heading title (exact/contains match)
    - query: optional case-insensitive text search
    """
    safe_max_chars = max(500, min(120000, int(max_chars)))
    text = _read_agent_readme_text()
    headings = _parse_markdown_headings(text)
    content = text
    selected_heading: dict[str, Any] | None = None
    warning: str | None = None

    if section:
        selected_heading = _resolve_heading_query(headings, section)
        if selected_heading is None:
            warning = f"Section '{section}' not found. Returning full guide."
        else:
            content = _extract_heading_section(text, headings, selected_heading)

    search_matches: list[dict[str, Any]] = []
    if query:
        search_matches = _search_text_lines(content, query, limit=20)
        if not search_matches and selected_heading is not None:
            warning = (
                warning
                or f"No matches for query '{query}' in section '{selected_heading.get('title')}'."
            )
        elif not search_matches:
            warning = warning or f"No matches for query '{query}' in guide."

    truncated = len(content) > safe_max_chars
    trimmed_content = content[:safe_max_chars]
    return {
        "path": str(_AGENT_README_PATH),
        "section_requested": section,
        "section_resolved": selected_heading,
        "query": query,
        "match_count": len(search_matches),
        "matches": search_matches,
        "include_headings": bool(include_headings),
        "headings": headings if include_headings else None,
        "max_chars": safe_max_chars,
        "content_chars": len(content),
        "truncated": truncated,
        "content": trimmed_content,
        "warning": warning,
    }


@mcp.tool()
def parseMeasResults(
    run_id: str | None = None,
    log_path: str | None = None,
) -> dict[str, Any]:
    """Parse LTspice .meas results from a run log or explicit log path."""
    if log_path:
        target = Path(log_path).expanduser().resolve()
        if not target.exists():
            raise FileNotFoundError(f"Log file not found: {target}")
        text = read_text_auto(target)
        parsed = _parse_meas_results_from_text(text)
        return {
            "run_id": run_id,
            "log_path": str(target),
            **parsed,
        }

    run = _resolve_run(run_id)
    log_target = run.log_utf8_path or run.log_path
    if log_target is None or not log_target.exists():
        return {
            "run_id": run.run_id,
            "log_path": None,
            "count": 0,
            "measurements": {},
            "items": [],
            "warning": "No log file available for the selected run.",
        }
    text = read_text_auto(log_target)
    parsed = _parse_meas_results_from_text(text)
    return {
        "run_id": run.run_id,
        "log_path": str(log_target),
        **parsed,
    }


@mcp.tool()
def runMeasAutomation(
    measurements: list[dict[str, Any] | str],
    netlist_path: str | None = None,
    netlist_content: str | None = None,
    circuit_name: str = "meas_automation",
    asc_path: str | None = None,
    ascii_raw: bool = False,
    timeout_seconds: int | None = None,
    show_ui: bool | None = None,
    open_raw_after_run: bool = False,
) -> dict[str, Any]:
    """Inject .meas directives into a netlist, run simulation, and parse measurement values."""
    statements = _build_meas_statements(measurements)
    source_netlist: Path
    if netlist_content is not None:
        source_netlist = _runner.write_netlist(netlist_content, circuit_name=circuit_name)
    elif netlist_path is not None:
        source_netlist = Path(netlist_path).expanduser().resolve()
        if not source_netlist.exists():
            raise FileNotFoundError(f"Netlist file not found: {source_netlist}")
    elif asc_path is not None:
        asc_resolved = Path(asc_path).expanduser().resolve()
        if not asc_resolved.exists():
            raise FileNotFoundError(f"Schematic file not found: {asc_resolved}")
        target = _resolve_schematic_simulation_target(asc_resolved, require_sidecar_on_macos=True)
        if not bool(target.get("can_batch_simulate")):
            raise ValueError(str(target.get("error")))
        source_netlist = Path(str(target["run_target_path"])).expanduser().resolve()
    else:
        raise ValueError("Provide one of netlist_content, netlist_path, or asc_path.")

    meas_netlist = _write_meas_netlist(
        base_netlist_path=source_netlist,
        statements=statements,
        suffix="meas",
    )
    run_payload = simulateNetlistFile(
        netlist_path=str(meas_netlist),
        ascii_raw=ascii_raw,
        timeout_seconds=timeout_seconds,
        show_ui=show_ui,
        open_raw_after_run=open_raw_after_run,
    )
    meas_payload = parseMeasResults(run_id=str(run_payload["run_id"]))
    return {
        "source_netlist_path": str(source_netlist),
        "meas_netlist_path": str(meas_netlist),
        "meas_statements": statements,
        "run": run_payload,
        "measurements": meas_payload,
    }


@mcp.tool()
def runVerificationPlan(
    assertions: list[dict[str, Any]],
    run_id: str | None = None,
    netlist_path: str | None = None,
    netlist_content: str | None = None,
    asc_path: str | None = None,
    circuit_name: str | None = None,
    measurements: list[dict[str, Any] | str] | None = None,
    ascii_raw: bool = False,
    timeout_seconds: int | None = None,
    show_ui: bool | None = None,
    open_raw_after_run: bool = False,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """
    Run simulation (or reuse a run) and evaluate assertion checks in one call.

    Assertion types:
    - `vector_stat`: vector + statistic(min|max|avg|rms|pp|final|abs_max)
    - `bandwidth`: vector (+ optional drop_db/reference/metric)
    - `gain_phase_margin`: vector (+ optional metric)
    - `rise_fall_time`: vector (+ optional metric)
    - `settling_time`: vector (+ optional tolerance_percent/target_value)
    - `meas`: name from .meas results
    - `all_of`: all nested assertions must pass
    - `any_of`: at least one nested assertion must pass
    Bounds:
    - `min`, `max`
    Tolerances:
    - `target` (+ optional `rel_tol_pct`, `abs_tol`)
    """
    if not assertions:
        raise ValueError("assertions must contain at least one assertion object")

    context = _resolve_run_target_for_input(
        run_id=run_id,
        netlist_path=netlist_path,
        netlist_content=netlist_content,
        circuit_name=circuit_name,
        asc_path=asc_path,
        ascii_raw=ascii_raw,
        timeout_seconds=timeout_seconds,
        show_ui=show_ui,
        open_raw_after_run=open_raw_after_run,
    )
    run: SimulationRun = context["run"]
    run_payload: dict[str, Any] = context["run_payload"]

    measurement_map: dict[str, float] = {}
    measurement_report: dict[str, Any] | None = None
    if measurements:
        meas_source_netlist = netlist_path
        meas_source_content = netlist_content
        meas_source_asc = asc_path
        if run_id and not any([netlist_path, netlist_content, asc_path]):
            measurement_report = parseMeasResults(run_id=run.run_id)
        else:
            measurement_report = runMeasAutomation(
                measurements=measurements,
                netlist_path=meas_source_netlist,
                netlist_content=meas_source_content,
                asc_path=meas_source_asc,
                circuit_name=circuit_name or "verification_meas",
                ascii_raw=ascii_raw,
                timeout_seconds=timeout_seconds,
                show_ui=show_ui,
                open_raw_after_run=open_raw_after_run,
            )
            run_payload = measurement_report["run"]
            run = _resolve_run(str(run_payload["run_id"]))
            measurement_report = measurement_report["measurements"]
        measurement_map = {
            str(key): float(value)
            for key, value in (measurement_report.get("measurements", {}) if measurement_report else {}).items()
        }
    else:
        parsed_meas = parseMeasResults(run_id=run.run_id)
        measurement_map = {
            str(key): float(value) for key, value in parsed_meas.get("measurements", {}).items()
        }
        measurement_report = parsed_meas

    requires_dataset = any(_assertion_requires_dataset(item) for item in assertions if isinstance(item, dict))
    dataset = _resolve_dataset(plot=None, run_id=run.run_id, raw_path=None) if requires_dataset else None
    checks: list[dict[str, Any]] = []

    for index, assertion in enumerate(assertions, start=1):
        if not isinstance(assertion, dict):
            checks.append(
                {
                    "id": f"check_{index}",
                    "type": "invalid",
                    "passed": False,
                    "value": None,
                    "error": "Assertion entries must be objects.",
                    "details": {"assertion": assertion},
                }
            )
            if fail_fast:
                break
            continue
        try:
            result = _evaluate_verification_assertion(
                assertion=assertion,
                dataset=dataset,
                measurement_map=measurement_map,
                default_id=f"check_{index}",
                fail_fast=fail_fast,
            )
            checks.append(result)
            if fail_fast and not bool(result.get("passed")):
                break
        except Exception as exc:  # noqa: BLE001
            checks.append(
                {
                    "id": str(assertion.get("id") or assertion.get("name") or f"check_{index}"),
                    "type": str(assertion.get("type", "vector_stat")).strip().lower(),
                    "passed": False,
                    "value": None,
                    "min": assertion.get("min"),
                    "max": assertion.get("max"),
                    "error": str(exc),
                    "details": {"assertion": assertion},
                }
            )
            if fail_fast:
                break

    passed_count = sum(1 for item in checks if item.get("passed"))
    failed_count = len(checks) - passed_count
    overall_passed = failed_count == 0 and bool(run_payload.get("succeeded", False))
    return {
        "overall_passed": overall_passed,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "run_source": context["source"],
        "run": run_payload,
        "checks": checks,
        "measurements": measurement_report,
    }


@mcp.tool()
def runSweepStudy(
    parameter: str,
    mode: Literal["step", "monte_carlo"] = "step",
    netlist_path: str | None = None,
    netlist_content: str | None = None,
    circuit_name: str = "sweep_study",
    values: list[float] | None = None,
    start: float | None = None,
    stop: float | None = None,
    step: float | None = None,
    samples: int = 8,
    nominal: float | None = None,
    sigma_pct: float = 5.0,
    distribution: Literal["gaussian", "uniform"] = "gaussian",
    metric_vector: str = "V(out)",
    metric_statistic: Literal["min", "max", "avg", "rms", "pp", "final", "abs_max"] = "final",
    timeout_seconds: int | None = None,
    ascii_raw: bool = False,
) -> dict[str, Any]:
    """Run stepped or Monte-Carlo parameter studies and return aggregate metrics."""
    param_name = parameter.strip()
    if not param_name:
        raise ValueError("parameter must not be empty")
    base_path: Path
    if netlist_content is not None:
        base_path = _runner.write_netlist(netlist_content, circuit_name=circuit_name)
    elif netlist_path is not None:
        base_path = Path(netlist_path).expanduser().resolve()
        if not base_path.exists():
            raise FileNotFoundError(f"Netlist file not found: {base_path}")
    else:
        raise ValueError("Provide netlist_content or netlist_path")

    mode_norm = mode.strip().lower()
    records: list[dict[str, Any]] = []

    if mode_norm == "step":
        sweep_values = values[:] if values else []
        if not sweep_values:
            if start is None or stop is None or step is None or step == 0:
                raise ValueError("For mode='step', provide values or (start, stop, step).")
            current = float(start)
            direction = 1.0 if step > 0 else -1.0
            while (direction > 0 and current <= float(stop)) or (direction < 0 and current >= float(stop)):
                sweep_values.append(current)
                current += float(step)
        step_line = ".step param " + param_name + " list " + " ".join(f"{float(value):.12g}" for value in sweep_values)
        text = _read_netlist_text(base_path)
        stepped_text = _append_unique_lines_before_end(text, [step_line])
        stepped_path = base_path.with_name(f"{base_path.stem}_step_study{base_path.suffix}")
        stepped_path.write_text(stepped_text, encoding="utf-8")

        run_payload = simulateNetlistFile(
            netlist_path=str(stepped_path),
            ascii_raw=ascii_raw,
            timeout_seconds=timeout_seconds,
            show_ui=False,
            open_raw_after_run=False,
        )
        run = _resolve_run(str(run_payload["run_id"]))
        dataset = _resolve_dataset(plot=None, run_id=run.run_id, raw_path=None)
        step_indices = list(range(dataset.step_count))
        metric_values: list[float] = []
        for idx in step_indices:
            selected = _resolve_step_index(dataset, idx)
            series = dataset.get_vector(metric_vector, step_index=selected)
            value = _metric_from_series(series, metric_statistic)
            metric_values.append(value)
            label = dataset.steps[idx].label if dataset.steps and idx < len(dataset.steps) else None
            records.append(
                {
                    "index": idx,
                    "parameter_value": sweep_values[idx] if idx < len(sweep_values) else None,
                    "step_label": label,
                    "metric_value": value,
                    "run_id": run.run_id,
                }
            )
    else:
        if nominal is None:
            raise ValueError("For mode='monte_carlo', nominal is required.")
        sample_count = max(1, min(500, int(samples)))
        spread = abs(float(nominal)) * (abs(float(sigma_pct)) / 100.0)
        base_text = _read_netlist_text(base_path)
        metric_values: list[float] = []
        for idx in range(sample_count):
            if distribution == "uniform":
                delta = random.uniform(-spread, spread)
            else:
                delta = random.gauss(0.0, spread)
            sample_value = float(nominal) + delta
            line = f".param {param_name}={sample_value:.12g}"
            candidate_text = _append_unique_lines_before_end(base_text, [line])
            candidate_path = base_path.with_name(f"{base_path.stem}_mc_{idx + 1}{base_path.suffix}")
            candidate_path.write_text(candidate_text, encoding="utf-8")
            run_payload = simulateNetlistFile(
                netlist_path=str(candidate_path),
                ascii_raw=ascii_raw,
                timeout_seconds=timeout_seconds,
                show_ui=False,
                open_raw_after_run=False,
            )
            run = _resolve_run(str(run_payload["run_id"]))
            record: dict[str, Any] = {
                "index": idx,
                "parameter_value": sample_value,
                "run_id": run.run_id,
                "succeeded": bool(run_payload.get("succeeded", False)),
            }
            if run_payload.get("succeeded", False):
                dataset = _resolve_dataset(plot=None, run_id=run.run_id, raw_path=None)
                selected = _resolve_step_index(dataset, 0)
                series = dataset.get_vector(metric_vector, step_index=selected)
                value = _metric_from_series(series, metric_statistic)
                metric_values.append(value)
                record["metric_value"] = value
            else:
                record["metric_value"] = None
                record["issues"] = run_payload.get("issues", [])
            records.append(record)

    numeric_values = [float(item["metric_value"]) for item in records if item.get("metric_value") is not None]
    aggregate = _aggregate_numeric(numeric_values)
    worst_case = None
    if records:
        if metric_statistic in {"max", "abs_max", "pp"}:
            worst_case = max(records, key=lambda item: float(item.get("metric_value") or float("-inf")))
        elif metric_statistic == "min":
            worst_case = min(records, key=lambda item: float(item.get("metric_value") or float("inf")))
        else:
            center = aggregate["mean"] if aggregate.get("mean") is not None else 0.0
            worst_case = max(
                records,
                key=lambda item: abs(float(item.get("metric_value") or center) - float(center)),
            )

    return {
        "mode": mode_norm,
        "parameter": param_name,
        "metric_vector": metric_vector,
        "metric_statistic": metric_statistic,
        "record_count": len(records),
        "records": records,
        "aggregate": aggregate,
        "worst_case": worst_case,
    }


@mcp.tool()
def generateVerifyAndCleanCircuit(
    intent: str,
    parameters: dict[str, Any] | None = None,
    assertions: list[dict[str, Any]] | None = None,
    measurements: list[dict[str, Any] | str] | None = None,
    clean_in_place: bool = True,
    clean_output_path: str | None = None,
    strict_lint: bool = False,
    strict_style_lint: bool = True,
    min_style_score: float = 96.0,
    auto_clean: bool = True,
    grid: int = 16,
    render_after_clean: bool = True,
    downscale_factor: float = 1.0,
    show_ui: bool | None = None,
    fail_fast_verification: bool = False,
) -> dict[str, Any]:
    """
    One-shot orchestration: create intent circuit, lint, simulate, verify, clean, and inspect.
    """
    safe_params = parameters or {}
    created = createIntentCircuit(
        intent=intent,
        parameters=safe_params,
        validate_schematic=True,
        open_ui=False,
    )
    asc_path = str(created["asc_path"])
    lint_before = lintSchematic(
        asc_path=asc_path,
        strict=strict_lint,
        strict_style=strict_style_lint,
        min_style_score=min_style_score,
    )
    simulation = simulateSchematicFile(
        asc_path=asc_path,
        show_ui=show_ui,
        open_raw_after_run=False,
        validate_first=True,
        abort_on_validation_error=False,
    )
    verification_assertions = assertions or [
        {
            "id": "default_input_sanity",
            "type": "vector_stat",
            "vector": "V(in)",
            "statistic": "abs_max",
            "min": 0.0,
        }
    ]
    verification = runVerificationPlan(
        assertions=verification_assertions,
        run_id=str(simulation["run_id"]),
        measurements=measurements,
        fail_fast=fail_fast_verification,
    )

    target_path = asc_path
    clean_result: dict[str, Any] | None = None
    if auto_clean:
        if not clean_in_place:
            if clean_output_path:
                target_path = str(Path(clean_output_path).expanduser().resolve())
            else:
                source = Path(asc_path).expanduser().resolve()
                target_path = str(source.with_name(f"{source.stem}_clean{source.suffix}"))
        clean_result = autoCleanSchematicLayout(
            asc_path=asc_path,
            output_path=target_path,
            prefer_sidecar_regeneration=False,
            grid=grid,
            render_after=render_after_clean,
            downscale_factor=downscale_factor,
        )
        target_path = str(clean_result["target_path"])

    lint_after = lintSchematic(
        asc_path=target_path,
        strict=strict_lint,
        strict_style=strict_style_lint,
        min_style_score=min_style_score,
    )
    inspection = inspectSchematicVisualQuality(
        asc_path=target_path,
        render=render_after_clean,
        downscale_factor=downscale_factor,
    )
    overall_passed = bool(
        simulation.get("succeeded", False)
        and verification.get("overall_passed", False)
        and lint_after.get("valid", False)
    )
    return {
        "overall_passed": overall_passed,
        "intent": intent,
        "parameters": safe_params,
        "created": created,
        "lint_before": lint_before,
        "simulation": simulation,
        "verification": verification,
        "clean": clean_result,
        "lint_after": lint_after,
        "inspection": inspection,
        "final_schematic_path": target_path,
    }


@mcp.tool()
def autoCleanSchematicLayout(
    asc_path: str,
    output_path: str | None = None,
    prefer_sidecar_regeneration: bool = True,
    grid: int = 16,
    render_after: bool = True,
    settle_seconds: float = 1.0,
    downscale_factor: float = 1.0,
) -> dict[str, Any]:
    """Auto-clean schematic layout and return before/after quality analysis."""
    source = Path(asc_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Schematic file not found: {source}")
    target = Path(output_path).expanduser().resolve() if output_path else source
    target.parent.mkdir(parents=True, exist_ok=True)

    before = _analyze_schematic_visual_quality(source)
    action_details: dict[str, Any]
    sidecar = _resolve_sidecar_netlist_path(source)
    if prefer_sidecar_regeneration and sidecar is not None and sidecar.exists():
        result = build_schematic_from_netlist(
            workdir=_runner.workdir,
            netlist_content=_read_netlist_text(sidecar),
            circuit_name=target.stem,
            output_path=str(target),
            sheet_width=1200,
            sheet_height=900,
            placement_mode="smart",
        )
        action_details = {
            "action": "regenerated_from_sidecar",
            "sidecar_netlist_path": str(sidecar),
            "generation": result,
        }
    else:
        action_details = {
            "action": "normalized_grid",
            "normalization": _normalize_schematic_grid(
                source=source,
                output=target,
                grid=max(2, int(grid)),
            ),
        }

    after = _analyze_schematic_visual_quality(target)
    render_payload: dict[str, Any] | None = None
    if render_after:
        render_result = renderLtspiceSchematicImage(
            asc_path=str(target),
            settle_seconds=settle_seconds,
            downscale_factor=downscale_factor,
        )
        render_payload = dict(render_result.structuredContent or {})
    return {
        "source_path": str(source),
        "target_path": str(target),
        "before": before,
        "after": after,
        "score_delta": round(float(after["score"]) - float(before["score"]), 3),
        "action_details": action_details,
        "render": render_payload,
    }


@mcp.tool()
def inspectSchematicVisualQuality(
    asc_path: str,
    render: bool = True,
    settle_seconds: float = 1.0,
    downscale_factor: float = 1.0,
) -> dict[str, Any]:
    """Inspect schematic visual quality and suggest coordinate-level fixes."""
    path = Path(asc_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Schematic file not found: {path}")
    quality = _analyze_schematic_visual_quality(path)
    rendered: dict[str, Any] | None = None
    if render:
        render_result = renderLtspiceSchematicImage(
            asc_path=str(path),
            settle_seconds=settle_seconds,
            downscale_factor=downscale_factor,
        )
        rendered = dict(render_result.structuredContent or {})
    return {
        "quality": quality,
        "render": rendered,
    }


@mcp.tool()
def daemonDoctor(
    include_recent_warnings: bool = True,
    deep_checks: bool = False,
) -> dict[str, Any]:
    """Run a daemon/system health check and return actionable recommendations."""
    status = getLtspiceStatus()
    ui_status = getLtspiceUiStatus()
    recent_errors = getRecentErrors(
        limit=60,
        log_count=4,
        include_warnings=include_recent_warnings,
    )
    capture_health = getCaptureHealth(limit=400, include_recent_events=True)

    sck_helper_path = os.getenv(
        "LTSPICE_MCP_SCK_HELPER_PATH",
        str(Path.home() / "Library/Application Support/ltspice-mcp/bin/ltspice-sck-helper"),
    )
    ax_helper_path = str(Path.home() / "Library/Application Support/ltspice-mcp/bin/ltspice-ax-close-helper")
    helper_checks = {
        "sck_helper_path": str(Path(sck_helper_path).expanduser().resolve()),
        "sck_helper_exists": Path(sck_helper_path).expanduser().resolve().exists(),
        "ax_helper_path": str(Path(ax_helper_path).expanduser().resolve()),
        "ax_helper_exists": Path(ax_helper_path).expanduser().resolve().exists(),
    }

    issues: list[str] = []
    recommendations: list[str] = []

    if not status.get("ltspice_executable"):
        issues.append("LTspice executable is not configured or auto-discovery failed.")
        recommendations.append("Set LTSPICE_BINARY or pass --ltspice-binary when starting the daemon.")
    if recent_errors.get("entry_count", 0) > 0:
        issues.append(f"Recent daemon logs contain {recent_errors['entry_count']} warning/error entries.")
        recommendations.append("Review getRecentErrors output and fix recurring failures first.")
    success_rate = capture_health.get("success_rate")
    if success_rate is not None and float(success_rate) < 0.8:
        issues.append(f"Capture success rate is low ({success_rate}).")
        recommendations.append("Re-run permission triggers and inspect close_event/capture diagnostics.")
    if not helper_checks["sck_helper_exists"]:
        issues.append("ScreenCaptureKit helper binary is missing.")
        recommendations.append("Run a render tool once to trigger helper compilation and permission prompts.")
    if not helper_checks["ax_helper_exists"]:
        issues.append("Accessibility close helper binary is missing.")
        recommendations.append("Call closeLtspiceWindow once and grant Accessibility permissions if prompted.")

    deep_payload: dict[str, Any] | None = None
    if deep_checks:
        deep_payload = {
            "library_status": getLtspiceLibraryStatus(),
            "latest_log_tail": tailDaemonLog(lines=120),
        }

    health = "ok"
    if issues:
        health = "warning"
    if not status.get("ltspice_executable"):
        health = "fail"

    return {
        "health": health,
        "status": status,
        "ui_status": ui_status,
        "helper_checks": helper_checks,
        "capture_health": capture_health,
        "recent_errors": recent_errors,
        "issues": issues,
        "recommendations": recommendations,
        "deep_checks": deep_payload,
    }


@mcp.tool()
def queueSimulationJob(
    netlist_path: str | None = None,
    netlist_content: str | None = None,
    circuit_name: str = "queued_job",
    ascii_raw: bool = False,
    timeout_seconds: int | None = None,
    priority: int = 50,
    max_retries: int = 0,
) -> dict[str, Any]:
    """Queue a simulation job and return a job id for status polling/cancelation."""
    global _loaded_netlist
    if netlist_content is None and netlist_path is None:
        raise ValueError("Provide either netlist_content or netlist_path.")
    source: Path
    if netlist_content is not None:
        source = _runner.write_netlist(netlist_content, circuit_name=circuit_name)
    else:
        source = Path(str(netlist_path)).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Netlist file not found: {source}")
    _loaded_netlist = source
    return _queue_simulation_job(
        source_path=source,
        ascii_raw=ascii_raw,
        timeout_seconds=timeout_seconds,
        kind="netlist",
        priority=max(0, min(1000, int(priority))),
        max_retries=max(0, int(max_retries)),
    )


@mcp.tool()
def listJobs(
    limit: int = 50,
    status: str | None = None,
    order_by: Literal["created_at", "priority"] = "created_at",
    include_history: bool = False,
) -> dict[str, Any]:
    """List queued/running/completed simulation jobs."""
    safe_limit = max(1, min(500, int(limit)))
    status_filter = status.strip().lower() if status else None
    with _job_lock:
        pool = [job for job_id in _job_order if (job := _jobs.get(job_id)) is not None]
        if order_by == "priority":
            selected_jobs = sorted(
                pool,
                key=lambda job: (
                    int(job.get("priority", 50)),
                    int(job.get("queue_seq", 1_000_000)),
                    str(job.get("created_at", "")),
                ),
            )
        else:
            selected_jobs = list(reversed(pool))
        selected = []
        for job in selected_jobs:
            if status_filter and str(job.get("status", "")).lower() != status_filter:
                continue
            selected.append(_job_public_payload(job))
            if len(selected) >= safe_limit:
                break
        if include_history and len(selected) < safe_limit:
            for item in reversed(_job_history):
                job_id = str(item.get("job_id", "")).strip()
                if not job_id:
                    continue
                if any(existing.get("job_id") == job_id for existing in selected):
                    continue
                if status_filter and str(item.get("status", "")).lower() != status_filter:
                    continue
                enriched = dict(item)
                selected.append(_job_public_payload(enriched))
                if len(selected) >= safe_limit:
                    break
    return {
        "limit": safe_limit,
        "status_filter": status_filter,
        "order_by": order_by,
        "include_history": bool(include_history),
        "count": len(selected),
        "jobs": selected,
    }


@mcp.tool()
def jobStatus(job_id: str, include_run: bool = True) -> dict[str, Any]:
    """Get status for one queued/running/completed simulation job."""
    with _job_lock:
        job = _jobs.get(job_id)
        if job is None:
            history_index = _job_history_index.get(job_id)
            if history_index is None:
                raise ValueError(f"Unknown job_id '{job_id}'")
            history_job = _job_history[history_index]
            payload = _job_public_payload(history_job)
            payload["from_history"] = True
        else:
            payload = _job_public_payload(job)
    if include_run and payload.get("run_id"):
        run = _resolve_run(str(payload["run_id"]))
        payload["run"] = _run_payload(run, include_output=False, log_tail_lines=120)
    return payload


@mcp.tool()
def cancelJob(job_id: str, force: bool = False) -> dict[str, Any]:
    """Request cancellation for a queued/running job."""
    with _job_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise ValueError(f"Unknown job_id '{job_id}'")
        if job.get("status") in {"succeeded", "failed", "canceled"}:
            return _job_public_payload(job)
        job["cancel_requested"] = True
        if job.get("status") == "queued":
            job["status"] = "canceled"
            job["summary"] = "Canceled before start."
            job["finished_at"] = _now_iso()
            _archive_job_locked(job)
        elif force:
            job["summary"] = "Force-cancel requested. Worker will terminate process on next poll."
        _save_job_state()
        _save_job_history_state()
    return jobStatus(job_id=job_id, include_run=False)


@mcp.tool()
def listJobHistory(
    limit: int = 100,
    status: str | None = None,
) -> dict[str, Any]:
    """List archived (terminal) queue jobs retained across daemon restarts."""
    safe_limit = max(1, min(2000, int(limit)))
    status_filter = status.strip().lower() if status else None
    with _job_lock:
        rows: list[dict[str, Any]] = []
        for item in reversed(_job_history):
            if status_filter and str(item.get("status", "")).lower() != status_filter:
                continue
            payload = _job_public_payload(item)
            payload["from_history"] = True
            payload["archived_at"] = item.get("archived_at")
            rows.append(payload)
            if len(rows) >= safe_limit:
                break
    return {
        "limit": safe_limit,
        "status_filter": status_filter,
        "retention": int(_job_history_retention),
        "count": len(rows),
        "jobs": rows,
    }


@mcp.tool()
def getLtspiceStatus() -> dict[str, Any]:
    """Get LTspice executable status and server configuration."""
    executable = _runner.executable or find_ltspice_executable()
    version = get_ltspice_version(executable) if executable else None
    return {
        "mcp_server_version": LTSPICE_MCP_VERSION,
        "ltspice_executable": str(executable) if executable else None,
        "ltspice_version": version,
        "workdir": str(_runner.workdir),
        "default_timeout_seconds": _runner.default_timeout_seconds,
        "runs_recorded": len(_run_order),
        "run_state_path": str(_state_path),
        "jobs_recorded": len(_job_order),
        "job_state_path": str(_job_state_path),
        "job_history_count": len(_job_history),
        "job_history_retention": int(_job_history_retention),
        "job_history_path": str(_job_history_path),
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
    settle_seconds: float = 1.0,
    include_pins: bool = True,
    include_pin_labels: bool = True,
    lib_zip_path: str | None = None,
) -> CallToolResult:
    """
    Render an LTspice symbol to an image and return the image through MCP.

    The response includes both image content and structured metadata (image_path, bounds, etc.).
    """
    if not symbol.strip():
        raise ValueError("symbol must be a non-empty string")

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
    payload: dict[str, Any] = {
        **screenshot_payload,
        "symbol": symbol,
        "preview_asc_path": preview["asc_path"],
        "backend_used": "ltspice",
    }
    ignored = []
    if not include_pins:
        ignored.append("include_pins")
    if not include_pin_labels:
        ignored.append("include_pin_labels")
    if lib_zip_path:
        ignored.append("lib_zip_path")
    if ignored:
        payload["warnings"] = [
            "The following parameters are ignored for LTspice-native rendering: "
            + ", ".join(ignored)
        ]
    return _image_tool_result(payload)


@mcp.tool()
def renderLtspiceSchematicImage(
    asc_path: str,
    output_path: str | None = None,
    width: int = 1400,
    height: int = 900,
    downscale_factor: float = 1.0,
    settle_seconds: float = 1.0,
    include_symbol_graphics: bool = True,
    lib_zip_path: str | None = None,
    render_session_id: str | None = None,
) -> CallToolResult:
    """
    Render an LTspice schematic (.asc) to an image and return it through MCP.

    `downscale_factor` lets clients request smaller rendered images.
    """
    asc_resolved = Path(asc_path).expanduser().resolve()
    open_path = asc_resolved
    title_hint = asc_resolved.name
    close_after_capture = True
    if render_session_id:
        session = _render_sessions.get(render_session_id)
        if not session:
            raise ValueError(f"Unknown render_session_id '{render_session_id}'")
        if session.get("path") and Path(session["path"]) != asc_resolved:
            raise ValueError("render_session_id is bound to a different path")
        open_path = None
        title_hint = str(session.get("title_hint") or asc_resolved.name)
        close_after_capture = False
    screenshot_payload = capture_ltspice_window_screenshot(
        output_path=_resolve_png_output_path(
            kind="schematics",
            name=asc_resolved.stem,
            output_path=output_path,
        ),
        open_path=open_path,
        title_hint=title_hint,
        settle_seconds=settle_seconds,
        downscale_factor=downscale_factor,
        avoid_space_switch=True,
        prefer_screencapturekit=True,
        close_after_capture=close_after_capture,
    )
    payload: dict[str, Any] = {
        **screenshot_payload,
        "asc_path": str(asc_resolved),
        "backend_used": "ltspice",
        "render_session_id": render_session_id,
    }
    ignored = []
    if width != 1400:
        ignored.append("width")
    if height != 900:
        ignored.append("height")
    if not include_symbol_graphics:
        ignored.append("include_symbol_graphics")
    if lib_zip_path:
        ignored.append("lib_zip_path")
    if ignored:
        payload["warnings"] = [
            "The following parameters are ignored for LTspice-native rendering: "
            + ", ".join(ignored)
        ]
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
    settle_seconds: float = 1.0,
    max_points: int = 2000,
    y_mode: Literal["magnitude", "phase", "real", "imag", "db"] = "magnitude",
    mode: Literal["auto", "db", "phase", "real", "imag"] = "auto",
    x_log: bool | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    dual_axis: bool | None = None,
    pane_layout: Literal["single", "split", "per_trace"] = "single",
    title: str | None = None,
    validate_capture: bool = True,
    render_session_id: str | None = None,
) -> CallToolResult:
    """
    Render one or more vectors from a RAW dataset to a plot image and return it through MCP.

    Supports run_id/raw_path resolution and optional step filtering for stepped runs.
    pane_layout: single | split | per_trace
    """
    if not vectors:
        raise ValueError("vectors must contain at least one vector name")
    dataset = _resolve_dataset(plot=plot, run_id=run_id, raw_path=raw_path)
    selected_step = _resolve_step_index(dataset, step_index)
    normalized_mode = _normalize_plot_mode(mode)
    normalized_y_mode = _normalize_plot_y_mode(y_mode)
    normalized_pane_layout = _normalize_pane_layout(pane_layout)
    warnings: list[str] = []

    effective_mode = normalized_mode
    if normalized_mode == "auto":
        if normalized_y_mode == "phase":
            effective_mode = "phase"
        elif normalized_y_mode == "real":
            effective_mode = "real"
        elif normalized_y_mode == "imag":
            effective_mode = "imag"
        elif normalized_y_mode in {"magnitude", "db"}:
            effective_mode = "db" if _infer_plot_type(dataset) == "ac" else "real"

    render_dataset, step_rendering = _materialize_plot_step_dataset(
        dataset=dataset,
        selected_step=selected_step,
    )
    plt_payload = _write_ltspice_plot_settings_file(
        dataset=render_dataset,
        vectors=vectors,
        mode=effective_mode,
        pane_layout=normalized_pane_layout,
        dual_axis=dual_axis,
        x_log=x_log,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )
    open_path = render_dataset.path
    title_hint = render_dataset.path.name
    close_after_capture = True
    if render_session_id:
        session = _render_sessions.get(render_session_id)
        if not session:
            raise ValueError(f"Unknown render_session_id '{render_session_id}'")
        session_path = Path(str(session.get("path") or "")).expanduser().resolve()
        if session.get("path") and session_path != render_dataset.path:
            raise ValueError(
                "render_session_id is bound to a different path. "
                "Start a render session for the selected step/raw file first."
            )
        title_hint = str(session.get("title_hint") or render_dataset.path.name)
        open_path = None
        close_after_capture = False
    screenshot_payload = capture_ltspice_window_screenshot(
        output_path=_resolve_png_output_path(
            kind="plots",
            name=f"{render_dataset.path.stem}_{vectors[0]}",
            output_path=output_path,
        ),
        open_path=open_path,
        title_hint=title_hint,
        settle_seconds=settle_seconds,
        downscale_factor=downscale_factor,
        avoid_space_switch=True,
        prefer_screencapturekit=True,
        close_after_capture=close_after_capture,
    )
    if any(
        [
            title is not None,
            max_points != 2000,
            width != 1280,
            height != 720,
        ]
    ):
        warnings.append(
            "LTspice plot rendering uses .plt settings and ignores title/max_points/width/height."
        )
    validation = None
    if validate_capture:
        validation = _validate_plot_capture(Path(str(screenshot_payload["image_path"])))
        if not validation["valid"]:
            raise RuntimeError(
                "LTspice plot capture appears empty or missing traces "
                f"(trace_pixels={validation['trace_pixels']}, "
                f"required_min={validation['min_trace_pixels']})."
            )
    payload = {
        **screenshot_payload,
        "raw_path": str(dataset.path),
        "render_raw_path": str(render_dataset.path),
        "plot_name": dataset.plot_name,
        "vectors": vectors,
        "plot_settings": plt_payload,
        "mode_requested": normalized_mode,
        "y_mode_requested": normalized_y_mode,
        "mode_used": plt_payload["mode_used"],
        "pane_layout": normalized_pane_layout,
        "dual_axis_requested": dual_axis,
        "validate_capture": bool(validate_capture),
        "backend_used": "ltspice",
        "render_session_id": render_session_id,
        "step_rendering": step_rendering,
        **_step_payload(dataset, selected_step),
    }
    if validation is not None:
        payload["capture_validation"] = validation
    if warnings:
        payload["warnings"] = warnings
    return _image_tool_result(payload)


@mcp.tool()
def generatePlotSettings(
    vectors: list[str],
    plot: str | None = None,
    run_id: str | None = None,
    raw_path: str | None = None,
    step_index: int | None = None,
    mode: Literal["auto", "db", "phase", "real", "imag"] = "auto",
    x_log: bool | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    dual_axis: bool | None = None,
    pane_layout: Literal["single", "split", "per_trace"] = "single",
    output_path: str | None = None,
) -> dict[str, Any]:
    """Generate a LTspice .plt file from vectors and return parsed settings for debugging."""
    if not vectors:
        raise ValueError("vectors must contain at least one vector name")
    dataset = _resolve_dataset(plot=plot, run_id=run_id, raw_path=raw_path)
    selected_step = _resolve_step_index(dataset, step_index)
    normalized_mode = _normalize_plot_mode(mode)
    normalized_layout = _normalize_pane_layout(pane_layout)
    render_dataset, step_rendering = _materialize_plot_step_dataset(
        dataset=dataset,
        selected_step=selected_step,
    )
    plt_target = (
        Path(output_path).expanduser().resolve()
        if output_path
        else render_dataset.path.with_suffix(".plt")
    )
    payload = _write_ltspice_plot_settings_file(
        dataset=render_dataset,
        vectors=vectors,
        mode=normalized_mode,
        pane_layout=normalized_layout,
        dual_axis=dual_axis,
        x_log=x_log,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        output_path=plt_target,
    )
    text = read_text_auto(plt_target)
    payload.update(
        {
            "raw_path": str(dataset.path),
            "render_raw_path": str(render_dataset.path),
            "mode_requested": normalized_mode,
            "pane_layout_requested": normalized_layout,
            "dual_axis_requested": dual_axis,
            "step_rendering": step_rendering,
            "preview": text[:4000],
            "preview_truncated": len(text) > 4000,
        }
    )
    payload.update(_step_payload(dataset, selected_step))
    return payload


@mcp.tool()
def listPlotPresets() -> dict[str, Any]:
    """List built-in deterministic LTspice plot presets."""
    return {
        "presets": [
            {"name": name, **settings}
            for name, settings in sorted(_PLOT_PRESETS.items())
        ]
    }


@mcp.tool()
def generatePlotPresetSettings(
    preset: str,
    vectors: list[str] | None = None,
    plot: str | None = None,
    run_id: str | None = None,
    raw_path: str | None = None,
    step_index: int | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Generate deterministic .plt settings for a named preset."""
    normalized_preset = _normalize_plot_preset(preset)
    dataset = _resolve_dataset(plot=plot, run_id=run_id, raw_path=raw_path)
    selected_vectors = vectors or _default_vectors_for_plot_preset(dataset, normalized_preset)
    settings = _PLOT_PRESETS[normalized_preset]
    payload = generatePlotSettings(
        vectors=selected_vectors,
        plot=plot,
        run_id=run_id,
        raw_path=raw_path,
        step_index=step_index,
        mode=str(settings["mode"]),
        x_log=bool(settings["x_log"]),
        dual_axis=bool(settings["dual_axis"]),
        pane_layout=str(settings["pane_layout"]),
        output_path=output_path,
    )
    payload["plot_preset"] = normalized_preset
    payload["preset_settings"] = settings
    payload["vectors_selected"] = selected_vectors
    return payload


@mcp.tool()
def renderLtspicePlotPresetImage(
    preset: str,
    vectors: list[str] | None = None,
    plot: str | None = None,
    run_id: str | None = None,
    raw_path: str | None = None,
    step_index: int | None = None,
    output_path: str | None = None,
    settle_seconds: float = 1.0,
    downscale_factor: float = 1.0,
    validate_capture: bool = True,
    render_session_id: str | None = None,
) -> CallToolResult:
    """Render a plot image using a built-in deterministic preset."""
    normalized_preset = _normalize_plot_preset(preset)
    dataset = _resolve_dataset(plot=plot, run_id=run_id, raw_path=raw_path)
    selected_vectors = vectors or _default_vectors_for_plot_preset(dataset, normalized_preset)
    settings = _PLOT_PRESETS[normalized_preset]
    result = renderLtspicePlotImage(
        vectors=selected_vectors,
        plot=plot,
        run_id=run_id,
        raw_path=raw_path,
        step_index=step_index,
        output_path=output_path,
        settle_seconds=settle_seconds,
        downscale_factor=downscale_factor,
        mode=str(settings["mode"]),
        x_log=bool(settings["x_log"]),
        dual_axis=bool(settings["dual_axis"]),
        pane_layout=str(settings["pane_layout"]),
        validate_capture=validate_capture,
        render_session_id=render_session_id,
    )
    payload = result.structuredContent
    if not isinstance(payload, dict):
        return result
    merged = dict(payload)
    merged["plot_preset"] = normalized_preset
    merged["preset_settings"] = settings
    merged["vectors_selected"] = selected_vectors
    return _image_tool_result(merged)


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
def closeLtspiceWindow(title_hint: str) -> dict[str, Any]:
    """Close LTspice windows whose title contains title_hint."""
    return close_ltspice_window(title_hint)


@mcp.tool()
def startLtspiceRenderSession(
    path: str,
    title_hint: str | None = None,
) -> dict[str, Any]:
    """Open an LTspice window once for repeated image rendering."""
    session_id = _new_render_session_id()
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Render session path not found: {resolved}")
    resolved_title = title_hint or resolved.name
    open_event = open_in_ltspice_ui(resolved, background=True)
    _render_sessions[session_id] = {
        "path": str(resolved),
        "title_hint": resolved_title,
        "opened": open_event.get("opened", False),
    }
    return {
        "render_session_id": session_id,
        "path": str(resolved),
        "title_hint": resolved_title,
        "open_event": open_event,
    }


@mcp.tool()
def endLtspiceRenderSession(render_session_id: str) -> dict[str, Any]:
    """Close the LTspice window associated with a render session."""
    session = _render_sessions.get(render_session_id)
    if not session:
        raise ValueError(f"Unknown render_session_id '{render_session_id}'")
    close_event = close_ltspice_window(str(session.get("title_hint") or ""))
    _render_sessions.pop(render_session_id, None)
    return {
        "render_session_id": render_session_id,
        "close_event": close_event,
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
    placement_mode: str = "smart",
    open_ui: bool | None = None,
) -> dict[str, Any]:
    """
    Create an LTspice .asc schematic from a SPICE netlist using auto-placement/routing.

    Supports common two-pin primitives (R, C, L, D, V, I) plus multi-pin active elements
    such as X-subcircuits, BJTs (Q), and MOSFETs (M) when symbols can be resolved.
    """
    result = build_schematic_from_netlist(
        workdir=_runner.workdir,
        netlist_content=netlist_content,
        circuit_name=circuit_name,
        output_path=output_path,
        sheet_width=sheet_width,
        sheet_height=sheet_height,
        placement_mode=placement_mode,
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
    placement_mode: str | None = None,
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
        placement_mode=placement_mode,
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
def listIntentCircuitTemplates() -> dict[str, Any]:
    """List high-level circuit intent templates and default parameters."""
    intents: list[dict[str, Any]] = []
    for intent_name in sorted(_INTENT_TEMPLATE_MAP.keys()):
        intents.append(
            {
                "intent": intent_name,
                "template_name": _INTENT_TEMPLATE_MAP[intent_name],
                "default_parameters": _INTENT_DEFAULT_PARAMETERS.get(intent_name, {}),
            }
        )
    return {"intents": intents}


@mcp.tool()
def createIntentCircuit(
    intent: str,
    parameters: dict[str, Any] | None = None,
    circuit_name: str | None = None,
    output_path: str | None = None,
    sheet_width: int | None = None,
    sheet_height: int | None = None,
    placement_mode: str = "smart",
    open_ui: bool | None = None,
    validate_schematic: bool = True,
) -> dict[str, Any]:
    """
    Create a circuit from high-level intent templates (filters, amplifier, regulator).

    Returns schematic and sidecar netlist paths, and optionally validation payload.
    """
    normalized_intent = _normalize_intent(intent)
    merged_parameters = _merge_intent_parameters(normalized_intent, parameters)
    template_name = _INTENT_TEMPLATE_MAP[normalized_intent]
    result = build_schematic_from_template(
        workdir=_runner.workdir,
        template_name=template_name,
        parameters=merged_parameters,
        circuit_name=circuit_name or normalized_intent,
        output_path=output_path,
        sheet_width=sheet_width,
        sheet_height=sheet_height,
        placement_mode=placement_mode,
    )
    if validate_schematic:
        result["schematic_validation"] = _validate_schematic_file(
            Path(str(result["asc_path"])).expanduser().resolve()
        )
    result["intent"] = normalized_intent
    result["template_name"] = template_name
    result["parameters_resolved"] = {
        str(key): str(value) for key, value in merged_parameters.items()
    }

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
    placement_mode: str = "smart",
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
        placement_mode=placement_mode,
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
    placement_mode: str = "smart",
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
        placement_mode=placement_mode,
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
            placement_mode="smart",
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

    run, ui_events, effective_ui, retry_payload = _run_simulation_with_auto_convergence_retry(
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
    response["auto_convergence_retry"] = retry_payload
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
    run, ui_events, effective_ui, retry_payload = _run_simulation_with_auto_convergence_retry(
        netlist_path=netlist_path,
        ascii_raw=ascii_raw,
        timeout_seconds=timeout_seconds,
        show_ui=show_ui,
        open_raw_after_run=open_raw_after_run,
    )
    response = _run_payload(run, include_output=False, log_tail_lines=120)
    response["ui_enabled"] = effective_ui
    response["auto_convergence_retry"] = retry_payload
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
    run, ui_events, effective_ui, retry_payload = _run_simulation_with_auto_convergence_retry(
        netlist_path=path,
        ascii_raw=ascii_raw,
        timeout_seconds=timeout_seconds,
        show_ui=show_ui,
        open_raw_after_run=open_raw_after_run,
    )
    response = _run_payload(run, include_output=False, log_tail_lines=120)
    response["ui_enabled"] = effective_ui
    response["auto_convergence_retry"] = retry_payload
    if ui_events:
        response["ui_events"] = ui_events
    return response


@mcp.tool()
def validateSchematic(asc_path: str) -> dict[str, Any]:
    """
    Validate a schematic (.asc) for simulation readiness.

    Checks for components, ground flag, and simulation directives in TEXT commands.
    """
    path = Path(asc_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Schematic file not found: {path}")
    return _validate_schematic_file(path)


@mcp.tool()
def lintSchematic(
    asc_path: str,
    strict: bool = False,
    strict_style: bool = False,
    min_style_score: float = 96.0,
    max_component_overlap: int = 0,
    max_component_crowding: int = 0,
    max_wire_crossing: int = 0,
    lib_zip_path: str | None = None,
) -> dict[str, Any]:
    """Run structural lint checks on a schematic, including pin connectivity and dangling wires."""
    path = Path(asc_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Schematic file not found: {path}")
    symbol_lib: SymbolLibrary | None
    symbol_lib_error: str | None = None
    try:
        symbol_lib = _resolve_symbol_library(lib_zip_path)
    except Exception as exc:
        symbol_lib = None
        symbol_lib_error = str(exc)
    payload = _lint_schematic_file(path, library=symbol_lib, strict=strict)
    if symbol_lib_error:
        payload.setdefault("warnings", []).append(
            f"Symbol library unavailable for pin-level lint: {symbol_lib_error}"
        )
        payload["symbol_library_available"] = False
    else:
        payload["symbol_library_available"] = True

    style_report = _analyze_schematic_visual_quality(path)
    payload["style"] = style_report
    payload["strict_style"] = bool(strict_style)
    payload["style_thresholds"] = {
        "min_style_score": float(min_style_score),
        "max_component_overlap": int(max_component_overlap),
        "max_component_crowding": int(max_component_crowding),
        "max_wire_crossing": int(max_wire_crossing),
    }

    if strict_style:
        findings = style_report.get("findings", [])
        overlap_count = sum(1 for item in findings if str(item.get("type", "")).lower() == "component_overlap")
        crowding_count = sum(1 for item in findings if str(item.get("type", "")).lower() == "component_crowding")
        crossing_count = sum(1 for item in findings if str(item.get("type", "")).lower() == "wire_crossing")
        score = float(style_report.get("score", 0.0))
        if score < float(min_style_score):
            payload.setdefault("errors", []).append(
                f"Style score {score:.3f} is below minimum {float(min_style_score):.3f}."
            )
        if overlap_count > int(max_component_overlap):
            payload.setdefault("errors", []).append(
                f"component_overlap count {overlap_count} exceeds max_component_overlap {int(max_component_overlap)}."
            )
        if crowding_count > int(max_component_crowding):
            payload.setdefault("errors", []).append(
                f"component_crowding count {crowding_count} exceeds max_component_crowding {int(max_component_crowding)}."
            )
        if crossing_count > int(max_wire_crossing):
            payload.setdefault("errors", []).append(
                f"wire_crossing count {crossing_count} exceeds max_wire_crossing {int(max_wire_crossing)}."
            )
        payload["style_counts"] = {
            "component_overlap": overlap_count,
            "component_crowding": crowding_count,
            "wire_crossing": crossing_count,
        }
        payload["valid"] = len(payload.get("errors", [])) == 0
    return payload


@mcp.tool()
def simulateSchematicFile(
    asc_path: str,
    ascii_raw: bool = False,
    timeout_seconds: int | None = None,
    show_ui: bool | None = None,
    open_raw_after_run: bool = False,
    validate_first: bool = True,
    abort_on_validation_error: bool = False,
) -> dict[str, Any]:
    """
    Run LTspice batch simulation for an existing schematic (.asc) file.

    Optional preflight validation is included to help agents debug schematics before simulation.
    """
    global _loaded_netlist
    path = Path(asc_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Schematic file not found: {path}")

    validation: dict[str, Any] | None = None
    if validate_first:
        validation = _validate_schematic_file(path)
        if abort_on_validation_error and not validation.get("valid", False):
            raise ValueError(
                "Schematic validation failed before simulation. "
                f"Issues: {validation.get('issues', [])}"
            )

    target_info = _resolve_schematic_simulation_target(path, require_sidecar_on_macos=True)
    if not bool(target_info.get("can_batch_simulate")):
        raise ValueError(str(target_info.get("error")))
    run_target = Path(str(target_info["run_target_path"])).expanduser().resolve()
    sidecar_netlist: Path | None = (
        Path(str(target_info["sidecar_path"])).expanduser().resolve()
        if target_info.get("sidecar_path")
        else None
    )

    _loaded_netlist = run_target
    run, ui_events, effective_ui, retry_payload = _run_simulation_with_auto_convergence_retry(
        netlist_path=run_target,
        ascii_raw=ascii_raw,
        timeout_seconds=timeout_seconds,
        show_ui=show_ui,
        open_raw_after_run=open_raw_after_run,
    )
    response = _run_payload(run, include_output=False, log_tail_lines=120)
    response["schematic_path"] = str(path)
    response["run_target_path"] = str(run_target)
    response["used_sidecar_netlist"] = sidecar_netlist is not None
    if sidecar_netlist is not None:
        response["sidecar_netlist_path"] = str(sidecar_netlist)
    response["ui_enabled"] = effective_ui
    response["auto_convergence_retry"] = retry_payload
    if validation is not None:
        response["schematic_validation"] = validation
    if ui_events:
        response["ui_events"] = ui_events
    return response


@mcp.tool()
def resolveSchematicSimulationTarget(
    asc_path: str,
    require_sidecar_on_macos: bool = True,
) -> dict[str, Any]:
    """
    Resolve which file simulateSchematicFile will execute and explain sidecar requirements.
    """
    path = Path(asc_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Schematic file not found: {path}")
    return _resolve_schematic_simulation_target(
        path,
        require_sidecar_on_macos=require_sidecar_on_macos,
    )


@mcp.tool()
def scanModelIssues(
    run_id: str | None = None,
    log_path: str | None = None,
    search_paths: list[str] | None = None,
    suggest_matches: bool = True,
    max_scan_files: int = 300,
    max_suggestions_per_issue: int = 5,
) -> dict[str, Any]:
    """Scan LTspice log text for missing model/include/subcircuit issues."""
    resolved_log: Path | None = None
    source_run_id: str | None = None
    source_path: Path | None = None
    if log_path:
        resolved_log = Path(log_path).expanduser().resolve()
    else:
        run = _resolve_run(run_id)
        source_run_id = run.run_id
        source_path = run.netlist_path
        resolved_log = run.log_path
    if resolved_log is None or not resolved_log.exists():
        return {
            "run_id": source_run_id or run_id,
            "log_path": str(resolved_log) if resolved_log else None,
            "has_model_issues": False,
            "missing_include_files": [],
            "missing_subcircuits": [],
            "missing_models": [],
            "matched_lines": [],
            "resolution_suggestions": [],
        }
    text = read_text_auto(resolved_log)
    payload = _extract_model_issues_from_text(text)
    payload["run_id"] = source_run_id or run_id
    payload["log_path"] = str(resolved_log)
    payload["suggestions"] = []
    if payload["missing_include_files"]:
        payload["suggestions"].append(
            "Import missing files with importModelFile and add .include lines via patchNetlistModelBindings."
        )
    if payload["missing_subcircuits"] or payload["missing_models"]:
        payload["suggestions"].append(
            "Use patchNetlistModelBindings with subckt_aliases/model_aliases to remap references."
        )

    payload["resolution_suggestions"] = []
    payload["model_search"] = {
        "search_paths": [],
        "scanned_file_count": 0,
        "scan_truncated": False,
        "enabled": bool(suggest_matches),
    }
    if payload["has_model_issues"] and suggest_matches:
        resolved_search_paths = _resolve_model_search_paths(
            run_workdir=_runner.workdir,
            extra_paths=search_paths,
            source_path=source_path,
        )
        inventory = _scan_model_inventory(
            search_paths=resolved_search_paths,
            max_scan_files=max_scan_files,
        )
        payload["resolution_suggestions"] = _suggest_model_resolutions(
            issues=payload,
            inventory=inventory,
            limit_per_issue=max_suggestions_per_issue,
        )
        payload["model_search"] = {
            "search_paths": inventory["search_paths"],
            "scanned_file_count": inventory["scanned_file_count"],
            "scan_truncated": inventory["scan_truncated"],
            "enabled": True,
        }
    return payload


@mcp.tool()
def importModelFile(
    source_path: str,
    destination_name: str | None = None,
    models_dir: str | None = None,
) -> dict[str, Any]:
    """Import a model file into the MCP workdir for reproducible .include usage."""
    source = Path(source_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Model file not found: {source}")
    if not source.is_file():
        raise ValueError(f"Model source_path must be a file: {source}")
    target_dir = (
        Path(models_dir).expanduser().resolve()
        if models_dir
        else (_runner.workdir / "models").resolve()
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    target_name = destination_name or source.name
    target = (target_dir / target_name).resolve()
    if target.exists() and target.read_bytes() == source.read_bytes():
        imported = False
    else:
        shutil.copy2(source, target)
        imported = True

    model_text = read_text_auto(target)
    return {
        "source_path": str(source),
        "model_path": str(target),
        "imported": imported,
        "include_directive": f'.include "{target}"',
        "subckt_names": _discover_subckt_names(model_text),
        "model_names": _discover_model_names(model_text),
    }


@mcp.tool()
def patchNetlistModelBindings(
    netlist_path: str,
    include_files: list[str] | None = None,
    subckt_aliases: dict[str, str] | None = None,
    model_aliases: dict[str, str] | None = None,
    backup: bool = True,
) -> dict[str, Any]:
    """Patch netlist model bindings by adding includes and remapping model/subckt names."""
    path = Path(netlist_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Netlist file not found: {path}")
    if not path.is_file():
        raise ValueError(f"netlist_path must be a file: {path}")

    backup_path: Path | None = None
    original_text = path.read_text(encoding="utf-8", errors="ignore")
    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        backup_path.write_text(original_text, encoding="utf-8")

    include_lines: list[str] = []
    for raw_include in include_files or []:
        include_target = Path(raw_include).expanduser().resolve()
        include_lines.append(f'.include "{include_target}"')
    added_include_lines = _append_lines_if_missing(path, include_lines) if include_lines else []

    subckt_lookup = {
        str(key).strip().lower(): str(value).strip()
        for key, value in (subckt_aliases or {}).items()
        if str(key).strip() and str(value).strip()
    }
    model_lookup = {
        str(key).strip().lower(): str(value).strip()
        for key, value in (model_aliases or {}).items()
        if str(key).strip() and str(value).strip()
    }
    model_pattern = None
    if model_lookup:
        model_pattern = re.compile(
            r"\b(" + "|".join(re.escape(key) for key in sorted(model_lookup.keys(), key=len, reverse=True)) + r")\b",
            re.IGNORECASE,
        )

    updated_lines: list[str] = []
    replacement_count = 0
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line
        stripped = raw_line.strip()
        if stripped and not stripped.startswith("*"):
            tokens = raw_line.split()
            if tokens and tokens[0][:1].upper() == "X" and subckt_lookup:
                model_idx = len(tokens) - 1
                while model_idx > 0 and "=" in tokens[model_idx]:
                    model_idx -= 1
                old_symbol = tokens[model_idx]
                new_symbol = subckt_lookup.get(old_symbol.lower())
                if new_symbol and new_symbol != old_symbol:
                    tokens[model_idx] = new_symbol
                    line = " ".join(tokens)
                    replacement_count += 1
            if model_pattern is not None:
                replaced_line = model_pattern.sub(
                    lambda match: model_lookup.get(match.group(1).lower(), match.group(1)),
                    line,
                )
                if replaced_line != line:
                    replacement_count += 1
                line = replaced_line
        updated_lines.append(line)

    path.write_text("\n".join(updated_lines).rstrip() + "\n", encoding="utf-8")
    return {
        "netlist_path": str(path),
        "backup_path": str(backup_path) if backup_path else None,
        "include_lines_added": added_include_lines,
        "replacements": replacement_count,
        "subckt_aliases": subckt_lookup,
        "model_aliases": model_lookup,
    }


@mcp.tool()
def autoDebugSchematic(
    asc_path: str,
    max_iterations: int = 3,
    ascii_raw: bool = False,
    timeout_seconds: int | None = None,
    show_ui: bool | None = None,
    open_raw_after_run: bool = False,
    auto_fix_preflight: bool = True,
    auto_fix_runtime: bool = True,
    auto_import_models: bool = False,
    auto_fix_convergence: bool = True,
    model_search_paths: list[str] | None = None,
) -> dict[str, Any]:
    """
    Iteratively validate, simulate, and apply targeted fixes to a schematic until it runs or stalls.
    """
    if max_iterations <= 0:
        raise ValueError("max_iterations must be > 0")
    path = Path(asc_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Schematic file not found: {path}")

    history: list[dict[str, Any]] = []
    final_result: dict[str, Any] | None = None
    all_actions: list[dict[str, Any]] = []

    for iteration in range(1, max_iterations + 1):
        validation = _validate_schematic_file(path)
        preflight_actions: list[dict[str, Any]] = []
        if auto_fix_preflight and not validation.get("valid", False):
            preflight_actions = _apply_schematic_preflight_fixes(path, validation)
            for action in preflight_actions:
                action["iteration"] = iteration
            if preflight_actions:
                validation = _validate_schematic_file(path)
                all_actions.extend(preflight_actions)

        run_result = simulateSchematicFile(
            asc_path=str(path),
            ascii_raw=ascii_raw,
            timeout_seconds=timeout_seconds,
            show_ui=show_ui,
            open_raw_after_run=open_raw_after_run,
            validate_first=True,
            abort_on_validation_error=False,
        )
        final_result = run_result
        diagnostics_messages = [
            str(item.get("message", ""))
            for item in run_result.get("diagnostics", [])
            if isinstance(item, dict)
        ]
        issue_messages = [str(item) for item in run_result.get("issues", [])]
        log_tail = str(run_result.get("log_tail", ""))
        all_messages = [*diagnostics_messages, *issue_messages, log_tail]
        floating_nodes = _extract_floating_nodes(all_messages)
        convergence_detected = _has_convergence_issue(all_messages)

        log_path_value = run_result.get("log_path")
        if isinstance(log_path_value, str) and log_path_value.strip():
            model_scan = scanModelIssues(
                run_id=None,
                log_path=log_path_value,
                search_paths=model_search_paths,
                suggest_matches=True,
                max_scan_files=300,
                max_suggestions_per_issue=5,
            )
        else:
            model_scan = _extract_model_issues_from_text(log_tail)
            model_scan["resolution_suggestions"] = []

        runtime_actions: list[dict[str, Any]] = []
        if auto_fix_runtime:
            try:
                netlist_path = _ensure_debug_fix_netlist(path)
            except Exception:
                netlist_path = None

            if netlist_path is not None and floating_nodes:
                additions = _apply_floating_node_bleeders(netlist_path, floating_nodes)
                if additions:
                    runtime_actions.append(
                        {
                            "action": "add_bleeder_resistors",
                            "path": str(netlist_path),
                            "lines": additions,
                        }
                    )

            if auto_fix_convergence and netlist_path is not None and convergence_detected:
                convergence_lines = _apply_convergence_options(netlist_path)
                if convergence_lines:
                    runtime_actions.append(
                        {
                            "action": "add_convergence_options",
                            "path": str(netlist_path),
                            "lines": convergence_lines,
                        }
                    )

            if (
                auto_import_models
                and netlist_path is not None
                and model_scan.get("has_model_issues")
            ):
                include_candidates: list[str] = []
                for item in model_scan.get("resolution_suggestions", []):
                    if not isinstance(item, dict):
                        continue
                    if item.get("issue_type") != "missing_include_file":
                        continue
                    direct_matches = item.get("direct_matches") or []
                    best_matches = item.get("best_matches") or []
                    if direct_matches:
                        include_candidates.append(str(direct_matches[0]))
                        continue
                    if best_matches and isinstance(best_matches[0], dict):
                        candidate_path = best_matches[0].get("path")
                        if candidate_path:
                            include_candidates.append(str(candidate_path))
                if include_candidates:
                    include_lines = [
                        f'.include "{Path(item).expanduser().resolve()}"'
                        for item in include_candidates
                    ]
                    added_lines = _append_lines_if_missing(netlist_path, include_lines)
                    if added_lines:
                        runtime_actions.append(
                            {
                                "action": "add_model_includes",
                                "path": str(netlist_path),
                                "lines": added_lines,
                            }
                        )

        for action in runtime_actions:
            action["iteration"] = iteration
        all_actions.extend(runtime_actions)

        history.append(
            {
                "iteration": iteration,
                "validation": validation,
                "run_result": run_result,
                "floating_nodes": floating_nodes,
                "convergence_detected": convergence_detected,
                "model_issues": model_scan,
                "preflight_actions": preflight_actions,
                "runtime_actions": runtime_actions,
            }
        )

        if run_result.get("succeeded", False) and not run_result.get("issues"):
            break
        if not preflight_actions and not runtime_actions:
            break

    succeeded = bool(final_result and final_result.get("succeeded", False))
    confidence = _compute_auto_debug_confidence(
        succeeded=succeeded,
        final_result=final_result,
        actions_applied=all_actions,
        iterations_run=len(history),
    )
    return {
        "asc_path": str(path),
        "iterations_run": len(history),
        "max_iterations": max_iterations,
        "succeeded": succeeded,
        "actions_applied": all_actions,
        "history": history,
        "final_result": final_result,
        "confidence": confidence,
    }


@mcp.tool()
def getToolTelemetry(tool_name: str | None = None) -> dict[str, Any]:
    """Return rolling performance telemetry for MCP tools."""
    payload = _telemetry_payload()
    if tool_name is None:
        return payload
    target = tool_name.strip()
    if not target:
        return payload
    filtered = [entry for entry in payload["tools"] if str(entry.get("tool", "")).lower() == target.lower()]
    return {
        "window_size": payload["window_size"],
        "tool_count": len(filtered),
        "tools": filtered,
    }


@mcp.tool()
def resetToolTelemetry(tool_name: str | None = None) -> dict[str, Any]:
    """Reset rolling performance telemetry for one tool or all tools."""
    if tool_name is None:
        cleared = len(_tool_telemetry)
        _tool_telemetry.clear()
        return {"cleared": cleared, "tool_name": None}
    target = tool_name.strip().lower()
    if not target:
        cleared = len(_tool_telemetry)
        _tool_telemetry.clear()
        return {"cleared": cleared, "tool_name": None}
    removed = 0
    for key in list(_tool_telemetry.keys()):
        if key.lower() == target:
            _tool_telemetry.pop(key, None)
            removed += 1
    return {"cleared": removed, "tool_name": tool_name}


@mcp.tool()
def tailDaemonLog(
    lines: int = 200,
    log_path: str | None = None,
    log_dir: str | None = None,
) -> dict[str, Any]:
    """Return tail text from the active daemon log file."""
    max_lines = max(1, min(5000, int(lines)))
    try:
        resolved = _resolve_daemon_log_path(log_path=log_path, log_dir=log_dir)
    except FileNotFoundError as exc:
        return {
            "log_path": None,
            "line_count": 0,
            "lines_requested": max_lines,
            "text": "",
            "warning": str(exc),
        }
    text = tail_text_file(resolved, max_lines=max_lines)
    return {
        "log_path": str(resolved),
        "line_count": len(text.splitlines()) if text else 0,
        "lines_requested": max_lines,
        "text": text,
    }


@mcp.tool()
def getRecentErrors(
    limit: int = 80,
    log_count: int = 5,
    include_warnings: bool = False,
    log_dir: str | None = None,
) -> dict[str, Any]:
    """Scan recent daemon logs and return structured error entries."""
    max_items = max(1, min(1000, int(limit)))
    entries = _collect_recent_log_entries(
        limit=max_items,
        include_warnings=bool(include_warnings),
        log_count=max(1, min(30, int(log_count))),
        log_dir=log_dir,
    )
    return {
        "entry_count": len(entries),
        "limit": max_items,
        "log_count": max(1, min(30, int(log_count))),
        "include_warnings": bool(include_warnings),
        "entries": entries,
    }


@mcp.tool()
def getCaptureHealth(
    limit: int = 400,
    include_recent_events: bool = False,
) -> dict[str, Any]:
    """Summarize ScreenCapture/LTspice capture health from in-process capture events."""
    max_items = max(1, min(2000, int(limit)))
    health = get_capture_health_snapshot(limit=max_items)
    health["limit"] = max_items
    if include_recent_events:
        health["recent_events"] = get_capture_event_history(limit=min(120, max_items))
    return health


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
    global _runner, _loaded_netlist, _raw_cache, _state_path, _ui_enabled, _job_state_path, _job_history_path
    global _schematic_single_window_enabled, _schematic_live_path
    global _symbol_library, _symbol_library_zip_path
    _stop_job_worker()
    _runner = LTspiceRunner(
        workdir=workdir,
        executable=ltspice_binary,
        default_timeout_seconds=timeout,
    )
    _state_path = workdir / ".ltspice_mcp_runs.json"
    _job_state_path = workdir / ".ltspice_mcp_jobs.json"
    _job_history_path = workdir / ".ltspice_mcp_job_history.json"
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
    _load_job_state()
    _load_job_history_state()
    if any(
        str(_jobs[job_id].get("status", "")).lower() == "queued" and not _jobs[job_id].get("cancel_requested")
        for job_id in _job_order
        if job_id in _jobs
    ):
        _ensure_job_worker()


def _run_streamable_http_with_uvicorn(
    *,
    host: str,
    port: int,
    log_level: str,
    streamable_http_path: str,
) -> None:
    import anyio
    import uvicorn

    _install_uvicorn_noise_filters(streamable_http_path)

    async def _serve() -> None:
        app = mcp.streamable_http_app()
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level=log_level.lower(),
            timeout_graceful_shutdown=5,
        )
        server = uvicorn.Server(config)
        await server.serve()

    anyio.run(_serve)


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
        "--log-level",
        default=_DEFAULT_LOG_LEVEL,
        help="Python log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
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
        default=os.getenv("LTSPICE_MCP_TRANSPORT", "stdio"),
        choices=["stdio", "sse", "streamable-http"],
        help="MCP transport",
    )
    parser.add_argument(
        "--daemon-http",
        action="store_true",
        help=(
            "Run as a long-lived HTTP MCP daemon "
            "(equivalent to --transport streamable-http)."
        ),
    )
    parser.add_argument(
        "--host",
        default=os.getenv("LTSPICE_MCP_HOST", "127.0.0.1"),
        help="Bind host for HTTP transports (sse/streamable-http).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("LTSPICE_MCP_PORT", "8000")),
        help="Bind port for HTTP transports (sse/streamable-http).",
    )
    parser.add_argument(
        "--mount-path",
        default=os.getenv("LTSPICE_MCP_MOUNT_PATH", "/"),
        help="Mount path for HTTP transports (sse/streamable-http).",
    )
    parser.add_argument(
        "--sse-path",
        default=os.getenv("LTSPICE_MCP_SSE_PATH", "/sse"),
        help="SSE endpoint path when --transport sse is used.",
    )
    parser.add_argument(
        "--message-path",
        default=os.getenv("LTSPICE_MCP_MESSAGE_PATH", "/messages/"),
        help="SSE message endpoint path when --transport sse is used.",
    )
    parser.add_argument(
        "--http-path",
        dest="streamable_http_path",
        default=os.getenv("LTSPICE_MCP_STREAMABLE_HTTP_PATH", "/mcp"),
        help="Streamable HTTP endpoint path when --transport streamable-http is used.",
    )
    parser.add_argument(
        "--json-response",
        dest="json_response",
        action="store_true",
        help=(
            "Enable JSON responses for Streamable HTTP tool calls "
            "(compatibility mode for clients that do not consume SSE responses)."
        ),
    )
    parser.add_argument(
        "--sse-response",
        dest="json_response",
        action="store_false",
        help="Use SSE responses for Streamable HTTP tool calls.",
    )
    parser.set_defaults(json_response=None)
    parser.add_argument(
        "--stateless-http",
        dest="stateless_http",
        action="store_true",
        help=(
            "Disable sticky MCP session requirement for Streamable HTTP "
            "(creates a fresh transport per request)."
        ),
    )
    parser.add_argument(
        "--stateful-http",
        dest="stateless_http",
        action="store_false",
        help="Require sticky MCP session IDs for Streamable HTTP.",
    )
    parser.set_defaults(stateless_http=None)
    args = parser.parse_args()
    configured_log_level = _configure_logging(args.log_level)

    transport = args.transport
    if args.daemon_http:
        if args.transport not in {"stdio", "streamable-http"}:
            parser.error("--daemon-http can only be combined with stdio or streamable-http transport.")
        transport = "streamable-http"

    def _normalize_http_path(raw: str, *, trailing_slash: bool = False) -> str:
        value = raw.strip() or "/"
        if not value.startswith("/"):
            value = f"/{value}"
        if trailing_slash and not value.endswith("/"):
            value = f"{value}/"
        return value

    mount_path = _normalize_http_path(args.mount_path)
    sse_path = _normalize_http_path(args.sse_path)
    message_path = _normalize_http_path(args.message_path, trailing_slash=True)
    streamable_http_path = _normalize_http_path(args.streamable_http_path)
    json_response = _DEFAULT_JSON_RESPONSE if args.json_response is None else bool(args.json_response)
    stateless_http = _DEFAULT_STATELESS_HTTP if args.stateless_http is None else bool(args.stateless_http)

    _LOGGER.info(
        "ltspice_mcp_server_start %s",
        json.dumps(
            {
                "transport": transport,
                "host": args.host,
                "port": args.port,
                "mount_path": mount_path,
                "streamable_http_path": streamable_http_path,
                "json_response": json_response,
                "stateless_http": stateless_http,
                "workdir": str(Path(args.workdir).expanduser().resolve()),
                "log_level": configured_log_level,
                "tool_logging_enabled": _TOOL_LOGGING_ENABLED,
            },
            sort_keys=True,
        ),
    )

    _configure_runner(
        workdir=Path(args.workdir).expanduser().resolve(),
        ltspice_binary=args.ltspice_binary,
        timeout=args.timeout,
        ui_enabled=args.ui_enabled,
        schematic_single_window=args.schematic_single_window,
        schematic_live_path=args.schematic_live_path,
    )
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.settings.log_level = configured_log_level
    mcp.settings.mount_path = mount_path
    mcp.settings.sse_path = sse_path
    mcp.settings.message_path = message_path
    mcp.settings.streamable_http_path = streamable_http_path
    mcp.settings.json_response = json_response
    mcp.settings.stateless_http = stateless_http

    if transport == "sse":
        mcp.run(transport=transport, mount_path=mount_path)
        return
    if transport == "streamable-http":
        _run_streamable_http_with_uvicorn(
            host=args.host,
            port=args.port,
            log_level=configured_log_level,
            streamable_http_path=streamable_http_path,
        )
        return
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
