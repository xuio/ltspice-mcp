#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

_DENIED_MARKERS = (
    "assistive access",
    "not allowed assistive access",
    "(-25211)",
)


def _extract_call_result(payload: Any) -> Any:
    structured = getattr(payload, "structuredContent", None)
    if structured is not None:
        if isinstance(structured, dict) and "result" in structured:
            return structured["result"]
        return structured

    content = getattr(payload, "content", None) or []
    if not content:
        return None

    for entry in content:
        text = getattr(entry, "text", None)
        if text is None:
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return None


def _looks_like_accessibility_denied(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _DENIED_MARKERS)


def _open_accessibility_settings() -> None:
    subprocess.run(
        [
            "open",
            "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


async def _run(args: argparse.Namespace) -> int:
    command = args.command
    if not shutil.which(command):
        print(f"Command not found: {command}", file=sys.stderr)
        return 2

    asc_path = Path(args.asc_path).expanduser().resolve()
    if not asc_path.exists():
        print(f"Schematic path not found: {asc_path}", file=sys.stderr)
        return 2

    command_args = args.command_args if args.command_args else ["-y", "mcp-remote"]
    if command_args and command_args[0] == "--":
        command_args = command_args[1:]

    server_params = StdioServerParameters(
        command=command,
        args=[*command_args, args.url],
    )

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                init = await session.initialize()
                print(f"Connected via {command} to {args.url}")
                print(f"Server: {init.serverInfo.name} {init.serverInfo.version}")

                open_result = await session.call_tool(
                    "openLtspiceUi",
                    {
                        "path": str(asc_path),
                    },
                )
                open_payload = _extract_call_result(open_result)
                if not isinstance(open_payload, dict) or not open_payload.get("opened", False):
                    print(
                        "Failed to open LTspice UI before accessibility check.",
                        file=sys.stderr,
                    )
                    print(f"openLtspiceUi payload: {open_payload}", file=sys.stderr)
                    return 1

                await anyio.sleep(max(0.0, float(args.settle_seconds)))

                close_result = await session.call_tool(
                    "closeLtspiceWindow",
                    {
                        "title_hint": asc_path.name,
                    },
                )
                close_payload = _extract_call_result(close_result)
                if not isinstance(close_payload, dict):
                    print("Unexpected closeLtspiceWindow payload", file=sys.stderr)
                    print(f"payload: {close_payload}", file=sys.stderr)
                    return 1

                stderr = str(close_payload.get("stderr") or "")
                denied = _looks_like_accessibility_denied(stderr)
                closed = bool(close_payload.get("closed"))

                print(
                    json.dumps(
                        {
                            "asc_path": str(asc_path),
                            "open_event": open_payload,
                            "close_event": close_payload,
                            "accessibility_denied": denied,
                        },
                        indent=2,
                    )
                )

                if closed:
                    print("Accessibility check passed: LTspice window was closed through System Events.")
                    return 0

                if denied:
                    print(
                        "Accessibility access is currently denied for the daemon process.",
                        file=sys.stderr,
                    )
                    print(
                        "Enable access in System Settings > Privacy & Security > Accessibility.",
                        file=sys.stderr,
                    )
                    if args.mode == "trigger":
                        if args.open_settings:
                            _open_accessibility_settings()
                            print("Opened Accessibility settings pane.", file=sys.stderr)
                        print(
                            "If macOS shows a permission dialog, allow it and rerun check-accessibility.",
                            file=sys.stderr,
                        )
                        return 0
                    return 1

                print(
                    "Accessibility check did not close the LTspice window, but denial markers were not detected.",
                    file=sys.stderr,
                )
                return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Accessibility flow failed: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check or trigger macOS Accessibility permissions for LTspice UI control "
            "through MCP over HTTP (via mcp-remote)."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["check", "trigger"],
        default="check",
        help="check: verify accessibility can close LTspice window; trigger: intentionally exercise flow",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8765/mcp",
        help="HTTP MCP endpoint URL (default: http://127.0.0.1:8765/mcp)",
    )
    parser.add_argument(
        "--command",
        default="npx",
        help="Bridge command to run (default: npx)",
    )
    parser.add_argument(
        "--command-args",
        nargs=argparse.REMAINDER,
        default=None,
        help=(
            "Arguments for the bridge command before URL. "
            "Example: --command-args -y mcp-remote (default: -y mcp-remote)"
        ),
    )
    parser.add_argument(
        "--asc-path",
        default="tests/fixtures/schematic/common_circuits/rc_lowpass_ac.asc",
        help="Path to schematic used for open/close accessibility verification",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.8,
        help="Wait time after opening LTspice before close attempt (default: 0.8)",
    )
    parser.add_argument(
        "--open-settings",
        action="store_true",
        help="Open macOS Accessibility settings when mode=trigger and access is denied",
    )

    args = parser.parse_args()
    return anyio.run(_run, args)


if __name__ == "__main__":
    raise SystemExit(main())
