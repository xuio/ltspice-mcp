#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from typing import Any

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


_PERMISSION_MARKERS = (
    "declined tcc",
    "the user declined tccs",
    "screencapturekit capture failed",
    "screen recording",
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


def _looks_like_permission_denied(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _PERMISSION_MARKERS)


async def _trigger(args: argparse.Namespace) -> int:
    command = args.command
    if not shutil.which(command):
        print(f"Command not found: {command}", file=sys.stderr)
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
                result = await session.call_tool(
                    "renderLtspiceSymbolImage",
                    {
                        "symbol": args.symbol,
                        "backend": "ltspice",
                        "settle_seconds": args.settle_seconds,
                    },
                )
                payload = _extract_call_result(result)
                if isinstance(payload, dict):
                    image_path = payload.get("image_path")
                    if image_path:
                        print(f"Capture succeeded. image_path={image_path}")
                        return 0
                print("Capture call completed.")
                return 0
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        if _looks_like_permission_denied(message):
            print("Permission prompt path triggered, but Screen Recording is still denied.", file=sys.stderr)
            print(
                "Open macOS System Settings > Privacy & Security > Screen Recording and enable access.",
                file=sys.stderr,
            )
            print(f"Tool error: {message}", file=sys.stderr)
            return 0
        print(f"Failed to trigger ScreenCaptureKit prompt: {message}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Trigger LTspice ScreenCaptureKit flow through MCP over HTTP "
            "(via mcp-remote) so macOS can present Screen Recording permissions."
        )
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
        "--symbol",
        default="res",
        help="LTspice symbol name to render (default: res)",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=1.5,
        help="Window settle duration before capture (default: 1.5)",
    )
    args = parser.parse_args()
    return anyio.run(_trigger, args)


if __name__ == "__main__":
    raise SystemExit(main())
