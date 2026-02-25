from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

import anyio

from ltspice_mcp.ltspice import find_ltspice_executable

try:
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client
except Exception:  # pragma: no cover - optional dependency for integration runs
    ClientSession = None  # type: ignore[assignment]
    streamable_http_client = None  # type: ignore[assignment]


def _enabled() -> bool:
    return os.getenv("LTSPICE_MCP_RUN_HTTP_INTEGRATION", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _extract_call_result(payload: Any) -> Any:
    structured = getattr(payload, "structuredContent", None)
    if structured is not None:
        if isinstance(structured, dict) and "result" in structured:
            return structured["result"]
        return structured
    content = getattr(payload, "content", None) or []
    for entry in content:
        text = getattr(entry, "text", None)
        if isinstance(text, str):
            return text
    return None


@unittest.skipUnless(_enabled(), "Set LTSPICE_MCP_RUN_HTTP_INTEGRATION=1 to run HTTP integration tests.")
class TestStreamableHttpIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if ClientSession is None or streamable_http_client is None:
            raise unittest.SkipTest("MCP streamable HTTP client dependencies are unavailable.")
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.workdir = Path(tempfile.mkdtemp(prefix="ltspice_http_integration_")).resolve()
        cls.port = _free_port()
        cls.url = f"http://127.0.0.1:{cls.port}/mcp"
        ltspice_binary = find_ltspice_executable()

        command = [
            sys.executable,
            "-m",
            "ltspice_mcp.server",
            "--daemon-http",
            "--host",
            "127.0.0.1",
            "--port",
            str(cls.port),
            "--http-path",
            "/mcp",
            "--workdir",
            str(cls.workdir),
            "--timeout",
            "180",
        ]
        if ltspice_binary:
            command.extend(["--ltspice-binary", str(ltspice_binary)])

        cls.proc = subprocess.Popen(
            command,
            cwd=str(cls.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        cls._wait_until_ready()

    @classmethod
    def tearDownClass(cls) -> None:
        proc = getattr(cls, "proc", None)
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    @classmethod
    def _wait_until_ready(cls) -> None:
        deadline = time.time() + 45
        while time.time() < deadline:
            proc = cls.proc
            assert proc is not None
            if proc.poll() is not None:
                output = ""
                if proc.stdout is not None:
                    output = proc.stdout.read()
                raise RuntimeError(
                    f"HTTP integration server exited with code {proc.returncode} before ready.\n{output}"
                )
            try:
                anyio.run(cls._probe_once)
                return
            except Exception:  # noqa: BLE001
                time.sleep(0.4)
        raise TimeoutError(f"Timed out waiting for HTTP server readiness: {cls.url}")

    @classmethod
    async def _probe_once(cls) -> None:
        assert streamable_http_client is not None
        assert ClientSession is not None
        async with streamable_http_client(cls.url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                await session.list_tools()

    async def _with_session(self, callback):
        assert streamable_http_client is not None
        assert ClientSession is not None
        async with streamable_http_client(self.url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                return await callback(session)

    def test_http_tools_schema_and_mode_enums(self) -> None:
        async def _run(session):
            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            self.assertIn("renderLtspicePlotImage", names)
            self.assertIn("resolveSchematicSimulationTarget", names)

            plot_tool = next(tool for tool in tools.tools if tool.name == "renderLtspicePlotImage")
            props = plot_tool.inputSchema.get("properties", {})
            self.assertNotIn("backend", props)
            self.assertEqual(set(props["pane_layout"]["enum"]), {"single", "split", "per_trace"})
            self.assertEqual(set(props["mode"]["enum"]), {"auto", "db", "phase", "real", "imag"})
            self.assertEqual(set(props["y_mode"]["enum"]), {"magnitude", "phase", "real", "imag", "db"})

        anyio.run(self._with_session, _run)

    def test_http_rejects_retired_backend_argument(self) -> None:
        async def _run(session):
            asc_path = str(
                (self.repo_root / "tests" / "fixtures" / "schematic" / "common_circuits" / "rc_lowpass_ac.asc")
                .resolve()
            )
            result = await session.call_tool(
                "renderLtspiceSchematicImage",
                {"asc_path": asc_path, "backend": "ltspice"},
            )
            self.assertTrue(result.isError)
            text = _extract_call_result(result)
            self.assertIsInstance(text, str)
            assert isinstance(text, str)
            self.assertIn("Extra inputs are not permitted", text)
            self.assertIn("backend", text)

        anyio.run(self._with_session, _run)

    def test_http_sidecar_resolution_guidance(self) -> None:
        async def _run(session):
            asc_path = str(
                (self.repo_root / "tests" / "fixtures" / "schematic" / "common_circuits" / "rc_lowpass_ac.asc")
                .resolve()
            )
            result = await session.call_tool(
                "resolveSchematicSimulationTarget",
                {"asc_path": asc_path},
            )
            self.assertFalse(result.isError)
            payload = _extract_call_result(result)
            self.assertIsInstance(payload, dict)
            assert isinstance(payload, dict)
            self.assertIn("can_batch_simulate", payload)
            self.assertIn("candidate_sidecar_paths", payload)
            if payload.get("platform") == "Darwin":
                self.assertFalse(payload.get("can_batch_simulate"))
                self.assertEqual(payload.get("reason"), "missing_sidecar_required_on_macos")

        anyio.run(self._with_session, _run)

    def test_http_schematic_render_returns_payload(self) -> None:
        if os.getenv("LTSPICE_MCP_RUN_HTTP_RENDER", "").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            self.skipTest("Set LTSPICE_MCP_RUN_HTTP_RENDER=1 to run live render check.")

        async def _run(session):
            asc_path = str(
                (self.repo_root / "tests" / "fixtures" / "schematic" / "common_circuits" / "bridge_rectifier_filter.asc")
                .resolve()
            )
            result = await session.call_tool(
                "renderLtspiceSchematicImage",
                {"asc_path": asc_path, "settle_seconds": 0.8},
            )
            self.assertFalse(result.isError)
            payload = _extract_call_result(result)
            self.assertIsInstance(payload, dict)
            assert isinstance(payload, dict)
            image_path = Path(str(payload.get("image_path", "")))
            self.assertTrue(image_path.exists())
            self.assertEqual(payload.get("backend_used"), "ltspice")

        anyio.run(self._with_session, _run)


if __name__ == "__main__":
    unittest.main()
