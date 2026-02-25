from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import anyio

from ltspice_mcp.ltspice import find_ltspice_executable

try:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
except Exception:  # pragma: no cover - optional dependency for real integration runs
    ClientSession = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment]
    stdio_client = None  # type: ignore[assignment]


def _real_plot_tests_enabled() -> bool:
    value = os.getenv("LTSPICE_MCP_RUN_REAL_SCK", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _sck_helper_runtime_available() -> tuple[bool, str]:
    helper_override = os.getenv("LTSPICE_MCP_SCK_HELPER_PATH")
    if helper_override:
        helper_path = Path(helper_override).expanduser()
        if helper_path.exists():
            return True, ""
        return False, f"LTSPICE_MCP_SCK_HELPER_PATH does not exist: {helper_path}"
    if shutil.which("swiftc") is None:
        return False, "swiftc is required to build the ScreenCaptureKit helper."
    return True, ""


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


@unittest.skipUnless(platform.system() == "Darwin", "Real LTspice plot tests require macOS")
class TestPlotRenderMCPReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not _real_plot_tests_enabled():
            raise unittest.SkipTest(
                "Set LTSPICE_MCP_RUN_REAL_SCK=1 to run real MCP plot render integration tests."
            )
        if ClientSession is None or StdioServerParameters is None or stdio_client is None:
            raise unittest.SkipTest("MCP python client is not available in this environment.")
        helper_ok, helper_reason = _sck_helper_runtime_available()
        if not helper_ok:
            raise unittest.SkipTest(helper_reason)
        ltspice_binary = os.getenv("LTSPICE_BINARY") or str(find_ltspice_executable() or "")
        if not ltspice_binary:
            raise unittest.SkipTest("LTspice binary not found.")
        cls.ltspice_binary = ltspice_binary
        cls.workdir = Path(tempfile.mkdtemp(prefix="ltspice_real_plot_mcp_")).resolve()

    async def _with_session(self, callback):
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "ltspice_mcp.server",
                "--transport",
                "stdio",
                "--workdir",
                str(self.workdir),
                "--timeout",
                "180",
                "--ltspice-binary",
                self.ltspice_binary,
            ],
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                return await callback(session)

    def test_real_ac_multi_trace_render(self) -> None:
        async def _run(session):
            netlist = (
                "* ac multi trace\n"
                "V1 in 0 AC 1\n"
                "R1 in out 1k\n"
                "C1 out 0 1u\n"
                ".ac dec 20 10 100k\n"
                ".end\n"
            )
            sim = _extract_call_result(
                await session.call_tool(
                    "simulateNetlist",
                    {"netlist_content": netlist, "circuit_name": "real_ac_multi_trace"},
                )
            )
            self.assertTrue(sim.get("succeeded"), sim)
            payload = _extract_call_result(
                await session.call_tool(
                    "renderLtspicePlotImage",
                    {
                        "run_id": sim["run_id"],
                        "vectors": ["V(out)", "V(in)"],
                        "mode": "db",
                        "dual_axis": False,
                        "pane_layout": "single",
                        "backend": "auto",
                    },
                )
            )
            self.assertEqual(payload.get("backend_used"), "ltspice")
            self.assertEqual(payload.get("plot_settings", {}).get("trace_count"), 2)
            self.assertTrue(payload.get("capture_validation", {}).get("valid"))
            self.assertTrue(Path(str(payload.get("image_path"))).exists())

        anyio.run(self._with_session, _run)

    def test_real_transient_render_with_panes(self) -> None:
        async def _run(session):
            netlist = (
                "* transient pane test\n"
                "V1 in 0 PULSE(0 1 0 1u 1u 1m 2m)\n"
                "R1 in out 1k\n"
                "C1 out 0 1u\n"
                ".tran 0 8m 0 10u\n"
                ".end\n"
            )
            sim = _extract_call_result(
                await session.call_tool(
                    "simulateNetlist",
                    {"netlist_content": netlist, "circuit_name": "real_tran_panes"},
                )
            )
            self.assertTrue(sim.get("succeeded"), sim)
            payload = _extract_call_result(
                await session.call_tool(
                    "renderLtspicePlotImage",
                    {
                        "run_id": sim["run_id"],
                        "vectors": ["V(out)", "V(in)"],
                        "mode": "real",
                        "pane_layout": "per_trace",
                        "backend": "auto",
                        "x_log": False,
                    },
                )
            )
            self.assertEqual(payload.get("backend_used"), "ltspice")
            self.assertEqual(payload.get("plot_settings", {}).get("parsed", {}).get("npanes"), 2)
            self.assertTrue(payload.get("capture_validation", {}).get("valid"))
            self.assertTrue(Path(str(payload.get("image_path"))).exists())

        anyio.run(self._with_session, _run)

    def test_real_stepped_render_step_index(self) -> None:
        async def _run(session):
            netlist = (
                "* stepped test\n"
                ".param rval=1k\n"
                "V1 in 0 PULSE(0 1 0 1u 1u 500u 1m)\n"
                "R1 in out {rval}\n"
                "C1 out 0 1u\n"
                ".step param rval list 1k 2k\n"
                ".tran 0 4m 0 20u\n"
                ".end\n"
            )
            sim = _extract_call_result(
                await session.call_tool(
                    "simulateNetlist",
                    {"netlist_content": netlist, "circuit_name": "real_step_render"},
                )
            )
            self.assertTrue(sim.get("succeeded"), sim)
            payload = _extract_call_result(
                await session.call_tool(
                    "renderLtspicePlotImage",
                    {
                        "run_id": sim["run_id"],
                        "vectors": ["V(out)"],
                        "step_index": 1,
                        "mode": "real",
                        "backend": "auto",
                    },
                )
            )
            self.assertEqual(payload.get("selected_step"), 1)
            step_rendering = payload.get("step_rendering", {})
            self.assertTrue(step_rendering.get("step_materialized"))
            self.assertTrue(str(payload.get("render_raw_path", "")).endswith("__step1.raw"))
            self.assertTrue(Path(str(payload.get("render_raw_path"))).exists())
            self.assertTrue(payload.get("capture_validation", {}).get("valid"))

        anyio.run(self._with_session, _run)

    def test_real_generate_plot_settings_tool(self) -> None:
        async def _run(session):
            netlist = (
                "* generate plt tool test\n"
                "V1 in 0 AC 1\n"
                "R1 in out 1k\n"
                "C1 out 0 1u\n"
                ".ac dec 10 10 10k\n"
                ".end\n"
            )
            sim = _extract_call_result(
                await session.call_tool(
                    "simulateNetlist",
                    {"netlist_content": netlist, "circuit_name": "real_generate_plt"},
                )
            )
            self.assertTrue(sim.get("succeeded"), sim)
            payload = _extract_call_result(
                await session.call_tool(
                    "generatePlotSettings",
                    {
                        "run_id": sim["run_id"],
                        "vectors": ["V(out)", "V(in)"],
                        "mode": "phase",
                        "pane_layout": "split",
                    },
                )
            )
            self.assertEqual(payload.get("mode_used"), "phase")
            self.assertEqual(payload.get("pane_layout"), "split")
            self.assertTrue(Path(str(payload.get("plt_path"))).exists())
            self.assertGreaterEqual(payload.get("parsed", {}).get("npanes", 0), 1)

        anyio.run(self._with_session, _run)


if __name__ == "__main__":
    unittest.main()
