#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def _extract_call_result(payload: Any) -> Any:
    structured = getattr(payload, "structuredContent", None)
    if structured is not None:
        if isinstance(structured, dict) and "result" in structured:
            return structured["result"]
        return structured

    content = getattr(payload, "content", None) or []
    if not content:
        return None

    text = getattr(content[0], "text", None)
    if text is not None:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


async def _run_smoke_test(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    server_params = StdioServerParameters(
        command=args.server_command,
        args=[
            "--transport",
            "stdio",
            "--workdir",
            str(workdir),
            "--timeout",
            str(args.timeout),
            *(
                ["--ltspice-binary", args.ltspice_binary]
                if args.ltspice_binary
                else []
            ),
        ],
        cwd=str(Path(args.server_cwd).expanduser().resolve()),
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            init = await session.initialize()
            print(f"Connected to {init.serverInfo.name} {init.serverInfo.version}")

            tools_result = await session.list_tools()
            tool_names = {tool.name for tool in tools_result.tools}
            required_tools = {
                "getLtspiceStatus",
                "simulateNetlist",
                "getPlotNames",
                "getVectorsInfo",
                "getVectorData",
                "getLocalExtrema",
                "getBandwidth",
                "getGainPhaseMargin",
                "getRiseFallTime",
                "getSettlingTime",
                "listRuns",
                "getRunDetails",
            }
            missing_tools = required_tools - tool_names
            _require(not missing_tools, f"Missing required tools: {sorted(missing_tools)}")
            print(f"Tool check passed ({len(tool_names)} tools)")

            status = _extract_call_result(await session.call_tool("getLtspiceStatus", {}))
            _require(isinstance(status, dict), "getLtspiceStatus did not return an object")
            ltspice_executable = status.get("ltspice_executable")
            _require(ltspice_executable, "LTspice executable not detected by server")
            print(f"LTspice executable: {ltspice_executable}")

            netlist = """
* RC low-pass smoke test
V1 in 0 AC 1
R1 in out 1k
C1 out 0 1u
.ac dec 20 10 100k
.op
.end
""".strip()

            run = _extract_call_result(
                await session.call_tool(
                    "simulateNetlist",
                    {
                        "netlist_content": netlist,
                        "circuit_name": "smoke_rc_lowpass",
                    },
                )
            )
            _require(isinstance(run, dict), "simulateNetlist did not return an object")
            _require(run.get("succeeded") is True, f"Simulation failed: {run}")
            run_id = run.get("run_id")
            _require(isinstance(run_id, str) and run_id, "Missing run_id in simulation result")
            _require(len(run.get("raw_files", [])) > 0, "Simulation produced no RAW files")
            print(f"Simulation passed (run_id={run_id})")

            plots = _extract_call_result(
                await session.call_tool("getPlotNames", {"run_id": run_id})
            )
            _require(isinstance(plots, dict), "getPlotNames did not return an object")
            _require(len(plots.get("plots", [])) > 0, "No plots returned")
            print(f"Plot check passed ({len(plots['plots'])} plots)")

            vectors_info = _extract_call_result(
                await session.call_tool("getVectorsInfo", {"run_id": run_id})
            )
            _require(isinstance(vectors_info, dict), "getVectorsInfo did not return an object")
            vectors = vectors_info.get("vectors", [])
            _require(len(vectors) > 0, "No vectors returned")
            vector_names = {entry["name"] for entry in vectors if "name" in entry}
            _require("V(out)" in vector_names, "Expected V(out) vector not found")
            print(f"Vector check passed ({len(vectors)} vectors)")

            vector_data = _extract_call_result(
                await session.call_tool(
                    "getVectorData",
                    {
                        "run_id": run_id,
                        "vectors": ["V(out)"],
                        "max_points": 8,
                        "representation": "magnitude-phase",
                    },
                )
            )
            _require(isinstance(vector_data, dict), "getVectorData did not return an object")
            _require(len(vector_data.get("scale_points", [])) > 0, "getVectorData has no samples")
            vout = vector_data.get("vectors", {}).get("V(out)", {})
            _require("magnitude" in vout, "V(out) magnitude data missing")
            print(f"Trace check passed ({len(vector_data['scale_points'])} sampled points)")

            extrema = _extract_call_result(
                await session.call_tool(
                    "getLocalExtrema",
                    {
                        "run_id": run_id,
                        "vectors": ["V(out)"],
                        "options": {"minima": True, "maxima": True, "threshold": 1e-12},
                    },
                )
            )
            _require(isinstance(extrema, dict), "getLocalExtrema did not return an object")
            _require(
                isinstance(extrema.get("extrema"), dict) and "V(out)" in extrema["extrema"],
                "getLocalExtrema missing V(out) result",
            )
            print("Extrema check passed")

            bandwidth = _extract_call_result(
                await session.call_tool(
                    "getBandwidth",
                    {"run_id": run_id, "vector": "V(out)"},
                )
            )
            _require(isinstance(bandwidth, dict), "getBandwidth did not return an object")
            _require("lowpass_bandwidth_hz" in bandwidth, "getBandwidth missing expected output")
            print("Bandwidth check passed")

            margins = _extract_call_result(
                await session.call_tool(
                    "getGainPhaseMargin",
                    {"run_id": run_id, "vector": "V(out)"},
                )
            )
            _require(isinstance(margins, dict), "getGainPhaseMargin did not return an object")
            _require("gain_crossover_hz" in margins, "getGainPhaseMargin missing expected output")
            print("Gain/phase margin check passed")

            runs = _extract_call_result(await session.call_tool("listRuns", {"limit": 10}))
            _require(isinstance(runs, list), "listRuns did not return a list")
            _require(any(item.get("run_id") == run_id for item in runs), "run_id not found in listRuns")
            print(f"Run listing check passed ({len(runs)} returned)")

            details = _extract_call_result(
                await session.call_tool("getRunDetails", {"run_id": run_id, "include_output": False})
            )
            _require(isinstance(details, dict), "getRunDetails did not return an object")
            _require(details.get("run_id") == run_id, "getRunDetails returned wrong run_id")
            print("Run detail check passed")

            transient_netlist = """
* RC transient stepped smoke test
.param rval=1k
V1 in 0 PULSE(0 1 0 1u 1u 500u 1m)
R1 in out {rval}
C1 out 0 1u
.step param rval list 1k 2k
.tran 0 4m 0 20u
.end
""".strip()
            tran_run = _extract_call_result(
                await session.call_tool(
                    "simulateNetlist",
                    {
                        "netlist_content": transient_netlist,
                        "circuit_name": "smoke_rc_transient",
                    },
                )
            )
            _require(isinstance(tran_run, dict), "Transient simulateNetlist did not return an object")
            _require(tran_run.get("succeeded") is True, "Transient simulation failed")
            tran_id = tran_run.get("run_id")

            stepped_data = _extract_call_result(
                await session.call_tool(
                    "getVectorData",
                    {"run_id": tran_id, "vectors": ["V(out)"], "step_index": 1},
                )
            )
            _require(isinstance(stepped_data, dict), "Stepped getVectorData did not return an object")
            _require(stepped_data.get("step_count", 1) >= 2, "Expected stepped run with step_count >= 2")
            _require(stepped_data.get("selected_step") == 1, "Step filter did not select step_index=1")
            print("Step-filter check passed")

            rise_fall = _extract_call_result(
                await session.call_tool(
                    "getRiseFallTime",
                    {"run_id": tran_id, "vector": "V(out)"},
                )
            )
            _require(isinstance(rise_fall, dict), "getRiseFallTime did not return an object")
            _require("rise_time_s" in rise_fall, "getRiseFallTime missing expected output")
            print("Rise/fall check passed")

            settling = _extract_call_result(
                await session.call_tool(
                    "getSettlingTime",
                    {"run_id": tran_id, "vector": "V(out)"},
                )
            )
            _require(isinstance(settling, dict), "getSettlingTime did not return an object")
            _require("settling_time_s" in settling, "getSettlingTime missing expected output")
            print("Settling-time check passed")

    print("MCP smoke test passed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end smoke test for ltspice-mcp via MCP stdio transport"
    )
    parser.add_argument(
        "--server-command",
        default="ltspice-mcp",
        help="Command used to launch the MCP server",
    )
    parser.add_argument(
        "--server-cwd",
        default=str(Path(__file__).resolve().parent),
        help="Working directory for launching the server",
    )
    parser.add_argument(
        "--workdir",
        default=str((Path(__file__).resolve().parent / ".tmp_smoke").resolve()),
        help="LTspice MCP workdir used during the smoke test",
    )
    parser.add_argument(
        "--ltspice-binary",
        default=None,
        help="Explicit LTspice binary path (optional)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Simulation timeout passed to the server",
    )
    args = parser.parse_args()

    try:
        anyio.run(_run_smoke_test, args)
        return 0
    except Exception as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
