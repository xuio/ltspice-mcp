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
                "getLtspiceLibraryStatus",
                "listLtspiceLibraryEntries",
                "listLtspiceSymbols",
                "getLtspiceSymbolInfo",
                "renderLtspiceSymbolImage",
                "renderLtspiceSchematicImage",
                "renderLtspicePlotImage",
                "setSchematicUiSingleWindow",
                "closeLtspiceWindow",
                "startLtspiceRenderSession",
                "endLtspiceRenderSession",
                "createSchematic",
                "createSchematicFromNetlist",
                "listSchematicTemplates",
                "createSchematicFromTemplate",
                "syncSchematicFromNetlistFile",
                "watchSchematicFromNetlistFile",
                "validateSchematic",
                "simulateNetlist",
                "simulateSchematicFile",
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

            lib_status = _extract_call_result(await session.call_tool("getLtspiceLibraryStatus", {}))
            _require(isinstance(lib_status, dict), "getLtspiceLibraryStatus did not return an object")
            _require(lib_status.get("exists") is True, "LTspice lib.zip not detected by server")
            _require(lib_status.get("symbols_count", 0) > 0, "LTspice symbol library appears empty")
            print(f"Library status check passed ({lib_status.get('symbols_count')} symbols)")

            lib_entries = _extract_call_result(
                await session.call_tool(
                    "listLtspiceLibraryEntries",
                    {"query": "OpAmps/opamp2.asy", "limit": 5},
                )
            )
            _require(isinstance(lib_entries, dict), "listLtspiceLibraryEntries did not return an object")
            _require(lib_entries.get("returned_count", 0) >= 1, "Expected opamp2 entry in library listing")
            print("Library entry search check passed")

            symbols = _extract_call_result(
                await session.call_tool(
                    "listLtspiceSymbols",
                    {"query": "opamp2", "limit": 20},
                )
            )
            _require(isinstance(symbols, dict), "listLtspiceSymbols did not return an object")
            _require(symbols.get("returned_count", 0) >= 1, "Expected opamp2 symbol in symbol search")
            print("Symbol search check passed")

            symbol_info = _extract_call_result(
                await session.call_tool(
                    "getLtspiceSymbolInfo",
                    {"symbol": "opamp2", "include_source": True, "source_max_chars": 1200},
                )
            )
            _require(isinstance(symbol_info, dict), "getLtspiceSymbolInfo did not return an object")
            _require(symbol_info.get("pin_count", 0) >= 5, "opamp2 pin map seems invalid")
            _require("source" in symbol_info, "getLtspiceSymbolInfo missing source field")
            print("Symbol info check passed")

            symbol_image = _extract_call_result(
                await session.call_tool(
                    "renderLtspiceSymbolImage",
                    {"symbol": "opamp2", "downscale_factor": 0.5, "backend": "auto"},
                )
            )
            _require(isinstance(symbol_image, dict), "renderLtspiceSymbolImage did not return an object")
            symbol_image_path = Path(symbol_image.get("image_path", ""))
            _require(symbol_image_path.exists(), "renderLtspiceSymbolImage output missing")
            _require(symbol_image.get("backend_used") in {"ltspice", "svg"}, "Unexpected symbol image backend")
            if symbol_image.get("backend_used") == "svg":
                _require(symbol_image.get("width", 0) < 640, "Symbol image downscale did not apply (svg)")
            else:
                downscale_info = symbol_image.get("downscale", {})
                _require(
                    isinstance(downscale_info, dict) and (downscale_info.get("downscaled") is True or symbol_image.get("downscale_factor") < 1.0),
                    "Symbol image downscale metadata missing for ltspice backend",
                )
            print("Symbol image render check passed")

            templates = _extract_call_result(await session.call_tool("listSchematicTemplates", {}))
            _require(isinstance(templates, dict), "listSchematicTemplates did not return an object")
            template_entries = templates.get("templates", [])
            _require(isinstance(template_entries, list) and len(template_entries) >= 1, "No schematic templates available")
            print(f"Template list check passed ({len(template_entries)} templates)")

            template_schematic = _extract_call_result(
                await session.call_tool(
                    "createSchematicFromTemplate",
                    {
                        "template_name": "rc_lowpass_ac",
                        "parameters": {
                            "vin_ac": "1",
                            "r_value": "1k",
                            "c_value": "1u",
                            "ac_points": "20",
                            "f_start": "10",
                            "f_stop": "100k",
                        },
                        "circuit_name": "smoke_template_schematic",
                        "open_ui": False,
                    },
                )
            )
            _require(isinstance(template_schematic, dict), "createSchematicFromTemplate did not return an object")
            _require(template_schematic.get("asc_path"), "createSchematicFromTemplate missing asc_path")
            print("Template schematic check passed")

            schematic_validation = _extract_call_result(
                await session.call_tool(
                    "validateSchematic",
                    {"asc_path": template_schematic.get("asc_path")},
                )
            )
            _require(isinstance(schematic_validation, dict), "validateSchematic did not return an object")
            _require(schematic_validation.get("valid") is True, "validateSchematic reported invalid template schematic")
            print("Schematic validation check passed")

            schematic_run = _extract_call_result(
                await session.call_tool(
                    "simulateSchematicFile",
                    {
                        "asc_path": template_schematic.get("asc_path"),
                        "validate_first": True,
                        "abort_on_validation_error": True,
                    },
                )
            )
            _require(isinstance(schematic_run, dict), "simulateSchematicFile did not return an object")
            _require(schematic_run.get("succeeded") is True, f"simulateSchematicFile failed: {schematic_run}")
            print("Schematic simulation check passed")

            schematic_image = _extract_call_result(
                await session.call_tool(
                    "renderLtspiceSchematicImage",
                    {
                        "asc_path": template_schematic.get("asc_path"),
                        "downscale_factor": 0.5,
                        "backend": "auto",
                    },
                )
            )
            _require(isinstance(schematic_image, dict), "renderLtspiceSchematicImage did not return an object")
            schematic_image_path = Path(schematic_image.get("image_path", ""))
            _require(schematic_image_path.exists(), "renderLtspiceSchematicImage output missing")
            _require(schematic_image.get("backend_used") in {"ltspice", "svg"}, "Unexpected schematic image backend")
            if schematic_image.get("backend_used") == "svg":
                _require(schematic_image.get("width", 0) < 1400, "Schematic image downscale did not apply (svg)")
            else:
                downscale_info = schematic_image.get("downscale", {})
                _require(
                    isinstance(downscale_info, dict) and (downscale_info.get("downscaled") is True or schematic_image.get("downscale_factor") < 1.0),
                    "Schematic image downscale metadata missing for ltspice backend",
                )
            print("Schematic image render check passed")

            netlist = """
* RC low-pass smoke test
V1 in 0 AC 1
R1 in out 1k
C1 out 0 1u
.ac dec 20 10 100k
.op
.end
""".strip()

            loaded = _extract_call_result(
                await session.call_tool(
                    "loadCircuit",
                    {
                        "netlist": netlist,
                        "circuit_name": "smoke_rc_load_only",
                    },
                )
            )
            _require(isinstance(loaded, dict), "loadCircuit did not return an object")
            _require(loaded.get("loaded") is True, "loadCircuit did not mark circuit as loaded")
            _require(loaded.get("asc_path"), "loadCircuit missing generated asc_path")
            _require(
                Path(str(loaded.get("asc_path"))).exists(),
                "loadCircuit reported asc_path but file is missing",
            )
            print("loadCircuit schematic generation check passed")

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

            synced = _extract_call_result(
                await session.call_tool(
                    "syncSchematicFromNetlistFile",
                    {
                        "netlist_path": run.get("netlist_path"),
                        "circuit_name": "smoke_synced_schematic",
                        "open_ui": False,
                    },
                )
            )
            _require(isinstance(synced, dict), "syncSchematicFromNetlistFile did not return an object")
            _require(synced.get("asc_path"), "syncSchematicFromNetlistFile missing asc_path")
            _require(
                synced.get("reason") in {"forced", "missing_output", "source_changed", "unchanged"},
                f"Unexpected sync reason: {synced.get('reason')}",
            )
            print("Schematic sync check passed")

            watch_result = _extract_call_result(
                await session.call_tool(
                    "watchSchematicFromNetlistFile",
                    {
                        "netlist_path": run.get("netlist_path"),
                        "duration_seconds": 0.1,
                        "poll_interval_seconds": 0.05,
                        "max_updates": 2,
                        "open_ui": False,
                    },
                )
            )
            _require(isinstance(watch_result, dict), "watchSchematicFromNetlistFile did not return an object")
            _require(isinstance(watch_result.get("updates_count"), int), "watchSchematicFromNetlistFile missing updates_count")
            print("Schematic watch check passed")

            auto_schematic = _extract_call_result(
                await session.call_tool(
                    "createSchematicFromNetlist",
                    {
                        "netlist_content": netlist,
                        "circuit_name": "smoke_auto_schematic",
                        "open_ui": False,
                    },
                )
            )
            _require(isinstance(auto_schematic, dict), "createSchematicFromNetlist did not return an object")
            _require(auto_schematic.get("asc_path"), "createSchematicFromNetlist missing asc_path")
            print("Auto-schematic check passed")

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

            plot_image = _extract_call_result(
                await session.call_tool(
                    "renderLtspicePlotImage",
                    {
                        "run_id": run_id,
                        "vectors": ["V(out)"],
                        "y_mode": "magnitude",
                        "downscale_factor": 0.5,
                        "backend": "auto",
                    },
                )
            )
            _require(isinstance(plot_image, dict), "renderLtspicePlotImage did not return an object")
            plot_image_path = Path(plot_image.get("image_path", ""))
            _require(plot_image_path.exists(), "renderLtspicePlotImage output missing")
            _require(plot_image.get("backend_used") in {"ltspice", "svg"}, "Unexpected plot image backend")
            if plot_image.get("backend_used") == "svg":
                _require(plot_image.get("width", 0) < 1280, "Plot image downscale did not apply (svg)")
            else:
                downscale_info = plot_image.get("downscale", {})
                _require(
                    isinstance(downscale_info, dict) and (downscale_info.get("downscaled") is True or plot_image.get("downscale_factor") < 1.0),
                    "Plot image downscale metadata missing for ltspice backend",
                )
            print("Plot image render check passed")

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
