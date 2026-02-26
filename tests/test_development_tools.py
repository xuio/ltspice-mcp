from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import anyio

from ltspice_mcp import server
from ltspice_mcp.models import RawDataset, RawVariable, SimulationDiagnostic, SimulationRun


class _FakeCallResult:
    def __init__(self, payload: dict) -> None:
        self.structuredContent = payload
        self.content = []


class TestDevelopmentTools(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_dev_tools_test_"))
        server._configure_runner(workdir=self.temp_dir, ltspice_binary=None, timeout=10)

    def test_list_intent_templates(self) -> None:
        payload = server.listIntentCircuitTemplates()
        intents = {entry["intent"] for entry in payload["intents"]}
        self.assertIn("rc_lowpass", intents)
        self.assertIn("rc_highpass", intents)
        self.assertIn("non_inverting_amplifier", intents)
        self.assertIn("zener_regulator", intents)

    def test_create_intent_circuit_calls_template_and_validation(self) -> None:
        asc_path = self.temp_dir / "intent.asc"
        asc_path.write_text("Version 4\nSHEET 1 100 100\n", encoding="utf-8")
        with (
            patch("ltspice_mcp.server.build_schematic_from_template") as build_mock,
            patch("ltspice_mcp.server._validate_schematic_file") as validate_mock,
        ):
            build_mock.return_value = {
                "asc_path": str(asc_path),
                "netlist_path": str(asc_path.with_suffix(".cir")),
            }
            validate_mock.return_value = {"valid": True, "issues": []}
            payload = server.createIntentCircuit(
                intent="rc_lowpass",
                parameters={"r_value": "2k"},
                validate_schematic=True,
                open_ui=False,
            )
        self.assertEqual(payload["intent"], "rc_lowpass")
        self.assertTrue(payload["schematic_validation"]["valid"])
        self.assertEqual(payload["parameters_resolved"]["r_value"], "2k")
        build_mock.assert_called_once()

    def test_generate_plot_preset_settings_uses_defaults(self) -> None:
        raw_path = self.temp_dir / "preset.raw"
        raw_path.write_text("raw\n", encoding="utf-8")
        dataset = RawDataset(
            path=raw_path.resolve(),
            plot_name="AC Analysis",
            flags={"complex"},
            metadata={},
            variables=[
                RawVariable(index=0, name="frequency", kind="frequency"),
                RawVariable(index=1, name="V(out)", kind="voltage"),
                RawVariable(index=2, name="V(in)", kind="voltage"),
            ],
            values=[
                [10.0 + 0j, 100.0 + 0j],
                [1.0 + 0j, 0.5 + 0j],
                [1.0 + 0j, 1.0 + 0j],
            ],
            steps=[],
        )
        with (
            patch("ltspice_mcp.server._resolve_dataset", return_value=dataset),
            patch("ltspice_mcp.server.generatePlotSettings") as plt_mock,
        ):
            plt_mock.return_value = {"plt_path": str(self.temp_dir / "plot.plt"), "mode_used": "db"}
            payload = server.generatePlotPresetSettings(
                preset="bode",
                run_id="run-1",
            )
        self.assertEqual(payload["plot_preset"], "bode")
        self.assertEqual(payload["vectors_selected"], ["V(out)"])
        plt_mock.assert_called_once()

    def test_render_plot_preset_image_wraps_payload(self) -> None:
        raw_path = self.temp_dir / "preset2.raw"
        raw_path.write_text("raw\n", encoding="utf-8")
        image_path = self.temp_dir / "plot.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
        dataset = RawDataset(
            path=raw_path.resolve(),
            plot_name="AC Analysis",
            flags={"complex"},
            metadata={},
            variables=[
                RawVariable(index=0, name="frequency", kind="frequency"),
                RawVariable(index=1, name="V(out)", kind="voltage"),
            ],
            values=[
                [10.0 + 0j, 100.0 + 0j],
                [1.0 + 0j, 0.5 + 0j],
            ],
            steps=[],
        )
        fake_result = _FakeCallResult({"image_path": str(image_path)})
        with (
            patch("ltspice_mcp.server._resolve_dataset", return_value=dataset),
            patch("ltspice_mcp.server.renderLtspicePlotImage", return_value=fake_result),
            patch("ltspice_mcp.server._image_tool_result", side_effect=lambda payload: payload),
        ):
            payload = server.renderLtspicePlotPresetImage(
                preset="bode",
                run_id="run-1",
            )
        self.assertEqual(payload["plot_preset"], "bode")
        self.assertEqual(payload["vectors_selected"], ["V(out)"])

    def test_scan_model_issues_from_log(self) -> None:
        log_path = self.temp_dir / "model.log"
        log_path.write_text(
            "\n".join(
                [
                    "Could not open include file: missing.lib",
                    "Unknown subcircuit called in: XU1 out in foo_subckt",
                    "can't find definition of model DFAST",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        payload = server.scanModelIssues(log_path=str(log_path))
        self.assertTrue(payload["has_model_issues"])
        self.assertIn("missing.lib", payload["missing_include_files"])
        self.assertIn("DFAST", payload["missing_models"])

    def test_scan_model_issues_provides_resolution_suggestions(self) -> None:
        log_path = self.temp_dir / "model_suggest.log"
        model_dir = self.temp_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        candidate = model_dir / "fast_diode.lib"
        candidate.write_text(
            ".model DFAST D(Is=1e-12)\n.subckt foo_subckt in out\n.ends foo_subckt\n",
            encoding="utf-8",
        )
        log_path.write_text(
            "\n".join(
                [
                    "Could not open include file: fast_diode.lib",
                    "Unknown subcircuit called in: XU1 out in foo_subckt",
                    "can't find definition of model DFAST",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        payload = server.scanModelIssues(
            log_path=str(log_path),
            search_paths=[str(model_dir)],
            suggest_matches=True,
            max_scan_files=50,
        )
        self.assertTrue(payload["has_model_issues"])
        self.assertGreaterEqual(payload["model_search"]["scanned_file_count"], 1)
        suggestions = payload.get("resolution_suggestions", [])
        self.assertTrue(suggestions)
        include_suggestions = [item for item in suggestions if item.get("issue_type") == "missing_include_file"]
        self.assertTrue(include_suggestions)
        self.assertTrue(
            include_suggestions[0].get("direct_matches") or include_suggestions[0].get("best_matches")
        )

    def test_import_model_and_patch_bindings(self) -> None:
        model_source = self.temp_dir / "foo.lib"
        model_source.write_text(".subckt myamp in out\n.ends myamp\n", encoding="utf-8")
        imported = server.importModelFile(source_path=str(model_source))
        self.assertTrue(Path(imported["model_path"]).exists())
        self.assertIn("myamp", imported["subckt_names"])

        netlist_path = self.temp_dir / "circuit.cir"
        netlist_path.write_text(
            "* model patch\nXU1 in out oldamp\n.end\n",
            encoding="utf-8",
        )
        patched = server.patchNetlistModelBindings(
            netlist_path=str(netlist_path),
            include_files=[imported["model_path"]],
            subckt_aliases={"oldamp": "myamp"},
            backup=True,
        )
        self.assertGreaterEqual(patched["replacements"], 1)
        text = netlist_path.read_text(encoding="utf-8")
        self.assertIn("myamp", text)
        self.assertIn(".include", text)
        self.assertTrue(Path(str(patched["backup_path"])).exists())

    def test_auto_debug_schematic_adds_bleeder_and_reruns(self) -> None:
        asc_path = self.temp_dir / "debug.asc"
        asc_path.write_text(
            "Version 4\nSHEET 1 100 100\nSYMBOL res 120 120 R0\nSYMATTR InstName R1\nFLAG 120 200 0\nTEXT 48 80 Left 2 !.op\n",
            encoding="utf-8",
        )
        sidecar = asc_path.with_suffix(".cir")
        sidecar.write_text("* debug\nR1 out in 1k\n.op\n.end\n", encoding="utf-8")

        failed = {
            "succeeded": False,
            "issues": ["Singular matrix: node out is floating"],
            "diagnostics": [{"message": "node out is floating", "category": "floating_node"}],
            "log_tail": "node out is floating",
            "run_target_path": str(sidecar),
            "used_sidecar_netlist": True,
        }
        succeeded = {
            "succeeded": True,
            "issues": [],
            "diagnostics": [],
            "log_tail": "",
            "run_target_path": str(sidecar),
            "used_sidecar_netlist": True,
        }
        with patch("ltspice_mcp.server.simulateSchematicFile", side_effect=[failed, succeeded]):
            payload = server.autoDebugSchematic(
                asc_path=str(asc_path),
                max_iterations=3,
                auto_fix_preflight=True,
                auto_fix_runtime=True,
            )
        self.assertTrue(payload["succeeded"])
        self.assertGreaterEqual(payload["iterations_run"], 2)
        self.assertIn("R__BLEED", sidecar.read_text(encoding="utf-8"))
        self.assertIn("confidence", payload)
        self.assertIn("score", payload["confidence"])

    def test_lint_schematic_detects_structural_issues(self) -> None:
        asc_path = self.temp_dir / "lint_case.asc"
        asc_path.write_text(
            "\n".join(
                [
                    "Version 4",
                    "SHEET 1 200 200",
                    "SYMBOL res 100 100 R0",
                    "WIRE 10 10 30 10",
                    "FLAG 30 10 net_a",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        payload = server.lintSchematic(asc_path=str(asc_path), strict=False)
        self.assertFalse(payload["valid"])
        self.assertGreaterEqual(len(payload["errors"]), 1)

    def test_lint_and_validate_support_utf16_ltspice_style(self) -> None:
        asc_path = self.temp_dir / "lint_utf16_style.asc"
        asc_path.write_bytes(
            (
                "Version 4\n"
                "SHEET 1 880 680\n"
                "WIRE 192 160 96 160\n"
                "WIRE 288 160 192 160\n"
                "WIRE 288 224 288 160\n"
                "FLAG 288 224 0\n"
                "SYMBOL voltage 96 144 R0\n"
                "SYMATTR InstName V1\n"
                "SYMATTR Value AC 1\n"
                "SYMBOL Diode 192 176 R0\n"
                "WINDOW 0 24 64 Left 2\n"
                "WINDOW 3 24 0 Left 2\n"
                "SYMATTR InstName D1\n"
                "SYMBOL Res 272 160 R0\n"
                "SYMATTR InstName R1\n"
                "SYMATTR Value 1k\n"
                "TEXT 48 560 Left 2 !.op\n"
            ).encode("utf-16le")
        )

        validation = server.validateSchematic(str(asc_path))
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["components"], 3)
        self.assertEqual(validation["wires"], 3)
        self.assertTrue(validation["has_ground"])
        self.assertIn(".op", [item.lower() for item in validation["simulation_directives"]])

        lint_payload = server.lintSchematic(
            asc_path=str(asc_path),
            strict=False,
            lib_zip_path=str(self.temp_dir / "missing_lib.zip"),
        )
        self.assertTrue(lint_payload["valid"])
        self.assertEqual(lint_payload["component_count"], 3)
        self.assertEqual(lint_payload["wire_count"], 3)
        self.assertEqual(lint_payload["flag_count"], 1)

    def test_tool_telemetry_records_calls(self) -> None:
        server.resetToolTelemetry()
        server.listIntentCircuitTemplates()
        server.listIntentCircuitTemplates()
        payload = server.getToolTelemetry(tool_name="listIntentCircuitTemplates")
        self.assertEqual(payload["tool_count"], 1)
        self.assertGreaterEqual(payload["tools"][0]["calls_total"], 2)

    def test_tool_logging_emits_start_and_success_events(self) -> None:
        previous = server._TOOL_LOGGING_ENABLED
        server._TOOL_LOGGING_ENABLED = True
        try:
            with self.assertLogs(server.__name__, level="INFO") as log_capture:
                server.listIntentCircuitTemplates()
        finally:
            server._TOOL_LOGGING_ENABLED = previous

        events = []
        for line in log_capture.output:
            if "mcp_tool " not in line:
                continue
            payload = json.loads(line.split("mcp_tool ", 1)[1])
            events.append(payload)

        self.assertTrue(events, "Expected structured mcp_tool log entries.")
        starts = [event for event in events if event.get("event") == "tool_call_start"]
        successes = [event for event in events if event.get("event") == "tool_call_success"]
        self.assertTrue(starts, "Expected tool_call_start log entry.")
        self.assertTrue(successes, "Expected tool_call_success log entry.")
        self.assertEqual(starts[0]["tool"], "listIntentCircuitTemplates")
        self.assertEqual(successes[0]["tool"], "listIntentCircuitTemplates")
        self.assertEqual(starts[0]["call_id"], successes[0]["call_id"])

    def test_tool_logging_emits_error_event(self) -> None:
        previous = server._TOOL_LOGGING_ENABLED
        server._TOOL_LOGGING_ENABLED = True
        try:
            with self.assertLogs(server.__name__, level="INFO") as log_capture:
                with self.assertRaises(ValueError):
                    server.getLtspiceSymbolInfo(symbol="")
        finally:
            server._TOOL_LOGGING_ENABLED = previous

        events = []
        for line in log_capture.output:
            if "mcp_tool " not in line:
                continue
            payload = json.loads(line.split("mcp_tool ", 1)[1])
            events.append(payload)

        starts = [event for event in events if event.get("event") == "tool_call_start"]
        errors = [event for event in events if event.get("event") == "tool_call_error"]
        self.assertTrue(starts, "Expected tool_call_start log entry.")
        self.assertTrue(errors, "Expected tool_call_error log entry.")
        self.assertEqual(starts[0]["tool"], "getLtspiceSymbolInfo")
        self.assertEqual(errors[0]["tool"], "getLtspiceSymbolInfo")
        self.assertEqual(errors[0]["error_type"], "ValueError")
        self.assertEqual(starts[0]["call_id"], errors[0]["call_id"])

    def test_uvicorn_access_noise_filter_suppresses_known_probe_lines(self) -> None:
        filt = server._UvicornAccessNoiseFilter("/mcp")

        oauth_record = Mock()
        oauth_record.getMessage.return_value = (
            '127.0.0.1 - "GET /.well-known/oauth-authorization-server HTTP/1.1" 404 Not Found'
        )
        self.assertFalse(filt.filter(oauth_record))

        transient_record = Mock()
        transient_record.getMessage.return_value = (
            '127.0.0.1 - "GET /mcp HTTP/1.1" 400 Bad Request'
        )
        self.assertFalse(filt.filter(transient_record))

        delete_record = Mock()
        delete_record.getMessage.return_value = (
            '127.0.0.1 - "DELETE /mcp HTTP/1.1" 405 Method Not Allowed'
        )
        self.assertFalse(filt.filter(delete_record))

        normal_record = Mock()
        normal_record.getMessage.return_value = '127.0.0.1 - "POST /mcp HTTP/1.1" 200 OK'
        self.assertTrue(filt.filter(normal_record))

    def test_uvicorn_error_noise_filter_suppresses_shutdown_false_positive(self) -> None:
        filt = server._UvicornErrorNoiseFilter()
        noisy = Mock()
        noisy.getMessage.return_value = "ASGI callable returned without completing response."
        self.assertFalse(filt.filter(noisy))
        normal = Mock()
        normal.getMessage.return_value = "Application startup complete."
        self.assertTrue(filt.filter(normal))

    def test_root_noise_filter_suppresses_stateless_termination_noise(self) -> None:
        filt = server._RootNoiseFilter()
        noisy = Mock()
        noisy.getMessage.return_value = "Terminating session: None"
        self.assertFalse(filt.filter(noisy))
        normal = Mock()
        normal.getMessage.return_value = "Terminating session: abc123"
        self.assertTrue(filt.filter(normal))

    def test_simulate_netlist_file_retries_on_convergence(self) -> None:
        netlist = self.temp_dir / "conv_case.cir"
        netlist.write_text("* conv case\nR1 in out 1k\n.tran 0 10m\n.end\n", encoding="utf-8")
        retry_netlist = self.temp_dir / "conv_case_convergence_retry.cir"
        retry_netlist.write_text(netlist.read_text(encoding="utf-8"), encoding="utf-8")

        started_at = datetime.now().astimezone().isoformat()
        executable = Path("/Applications/LTspice.app/Contents/MacOS/LTspice")
        primary_run = SimulationRun(
            run_id="run-primary",
            netlist_path=netlist,
            command=["LTspice", "-b", str(netlist)],
            ltspice_executable=executable,
            started_at=started_at,
            duration_seconds=0.2,
            return_code=255,
            stdout="",
            stderr="",
            log_path=netlist.with_suffix(".log"),
            raw_files=[],
            artifacts=[netlist],
            issues=["Time step too small"],
            warnings=[],
            diagnostics=[
                SimulationDiagnostic(
                    category="convergence",
                    severity="error",
                    message="Time step too small",
                    suggestion=None,
                )
            ],
        )
        fallback_run = SimulationRun(
            run_id="run-fallback",
            netlist_path=retry_netlist,
            command=["LTspice", "-b", str(retry_netlist)],
            ltspice_executable=executable,
            started_at=started_at,
            duration_seconds=0.18,
            return_code=0,
            stdout="",
            stderr="",
            log_path=retry_netlist.with_suffix(".log"),
            raw_files=[retry_netlist.with_suffix(".raw")],
            artifacts=[retry_netlist],
            issues=[],
            warnings=[],
            diagnostics=[],
        )

        with (
            patch(
                "ltspice_mcp.server._prepare_convergence_retry_netlist",
                return_value=(retry_netlist, [".options method=gear"]),
            ),
            patch(
                "ltspice_mcp.server._run_simulation_with_ui",
                side_effect=[
                    (primary_run, [{"stage": "before_run"}], False),
                    (fallback_run, [], False),
                ],
            ),
        ):
            payload = server.simulateNetlistFile(str(netlist))

        self.assertEqual(payload["run_id"], "run-fallback")
        self.assertTrue(payload["succeeded"])
        retry = payload["auto_convergence_retry"]
        self.assertTrue(retry["triggered"])
        self.assertEqual(retry["reason"], "convergence_detected")
        self.assertEqual(retry["selected_attempt"], "fallback")
        self.assertEqual(retry["fallback_run_id"], "run-fallback")
        self.assertEqual(len(retry["attempts"]), 2)

    def test_tail_daemon_log_and_recent_errors(self) -> None:
        log_dir = self.temp_dir / "daemon" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "ltspice-mcp-daemon-20260225-120000.log"
        log_file.write_text(
            "\n".join(
                [
                    'INFO:     127.0.0.1 - "GET /.well-known/oauth-authorization-server HTTP/1.1" 404 Not Found',
                    "2026-02-25 12:00:01 ERROR ltspice_mcp.server run failed",
                    '2026-02-25 12:00:02 INFO ltspice_mcp.server mcp_tool {"event":"tool_call_error","tool":"renderLtspiceSchematicImage"}',
                    '2026-02-25 12:00:03 WARNING ltspice_mcp.ltspice ltspice_capture {"event":"capture_close_incomplete"}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        tail = server.tailDaemonLog(lines=2, log_dir=str(log_dir))
        self.assertEqual(tail["line_count"], 2)
        self.assertIn("capture_close_incomplete", tail["text"])

        errors = server.getRecentErrors(limit=10, log_count=1, include_warnings=True, log_dir=str(log_dir))
        self.assertGreaterEqual(errors["entry_count"], 3)
        messages = [entry["message"] for entry in errors["entries"]]
        self.assertTrue(any("run failed" in message for message in messages))
        self.assertTrue(any("tool_call_error" in message for message in messages))
        self.assertTrue(any("capture_close_incomplete" in message for message in messages))

    def test_get_capture_health_can_include_recent_events(self) -> None:
        with (
            patch(
                "ltspice_mcp.server.get_capture_health_snapshot",
                return_value={"capture_starts": 2, "capture_successes": 1},
            ),
            patch(
                "ltspice_mcp.server.get_capture_event_history",
                return_value=[{"event": "capture_start"}],
            ),
        ):
            payload = server.getCaptureHealth(limit=50, include_recent_events=True)
        self.assertEqual(payload["capture_starts"], 2)
        self.assertEqual(payload["limit"], 50)
        self.assertEqual(payload["recent_events"], [{"event": "capture_start"}])

    def test_run_payload_includes_utf8_log_tail(self) -> None:
        netlist = self.temp_dir / "payload_case.cir"
        netlist.write_text("* payload\n.end\n", encoding="utf-8")
        log_path = self.temp_dir / "payload_case.log"
        log_path.write_text("A\nB\n", encoding="utf-8")
        log_utf8 = self.temp_dir / "payload_case.log.utf8.txt"
        log_utf8.write_text("A\nB\n", encoding="utf-8")
        run = SimulationRun(
            run_id="payload-run",
            netlist_path=netlist,
            command=["LTspice", "-b", str(netlist)],
            ltspice_executable=Path("/Applications/LTspice.app/Contents/MacOS/LTspice"),
            started_at=datetime.now().astimezone().isoformat(),
            duration_seconds=0.1,
            return_code=0,
            stdout="",
            stderr="",
            log_path=log_path,
            raw_files=[],
            artifacts=[netlist, log_path, log_utf8],
            log_utf8_path=log_utf8,
            issues=[],
            warnings=[],
            diagnostics=[],
        )
        payload = server._run_payload(run, include_output=False, log_tail_lines=10)
        self.assertIn("log_tail_utf8", payload)
        self.assertEqual(payload["log_tail_utf8"], "A\nB")

    def test_read_agent_guide_supports_section_and_query(self) -> None:
        payload = server.readAgentGuide(
            section="1",
            query="daemon",
            include_headings=True,
            max_chars=8000,
        )
        self.assertTrue(payload["path"].endswith("AGENT_README.md"))
        self.assertIsNotNone(payload["section_resolved"])
        self.assertGreaterEqual(payload["match_count"], 1)
        self.assertIsInstance(payload["headings"], list)
        self.assertIn("LTspice MCP Agent Playbook", payload["content"])

    def test_agent_readme_resource_is_registered_and_readable(self) -> None:
        resources = server.mcp._resource_manager.list_resources()  # type: ignore[attr-defined]
        uris = {str(item.uri) for item in resources}
        self.assertIn("docs://agent-readme", uris)

        async def _read_resource() -> str:
            resource = await server.mcp._resource_manager.get_resource(  # type: ignore[attr-defined]
                "docs://agent-readme"
            )
            return await resource.read()

        content = anyio.run(_read_resource)
        self.assertIn("LTspice MCP Agent Playbook", content)


if __name__ == "__main__":
    unittest.main()
