from __future__ import annotations

import math
import json
import threading
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from ltspice_mcp import server
from ltspice_mcp.models import RawDataset, RawStep, RawVariable, SimulationRun


def _make_run(temp_dir: Path, run_id: str = "run-test") -> SimulationRun:
    netlist = temp_dir / f"{run_id}.cir"
    netlist.write_text("* test\n.end\n", encoding="utf-8")
    log_path = temp_dir / f"{run_id}.log"
    log_path.write_text("", encoding="utf-8")
    raw_path = temp_dir / f"{run_id}.raw"
    raw_path.write_text("", encoding="utf-8")
    return SimulationRun(
        run_id=run_id,
        netlist_path=netlist,
        command=["LTspice", "-b", str(netlist)],
        ltspice_executable=Path("/Applications/LTspice.app/Contents/MacOS/LTspice"),
        started_at=datetime.now().astimezone().isoformat(),
        duration_seconds=0.1,
        return_code=0,
        stdout="",
        stderr="",
        log_path=log_path,
        raw_files=[raw_path],
        artifacts=[netlist, log_path, raw_path],
        issues=[],
        warnings=[],
        diagnostics=[],
    )


class TestVerificationAndQueueTools(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ltspice_verify_queue_test_"))
        server._configure_runner(workdir=self.temp_dir, ltspice_binary=None, timeout=10)

    def tearDown(self) -> None:
        server._stop_job_worker()

    def test_parse_meas_results_from_explicit_log(self) -> None:
        log_path = self.temp_dir / "meas.log"
        log_path.write_text(
            "\n".join(
                [
                    "Measurement: gain",
                    "gain: 2.500000e+00",
                    "phase = -4.200000e+01",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        payload = server.parseMeasResults(log_path=str(log_path))
        self.assertEqual(payload["count"], 2)
        self.assertAlmostEqual(payload["measurements"]["gain"], 2.5)
        self.assertAlmostEqual(payload["measurements"]["phase"], -42.0)
        self.assertEqual(payload["measurements_text"]["gain"], "2.500000e+00")

    def test_parse_meas_results_uses_value_column_for_step_tables(self) -> None:
        log_path = self.temp_dir / "meas_step_table.log"
        log_path.write_text(
            "\n".join(
                [
                    "Measurement: ipeak",
                    "  step\tMAX(i(lprobe))\tFROM\tTO",
                    "     1\t58.3693\t0.0006\t0.0007",
                    "     2\t68.6942\t0.0006\t0.0007",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        payload = server.parseMeasResults(log_path=str(log_path))
        self.assertEqual(payload["count"], 1)
        self.assertAlmostEqual(payload["measurements"]["ipeak"], 68.6942, places=6)
        self.assertEqual(payload["measurement_steps"]["ipeak"][0]["step"], 1)
        self.assertAlmostEqual(payload["measurement_steps"]["ipeak"][0]["value"], 58.3693, places=6)
        self.assertEqual(payload["measurement_steps"]["ipeak"][1]["step"], 2)
        self.assertAlmostEqual(payload["measurement_steps"]["ipeak"][1]["value"], 68.6942, places=6)

    def test_parse_meas_results_preserves_high_precision_text(self) -> None:
        log_path = self.temp_dir / "meas_precision.log"
        raw_value = "1.234567890123456e-12"
        log_path.write_text(
            "\n".join(
                [
                    "Measurement: gain",
                    f"gain: {raw_value}",
                    "phase: -3.125m",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        payload = server.parseMeasResults(log_path=str(log_path))
        self.assertEqual(payload["measurements_text"]["gain"], raw_value)
        self.assertAlmostEqual(payload["measurements"]["gain"], 1.234567890123456e-12, places=24)
        self.assertAlmostEqual(payload["measurements"]["phase"], -0.003125, places=12)

    def test_parse_meas_results_ignores_total_elapsed_after_step_table(self) -> None:
        log_path = self.temp_dir / "meas_elapsed_tail.log"
        log_path.write_text(
            "\n".join(
                [
                    "Measurement: vout_pp",
                    "  step\tPP(v(out))\tFROM\tTO",
                    "     1\t1.70893\t0\t0.005",
                    "     2\t1.70893\t0\t0.005",
                    "",
                    "Total elapsed time: 0.006 seconds.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        payload = server.parseMeasResults(log_path=str(log_path))
        self.assertEqual(payload["count"], 1)
        self.assertAlmostEqual(payload["measurements"]["vout_pp"], 1.70893, places=6)
        rows = payload["measurement_steps"]["vout_pp"]
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]["value"], 1.70893, places=6)
        self.assertAlmostEqual(rows[1]["value"], 1.70893, places=6)

    def test_parse_meas_results_step_table_with_axis_column(self) -> None:
        log_path = self.temp_dir / "meas_step_axis.log"
        log_path.write_text(
            "\n".join(
                [
                    "Measurement: phase",
                    "  step\tfreq\tphase",
                    "     1\t1.00000e+03\t-3.125m",
                    "     2\t2.00000e+03\t6.250m",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        payload = server.parseMeasResults(log_path=str(log_path))
        self.assertEqual(payload["count"], 1)
        self.assertAlmostEqual(payload["measurements"]["phase"], 0.00625, places=12)
        self.assertEqual(payload["measurements_text"]["phase"], "6.250m")
        rows = payload["measurement_steps"]["phase"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["step"], 1)
        self.assertAlmostEqual(rows[0]["value"], -0.003125, places=12)
        self.assertEqual(rows[1]["step"], 2)
        self.assertAlmostEqual(rows[1]["value"], 0.00625, places=12)

    def test_parse_meas_results_step_table_header_with_expression_equals(self) -> None:
        log_path = self.temp_dir / "meas_step_expr_equals.log"
        log_path.write_text(
            "\n".join(
                [
                    "Measurement: __mcp_rise_start",
                    "  step\tv(out)=0.06118127703666687",
                    "     1\t6.38245e-05",
                    "     2\t0.000126977",
                    "",
                    "Measurement: __mcp_rise_end",
                    "  step\tv(out)=0.5506314933300018",
                    "     1\t0.00240613",
                    "     2\t0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        payload = server.parseMeasResults(log_path=str(log_path))
        self.assertEqual(payload["count"], 2)
        rows = payload["measurement_steps"]["__mcp_rise_start"]
        self.assertEqual(rows[0]["step"], 1)
        self.assertAlmostEqual(rows[0]["value"], 6.38245e-05, places=12)
        self.assertEqual(rows[1]["step"], 2)
        self.assertAlmostEqual(rows[1]["value"], 0.000126977, places=12)
        self.assertAlmostEqual(payload["measurements"]["__mcp_rise_start"], 0.000126977, places=12)
        self.assertEqual(
            server._measurement_value_for_step(
                parsed=payload,
                measurement_name="__mcp_rise_start",
                step_index=0,
            ),
            rows[0]["value"],
        )
        self.assertEqual(
            server._measurement_value_for_step(
                parsed=payload,
                measurement_name="__mcp_rise_start",
                step_index=1,
            ),
            rows[1]["value"],
        )

    def test_parse_meas_results_inline_expression_value(self) -> None:
        log_path = self.temp_dir / "meas_inline_expr.log"
        log_path.write_text(
            "\n".join(
                [
                    "vpp: PP(v(out))=0.731107 FROM 0 TO 0.008",
                    "mag_1k: mag(v(out))=(-1.44507dB,0deg) at 1000",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        payload = server.parseMeasResults(log_path=str(log_path))
        self.assertEqual(payload["count"], 2)
        self.assertAlmostEqual(payload["measurements"]["vpp"], 0.731107, places=6)
        self.assertAlmostEqual(payload["measurements"]["mag_1k"], -1.44507, places=6)
        self.assertEqual(payload["measurements_text"]["vpp"], "0.731107")
        self.assertEqual(payload["measurements_text"]["mag_1k"], "-1.44507dB")
        self.assertIn("PP(v(out))=0.731107", payload["measurements_display"]["vpp"])

    def test_parse_meas_results_when_expression_uses_at_value(self) -> None:
        log_path = self.temp_dir / "meas_when_at.log"
        log_path.write_text(
            "\n".join(
                [
                    "tsettle_up: abs(v(out)-1)-0.02=0 AT 0.00406703",
                    "tsettle_dn: abs(v(out)-0)-0.02=0 AT 0.00407218",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        payload = server.parseMeasResults(log_path=str(log_path))
        self.assertEqual(payload["count"], 2)
        self.assertAlmostEqual(payload["measurements"]["tsettle_up"], 0.00406703, places=9)
        self.assertAlmostEqual(payload["measurements"]["tsettle_dn"], 0.00407218, places=9)
        self.assertEqual(payload["measurements_text"]["tsettle_up"], "0.00406703")
        self.assertEqual(payload["measurements_text"]["tsettle_dn"], "0.00407218")

    def test_parse_meas_results_handles_lowercase_at_without_breaking_find_values(self) -> None:
        log_path = self.temp_dir / "meas_lowercase_at.log"
        log_path.write_text(
            "\n".join(
                [
                    "bw: db(v(out))=-3 at 1.5879k",
                    "mag_1k: mag(v(out))=(-1.44507dB,0deg) at 1000",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        payload = server.parseMeasResults(log_path=str(log_path))
        self.assertEqual(payload["count"], 2)
        self.assertAlmostEqual(payload["measurements"]["bw"], 1587.9, places=9)
        self.assertEqual(payload["measurements_text"]["bw"], "1.5879k")
        self.assertAlmostEqual(payload["measurements"]["mag_1k"], -1.44507, places=6)
        self.assertEqual(payload["measurements_text"]["mag_1k"], "-1.44507dB")

    def test_parse_meas_results_ignores_simulator_preamble_scalars(self) -> None:
        log_path = self.temp_dir / "meas_ignore_preamble.log"
        log_path.write_text(
            "\n".join(
                [
                    "tnom = 27",
                    "temp = 27",
                    "Measurement: gain",
                    "gain: 1.234",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        payload = server.parseMeasResults(log_path=str(log_path))
        self.assertEqual(payload["count"], 1)
        self.assertIn("gain", payload["measurements"])
        self.assertNotIn("tnom", payload["measurements"])
        self.assertNotIn("temp", payload["measurements"])

    def test_parse_meas_results_run_id_ignores_timeout_header_values(self) -> None:
        run = _make_run(self.temp_dir, run_id="failed_meas_headers")
        run.return_code = -1
        assert run.log_path is not None
        run.log_path.write_text(
            "\n".join(
                [
                    "Circuit = 250",
                    "AsciiRawFile = 1",
                    "No. Variables: 5",
                    "No. Points: 1001",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        server._register_run(run)

        payload = server.parseMeasResults(run_id=run.run_id)
        self.assertEqual(payload["count"], 0)
        self.assertEqual(payload["measurements"], {})

    def test_register_run_snapshots_artifacts_for_immutable_history(self) -> None:
        netlist = self.temp_dir / "mutable.cir"
        netlist.write_text(
            "\n".join(
                [
                    "* mutable run test",
                    ".meas tran vout_max max V(out)",
                    ".end",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        log_path = self.temp_dir / "mutable.log"
        log_path.write_text("vout_max: MAX(v(out))=3.09171\n", encoding="utf-8")
        raw_path = self.temp_dir / "mutable.raw"
        raw_path.write_text("raw-a\n", encoding="utf-8")
        run = SimulationRun(
            run_id="immutable-a",
            netlist_path=netlist,
            command=["LTspice", "-b", str(netlist)],
            ltspice_executable=Path("/Applications/LTspice.app/Contents/MacOS/LTspice"),
            started_at=datetime.now().astimezone().isoformat(),
            duration_seconds=0.1,
            return_code=0,
            stdout="",
            stderr="",
            log_path=log_path,
            log_utf8_path=None,
            raw_files=[raw_path],
            artifacts=[netlist, log_path, raw_path],
            issues=[],
            warnings=[],
            diagnostics=[],
        )
        server._register_run(run)

        # Simulate a later run mutating basename artifacts in-place.
        log_path.write_text("vout_max: MAX(v(out))=1.00845\n", encoding="utf-8")
        raw_path.write_text("raw-b\n", encoding="utf-8")

        payload = server.parseMeasResults(run_id="immutable-a")
        self.assertEqual(payload["count"], 1)
        self.assertAlmostEqual(payload["measurements"]["vout_max"], 3.09171, places=6)
        snap_run = server._resolve_run("immutable-a")
        self.assertIn("run_artifacts/immutable-a", str(snap_run.log_path))
        self.assertIn("run_artifacts/immutable-a", str(snap_run.netlist_path))
        details = server.getRunDetails(run_id="immutable-a", include_output=False)
        self.assertIn("3.09171", details["log_tail"])
        self.assertNotIn("1.00845", details["log_tail"])

    def test_load_run_state_migrates_legacy_paths_to_snapshots(self) -> None:
        netlist = self.temp_dir / "legacy.cir"
        netlist.write_text(
            "\n".join(
                [
                    "* legacy run",
                    ".meas tran vout_max max V(out)",
                    ".end",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        log_path = self.temp_dir / "legacy.log"
        log_path.write_text("vout_max: MAX(v(out))=2.5\n", encoding="utf-8")
        raw_path = self.temp_dir / "legacy.raw"
        raw_path.write_text("legacy-raw-a\n", encoding="utf-8")
        legacy_run = SimulationRun(
            run_id="legacy-run",
            netlist_path=netlist,
            command=["LTspice", "-b", str(netlist)],
            ltspice_executable=Path("/Applications/LTspice.app/Contents/MacOS/LTspice"),
            started_at=datetime.now().astimezone().isoformat(),
            duration_seconds=0.1,
            return_code=0,
            stdout="",
            stderr="",
            log_path=log_path,
            log_utf8_path=None,
            raw_files=[raw_path],
            artifacts=[netlist, log_path, raw_path],
            issues=[],
            warnings=[],
            diagnostics=[],
        )
        state_path = self.temp_dir / ".ltspice_mcp_runs.json"
        state_path.write_text(
            json.dumps({"version": 1, "runs": [legacy_run.to_storage_dict()]}, indent=2),
            encoding="utf-8",
        )

        server._configure_runner(workdir=self.temp_dir, ltspice_binary=None, timeout=10)
        loaded = server._resolve_run("legacy-run")
        self.assertIn("run_artifacts/legacy-run", str(loaded.netlist_path))
        self.assertIn("run_artifacts/legacy-run", str(loaded.log_path))

        # Mutate original basename artifacts; loaded run must remain immutable.
        log_path.write_text("vout_max: MAX(v(out))=9.9\n", encoding="utf-8")
        raw_path.write_text("legacy-raw-b\n", encoding="utf-8")
        parsed = server.parseMeasResults(run_id="legacy-run")
        self.assertEqual(parsed["count"], 1)
        self.assertAlmostEqual(parsed["measurements"]["vout_max"], 2.5, places=6)

        persisted = json.loads(state_path.read_text(encoding="utf-8"))
        run_entry = persisted["runs"][0]
        self.assertIn("run_artifacts/legacy-run", run_entry["netlist_path"])
        self.assertIn("run_artifacts/legacy-run", run_entry["log_path"])

    def test_run_process_simulation_with_cancel_uses_subprocess_module(self) -> None:
        class _FakeProc:
            def __init__(self, *, log_target: Path) -> None:
                self._poll_calls = 0
                self.returncode = 0
                self._log_target = log_target

            def poll(self):  # noqa: ANN001
                self._poll_calls += 1
                if self._poll_calls == 2:
                    self._log_target.write_text("ok\n", encoding="utf-8")
                return None if self._poll_calls == 1 else self.returncode

            def communicate(self):  # noqa: ANN001
                return ("", "")

            def terminate(self) -> None:
                return None

            def wait(self, timeout=None):  # noqa: ANN001
                return self.returncode

            def kill(self) -> None:
                self.returncode = -9

        netlist = self.temp_dir / "queue_subprocess.cir"
        netlist.write_text("* queue subprocess\n.end\n", encoding="utf-8")
        fake_proc = _FakeProc(log_target=netlist.with_suffix(".log"))
        with (
            patch.object(
                server._runner,
                "ensure_executable",
                return_value=Path("/Applications/LTspice.app/Contents/MacOS/LTspice"),
            ),
            patch("ltspice_mcp.server.subprocess.Popen", return_value=fake_proc) as popen_mock,
            patch("ltspice_mcp.server.time.sleep", return_value=None),
        ):
            run, canceled = server._run_process_simulation_with_cancel(
                netlist_path=netlist,
                ascii_raw=False,
                timeout_seconds=5,
                cancel_requested=lambda: False,
            )
        self.assertFalse(canceled)
        self.assertEqual(run.return_code, 0)
        self.assertEqual(run.command[0], "/Applications/LTspice.app/Contents/MacOS/LTspice")
        self.assertEqual(run.command[1], "-b")
        self.assertEqual(Path(run.command[2]).resolve(), netlist.resolve())
        self.assertIsNotNone(run.log_utf8_path)
        self.assertTrue(run.log_utf8_path is not None and run.log_utf8_path.exists())
        self.assertIn(run.log_utf8_path, run.artifacts)
        popen_mock.assert_called_once()

    def test_run_simulation_with_ui_preserves_requested_timeout(self) -> None:
        netlist = self.temp_dir / "timeout_margin.cir"
        netlist.write_text("* timeout margin\n.end\n", encoding="utf-8")
        run = _make_run(self.temp_dir, run_id="timeout_margin_run")
        original_margin = server._SYNC_TOOL_TIMEOUT_MARGIN_SECONDS
        try:
            server._SYNC_TOOL_TIMEOUT_MARGIN_SECONDS = 10
            with (
                patch.object(server._runner, "run_file", return_value=run) as run_file_mock,
                patch("ltspice_mcp.server._register_run", side_effect=lambda value: value),
            ):
                _run, _events, _effective_ui = server._run_simulation_with_ui(
                    netlist_path=netlist,
                    ascii_raw=False,
                    timeout_seconds=180,
                    show_ui=False,
                    open_raw_after_run=False,
                )
            self.assertEqual(run_file_mock.call_args.kwargs["timeout_seconds"], 180)
            self.assertFalse(
                any("Requested timeout was reduced" in warning for warning in run.warnings),
            )
        finally:
            server._SYNC_TOOL_TIMEOUT_MARGIN_SECONDS = original_margin

    def test_run_meas_automation_injects_meas_netlist(self) -> None:
        netlist = self.temp_dir / "base.cir"
        netlist.write_text("* base\nR1 in out 1k\n.end\n", encoding="utf-8")
        resolved_run = _make_run(self.temp_dir, run_id="run-meas")
        with (
            patch("ltspice_mcp.server.simulateNetlistFile") as simulate_mock,
            patch("ltspice_mcp.server.parseMeasResults") as parse_mock,
            patch("ltspice_mcp.server._resolve_run", return_value=resolved_run),
        ):
            simulate_mock.return_value = {"run_id": "run-meas", "succeeded": True}
            parse_mock.return_value = {
                "run_id": "run-meas",
                "count": 1,
                "measurements": {"gain": 1.23},
                "items": [{"name": "gain", "value": 1.23}],
            }
            payload = server.runMeasAutomation(
                measurements=[
                    {"analysis": "ac", "name": "gain", "operation": "max", "target": "mag(V(out))"}
                ],
                netlist_path=str(netlist),
            )

        meas_netlist = Path(payload["meas_netlist_path"])
        self.assertTrue(meas_netlist.exists())
        text = meas_netlist.read_text(encoding="utf-8")
        self.assertIn(".meas ac gain max mag(v(out))", text.lower())
        self.assertIn(".end", text.lower())
        self.assertEqual(payload["measurements"]["measurements"]["gain"], 1.23)
        simulate_mock.assert_called_once()
        parse_mock.assert_called_once()

    def test_run_meas_automation_reports_failed_requested_measurements(self) -> None:
        netlist = self.temp_dir / "meas_fail_base.cir"
        netlist.write_text(
            "\n".join(
                [
                    "* meas fail base",
                    ".meas tran good FIND V(out) AT=5u",
                    ".meas tran bad FIND V(nope) AT=5u",
                    ".end",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        run = _make_run(self.temp_dir, run_id="run-meas-fail")
        run.netlist_path.write_text(
            "\n".join(
                [
                    "* synthetic run netlist",
                    ".meas tran good FIND V(out) AT=5u",
                    ".meas tran bad FIND V(nope) AT=5u",
                    ".end",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        assert run.log_path is not None
        run.log_path.write_text(
            "\n".join(
                [
                    'Measurement "bad" FAIL\'ed: no such vector',
                    "Measurement: good",
                    "good: 1.234",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        server._register_run(run)
        with patch(
            "ltspice_mcp.server.simulateNetlistFile",
            return_value={
                "run_id": run.run_id,
                "succeeded": True,
                "issues": [],
                "warnings": [],
            },
        ):
            payload = server.runMeasAutomation(
                measurements=[
                    ".meas tran bad FIND V(nope) AT=5u",
                    ".meas tran good FIND V(out) AT=5u",
                ],
                netlist_path=str(netlist),
            )

        self.assertFalse(payload["requested_measurements_succeeded"])
        self.assertFalse(payload["run"]["requested_measurements_succeeded"])
        self.assertTrue(payload["run"]["simulation_succeeded"])
        self.assertFalse(payload["run"]["overall_succeeded"])
        failed = payload["failed_measurements"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["name"], "bad")
        self.assertIn("no such vector", str(failed[0].get("reason", "")))
        self.assertEqual(payload["missing_requested_measurements"], [])
        self.assertTrue(payload["warnings"])

    def test_run_meas_automation_scopes_requested_measurements_and_isolates_legacy_meas(self) -> None:
        netlist = self.temp_dir / "meas_scope_base.cir"
        netlist.write_text(
            "\n".join(
                [
                    "* scope base",
                    ".meas op oldbad FIND V(nope)",
                    "V1 in 0 1",
                    ".op",
                    ".end",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        run = _make_run(self.temp_dir, run_id="run-meas-scope")
        assert run.log_path is not None
        run.log_path.write_text(
            "\n".join(
                [
                    ".meas op oldbad FIND V(nope)",
                    "Error: FIND can not be evaluated over an interval.",
                    "Measurement: newok",
                    "newok: 2",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        with (
            patch(
                "ltspice_mcp.server.simulateNetlistFile",
                return_value={"run_id": run.run_id, "succeeded": True, "issues": [], "warnings": []},
            ),
            patch(
                "ltspice_mcp.server.parseMeasResults",
                return_value={
                    "run_id": run.run_id,
                    "count": 2,
                    "measurements": {"oldbad": 0.0, "newok": 2.0},
                    "measurements_text": {"oldbad": "0", "newok": "2"},
                    "measurements_display": {"oldbad": "0", "newok": "2"},
                    "measurement_steps": {},
                    "items": [
                        {"name": "oldbad", "value": 0.0, "value_text": "0", "steps": []},
                        {"name": "newok", "value": 2.0, "value_text": "2", "steps": []},
                    ],
                },
            ),
            patch("ltspice_mcp.server._resolve_run", return_value=run),
        ):
            payload = server.runMeasAutomation(
                netlist_path=str(netlist),
                measurements=[".meas op NEWOK PARAM 2"],
            )

        self.assertEqual(payload["requested_measurements"], ["NEWOK"])
        self.assertEqual(payload["measurements"]["measurements"], {"newok": 2.0})
        self.assertTrue(payload["requested_measurements_succeeded"])
        self.assertTrue(payload["overall_succeeded"])
        self.assertEqual(payload["failed_measurements"], [])

    def test_parse_meas_results_reports_mixed_success_and_failure(self) -> None:
        run = _make_run(self.temp_dir, run_id="run-meas-mixed")
        assert run.log_path is not None
        run.log_path.write_text(
            "\n".join(
                [
                    ".meas op oldbad FIND V(nope)",
                    "Error: FIND can not be evaluated over an interval.",
                    "Measurement: newok",
                    "newok: 2",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        server._register_run(run)

        payload = server.parseMeasResults(run_id=run.run_id)
        self.assertEqual(payload["measurements"], {"newok": 2.0})
        self.assertEqual(payload["count"], 1)
        self.assertTrue(payload["failed_measurements"])
        self.assertEqual(payload["failed_measurements"][0]["name"].lower(), "oldbad")
        self.assertIn(
            "find can not be evaluated over an interval",
            str(payload["failed_measurements"][0].get("reason", "")).lower(),
        )

    def test_run_verification_plan_supports_vector_and_meas_checks(self) -> None:
        run = _make_run(self.temp_dir, run_id="verify-run")
        dataset = RawDataset(
            path=self.temp_dir / "verify.raw",
            plot_name="Transient Analysis",
            flags=set(),
            metadata={},
            variables=[
                RawVariable(index=0, name="time", kind="time"),
                RawVariable(index=1, name="V(out)", kind="voltage"),
            ],
            values=[
                [0.0 + 0j, 1e-3 + 0j, 2e-3 + 0j],
                [0.0 + 0j, 0.75 + 0j, 1.5 + 0j],
            ],
            steps=[],
        )
        with (
            patch(
                "ltspice_mcp.server._resolve_run_target_for_input",
                return_value={
                    "source": "existing_run",
                    "run": run,
                    "run_payload": {"run_id": run.run_id, "succeeded": True},
                },
            ),
            patch(
                "ltspice_mcp.server.parseMeasResults",
                return_value={
                    "run_id": run.run_id,
                    "count": 1,
                    "measurements": {"gain_check": 1.5},
                    "items": [{"name": "gain_check", "value": 1.5}],
                },
            ),
            patch("ltspice_mcp.server._resolve_dataset", return_value=dataset),
        ):
            payload = server.runVerificationPlan(
                assertions=[
                    {
                        "id": "vout_final",
                        "type": "vector_stat",
                        "vector": "V(out)",
                        "statistic": "final",
                        "min": 1.4,
                        "max": 1.6,
                    },
                    {"id": "meas_gain", "type": "meas", "name": "gain_check", "min": 1.4},
                ]
            )

        self.assertTrue(payload["overall_passed"])
        self.assertEqual(payload["passed_count"], 2)
        self.assertEqual(payload["failed_count"], 0)

    def test_run_verification_plan_supports_groups_and_tolerance(self) -> None:
        run = _make_run(self.temp_dir, run_id="verify-group")
        dataset = RawDataset(
            path=self.temp_dir / "verify_group.raw",
            plot_name="Transient Analysis",
            flags=set(),
            metadata={},
            variables=[
                RawVariable(index=0, name="time", kind="time"),
                RawVariable(index=1, name="V(out)", kind="voltage"),
            ],
            values=[
                [0.0 + 0j, 1e-3 + 0j],
                [0.95 + 0j, 1.05 + 0j],
            ],
            steps=[],
        )
        with (
            patch(
                "ltspice_mcp.server._resolve_run_target_for_input",
                return_value={
                    "source": "existing_run",
                    "run": run,
                    "run_payload": {"run_id": run.run_id, "succeeded": True},
                },
            ),
            patch(
                "ltspice_mcp.server.parseMeasResults",
                return_value={
                    "run_id": run.run_id,
                    "count": 1,
                    "measurements": {"gain_check": 2.0},
                    "items": [{"name": "gain_check", "value": 2.0}],
                },
            ),
            patch("ltspice_mcp.server._resolve_dataset", return_value=dataset),
        ):
            payload = server.runVerificationPlan(
                assertions=[
                    {
                        "id": "group_all",
                        "type": "all_of",
                        "assertions": [
                            {
                                "id": "vout_tol",
                                "type": "vector_stat",
                                "vector": "V(out)",
                                "statistic": "final",
                                "target": 1.0,
                                "rel_tol_pct": 10.0,
                            },
                            {
                                "id": "group_any_meas",
                                "type": "any_of",
                                "assertions": [
                                    {"id": "meas_fail", "type": "meas", "name": "gain_check", "min": 3.0},
                                    {"id": "meas_pass", "type": "meas", "name": "gain_check", "min": 1.5},
                                ],
                            },
                        ],
                    }
                ]
            )

        self.assertTrue(payload["overall_passed"])
        self.assertEqual(payload["passed_count"], 1)
        self.assertEqual(payload["failed_count"], 0)
        root = payload["checks"][0]
        self.assertEqual(root["type"], "all_of")
        self.assertEqual(root["child_count"], 2)
        self.assertTrue(root["passed"])

    def test_validate_ltspice_measurements_transient_metrics(self) -> None:
        run = _make_run(self.temp_dir, run_id="tran-validate")
        dataset = RawDataset(
            path=self.temp_dir / "tran_validate.raw",
            plot_name="Transient Analysis",
            flags=set(),
            metadata={},
            variables=[
                RawVariable(index=0, name="time", kind="time"),
                RawVariable(index=1, name="V(out)", kind="voltage"),
            ],
            values=[
                [0.0 + 0j, 0.1 + 0j, 0.2 + 0j, 0.3 + 0j, 0.4 + 0j],
                [0.0 + 0j, 0.2 + 0j, 0.8 + 0j, 1.0 + 0j, 1.0 + 0j],
            ],
            steps=[],
        )
        with (
            patch("ltspice_mcp.server._resolve_dataset", return_value=dataset),
            patch("ltspice_mcp.server._resolve_run_for_dataset_context", return_value=run),
            patch("ltspice_mcp.server.runMeasAutomation") as meas_mock,
        ):
            meas_mock.return_value = {
                "run": {"run_id": "tran-meas", "succeeded": True},
                "meas_netlist_path": str(self.temp_dir / "tran_meas.cir"),
                "measurements": {
                    "measurements": {
                        "__mcp_rise_start": 0.05,
                        "__mcp_rise_end": 0.25,
                        "__mcp_rise_time": 0.2,
                        "__mcp_settle_first": 0.29,
                        "__mcp_settle_time": 0.29,
                    },
                    "measurements_text": {
                        "__mcp_rise_start": "0.05",
                        "__mcp_rise_end": "0.25",
                        "__mcp_rise_time": "0.2",
                        "__mcp_settle_first": "0.29",
                        "__mcp_settle_time": "0.29",
                    },
                    "measurement_steps": {},
                },
            }
            payload = server.validateLtspiceMeasurements(
                vector="V(out)",
                run_id=run.run_id,
                low_threshold_pct=10.0,
                high_threshold_pct=90.0,
                tolerance_percent=2.0,
                target_value=1.0,
                abs_tolerance=1e-9,
                rel_tolerance_pct=0.001,
            )

        self.assertTrue(payload["overall_passed"])
        self.assertEqual(payload["failure_count"], 0)
        self.assertAlmostEqual(payload["comparisons"]["rise_time_s"]["analysis_value"], 0.2, places=9)
        self.assertAlmostEqual(payload["comparisons"]["rise_time_s"]["ltspice_value"], 0.2, places=9)
        self.assertAlmostEqual(payload["authoritative_values"]["rise_time_s"], 0.2, places=9)
        self.assertEqual(payload["authoritative_values_text"]["rise_time_s"], "0.2")
        self.assertTrue(payload["comparisons"]["fall_time_s"]["passed"])
        meas_mock.assert_called_once()

    def test_validate_ltspice_measurements_ac_metrics(self) -> None:
        run = _make_run(self.temp_dir, run_id="ac-validate")
        freq = [10.0, 100.0, 1_000.0, 10_000.0, 100_000.0]
        magnitudes = [10.0, 2.0, 1.0, 0.5, 0.1]
        phases_deg = [-90.0, -120.0, -135.0, -180.0, -225.0]
        response = [
            mag * complex(math.cos(math.radians(phase)), math.sin(math.radians(phase)))
            for mag, phase in zip(magnitudes, phases_deg, strict=True)
        ]
        dataset = RawDataset(
            path=self.temp_dir / "ac_validate.raw",
            plot_name="AC Analysis",
            flags=set(),
            metadata={},
            variables=[
                RawVariable(index=0, name="frequency", kind="frequency"),
                RawVariable(index=1, name="V(out)", kind="voltage"),
            ],
            values=[
                [value + 0j for value in freq],
                response,
            ],
            steps=[],
        )
        expected_bw = server.compute_bandwidth(
            frequency_hz=freq,
            response=response,
            reference="first",
            drop_db=3.0,
        )
        expected_gp = server.compute_gain_phase_margin(
            frequency_hz=freq,
            response=response,
        )
        with (
            patch("ltspice_mcp.server._resolve_dataset", return_value=dataset),
            patch("ltspice_mcp.server._resolve_run_for_dataset_context", return_value=run),
            patch("ltspice_mcp.server.runMeasAutomation") as meas_mock,
        ):
            meas_mock.return_value = {
                "run": {"run_id": "ac-meas", "succeeded": True},
                "meas_netlist_path": str(self.temp_dir / "ac_meas.cir"),
                "measurements": {
                    "measurements": {
                        "__mcp_bw_low": expected_bw["lowpass_bandwidth_hz"],
                        "__mcp_gain_cross": expected_gp["gain_crossover_hz"],
                        "__mcp_phase_cross": expected_gp["phase_crossover_hz"],
                        "__mcp_phase_margin": expected_gp["phase_margin_deg"],
                        "__mcp_gain_margin": expected_gp["gain_margin_db"],
                    },
                    "measurements_text": {
                        "__mcp_bw_low": f"{expected_bw['lowpass_bandwidth_hz']:.12g}",
                        "__mcp_gain_cross": f"{expected_gp['gain_crossover_hz']:.12g}",
                        "__mcp_phase_cross": f"{expected_gp['phase_crossover_hz']:.12g}",
                        "__mcp_phase_margin": f"{expected_gp['phase_margin_deg']:.12g}",
                        "__mcp_gain_margin": f"{expected_gp['gain_margin_db']:.12g}",
                    },
                    "measurement_steps": {},
                },
            }
            payload = server.validateLtspiceMeasurements(
                vector="V(out)",
                run_id=run.run_id,
                reference="first",
                drop_db=3.0,
                abs_tolerance=1e-9,
                rel_tolerance_pct=0.001,
            )

        self.assertTrue(payload["overall_passed"])
        self.assertEqual(payload["failure_count"], 0)
        self.assertAlmostEqual(
            payload["comparisons"]["lowpass_bandwidth_hz"]["analysis_value"],
            expected_bw["lowpass_bandwidth_hz"],
            places=9,
        )
        self.assertTrue(payload["comparisons"]["highpass_bandwidth_hz"]["passed"])
        self.assertIsNotNone(payload["authoritative_values_text"]["lowpass_bandwidth_hz"])
        self.assertIn("__mcp_phase_margin", "\n".join(payload["measurement_statements"]))
        meas_mock.assert_called_once()

    def test_validate_ltspice_measurements_fails_when_generated_meas_execution_fails(self) -> None:
        run = _make_run(self.temp_dir, run_id="ac-validate-fail")
        dataset = RawDataset(
            path=self.temp_dir / "ac_validate_fail.raw",
            plot_name="AC Analysis",
            flags={"complex"},
            metadata={},
            variables=[
                RawVariable(index=0, name="frequency", kind="frequency"),
                RawVariable(index=1, name="V(out)", kind="voltage"),
            ],
            values=[
                [10.0 + 0j, 100.0 + 0j, 1000.0 + 0j],
                [1.0 + 0j, 0.5 + 0j, 0.25 + 0j],
            ],
            steps=[],
        )
        with (
            patch("ltspice_mcp.server._resolve_dataset", return_value=dataset),
            patch("ltspice_mcp.server._resolve_run_for_dataset_context", return_value=run),
            patch("ltspice_mcp.server.runMeasAutomation") as meas_mock,
        ):
            meas_mock.return_value = {
                "run": {"run_id": "ac-meas-fail", "overall_succeeded": False, "succeeded": False},
                "meas_netlist_path": str(self.temp_dir / "ac_meas_fail.cir"),
                "requested_measurements_succeeded": False,
                "measurements": {
                    "measurements": {},
                    "measurements_text": {},
                    "measurement_steps": {},
                },
            }
            payload = server.validateLtspiceMeasurements(
                vector="V(out)",
                run_id=run.run_id,
                reference="first",
                drop_db=3.0,
            )

        self.assertFalse(payload["overall_passed"])
        self.assertFalse(payload["measurement_execution_succeeded"])
        self.assertFalse(payload["validation_run_succeeded"])
        self.assertIn("ltspice_measurements", payload["failures"])

    def test_run_sweep_study_step_mode(self) -> None:
        netlist = self.temp_dir / "step_study.cir"
        netlist.write_text("* step study\nR1 in out 1k\n.end\n", encoding="utf-8")
        run = _make_run(self.temp_dir, run_id="step-run")
        dataset = RawDataset(
            path=self.temp_dir / "step.raw",
            plot_name="Transient Analysis",
            flags={"stepped"},
            metadata={},
            variables=[
                RawVariable(index=0, name="time", kind="time"),
                RawVariable(index=1, name="V(out)", kind="voltage"),
            ],
            values=[
                [0.0 + 0j, 1.0 + 0j, 0.0 + 0j, 1.0 + 0j],
                [1.0 + 0j, 2.0 + 0j, 3.0 + 0j, 4.0 + 0j],
            ],
            steps=[
                RawStep(index=0, start=0, end=2, label="R=1k"),
                RawStep(index=1, start=2, end=4, label="R=2k"),
            ],
        )
        with (
            patch(
                "ltspice_mcp.server.simulateNetlistFile",
                return_value={"run_id": run.run_id, "succeeded": True},
            ),
            patch("ltspice_mcp.server._resolve_run", return_value=run),
            patch("ltspice_mcp.server._resolve_dataset", return_value=dataset),
        ):
            payload = server.runSweepStudy(
                parameter="R",
                mode="step",
                netlist_path=str(netlist),
                values=[1000.0, 2000.0],
                metric_vector="V(out)",
                metric_statistic="final",
            )

        self.assertEqual(payload["record_count"], 2)
        self.assertEqual([row["metric_value"] for row in payload["records"]], [2.0, 4.0])
        self.assertAlmostEqual(payload["aggregate"]["mean"], 3.0)

    def test_run_sweep_study_warns_when_parameter_appears_unused(self) -> None:
        netlist = self.temp_dir / "step_unused_param.cir"
        netlist.write_text("* step study\nR1 in out 1k\n.end\n", encoding="utf-8")
        run = _make_run(self.temp_dir, run_id="step-unused")
        dataset = RawDataset(
            path=self.temp_dir / "step_unused.raw",
            plot_name="Transient Analysis",
            flags={"stepped"},
            metadata={},
            variables=[
                RawVariable(index=0, name="time", kind="time"),
                RawVariable(index=1, name="V(out)", kind="voltage"),
            ],
            values=[
                [0.0 + 0j, 1.0 + 0j, 0.0 + 0j, 1.0 + 0j],
                [1.0 + 0j, 2.0 + 0j, 3.0 + 0j, 4.0 + 0j],
            ],
            steps=[
                RawStep(index=0, start=0, end=2, label="NOSUCH=1"),
                RawStep(index=1, start=2, end=4, label="NOSUCH=2"),
            ],
        )
        with (
            patch(
                "ltspice_mcp.server.simulateNetlistFile",
                return_value={"run_id": run.run_id, "succeeded": True},
            ),
            patch("ltspice_mcp.server._resolve_run", return_value=run),
            patch("ltspice_mcp.server._resolve_dataset", return_value=dataset),
        ):
            payload = server.runSweepStudy(
                parameter="NO_SUCH",
                mode="step",
                netlist_path=str(netlist),
                values=["1", "2"],
                metric_vector="V(out)",
                metric_statistic="final",
            )
        self.assertTrue(payload["warnings"])
        self.assertIn("does not appear to be referenced", payload["warnings"][0])

    def test_run_sweep_study_rejects_unknown_metric_vector_with_value_error(self) -> None:
        netlist = self.temp_dir / "step_metric_vector_validation.cir"
        netlist.write_text("* step study\nR1 in out 1k\n.end\n", encoding="utf-8")
        run = _make_run(self.temp_dir, run_id="step-metric-vector")
        dataset = RawDataset(
            path=self.temp_dir / "step_metric_vector.raw",
            plot_name="Transient Analysis",
            flags={"stepped"},
            metadata={},
            variables=[
                RawVariable(index=0, name="time", kind="time"),
                RawVariable(index=1, name="V(in)", kind="voltage"),
            ],
            values=[
                [0.0 + 0j, 1.0 + 0j, 0.0 + 0j, 1.0 + 0j],
                [1.0 + 0j, 2.0 + 0j, 3.0 + 0j, 4.0 + 0j],
            ],
            steps=[
                RawStep(index=0, start=0, end=2, label="R=1k"),
                RawStep(index=1, start=2, end=4, label="R=2k"),
            ],
        )
        with (
            patch(
                "ltspice_mcp.server.simulateNetlistFile",
                return_value={"run_id": run.run_id, "succeeded": True},
            ),
            patch("ltspice_mcp.server._resolve_run", return_value=run),
            patch("ltspice_mcp.server._resolve_dataset", return_value=dataset),
        ):
            with self.assertRaisesRegex(ValueError, "Unknown vector 'db\\(V\\(out\\)\\)'"):
                server.runSweepStudy(
                    parameter="RVAL",
                    mode="step",
                    netlist_path=str(netlist),
                    values=["1k", "2k"],
                    metric_vector="db(V(out))",
                    metric_statistic="final",
                )

    def test_run_sweep_study_step_mode_operating_point_fallback_aligns_points(self) -> None:
        netlist = self.temp_dir / "step_study_op.cir"
        netlist.write_text("* step study op\nR1 in out 1k\n.end\n", encoding="utf-8")
        run = _make_run(self.temp_dir, run_id="step-op-run")
        dataset = RawDataset(
            path=self.temp_dir / "step_op.raw",
            plot_name="Operating Point",
            flags={"stepped"},
            metadata={},
            variables=[
                RawVariable(index=0, name="rval", kind="param"),
                RawVariable(index=1, name="V(out)", kind="voltage"),
            ],
            values=[
                [1000.0 + 0j, 2000.0 + 0j, 4000.0 + 0j],
                [5.0 + 0j, 3.333333333333 + 0j, 2.0 + 0j],
            ],
            steps=[RawStep(index=0, start=0, end=3, label=None)],
        )
        with (
            patch(
                "ltspice_mcp.server.simulateNetlistFile",
                return_value={"run_id": run.run_id, "succeeded": True},
            ),
            patch("ltspice_mcp.server._resolve_run", return_value=run),
            patch("ltspice_mcp.server._resolve_dataset", return_value=dataset),
        ):
            payload = server.runSweepStudy(
                parameter="RVAL",
                mode="step",
                netlist_path=str(netlist),
                values=["1k", "2k", "4k"],
                metric_vector="V(out)",
                metric_statistic="final",
            )

        self.assertEqual(payload["record_count"], 3)
        self.assertEqual([row["parameter_value"] for row in payload["records"]], [1000.0, 2000.0, 4000.0])
        self.assertAlmostEqual(payload["records"][0]["metric_value"], 5.0, places=6)
        self.assertAlmostEqual(payload["records"][1]["metric_value"], 3.333333333333, places=6)
        self.assertAlmostEqual(payload["records"][2]["metric_value"], 2.0, places=6)
        self.assertEqual(
            [row["step_label"] for row in payload["records"]],
            ["RVAL=1000", "RVAL=2000", "RVAL=4000"],
        )

    def test_run_sweep_study_step_mode_accepts_engineering_range_tokens(self) -> None:
        netlist = self.temp_dir / "step_study_eng_range.cir"
        netlist.write_text("* step study\nR1 in out 1k\n.end\n", encoding="utf-8")
        run = _make_run(self.temp_dir, run_id="step-eng-range")
        dataset = RawDataset(
            path=self.temp_dir / "step_eng_range.raw",
            plot_name="Transient Analysis",
            flags={"stepped"},
            metadata={},
            variables=[
                RawVariable(index=0, name="time", kind="time"),
                RawVariable(index=1, name="V(out)", kind="voltage"),
            ],
            values=[
                [0.0 + 0j, 1.0 + 0j, 0.0 + 0j, 1.0 + 0j],
                [1.0 + 0j, 2.0 + 0j, 3.0 + 0j, 4.0 + 0j],
            ],
            steps=[
                RawStep(index=0, start=0, end=2, label="R=1k"),
                RawStep(index=1, start=2, end=4, label="R=2k"),
            ],
        )
        captured_step_netlist: list[Path] = []
        with (
            patch(
                "ltspice_mcp.server.simulateNetlistFile",
                side_effect=lambda netlist_path, **kwargs: (
                    captured_step_netlist.append(Path(str(netlist_path)).expanduser().resolve()),
                    {"run_id": run.run_id, "succeeded": True},
                )[1],
            ),
            patch("ltspice_mcp.server._resolve_run", return_value=run),
            patch("ltspice_mcp.server._resolve_dataset", return_value=dataset),
        ):
            payload = server.runSweepStudy(
                parameter="RVAL",
                mode="step",
                netlist_path=str(netlist),
                start="1k",
                stop="2k",
                step="1k",
                metric_vector="V(out)",
                metric_statistic="final",
            )

        self.assertEqual(payload["record_count"], 2)
        self.assertEqual([row["parameter_value"] for row in payload["records"]], [1000.0, 2000.0])
        self.assertEqual(len(captured_step_netlist), 1)
        stepped_text = captured_step_netlist[0].read_text(encoding="utf-8").lower()
        self.assertIn(".step param rval list 1000 2000", stepped_text)

    def test_run_sweep_study_step_mode_accepts_scalar_values_string(self) -> None:
        netlist = self.temp_dir / "step_study_eng_values.cir"
        netlist.write_text("* step study\nR1 in out 1k\n.end\n", encoding="utf-8")
        run = _make_run(self.temp_dir, run_id="step-eng-values")
        dataset = RawDataset(
            path=self.temp_dir / "step_eng_values.raw",
            plot_name="Transient Analysis",
            flags={"stepped"},
            metadata={},
            variables=[
                RawVariable(index=0, name="time", kind="time"),
                RawVariable(index=1, name="V(out)", kind="voltage"),
            ],
            values=[
                [0.0 + 0j, 1.0 + 0j, 0.0 + 0j, 1.0 + 0j, 0.0 + 0j, 1.0 + 0j],
                [1.0 + 0j, 2.0 + 0j, 3.0 + 0j, 4.0 + 0j, 5.0 + 0j, 6.0 + 0j],
            ],
            steps=[
                RawStep(index=0, start=0, end=2, label="R=1k"),
                RawStep(index=1, start=2, end=4, label="R=2k"),
                RawStep(index=2, start=4, end=6, label="R=4k"),
            ],
        )
        captured_step_netlist: list[Path] = []
        with (
            patch(
                "ltspice_mcp.server.simulateNetlistFile",
                side_effect=lambda netlist_path, **kwargs: (
                    captured_step_netlist.append(Path(str(netlist_path)).expanduser().resolve()),
                    {"run_id": run.run_id, "succeeded": True},
                )[1],
            ),
            patch("ltspice_mcp.server._resolve_run", return_value=run),
            patch("ltspice_mcp.server._resolve_dataset", return_value=dataset),
        ):
            payload = server.runSweepStudy(
                parameter="RVAL",
                mode="step",
                netlist_path=str(netlist),
                values="1k,2k,4k",
                metric_vector="V(out)",
                metric_statistic="final",
            )

        self.assertEqual(payload["record_count"], 3)
        self.assertEqual([row["parameter_value"] for row in payload["records"]], [1000.0, 2000.0, 4000.0])
        self.assertEqual(len(captured_step_netlist), 1)
        stepped_text = captured_step_netlist[0].read_text(encoding="utf-8").lower()
        self.assertIn(".step param rval list 1000 2000 4000", stepped_text)

    def test_run_sweep_study_monte_carlo_mode(self) -> None:
        netlist = self.temp_dir / "mc_study.cir"
        netlist.write_text("* mc study\nR1 in out {RVAL}\n.end\n", encoding="utf-8")
        run = _make_run(self.temp_dir, run_id="mc-run")
        dataset = RawDataset(
            path=self.temp_dir / "mc.raw",
            plot_name="Transient Analysis",
            flags=set(),
            metadata={},
            variables=[
                RawVariable(index=0, name="time", kind="time"),
                RawVariable(index=1, name="V(out)", kind="voltage"),
            ],
            values=[
                [0.0 + 0j, 1e-3 + 0j],
                [0.9 + 0j, 1.1 + 0j],
            ],
            steps=[],
        )
        with (
            patch(
                "ltspice_mcp.server.simulateNetlistFile",
                return_value={"run_id": run.run_id, "succeeded": True},
            ),
            patch("ltspice_mcp.server._resolve_run", return_value=run),
            patch("ltspice_mcp.server._resolve_dataset", return_value=dataset),
            patch("ltspice_mcp.server.random.gauss", return_value=0.0),
        ):
            payload = server.runSweepStudy(
                parameter="RVAL",
                mode="monte_carlo",
                netlist_path=str(netlist),
                samples=3,
                nominal=1000.0,
                sigma_pct=5.0,
                metric_vector="V(out)",
                metric_statistic="final",
            )

        self.assertEqual(payload["record_count"], 3)
        self.assertTrue(all(item["metric_value"] == 1.1 for item in payload["records"]))
        self.assertAlmostEqual(payload["aggregate"]["mean"], 1.1)

    def test_run_sweep_study_monte_carlo_replaces_existing_param(self) -> None:
        netlist = self.temp_dir / "mc_param_replace.cir"
        netlist.write_text(
            "\n".join(
                [
                    "* mc replace test",
                    ".param RVAL=1000",
                    "V1 in 0 1",
                    "R1 in out {RVAL}",
                    ".tran 0 1m",
                    ".end",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        captured_paths: list[Path] = []
        with (
            patch(
                "ltspice_mcp.server.simulateNetlistFile",
                side_effect=lambda netlist_path, **kwargs: (
                    captured_paths.append(Path(str(netlist_path)).expanduser().resolve()),
                    {"run_id": f"mc-replace-{len(captured_paths)}", "succeeded": False, "issues": []},
                )[1],
            ),
            patch(
                "ltspice_mcp.server._resolve_run",
                return_value=_make_run(self.temp_dir, run_id="mc-replace"),
            ),
        ):
            payload = server.runSweepStudy(
                parameter="RVAL",
                mode="monte_carlo",
                netlist_path=str(netlist),
                samples=3,
                nominal=1000.0,
                sigma_pct=5.0,
            )

        self.assertEqual(payload["record_count"], 3)
        self.assertEqual(len(captured_paths), 3)
        for candidate in captured_paths:
            text = candidate.read_text(encoding="utf-8")
            self.assertEqual(text.lower().count(".param rval="), 1)

    def test_auto_clean_and_visual_inspection_tools(self) -> None:
        asc = self.temp_dir / "messy.asc"
        asc.write_text(
            "\n".join(
                [
                    "Version 4",
                    "SHEET 1 400 300",
                    "SYMBOL res 19 23 R0",
                    "SYMATTR InstName R1",
                    "SYMBOL cap 25 21 R0",
                    "SYMATTR InstName C1",
                    "WIRE 11 21 51 21",
                    "WIRE 31 1 31 41",
                    "TEXT 7 9 Left 2 !.op",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        cleaned = self.temp_dir / "messy_clean.asc"
        clean_payload = server.autoCleanSchematicLayout(
            asc_path=str(asc),
            output_path=str(cleaned),
            prefer_sidecar_regeneration=False,
            grid=16,
            render_after=False,
        )
        self.assertEqual(clean_payload["action_details"]["action"], "normalized_grid")
        self.assertTrue(cleaned.exists())
        normalization = clean_payload["action_details"]["normalization"]
        self.assertEqual(normalization["applied_style_profile"]["anchor_x"], 120)
        self.assertEqual(normalization["applied_style_profile"]["directive_x"], 48)
        cleaned_text = cleaned.read_text(encoding="utf-8")
        self.assertIn("TEXT 48", cleaned_text)

        inspect_payload = server.inspectSchematicVisualQuality(
            asc_path=str(cleaned),
            render=False,
        )
        self.assertIn("quality", inspect_payload)
        self.assertGreaterEqual(inspect_payload["quality"]["component_count"], 2)

    def test_lint_schematic_supports_strict_style_mode(self) -> None:
        asc = self.temp_dir / "style_lint.asc"
        asc.write_text(
            "\n".join(
                [
                    "Version 4",
                    "SHEET 1 400 300",
                    "SYMBOL res 100 100 R0",
                    "SYMATTR InstName R1",
                    "SYMBOL cap 110 100 R0",
                    "SYMATTR InstName C1",
                    "WIRE 100 100 140 100",
                    "TEXT 48 200 Left 2 !.op",
                    "FLAG 100 160 0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        payload = server.lintSchematic(
            asc_path=str(asc),
            strict=False,
            strict_style=True,
            min_style_score=99.0,
            max_component_overlap=0,
            max_component_crowding=0,
            max_wire_crossing=0,
            lib_zip_path=str(self.temp_dir / "missing_lib.zip"),
        )
        self.assertFalse(payload["valid"])
        self.assertTrue(any("component_overlap" in err or "Style score" in err for err in payload["errors"]))
        self.assertTrue(payload["strict_style"])

    def test_daemon_doctor_reports_missing_setup(self) -> None:
        with (
            patch(
                "ltspice_mcp.server.getLtspiceStatus",
                return_value={"ltspice_executable": None},
            ),
            patch("ltspice_mcp.server.getLtspiceUiStatus", return_value={"ui_running": False}),
            patch("ltspice_mcp.server.getRecentErrors", return_value={"entry_count": 2, "entries": []}),
            patch("ltspice_mcp.server.getCaptureHealth", return_value={"success_rate": 0.5}),
        ):
            payload = server.daemonDoctor(include_recent_warnings=True, deep_checks=False)
        self.assertEqual(payload["health"], "fail")
        self.assertGreaterEqual(len(payload["issues"]), 1)
        self.assertGreaterEqual(len(payload["recommendations"]), 1)

    def test_queue_list_status_and_cancel(self) -> None:
        netlist = self.temp_dir / "queued.cir"
        netlist.write_text("* queue\n.end\n", encoding="utf-8")
        with patch("ltspice_mcp.server._ensure_job_worker", return_value=None):
            queued = server.queueSimulationJob(netlist_path=str(netlist), priority=20, max_retries=2)
        self.assertEqual(queued["status"], "queued")
        self.assertEqual(queued["priority"], 20)
        self.assertEqual(queued["max_retries"], 2)
        job_id = queued["job_id"]

        listed = server.listJobs(limit=10)
        self.assertEqual(listed["count"], 1)
        self.assertEqual(listed["jobs"][0]["job_id"], job_id)

        status = server.jobStatus(job_id=job_id, include_run=False)
        self.assertEqual(status["status"], "queued")

        canceled = server.cancelJob(job_id=job_id)
        self.assertEqual(canceled["status"], "canceled")

    def test_job_status_include_run_handles_inflight_run_materialization_gap(self) -> None:
        job_id = "job-materialization-gap"
        with server._job_lock:
            server._jobs[job_id] = {
                "job_id": job_id,
                "status": "running",
                "created_at": datetime.now().astimezone().isoformat(),
                "cancel_requested": False,
                "priority": 50,
                "max_retries": 0,
                "retry_count": 0,
                "queue_seq": 1,
                "run_id": "run-not-yet-registered",
                "attempt_run_ids": ["run-not-yet-registered"],
            }
            if job_id not in server._job_order:
                server._job_order.append(job_id)
        try:
            payload = server.jobStatus(job_id=job_id, include_run=True)
        finally:
            with server._job_lock:
                server._jobs.pop(job_id, None)
                if job_id in server._job_order:
                    server._job_order.remove(job_id)

        self.assertEqual(payload["status"], "running")
        self.assertEqual(payload["run_id"], "run-not-yet-registered")
        self.assertIn("run", payload)
        self.assertFalse(payload["run"]["materialized"])
        self.assertIn("not materialized yet", payload["run"]["message"])

    def test_queue_running_status_includes_active_attempt_run_id(self) -> None:
        netlist = self.temp_dir / "running_attempt.cir"
        netlist.write_text("* running attempt\n.end\n", encoding="utf-8")
        started = threading.Event()
        release = threading.Event()

        def fake_run(*, netlist_path: Path, ascii_raw: bool, timeout_seconds: int | None, cancel_requested):
            _ = (netlist_path, ascii_raw, timeout_seconds, cancel_requested)
            started.set()
            release.wait(2.0)
            run = _make_run(self.temp_dir, run_id="worker-final-id")
            return run, False

        with patch("ltspice_mcp.server._run_process_simulation_with_cancel", side_effect=fake_run):
            queued = server.queueSimulationJob(netlist_path=str(netlist), priority=5, max_retries=0)
            self.assertTrue(started.wait(2.0))
            running = None
            deadline = time.time() + 2.0
            while time.time() < deadline:
                status = server.jobStatus(queued["job_id"], include_run=False)
                if status["status"] == "running":
                    running = status
                    break
                time.sleep(0.02)
            self.assertIsNotNone(running)
            assert running is not None
            self.assertTrue(running["run_id"])
            self.assertIn(running["run_id"], running["attempt_run_ids"])
            release.set()
            deadline = time.time() + 4.0
            final_status = None
            while time.time() < deadline:
                polled = server.jobStatus(queued["job_id"], include_run=False)
                if polled["status"] in {"succeeded", "failed", "canceled"}:
                    final_status = polled
                    break
                time.sleep(0.02)
            self.assertIsNotNone(final_status)
            assert final_status is not None
            self.assertEqual(final_status["status"], "succeeded")
            self.assertTrue(final_status["attempt_run_ids"])
            self.assertIn(final_status["run_id"], final_status["attempt_run_ids"])

    def test_queue_job_uses_submission_snapshot_even_if_source_changes(self) -> None:
        netlist = self.temp_dir / "queued_snapshot_clean.cir"
        netlist.write_text(
            "\n".join(
                [
                    "* queued snapshot clean VERSION_A",
                    "V1 in 0 10",
                    "R1 in out 1k",
                    "R2 out 0 1k",
                    ".op",
                    ".end",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        with patch("ltspice_mcp.server._ensure_job_worker", return_value=None):
            queued = server.queueSimulationJob(netlist_path=str(netlist), priority=10, max_retries=0)

        run_path = Path(str(queued["run_path"])).expanduser().resolve()
        self.assertTrue(run_path.exists())
        self.assertNotEqual(run_path, netlist.resolve())
        self.assertIn("VERSION_A", run_path.read_text(encoding="utf-8"))

        # Mutate the original source after queue submission; the queued run must
        # continue to use the immutable snapshot captured at submission time.
        netlist.write_text(
            "\n".join(
                [
                    "* queued snapshot clean VERSION_B",
                    "V1 in 0 10",
                    "R1 in out 2k",
                    "R2 out 0 1k",
                    ".op",
                    ".end",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        observed: dict[str, str] = {}

        def fake_run(*, netlist_path: Path, ascii_raw: bool, timeout_seconds: int | None, cancel_requested):
            observed["path"] = str(netlist_path)
            observed["text"] = netlist_path.read_text(encoding="utf-8")
            run = _make_run(self.temp_dir, run_id="queued-snapshot-run")
            run.netlist_path = netlist_path
            run.command = ["/Applications/LTspice.app/Contents/MacOS/LTspice", "-b", str(netlist_path)]
            return run, False

        with patch("ltspice_mcp.server._run_process_simulation_with_cancel", side_effect=fake_run):
            server._ensure_job_worker()
            deadline = time.time() + 8.0
            while time.time() < deadline:
                status = server.jobStatus(queued["job_id"], include_run=False)
                if status["status"] in {"succeeded", "failed", "canceled"}:
                    break
                time.sleep(0.05)

        final_status = server.jobStatus(queued["job_id"], include_run=False)
        self.assertEqual(final_status["status"], "succeeded")
        self.assertEqual(Path(observed["path"]).expanduser().resolve(), run_path)
        self.assertIn("VERSION_A", observed["text"])
        self.assertNotIn("VERSION_B", observed["text"])

    def test_queue_priority_controls_dispatch_order(self) -> None:
        low_path = self.temp_dir / "low_priority.cir"
        high_path = self.temp_dir / "high_priority.cir"
        low_path.write_text("* low\n.end\n", encoding="utf-8")
        high_path.write_text("* high\n.end\n", encoding="utf-8")

        order: list[str] = []

        def fake_run(*, netlist_path: Path, ascii_raw: bool, timeout_seconds: int | None, cancel_requested):
            order.append(netlist_path.name)
            run = _make_run(self.temp_dir, run_id=f"priority-{len(order)}")
            return run, False

        with patch("ltspice_mcp.server._ensure_job_worker", return_value=None):
            low_job = server.queueSimulationJob(netlist_path=str(low_path), priority=200, max_retries=0)
            high_job = server.queueSimulationJob(netlist_path=str(high_path), priority=10, max_retries=0)

        with patch("ltspice_mcp.server._run_process_simulation_with_cancel", side_effect=fake_run):
            server._ensure_job_worker()
            deadline = time.time() + 8.0
            while time.time() < deadline:
                low_status = server.jobStatus(low_job["job_id"], include_run=False)["status"]
                high_status = server.jobStatus(high_job["job_id"], include_run=False)["status"]
                if low_status in {"succeeded", "failed", "canceled"} and high_status in {
                    "succeeded",
                    "failed",
                    "canceled",
                }:
                    break
                time.sleep(0.05)

        self.assertGreaterEqual(len(order), 2)
        self.assertEqual(order[0], "high_priority.cir")
        self.assertEqual(order[1], "low_priority.cir")

    def test_queue_retry_policy_retries_then_succeeds(self) -> None:
        retry_path = self.temp_dir / "retry_job.cir"
        retry_path.write_text("* retry\n.end\n", encoding="utf-8")
        calls = {"count": 0}

        def fake_run(*, netlist_path: Path, ascii_raw: bool, timeout_seconds: int | None, cancel_requested):
            calls["count"] += 1
            run = _make_run(self.temp_dir, run_id=f"retry-{calls['count']}")
            if calls["count"] == 1:
                run.return_code = 255
                run.issues = ["Intentional failure"]
            return run, False

        with patch("ltspice_mcp.server._run_process_simulation_with_cancel", side_effect=fake_run):
            queued = server.queueSimulationJob(
                netlist_path=str(retry_path),
                priority=50,
                max_retries=1,
            )
            deadline = time.time() + 8.0
            while time.time() < deadline:
                status = server.jobStatus(queued["job_id"], include_run=False)
                if status["status"] in {"succeeded", "failed", "canceled"}:
                    break
                time.sleep(0.05)

        final_status = server.jobStatus(queued["job_id"], include_run=False)
        self.assertEqual(final_status["status"], "succeeded")
        self.assertEqual(final_status["retry_count"], 1)
        self.assertEqual(calls["count"], 2)

    def test_queue_persists_across_reconfigure(self) -> None:
        netlist = self.temp_dir / "persist_job.cir"
        netlist.write_text("* persist\n.end\n", encoding="utf-8")
        with patch("ltspice_mcp.server._ensure_job_worker", return_value=None):
            queued = server.queueSimulationJob(
                netlist_path=str(netlist),
                priority=7,
                max_retries=3,
            )
            server._configure_runner(workdir=self.temp_dir, ltspice_binary=None, timeout=10)

        restored = server.jobStatus(queued["job_id"], include_run=False)
        self.assertEqual(restored["status"], "queued")
        self.assertEqual(restored["priority"], 7)
        self.assertEqual(restored["max_retries"], 3)
        self.assertIn("persisted queue state", str(restored.get("summary", "")).lower())

    def test_queue_history_persists_and_is_queryable(self) -> None:
        netlist = self.temp_dir / "history_job.cir"
        netlist.write_text("* history\n.end\n", encoding="utf-8")
        with patch("ltspice_mcp.server._ensure_job_worker", return_value=None):
            queued = server.queueSimulationJob(netlist_path=str(netlist), priority=5, max_retries=0)
        canceled = server.cancelJob(job_id=queued["job_id"])
        self.assertEqual(canceled["status"], "canceled")

        history = server.listJobHistory(limit=50)
        self.assertGreaterEqual(history["count"], 1)
        self.assertTrue(any(item["job_id"] == queued["job_id"] for item in history["jobs"]))

        server._configure_runner(workdir=self.temp_dir, ltspice_binary=None, timeout=10)
        history_after = server.listJobHistory(limit=50)
        self.assertTrue(any(item["job_id"] == queued["job_id"] for item in history_after["jobs"]))

    def test_list_jobs_include_history_flag_controls_terminal_visibility(self) -> None:
        netlist = self.temp_dir / "history_visibility.cir"
        netlist.write_text("* history visibility\n.end\n", encoding="utf-8")
        with patch("ltspice_mcp.server._ensure_job_worker", return_value=None):
            queued = server.queueSimulationJob(netlist_path=str(netlist), priority=5, max_retries=0)
        canceled = server.cancelJob(job_id=queued["job_id"])
        self.assertEqual(canceled["status"], "canceled")

        without_history = server.listJobs(limit=10, include_history=False)
        self.assertEqual(without_history["count"], 0)

        with_history = server.listJobs(limit=10, include_history=True)
        self.assertGreaterEqual(with_history["count"], 1)
        self.assertTrue(any(item["job_id"] == queued["job_id"] for item in with_history["jobs"]))

    def test_cancel_job_returns_terminal_noop_for_history_only_job(self) -> None:
        netlist = self.temp_dir / "history_only_job.cir"
        netlist.write_text("* history only\n.end\n", encoding="utf-8")
        with patch("ltspice_mcp.server._ensure_job_worker", return_value=None):
            queued = server.queueSimulationJob(netlist_path=str(netlist), priority=5, max_retries=0)
        canceled = server.cancelJob(job_id=queued["job_id"])
        self.assertEqual(canceled["status"], "canceled")
        self.assertIsNotNone(canceled["archived_at"])

        # Simulate a terminal job that exists only in archived history.
        with server._job_lock:
            server._jobs.pop(queued["job_id"], None)
            if queued["job_id"] in server._job_order:
                server._job_order.remove(queued["job_id"])
        history_only = server.cancelJob(job_id=queued["job_id"])
        self.assertTrue(history_only["from_history"])
        self.assertTrue(history_only["no_op"])
        self.assertEqual(history_only["reason"], "job_already_terminal")
        self.assertEqual(history_only["status"], "canceled")
        self.assertIsNotNone(history_only["archived_at"])

    def test_queue_rejects_invalid_priority_retry_and_timeout_inputs(self) -> None:
        netlist = self.temp_dir / "queue_validation.cir"
        netlist.write_text("* queue validation\n.end\n", encoding="utf-8")
        with patch("ltspice_mcp.server._ensure_job_worker", return_value=None):
            with self.assertRaisesRegex(ValueError, "priority must be >= 0"):
                server.queueSimulationJob(netlist_path=str(netlist), priority=-1)
            with self.assertRaisesRegex(ValueError, "max_retries must be >= 0"):
                server.queueSimulationJob(netlist_path=str(netlist), max_retries=-1)
            with self.assertRaisesRegex(ValueError, "timeout_seconds must be >= 1"):
                server.queueSimulationJob(netlist_path=str(netlist), timeout_seconds=0)
            with self.assertRaisesRegex(ValueError, "max_retries must be an integer, not boolean"):
                server.queueSimulationJob(netlist_path=str(netlist), max_retries=True)
            with self.assertRaisesRegex(ValueError, "priority must be an integer, not boolean"):
                server.queueSimulationJob(netlist_path=str(netlist), priority=False)

    def test_parse_meas_results_rejects_conflicting_selectors(self) -> None:
        log_path = self.temp_dir / "selector_conflict.log"
        log_path.write_text("gain: 1\n", encoding="utf-8")
        run = _make_run(self.temp_dir, run_id="selector-conflict-run")
        server._register_run(run)
        with self.assertRaisesRegex(ValueError, "accepts only one source selector"):
            server.parseMeasResults(run_id=run.run_id, log_path=str(log_path))

    def test_parse_meas_results_run_and_log_path_are_equivalent(self) -> None:
        run = _make_run(self.temp_dir, run_id="selector-eq-run")
        assert run.log_path is not None
        run.log_path.write_text("foo: v(in)=1\n", encoding="utf-8")
        server._register_run(run)

        via_run = server.parseMeasResults(run_id=run.run_id)
        via_path = server.parseMeasResults(log_path=str(run.log_path))
        self.assertEqual(via_run["measurements"], via_path["measurements"])
        self.assertEqual(via_run["count"], via_path["count"])

    def test_run_meas_automation_supports_short_form_measurements(self) -> None:
        netlist = self.temp_dir / "meas_short_form_base.cir"
        netlist.write_text(
            "\n".join(
                [
                    "* short-form measurement",
                    "V1 in 0 1",
                    ".op",
                    ".end",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        run = _make_run(self.temp_dir, run_id="meas-short-form-run")
        assert run.log_path is not None
        run.log_path.write_text("foo: v(in)=1\n", encoding="utf-8")
        with (
            patch("ltspice_mcp.server.simulateNetlistFile", return_value={"run_id": run.run_id, "succeeded": True}),
            patch("ltspice_mcp.server._resolve_run", return_value=run),
        ):
            payload = server.runMeasAutomation(
                netlist_path=str(netlist),
                measurements=["foo PARAM V(in)"],
            )
        self.assertEqual(payload["requested_measurements"], ["foo"])
        self.assertEqual(payload["measurements"]["requested_measurements"], ["foo"])
        self.assertEqual(payload["measurements"]["missing_requested_measurements"], [])
        self.assertTrue(payload["requested_measurements_succeeded"])

    def test_run_verification_plan_meas_works_with_short_form_results(self) -> None:
        run = _make_run(self.temp_dir, run_id="verify-short-form-run")
        assert run.log_path is not None
        run.log_path.write_text("foo: v(in)=1\n", encoding="utf-8")
        server._register_run(run)
        payload = server.runVerificationPlan(
            run_id=run.run_id,
            assertions=[{"id": "mfoo", "type": "meas", "name": "FOO", "min": 0.0}],
        )
        self.assertTrue(payload["overall_passed"])
        self.assertEqual(payload["checks"][0]["id"], "mfoo")
        self.assertTrue(payload["checks"][0]["passed"])
        self.assertAlmostEqual(payload["checks"][0]["value"], 1.0, places=9)

    def test_run_verification_plan_rejects_conflicting_or_unconstrained_inputs(self) -> None:
        run = _make_run(self.temp_dir, run_id="verify-conflict-run")
        server._register_run(run)
        netlist = self.temp_dir / "verify_conflict.cir"
        netlist.write_text("* verify conflict\n.end\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "accepts only one source selector"):
            server.runVerificationPlan(
                run_id=run.run_id,
                netlist_path=str(netlist),
                assertions=[{"type": "meas", "name": "foo", "min": 0.0}],
            )
        with self.assertRaisesRegex(ValueError, "must include at least one acceptance criterion"):
            server.runVerificationPlan(
                run_id=run.run_id,
                assertions=[{"type": "meas", "name": "foo"}],
            )
        with self.assertRaisesRegex(ValueError, "does not support `run_id` together with `measurements`"):
            server.runVerificationPlan(
                run_id=run.run_id,
                measurements=["foo PARAM V(in)"],
                assertions=[{"type": "meas", "name": "foo", "min": 0.0}],
            )

    def test_get_plot_names_rejects_conflicting_selectors_and_reports_missing_raw(self) -> None:
        raw_path = self.temp_dir / "plot_names.raw"
        raw_path.write_text("placeholder\n", encoding="utf-8")
        run = _make_run(self.temp_dir, run_id="plot-names-run")
        run.raw_files = []
        server._register_run(run)

        with self.assertRaisesRegex(ValueError, "accepts only one source selector"):
            server.getPlotNames(run_id=run.run_id, raw_path=str(raw_path))
        with self.assertRaisesRegex(ValueError, "has no RAW files"):
            server.getPlotNames(run_id=run.run_id)

    def test_get_plot_names_supports_ascii_complex_raw(self) -> None:
        raw_path = self.temp_dir / "ascii_complex_ac.raw"
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
                [10.0 + 0j, 100.0 + 0j, 1000.0 + 0j],
                [1.0 + 0.0j, 0.5 - 0.5j, 0.1 - 0.9j],
            ],
            steps=[],
        )
        server._write_raw_dataset_ascii(dataset, raw_path)
        payload = server.getPlotNames(raw_path=str(raw_path))
        self.assertEqual(payload["run_id"], None)
        self.assertEqual(len(payload["plots"]), 1)
        self.assertEqual(payload["plots"][0]["plot_name"], "AC Analysis")

    def test_run_sweep_study_rejects_conflicting_step_specs_and_existing_step(self) -> None:
        netlist = self.temp_dir / "step_conflict.cir"
        netlist.write_text(
            "\n".join(
                [
                    "* sweep conflict",
                    ".step param A list 1k 2k",
                    "R1 in out {A}",
                    ".op",
                    ".end",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "does not support source netlists that already contain \\.step"):
            server.runSweepStudy(
                parameter="B",
                mode="step",
                netlist_path=str(netlist),
                values=["1k", "2k"],
            )

        plain = self.temp_dir / "step_conflict_plain.cir"
        plain.write_text("* plain\nR1 in out 1k\n.end\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Provide either values OR start/stop/step"):
            server.runSweepStudy(
                parameter="RVAL",
                mode="step",
                netlist_path=str(plain),
                values=["1k", "2k"],
                start="1k",
                stop="2k",
                step="1k",
            )

    def test_generate_verify_and_clean_circuit_orchestration(self) -> None:
        asc = self.temp_dir / "orchestration.asc"
        asc.write_text("Version 4\nSHEET 1 200 200\n", encoding="utf-8")
        cleaned = self.temp_dir / "orchestration_clean.asc"
        cleaned.write_text("Version 4\nSHEET 1 200 200\n", encoding="utf-8")
        with (
            patch(
                "ltspice_mcp.server.createIntentCircuit",
                return_value={"intent": "rc_lowpass", "asc_path": str(asc), "schematic_validation": {"valid": True}},
            ),
            patch("ltspice_mcp.server.lintSchematic") as lint_mock,
            patch(
                "ltspice_mcp.server.simulateSchematicFile",
                return_value={"run_id": "run-orch", "succeeded": True},
            ),
            patch(
                "ltspice_mcp.server.runVerificationPlan",
                return_value={"overall_passed": True, "checks": []},
            ),
            patch(
                "ltspice_mcp.server.autoCleanSchematicLayout",
                return_value={"target_path": str(cleaned), "after": {"score": 99.0}},
            ),
            patch(
                "ltspice_mcp.server.inspectSchematicVisualQuality",
                return_value={"quality": {"score": 99.0}, "render": None},
            ),
        ):
            lint_mock.side_effect = [
                {"valid": True, "errors": [], "warnings": []},
                {"valid": True, "errors": [], "warnings": []},
            ]
            payload = server.generateVerifyAndCleanCircuit(
                intent="rc_lowpass",
                parameters={"r_value": "1k", "c_value": "100n"},
                auto_clean=True,
                render_after_clean=False,
            )
        self.assertTrue(payload["overall_passed"])
        self.assertEqual(payload["final_schematic_path"], str(cleaned))
        self.assertEqual(lint_mock.call_count, 2)

    def test_generate_verify_and_clean_uses_asc_mode_when_measurements_requested(self) -> None:
        asc = self.temp_dir / "orchestration_meas.asc"
        asc.write_text("Version 4\nSHEET 1 200 200\n", encoding="utf-8")
        with (
            patch(
                "ltspice_mcp.server.createIntentCircuit",
                return_value={"intent": "rc_lowpass", "asc_path": str(asc), "schematic_validation": {"valid": True}},
            ),
            patch("ltspice_mcp.server.lintSchematic", return_value={"valid": True, "errors": [], "warnings": []}),
            patch(
                "ltspice_mcp.server.simulateSchematicFile",
                return_value={"run_id": "run-orch-meas", "succeeded": True},
            ),
            patch("ltspice_mcp.server.autoCleanSchematicLayout", return_value=None),
            patch(
                "ltspice_mcp.server.inspectSchematicVisualQuality",
                return_value={"quality": {"score": 99.0}, "render": None},
            ),
            patch("ltspice_mcp.server.runVerificationPlan") as verify_mock,
        ):
            verify_mock.return_value = {"overall_passed": True, "checks": []}
            payload = server.generateVerifyAndCleanCircuit(
                intent="rc_lowpass",
                measurements=["foo PARAM V(in)"],
                auto_clean=False,
                render_after_clean=False,
            )
        self.assertTrue(payload["overall_passed"])
        verify_kwargs = verify_mock.call_args.kwargs
        self.assertEqual(verify_kwargs.get("asc_path"), str(asc))
        self.assertNotIn("run_id", verify_kwargs)


if __name__ == "__main__":
    unittest.main()
