from __future__ import annotations

import tempfile
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

    def test_run_meas_automation_injects_meas_netlist(self) -> None:
        netlist = self.temp_dir / "base.cir"
        netlist.write_text("* base\nR1 in out 1k\n.end\n", encoding="utf-8")
        with (
            patch("ltspice_mcp.server.simulateNetlistFile") as simulate_mock,
            patch("ltspice_mcp.server.parseMeasResults") as parse_mock,
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

        inspect_payload = server.inspectSchematicVisualQuality(
            asc_path=str(cleaned),
            render=False,
        )
        self.assertIn("quality", inspect_payload)
        self.assertGreaterEqual(inspect_payload["quality"]["component_count"], 2)

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
            queued = server.queueSimulationJob(netlist_path=str(netlist))
        self.assertEqual(queued["status"], "queued")
        job_id = queued["job_id"]

        listed = server.listJobs(limit=10)
        self.assertEqual(listed["count"], 1)
        self.assertEqual(listed["jobs"][0]["job_id"], job_id)

        status = server.jobStatus(job_id=job_id, include_run=False)
        self.assertEqual(status["status"], "queued")

        canceled = server.cancelJob(job_id=job_id)
        self.assertEqual(canceled["status"], "canceled")


if __name__ == "__main__":
    unittest.main()
