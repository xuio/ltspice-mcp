from __future__ import annotations

import json
import os
import platform
import re
import sys
import tempfile
import unittest
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

try:
    import anyio
except Exception:  # pragma: no cover - optional dependency for real integration runs
    anyio = None  # type: ignore[assignment]

from ltspice_mcp.ltspice import find_ltspice_executable

try:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
except Exception:  # pragma: no cover - optional dependency for real integration runs
    ClientSession = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment]
    stdio_client = None  # type: ignore[assignment]


_NUMERIC_TOKEN_RE = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?(?:[A-Za-z]+)?"
)
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
_AUX_HEADER_TOKENS = {
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


def _real_ax_tests_enabled() -> bool:
    value = os.getenv("LTSPICE_MCP_RUN_REAL_AX", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


def _extract_ui_line(ui_text: str, name: str) -> str:
    pattern = re.compile(rf"(?im)^\s*{re.escape(name)}\s*(?::|=)\s*(?P<rhs>.+?)\s*$")
    match = pattern.search(ui_text)
    if not match:
        raise AssertionError(f"UI line for measurement '{name}' was not found.")
    return match.group("rhs").strip()


def _extract_ui_value(rhs: str, *, mode: str) -> tuple[float, str]:
    if mode == "when":
        at_match = re.search(
            r"\bat\s*=?\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?(?:[A-Za-z]+)?)\b",
            rhs,
            re.IGNORECASE,
        )
        if not at_match:
            raise AssertionError(f"Expected AT value in WHEN measurement line: {rhs}")
        token = at_match.group("value")
        parsed = _parse_spice_number_token(token)
        if parsed is None:
            raise AssertionError(f"Failed to parse UI WHEN token '{token}' in line: {rhs}")
        return parsed, token

    candidate = rhs.split("=", 1)[1] if "=" in rhs else rhs
    for token in _NUMERIC_TOKEN_RE.findall(candidate):
        parsed = _parse_spice_number_token(token)
        if parsed is not None:
            return parsed, token
    raise AssertionError(f"Failed to extract numeric value from UI measurement line: {rhs}")


def _extract_measurement_section(ui_text: str, name: str) -> str:
    start_match = re.search(
        rf"(?im)^\s*Measurement:\s*{re.escape(name)}\s*$",
        ui_text,
    )
    if not start_match:
        raise AssertionError(f"UI section 'Measurement: {name}' was not found.")
    next_match = re.search(r"(?im)^\s*Measurement:\s*[A-Za-z_][\w.$-]*\s*$", ui_text[start_match.end() :])
    if next_match:
        return ui_text[start_match.end() : start_match.end() + next_match.start()]
    return ui_text[start_match.end() :]


def _extract_ui_step_rows(section_text: str, measurement_name: str) -> list[dict[str, Any]]:
    lines = [line for line in section_text.splitlines() if line.strip()]
    header_tokens: list[str] | None = None
    for line in lines:
        tokens = [token for token in re.split(r"\s+", line.strip()) if token]
        if tokens and tokens[0].lower() == "step":
            header_tokens = tokens
            break
    if not header_tokens:
        raise AssertionError(f"Could not locate stepped table header for '{measurement_name}'.")

    lowered_header = [token.lower() for token in header_tokens]
    step_col = lowered_header.index("step") if "step" in lowered_header else None
    if measurement_name.lower() in lowered_header:
        value_col = lowered_header.index(measurement_name.lower())
    else:
        candidates = [
            idx
            for idx, token in enumerate(lowered_header)
            if token not in _AUX_HEADER_TOKENS and idx != (step_col if step_col is not None else -1)
        ]
        if not candidates:
            raise AssertionError(f"Could not infer value column for '{measurement_name}'.")
        value_col = candidates[0]

    rows: list[dict[str, Any]] = []
    header_found = False
    for line in lines:
        tokens = [token for token in re.split(r"\s+", line.strip()) if token]
        if not tokens:
            continue
        lowered = [token.lower() for token in tokens]
        if not header_found:
            if tokens == header_tokens:
                header_found = True
            continue
        if len(tokens) <= value_col:
            continue
        step_value: int | None = None
        if step_col is not None:
            if len(tokens) <= step_col:
                continue
            parsed_step = _parse_spice_number_token(tokens[step_col])
            if parsed_step is None:
                continue
            step_value = int(parsed_step)
        value_token = tokens[value_col]
        value = _parse_spice_number_token(value_token)
        if value is None:
            for candidate_token in _NUMERIC_TOKEN_RE.findall(value_token):
                value = _parse_spice_number_token(candidate_token)
                if value is not None:
                    value_token = candidate_token
                    break
        if value is None:
            continue
        entry: dict[str, Any] = {"value": value, "value_text": value_token}
        if step_value is not None:
            entry["step"] = step_value
        rows.append(entry)
    if not rows:
        raise AssertionError(f"No stepped rows parsed for '{measurement_name}'.")
    return rows


def _assert_close_numeric(test_case: unittest.TestCase, lhs: float, rhs: float) -> None:
    tolerance = max(1e-12, abs(rhs) * 1e-9)
    test_case.assertLessEqual(abs(lhs - rhs), tolerance, f"Mismatch: lhs={lhs} rhs={rhs}")


@unittest.skipUnless(platform.system() == "Darwin", "Accessibility parity tests require macOS")
class TestMeasUiAccessibilityParityReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not _real_ax_tests_enabled():
            raise unittest.SkipTest(
                "Set LTSPICE_MCP_RUN_REAL_AX=1 to run real LTspice AX measurement parity tests."
            )
        if ClientSession is None or StdioServerParameters is None or stdio_client is None:
            raise unittest.SkipTest("MCP python client is not available in this environment.")
        if anyio is None:
            raise unittest.SkipTest("anyio is not available in this environment.")
        ltspice_binary = os.getenv("LTSPICE_BINARY") or str(find_ltspice_executable() or "")
        if not ltspice_binary:
            raise unittest.SkipTest("LTspice binary not found.")
        cls.ltspice_binary = ltspice_binary
        cls.workdir = Path(tempfile.mkdtemp(prefix="ltspice_real_ax_parity_")).resolve()

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

    async def _run_ui_parity_check(
        self,
        session: ClientSession,
        *,
        circuit_name: str,
        netlist: str,
        measurement_modes: dict[str, str],
        stepped_measurements: list[str] | None = None,
    ) -> None:
        sim = _extract_call_result(
            await session.call_tool(
                "simulateNetlist",
                {"netlist_content": netlist, "circuit_name": circuit_name},
            )
        )
        self.assertTrue(sim.get("succeeded"), sim)
        run_id = str(sim["run_id"])

        parsed = _extract_call_result(await session.call_tool("parseMeasResults", {"run_id": run_id}))
        self.assertGreater(int(parsed.get("count", 0)), 0, parsed)
        log_path = Path(str(parsed.get("log_path") or ""))
        self.assertTrue(log_path.exists(), parsed)
        ui_text = log_path.read_text(encoding="utf-8", errors="ignore")
        self.assertTrue(ui_text.strip(), {"log_path": str(log_path)})

        parsed_values = parsed.get("measurements", {})
        parsed_text = parsed.get("measurements_text", {})
        for name, mode in measurement_modes.items():
            self.assertIn(name, parsed_values, f"Parser missing measurement '{name}'")
            rhs = _extract_ui_line(ui_text, name)
            ui_value, ui_token = _extract_ui_value(rhs, mode=mode)
            parser_value = float(parsed_values[name])
            _assert_close_numeric(self, parser_value, ui_value)
            parser_token = str(parsed_text.get(name) or "").strip()
            self.assertEqual(
                parser_token.lower(),
                ui_token.lower(),
                f"Token mismatch for {name}: parser={parser_token} ui={ui_token}",
            )

        for measurement_name in stepped_measurements or []:
            self.assertIn(measurement_name, parsed_values, f"Parser missing stepped '{measurement_name}'")
            section = _extract_measurement_section(ui_text, measurement_name)
            ui_steps = _extract_ui_step_rows(section, measurement_name)
            parser_steps = list(parsed.get("measurement_steps", {}).get(measurement_name, []))
            self.assertEqual(len(parser_steps), len(ui_steps), f"Step count mismatch for '{measurement_name}'")
            for parser_entry, ui_entry in zip(parser_steps, ui_steps):
                if "step" in ui_entry:
                    self.assertEqual(int(parser_entry.get("step")), int(ui_entry["step"]))
                _assert_close_numeric(self, float(parser_entry["value"]), float(ui_entry["value"]))
                self.assertEqual(
                    str(parser_entry.get("value_text", "")).lower(),
                    str(ui_entry.get("value_text", "")).lower(),
                )
            _assert_close_numeric(self, float(parsed_values[measurement_name]), float(ui_steps[-1]["value"]))

    def test_ui_parity_transient_measurements(self) -> None:
        async def _run(session: ClientSession) -> None:
            netlist = (
                "* ui parity transient\n"
                "V1 in 0 PULSE(0 1 0 1u 1u 5m 10m)\n"
                "R1 in out 1k\n"
                "C1 out 0 1u\n"
                ".tran 0 25m 0 10u\n"
                ".meas tran v_max MAX V(out)\n"
                ".meas tran v_min MIN V(out)\n"
                ".meas tran v_pp PP V(out)\n"
                ".meas tran rise_t TRIG V(out) VAL=0.1 RISE=1 TARG V(out) VAL=0.9 RISE=1\n"
                ".meas tran fall_t TRIG V(out) VAL=0.9 FALL=1 TARG V(out) VAL=0.1 FALL=1\n"
                ".meas tran t_half_when WHEN V(out)=0.5 RISE=1\n"
                ".meas tran v_at_5m FIND V(out) AT=5m\n"
                ".meas tran avg_0_8m AVG V(out) FROM 0 TO 8m\n"
                ".meas tran integ_0_8m INTEG V(out) FROM 0 TO 8m\n"
                ".end\n"
            )
            await self._run_ui_parity_check(
                session,
                circuit_name="ui_parity_tran",
                netlist=netlist,
                measurement_modes={
                    "v_max": "standard",
                    "v_min": "standard",
                    "v_pp": "standard",
                    "rise_t": "standard",
                    "fall_t": "standard",
                    "t_half_when": "when",
                    "v_at_5m": "standard",
                    "avg_0_8m": "standard",
                    "integ_0_8m": "standard",
                },
            )

        anyio.run(self._with_session, _run)

    def test_ui_parity_ac_measurements(self) -> None:
        async def _run(session: ClientSession) -> None:
            netlist = (
                "* ui parity ac\n"
                "V1 in 0 AC 1\n"
                "R1 in out 1k\n"
                "C1 out 0 1u\n"
                ".ac dec 40 10 100k\n"
                ".meas ac gain_db_1k FIND db(V(out)) AT=1k\n"
                ".meas ac phase_1k FIND ph(V(out)) AT=1k\n"
                ".meas ac mag_max MAX mag(V(out))\n"
                ".meas ac mag_min MIN mag(V(out))\n"
                ".meas ac f_when_m3db WHEN db(V(out))=-3 FALL=1\n"
                ".end\n"
            )
            await self._run_ui_parity_check(
                session,
                circuit_name="ui_parity_ac",
                netlist=netlist,
                measurement_modes={
                    "gain_db_1k": "standard",
                    "phase_1k": "standard",
                    "mag_max": "standard",
                    "mag_min": "standard",
                    "f_when_m3db": "when",
                },
            )

        anyio.run(self._with_session, _run)

    def test_ui_parity_stepped_measurements(self) -> None:
        async def _run(session: ClientSession) -> None:
            netlist = (
                "* ui parity stepped\n"
                ".param rval=1k\n"
                "V1 in 0 AC 1\n"
                "R1 in out {rval}\n"
                "C1 out 0 1u\n"
                ".step param rval list 1k 2k 5k\n"
                ".ac dec 30 10 100k\n"
                ".meas ac gain_step FIND db(V(out)) AT=1k\n"
                ".meas ac f3_step WHEN db(V(out))=-3 FALL=1\n"
                ".end\n"
            )
            await self._run_ui_parity_check(
                session,
                circuit_name="ui_parity_step",
                netlist=netlist,
                measurement_modes={},
                stepped_measurements=["gain_step", "f3_step"],
            )

        anyio.run(self._with_session, _run)


if __name__ == "__main__":
    unittest.main()
