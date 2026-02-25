from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RawVariable:
    index: int
    name: str
    kind: str

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "name": self.name, "kind": self.kind}


@dataclass(slots=True)
class RawStep:
    index: int
    start: int
    end: int
    label: str | None = None

    @property
    def points(self) -> int:
        return max(0, self.end - self.start)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "step_index": self.index,
            "start_index": self.start,
            "end_index": self.end,
            "points": self.points,
        }
        if self.label:
            payload["label"] = self.label
        return payload


@dataclass(slots=True)
class RawDataset:
    path: Path
    plot_name: str
    flags: set[str]
    metadata: dict[str, str]
    variables: list[RawVariable]
    values: list[list[complex]]
    steps: list[RawStep] = field(default_factory=list)

    @property
    def points(self) -> int:
        if not self.values:
            return 0
        return len(self.values[0])

    @property
    def step_count(self) -> int:
        return len(self.steps) if self.steps else 1

    @property
    def is_stepped(self) -> bool:
        return self.step_count > 1 or "stepped" in self.flags

    @property
    def scale_name(self) -> str:
        if self.has_natural_scale() and self.variables:
            return self.variables[0].name
        return "index"

    def has_natural_scale(self) -> bool:
        if not self.variables:
            return False
        return self.variables[0].kind.lower() in {"time", "frequency"}

    def _resolve_step(self, step_index: int | None) -> RawStep:
        if not self.steps:
            return RawStep(index=0, start=0, end=self.points, label=None)
        if step_index is None:
            return self.steps[0]
        if step_index < 0 or step_index >= len(self.steps):
            raise IndexError(f"step_index {step_index} is out of range [0, {len(self.steps) - 1}]")
        return self.steps[step_index]

    def scale_values(self, step_index: int | None = None) -> list[float]:
        step = self._resolve_step(step_index)
        if self.has_natural_scale() and self.values:
            return [value.real for value in self.values[0][step.start : step.end]]
        return [float(index) for index in range(step.points)]

    def get_vector(self, name: str, step_index: int | None = None) -> list[complex]:
        step = self._resolve_step(step_index)
        for variable in self.variables:
            if variable.name == name:
                return self.values[variable.index][step.start : step.end]
        raise KeyError(f"Unknown vector '{name}'")


@dataclass(slots=True)
class SimulationDiagnostic:
    category: str
    severity: str
    message: str
    suggestion: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "category": self.category,
            "severity": self.severity,
            "message": self.message,
        }
        if self.suggestion:
            payload["suggestion"] = self.suggestion
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SimulationDiagnostic":
        return cls(
            category=str(payload.get("category", "unknown")),
            severity=str(payload.get("severity", "info")),
            message=str(payload.get("message", "")),
            suggestion=payload.get("suggestion"),
        )


@dataclass(slots=True)
class SimulationRun:
    run_id: str
    netlist_path: Path
    command: list[str]
    ltspice_executable: Path
    started_at: str
    duration_seconds: float
    return_code: int
    stdout: str
    stderr: str
    log_path: Path | None
    raw_files: list[Path]
    artifacts: list[Path]
    log_utf8_path: Path | None = None
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: list[SimulationDiagnostic] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.return_code == 0 and not self.issues

    def as_dict(self, include_output: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "succeeded": self.succeeded,
            "return_code": self.return_code,
            "started_at": self.started_at,
            "duration_seconds": round(self.duration_seconds, 3),
            "netlist_path": str(self.netlist_path),
            "command": self.command,
            "ltspice_executable": str(self.ltspice_executable),
            "log_path": str(self.log_path) if self.log_path else None,
            "log_utf8_path": str(self.log_utf8_path) if self.log_utf8_path else None,
            "raw_files": [str(path) for path in self.raw_files],
            "artifacts": [str(path) for path in self.artifacts],
            "issues": self.issues,
            "warnings": self.warnings,
            "diagnostics": [diagnostic.as_dict() for diagnostic in self.diagnostics],
        }
        if include_output:
            payload["stdout"] = self.stdout
            payload["stderr"] = self.stderr
        return payload

    def to_storage_dict(self) -> dict[str, Any]:
        payload = self.as_dict(include_output=False)
        return payload

    @classmethod
    def from_storage_dict(cls, payload: dict[str, Any]) -> "SimulationRun":
        return cls(
            run_id=str(payload["run_id"]),
            netlist_path=Path(payload["netlist_path"]).expanduser().resolve(),
            command=[str(item) for item in payload.get("command", [])],
            ltspice_executable=Path(payload["ltspice_executable"]).expanduser().resolve(),
            started_at=str(payload.get("started_at", "")),
            duration_seconds=float(payload.get("duration_seconds", 0.0)),
            return_code=int(payload.get("return_code", 0)),
            stdout="",
            stderr="",
            log_path=(
                Path(payload["log_path"]).expanduser().resolve()
                if payload.get("log_path")
                else None
            ),
            log_utf8_path=(
                Path(payload["log_utf8_path"]).expanduser().resolve()
                if payload.get("log_utf8_path")
                else None
            ),
            raw_files=[
                Path(item).expanduser().resolve() for item in payload.get("raw_files", [])
            ],
            artifacts=[
                Path(item).expanduser().resolve() for item in payload.get("artifacts", [])
            ],
            issues=[str(item) for item in payload.get("issues", [])],
            warnings=[str(item) for item in payload.get("warnings", [])],
            diagnostics=[
                SimulationDiagnostic.from_dict(item)
                for item in payload.get("diagnostics", [])
                if isinstance(item, dict)
            ],
        )
