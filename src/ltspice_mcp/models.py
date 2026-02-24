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
class RawDataset:
    path: Path
    plot_name: str
    flags: set[str]
    metadata: dict[str, str]
    variables: list[RawVariable]
    values: list[list[complex]]

    @property
    def points(self) -> int:
        if not self.values:
            return 0
        return len(self.values[0])

    @property
    def scale_name(self) -> str:
        if self.has_natural_scale() and self.variables:
            return self.variables[0].name
        return "index"

    def has_natural_scale(self) -> bool:
        if not self.variables:
            return False
        return self.variables[0].kind.lower() in {"time", "frequency"}

    def scale_values(self) -> list[float]:
        if self.has_natural_scale() and self.values:
            return [value.real for value in self.values[0]]
        return [float(index) for index in range(self.points)]

    def get_vector(self, name: str) -> list[complex]:
        for variable in self.variables:
            if variable.name == name:
                return self.values[variable.index]
        raise KeyError(f"Unknown vector '{name}'")


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
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

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
            "raw_files": [str(path) for path in self.raw_files],
            "artifacts": [str(path) for path in self.artifacts],
            "issues": self.issues,
            "warnings": self.warnings,
        }
        if include_output:
            payload["stdout"] = self.stdout
            payload["stderr"] = self.stderr
        return payload
