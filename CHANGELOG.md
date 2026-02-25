# Changelog

## v1.27.0 - 2026-02-25

### Added
- Multi-pin netlist-to-schematic support in smart/legacy auto-layout for active elements (`X`, `Q`, `M`) with symbol resolution.
- `lintSchematic` MCP tool for structural schematic linting (pin connectivity, dangling wires, duplicate/missing `InstName`).
- `getToolTelemetry` and `resetToolTelemetry` MCP tools for rolling per-tool performance stats.
- Model issue scanning enhancements with optional search-path indexing and ranked resolution suggestions.

### Improved
- `autoDebugSchematic` now includes convergence fix actions (`.options` insertion) and returns a confidence score with factor breakdown.
- Model import now reports discovered `.model` names in addition to `.subckt` names.
- Smoke tests and unit tests expanded for multi-pin layout, linting, model suggestions, and telemetry.

### Notes
- Existing tool names remain stable; new optional parameters were added in a backward-compatible way.
