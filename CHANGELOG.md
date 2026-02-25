# Changelog

## v1.27.1 - 2026-02-25

### Added
- `resolveSchematicSimulationTarget` MCP tool to explain sidecar requirements and the effective batch simulation target for schematics.
- New streamable-http integration test suite (`tests/test_streamable_http_integration.py`) covering tool schema visibility, retired-argument rejection, sidecar guidance, and optional live render checks.
- Tool contract tests for image-tool output schema handling and strict argument validation.

### Improved
- Fixed image tool output handling under MCP so `renderLtspiceSymbolImage`, `renderLtspiceSchematicImage`, and `renderLtspicePlotImage` return `CallToolResult` without Pydantic output-schema misvalidation.
- `simulateSchematicFile` now returns a clear, explicit macOS error when no sidecar netlist exists for `.asc` inputs.
- Plot tool schemas now expose enums for `mode`, `y_mode`, and `pane_layout`.
- Unknown MCP tool arguments are now rejected (`extra=forbid`) instead of silently ignored.
- `smoke_test_mcp.py` now supports both `stdio` and `streamable-http` transport modes and validates `resolveSchematicSimulationTarget`.

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
