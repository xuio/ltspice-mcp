# Changelog

## v1.30.0 - 2026-02-25

### Added
- Verification and study automation MCP tools:
  - `runVerificationPlan`
  - `runMeasAutomation`
  - `parseMeasResults`
  - `runSweepStudy`
- Schematic quality tools:
  - `autoCleanSchematicLayout`
  - `inspectSchematicVisualQuality`
- Daemon health umbrella tool:
  - `daemonDoctor`
- Queue/cancellation MCP tools:
  - `queueSimulationJob`
  - `listJobs`
  - `jobStatus`
  - `cancelJob`
- New test module `tests/test_verification_queue_tools.py` covering the features above.

### Improved
- Queue worker shutdown is now deadlock-safe and is reset automatically during `_configure_runner`.
- README and AGENT_README now include the new verification, sweep, visual QA, doctor, and queue workflows.

## v1.29.0 - 2026-02-25

### Added
- MCP resource `docs://agent-readme` to serve `AGENT_README.md` directly through MCP.
- MCP tool `readAgentGuide` with section selection and text search for interactive agent guidance.

### Improved
- Reworked `README.md` into a quickstart-first layout with clearer sections and reduced duplication.
- Reworked `AGENT_README.md` into a concise operational playbook focused on the real agent workflow.

### Tests
- Added regression coverage for `readAgentGuide` and MCP resource registration/readability for `docs://agent-readme`.

## v1.28.0 - 2026-02-25

### Added
- New MCP diagnostics tools for daemon operations:
  - `tailDaemonLog`
  - `getRecentErrors`
  - `getCaptureHealth`
- In-memory ScreenCapture/LTspice capture health tracking with structured event summaries.
- UTF-8 normalized LTspice log sidecars (`*.log.utf8.txt`) recorded per simulation run.

### Improved
- Streamable HTTP daemon startup now uses explicit uvicorn wiring with graceful shutdown timeout and built-in noise filters.
- Daemon logs now suppress known OAuth discovery probe 404s and transient `GET /mcp` 400 probe noise by default.
- Suppressed known uvicorn shutdown false-positive (`ASGI callable returned without completing response`) in daemon logs.
- `runSimulation`, `simulateNetlist`, `simulateNetlistFile`, and `simulateSchematicFile` now auto-retry once with convergence options when convergence failure is detected.
- LTspice window close flow now performs post-close verification using exact selectors (window id/title), reducing false close positives.

### Tests
- Added regression coverage for close verification, convergence auto-retry behavior, uvicorn noise filtering, daemon log diagnostics tools, and UTF-8 log sidecar generation.

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
