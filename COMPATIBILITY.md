# Compatibility Notes

## Release
- Current: `v1.27.0`

## Runtime
- Python `>=3.11`
- macOS LTspice installation required for real LTspice rendering and simulation execution.

## Backward compatibility
- Existing MCP tools and core parameters remain available.
- New optional parameters:
  - `scanModelIssues`: `search_paths`, `suggest_matches`, `max_scan_files`, `max_suggestions_per_issue`
  - `autoDebugSchematic`: `auto_fix_convergence`, `model_search_paths`
- New MCP tools:
  - `lintSchematic`
  - `getToolTelemetry`
  - `resetToolTelemetry`

## Behavioral updates
- Netlist auto-layout now attempts multi-pin symbol placement for `X`, `Q`, `M` elements when symbol resolution succeeds.
- `autoDebugSchematic` now reports a `confidence` object and can inject convergence options into sidecar netlists.
- Tool telemetry is rolling-window based (`LTSPICE_MCP_TELEMETRY_WINDOW`, default `200` samples/tool).
