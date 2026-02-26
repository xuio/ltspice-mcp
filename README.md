# ltspice-mcp (macOS)

MCP server for LTspice on macOS with:
- simulation execution,
- schematic generation/lint/debug,
- native LTspice screenshot rendering for symbols, schematics, and plots,
- RAW data queries and measurement tools.

Inspired by:
- [gtnoble/ngspice-mcp](https://github.com/gtnoble/ngspice-mcp)
- [luc-me/ltspiceMCP](https://github.com/luc-me/ltspiceMCP)

For agent-specific operating guidance, see [AGENT_README.md](AGENT_README.md).

## Quick Start

### 1) Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2) Start daemon (recommended)

```bash
./scripts/ltspice_mcp_daemon.sh start
```

Default endpoint:
- `http://127.0.0.1:8765/mcp`

### 3) First-time permissions (required for screenshots/window control)

```bash
./scripts/ltspice_mcp_daemon.sh trigger-initial-permissions
./scripts/ltspice_mcp_daemon.sh check-accessibility
```

### 4) Run smoke test

```bash
python3 smoke_test_mcp.py \
  --transport streamable-http \
  --server-url http://127.0.0.1:8765/mcp
```

## Run Modes

- `stdio`: spawned subprocess mode.
- `sse`: HTTP + SSE mode.
- `streamable-http`: long-lived HTTP MCP mode (recommended for multi-client stability).

`streamable-http` now defaults to compatibility mode:
- JSON responses enabled.
- Stateless HTTP enabled (no sticky session requirement).

This improves interoperability with MCP clients that do not reliably persist
session headers across requests.

Direct run examples:

```bash
ltspice-mcp --transport stdio
```

```bash
ltspice-mcp \
  --daemon-http \
  --host 127.0.0.1 \
  --port 8765 \
  --http-path /mcp
```

## What It Provides

- LTspice executable discovery (`LTSPICE_BINARY` override).
- Netlist simulation tools with automatic one-shot convergence retry.
- Schematic generation from explicit specs, netlists, and templates.
- Schematic validation and structural linting.
- Iterative schematic auto-debug (`autoDebugSchematic`).
- LTspice symbol library inspection (`lib.zip`).
- Native LTspice rendering for symbol/schematic/plot images.
- Plot control via deterministic `.plt` generation.
- RAW vector access and analysis:
  - bandwidth,
  - gain/phase margin,
  - rise/fall time,
  - settling time.
- Verification and automation:
  - `.meas` generation + parsing (`runMeasAutomation`, `parseMeasResults`),
  - one-shot assertion plans (`runVerificationPlan`) with assertion groups (`all_of`/`any_of`) and tolerance checks (`target`, `rel_tol_pct`, `abs_tol`),
  - stepped/Monte-Carlo studies (`runSweepStudy`).
- Schematic visual quality tools:
  - quality inspection (`inspectSchematicVisualQuality`),
  - auto-cleaner (`autoCleanSchematicLayout`),
  - strict style lint in `lintSchematic(strict_style=true, ...)`.
- End-to-end orchestration:
  - `generateVerifyAndCleanCircuit` (create → lint → simulate → verify → clean → inspect).
- Queue-based simulation control:
  - `queueSimulationJob`, `listJobs`, `jobStatus`, `cancelJob`,
  - persisted queue state across daemon restart,
  - priority scheduling and per-job retry policy.
  - persisted terminal job history with retention (`listJobHistory`).
- Daemon diagnostics:
  - `tailDaemonLog`,
  - `getRecentErrors`,
  - `getCaptureHealth`.
  - `daemonDoctor`.
- Run persistence and artifacts with UTF-8 log sidecars (`*.log.utf8.txt`).

## MCP Tool Overview

### Status and setup
- `getLtspiceStatus`
- `getLtspiceUiStatus`
- `setLtspiceUiEnabled`
- `setSchematicUiSingleWindow`

### Rendering and UI
- `renderLtspiceSymbolImage`
- `renderLtspiceSchematicImage`
- `renderLtspicePlotImage`
- `renderLtspicePlotPresetImage`
- `startLtspiceRenderSession`
- `endLtspiceRenderSession`
- `openLtspiceUi`
- `closeLtspiceWindow`

### Schematic generation/debug
- `createSchematic`
- `createSchematicFromNetlist`
- `createSchematicFromTemplate`
- `listSchematicTemplates`
- `listIntentCircuitTemplates`
- `createIntentCircuit`
- `syncSchematicFromNetlistFile`
- `watchSchematicFromNetlistFile`
- `validateSchematic`
- `lintSchematic`
- `resolveSchematicSimulationTarget`
- `simulateSchematicFile`
- `autoDebugSchematic`
- `autoCleanSchematicLayout`
- `inspectSchematicVisualQuality`
- `generateVerifyAndCleanCircuit`

### Simulation
- `loadCircuit`
- `loadNetlistFromFile`
- `runSimulation`
- `simulateNetlist`
- `simulateNetlistFile`
- `queueSimulationJob`
- `listJobs`
- `jobStatus`
- `cancelJob`
- `listJobHistory`

### Plot/data/analysis
- `getPlotNames`
- `getVectorsInfo`
- `getVectorData`
- `getLocalExtrema`
- `getBandwidth`
- `getGainPhaseMargin`
- `getRiseFallTime`
- `getSettlingTime`
- `runSweepStudy`
- `parseMeasResults`
- `runMeasAutomation`
- `runVerificationPlan`

### Model/library/debug telemetry
- `getLtspiceLibraryStatus`
- `listLtspiceLibraryEntries`
- `listLtspiceSymbols`
- `getLtspiceSymbolInfo`
- `scanModelIssues`
- `importModelFile`
- `patchNetlistModelBindings`
- `getToolTelemetry`
- `resetToolTelemetry`
- `tailDaemonLog`
- `getRecentErrors`
- `getCaptureHealth`
- `daemonDoctor`
- `readAgentGuide`

## MCP Resources

- `docs://agent-readme`  
  Returns the full `AGENT_README.md` content as markdown via MCP resources.

## Example Client Config

### `stdio` (subprocess)

```json
{
  "mcpServers": {
    "ltspice-mcp": {
      "command": "ltspice-mcp",
      "args": ["--transport", "stdio"],
      "cwd": "/absolute/path/to/ltspice-mcp"
    }
  }
}
```

### Daemon URL clients

```toml
[mcp_servers.ltspice]
url = "http://127.0.0.1:8765/mcp"
enabled = true
```

### Claude Desktop via `mcp-remote`

```json
{
  "mcpServers": {
    "ltspice-mcp": {
      "command": "/opt/homebrew/bin/npx",
      "args": ["-y", "mcp-remote", "http://127.0.0.1:8765/mcp"]
    }
  }
}
```

## Daemon Operations

Use the helper script from repo root:

```bash
./scripts/ltspice_mcp_daemon.sh start
./scripts/ltspice_mcp_daemon.sh restart
./scripts/ltspice_mcp_daemon.sh stop
./scripts/ltspice_mcp_daemon.sh status
./scripts/ltspice_mcp_daemon.sh logs --lines 200
./scripts/ltspice_mcp_daemon.sh logs --follow
```

Additional helpers:

```bash
./scripts/ltspice_mcp_daemon.sh trigger-initial-permissions
./scripts/ltspice_mcp_daemon.sh trigger-screen-recording-permission
./scripts/ltspice_mcp_daemon.sh trigger-accessibility-permission
./scripts/ltspice_mcp_daemon.sh check-accessibility
```

## UI and Rendering Notes

- UI integration is disabled by default.
- Schematic single-window mode is enabled by default.
- Native LTspice rendering uses ScreenCaptureKit direct-window capture.
- Rendering tools return image content through MCP and include metadata (`image_path`, diagnostics).
- Plot rendering uses `.plt` files to preconfigure traces and axes without UI clicking.

## Important Environment Variables

Core:
- `LTSPICE_BINARY`
- `LTSPICE_MCP_WORKDIR`
- `LTSPICE_MCP_TIMEOUT`
- `LTSPICE_MCP_TRANSPORT`
- `LTSPICE_MCP_HOST`
- `LTSPICE_MCP_PORT`
- `LTSPICE_MCP_STREAMABLE_HTTP_PATH`
- `LTSPICE_MCP_JSON_RESPONSE`
- `LTSPICE_MCP_STATELESS_HTTP`

UI/render:
- `LTSPICE_MCP_UI_ENABLED`
- `LTSPICE_MCP_SCHEMATIC_SINGLE_WINDOW`
- `LTSPICE_MCP_SCHEMATIC_LIVE_PATH`
- `LTSPICE_MCP_SCK_HELPER_DIR`
- `LTSPICE_MCP_SCK_HELPER_PATH`
- `LTSPICE_MCP_VERIFY_WINDOW_CLOSE`

Logging/diagnostics:
- `LTSPICE_MCP_LOG_LEVEL`
- `LTSPICE_MCP_TOOL_LOGGING`
- `LTSPICE_MCP_TOOL_LOG_MAX_ITEMS`
- `LTSPICE_MCP_TOOL_LOG_MAX_CHARS`
- `LTSPICE_MCP_DISABLE_UVICORN_NOISE_FILTERS`
- `LTSPICE_MCP_DAEMON_LOG_DIR`

## Testing

Run full test suite:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Run real ScreenCaptureKit integration tests (disabled by default):

```bash
LTSPICE_MCP_RUN_REAL_SCK=1 PYTHONPATH=src .venv/bin/python -m unittest tests.test_screencapturekit_integration -v
LTSPICE_MCP_RUN_REAL_SCK=1 PYTHONPATH=src .venv/bin/python -m unittest tests.test_plot_render_mcp_real -v
```

## Related Docs

- [AGENT_README.md](AGENT_README.md)
- [CHANGELOG.md](CHANGELOG.md)
- [COMPATIBILITY.md](COMPATIBILITY.md)
