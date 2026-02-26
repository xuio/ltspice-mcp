# LTspice MCP for macOS

Model Context Protocol (MCP) server that lets agents and MCP clients control LTspice on macOS for simulation, schematic generation, data extraction, verification, and rendering.

This project is designed for practical automation: reliable runs, reproducible artifacts, and outputs that match LTspice behavior closely.

Inspired by:
- [gtnoble/ngspice-mcp](https://github.com/gtnoble/ngspice-mcp)
- [luc-me/ltspiceMCP](https://github.com/luc-me/ltspiceMCP)

## What You Can Do

- Run LTspice simulations from MCP (`simulateNetlistFile`, `runSimulation`, queue tools).
- Generate and refine schematics (`createSchematic*`, lint/clean/debug tools).
- Render real LTspice images for schematics, plots, and symbols.
- Query RAW vectors and analysis metrics (bandwidth, margins, rise/fall, settling).
- Automate `.meas` and assertion-driven verification workflows.
- Run stepped and Monte Carlo studies with structured results.

## Quick Start (5 Minutes)

### 1) Prerequisites

- macOS with LTspice installed (`/Applications/LTspice.app` expected by default).
- Python 3.11+.
- `uv` (recommended) or `pip`.

### 2) Install

```bash
uv sync
```

If you prefer pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3) Start the MCP daemon

```bash
./scripts/ltspice_mcp_daemon.sh start
./scripts/ltspice_mcp_daemon.sh status
```

Default endpoint:
- `http://127.0.0.1:8765/mcp`

### 4) Grant macOS permissions once

Required for screenshot/render and some UI automation features.

```bash
./scripts/ltspice_mcp_daemon.sh trigger-initial-permissions
./scripts/ltspice_mcp_daemon.sh check-accessibility
```

### 5) Run a smoke test

```bash
uv run python smoke_test_mcp.py \
  --transport streamable-http \
  --server-url http://127.0.0.1:8765/mcp
```

## Client Configuration

### URL-capable MCP clients

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

### `stdio` (subprocess) mode

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

## Core Capabilities by Category

### Setup and diagnostics

- `getLtspiceStatus`, `getLtspiceUiStatus`
- `daemonDoctor`, `tailDaemonLog`, `getRecentErrors`, `getCaptureHealth`

### Simulation and queueing

- `simulateNetlist`, `simulateNetlistFile`, `runSimulation`
- `queueSimulationJob`, `listJobs`, `jobStatus`, `cancelJob`, `listJobHistory`

### Schematic workflows

- `createSchematic`, `createSchematicFromNetlist`, `createSchematicFromTemplate`
- `validateSchematic`, `lintSchematic`, `autoDebugSchematic`
- `inspectSchematicVisualQuality`, `autoCleanSchematicLayout`

### Data, measurements, and verification

- `getPlotNames`, `getVectorsInfo`, `getVectorData`, `getLocalExtrema`
- `getBandwidth`, `getGainPhaseMargin`, `getRiseFallTime`, `getSettlingTime`
- `parseMeasResults`, `runMeasAutomation`, `runVerificationPlan`, `runSweepStudy`

### Native LTspice rendering

- `renderLtspiceSchematicImage`
- `renderLtspicePlotImage`, `renderLtspicePlotPresetImage`
- `renderLtspiceSymbolImage`
- `startLtspiceRenderSession`, `endLtspiceRenderSession`

## Reliability Notes

- Streamable HTTP defaults are tuned for compatibility:
  - `json_response = true`
  - `stateless_http = true`
- UI integration is disabled by default (`LTSPICE_MCP_UI_ENABLED=0`).
- Schematic single-window updates are enabled by default.
- Rendering uses LTspice + ScreenCaptureKit direct-window capture.
- Run artifacts are stored per run to keep historical results stable.

## Daemon Operations

```bash
./scripts/ltspice_mcp_daemon.sh start
./scripts/ltspice_mcp_daemon.sh restart
./scripts/ltspice_mcp_daemon.sh stop
./scripts/ltspice_mcp_daemon.sh status
./scripts/ltspice_mcp_daemon.sh logs --lines 200
./scripts/ltspice_mcp_daemon.sh logs --follow
```

Permission helpers:

```bash
./scripts/ltspice_mcp_daemon.sh trigger-screen-recording-permission
./scripts/ltspice_mcp_daemon.sh trigger-accessibility-permission
```

## Testing

Run core tests:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Run real ScreenCaptureKit integration tests (opt-in):

```bash
LTSPICE_MCP_RUN_REAL_SCK=1 PYTHONPATH=src .venv/bin/python -m unittest tests.test_screencapturekit_integration -v
LTSPICE_MCP_RUN_REAL_SCK=1 PYTHONPATH=src .venv/bin/python -m unittest tests.test_plot_render_mcp_real -v
```

## Contributing

Contributions are welcome. Start with:
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [AGENT_README.md](AGENT_README.md) for agent-specific workflows

When filing bugs, include:
- MCP server version,
- LTspice version,
- transport mode,
- exact tool call + parameters,
- daemon log excerpts.

## Documentation Map

- [docs/README.md](docs/README.md)
- [AGENT_README.md](AGENT_README.md)
- [CHANGELOG.md](CHANGELOG.md)
- [COMPATIBILITY.md](COMPATIBILITY.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [SUPPORT.md](SUPPORT.md)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [SECURITY.md](SECURITY.md)
- MCP resource: `docs://agent-readme`
- MCP tool: `readAgentGuide`

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

Logging:
- `LTSPICE_MCP_LOG_LEVEL`
- `LTSPICE_MCP_TOOL_LOGGING`
- `LTSPICE_MCP_TOOL_LOG_MAX_ITEMS`
- `LTSPICE_MCP_TOOL_LOG_MAX_CHARS`
- `LTSPICE_MCP_DISABLE_UVICORN_NOISE_FILTERS`
- `LTSPICE_MCP_DAEMON_LOG_DIR`
