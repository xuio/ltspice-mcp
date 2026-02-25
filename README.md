# ltspice-mcp (macOS)

MCP server for running LTspice on macOS and querying simulation results from `.raw` files.

This implementation is inspired by:
- [gtnoble/ngspice-mcp](https://github.com/gtnoble/ngspice-mcp)
- [luc-me/ltspiceMCP](https://github.com/luc-me/ltspiceMCP)

## Agent guide

For agent-focused operation details (daemon lifecycle, permissions, MCP tool workflow, close verification, and troubleshooting), see:
- [`AGENT_README.md`](AGENT_README.md)

## What it provides

- LTspice executable auto-discovery on macOS (`LTSPICE_BINARY` override supported)
- Optional LTspice UI integration (disabled by default)
- Schematic UI single-window mode (enabled by default)
- Batch simulation from netlist text or existing netlist file
- Schematic generation (`.asc`) from structured data or pin-aware netlist auto-layout (`placement_mode=smart|legacy`)
- Netlist auto-layout supports both two-pin primitives and multi-pin active elements (`X`, `Q`, `M`) when symbols resolve
- Intent-driven circuit generation (filters, non-inverting amplifier, zener regulator)
- Iterative schematic auto-debug loop with targeted fix application, convergence option fixes, and confidence scoring
- LTspice symbol library inspection tools (lib.zip entry/symbol/pin/source queries)
- Model management tools for missing include/model discovery, search-path suggestions, import, and netlist binding patching
- Schematic lint tool with pin-connectivity, dangling-wire, and duplicate-reference checks
- MCP-served image rendering for symbols, schematics, and plots
- Deterministic plot presets (`bode`, `transient_startup`, `noise`, `step_compare`) backed by `.plt`
- Tool-level performance telemetry with rolling timing stats and reset controls
- JSON template-driven schematic generation
- Netlist-file schematic sync/watch workflow with JSON state files
- Run history with artifacts (`.log`, `.raw`, `.op.raw`)
- JSON-backed run metadata persistence across server restarts (`.ltspice_mcp_runs.json`)
- RAW parser with support for:
  - real/complex datasets
  - standard and `FastAccess` binary layouts
  - ASCII `Values:` layout
  - stepped sweeps (`.step`) with per-step segmentation
- Vector queries:
  - list plots and vectors
  - sample/downsample vector traces
  - interpolate vector values at explicit points
  - local minima/maxima extraction
- Structured diagnostics from LTspice logs with categorized issues/suggestions
- Analysis tools:
  - bandwidth
  - gain/phase margin
  - rise/fall time
  - settling time

## Requirements

- Python 3.11+
- LTspice for macOS installed (typically `/Applications/LTspice.app`)

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## First-time setup (required for screenshots)

Use the daemon helper script. On the first `start`, it now runs a one-time permission setup automatically.

```bash
./scripts/ltspice_mcp_daemon.sh start
```

What happens on first run:
- Screen Recording flow is triggered so macOS can grant capture access.
- Accessibility flow is triggered so the daemon can close LTspice windows after screenshots.

If macOS opens permission dialogs or settings panes, allow access and then verify:

```bash
./scripts/ltspice_mcp_daemon.sh check-accessibility
./scripts/ltspice_mcp_daemon.sh trigger-screen-recording-permission
```

You can re-run both permission triggers anytime:

```bash
./scripts/ltspice_mcp_daemon.sh trigger-initial-permissions
```

## Run

```bash
ltspice-mcp --transport stdio
```

Transport modes:
- `stdio` (default): subprocess mode for MCP clients that spawn the server
- `sse`: HTTP + SSE endpoint mode
- `streamable-http`: HTTP MCP endpoint mode (best for standalone daemon setups)

Optional flags:

```bash
ltspice-mcp \
  --workdir /absolute/path/to/workdir \
  --ltspice-binary /Applications/LTspice.app/Contents/MacOS/LTspice \
  --timeout 180 \
  --transport stdio
```

HTTP daemon example:

```bash
ltspice-mcp \
  --daemon-http \
  --host 127.0.0.1 \
  --port 8765 \
  --http-path /mcp \
  --workdir /absolute/path/to/workdir \
  --ltspice-binary /Applications/LTspice.app/Contents/MacOS/LTspice
```

Environment variables:

- `LTSPICE_BINARY`
- `LTSPICE_MCP_WORKDIR`
- `LTSPICE_MCP_TIMEOUT`
- `LTSPICE_MCP_TRANSPORT` (`stdio`/`sse`/`streamable-http`)
- `LTSPICE_MCP_HOST` (HTTP transports)
- `LTSPICE_MCP_PORT` (HTTP transports)
- `LTSPICE_MCP_MOUNT_PATH` (HTTP transports)
- `LTSPICE_MCP_SSE_PATH` (SSE transport endpoint path)
- `LTSPICE_MCP_MESSAGE_PATH` (SSE message endpoint path)
- `LTSPICE_MCP_STREAMABLE_HTTP_PATH` (streamable-http endpoint path, default `/mcp`)
- `LTSPICE_MCP_UI_ENABLED` (`true`/`false`)
- `LTSPICE_MCP_SCHEMATIC_SINGLE_WINDOW` (`true`/`false`, default `true`)
- `LTSPICE_MCP_SCHEMATIC_LIVE_PATH` (optional live schematic path override)
- `LTSPICE_MCP_SCK_HELPER_DIR` (optional ScreenCaptureKit helper cache dir; default `~/Library/Application Support/ltspice-mcp/bin`)
- `LTSPICE_MCP_SCK_HELPER_PATH` (optional absolute path to a prebuilt ScreenCaptureKit helper executable)
- `LTSPICE_MCP_LOG_LEVEL` (`DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL`, default `INFO`)
- `LTSPICE_MCP_TOOL_LOGGING` (`true`/`false`, default `true`)
- `LTSPICE_MCP_TOOL_LOG_MAX_ITEMS` (max collection items summarized per tool log event, default `16`)
- `LTSPICE_MCP_TOOL_LOG_MAX_CHARS` (max string size summarized per tool log event, default `300`)

Tool-call logs are emitted as structured JSON with the `mcp_tool` prefix, including:
- `tool_call_start` (tool name + summarized args/kwargs)
- `tool_call_success` (elapsed time + summarized result)
- `tool_call_error` (elapsed time + exception type/message)

## MCP tools

Simulation and setup:

- `getLtspiceStatus`
- `getLtspiceUiStatus`
- `getLtspiceLibraryStatus`
- `listLtspiceLibraryEntries`
- `listLtspiceSymbols`
- `getLtspiceSymbolInfo`
- `renderLtspiceSymbolImage`
- `renderLtspiceSchematicImage`
- `renderLtspicePlotImage`
- `listPlotPresets`
- `generatePlotPresetSettings`
- `renderLtspicePlotPresetImage`
- `generatePlotSettings`
- `setLtspiceUiEnabled`
- `setSchematicUiSingleWindow`
- `closeLtspiceWindow`
- `startLtspiceRenderSession`
- `endLtspiceRenderSession`
- `openLtspiceUi`
- `createSchematic`
- `createSchematicFromNetlist`
- `listSchematicTemplates`
- `createSchematicFromTemplate`
- `listIntentCircuitTemplates`
- `createIntentCircuit`
- `syncSchematicFromNetlistFile`
- `watchSchematicFromNetlistFile`
- `validateSchematic`
- `lintSchematic`
- `loadCircuit`
- `loadNetlistFromFile`
- `runSimulation`
- `simulateNetlist`
- `simulateNetlistFile`
- `simulateSchematicFile`
- `autoDebugSchematic`
- `getToolTelemetry`
- `resetToolTelemetry`
- `scanModelIssues`
- `importModelFile`
- `patchNetlistModelBindings`
- `listRuns`
- `getRunDetails`

Data access:

- `getPlotNames`
- `getVectorsInfo`
- `getVectorData`
- `getLocalExtrema`
- `getBandwidth`
- `getGainPhaseMargin`
- `getRiseFallTime`
- `getSettlingTime`

### Stepped sweeps

For stepped RAW datasets, `getVectorsInfo`, `getVectorData`, `getLocalExtrema`, and the analysis tools accept `step_index`.

- `step_index` omitted: defaults to step `0`
- `step_index=1` (etc.): filter to a specific sweep step

Responses include:

- `step_count`
- `selected_step`
- `steps` (index/range/label metadata)

## Example client config

Subprocess (`stdio`) mode:
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

Daemon (`streamable-http`) mode:
```toml
[mcp_servers.ltspice]
url = "http://127.0.0.1:8765/mcp"
enabled = true
```

Claude Desktop daemon config example:
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

Note: on some Claude Desktop builds, a direct `"url"` entry can cause startup failures. The `mcp-remote` bridge above is the stable option.

## Standalone daemon mode (optional)

Use this mode when you want one long-lived MCP server process shared by multiple clients (Codex, Claude, etc.) over HTTP.

1. Start daemon in a dedicated terminal:
```bash
ltspice-mcp --daemon-http --host 127.0.0.1 --port 8765 --http-path /mcp
```

2. Point clients to `http://127.0.0.1:8765/mcp`:
   - Codex/URL-capable clients: direct URL config
   - Claude Desktop: use `mcp-remote` bridge (`command` + `args`) as shown above

3. Keep daemon running (or run it via `launchd`) so clients connect to it rather than spawning separate subprocesses.

Daemon helper script (recommended, uses `uv run` internally):
```bash
./scripts/ltspice_mcp_daemon.sh start
```

Agent/LLM operations after code changes:
```bash
./scripts/ltspice_mcp_daemon.sh restart
./scripts/ltspice_mcp_daemon.sh status
./scripts/ltspice_mcp_daemon.sh trigger-initial-permissions
./scripts/ltspice_mcp_daemon.sh check-accessibility
./scripts/ltspice_mcp_daemon.sh trigger-accessibility-permission
./scripts/ltspice_mcp_daemon.sh trigger-screen-recording-permission
./scripts/ltspice_mcp_daemon.sh follow
./scripts/ltspice_mcp_daemon.sh latest-log
./scripts/ltspice_mcp_daemon.sh list-logs 5
./scripts/ltspice_mcp_daemon.sh logs --lines 200
./scripts/ltspice_mcp_daemon.sh logs --follow
```

Permission prompt helper:
- On the first daemon start, permission setup runs automatically once and drops a marker at `.mcp-workdir/daemon/first-run-permissions.done`.
- `./scripts/ltspice_mcp_daemon.sh trigger-initial-permissions` runs both permission triggers manually (Screen Recording + Accessibility).
- `./scripts/ltspice_mcp_daemon.sh trigger-screen-recording-permission` forces an MCP-over-HTTP image render call (through `mcp-remote`) that exercises ScreenCaptureKit so macOS can show/refresh Screen Recording permission state.
- `./scripts/ltspice_mcp_daemon.sh check-accessibility` verifies whether the daemon process can use macOS Accessibility APIs to close LTspice windows.
- `./scripts/ltspice_mcp_daemon.sh trigger-accessibility-permission` intentionally exercises LTspice UI open/close control through MCP and opens the Accessibility settings pane when access is denied.
- Set `LTSPICE_MCP_DAEMON_AUTO_FIRST_RUN_PERMISSION_SETUP=0` to disable automatic first-run permission setup.

## UI integration (optional)

UI is disabled by default. You can enable or override it either globally or per call.

Global:
- default behavior is equivalent to `--ui-disabled`
- enable with `--ui-enabled` (or `LTSPICE_MCP_UI_ENABLED=true`)
- disable explicitly with `--ui-disabled` (or `LTSPICE_MCP_UI_ENABLED=false`)
- change at runtime via `setLtspiceUiEnabled`

Schematic single-window mode:
- enabled by default (`--schematic-single-window` / `LTSPICE_MCP_SCHEMATIC_SINGLE_WINDOW=true`)
- routes schematic UI opens through one live path (default: `<workdir>/.ui/live_schematic.asc`)
- disable via `--schematic-multi-window` or `setSchematicUiSingleWindow(false)`
- override live path with `--schematic-live-path` or `LTSPICE_MCP_SCHEMATIC_LIVE_PATH`

Per simulation call:
- `show_ui=true|false` (overrides default)
- `open_raw_after_run=true` to open waveform output after batch simulation

## Schematic generation

- `createSchematic`: build `.asc` from explicit components/wires/labels/directives
- `createSchematicFromNetlist`: parse a netlist and auto-place primitives (`R`, `C`, `L`, `D`, `V`, `I`) and active multi-pin elements (`X`, `Q`, `M`) when symbols resolve
  - `placement_mode=smart` (default): net-layered, pin-aware placement and cleaner trunk routing
  - `placement_mode=legacy`: original simple grid layout
- `listSchematicTemplates`: inspect built-in or user-supplied JSON templates
- `createSchematicFromTemplate`: generate `.asc` from template type `netlist` or `spec`
- `listIntentCircuitTemplates` / `createIntentCircuit`: high-level templates for common analog intents
- `syncSchematicFromNetlistFile`: only regenerate `.asc` when source netlist content changes
- `watchSchematicFromNetlistFile`: poll netlist changes and emit rebuild events
- `loadCircuit`: also attempts netlist-to-schematic generation and returns `asc_path` when successful

All create/sync/watch schematic tools return an `asc_path` and support `open_ui` to open the resulting schematic in LTspice.

Schematic debug workflow:
- `validateSchematic`: preflight checks for `.asc` files (components, ground flag, simulation directives)
- `lintSchematic`: deeper structural linting (pin connectivity, dangling wire endpoints, duplicate InstName detection)
- `simulateSchematicFile`: runs `.asc` directly in batch mode and optionally includes preflight validation in the response
- use `abort_on_validation_error=true` when you want to block execution until preflight issues are fixed
- schematics generated from netlists/templates include a sidecar `.cir`; `simulateSchematicFile` uses this sidecar
- macOS note: LTspice batch simulation does not run `.asc` directly; `simulateSchematicFile` requires a sidecar netlist (`.cir`/`.net`/`.sp`/`.spi`) next to the schematic

Template notes:
- built-in template JSON: `src/ltspice_mcp/schematic_templates.json`
- built-in examples include `rc_lowpass_ac`, `rc_highpass_ac`, `rl_highpass_ac`, `zener_regulator_dc`, `resistor_divider_spec`, and `non_inverting_opamp_spec`
- string fields support `{placeholder}` substitution via `parameters`
- missing placeholders are left as-is (useful for LTspice param braces like `{rval}`)
- spec templates can include `sidecar_netlist_content` to emit a validated sidecar `.cir` next to `.asc`

Model/debug workflow:
- `scanModelIssues`: parse missing include/model/subckt diagnostics from logs and optionally scan model search paths for best-match suggestions
- `importModelFile`: copy model files into `<workdir>/models` and return include directives + discovered `.subckt` names
- `patchNetlistModelBindings`: add `.include` lines and remap model/subckt tokens in netlists
- `autoDebugSchematic`: iterative validate/simulate/fix loop (preflight + floating-node bleeder fixes + convergence option fixes + optional model include injection) with `confidence` scoring in output

Telemetry:
- `getToolTelemetry`: inspect rolling per-tool latency/call stats (`avg_ms`, `p50_ms`, `p95_ms`, error counts)
- `resetToolTelemetry`: clear telemetry globally or for one tool

## LTspice library inspection

- `getLtspiceLibraryStatus`: verify `lib.zip` path and symbol counts
- `listLtspiceLibraryEntries`: list raw `.asy` zip entries (useful for path discovery)
- `listLtspiceSymbols`: search symbols by name/category (e.g. `opamp2`, `UniversalOpAmp2`)
- `getLtspiceSymbolInfo`: return parsed pin map and optional `.asy` source text

## Image rendering (MCP-served)

- `renderLtspiceSymbolImage`: returns symbol image content plus metadata
- `renderLtspiceSchematicImage`: returns rendered `.asc` schematic image content plus metadata
- `renderLtspicePlotImage`: returns rendered RAW plot image content plus metadata

All three tools return image content blocks through MCP (not just file paths), and also include `image_path` in structured metadata for traceability.
When these tools open LTspice to render an image, the window is closed automatically afterwards.

Render sessions:
- `startLtspiceRenderSession` opens a window once and returns `render_session_id`.
- `renderLtspiceSchematicImage` and `renderLtspicePlotImage` accept `render_session_id` to reuse that window and skip auto-close.
- `endLtspiceRenderSession` closes the associated window.

Plot rendering specifics:
- `renderLtspicePlotImage` writes a companion `.plt` file next to the RAW file so LTspice opens with the requested traces preselected.
- This avoids UI button/menu interaction for trace selection and keeps rendering on LTspice's native plot engine.
- Preset tools:
  - `listPlotPresets`: discover built-ins
  - `generatePlotPresetSettings`: deterministic preset `.plt` generation
  - `renderLtspicePlotPresetImage`: render image using preset settings
- Plot controls supported by `renderLtspicePlotImage` and `generatePlotSettings`:
  - `mode`: `auto`, `db`, `phase`, `real`, `imag`
  - `pane_layout`: `single`, `split`, `per_trace`
  - `dual_axis`: AC Bode dual-axis toggle (magnitude + phase)
  - `x_log`, `x_min`, `x_max`, `y_min`, `y_max`: explicit axis controls
  - `step_index`: for stepped runs, selected step is materialized to a temporary step-specific RAW for LTspice rendering
- Plot captures include `capture_validation` metadata, and `validate_capture=true` (default) fails fast when LTspice returns an empty/missing-trace plot image.

Downscale:
- `downscale_factor` (e.g. `0.5`) is supported for symbol, schematic, and plot image tools.

LTspice screenshot behavior:
- uses ScreenCaptureKit direct-window capture (`SCContentFilter(desktopIndependentWindow:)`)
- runs capture through a persistent helper binary (`ltspice-sck-helper`) so macOS Screen Recording can be approved once per helper path
- opens LTspice in the background (`open -g -j`) to reduce Space switching
- captures the first frame from an `SCStream` for reliability, including LTspice windows that are off-screen/in other Spaces
- returns detailed `capture_diagnostics` metadata (timing, preflight state, title matching, candidate window info)

macOS Screen Recording tip:
- if you use multiple MCP hosts/models, point them to the same helper path (`LTSPICE_MCP_SCK_HELPER_PATH`) or the same helper dir (`LTSPICE_MCP_SCK_HELPER_DIR`) to avoid repeated permission prompts.

## Run tests

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Schematic tests include fixture snapshots under `tests/fixtures/schematic/*.asc` for deterministic rendering checks.

Real ScreenCaptureKit integration tests are available and disabled by default:

```bash
LTSPICE_MCP_RUN_REAL_SCK=1 PYTHONPATH=src .venv/bin/python -m unittest tests.test_screencapturekit_integration -v
LTSPICE_MCP_RUN_REAL_SCK=1 PYTHONPATH=src .venv/bin/python -m unittest tests.test_plot_render_mcp_real -v
```

## Run MCP smoke test

This performs an end-to-end MCP stdio session, runs LTspice on a sample circuit, and validates the core tools.

```bash
source .venv/bin/activate
python3 smoke_test_mcp.py \
  --server-command .venv/bin/ltspice-mcp \
  --ltspice-binary /Applications/LTspice.app/Contents/MacOS/LTspice
```

Run responses now include `diagnostics` with categories like `convergence`, `floating_node`, and `model_missing`, each with suggested fixes.

## Changelog and compatibility

- Changelog: [`CHANGELOG.md`](CHANGELOG.md)
- Compatibility notes: [`COMPATIBILITY.md`](COMPATIBILITY.md)
