# ltspice-mcp (macOS)

MCP server for running LTspice on macOS and querying simulation results from `.raw` files.

This implementation is inspired by:
- [gtnoble/ngspice-mcp](https://github.com/gtnoble/ngspice-mcp)
- [luc-me/ltspiceMCP](https://github.com/luc-me/ltspiceMCP)

## What it provides

- LTspice executable auto-discovery on macOS (`LTSPICE_BINARY` override supported)
- Optional LTspice UI integration (disabled by default)
- Schematic UI single-window mode (enabled by default)
- Batch simulation from netlist text or existing netlist file
- Schematic generation (`.asc`) from structured data or netlist auto-layout
- LTspice symbol library inspection tools (lib.zip entry/symbol/pin/source queries)
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

## Run

```bash
ltspice-mcp --transport stdio
```

Optional flags:

```bash
ltspice-mcp \
  --workdir /absolute/path/to/workdir \
  --ltspice-binary /Applications/LTspice.app/Contents/MacOS/LTspice \
  --timeout 180 \
  --transport stdio
```

Environment variables:

- `LTSPICE_BINARY`
- `LTSPICE_MCP_WORKDIR`
- `LTSPICE_MCP_TIMEOUT`
- `LTSPICE_MCP_UI_ENABLED` (`true`/`false`)
- `LTSPICE_MCP_SCHEMATIC_SINGLE_WINDOW` (`true`/`false`, default `true`)
- `LTSPICE_MCP_SCHEMATIC_LIVE_PATH` (optional live schematic path override)

## MCP tools

Simulation and setup:

- `getLtspiceStatus`
- `getLtspiceUiStatus`
- `getLtspiceLibraryStatus`
- `listLtspiceLibraryEntries`
- `listLtspiceSymbols`
- `getLtspiceSymbolInfo`
- `setLtspiceUiEnabled`
- `setSchematicUiSingleWindow`
- `openLtspiceUi`
- `createSchematic`
- `createSchematicFromNetlist`
- `listSchematicTemplates`
- `createSchematicFromTemplate`
- `syncSchematicFromNetlistFile`
- `watchSchematicFromNetlistFile`
- `loadCircuit`
- `loadNetlistFromFile`
- `runSimulation`
- `simulateNetlist`
- `simulateNetlistFile`
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
- `createSchematicFromNetlist`: parse a netlist and auto-place common two-pin primitives (`R`, `C`, `L`, `D`, `V`, `I`)
- `listSchematicTemplates`: inspect built-in or user-supplied JSON templates
- `createSchematicFromTemplate`: generate `.asc` from template type `netlist` or `spec`
- `syncSchematicFromNetlistFile`: only regenerate `.asc` when source netlist content changes
- `watchSchematicFromNetlistFile`: poll netlist changes and emit rebuild events

All create/sync/watch schematic tools return an `asc_path` and support `open_ui` to open the resulting schematic in LTspice.

Template notes:
- built-in template JSON: `src/ltspice_mcp/schematic_templates.json`
- string fields support `{placeholder}` substitution via `parameters`
- missing placeholders are left as-is (useful for LTspice param braces like `{rval}`)

## LTspice library inspection

- `getLtspiceLibraryStatus`: verify `lib.zip` path and symbol counts
- `listLtspiceLibraryEntries`: list raw `.asy` zip entries (useful for path discovery)
- `listLtspiceSymbols`: search symbols by name/category (e.g. `opamp2`, `UniversalOpAmp2`)
- `getLtspiceSymbolInfo`: return parsed pin map and optional `.asy` source text

## Run tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Schematic tests include fixture snapshots under `tests/fixtures/schematic/*.asc` for deterministic rendering checks.

## Run MCP smoke test

This performs an end-to-end MCP stdio session, runs LTspice on a sample circuit, and validates the core tools.

```bash
source .venv/bin/activate
python3 smoke_test_mcp.py \
  --server-command .venv/bin/ltspice-mcp \
  --ltspice-binary /Applications/LTspice.app/Contents/MacOS/LTspice
```

Run responses now include `diagnostics` with categories like `convergence`, `floating_node`, and `model_missing`, each with suggested fixes.
