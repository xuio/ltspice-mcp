# ltspice-mcp (macOS)

MCP server for running LTspice on macOS and querying simulation results from `.raw` files.

This implementation is inspired by:
- [gtnoble/ngspice-mcp](https://github.com/gtnoble/ngspice-mcp)
- [luc-me/ltspiceMCP](https://github.com/luc-me/ltspiceMCP)

## What it provides

- LTspice executable auto-discovery on macOS (`LTSPICE_BINARY` override supported)
- Batch simulation from netlist text or existing netlist file
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

## MCP tools

Simulation and setup:

- `getLtspiceStatus`
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

## Run tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
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
