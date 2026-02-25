# LTspice MCP Agent Playbook

Operational guide for agents using this repository's MCP server on macOS.

This guide is also available through MCP:
- Resource: `docs://agent-readme`
- Tool: `readAgentGuide`

## 1) Fast Path

From repo root:

```bash
./scripts/ltspice_mcp_daemon.sh status
./scripts/ltspice_mcp_daemon.sh restart
./scripts/ltspice_mcp_daemon.sh logs --lines 200
```

Then use MCP endpoint:
- `http://127.0.0.1:8765/mcp`

## 2) Required Permission Checks

Run once or whenever captures fail:

```bash
./scripts/ltspice_mcp_daemon.sh trigger-initial-permissions
./scripts/ltspice_mcp_daemon.sh check-accessibility
```

Granular permission helpers:

```bash
./scripts/ltspice_mcp_daemon.sh trigger-screen-recording-permission
./scripts/ltspice_mcp_daemon.sh trigger-accessibility-permission
```

## 3) Core MCP Workflow

1. `getLtspiceStatus` to validate executable/workdir/defaults.
2. Create/load/simulate:
   - `createSchematic*` or `loadCircuit` / `loadNetlistFromFile`
   - `simulateNetlist*` / `simulateSchematicFile`
3. Inspect results:
   - `getPlotNames`, `getVectorsInfo`, `getVectorData`
   - `getBandwidth`, `getGainPhaseMargin`, `getRiseFallTime`, `getSettlingTime`
   - `parseMeasResults`, `runVerificationPlan`
   - `runSweepStudy`
4. Render visuals:
   - `renderLtspiceSchematicImage`
   - `renderLtspicePlotImage` or `renderLtspicePlotPresetImage`
5. For long simulations:
   - `queueSimulationJob`, `listJobs`, `jobStatus`, `cancelJob`
6. For schematic cleanup:
   - `inspectSchematicVisualQuality`, `autoCleanSchematicLayout`

## 4) Screenshot and Window Behavior

Rendering uses LTspice + ScreenCaptureKit direct-window capture.

Expected successful capture indicators:
- `capture_backend == "screencapturekit"`
- `close_event.closed == true`
- `close_event.post_verify.verified_closed == true` (when verification is available)

If you pass `render_session_id`, window auto-close is skipped until:
- `endLtspiceRenderSession(render_session_id)`

## 5) Recommended Debug Sequence

1. Check daemon:
   - `tailDaemonLog`
   - `getRecentErrors`
   - `daemonDoctor`
2. Check capture health:
   - `getCaptureHealth`
3. Retry permission flow if needed.
4. Re-run one minimal render call.
5. Confirm close verification fields in `close_event`.

## 6) Common Failure Modes

### A) MCP connection failures

Use:

```bash
./scripts/ltspice_mcp_daemon.sh status
./scripts/ltspice_mcp_daemon.sh logs --lines 200
./scripts/ltspice_mcp_daemon.sh restart
```

### B) Permission denials

Run:

```bash
./scripts/ltspice_mcp_daemon.sh trigger-screen-recording-permission
./scripts/ltspice_mcp_daemon.sh trigger-accessibility-permission
./scripts/ltspice_mcp_daemon.sh check-accessibility
```

### C) Convergence failures

Simulation tools now perform one automatic convergence retry with `.options`.
Inspect response field:
- `auto_convergence_retry`

## 7) MCP Tools for Agent Guidance

- `readAgentGuide`: reads this file with:
  - section selection (index/title),
  - optional text search,
  - heading index output.
- `docs://agent-readme`: full markdown resource.

## 8) Client Notes

- URL-capable MCP clients can use `http://127.0.0.1:8765/mcp` directly.
- Claude Desktop typically works best with `mcp-remote`.

## 9) Helper Binary Identity

macOS permissions are tied to executable path/signature.

Keep helper paths stable:
- `~/Library/Application Support/ltspice-mcp/bin/ltspice-sck-helper`
- `~/Library/Application Support/ltspice-mcp/bin/ltspice-ax-close-helper`
