# LTspice MCP Agent Playbook

This guide is for LLM agents operating this repository's MCP server on macOS.

It focuses on:
- how to control the HTTP daemon safely,
- how to trigger and verify macOS permissions,
- which MCP tools to call for screenshots/renders/simulations,
- how to verify window auto-close behavior and troubleshoot failures.

## 1) Preferred operating mode

Use one long-lived HTTP daemon and connect clients through MCP over HTTP:

- URL: `http://127.0.0.1:8765/mcp`
- Start/restart/status/logs script: `./scripts/ltspice_mcp_daemon.sh`

Why:
- avoids per-client process churn,
- stabilizes macOS Screen Recording/Accessibility permissions,
- keeps helper binary identity stable.

## 2) Daemon control commands (shell)

Use these commands from the repo root:

```bash
./scripts/ltspice_mcp_daemon.sh start
./scripts/ltspice_mcp_daemon.sh restart
./scripts/ltspice_mcp_daemon.sh stop
./scripts/ltspice_mcp_daemon.sh status
./scripts/ltspice_mcp_daemon.sh logs --lines 200
./scripts/ltspice_mcp_daemon.sh logs --follow
./scripts/ltspice_mcp_daemon.sh latest-log
./scripts/ltspice_mcp_daemon.sh list-logs 5
./scripts/ltspice_mcp_daemon.sh follow
```

### After code changes

Always reload the daemon before testing MCP behavior:

```bash
./scripts/ltspice_mcp_daemon.sh restart
```

## 3) Permission flows (macOS)

The project has explicit trigger/check commands:

```bash
./scripts/ltspice_mcp_daemon.sh trigger-initial-permissions
./scripts/ltspice_mcp_daemon.sh trigger-screen-recording-permission
./scripts/ltspice_mcp_daemon.sh check-accessibility
./scripts/ltspice_mcp_daemon.sh trigger-accessibility-permission
```

What each one does:
- `trigger-screen-recording-permission`:
  runs a real MCP render path through ScreenCaptureKit so macOS can grant Screen Recording.
- `check-accessibility`:
  opens/closes LTspice through MCP and reports whether Accessibility is functional.
- `trigger-accessibility-permission`:
  intentionally runs the Accessibility flow and opens settings if access is denied.
- `trigger-initial-permissions`:
  runs both Screen Recording and Accessibility triggers.

First daemon start can run these automatically once (marker file):
- `.mcp-workdir/daemon/first-run-permissions.done`

## 4) MCP tool quick-reference

### Core status/setup
- `getLtspiceStatus`
- `getLtspiceUiStatus`
- `setLtspiceUiEnabled`

### Rendering and screenshots
- `renderLtspiceSymbolImage`
- `renderLtspiceSchematicImage`
- `renderLtspicePlotImage`
- `renderLtspicePlotPresetImage`
- `startLtspiceRenderSession`
- `endLtspiceRenderSession`
- `openLtspiceUi`
- `closeLtspiceWindow`

### Simulation/data
- `simulateNetlist`
- `simulateNetlistFile`
- `simulateSchematicFile`
- `getPlotNames`
- `getVectorsInfo`
- `getVectorData`

## 5) Screenshot/render behavior to expect

When using LTspice render tools:
- capture uses ScreenCaptureKit helper (`ltspice-sck-helper`),
- LTspice window should auto-close after capture by default,
- response includes diagnostics in:
  - `capture_window_info.capture_diagnostics`
  - `close_event`

Expected successful close indicators:
- `close_event.closed = true`
- `close_event.remaining_windows = 0`

If you pass `render_session_id`, auto-close is intentionally skipped until:
- `endLtspiceRenderSession(render_session_id)`

## 6) Standard agent validation sequence

1. Check daemon:
```bash
./scripts/ltspice_mcp_daemon.sh status
```

2. If code changed, restart:
```bash
./scripts/ltspice_mcp_daemon.sh restart
```

3. Run one MCP render call (`renderLtspiceSymbolImage`).

4. Inspect result fields:
- `capture_backend` should be `screencapturekit`
- `close_event.closed` should be `true`
- `close_event.remaining_windows` should be `0`

5. Run a short repeat loop (for reliability), not only one call.

## 7) Known failure modes and fixes

### A) Daemon says started, but MCP connection is refused

Actions:
1. `./scripts/ltspice_mcp_daemon.sh status`
2. `./scripts/ltspice_mcp_daemon.sh logs --lines 200`
3. `./scripts/ltspice_mcp_daemon.sh restart`
4. Re-check status and retry MCP call

### B) Permission denied errors on captures/closes

Actions:
1. `./scripts/ltspice_mcp_daemon.sh trigger-screen-recording-permission`
2. `./scripts/ltspice_mcp_daemon.sh trigger-accessibility-permission`
3. `./scripts/ltspice_mcp_daemon.sh check-accessibility`

### C) Render succeeds but windows remain open

Actions:
1. Inspect `close_event` in tool result.
2. Use `closeLtspiceWindow` with strong selectors:
   - `exact_title` when available,
   - `window_id` when available.
3. Re-run render and verify `remaining_windows = 0`.

## 8) Client connection examples

### URL-capable MCP clients
Use direct URL:
- `http://127.0.0.1:8765/mcp`

### Claude Desktop bridge
Use `mcp-remote`:

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

## 9) Important helper identity notes

macOS permissions are tied to executable identity/path.

Keep helper paths stable:
- ScreenCaptureKit helper:
  `~/Library/Application Support/ltspice-mcp/bin/ltspice-sck-helper`
- Accessibility close helper:
  `~/Library/Application Support/ltspice-mcp/bin/ltspice-ax-close-helper`

Changing helper path/binary identity can cause additional permission prompts.
