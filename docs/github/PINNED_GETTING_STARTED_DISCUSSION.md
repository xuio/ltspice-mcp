# Pinned Discussion Draft: Start Here - LTspice MCP

Welcome to LTspice MCP.

If this is your first time here, this thread gives you the fastest path to a successful setup.

## 1) Install and start daemon

```bash
uv sync
./scripts/ltspice_mcp_daemon.sh start
./scripts/ltspice_mcp_daemon.sh status
```

Endpoint:
- `http://127.0.0.1:8765/mcp`

## 2) Grant macOS permissions once

```bash
./scripts/ltspice_mcp_daemon.sh trigger-initial-permissions
./scripts/ltspice_mcp_daemon.sh check-accessibility
```

## 3) Validate with three MCP calls

1. `getLtspiceStatus`
2. `simulateNetlist` (small RC circuit)
3. `getVectorsInfo` on returned `run_id`

A complete copy-paste version is here:
- [docs/README.md](../README.md)

## 4) If something fails

When opening a bug report, include:
- MCP server version
- LTspice version
- transport mode
- exact tool call + arguments
- `tailDaemonLog` excerpt

Bug template:
- `.github/ISSUE_TEMPLATE/bug_report.yml`

## 5) Good first contributions

- improve docs/examples
- add regression tests for parsing/rendering edge cases
- improve diagnostics for MCP client interoperability

Thanks for trying LTspice MCP.
