# Contributing

Thanks for helping improve LTspice MCP.

## Before You Start

- Read [README.md](README.md) for setup and runtime basics.
- Read [AGENT_README.md](AGENT_README.md) if you are developing via MCP-driven workflows.
- Keep changes focused and easy to review.

## Local Setup

```bash
uv sync
```

If you do not use `uv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Running Tests

Primary test command:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Optional real integration tests (require macOS permissions and LTspice installed):

```bash
LTSPICE_MCP_RUN_REAL_SCK=1 PYTHONPATH=src .venv/bin/python -m unittest tests.test_screencapturekit_integration -v
LTSPICE_MCP_RUN_REAL_SCK=1 PYTHONPATH=src .venv/bin/python -m unittest tests.test_plot_render_mcp_real -v
```

## Coding Guidelines

- Preserve existing behavior unless the PR explicitly changes it.
- Prefer deterministic outputs and clear error diagnostics.
- Keep tool responses structured and backward-compatible.
- Add regression tests for every bug fix.
- Avoid adding fallback behavior that can hide real failures.
- For documentation images, only commit LTspice window-only captures produced by MCP rendering tools (ScreenCaptureKit direct-window path). Never commit full-desktop screenshots.

## Bug Reports

Please include:

- MCP server version
- LTspice version
- transport mode (`stdio`, `sse`, `streamable-http`)
- exact tool call and parameters
- expected behavior vs actual behavior
- relevant daemon logs (`tailDaemonLog`, `getRecentErrors`)

Use the GitHub issue templates when possible.

## Pull Requests

- Use a clear title and concise summary.
- Mention affected tools/endpoints explicitly.
- Include test evidence in the PR description.
- Keep unrelated formatting/refactor changes out of the same PR.

## Versioning Notes

This project tracks behavior changes in [CHANGELOG.md](CHANGELOG.md). If your change is user-visible, update the changelog in the same PR.
