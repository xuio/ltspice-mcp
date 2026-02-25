from __future__ import annotations

import inspect
import unittest

from ltspice_mcp import server


class TestToolContracts(unittest.TestCase):
    def test_image_tools_use_calltoolresult_annotation(self) -> None:
        for name in (
            "renderLtspiceSymbolImage",
            "renderLtspiceSchematicImage",
            "renderLtspicePlotImage",
            "renderLtspicePlotPresetImage",
        ):
            func = getattr(server, name)
            sig = inspect.signature(func)
            self.assertIs(sig.return_annotation, server.CallToolResult, name)

    def test_image_tools_do_not_expose_structured_output_schema(self) -> None:
        for name in (
            "renderLtspiceSymbolImage",
            "renderLtspiceSchematicImage",
            "renderLtspicePlotImage",
            "renderLtspicePlotPresetImage",
        ):
            tool = server.mcp._tool_manager.get_tool(name)  # type: ignore[attr-defined]
            self.assertIsNotNone(tool, name)
            self.assertIsNone(tool.output_schema, name)

    def test_plot_render_schema_lists_pane_layout_enum_and_no_backend(self) -> None:
        tool = server.mcp._tool_manager.get_tool("renderLtspicePlotImage")  # type: ignore[attr-defined]
        self.assertIsNotNone(tool)
        assert tool is not None
        props = tool.parameters.get("properties", {})
        self.assertNotIn("backend", props)
        pane_layout = props.get("pane_layout", {})
        self.assertEqual(
            set(pane_layout.get("enum", [])),
            {"single", "split", "per_trace"},
        )


if __name__ == "__main__":
    unittest.main()
