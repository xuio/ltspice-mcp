from __future__ import annotations

import inspect
import unittest

from pydantic import ValidationError

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
        self.assertEqual(set(props.get("mode", {}).get("enum", [])), {"auto", "db", "phase", "real", "imag"})
        self.assertEqual(
            set(props.get("y_mode", {}).get("enum", [])),
            {"magnitude", "phase", "real", "imag", "db"},
        )
        pane_layout = props.get("pane_layout", {})
        self.assertEqual(
            set(pane_layout.get("enum", [])),
            {"single", "split", "per_trace"},
        )

    def test_plot_render_rejects_unknown_arguments(self) -> None:
        tool = server.mcp._tool_manager.get_tool("renderLtspicePlotImage")  # type: ignore[attr-defined]
        self.assertIsNotNone(tool)
        assert tool is not None
        with self.assertRaises(ValidationError):
            tool.fn_metadata.arg_model.model_validate({"vectors": ["V(out)"], "backend": "ltspice"})


if __name__ == "__main__":
    unittest.main()
