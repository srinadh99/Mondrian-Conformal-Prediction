"""
Notebook-friendly package exports for the MCP tool.

Example
-------
    from mcp import MCPTool

    mcp = MCPTool("predictions.csv", alpha=0.1).run()
"""

from mcp_tool import (
    MCPTool,
    calibration,
    testing,
    get_nonconformity_dict,
    load_prediction_csv,
    make_alpha_grid,
    compute_qhat_sweep,
    export_qhat_csv,
    load_qhat_csv,
    qhat_dict_from_row,
    select_qhat_for_alpha,
    compute_sweep_from_qhat,
    compute_sweep,
    compute_metrics_at_alpha,
    compute_metrics_from_qhat,
    compute_macro_roc,
    plot_coverage,
    plot_set_size,
    plot_roc,
    plot_alpha_metrics,
    plot_prediction_set_scatter,
    export_metrics_csv,
)

__all__ = [
    "MCPTool",
    "calibration",
    "testing",
    "get_nonconformity_dict",
    "load_prediction_csv",
    "make_alpha_grid",
    "compute_qhat_sweep",
    "export_qhat_csv",
    "load_qhat_csv",
    "qhat_dict_from_row",
    "select_qhat_for_alpha",
    "compute_sweep_from_qhat",
    "compute_sweep",
    "compute_metrics_at_alpha",
    "compute_metrics_from_qhat",
    "compute_macro_roc",
    "plot_coverage",
    "plot_set_size",
    "plot_roc",
    "plot_alpha_metrics",
    "plot_prediction_set_scatter",
    "export_metrics_csv",
]
