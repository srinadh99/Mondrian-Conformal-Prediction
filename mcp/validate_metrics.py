"""
Package entry point for MCP metric validation.

Usage
-----
    python -m mcp.validate_metrics --help
"""

from validate_metrics import (
    independent_metrics,
    independent_macro_auc,
    independent_mcp_auc,
    check_close,
    main,
)

__all__ = [
    "independent_metrics",
    "independent_macro_auc",
    "independent_mcp_auc",
    "check_close",
    "main",
]


if __name__ == "__main__":
    main()
