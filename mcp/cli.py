"""
Package entry point for the MCP CLI.

Usage
-----
    python -m mcp.cli --help
"""

from cli import build_parser, resolve_mode, main

__all__ = ["build_parser", "resolve_mode", "main"]


if __name__ == "__main__":
    main()
