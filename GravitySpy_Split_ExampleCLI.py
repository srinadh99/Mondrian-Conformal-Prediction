#!/usr/bin/env python3
"""
Example CLI for the split calibration/test MCP workflow.

Usage
-----
    python GravitySpy_Split_ExampleCLI.py calibration.csv test.csv

Outputs
-------
    <output>/<prefix>_qhat.csv
    <output>/<prefix>_coverage.pdf
    <output>/<prefix>_coverage.png
    <output>/<prefix>_setsize.pdf
    <output>/<prefix>_setsize.png
    <output>/<prefix>_roc.pdf
    <output>/<prefix>_roc.png
    <output>/<prefix>_alpha_metrics.pdf
    <output>/<prefix>_alpha_metrics.png
    <output>/<prefix>_fig2_scatter.pdf
    <output>/<prefix>_fig2_scatter.png
    <output>/<prefix>_metrics.csv
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from mcp import MCPTool
from mcp.example_utils import save_plot_bundle


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the MCP workflow from separate calibration and test CSVs.",
    )
    parser.add_argument("calibration_csv", help="Calibration CSV with 'true_label' and per-class scores.")
    parser.add_argument("test_csv", help="Test CSV with 'true_label' and per-class scores.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Nominal error rate.")
    parser.add_argument("--random-state", type=int, default=123, help="Random seed for reproducibility.")
    parser.add_argument("--n-alphas", type=int, default=51, help="Number of alpha sweep points.")
    parser.add_argument("--output", default="output_split", help="Output directory.")
    parser.add_argument("--prefix", default="GravitySpy", help="Filename prefix for saved outputs.")
    parser.add_argument(
        "--prediction-label",
        default="Model prediction",
        help="Legend/axis label for the dark-blue point in the scatter plot.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    qhat_csv = output_dir / f"{args.prefix}_qhat.csv"

    qhat_builder = MCPTool.from_calibration_csv(
        args.calibration_csv,
        alpha=args.alpha,
        random_state=args.random_state,
        n_alphas=args.n_alphas,
    )
    qhat_builder.export_qhat_csv(qhat_csv)

    mcp = MCPTool.from_qhat_csv(
        str(qhat_csv),
        args.test_csv,
        alpha=args.alpha,
        random_state=args.random_state,
        n_alphas=args.n_alphas,
    ).run()

    save_plot_bundle(
        mcp,
        output_dir=output_dir,
        prefix=args.prefix,
        prediction_label=args.prediction_label,
    )

    print(f"Saved qhat CSV to: {qhat_csv}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
