#!/usr/bin/env python3
"""
Example CLI for the combined-CSV MCP workflow.

Usage
-----
    python GravitySpy_ExampleCLI.py glitch_confidence_data_1500.csv

Outputs
-------
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
    <output>/<prefix>_qhat.csv
    <output>/calibration.csv
    <output>/test.csv
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from mcp import MCPTool
from mcp.example_utils import save_plot_bundle

def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the MCP workflow from one combined CSV and save the outputs.",
    )
    parser.add_argument("csv", help="Combined CSV with 'true_label' and per-class score columns.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Nominal error rate.")
    parser.add_argument("--test-size", type=float, default=0.5, help="Test split fraction.")
    parser.add_argument("--random-state", type=int, default=123, help="Random seed for the split.")
    parser.add_argument("--n-alphas", type=int, default=51, help="Number of alpha sweep points.")
    parser.add_argument("--output", default="output", help="Output directory.")
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

    mcp = MCPTool(
        str(args.csv),
        alpha=args.alpha,
        test_size=args.test_size,
        random_state=args.random_state,
        n_alphas=args.n_alphas,
    ).run()

    save_plot_bundle(
        mcp,
        output_dir=output_dir,
        prefix=args.prefix,
        prediction_label=args.prediction_label,
    )

    mcp.export_qhat_csv(output_dir / f"{args.prefix}_qhat.csv")
    mcp.export_split_csvs(
        calibration_path=output_dir / "calibration.csv",
        test_path=output_dir / "test.csv",
    )

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
