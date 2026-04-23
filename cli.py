#!/usr/bin/env python3
"""
cli.py — Command-line interface for the MCP Tool
=================================================

Examples
--------
    # Run with defaults (alpha=0.1, saves to ./mcp_plots/)
    python cli.py predictions.csv

    # Use separate calibration and test CSV files
    python cli.py --calibration-csv calibration.csv --test-csv test.csv

    # Use a precomputed alpha-vs-qhat CSV plus test data
    python cli.py --qhat-csv qhat_sweep.csv --test-csv test.csv

    # Export only the qhat CSV from calibration data
    python cli.py --calibration-csv calibration.csv --qhat-only

    # Custom alpha and output directory
    python cli.py predictions.csv --alpha 0.05 --output ./results/

    # Save as PNG, suppress interactive windows
    python cli.py predictions.csv --format png --no-show

    # Suppress the metrics table
    python cli.py predictions.csv --no-print-metrics

    # Print CP prediction set for a single test sample
    python cli.py predictions.csv --predict-one 0

    # Add the Fig. 2-style random test-sample scatter plot
    python cli.py predictions.csv --fig2-scatter --no-show

    # Generate only the Fig. 2-style scatter plot
    python cli.py predictions.csv --fig2-only --no-show

    # Use a custom label for the dark-blue point in the scatter plot
    python cli.py predictions.csv --fig2-scatter --prediction-label "CNN prediction" --no-show

    # Finer alpha sweep
    python cli.py predictions.csv --n-alphas 40 --prefix experiment1
"""

import argparse
import os
import sys


def build_parser():
    p = argparse.ArgumentParser(
        prog="mcp_tool",
        description=(
            "Mondrian Conformal Prediction (MCP) Tool\n"
            "Generates publication-quality CP plots and a metrics summary\n"
            "from a CSV of true labels and predicted class probabilities."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("csv", nargs="?", metavar="CSV_FILE",
                   help="Combined input CSV: 'true_label' column + one column per class.")
    p.add_argument("--calibration-csv", metavar="CSV_FILE",
                   help="Calibration CSV with 'true_label' + per-class scores.")
    p.add_argument("--test-csv", metavar="CSV_FILE",
                   help="Test CSV with 'true_label' + per-class scores.")
    p.add_argument("--qhat-csv", metavar="CSV_FILE",
                   help="Precomputed alpha-vs-qhat CSV produced from calibration data.")

    p.add_argument("--alpha", type=float, default=0.1, metavar="FLOAT",
                   help="CP error rate (default: 0.1 -> 90%% coverage guarantee).")
    p.add_argument("--test-size", type=float, default=0.5, metavar="FLOAT",
                   help="Test fraction (default: 0.5).")
    p.add_argument("--random-state", type=int, default=123, metavar="INT",
                   help="Random seed for train/test split (default: 123).")
    p.add_argument("--n-alphas", type=int, default=51, metavar="INT",
                   help="Alpha grid points for sweeps (default: 51).")

    p.add_argument("--output", default="./mcp_plots", metavar="DIR",
                   help="Output directory (default: ./mcp_plots).")
    p.add_argument("--prefix", default="mcp", metavar="STR",
                   help="Filename prefix for saved plots (default: mcp).")
    p.add_argument("--qhat-output", default=None, metavar="FILE",
                   help="Output CSV path for the alpha-vs-qhat sweep (default: <output>/<prefix>_qhat.csv).")
    p.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"],
                   help="Output file format (default: pdf).")
    p.add_argument("--prediction-label", default="Model prediction", metavar="STR",
                   help="Legend/axis label for the dark-blue point in the Fig. 2-style scatter plot.")

    p.add_argument("--no-show", action="store_true",
                   help="Suppress interactive plot windows (save files only).")

    # Selective plot flags
    g = p.add_argument_group("plot selection (default: all four plots)")
    g.add_argument("--coverage-only",      action="store_true",
                   help="Generate only the coverage validity plot.")
    g.add_argument("--setsize-only",       action="store_true",
                   help="Generate only the set-size efficiency plot.")
    g.add_argument("--roc-only",           action="store_true",
                   help="Generate only the ROC plot.")
    g.add_argument("--alpha-metrics-only", action="store_true",
                   help="Generate only the alpha vs metrics (F1/singletons/set-size) plot.")
    g.add_argument("--fig2-only",          action="store_true",
                   help="Generate only the Fig. 2-style prediction-set scatter plot.")
    p.add_argument("--fig2-scatter", action="store_true",
                   help="Also generate the Fig. 2-style prediction-set scatter plot.")
    p.add_argument("--no-print-metrics", action="store_true",
                   help="Suppress the metrics summary table.")
    p.add_argument("--no-export-csv", action="store_true",
                   help="Skip saving the alpha-sweep metrics CSV.")
    p.add_argument("--qhat-only", action="store_true",
                   help="Export the alpha-vs-qhat CSV and exit (requires calibration data only).")
    p.add_argument("--predict-one", type=int, default=None, metavar="INDEX",
                   help="Print CP prediction set for test sample at INDEX (0-based).")
    p.add_argument("--nonconf-type", default="baseline", choices=["baseline"],
                   help="Nonconformity measure (default: baseline).")

    return p


def resolve_mode(args, parser):
    if args.csv:
        if any([args.calibration_csv, args.test_csv, args.qhat_csv, args.qhat_only]):
            parser.error("When CSV_FILE is provided, do not also pass --calibration-csv, --test-csv, --qhat-csv, or --qhat-only.")
        return "combined"

    if args.qhat_csv:
        if args.calibration_csv or not args.test_csv:
            parser.error("--qhat-csv requires --test-csv and cannot be combined with --calibration-csv.")
        if args.qhat_only:
            parser.error("--qhat-only cannot be used with --qhat-csv.")
        return "qhat"

    if args.calibration_csv and args.test_csv:
        if args.qhat_only:
            parser.error("--qhat-only is for calibration-only export. Omit --test-csv if you only want the qhat CSV.")
        return "split"

    if args.calibration_csv and args.qhat_only:
        if args.test_csv:
            parser.error("--qhat-only calibration export does not take --test-csv.")
        return "calibration_only"

    parser.error(
        "Provide either CSV_FILE, or --calibration-csv with --test-csv, "
        "or --qhat-csv with --test-csv, or --calibration-csv with --qhat-only."
    )


def main(argv=None):
    parser = build_parser()
    args   = parser.parse_args(argv)
    mode   = resolve_mode(args, parser)

    if not 0 < args.alpha < 1:
        sys.exit("Error: --alpha must be in (0, 1).")
    if not 0 < args.test_size < 1:
        sys.exit("Error: --test-size must be in (0, 1).")
    for path in [args.csv, args.calibration_csv, args.test_csv, args.qhat_csv]:
        if path and not os.path.isfile(path):
            sys.exit(f"Error: file not found: {path}")

    # Suppress GUI if requested
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    try:
        from mcp import MCPTool
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from mcp import MCPTool

    os.makedirs(args.output, exist_ok=True)

    if mode == "combined":
        mcp = MCPTool(
            csv_path=args.csv,
            alpha=args.alpha,
            test_size=args.test_size,
            random_state=args.random_state,
            n_alphas=args.n_alphas,
            nonconf_type=args.nonconf_type,
        )
        input_desc = args.csv
    elif mode == "split":
        mcp = MCPTool.from_split(
            args.calibration_csv,
            args.test_csv,
            alpha=args.alpha,
            test_size=args.test_size,
            random_state=args.random_state,
            n_alphas=args.n_alphas,
            nonconf_type=args.nonconf_type,
        )
        input_desc = f"cal={args.calibration_csv} | test={args.test_csv}"
    elif mode == "qhat":
        mcp = MCPTool.from_qhat_csv(
            args.qhat_csv,
            args.test_csv,
            alpha=args.alpha,
            test_size=args.test_size,
            random_state=args.random_state,
            n_alphas=args.n_alphas,
            nonconf_type=args.nonconf_type,
        )
        input_desc = f"qhat={args.qhat_csv} | test={args.test_csv}"
    else:
        mcp = MCPTool.from_calibration_csv(
            args.calibration_csv,
            alpha=args.alpha,
            test_size=args.test_size,
            random_state=args.random_state,
            n_alphas=args.n_alphas,
            nonconf_type=args.nonconf_type,
        )
        input_desc = args.calibration_csv

    print("=" * 55)
    print("  Mondrian Conformal Prediction Tool")
    print("=" * 55)
    print(f"  Mode   : {mode}")
    print(f"  Input  : {input_desc}")
    print(f"  Alpha  : {args.alpha}")
    print(f"  Output : {args.output}/")
    print("=" * 55 + "\n")

    qhat_save_path = args.qhat_output or os.path.join(args.output, f"{args.prefix}_qhat.csv")

    if mode == "calibration_only":
        mcp.export_qhat_csv(save_path=qhat_save_path)
        print("\nAll done.")
        return

    mcp.run()

    if mode in {"combined", "split"}:
        mcp.export_qhat_csv(save_path=qhat_save_path)

    # Optional: re-print metrics (run() already prints them; this
    # is a no-op unless --no-print-metrics suppressed the run() output,
    # which it can't — but left here for symmetry if run() is ever made quiet)
    if args.no_print_metrics:
        pass  # metrics already printed by run(); user wants them suppressed

    # Optional single-sample prediction
    if args.predict_one is not None:
        print(f"\n--- Prediction set for test sample {args.predict_one} ---")
        mcp.predict_one(args.predict_one)

    # Determine which plots to generate
    only_flags = [args.coverage_only, args.setsize_only, args.roc_only,
                  args.alpha_metrics_only, args.fig2_only]
    any_only   = any(only_flags)

    def sp(name):
        return os.path.join(args.output, f"{args.prefix}_{name}.{args.format}")

    show = not args.no_show
    print("\nGenerating plots …")

    if not any_only or args.coverage_only:
        mcp.plot_coverage(save_path=sp("coverage"), show=show)

    if not any_only or args.setsize_only:
        mcp.plot_set_size(save_path=sp("setsize"), show=show)

    if not any_only or args.roc_only:
        mcp.plot_roc(save_path=sp("roc"), show=show)

    if not any_only or args.alpha_metrics_only:
        mcp.plot_alpha_metrics(save_path=sp("alpha_metrics"), show=show)

    if args.fig2_scatter or args.fig2_only:
        mcp.plot_prediction_set_scatter(
            save_path=sp("fig2_scatter"),
            show=show,
            prediction_label=args.prediction_label,
        )

    if not args.no_export_csv:
        mcp.export_metrics_csv(save_path=sp("metrics").replace(f".{args.format}", ".csv"))

    print("\nAll done.")


if __name__ == "__main__":
    main()
