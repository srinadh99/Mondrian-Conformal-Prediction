"""
Shared helpers for the example CLI scripts.
"""

from pathlib import Path


def save_plot_bundle(mcp, output_dir, prefix="GravitySpy", prediction_label="Model prediction"):
    """
    Save the standard MCP plot bundle as both PDF and PNG, plus metrics CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mcp.plot_all(
        output_dir=str(output_dir),
        prefix=prefix,
        show=False,
        include_fig2_scatter=True,
        prediction_label=prediction_label,
    )

    mcp.plot_coverage(
        save_path=output_dir / f"{prefix}_coverage.png",
        show=False,
    )
    mcp.plot_set_size(
        save_path=output_dir / f"{prefix}_setsize.png",
        show=False,
    )
    mcp.plot_roc(
        save_path=output_dir / f"{prefix}_roc.png",
        show=False,
    )
    mcp.plot_alpha_metrics(
        save_path=output_dir / f"{prefix}_alpha_metrics.png",
        show=False,
    )
    mcp.plot_prediction_set_scatter(
        save_path=output_dir / f"{prefix}_fig2_scatter.png",
        show=False,
        prediction_label=prediction_label,
    )
