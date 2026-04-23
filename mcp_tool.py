"""
mcp_tool.py — Mondrian Conformal Prediction (MCP) Tool
=======================================================
Faithful implementation of the CP pipeline from:
  "Classification uncertainty for transient gravitational-wave noise artefacts
   with optimised conformal prediction" (Malz, Ashton, Colombo)

Provides:
  - Core CP functions       : calibration, testing, nonconformity scores
  - Unified metrics engine  : single-pass computation reused by all sweeps
  - Publication-quality plots: coverage, set-size, ROC
  - Metrics summary table   : printed automatically and queryable at any alpha

Usage (notebook):
    from mcp import MCPTool
    mcp = MCPTool("predictions.csv", alpha=0.1).run()
    mcp.plot_all(output_dir="./figures")   # saves 4 plots + metrics CSV
    mcp.plot_prediction_set_scatter()      # Fig. 2-style random test-sample plot
    mcp.plot_prediction_set_scatter(prediction_label="CNN prediction")
    mcp.export_metrics_csv("results.csv")  # alpha-sweep table (standalone)
    mcp.export_qhat_csv("qhat_sweep.csv")  # alpha vs qhat per class
    mcp.print_metrics()                    # summary at chosen alpha
    mcp.print_metrics(alpha=0.05)          # re-query at any alpha

Usage (CLI):
    python cli.py predictions.csv --alpha 0.1 --output ./plots/
    python cli.py --calibration-csv cal.csv --test-csv test.csv --output ./plots/
    python cli.py --qhat-csv qhat_sweep.csv --test-csv test.csv --output ./plots/
    python cli.py predictions.csv --fig2-scatter --no-show
    python cli.py predictions.csv --fig2-scatter --prediction-label "CNN prediction" --no-show
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# NumPy compat: np.trapz was renamed np.trapezoid in NumPy 2.0.
# Restore np.trapz so all code (including user notebooks) can call it uniformly.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

# ---------------------------------------------------------------------------
# Publication-style RC parameters  (Nature / PRD style guides)
# ---------------------------------------------------------------------------
PUBLICATION_RC = {
    "font.family":       "serif",
    "font.size":         12,
    "axes.labelsize":    13,
    "axes.titlesize":    13,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   9,
    "legend.framealpha": 0.85,
    "lines.linewidth":   1.8,
    "axes.linewidth":    1.0,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "ytick.right":       True,
    "grid.alpha":        0.4,
    "grid.linestyle":    "--",
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "figure.dpi":        100,
}


def _pub():
    """Return a matplotlib RC context manager with publication style."""
    return matplotlib.rc_context(PUBLICATION_RC)


def _fmt(label: str) -> str:
    """Replace underscores with spaces for axis / legend labels."""
    return label.replace("_", " ")


def _save_or_show(fig, save_path, show):
    """Shared helper: tight-layout, optionally save, optionally display."""
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    if show:
        plt.show()


# ---------------------------------------------------------------------------
# Wilson binomial confidence interval  (replaces statsmodels dependency)
# ---------------------------------------------------------------------------

def _wilson_ci_band(n_total, alpha_ci=0.01):
    """
    99 % Wilson CI band for proportions x/n, x in [1, n_total].
    Returns (x_norm, lower, upper) as 1-D length-100 arrays.
    """
    x = np.linspace(1, n_total, 100)
    p = x / n_total
    try:
        from scipy.stats import norm
        z = norm.ppf(1 - alpha_ci / 2)
    except ImportError:
        z = 2.576                               # 99 % fallback
    denom  = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return p, centre - margin, centre + margin


# ===========================================================================
# Core CP functions  — identical logic to the reference notebook
# ===========================================================================

def get_nonconformity_dict(labels, nonconf_type="baseline"):
    """
    Build a dict {label: scoring_function} for the chosen nonconformity measure.
    baseline: score(label, row) = 1 - row[label]
    """
    if nonconf_type != "baseline":
        raise ValueError(f"Unknown nonconf_type: {nonconf_type!r}. Only 'baseline' is supported.")
    def _make(label):
        return lambda row: 1.0 - row[label]
    return {label: _make(label) for label in labels}


def calibration(calibration_data, alpha=0.1, nonconf_type="baseline"):
    """
    Mondrian CP calibration: compute per-label quantile thresholds (qhat).

    Parameters
    ----------
    calibration_data : pd.DataFrame
        Columns: 'true_label', then one column per class with probabilities.
    alpha : float
        Error rate (e.g. 0.1 -> 90% coverage guarantee).
    nonconf_type : str
        Nonconformity measure ('baseline' only).

    Returns
    -------
    qhat_dict : dict {label: float}
    """
    labels  = list(calibration_data.columns[1:])
    scores  = {l: [] for l in labels}
    nonconf = get_nonconformity_dict(labels, nonconf_type)

    for _, row in calibration_data.iterrows():
        scores[row.true_label].append(nonconf[row.true_label](row))

    qhat_dict = {}
    for label, sc in scores.items():
        n_cal     = len(sc)
        q_level   = int(np.ceil((n_cal + 1) * (1 - alpha)))
        sorted_sc = np.sort(sc)
        qhat_dict[label] = 1.0 if q_level > n_cal else sorted_sc[q_level - 1]

    return qhat_dict


def testing(test_row, qhat_dict, nonconf_type="baseline"):
    """
    Return the CP prediction set for a single test sample.

    Parameters
    ----------
    test_row  : pd.Series — row with 'true_label' + probability columns
    qhat_dict : dict      — calibrated thresholds from calibration()

    Returns
    -------
    list of str — labels included in the prediction set
    """
    labels  = list(test_row.index[1:])
    nonconf = get_nonconformity_dict(labels, nonconf_type)
    return [lab for lab, fn in nonconf.items() if fn(test_row) < qhat_dict[lab]]


def _ensure_prediction_dataframe(df, labels=None, source_name="CSV"):
    """
    Validate prediction CSV shape and normalise class-column order.
    """
    if "true_label" not in df.columns:
        raise ValueError(f"{source_name} must contain a 'true_label' column.")
    if df.shape[1] < 3:
        raise ValueError(f"{source_name} must have at least two class probability columns.")

    found_labels = list(df.columns[1:])
    if labels is None:
        labels = sorted(found_labels)
    else:
        if set(found_labels) != set(labels):
            raise ValueError(
                f"{source_name} class columns do not match the expected label set."
            )
    return df[["true_label"] + list(labels)].copy(), list(labels)


def load_prediction_csv(path, labels=None):
    """Read and validate a prediction CSV."""
    df = pd.read_csv(path)
    return _ensure_prediction_dataframe(df, labels=labels, source_name=path)


def make_alpha_grid(n_alphas=51, alpha=None):
    """
    Build the alpha sweep grid and ensure the chosen alpha is included exactly.
    """
    alphas = np.linspace(0.01, 0.99, n_alphas)
    if alpha is not None and not np.any(np.isclose(alphas, alpha)):
        alphas = np.append(alphas, alpha)
    return np.asarray(sorted(set(float(a) for a in alphas)))


def compute_qhat_sweep(calibration_data, alphas, nonconf_type="baseline"):
    """
    Compute qhat per class for every alpha in the sweep.
    Returns a DataFrame with columns: alpha, n_cal, qhat_<label>, ...
    """
    labels = sorted(calibration_data.columns[1:].tolist())
    rows = []
    for alpha in alphas:
        qhat = calibration(calibration_data, alpha=float(alpha), nonconf_type=nonconf_type)
        row = {"alpha": float(alpha), "n_cal": int(len(calibration_data))}
        for label in labels:
            row[f"qhat_{label}"] = float(qhat[label])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)


def export_qhat_csv(qhat_table, save_path):
    """Write the alpha-vs-qhat table to CSV."""
    # Preserve threshold values faithfully so reloading the CSV reproduces
    # the same prediction-set boundaries as the in-memory calibration run.
    qhat_table.to_csv(save_path, index=False, float_format="%.17g")
    print(f"  Saved: {save_path}")
    return qhat_table


def load_qhat_csv(path):
    """
    Read and validate an alpha-vs-qhat CSV.
    """
    df = pd.read_csv(path, float_precision="round_trip")
    if "alpha" not in df.columns:
        raise ValueError(f"{path} must contain an 'alpha' column.")
    qhat_cols = [col for col in df.columns if col.startswith("qhat_")]
    if not qhat_cols:
        raise ValueError(f"{path} must contain one or more 'qhat_<label>' columns.")
    labels = sorted(col[len("qhat_"):] for col in qhat_cols)
    ordered = ["alpha"] + (["n_cal"] if "n_cal" in df.columns else []) + [f"qhat_{label}" for label in labels]
    if df["alpha"].duplicated().any():
        raise ValueError(f"{path} contains duplicate alpha rows.")
    return df[ordered].sort_values("alpha").reset_index(drop=True), labels


def qhat_dict_from_row(row, labels):
    """Convert one qhat-table row to a plain {label: qhat} dict."""
    return {label: float(row[f"qhat_{label}"]) for label in labels}


def select_qhat_for_alpha(qhat_table, labels, alpha, allow_nearest=True):
    """
    Return the qhat dict for the requested alpha.
    If the exact alpha is unavailable and allow_nearest=True, the nearest row is used.
    """
    alpha_values = qhat_table["alpha"].to_numpy(dtype=float)
    idx = int(np.argmin(np.abs(alpha_values - alpha)))
    chosen_alpha = float(alpha_values[idx])
    if not np.isclose(chosen_alpha, alpha):
        if not allow_nearest:
            raise ValueError(f"Requested alpha={alpha:.6f} was not found in the qhat CSV.")
        print(f"  Warning: alpha={alpha:.4f} not found in qhat CSV; using nearest stored alpha={chosen_alpha:.4f}.")
    return qhat_dict_from_row(qhat_table.iloc[idx], labels), chosen_alpha


# ===========================================================================
# Unified single-pass metrics engine
# ===========================================================================

def _metrics_pass(test_data, qhat_dict, nonconf_type="baseline"):
    """
    Iterate test_data exactly once, accumulating all statistics needed by
    every higher-level function.  No metric forces a second pass.

    Returns a raw accumulator dict with keys:
        labels, tp, fp, fn, tn, correct, single,
        cov_lengths, all_lengths, cond_correct, cond_setlens, n
    """
    labels = sorted(test_data.columns[1:].tolist())
    tp = {l: 0 for l in labels}
    fp = {l: 0 for l in labels}
    fn = {l: 0 for l in labels}
    tn = {l: 0 for l in labels}
    cond_correct = {l: 0 for l in labels}
    cond_setlens = {l: 0 for l in labels}
    correct = single = cov_lengths = all_lengths = 0

    for _, row in test_data.iterrows():
        true_label = row["true_label"]
        pred_set   = testing(row, qhat_dict, nonconf_type)
        n_set      = len(pred_set)

        all_lengths += n_set
        cond_setlens[true_label] += n_set

        if true_label in pred_set:
            correct     += 1
            cov_lengths += n_set
            cond_correct[true_label] += 1

        if n_set == 1 and pred_set[0] == true_label:
            single += 1

        for l in labels:
            tp[l] += int(l in pred_set and l == true_label)
            fp[l] += int(l in pred_set and l != true_label)
            fn[l] += int(l not in pred_set and l == true_label)
            tn[l] += int(l not in pred_set and l != true_label)

    return dict(labels=labels, tp=tp, fp=fp, fn=fn, tn=tn,
                correct=correct, single=single,
                cov_lengths=cov_lengths, all_lengths=all_lengths,
                cond_correct=cond_correct, cond_setlens=cond_setlens,
                n=len(test_data))


def _summarise(acc):
    """
    Derive all scalar metrics from a _metrics_pass accumulator.
    Returns: coverage, macro_f1, avg_set_size, singleton_rate, fpr, tpr, per_label_f1
    """
    labels = acc["labels"]
    n      = acc["n"]
    tp, fp, fn, tn = acc["tp"], acc["fp"], acc["fn"], acc["tn"]

    per_label_f1 = {
        l: 2 * tp[l] / (2 * tp[l] + fp[l] + fn[l] + 1e-12) for l in labels
    }
    total_fp = sum(fp.values())
    total_tp = sum(tp.values())
    total_fn = sum(fn.values())
    total_tn = sum(tn.values())

    return {
        "coverage":       acc["correct"] / n,
        "macro_f1":       float(np.mean(list(per_label_f1.values()))),
        "avg_set_size":   acc["all_lengths"] / n,
        "singleton_rate": acc["single"] / n,
        "fpr":            total_fp / (total_fp + total_tn + 1e-12),
        "tpr":            total_tp / (total_tp + total_fn + 1e-12),
        "per_label_f1":   per_label_f1,
    }


# ===========================================================================
# Sweep  — single unified function replacing all previous separate sweeps
# ===========================================================================

def compute_sweep_from_qhat(test_data, qhat_table, nonconf_type="baseline"):
    """
    Run the full metric sweep using a precomputed alpha-vs-qhat table.
    """
    qhat_cols = [col for col in qhat_table.columns if col.startswith("qhat_")]
    labels = sorted(col[len("qhat_"):] for col in qhat_cols)
    test_data, labels = _ensure_prediction_dataframe(
        test_data, labels=labels, source_name="test_data"
    )
    n_per_label = {l: (test_data["true_label"] == l).sum() for l in labels}

    marginal_coverage = []
    marginal_setsize = []
    conditional_coverage = {l: [] for l in labels}
    conditional_setsize = {l: [] for l in labels}
    f1_list, singleton_list, avset_list = [], [], []
    mcp_fpr, mcp_tpr = [], []
    alphas = qhat_table["alpha"].to_numpy(dtype=float)

    for _, row in qhat_table.iterrows():
        qhat = qhat_dict_from_row(row, labels)
        acc = _metrics_pass(test_data, qhat, nonconf_type)
        m = _summarise(acc)
        n = acc["n"]

        marginal_coverage.append(acc["correct"] / n)
        marginal_setsize.append(acc["all_lengths"] / n)
        for label in labels:
            nl = max(1, n_per_label[label])
            conditional_coverage[label].append(acc["cond_correct"][label] / nl)
            conditional_setsize[label].append(acc["cond_setlens"][label] / nl)

        f1_list.append(m["macro_f1"])
        singleton_list.append(m["singleton_rate"])
        avset_list.append(acc["all_lengths"] / n)
        mcp_fpr.append(m["fpr"])
        mcp_tpr.append(m["tpr"])

    order = np.argsort(mcp_fpr)
    fpr_s = np.concatenate([[0.0], np.array(mcp_fpr)[order], [1.0]])
    tpr_s = np.concatenate([[0.0], np.array(mcp_tpr)[order], [1.0]])
    mcp_auc = float(np.trapz(tpr_s, fpr_s))

    return {
        "labels":               labels,
        "alphas":               np.asarray(alphas),
        "marginal_coverage":    marginal_coverage,
        "conditional_coverage": conditional_coverage,
        "marginal_setsize":     marginal_setsize,
        "conditional_setsize":  conditional_setsize,
        "f1_list":              f1_list,
        "singleton_list":       singleton_list,
        "avset_list":           avset_list,
        "mcp_fpr":              np.array(mcp_fpr),
        "mcp_tpr":              np.array(mcp_tpr),
        "mcp_auc":              mcp_auc,
    }


def compute_sweep(calibration_data, test_data, alphas, nonconf_type="baseline"):
    """
    Run one _metrics_pass per alpha value and collect results for all plots.
    Replaces the former compute_coverage_and_setsize(), compute_mcp_roc(),
    and compute_alpha_metrics_sweep() — no redundant iterations.

    Returns
    -------
    dict with keys:
        labels, alphas,
        marginal_coverage, conditional_coverage,
        marginal_setsize,  conditional_setsize,
        f1_list, singleton_list, avset_list,
        mcp_fpr, mcp_tpr, mcp_auc
    """
    qhat_table = compute_qhat_sweep(calibration_data, alphas, nonconf_type=nonconf_type)
    return compute_sweep_from_qhat(test_data, qhat_table, nonconf_type=nonconf_type)


def compute_metrics_at_alpha(calibration_data, test_data, alpha, nonconf_type="baseline"):
    """
    Compute all key CP metrics at a single alpha in one pass.

    Returns
    -------
    dict: alpha, coverage, macro_f1, avg_set_size, singleton_rate,
          fpr, tpr, per_label_f1, n_cal, n_test, n_classes
    """
    qhat   = calibration(calibration_data, alpha=alpha, nonconf_type=nonconf_type)
    acc    = _metrics_pass(test_data, qhat, nonconf_type)
    m      = _summarise(acc)
    return {
        "alpha":          alpha,
        "n_cal":          len(calibration_data),
        "n_test":         acc["n"],
        "n_classes":      len(acc["labels"]),
        **{k: m[k] for k in ("coverage", "macro_f1", "avg_set_size",
                              "singleton_rate", "fpr", "tpr", "per_label_f1")},
    }


def compute_metrics_from_qhat(test_data, qhat_dict, alpha, nonconf_type="baseline", n_cal=None):
    """
    Compute all key CP metrics at a single alpha using a precomputed qhat dict.
    """
    acc = _metrics_pass(test_data, qhat_dict, nonconf_type)
    m = _summarise(acc)
    return {
        "alpha":          alpha,
        "n_cal":          n_cal,
        "n_test":         acc["n"],
        "n_classes":      len(acc["labels"]),
        **{k: m[k] for k in ("coverage", "macro_f1", "avg_set_size",
                              "singleton_rate", "fpr", "tpr", "per_label_f1")},
    }


def compute_macro_roc(test_data):
    """
    Macro-average ROC from raw model probability scores (one-vs-rest per class).

    Returns
    -------
    mean_fpr : np.ndarray
    mean_tpr : np.ndarray
    roc_auc  : float
    """
    labels  = sorted(test_data.columns[1:].tolist())
    y_true  = test_data["true_label"].values
    y_score = test_data[labels].values
    y_bin   = label_binarize(y_true, classes=labels)

    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    mean_fpr = np.linspace(0, 1, 300)
    tprs, aucs = [], []
    for i in range(len(labels)):
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_score[:, i])
        aucs.append(auc(fpr_i, tpr_i))
        interp    = np.interp(mean_fpr, fpr_i, tpr_i)
        interp[0] = 0.0
        tprs.append(interp)

    mean_tpr     = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    return mean_fpr, mean_tpr, float(np.mean(aucs))


# ===========================================================================
# Plotting functions
# ===========================================================================

def plot_coverage(labels, alphas, marginal_coverage, conditional_coverage,
                  n_test, save_path=None, show=True, ax=None):
    """
    Validity plot: empirical coverage vs 1-alpha, per label and marginal.
    Includes 99 % Wilson binomial CI band.
    """
    standalone = ax is None
    with _pub():
        if standalone:
            fig, ax = plt.subplots(figsize=(8, 5.5))

        x_norm, lo, hi = _wilson_ci_band(n_test)
        # ax.fill_between(x_norm, lo, hi, color="k", alpha=0.12,
        #                 label="99% CI (test set size)")

        cmap = plt.get_cmap("tab20", len(labels))
        for i, l in enumerate(labels):
            ax.plot(1 - alphas, conditional_coverage[l],
                    label=_fmt(l), color=cmap(i), linewidth=1.5)

        ax.plot(1 - alphas, marginal_coverage,
                label="Marginal", color="black", linewidth=2.2,
                linestyle="--", zorder=5)
        # ax.plot([0, 1], [0, 1], "k:", linewidth=1, alpha=0.5, label="Ideal")

        ax.set_xlabel(r"$1 - \alpha$ (coverage level)")
        ax.set_ylabel("Empirical coverage")
        ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
        ax.legend(loc="upper left", ncol=2)
        ax.grid(visible=True)
        # ax.set_title("Validity: Empirical Coverage vs. Coverage Level")

        if standalone:
            _save_or_show(fig, save_path, show)
            return fig, ax


def plot_set_size(labels, alphas, marginal_setsize, conditional_setsize,
                  cutoff=3, save_path=None, show=True, ax=None):
    """
    Efficiency plot: average prediction set size vs 1-alpha, per label and marginal.
    """
    standalone = ax is None
    with _pub():
        if standalone:
            fig, ax = plt.subplots(figsize=(8, 5.5))

        cmap = plt.get_cmap("tab20", len(labels))
        for i, l in enumerate(labels):
            ax.plot(1 - alphas[cutoff:], conditional_setsize[l][:-cutoff],
                    label=_fmt(l), color=cmap(i), linewidth=1.5)

        ax.plot(1 - alphas[cutoff:], marginal_setsize[:-cutoff],
                label="Marginal", color="black", linewidth=2.2,
                linestyle="--", zorder=5)

        ax.set_xlabel(r"$1 - \alpha$ (coverage level)")
        ax.set_ylabel("Average prediction set size")
        ax.legend(loc="upper left", ncol=2)
        ax.grid(visible=True)
        # ax.set_title("Efficiency: Average Prediction Set Size vs. Coverage Level")
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        if standalone:
            _save_or_show(fig, save_path, show)
            return fig, ax


def plot_roc(mcp_fpr, mcp_tpr, mcp_auc, alphas,
             macro_fpr, macro_tpr, macro_auc,
             save_path=None, show=True, ax=None):
    """
    ROC plot combining:
      - Macro ROC line from raw model probabilities (viridis, coloured by threshold)
      - MCP operating-point scatter (inferno, coloured by alpha)
    Dual colorbars: inner = error rate alpha, outer = decision threshold.
    """
    standalone = ax is None
    with _pub():
        if standalone:
            fig, ax = plt.subplots(figsize=(8, 5.5))

        # Threshold-coloured macro ROC line
        thresholds = np.linspace(1.0, 0.0, len(macro_fpr))
        points   = np.array([macro_fpr, macro_tpr]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm_t   = plt.Normalize(0.0, 1.0)
        lc = LineCollection(segments, cmap="viridis", norm=norm_t,
                            linewidth=2.5, zorder=3,
                            label=f"Macro ROC – model probabilities (AUC = {macro_auc:.3f})")
        lc.set_array(thresholds[:-1])
        ax.add_collection(lc)

        # Threshold colourbar (outer)
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm_t)
        sm.set_array([])
        cbar_t = plt.colorbar(sm, ax=ax, location="right", pad=0.06, aspect=30)
        cbar_t.set_label("Threshold", fontsize=11)
        cbar_t.ax.tick_params(labelsize=9)
        cbar_t.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

        # MCP scatter (inner)
        sc = ax.scatter(mcp_fpr, mcp_tpr, s=55, c=alphas, cmap="inferno",
                        zorder=4, edgecolors="white", linewidths=0.4,
                        label=f"MCP operating points (AUC ≈ {mcp_auc:.3f})")
        cbar_a = plt.colorbar(sc, ax=ax, location="right", pad=0.02, aspect=30)
        cbar_a.set_label(r"Error rate $\alpha$", fontsize=11)
        cbar_a.ax.tick_params(labelsize=9)

        ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.5,
                label="Random classifier")

        ax.set_xlabel("False Positive Rate  FP / (FP + TN)")
        ax.set_ylabel("True Positive Rate  TP / (TP + FN)")
        ax.set_xlim(-0.02, 1.02);  ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="lower right")
        ax.grid(visible=True)
        # ax.set_title("ROC Curve: MCP vs. Model Probabilities (Macro-Average)")

        if standalone:
            _save_or_show(fig, save_path, show)
            return fig, ax


# ===========================================================================
# Alpha vs metrics plot  (3-panel: F1 / singleton rate / avg set size)
# ===========================================================================

def plot_alpha_metrics(alphas, f1_list, singleton_list, avset_list,
                       save_path=None, show=True):
    """
    Three-panel figure: alpha vs macro-F1 | singleton rate | average set size.
    """
    with _pub():
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        # fig.suptitle(r"CP Metrics vs. Error Rate $\alpha$", fontsize=14, y=1.02)

        panels = [
            (axes[0], f1_list,        "Macro F1-score",
             "tab:blue",   r"Macro F1-score",             None),
            (axes[1], singleton_list, "Singleton rate",
             "tab:orange", r"Singleton rate",              None),
            (axes[2], avset_list,     "Average set size",
             "tab:green",  r"Average prediction set size", None),
        ]

        for ax, values, title, color, ylabel, ylim in panels:
            ax.plot(alphas, values, color=color, linewidth=2.0,
                    marker="o", markersize=4,
                    markerfacecolor="white", markeredgewidth=1.2)
            ax.set_xlabel(r"Error rate $\alpha$")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xlim(float(alphas[0]) - 0.01, float(alphas[-1]) + 0.01)
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(visible=True)

        _save_or_show(fig, save_path, show)
        return fig, axes


# ===========================================================================
# Fig. 2-style prediction-set scatter plot
# ===========================================================================

def plot_prediction_set_scatter(test_data, qhat_dict, labels, alpha,
                                nonconf_type="baseline", random_state=None,
                                prediction_label="Model prediction",
                                cp_set_label=None,
                                save_path=None, show=True, ax=None):
    """
    Fig. 2-style scatter plot:
      - x-axis: one random test sample per true label
      - dark blue point: model / point prediction
      - light blue points: labels included in the CP prediction set
    """
    sampled_rows = []
    rng = np.random.default_rng(random_state)

    for label in labels:
        label_rows = test_data[test_data["true_label"] == label]
        if label_rows.empty:
            continue
        row = label_rows.iloc[int(rng.integers(0, len(label_rows)))]
        sampled_rows.append((label, row))

    if not sampled_rows:
        raise ValueError("Test split is empty; cannot build a prediction-set scatter plot.")

    standalone = ax is None
    with _pub():
        if standalone:
            fig, ax = plt.subplots(figsize=(12, 9))

        ax.set_axisbelow(True)

        y_pos = {label: i for i, label in enumerate(labels)}
        cp_x, cp_y = [], []
        pred_x, pred_y = [], []

        if cp_set_label is None:
            cp_set_label = fr"CP set ($\alpha={alpha:.2f}$)"

        for x_pos, (_, row) in enumerate(sampled_rows):
            pred_set = testing(row, qhat_dict, nonconf_type)
            point_pred = row.iloc[1:].idxmax()

            cp_x.extend([x_pos] * len(pred_set))
            cp_y.extend([y_pos[label] for label in pred_set])
            pred_x.append(x_pos)
            pred_y.append(y_pos[point_pred])

        n_x = len(sampled_rows)
        ax.plot([-0.5, n_x - 0.5], [-0.5, n_x - 0.5],
                linestyle="--", linewidth=1.0, color="0.55", alpha=0.8,
                zorder=1)

        ax.scatter(cp_x, cp_y, s=220, c="#9ecae1", edgecolors="white",
                   linewidths=0.8, label=cp_set_label,
                   zorder=2)
        ax.scatter(pred_x, pred_y, s=60, c="#08306b",
                   label=prediction_label, zorder=3)

        ax.set_xlim(-0.5, n_x - 0.5)
        ax.set_ylim(-0.5, len(labels) - 0.5)
        ax.set_xticks(range(n_x))
        ax.set_xticklabels([_fmt(label) for label, _ in sampled_rows],
                           rotation=60, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([_fmt(label) for label in labels])
        ax.set_xlabel("True label (one random test sample per class)")
        ax.set_ylabel(f"{prediction_label} and labels in the CP set")
        ax.grid(visible=True, linewidth=0.5, alpha=0.2)
        ax.legend(loc="upper left")

        if standalone:
            _save_or_show(fig, save_path, show)
            return fig, ax


# ===========================================================================
# Metrics CSV export
# ===========================================================================

def export_metrics_csv(sweep, save_path):
    """
    Write the full alpha-sweep metrics to a CSV file.

    Columns: alpha, coverage, macro_f1, avg_set_size, singleton_rate,
             fpr, tpr, mcp_fpr, mcp_tpr  + one f1_<label> column per class.

    Parameters
    ----------
    sweep     : dict — result from compute_sweep()
    save_path : str  — output path (e.g. "results/mcp_metrics.csv")
    """
    alphas = sweep["alphas"]
    rows = []
    labels = sweep["labels"]

    for i, a in enumerate(alphas):
        row = {
            "alpha":           round(float(a), 6),
            "coverage":        round(sweep["marginal_coverage"][i], 6),
            "macro_f1":        round(sweep["f1_list"][i], 6),
            "avg_set_size":    round(sweep["avset_list"][i], 6),
            "singleton_rate":  round(sweep["singleton_list"][i], 6),
            "tpr":             round(float(sweep["mcp_tpr"][i]), 6),
            "fpr":             round(float(sweep["mcp_fpr"][i]), 6),
        }
        # Per-label conditional coverage
        for l in labels:
            row[f"coverage_{l}"] = round(sweep["conditional_coverage"][l][i], 6)
        # Per-label conditional avg set size
        for l in labels:
            row[f"setsize_{l}"] = round(sweep["conditional_setsize"][l][i], 6)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"  Saved: {save_path}")
    return df


# ===========================================================================
# High-level MCPTool class
# ===========================================================================

class MCPTool:
    """
    High-level interface for Mondrian Conformal Prediction analysis.

    Parameters
    ----------
    csv_path         : str or None — one combined CSV to split into calibration/test
    calibration_csv  : str or None — separate calibration CSV
    test_csv         : str or None — separate test CSV
    qhat_csv         : str or None — alpha-vs-qhat CSV produced from calibration data
    alpha            : float       — nominal error rate, default 0.1
    test_size        : float       — test fraction for combined CSV mode, default 0.5
    random_state     : int         — random seed for combined split, default 123
    n_alphas         : int         — alpha grid points for sweeps, default 51
    nonconf_type     : str         — nonconformity measure ('baseline' only)
    """

    def __init__(self, csv_path: str | None = None, alpha: float = 0.1,
                 test_size: float = 0.5, random_state: int = 123,
                 n_alphas: int = 51, nonconf_type: str = "baseline",
                 calibration_csv: str | None = None,
                 test_csv: str | None = None,
                 qhat_csv: str | None = None):
        self.csv_path         = csv_path
        self.calibration_csv  = calibration_csv
        self.test_csv         = test_csv
        self.qhat_csv         = qhat_csv
        self.alpha            = alpha
        self.test_size        = test_size
        self.random_state     = random_state
        self.n_alphas         = n_alphas
        self.nonconf_type     = nonconf_type
        self._mode            = self._resolve_mode()

        # Populated by load()
        self._data     = None
        self._cal      = None
        self._test     = None
        self._labels   = None
        self._n_cal    = None

        # Populated by run()
        self._qhat             = None
        self._qhat_table       = None
        self._qhat_alpha       = None
        self._alphas_sweep     = None
        self._sweep            = None
        self._macro_fpr        = None
        self._macro_tpr        = None
        self._macro_auc        = None
        self._metrics_at_alpha = None

    @classmethod
    def from_split(cls, calibration_csv: str, test_csv: str, **kwargs):
        """Construct MCPTool from separate calibration and test CSV files."""
        return cls(calibration_csv=calibration_csv, test_csv=test_csv, **kwargs)

    @classmethod
    def from_qhat_csv(cls, qhat_csv: str, test_csv: str, **kwargs):
        """Construct MCPTool from a qhat sweep CSV and a test CSV."""
        return cls(qhat_csv=qhat_csv, test_csv=test_csv, **kwargs)

    @classmethod
    def from_calibration_csv(cls, calibration_csv: str, **kwargs):
        """Construct MCPTool from calibration data only (for qhat export)."""
        return cls(calibration_csv=calibration_csv, **kwargs)

    def _resolve_mode(self):
        if self.csv_path and not any([self.calibration_csv, self.test_csv, self.qhat_csv]):
            return "combined"
        if self.calibration_csv and self.test_csv and not any([self.csv_path, self.qhat_csv]):
            return "split"
        if self.qhat_csv and self.test_csv and not any([self.csv_path, self.calibration_csv]):
            return "qhat"
        if self.calibration_csv and not any([self.csv_path, self.test_csv, self.qhat_csv]):
            return "calibration_only"
        raise ValueError(
            "Provide either: csv_path, or calibration_csv + test_csv, or qhat_csv + test_csv, "
            "or calibration_csv alone for qhat export."
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load(self):
        """Load and validate the configured input files."""
        if self._mode == "combined":
            df, labels = load_prediction_csv(self.csv_path)
            self._data = df
            self._cal, self._test = train_test_split(
                df, test_size=self.test_size, random_state=self.random_state
            )
            self._labels = labels
            self._n_cal = len(self._cal)
        elif self._mode == "split":
            self._cal, labels = load_prediction_csv(self.calibration_csv)
            self._test, self._labels = load_prediction_csv(self.test_csv, labels=labels)
            self._n_cal = len(self._cal)
        elif self._mode == "qhat":
            self._qhat_table, qhat_labels = load_qhat_csv(self.qhat_csv)
            self._test, self._labels = load_prediction_csv(self.test_csv, labels=qhat_labels)
            if "n_cal" in self._qhat_table.columns:
                n_cal_values = self._qhat_table["n_cal"].dropna().unique()
                self._n_cal = int(n_cal_values[0]) if len(n_cal_values) else None
        elif self._mode == "calibration_only":
            self._cal, self._labels = load_prediction_csv(self.calibration_csv)
            self._n_cal = len(self._cal)
        return self

    # ------------------------------------------------------------------
    # Core run
    # ------------------------------------------------------------------

    def run(self):
        """
        Load data, run the alpha sweep, compute macro ROC, and print metrics.
        Returns self for method chaining.
        """
        if self._mode == "calibration_only":
            raise RuntimeError("Calibration-only mode can export qhat CSVs but cannot run plots without test data.")
        if self._labels is None:
            self.load()

        if self._mode == "combined":
            print(f"Dataset     : {len(self._data)} samples, {len(self._labels)} classes")
            print(f"Split       : {len(self._cal)} calibration | {len(self._test)} test")
        elif self._mode == "split":
            print(f"Calibration : {len(self._cal)} samples")
            print(f"Test        : {len(self._test)} samples")
        elif self._mode == "qhat":
            print(f"Qhat rows   : {len(self._qhat_table)} alpha values")
            print(f"Test        : {len(self._test)} samples")
        print(f"Classes     : {self._labels}\n")

        # Prepare qhat sweep
        if self._mode in {"combined", "split"}:
            self._alphas_sweep = make_alpha_grid(self.n_alphas, self.alpha)
            self._qhat_table = compute_qhat_sweep(
                self._cal, self._alphas_sweep, self.nonconf_type
            )
            self._qhat, self._qhat_alpha = select_qhat_for_alpha(
                self._qhat_table, self._labels, self.alpha, allow_nearest=False
            )
        else:
            self._alphas_sweep = self._qhat_table["alpha"].to_numpy(dtype=float)
            self._qhat, self._qhat_alpha = select_qhat_for_alpha(
                self._qhat_table, self._labels, self.alpha, allow_nearest=True
            )

        # Single unified sweep from qhat table
        print("Running unified alpha sweep …")
        self._sweep = compute_sweep_from_qhat(
            self._test, self._qhat_table, self.nonconf_type
        )

        # Macro ROC from raw probabilities
        print("Computing macro ROC …")
        self._macro_fpr, self._macro_tpr, self._macro_auc = \
            compute_macro_roc(self._test)

        self._metrics_at_alpha = compute_metrics_from_qhat(
            self._test, self._qhat, self._qhat_alpha,
            nonconf_type=self.nonconf_type, n_cal=self._n_cal
        )

        print(f"\nMacro AUC = {self._macro_auc:.4f} | "
              f"MCP AUC ≈ {self._sweep['mcp_auc']:.4f}\n")
        self.print_metrics()
        return self

    # ------------------------------------------------------------------
    # Metrics CSV export
    # ------------------------------------------------------------------

    def export_metrics_csv(self, save_path=None):
        """
        Export the full alpha-sweep metrics to a CSV file.

        Columns: alpha, coverage, macro_f1, avg_set_size, singleton_rate,
                 fpr, tpr, + coverage_<label> and setsize_<label> per class.

        Parameters
        ----------
        save_path : str or None
            Output path. Defaults to '<csv_stem>_metrics.csv' next to the input file.

        Returns
        -------
        pd.DataFrame
        """
        self._check_run()
        if save_path is None:
            import os
            source = self.csv_path or self.test_csv or self.calibration_csv or self.qhat_csv
            stem = os.path.splitext(os.path.basename(source))[0]
            save_path = os.path.join(os.path.dirname(source) or ".",
                                     f"{stem}_metrics.csv")
        return export_metrics_csv(self._sweep, save_path)

    def export_qhat_csv(self, save_path=None):
        """
        Export the alpha-vs-qhat table to CSV.
        """
        if self._labels is None and self._cal is None and self._qhat_table is None:
            self.load()
        if self._qhat_table is None:
            if self._cal is None:
                raise RuntimeError("No calibration data available to build a qhat CSV.")
            self._alphas_sweep = make_alpha_grid(self.n_alphas, self.alpha)
            self._qhat_table = compute_qhat_sweep(
                self._cal, self._alphas_sweep, self.nonconf_type
            )
        if save_path is None:
            import os
            source = self.csv_path or self.calibration_csv or self.qhat_csv or "mcp"
            stem = os.path.splitext(os.path.basename(source))[0]
            base_dir = os.path.dirname(source) or "."
            save_path = os.path.join(base_dir, f"{stem}_qhat.csv")
        return export_qhat_csv(self._qhat_table, save_path)

    def export_split_csvs(self, calibration_path=None, test_path=None):
        """
        Export the loaded calibration/test split to separate CSV files.

        This is available for combined-CSV mode after the internal split is made,
        and for explicit split mode after the input files are loaded.
        """
        if self._cal is None and self._test is None:
            self.load()
        if self._cal is None:
            raise RuntimeError("No calibration data available to export.")
        if self._test is None:
            raise RuntimeError("No test data available to export.")

        import os

        source = self.csv_path or self.calibration_csv or self.test_csv or self.qhat_csv or "mcp"
        base_dir = os.path.dirname(source) or "."
        calibration_path = calibration_path or os.path.join(base_dir, "calibration.csv")
        test_path = test_path or os.path.join(base_dir, "test.csv")

        self._cal.to_csv(calibration_path, index=False)
        self._test.to_csv(test_path, index=False)
        print(f"  Saved: {calibration_path}")
        print(f"  Saved: {test_path}")
        return calibration_path, test_path

    # ------------------------------------------------------------------
    # Metrics summary
    # ------------------------------------------------------------------

    def _check_run(self):
        if self._qhat is None or self._test is None:
            raise RuntimeError("Call .run() before using this method.")

    def print_metrics(self, alpha=None):
        """
        Print a formatted metrics summary table.

        Parameters
        ----------
        alpha : float or None
            If given (and different from self.alpha), recomputes metrics at
            that alpha.  Otherwise uses the cached result from run().
        """
        self._check_run()
        if alpha is not None and not np.isclose(alpha, self._metrics_at_alpha["alpha"]):
            if self._cal is not None:
                m = compute_metrics_at_alpha(
                    self._cal, self._test, alpha, self.nonconf_type
                )
            else:
                qhat, chosen_alpha = select_qhat_for_alpha(
                    self._qhat_table, self._labels, alpha, allow_nearest=True
                )
                m = compute_metrics_from_qhat(
                    self._test, qhat, chosen_alpha,
                    nonconf_type=self.nonconf_type, n_cal=self._n_cal
                )
        else:
            m = self._metrics_at_alpha

        sep = "-" * 47
        print(sep)
        print(f"  MCP Metrics Summary   alpha = {m['alpha']:.4f}")
        print(sep)
        print(f"  Calibration samples : {m['n_cal'] if m['n_cal'] is not None else 'unknown'}")
        print(f"  Test samples        : {m['n_test']}")
        print(f"  Classes             : {m['n_classes']}")
        print(sep)
        print(f"  Coverage            : {m['coverage']:.4f}"
              f"   (target >= {1 - m['alpha']:.4f})")
        print(f"  Macro F1-score      : {m['macro_f1']:.4f}")
        print(f"  Average set size    : {m['avg_set_size']:.4f}")
        print(f"  Singleton rate      : {m['singleton_rate']:.4f}")
        print(f"  True positive rate  : {m['tpr']:.4f}")
        print(f"  False positive rate : {m['fpr']:.4f}")
        print(sep)
        print("  Per-class F1:")
        for label, score in sorted(m["per_label_f1"].items()):
            print(f"    {_fmt(label):<28} {score:.4f}")
        print(sep)
        return m

    # ------------------------------------------------------------------
    # Plotting wrappers  — thin delegates to module-level plot functions
    # ------------------------------------------------------------------

    def plot_coverage(self, save_path=None, show=True, ax=None):
        """Validity / coverage plot."""
        self._check_run()
        s = self._sweep
        return plot_coverage(
            s["labels"], s["alphas"],
            s["marginal_coverage"], s["conditional_coverage"],
            n_test=len(self._test),
            save_path=save_path, show=show, ax=ax)

    def plot_set_size(self, save_path=None, show=True, ax=None, cutoff=3):
        """Efficiency / set-size plot."""
        self._check_run()
        s = self._sweep
        return plot_set_size(
            s["labels"], s["alphas"],
            s["marginal_setsize"], s["conditional_setsize"],
            cutoff=cutoff,
            save_path=save_path, show=show, ax=ax)

    def plot_roc(self, save_path=None, show=True, ax=None):
        """ROC plot with dual colourbars (threshold + alpha)."""
        self._check_run()
        s = self._sweep
        return plot_roc(
            s["mcp_fpr"], s["mcp_tpr"], s["mcp_auc"], s["alphas"],
            self._macro_fpr, self._macro_tpr, self._macro_auc,
            save_path=save_path, show=show, ax=ax)

    def plot_alpha_metrics(self, save_path=None, show=True):
        """Three-panel plot: alpha vs macro-F1 | singleton rate | average set size."""
        self._check_run()
        s = self._sweep
        return plot_alpha_metrics(
            s["alphas"], s["f1_list"], s["singleton_list"], s["avset_list"],
            save_path=save_path, show=show)

    def plot_prediction_set_scatter(self, save_path=None, show=True, ax=None,
                                    random_state=None,
                                    prediction_label="Model prediction",
                                    cp_set_label=None):
        """Fig. 2-style random test-sample scatter plot."""
        self._check_run()
        return plot_prediction_set_scatter(
            self._test, self._qhat, self._labels, self._qhat_alpha,
            nonconf_type=self.nonconf_type,
            random_state=self.random_state if random_state is None else random_state,
            prediction_label=prediction_label,
            cp_set_label=cp_set_label,
            save_path=save_path, show=show, ax=ax)

    # Convenient alias for notebook usage
    plot_fig2_scatter = plot_prediction_set_scatter

    def plot_all(self, output_dir=".", prefix="mcp", show=True,
                 include_fig2_scatter=False,
                 prediction_label="Model prediction",
                 cp_set_label=None):
        """
        Save all four plots and the metrics CSV to output_dir.
        Files written: <prefix>_coverage.pdf, <prefix>_setsize.pdf,
                       <prefix>_roc.pdf, <prefix>_alpha_metrics.pdf,
                       <prefix>_metrics.csv
        Set include_fig2_scatter=True to also save
        <prefix>_fig2_scatter.pdf. The Fig. 2-style scatter legend for the
        dark-blue point uses prediction_label.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        for name, method in [
            ("coverage",      self.plot_coverage),
            ("setsize",       self.plot_set_size),
            ("roc",           self.plot_roc),
            ("alpha_metrics", self.plot_alpha_metrics),
        ]:
            method(save_path=os.path.join(output_dir, f"{prefix}_{name}.pdf"),
                   show=show)
        if include_fig2_scatter:
            self.plot_prediction_set_scatter(
                save_path=os.path.join(output_dir, f"{prefix}_fig2_scatter.pdf"),
                show=show,
                prediction_label=prediction_label,
                cp_set_label=cp_set_label)
        self.export_metrics_csv(
            save_path=os.path.join(output_dir, f"{prefix}_metrics.csv"))

    # ------------------------------------------------------------------
    # Single-sample prediction
    # ------------------------------------------------------------------

    def predict_one(self, index=0):
        """
        Print and return the CP prediction set for one test sample.

        Parameters
        ----------
        index : int — row index within the test set (0-based)
        """
        self._check_run()
        row      = self._test.iloc[index]
        pred_set = testing(row, self._qhat, self.nonconf_type)
        ml_pred  = row.iloc[1:].idxmax()
        print(f"True label       : {row['true_label']}")
        print(f"ML prediction    : {ml_pred}")
        print(f"CP prediction set (alpha={self.alpha}): {pred_set}")
        return pred_set
