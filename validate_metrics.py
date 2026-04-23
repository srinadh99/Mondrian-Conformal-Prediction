#!/usr/bin/env python3
"""
Independent metric validation for the MCP tool.

Checks:
  - summary metrics at a chosen alpha
  - macro ROC AUC against an independent one-vs-rest calculation
  - MCP ROC AUC against an independent sweep recomputation
"""

import argparse
import math

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from mcp import MCPTool, calibration, testing


def independent_metrics(test_data, qhat, labels):
    total_tp = total_fp = total_fn = total_tn = 0
    correct = single = all_lengths = 0
    per_label = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}

    for _, row in test_data.iterrows():
        true_label = row["true_label"]
        pred_set = testing(row, qhat)
        n_set = len(pred_set)

        all_lengths += n_set
        if true_label in pred_set:
            correct += 1
        if n_set == 1 and pred_set[0] == true_label:
            single += 1

        for label in labels:
            in_set = label in pred_set
            is_true = label == true_label
            total_tp += int(in_set and is_true)
            total_fp += int(in_set and not is_true)
            total_fn += int((not in_set) and is_true)
            total_tn += int((not in_set) and (not is_true))
            per_label[label]["tp"] += int(in_set and is_true)
            per_label[label]["fp"] += int(in_set and not is_true)
            per_label[label]["fn"] += int((not in_set) and is_true)

    n_test = len(test_data)
    per_label_f1 = {}
    for label, stats in per_label.items():
        denom = 2 * stats["tp"] + stats["fp"] + stats["fn"]
        per_label_f1[label] = 0.0 if denom == 0 else (2 * stats["tp"] / denom)

    return {
        "coverage": correct / n_test,
        "avg_set_size": all_lengths / n_test,
        "singleton_rate": single / n_test,
        "tpr": total_tp / (total_tp + total_fn),
        "fpr": total_fp / (total_fp + total_tn),
        "macro_f1": float(np.mean(list(per_label_f1.values()))),
    }


def independent_macro_auc(test_data, labels):
    y_true = test_data["true_label"].values
    y_score = test_data[labels].values
    y_bin = label_binarize(y_true, classes=labels)
    per_class_aucs = [
        roc_auc_score(y_bin[:, i], y_score[:, i]) for i in range(len(labels))
    ]
    return float(np.mean(per_class_aucs))


def independent_mcp_auc(cal_data, test_data, labels, alphas):
    points = []
    for alpha in alphas:
        qhat = calibration(cal_data, alpha=alpha)
        m = independent_metrics(test_data, qhat, labels)
        points.append((m["fpr"], m["tpr"]))
    points = np.asarray(points)
    order = np.argsort(points[:, 0])
    fpr_s = np.concatenate([[0.0], points[order, 0], [1.0]])
    tpr_s = np.concatenate([[0.0], points[order, 1], [1.0]])
    return float(auc(fpr_s, tpr_s))


def check_close(name, actual, expected, atol):
    diff = abs(actual - expected)
    ok = math.isfinite(actual) and math.isfinite(expected) and diff <= atol
    status = "OK" if ok else "FAIL"
    print(
        f"{status:4s}  {name:<20} actual={actual:.12f}  expected={expected:.12f}  diff={diff:.3e}"
    )
    return ok


def main():
    parser = argparse.ArgumentParser(description="Independently validate MCP metrics.")
    parser.add_argument("csv", help="Input CSV with true_label and per-class scores.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Alpha to validate.")
    parser.add_argument("--test-size", type=float, default=0.5, help="Test split fraction.")
    parser.add_argument("--random-state", type=int, default=123, help="Split seed.")
    parser.add_argument("--n-alphas", type=int, default=51, help="Sweep points for MCP AUC.")
    parser.add_argument("--atol", type=float, default=1e-12, help="Absolute tolerance.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    cal_data, test_data = train_test_split(
        df, test_size=args.test_size, random_state=args.random_state
    )
    labels = sorted(df.columns[1:].tolist())

    mcp = MCPTool(
        args.csv,
        alpha=args.alpha,
        test_size=args.test_size,
        random_state=args.random_state,
        n_alphas=args.n_alphas,
    ).run()

    qhat = calibration(cal_data, alpha=args.alpha)
    expected = independent_metrics(test_data, qhat, labels)
    expected_macro_auc = independent_macro_auc(test_data, labels)
    expected_mcp_auc = independent_mcp_auc(
        cal_data, test_data, labels, np.linspace(0.01, 0.99, args.n_alphas)
    )

    checks = [
        check_close("coverage", mcp._metrics_at_alpha["coverage"], expected["coverage"], args.atol),
        check_close("macro_f1", mcp._metrics_at_alpha["macro_f1"], expected["macro_f1"], args.atol),
        check_close("avg_set_size", mcp._metrics_at_alpha["avg_set_size"], expected["avg_set_size"], args.atol),
        check_close("singleton_rate", mcp._metrics_at_alpha["singleton_rate"], expected["singleton_rate"], args.atol),
        check_close("tpr", mcp._metrics_at_alpha["tpr"], expected["tpr"], args.atol),
        check_close("fpr", mcp._metrics_at_alpha["fpr"], expected["fpr"], args.atol),
        check_close("macro_roc_auc", mcp._macro_auc, expected_macro_auc, args.atol),
        check_close("mcp_auc", mcp._sweep["mcp_auc"], expected_mcp_auc, args.atol),
    ]

    if not all(checks):
        raise SystemExit(1)

    print("\nMetric validation passed.")


if __name__ == "__main__":
    main()
