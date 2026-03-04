from __future__ import annotations

import argparse
import json
import os
from typing import Iterable

import pandas as pd

from ml_pipeline.evaluate_holdout import run_holdout_evaluation
from ml_pipeline.prediction_core import PredictionArtifacts


def _parse_cols(raw: str) -> list[str]:
    return [c.strip() for c in raw.split(",") if c.strip()]


def _group_metrics(per_user_df: pd.DataFrame, users_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    users = users_df[["id", group_col]].copy()
    users = users.rename(columns={"id": "user_id"})
    merged = per_user_df.merge(users, on="user_id", how="left")
    merged = merged.dropna(subset=[group_col])
    if merged.empty:
        return pd.DataFrame()

    grouped = (
        merged.groupby([group_col, "k"])
        .agg(
            users=("user_id", "nunique"),
            mean_precision_at_k=("precision_at_k", "mean"),
            mean_recall_at_k=("recall_at_k", "mean"),
        )
        .reset_index()
    )
    return grouped


def _summarize_disparity(grouped: pd.DataFrame) -> dict:
    if grouped.empty:
        return {"status": "skipped", "reason": "no_groups"}
    summary = {}
    for k, df_k in grouped.groupby("k"):
        p_min = float(df_k["mean_precision_at_k"].min())
        p_max = float(df_k["mean_precision_at_k"].max())
        r_min = float(df_k["mean_recall_at_k"].min())
        r_max = float(df_k["mean_recall_at_k"].max())
        summary[int(k)] = {
            "precision_min": p_min,
            "precision_max": p_max,
            "precision_gap": p_max - p_min,
            "recall_min": r_min,
            "recall_max": r_max,
            "recall_gap": r_max - r_min,
        }
    return {"status": "ok", "by_k": summary}


def main() -> None:
    parser = argparse.ArgumentParser("Fairness evaluation for IAM recommender")
    parser.add_argument("--out-dir", type=str, default="artifacts")
    parser.add_argument("--groups", type=str, default="NOrganisationId,NBusinessRoleId,ManagerId")
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--k-target", type=int, default=10)
    parser.add_argument("--holdout-ratio", type=float, default=0.3)
    parser.add_argument("--min-truth", type=int, default=3)
    parser.add_argument("--initial-candidates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-precision-gap", type=float, default=0.30)
    parser.add_argument("--max-recall-gap", type=float, default=0.30)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    summary_df, per_user_df = run_holdout_evaluation(
        users=args.users,
        k_values=args.k,
        holdout_ratio=args.holdout_ratio,
        min_truth=args.min_truth,
        initial_candidates=args.initial_candidates,
        seed=args.seed,
        out_dir=args.out_dir,
    )

    artifacts = PredictionArtifacts.get_artifacts()
    users_df = artifacts["graph_dfs"]["users"]
    group_cols = [c for c in _parse_cols(args.groups) if c in users_df.columns]

    fairness_summary = {
        "groups": {},
        "holdout_summary_path": os.path.join(args.out_dir, "metrics_holdout_summary.csv"),
        "holdout_per_user_path": os.path.join(args.out_dir, "metrics_holdout_per_user.csv"),
        "thresholds": {
            "k_target": int(args.k_target),
            "max_precision_gap": float(args.max_precision_gap),
            "max_recall_gap": float(args.max_recall_gap),
        },
    }

    failures = []

    for col in group_cols:
        grouped = _group_metrics(per_user_df, users_df, col)
        out_path = os.path.join(args.out_dir, f"metrics_fairness_by_{col}.csv")
        if not grouped.empty:
            grouped.to_csv(out_path, index=False)
        disparity = _summarize_disparity(grouped)
        fairness_summary["groups"][col] = {
            "status": "ok" if not grouped.empty else "skipped",
            "path": out_path if not grouped.empty else None,
            "disparity": disparity,
        }
        if disparity.get("status") == "ok":
            by_k = disparity.get("by_k", {})
            k_data = by_k.get(int(args.k_target))
            if k_data:
                if k_data["precision_gap"] > args.max_precision_gap:
                    failures.append(
                        f"{col}: precision_gap {k_data['precision_gap']:.4f} > {args.max_precision_gap:.4f}"
                    )
                if k_data["recall_gap"] > args.max_recall_gap:
                    failures.append(
                        f"{col}: recall_gap {k_data['recall_gap']:.4f} > {args.max_recall_gap:.4f}"
                    )

    summary_path = os.path.join(args.out_dir, "metrics_fairness_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(fairness_summary, f, indent=2)

    print("\nFAIRNESS SUMMARY")
    print(json.dumps(fairness_summary, indent=2))
    print(f"\nSaved: {summary_path}")
    if failures:
        print("\nFAIRNESS GATE: FAIL")
        for f in failures:
            print(f" - {f}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
