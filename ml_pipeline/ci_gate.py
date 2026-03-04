from __future__ import annotations

import argparse
import os
from typing import Iterable

from ml_pipeline.evaluate_coverage import _sample_users, run_coverage_evaluation
from ml_pipeline.evaluate_holdout import run_holdout_evaluation
from ml_pipeline.prediction_core import PredictionArtifacts


def _parse_int_list(vals: Iterable[int]) -> list[int]:
    return [int(v) for v in vals]


def _get_threshold_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def main() -> int:
    parser = argparse.ArgumentParser("CI gate for IAM model metrics")
    parser.add_argument("--train", action="store_true", help="Run training before evaluation")
    parser.add_argument("--out-dir", type=str, default="artifacts")

    # Holdout evaluation params
    parser.add_argument("--holdout-users", type=int, default=100)
    parser.add_argument("--holdout-k", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--holdout-k-target", type=int, default=10)
    parser.add_argument("--holdout-ratio", type=float, default=0.3)
    parser.add_argument("--min-truth", type=int, default=3)
    parser.add_argument("--initial-candidates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    # Coverage evaluation params
    parser.add_argument("--coverage-users", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=20)

    # Thresholds (can be overridden via env vars)
    parser.add_argument("--min-precision", type=float, default=_get_threshold_env("IAM_MIN_PRECISION", 0.05))
    parser.add_argument("--min-recall", type=float, default=_get_threshold_env("IAM_MIN_RECALL", 0.05))
    parser.add_argument("--min-user-coverage", type=float, default=_get_threshold_env("IAM_MIN_USER_COVERAGE", 0.80))
    parser.add_argument("--min-catalog-coverage", type=float, default=_get_threshold_env("IAM_MIN_CATALOG_COVERAGE", 0.02))
    parser.add_argument("--max-gini", type=float, default=_get_threshold_env("IAM_MAX_GINI", 0.98))

    args = parser.parse_args()

    if args.train:
        from ml_pipeline.train import run_training
        run_training()

    # Holdout evaluation gate
    summary_df, _ = run_holdout_evaluation(
        users=args.holdout_users,
        k_values=_parse_int_list(args.holdout_k),
        holdout_ratio=args.holdout_ratio,
        min_truth=args.min_truth,
        initial_candidates=args.initial_candidates,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    target_k = int(args.holdout_k_target)
    target_row = summary_df[summary_df["k"] == target_k]
    if target_row.empty:
        print(f"FAIL: Holdout summary missing k={target_k}")
        return 2

    precision = float(target_row["mean_precision_at_k"].iloc[0])
    recall = float(target_row["mean_recall_at_k"].iloc[0])

    # Coverage evaluation gate
    artifacts = PredictionArtifacts.get_artifacts()
    users_df = artifacts["graph_dfs"]["users"]
    sampled_users = _sample_users(users_df, args.coverage_users, args.seed)
    coverage_metrics = run_coverage_evaluation(
        users=sampled_users,
        artifacts=artifacts,
        top_n=args.top_n,
        initial_candidates=args.initial_candidates,
        seed=args.seed,
        out_dir=args.out_dir,
    )

    failures = []
    if precision < args.min_precision:
        failures.append(f"precision@{target_k} {precision:.4f} < {args.min_precision:.4f}")
    if recall < args.min_recall:
        failures.append(f"recall@{target_k} {recall:.4f} < {args.min_recall:.4f}")
    if coverage_metrics["user_coverage"] < args.min_user_coverage:
        failures.append(
            f"user_coverage {coverage_metrics['user_coverage']:.4f} < {args.min_user_coverage:.4f}"
        )
    if coverage_metrics["catalog_coverage"] < args.min_catalog_coverage:
        failures.append(
            f"catalog_coverage {coverage_metrics['catalog_coverage']:.4f} < {args.min_catalog_coverage:.4f}"
        )
    if coverage_metrics["gini_coefficient"] > args.max_gini:
        failures.append(
            f"gini {coverage_metrics['gini_coefficient']:.4f} > {args.max_gini:.4f}"
        )

    if failures:
        print("CI GATE: FAIL")
        for f in failures:
            print(f" - {f}")
        return 2

    print("CI GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
