from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

from ml_pipeline.prediction_core import PredictionArtifacts, run_prediction_pipeline


def _gini(values: list[int]) -> float:
    if not values:
        return 0.0
    arr = np.array(sorted(values), dtype=float)
    if np.sum(arr) == 0:
        return 0.0
    n = len(arr)
    return float((2 * np.sum((np.arange(1, n + 1) * arr)) - (n + 1) * np.sum(arr)) / (n * np.sum(arr)))


def run_coverage_evaluation(
    users: list[int],
    artifacts: dict,
    top_n: int,
    initial_candidates: int,
    seed: int,
    out_dir: str,
) -> dict:
    rng = np.random.default_rng(seed)
    sampled_users = users.copy()
    rng.shuffle(sampled_users)

    all_recommended: list[str] = []
    users_with_recs = 0
    users_failed = 0

    for uid in sampled_users:
        try:
            res = run_prediction_pipeline(
                user_id=int(uid),
                top_n=top_n,
                initial_candidates=initial_candidates,
            )
            if res is None:
                continue
            recs = res["predictions"]["EntitlementId"].astype(str).tolist()
            if recs:
                users_with_recs += 1
                all_recommended.extend(recs)
        except Exception:
            users_failed += 1

    entitlements_df = artifacts["graph_dfs"]["entitlements"]
    all_entitlements = set(entitlements_df["id"].astype(str).tolist())

    counts = Counter(all_recommended)
    unique_recommended = set(counts.keys())
    never_recommended = all_entitlements - unique_recommended

    metrics = {
        "users_evaluated": len(sampled_users),
        "users_with_recommendations": users_with_recs,
        "users_failed": users_failed,
        "user_coverage": (users_with_recs / len(sampled_users)) if sampled_users else 0.0,
        "total_entitlements": len(all_entitlements),
        "unique_recommended_entitlements": len(unique_recommended),
        "catalog_coverage": (len(unique_recommended) / len(all_entitlements)) if all_entitlements else 0.0,
        "total_recommendations": len(all_recommended),
        "gini_coefficient": _gini(list(counts.values())),
        "never_recommended_count": len(never_recommended),
        "top_10_recommended": counts.most_common(10),
    }

    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "coverage_summary.json")
    freq_path = os.path.join(out_dir, "coverage_frequency.csv")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    freq_df = pd.DataFrame(
        [{"entitlement_id": k, "count": v} for k, v in counts.most_common()]
    )
    freq_df.to_csv(freq_path, index=False)

    print("\nCOVERAGE METRICS")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {freq_path}")
    return metrics


def _sample_users(users_df: pd.DataFrame, sample_n: int, seed: int) -> list[int]:
    ids = pd.to_numeric(users_df["id"], errors="coerce").dropna().astype("int64")
    ids = ids[ids != 1].unique().tolist()
    rng = np.random.default_rng(seed)
    if sample_n < len(ids):
        return rng.choice(ids, size=sample_n, replace=False).tolist()
    return ids


def main() -> None:
    parser = argparse.ArgumentParser("Coverage/diversity evaluation for IAM recommender")
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--initial-candidates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="artifacts")
    args = parser.parse_args()

    artifacts = PredictionArtifacts.get_artifacts()
    users_df = artifacts["graph_dfs"]["users"]
    sampled = _sample_users(users_df, args.users, args.seed)

    run_coverage_evaluation(
        users=sampled,
        artifacts=artifacts,
        top_n=args.top_n,
        initial_candidates=args.initial_candidates,
        seed=args.seed,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
