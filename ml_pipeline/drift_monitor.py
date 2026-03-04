from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import pandas as pd

from ml_pipeline.prediction_core import PredictionArtifacts, run_prediction_pipeline


def _load_baseline(path: str) -> dict[str, dict[str, float]]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _compute_feature_stats(X: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats = {}
    if X is None or X.empty:
        return stats
    desc = X.describe(percentiles=[0.5, 0.95]).T
    for col, row in desc.iterrows():
        stats[str(col)] = {
            "mean": float(row.get("mean", 0.0)),
            "std": float(row.get("std", 0.0)),
            "p50": float(row.get("50%", 0.0)),
            "p95": float(row.get("95%", 0.0)),
        }
    return stats


def _relative_delta(new: float, base: float) -> float:
    denom = abs(base) if abs(base) > 1e-9 else 1.0
    return (new - base) / denom


def _compare_stats(
    current: dict[str, dict[str, float]],
    baseline: dict[str, dict[str, float]],
    mean_threshold: float,
    std_threshold: float,
) -> list[dict[str, Any]]:
    drifted = []
    for feature, cur in current.items():
        base = baseline.get(feature)
        if not base:
            continue
        mean_delta = _relative_delta(cur.get("mean", 0.0), base.get("mean", 0.0))
        std_delta = _relative_delta(cur.get("std", 0.0), base.get("std", 0.0))
        if abs(mean_delta) >= mean_threshold or abs(std_delta) >= std_threshold:
            drifted.append(
                {
                    "feature": feature,
                    "mean_delta": float(mean_delta),
                    "std_delta": float(std_delta),
                    "base_mean": base.get("mean", 0.0),
                    "cur_mean": cur.get("mean", 0.0),
                    "base_std": base.get("std", 0.0),
                    "cur_std": cur.get("std", 0.0),
                }
            )
    return drifted


def _sample_users(users_df: pd.DataFrame, sample_n: int, seed: int) -> list[int]:
    ids = pd.to_numeric(users_df["id"], errors="coerce").dropna().astype("int64")
    ids = ids[ids != 1].unique().tolist()
    rng = np.random.default_rng(seed)
    if sample_n < len(ids):
        return rng.choice(ids, size=sample_n, replace=False).tolist()
    return ids


def _collect_feature_matrices(user_ids: list[int], top_n: int, initial_candidates: int) -> dict[str, pd.DataFrame]:
    cand_frames = []
    rerank_frames = []
    for uid in user_ids:
        res = run_prediction_pipeline(
            user_id=int(uid),
            top_n=top_n,
            initial_candidates=initial_candidates,
        )
        if res is None:
            continue
        if res.get("X_cand") is not None:
            cand_frames.append(res["X_cand"])
        if res.get("X_rerank") is not None:
            rerank_frames.append(res["X_rerank"])
    X_cand = pd.concat(cand_frames, ignore_index=True) if cand_frames else pd.DataFrame()
    X_rerank = pd.concat(rerank_frames, ignore_index=True) if rerank_frames else pd.DataFrame()
    return {"candidate": X_cand, "reranker": X_rerank}


def main() -> int:
    parser = argparse.ArgumentParser("Drift monitoring for IAM recommender")
    parser.add_argument("--out-dir", type=str, default="artifacts")
    parser.add_argument("--sample-users", type=int, default=50)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--initial-candidates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mean-threshold", type=float, default=0.25)
    parser.add_argument("--std-threshold", type=float, default=0.50)
    args = parser.parse_args()

    artifacts = PredictionArtifacts.get_artifacts()
    users_df = artifacts["graph_dfs"]["users"]
    sampled_users = _sample_users(users_df, args.sample_users, args.seed)

    baseline_cand = _load_baseline(os.path.join(args.out_dir, "drift_baseline_candidate.json"))
    baseline_rerank = _load_baseline(os.path.join(args.out_dir, "drift_baseline_reranker.json"))

    matrices = _collect_feature_matrices(sampled_users, args.top_n, args.initial_candidates)
    current_cand = _compute_feature_stats(matrices["candidate"])
    current_rerank = _compute_feature_stats(matrices["reranker"])

    drift_cand = _compare_stats(current_cand, baseline_cand, args.mean_threshold, args.std_threshold)
    drift_rerank = _compare_stats(current_rerank, baseline_rerank, args.mean_threshold, args.std_threshold)

    report = {
        "sample_users": len(sampled_users),
        "mean_threshold": args.mean_threshold,
        "std_threshold": args.std_threshold,
        "candidate": {
            "features_checked": len(current_cand),
            "drifted_features": drift_cand,
        },
        "reranker": {
            "features_checked": len(current_rerank),
            "drifted_features": drift_rerank,
        },
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "drift_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nDRIFT REPORT")
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {out_path}")

    if drift_cand or drift_rerank:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
