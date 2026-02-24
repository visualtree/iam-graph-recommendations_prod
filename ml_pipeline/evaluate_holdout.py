from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml_pipeline import feature_engineering
from ml_pipeline.prediction_core import PredictionArtifacts, _hard_fail_feature_alignment


@dataclass
class HoldoutResult:
    user_id: int
    k: int
    hits: int
    precision_at_k: float
    recall_at_k: float
    hidden_size: int
    visible_size: int
    total_size: int


def _precision_recall_at_k(recommended_ids: list[str], true_hidden_ids: list[str], k: int) -> tuple[int, float, float]:
    rec_k = [str(x).strip() for x in recommended_ids[:k]]
    true_set = {str(x).strip() for x in true_hidden_ids}
    hits = len(set(rec_k) & true_set)
    precision = hits / k if k > 0 else 0.0
    recall = hits / len(true_set) if len(true_set) > 0 else 0.0
    return hits, precision, recall


def _predict_with_visible_only(
    artifacts: dict,
    user_id: int,
    visible_set: set[str],
    top_n: int,
    initial_candidates: int,
) -> list[str]:
    """
    Run the 2-stage scoring while pretending the user only has `visible_set`.
    This is a proper holdout simulation for NEW-access recommendation.
    """
    graph_dfs = artifacts["graph_dfs"]
    ent_df = graph_dfs["entitlements"]

    # Candidate pool includes hidden truths + all other non-visible entitlements.
    candidate_ents_df = ent_df[~ent_df["id"].astype(str).isin(visible_set)].copy()
    if candidate_ents_df.empty:
        return []

    candidates_df = pd.DataFrame(
        {
            "UserId": [int(user_id)] * len(candidate_ents_df),
            "EntitlementId": candidate_ents_df["id"].astype(str).tolist(),
        }
    )
    candidates_df["UserId"] = candidates_df["UserId"].astype("int64")
    candidates_df["EntitlementId"] = candidates_df["EntitlementId"].astype("string")

    # Stage-1
    X_cand, _, _ = feature_engineering.create_candidate_model_features(
        candidates_df.copy(), artifacts["embeddings_df"]
    )
    X_cand = _hard_fail_feature_alignment(X_cand, artifacts["candidate_features"], "candidate_eval")
    cand_scores = artifacts["candidate_model"].predict_proba(X_cand)[:, 1]
    candidates_df["CandidateScore"] = cand_scores
    top_candidates = candidates_df.sort_values("CandidateScore", ascending=False).head(initial_candidates).copy()
    if top_candidates.empty:
        return []

    # For peer features in holdout, remove hidden entitlements from this user's history.
    modified_graph_dfs = dict(graph_dfs)
    entre = graph_dfs["entrecon"].copy()
    user_mask = entre["UserId"] == int(user_id)
    keep_for_user = entre["EntitlementId"].astype(str).isin(visible_set)
    modified_graph_dfs["entrecon"] = pd.concat(
        [entre[~user_mask], entre[user_mask & keep_for_user]],
        ignore_index=True,
    )
    peer_lookup = feature_engineering.build_peer_lookup_cache(modified_graph_dfs)

    # Stage-2
    X_rerank, _, _ = feature_engineering.create_enhanced_reranker_features(
        top_candidates.copy(),
        artifacts["embeddings_df"],
        modified_graph_dfs,
        peer_lookup_cache=peer_lookup,
    )
    X_rerank = _hard_fail_feature_alignment(X_rerank, artifacts["reranker_features"], "reranker_eval")
    rerank_scores = artifacts["reranker_model"].predict_proba(X_rerank)[:, 1]
    top_candidates["FinalScore"] = rerank_scores
    final_recs = top_candidates.sort_values("FinalScore", ascending=False).head(top_n)
    return final_recs["EntitlementId"].astype(str).tolist()


def run_holdout_evaluation(
    users: int,
    k_values: list[int],
    holdout_ratio: float,
    min_truth: int,
    initial_candidates: int,
    seed: int,
    out_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    artifacts = PredictionArtifacts.get_artifacts()
    entrecon_df = artifacts["graph_dfs"]["entrecon"][["UserId", "EntitlementId"]].copy()
    entrecon_df["UserId"] = pd.to_numeric(entrecon_df["UserId"], errors="coerce").astype("Int64")
    entrecon_df["EntitlementId"] = entrecon_df["EntitlementId"].astype(str)
    entrecon_df = entrecon_df.dropna(subset=["UserId"]).copy()
    entrecon_df["UserId"] = entrecon_df["UserId"].astype("int64")

    truth_map = entrecon_df.groupby("UserId")["EntitlementId"].apply(lambda s: sorted(set(s.tolist()))).to_dict()
    eligible_users = [uid for uid, ents in truth_map.items() if len(ents) >= min_truth]
    if not eligible_users:
        raise RuntimeError("No eligible users for holdout evaluation.")

    rng = np.random.default_rng(seed)
    sample_n = min(users, len(eligible_users))
    sampled_users = rng.choice(eligible_users, size=sample_n, replace=False).tolist()

    rows: list[HoldoutResult] = []
    max_k = max(k_values)

    for uid in sampled_users:
        full_truth = list(truth_map[uid])
        rng.shuffle(full_truth)
        hidden_n = max(1, int(len(full_truth) * holdout_ratio))
        hidden = full_truth[:hidden_n]
        visible = set(full_truth[hidden_n:])

        recommended = _predict_with_visible_only(
            artifacts=artifacts,
            user_id=int(uid),
            visible_set=visible,
            top_n=max_k,
            initial_candidates=initial_candidates,
        )

        for k in sorted(set(k_values)):
            hits, p, r = _precision_recall_at_k(recommended, hidden, k)
            rows.append(
                HoldoutResult(
                    user_id=int(uid),
                    k=int(k),
                    hits=int(hits),
                    precision_at_k=float(p),
                    recall_at_k=float(r),
                    hidden_size=int(len(hidden)),
                    visible_size=int(len(visible)),
                    total_size=int(len(full_truth)),
                )
            )

    per_user_df = pd.DataFrame([r.__dict__ for r in rows])
    if per_user_df.empty:
        raise RuntimeError("Holdout evaluation produced no rows.")

    summary_df = (
        per_user_df.groupby("k")
        .agg(
            mean_precision_at_k=("precision_at_k", "mean"),
            mean_recall_at_k=("recall_at_k", "mean"),
            mean_hits=("hits", "mean"),
            users_evaluated=("user_id", "nunique"),
        )
        .reset_index()
        .sort_values("k")
    )

    os.makedirs(out_dir, exist_ok=True)
    per_user_path = os.path.join(out_dir, "metrics_holdout_per_user.csv")
    summary_path = os.path.join(out_dir, "metrics_holdout_summary.csv")
    meta_path = os.path.join(out_dir, "metrics_holdout_meta.json")

    per_user_df.to_csv(per_user_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "users_requested": users,
                "users_evaluated": int(per_user_df["user_id"].nunique()),
                "k_values": sorted(set(k_values)),
                "holdout_ratio": holdout_ratio,
                "min_truth": min_truth,
                "initial_candidates": initial_candidates,
                "seed": seed,
            },
            f,
            indent=2,
        )

    print("\nHOLDOUT TOP-K METRICS")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {per_user_path}")
    print(f"Saved: {meta_path}")
    return summary_df, per_user_df


def main() -> None:
    parser = argparse.ArgumentParser("Holdout evaluation for IAM recommender")
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--holdout-ratio", type=float, default=0.3)
    parser.add_argument("--min-truth", type=int, default=3)
    parser.add_argument("--initial-candidates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="artifacts")
    args = parser.parse_args()

    run_holdout_evaluation(
        users=args.users,
        k_values=args.k,
        holdout_ratio=args.holdout_ratio,
        min_truth=args.min_truth,
        initial_candidates=args.initial_candidates,
        seed=args.seed,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
