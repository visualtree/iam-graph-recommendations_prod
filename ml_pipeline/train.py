from __future__ import annotations

import glob
import json
import os
import random
import shutil
import time
from collections import defaultdict

import joblib
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from . import config, data_loader, feature_engineering

EXPECTED_PEER_COLS = [
    "close_peer_adoption_rate",
    "direct_team_adoption_rate",
    "role_peer_adoption_rate",
    "dept_peer_adoption_rate",
    "close_peer_count",
    "direct_team_count",
    "role_peer_count",
    "dept_peer_count",
]

MONOTONE_POSITIVE_PEER_RATE_COLS = [
    "close_peer_adoption_rate",
    "direct_team_adoption_rate",
    "role_peer_adoption_rate",
    "dept_peer_adoption_rate",
]


def _build_monotone_constraints(feature_names: list[str], positive_cols: list[str]) -> tuple[int, ...]:
    """Build XGBoost monotone constraints vector aligned to feature order."""
    positive = set(positive_cols)
    return tuple(1 if f in positive else 0 for f in feature_names)


def cleanup_artifacts(dir_path: str) -> None:
    """Safely remove old training artifacts from the target directory."""
    if not dir_path or dir_path.strip() in {"/", "C:\\", "C:/", "\\"}:
        raise RuntimeError(f"Refusing to clean a dangerous path: {dir_path!r}")

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        return

    patterns = [
        "candidate_model.joblib",
        "candidate_model_features.joblib",
        "reranker_model.joblib",
        "reranker_model_features.joblib",
        "reranker_model_manifest.joblib",
        "embeddings.pkl",
        "users.pkl",
        "entitlements.pkl",
        "entrecon.pkl",
        "orgs.pkl",
        "endpoints.pkl",
        "designations.pkl",
        "shap_*.html",
        "*.joblib.bak",
        "*.pkl.bak",
    ]

    removed = []
    for pat in patterns:
        for fp in glob.glob(os.path.join(dir_path, pat)):
            try:
                if os.path.isfile(fp):
                    os.remove(fp)
                    removed.append(fp)
                elif os.path.isdir(fp):
                    shutil.rmtree(fp)
                    removed.append(fp)
            except Exception as exc:  # best-effort cleanup
                print(f"WARN: Could not remove {fp}: {exc}")

    print(f"Artifacts cleanup done. Removed {len(removed)} items from {dir_path}")


def _parse_endpoint_system(ent_id: str) -> int | None:
    try:
        return int(str(ent_id).split("_")[0])
    except Exception:
        return None


def _build_training_pairs(
    graph_dfs: dict,
    hard_negative_ratio: float = 0.5,
    neg_multiplier: float = 2.0,
) -> pd.DataFrame:
    """Create labeled (UserId, EntitlementId, HasAccess) pairs with mixed hard negatives."""
    entre = graph_dfs["entrecon"][["UserId", "EntitlementId"]].drop_duplicates()
    users = graph_dfs["users"][["id", "NBusinessRoleId", "NOrganisationId"]].rename(columns={"id": "UserId"})

    pos_df = entre.copy()
    pos_df["HasAccess"] = 1

    user_ids = pos_df["UserId"].unique().tolist()
    entitlement_ids = graph_dfs["entitlements"]["id"].unique().tolist()
    positive_pairs = set(map(tuple, pos_df[["UserId", "EntitlementId"]].itertuples(index=False, name=None)))

    num_neg_total = int(len(pos_df) * neg_multiplier)
    num_hard = int(num_neg_total * hard_negative_ratio)
    print(f"Sampling negatives: total={num_neg_total} (hard={num_hard}, easy={num_neg_total - num_hard})")

    ent_to_sys = {e: _parse_endpoint_system(e) for e in entitlement_ids}
    labeled_ctx = entre.merge(users, on="UserId", how="left")
    role_pop = labeled_ctx.groupby(["NBusinessRoleId", "EntitlementId"]).size().reset_index(name="cnt")
    org_pop = labeled_ctx.groupby(["NOrganisationId", "EntitlementId"]).size().reset_index(name="cnt")

    user_ent_map, user_sys_map = defaultdict(set), defaultdict(set)
    for u, e in entre[["UserId", "EntitlementId"]].itertuples(index=False):
        user_ent_map[u].add(e)
        sys_id = ent_to_sys.get(e)
        if sys_id is not None:
            user_sys_map[u].add(sys_id)

    def hard_pool_for_user(u):
        own = user_ent_map.get(u, set())
        sys_set = user_sys_map.get(u, set())
        role_id = users.loc[users["UserId"] == u, "NBusinessRoleId"]
        org_id = users.loc[users["UserId"] == u, "NOrganisationId"]
        role_id = int(role_id.iloc[0]) if len(role_id) > 0 and pd.notna(role_id.iloc[0]) else None
        org_id = int(org_id.iloc[0]) if len(org_id) > 0 and pd.notna(org_id.iloc[0]) else None

        pool = set()
        if sys_set:
            pool.update([e for e in entitlement_ids if ent_to_sys.get(e) in sys_set])
        if role_id is not None:
            pool.update(role_pop[role_pop["NBusinessRoleId"] == role_id]["EntitlementId"].tolist())
        if org_id is not None:
            pool.update(org_pop[org_pop["NOrganisationId"] == org_id]["EntitlementId"].tolist())
        pool.difference_update(own)
        return list(pool)

    rng = random.Random(config.RANDOM_STATE)
    neg_rows = []
    neg_pairs = set()

    guard = 0
    while len(neg_rows) < num_hard and guard < num_hard * 20:
        guard += 1
        u = rng.choice(user_ids)
        pool = hard_pool_for_user(u)
        if not pool:
            continue
        e = rng.choice(pool)
        pair = (u, e)
        if pair not in positive_pairs and pair not in neg_pairs:
            neg_rows.append((u, e, 0))
            neg_pairs.add(pair)

    guard = 0
    while len(neg_rows) < num_neg_total and guard < num_neg_total * 20:
        guard += 1
        u = rng.choice(user_ids)
        e = rng.choice(entitlement_ids)
        pair = (u, e)
        if pair not in positive_pairs and pair not in neg_pairs:
            neg_rows.append((u, e, 0))
            neg_pairs.add(pair)

    neg_df = pd.DataFrame(neg_rows, columns=["UserId", "EntitlementId", "HasAccess"])
    labeled = pd.concat([pos_df[["UserId", "EntitlementId", "HasAccess"]], neg_df], ignore_index=True)
    print(f"Labeled pairs: positives={len(pos_df)}, negatives={len(neg_df)}, total={len(labeled)}")
    return labeled


def _fit_xgb_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    params: dict,
    model_name: str,
    monotone_constraints: tuple[int, ...] | None = None,
):
    clf = xgb.XGBClassifier(
        max_depth=int(params.get("max_depth", 6)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        n_estimators=int(params.get("n_estimators", 300)),
        subsample=float(params.get("subsample", 0.9)),
        colsample_bytree=float(params.get("colsample_bytree", 0.8)),
        random_state=config.RANDOM_STATE,
        early_stopping_rounds=int(params.get("early_stopping_rounds", 50)),
        eval_metric="auc",
        tree_method=params.get("tree_method", "hist"),
        n_jobs=int(params.get("n_jobs", -1)),
        monotone_constraints=monotone_constraints,
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    val_pred = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    print(f"{model_name} AUC={auc:.6f}")
    return clf, auc


def _optuna_objective(
    trial,
    X_train,
    y_train,
    X_val,
    y_val,
    model_name: str,
    monotone_constraints: tuple[int, ...] | None = None,
):
    params = {
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "early_stopping_rounds": 50,
        "tree_method": "hist",
    }
    _, auc = _fit_xgb_classifier(
        X_train, y_train, X_val, y_val, params, model_name, monotone_constraints=monotone_constraints
    )
    return auc


def _split_train_val(X, y):
    stratify = y if y.nunique() > 1 else None
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=config.RANDOM_STATE,
        stratify=stratify,
    )


def _build_user_splits(
    labeled_df: pd.DataFrame,
    out_dir: str,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = config.RANDOM_STATE,
    min_users: int = 10,
) -> dict:
    """Create and persist user-disjoint train/val/test splits for locked evaluation."""
    users = pd.Series(labeled_df["UserId"]).dropna().unique().tolist()
    total_users = len(users)
    if total_users < min_users:
        return {
            "status": "skipped",
            "reason": "too_few_unique_users",
            "total_users": total_users,
        }

    rng = random.Random(seed)
    rng.shuffle(users)

    n_test = max(1, int(total_users * test_ratio))
    n_val = max(1, int(total_users * val_ratio))
    n_train = total_users - n_val - n_test
    if n_train <= 0:
        return {
            "status": "skipped",
            "reason": "insufficient_users_for_split",
            "total_users": total_users,
            "val_users": n_val,
            "test_users": n_test,
        }

    test_users = users[:n_test]
    val_users = users[n_test:n_test + n_val]
    train_users = users[n_test + n_val:]

    splits = {
        "status": "ok",
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "total_users": total_users,
        "train_users": train_users,
        "val_users": val_users,
        "test_users": test_users,
    }

    os.makedirs(out_dir, exist_ok=True)
    split_path = os.path.join(out_dir, "user_splits.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "status": splits["status"],
                "seed": seed,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "total_users": total_users,
                "train_users": train_users,
                "val_users": val_users,
                "test_users": test_users,
            },
            f,
            indent=2,
        )
    return splits


def _split_by_users(X: pd.DataFrame, y: pd.Series, labeled_df: pd.DataFrame, splits: dict) -> dict:
    train_users = set(splits.get("train_users", []))
    val_users = set(splits.get("val_users", []))
    test_users = set(splits.get("test_users", []))

    train_mask = labeled_df["UserId"].isin(train_users).to_numpy()
    val_mask = labeled_df["UserId"].isin(val_users).to_numpy()
    test_mask = labeled_df["UserId"].isin(test_users).to_numpy()

    return {
        "X_train": X.loc[train_mask],
        "y_train": y.loc[train_mask],
        "X_val": X.loc[val_mask],
        "y_val": y.loc[val_mask],
        "X_test": X.loc[test_mask],
        "y_test": y.loc[test_mask],
    }


def _evaluate_auc_on_split(model, X_split, y_split, split_name: str) -> dict:
    if X_split is None or y_split is None or len(X_split) == 0:
        return {"status": "skipped", "reason": "empty_split", "rows": 0}
    if y_split.nunique() < 2:
        return {"status": "skipped", "reason": "single_class", "rows": int(len(y_split))}
    preds = model.predict_proba(X_split)[:, 1]
    auc = roc_auc_score(y_split, preds)
    return {"status": "ok", "rows": int(len(y_split)), "auc": float(auc)}


def _fit_calibrator(model, X_val, y_val, method: str) -> dict:
    """Fit a probability calibrator on validation data (prefit base model)."""
    if X_val is None or y_val is None or len(X_val) == 0:
        return {"status": "skipped", "reason": "empty_split", "calibrator": None}
    if y_val.nunique() < 2:
        return {"status": "skipped", "reason": "single_class", "calibrator": None}

    method = method if method in {"sigmoid", "isotonic"} else "sigmoid"
    calibrator = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrator.fit(X_val, y_val)
    return {"status": "ok", "method": method, "calibrator": calibrator}


def _aligned_labeled_df_for_features(labeled_df: pd.DataFrame, X: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Align labeled pairs to feature matrix row-count for downstream split diagnostics."""
    labeled_aligned = labeled_df.reset_index(drop=True)
    if len(labeled_aligned) != len(X):
        print(
            f"WARN: {model_name} row mismatch for leakage check: "
            f"labeled={len(labeled_aligned)} features={len(X)}; truncating to min length"
        )
        n = min(len(labeled_aligned), len(X))
        labeled_aligned = labeled_aligned.iloc[:n].reset_index(drop=True)
    return labeled_aligned


def _evaluate_user_holdout_auc(
    X: pd.DataFrame,
    y: pd.Series,
    labeled_df: pd.DataFrame,
    params: dict,
    model_name: str,
    monotone_constraints: tuple[int, ...] | None = None,
) -> dict:
    """
    Evaluate generalization with a user-disjoint split.
    This is a leakage guardrail: train users and validation users do not overlap.
    """
    users = pd.Series(labeled_df["UserId"]).dropna().unique().tolist()
    if len(users) < 10:
        return {
            "status": "skipped",
            "reason": "too_few_unique_users",
            "train_users": len(users),
        }

    rng = random.Random(config.RANDOM_STATE)
    val_count = max(1, int(0.2 * len(users)))
    val_users = set(rng.sample(users, val_count))
    val_mask = labeled_df["UserId"].isin(val_users).to_numpy()

    if val_mask.all() or (~val_mask).all():
        return {
            "status": "skipped",
            "reason": "degenerate_user_split",
            "train_users": len(users) - len(val_users),
            "val_users": len(val_users),
        }

    X_train = X.loc[~val_mask]
    y_train = y.loc[~val_mask]
    X_val = X.loc[val_mask]
    y_val = y.loc[val_mask]

    if y_val.nunique() < 2 or y_train.nunique() < 2:
        return {
            "status": "skipped",
            "reason": "single_class_after_user_split",
            "train_users": len(users) - len(val_users),
            "val_users": len(val_users),
        }

    model, val_auc = _fit_xgb_classifier(
        X_train,
        y_train,
        X_val,
        y_val,
        params,
        f"{model_name}_user_holdout",
        monotone_constraints=monotone_constraints,
    )
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])

    return {
        "status": "ok",
        "train_users": len(users) - len(val_users),
        "val_users": len(val_users),
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "train_auc": float(train_auc),
        "val_auc": float(val_auc),
        "auc_gap": float(train_auc - val_auc),
    }


def run_training() -> None:
    os.makedirs(config.ARTIFACT_DIR, exist_ok=True)
    cleanup_artifacts(config.ARTIFACT_DIR)

    print("STARTING IAM TRAINING PIPELINE")
    start_ts = time.time()

    graph_dfs = data_loader.get_all_graph_data()
    embeddings_df = data_loader.get_neo4j_embeddings()
    joblib.dump(embeddings_df, os.path.join(config.ARTIFACT_DIR, "embeddings.pkl"))

    labeled = _build_training_pairs(graph_dfs, hard_negative_ratio=0.5, neg_multiplier=2.0)
    user_splits = _build_user_splits(labeled, out_dir=config.ARTIFACT_DIR)

    # Candidate model
    X_cand, y_cand, _ = feature_engineering.create_candidate_model_features(labeled.copy(), embeddings_df)
    X_cand = X_cand.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if y_cand is None or X_cand.empty:
        raise RuntimeError("Candidate feature generation failed: empty matrix or labels")

    labeled_cand = _aligned_labeled_df_for_features(labeled, X_cand, "candidate")
    if user_splits.get("status") == "ok":
        cand_split = _split_by_users(X_cand, y_cand, labeled_cand, user_splits)
        Xc_train, yc_train = cand_split["X_train"], cand_split["y_train"]
        Xc_val, yc_val = cand_split["X_val"], cand_split["y_val"]
        Xc_test, yc_test = cand_split["X_test"], cand_split["y_test"]
        if len(Xc_val) == 0 or yc_val.nunique() < 2 or yc_train.nunique() < 2:
            print("WARN: Candidate user split invalid; falling back to random split")
            Xc_train, Xc_val, yc_train, yc_val = _split_train_val(X_cand, y_cand)
            Xc_test, yc_test = None, None
    else:
        Xc_train, Xc_val, yc_train, yc_val = _split_train_val(X_cand, y_cand)
        Xc_test, yc_test = None, None
    print("Running Optuna for candidate model...")
    cand_study = optuna.create_study(direction="maximize")
    cand_study.optimize(
        lambda trial: _optuna_objective(trial, Xc_train, yc_train, Xc_val, yc_val, "candidate_model"),
        n_trials=config.OPTUNA_CANDIDATE_TRIALS,
    )
    candidate_model, cand_auc = _fit_xgb_classifier(
        Xc_train,
        yc_train,
        Xc_val,
        yc_val,
        cand_study.best_params,
        model_name="candidate_model",
    )
    cand_test_metrics = _evaluate_auc_on_split(candidate_model, Xc_test, yc_test, "candidate_test")
    cand_cal = _fit_calibrator(candidate_model, Xc_val, yc_val, config.CALIBRATION_METHOD)
    if cand_cal.get("status") == "ok":
        joblib.dump(cand_cal["calibrator"], os.path.join(config.ARTIFACT_DIR, "candidate_calibrator.joblib"))
    joblib.dump(candidate_model, os.path.join(config.ARTIFACT_DIR, "candidate_model.joblib"))
    joblib.dump(list(X_cand.columns), os.path.join(config.ARTIFACT_DIR, "candidate_model_features.joblib"))

    # Reranker model
    X_rerank, y_rerank, _ = feature_engineering.create_enhanced_reranker_features(
        labeled.copy(),
        embeddings_df,
        graph_dfs,
    )
    missing_peers = sorted(set(EXPECTED_PEER_COLS) - set(X_rerank.columns))
    if missing_peers:
        raise RuntimeError(f"Peer features missing from X_rerank: {missing_peers}")

    X_rerank = X_rerank.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if y_rerank is None or X_rerank.empty:
        raise RuntimeError("Reranker feature generation failed: empty matrix or labels")

    labeled_rerank = _aligned_labeled_df_for_features(labeled, X_rerank, "reranker")
    if user_splits.get("status") == "ok":
        rerank_split = _split_by_users(X_rerank, y_rerank, labeled_rerank, user_splits)
        Xr_train, yr_train = rerank_split["X_train"], rerank_split["y_train"]
        Xr_val, yr_val = rerank_split["X_val"], rerank_split["y_val"]
        Xr_test, yr_test = rerank_split["X_test"], rerank_split["y_test"]
        if len(Xr_val) == 0 or yr_val.nunique() < 2 or yr_train.nunique() < 2:
            print("WARN: Reranker user split invalid; falling back to random split")
            Xr_train, Xr_val, yr_train, yr_val = _split_train_val(X_rerank, y_rerank)
            Xr_test, yr_test = None, None
    else:
        Xr_train, Xr_val, yr_train, yr_val = _split_train_val(X_rerank, y_rerank)
        Xr_test, yr_test = None, None
    reranker_monotone_constraints = _build_monotone_constraints(
        list(X_rerank.columns),
        MONOTONE_POSITIVE_PEER_RATE_COLS,
    )
    constrained_count = sum(1 for c in reranker_monotone_constraints if c != 0)
    print(
        f"Applying reranker monotonic constraints on peer adoption rates: "
        f"{constrained_count}/{len(reranker_monotone_constraints)} constrained"
    )
    print("Running Optuna for reranker model...")
    rerank_study = optuna.create_study(direction="maximize")
    rerank_study.optimize(
        lambda trial: _optuna_objective(
            trial,
            Xr_train,
            yr_train,
            Xr_val,
            yr_val,
            "reranker_model",
            monotone_constraints=reranker_monotone_constraints,
        ),
        n_trials=config.OPTUNA_RERANKER_TRIALS,
    )
    reranker_model, rerank_auc = _fit_xgb_classifier(
        Xr_train,
        yr_train,
        Xr_val,
        yr_val,
        rerank_study.best_params,
        model_name="reranker_model",
        monotone_constraints=reranker_monotone_constraints,
    )
    rerank_test_metrics = _evaluate_auc_on_split(reranker_model, Xr_test, yr_test, "reranker_test")
    rerank_cal = _fit_calibrator(reranker_model, Xr_val, yr_val, config.CALIBRATION_METHOD)
    if rerank_cal.get("status") == "ok":
        joblib.dump(rerank_cal["calibrator"], os.path.join(config.ARTIFACT_DIR, "reranker_calibrator.joblib"))
    rerank_train_auc = roc_auc_score(yr_train, reranker_model.predict_proba(Xr_train)[:, 1])

    # Leakage/overfitting diagnostic: user-disjoint holdout AUC for reranker
    user_holdout_diag = _evaluate_user_holdout_auc(
        X_rerank.reset_index(drop=True),
        y_rerank.reset_index(drop=True),
        labeled_rerank,
        rerank_study.best_params,
        "reranker_model",
        monotone_constraints=reranker_monotone_constraints,
    )

    joblib.dump(reranker_model, os.path.join(config.ARTIFACT_DIR, "reranker_model.joblib"))
    joblib.dump(list(X_rerank.columns), os.path.join(config.ARTIFACT_DIR, "reranker_model_features.joblib"))

    # Save graph data used at inference time
    for name in ["users", "entitlements", "entrecon", "orgs", "endpoints", "designations"]:
        joblib.dump(graph_dfs[name], os.path.join(config.ARTIFACT_DIR, f"{name}.pkl"))

    manifest = {
        "model": "reranker_model",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "code_version": os.getenv("GIT_COMMIT", "unknown"),
        "feature_count": int(X_rerank.shape[1]),
        "candidate_feature_count": int(X_cand.shape[1]),
        "best_candidate_params": cand_study.best_params,
        "best_reranker_params": rerank_study.best_params,
        "candidate_auc_validation": float(cand_auc),
        "candidate_auc_test_user_split": cand_test_metrics,
        "candidate_calibration": {
            "status": cand_cal.get("status"),
            "method": cand_cal.get("method"),
            "reason": cand_cal.get("reason"),
            "path": os.path.join(config.ARTIFACT_DIR, "candidate_calibrator.joblib")
            if cand_cal.get("status") == "ok"
            else None,
        },
        "reranker_auc_train_random_split": float(rerank_train_auc),
        "reranker_auc_validation_random_split": float(rerank_auc),
        "reranker_auc_test_user_split": rerank_test_metrics,
        "reranker_calibration": {
            "status": rerank_cal.get("status"),
            "method": rerank_cal.get("method"),
            "reason": rerank_cal.get("reason"),
            "path": os.path.join(config.ARTIFACT_DIR, "reranker_calibrator.joblib")
            if rerank_cal.get("status") == "ok"
            else None,
        },
        "reranker_user_holdout_diagnostic": user_holdout_diag,
        "user_split_metadata": {
            "status": user_splits.get("status"),
            "total_users": user_splits.get("total_users"),
            "val_ratio": user_splits.get("val_ratio"),
            "test_ratio": user_splits.get("test_ratio"),
            "train_users": len(user_splits.get("train_users", [])),
            "val_users": len(user_splits.get("val_users", [])),
            "test_users": len(user_splits.get("test_users", [])),
            "path": os.path.join(config.ARTIFACT_DIR, "user_splits.json")
            if user_splits.get("status") == "ok"
            else None,
        },
        "reranker_monotone_constraints_positive_cols": MONOTONE_POSITIVE_PEER_RATE_COLS,
        "reranker_monotone_constraints_vector": list(reranker_monotone_constraints),
    }
    joblib.dump(manifest, os.path.join(config.ARTIFACT_DIR, "reranker_model_manifest.joblib"))
    with open(os.path.join(config.ARTIFACT_DIR, "leakage_report.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if user_holdout_diag.get("status") == "ok":
        holdout_auc = user_holdout_diag["val_auc"]
        auc_drop = rerank_auc - holdout_auc
        print(
            f"Reranker leakage check | random_split_val_auc={rerank_auc:.6f} "
            f"user_holdout_val_auc={holdout_auc:.6f} drop={auc_drop:.6f}"
        )
        if rerank_auc >= 0.995 and auc_drop > 0.02:
            print(
                "WARN: Potential leakage/overfitting signal: reranker AUC is near-perfect on "
                "random split but drops materially on user-disjoint holdout."
            )
    else:
        print(f"Reranker leakage check skipped: {user_holdout_diag}")

    print("TRAINING COMPLETE")
    print(f"Candidate AUC: {cand_auc:.6f} | Reranker AUC: {rerank_auc:.6f}")
    print(f"Artifacts dir: {config.ARTIFACT_DIR}")
    print(f"Total training time: {time.time() - start_ts:.2f}s")


if __name__ == "__main__":
    run_training()
