# # ml_pipeline/train.py (UPDATED FOR TYPE SAFETY)

# import os, joblib, time, pandas as pd, xgboost as xgb, optuna, random
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score
# from . import config, data_loader, feature_engineering


# # --- add near the top of train.py ---
# import glob, shutil

# def cleanup_artifacts(dir_path: str):
    # """
    # Safely remove old model/feature/embedding artifacts before training.
    # Only deletes known patterns inside dir_path; refuses to run on suspicious paths.
    # """
    # # Safety rails
    # if not dir_path or dir_path.strip() in {"/", "C:\\", "C:/", "\\"}:
        # raise RuntimeError(f"Refusing to clean a dangerous path: {dir_path!r}")
    # if not os.path.isdir(dir_path):
        # os.makedirs(dir_path, exist_ok=True)
        # return

    # patterns = [
        # "candidate_model.joblib",
        # "candidate_model_features.joblib",
        # "reranker_model.joblib",
        # "reranker_model_features.joblib",
        # "reranker_model_manifest.joblib",
        # "embeddings.pkl",
        # "users.pkl", "entitlements.pkl", "entrecon.pkl",
        # "orgs.pkl", "endpoints.pkl", "designations.pkl", "accounts.pkl",
        # # any SHAP/html explainability files from previous runs
        # "shap_*.html",
        # # any stray joblibs/pkls you know you generate
        # "*.joblib.bak", "*.pkl.bak"
    # ]

    # removed = []
    # for pat in patterns:
        # for fp in glob.glob(os.path.join(dir_path, pat)):
            # try:
                # if os.path.isfile(fp):
                    # os.remove(fp); removed.append(fp)
                # elif os.path.isdir(fp):
                    # shutil.rmtree(fp); removed.append(fp)
            # except Exception as e:
                # print(f"⚠️ Could not remove {fp}: {e}")

    # print(f"🧹 Artifacts cleanup done. Removed {len(removed)} items.")


# def train_model(X, y, model_name):
    # print(f"\n--- Training Model: {model_name} ---")

    # # Validate inputs
    # if y is None:
        # raise ValueError(f"{model_name}: labels 'y' are None")

    # # Drop rows where y is NaN and align X
    # mask = y.notna()
    # if mask.sum() != len(y):
        # dropped = int((~mask).sum())
        # print(f"⚠️  Dropping {dropped} row(s) with NaN labels before train/test split.")
    # X = X.loc[mask].copy()
    # y = y.loc[mask].astype(int)

    # # Handle single class case
    # stratify_arg = y if y.nunique() > 1 else None
    # if stratify_arg is None:
        # print("⚠️  Only one class present in y; proceeding without stratify.")

    # X_train, X_test, y_train, y_test = train_test_split(
        # X, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=stratify_arg
    # )

    # def objective(trial):
        # if y_train.value_counts().get(1, 0) == 0:
            # return 0.5
        # scale_pos_weight = y_train.value_counts().get(0, 1) / y_train.value_counts().get(1, 1)
        # param = {
            # 'objective': 'binary:logistic', 'eval_metric': 'logloss', 'booster': 'gbtree',
            # 'scale_pos_weight': scale_pos_weight,
            # 'max_depth': trial.suggest_int('max_depth', 4, 10),
            # 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            # 'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            # 'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            # 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        # }
        # model = xgb.XGBClassifier(**param, n_jobs=-1, random_state=config.RANDOM_STATE)
        # model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
        # preds = model.predict_proba(X_test)[:, 1]
        # return roc_auc_score(y_test, preds)

    # study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE))
    # study.optimize(objective, n_trials=config.OPTUNA_N_TRIALS)
    # print(f"✅ Optuna best ROC AUC for {model_name}: {study.best_value:.4f}")

    # print(f"🚀 Training final {model_name} on all data...")
    # final_params = {**config.XGB_PARAMS, **study.best_params}
    # final_model = xgb.XGBClassifier(**final_params)
    # final_model.fit(X, y)
    
    # joblib.dump(final_model, os.path.join(config.ARTIFACT_DIR, f'{model_name}.joblib'))
    # joblib.dump(list(X.columns), os.path.join(config.ARTIFACT_DIR, f'{model_name}_features.joblib'))
    # print(f"✅ {model_name} and its features saved.")

# # OPTIONAL ENHANCEMENT: train.py - Add feature validation

# def run_training():



    # start_time = time.time()
    # os.makedirs(config.ARTIFACT_DIR, exist_ok=True)
    # cleanup_artifacts(config.ARTIFACT_DIR)
    
    # print("====== STARTING SCALABLE IAM ML TRAINING PIPELINE ======")
    # graph_dfs = data_loader.get_all_graph_data()
    # embeddings_df = data_loader.get_neo4j_embeddings()
    
    # # Add this after: embeddings_df = data_loader.get_neo4j_embeddings()
    # print(f"🔍 EMBEDDING DEBUG:")
    # print(f"   Embeddings shape: {embeddings_df.shape}")
    # print(f"   Sample embedding type: {type(embeddings_df['embedding'].iloc[0])}")
    # print(f"   Sample embedding length: {len(embeddings_df['embedding'].iloc[0])}")
    # print(f"   Sample embedding: {embeddings_df['embedding'].iloc[0][:10]}")

    # # Also check what happens in _expand_embeddings
    # # sample_user_ids = ml_labeled['UserId'].unique()[:5]
    # # user_raw = embeddings_df[embeddings_df['originalId'].isin(sample_user_ids)].copy()
    # # print(f"   User embeddings found: {len(user_raw)}")
    # # if len(user_raw) > 0:
        # # sample_emb = user_raw['embedding'].iloc[0]
        # # print(f"   User embedding length: {len(sample_emb)}")
    
    
    # print("🚀 Preparing labeled dataset with memory-efficient negative sampling...")
    # pos_df = graph_dfs['entrecon'].copy()
    # if pos_df.empty:
        # raise ValueError("No positive access examples (HAS_ACCESS_TO) found in graph. Check ETL script.")
    # pos_df['HasAccess'] = 1
    
    # # TYPE SAFETY: Ensure consistent types for negative sampling
    # print(f"🔧 Ensuring type consistency for negative sampling...")
    # user_ids = graph_dfs['users']['id'].tolist()
    # entitlement_ids = graph_dfs['entitlements']['id'].tolist()
    
    # # Verify types match between positive and negative sampling
    # print(f"   User IDs type: {type(user_ids[0]) if user_ids else 'None'}")
    # print(f"   Entitlement IDs type: {type(entitlement_ids[0]) if entitlement_ids else 'None'}")
    # print(f"   Positive pair types: UserId={pos_df['UserId'].dtype}, EntitlementId={pos_df['EntitlementId'].dtype}")
    
    # if not user_ids or not entitlement_ids:
        # raise ValueError("User or Entitlement data is empty. Cannot perform negative sampling.")
    
    # # Create positive pairs set with proper types
    # positive_pairs = set()
    # for _, row in pos_df.iterrows():
        # user_id = row['UserId']
        # ent_id = str(row['EntitlementId'])  # Ensure string for consistency
        # positive_pairs.add((user_id, ent_id))
    
    # num_negatives_to_sample = len(pos_df) * 2
    # neg_samples = []
    
    # print(f"Sampling {num_negatives_to_sample} negative examples...")
    # random.seed(config.RANDOM_STATE)
    # attempts = 0
    # max_attempts = num_negatives_to_sample * 10  # Prevent infinite loop
    
    # while len(neg_samples) < num_negatives_to_sample and attempts < max_attempts:
        # rand_user = random.choice(user_ids)
        # rand_ent = str(random.choice(entitlement_ids))  # Ensure string consistency
        
        # if (rand_user, rand_ent) not in positive_pairs:
            # neg_samples.append({
                # 'UserId': rand_user, 
                # 'EntitlementId': rand_ent,
                # 'HasAccess': 0
            # })
        # attempts += 1
    
    # if len(neg_samples) < num_negatives_to_sample:
        # print(f"⚠️ Warning: Only generated {len(neg_samples)} negative samples out of {num_negatives_to_sample} requested")
    
    # neg_df = pd.DataFrame(neg_samples)
    
    # # TYPE SAFETY: Ensure consistent types in final dataset
    # if not neg_df.empty:
        # neg_df['UserId'] = neg_df['UserId'].astype('int64')
        # neg_df['EntitlementId'] = neg_df['EntitlementId'].astype('string')
    
    # # Ensure positive df has consistent types too
    # pos_df['UserId'] = pos_df['UserId'].astype('int64')
    # pos_df['EntitlementId'] = pos_df['EntitlementId'].astype('string')
    
    # ml_labeled = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)
    
    # print(f"🔍 EMBEDDING DEBUG:")
    # print(f"   Embeddings shape: {embeddings_df.shape}")
    # print(f"   Sample embedding type: {type(embeddings_df['embedding'].iloc[0])}")
    # print(f"   Sample embedding length: {len(embeddings_df['embedding'].iloc[0])}")
    # print(f"   Sample embedding: {embeddings_df['embedding'].iloc[0][:10]}")
        
    # # ADDITIONAL TYPE SAFETY: Check for and remove any duplicate pairs
    # print(f"\n🔍 TRAINING DATA QUALITY CHECK:")
    # print(f"   Total ml_labeled rows: {len(ml_labeled)}")
    # duplicate_pairs = ml_labeled.duplicated(subset=['UserId', 'EntitlementId']).sum()
    # if duplicate_pairs > 0:
        # print(f"   ⚠️ Found {duplicate_pairs} duplicate user-entitlement pairs")
        # ml_labeled = ml_labeled.drop_duplicates(subset=['UserId', 'EntitlementId']).reset_index(drop=True)
        # print(f"   🔧 Cleaned to {len(ml_labeled)} unique pairs")
    
    # print(f"✅ Labeled dataset prepared with {len(ml_labeled)} samples.")
    # print(f"   Positive: {len(pos_df)}, Negative: {len(neg_df)}")
    # print(f"   Final types - UserId: {ml_labeled['UserId'].dtype}, EntitlementId: {ml_labeled['EntitlementId'].dtype}")
    
    # print("\n--- Generating features for Candidate Model ---")
    # X_cand, y_cand, cand_features = feature_engineering.create_candidate_model_features(ml_labeled.copy(), embeddings_df)
    
    # # ✅ VALIDATION: Check candidate features
    # print(f"✅ Candidate model features ({len(cand_features)}): {cand_features[:5]}...")
    
    # train_model(X_cand, y_cand, 'candidate_model')

    # print("\n--- Generating features for Re-Ranker Model (Enhanced with ALL Peer Features) ---")
    # X_rerank, y_rerank, rerank_features = feature_engineering.create_enhanced_reranker_features(ml_labeled.copy(), embeddings_df, graph_dfs)
    
    # # ✅ VALIDATION: Check reranker features, especially peer features
    # peer_features_found = [f for f in rerank_features if 'peer' in f]
    # print(f"✅ Reranker model features ({len(rerank_features)} total)")
    # print(f"✅ Peer features found ({len(peer_features_found)}): {peer_features_found}")
    
    # # Validate we have all 8 expected peer features
    # expected_peer_features = [
        # 'close_peer_adoption_rate',
        # 'direct_team_adoption_rate',
        # 'role_peer_adoption_rate',
        # 'dept_peer_adoption_rate',
        # 'close_peer_count', 
        # 'direct_team_count',
        # 'role_peer_count', 
        # 'dept_peer_count'
    # ]
    
    # missing_peer_features = set(expected_peer_features) - set(peer_features_found)
    # if missing_peer_features:
        # print(f"⚠️ WARNING: Missing expected peer features: {missing_peer_features}")
    # else:
        # print(f"✅ All 8 peer features present in training data")
    
    # train_model(X_rerank, y_rerank, 'reranker_model')

    # print("\n🚀 Saving supporting artifacts...")
    # for name, df in graph_dfs.items():
        # df.to_pickle(os.path.join(config.ARTIFACT_DIR, f'{name}.pkl'))
    # embeddings_df.to_pickle(os.path.join(config.ARTIFACT_DIR, 'embeddings.pkl'))
    # print("✅ Supporting artifacts saved.")
    
    # print(f"\n====== SCALABLE TRAINING PIPELINE COMPLETED IN {time.time() - start_time:.2f} SECONDS ======")
    
    # # ✅ FINAL VALIDATION: Print feature counts for verification
    # print(f"\n📊 FINAL MODEL SPECIFICATIONS:")
    # print(f"   Candidate Model: {len(cand_features)} features")
    # print(f"   Reranker Model: {len(rerank_features)} features ({len(peer_features_found)} peer)")

# if __name__ == '__main__':
    # run_training()
    
    
# -*- coding: utf-8 -*-
"""
Regenerated train.py
- Safe artifacts cleanup at start of training
- Early stopping moved to XGBClassifier constructor (no deprecation warning)
- Hard‑fail assertions that ALL 8 peer features exist for the reranker
- Optional hard‑negative sampling (same‑system / peer‑popular) mixed with random negatives
- Saves feature lists + manifest for traceability
- **Optuna integration re‑added** for hyperparameter tuning (candidate and reranker models)

Assumptions:
- feature_engineering.create_candidate_model_features(labeled_df, embeddings_df, graph_dfs=None)
- feature_engineering.create_enhanced_reranker_features(labeled_df, embeddings_df, graph_dfs)
- data_loader.load_cached_graph_dfs() returns dict of cached DataFrames
- A precomputed embeddings.pkl is created in this run and saved to ARTIFACT_DIR
"""
from __future__ import annotations
import os
import time
import json
import glob
import shutil
import random
from collections import Counter, defaultdict

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import xgboost as xgb
import optuna

from . import config
from . import data_loader
from . import feature_engineering

# ----------------------
# Constants
# ----------------------
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

# ----------------------
# Utilities
# ----------------------



# (unchanged cleanup_artifacts and _parse_endpoint_system definitions...)

# ----------------------
# Training utils
# ----------------------

def _fit_xgb_classifier(X_train, y_train, X_val, y_val, params: dict, model_name: str):
    clf = xgb.XGBClassifier(
        max_depth=int(params.get('max_depth', 6)),
        learning_rate=float(params.get('learning_rate', 0.1)),
        n_estimators=int(params.get('n_estimators', 300)),
        subsample=float(params.get('subsample', 0.9)),
        colsample_bytree=float(params.get('colsample_bytree', 0.8)),
        random_state=config.RANDOM_STATE,
        early_stopping_rounds=int(params.get('early_stopping_rounds', 50)),
        eval_metric='auc',
        tree_method=params.get('tree_method', 'auto'),
        n_jobs=params.get('n_jobs', -1),
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    val_pred = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    print(f"✅ {model_name} AUC={auc:.6f}")
    return clf, auc


def _optuna_objective(trial, X_train, y_train, X_val, y_val, model_name: str):
    params = {
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'early_stopping_rounds': 50,
        'tree_method': 'hist'
    }
    model, auc = _fit_xgb_classifier(X_train, y_train, X_val, y_val, params, model_name)
    return auc


# ----------------------
# Training entrypoint
# ----------------------

def run_training():
    os.makedirs(config.ARTIFACT_DIR, exist_ok=True)
    cleanup_artifacts(config.ARTIFACT_DIR)

    print("====== STARTING SCALABLE IAM ML TRAINING PIPELINE ======")
    start_ts = time.time()

    graph_dfs = data_loader.get_all_graph_data()
    embeddings_df = data_loader.get_neo4j_embeddings()
    
    joblib.dump(embeddings_df, os.path.join(config.ARTIFACT_DIR, 'embeddings.pkl'))
    print("💾 Saved embeddings.pkl")

    labeled = _build_training_pairs(graph_dfs, hard_negative_ratio=0.5, neg_multiplier=2.0)

    # Candidate features
    X_cand, y_cand, cand_feature_names = feature_engineering.create_candidate_model_features(labeled.copy(), embeddings_df)
    X_cand = X_cand.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    
    
    
    
    
    # Add this debug code right before the train_test_split line in run_training():

    # Debug the labeled data first
    print(f"🔍 DEBUGGING LABELED DATA:")
    print(f"   Labeled shape: {labeled.shape}")
    print(f"   Labeled columns: {labeled.columns.tolist()}")
    print(f"   Labeled head:\n{labeled.head()}")
    print(f"   Labeled dtypes:\n{labeled.dtypes}")
    print(f"   Null values in labeled:\n{labeled.isnull().sum()}")

    # Debug embeddings
    print(f"\n🔍 DEBUGGING EMBEDDINGS:")
    print(f"   Embeddings shape: {embeddings_df.shape}")
    print(f"   Embeddings columns: {embeddings_df.columns.tolist()}")
    print(f"   Sample embedding IDs: {embeddings_df['originalId'].head().tolist()}")
    print(f"   Embedding types: {embeddings_df.dtypes}")

    # Now debug the feature creation
    print(f"\n🔍 CALLING create_candidate_model_features...")
    try:
        X_cand, y_cand, cand_feature_names = feature_engineering.create_candidate_model_features(labeled.copy(), embeddings_df)
        
        print(f"✅ Feature creation completed!")
        print(f"   X_cand type: {type(X_cand)}")
        print(f"   y_cand type: {type(y_cand)}")
        print(f"   cand_feature_names type: {type(cand_feature_names)}")
        
        if X_cand is not None:
            print(f"   X_cand shape: {X_cand.shape}")
            print(f"   X_cand columns: {X_cand.columns.tolist()[:10]}...")  # First 10 columns
        else:
            print("   ❌ X_cand is None!")
            
        if y_cand is not None:
            print(f"   y_cand shape: {y_cand.shape}")
            print(f"   y_cand unique values: {y_cand.unique()}")
            print(f"   y_cand dtype: {y_cand.dtype}")
        else:
            print("   ❌ y_cand is None!")
            
        if cand_feature_names is not None:
            print(f"   Feature names count: {len(cand_feature_names)}")
        else:
            print("   ❌ cand_feature_names is None!")
            
    except Exception as e:
        print(f"❌ ERROR in create_candidate_model_features: {e}")
        import traceback
        traceback.print_exc()
        return  # Exit early to avoid the train_test_split error

    # Only proceed if we have valid data
    if X_cand is None or y_cand is None:
        print("❌ STOPPING: X_cand or y_cand is None - cannot proceed with train_test_split")
        return
    
    
    
    Xc_train, Xc_val, yc_train, yc_val = train_test_split(X_cand, y_cand, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y_cand)

    # Optuna optimization for candidate model
    print("🔎 Starting Optuna search for candidate model...")
    cand_study = optuna.create_study(direction="maximize")
    cand_study.optimize(lambda trial: _optuna_objective(trial, Xc_train, yc_train, Xc_val, yc_val, "candidate_model"), n_trials=config.OPTUNA_CANDIDATE_TRIALS)
    best_cand_params = cand_study.best_params
    candidate_model, cand_auc = _fit_xgb_classifier(Xc_train, yc_train, Xc_val, yc_val, best_cand_params, model_name="candidate_model")

    joblib.dump(candidate_model, os.path.join(config.ARTIFACT_DIR, 'candidate_model.joblib'))
    joblib.dump(list(X_cand.columns), os.path.join(config.ARTIFACT_DIR, 'candidate_model_features.joblib'))

    # Reranker features
    X_rerank, y_rerank, rerank_feature_names = feature_engineering.create_enhanced_reranker_features(labeled.copy(), embeddings_df, graph_dfs)
    missing_peers = set(EXPECTED_PEER_COLS) - set(X_rerank.columns)
    if missing_peers:
        raise RuntimeError(f"❌ Peer features missing from X_rerank: {sorted(missing_peers)}")
    X_rerank = X_rerank.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    Xr_train, Xr_val, yr_train, yr_val = train_test_split(X_rerank, y_rerank, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y_rerank)

    print("🔎 Starting Optuna search for reranker model...")
    rerank_study = optuna.create_study(direction="maximize")
    rerank_study.optimize(lambda trial: _optuna_objective(trial, Xr_train, yr_train, Xr_val, yr_val, "reranker_model"), n_trials=config.OPTUNA_RERANKER_TRIALS)
    best_rerank_params = rerank_study.best_params
    reranker_model, rerank_auc = _fit_xgb_classifier(Xr_train, yr_train, Xr_val, yr_val, best_rerank_params, model_name="reranker_model")

    joblib.dump(reranker_model, os.path.join(config.ARTIFACT_DIR, 'reranker_model.joblib'))
    joblib.dump(list(X_rerank.columns), os.path.join(config.ARTIFACT_DIR, 'reranker_model_features.joblib'))

    manifest = {
        'model': 'reranker_model',
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'code_version': os.getenv('GIT_COMMIT', 'unknown'),
        'feature_count': int(X_rerank.shape[1]),
        'candidate_feature_count': int(X_cand.shape[1]),
        'best_candidate_params': best_cand_params,
        'best_reranker_params': best_rerank_params
    }
    joblib.dump(manifest, os.path.join(config.ARTIFACT_DIR, 'reranker_model_manifest.joblib'))

    print("====== TRAINING COMPLETE ======")
    print(f"Candidate AUC: {cand_auc:.6f}  |  Reranker AUC: {rerank_auc:.6f}")
    print(f"Artifacts dir: {config.ARTIFACT_DIR}")
    print(f"⏱️ Total training time: {time.time() - start_ts:.2f}s")

# ----------------------
# Utilities (restore)
# ----------------------
import glob, shutil

def cleanup_artifacts(dir_path: str):
    """Safely remove prior artifacts before training. Deletes only known patterns."""
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
        "shap_*.html",
        "*.joblib.bak", "*.pkl.bak",
    ]
    removed = []
    for pat in patterns:
        for fp in glob.glob(os.path.join(dir_path, pat)):
            try:
                if os.path.isfile(fp): os.remove(fp); removed.append(fp)
                elif os.path.isdir(fp): shutil.rmtree(fp); removed.append(fp)
            except Exception as e:
                print(f"⚠️ Could not remove {fp}: {e}")
    print(f"🧹 Artifacts cleanup done. Removed {len(removed)} items from {dir_path}")

def _parse_endpoint_system(ent_id):
    try:
        return int(str(ent_id).split("_")[0])
    except Exception:
        return None



# ----------------------
# Negative sampling (restore)
# ----------------------
from collections import defaultdict

def _build_training_pairs(graph_dfs: dict,
                          hard_negative_ratio: float = 0.5,
                          neg_multiplier: float = 2.0) -> pd.DataFrame:
    """Create labeled (UserId, EntitlementId, HasAccess) pairs with mixed hard negatives."""
    entre = graph_dfs['entrecon'][['UserId','EntitlementId']].drop_duplicates()
    users = graph_dfs['users'][['id','NBusinessRoleId','NOrganisationId']].rename(columns={'id':'UserId'})

    pos_df = entre.copy()
    pos_df['HasAccess'] = 1

    user_ids = pos_df['UserId'].unique().tolist()
    entitlement_ids = graph_dfs['entitlements']['id'].unique().tolist()
    positive_pairs = set(map(tuple, pos_df[['UserId','EntitlementId']].itertuples(index=False, name=None)))

    num_neg_total = int(len(pos_df) * neg_multiplier)
    num_hard = int(num_neg_total * hard_negative_ratio)
    num_easy = num_neg_total - num_hard
    print(f"🧪 Sampling negatives: total={num_neg_total} (hard={num_hard}, easy={num_easy})")

    # Precompute mapping entitlement->system and popularity by role/org
    ent_to_sys = {e: _parse_endpoint_system(e) for e in entitlement_ids}
    labeled_ctx = entre.merge(users, on='UserId', how='left')
    role_pop = (labeled_ctx.groupby(['NBusinessRoleId','EntitlementId']).size().reset_index(name='cnt'))
    org_pop  = (labeled_ctx.groupby(['NOrganisationId','EntitlementId']).size().reset_index(name='cnt'))

    user_ent_map, user_sys_map = defaultdict(set), defaultdict(set)
    for u, e in entre[['UserId','EntitlementId']].itertuples(index=False):
        user_ent_map[u].add(e)
        sys_id = ent_to_sys.get(e)
        if sys_id is not None: user_sys_map[u].add(sys_id)

    def hard_pool_for_user(u):
        own = user_ent_map.get(u, set())
        sys_set = user_sys_map.get(u, set())
        role_id = users.loc[users['UserId']==u, 'NBusinessRoleId']
        org_id  = users.loc[users['UserId']==u, 'NOrganisationId']
        role_id = int(role_id.iloc[0]) if len(role_id)>0 and pd.notna(role_id.iloc[0]) else None
        org_id  = int(org_id.iloc[0])  if len(org_id)>0  and pd.notna(org_id.iloc[0])  else None
        pool = set()
        if sys_set:
            pool.update([e for e in entitlement_ids if ent_to_sys.get(e) in sys_set])
        if role_id is not None:
            pool.update(role_pop[role_pop['NBusinessRoleId']==role_id]['EntitlementId'].tolist())
        if org_id is not None:
            pool.update(org_pop[org_pop['NOrganisationId']==org_id]['EntitlementId'].tolist())
        pool.difference_update(own)
        return list(pool)

    rng = random.Random(config.RANDOM_STATE)
    neg_rows = []

    # Hard negatives
    guard = 0
    while len(neg_rows) < num_hard and guard < num_hard * 20:
        guard += 1
        u = rng.choice(user_ids)
        pool = hard_pool_for_user(u)
        if not pool: continue
        e = rng.choice(pool)
        if (u, e) not in positive_pairs:
            neg_rows.append((u, e, 0))

    # Easy (random) negatives
    guard = 0
    while len(neg_rows) < num_neg_total and guard < num_neg_total * 20:
        guard += 1
        u = rng.choice(user_ids)
        e = rng.choice(entitlement_ids)
        if (u, e) not in positive_pairs:
            neg_rows.append((u, e, 0))

    neg_df = pd.DataFrame(neg_rows, columns=['UserId','EntitlementId','HasAccess'])
    labeled = pd.concat([pos_df[['UserId','EntitlementId','HasAccess']], neg_df], ignore_index=True)
    print(f"📦 Labeled pairs: positives={len(pos_df)}, negatives={len(neg_df)}, total={len(labeled)}")
    return labeled



if __name__ == "__main__":
    run_training()
