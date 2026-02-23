# ml_pipeline/train.py (UPDATED FOR TYPE SAFETY)

import os, joblib, time, pandas as pd, xgboost as xgb, optuna, random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from . import config, data_loader, feature_engineering


# --- add near the top of train.py ---
import glob, shutil

def cleanup_artifacts(dir_path: str):
    """
    Safely remove old model/feature/embedding artifacts before training.
    Only deletes known patterns inside dir_path; refuses to run on suspicious paths.
    """
    # Safety rails
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
        "users.pkl", "entitlements.pkl", "entrecon.pkl",
        "orgs.pkl", "endpoints.pkl", "designations.pkl", "accounts.pkl",
        # any SHAP/html explainability files from previous runs
        "shap_*.html",
        # any stray joblibs/pkls you know you generate
        "*.joblib.bak", "*.pkl.bak"
    ]

    removed = []
    for pat in patterns:
        for fp in glob.glob(os.path.join(dir_path, pat)):
            try:
                if os.path.isfile(fp):
                    os.remove(fp); removed.append(fp)
                elif os.path.isdir(fp):
                    shutil.rmtree(fp); removed.append(fp)
            except Exception as e:
                print(f"⚠️ Could not remove {fp}: {e}")

    print(f"🧹 Artifacts cleanup done. Removed {len(removed)} items.")


def train_model(X, y, model_name):
    print(f"\n--- Training Model: {model_name} ---")

    # Validate inputs
    if y is None:
        raise ValueError(f"{model_name}: labels 'y' are None")

    # Drop rows where y is NaN and align X
    mask = y.notna()
    if mask.sum() != len(y):
        dropped = int((~mask).sum())
        print(f"⚠️  Dropping {dropped} row(s) with NaN labels before train/test split.")
    X = X.loc[mask].copy()
    y = y.loc[mask].astype(int)

    # Handle single class case
    stratify_arg = y if y.nunique() > 1 else None
    if stratify_arg is None:
        print("⚠️  Only one class present in y; proceeding without stratify.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=stratify_arg
    )

    def objective(trial):
        if y_train.value_counts().get(1, 0) == 0:
            return 0.5
        scale_pos_weight = y_train.value_counts().get(0, 1) / y_train.value_counts().get(1, 1)
        param = {
            'objective': 'binary:logistic', 'eval_metric': 'logloss', 'booster': 'gbtree',
            'scale_pos_weight': scale_pos_weight,
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        }
        model = xgb.XGBClassifier(**param, n_jobs=-1, random_state=config.RANDOM_STATE)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
        preds = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, preds)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE))
    study.optimize(objective, n_trials=config.OPTUNA_N_TRIALS)
    print(f"✅ Optuna best ROC AUC for {model_name}: {study.best_value:.4f}")

    print(f"🚀 Training final {model_name} on all data...")
    final_params = {**config.XGB_PARAMS, **study.best_params}
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X, y)
    
    joblib.dump(final_model, os.path.join(config.ARTIFACT_DIR, f'{model_name}.joblib'))
    joblib.dump(list(X.columns), os.path.join(config.ARTIFACT_DIR, f'{model_name}_features.joblib'))
    print(f"✅ {model_name} and its features saved.")

# OPTIONAL ENHANCEMENT: train.py - Add feature validation

def run_training():



    start_time = time.time()
    os.makedirs(config.ARTIFACT_DIR, exist_ok=True)
    cleanup_artifacts(config.ARTIFACT_DIR)
    
    print("====== STARTING SCALABLE IAM ML TRAINING PIPELINE ======")
    graph_dfs = data_loader.get_all_graph_data()
    embeddings_df = data_loader.get_neo4j_embeddings()
    
    # Add this after: embeddings_df = data_loader.get_neo4j_embeddings()
    print(f"🔍 EMBEDDING DEBUG:")
    print(f"   Embeddings shape: {embeddings_df.shape}")
    print(f"   Sample embedding type: {type(embeddings_df['embedding'].iloc[0])}")
    print(f"   Sample embedding length: {len(embeddings_df['embedding'].iloc[0])}")
    print(f"   Sample embedding: {embeddings_df['embedding'].iloc[0][:10]}")

    # Also check what happens in _expand_embeddings
    # sample_user_ids = ml_labeled['UserId'].unique()[:5]
    # user_raw = embeddings_df[embeddings_df['originalId'].isin(sample_user_ids)].copy()
    # print(f"   User embeddings found: {len(user_raw)}")
    # if len(user_raw) > 0:
        # sample_emb = user_raw['embedding'].iloc[0]
        # print(f"   User embedding length: {len(sample_emb)}")
    
    
    print("🚀 Preparing labeled dataset with memory-efficient negative sampling...")
    pos_df = graph_dfs['entrecon'].copy()
    if pos_df.empty:
        raise ValueError("No positive access examples (HAS_ACCESS_TO) found in graph. Check ETL script.")
    pos_df['HasAccess'] = 1
    
    # TYPE SAFETY: Ensure consistent types for negative sampling
    print(f"🔧 Ensuring type consistency for negative sampling...")
    user_ids = graph_dfs['users']['id'].tolist()
    entitlement_ids = graph_dfs['entitlements']['id'].tolist()
    
    # Verify types match between positive and negative sampling
    print(f"   User IDs type: {type(user_ids[0]) if user_ids else 'None'}")
    print(f"   Entitlement IDs type: {type(entitlement_ids[0]) if entitlement_ids else 'None'}")
    print(f"   Positive pair types: UserId={pos_df['UserId'].dtype}, EntitlementId={pos_df['EntitlementId'].dtype}")
    
    if not user_ids or not entitlement_ids:
        raise ValueError("User or Entitlement data is empty. Cannot perform negative sampling.")
    
    # Create positive pairs set with proper types
    positive_pairs = set()
    for _, row in pos_df.iterrows():
        user_id = row['UserId']
        ent_id = str(row['EntitlementId'])  # Ensure string for consistency
        positive_pairs.add((user_id, ent_id))
    
    num_negatives_to_sample = len(pos_df) * 2
    neg_samples = []
    
    print(f"Sampling {num_negatives_to_sample} negative examples...")
    random.seed(config.RANDOM_STATE)
    attempts = 0
    max_attempts = num_negatives_to_sample * 10  # Prevent infinite loop
    
    while len(neg_samples) < num_negatives_to_sample and attempts < max_attempts:
        rand_user = random.choice(user_ids)
        rand_ent = str(random.choice(entitlement_ids))  # Ensure string consistency
        
        if (rand_user, rand_ent) not in positive_pairs:
            neg_samples.append({
                'UserId': rand_user, 
                'EntitlementId': rand_ent,
                'HasAccess': 0
            })
        attempts += 1
    
    if len(neg_samples) < num_negatives_to_sample:
        print(f"⚠️ Warning: Only generated {len(neg_samples)} negative samples out of {num_negatives_to_sample} requested")
    
    neg_df = pd.DataFrame(neg_samples)
    
    # TYPE SAFETY: Ensure consistent types in final dataset
    if not neg_df.empty:
        neg_df['UserId'] = neg_df['UserId'].astype('int64')
        neg_df['EntitlementId'] = neg_df['EntitlementId'].astype('string')
    
    # Ensure positive df has consistent types too
    pos_df['UserId'] = pos_df['UserId'].astype('int64')
    pos_df['EntitlementId'] = pos_df['EntitlementId'].astype('string')
    
    ml_labeled = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)
    
    print(f"🔍 EMBEDDING DEBUG:")
    print(f"   Embeddings shape: {embeddings_df.shape}")
    print(f"   Sample embedding type: {type(embeddings_df['embedding'].iloc[0])}")
    print(f"   Sample embedding length: {len(embeddings_df['embedding'].iloc[0])}")
    print(f"   Sample embedding: {embeddings_df['embedding'].iloc[0][:10]}")
        
    # ADDITIONAL TYPE SAFETY: Check for and remove any duplicate pairs
    print(f"\n🔍 TRAINING DATA QUALITY CHECK:")
    print(f"   Total ml_labeled rows: {len(ml_labeled)}")
    duplicate_pairs = ml_labeled.duplicated(subset=['UserId', 'EntitlementId']).sum()
    if duplicate_pairs > 0:
        print(f"   ⚠️ Found {duplicate_pairs} duplicate user-entitlement pairs")
        ml_labeled = ml_labeled.drop_duplicates(subset=['UserId', 'EntitlementId']).reset_index(drop=True)
        print(f"   🔧 Cleaned to {len(ml_labeled)} unique pairs")
    
    print(f"✅ Labeled dataset prepared with {len(ml_labeled)} samples.")
    print(f"   Positive: {len(pos_df)}, Negative: {len(neg_df)}")
    print(f"   Final types - UserId: {ml_labeled['UserId'].dtype}, EntitlementId: {ml_labeled['EntitlementId'].dtype}")
    
    print("\n--- Generating features for Candidate Model ---")
    X_cand, y_cand, cand_features = feature_engineering.create_candidate_model_features(ml_labeled.copy(), embeddings_df)
    
    # ✅ VALIDATION: Check candidate features
    print(f"✅ Candidate model features ({len(cand_features)}): {cand_features[:5]}...")
    
    train_model(X_cand, y_cand, 'candidate_model')

    print("\n--- Generating features for Re-Ranker Model (Enhanced with ALL Peer Features) ---")
    X_rerank, y_rerank, rerank_features = feature_engineering.create_enhanced_reranker_features(ml_labeled.copy(), embeddings_df, graph_dfs)
    
    # ✅ VALIDATION: Check reranker features, especially peer features
    peer_features_found = [f for f in rerank_features if 'peer' in f]
    print(f"✅ Reranker model features ({len(rerank_features)} total)")
    print(f"✅ Peer features found ({len(peer_features_found)}): {peer_features_found}")
    
    # Validate we have all 8 expected peer features
    expected_peer_features = [
        'close_peer_adoption_rate',
        'direct_team_adoption_rate',
        'role_peer_adoption_rate',
        'dept_peer_adoption_rate',
        'close_peer_count', 
        'direct_team_count',
        'role_peer_count', 
        'dept_peer_count'
    ]
    
    missing_peer_features = set(expected_peer_features) - set(peer_features_found)
    if missing_peer_features:
        print(f"⚠️ WARNING: Missing expected peer features: {missing_peer_features}")
    else:
        print(f"✅ All 8 peer features present in training data")
    
    train_model(X_rerank, y_rerank, 'reranker_model')

    print("\n🚀 Saving supporting artifacts...")
    for name, df in graph_dfs.items():
        df.to_pickle(os.path.join(config.ARTIFACT_DIR, f'{name}.pkl'))
    embeddings_df.to_pickle(os.path.join(config.ARTIFACT_DIR, 'embeddings.pkl'))
    print("✅ Supporting artifacts saved.")
    
    print(f"\n====== SCALABLE TRAINING PIPELINE COMPLETED IN {time.time() - start_time:.2f} SECONDS ======")
    
    # ✅ FINAL VALIDATION: Print feature counts for verification
    print(f"\n📊 FINAL MODEL SPECIFICATIONS:")
    print(f"   Candidate Model: {len(cand_features)} features")
    print(f"   Reranker Model: {len(rerank_features)} features ({len(peer_features_found)} peer)")

if __name__ == '__main__':
    run_training()