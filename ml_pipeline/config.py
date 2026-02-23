# ml_pipeline/config.py (FINAL, COMPLETE PRODUCTION VERSION)

import os

# --- DATABASE CREDENTIALS ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "@bcd1234" # Ensure this is your correct Neo4j password
NEO4J_DATABASE = "neo4j"

# --- MODEL & ARTIFACT PARAMETERS ---
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "artifacts")
EMBEDDING_DIMENSION = 64
RANDOM_STATE = 42

# --- OPTUNA HYPERPARAMETER TUNING ---
OPTUNA_N_TRIALS = 30 
OPTUNA_CANDIDATE_TRIALS = 30
OPTUNA_RERANKER_TRIALS  = 30


# --- XGBOOST BASE PARAMETERS ---
XGB_PARAMS = {
    'objective': 'binary:logistic', 'eval_metric': 'logloss', 'booster': 'gbtree',
    'tree_method': 'auto', 'random_state': RANDOM_STATE, 'n_jobs': -1
}