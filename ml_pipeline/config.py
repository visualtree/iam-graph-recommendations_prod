import os

# --- DATABASE CREDENTIALS (ENV-DRIVEN) ---
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASS")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

# --- SQL SOURCE (FOR ETL) ---
SQL_DB_HOST = os.environ.get("SQL_DB_HOST", "localhost")
SQL_DB_USER = os.environ.get("SQL_DB_USER", "sa")
SQL_DB_PASS = os.environ.get("SQL_DB_PASS")
SQL_DB_NAME = os.environ.get("SQL_DB_NAME", "F_IACM_Demo")

# --- MODEL & ARTIFACT PARAMETERS ---
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "artifacts")
EMBEDDING_DIMENSION = 64
RANDOM_STATE = 42

# --- OPTUNA HYPERPARAMETER TUNING ---
OPTUNA_N_TRIALS = 30
OPTUNA_CANDIDATE_TRIALS = 30
OPTUNA_RERANKER_TRIALS  = 30

# --- CALIBRATION ---
# Options: "sigmoid" (Platt scaling) or "isotonic"
CALIBRATION_METHOD = os.environ.get("CALIBRATION_METHOD", "sigmoid").lower()


# --- XGBOOST BASE PARAMETERS ---
XGB_PARAMS = {
    'objective': 'binary:logistic', 'eval_metric': 'logloss', 'booster': 'gbtree',
    'tree_method': 'auto', 'random_state': RANDOM_STATE, 'n_jobs': -1
}


def require_neo4j_config() -> None:
    """Fail fast if required Neo4j credentials are missing."""
    if not NEO4J_PASS:
        raise RuntimeError("Missing NEO4J_PASS. Set it via environment variable.")


def require_sql_config() -> None:
    """Fail fast if required SQL credentials are missing."""
    if not SQL_DB_PASS:
        raise RuntimeError("Missing SQL_DB_PASS. Set it via environment variable.")
