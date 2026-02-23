# ml_pipeline/prediction_core.py - SINGLE SOURCE OF TRUTH FOR ALL PREDICTION LOGIC

import os
import threading
import logging
import time
import joblib
import pandas as pd
import numpy as np
import warnings
from . import config, feature_engineering

warnings.filterwarnings("ignore", category=UserWarning, module='shap')
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


class PredictionArtifacts:
    """Centralized artifact loading and management"""
    _instance = None
    _artifacts = None
    _lock = threading.Lock()

    @classmethod
    def get_artifacts(cls):
        """Thread-safe singleton — loads artifacts only once across all threads."""
        # Fast path: already loaded, no lock needed (read after guaranteed write)
        if cls._artifacts is not None:
            return cls._artifacts
        # Slow path: first load — acquire lock and double-check
        with cls._lock:
            if cls._artifacts is None:
                cls._instance = cls()
                cls._artifacts = cls._load_all_artifacts()
        return cls._artifacts

    @staticmethod
    def _load_all_artifacts():
        """Load all prediction artifacts"""
        d = config.ARTIFACT_DIR
        logger.info("Loading all production artifacts from: %s", d)
        try:
            artifacts = {
                'candidate_model':    joblib.load(os.path.join(d, 'candidate_model.joblib')),
                'candidate_features': joblib.load(os.path.join(d, 'candidate_model_features.joblib')),
                'reranker_model':     joblib.load(os.path.join(d, 'reranker_model.joblib')),
                'reranker_features':  joblib.load(os.path.join(d, 'reranker_model_features.joblib')),
                'embeddings_df':      joblib.load(os.path.join(d, 'embeddings.pkl')),
                'graph_dfs': {
                    'users':        joblib.load(os.path.join(d, 'users.pkl')),
                    'entitlements': joblib.load(os.path.join(d, 'entitlements.pkl')),
                    'entrecon':     joblib.load(os.path.join(d, 'entrecon.pkl')),
                    'orgs':         joblib.load(os.path.join(d, 'orgs.pkl')),
                    'endpoints':    joblib.load(os.path.join(d, 'endpoints.pkl')),
                    'designations': joblib.load(os.path.join(d, 'designations.pkl')),
                }
            }
            artifacts['peer_lookup'] = feature_engineering.build_peer_lookup_cache(artifacts['graph_dfs'])
            # Cap n_jobs to 1 to prevent thread oversubscription under concurrent requests
            for key in ('candidate_model', 'reranker_model'):
                if hasattr(artifacts[key], 'n_jobs'):
                    artifacts[key].n_jobs = 1
            logger.info("All artifacts loaded successfully")
            return artifacts
        except FileNotFoundError as e:
            logger.error("Artifact not found: %s", e)
            raise FileNotFoundError(f"Artifact not found: {e}. Please run the training pipeline first.")


def _hard_fail_feature_alignment(X_df, expected_features, model_name="model"):
    """Hard-fail feature alignment - ensures exact match"""
    exp = list(expected_features)
    act = list(X_df.columns)

    missing_in_calc = [f for f in exp if f not in act]
    extra_in_calc   = [f for f in act if f not in exp]

    logger.info("%s expected features: %d", model_name, len(exp))
    logger.info("%s calculated features: %d", model_name, len(act))

    if missing_in_calc or extra_in_calc:
        raise RuntimeError(
            f"[{model_name.upper()} FEATURE MISMATCH]\n"
            f"  • Missing in calculated: {missing_in_calc}\n"
            f"  • Extra in calculated: {extra_in_calc}\n"
        )

    X_ordered = X_df[exp].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    logger.info("%s feature alignment OK; shape: %s", model_name, X_ordered.shape)
    return X_ordered


def run_prediction_pipeline(
    user_id,
    top_n=5,
    initial_candidates=100,
    endpoint_id=None,
    progress_callback=None,
):
    """
    Core prediction pipeline - used by both console and Streamlit

    Args:
        user_id: User ID to generate predictions for
        top_n: Number of final recommendations
        initial_candidates: Number of initial candidates to generate
        endpoint_id: Optional endpoint/system ID to scope candidate entitlements
        progress_callback: Optional callback function for progress updates (used by Streamlit)

    Returns:
        Dictionary with predictions and metadata, or None if no candidates exist.
    """

    def update_progress(step, total, message=""):
        if progress_callback:
            progress_callback(step, total, message)
        else:
            logger.info("Progress: %d/%d - %s", step, total, message)

    pipeline_start = time.perf_counter()
    timings_ms = {}

    try:
        user_id = int(user_id)
    except (ValueError, TypeError):
        raise ValueError(f"User ID must be a valid integer. You provided '{user_id}'.")

    # Load artifacts (cached after first load)
    update_progress(1, 10, "Loading artifacts...")
    t0 = time.perf_counter()
    artifacts = PredictionArtifacts.get_artifacts()
    timings_ms["artifacts_load"] = (time.perf_counter() - t0) * 1000

    # Get candidate entitlements
    update_progress(2, 10, "Finding candidate entitlements...")
    t0 = time.perf_counter()
    candidate_ents_df = artifacts['graph_dfs']['entitlements'][
        ~artifacts['graph_dfs']['entitlements']['id'].isin(
            artifacts['graph_dfs']['entrecon'][
                artifacts['graph_dfs']['entrecon']['UserId'] == user_id
            ]['EntitlementId']
        )
    ].copy()

    if endpoint_id is not None:
        try:
            endpoint_id = int(endpoint_id)
        except (ValueError, TypeError):
            raise ValueError(f"endpoint_id must be a valid integer. You provided '{endpoint_id}'.")

        if 'EndpointSystemId' in candidate_ents_df.columns:
            candidate_ents_df = candidate_ents_df[
                pd.to_numeric(candidate_ents_df['EndpointSystemId'], errors='coerce') == endpoint_id
            ].copy()
        else:
            parsed_endpoint_ids = (
                candidate_ents_df['id']
                .astype(str)
                .str.split('_')
                .str[0]
                .pipe(pd.to_numeric, errors='coerce')
            )
            candidate_ents_df = candidate_ents_df[parsed_endpoint_ids == endpoint_id].copy()
    timings_ms["candidate_filter"] = (time.perf_counter() - t0) * 1000

    if candidate_ents_df.empty:
        return None

    # Create candidates DataFrame
    update_progress(3, 10, "Creating candidate matrix...")
    t0 = time.perf_counter()
    candidates_df = pd.DataFrame({
        'UserId': [user_id] * len(candidate_ents_df),
        'EntitlementId': candidate_ents_df['id'].tolist()
    })
    candidates_df['UserId'] = candidates_df['UserId'].astype('int64')
    candidates_df['EntitlementId'] = candidates_df['EntitlementId'].astype('string')
    timings_ms["candidate_matrix"] = (time.perf_counter() - t0) * 1000

    # Stage 1: Candidate scoring
    update_progress(4, 10, "Generating candidate features...")
    t0 = time.perf_counter()
    X_cand, _, _ = feature_engineering.create_candidate_model_features(
        candidates_df.copy(), artifacts['embeddings_df']
    )
    timings_ms["candidate_features"] = (time.perf_counter() - t0) * 1000

    update_progress(5, 10, "Aligning candidate features...")
    t0 = time.perf_counter()
    X_cand = _hard_fail_feature_alignment(X_cand, artifacts['candidate_features'], "candidate")
    timings_ms["candidate_align"] = (time.perf_counter() - t0) * 1000

    update_progress(6, 10, "Stage 1: Scoring candidates...")
    t0 = time.perf_counter()
    pred_probs_cand = artifacts['candidate_model'].predict_proba(X_cand)[:, 1]
    candidates_df['CandidateScore'] = pred_probs_cand
    top_candidates = candidates_df.sort_values('CandidateScore', ascending=False).head(initial_candidates)
    timings_ms["candidate_score_rank"] = (time.perf_counter() - t0) * 1000

    # Stage 2: Reranking
    update_progress(7, 10, "Creating reranker features...")
    t0 = time.perf_counter()
    X_rerank, _, _ = feature_engineering.create_enhanced_reranker_features(
        top_candidates.copy(),
        artifacts['embeddings_df'],
        artifacts['graph_dfs'],
        peer_lookup_cache=artifacts.get('peer_lookup'),
    )
    timings_ms["reranker_features"] = (time.perf_counter() - t0) * 1000

    update_progress(8, 10, "Aligning reranker features...")
    t0 = time.perf_counter()
    X_rerank = _hard_fail_feature_alignment(X_rerank, artifacts['reranker_features'], "reranker")
    timings_ms["reranker_align"] = (time.perf_counter() - t0) * 1000

    update_progress(9, 10, "Stage 2: Final reranking...")
    t0 = time.perf_counter()
    pred_probs_rerank = artifacts['reranker_model'].predict_proba(X_rerank)[:, 1]
    top_candidates['FinalScore'] = pred_probs_rerank

    final_recs = top_candidates.sort_values('FinalScore', ascending=False).head(top_n)
    final_recs = final_recs.copy()
    final_recs['OriginalEntitlementId'] = final_recs['EntitlementId'].astype(str).str.split('_').str[1]
    timings_ms["reranker_score_rank"] = (time.perf_counter() - t0) * 1000

    update_progress(10, 10, "Prediction pipeline completed!")
    total_ms = (time.perf_counter() - pipeline_start) * 1000
    logger.info(
        "Pipeline timing | user_id=%d endpoint_id=%s candidates=%d stage1_count=%d total_ms=%.2f "
        "artifacts_load_ms=%.2f candidate_filter_ms=%.2f candidate_matrix_ms=%.2f "
        "candidate_features_ms=%.2f candidate_align_ms=%.2f candidate_score_rank_ms=%.2f "
        "reranker_features_ms=%.2f reranker_align_ms=%.2f reranker_score_rank_ms=%.2f",
        user_id,
        endpoint_id,
        len(candidate_ents_df),
        len(top_candidates),
        total_ms,
        timings_ms.get("artifacts_load", 0.0),
        timings_ms.get("candidate_filter", 0.0),
        timings_ms.get("candidate_matrix", 0.0),
        timings_ms.get("candidate_features", 0.0),
        timings_ms.get("candidate_align", 0.0),
        timings_ms.get("candidate_score_rank", 0.0),
        timings_ms.get("reranker_features", 0.0),
        timings_ms.get("reranker_align", 0.0),
        timings_ms.get("reranker_score_rank", 0.0),
    )

    return {
        'predictions': final_recs,
        'total_candidates': len(candidate_ents_df),
        'stage1_count': len(top_candidates),
        'user_id': user_id,
        'X_rerank': X_rerank,
        'artifacts': artifacts
    }


def calculate_peer_insights(user_id, entitlement_id):
    """Calculate peer adoption insights"""
    try:
        artifacts = PredictionArtifacts.get_artifacts()
        graph_dfs = artifacts['graph_dfs']

        test_df = pd.DataFrame({
            'UserId': [user_id],
            'EntitlementId': [entitlement_id]
        })

        peer_features = feature_engineering.calculate_peer_adoption_features(test_df, graph_dfs)

        if peer_features.empty:
            return None

        peer_data = peer_features.iloc[0]

        return {
            'close_peers': {
                'adoption_rate': peer_data.get('close_peer_adoption_rate', 0),
                'total': peer_data.get('close_peer_count', 0)
            },
            'direct_team': {
                'adoption_rate': peer_data.get('direct_team_adoption_rate', 0),
                'total': peer_data.get('direct_team_count', 0)
            },
            'role_peers': {
                'adoption_rate': peer_data.get('role_peer_adoption_rate', 0),
                'total': peer_data.get('role_peer_count', 0)
            },
            'dept_peers': {
                'adoption_rate': peer_data.get('dept_peer_adoption_rate', 0),
                'total': peer_data.get('dept_peer_count', 0)
            }
        }

    except Exception as e:
        logger.exception("Error calculating peer insights for user=%s entitlement=%s", user_id, entitlement_id)
        return None


def generate_shap_explanation(results, output_path=None):
    """Generate SHAP explanation for the top recommendation"""
    try:
        import shap

        final_recs = results['predictions']
        X_rerank = results['X_rerank']
        reranker_model = results['artifacts']['reranker_model']
        reranker_features = results['artifacts']['reranker_features']
        user_id = results['user_id']

        if final_recs.empty:
            return None

        logger.info("Generating SHAP explanation for the top recommendation (user=%s)", user_id)

        top_rec_details = final_recs.iloc[0]
        prediction_features_series = X_rerank.iloc[0]
        prediction_features_reshaped = prediction_features_series.values.reshape(1, -1)

        explainer = shap.TreeExplainer(reranker_model)
        shap_values = explainer.shap_values(prediction_features_reshaped)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification, take positive class

        shap.initjs()
        force_plot = shap.force_plot(
            explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
            shap_values,
            prediction_features_series,
            feature_names=reranker_features
        )

        if output_path is None:
            output_path = f"shap_force_plot_user_{user_id}_ent_{top_rec_details['OriginalEntitlementId']}.html"

        shap.save_html(output_path, force_plot)
        logger.info("SHAP force plot saved to '%s'", output_path)

        return output_path

    except ImportError:
        logger.warning("SHAP not available. Install with: pip install shap")
        return None
    except Exception as e:
        logger.warning("Could not generate SHAP explanation: %s", e)
        return None
