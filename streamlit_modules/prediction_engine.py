"""Prediction helpers for Streamlit app."""

import json
import os
import urllib.error
import urllib.request

import pandas as pd
import streamlit as st

from ml_pipeline.prediction_core import run_prediction_pipeline as core_run_prediction_pipeline


def _api_predict(user_id, top_n, candidates, endpoint_id=None):
    """Call FastAPI /predict and return decoded JSON."""
    base_url = os.environ.get("IAM_API_BASE_URL", "http://127.0.0.1:8010").rstrip("/")
    timeout_seconds = float(os.environ.get("IAM_API_TIMEOUT_SECONDS", "120"))
    payload = {
        "user_id": int(user_id),
        "top_n": int(top_n),
        "initial_candidates": int(candidates),
    }
    if endpoint_id is not None:
        payload["endpoint_id"] = int(endpoint_id)

    api_token = os.environ.get("IAM_API_TOKEN", "").strip()
    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["X-API-Key"] = api_token

    request = urllib.request.Request(
        url=f"{base_url}/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"API connection failed: {exc.reason}") from exc


def _adapt_api_response(api_data, models_data):
    """Convert API response into existing Streamlit result contract."""
    rows = []
    for rec in api_data.get("recommendations", []):
        rows.append(
            {
                "EntitlementId": str(rec.get("entitlement_id")),
                "OriginalEntitlementId": str(rec.get("original_entitlement_id")),
                "CandidateScore": float(rec.get("candidate_score", 0.0)),
                "FinalScore": float(rec.get("final_score", 0.0)),
                "Name": rec.get("entitlement_name"),
                "Description": rec.get("entitlement_description"),
                "ApplicationCode": rec.get("endpoint_name"),
            }
        )

    final_recs = pd.DataFrame(rows)
    if not final_recs.empty:
        final_recs = final_recs.sort_values("FinalScore", ascending=False).reset_index(drop=True)
        final_recs = add_entitlement_details_streamlit(
            final_recs,
            {"graph_dfs": models_data["graph_dfs"]},
        )

    return {
        "predictions": final_recs,
        "candidate_features": None,
        "reranker_features": None,
        "stage1_count": int(api_data.get("stage1_count", 0)),
        "total_candidates": int(api_data.get("total_candidates", 0)),
        "user_id": int(api_data.get("user_id", 0)),
        "artifacts": {"graph_dfs": models_data["graph_dfs"]},
        "pipeline_timings_ms": {"api_predict_duration_ms": api_data.get("duration_ms")},
        "pipeline_total_ms": api_data.get("duration_ms"),
    }


def _run_prediction_pipeline_core(user_id, top_n, candidates):
    """Run shared core pipeline and adapt outputs for Streamlit screens."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    def _progress_callback(step, total, message=""):
        pct = int((step / total) * 100) if total else 0
        progress_bar.progress(pct)
        status_text.text(f"{message} ({step}/{total})")

    core_results = core_run_prediction_pipeline(
        user_id=user_id,
        top_n=top_n,
        initial_candidates=candidates,
        progress_callback=_progress_callback,
    )
    if core_results is None:
        st.warning("No candidate entitlements found for this user.")
        return None

    final_recs = add_entitlement_details_streamlit(
        core_results["predictions"].copy(),
        {"graph_dfs": core_results["artifacts"]["graph_dfs"]},
    )

    # Align reranker feature rows to final recommendation order for explanation panels.
    rerank_aligned = None
    rerank_scored = core_results.get("X_rerank_scored")
    if rerank_scored is not None and not rerank_scored.empty:
        rerank_indexed = rerank_scored.set_index("EntitlementId", drop=True)
        ordered_ids = final_recs["EntitlementId"].astype(str).tolist()
        rerank_aligned = rerank_indexed.reindex(ordered_ids).reset_index(drop=True)
    else:
        rerank_aligned = core_results.get("X_rerank")

    progress_bar.progress(100)
    status_text.text("Prediction pipeline completed.")

    return {
        "predictions": final_recs,
        "candidate_features": core_results.get("X_cand"),
        "reranker_features": rerank_aligned,
        "stage1_count": core_results["stage1_count"],
        "total_candidates": core_results["total_candidates"],
        "user_id": core_results["user_id"],
        "artifacts": core_results["artifacts"],
        "pipeline_timings_ms": core_results.get("timings_ms", {}),
        "pipeline_total_ms": core_results.get("total_ms"),
    }


def run_prediction_pipeline(user_id, models_data, top_n=5, candidates=100, endpoint_id=None):
    """API-first mode with local-core fallback."""
    backend_mode = os.environ.get("IAM_PREDICTION_BACKEND", "api_first").strip().lower()
    fallback_enabled = os.environ.get("IAM_API_FALLBACK_TO_CORE", "true").strip().lower() in {
        "1",
        "true",
        "yes",
    }

    try:
        if backend_mode in {"core", "local"}:
            return _run_prediction_pipeline_core(user_id, top_n, candidates)

        api_data = _api_predict(
            user_id=user_id,
            top_n=top_n,
            candidates=candidates,
            endpoint_id=endpoint_id,
        )
        return _adapt_api_response(api_data, models_data)
    except Exception as exc:
        if backend_mode in {"api", "api_only"} or not fallback_enabled:
            st.error(f"Prediction pipeline failed: {exc}")
            return None
        st.warning(f"API unavailable, switching to local pipeline: {exc}")
        try:
            return _run_prediction_pipeline_core(user_id, top_n, candidates)
        except Exception as inner_exc:
            st.error(f"Prediction pipeline failed: {inner_exc}")
            return None


def add_entitlement_details_streamlit(predictions_df, artifacts):
    """Add entitlement/system labels for Streamlit rendering."""
    graph_dfs = artifacts["graph_dfs"]

    predictions_df = predictions_df.merge(
        graph_dfs["entitlements"][["id", "Name", "Description"]],
        left_on="EntitlementId",
        right_on="id",
        how="left",
        suffixes=("", "_ent"),
    )

    predictions_df["EndpointSystemId"] = (
        predictions_df["EntitlementId"].astype(str).str.split("_").str[0].astype("Int64")
    )

    if "endpoints" in graph_dfs:
        endpoint_cols = ["id", "ApplicationCode"]
        if "DisplayName" in graph_dfs["endpoints"].columns:
            endpoint_cols.append("DisplayName")

        predictions_df = predictions_df.merge(
            graph_dfs["endpoints"][endpoint_cols],
            left_on="EndpointSystemId",
            right_on="id",
            how="left",
            suffixes=("", "_sys"),
        )

    return predictions_df


def calculate_peer_insights(user_id, entitlement_id, graph_dfs):
    """Peer adoption stats used by Streamlit detail cards."""
    try:
        user_info = graph_dfs["users"][graph_dfs["users"]["id"] == user_id]
        if user_info.empty:
            return None

        user_row = user_info.iloc[0]
        user_role = user_row.get("NBusinessRoleId")
        user_org = user_row.get("NOrganisationId")
        user_manager = user_row.get("ManagerId")

        def _peer_block(peers_df):
            access_df = graph_dfs["entrecon"][
                (graph_dfs["entrecon"]["UserId"].isin(peers_df["id"]))
                & (graph_dfs["entrecon"]["EntitlementId"] == entitlement_id)
            ]
            users_with_access = peers_df[peers_df["id"].isin(access_df["UserId"])]
            total = len(peers_df)
            with_access = len(access_df)
            return {
                "total": total,
                "with_access": with_access,
                "adoption_rate": (with_access / total) if total else 0.0,
                "peer_names": users_with_access["UserName"].tolist()[:5],
            }

        insights = {}

        if pd.notna(user_role) and pd.notna(user_org):
            close_peers = graph_dfs["users"][
                (graph_dfs["users"]["NBusinessRoleId"] == user_role)
                & (graph_dfs["users"]["NOrganisationId"] == user_org)
                & (graph_dfs["users"]["id"] != user_id)
                & (graph_dfs["users"]["IsActive"] == True)
            ]
            insights["close_peers"] = _peer_block(close_peers)

        if pd.notna(user_manager):
            team_peers = graph_dfs["users"][
                (graph_dfs["users"]["ManagerId"] == user_manager)
                & (graph_dfs["users"]["id"] != user_id)
                & (graph_dfs["users"]["IsActive"] == True)
            ]
            insights["direct_team"] = _peer_block(team_peers)

        if pd.notna(user_role):
            role_peers = graph_dfs["users"][
                (graph_dfs["users"]["NBusinessRoleId"] == user_role)
                & (graph_dfs["users"]["id"] != user_id)
                & (graph_dfs["users"]["IsActive"] == True)
            ]
            insights["role_peers"] = _peer_block(role_peers)

        if pd.notna(user_org):
            dept_peers = graph_dfs["users"][
                (graph_dfs["users"]["NOrganisationId"] == user_org)
                & (graph_dfs["users"]["id"] != user_id)
                & (graph_dfs["users"]["IsActive"] == True)
            ]
            insights["dept_peers"] = _peer_block(dept_peers)

        return insights
    except Exception:
        return None


def format_predictions_for_streamlit(results):
    """Format predictions for compact table rendering."""
    if not results or results["predictions"].empty:
        return None

    final_recs = results["predictions"].copy()
    artifacts = results.get("artifacts")
    if artifacts:
        final_recs = add_entitlement_details_streamlit(final_recs, artifacts)

    final_recs["OriginalEntitlementId"] = final_recs["EntitlementId"].astype(str).str.split("_").str[1]

    display_cols = ["OriginalEntitlementId", "Name", "FinalScore"]
    if "ApplicationCode" in final_recs.columns:
        display_cols.insert(-1, "ApplicationCode")
    if "DisplayName" in final_recs.columns:
        display_cols.insert(-1, "DisplayName")

    display_df = final_recs[display_cols].copy().rename(
        columns={
            "OriginalEntitlementId": "Entitlement ID",
            "Name": "Entitlement Name",
            "ApplicationCode": "Application",
            "DisplayName": "System Name",
            "FinalScore": "Confidence Score",
        }
    )

    if "Confidence Score" in display_df.columns:
        display_df["Confidence Score"] = display_df["Confidence Score"].apply(lambda x: f"{x:.1%}")

    return display_df
