import time
import logging

from fastapi import APIRouter

from api.schemas import HealthArtifact, HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()

_TOP_LEVEL_KEYS = [
    "candidate_model",
    "candidate_features",
    "reranker_model",
    "reranker_features",
    "embeddings_df",
]
_GRAPH_KEYS = [
    "users",
    "entitlements",
    "entrecon",
    "orgs",
    "endpoints",
    "designations",
]


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Reports service readiness and artifact load status.
    Use as a Kubernetes readiness probe: wait for status == "ok".
    """
    from ml_pipeline.prediction_core import PredictionArtifacts
    from ml_pipeline import config
    import api.app as app_module

    artifacts = PredictionArtifacts._artifacts  # None if not loaded or load failed
    details: list[HealthArtifact] = []

    if artifacts is not None:
        for key in _TOP_LEVEL_KEYS:
            details.append(HealthArtifact(name=key, loaded=(key in artifacts)))
        graph_dfs = artifacts.get("graph_dfs", {})
        for key in _GRAPH_KEYS:
            details.append(HealthArtifact(name=f"graph_dfs.{key}", loaded=(key in graph_dfs)))
        all_loaded = all(d.loaded for d in details)
        api_status = "ok" if all_loaded else "degraded"
    else:
        details = [HealthArtifact(name="artifacts", loaded=False)]
        api_status = "starting"

    uptime = time.monotonic() - app_module._startup_time

    return HealthResponse(
        status=api_status,
        artifacts_loaded=(artifacts is not None),
        artifact_dir=config.ARTIFACT_DIR,
        details=details,
        uptime_seconds=round(uptime, 1),
    )
