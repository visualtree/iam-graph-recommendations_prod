from __future__ import annotations

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict

from fastapi import APIRouter, HTTPException, Request, status

from api.dependencies import ArtifactsDep, AuthDep, ExecutorDep, get_semaphore
from api.schemas import (
    PeerGroup,
    PeerInsightsRequest,
    PeerInsightsResponse,
    PredictRequest,
    PredictResponse,
    Recommendation,
    ShapJobAccepted,
    ShapJobStatus,
    ShapRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()
ENDPOINT_SCOPED_INITIAL_CANDIDATES = 25


# ── In-Memory SHAP Job Store ─────────────────────────────────────────────────
# Protected by an asyncio.Lock: all reads/writes happen from async coroutines.
# The sync thread (_sync_shap) returns a value to the coroutine; it never
# touches _job_store directly.
#
# Cleanup: jobs older than JOB_TTL_SECONDS are evicted on the next GET
# for that job (TTL-on-access). No background sweep needed for MVP.

_job_store: Dict[str, dict] = {}
_job_store_lock = asyncio.Lock()
JOB_TTL_SECONDS = 3600  # 1 hour


async def _register_job(job_id: str) -> None:
    async with _job_store_lock:
        _job_store[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "output_path": None,
            "error": None,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }


async def _update_job(job_id: str, **kwargs) -> None:
    async with _job_store_lock:
        if job_id in _job_store:
            _job_store[job_id].update(kwargs)
            _job_store[job_id]["updated_at"] = datetime.now(timezone.utc)


# ── POST /predict ─────────────────────────────────────────────────────────────

@router.post("", response_model=PredictResponse, status_code=200)
async def predict(
    body: PredictRequest,
    request: Request,
    _: AuthDep,
    artifacts: ArtifactsDep,
    executor: ExecutorDep,
):
    """
    Run the full 2-stage entitlement recommendation pipeline for a user.

    CPU-heavy work (feature engineering + two XGBoost predict_proba calls)
    runs in a thread pool so the event loop stays free for health checks and
    SHAP status polls.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    effective_initial_candidates = body.initial_candidates
    if body.endpoint_id is not None and "initial_candidates" not in body.model_fields_set:
        effective_initial_candidates = ENDPOINT_SCOPED_INITIAL_CANDIDATES
    logger.info(
        "predict: user_id=%d endpoint_id=%s top_n=%d initial_candidates=%d request_id=%s",
        body.user_id,
        body.endpoint_id,
        body.top_n,
        effective_initial_candidates,
        request_id,
    )

    semaphore = get_semaphore()
    request_start = time.perf_counter()
    queue_wait_ms = 0.0
    compute_ms = 0.0

    async with semaphore:
        queue_wait_ms = (time.perf_counter() - request_start) * 1000
        compute_start = time.perf_counter()
        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                executor,
                _sync_predict,
                body.user_id,
                body.top_n,
                effective_initial_candidates,
                body.endpoint_id,
            )
        except ValueError as exc:
            safe_detail = str(exc).encode("ascii", errors="backslashreplace").decode("ascii")
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=safe_detail)
        except RuntimeError as exc:
            safe_detail = str(exc).encode("ascii", errors="backslashreplace").decode("ascii")
            logger.error("Feature alignment error (request_id=%s): %s", request_id, safe_detail)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=safe_detail)
        except Exception as exc:
            safe_detail = str(exc).encode("ascii", errors="backslashreplace").decode("ascii")
            logger.error(
                "Unexpected error in predict for user_id=%d (request_id=%s): %s",
                body.user_id,
                request_id,
                safe_detail,
            )
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction pipeline failed")
        finally:
            compute_ms = (time.perf_counter() - compute_start) * 1000

    if results is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {body.user_id} has no unassigned entitlement candidates."
        )

    duration_ms = (time.perf_counter() - request_start) * 1000
    logger.info(
        "predict timing: request_id=%s user_id=%d queue_wait_ms=%.2f compute_ms=%.2f total_ms=%.2f",
        request_id,
        body.user_id,
        queue_wait_ms,
        compute_ms,
        duration_ms,
    )

    # Build lookup tables from artifacts for entitlement enrichment
    artifacts      = results["artifacts"]
    ents_df        = artifacts["graph_dfs"]["entitlements"].set_index("id")
    endpoints_df   = artifacts["graph_dfs"]["endpoints"].set_index("id")

    recommendations = []
    for _, row in results["predictions"].iterrows():
        ent_id   = str(row["EntitlementId"])
        ent_row  = ents_df.loc[ent_id] if ent_id in ents_df.index else None
        ep_id    = int(ent_row["EndpointSystemId"]) if ent_row is not None else None
        ep_row   = endpoints_df.loc[ep_id] if (ep_id is not None and ep_id in endpoints_df.index) else None

        recommendations.append(Recommendation(
            entitlement_id=ent_id,
            original_entitlement_id=str(row["OriginalEntitlementId"]),
            entitlement_name=str(ent_row["Name"]) if ent_row is not None else None,
            entitlement_description=str(ent_row["Description"]) if ent_row is not None else None,
            endpoint_id=ep_id,
            endpoint_name=str(ep_row["DisplayName"]) if ep_row is not None else None,
            candidate_score=float(row["CandidateScore"]),
            final_score=float(row["FinalScore"]),
        ))

    return PredictResponse(
        user_id=body.user_id,
        total_candidates=results["total_candidates"],
        stage1_count=results["stage1_count"],
        recommendations=recommendations,
        request_id=request_id,
        duration_ms=round(duration_ms, 2),
    )


def _sync_predict(
    user_id: int,
    top_n: int,
    initial_candidates: int,
    endpoint_id: int | None,
) -> dict | None:
    """Thin synchronous wrapper — runs in ThreadPoolExecutor."""
    from ml_pipeline.prediction_core import run_prediction_pipeline
    return run_prediction_pipeline(
        user_id=user_id,
        top_n=top_n,
        initial_candidates=initial_candidates,
        endpoint_id=endpoint_id,
        progress_callback=None,
    )


# ── POST /predict/peer-insights ───────────────────────────────────────────────

@router.post("/peer-insights", response_model=PeerInsightsResponse)
async def peer_insights(
    body: PeerInsightsRequest,
    request: Request,
    _: AuthDep,
    artifacts: ArtifactsDep,
    executor: ExecutorDep,
):
    """
    Per user-entitlement pair peer adoption metrics.
    Also CPU-bound so runs in the thread pool behind the semaphore.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    semaphore = get_semaphore()

    async with semaphore:
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                executor,
                _sync_peer_insights,
                body.user_id,
                body.entitlement_id,
            )
        except Exception:
            logger.exception(
                "peer_insights failed for user_id=%d entitlement_id=%s (request_id=%s)",
                body.user_id, body.entitlement_id, request_id,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Peer insights calculation failed"
            )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No peer data found for user {body.user_id}."
        )

    def _pg(d: dict) -> PeerGroup:
        return PeerGroup(adoption_rate=float(d["adoption_rate"]), total=int(d["total"]))

    return PeerInsightsResponse(
        user_id=body.user_id,
        entitlement_id=body.entitlement_id,
        close_peers=_pg(result["close_peers"]),
        direct_team=_pg(result["direct_team"]),
        role_peers=_pg(result["role_peers"]),
        dept_peers=_pg(result["dept_peers"]),
        request_id=request_id,
    )


def _sync_peer_insights(user_id: int, entitlement_id: str) -> dict | None:
    from ml_pipeline.prediction_core import calculate_peer_insights
    return calculate_peer_insights(user_id, entitlement_id)


# ── POST /predict/shap ────────────────────────────────────────────────────────

@router.post("/shap", response_model=ShapJobAccepted, status_code=202)
async def submit_shap(
    body: ShapRequest,
    request: Request,
    _: AuthDep,
    artifacts: ArtifactsDep,
    executor: ExecutorDep,
):
    """
    Submit a SHAP explanation job. Returns a job_id immediately (HTTP 202).
    SHAP generation runs as a background task in the thread pool.
    Poll GET /predict/shap/{job_id} for status.

    SHAP is intentionally NOT behind the concurrency semaphore: it is a
    background task and should not occupy a slot that blocks real-time /predict
    requests.
    """
    job_id = str(uuid.uuid4())
    await _register_job(job_id)

    loop = asyncio.get_event_loop()
    asyncio.ensure_future(
        _run_shap_background(job_id, body.user_id, body.top_n, body.initial_candidates, loop, executor)
    )

    logger.info("shap job submitted: job_id=%s user_id=%d", job_id, body.user_id)
    return ShapJobAccepted(
        job_id=job_id,
        status="pending",
        message=f"SHAP job accepted. Poll GET /predict/shap/{job_id} for status.",
    )


async def _run_shap_background(
    job_id: str,
    user_id: int,
    top_n: int,
    initial_candidates: int,
    loop: asyncio.AbstractEventLoop,
    executor: ThreadPoolExecutor,
) -> None:
    """Coroutine that drives SHAP generation in a thread and updates the job store."""
    await _update_job(job_id, status="running")
    try:
        output_path = await loop.run_in_executor(
            executor,
            _sync_shap,
            user_id,
            top_n,
            initial_candidates,
        )
        if output_path:
            await _update_job(job_id, status="done", output_path=output_path)
        else:
            await _update_job(job_id, status="failed", error="SHAP returned no output (SHAP may not be installed or no recommendations found)")
    except Exception as exc:
        logger.exception("SHAP background job %s failed", job_id)
        await _update_job(job_id, status="failed", error=str(exc))


def _sync_shap(user_id: int, top_n: int, initial_candidates: int) -> str | None:
    """
    Synchronous SHAP execution in a thread.
    Runs the prediction pipeline first to obtain results, then generates the
    SHAP force plot HTML. Returns the saved file path.
    """
    import os
    from ml_pipeline.prediction_core import generate_shap_explanation, run_prediction_pipeline

    results = run_prediction_pipeline(user_id, top_n=top_n, initial_candidates=initial_candidates)
    if results is None:
        return None

    output_dir = os.environ.get("SHAP_OUTPUT_DIR", "shap_outputs")
    os.makedirs(output_dir, exist_ok=True)
    ent_id = results["predictions"].iloc[0]["OriginalEntitlementId"]
    output_path = os.path.join(output_dir, f"shap_user_{user_id}_ent_{ent_id}_{uuid.uuid4().hex[:8]}.html")

    return generate_shap_explanation(results, output_path=output_path)


# ── GET /predict/shap/{job_id} ────────────────────────────────────────────────

@router.get("/shap/{job_id}", response_model=ShapJobStatus)
async def get_shap_status(job_id: str, _: AuthDep):
    """Poll the status of a submitted SHAP job."""
    async with _job_store_lock:
        job = _job_store.get(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found."
        )

    # TTL-on-access cleanup: evict completed/failed jobs older than JOB_TTL_SECONDS
    age_seconds = (datetime.now(timezone.utc) - job["created_at"]).total_seconds()
    if age_seconds > JOB_TTL_SECONDS and job["status"] in ("done", "failed"):
        async with _job_store_lock:
            _job_store.pop(job_id, None)
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail=f"Job '{job_id}' has expired and been cleaned up."
        )

    return ShapJobStatus(**job)
