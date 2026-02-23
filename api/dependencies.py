import os
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated

from fastapi import Depends, HTTPException, status

# ── Thread Pool ──────────────────────────────────────────────────────────────
# With n_jobs=1 per model call, each prediction uses ~1 CPU core at peak.
# Setting max_workers = CPU count gives full utilisation without oversubscription.
# Tune via IAM_API_WORKERS env var in production.
_MAX_WORKERS = int(os.environ.get("IAM_API_WORKERS", "4"))

_executor = ThreadPoolExecutor(
    max_workers=_MAX_WORKERS,
    thread_name_prefix="iam-predict",
)


def get_executor() -> ThreadPoolExecutor:
    return _executor


# ── Concurrency Semaphore ────────────────────────────────────────────────────
# Caps simultaneous predict_proba calls to the pool size, preventing
# N concurrent requests from each allocating a large feature DataFrame
# before any thread becomes free.
#
# Note: asyncio.Semaphore must be used from async context (route handlers).
# It is created lazily at first use to ensure it binds to the running event loop.
_semaphore = None


def get_semaphore():
    import asyncio
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(_MAX_WORKERS)
    return _semaphore


# ── Artifact Dependency ──────────────────────────────────────────────────────
def get_artifacts() -> dict:
    """
    FastAPI dependency that returns the pre-loaded artifact dict.
    The lifespan handler pre-loads on startup; this is always a fast dict access.
    Returns HTTP 503 if artifacts failed to load at startup.
    """
    from ml_pipeline.prediction_core import PredictionArtifacts
    artifacts = PredictionArtifacts._artifacts
    if artifacts is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction artifacts are not loaded. Service is starting or startup failed."
        )
    return artifacts


# ── Annotated shorthands for route injection ─────────────────────────────────
ArtifactsDep = Annotated[dict, Depends(get_artifacts)]
ExecutorDep   = Annotated[ThreadPoolExecutor, Depends(get_executor)]
