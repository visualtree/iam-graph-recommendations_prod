from __future__ import annotations

import logging
import logging.config
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager

# Force UTF-8 for the whole process (works even on Windows)
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

# Force UTF-8 on stdout/stderr before any logging is configured.
# Windows defaults to cp1252 which cannot encode emoji or most non-Latin characters.
# This prevents UnicodeEncodeError from propagating as HTTP 500 responses.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

# ── Structured Logging ────────────────────────────────────────────────────────
_LOG_FORMAT = os.environ.get("LOG_FORMAT", "plain")
_LOG_LEVEL  = os.environ.get("LOG_LEVEL", "INFO")

_FORMATTER = (
    "json"
    if _LOG_FORMAT == "json"
    else "plain"
)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "plain": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": _FORMATTER,
        }
    },
    "root": {"level": _LOG_LEVEL, "handlers": ["console"]},
}

try:
    logging.config.dictConfig(LOGGING_CONFIG)
except Exception:
    # Fall back to plain logging if python-json-logger is not installed
    LOGGING_CONFIG["handlers"]["console"]["formatter"] = "plain"
    logging.config.dictConfig(LOGGING_CONFIG)

logger = logging.getLogger(__name__)


def _safe_text(value: object) -> str:
    """Convert text to an ASCII-safe representation for logs and error payloads."""
    if value is None:
        return ""
    return str(value).encode("ascii", errors="backslashreplace").decode("ascii")


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _validate_runtime_config() -> None:
    """
    Validate runtime env configuration.

    Behavior:
    - Logs warnings/errors in all modes.
    - Raises RuntimeError only when IAM_STRICT_ENV_VALIDATION=true.
    """
    strict = _is_truthy(os.environ.get("IAM_STRICT_ENV_VALIDATION"))
    require_api_token = _is_truthy(os.environ.get("IAM_REQUIRE_API_TOKEN"))

    errors: list[str] = []
    warnings: list[str] = []

    workers_raw = os.environ.get("IAM_API_WORKERS", "4")
    try:
        workers = int(workers_raw)
        if workers <= 0:
            errors.append("IAM_API_WORKERS must be a positive integer.")
    except ValueError:
        errors.append("IAM_API_WORKERS must be an integer.")

    api_port_raw = os.environ.get("API_PORT")
    if api_port_raw:
        try:
            api_port = int(api_port_raw)
            if api_port < 1 or api_port > 65535:
                errors.append("API_PORT must be between 1 and 65535.")
        except ValueError:
            errors.append("API_PORT must be an integer.")

    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    valid_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
    if log_level not in valid_levels:
        errors.append(f"LOG_LEVEL must be one of: {', '.join(sorted(valid_levels))}.")

    artifact_dir = os.environ.get("ARTIFACT_DIR", "artifacts")
    if not os.path.isdir(artifact_dir):
        warnings.append(f"ARTIFACT_DIR directory does not exist yet: {artifact_dir}")

    api_token = os.environ.get("IAM_API_TOKEN", "").strip()
    if require_api_token and not api_token:
        errors.append("IAM_REQUIRE_API_TOKEN=true but IAM_API_TOKEN is not set.")
    if not require_api_token and not api_token:
        warnings.append("IAM_API_TOKEN is not set. API auth guard is disabled.")

    for msg in warnings:
        logger.warning("CONFIG WARNING: %s", msg)
    for msg in errors:
        logger.error("CONFIG ERROR: %s", msg)

    if strict and errors:
        raise RuntimeError(
            "Runtime configuration validation failed. "
            "Set IAM_STRICT_ENV_VALIDATION=false to continue in non-strict mode."
        )


# Recorded at module import time; health endpoint uses this for uptime reporting
_startup_time = time.monotonic()


# ── Lifespan: Pre-load Artifacts ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pre-loads the 18 MB artifact singleton before the first request arrives.
    Uses a dedicated 1-worker executor so startup I/O does not consume a slot
    from the main serving thread pool.

    On failure: logs the error but does NOT crash the process. /health will
    report status="starting" and prediction endpoints will return HTTP 503
    until the issue is resolved and the service restarted.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from ml_pipeline.prediction_core import PredictionArtifacts

    _validate_runtime_config()
    logger.info("API startup: pre-loading prediction artifacts...")
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="iam-startup") as startup_pool:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(startup_pool, PredictionArtifacts.get_artifacts)
        logger.info("Artifacts pre-loaded. Service is ready.")
    except Exception:
        logger.exception("STARTUP FAILURE: could not pre-load artifacts. Predictions will return 503.")

    yield  # Service is live

    logger.info("API shutdown: cleaning up.")


# ── FastAPI Application ───────────────────────────────────────────────────────
app = FastAPI(
    title="IAM Entitlement Recommendation API",
    description=(
        "Production REST API for the IAM graph-based ML recommendation pipeline. "
        "Exposes two-stage (candidate + reranker) XGBoost predictions, peer adoption "
        "insights, and async SHAP explanations."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Middleware: Request ID + Timing ───────────────────────────────────────────
@app.middleware("http")
async def request_instrumentation(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()

    try:
        response: Response = await call_next(request)
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        safe_error = _safe_text(exc)
        logger.error(
            "Unhandled exception in %s %s | request_id=%s duration_ms=%.2f error=%s",
            request.method, request.url.path, request_id, duration_ms, safe_error,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id},
        )

    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Duration-Ms"] = f"{duration_ms:.2f}"

    logger.info(
        "%s %s | status=%d request_id=%s duration_ms=%.2f",
        request.method, request.url.path, response.status_code, request_id, duration_ms,
    )
    return response


# ── Exception Handlers ────────────────────────────────────────────────────────
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    safe_detail = _safe_text(exc)
    return JSONResponse(
        status_code=422,
        content={
            "detail": safe_detail,
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    safe_detail = _safe_text(exc)
    logger.error(
        "RuntimeError | request_id=%s error=%s",
        getattr(request.state, "request_id", "?"), safe_detail,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": safe_detail,
            "request_id": getattr(request.state, "request_id", None),
        },
    )


# ── Routers ───────────────────────────────────────────────────────────────────
from api.routes import health, predictions  # noqa: E402  (import after app creation)

app.include_router(health.router)
app.include_router(predictions.router, prefix="/predict", tags=["Predictions"])
