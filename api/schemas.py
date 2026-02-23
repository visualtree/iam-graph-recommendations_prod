from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ── Request Models ───────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    user_id: int = Field(..., ge=1, description="User ID to generate recommendations for")
    endpoint_id: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional target endpoint/system ID to scope recommendations",
    )
    top_n: int = Field(default=5, ge=1, le=50, description="Number of final recommendations")
    initial_candidates: int = Field(
        default=100, ge=10, le=500,
        description="Number of Stage-1 candidates to pass to the reranker"
    )


class PeerInsightsRequest(BaseModel):
    user_id: int = Field(..., ge=1)
    entitlement_id: str = Field(
        ...,
        description="Composite entitlement ID in format '{endpoint}_{entitlement}'"
    )

    @field_validator("entitlement_id")
    @classmethod
    def validate_entitlement_format(cls, v: str) -> str:
        if len(v.split("_")) < 2:
            raise ValueError(
                "entitlement_id must be a composite ID in format '{endpoint}_{entitlement}'"
            )
        return v


class ShapRequest(BaseModel):
    user_id: int = Field(..., ge=1)
    top_n: int = Field(default=5, ge=1, le=50)
    initial_candidates: int = Field(default=100, ge=10, le=500)


# ── Response Models ──────────────────────────────────────────────────────────

class Recommendation(BaseModel):
    entitlement_id: str
    original_entitlement_id: str
    entitlement_name: Optional[str] = None
    entitlement_description: Optional[str] = None
    endpoint_id: Optional[int] = None
    endpoint_name: Optional[str] = None
    candidate_score: float
    final_score: float


class PredictResponse(BaseModel):
    user_id: int
    total_candidates: int
    stage1_count: int
    recommendations: List[Recommendation]
    request_id: str
    duration_ms: float


class PeerGroup(BaseModel):
    adoption_rate: float
    total: int


class PeerInsightsResponse(BaseModel):
    user_id: int
    entitlement_id: str
    close_peers: PeerGroup
    direct_team: PeerGroup
    role_peers: PeerGroup
    dept_peers: PeerGroup
    request_id: str


class ShapJobAccepted(BaseModel):
    job_id: str
    status: Literal["pending"]
    message: str


class ShapJobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "running", "done", "failed"]
    output_path: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class HealthArtifact(BaseModel):
    name: str
    loaded: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "starting"]
    artifacts_loaded: bool
    artifact_dir: str
    details: List[HealthArtifact]
    uptime_seconds: float
