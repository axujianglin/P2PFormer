"""Pydantic response schemas for the SHORTER protocol."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: str
    busy: bool
    config_exists: bool
    checkpoint_exists: bool
    cuda_available: bool
    model_name: str
    model_version: str
    device: str
    current_jobs: int
    queue_length: int
    max_concurrency: int


class CapabilityFlags(BaseModel):
    async_jobs: bool
    sync_inference: bool
    job_cancellation: bool


class CapabilitiesResponse(BaseModel):
    protocol_version: str
    service_version: str
    model_version: str
    model_name: str
    device: str
    cuda_available: bool
    config_exists: bool
    checkpoint_exists: bool
    max_concurrency: int
    capabilities: CapabilityFlags


class ImageInfo(BaseModel):
    file_name: str
    image_id: str
    width: int
    height: int


class PolygonResponse(BaseModel):
    id: str
    score: float
    score_cls: Optional[float] = None
    bbox_xywh: List[float]
    points: List[List[float]]
    metadata: Dict[str, Any]


class InferenceSummary(BaseModel):
    polygon_count: int
    score_threshold: float


class InferenceResponse(BaseModel):
    success: bool
    protocol_version: str
    request_id: str
    image: ImageInfo
    coordinate_space: str
    polygons: List[PolygonResponse]
    summary: InferenceSummary

