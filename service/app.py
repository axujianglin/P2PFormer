"""FastAPI application implementing the SHORTER SAMPoly protocol."""

import asyncio
import hashlib
import logging
import threading
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any, Dict, Optional

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import ServiceSettings, cuda_is_available
from .predictor import P2PFormerPredictor, PredictedPolygon
from .schemas import (
    CapabilitiesResponse,
    CapabilityFlags,
    HealthResponse,
    ImageInfo,
    InferenceResponse,
    InferenceSummary,
    PolygonResponse,
)

LOGGER = logging.getLogger("p2pformer.service")
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
SUPPORTED_PIL_FORMATS = {"PNG", "JPEG", "TIFF"}


class ServiceError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


class RuntimeState:
    def __init__(self) -> None:
        self.predictor: Optional[Any] = None
        self.model_ready = False
        self.model_error = False
        self.current_jobs = 0
        self._jobs_guard = threading.Lock()

    def try_start_job(self) -> bool:
        with self._jobs_guard:
            if self.current_jobs >= 1:
                return False
            self.current_jobs += 1
            return True

    def finish_job(self) -> None:
        with self._jobs_guard:
            self.current_jobs = max(0, self.current_jobs - 1)


def _status(settings: ServiceSettings, state: RuntimeState,
            cuda_available: bool) -> str:
    if not settings.config_exists:
        return "config_missing"
    if not settings.checkpoint_exists:
        return "checkpoint_missing"
    if settings.device.lower().startswith("cuda") and not cuda_available:
        return "cuda_unavailable"
    if not settings.api_key and not settings.allow_empty_api_key:
        return "api_key_missing"
    if state.model_error:
        return "model_error"
    if not state.model_ready:
        return "starting"
    return "ready"


def _safe_filename(filename: Optional[str]) -> str:
    if not filename:
        raise ServiceError(400, "invalid_request", "Image filename is required")
    # The value is echoed as protocol data, never used as a filesystem path.
    return filename


def _extension(filename: str) -> str:
    dot = filename.rfind(".")
    return filename[dot:].lower() if dot >= 0 else ""


def _decode_image(data: bytes, filename: str) -> np.ndarray:
    if _extension(filename) not in SUPPORTED_EXTENSIONS:
        raise ServiceError(
            415,
            "unsupported_image_format",
            "Supported image formats are PNG, JPG, JPEG, TIF and TIFF",
        )
    try:
        with Image.open(BytesIO(data)) as source:
            if source.format not in SUPPORTED_PIL_FORMATS:
                raise ServiceError(
                    415,
                    "unsupported_image_format",
                    "The uploaded file is not a supported image",
                )
            source.load()
            rgb = source.convert("RGB")
            # MMDetection's ndarray path expects OpenCV-style BGR input.
            return np.asarray(rgb, dtype=np.uint8)[:, :, ::-1].copy()
    except ServiceError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ServiceError(
            400, "invalid_image", "The uploaded image could not be decoded"
        ) from exc


def _build_polygons(
    predictions: Any, width: int, height: int, model_name: str
) -> Any:
    response = []
    tolerance = 1e-6
    for index, prediction in enumerate(predictions, start=1):
        if not isinstance(prediction, PredictedPolygon):
            raise RuntimeError("Predictor returned an unsupported result")
        score = float(prediction.score)
        score_cls = float(prediction.score_cls)
        if not 0.0 <= score <= 1.0 or not 0.0 <= score_cls <= 1.0:
            raise RuntimeError("Predictor returned an invalid confidence score")
        points = [[float(x), float(y)] for x, y in prediction.points]
        if len(points) < 3 or len({tuple(point) for point in points}) < 3:
            raise RuntimeError("Predictor returned an invalid polygon")
        if any(
            not np.isfinite(point).all()
            or point[0] < -tolerance
            or point[0] > width + tolerance
            or point[1] < -tolerance
            or point[1] > height + tolerance
            for point in map(np.asarray, points)
        ):
            raise RuntimeError("Predictor returned an out-of-bounds polygon")
        for point in points:
            point[0] = min(float(width), max(0.0, point[0]))
            point[1] = min(float(height), max(0.0, point[1]))
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        response.append(
            PolygonResponse(
                id=f"building_{index:04d}",
                score=score,
                score_cls=score_cls,
                bbox_xywh=[
                    min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
                ],
                points=points,
                metadata={"model": model_name},
            )
        )
    return response


def create_app(
    settings: Optional[ServiceSettings] = None,
    predictor: Optional[Any] = None,
) -> FastAPI:
    settings = settings or ServiceSettings.from_env()
    state = RuntimeState()
    injected_predictor = predictor

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        can_load = (
            settings.config_exists
            and settings.checkpoint_exists
            and (
                not settings.device.lower().startswith("cuda")
                or cuda_is_available()
            )
        )
        if injected_predictor is not None:
            state.predictor = injected_predictor
            state.model_ready = True
        elif can_load:
            try:
                state.predictor = P2PFormerPredictor(settings)
                await run_in_threadpool(state.predictor.load)
                state.model_ready = True
            except Exception:
                state.model_error = True
                LOGGER.exception("P2PFormer model initialization failed")
        yield

    app = FastAPI(
        title="P2PFormer SHORTER inference service",
        version=settings.service_version,
        lifespan=lifespan,
    )
    app.state.service_settings = settings
    app.state.runtime = state

    @app.exception_handler(ServiceError)
    async def service_error_handler(_: Request, exc: ServiceError):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError):
        fields = [
            ".".join(str(part) for part in error.get("loc", []))
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": "The request format is invalid",
                    "details": {"fields": fields},
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(_: Request, exc: StarletteHTTPException):
        code = "not_found" if exc.status_code == 404 else "http_error"
        message = "The requested endpoint was not found" \
            if exc.status_code == 404 else "The request could not be completed"
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {"code": code, "message": message, "details": {}}
            },
        )

    async def authorize(
        x_api_key: Optional[str] = Header(None, alias="X-API-Key")
    ) -> None:
        configured = settings.api_key or ""
        provided = x_api_key or ""
        if not configured and settings.allow_empty_api_key:
            return
        if not configured or not provided:
            raise ServiceError(401, "unauthorized", "A valid API key is required")
        import hmac

        if not hmac.compare_digest(provided, configured):
            raise ServiceError(401, "unauthorized", "A valid API key is required")

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        cuda_available = cuda_is_available()
        return HealthResponse(
            status=_status(settings, state, cuda_available),
            busy=state.current_jobs > 0,
            config_exists=settings.config_exists,
            checkpoint_exists=settings.checkpoint_exists,
            cuda_available=cuda_available,
            model_name=settings.model_name,
            model_version=settings.model_version,
            device=settings.display_device,
            current_jobs=state.current_jobs,
            queue_length=0,
            max_concurrency=settings.max_concurrency,
        )

    @app.get(
        "/v1/capabilities",
        response_model=CapabilitiesResponse,
        dependencies=[Depends(authorize)],
    )
    async def capabilities() -> CapabilitiesResponse:
        return CapabilitiesResponse(
            protocol_version=settings.protocol_version,
            service_version=settings.service_version,
            model_version=settings.model_version,
            model_name=settings.model_name,
            device=settings.display_device,
            cuda_available=cuda_is_available(),
            config_exists=settings.config_exists,
            checkpoint_exists=settings.checkpoint_exists,
            max_concurrency=settings.max_concurrency,
            capabilities=CapabilityFlags(
                async_jobs=False,
                sync_inference=True,
                job_cancellation=False,
            ),
        )

    @app.post(
        "/v1/auto/infer",
        response_model=InferenceResponse,
        dependencies=[Depends(authorize)],
    )
    async def infer(
        image: UploadFile = File(...),
        score_threshold: float = Form(...),
        x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    ) -> InferenceResponse:
        if x_request_id is None or not x_request_id.strip():
            raise ServiceError(
                400, "missing_request_id", "X-Request-ID is required"
            )
        if not 0.0 <= score_threshold <= 1.0:
            raise ServiceError(
                400,
                "invalid_score_threshold",
                "score_threshold must be between 0.0 and 1.0",
            )
        if not state.model_ready or state.predictor is None:
            raise ServiceError(
                503, "model_unavailable", "The inference model is unavailable"
            )
        filename = _safe_filename(image.filename)
        data = await image.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise ServiceError(
                413, "image_too_large", "The uploaded image is too large"
            )
        if not data:
            raise ServiceError(400, "invalid_image", "The uploaded image is empty")
        image_bgr = _decode_image(data, filename)
        height, width = image_bgr.shape[:2]

        if not state.try_start_job():
            raise ServiceError(
                503, "service_busy", "The GPU is busy; retry this request later"
            )
        try:
            predictions = await run_in_threadpool(
                state.predictor.infer, image_bgr, score_threshold
            )
            polygons = _build_polygons(
                predictions, width, height, settings.model_name
            )
        except ServiceError:
            raise
        except Exception:
            LOGGER.exception("Inference failed for request %r", x_request_id)
            raise ServiceError(
                500, "inference_failed", "Inference failed"
            )
        finally:
            state.finish_job()

        return InferenceResponse(
            protocol_version=settings.protocol_version,
            request_id=x_request_id,
            image=ImageInfo(
                file_name=filename,
                image_id=hashlib.sha256(data).hexdigest(),
                width=width,
                height=height,
            ),
            coordinate_space="original_image_pixel",
            polygons=polygons,
            summary=InferenceSummary(
                polygon_count=len(polygons),
                score_threshold=score_threshold,
            ),
        )

    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = app.state.service_settings
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()

