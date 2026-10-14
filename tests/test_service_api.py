"""Protocol-level tests that never load a real P2PFormer checkpoint."""

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from service.app import create_app
from service.config import ServiceSettings
from service.predictor import PredictedPolygon, restore_polygon_to_original


class MockPredictor:
    def __init__(self, predictions=None, error=None):
        self.predictions = predictions or []
        self.error = error

    def infer(self, image_bgr, score_threshold):
        if self.error:
            raise self.error
        return [
            prediction
            for prediction in self.predictions
            if prediction.score >= score_threshold
        ]


@pytest.fixture
def settings(tmp_path):
    config = tmp_path / "model.py"
    checkpoint = tmp_path / "weights-v1.pth"
    config.write_text("model = {}", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint-placeholder")
    return ServiceSettings(
        api_key="test-secret",
        allow_empty_api_key=False,
        config_path=config,
        checkpoint_path=checkpoint,
        device="cpu",
        host="127.0.0.1",
        port=8000,
        max_upload_bytes=1024 * 1024,
        model_version_override="test-weights-v1",
    )


def image_bytes(image_format, size=(32, 20)):
    output = BytesIO()
    Image.new("RGB", size, color=(12, 34, 56)).save(
        output, format=image_format
    )
    return output.getvalue()


def infer_request(client, data, filename="sample.png", threshold="0.5",
                  request_id="request-001", api_key="test-secret"):
    headers = {"X-API-Key": api_key}
    if request_id is not None:
        headers["X-Request-ID"] = request_id
    return client.post(
        "/v1/auto/infer",
        headers=headers,
        files={"image": (filename, data, "application/octet-stream")},
        data={"score_threshold": threshold},
    )


def polygon(points, score=0.9):
    return PredictedPolygon(points=points, score=score, score_cls=score)


def test_health_has_required_fields_and_types(settings):
    with TestClient(create_app(settings, MockPredictor())) as client:
        response = client.get("/health", headers={"X-Request-ID": "health-1"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    for field in ("busy", "config_exists", "checkpoint_exists",
                  "cuda_available"):
        assert type(body[field]) is bool
    assert body["model_name"] == "P2PFormer"
    assert body["model_version"] == "test-weights-v1"
    assert body["current_jobs"] == 0
    assert body["queue_length"] == 0
    assert body["max_concurrency"] == 1


def test_health_is_diagnostic_when_configuration_is_missing(settings):
    missing = ServiceSettings(
        **{
            **settings.__dict__,
            "config_path": settings.config_path.parent / "missing.py",
        }
    )
    with TestClient(create_app(missing, MockPredictor())) as client:
        body = client.get("/health").json()
    assert body["status"] == "config_missing"
    assert body["config_exists"] is False


def test_capabilities_declares_sync_only(settings):
    with TestClient(create_app(settings, MockPredictor())) as client:
        response = client.get(
            "/v1/capabilities", headers={"X-API-Key": "test-secret"}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["protocol_version"] == "1.0"
    assert body["capabilities"] == {
        "async_jobs": False,
        "sync_inference": True,
        "job_cancellation": False,
    }


@pytest.mark.parametrize(
    "headers,status",
    [({}, 401), ({"X-API-Key": "wrong"}, 401),
     ({"X-API-Key": "test-secret"}, 200)],
)
def test_api_key_required_for_capabilities(settings, headers, status):
    with TestClient(create_app(settings, MockPredictor())) as client:
        response = client.get("/v1/capabilities", headers=headers)
    assert response.status_code == status
    if status == 401:
        assert response.json()["error"]["code"] == "unauthorized"


def test_explicit_empty_api_key_mode(settings):
    empty_key = ServiceSettings(
        **{**settings.__dict__, "api_key": None, "allow_empty_api_key": True}
    )
    with TestClient(create_app(empty_key, MockPredictor())) as client:
        response = client.get("/v1/capabilities")
    assert response.status_code == 200


@pytest.mark.parametrize(
    "image_format,filename",
    [("PNG", "image.png"), ("JPEG", "image.jpg"), ("TIFF", "image.tiff")],
)
def test_supported_upload_formats_preserve_original_size(
    settings, image_format, filename
):
    with TestClient(create_app(settings, MockPredictor())) as client:
        response = infer_request(
            client, image_bytes(image_format, (37, 23)), filename=filename
        )
    assert response.status_code == 200
    assert response.json()["image"]["width"] == 37
    assert response.json()["image"]["height"] == 23
    assert response.json()["image"]["file_name"] == filename


@pytest.mark.parametrize("threshold", ["0", "1", "0.0", "1.0"])
def test_score_threshold_boundaries(settings, threshold):
    with TestClient(create_app(settings, MockPredictor())) as client:
        response = infer_request(
            client, image_bytes("PNG"), threshold=threshold
        )
    assert response.status_code == 200
    assert response.json()["summary"]["score_threshold"] == float(threshold)


@pytest.mark.parametrize("threshold", ["-0.01", "1.01", "nan", "abc"])
def test_invalid_score_threshold(settings, threshold):
    with TestClient(create_app(settings, MockPredictor())) as client:
        response = infer_request(
            client, image_bytes("PNG"), threshold=threshold
        )
    assert response.status_code == 400
    assert "error" in response.json()


@pytest.mark.parametrize("request_id", [None, "", "   "])
def test_request_id_is_required(settings, request_id):
    with TestClient(create_app(settings, MockPredictor())) as client:
        response = infer_request(
            client, image_bytes("PNG"), request_id=request_id
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] in {
        "missing_request_id", "invalid_request"
    }


def test_request_id_is_returned_exactly(settings):
    request_id = "client-ID_case.SENSITIVE_42"
    with TestClient(create_app(settings, MockPredictor())) as client:
        response = infer_request(
            client, image_bytes("PNG"), request_id=request_id
        )
    assert response.status_code == 200
    assert response.json()["request_id"] == request_id


def test_empty_prediction_is_success(settings):
    with TestClient(create_app(settings, MockPredictor())) as client:
        response = infer_request(client, image_bytes("PNG"))
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["polygons"] == []
    assert response.json()["summary"]["polygon_count"] == 0


def test_one_and_multiple_polygons_are_validated_and_thresholded(settings):
    predictions = [
        polygon([[1, 2], [11, 2], [11, 12], [1, 12]], 0.95),
        polygon([[20, 3], [30, 3], [25, 15]], 0.75),
        polygon([[2, 2], [3, 2], [2, 3]], 0.2),
    ]
    with TestClient(create_app(settings, MockPredictor(predictions))) as client:
        response = infer_request(
            client, image_bytes("PNG", (32, 20)), threshold="0.5"
        )
    assert response.status_code == 200
    body = response.json()
    assert body["coordinate_space"] == "original_image_pixel"
    assert body["summary"]["polygon_count"] == len(body["polygons"]) == 2
    assert body["polygons"][0]["id"] == "building_0001"
    assert body["polygons"][0]["bbox_xywh"] == [1.0, 2.0, 10.0, 10.0]
    for result in body["polygons"]:
        assert 0 <= result["score"] <= 1
        assert 0 <= result["score_cls"] <= 1
        assert len(set(map(tuple, result["points"]))) >= 3
        for x, y in result["points"]:
            assert 0 <= x <= 32
            assert 0 <= y <= 20


def test_predictor_exception_uses_standard_error_and_no_traceback(settings):
    predictor = MockPredictor(error=RuntimeError("secret /internal/model.pth"))
    with TestClient(create_app(settings, predictor)) as client:
        response = infer_request(client, image_bytes("PNG"))
    assert response.status_code == 500
    body = response.json()
    assert body == {
        "error": {
            "code": "inference_failed",
            "message": "Inference failed",
            "details": {},
        }
    }
    assert "internal" not in response.text


def test_unsupported_and_invalid_images(settings):
    with TestClient(create_app(settings, MockPredictor())) as client:
        unsupported = infer_request(client, b"data", filename="image.bmp")
        invalid = infer_request(client, b"not a png", filename="image.png")
    assert unsupported.status_code == 415
    assert invalid.status_code == 400


def test_upload_size_limit(settings):
    tiny_limit = ServiceSettings(
        **{**settings.__dict__, "max_upload_bytes": 10}
    )
    with TestClient(create_app(tiny_limit, MockPredictor())) as client:
        response = infer_request(client, image_bytes("PNG"))
    assert response.status_code == 413


def test_busy_service_rejects_without_queueing(settings):
    app = create_app(settings, MockPredictor())
    with TestClient(app) as client:
        app.state.runtime.current_jobs = 1
        response = infer_request(client, image_bytes("PNG"))
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_busy"


def test_unknown_endpoint_uses_standard_error(settings):
    with TestClient(create_app(settings, MockPredictor())) as client:
        response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_restore_polygon_uses_independent_xy_scale_factors():
    metadata = {
        "img_shape": (200, 400, 3),
        "ori_shape": (100, 100, 3),
        "scale_factor": [4.0, 2.0, 4.0, 2.0],
        "flip": False,
    }
    restored = restore_polygon_to_original(
        [[40, 20], [360, 20], [360, 180], [40, 180]],
        metadata,
        original_width=100,
        original_height=100,
    )
    assert restored == [[10.0, 10.0], [90.0, 10.0],
                        [90.0, 90.0], [10.0, 90.0]]


def test_restored_polygon_clips_only_small_rounding_error():
    metadata = {
        "img_shape": (50, 50, 3),
        "scale_factor": [0.5, 0.5, 0.5, 0.5],
        "flip": False,
    }
    restored = restore_polygon_to_original(
        [[-0.1, 0], [50.1, 0], [50, 50]], metadata, 100, 100
    )
    assert restored == [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]]


def test_restore_polygon_accepts_model_roi_expansion_at_image_edge():
    metadata = {
        "img_shape": (100, 100, 3),
        "scale_factor": [1.0, 1.0, 1.0, 1.0],
        "flip": False,
    }
    restored = restore_polygon_to_original(
        [[-2.5, 10], [40, 10], [40, 50], [-2.5, 50]],
        metadata,
        100,
        100,
        bbox_xyxy=[0, 10, 40, 50],
        roi_expand_scale=1.1,
    )
    assert restored == [[0.0, 10.0], [40.0, 10.0],
                        [40.0, 50.0], [0.0, 50.0]]


def test_restore_polygon_rejects_large_excursion_without_raising():
    metadata = {
        "img_shape": (100, 100, 3),
        "scale_factor": [1.0, 1.0, 1.0, 1.0],
        "flip": False,
    }
    restored = restore_polygon_to_original(
        [[-30, 10], [40, 10], [40, 50], [-30, 50]],
        metadata,
        100,
        100,
        bbox_xyxy=[0, 10, 40, 50],
        roi_expand_scale=1.1,
    )
    assert restored == []

