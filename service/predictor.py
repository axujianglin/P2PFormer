"""Adapter around the repository's existing single-image inference path."""

import logging
import math
from dataclasses import dataclass
from types import MethodType
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .config import ServiceSettings

LOGGER = logging.getLogger("p2pformer.service.predictor")


@dataclass
class PredictedPolygon:
    points: List[List[float]]
    score: float
    score_cls: float


def restore_polygon_to_original(
    points: Sequence[Sequence[float]],
    image_meta: Dict[str, Any],
    original_width: int,
    original_height: int,
    bbox_xyxy: Sequence[float] = (),
    roi_expand_scale: float = 1.0,
) -> List[List[float]]:
    """Undo MMDetection flip and resize for a polygon.

    P2PFormer creates line/polygon coordinates in ``img_shape`` (the resized
    image before right/bottom padding). MMDetection's ``scale_factor`` is
    ``[w_scale, h_scale, w_scale, h_scale]``. No crop is present in the
    repository test pipelines.
    """

    array = np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()
    if not np.isfinite(array).all():
        return []

    image_height, image_width = image_meta["img_shape"][:2]
    if image_meta.get("flip", False):
        direction = image_meta.get("flip_direction", "horizontal")
        if direction in {"horizontal", "diagonal"}:
            array[:, 0] = image_width - array[:, 0]
        if direction in {"vertical", "diagonal"}:
            array[:, 1] = image_height - array[:, 1]

    scale_factor = np.asarray(image_meta["scale_factor"], dtype=np.float64)
    if scale_factor.size == 1:
        width_scale = height_scale = float(scale_factor[0])
    else:
        width_scale, height_scale = map(float, scale_factor[:2])
    if width_scale <= 0.0 or height_scale <= 0.0:
        raise ValueError("Invalid image scale factor produced by preprocessing")

    array[:, 0] /= width_scale
    array[:, 1] /= height_scale

    # Line points are sigmoid-bounded inside an RoI enlarged by
    # ``line_head.expand_scale`` (1.1 in the supplied configs). A detection
    # touching an image edge can therefore legitimately produce a point just
    # outside the image. Permit only that architecture-defined margin plus a
    # small resize-rounding allowance; reject larger excursions so clipping
    # cannot conceal a bad coordinate transform.
    tolerance_x = 1.0
    tolerance_y = 1.0
    if len(bbox_xyxy) >= 4:
        bbox = np.asarray(bbox_xyxy[:4], dtype=np.float64)
        if np.isfinite(bbox).all():
            expansion = max(0.0, float(roi_expand_scale) - 1.0) / 2.0
            tolerance_x += max(0.0, float(bbox[2] - bbox[0])) * expansion
            tolerance_y += max(0.0, float(bbox[3] - bbox[1])) * expansion
    if (
        (array[:, 0] < -tolerance_x).any()
        or (array[:, 0] > original_width + tolerance_x).any()
        or (array[:, 1] < -tolerance_y).any()
        or (array[:, 1] > original_height + tolerance_y).any()
    ):
        return []

    array[:, 0] = np.clip(array[:, 0], 0.0, float(original_width))
    array[:, 1] = np.clip(array[:, 1], 0.0, float(original_height))
    restored = array.tolist()
    if len(restored) > 1 and np.allclose(restored[0], restored[-1]):
        restored.pop()
    if len({(point[0], point[1]) for point in restored}) < 3:
        return []
    return [[float(x), float(y)] for x, y in restored]


class P2PFormerPredictor:
    """Load P2PFormer once and expose vector polygon inference."""

    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        self.model = None

    def load(self) -> None:
        """Load the existing MMDetection model and checkpoint once."""

        # Imports stay lazy so health/config tests do not require CUDA/MMCV.
        from mmdet.apis import init_detector

        self.model = init_detector(
            str(self.settings.config_path),
            str(self.settings.checkpoint_path),
            device=self.settings.device,
        )

    def infer(
        self, image_bgr: np.ndarray, score_threshold: float
    ) -> List[PredictedPolygon]:
        if self.model is None:
            raise RuntimeError("The model is not loaded")

        from mmdet.apis import inference_detector

        captured: Dict[str, Any] = {}

        def capture_converter(
            model: Any,
            contours: Any,
            labels: Any,
            bboxes: Any,
            image_metas: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            captured["contours"] = [
                [np.asarray(polygon, dtype=np.float64).copy()
                 for polygon in image_polygons]
                for image_polygons in contours
            ]
            captured["bboxes"] = [
                item.detach().cpu().numpy().copy() for item in bboxes
            ]
            captured["image_metas"] = [dict(meta) for meta in image_metas]
            # P2PFormer's public inference result rasterizes these vectors to
            # full-size masks. The HTTP protocol needs the vectors, so avoid
            # that lossy and memory-heavy conversion while preserving the
            # tuple shape consumed by P2PFormerSegmentor.simple_test.
            return tuple(
                (
                    [[] for _ in range(model.bbox_head.num_classes)],
                    image_bboxes,
                    image_labels,
                )
                for image_bboxes, image_labels in zip(bboxes, labels)
            )

        self.model.convert_contour2mask = MethodType(
            capture_converter, self.model
        )
        try:
            inference_detector(self.model, image_bgr)
        finally:
            del self.model.convert_contour2mask

        if "contours" not in captured:
            # The model bypasses polygon construction when no boxes exist.
            return []

        height, width = image_bgr.shape[:2]
        polygons = captured["contours"][0]
        bboxes = captured["bboxes"][0]
        image_meta = captured["image_metas"][0]
        roi_expand_scale = float(
            getattr(self.model.line_head, "expand_scale", 1.0)
        )
        if len(polygons) != len(bboxes):
            raise RuntimeError("P2PFormer returned misaligned polygons and boxes")

        predictions = []
        invalid_polygon_count = 0
        for polygon, bbox in zip(polygons, bboxes):
            score = float(bbox[4])
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                continue
            if score < score_threshold:
                continue
            points = restore_polygon_to_original(
                polygon,
                image_meta,
                width,
                height,
                bbox_xyxy=bbox[:4],
                roi_expand_scale=roi_expand_scale,
            )
            if len(points) < 3:
                invalid_polygon_count += 1
                continue
            predictions.append(
                PredictedPolygon(
                    points=points, score=score, score_cls=score
                )
            )
        if invalid_polygon_count:
            LOGGER.warning(
                "Skipped %d invalid or out-of-bounds polygon(s) out of %d",
                invalid_polygon_count,
                len(polygons),
            )
        return predictions

