"""Pen detection using hand geometry with optional vision refinement."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

REF_HEIGHT = 720


@dataclass(frozen=True)
class PenDetectionResult:
    """Detected pen geometry in image coordinates."""

    tip: tuple[float, float]
    grip: tuple[float, float]
    axis: tuple[float, float]
    confidence: float


class PenTracker:
    """Estimates pen tip from hand pose; vision only refines when the pen is visible."""

    def __init__(
        self,
        *,
        max_value: int = 160,
        axis_distance: float = 20.0,
        min_candidates: int = 5,
        smoothing: float = 0.65,
        detection_height: int = REF_HEIGHT,
        extension_factor: float = 1.35,
    ) -> None:
        self.max_value = max_value
        self.axis_distance = axis_distance
        self.min_candidates = min_candidates
        self.smoothing = smoothing
        self.detection_height = detection_height
        self.extension_factor = extension_factor
        self._last: PenDetectionResult | None = None

    def detect(self, frame_bgr, hand_landmarks) -> PenDetectionResult | None:
        raw = self._detect_raw(frame_bgr, hand_landmarks)
        if raw is None:
            return self._last

        if self._last is None:
            self._last = raw
            return raw

        alpha = self.smoothing
        tip = (
            alpha * raw.tip[0] + (1 - alpha) * self._last.tip[0],
            alpha * raw.tip[1] + (1 - alpha) * self._last.tip[1],
        )
        grip = (
            alpha * raw.grip[0] + (1 - alpha) * self._last.grip[0],
            alpha * raw.grip[1] + (1 - alpha) * self._last.grip[1],
        )
        smoothed = PenDetectionResult(
            tip=tip,
            grip=grip,
            axis=raw.axis,
            confidence=raw.confidence,
        )
        self._last = smoothed
        return smoothed

    def reset(self) -> None:
        self._last = None

    def _hand_geometry(
        self,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
        """Derive grip, pen axis, index tip, and finger span from landmarks."""
        index_mcp = points[5]
        index_tip = points[8]
        thumb_tip = points[4]

        finger_vec = index_tip - index_mcp
        finger_span = float(np.linalg.norm(finger_vec))
        if finger_span < 1e-3:
            return None

        axis = finger_vec / finger_span
        grip = (thumb_tip + index_tip) / 2.0
        return grip, axis, index_tip, finger_span

    def _geometry_tip(
        self,
        index_tip: np.ndarray,
        axis: np.ndarray,
        finger_span: float,
    ) -> np.ndarray:
        """Extrapolate pen tip beyond the index fingertip along the pen axis."""
        extension = finger_span * self.extension_factor
        return index_tip + axis * extension

    def _refine_tip_with_vision(
        self,
        gray: np.ndarray,
        index_tip: np.ndarray,
        axis: np.ndarray,
        finger_span: float,
        geometry_tip: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        """
        Optionally nudge the tip along the pen axis using darkness in a narrow band.
        Falls back to geometry when the blurry pen shaft is not visible.
        """
        perp = np.array([-axis[1], axis[0]])
        half_width = max(4.0, 10.0 * scale)
        step = max(2.0, 4.0 * scale)
        min_extension = finger_span * 0.5
        max_extension = finger_span * 2.2
        h, w = gray.shape[:2]

        best_along = float(np.dot(geometry_tip - index_tip, axis))
        best_score = -1.0

        along = min_extension
        while along <= max_extension:
            center = index_tip + axis * along
            darkness = 0.0
            samples = 0
            for offset in np.linspace(-half_width, half_width, 5):
                point = center + perp * offset
                x, y = int(round(point[0])), int(round(point[1]))
                if 0 <= y < h and 0 <= x < w:
                    darkness += 255.0 - float(gray[y, x])
                    samples += 1
            if samples > 0:
                score = darkness / samples
                if score > best_score:
                    best_score = score
                    best_along = along
            along += step

        geo_along = float(np.dot(geometry_tip - index_tip, axis))
        # Only trust vision if it finds a clearly darker corridor near the geometry estimate.
        if best_score < 80.0 or abs(best_along - geo_along) > finger_span * 0.6:
            return geometry_tip
        return index_tip + axis * best_along

    def _detect_raw(self, frame_bgr, hand_landmarks) -> PenDetectionResult | None:
        height, width = frame_bgr.shape[:2]
        points = np.array(
            [[landmark.x * width, landmark.y * height] for landmark in hand_landmarks]
        )

        geometry = self._hand_geometry(points)
        if geometry is None:
            return None

        grip, axis, index_tip, finger_span = geometry
        geometry_tip = self._geometry_tip(index_tip, axis, finger_span)

        scale = height / REF_HEIGHT
        padding = int(40 * scale)
        x1 = int(max(0, points[:, 0].min() - padding))
        y1 = int(max(0, points[:, 1].min() - padding))
        x2 = int(min(width, points[:, 0].max() + padding))
        y2 = int(min(height, points[:, 1].max() + padding + int(220 * scale)))
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            tip = geometry_tip
            confidence = 0.6
        else:
            roi_h = roi.shape[0]
            det_scale = min(1.0, self.detection_height / roi_h)
            if det_scale < 1.0:
                det_roi = cv2.resize(
                    roi,
                    (int(roi.shape[1] * det_scale), int(roi_h * det_scale)),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                det_roi = roi
                det_scale = 1.0

            local_index_tip = (index_tip - np.array([x1, y1])) * det_scale
            local_axis = axis  # unit vector, unchanged by uniform scale
            local_span = finger_span * det_scale
            local_geo_tip = self._geometry_tip(local_index_tip, local_axis, local_span)

            gray = cv2.cvtColor(det_roi, cv2.COLOR_BGR2GRAY)
            local_tip = self._refine_tip_with_vision(
                gray,
                local_index_tip,
                local_axis,
                local_span,
                local_geo_tip,
                det_scale,
            )
            tip = local_tip / det_scale + np.array([x1, y1])
            confidence = 0.85

        # Tip must extend beyond the index fingertip along the pen axis.
        tip_along = float(np.dot(tip - index_tip, axis))
        min_along = finger_span * 0.4
        if tip_along < min_along:
            tip = geometry_tip
            confidence = 0.7

        return PenDetectionResult(
            tip=(float(tip[0]), float(tip[1])),
            grip=(float(grip[0]), float(grip[1])),
            axis=(float(axis[0]), float(axis[1])),
            confidence=confidence,
        )
