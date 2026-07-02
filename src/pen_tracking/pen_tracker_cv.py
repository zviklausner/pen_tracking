"""Pen detection using hand-guided color segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PenDetectionResult:
    """Detected pen geometry in image coordinates."""

    tip: tuple[float, float]
    grip: tuple[float, float]
    axis: tuple[float, float]
    confidence: float


class PenTracker:
    """Detects a dark pen held in a tracked hand."""

    def __init__(
        self,
        *,
        max_value: int = 100,
        axis_distance: float = 20.0,
        min_candidates: int = 5,
        smoothing: float = 0.65,
    ) -> None:
        self.max_value = max_value
        self.axis_distance = axis_distance
        self.min_candidates = min_candidates
        self.smoothing = smoothing
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

    def _detect_raw(self, frame_bgr, hand_landmarks) -> PenDetectionResult | None:
        height, width = frame_bgr.shape[:2]
        points = np.array(
            [[landmark.x * width, landmark.y * height] for landmark in hand_landmarks]
        )

        index_mcp = points[5]
        index_tip = points[8]
        thumb_tip = points[4]
        axis = index_tip - index_mcp
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-3:
            return None

        axis = axis / axis_norm
        perpendicular = np.array([-axis[1], axis[0]])
        grip = (thumb_tip + index_tip) / 2

        padding = 60
        x1 = int(max(0, points[:, 0].min() - padding))
        y1 = int(max(0, points[:, 1].min() - padding))
        x2 = int(min(width, points[:, 0].max() + padding))
        y2 = int(min(height, points[:, 1].max() + padding + 180))
        roi = frame_bgr[y1:y2, x1:x2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, self.max_value))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None

        candidates: list[tuple[float, float, float, float]] = []
        for x, y in zip(xs, ys):
            px = float(x + x1)
            py = float(y + y1)
            relative = np.array([px, py]) - grip
            along = float(np.dot(relative, axis))
            across = abs(float(np.dot(relative, perpendicular)))
            if along < -10 or along > 260 or across > self.axis_distance:
                continue
            if py < index_tip[1] - 15:
                continue
            candidates.append((px, py, along, across))

        if len(candidates) < self.min_candidates:
            fallback = self._detect_from_contour(mask, x1, y1, grip, index_tip)
            if fallback is None:
                return None
            tip, grip_point, confidence = fallback
            return PenDetectionResult(
                tip=tip,
                grip=grip_point,
                axis=(float(axis[0]), float(axis[1])),
                confidence=confidence,
            )

        arr = np.array(candidates)
        tip = tuple(arr[np.argmax(arr[:, 1]), :2])
        near_grip = arr[arr[:, 2] < 40]
        if len(near_grip):
            grip_point = tuple(
                near_grip[
                    np.argmin(np.linalg.norm(near_grip[:, :2] - grip, axis=1))
                ][:2]
            )
        else:
            grip_point = (float(grip[0]), float(grip[1]))

        confidence = min(1.0, len(candidates) / 40.0)
        return PenDetectionResult(
            tip=tip,
            grip=grip_point,
            axis=(float(axis[0]), float(axis[1])),
            confidence=confidence,
        )

    @staticmethod
    def _detect_from_contour(
        mask,
        x_offset: int,
        y_offset: int,
        grip: np.ndarray,
        index_tip: np.ndarray,
    ) -> tuple[tuple[float, float], tuple[float, float], float] | None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_score = -1.0
        best: tuple[tuple[float, float], tuple[float, float], float] | None = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 80 or area > 8000:
                continue

            rect = cv2.minAreaRect(contour)
            _, (rect_w, rect_h), _ = rect
            major = max(rect_w, rect_h)
            minor = min(rect_w, rect_h)
            if minor < 1 or major / minor < 2.0:
                continue

            contour_points = contour.reshape(-1, 2).astype(float)
            contour_points[:, 0] += x_offset
            contour_points[:, 1] += y_offset

            if np.linalg.norm(contour_points - grip, axis=1).min() > 120:
                continue
            if contour_points[:, 1].max() < index_tip[1] - 10:
                continue

            tip = tuple(contour_points[np.argmax(contour_points[:, 1])])
            grip_point = tuple(
                contour_points[np.argmin(np.linalg.norm(contour_points - grip, axis=1))]
            )
            score = area * (major / minor)
            if score > best_score:
                best_score = score
                best = (tip, grip_point, min(1.0, score / 500.0))

        return best
