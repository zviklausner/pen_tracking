"""A region of the frame the tracker is allowed to search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

RegionShape = Literal["rectangle", "square", "ellipse", "circle", "semicircle"]
REGION_SHAPES: tuple[RegionShape, ...] = (
    "rectangle",
    "square",
    "ellipse",
    "circle",
    "semicircle",
)


@dataclass(frozen=True)
class InterestRegion:
    """A shape drawn on the source frame, in full-frame pixels.

    Rectangle, square, ellipse, and circle store the opposite corners of the
    bounding box. A semicircle stores the two ends of its straight edge; the
    arc bulges to the right of the direction from the first point to the second.
    """

    shape: RegionShape
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.shape not in REGION_SHAPES:
            raise ValueError(f"Region shape must be one of {REGION_SHAPES}")
        if self._span() < 8.0:
            raise ValueError("The region is too small. Drag a larger shape.")

    @staticmethod
    def from_drag(shape: RegionShape, x0: float, y0: float, x1: float, y1: float) -> InterestRegion:
        """Build a region from a mouse drag. Square and circle stay equal-sided."""
        if shape in ("square", "circle"):
            dx = x1 - x0
            dy = y1 - y0
            side = max(abs(dx), abs(dy))
            x1 = x0 + side * (1.0 if dx >= 0.0 else -1.0)
            y1 = y0 + side * (1.0 if dy >= 0.0 else -1.0)
        return InterestRegion(shape, float(x0), float(y0), float(x1), float(y1))

    def flipped(self) -> InterestRegion:
        """Put a semicircle's arc on the other side of its straight edge."""
        return InterestRegion(self.shape, self.x1, self.y1, self.x0, self.y0)

    def _span(self) -> float:
        return float(np.hypot(self.x1 - self.x0, self.y1 - self.y0))

    def _box(self) -> tuple[float, float, float, float]:
        left, right = sorted((self.x0, self.x1))
        top, bottom = sorted((self.y0, self.y1))
        return left, top, right, bottom

    def _semicircle(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        start = np.array([self.x0, self.y0], dtype=np.float64)
        end = np.array([self.x1, self.y1], dtype=np.float64)
        chord = end - start
        length = float(np.linalg.norm(chord))
        center = 0.5 * (start + end)
        direction = chord / length
        # Screen y grows downward, so this normal points to the right of the drag.
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        return center, normal, direction, length * 0.5

    def bounds(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Crop rectangle as x0, y0, x1, y1, clamped to the frame and exclusive at the far edge."""
        if self.shape == "semicircle":
            center, normal, _direction, radius = self._semicircle()
            points = [
                np.array([self.x0, self.y0]),
                np.array([self.x1, self.y1]),
                center + normal * radius,
            ]
            for step in ((radius, 0.0), (-radius, 0.0), (0.0, radius), (0.0, -radius)):
                point = center + np.array(step)
                if float(np.dot(point - center, normal)) >= -1e-3:
                    points.append(point)
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
        else:
            left, top, right, bottom = self._box()
        x0 = int(np.floor(left))
        y0 = int(np.floor(top))
        x1 = int(np.ceil(right))
        y1 = int(np.ceil(bottom))
        x0 = max(0, min(width - 1, x0))
        y0 = max(0, min(height - 1, y0))
        x1 = max(x0 + 1, min(width, x1))
        y1 = max(y0 + 1, min(height, y1))
        return x0, y0, x1, y1

    def contains(self, x: float, y: float) -> bool:
        point = np.array([x, y], dtype=np.float64)
        if self.shape in ("rectangle", "square"):
            left, top, right, bottom = self._box()
            return left <= x <= right and top <= y <= bottom
        if self.shape in ("ellipse", "circle"):
            left, top, right, bottom = self._box()
            center_x = 0.5 * (left + right)
            center_y = 0.5 * (top + bottom)
            radius_x = max(0.5 * (right - left), 1e-6)
            radius_y = max(0.5 * (bottom - top), 1e-6)
            return ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2 <= 1.0
        center, normal, _direction, radius = self._semicircle()
        offset = point - center
        return float(np.dot(offset, offset)) <= radius * radius + 1e-6 and float(
            np.dot(offset, normal)
        ) >= -1e-6

    def mask(self, origin_x: int, origin_y: int, height: int, width: int) -> np.ndarray:
        """True for pixels of a crop that lie inside the shape."""
        ys, xs = np.ogrid[0:height, 0:width]
        global_x = xs + origin_x
        global_y = ys + origin_y
        if self.shape in ("rectangle", "square"):
            return np.ones((height, width), dtype=bool)
        if self.shape in ("ellipse", "circle"):
            left, top, right, bottom = self._box()
            center_x = 0.5 * (left + right)
            center_y = 0.5 * (top + bottom)
            radius_x = max(0.5 * (right - left), 1e-6)
            radius_y = max(0.5 * (bottom - top), 1e-6)
            return ((global_x - center_x) / radius_x) ** 2 + (
                (global_y - center_y) / radius_y
            ) ** 2 <= 1.0
        center, normal, _direction, radius = self._semicircle()
        offset_x = global_x - center[0]
        offset_y = global_y - center[1]
        return (offset_x**2 + offset_y**2 <= radius * radius + 1e-6) & (
            offset_x * normal[0] + offset_y * normal[1] >= -1e-6
        )

    def crop(self, frame: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Return the cropped frame and the crop origin in full-frame pixels.

        Pixels outside an ellipse, circle, or semicircle are painted white so the
        search cannot lock onto them, while the smaller image is what gets processed.
        """
        height, width = frame.shape[:2]
        x0, y0, x1, y1 = self.bounds(width, height)
        cropped = frame[y0:y1, x0:x1].copy()
        if self.shape not in ("rectangle", "square"):
            inside = self.mask(x0, y0, cropped.shape[0], cropped.shape[1])
            cropped[~inside] = 255
        return cropped, x0, y0

    def outline(self, samples: int = 64) -> np.ndarray:
        """Polyline of the shape, in full-frame pixels, for the drawing preview."""
        if self.shape in ("rectangle", "square"):
            left, top, right, bottom = self._box()
            return np.array(
                [[left, top], [right, top], [right, bottom], [left, bottom], [left, top]]
            )
        if self.shape in ("ellipse", "circle"):
            left, top, right, bottom = self._box()
            center = np.array([0.5 * (left + right), 0.5 * (top + bottom)])
            radius = np.array([0.5 * (right - left), 0.5 * (bottom - top)])
            theta = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=True)
            return center + np.stack((np.cos(theta), np.sin(theta)), axis=1) * radius
        center, normal, direction, radius = self._semicircle()
        theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, samples // 2)
        arc = center + (direction * np.sin(theta)[:, None] + normal * np.cos(theta)[:, None]) * radius
        return np.vstack((arc, arc[:1]))


def read_first_frame(video_path: str) -> np.ndarray | None:
    """Open one frame so the region can be drawn before processing starts."""
    capture = cv2.VideoCapture(video_path)
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        return None
    return frame
