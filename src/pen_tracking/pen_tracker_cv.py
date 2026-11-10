"""Pen detection using hand geometry with optional vision refinement."""

from __future__ import annotations

from collections import deque
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
    from_hand: bool = True


@dataclass(frozen=True)
class _ShaftSegment:
    """A dark, thin, elongated blob that may be (part of) the pen shaft."""

    points: np.ndarray
    center: np.ndarray
    direction: np.ndarray
    length: float
    thickness: float
    contrast: float


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
        shaft_max_value: int = 70,
        tip_cone_max_saturation: int = 50,
        tip_cone_min_value: int = 170,
        max_missed_frames: int = 3,
        history_size: int = 5,
        max_jump_fraction: float = 0.05,
        min_tip_cone_pixels: int = 2,
        max_rejected_frames: int = 3,
    ) -> None:
        self.max_value = max_value
        self.axis_distance = axis_distance
        self.min_candidates = min_candidates
        self.smoothing = smoothing
        self.detection_height = detection_height
        self.extension_factor = extension_factor
        self.shaft_max_value = shaft_max_value
        self.tip_cone_max_saturation = tip_cone_max_saturation
        self.tip_cone_min_value = tip_cone_min_value
        self.max_missed_frames = max_missed_frames
        self.max_jump_fraction = max_jump_fraction
        self.min_tip_cone_pixels = min_tip_cone_pixels
        self.max_rejected_frames = max_rejected_frames
        self._last: PenDetectionResult | None = None
        self._missed = 0
        self._rejected = 0
        self._history: deque[PenDetectionResult] = deque(maxlen=history_size)

    def track(self, frame_bgr, hand_landmarks=None) -> PenDetectionResult | None:
        """Follow the pen by its appearance; the hand, when given, only says where to look first."""
        hand_hint = None
        fingertip_pixels: list[np.ndarray] = []
        if hand_landmarks:
            height, width = frame_bgr.shape[:2]
            for landmark in (hand_landmarks[4], hand_landmarks[8]):
                fingertip_pixels.append(
                    np.array([landmark.x * width, landmark.y * height], dtype=np.float64)
                )
            hand_hint = (fingertip_pixels[0] + fingertip_pixels[1]) / 2.0

        frame_height = frame_bgr.shape[0]
        raw = self._detect_raw_without_hand(frame_bgr, hand_hint)
        if raw is not None and fingertip_pixels and self._tip_is_on_the_hand(
            raw, fingertip_pixels, frame_height
        ):
            # A dark gap between fingers can look like a short pen; never take that tip.
            raw = None
        if raw is not None:
            self._missed = 0
            if self._is_outlier(raw, frame_height):
                self._rejected += 1
                if self._rejected <= self.max_rejected_frames:
                    return self._last
                # The new position has persisted, so the pen really is there.
                self._history.clear()
                self._last = None
            self._rejected = 0
            self._history.append(raw)
            return self._update(raw, frame_height)

        # Brief misses (motion blur, a finger over the tip) keep the last position;
        # longer ones report no pen rather than a guess.
        self._missed += 1
        if self._missed > self.max_missed_frames:
            self._last = None
            self._history.clear()
            self._rejected = 0
        return self._last

    def _is_outlier(self, raw: PenDetectionResult, frame_height: int) -> bool:
        """Reject a tip far from where the recent average motion says it should be."""
        if not self._history:
            return False
        tips = np.array([p.tip for p in self._history])
        expected_tip = tips[-1]
        if len(tips) >= 2:
            velocity = (tips[-1] - tips[0]) / (len(tips) - 1)
            expected_tip = tips[-1] + velocity
        jump = float(np.linalg.norm(np.array(raw.tip) - expected_tip))
        return jump > self.max_jump_fraction * frame_height

    @staticmethod
    def _tip_is_on_the_hand(
        raw: PenDetectionResult,
        fingertips: list[np.ndarray],
        frame_height: int,
    ) -> bool:
        """True when the tip sits on the fingers instead of past them on the pencil."""
        tip = np.array(raw.tip, dtype=np.float64)
        grip = np.array(raw.grip, dtype=np.float64)
        axis = tip - grip
        length = float(np.linalg.norm(axis))
        if length < 1e-3:
            return True
        axis /= length
        nearest = min(float(np.linalg.norm(tip - finger)) for finger in fingertips)
        # Tip on a knuckle / finger gap is close to a fingertip.
        if nearest < 0.04 * frame_height:
            return True
        # The real tip extends past the fingertips along the shaft.
        tip_along = float(np.dot(tip - grip, axis))
        finger_along = max(float(np.dot(finger - grip, axis)) for finger in fingertips)
        return tip_along < finger_along + 0.02 * frame_height

    def detect(self, frame_bgr, hand_landmarks) -> PenDetectionResult | None:
        """Estimate the pen from the hand pose, refined by the image."""
        return self._update(self._detect_raw(frame_bgr, hand_landmarks), frame_bgr.shape[0])

    def detect_without_hand(self, frame_bgr) -> PenDetectionResult | None:
        """Find the pen from its appearance alone (dark thin shaft, light sharpened tip)."""
        return self.track(frame_bgr)

    def _update(self, raw: PenDetectionResult | None, frame_height: int) -> PenDetectionResult | None:
        if raw is None:
            return self._last

        if self._last is None or self._last.from_hand != raw.from_hand:
            # The two methods place the tip differently; blending across a switch
            # would drag the marker between them for several frames.
            self._last = raw
            return raw

        if not raw.from_hand:
            # Appearance-based tips are accurate per frame, so only damp sub-jitter
            # moves; blending real motion makes the marker trail behind the pen.
            jitter = 0.004 * frame_height
            if np.linalg.norm(np.subtract(raw.tip, self._last.tip)) > jitter:
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
            from_hand=raw.from_hand,
        )
        self._last = smoothed
        return smoothed

    def reset(self) -> None:
        self._last = None
        self._missed = 0
        self._rejected = 0
        self._history.clear()

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

    def _detect_raw_without_hand(
        self,
        frame_bgr,
        hand_hint: np.ndarray | None = None,
    ) -> PenDetectionResult | None:
        height, width = frame_bgr.shape[:2]
        scale = min(1.0, self.detection_height / height)
        if scale < 1.0:
            small = cv2.resize(
                frame_bgr,
                (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = frame_bgr
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        segments = self._find_shaft_segments(hsv)
        if not segments:
            return None

        previous_tip = None
        if self._last is not None and not self._last.from_hand:
            previous_tip = np.array(self._last.tip) * scale
        hint = hand_hint * scale if hand_hint is not None else None
        best = self._pick_shaft(segments, previous_tip, hint, hsv.shape[0])
        if best is None:
            return None
        shaft = self._merge_collinear(best, segments)

        proj = (shaft.points - shaft.center) @ shaft.direction
        end_a = shaft.center + shaft.direction * proj.min()
        end_b = shaft.center + shaft.direction * proj.max()
        tip_end, back_end = self._orient_shaft(end_a, end_b, hsv.shape, previous_tip)

        axis = tip_end - back_end
        axis /= np.linalg.norm(axis)
        tip, cone_steps = self._extend_through_tip_cone(hsv, tip_end, axis, shaft)
        if cone_steps < self.min_tip_cone_pixels:
            # No sharpened point at this end: a dark streak (finger gap, shadow), not the tip.
            return None

        tip_full = tip / scale
        if scale < 1.0:
            tip_full = self._refine_at_full_resolution(frame_bgr, tip_full, axis, scale)
        back_full = back_end / scale
        return PenDetectionResult(
            tip=(float(tip_full[0]), float(tip_full[1])),
            grip=(float(back_full[0]), float(back_full[1])),
            axis=(float(axis[0]), float(axis[1])),
            confidence=0.6,
            from_hand=False,
        )

    def _find_shaft_segments(self, hsv: np.ndarray) -> list[_ShaftSegment]:
        img_h = hsv.shape[0]
        value = hsv[..., 2]
        dark = (value < self.shaft_max_value).astype(np.uint8) * 255
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        min_length = 0.05 * img_h
        max_thickness = 0.04 * img_h
        ring_kernel = np.ones((7, 7), np.uint8)

        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        segments: list[_ShaftSegment] = []
        for contour in contours:
            if len(contour) < 10:
                continue
            (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(contour)
            length, thickness = max(rect_w, rect_h), max(1.0, min(rect_w, rect_h))
            if length < min_length or thickness > max_thickness or length / thickness < 4.0:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            pad = 6
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(hsv.shape[1], x + w + pad), min(img_h, y + h + pad)
            blob = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.drawContours(blob, [contour], -1, 255, -1, offset=(-x0, -y0))
            ring = cv2.dilate(blob, ring_kernel) & ~blob
            patch = value[y0:y1, x0:x1]
            if not ring.any() or not blob.any():
                continue
            # A pen on a table or in a hand is a dark line on a much brighter
            # surround; dark background clutter (books, shadows) is not.
            contrast = float(patch[ring > 0].mean()) - float(patch[blob > 0].mean())
            if contrast < 60.0:
                continue

            points = contour.reshape(-1, 2).astype(np.float64)
            vx, vy, cx, cy = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            segments.append(
                _ShaftSegment(
                    points=points,
                    center=np.array([cx, cy], dtype=np.float64),
                    direction=np.array([vx, vy], dtype=np.float64),
                    length=float(length),
                    thickness=float(thickness),
                    contrast=contrast,
                )
            )
        return segments

    def _refine_at_full_resolution(
        self,
        frame_bgr,
        tip: np.ndarray,
        axis: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        """Re-snap the tip on the original pixels; one detection pixel spans several here."""
        height, width = frame_bgr.shape[:2]
        reach = int(np.ceil(3.0 / scale))
        x0, y0 = max(0, int(tip[0]) - reach), max(0, int(tip[1]) - reach)
        x1, y1 = min(width, int(tip[0]) + reach + 1), min(height, int(tip[1]) + reach + 1)
        if x1 <= x0 or y1 <= y0:
            return tip
        hsv = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
        offset = np.array([x0, y0], dtype=np.float64)
        local = self._snap_to_lead_apex(hsv, tip - offset, axis, 2.0 / scale)
        return local + offset

    @staticmethod
    def _pick_shaft(
        segments: list[_ShaftSegment],
        previous_tip: np.ndarray | None,
        hand_hint: np.ndarray | None,
        img_h: int,
    ) -> _ShaftSegment | None:
        def distance_to_segment(segment: _ShaftSegment, point: np.ndarray) -> float:
            ends = (segment.points - segment.center) @ segment.direction
            proj = float(np.clip(np.dot(point - segment.center, segment.direction), ends.min(), ends.max()))
            closest = segment.center + segment.direction * proj
            return float(np.linalg.norm(point - closest))

        def distance_to_ends(segment: _ShaftSegment, point: np.ndarray) -> float:
            proj = (segment.points - segment.center) @ segment.direction
            ends = [
                segment.center + segment.direction * proj.min(),
                segment.center + segment.direction * proj.max(),
            ]
            return min(float(np.linalg.norm(end - point)) for end in ends)

        if previous_tip is not None:
            max_jump = 0.15 * img_h
            nearby = [s for s in segments if distance_to_ends(s, previous_tip) < max_jump]
            if nearby:
                return min(nearby, key=lambda s: distance_to_ends(s, previous_tip))
        if hand_hint is not None:
            # The pen passes between thumb and index finger.
            max_gap = 0.08 * img_h
            held = [s for s in segments if distance_to_segment(s, hand_hint) < max_gap]
            if held:
                return max(held, key=lambda s: s.length * s.contrast)
        return max(segments, key=lambda s: s.length * s.contrast)

    @staticmethod
    def _merge_collinear(best: _ShaftSegment, segments: list[_ShaftSegment]) -> _ShaftSegment:
        """Join pieces of the same shaft split by fingers or the pencil's printed band."""
        normal = np.array([-best.direction[1], best.direction[0]])
        max_offset = best.thickness * 1.5 + 4.0
        merged = [best.points]
        for segment in segments:
            if segment is best:
                continue
            if abs(float(np.dot(segment.direction, best.direction))) < np.cos(np.radians(10)):
                continue
            if abs(float(np.dot(segment.center - best.center, normal))) > max_offset:
                continue
            merged.append(segment.points)
        if len(merged) == 1:
            return best

        points = np.vstack(merged)
        vx, vy, cx, cy = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        direction = np.array([vx, vy], dtype=np.float64)
        center = np.array([cx, cy], dtype=np.float64)
        proj = (points - center) @ direction
        return _ShaftSegment(
            points=points,
            center=center,
            direction=direction,
            length=float(proj.max() - proj.min()),
            thickness=best.thickness,
            contrast=best.contrast,
        )

    @staticmethod
    def _orient_shaft(
        end_a: np.ndarray,
        end_b: np.ndarray,
        shape: tuple[int, ...],
        previous_tip: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (tip_end, back_end)."""
        img_h, img_w = shape[:2]
        margin = 4.0

        def at_border(point: np.ndarray) -> bool:
            return (
                point[0] < margin
                or point[1] < margin
                or point[0] > img_w - 1 - margin
                or point[1] > img_h - 1 - margin
            )

        a_border, b_border = at_border(end_a), at_border(end_b)
        if a_border != b_border:
            return (end_b, end_a) if a_border else (end_a, end_b)
        if previous_tip is not None:
            if np.linalg.norm(end_a - previous_tip) <= np.linalg.norm(end_b - previous_tip):
                return end_a, end_b
            return end_b, end_a
        # A pen being used points down towards the writing surface.
        return (end_a, end_b) if end_a[1] >= end_b[1] else (end_b, end_a)

    def _extend_through_tip_cone(
        self,
        hsv: np.ndarray,
        shaft_end: np.ndarray,
        axis: np.ndarray,
        shaft: _ShaftSegment,
    ) -> tuple[np.ndarray, int]:
        """Walk past the dark shaft over the light sharpened wood and the lead point.

        Returns the tip and how many steps crossed the light sharpened wood.
        """
        img_h, img_w = hsv.shape[:2]
        normal = np.array([-axis[1], axis[0]])
        half_width = max(1.0, shaft.thickness / 2.5)
        max_walk = max(6.0, 0.35 * shaft.length)

        tip = shaft_end.copy()
        cone_steps = 0
        misses = 0
        along = 1.0
        while along <= max_walk:
            center = shaft_end + axis * along
            hit = False
            on_cone = False
            for offset in (-half_width, 0.0, half_width):
                x, y = (center + normal * offset).round().astype(int)
                if not (0 <= x < img_w and 0 <= y < img_h):
                    continue
                sat, val = int(hsv[y, x, 1]), int(hsv[y, x, 2])
                is_cone = sat < self.tip_cone_max_saturation and val > self.tip_cone_min_value
                # Graphite is shiny mid-grey: darker than the wood and nearly colourless.
                is_lead = val < self.shaft_max_value or (val < 140 and sat < 70)
                on_cone = on_cone or is_cone
                hit = hit or is_cone or is_lead
            if hit:
                tip = center
                misses = 0
                cone_steps += int(on_cone)
            else:
                misses += 1
                if misses > 2:
                    break
            along += 1.0
        return self._snap_to_lead_apex(hsv, tip, axis, shaft.thickness), cone_steps

    def _snap_to_lead_apex(
        self,
        hsv: np.ndarray,
        tip: np.ndarray,
        axis: np.ndarray,
        thickness: float,
    ) -> np.ndarray:
        """Move the tip onto the furthest graphite pixel; the fitted axis is rarely centred on it."""
        img_h, img_w = hsv.shape[:2]
        radius = int(np.ceil(max(3.0, thickness)))
        x0, y0 = max(0, int(tip[0]) - radius), max(0, int(tip[1]) - radius)
        x1, y1 = min(img_w, int(tip[0]) + radius + 1), min(img_h, int(tip[1]) + radius + 1)
        patch = hsv[y0:y1, x0:x1]
        if patch.size == 0:
            return tip
        sat, val = patch[..., 1].astype(int), patch[..., 2].astype(int)
        lead = (val < self.shaft_max_value) | ((val < 140) & (sat < 70))
        ys, xs = np.nonzero(lead)
        if len(xs) == 0:
            return tip
        pixels = np.stack([xs + x0, ys + y0], axis=1).astype(np.float64)
        along = (pixels - tip) @ axis
        apex = pixels[along >= along.max() - 1.5].mean(axis=0)
        return apex
