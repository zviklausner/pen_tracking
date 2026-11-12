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
        self._shaft_length_px: float | None = None

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
        # Silver / metallic pens are too bright for the dark-shaft mask; fall back to a
        # relative-darkness scan guided by the writing grip.
        if raw is None and hand_landmarks is not None:
            raw = self._detect_relative_shaft(frame_bgr, hand_landmarks)
            if raw is not None and fingertip_pixels and self._tip_is_on_the_hand(
                raw, fingertip_pixels, frame_height
            ):
                raw = None
        if raw is not None:
            # The downscaled search stops a few pixels short of the lead, so the
            # trail kinks off the point. Seat the tip on the last dark pixel.
            raw = self._seat_tip_on_apex(frame_bgr, raw)
        if raw is not None and self._tip_is_on_paper(frame_bgr, raw):
            # A tip on blank paper is always wrong; drop it rather than lock onto drift.
            raw = None
        if raw is not None and hand_hint is not None and self._shaft_misses_the_grip(
            raw, hand_hint, frame_height
        ):
            # Line drifted beside the pen instead of through the fingers.
            raw = None
            if self._last is not None and self._shaft_misses_the_grip(
                self._last, hand_hint, frame_height
            ):
                self._last = None
                self._history.clear()
                self._shaft_length_px = None
        if raw is not None:
            self._missed = 0
            if self._is_outlier(raw, frame_height):
                self._rejected += 1
                if self._rejected <= self.max_rejected_frames:
                    return self._last
                # The new position has persisted, so the pen really is there.
                self._history.clear()
                self._last = None
                self._shaft_length_px = None
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
            self._shaft_length_px = None
        if (
            self._last is not None
            and self._missed > 1
            and self._tip_is_on_paper(frame_bgr, self._last)
        ):
            self._last = None
            self._history.clear()
            self._rejected = 0
            self._shaft_length_px = None
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
        if jump > self.max_jump_fraction * frame_height:
            return True
        # Also reject a sudden sideways escape while length stays similar.
        last = self._history[-1]
        last_axis = np.array(last.axis, dtype=np.float64)
        tip_delta = np.array(raw.tip, dtype=np.float64) - np.array(last.tip, dtype=np.float64)
        lateral = abs(float(last_axis[0] * tip_delta[1] - last_axis[1] * tip_delta[0]))
        return lateral > 0.032 * frame_height

    @staticmethod
    def _shaft_misses_the_grip(
        raw: PenDetectionResult,
        hand_hint: np.ndarray,
        frame_height: int,
    ) -> bool:
        """True when the tip–grip line does not pass near the writing grip."""
        tip = np.array(raw.tip, dtype=np.float64)
        grip = np.array(raw.grip, dtype=np.float64)
        axis = tip - grip
        length = float(np.linalg.norm(axis))
        if length < 1e-3:
            return True
        axis /= length
        along = float(np.dot(hand_hint - grip, axis))
        closest = grip + axis * float(np.clip(along, 0.0, length))
        return float(np.linalg.norm(hand_hint - closest)) > 0.048 * frame_height

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

    @staticmethod
    def _seat_tip_on_apex(frame_bgr, raw: PenDetectionResult) -> PenDetectionResult:
        """Move the tip forward onto the lead, where paper starts just beyond it."""
        tip = np.array(raw.tip, dtype=np.float64)
        grip = np.array(raw.grip, dtype=np.float64)
        axis = tip - grip
        length = float(np.linalg.norm(axis))
        if length < 1e-3:
            return raw
        axis = axis / length
        normal = np.array([-axis[1], axis[0]], dtype=np.float64)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        height, width = hsv.shape[:2]
        best: np.ndarray | None = None
        best_score = -1e9
        for along in np.arange(0.0, 24.0, 1.0):
            for off in np.arange(-8.0, 8.1, 1.0):
                point = tip + axis * along + normal * off
                x, y = int(round(point[0])), int(round(point[1]))
                if not (1 <= x < width - 1 and 1 <= y < height - 1):
                    continue
                brightness = float(hsv[y, x, 2])
                saturation = float(hsv[y, x, 1])
                if brightness > 105 or saturation > 90:
                    continue
                ahead = point + axis * 5.0
                ax, ay = int(round(ahead[0])), int(round(ahead[1]))
                if not (0 <= ax < width and 0 <= ay < height):
                    continue
                if float(hsv[ay, ax, 2]) <= 145 or float(hsv[ay, ax, 1]) >= 45:
                    continue
                score = float(along) - 0.4 * abs(float(off))
                if score > best_score:
                    best_score = score
                    best = point
        if best is None:
            return raw
        # The marker drawn on the apex hangs past the lead onto the paper.
        best = best - axis * 4.0
        if float(np.dot(best - tip, axis)) < 0.0:
            best = tip
        new_axis = best - grip
        new_len = float(np.linalg.norm(new_axis))
        if new_len < 1e-3:
            return raw
        new_axis = new_axis / new_len
        return PenDetectionResult(
            tip=(float(best[0]), float(best[1])),
            grip=raw.grip,
            axis=(float(new_axis[0]), float(new_axis[1])),
            confidence=raw.confidence,
            from_hand=raw.from_hand,
        )

    @staticmethod
    def _tip_is_on_paper(frame_bgr, raw: PenDetectionResult) -> bool:
        """True when the reported tip lands on blank bright paper instead of the pen.

        Only flags empty paper: bright, nearly colourless, with no dark tip nearby.
        A light wood cone next to graphite must not be rejected.
        """
        height, width = frame_bgr.shape[:2]
        x, y = int(round(raw.tip[0])), int(round(raw.tip[1]))
        if not (0 <= x < width and 0 <= y < height):
            return True
        patch = frame_bgr[max(0, y - 4) : y + 5, max(0, x - 4) : x + 5]
        if patch.size == 0:
            return True
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        value = float(hsv[..., 2].mean())
        sat = float(hsv[..., 1].mean())
        darkest = float(hsv[..., 2].min())
        return value > 150 and sat < 40 and darkest > 130

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
        self._shaft_length_px = None

    def _stabilize_shaft_length(
        self,
        tip: np.ndarray,
        back: np.ndarray,
        axis: np.ndarray,
        frame_height: int,
        *,
        appearance_length: float | None = None,
    ) -> np.ndarray:
        """Keep overlay length steady, but never past the visible upper end."""
        length = float(np.linalg.norm(tip - back))
        min_len = 0.08 * frame_height
        max_len = 0.28 * frame_height
        if appearance_length is not None and appearance_length > min_len:
            length = min(length, appearance_length)
        length = float(np.clip(length, min_len, max_len))
        if self._shaft_length_px is None:
            self._shaft_length_px = length
        else:
            delta = length - self._shaft_length_px
            if delta >= 0:
                # Reach a newly visible eraser within a frame or two.
                step = min(delta, 0.06 * frame_height)
            else:
                # One frame that loses the upper shaft must not collapse the line.
                step = max(delta, -0.01 * frame_height)
            updated = self._shaft_length_px + step
            if (
                appearance_length is not None
                and appearance_length > min_len
                and appearance_length >= self._shaft_length_px - 0.02 * frame_height
            ):
                updated = min(updated, appearance_length)
            self._shaft_length_px = float(np.clip(updated, min_len, max_len))
        return tip - axis * self._shaft_length_px

    def _commit_upper_end(
        self,
        tip: np.ndarray,
        axis: np.ndarray,
        visible_back: np.ndarray,
        img_h: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Keep the eraser when one frame's walk stops short on the hand."""
        delta = tip - visible_back
        dist = float(np.linalg.norm(delta))
        if (
            self._shaft_length_px is not None
            and dist < self._shaft_length_px - 0.03 * img_h
            and self._last is not None
        ):
            previous = np.array(self._last.axis, dtype=np.float64)
            norm = float(np.linalg.norm(previous))
            if norm > 1e-6:
                axis = previous / norm
        elif dist > 1e-3:
            axis = delta / dist
        back = self._stabilize_shaft_length(
            tip, visible_back, axis, img_h, appearance_length=dist
        )
        return tip, back, axis

    @staticmethod
    def _visible_upper_end(
        tip: np.ndarray,
        axis: np.ndarray,
        blackhat: np.ndarray,
        value: np.ndarray,
        sat: np.ndarray,
        img_w: int,
        img_h: int,
        hint: np.ndarray,
    ) -> np.ndarray:
        """Follow the shaft from the grip up to the eraser, staying on the ridge.

        The writing axis is often a few pixels off, so each step may slide
        sideways onto the dark ridge. The walk stops at blank background.
        """
        del sat  # brightness and the ridge are enough; the shaft can look tinted
        direction = -axis
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return hint.copy()
        back_axis = direction / norm
        normal = np.array([-back_axis[1], back_axis[0]], dtype=np.float64)
        grip_along = max(0.02 * img_h, float(np.dot(hint - tip, back_axis)))

        def sample(along: float, off: float) -> tuple[float, float, np.ndarray] | None:
            point = tip + back_axis * along + normal * off
            x, y = int(round(point[0])), int(round(point[1]))
            if not (0 <= x < img_w and 0 <= y < img_h):
                return None
            return float(blackhat[y, x]), float(value[y, x]), point

        # Measure length along the tip axis. Look a few pixels sideways only
        # when the axis itself has left the shaft, so the line cannot drift
        # onto a parallel edge of the hand.
        last_along = grip_along
        end_off = 0.0
        gaps = 0
        limit = min(0.34 * img_h, grip_along + 0.18 * img_h)
        for along in np.arange(4.0, limit, 1.0):
            found_off = None
            center = sample(float(along), 0.0)
            if center is not None and center[0] >= 8 and center[1] < 160:
                found_off = 0.0
            elif along > grip_along:
                for off in (-3.0, 3.0, -6.0, 6.0, -8.0, 8.0):
                    hit = sample(float(along), off)
                    if hit is not None and hit[0] >= 12 and hit[1] < 145:
                        found_off = off
                        break
            if found_off is not None:
                last_along = float(along)
                end_off = found_off
                gaps = 0
                continue
            if along < grip_along + 0.02 * img_h:
                continue
            gaps += 1
            if gaps > 5:
                break
        for step in range(0, 8):
            seated = sample(last_along - float(step), end_off)
            if seated is not None and seated[0] >= 12 and seated[1] < 115:
                # The cap drawn on this point hangs past the metal, and the
                # black-hat response continues a few pixels into the wall.
                inset = min(3.0, float(step) + 3.0)
                return tip + back_axis * (last_along - float(step) - inset) + normal * end_off
        return tip + back_axis * last_along + normal * end_off

    def _appearance_back_length(
        self,
        tip: np.ndarray,
        axis: np.ndarray,
        blackhat: np.ndarray,
        value: np.ndarray,
        sat: np.ndarray,
        img_w: int,
        img_h: int,
        hint: np.ndarray,
    ) -> float:
        """How far the shaft still looks like a pen when walking tip → upper end."""
        back_axis = -axis
        normal = np.array([-axis[1], axis[0]], dtype=np.float64)
        grip_along = max(0.02 * img_h, float(np.dot(hint - tip, back_axis)))
        last_along = grip_along + 0.02 * img_h
        gaps = 0
        for along in np.arange(1.0, 0.24 * img_h, 1.0):
            point = tip + back_axis * along
            x, y = int(round(point[0])), int(round(point[1]))
            if not (0 <= x < img_w and 0 <= y < img_h):
                break
            bh = float(blackhat[y, x])
            brightness = float(value[y, x])
            saturation = float(sat[y, x])
            side = 0.0
            samples = 0
            for offset in (3.0, 5.0):
                for sign in (-1.0, 1.0):
                    neighbor = point + normal * sign * offset
                    nx, ny = int(round(neighbor[0])), int(round(neighbor[1]))
                    if 0 <= nx < img_w and 0 <= ny < img_h:
                        side += float(blackhat[ny, nx])
                        samples += 1
            thin = bh - (side / samples if samples else 0.0)
            is_paper = brightness > 150 and saturation < 45 and bh < 6
            # Silver shafts stay bright; trust a thin black-hat ridge, not only dark pixels.
            is_pen = (
                (bh >= 6 and thin > 1.5)
                or (brightness < 135 and saturation < 80 and bh >= 4)
                or (thin > 2.5 and bh >= 5 and saturation < 85)
            )
            is_skin = saturation > 95 and brightness > 100
            if along < grip_along + 0.02 * img_h:
                if is_pen or is_skin:
                    last_along = along
                continue
            if is_paper:
                break
            # Past the fingers the upper end must stay a clear thin ridge, not hand edges.
            if is_skin:
                # Cross the finger zone, but do not grow length on skin alone.
                gaps += 1
                if gaps > 5:
                    break
                continue
            if is_pen and thin > 2.0:
                last_along = along
                gaps = 0
                continue
            gaps += 1
            if gaps > 1:
                break
        # Stop at the visible eraser, which sits well above the fingers.
        return float(min(last_along, grip_along + 0.20 * img_h))

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

    def _detect_relative_shaft(self, frame_bgr, hand_landmarks) -> PenDetectionResult | None:
        """Find a bright metallic pen from its edges/shaft, not from millimetre priors."""
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
        img_h, img_w = small.shape[:2]

        points = np.array(
            [[landmark.x * img_w, landmark.y * img_h] for landmark in hand_landmarks],
            dtype=np.float64,
        )
        thumb, index, mcp = points[4], points[8], points[5]
        hint = (thumb + index) / 2.0
        finger = index - mcp
        finger_span = float(np.linalg.norm(finger))
        if finger_span < 1e-3:
            return None
        finger = finger / finger_span

        prev_tip = None
        if self._last is not None and not self._last.from_hand:
            prev_tip = np.array(self._last.tip, dtype=np.float64) * scale

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        blackhat = cv2.morphologyEx(
            gray,
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
        ).astype(np.float32)
        sat = hsv[..., 1].astype(np.float32)
        value = hsv[..., 2].astype(np.float32)

        # Upper metallic shaft first (visible above the fingers), then tip along that axis.
        upper = self._find_upper_shaft_point(
            hint, finger, blackhat, value, sat, img_w, img_h
        )
        candidates: list[tuple[np.ndarray, np.ndarray]] = []
        if upper is not None:
            tip_try, axis_try = self._tip_from_upper_shaft(
                upper, finger, hint, value, sat, blackhat, img_w, img_h
            )
            if tip_try is not None and axis_try is not None:
                candidates.append((tip_try, axis_try))
        hough = self._hough_shaft_segment(
            gray, blackhat, hint, finger, value, sat, img_h, img_w, prev_tip
        )
        if hough is not None:
            candidates.append((hough[0], hough[2]))
        saved_len = self._shaft_length_px
        ray = self._detect_relative_shaft_rays(
            small, scale, hint, finger, gray, blackhat, sat, value, img_h, img_w
        )
        self._shaft_length_px = saved_len
        if ray is not None:
            tip_r = np.array(ray.tip, dtype=np.float64) * scale
            axis_r = np.array(ray.axis, dtype=np.float64)
            axis_n = float(np.linalg.norm(axis_r))
            if axis_n > 1e-6:
                candidates.append((tip_r, axis_r / axis_n))

        best: tuple[float, np.ndarray, np.ndarray, float] | None = None
        for tip0, axis0 in candidates:
            tip = self._lateral_snap_whole_shaft(
                tip0, axis0, blackhat, hint, img_w, img_h
            )
            tip = self._snap_tip_before_paper(tip, axis0, value, sat, img_w, img_h)
            tip = self._recenter_tip_on_dark(tip, value, sat, img_w, img_h, axis=axis0)
            tip = self._pull_tip_onto_lead(tip, axis0, value, sat, img_w, img_h)
            if not self._relative_shaft_geometry_ok(tip, axis0, hint, finger, img_h):
                continue
            if not self._tip_looks_like_graphite(tip, value, sat, img_w, img_h):
                continue
            if not self._tip_has_shaft_support(tip, axis0, blackhat, hint, img_w, img_h):
                continue
            appear_len = self._appearance_back_length(
                tip, axis0, blackhat, value, sat, img_w, img_h, hint
            )
            grip_along = float(np.dot(hint - tip, -axis0))
            soft_cap = grip_along + 0.20 * img_h
            if upper is not None:
                upper_along = float(np.dot(upper - tip, -axis0))
                upper_lat = abs(
                    float(
                        axis0[0] * (upper[1] - tip[1]) - axis0[1] * (upper[0] - tip[0])
                    )
                )
                # Trust upper only when it sits on this axis and isn't absurdly long.
                if (
                    upper_lat <= 0.025 * img_h
                    and grip_along < upper_along <= grip_along + 0.20 * img_h
                ):
                    appear_len = max(appear_len, upper_along)
                    soft_cap = max(soft_cap, upper_along)
            appear_len = min(appear_len, soft_cap)
            ridge = self._shaft_ridge_score(tip, axis0, blackhat, appear_len, img_w, img_h)
            quality = (
                float(np.dot(axis0, finger)) * 1.5
                + appear_len / max(img_h, 1.0)
                + ridge * 0.02
            )
            if prev_tip is not None:
                quality -= float(np.linalg.norm(tip - prev_tip)) / max(img_h, 1.0)
            if best is None or quality > best[0]:
                best = (quality, tip, axis0, appear_len)
        if best is None:
            return None
        _, tip, axis, _appear_len = best
        visible_back = self._visible_upper_end(
            tip, axis, blackhat, value, sat, img_w, img_h, hint
        )
        _tip, back, axis = self._commit_upper_end(tip, axis, visible_back, img_h)
        tip_full = tip / scale
        back_full = back / scale
        return PenDetectionResult(
            tip=(float(tip_full[0]), float(tip_full[1])),
            grip=(float(back_full[0]), float(back_full[1])),
            axis=(float(axis[0]), float(axis[1])),
            confidence=0.58,
            from_hand=False,
        )

    @staticmethod
    def _relative_shaft_geometry_ok(
        tip: np.ndarray,
        axis: np.ndarray,
        hint: np.ndarray,
        finger: np.ndarray,
        img_h: int,
    ) -> bool:
        """Reject shafts that leave the finger corridor (vertical / sideways escapes)."""
        tip_along = float(np.dot(tip - hint, axis))
        if tip_along < 0.025 * img_h or tip_along > 0.22 * img_h:
            return False
        if float(np.dot(axis, finger)) < 0.45:
            return False
        finger_n = np.array([-finger[1], finger[0]], dtype=np.float64)
        if abs(float(np.dot(tip - hint, finger_n))) > 0.06 * img_h:
            return False
        return True

    @staticmethod
    def _find_upper_shaft_point(
        hint: np.ndarray,
        finger: np.ndarray,
        blackhat: np.ndarray,
        value: np.ndarray,
        sat: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> np.ndarray | None:
        """Locate the visible metallic shaft just above the grip (away from the tip)."""
        # Same cone as the old loop: angles, then sideways offsets, then steps up the shaft.
        base_ang = float(np.arctan2(-finger[1], -finger[0]))
        degrees = np.arange(-40, 41, 4)
        angles = base_ang + np.radians(degrees)
        back = np.stack((np.cos(angles), np.sin(angles)), axis=1)
        normal = np.stack((-back[:, 1], back[:, 0]), axis=1)
        offsets = np.arange(-0.02 * img_h, 0.02 * img_h + 0.1, 1.5)
        along = np.arange(0.045 * img_h, 0.14 * img_h, 2.0)
        points = (
            hint
            + back[:, None, None, :] * along[None, None, :, None]
            + normal[:, None, None, :] * offsets[None, :, None, None]
        )
        xy = np.rint(points).astype(np.int32)
        inside = (
            (xy[..., 0] >= 0)
            & (xy[..., 0] < img_w)
            & (xy[..., 1] >= 0)
            & (xy[..., 1] < img_h)
        )
        brightness = np.zeros(inside.shape, dtype=np.float64)
        saturation = np.zeros(inside.shape, dtype=np.float64)
        ridge = np.zeros(inside.shape, dtype=np.float64)
        if np.any(inside):
            ys = xy[..., 1][inside]
            xs = xy[..., 0][inside]
            brightness[inside] = value[ys, xs]
            saturation[inside] = sat[ys, xs]
            ridge[inside] = blackhat[ys, xs]
        sign = np.array([-4.0, 4.0], dtype=np.float64).reshape(1, 1, 1, 2, 1)
        neighbors = points[..., None, :] + normal[:, None, None, None, :] * sign
        nxy = np.rint(neighbors).astype(np.int32)
        neighbor_inside = (
            (nxy[..., 0] >= 0)
            & (nxy[..., 0] < img_w)
            & (nxy[..., 1] >= 0)
            & (nxy[..., 1] < img_h)
        )
        neighbor_ridge = np.zeros(neighbor_inside.shape, dtype=np.float64)
        if np.any(neighbor_inside):
            neighbor_ridge[neighbor_inside] = blackhat[
                nxy[..., 1][neighbor_inside], nxy[..., 0][neighbor_inside]
            ]
        neighbor_count = neighbor_inside.sum(axis=-1)
        side = np.zeros(inside.shape, dtype=np.float64)
        np.divide(
            neighbor_ridge.sum(axis=-1),
            neighbor_count,
            out=side,
            where=neighbor_count > 0,
        )
        thin = ridge - side
        kept = (
            inside
            & ~((saturation > 100) & (brightness > 110))
            & (thin >= 1.2)
            & (ridge >= 3.5)
        )
        metallic = np.ones(inside.shape, dtype=np.float64)
        metallic[(brightness >= 80) & (brightness <= 220) & (saturation < 90)] = 3.0
        metallic[kept & (brightness < 75) & (metallic == 1.0)] = 0.25
        contribution = np.where(kept, (ridge + thin * 3.0) * metallic, 0.0)
        kept_count = kept.sum(axis=-1)
        align = np.maximum(0.0, (-back) @ finger)
        scores = contribution.sum(axis=-1)
        scores *= 0.45 + 0.55 * align[:, None]
        scores *= 1.0 / (1.0 + np.abs(offsets) / (0.025 * img_h))
        scores = np.where(kept_count >= 3, scores, -1.0)
        if float(scores.max()) < 0.0:
            return None
        flat = int(np.argmax(scores))
        chosen_angle, chosen_offset = np.unravel_index(flat, scores.shape)
        kept_steps = np.flatnonzero(kept[chosen_angle, chosen_offset])
        middle = kept_steps[len(kept_steps) // 2]
        return points[chosen_angle, chosen_offset, middle].copy()

    @staticmethod
    def _tip_from_upper_shaft(
        upper: np.ndarray,
        finger: np.ndarray,
        hint: np.ndarray,
        value: np.ndarray,
        sat: np.ndarray,
        blackhat: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """From the metallic top, search near the finger direction for the dark tip."""
        toward_hint = hint - upper
        if float(np.linalg.norm(toward_hint)) > 1e-3:
            base = toward_hint / float(np.linalg.norm(toward_hint))
            if float(np.dot(base, finger)) < 0:
                base = -base
            base_ang = float(np.arctan2(base[1], base[0]))
        else:
            base_ang = float(np.arctan2(finger[1], finger[0]))
        best: tuple[float, np.ndarray, np.ndarray] | None = None
        for deg in range(-24, 25, 2):
            ang = base_ang + np.radians(deg)
            axis = np.array([np.cos(ang), np.sin(ang)], dtype=np.float64)
            if float(np.dot(axis, finger)) < 0.25:
                continue
            grip_along = float(np.dot(hint - upper, axis))
            if grip_along < 0.01 * img_h:
                continue
            # Ridge support between upper and the grip.
            support = 0.0
            support_n = 0
            for along in np.arange(2.0, grip_along, 2.0):
                point = upper + axis * along
                x, y = int(round(point[0])), int(round(point[1]))
                if not (0 <= x < img_w and 0 <= y < img_h):
                    continue
                support += float(blackhat[y, x])
                support_n += 1
            if support_n < 3 or support / support_n < 3.0:
                continue
            tip = None
            darkest = 1e9
            # Skip the finger zone; lock onto the dark tip past the grip.
            for along in np.arange(grip_along + 0.045 * img_h, grip_along + 0.22 * img_h, 1.0):
                point = upper + axis * along
                x, y = int(round(point[0])), int(round(point[1]))
                if not (0 <= x < img_w and 0 <= y < img_h):
                    break
                brightness = float(value[y, x])
                saturation = float(sat[y, x])
                bh = float(blackhat[y, x])
                if brightness > 155 and saturation < 50 and bh < 5:
                    break
                if saturation > 80:
                    continue
                if brightness < darkest and brightness < 135:
                    darkest = brightness
                    tip = point.copy()
            if tip is None:
                continue
            tip_along = float(np.dot(tip - hint, axis))
            if tip_along < 0.035 * img_h:
                continue
            grip_lat = abs(
                float(axis[0] * (hint[1] - tip[1]) - axis[1] * (hint[0] - tip[0]))
            )
            quality = (
                support / max(support_n, 1)
                + (135.0 - darkest) * 0.35
                - grip_lat * 0.35
                + float(np.dot(axis, finger)) * 12.0
                + tip_along / max(img_h, 1.0) * 30.0
            )
            if best is None or quality > best[0]:
                best = (quality, tip, axis)
        if best is None:
            return None, None
        return best[1], best[2]

    @staticmethod
    def _lateral_snap_whole_shaft(
        tip: np.ndarray,
        axis: np.ndarray,
        blackhat: np.ndarray,
        hint: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> np.ndarray:
        """Translate the shaft sideways onto the strongest ridge (keeps angle)."""
        normal = np.array([-axis[1], axis[0]], dtype=np.float64)
        grip_along = max(8.0, float(np.dot(hint - tip, -axis)))
        offsets = np.arange(-0.055 * img_h, 0.055 * img_h + 0.1, 1.0)
        along = np.arange(4.0, grip_along + 0.07 * img_h, 2.0)
        if offsets.size == 0:
            return tip
        samples = tip + normal * offsets[:, None, None] - axis * along[None, :, None]
        xy = np.rint(samples).astype(np.int32)
        inside = (
            (xy[..., 0] >= 0)
            & (xy[..., 0] < img_w)
            & (xy[..., 1] >= 0)
            & (xy[..., 1] < img_h)
        )
        ridge = np.zeros(inside.shape, dtype=np.float64)
        if np.any(inside):
            ridge[inside] = blackhat[xy[..., 1][inside], xy[..., 0][inside]]
        # The first strongest offset wins, matching the old left-to-right scan.
        return tip + normal * float(offsets[int(np.argmax(ridge.sum(axis=1)))])

    @staticmethod
    def _shaft_ridge_score(
        tip: np.ndarray,
        axis: np.ndarray,
        blackhat: np.ndarray,
        length: float,
        img_w: int,
        img_h: int,
    ) -> float:
        score = 0.0
        n = 0
        for along in np.arange(4.0, max(length, 8.0), 3.0):
            point = tip - axis * along
            x, y = int(round(point[0])), int(round(point[1]))
            if 0 <= x < img_w and 0 <= y < img_h:
                score += float(blackhat[y, x])
                n += 1
        return score / max(n, 1)

    @staticmethod
    def _tip_looks_like_graphite(
        tip: np.ndarray,
        value: np.ndarray,
        sat: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> bool:
        """Reject tips that landed on paper, skin, or a coloured ball."""
        x, y = int(round(tip[0])), int(round(tip[1]))
        if not (0 <= x < img_w and 0 <= y < img_h):
            return False
        patch_v = value[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3]
        patch_s = sat[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3]
        if patch_v.size == 0:
            return False
        brightness = float(patch_v.mean())
        saturation = float(patch_s.mean())
        darkest = float(patch_v.min())
        # The lead is only a few pixels. Paper beside it pulls the mean just
        # over 140, which used to drop the whole pen for that frame.
        if darkest > 115 or brightness > 170:
            return False
        if saturation > 70:
            return False
        # Coloured balls near the tip are a common false lock.
        if float(patch_s.max()) > 110:
            return False
        return True

    @staticmethod
    def _tip_has_shaft_support(
        tip: np.ndarray,
        axis: np.ndarray,
        blackhat: np.ndarray,
        hint: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> bool:
        """True when a black-hat ridge continues from the tip toward the grip."""
        grip_along = float(np.dot(hint - tip, -axis))
        if grip_along < 0.02 * img_h:
            return False
        hits = 0
        samples = 0
        for along in np.arange(4.0, grip_along, 3.0):
            point = tip - axis * along
            x, y = int(round(point[0])), int(round(point[1]))
            if not (0 <= x < img_w and 0 <= y < img_h):
                continue
            samples += 1
            if float(blackhat[y, x]) >= 4.0:
                hits += 1
        return samples >= 3 and hits / samples >= 0.30

    def _hough_shaft_segment(
        self,
        gray: np.ndarray,
        blackhat: np.ndarray,
        hint: np.ndarray,
        finger: np.ndarray,
        value: np.ndarray,
        sat: np.ndarray,
        img_h: int,
        img_w: int,
        prev_tip: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Recover tip/back from parallel pen edges near the grip."""
        x0 = int(max(0, hint[0] - 0.15 * img_h))
        x1 = int(min(img_w, hint[0] + 0.15 * img_h))
        y0 = int(max(0, hint[1] - 0.32 * img_h))
        y1 = int(min(img_h, hint[1] + 0.28 * img_h))
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            return None
        edges = cv2.Canny(cv2.GaussianBlur(roi, (5, 5), 0), 35, 120)
        edges = cv2.bitwise_and(edges, (blackhat[y0:y1, x0:x1] > 4).astype(np.uint8) * 255)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            16,
            minLineLength=int(0.09 * img_h),
            maxLineGap=16,
        )
        if lines is None:
            return None

        candidates: list[tuple[float, float, np.ndarray, np.ndarray, np.ndarray]] = []
        for x1l, y1l, x2l, y2l in lines.reshape(-1, 4):
            p1 = np.array([x1l + x0, y1l + y0], dtype=np.float64)
            p2 = np.array([x2l + x0, y2l + y0], dtype=np.float64)
            vec = p2 - p1
            length = float(np.linalg.norm(vec))
            if length < 1.0:
                continue
            direction = vec / length
            if float(np.dot(direction, finger)) < 0:
                direction = -direction
                p1, p2 = p2, p1
            align = float(np.dot(direction, finger))
            if align < 0.35:
                continue
            proj = float(np.clip(np.dot(hint - p1, direction), 0.0, length))
            dist = float(np.linalg.norm(hint - (p1 + direction * proj)))
            if dist > 0.038 * img_h:
                continue
            candidates.append((length, dist, align, p1, p2, direction))
        if not candidates:
            return None

        # Prefer long, finger-aligned shafts that stay close to the grip.
        candidates.sort(
            key=lambda item: item[0] * (item[2] ** 2) / (1.0 + 4.0 * item[1]),
            reverse=True,
        )
        base = candidates[0]
        points = [base[3], base[4]]
        for cand in candidates[1:6]:
            if abs(float(np.dot(cand[5], base[5]))) < 0.95:
                continue
            mid = 0.5 * (cand[3] + cand[4])
            offset = abs(
                float(base[5][0] * (mid[1] - base[3][1]) - base[5][1] * (mid[0] - base[3][0]))
            )
            if offset > 0.035 * img_h:
                continue
            points.extend([cand[3], cand[4]])

        pts = np.asarray(points, dtype=np.float64)
        vx, vy, cx, cy = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        axis = np.array([vx, vy], dtype=np.float64)
        if float(np.dot(axis, finger)) < 0:
            axis = -axis
        center = np.array([cx, cy], dtype=np.float64)
        proj = (pts - center) @ axis
        end_hi = center + axis * float(proj.max())
        end_lo = center + axis * float(proj.min())

        # Tip is the end past the fingertips (writing side), not the eraser.
        if float(np.dot(end_hi - hint, finger)) >= float(np.dot(end_lo - hint, finger)):
            tip, back = end_hi, end_lo
        else:
            tip, back = end_lo, end_hi
            axis = -axis

        # If one end is clearly darker graphite, prefer that as tip.
        def end_value(point: np.ndarray) -> float:
            x, y = int(round(point[0])), int(round(point[1]))
            patch = value[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3]
            return float(patch.mean()) if patch.size else 255.0

        tip_v, back_v = end_value(tip), end_value(back)
        if back_v + 20 < tip_v and float(np.dot(back - hint, finger)) > 0:
            tip, back = back, tip
            axis = -axis
        if prev_tip is not None and np.linalg.norm(tip - prev_tip) > np.linalg.norm(
            back - prev_tip
        ) + 0.04 * img_h:
            # Only flip for continuity when the other end is also past the fingers.
            if float(np.dot(back - hint, finger)) > float(np.dot(tip - hint, finger)):
                tip, back = back, tip
                axis = -axis

        tip = self._snap_tip_before_paper(tip, axis, value, sat, img_w, img_h)
        tip = self._recenter_tip_on_dark(tip, value, sat, img_w, img_h, axis=axis)
        tx, ty = int(round(tip[0])), int(round(tip[1]))
        if not (0 <= tx < img_w and 0 <= ty < img_h):
            return None
        if float(value[ty, tx]) > 135 and float(sat[ty, tx]) < 55:
            return None
        if float(np.dot(tip - hint, axis)) < 0.02 * img_h:
            return None
        # Keep back at least slightly past the grip; appearance clamp shortens later.
        grip_along = float(np.dot(hint - tip, -axis))
        back_along = float(np.dot(back - tip, -axis))
        if back_along < grip_along + 0.03 * img_h:
            back = tip - axis * (grip_along + 0.06 * img_h)
        return tip, back, axis

    def _detect_relative_shaft_rays(
        self,
        small,
        scale: float,
        hint: np.ndarray,
        finger: np.ndarray,
        gray: np.ndarray,
        blackhat: np.ndarray,
        sat: np.ndarray,
        value: np.ndarray,
        img_h: int,
        img_w: int,
    ) -> PenDetectionResult | None:
        """Ray fallback when edge pairing does not find a shaft."""
        blur = cv2.GaussianBlur(gray.astype(np.float32), (31, 31), 0)
        relative = (blur - gray.astype(np.float32)).clip(0, 255)
        score = relative * 0.35 + blackhat * 0.65
        finger_ang = float(np.arctan2(finger[1], finger[0]))
        prev_axis = None
        if self._last is not None and not self._last.from_hand:
            prev_axis = np.array(self._last.axis, dtype=np.float64)
            norm = float(np.linalg.norm(prev_axis))
            prev_axis = prev_axis / norm if norm > 1e-6 else None

        center_ang = finger_ang
        search_span = 48
        if prev_axis is not None:
            center_ang = float(np.arctan2(prev_axis[1], prev_axis[0]))
            search_span = 24

        best: tuple[float, np.ndarray, np.ndarray, np.ndarray] | None = None
        for deg in range(-search_span, search_span + 1, 2):
            axis = np.array(
                [np.cos(center_ang + np.radians(deg)), np.sin(center_ang + np.radians(deg))],
                dtype=np.float64,
            )
            if float(np.dot(axis, finger)) < 0.45:
                continue
            tip_vals, tip_pts, tip_value, tip_sat = self._sample_shaft_ray(
                hint, axis, 0.30 * img_h, 1.0, score, sat, value, img_w, img_h
            )
            if len(tip_vals) < 12:
                continue
            tip_run, tip_end, tip_mean = self._shaft_run_until_paper(
                tip_vals, tip_value, tip_sat
            )
            if tip_run < 11:
                continue
            tip = tip_pts[tip_end].copy()
            tip = self._snap_tip_before_paper(tip, axis, value, sat, img_w, img_h)
            tip = self._recenter_tip_on_dark(tip, value, sat, img_w, img_h, axis=axis)
            tx, ty = int(round(tip[0])), int(round(tip[1]))
            if not (0 <= tx < img_w and 0 <= ty < img_h):
                continue
            if float(value[ty, tx]) > 130:
                continue
            if float(np.dot(tip - hint, axis)) < 0.028 * img_h:
                continue
            back = self._walk_shaft_from_tip(
                tip, axis, hint, score, sat, value, img_w, img_h
            )
            quality = tip_run * tip_mean * (0.5 + float(np.dot(axis, finger)))
            if best is None or quality > best[0]:
                best = (quality, tip, back, axis)
        if best is None:
            return None
        _, tip, _back, axis = best
        visible_back = self._visible_upper_end(
            tip, axis, blackhat, value, sat, img_w, img_h, hint
        )
        _tip, back, axis = self._commit_upper_end(tip, axis, visible_back, img_h)
        tip_full = tip / scale
        back_full = back / scale
        return PenDetectionResult(
            tip=(float(tip_full[0]), float(tip_full[1])),
            grip=(float(back_full[0]), float(back_full[1])),
            axis=(float(axis[0]), float(axis[1])),
            confidence=0.52,
            from_hand=False,
        )

    @staticmethod
    def _sample_shaft_ray(
        origin: np.ndarray,
        axis: np.ndarray,
        max_dist: float,
        step: float,
        score: np.ndarray,
        sat: np.ndarray,
        value: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
        """Score a ray for a thin, locally-dark shaft; skin-coloured pixels score zero."""
        along = np.arange(1.0, max_dist, step)
        empty = (
            np.zeros(0, dtype=np.float64),
            [],
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
        if along.size == 0:
            return empty
        points = origin + axis * along[:, None]
        xy = np.rint(points).astype(np.int32)
        inside = (
            (xy[:, 0] >= 0) & (xy[:, 0] < img_w) & (xy[:, 1] >= 0) & (xy[:, 1] < img_h)
        )
        if not np.all(inside):
            stop = int(np.argmax(~inside))
            if stop == 0:
                return empty
            points = points[:stop]
            xy = xy[:stop]
        brightness = value[xy[:, 1], xy[:, 0]].astype(np.float64, copy=False)
        saturation = sat[xy[:, 1], xy[:, 0]].astype(np.float64, copy=False)
        center = score[xy[:, 1], xy[:, 0]].astype(np.float64, copy=False)
        normal = np.array([-axis[1], axis[0]], dtype=np.float64)
        pair = (np.array([3.0, 5.0])[:, None] * np.array([-1.0, 1.0])[None, :]).reshape(-1)
        neighbors = points[:, None, :] + normal * pair.reshape(1, -1, 1)
        nxy = np.rint(neighbors).astype(np.int32)
        neighbor_inside = (
            (nxy[..., 0] >= 0)
            & (nxy[..., 0] < img_w)
            & (nxy[..., 1] >= 0)
            & (nxy[..., 1] < img_h)
        )
        gathered = np.zeros(neighbor_inside.shape, dtype=np.float64)
        if np.any(neighbor_inside):
            gathered[neighbor_inside] = score[
                nxy[..., 1][neighbor_inside], nxy[..., 0][neighbor_inside]
            ]
        neighbor_count = neighbor_inside.sum(axis=1)
        side = np.zeros(points.shape[0], dtype=np.float64)
        np.divide(gathered.sum(axis=1), neighbor_count, out=side, where=neighbor_count > 0)
        vals = center + 0.8 * np.maximum(0.0, center - side)
        vals[(brightness < 105) & (saturation < 70)] += 25.0
        vals[(saturation > 95) & (brightness > 105)] = 0.0
        return (
            vals,
            [points[i] for i in range(points.shape[0])],
            brightness,
            saturation,
        )

    @staticmethod
    def _shaft_run_until_paper(
        vals: np.ndarray,
        value: np.ndarray,
        sat: np.ndarray,
    ) -> tuple[int, int, float]:
        """Shaft run that starts at the grip and stops when the ray hits blank paper."""
        if len(vals) == 0:
            return 0, 0, 0.0
        positive = vals[vals > 0]
        threshold = max(6.0, float(np.median(positive)) * 0.55) if len(positive) else 6.0
        run_start = -1
        run_end = -1
        count = 0
        for i, score in enumerate(vals):
            is_paper = value[i] > 152 and sat[i] < 35 and score < 8
            is_pen = score >= threshold or (value[i] < 125 and sat[i] < 85 and score > 4)
            if run_start < 0:
                if is_pen and i <= 14:
                    run_start = i
                    run_end = i
                    count = 1
                continue
            if is_paper and count >= 8:
                break
            if is_pen:
                run_end = i
                count += 1
                continue
            if i + 1 < len(vals) and vals[i + 1] >= threshold:
                continue
            if count >= 10:
                break
            if i <= 14:
                run_start = -1
                count = 0
            else:
                break
        if count < 1 or run_end < 0:
            return 0, 0, 0.0
        mean_score = float(vals[run_start : run_end + 1].mean())
        return count, run_end, mean_score

    @staticmethod
    def _snap_tip_before_paper(
        tip: np.ndarray,
        axis: np.ndarray,
        value: np.ndarray,
        sat: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> np.ndarray:
        """Move onto the darkest tip pixel, but never continue onto bright paper."""
        apex = tip.copy()
        darkest = 1e9
        x0, y0 = int(round(tip[0])), int(round(tip[1]))
        if 0 <= x0 < img_w and 0 <= y0 < img_h:
            darkest = float(value[y0, x0])
        for along in np.arange(0.0, 18.0, 0.5):
            point = tip + axis * along
            x, y = int(round(point[0])), int(round(point[1]))
            if not (0 <= x < img_w and 0 <= y < img_h):
                break
            brightness = float(value[y, x])
            saturation = float(sat[y, x])
            if brightness > 145 and saturation < 50:
                break
            if brightness <= darkest + 8 and brightness < 125:
                apex = point.copy()
                darkest = min(darkest, brightness)
        return apex

    @staticmethod
    def _pull_tip_onto_lead(
        tip: np.ndarray,
        axis: np.ndarray,
        value: np.ndarray,
        sat: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> np.ndarray:
        """If the tip landed on blank paper past the lead, step back onto it."""
        x0, y0 = int(round(tip[0])), int(round(tip[1]))
        if not (0 <= x0 < img_w and 0 <= y0 < img_h):
            return tip
        if float(value[y0, x0]) < 135:
            return tip
        for along in np.arange(1.0, 36.0, 0.5):
            point = tip - axis * along
            x, y = int(round(point[0])), int(round(point[1]))
            if not (0 <= x < img_w and 0 <= y < img_h):
                break
            brightness = float(value[y, x])
            saturation = float(sat[y, x])
            if brightness < 125 and saturation < 100:
                return point
        return tip

    @staticmethod
    def _recenter_tip_on_dark(
        tip: np.ndarray,
        value: np.ndarray,
        sat: np.ndarray,
        img_w: int,
        img_h: int,
        axis: np.ndarray | None = None,
        radius: int = 5,
    ) -> np.ndarray:
        """Snap onto the dark tip apex (darkest, furthest along the shaft)."""
        x0, y0 = int(round(tip[0])), int(round(tip[1]))
        x1, y1 = max(0, x0 - radius), max(0, y0 - radius)
        x2, y2 = min(img_w, x0 + radius + 1), min(img_h, y0 + radius + 1)
        patch_v = value[y1:y2, x1:x2]
        patch_s = sat[y1:y2, x1:x2]
        if patch_v.size == 0:
            return tip
        mask = (patch_v < 125) & (patch_s < 90)
        if not np.any(mask):
            return tip
        ys, xs = np.nonzero(mask)
        pixels = np.stack([xs + x1, ys + y1], axis=1).astype(np.float64)
        darkness = patch_v[ys, xs].astype(np.float64)
        if axis is None:
            order = np.argsort(darkness)
            return pixels[order[: max(1, min(5, len(order)))]].mean(axis=0)
        along = (pixels - tip) @ axis
        # Apex = furthest along among the darkest tip pixels only.
        dark_enough = darkness <= float(np.min(darkness)) + 15
        candidates = pixels[dark_enough]
        along_c = along[dark_enough]
        if len(candidates) == 0:
            return tip
        return candidates[int(np.argmax(along_c))]

    def _walk_shaft_from_tip(
        self,
        tip: np.ndarray,
        tip_axis: np.ndarray,
        hint: np.ndarray,
        score: np.ndarray,
        sat: np.ndarray,
        value: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> np.ndarray:
        """Place the back end past the grip along the shaft axis.

        Length is grip-relative (stable) and only nudged by appearance so it does
        not depend on ball/pen millimetre priors.
        """
        back_axis = -tip_axis
        grip_along = max(0.02 * img_h, float(np.dot(hint - tip, back_axis)))
        # Default: cover tip→grip plus a shaft stub above the fingers.
        target = grip_along + 0.11 * img_h
        target = min(target, 0.26 * img_h)
        # Nudge to the furthest still-pen-like point near that target.
        best = tip + back_axis * target
        for along in np.arange(grip_along + 0.04 * img_h, target + 0.04 * img_h, 1.0):
            point = tip + back_axis * along
            x, y = int(round(point[0])), int(round(point[1]))
            if not (0 <= x < img_w and 0 <= y < img_h):
                break
            brightness = float(value[y, x])
            saturation = float(sat[y, x])
            center = float(score[y, x])
            if brightness > 155 and saturation < 40 and center < 6:
                break
            if center >= 5 or (brightness < 140 and saturation < 85):
                best = point.copy()
            elif along > target:
                break
        return best

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
