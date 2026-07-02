"""Shared pytest fixtures."""

import cv2
import mediapipe as mp
import pytest
from mediapipe.tasks.python import BaseOptions, vision

from pen_tracking.paths import DEFAULT_MODEL, DEFAULT_VIDEO


@pytest.fixture(scope="session")
def hand_landmarks_from_video():
    cap = cv2.VideoCapture(str(DEFAULT_VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 40)
    ret, frame = cap.read()
    cap.release()
    assert ret

    height, width = frame.shape[:2]
    small = cv2.resize(frame, (int(width * 720 / height), 720))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(DEFAULT_MODEL)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    landmarker.close()
    assert result.hand_landmarks
    return result.hand_landmarks[0]
