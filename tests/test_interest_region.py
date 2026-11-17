"""Region shapes that limit where tracking runs."""

import numpy as np
import pytest

from pen_tracking.interest_region import InterestRegion


def test_square_drag_keeps_equal_sides() -> None:
    region = InterestRegion.from_drag("square", 10, 20, 40, 30)
    assert region.x1 - region.x0 == region.y1 - region.y0 == 30


def test_circle_drag_keeps_equal_sides() -> None:
    region = InterestRegion.from_drag("circle", 0, 0, 10, 50)
    assert abs(region.x1 - region.x0) == abs(region.y1 - region.y0) == 50


def test_rectangle_contains_its_box_and_not_the_outside() -> None:
    region = InterestRegion("rectangle", 10, 10, 50, 40)
    assert region.contains(30, 20)
    assert not region.contains(0, 0)


def test_ellipse_excludes_the_corner_of_its_box() -> None:
    region = InterestRegion("ellipse", 0, 0, 100, 40)
    assert region.contains(50, 20)
    assert not region.contains(1, 1)


def test_semicircle_bulges_to_the_right_of_the_drag() -> None:
    # Dragging left to right puts the arc downward in image coordinates.
    region = InterestRegion("semicircle", 0, 10, 20, 10)
    assert region.contains(10, 16)
    assert not region.contains(10, 4)


def test_flipped_semicircle_bulges_the_other_way() -> None:
    region = InterestRegion("semicircle", 0, 10, 20, 10).flipped()
    assert region.contains(10, 4)
    assert not region.contains(10, 16)


def test_crop_is_smaller_than_the_frame_and_whites_out_the_ellipse_corner() -> None:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    region = InterestRegion("ellipse", 10, 10, 70, 50)
    cropped, origin_x, origin_y = region.crop(frame)
    assert cropped.shape[0] < frame.shape[0]
    assert cropped.shape[1] < frame.shape[1]
    assert origin_x == 10
    assert origin_y == 10
    assert tuple(cropped[0, 0]) == (255, 255, 255)
    assert tuple(cropped[cropped.shape[0] // 2, cropped.shape[1] // 2]) == (0, 0, 0)


def test_rectangle_crop_keeps_the_pixels() -> None:
    frame = np.full((40, 40, 3), 7, dtype=np.uint8)
    region = InterestRegion("rectangle", 5, 6, 15, 18)
    cropped, origin_x, origin_y = region.crop(frame)
    assert (origin_x, origin_y) == (5, 6)
    assert cropped.shape[:2] == (12, 10)
    assert int(cropped[0, 0, 0]) == 7


def test_region_rejects_a_tiny_drag() -> None:
    with pytest.raises(ValueError):
        InterestRegion("rectangle", 0, 0, 3, 3)
