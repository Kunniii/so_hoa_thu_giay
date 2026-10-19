"""Tests for imaging/geometry.py: point ordering, validation, and rectification."""

import numpy as np
import pytest

from letter_worker.errors import InvalidCornersError
from letter_worker.imaging.geometry import (
    estimate_3d_aspect_ratio,
    estimate_rectified_dimensions,
    order_corners,
    rectify_perspective,
    validate_quadrilateral,
)


def test_order_corners_standard():
    # Points in random order: BR, TL, BL, TR
    pts = np.array([[300.0, 400.0], [50.0, 60.0], [60.0, 410.0], [310.0, 50.0]])
    ordered = order_corners(pts)

    tl, tr, br, bl = ordered
    assert np.allclose(tl, [50.0, 60.0])
    assert np.allclose(tr, [310.0, 50.0])
    assert np.allclose(br, [300.0, 400.0])
    assert np.allclose(bl, [60.0, 410.0])


def test_validate_quadrilateral_valid():
    quad = np.array([[10, 10], [200, 15], [195, 300], [15, 290]], dtype=np.float32)
    # Should not raise
    validate_quadrilateral(quad, image_shape=(500, 500))


def test_validate_quadrilateral_invalid_point_count():
    with pytest.raises(InvalidCornersError, match="must have 4 points"):
        validate_quadrilateral(np.array([[0, 0], [10, 0], [10, 10]]))


def test_validate_quadrilateral_concave():
    # Concave quadrilateral (arrowhead / boomerang)
    concave = np.array([[10, 10], [200, 10], [100, 100], [200, 200]], dtype=np.float32)
    with pytest.raises(InvalidCornersError, match="not convex"):
        validate_quadrilateral(concave)


def test_validate_quadrilateral_self_intersecting_bow_tie():
    # Bow-tie polygon (self-intersecting)
    bowtie = np.array([[10, 10], [200, 200], [200, 10], [10, 200]], dtype=np.float32)
    with pytest.raises(InvalidCornersError):
        validate_quadrilateral(bowtie)


def test_validate_quadrilateral_duplicate_points():
    dup = np.array([[10, 10], [10, 10], [100, 100], [0, 100]], dtype=np.float32)
    with pytest.raises(InvalidCornersError, match="almost identical"):
        validate_quadrilateral(dup)


def test_estimate_rectified_dimensions():
    # 300 wide by 400 high roughly in free mode
    corners = np.array([[0, 0], [300, 0], [300, 400], [0, 400]], dtype=np.float32)
    w, h = estimate_rectified_dimensions(corners, aspect_ratio_mode="free")
    assert w == 300
    assert h == 400

    # A4 preset mode
    w_a4, h_a4 = estimate_rectified_dimensions(corners, aspect_ratio_mode="a4")
    assert abs((w_a4 / h_a4) - (1.0 / 1.41421356)) < 0.01

    # Notebook preset mode (1:1.35)
    w_nb, h_nb = estimate_rectified_dimensions(corners, aspect_ratio_mode="notebook")
    assert abs((w_nb / h_nb) - (1.0 / 1.35)) < 0.01


def test_estimate_3d_aspect_ratio_on_tilted_perspective():
    # A standard A4 document (1000 x 1414, ratio = 0.707)
    # Photographed tilted: top edge is smaller due to perspective depth
    # Corners: TL, TR, BR, BL
    tilted_corners = np.array(
        [
            [350.0, 200.0],   # TL (farther away, narrower)
            [1650.0, 200.0],  # TR (width = 1300)
            [1800.0, 1900.0], # BR (width = 1600)
            [200.0, 1900.0],  # BL
        ],
        dtype=np.float32,
    )
    # Estimate 3D aspect ratio
    ratio = estimate_3d_aspect_ratio(tilted_corners, image_shape=(2500, 2000))
    assert ratio is not None
    # Should recover true physical aspect ratio (around 0.70 - 0.73), not distorted wide
    assert 0.65 <= ratio <= 0.78


def test_rectify_perspective_transforms_accurately(synthetic_desk_photo):
    photo, gt_corners = synthetic_desk_photo
    rectified, (w, h) = rectify_perspective(photo, gt_corners)

    assert rectified is not None
    assert rectified.ndim == 3
    assert rectified.shape[0] == h
    assert rectified.shape[1] == w
    assert 550 <= w <= 650
    assert 500 <= h <= 600
