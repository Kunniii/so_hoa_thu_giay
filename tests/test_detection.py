"""Tests for imaging/detection.py: contour finding, quad ranking, and confidence scoring."""

import numpy as np

from letter_worker.imaging.detection import detect_paper_corners


def test_detect_paper_corners_on_synthetic_photo(synthetic_desk_photo):
    photo, gt_corners = synthetic_desk_photo
    result = detect_paper_corners(photo)

    assert result.corners is not None
    assert result.confidence >= 0.70
    assert len(result.candidates) >= 1
    assert result.corners.shape == (4, 2)

    # Verify detected corners roughly match ground truth
    # Corners are ordered: TL, TR, BR, BL
    det_corners = result.corners
    # Mean Euclidean distance to ground truth should be small (< 25 px)
    diff = np.mean(np.linalg.norm(det_corners - gt_corners, axis=1))
    assert diff < 25.0


def test_detect_paper_corners_blank_canvas():
    # Completely uniform image with no paper
    blank = np.full((500, 500, 3), 128, dtype=np.uint8)
    result = detect_paper_corners(blank)

    # Should have no valid paper or very low confidence
    assert result.confidence < 0.50
