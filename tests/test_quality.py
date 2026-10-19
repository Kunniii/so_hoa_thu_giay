"""Tests for imaging/quality.py: quality assurance, blur scoring, and anomaly detection."""

import cv2
import numpy as np

from letter_worker.imaging.quality import evaluate_image_quality


def test_quality_passes_for_normal_letter(synthetic_letter_image):
    qa = evaluate_image_quality(synthetic_letter_image)
    assert qa.status in ("passed", "warning")
    assert qa.width == 400
    assert qa.height == 600
    assert qa.blur_score > 35.0
    assert qa.ink_coverage > 0.005


def test_quality_rejects_solid_white():
    white = np.full((300, 300), 255, dtype=np.uint8)
    qa = evaluate_image_quality(white)
    assert qa.status == "rejected"
    assert any("WHITE" in r for r in qa.reasons)


def test_quality_rejects_solid_black():
    black = np.full((300, 300), 0, dtype=np.uint8)
    qa = evaluate_image_quality(black)
    assert qa.status == "rejected"
    assert any("BLACK" in r for r in qa.reasons)


def test_quality_rejects_tiny_dimensions():
    tiny = np.full((50, 50), 200, dtype=np.uint8)
    qa = evaluate_image_quality(tiny, min_dimension=100)
    assert qa.status == "rejected"
    assert any("DIMENSIONS_TOO_SMALL" in r for r in qa.reasons)


def test_quality_warns_on_blurry_image(synthetic_letter_image):
    # Heavily blur synthetic letter
    blurry = cv2.GaussianBlur(synthetic_letter_image, (45, 45), 20.0)
    qa = evaluate_image_quality(blurry, min_blur_score=50.0)
    assert qa.status == "warning"
    assert any("BLUR" in r for r in qa.reasons)
