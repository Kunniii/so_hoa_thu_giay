"""Tests for imaging/illumination.py: background estimation and shadow removal."""

import numpy as np

from letter_worker.imaging.illumination import (
    correct_illumination_gray,
    correct_illumination_rgb,
    estimate_background,
)


def test_estimate_background_produces_smooth_field():
    gray = np.full((300, 300), 200, dtype=np.uint8)
    # Add high frequency noise
    noise = np.random.randint(-20, 20, (300, 300), dtype=np.int16)
    noisy = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    bg = estimate_background(noisy)
    assert bg.shape == (300, 300)
    # Background should be smooth with small variance compared to noisy
    assert np.std(bg) < np.std(noisy)


def test_correct_illumination_removes_linear_shadow():
    h, w = 300, 400
    # Simulate a page with a strong shadow gradient: left is dark (120), right is bright (240)
    gradient = np.tile(np.linspace(120, 240, w), (h, 1)).astype(np.uint8)

    corrected = correct_illumination_gray(gradient, target_paper_val=220.0, strength=1.0)

    # After correction, left and right mean brightness should be much closer
    left_mean = float(np.mean(corrected[:, :50]))
    right_mean = float(np.mean(corrected[:, -50:]))
    assert abs(left_mean - right_mean) < 30.0


def test_correct_illumination_rgb():
    rgb = np.full((200, 200, 3), 180, dtype=np.uint8)
    corrected = correct_illumination_rgb(rgb, target_paper_val=230.0, strength=1.0)
    assert corrected.shape == (200, 200, 3)
    assert np.mean(corrected) > np.mean(rgb)
