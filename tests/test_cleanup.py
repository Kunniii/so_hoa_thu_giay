"""Tests for imaging/cleanup.py: Pure B&W, Natural, and Vintage restoration presets."""

import numpy as np

from letter_worker.imaging.cleanup import (
    cleanup_natural,
    cleanup_pure_bw,
    cleanup_vintage,
    restore_image,
)
from letter_worker.schemas import RestoreSettings


def test_cleanup_pure_bw(synthetic_letter_image):
    settings = RestoreSettings(preset="pure_bw", threshold_method="adaptive")
    result = cleanup_pure_bw(synthetic_letter_image, settings)

    assert result.ndim == 2
    # Check that pixels are strictly binary: only 0 or 255
    unique_vals = set(np.unique(result))
    assert unique_vals.issubset({0, 255})
    # Paper background should be predominantly 255
    assert np.mean(result) > 200.0


def test_cleanup_pure_bw_otsu(synthetic_letter_image):
    settings = RestoreSettings(preset="pure_bw", threshold_method="otsu")
    result = cleanup_pure_bw(synthetic_letter_image, settings)
    assert result.ndim == 2
    assert set(np.unique(result)).issubset({0, 255})


def test_cleanup_natural(synthetic_letter_image):
    settings = RestoreSettings(preset="natural", denoise="light")
    result = cleanup_natural(synthetic_letter_image, settings)

    assert result.ndim == 2
    # Natural preset keeps smooth grayscale shades (not just 0 and 255)
    assert len(np.unique(result)) > 20
    # Paper should be bright
    assert np.mean(result) > 190.0


def test_cleanup_vintage(synthetic_letter_image):
    settings = RestoreSettings(preset="vintage")
    result = cleanup_vintage(synthetic_letter_image, settings)

    assert result.ndim == 3
    assert result.shape[2] == 3
    # Check vintage color signature: Red > Blue (warm/sepia tint)
    mean_r = np.mean(result[:, :, 0])
    mean_b = np.mean(result[:, :, 2])
    assert mean_r > mean_b


def test_restore_image_dispatcher(synthetic_letter_image):
    for preset in ["pure_bw", "natural", "vintage"]:
        settings = RestoreSettings(preset=preset)
        out = restore_image(synthetic_letter_image, settings)
        assert out is not None
