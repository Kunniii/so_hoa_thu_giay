"""Tests for the high-level Python API: process_letter and package exports."""

import numpy as np
import pytest

import letter_worker
from letter_worker import (
    InvalidCornersError,
    InvalidImageError,
    LetterProcessor,
    RestoreSettings,
    process_letter,
)


def test_package_exports():
    """Verify top-level package exposes all expected public symbols and version."""
    assert letter_worker.__version__ == "0.2.0"
    assert callable(process_letter)
    assert issubclass(LetterProcessor, object)
    assert issubclass(RestoreSettings, object)


def test_process_letter_in_memory_numpy(synthetic_letter_image):
    """Test process_letter directly with a NumPy array and manual corners."""
    corners = [
        [0.0, 0.0],
        [400.0, 0.0],
        [400.0, 600.0],
        [0.0, 600.0],
    ]
    restored, meta = process_letter(
        synthetic_letter_image,
        corners=corners,
        preset="vintage",
        aspect_ratio="auto",
        paper_warmth=0.80,
    )

    assert isinstance(restored, np.ndarray)
    assert restored.ndim == 3
    assert restored.shape[2] == 3
    assert meta["settings"]["preset"] == "vintage"
    assert meta["settings"]["paper_warmth"] == 0.80
    assert "quality" in meta
    assert meta["quality"]["status"] == "passed"


def test_process_letter_presets(synthetic_letter_image):
    """Test process_letter across all supported aesthetic presets."""
    corners = [[0, 0], [400, 0], [400, 600], [0, 600]]

    # Pure B&W
    bw, meta_bw = process_letter(synthetic_letter_image, corners=corners, preset="pure_bw")
    assert bw.ndim == 2
    assert set(np.unique(bw)).issubset({0, 255})

    # Natural
    nat, meta_nat = process_letter(synthetic_letter_image, corners=corners, preset="natural")
    assert nat.ndim == 2
    assert len(np.unique(nat)) > 10

    # Vintage
    vint, meta_vint = process_letter(synthetic_letter_image, corners=corners, preset="vintage")
    assert vint.ndim == 3
    # Check warm tone (Red > Blue)
    assert np.mean(vint[:, :, 0]) > np.mean(vint[:, :, 2])


def test_process_letter_file_save(synthetic_letter_image, tmp_path):
    """Test process_letter with output_path writes file atomically."""
    input_file = tmp_path / "raw_letter.png"
    output_file = tmp_path / "restored.png"

    from PIL import Image

    Image.fromarray(synthetic_letter_image).save(input_file)

    corners = [[0, 0], [400, 0], [400, 600], [0, 600]]
    restored, meta = process_letter(
        input_file,
        corners=corners,
        output_path=output_file,
        preset="vintage",
    )

    assert output_file.exists()
    assert output_file.stat().st_size > 0
    saved = np.array(Image.open(output_file))
    assert np.array_equal(restored, saved)


def test_process_letter_aspect_ratio_modes(synthetic_letter_image):
    """Test aspect ratio constraints in process_letter."""
    corners = [[0, 0], [400, 0], [400, 600], [0, 600]]

    # A4 mode
    res_a4, meta_a4 = process_letter(synthetic_letter_image, corners=corners, aspect_ratio="a4")
    dims = meta_a4["output_dimensions"]
    ratio = dims["width"] / dims["height"]
    assert abs(ratio - (1.0 / 1.41421356)) < 0.01

    # Notebook mode
    res_nb, meta_nb = process_letter(
        synthetic_letter_image, corners=corners, aspect_ratio="notebook"
    )
    dims_nb = meta_nb["output_dimensions"]
    ratio_nb = dims_nb["width"] / dims_nb["height"]
    assert abs(ratio_nb - (1.0 / 1.35)) < 0.01


def test_process_letter_invalid_inputs():
    """Test error handling on invalid image and corners."""
    # Invalid array shape
    with pytest.raises(InvalidImageError):
        process_letter(np.zeros((10, 10, 5), dtype=np.uint8))

    # Invalid corners count
    with pytest.raises(Exception):
        process_letter(
            np.full((300, 300, 3), 240, dtype=np.uint8),
            corners=[[0, 0], [100, 0]],  # Only 2 points
        )

    # Concave / non-convex corners
    with pytest.raises(InvalidCornersError):
        process_letter(
            np.full((300, 300, 3), 240, dtype=np.uint8),
            corners=[[10, 10], [200, 10], [100, 100], [200, 200]],
        )
