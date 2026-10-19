"""Tests for imaging/loader.py: validation, bounds checks, and EXIF normalization."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from letter_worker.errors import ImageTooLargeError, InvalidImageError
from letter_worker.imaging.loader import load_and_validate_image


def test_load_valid_image(tmp_path: Path):
    img_path = tmp_path / "valid.png"
    arr = np.full((100, 100, 3), 180, dtype=np.uint8)
    Image.fromarray(arr).save(img_path)

    loaded = load_and_validate_image(img_path)
    assert loaded.width == 100
    assert loaded.height == 100
    assert loaded.image_rgb.shape == (100, 100, 3)
    assert loaded.source_format in ("PNG", "PNG")


def test_load_nonexistent_file(tmp_path: Path):
    missing_path = tmp_path / "does_not_exist.jpg"
    with pytest.raises(InvalidImageError, match="does not exist"):
        load_and_validate_image(missing_path)


def test_load_empty_file(tmp_path: Path):
    empty_path = tmp_path / "empty.jpg"
    empty_path.write_bytes(b"")
    with pytest.raises(InvalidImageError, match="empty"):
        load_and_validate_image(empty_path)


def test_load_corrupt_file(tmp_path: Path):
    corrupt_path = tmp_path / "corrupt.jpg"
    corrupt_path.write_bytes(b"NOT_A_REAL_IMAGE_DATA_12345")
    with pytest.raises(InvalidImageError):
        load_and_validate_image(corrupt_path)


def test_load_file_exceeds_size_limit(tmp_path: Path):
    large_file = tmp_path / "large.png"
    arr = np.full((50, 50, 3), 100, dtype=np.uint8)
    Image.fromarray(arr).save(large_file)

    # Force a tiny max_file_size_bytes
    with pytest.raises(ImageTooLargeError, match="exceeds limit"):
        load_and_validate_image(large_file, max_file_size_bytes=10)


def test_load_file_exceeds_pixel_limit(tmp_path: Path):
    img_path = tmp_path / "huge_res.png"
    arr = np.full((200, 200, 3), 100, dtype=np.uint8)
    Image.fromarray(arr).save(img_path)

    with pytest.raises(ImageTooLargeError, match="exceeds limit"):
        load_and_validate_image(img_path, max_image_pixels=1000)


def test_exif_orientation_normalized(tmp_path: Path):
    """Test that image with EXIF orientation 6 (90 deg CW) is transposed right-side up."""
    # Create an asymmetrical image: 100 wide x 50 high
    base_img = Image.new("RGB", (100, 50), color=(255, 0, 0))
    exif = base_img.getexif()
    # 0x0112 is Orientation tag. 6 = Rotate 90 CW (so upright is 50 wide x 100 high)
    exif[0x0112] = 6
    img_path = tmp_path / "oriented.jpg"
    base_img.save(img_path, format="JPEG", exif=exif)

    loaded = load_and_validate_image(img_path)
    # Transposed shape should swap W and H
    assert loaded.width == 50
    assert loaded.height == 100
    assert loaded.image_rgb.shape == (100, 50, 3)
