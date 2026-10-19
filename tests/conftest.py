"""Shared pytest fixtures and synthetic image generators."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def temp_storage(tmp_path: Path) -> Path:
    """Provide a temporary storage directory."""
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


@pytest.fixture
def synthetic_letter_image() -> np.ndarray:
    """Generate a clean synthetic letter: white paper document with dark handwritten-like strokes."""
    # 600 x 400 paper
    paper = np.full((600, 400, 3), 245, dtype=np.uint8)

    # Draw text-like lines (ink strokes)
    for y in range(80, 520, 30):
        # Draw simulated handwriting lines with subtle variations
        pts = []
        for x in range(50, 350, 15):
            dy = int(np.sin(x * 0.1) * 4)
            pts.append([x, y + dy])
        cv2.polylines(paper, [np.array(pts, dtype=np.int32)], False, (30, 25, 20), thickness=2)

    return paper


@pytest.fixture
def synthetic_desk_photo(synthetic_letter_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate a realistic synthetic photo: dark desk background with a perspective-warped letter.

    Returns:
        (photo_rgb, ground_truth_corners)
    """
    canvas_h, canvas_w = 800, 1000
    # Dark desk background with slight texture
    desk = np.full((canvas_h, canvas_w, 3), 50, dtype=np.uint8)
    # Add subtle desk grain
    noise = np.random.randint(-10, 10, (canvas_h, canvas_w, 3), dtype=np.int16)
    desk = np.clip(desk.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Letter dimensions: 600 x 400
    src_h, src_w = synthetic_letter_image.shape[:2]
    src_pts = np.array(
        [[0, 0], [src_w - 1, 0], [src_w - 1, src_h - 1], [0, src_h - 1]],
        dtype=np.float32,
    )

    # Target quad on desk (clockwise: TL, TR, BR, BL)
    dst_pts = np.array(
        [[180.0, 120.0], [780.0, 160.0], [720.0, 680.0], [140.0, 620.0]],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_paper = cv2.warpPerspective(
        synthetic_letter_image,
        matrix,
        (canvas_w, canvas_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    mask = cv2.warpPerspective(
        np.full((src_h, src_w), 255, dtype=np.uint8),
        matrix,
        (canvas_w, canvas_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Composite paper onto desk
    mask_3d = mask[:, :, np.newaxis] > 0
    composite = np.where(mask_3d, warped_paper, desk)

    return composite, dst_pts


@pytest.fixture
def sample_image_file(temp_storage: Path, synthetic_desk_photo) -> Path:
    """Save a synthetic desk photo to a JPEG file in temp storage."""
    photo, _ = synthetic_desk_photo
    img_path = temp_storage / "sample_desk.jpg"
    pil_img = Image.fromarray(photo)
    pil_img.save(img_path, format="JPEG", quality=90)
    return img_path
