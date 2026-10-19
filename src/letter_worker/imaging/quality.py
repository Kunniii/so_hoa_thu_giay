"""Quality assurance checks and anomaly detection for restored letter images."""

from typing import Literal

import cv2
import numpy as np

from letter_worker.schemas import QualityCheckResult


def compute_blur_score(gray: np.ndarray) -> float:
    """Compute blurriness estimate via variance of Laplacian."""
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())
    return variance


def compute_ink_coverage(gray: np.ndarray, ink_threshold: int = 140) -> float:
    """Calculate proportion of pixels classified as ink (dark strokes).

    Args:
        gray: 2D uint8 grayscale image.
        ink_threshold: Pixel intensity below which a pixel is considered ink.

    Returns:
        Fraction between 0.0 and 1.0.
    """
    total_pixels = gray.size
    if total_pixels == 0:
        return 0.0
    ink_pixels = int(np.count_nonzero(gray < ink_threshold))
    return float(ink_pixels / total_pixels)


def evaluate_image_quality(
    image: np.ndarray,
    min_dimension: int = 100,
    min_blur_score: float = 35.0,
) -> QualityCheckResult:
    """Run comprehensive quality assurance checks on restored image.

    Args:
        image: 2D (grayscale) or 3D (RGB) numpy array.
        min_dimension: Minimum allowed width/height.
        min_blur_score: Threshold below which image is flagged as blurry.

    Returns:
        QualityCheckResult with status 'passed', 'warning', or 'rejected'.
    """
    reasons: list[str] = []
    details: dict = {}

    # 1. Dimensionality check
    if image is None or image.size == 0:
        return QualityCheckResult(
            status="rejected",
            blur_score=0.0,
            ink_coverage=0.0,
            width=0,
            height=0,
            reasons=["IMAGE_EMPTY_OR_CORRUPT"],
            details={"error": "Image array is None or empty"},
        )

    h, w = image.shape[:2]
    if w < min_dimension or h < min_dimension:
        reasons.append(f"DIMENSIONS_TOO_SMALL: {w}x{h} (minimum {min_dimension})")
        return QualityCheckResult(
            status="rejected",
            blur_score=0.0,
            ink_coverage=0.0,
            width=w,
            height=h,
            reasons=reasons,
            details={"width": w, "height": h},
        )

    # Convert to grayscale for statistical analysis
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    blur_score = compute_blur_score(gray)
    ink_coverage = compute_ink_coverage(gray)

    details["mean_pixel_value"] = round(mean_val, 2)
    details["std_pixel_value"] = round(std_val, 2)
    details["blur_score"] = round(blur_score, 2)
    details["ink_coverage"] = round(ink_coverage, 4)

    # 2. Pixel distribution check (all-black or all-white)
    if std_val < 1.0:
        if mean_val > 250.0:
            reasons.append("SOLID_WHITE_IMAGE: Variance is virtually zero")
            return QualityCheckResult(
                status="rejected",
                blur_score=blur_score,
                ink_coverage=ink_coverage,
                width=w,
                height=h,
                reasons=reasons,
                details=details,
            )
        elif mean_val < 5.0:
            reasons.append("SOLID_BLACK_IMAGE: Image is completely dark")
            return QualityCheckResult(
                status="rejected",
                blur_score=blur_score,
                ink_coverage=ink_coverage,
                width=w,
                height=h,
                reasons=reasons,
                details=details,
            )

    status: Literal["passed", "warning", "rejected"] = "passed"

    # 3. Ink coverage warnings
    if ink_coverage < 0.0005:  # Less than 0.05%
        reasons.append("LOW_INK_COVERAGE: Very few ink strokes detected (possible blank page)")
        status = "warning"
    elif ink_coverage > 0.65:  # Over 65% dark pixels
        reasons.append("EXCESSIVE_DARK_COVERAGE: Over 65% of image is dark pixels")
        status = "warning"

    # 4. Blur score warning
    if blur_score < min_blur_score:
        reasons.append(
            f"POTENTIAL_BLUR: Sharpness score {blur_score:.1f} is below threshold {min_blur_score}"
        )
        status = "warning"

    return QualityCheckResult(
        status=status,
        blur_score=round(blur_score, 2),
        ink_coverage=round(ink_coverage, 4),
        width=w,
        height=h,
        reasons=reasons,
        details=details,
    )
