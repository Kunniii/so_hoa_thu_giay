"""Illumination correction and shadow removal algorithms."""

import cv2
import numpy as np


def estimate_background(
    gray: np.ndarray,
    scale_factor: float = 0.25,
    morph_kernel_ratio: float = 0.05,
) -> np.ndarray:
    """Estimate true paper illumination field via morphological dilation and smoothing.

    Uses morphological dilation on a downsampled image to remove dark ink strokes
    and extract the pristine background lighting map across the page.

    Args:
        gray: 2D uint8 grayscale image.
        scale_factor: Downscaling factor for speed and wide-area smoothing (0.15 to 0.35).
        morph_kernel_ratio: Ratio of downscaled dimension used for morphological structuring element.

    Returns:
        2D float32 background illumination map of original shape.
    """
    h, w = gray.shape[:2]
    small_w = max(int(round(w * scale_factor)), 32)
    small_h = max(int(round(h * scale_factor)), 32)

    # 1. Downsample
    small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # 2. Morphological dilation to eliminate dark ink strokes (leaving pure paper intensity)
    k_dim = max(11, int(round(min(small_w, small_h) * morph_kernel_ratio)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dim, k_dim))
    dilated = cv2.dilate(small, kernel)

    # 3. Median and gentle Gaussian blur to smooth out local texture transitions
    med_k = max(7, (k_dim // 2) | 1)
    smoothed = cv2.medianBlur(dilated, med_k)
    blurred = cv2.GaussianBlur(smoothed.astype(np.float32), (k_dim, k_dim), sigmaX=k_dim / 3.0)

    # 4. Upsample back to original resolution
    background = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.maximum(background, 1.0)


def correct_illumination_gray(
    gray: np.ndarray,
    target_paper_val: float = 245.0,
    strength: float = 1.0,
) -> np.ndarray:
    """Normalize illumination across a grayscale image, removing uneven shading and shadows.

    Args:
        gray: 2D uint8 grayscale image.
        target_paper_val: Desired brightness for paper background (usually 240-250).
        strength: Correction blending factor (0.0 = no correction, 1.0 = full normalization).

    Returns:
        2D uint8 corrected grayscale image.
    """
    gray_f = gray.astype(np.float32)
    background = estimate_background(gray)

    # Normalized division
    normalized = (gray_f / background) * target_paper_val
    normalized = np.clip(normalized, 0, 255)

    if strength >= 0.999:
        return normalized.astype(np.uint8)

    # Blend with original according to strength
    blended = strength * normalized + (1.0 - strength) * gray_f
    return np.clip(blended, 0, 255).astype(np.uint8)


def correct_illumination_rgb(
    rgb: np.ndarray,
    target_paper_val: float = 245.0,
    strength: float = 1.0,
) -> np.ndarray:
    """Normalize illumination on color images via LAB luminance channel to preserve hues.

    Args:
        rgb: 3D uint8 RGB image.
        target_paper_val: Desired paper brightness level.
        strength: Blending strength.

    Returns:
        3D uint8 corrected RGB image.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    corrected_l = correct_illumination_gray(l_chan, target_paper_val=target_paper_val, strength=strength)
    corrected_lab = cv2.merge([corrected_l, a_chan, b_chan])
    return cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2RGB)
