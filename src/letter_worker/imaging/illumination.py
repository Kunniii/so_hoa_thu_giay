"""Illumination correction and shadow removal algorithms."""

import cv2
import numpy as np


def estimate_background(
    gray: np.ndarray,
    scale_factor: float = 0.25,
    blur_kernel_ratio: float = 0.08,
) -> np.ndarray:
    """Estimate low-frequency background illumination via scaled Gaussian filtering.

    Downscales for computational speed and wide-field smoothing, then upsamples back.

    Args:
        gray: 2D uint8 grayscale image.
        scale_factor: Downscaling factor for illumination estimation (0.1 to 0.5).
        blur_kernel_ratio: Ratio of image dimension used for Gaussian blur sigma.

    Returns:
        2D float32 background illumination map of original shape.
    """
    h, w = gray.shape[:2]
    small_w = max(int(w * scale_factor), 32)
    small_h = max(int(h * scale_factor), 32)

    # Downsample
    small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # Estimate blur sigma relative to downscaled dimension
    sigma = max(10.0, min(small_w, small_h) * blur_kernel_ratio)
    ksize = int(2 * round(3 * sigma) + 1)

    blurred_small = cv2.GaussianBlur(
        small.astype(np.float32),
        (ksize, ksize),
        sigmaX=sigma,
        sigmaY=sigma,
    )

    # Upsample back to original resolution
    background = cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.maximum(background, 1.0)


def correct_illumination_gray(
    gray: np.ndarray,
    target_paper_val: float = 230.0,
    strength: float = 1.0,
) -> np.ndarray:
    """Normalize illumination across a grayscale image, removing uneven shading and shadows.

    Args:
        gray: 2D uint8 grayscale image.
        target_paper_val: Desired mean brightness for paper background (usually 220-240).
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
    target_paper_val: float = 230.0,
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

    corrected_l = correct_illumination_gray(
        l_chan, target_paper_val=target_paper_val, strength=strength
    )
    corrected_lab = cv2.merge([corrected_l, a_chan, b_chan])
    return cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2RGB)
