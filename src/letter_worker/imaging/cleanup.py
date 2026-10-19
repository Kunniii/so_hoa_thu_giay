"""Image restoration and cleanup presets: Pure B&W, Natural, and Vintage."""

import cv2
import numpy as np

from letter_worker.imaging.illumination import correct_illumination_gray
from letter_worker.schemas import RestoreSettings


def apply_denoise(gray: np.ndarray, mode: str = "light") -> np.ndarray:
    """Apply non-destructive denoising to grayscale document."""
    if mode == "off":
        return gray
    elif mode == "light":
        # Fast gaussian or small bilateral to preserve fine ink lines
        return cv2.bilateralFilter(gray, d=5, sigmaColor=25, sigmaSpace=25)
    elif mode == "medium":
        # Stronger bilateral or fastNlMeans
        return cv2.fastNlMeansDenoising(gray, h=6, templateWindowSize=7, searchWindowSize=21)
    return gray


def cleanup_pure_bw(image: np.ndarray, settings: RestoreSettings) -> np.ndarray:
    """Preset A: Pure B&W binarization. White paper (255), black ink (0)."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # 1. Correct illumination
    corrected = correct_illumination_gray(gray, target_paper_val=235.0, strength=1.0)

    # 2. Denoise
    denoised = apply_denoise(corrected, mode=settings.denoise)

    # 3. Thresholding
    if settings.threshold_method == "otsu":
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Adaptive thresholding
        h, w = denoised.shape
        if settings.adaptive_block_size is not None:
            block_size = settings.adaptive_block_size
            if block_size % 2 == 0:
                block_size += 1
        else:
            # Scaled odd block size
            diag = int(np.hypot(h, w))
            block_size = max(15, (diag // 45) | 1)

        if settings.adaptive_c is not None:
            c_val = settings.adaptive_c
        else:
            # Derived from threshold strength: strength 0.0 -> C=4, strength 1.0 -> C=16
            c_val = 4.0 + (settings.threshold_strength * 12.0)

        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c_val,
        )

    return binary


def cleanup_natural(image: np.ndarray, settings: RestoreSettings) -> np.ndarray:
    """Preset B: Natural enhancement. Multi-level grayscale preserving subtle ink shades."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # 1. Illumination correction
    corrected = correct_illumination_gray(gray, target_paper_val=235.0, strength=0.92)

    # 2. Denoise
    denoised = apply_denoise(corrected, mode=settings.denoise)

    # 3. Controlled contrast enhancement via CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 4. Whiten background slightly while preserving ink nuances
    # Linear stretch above threshold: anything >= 220 pushed gracefully to 255
    lut = np.arange(256, dtype=np.float32)
    bg_thresh = 210.0
    mask_bg = lut >= bg_thresh
    lut[mask_bg] = bg_thresh + (lut[mask_bg] - bg_thresh) * (255.0 - bg_thresh) / (
        255.0 - bg_thresh + 1e-5
    )
    # Apply threshold strength to ink contrast
    ink_factor = 0.85 + (1.0 - settings.threshold_strength) * 0.3
    lut[~mask_bg] = np.clip(lut[~mask_bg] * ink_factor, 0, bg_thresh)

    stretched = cv2.LUT(enhanced, np.clip(lut, 0, 255).astype(np.uint8))
    return stretched


def cleanup_vintage(image: np.ndarray, settings: RestoreSettings) -> np.ndarray:
    """Preset C: Vintage parchment tone with dark sepia ink."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # 1. Gentle illumination correction (keeps subtle organic texture)
    corrected = correct_illumination_gray(gray, target_paper_val=225.0, strength=0.65)

    # 2. Denoise
    denoised = apply_denoise(corrected, mode=settings.denoise)

    # 3. Contrast adjustment
    clahe = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 4. Color mapping to warm aged parchment & antique brown ink
    paper_rgb = np.array([245.0, 238.0, 222.0], dtype=np.float32)  # Warm ivory
    ink_rgb = np.array([42.0, 32.0, 24.0], dtype=np.float32)  # Iron gall sepia

    norm = (enhanced.astype(np.float32) / 255.0)[:, :, np.newaxis]
    vintage_rgb = ink_rgb * (1.0 - norm) + paper_rgb * norm

    return np.clip(vintage_rgb, 0, 255).astype(np.uint8)


def restore_image(image: np.ndarray, settings: RestoreSettings) -> np.ndarray:
    """Dispatch image restoration according to chosen preset."""
    if settings.preset == "pure_bw":
        return cleanup_pure_bw(image, settings)
    elif settings.preset == "natural":
        return cleanup_natural(image, settings)
    elif settings.preset == "vintage":
        return cleanup_vintage(image, settings)
    else:
        return cleanup_natural(image, settings)
