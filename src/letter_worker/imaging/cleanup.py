"""Image restoration and cleanup presets: Pure B&W, Natural, and Vintage with bleed-through suppression."""

import cv2
import numpy as np

from letter_worker.imaging.illumination import estimate_background
from letter_worker.schemas import RestoreSettings


def apply_denoise(gray: np.ndarray, mode: str = "light") -> np.ndarray:
    """Apply non-destructive denoising to grayscale document."""
    if mode == "off":
        return gray
    elif mode == "light":
        # Bilateral filter preserves fine ink strokes and punctuation
        return cv2.bilateralFilter(gray, d=5, sigmaColor=25, sigmaSpace=25)
    elif mode == "medium":
        # Fast non-local means for high-ISO noise
        return cv2.fastNlMeansDenoising(gray, h=6, templateWindowSize=7, searchWindowSize=21)
    return gray


def normalize_and_suppress_bleed_through(
    gray: np.ndarray,
    bleed_through_reduction: float = 0.82,
    threshold_strength: float = 0.5,
) -> np.ndarray:
    """Normalize background to 1.0 and suppress faint show-through text from back of page.

    Args:
        gray: 2D uint8 grayscale image.
        bleed_through_reduction: White-point threshold ratio (0.50 to 0.98).
            Faint show-through text has lower contrast relative to paper than primary ink.
            Pixels >= this ratio are pushed to pure 1.0 paper.
        threshold_strength: Sensitivity controlling ink stroke darkness.

    Returns:
        float32 2D array in [0.0, 1.0] where 1.0 is pure paper and 0.0 is deep ink.
    """
    gray_f = gray.astype(np.float32)
    background = estimate_background(gray)
    norm = gray_f / background

    # White-point: anything >= white_point becomes pure paper background
    white_point = float(np.clip(bleed_through_reduction, 0.50, 0.98))

    # Black-point: primary ink boundary
    black_point = 0.20 + (1.0 - threshold_strength) * 0.25
    black_point = min(black_point, white_point - 0.15)

    # Stretch contrast between black_point and white_point
    stretched = np.clip((norm - black_point) / (white_point - black_point), 0.0, 1.0)

    # Power curve to retain ink stroke gradation while keeping paper pristine
    cleaned = np.power(stretched, 1.25)
    return cleaned


def cleanup_pure_bw(image: np.ndarray, settings: RestoreSettings) -> np.ndarray:
    """Preset A: Pure B&W binarization. Clean white paper (255) and crisp dark ink (0)."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    denoised = apply_denoise(gray, mode=settings.denoise)
    cleaned_norm = normalize_and_suppress_bleed_through(
        denoised,
        bleed_through_reduction=settings.bleed_through_reduction,
        threshold_strength=settings.threshold_strength,
    )
    cleaned_u8 = (cleaned_norm * 255.0).astype(np.uint8)

    if settings.threshold_method == "otsu":
        _, binary = cv2.threshold(cleaned_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Adaptive thresholding
        h, w = cleaned_u8.shape
        if settings.adaptive_block_size is not None:
            block_size = settings.adaptive_block_size
            if block_size % 2 == 0:
                block_size += 1
        else:
            diag = int(np.hypot(h, w))
            block_size = max(15, (diag // 45) | 1)

        if settings.adaptive_c is not None:
            c_val = settings.adaptive_c
        else:
            c_val = 4.0 + (settings.threshold_strength * 12.0)

        binary = cv2.adaptiveThreshold(
            cleaned_u8,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c_val,
        )

    return binary


def cleanup_natural(image: np.ndarray, settings: RestoreSettings) -> np.ndarray:
    """Preset B: Natural enhancement. Multi-level grayscale preserving subtle pencil/ink nuances."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    denoised = apply_denoise(gray, mode=settings.denoise)
    cleaned_norm = normalize_and_suppress_bleed_through(
        denoised,
        bleed_through_reduction=settings.bleed_through_reduction,
        threshold_strength=settings.threshold_strength,
    )
    return np.clip(cleaned_norm * 255.0, 0, 255).astype(np.uint8)


def cleanup_vintage(image: np.ndarray, settings: RestoreSettings) -> np.ndarray:
    """Preset C: Vintage parchment tone with warm ivory paper and antique dark sepia ink."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    denoised = apply_denoise(gray, mode=settings.denoise)
    cleaned_norm = normalize_and_suppress_bleed_through(
        denoised,
        bleed_through_reduction=settings.bleed_through_reduction,
        threshold_strength=settings.threshold_strength,
    )

    # 1. Extract subtle paper fiber grain to preserve the tactile physical character of real paper
    smooth_bg = cv2.GaussianBlur(gray, (45, 45), 0).astype(np.float32)
    grain = np.clip(gray.astype(np.float32) / (smooth_bg + 1e-5), 0.94, 1.05)
    paper_texture = np.clip(cleaned_norm * grain, 0.0, 1.0)
    norm_3d = paper_texture[:, :, np.newaxis]

    # 2. Dynamic paper warmth interpolation
    # p_light: Light clean cream (#FBF8EE)
    # p_deep: Authentic antique golden-aged parchment (#D0C6B0) matching original favorite letter tone
    p_light = np.array([248.0, 242.0, 232.0], dtype=np.float32)
    p_deep = np.array([208.0, 198.0, 176.0], dtype=np.float32)
    warmth = float(np.clip(settings.paper_warmth, 0.0, 1.0))
    paper_rgb = p_light * (1.0 - warmth) + p_deep * warmth

    # Deep iron-gall antique sepia ink
    ink_rgb = np.array([42.0, 32.0, 22.0], dtype=np.float32)

    vintage_rgb = ink_rgb * (1.0 - norm_3d) + paper_rgb * norm_3d

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
