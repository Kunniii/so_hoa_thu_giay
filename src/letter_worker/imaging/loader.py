"""Image loading, EXIF orientation correction, and safety validation."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from letter_worker.errors import ImageTooLargeError, InvalidImageError

SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP", "TIFF", "BMP", "HEIC"}


@dataclass
class LoadedImage:
    """Decoded, EXIF-normalized image in RGB format."""

    image_rgb: np.ndarray
    width: int
    height: int
    source_format: str


def load_and_validate_image(
    file_path: Path,
    max_file_size_bytes: int = 50 * 1024 * 1024,
    max_image_pixels: int = 50_000_000,
) -> LoadedImage:
    """Load, inspect safety constraints, orient via EXIF, and decode to RGB numpy array.

    Args:
        file_path: Absolute or safe path to target image.
        max_file_size_bytes: Maximum allowed file size in bytes.
        max_image_pixels: Maximum allowed pixel area (W x H).

    Returns:
        LoadedImage with upright RGB numpy array and metadata.

    Raises:
        InvalidImageError: If format unsupported, file corrupt, or unreadable.
        ImageTooLargeError: If size exceeds allowed safety thresholds.
    """
    if not file_path.exists() or not file_path.is_file():
        raise InvalidImageError(f"Image file does not exist: {file_path}")

    # 1. Check file size
    file_size = file_path.stat().st_size
    if file_size == 0:
        raise InvalidImageError(f"Image file is empty (0 bytes): {file_path}")

    if file_size > max_file_size_bytes:
        raise ImageTooLargeError(
            f"Image file size ({file_size} bytes) exceeds limit ({max_file_size_bytes} bytes)",
            details={"file_size": file_size, "limit": max_file_size_bytes},
        )

    # 2. Inspect image header before complete decode
    Image.MAX_IMAGE_PIXELS = max_image_pixels

    try:
        with Image.open(file_path) as probe:
            fmt = (probe.format or "").upper()
            if (
                fmt not in SUPPORTED_FORMATS
                and file_path.suffix.lstrip(".").upper() not in SUPPORTED_FORMATS
            ):
                raise InvalidImageError(
                    f"Unsupported image format '{fmt}' (extension '{file_path.suffix}')",
                    details={"format": fmt, "supported": list(SUPPORTED_FORMATS)},
                )

            width, height = probe.size
            total_pixels = width * height
            if total_pixels > max_image_pixels:
                raise ImageTooLargeError(
                    f"Image resolution {width}x{height} ({total_pixels} px) exceeds limit ({max_image_pixels} px)",
                    details={
                        "width": width,
                        "height": height,
                        "pixels": total_pixels,
                        "limit": max_image_pixels,
                    },
                )
    except (ImageTooLargeError, InvalidImageError):
        raise
    except Image.DecompressionBombError as e:
        raise ImageTooLargeError(f"Image decompression bomb limit exceeded: {e}") from e
    except Exception as e:
        raise InvalidImageError(f"Cannot inspect image headers: {e}") from e

    # 3. Full decode with EXIF transposition
    try:
        with Image.open(file_path) as pil_img:
            # Handle orientation if present
            transposed = ImageOps.exif_transpose(pil_img)
            if transposed is None:
                transposed = pil_img

            # Convert to RGB (handles RGBA, Palette, Grayscale, etc.)
            rgb_img = transposed.convert("RGB")
            arr = np.array(rgb_img, dtype=np.uint8)

            w, h = rgb_img.size
            return LoadedImage(
                image_rgb=arr,
                width=w,
                height=h,
                source_format=fmt or pil_img.format or "UNKNOWN",
            )
    except Exception as e:
        raise InvalidImageError(f"Failed to decode image data: {e}") from e
