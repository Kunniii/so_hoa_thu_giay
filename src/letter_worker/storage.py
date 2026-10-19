"""Safe filesystem operations, path sandboxing, and atomic file persistence."""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from letter_worker.errors import OutputValidationError, StorageSecurityError


class SafeStorage:
    """Manages secure reading, temporary writes, and atomic persistence of artifacts."""

    def __init__(self, root_dir: Path | str):
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def resolve_safe_path(self, path_str: str | Path, must_exist: bool = False) -> Path:
        """Resolve a path and verify it does not escape the configured storage root.

        Args:
            path_str: Path relative to root or absolute path.
            must_exist: Whether the target path must exist on disk.

        Returns:
            Resolved absolute Path.

        Raises:
            StorageSecurityError: If path escapes sandbox or fails existence check.
        """
        p = Path(path_str)
        if not p.is_absolute():
            resolved = (self.root_dir / p).resolve()
        else:
            resolved = p.resolve()

        # Check containment
        try:
            resolved.relative_to(self.root_dir)
        except ValueError as err:
            raise StorageSecurityError(
                f"Path '{path_str}' is outside storage root '{self.root_dir}'",
                details={"path": str(path_str), "root": str(self.root_dir)},
            ) from err

        if must_exist and not resolved.exists():
            raise StorageSecurityError(
                f"Required path does not exist: {resolved}",
                details={"path": str(resolved)},
            )

        return resolved

    def ensure_output_dir(self, output_dir: str | Path) -> Path:
        """Resolve and create output directory safely."""
        out_dir = self.resolve_safe_path(output_dir, must_exist=False)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def save_image_atomically(
        self,
        image: np.ndarray | Image.Image,
        target_path: Path,
        format_name: str = "PNG",
    ) -> Path:
        """Save image to a temp file, verify readability, and atomically rename to target.

        Args:
            image: numpy array (RGB or Grayscale) or PIL Image.
            target_path: Absolute destination path.
            format_name: "PNG", "JPEG", "WEBP".

        Returns:
            Final Path.

        Raises:
            OutputValidationError: If temporary file cannot be verified.
        """
        target_dir = target_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        temp_path = (
            target_dir / f".tmp_{target_path.stem}_{os.getpid()}_{id(image)}.{format_name.lower()}"
        )

        try:
            # Convert numpy array to PIL Image if necessary
            if isinstance(image, np.ndarray):
                if image.ndim == 2:
                    pil_img = Image.fromarray(image, mode="L")
                elif image.ndim == 3 and image.shape[2] == 3:
                    pil_img = Image.fromarray(image, mode="RGB")
                elif image.ndim == 3 and image.shape[2] == 4:
                    pil_img = Image.fromarray(image, mode="RGBA")
                else:
                    raise OutputValidationError(f"Unsupported image array shape: {image.shape}")
            else:
                pil_img = image

            # Save to temporary file
            pil_img.save(temp_path, format=format_name, quality=95)

            # Verification: reopen and verify integrity
            with Image.open(temp_path) as check_img:
                check_img.verify()

            # Atomic rename / replace
            os.replace(temp_path, target_path)
            return target_path

        except Exception as e:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            if isinstance(e, OutputValidationError):
                raise
            raise OutputValidationError(
                f"Failed to persist or verify output image: {e}",
                details={"target": str(target_path)},
            ) from e

    def save_json_atomically(self, data: dict[str, Any], target_path: Path) -> Path:
        """Atomically persist JSON dictionary to disk."""
        target_dir = target_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        temp_path = target_dir / f".tmp_{target_path.stem}_{os.getpid()}_{id(data)}.json"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, target_path)
            return target_path
        except Exception as e:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise StorageSecurityError(
                f"Failed to persist json metadata: {e}",
                details={"target": str(target_path)},
            ) from e
