"""Pipeline orchestrator for letter document processing."""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image, ImageOps

from letter_worker.config import WorkerConfig, default_config
from letter_worker.errors import (
    InvalidCornersError,
    InvalidImageError,
    OutputValidationError,
    PaperNotDetectedError,
    WorkerError,
)
from letter_worker.imaging.cleanup import restore_image
from letter_worker.imaging.detection import detect_paper_corners
from letter_worker.imaging.geometry import (
    order_corners,
    rectify_perspective,
    validate_quadrilateral,
)
from letter_worker.imaging.loader import load_and_validate_image
from letter_worker.imaging.quality import evaluate_image_quality
from letter_worker.schemas import (
    JobProgressEvent,
    JobResult,
    JobStage,
    LetterJob,
    Point,
    RestoreSettings,
)
from letter_worker.storage import SafeStorage

ProgressCallback = Callable[[JobProgressEvent], None]


class LetterProcessor:
    """Orchestrates end-to-end pipeline execution from photo to restored letter."""

    def __init__(self, config: WorkerConfig = default_config):
        self.config = config
        self.storage = SafeStorage(config.storage_root)

    def _notify(
        self,
        callback: ProgressCallback | None,
        job_id: str,
        stage: JobStage,
        progress: int,
        message_code: str,
        details: dict | None = None,
    ) -> None:
        if callback:
            callback(
                JobProgressEvent(
                    job_id=job_id,
                    stage=stage,
                    progress=progress,
                    message_code=message_code,
                    details=details or {},
                )
            )

    def process(
        self,
        job: LetterJob,
        on_progress: ProgressCallback | None = None,
    ) -> JobResult:
        """Execute the processing pipeline on a single LetterJob.

        Args:
            job: LetterJob payload containing input path, output dir, corners, and settings.
            on_progress: Optional callback function receiving JobProgressEvent.

        Returns:
            JobResult containing final status, paths, corners, and quality stats.
        """
        start_time = time.time()
        job_id = job.job_id

        try:
            # 1. Resolve safe paths
            input_file = self.storage.resolve_safe_path(job.input_path, must_exist=True)
            output_dir = self.storage.ensure_output_dir(job.output_dir)

            # 2. Stage: Loading & Validation
            self._notify(on_progress, job_id, JobStage.LOADING, 10, "IMAGE_LOADING")
            loaded = load_and_validate_image(
                input_file,
                max_file_size_bytes=self.config.max_image_file_size_bytes,
                max_image_pixels=self.config.max_image_pixels,
            )

            # 3. Stage: Corner Determination
            self._notify(on_progress, job_id, JobStage.DETECTING, 25, "CORNER_DETERMINATION")

            effective_corners: np.ndarray | None = None
            confidence: float = 1.0
            detected_pts_list: list[Point] | None = None

            if job.corners is not None and len(job.corners) == 4:
                # User provided manual / confirmed corners
                corners_arr = np.array([p.to_tuple() for p in job.corners], dtype=np.float32)
                ordered = order_corners(corners_arr)
                validate_quadrilateral(ordered, image_shape=(loaded.height, loaded.width))
                effective_corners = ordered
                confidence = 1.0
                detected_pts_list = [Point.from_array(pt) for pt in ordered]
                self._notify(on_progress, job_id, JobStage.DETECTING, 35, "USING_USER_CORNERS")
            else:
                # Automatic corner detection
                det_res = detect_paper_corners(loaded.image_rgb)
                confidence = det_res.confidence

                if det_res.corners is not None:
                    detected_pts_list = [Point.from_array(pt) for pt in det_res.corners]

                # If confidence is below threshold, request manual corner confirmation
                if det_res.corners is None or confidence < self.config.confidence_auto_accept:
                    self._notify(
                        on_progress,
                        job_id,
                        JobStage.AWAITING_CORNERS,
                        40,
                        "AWAITING_CORNER_CONFIRMATION",
                        {
                            "confidence": confidence,
                            "has_detected_candidates": det_res.corners is not None,
                        },
                    )
                    return JobResult(
                        job_id=job_id,
                        status="awaiting_corners",
                        detected_corners=detected_pts_list,
                        confidence=confidence,
                        duration_seconds=round(time.time() - start_time, 3),
                    )

                effective_corners = det_res.corners

            if effective_corners is None:
                raise InvalidCornersError("No valid corners available for rectification")

            # 4. Stage: Rectification
            self._notify(on_progress, job_id, JobStage.RECTIFYING, 50, "PERSPECTIVE_RECTIFICATION")
            rectified, (out_w, out_h) = rectify_perspective(
                loaded.image_rgb,
                effective_corners,
                aspect_ratio_mode=job.settings.aspect_ratio,
                max_dimension=self.config.max_output_dimension,
            )

            # Optional margin inset to trim away spiral bindings or dirty border edges
            if job.settings.margin_inset > 0.0:
                inset = min(job.settings.margin_inset, 0.15)
                dx = int(round(out_w * inset))
                dy = int(round(out_h * inset))
                if out_w - 2 * dx >= 100 and out_h - 2 * dy >= 100:
                    rectified = rectified[dy : out_h - dy, dx : out_w - dx]
                    out_h, out_w = rectified.shape[:2]

            # 5. Stage: Restoration (Illumination correction & preset cleanup)
            self._notify(on_progress, job_id, JobStage.RESTORING, 75, "RESTORATION_APPLIED")
            restored = restore_image(rectified, job.settings)

            # 6. Stage: Quality Assurance
            self._notify(on_progress, job_id, JobStage.QUALITY_CHECK, 90, "QUALITY_ASSURANCE")
            qa_res = evaluate_image_quality(restored)

            if qa_res.status == "rejected":
                raise OutputValidationError(
                    f"Quality check rejected: {'; '.join(qa_res.reasons)}",
                    details=qa_res.details,
                )

            # 7. Stage: Persist Result
            output_filename = f"{job_id}_{job.settings.preset}.png"
            output_path = output_dir / output_filename
            saved_img_path = self.storage.save_image_atomically(
                restored, output_path, format_name="PNG"
            )

            duration = round(time.time() - start_time, 3)

            # Build metadata payload
            metadata = {
                "job_id": job_id,
                "input_path": str(input_file),
                "output_path": str(saved_img_path),
                "settings": job.settings.model_dump(),
                "corners": [p.model_dump() for p in (detected_pts_list or [])],
                "confidence": confidence,
                "quality": qa_res.model_dump(),
                "duration_seconds": duration,
                "output_dimensions": {"width": out_w, "height": out_h},
            }
            meta_path = output_dir / f"{job_id}_metadata.json"
            saved_meta_path = self.storage.save_json_atomically(metadata, meta_path)

            self._notify(on_progress, job_id, JobStage.COMPLETED, 100, "PROCESSING_COMPLETED")

            return JobResult(
                job_id=job_id,
                status="completed",
                output_path=str(saved_img_path),
                metadata_path=str(saved_meta_path),
                detected_corners=detected_pts_list,
                confidence=confidence,
                quality=qa_res,
                duration_seconds=duration,
            )

        except WorkerError as e:
            self._notify(
                on_progress,
                job_id,
                JobStage.FAILED,
                100,
                e.code,
                {"error": e.message, "details": e.details},
            )
            return JobResult(
                job_id=job_id,
                status="failed",
                error_code=e.code,
                error_message=e.message,
                duration_seconds=round(time.time() - start_time, 3),
            )
        except Exception as e:
            self._notify(
                on_progress,
                job_id,
                JobStage.FAILED,
                100,
                "INTERNAL_ERROR",
                {"error": str(e)},
            )
            return JobResult(
                job_id=job_id,
                status="failed",
                error_code="INTERNAL_ERROR",
                error_message=str(e),
                duration_seconds=round(time.time() - start_time, 3),
            )


def process_letter(
    image: str | Path | np.ndarray | Image.Image,
    corners: list[tuple[float, float]] | list[list[float]] | np.ndarray | None = None,
    preset: Literal["pure_bw", "natural", "vintage"] = "vintage",
    aspect_ratio: Literal["auto", "a4", "notebook", "letter", "free"] = "auto",
    paper_warmth: float = 0.75,
    bleed_through_reduction: float = 0.82,
    margin_inset: float = 0.0,
    threshold_strength: float = 0.5,
    denoise: Literal["off", "light", "medium"] = "light",
    output_path: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convenience functional API to restore a letter document in-memory or from file.

    Performs full pipeline: loading, corner detection or ordering, 3D perspective
    rectification, illumination flattening, bleed-through suppression, aesthetic
    color/tone mapping, and quality assurance.

    Args:
        image: Filepath (str/Path), uint8 RGB NumPy array (H, W, 3), or PIL Image.
        corners: Optional 4 corner points [[x, y], ...]. If None, auto-detected.
        preset: Restoration style ('pure_bw', 'natural', 'vintage').
        aspect_ratio: Aspect ratio mode ('auto', 'a4', 'notebook', 'letter', 'free').
        paper_warmth: Patina warmth in vintage preset (0.0 to 1.0, default 0.75).
        bleed_through_reduction: Show-through ink suppression (0.50 to 0.98, default 0.82).
        margin_inset: Fraction of margin to crop (0.0 to 0.10, default 0.0).
        threshold_strength: Ink stroke darkness sensitivity (0.0 to 1.0, default 0.5).
        denoise: Denoising degree ('off', 'light', 'medium').
        output_path: Optional path to save the restored image to disk.

    Returns:
        tuple[np.ndarray, dict]: (restored_image_rgb, metadata_dict)

    Raises:
        InvalidImageError: If image cannot be read or shape is invalid.
        PaperNotDetectedError: If corners is None and auto-detection fails.
        InvalidCornersError: If provided corners are invalid or self-intersecting.
        OutputValidationError: If quality check fails.
    """
    start_time = time.time()

    # 1. Normalize input image into uint8 RGB array
    if isinstance(image, (str, Path)):
        loaded = load_and_validate_image(Path(image))
        rgb_img = loaded.image_rgb
    elif isinstance(image, Image.Image):
        transposed = ImageOps.exif_transpose(image)
        if transposed is None:
            transposed = image
        rgb_img = np.array(transposed.convert("RGB"), dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            rgb_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            rgb_img = image.copy()
        elif image.ndim == 3 and image.shape[2] == 4:
            rgb_img = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            raise InvalidImageError(f"Unsupported image array shape {image.shape}")
    else:
        raise InvalidImageError(f"Unsupported image type: {type(image)}")

    # 2. Determine corners
    confidence = 1.0
    if corners is not None:
        corners_arr = np.array(corners, dtype=np.float32).reshape(4, 2)
        effective_corners = order_corners(corners_arr)
        validate_quadrilateral(effective_corners, image_shape=rgb_img.shape)
    else:
        det_res = detect_paper_corners(rgb_img)
        if det_res.corners is None:
            raise PaperNotDetectedError("Automatic paper detection failed to find 4 corners")
        effective_corners = det_res.corners
        confidence = det_res.confidence

    # 3. Rectify perspective
    rectified, (out_w, out_h) = rectify_perspective(
        rgb_img,
        effective_corners,
        aspect_ratio_mode=aspect_ratio,
    )

    # 4. Optional margin inset
    if margin_inset > 0.0:
        inset = min(margin_inset, 0.15)
        dx = int(round(out_w * inset))
        dy = int(round(out_h * inset))
        if out_w - 2 * dx >= 100 and out_h - 2 * dy >= 100:
            rectified = rectified[dy : out_h - dy, dx : out_w - dx]
            out_h, out_w = rectified.shape[:2]

    # 5. Restore image
    settings = RestoreSettings(
        preset=preset,
        threshold_strength=threshold_strength,
        bleed_through_reduction=bleed_through_reduction,
        margin_inset=margin_inset,
        paper_warmth=paper_warmth,
        aspect_ratio=aspect_ratio,
        denoise=denoise,
    )
    restored = restore_image(rectified, settings)

    # 6. Quality assurance
    qa_res = evaluate_image_quality(restored)
    if qa_res.status == "rejected":
        raise OutputValidationError(
            f"Quality check rejected: {'; '.join(qa_res.reasons)}",
            details=qa_res.details,
        )

    # 7. Optional disk write
    if output_path is not None:
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(restored).save(out_p)

    duration = round(time.time() - start_time, 3)
    metadata = {
        "settings": settings.model_dump(),
        "corners": effective_corners.tolist(),
        "confidence": confidence,
        "quality": qa_res.model_dump(),
        "duration_seconds": duration,
        "output_dimensions": {"width": out_w, "height": out_h},
    }
    return restored, metadata
