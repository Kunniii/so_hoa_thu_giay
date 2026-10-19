"""Pydantic models and schemas for Letter Processing Worker."""

from enum import StrEnum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field


class Point(BaseModel):
    """2D coordinate representing an image corner."""

    x: float = Field(ge=0, description="X coordinate (column) in normalized EXIF space")
    y: float = Field(ge=0, description="Y coordinate (row) in normalized EXIF space")

    def to_tuple(self) -> tuple[float, float]:
        return (float(self.x), float(self.y))

    def to_list(self) -> list[float]:
        return [float(self.x), float(self.y)]

    @classmethod
    def from_array(cls, arr: np.ndarray | tuple | list) -> "Point":
        return cls(x=float(arr[0]), y=float(arr[1]))


class RestoreSettings(BaseModel):
    """Parameters controlling illumination correction and cleanup."""

    preset: Literal["pure_bw", "natural", "vintage"] = Field(
        default="natural",
        description="Restoration style: pure_bw, natural, or vintage",
    )
    denoise: Literal["off", "light", "medium"] = Field(
        default="light",
        description="Degree of denoising applied before enhancement",
    )
    threshold_method: Literal["adaptive", "otsu"] = Field(
        default="adaptive",
        description="Binarization algorithm for pure_bw preset",
    )
    threshold_strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Normalized strength (0.0 to 1.0) controlling threshold bias",
    )
    adaptive_block_size: int | None = Field(
        default=None,
        description="Custom odd block size for adaptive thresholding. If None, auto-calculated.",
    )
    adaptive_c: float | None = Field(
        default=None,
        description="Custom constant subtracted in adaptive thresholding. If None, derived from strength.",
    )


class JobStage(StrEnum):
    """Pipeline progression stages."""

    QUEUED = "queued"
    LOADING = "loading"
    DETECTING = "detecting"
    AWAITING_CORNERS = "awaiting_corners"
    RECTIFYING = "rectifying"
    RESTORING = "restoring"
    QUALITY_CHECK = "quality_check"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LetterJob(BaseModel):
    """Input payload describing a letter processing request."""

    job_id: str = Field(description="Unique identifier for the job")
    input_path: str = Field(description="Path to input photo relative to storage root")
    output_dir: str = Field(description="Destination directory for processed output")
    corners: list[Point] | None = Field(
        default=None,
        description="Optional pre-confirmed corners [TL, TR, BR, BL]. If None, auto-detected.",
    )
    settings: RestoreSettings = Field(
        default_factory=RestoreSettings,
        description="Restoration parameters",
    )


class JobProgressEvent(BaseModel):
    """Structured progress update emitted during job execution."""

    job_id: str
    stage: JobStage
    progress: int = Field(ge=0, le=100)
    message_code: str
    details: dict[str, Any] = Field(default_factory=dict)


class QualityCheckResult(BaseModel):
    """Outcome of quality assurance analysis on the restored image."""

    status: Literal["passed", "warning", "rejected"]
    blur_score: float = Field(ge=0.0)
    ink_coverage: float = Field(ge=0.0, le=1.0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    reasons: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class JobResult(BaseModel):
    """Final outcome and metadata of a processed job."""

    job_id: str
    status: Literal["completed", "awaiting_corners", "failed", "cancelled"]
    output_path: str | None = None
    metadata_path: str | None = None
    detected_corners: list[Point] | None = None
    confidence: float | None = None
    quality: QualityCheckResult | None = None
    error_code: str | None = None
    error_message: str | None = None
    duration_seconds: float = 0.0
