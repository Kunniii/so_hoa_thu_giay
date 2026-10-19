"""Configuration settings for Letter Worker."""

import os
from pathlib import Path

from pydantic import BaseModel, Field


class WorkerConfig(BaseModel):
    """Runtime configuration for worker processes and safety limits."""

    storage_root: Path = Field(
        default_factory=lambda: Path(os.getenv("STORAGE_ROOT", ".")).resolve()
    )
    database_path: Path = Field(
        default_factory=lambda: Path(os.getenv("DATABASE_PATH", "letter_worker.db")).resolve()
    )
    max_image_file_size_bytes: int = Field(
        default=int(os.getenv("MAX_IMAGE_FILE_SIZE_BYTES", str(50 * 1024 * 1024)))  # 50 MB
    )
    max_image_pixels: int = Field(
        default=int(os.getenv("MAX_IMAGE_PIXELS", "50000000"))  # 50 MP
    )
    max_output_dimension: int = Field(default=int(os.getenv("MAX_OUTPUT_DIMENSION", "8192")))
    job_lease_timeout_seconds: int = Field(
        default=int(os.getenv("JOB_LEASE_TIMEOUT_SECONDS", "300"))
    )
    job_max_attempts: int = Field(default=int(os.getenv("JOB_MAX_ATTEMPTS", "3")))
    confidence_auto_accept: float = Field(
        default=float(os.getenv("CONFIDENCE_AUTO_ACCEPT", "0.80"))
    )
    confidence_await_confirmation: float = Field(
        default=float(os.getenv("CONFIDENCE_AWAIT_CONFIRMATION", "0.50"))
    )
    poll_interval_seconds: float = Field(default=float(os.getenv("POLL_INTERVAL_SECONDS", "1.5")))
    worker_id: str = Field(default_factory=lambda: os.getenv("WORKER_ID", f"worker-{os.getpid()}"))


# Default global instance
default_config = WorkerConfig()
