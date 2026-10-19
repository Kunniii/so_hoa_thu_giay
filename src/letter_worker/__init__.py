"""Letter processing worker and high-performance document imaging engine.

Specialized in handwritten letter and document image enhancement:
- Automatic paper corner detection and interactive override.
- True 3D perspective rectification using vanishing geometry (MSR-TR-2003-39).
- Morphological illumination flattening without halo artifacts.
- Bleed-through / show-through text suppression for thin notebook paper.
- Aesthetic restoration presets: Pure B&W, Natural, and Vintage Parchment.
- Quality assurance checks (blur, ink coverage, dimensions).
- Standalone Python API, transactional SQLite worker daemon, and CLI.
"""

from letter_worker.config import WorkerConfig, default_config
from letter_worker.errors import (
    ImageTooLargeError,
    InternalWorkerError,
    InvalidCornersError,
    InvalidImageError,
    OutputValidationError,
    PaperNotDetectedError,
    ProcessingTimeoutError,
    RectificationError,
    StorageSecurityError,
    WorkerError,
)
from letter_worker.processor import LetterProcessor, process_letter
from letter_worker.queue import SQLiteJobQueue
from letter_worker.schemas import (
    JobProgressEvent,
    JobResult,
    JobStage,
    LetterJob,
    Point,
    QualityCheckResult,
    RestoreSettings,
)

__version__ = "0.2.0"

__all__ = [
    # Top-level API
    "LetterProcessor",
    "process_letter",
    "SQLiteJobQueue",
    "WorkerConfig",
    "default_config",
    # Data models
    "LetterJob",
    "JobResult",
    "JobStage",
    "JobProgressEvent",
    "Point",
    "QualityCheckResult",
    "RestoreSettings",
    # Errors
    "WorkerError",
    "InvalidImageError",
    "ImageTooLargeError",
    "PaperNotDetectedError",
    "InvalidCornersError",
    "RectificationError",
    "InternalWorkerError",
    "ProcessingTimeoutError",
    "OutputValidationError",
    "StorageSecurityError",
    # Version
    "__version__",
]
