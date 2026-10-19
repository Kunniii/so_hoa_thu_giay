"""Typed error definitions and standard error codes for Letter Worker."""

from typing import Any


class WorkerError(Exception):
    """Base exception for all letter worker errors."""

    def __init__(
        self,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.retryable = retryable

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "retryable": self.retryable,
        }


class InvalidImageError(WorkerError):
    """Raised when an image is corrupt, unsupported, or cannot be decoded."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            code="INVALID_IMAGE",
            message=message,
            details=details,
            retryable=False,
        )


class ImageTooLargeError(WorkerError):
    """Raised when an image file size or pixel count exceeds allowed limits."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            code="IMAGE_TOO_LARGE",
            message=message,
            details=details,
            retryable=False,
        )


class PaperNotDetectedError(WorkerError):
    """Raised when paper corners cannot be automatically detected with high confidence."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            code="PAPER_NOT_DETECTED",
            message=message,
            details=details,
            retryable=False,
        )


class InvalidCornersError(WorkerError):
    """Raised when supplied or detected corners form a degenerate or invalid quadrilateral."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            code="INVALID_CORNERS",
            message=message,
            details=details,
            retryable=False,
        )


class RectificationError(WorkerError):
    """Raised when perspective rectification computation fails."""

    def __init__(
        self, message: str, details: dict[str, Any] | None = None, retryable: bool = False
    ):
        super().__init__(
            code="RECTIFICATION_FAILED",
            message=message,
            details=details,
            retryable=retryable,
        )


class ProcessingTimeoutError(WorkerError):
    """Raised when processing exceeds maximum allowed execution time."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            code="PROCESSING_TIMEOUT",
            message=message,
            details=details,
            retryable=False,
        )


class OutputValidationError(WorkerError):
    """Raised when output image fails verification or quality checks."""

    def __init__(self, message: str, details: dict[str, Any] | None = None, retryable: bool = True):
        super().__init__(
            code="OUTPUT_VALIDATION_FAILED",
            message=message,
            details=details,
            retryable=retryable,
        )


class StorageSecurityError(WorkerError):
    """Raised when a file path violates sandbox boundaries or fails I/O operations."""

    def __init__(
        self, message: str, details: dict[str, Any] | None = None, retryable: bool = False
    ):
        super().__init__(
            code="STORAGE_ERROR",
            message=message,
            details=details,
            retryable=retryable,
        )


class InternalWorkerError(WorkerError):
    """Raised on unexpected worker failures."""

    def __init__(
        self, message: str, details: dict[str, Any] | None = None, retryable: bool = False
    ):
        super().__init__(
            code="INTERNAL_ERROR",
            message=message,
            details=details,
            retryable=retryable,
        )
