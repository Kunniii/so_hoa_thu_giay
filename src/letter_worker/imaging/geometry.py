"""Geometric operations, corner ordering, validation, and perspective rectification."""

import math

import cv2
import numpy as np

from letter_worker.errors import InvalidCornersError, RectificationError


def order_corners(points: np.ndarray | list) -> np.ndarray:
    """Order 4 corner coordinates in clockwise sequence: [top-left, top-right, bottom-right, bottom-left].

    Args:
        points: Array or list of shape (4, 2) or (4, 1, 2).

    Returns:
        np.ndarray of shape (4, 2) dtype float32 ordered [TL, TR, BR, BL].

    Raises:
        InvalidCornersError: If points array does not contain exactly 4 points.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise InvalidCornersError(f"Expected exactly 4 points, got {pts.shape[0]}")

    # Compute centroid
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))

    # Sort points clockwise by angle relative to centroid
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    clockwise_indices = np.argsort(angles)
    sorted_pts = pts[clockwise_indices]

    # Find which of the 4 points corresponds to top-left.
    # For a document, the top edge should head in positive X direction (left to right),
    # and the left edge should head in positive Y direction (top to bottom).
    best_score = -float("inf")
    best_start = 0

    for i in range(4):
        p0 = sorted_pts[i]
        p1 = sorted_pts[(i + 1) % 4]
        p2 = sorted_pts[(i + 2) % 4]
        p3 = sorted_pts[(i + 3) % 4]

        # Score alignment with upright orientation:
        # p0 -> p1 is top edge (should have positive dx)
        # p1 -> p2 is right edge (should have positive dy)
        # p3 -> p2 is bottom edge (should have positive dx)
        # p0 -> p3 is left edge (should have positive dy)
        top_dx = p1[0] - p0[0]
        right_dy = p2[1] - p1[1]
        bottom_dx = p2[0] - p3[0]
        left_dy = p3[1] - p0[1]

        # Also penalize large distance of p0 from origin
        origin_dist = math.hypot(p0[0], p0[1])

        score = (top_dx + right_dy + bottom_dx + left_dy) - 0.25 * origin_dist
        if score > best_score:
            best_score = score
            best_start = i

    ordered = np.roll(sorted_pts, -best_start, axis=0)
    return ordered.astype(np.float32)


def validate_quadrilateral(
    corners: np.ndarray,
    image_shape: tuple[int, ...] | None = None,
    min_area: float = 100.0,
) -> None:
    """Verify that 4 corners form a valid, non-degenerate, convex quadrilateral.

    Args:
        corners: Ordered or unordered points of shape (4, 2).
        image_shape: Optional (H, W) or (H, W, C) to ensure points are within reasonable bounds.
        min_area: Minimum acceptable pixel area.

    Raises:
        InvalidCornersError: If invalid or degenerate.
    """
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise InvalidCornersError(f"Quadrilateral must have 4 points, got {pts.shape[0]}")

    # Check for NaN or Inf
    if not np.isfinite(pts).all():
        raise InvalidCornersError("Corners contain NaN or Inf values")

    # Check uniqueness of points (distance between any pair > 5px)
    for i in range(4):
        for j in range(i + 1, 4):
            dist = float(np.linalg.norm(pts[i] - pts[j]))
            if dist < 5.0:
                raise InvalidCornersError(
                    f"Points {i} and {j} are almost identical (distance {dist:.2f}px)"
                )

    # Check contour area
    area = cv2.contourArea(pts.astype(np.float32))
    if area < min_area:
        raise InvalidCornersError(f"Quadrilateral area {area:.1f} is below minimum {min_area}")

    # Check convexity
    # cv2.isContourConvex requires int32 array of shape (N, 1, 2)
    int_pts = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(int_pts):
        raise InvalidCornersError("Quadrilateral is not convex (self-intersecting or concave)")

    # Bounds check if image_shape is provided
    if image_shape is not None:
        h, w = image_shape[:2]
        # Allow slight margin (e.g. 5% outside bounds)
        margin_x = w * 0.05
        margin_y = h * 0.05
        if (
            np.any(pts[:, 0] < -margin_x)
            or np.any(pts[:, 0] > w + margin_x)
            or np.any(pts[:, 1] < -margin_y)
            or np.any(pts[:, 1] > h + margin_y)
        ):
            raise InvalidCornersError("Corners lie significantly outside image boundaries")


def estimate_rectified_dimensions(
    ordered_corners: np.ndarray,
    max_dimension: int = 8192,
    min_dimension: int = 100,
) -> tuple[int, int]:
    """Calculate the target width and height for perspective rectification based on edge lengths.

    Args:
        ordered_corners: Shape (4, 2) ordered [TL, TR, BR, BL].
        max_dimension: Upper cap on width and height.
        min_dimension: Minimum allowed dimension.

    Returns:
        (width, height) in pixels.
    """
    tl, tr, br, bl = ordered_corners

    top_w = float(np.linalg.norm(tr - tl))
    bottom_w = float(np.linalg.norm(br - bl))
    width = max(int(round(max(top_w, bottom_w))), min_dimension)

    left_h = float(np.linalg.norm(bl - tl))
    right_h = float(np.linalg.norm(br - tr))
    height = max(int(round(max(left_h, right_h))), min_dimension)

    # Scale down proportionally if exceeding max_dimension
    max_side = max(width, height)
    if max_side > max_dimension:
        scale = max_dimension / float(max_side)
        width = max(int(round(width * scale)), min_dimension)
        height = max(int(round(height * scale)), min_dimension)

    return (width, height)


def rectify_perspective(
    image: np.ndarray,
    corners: np.ndarray,
    max_dimension: int = 8192,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Rectify perspective of the region defined by corners into an upright rectangular image.

    Args:
        image: Source image numpy array (H, W, C) or (H, W).
        corners: 4 corners in source image coordinates.
        max_dimension: Safety limit on output resolution.

    Returns:
        (rectified_image, (out_width, out_height))

    Raises:
        InvalidCornersError: If corners are degenerate or invalid.
        RectificationError: If transformation matrix computation fails.
    """
    ordered = order_corners(corners)
    validate_quadrilateral(ordered, image_shape=image.shape)

    out_w, out_h = estimate_rectified_dimensions(ordered, max_dimension=max_dimension)

    # Target rectangle destination points: [TL, TR, BR, BL]
    dst = np.array(
        [
            [0.0, 0.0],
            [float(out_w - 1), 0.0],
            [float(out_w - 1), float(out_h - 1)],
            [0.0, float(out_h - 1)],
        ],
        dtype=np.float32,
    )

    try:
        matrix = cv2.getPerspectiveTransform(ordered, dst)
        if matrix is None:
            raise RectificationError("Computed perspective transform matrix is None")

        rectified = cv2.warpPerspective(
            image,
            matrix,
            (out_w, out_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rectified, (out_w, out_h)
    except cv2.error as e:
        raise RectificationError(f"OpenCV perspective warp failed: {e}") from e
