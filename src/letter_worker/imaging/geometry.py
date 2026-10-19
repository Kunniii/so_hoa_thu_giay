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


STANDARD_ASPECT_RATIOS: dict[str, float] = {
    "a4": 1.0 / 1.41421356,  # ~0.7071 (Width / Height)
    "notebook": 1.0 / 1.35,  # ~0.7407 (Vietnamese student notebook / B5)
    "letter": 8.5 / 11.0,  # ~0.7727 (US Letter standard)
}


def estimate_3d_aspect_ratio(
    ordered_corners: np.ndarray,
    image_shape: tuple[int, ...] | None = None,
) -> float | None:
    """Estimate true physical aspect ratio (width / height) of a 3D rectangle from perspective view.

    Implements Zhang's single-view rectangle geometry (Microsoft Research MSR-TR-2003-39).
    Uses vanishing geometry to solve for camera focal length and true physical aspect ratio,
    preventing horizontal or vertical stretching under tilted camera angles.

    Args:
        ordered_corners: Shape (4, 2) array ordered [TL, TR, BR, BL].
        image_shape: (H, W) or (H, W, C) of original image for principal point center.

    Returns:
        float: True physical aspect ratio (width / height), or None if geometry is degenerate.
    """
    tl, tr, br, bl = ordered_corners
    if image_shape is not None:
        h, w = image_shape[:2]
        u0, v0 = w / 2.0, h / 2.0
    else:
        u0 = float(np.mean(ordered_corners[:, 0]))
        v0 = float(np.mean(ordered_corners[:, 1]))

    # Zhang coordinate indexing: p1=TL, p2=TR, p3=BL, p4=BR relative to optical center
    p1 = np.array([tl[0] - u0, tl[1] - v0, 1.0], dtype=np.float64)
    p2 = np.array([tr[0] - u0, tr[1] - v0, 1.0], dtype=np.float64)
    p3 = np.array([bl[0] - u0, bl[1] - v0, 1.0], dtype=np.float64)
    p4 = np.array([br[0] - u0, br[1] - v0, 1.0], dtype=np.float64)

    # Cross products
    cp_14 = np.cross(p1, p4)
    cp_24 = np.cross(p2, p4)
    cp_34 = np.cross(p3, p4)

    denom_k2 = float(np.dot(cp_24, p3))
    denom_k3 = float(np.dot(cp_34, p2))

    if abs(denom_k2) < 1e-9 or abs(denom_k3) < 1e-9:
        return None

    k2 = float(np.dot(cp_14, p3)) / denom_k2
    k3 = float(np.dot(cp_14, p2)) / denom_k3

    # Direction vectors along horizontal and vertical perspective edges
    n2 = k2 * p2 - p1
    n3 = k3 * p3 - p1

    denom_f2 = n2[2] * n3[2]
    # Solve for squared focal length f^2 via orthogonality: (n2 . n3) = 0 in 3D
    if abs(denom_f2) >= 1e-8:
        f2 = -(n2[0] * n3[0] + n2[1] * n3[1]) / denom_f2
    else:
        f2 = -1.0

    # If f2 <= 0 (edges near parallel / frontal), use nominal focal length of camera
    if f2 <= 0.0 or not math.isfinite(f2):
        f0 = math.hypot(w, h)
        f2 = f0 ** 2

    # Recover aspect ratio (width / height)
    num = n2[0] ** 2 + n2[1] ** 2 + f2 * (n2[2] ** 2)
    den = n3[0] ** 2 + n3[1] ** 2 + f2 * (n3[2] ** 2)

    if den <= 0.0 or num <= 0.0:
        return None

    ratio = math.sqrt(num / den)
    # Sanity bounds check for document aspect ratio (0.3 to 3.0)
    if not (0.3 <= ratio <= 3.0):
        return None

    return float(ratio)


def estimate_rectified_dimensions(
    ordered_corners: np.ndarray,
    image_shape: tuple[int, ...] | None = None,
    aspect_ratio_mode: str = "auto",
    max_dimension: int = 8192,
    min_dimension: int = 100,
) -> tuple[int, int]:
    """Calculate target width and height for perspective rectification.

    Avoids horizontal/vertical stretching under tilted camera angles by recovering the
    true physical aspect ratio or applying a standard document ratio.

    Args:
        ordered_corners: Shape (4, 2) ordered [TL, TR, BR, BL].
        image_shape: Source image shape (H, W, ...).
        aspect_ratio_mode: 'auto' (3D perspective recovery), 'a4', 'notebook', 'letter', or 'free'.
        max_dimension: Upper cap on width and height.
        min_dimension: Minimum allowed dimension.

    Returns:
        (width, height) in pixels.
    """
    tl, tr, br, bl = ordered_corners

    top_w = float(np.linalg.norm(tr - tl))
    bottom_w = float(np.linalg.norm(br - bl))
    raw_w = max(top_w, bottom_w)

    left_h = float(np.linalg.norm(bl - tl))
    right_h = float(np.linalg.norm(br - tr))
    raw_h = max(left_h, right_h)

    # 1. Determine target aspect ratio (W / H)
    target_ratio: float | None = None

    if aspect_ratio_mode in STANDARD_ASPECT_RATIOS:
        target_ratio = STANDARD_ASPECT_RATIOS[aspect_ratio_mode]

    elif aspect_ratio_mode == "auto":
        # Recover 3D physical aspect ratio using Zhang's perspective geometry
        ratio_3d = estimate_3d_aspect_ratio(ordered_corners, image_shape=image_shape)
        if ratio_3d is not None:
            target_ratio = ratio_3d
        else:
            # Fallback if image was taken head-on (frontal, no perspective convergence):
            # Check Euclidean ratio against standards; if close, snap to nearest standard,
            # otherwise use the Euclidean ratio.
            euclidean_ratio = raw_w / max(raw_h, 1.0)
            best_standard = None
            best_diff = float("inf")
            for std_ratio in STANDARD_ASPECT_RATIOS.values():
                diff = abs(euclidean_ratio - std_ratio)
                if diff < best_diff and diff < 0.08:
                    best_diff = diff
                    best_standard = std_ratio
            target_ratio = best_standard if best_standard is not None else euclidean_ratio

    elif aspect_ratio_mode == "free":
        target_ratio = None  # Naive 2D Euclidean

    # 2. Compute (width, height) based on anchor edge and target aspect ratio
    if target_ratio is not None and target_ratio > 0:
        if target_ratio <= 1.0:
            # Portrait:
            # Under pitch tilt, raw_h is foreshortened. Anchor to the longer dimension
            # ensuring neither dimension is compressed below maximum observed detail.
            height = max(raw_h, raw_w / target_ratio)
            width = height * target_ratio
        else:
            # Landscape:
            width = max(raw_w, raw_h * target_ratio)
            height = width / target_ratio

        width = max(int(round(width)), min_dimension)
        height = max(int(round(height)), min_dimension)
    else:
        # Unconstrained 2D Euclidean (naive)
        width = max(int(round(raw_w)), min_dimension)
        height = max(int(round(raw_h)), min_dimension)

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
    aspect_ratio_mode: str = "auto",
    max_dimension: int = 8192,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Rectify perspective of the region defined by corners into an upright rectangular image.

    Args:
        image: Source image numpy array (H, W, C) or (H, W).
        corners: 4 corners in source image coordinates.
        aspect_ratio_mode: Aspect ratio constraint mode ('auto', 'a4', 'notebook', 'letter', 'free').
        max_dimension: Safety limit on output resolution.

    Returns:
        (rectified_image, (out_width, out_height))

    Raises:
        InvalidCornersError: If corners are degenerate or invalid.
        RectificationError: If transformation matrix computation fails.
    """
    ordered = order_corners(corners)
    validate_quadrilateral(ordered, image_shape=image.shape)

    out_w, out_h = estimate_rectified_dimensions(
        ordered,
        image_shape=image.shape,
        aspect_ratio_mode=aspect_ratio_mode,
        max_dimension=max_dimension,
    )

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
