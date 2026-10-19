"""Paper boundary and four-corner detection using multi-strategy contour analysis."""

from dataclasses import dataclass

import cv2
import numpy as np

from letter_worker.imaging.geometry import order_corners


@dataclass
class DetectionResult:
    """Outcome of paper corner detection."""

    corners: np.ndarray | None  # (4, 2) float32 in original image coordinates, or None
    confidence: float  # 0.0 to 1.0
    method: str
    candidates: list[np.ndarray]  # list of (4, 2) float32 candidate quads


def _score_quadrilateral(
    quad: np.ndarray,
    img_w: int,
    img_h: int,
) -> float:
    """Score a candidate 4-point quadrilateral based on geometry, angles, and area.

    Returns:
        Score between 0.0 and 1.0.
    """
    total_area = float(img_w * img_h)
    area = cv2.contourArea(quad)
    area_ratio = area / total_area

    # Paper should occupy reasonable portion of frame (15% to 92%)
    if area_ratio < 0.12 or area_ratio > 0.98:
        return 0.0

    # Must be convex
    int_pts = np.round(quad).astype(np.int32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(int_pts):
        return 0.0

    ordered = order_corners(quad)
    tl, tr, br, bl = ordered

    # Edge vectors
    edges = [tr - tl, br - tr, bl - br, tl - bl]
    lengths = [float(np.linalg.norm(e)) for e in edges]
    if any(length < 20.0 for length in lengths):
        return 0.0

    # 1. Corner angle orthogonality (dot product of adjacent normalized vectors)
    cosines = []
    for i in range(4):
        v1 = edges[i] / lengths[i]
        v2 = -edges[(i + 3) % 4] / lengths[(i + 3) % 4]
        cos_theta = float(np.dot(v1, v2))
        cosines.append(abs(cos_theta))

    max_cos = max(cosines)
    mean_cos = sum(cosines) / 4.0
    if max_cos > 0.65:  # Angle deviated too far from 90 degrees
        return 0.0

    angle_score = max(0.0, 1.0 - mean_cos * 2.0)

    # 2. Area score: ideal paper coverage is 30% - 85%
    if 0.30 <= area_ratio <= 0.85:
        area_score = 1.0
    elif area_ratio < 0.30:
        area_score = area_ratio / 0.30
    else:
        area_score = max(0.0, 1.0 - (area_ratio - 0.85) / 0.13)

    # 3. Aspect ratio score
    top_w = lengths[0]
    bot_w = lengths[2]
    left_h = lengths[3]
    right_h = lengths[1]
    avg_w = (top_w + bot_w) / 2.0
    avg_h = (left_h + right_h) / 2.0
    aspect = avg_w / max(avg_h, 1.0)
    # Typical letter aspect ratio between 0.5 (tall) and 2.0 (wide)
    if 0.5 <= aspect <= 2.0:
        aspect_score = 1.0
    else:
        aspect_score = max(0.0, 1.0 - min(abs(aspect - 0.5), abs(aspect - 2.0)))

    # 4. Opposite edge parallelism: top vs bottom, left vs right length ratio
    w_ratio = min(top_w, bot_w) / max(top_w, bot_w, 1.0)
    h_ratio = min(left_h, right_h) / max(left_h, right_h, 1.0)
    parallel_score = (w_ratio + h_ratio) / 2.0

    score = 0.35 * angle_score + 0.25 * area_score + 0.20 * aspect_score + 0.20 * parallel_score
    return float(np.clip(score, 0.0, 1.0))


def detect_paper_corners(
    image: np.ndarray,
    max_detect_dimension: int = 1200,
) -> DetectionResult:
    """Detect paper corners in an image using multi-pass contour and edge analysis.

    Args:
        image: RGB or grayscale image numpy array.
        max_detect_dimension: Scale down factor for faster detection.

    Returns:
        DetectionResult containing best corners (in original coordinate space) and confidence.
    """
    orig_h, orig_w = image.shape[:2]

    # Convert to grayscale
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Scale down for robust detection if large
    scale = 1.0
    max_dim = max(orig_h, orig_w)
    if max_dim > max_detect_dimension:
        scale = max_detect_dimension / float(max_dim)
        proc_w = int(round(orig_w * scale))
        proc_h = int(round(orig_h * scale))
        proc_gray = cv2.resize(gray, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    else:
        proc_w, proc_h = orig_w, orig_h
        proc_gray = gray

    # Bilateral and Gaussian smoothing to preserve document boundary while smoothing noise
    blurred = cv2.GaussianBlur(proc_gray, (5, 5), sigmaX=1.2)

    # Strategy 1: Dynamic Canny with morphological closing
    med = float(np.median(blurred))
    lower_th = int(max(0, 0.60 * med))
    upper_th = int(min(255, 1.35 * med))
    edges = cv2.Canny(blurred, lower_th, upper_th)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Strategy 2: Adaptive & Otsu threshold masks
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)

    binary_inv = cv2.bitwise_not(otsu_closed)

    candidate_quads: list[tuple[float, np.ndarray, str]] = []

    # Run contour detection across multiple representations
    representations = [
        ("canny", closed_edges),
        ("otsu", otsu_closed),
        ("otsu_inv", binary_inv),
    ]

    for method_name, mask in representations:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # Sort by area descending
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            if peri < 100:
                continue

            # Sweep multiple approximation epsilons
            for eps_factor in [0.015, 0.02, 0.025, 0.035]:
                approx = cv2.approxPolyDP(cnt, eps_factor * peri, True)
                if len(approx) == 4:
                    pts = approx.reshape(4, 2).astype(np.float32)
                    score = _score_quadrilateral(pts, proc_w, proc_h)
                    if score > 0.25:
                        candidate_quads.append((score, pts, method_name))

    if not candidate_quads:
        return DetectionResult(
            corners=None,
            confidence=0.0,
            method="none",
            candidates=[],
        )

    # Sort candidates by score descending
    candidate_quads.sort(key=lambda x: x[0], reverse=True)

    # Remove near-duplicates from candidate list
    unique_candidates: list[tuple[float, np.ndarray, str]] = []
    for sc, quad, meth in candidate_quads:
        is_dup = False
        for _, uq, _ in unique_candidates:
            diff = np.mean(np.abs(np.sort(quad, axis=0) - np.sort(uq, axis=0)))
            if diff < 15.0:
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append((sc, quad, meth))
        if len(unique_candidates) >= 5:
            break

    best_score, best_quad_proc, best_method = unique_candidates[0]

    # Margin bonus: if top candidate is significantly better than second, increase confidence
    if len(unique_candidates) > 1:
        second_score = unique_candidates[1][0]
        margin = max(0.0, best_score - second_score)
        confidence = float(np.clip(best_score + 0.15 * margin, 0.0, 1.0))
    else:
        confidence = best_score

    # Map candidate quad points back to original image space
    inv_scale = 1.0 / scale
    orig_candidates = [(c[1] * inv_scale).astype(np.float32) for c in unique_candidates]

    best_orig_corners = order_corners(best_quad_proc * inv_scale)

    return DetectionResult(
        corners=best_orig_corners,
        confidence=round(confidence, 3),
        method=best_method,
        candidates=orig_candidates,
    )
