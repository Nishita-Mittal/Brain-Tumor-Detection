"""
Skull-Stripping Preprocessing Script (v4)
==========================================
Removes skull, scalp, face, and background from MRI images,
keeping only brain tissue. Uses distance-transform-based
separation to break narrow brain-face connections at skull base.

Usage:
    python src/skull_strip.py

Input:  data/Training/  and  data/Testing/
Output: data/Training_stripped/  and  data/Testing_stripped/
"""

import os
import cv2
import numpy as np
from tqdm import tqdm


_VIEW_MODEL = None
_VIEW_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "models", "view_detector.pth")


def _load_view_detector():
    global _VIEW_MODEL
    if _VIEW_MODEL is not None:
        return _VIEW_MODEL
    if not os.path.isfile(_VIEW_MODEL_PATH):
        return None
    try:
        from view_detector import load_view_detector
        _VIEW_MODEL = load_view_detector(_VIEW_MODEL_PATH)
    except Exception:
        _VIEW_MODEL = None
    return _VIEW_MODEL


def _basic_tissue_mask(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, tissue = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(tissue, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, None
    head_cnt = max(contours, key=cv2.contourArea)
    head_mask = np.zeros_like(gray)
    cv2.drawContours(head_mask, [head_cnt], -1, 255, cv2.FILLED)
    x, y, bw, bh = cv2.boundingRect(head_cnt)
    head_diam = max(bw, bh)
    return blurred, head_mask, (x, y, bw, bh), head_diam


def create_brain_mask(img):
    """
    Create a binary mask that isolates brain tissue.

    Distance-transform approach (axial-friendly). This is the default and
    matches the previous v4 behavior.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h, w = gray.shape
    blurred, head_mask, rect, head_diam = _basic_tissue_mask(gray)
    if head_mask is None:
        return np.ones_like(gray) * 255

    ero_px = max(int(head_diam * 0.08), 8)
    ero_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ero_px * 2 + 1, ero_px * 2 + 1)
    )
    eroded = cv2.erode(head_mask, ero_k, iterations=1)

    vals = blurred[eroded > 0]
    if len(vals) == 0:
        return eroded

    otsu_val, _ = cv2.threshold(vals, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tissue_thresh = max(int(otsu_val * 0.4), 15)

    brain_tissue = np.zeros_like(gray)
    brain_tissue[(blurred > tissue_thresh) & (eroded > 0)] = 255

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    brain_tissue = cv2.morphologyEx(brain_tissue, cv2.MORPH_CLOSE, close_k)

    dist = cv2.distanceTransform(brain_tissue, cv2.DIST_L2, 5)
    dist_max = min(float(dist.max()), float(head_diam))
    break_thresh = float(max(ero_px * 0.8, dist_max * 0.08, 5))

    separated = brain_tissue.copy()
    separated[dist < break_thresh] = 0

    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    separated = cv2.morphologyEx(separated, cv2.MORPH_OPEN, open_k)

    contours, _ = cv2.findContours(separated, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return brain_tissue

    target_x = w * 0.5
    target_y = h * 0.35

    best_cnt = None
    best_score = -999

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < h * w * 0.005:
            continue
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']

        perimeter = cv2.arcLength(cnt, True)
        circ = (4 * np.pi * area / perimeter**2) if perimeter > 0 else 0

        d = np.sqrt((cx - target_x)**2 + (cy - target_y)**2)
        max_d = np.sqrt(w**2 + h**2)

        score = (area / (h * w)) * 5 + circ + (1 - d / max_d)

        if score > best_score:
            best_score = score
            best_cnt = cnt

    if best_cnt is None:
        return brain_tissue

    core = np.zeros_like(gray)
    cv2.drawContours(core, [best_cnt], -1, 255, cv2.FILLED)

    rec_size = int(max(break_thresh * 0.7, 3)) * 2 + 1
    rec_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rec_size, rec_size))
    recovered = cv2.bitwise_and(cv2.dilate(core, rec_k), brain_tissue)

    flood = recovered.copy()
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask_ff, (0, 0), 255)
    recovered = cv2.bitwise_or(recovered, cv2.bitwise_not(flood))

    recovered = cv2.GaussianBlur(recovered, (9, 9), 0)
    _, recovered = cv2.threshold(recovered, 127, 255, cv2.THRESH_BINARY)

    return recovered


def create_brain_mask_sagittal(img):
    """Create brain mask for sagittal views (left/right erosion)."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    blurred, head_mask, rect, head_diam = _basic_tissue_mask(gray)
    if head_mask is None:
        return np.ones_like(gray) * 255

    h, w = gray.shape
    x, y, bw, bh = rect
    band_pad = int(bw * 0.18)
    roi = head_mask.copy()
    roi[:, :x + band_pad] = 0
    roi[:, x + bw - band_pad:] = 0

    ero_px = max(int(head_diam * 0.07), 6)
    ero_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ero_px * 2 + 1, ero_px * 2 + 1)
    )
    eroded = cv2.erode(roi, ero_k, iterations=1)

    vals = blurred[eroded > 0]
    if len(vals) == 0:
        return eroded

    otsu_val, _ = cv2.threshold(vals, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tissue_thresh = max(int(otsu_val * 0.45), 15)

    brain_tissue = np.zeros_like(gray)
    brain_tissue[(blurred > tissue_thresh) & (eroded > 0)] = 255

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    brain_tissue = cv2.morphologyEx(brain_tissue, cv2.MORPH_CLOSE, close_k)

    contours, _ = cv2.findContours(brain_tissue, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return brain_tissue

    best_cnt = max(contours, key=cv2.contourArea)
    core = np.zeros_like(gray)
    cv2.drawContours(core, [best_cnt], -1, 255, cv2.FILLED)

    core = cv2.GaussianBlur(core, (7, 7), 0)
    _, core = cv2.threshold(core, 127, 255, cv2.THRESH_BINARY)
    return core


def create_brain_mask_coronal(img):
    """Create brain mask for coronal views (side + top erosion)."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    blurred, head_mask, rect, head_diam = _basic_tissue_mask(gray)
    if head_mask is None:
        return np.ones_like(gray) * 255

    h, w = gray.shape
    x, y, bw, bh = rect
    side_pad = int(bw * 0.15)
    top_pad = int(bh * 0.12)
    roi = head_mask.copy()
    roi[:, :x + side_pad] = 0
    roi[:, x + bw - side_pad:] = 0
    roi[:y + top_pad, :] = 0

    ero_px = max(int(head_diam * 0.07), 6)
    ero_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ero_px * 2 + 1, ero_px * 2 + 1)
    )
    eroded = cv2.erode(roi, ero_k, iterations=1)

    vals = blurred[eroded > 0]
    if len(vals) == 0:
        return eroded

    otsu_val, _ = cv2.threshold(vals, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tissue_thresh = max(int(otsu_val * 0.45), 15)

    brain_tissue = np.zeros_like(gray)
    brain_tissue[(blurred > tissue_thresh) & (eroded > 0)] = 255

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    brain_tissue = cv2.morphologyEx(brain_tissue, cv2.MORPH_CLOSE, close_k)

    contours, _ = cv2.findContours(brain_tissue, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return brain_tissue

    best_cnt = max(contours, key=cv2.contourArea)
    core = np.zeros_like(gray)
    cv2.drawContours(core, [best_cnt], -1, 255, cv2.FILLED)

    core = cv2.GaussianBlur(core, (7, 7), 0)
    _, core = cv2.threshold(core, 127, 255, cv2.THRESH_BINARY)
    return core


def skull_strip_image(img):
    """Apply axial skull-stripping (legacy behavior)."""
    mask = create_brain_mask(img)

    if len(img.shape) == 3:
        mask_3ch = cv2.merge([mask, mask, mask])
        stripped = cv2.bitwise_and(img, mask_3ch)
    else:
        stripped = cv2.bitwise_and(img, mask)

    return stripped


def skull_strip_image_adaptive(img, view=None):
    """Skull-strip with view-aware masks. Falls back to axial when unsure."""
    detected_view = view
    if detected_view is None:
        model = _load_view_detector()
        if model is not None:
            try:
                from view_detector import detect_view
                detected_view, conf = detect_view(img, model)
                if conf < 0.6:
                    detected_view = None
            except Exception:
                detected_view = None

    if detected_view == "sagittal":
        mask = create_brain_mask_sagittal(img)
    elif detected_view == "coronal":
        mask = create_brain_mask_coronal(img)
    else:
        mask = create_brain_mask(img)

    if len(img.shape) == 3:
        mask_3ch = cv2.merge([mask, mask, mask])
        stripped = cv2.bitwise_and(img, mask_3ch)
    else:
        stripped = cv2.bitwise_and(img, mask)

    return stripped, detected_view or "axial"


def process_directory(input_dir, output_dir):
    """Process all images in a directory tree, preserving folder structure."""
    if not os.path.isdir(input_dir):
        print(f"  [SKIP] Directory not found: {input_dir}")
        return 0

    total = 0
    failed = 0
    classes = sorted([d for d in os.listdir(input_dir)
                      if os.path.isdir(os.path.join(input_dir, d))])

    print(f"\n  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Classes: {classes}\n")

    for cls in classes:
        cls_input = os.path.join(input_dir, cls)
        cls_output = os.path.join(output_dir, cls)
        os.makedirs(cls_output, exist_ok=True)

        files = [f for f in os.listdir(cls_input)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]

        for fname in tqdm(files, desc=f"  [{cls}]"):
            src_path = os.path.join(cls_input, fname)
            dst_path = os.path.join(cls_output, fname)

            img = cv2.imread(src_path)
            if img is None:
                failed += 1
                continue

            stripped, _ = skull_strip_image_adaptive(img)
            cv2.imwrite(dst_path, stripped)
            total += 1

    return total


def main():
    print("=" * 55)
    print("  SKULL-STRIPPING PREPROCESSOR (v4)")
    print("  Distance-transform based brain extraction")
    print("=" * 55)

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("\n[1/2] Processing TRAINING images...")
    t1 = process_directory(
        os.path.join(base, "data", "Training"),
        os.path.join(base, "data", "Training_stripped")
    )

    print("\n[2/2] Processing TESTING images...")
    t2 = process_directory(
        os.path.join(base, "data", "Testing"),
        os.path.join(base, "data", "Testing_stripped")
    )

    print("\n" + "=" * 55)
    print(f"  DONE! Processed {t1 + t2} images total")
    print(f"  Training: {t1} images → data/Training_stripped/")
    print(f"  Testing:  {t2} images → data/Testing_stripped/")
    print("=" * 55)
    print("\nNext step: Retrain model on stripped data:")
    print("  python src/train_cnn.py --train-dir data/Training_stripped --val-dir data/Testing_stripped")


if __name__ == "__main__":
    main()

