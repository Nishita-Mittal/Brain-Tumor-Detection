"""
Skull stripping v4 — Distance Transform based.
Breaks narrow brain-face connections at skull base.
"""
import os, cv2, numpy as np


def create_brain_mask_v4(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1: Head boundary
    _, tissue = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(tissue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones_like(gray) * 255
    head_cnt = max(contours, key=cv2.contourArea)
    head_mask = np.zeros_like(gray)
    cv2.drawContours(head_mask, [head_cnt], -1, 255, cv2.FILLED)

    _, _, bw, bh = cv2.boundingRect(head_cnt)
    head_diam = max(bw, bh)

    # 2: Erode to remove skull
    ero_px = max(int(head_diam * 0.08), 8)
    ero_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ero_px*2+1, ero_px*2+1))
    eroded = cv2.erode(head_mask, ero_k)

    # 3: Intensity threshold within eroded region
    vals = blurred[eroded > 0]
    if len(vals) == 0:
        return eroded
    otsu, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = max(int(otsu * 0.4), 15)
    brain_t = np.zeros_like(gray)
    brain_t[(blurred > thresh) & (eroded > 0)] = 255

    # Close gaps
    brain_t = cv2.morphologyEx(brain_t, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))

    # 4: Distance transform — break narrow connections
    dist = cv2.distanceTransform(brain_t, cv2.DIST_L2, 5)
    break_t = max(ero_px * 0.8, dist.max() * 0.08, 5)
    separated = brain_t.copy()
    separated[dist < break_t] = 0
    separated = cv2.morphologyEx(separated, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # 5: Pick brain component (large + round + upper-center)
    contours, _ = cv2.findContours(separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return brain_t

    tx, ty = w * 0.5, h * 0.35
    best, bscore = None, -999
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a < h * w * 0.005:
            continue
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx, cy = M['m10']/M['m00'], M['m01']/M['m00']
        p = cv2.arcLength(cnt, True)
        circ = (4 * np.pi * a / p**2) if p > 0 else 0
        d = np.sqrt((cx-tx)**2 + (cy-ty)**2) / np.sqrt(w**2+h**2)
        s = (a/(h*w))*5 + circ + (1-d)
        if s > bscore:
            bscore, best = s, cnt

    if best is None:
        return brain_t

    # 6: Recover edges (dilate core slightly within tissue)
    core = np.zeros_like(gray)
    cv2.drawContours(core, [best], -1, 255, cv2.FILLED)
    rec_px = max(int(break_t * 0.7), 3)
    rec_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rec_px*2+1, rec_px*2+1))
    recovered = cv2.bitwise_and(cv2.dilate(core, rec_k), brain_t)

    # 7: Fill holes
    flood = recovered.copy()
    cv2.floodFill(flood, np.zeros((h+2, w+2), np.uint8), (0, 0), 255)
    recovered = cv2.bitwise_or(recovered, cv2.bitwise_not(flood))

    # 8: Smooth
    recovered = cv2.GaussianBlur(recovered, (9, 9), 0)
    _, recovered = cv2.threshold(recovered, 127, 255, cv2.THRESH_BINARY)
    return recovered


def skull_strip_v4(img):
    mask = create_brain_mask_v4(img)
    if len(img.shape) == 3:
        return cv2.bitwise_and(img, cv2.merge([mask, mask, mask]))
    return cv2.bitwise_and(img, mask)


if __name__ == '__main__':
    samples = [
        ("glioma", "data/Training/glioma/Tr-glTr_0000.jpg"),
        ("glioma", "data/Training/glioma/Tr-glTr_0001.jpg"),
        ("glioma", "data/Training/glioma/Tr-glTr_0002.jpg"),
        ("meningioma", "data/Training/meningioma/Tr-meTr_0000.jpg"),
        ("meningioma", "data/Training/meningioma/Tr-meTr_0001.jpg"),
        ("notumor", "data/Training/notumor/Tr-noTr_0000.jpg"),
        ("notumor", "data/Training/notumor/Tr-noTr_0001.jpg"),
        ("pituitary", "data/Training/pituitary/Tr-piTr_0000.jpg"),
        ("pituitary", "data/Training/pituitary/Tr-piTr_0001.jpg"),
    ]
    out = "outputs/strip_v4"
    os.makedirs(out, exist_ok=True)
    for cls, p in samples:
        if not os.path.isfile(p):
            continue
        img = cv2.imread(p)
        s = skull_strip_v4(img)
        comb = np.hstack([img, s])
        cv2.putText(comb, "ORIGINAL", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(comb, "V4", (img.shape[1]+10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        name = f"{cls}_{os.path.basename(p)}"
        cv2.imwrite(os.path.join(out, name), comb)
        print(f"OK: {name}")
    print("Done!")
