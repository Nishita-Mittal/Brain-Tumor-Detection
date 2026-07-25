import os
import cv2
import numpy as np
import joblib
import torch
import argparse
import torchvision.transforms as T

from super_hybrid_model import SuperHybridModel
from skull_strip import skull_strip_image_adaptive

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model(weights_path):
    """Load the Super-Hybrid model with saved weights."""
    model = SuperHybridModel(num_classes=len(CLASSES))
    state = torch.load(weights_path, map_location=DEVICE)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE)
    model.eval()
    return model


def extract_feature(model, img):
    """Extract 1024-D feature from a BGR numpy image."""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inp = transform(img_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _, fused = model(inp)
    return fused.cpu().numpy().flatten()


def predict_proba(clf_data, feature):
    """Predict class probabilities using SVM classifier."""
    scaler = clf_data['scaler']
    classifier = clf_data['classifier']
    feat_scaled = scaler.transform([feature])
    return classifier.predict_proba(feat_scaled)[0]


def create_brain_mask(img):
    """
    Create a binary mask for XAI heatmap overlay.
    Uses same v4 distance-transform algorithm as skull_strip.py,
    with extra erosion for cleaner heatmap boundaries.
    """
    from skull_strip import create_brain_mask as _base_mask

    mask = _base_mask(img)

    h, w = mask.shape[:2]
    head_size = max(h, w)
    extra_ero = max(int(head_size * 0.02), 3)
    ero_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (extra_ero * 2 + 1, extra_ero * 2 + 1)
    )
    mask = cv2.erode(mask, ero_k, iterations=1)

    return mask


def _single_scale_occlusion(img, stripped_img, model, clf_data, brain_mask,
                            target_class, base_p,
                            patch_size, stride, scale_label=""):
    """
    Run occlusion at one scale using ZERO-FILL masking.
    Returns raw heatmap (float32, same size as img).

    Uses stripped_img for model predictions (matches training pipeline).
    Zero-fill creates strong signal → clear heatmaps on tumor regions.
    """
    h, w = stripped_img.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    counts  = np.zeros((h, w), dtype=np.float32)

    patch_positions = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y2 = min(y + patch_size, h)
            x2 = min(x + patch_size, w)
            patch_mask = brain_mask[y:y2, x:x2]
            if patch_mask.mean() < 50:
                continue
            patch_positions.append((y, x, y2, x2))

    total = len(patch_positions)
    print(f"  Scale {scale_label} (patch={patch_size}, stride={stride}): {total} patches")

    for i, (y, x, y2, x2) in enumerate(patch_positions):
        img_occl = stripped_img.copy()
        img_occl[y:y2, x:x2] = 0

        feat = extract_feature(model, img_occl)
        probs = predict_proba(clf_data, feat)
        drop = base_p - probs[target_class]

        drop = max(drop, 0.0)

        heatmap[y:y2, x:x2] += drop
        counts[y:y2, x:x2]  += 1

        done = i + 1
        if done % 50 == 0 or done == total:
            print(f"    Progress: {done}/{total} ({done*100//total}%)")

    counts[counts == 0] = 1
    heatmap = heatmap / counts

    return heatmap


def occlusion_map(image_path, model, clf_data, out_path=None,
                  patch_size=32, stride=16):
    """
    Generate a PRECISE occlusion-based heatmap using:
      1) Gaussian blur masking (eliminates edge bias)
      2) Multi-scale fusion (3 scales for precision + context)
      3) Positive-only drops (only truly important regions)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f'Cannot read image: {image_path}')

    h, w = img.shape[:2]

    brain_mask = create_brain_mask(img)
    mask_float = (brain_mask / 255.0).astype(np.float32)
    print(f"Brain mask: {np.count_nonzero(brain_mask)}/{h*w} pixels inside brain")

    stripped_img, _ = skull_strip_image_adaptive(img)

    base_feat = extract_feature(model, stripped_img)
    base_probs = predict_proba(clf_data, base_feat)
    target_class = np.argmax(base_probs)
    base_p = base_probs[target_class]
    print(f"Baseline: {CLASSES[target_class].upper()} ({base_p*100:.1f}%)\n")

    scales = [
        (12,  6,  "FINE"),
        (24, 12,  "MEDIUM"),
        (40, 20,  "COARSE"),
    ]

    heatmaps = []
    weights = [0.5, 0.3, 0.2]

    for (ps, st, label) in scales:
        hm = _single_scale_occlusion(
            img, stripped_img, model, clf_data, brain_mask,
            target_class, base_p,
            patch_size=ps, stride=st, scale_label=label
        )
        hm = hm * mask_float
        brain_vals = hm[brain_mask > 0]
        if len(brain_vals) > 0 and brain_vals.max() > 0:
            hm = hm / brain_vals.max()
        heatmaps.append(hm)

    fused = np.zeros((h, w), dtype=np.float32)
    for hm, wt in zip(heatmaps, weights):
        fused += wt * hm

    fused = fused * mask_float

    fused = cv2.GaussianBlur(fused, (31, 31), sigmaX=0)
    fused = fused * mask_float

    brain_values = fused[brain_mask > 0]
    if len(brain_values) > 0:
        low  = np.percentile(brain_values, 5)
        high = np.percentile(brain_values, 95)
    else:
        low, high = 0, 1

    hm = np.clip(fused, low, high)
    hm = hm - hm.min()
    if hm.max() > 0:
        hm = hm / hm.max()

    hm = np.power(hm, 1.5)

    hm = hm * mask_float

    hm_uint8 = (hm * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)

    mask_3ch = cv2.merge([brain_mask, brain_mask, brain_mask])
    overlay = cv2.addWeighted(img, 0.5, hm_color, 0.5, 0)
    overlay = np.where(mask_3ch > 0, overlay, img)

    if out_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join('outputs', f'occlusion_{base_name}.jpg')

    os.makedirs(os.path.dirname(out_path) or 'outputs', exist_ok=True)
    cv2.imwrite(out_path, overlay)
    print(f'\nOcclusion heatmap saved -> {out_path}')
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate occlusion-based XAI heatmap')
    parser.add_argument('image', type=str, help='Path to MRI image')
    parser.add_argument('--weights', type=str, default='models/super_hybrid.pth',
                        help='Path to Super-Hybrid model weights')
    parser.add_argument('--clf', type=str, default='models/classifier.joblib',
                        help='Path to SVM classifier (.joblib)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for heatmap image')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f'Image not found: {args.image}')

    print("Loading Super-Hybrid model...")
    model = load_model(args.weights)

    print("Loading SVM classifier...")
    clf_data = joblib.load(args.clf)

    occlusion_map(args.image, model, clf_data,
                  out_path=args.output)

