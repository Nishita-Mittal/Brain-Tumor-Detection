"""
ScoreCAM Explainability Module
==============================
ScoreCAM with CBAM refinement for precise, gradient-free localization.

Usage:
    python src/gradcam_explain.py <image_path>
"""

import os
import cv2
import numpy as np
import joblib
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import argparse

from super_hybrid_model import SuperHybridModel
from skull_strip import skull_strip_image_adaptive, create_brain_mask

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model(weights_path):
    model = SuperHybridModel(num_classes=len(CLASSES))
    state = torch.load(weights_path, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE)
    model.eval()
    return model


def _extract_cbam_map(model):
    cbam_maps = {}

    def hook(_m, _i, out):
        cbam_maps["resnet_cbam"] = torch.sigmoid(out).detach()

    handle = model.branch1.cbam.spatial_attn.conv.register_forward_hook(hook)
    return cbam_maps, handle


def _scorecam(model, inp, target_class, top_k=75):
    activations = {}

    def save_activation(_m, _i, out):
        activations["layer4"] = out.detach()

    handle = model.branch1.features[7].register_forward_hook(save_activation)

    with torch.no_grad():
        _ = model(inp)

    handle.remove()

    acts = activations.get("layer4")
    if acts is None:
        raise RuntimeError("Activation not captured for ScoreCAM")

    b, c, h, w = acts.size()
    scores = acts.mean(dim=(2, 3)).squeeze(0)
    k = min(top_k, c)
    top_idx = torch.topk(scores, k=k, dim=0).indices

    heatmap = torch.zeros((inp.size(2), inp.size(3)), device=inp.device)
    weight_sum = 0.0

    for idx in top_idx:
        act = acts[0, idx, :, :]
        act = act - act.min()
        if act.max() <= 0:
            continue
        act = act / act.max()
        act_up = F.interpolate(act[None, None, :, :], size=inp.shape[2:], mode="bilinear",
                               align_corners=False).squeeze(0)

        masked = inp * act_up
        with torch.no_grad():
            logits, _ = model(masked)
            prob = torch.softmax(logits, dim=1)[0, target_class].item()

        heatmap += prob * act_up.squeeze(0)
        weight_sum += prob

    if weight_sum > 0:
        heatmap = heatmap / weight_sum

    heatmap = heatmap.detach().cpu().numpy()
    return heatmap


def _create_overlay(img, hm, brain_mask):
    hm_uint8 = (hm * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    mask_3ch = cv2.merge([brain_mask, brain_mask, brain_mask])
    overlay = cv2.addWeighted(img, 0.5, hm_color, 0.5, 0)
    overlay = np.where(mask_3ch > 0, overlay, img)
    return overlay


def scorecam_map(image_path, model, clf_data=None, out_path=None, top_k=75):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]
    brain_mask = create_brain_mask(img)
    mask_float = (brain_mask / 255.0).astype(np.float32)

    stripped_img, _ = skull_strip_image_adaptive(img)

    if len(stripped_img.shape) == 2:
        stripped_img = cv2.cvtColor(stripped_img, cv2.COLOR_GRAY2BGR)
    img_rgb = cv2.cvtColor(stripped_img, cv2.COLOR_BGR2RGB)

    inp = transform(img_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _, fused_feats = model(inp)
    feat = fused_feats.cpu().numpy().flatten()

    if clf_data is None:
        raise RuntimeError("SVM classifier required for ScoreCAM target class")

    scaler = clf_data["scaler"]
    classifier = clf_data["classifier"]
    feat_scaled = scaler.transform([feat])
    svm_probs = classifier.predict_proba(feat_scaled)[0]
    target_class = int(np.argmax(svm_probs))

    cbam_maps, cbam_handle = _extract_cbam_map(model)
    with torch.no_grad():
        _ = model(inp)
    cbam_handle.remove()

    heatmap = _scorecam(model, inp, target_class, top_k=top_k)

    # Resize from model input size (224x224) to original image size
    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

    if "resnet_cbam" in cbam_maps:
        cbam = cbam_maps["resnet_cbam"].squeeze().cpu().numpy()
        cbam = cv2.resize(cbam, (w, h), interpolation=cv2.INTER_CUBIC)
        cbam = np.clip(cbam, 0, 1)
        if cbam.max() > 0:
            cbam = cbam / cbam.max()
        heatmap = heatmap * cbam

    heatmap = heatmap * mask_float

    brain_vals = heatmap[brain_mask > 0]
    if len(brain_vals) > 0 and brain_vals.max() > 0:
        heatmap = heatmap / brain_vals.max()

    heatmap = cv2.GaussianBlur(heatmap, (11, 11), sigmaX=0)
    heatmap = heatmap * mask_float

    brain_vals = heatmap[brain_mask > 0]
    if len(brain_vals) > 0:
        low = np.percentile(brain_vals, 60)
        high = np.percentile(brain_vals, 99)
    else:
        low, high = 0, 1

    hm = np.clip(heatmap, low, high)
    hm = hm - hm.min()
    if hm.max() > 0:
        hm = hm / hm.max()

    hm = np.power(hm, 1.7)
    hm = hm * mask_float

    overlay = _create_overlay(img, hm, brain_mask)

    if out_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join("outputs", f"scorecam_{base_name}.jpg")

    os.makedirs(os.path.dirname(out_path) or "outputs", exist_ok=True)
    cv2.imwrite(out_path, overlay)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ScoreCAM XAI heatmap")
    parser.add_argument("image", type=str, help="Path to MRI image")
    parser.add_argument("--weights", type=str, default="models/super_hybrid.pth",
                        help="Path to Super-Hybrid model weights")
    parser.add_argument("--clf", type=str, default="models/classifier.joblib",
                        help="Path to SVM classifier (.joblib)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for heatmap image")
    parser.add_argument("--top_k", type=int, default=75,
                        help="Top-K activation channels for ScoreCAM")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    print("Loading Super-Hybrid model...")
    mdl = load_model(args.weights)

    print("Loading SVM classifier...")
    clf = joblib.load(args.clf)

    out = scorecam_map(args.image, mdl, clf, out_path=args.output, top_k=args.top_k)
    print(f"ScoreCAM heatmap saved -> {out}")
