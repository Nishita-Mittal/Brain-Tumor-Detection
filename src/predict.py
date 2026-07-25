import argparse
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import joblib

from super_hybrid_model import SuperHybridModel
from skull_strip import skull_strip_image_adaptive

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

tta_transforms = [
    # Plain forward pass
    T.Compose([
        T.ToPILImage(), T.Resize((224, 224)),
        T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    # Horizontal flip (MRI can be mirrored)
    T.Compose([
        T.ToPILImage(), T.Resize((224, 224)), T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    # Small rotation +5°
    T.Compose([
        T.ToPILImage(), T.Resize((224, 224)), T.RandomRotation((5, 5)),
        T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    # Small rotation -5°
    T.Compose([
        T.ToPILImage(), T.Resize((224, 224)), T.RandomRotation((-5, -5)),
        T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
]


def load_backbone(weights_path):
    model = SuperHybridModel(num_classes=len(CLASSES))
    state = torch.load(weights_path, map_location=DEVICE)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE)
    model.eval()
    return model


def auto_crop_borders(img):
    """Remove black/dark borders from MRI screenshots."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    img_h, img_w = img.shape[:2]
    if x > img_w * 0.05 or y > img_h * 0.05 or \
       (img_w - x - w) > img_w * 0.05 or (img_h - y - h) > img_h * 0.05:
        cropped = img[y:y+h, x:x+w]
        if cropped.shape[0] > 50 and cropped.shape[1] > 50:
            return cropped
    return img


def preprocess_for_prediction(img):
    """
    Same pipeline as training data:
    raw image → skull_strip → feed to model

    NOTE: auto_crop_borders was REMOVED because it was never part
    of the training pipeline (training only did skull_strip) and was
    corrupting brain MRIs by mis-detecting dark brain regions as borders.
    """
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    img, view = skull_strip_image_adaptive(img)

    h, w = img.shape[:2]
    if h < 100 or w < 100:
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    return img, view


def predict_with_tta(model, img_bgr, scaler, classifier):
    """
    Predict with 4-way TTA (plain + hflip + small rotations).
    Returns averaged probability vector.
    """
    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    all_probs = []
    for t in tta_transforms:
        inp = t(img_rgb).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _, fused = model(inp)
        feat = fused.cpu().numpy().flatten()
        feat_scaled = scaler.transform([feat])
        probs = classifier.predict_proba(feat_scaled)[0]
        all_probs.append(probs)

    return np.mean(all_probs, axis=0)


def get_confidence_level(probs):
    """Assess prediction confidence level."""
    max_prob = np.max(probs)
    sorted_probs = np.sort(probs)[::-1]
    margin = sorted_probs[0] - sorted_probs[1]

    if max_prob >= 0.75 and margin >= 0.3:
        return '✅ HIGH CONFIDENCE'
    elif max_prob >= 0.50 and margin >= 0.15:
        return '🟡 MODERATE CONFIDENCE — verify recommended'
    elif max_prob >= 0.35:
        return '⚠️ LOW CONFIDENCE — uncertain, consult specialist'
    else:
        return '❌ VERY LOW CONFIDENCE — prediction unreliable'


def main():
    parser = argparse.ArgumentParser(description="Predict brain tumor type from MRI image")
    parser.add_argument("image", type=str, nargs='?', default=None)
    parser.add_argument("--weights", type=str, default="models/super_hybrid.pth")
    parser.add_argument("--clf", type=str, default="models/classifier.joblib")
    args = parser.parse_args()

    if args.image is None:
        print("Error: Please provide image path")
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py data/Testing/glioma/Te-gl_0001.jpg")
        exit(1)

    print("Loading Super-Hybrid model...")
    model = load_backbone(args.weights)

    print("Loading SVM classifier...")
    data = joblib.load(args.clf)
    scaler = data["scaler"]
    classifier = data["classifier"]

    print("Reading and preprocessing image...")
    img = cv2.imread(args.image)
    if img is None:
        raise ValueError("Cannot read image: " + args.image)

    print("Applying skull stripping (matching training data)...")
    img, view = preprocess_for_prediction(img)

    print(f"Detected view: {view}")

    print("Running 4-way Test-Time Augmentation...")
    probs = predict_with_tta(model, img, scaler, classifier)
    pred = np.argmax(probs)

    conf_level = get_confidence_level(probs)

    print("\n" + "=" * 50)
    print(f"  Prediction : {CLASSES[pred].upper()}")
    print(f"  Confidence : {probs[pred] * 100:.2f}%")
    print(f"  Assessment : {conf_level}")
    print("=" * 50)
    print("\nAll class probabilities:")
    for i, cls in enumerate(CLASSES):
        bar = "█" * int(probs[i] * 30)
        print(f"  {cls:12s} : {probs[i]*100:6.2f}%  {bar}")


if __name__ == "__main__":
    main()

