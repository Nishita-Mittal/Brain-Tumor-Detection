import os
import argparse
import numpy as np
from tqdm import tqdm
import cv2
import torch
import torchvision.transforms as T

from super_hybrid_model import SuperHybridModel
from skull_strip import skull_strip_image_adaptive

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_EXTS = ['.jpg', '.jpeg', '.png']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_backbone(weights_path, device):
    model = SuperHybridModel(num_classes=len(CLASSES))

    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt

    missing = model.load_state_dict(state, strict=False)
    print(f"Loaded weights. Incompatible/missing keys: {missing}")

    model = model.to(device)
    model.eval()
    return model

def extract_from_dir(base_dir, model, feat_save, label_save, apply_strip=True):
    features = []
    labels = []

    for idx, cls in enumerate(CLASSES):
        cls_folder = os.path.join(base_dir, cls)
        if not os.path.isdir(cls_folder):
            print(f"Warning: Missing folder → {cls_folder}")
            continue

        files = [f for f in os.listdir(cls_folder) if any(f.lower().endswith(ext) for ext in IMG_EXTS)]
        for fname in tqdm(files, desc=f"Extract [{cls}]"):
            path = os.path.join(cls_folder, fname)
            img = cv2.imread(path)
            if img is None:
                continue

            if apply_strip:
                img, _ = skull_strip_image_adaptive(img)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            inp = transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                _, fused = model(inp)
                feat = fused.cpu().numpy().squeeze()

            features.append(feat)
            labels.append(idx)

    features = np.array(features)
    labels = np.array(labels)

    os.makedirs(os.path.dirname(feat_save) or "features", exist_ok=True)
    np.save(feat_save, features)
    np.save(label_save, labels)

    print(f"Saved: {feat_save} {features.shape}, {label_save} {labels.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 3072-D features using Super-Hybrid Model")
    parser.add_argument("--train_dir", type=str, default="data/Training_stripped")
    parser.add_argument("--test_dir", type=str, default="data/Testing_stripped")
    parser.add_argument("--weights", type=str, default="models/super_hybrid.pth")
    parser.add_argument("--skip_strip", action="store_true",
                        help="Skip view-adaptive skull stripping before extraction")
    args = parser.parse_args()

    print(f"\nDevice: {DEVICE}")
    print("Loading Super-Hybrid backbone ...")
    model = load_backbone(args.weights, DEVICE)

    print("\n--- Extracting TRAINING features ---")
    extract_from_dir(args.train_dir, model,
                     "features/X_train_feats.npy", "features/y_train.npy",
                     apply_strip=not args.skip_strip)

    print("\n--- Extracting TESTING features ---")
    extract_from_dir(args.test_dir, model,
                     "features/X_test_feats.npy", "features/y_test.npy",
                     apply_strip=not args.skip_strip)

    print("\nDone!")

