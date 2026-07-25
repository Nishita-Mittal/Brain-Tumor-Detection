import os
import argparse
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

CLASSES = ["axial", "sagittal", "coronal"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _list_images(root_dir):
    paths = []
    for base, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(base, f))
    return paths


def _synthesize_view(img_bgr, view):
    if view == "axial":
        return img_bgr
    if view == "sagittal":
        out = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
        return out
    out = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    out = cv2.flip(out, 1)
    return out


class ViewDataset(Dataset):
    def __init__(self, root_dir, train=True):
        self.paths = _list_images(root_dir)
        if not self.paths:
            raise RuntimeError(f"No images found in {root_dir}")
        self.train = train
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.aug = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        view = random.choice(CLASSES)
        img = _synthesize_view(img, view)
        if self.train:
            img = np.array(self.aug(transforms.ToPILImage()(img)))
        img = self.base_transform(img)
        label = CLASSES.index(view)
        return img, label


def build_model(num_classes=3):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def load_view_detector(weights_path):
    model = build_model(num_classes=len(CLASSES))
    state = torch.load(weights_path, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE)
    model.eval()
    return model


def detect_view(img_bgr, model):
    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    inp = tfm(img_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(inp)
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
    idx = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx])


def train_view_detector(train_dir, val_dir=None, epochs=8, batch_size=32, lr=2e-4,
                        save_path="models/view_detector.pth"):
    train_ds = ViewDataset(train_dir, train=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)

    val_loader = None
    if val_dir:
        val_ds = ViewDataset(val_dir, train=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=2, pin_memory=True)

    model = build_model(num_classes=len(CLASSES)).to(DEVICE)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            logits = model(imgs)
            loss = crit(logits, labels)
            loss.backward()
            opt.step()

            running += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

        train_acc = correct / max(total, 1)
        train_loss = running / max(total, 1)

        val_acc = None
        if val_loader is not None:
            model.eval()
            v_correct = 0
            v_total = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    logits = model(imgs)
                    preds = logits.argmax(dim=1)
                    v_correct += (preds == labels).sum().item()
                    v_total += imgs.size(0)
            val_acc = v_correct / max(v_total, 1)

        if val_acc is None:
            print(f"Epoch {epoch}: loss={train_loss:.4f} acc={train_acc:.4f}")
        else:
            print(f"Epoch {epoch}: loss={train_loss:.4f} acc={train_acc:.4f} val_acc={val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)

    if val_loader is None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MRI view detector")
    parser.add_argument("--train_dir", type=str, default="data/Training")
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save_path", type=str, default="models/view_detector.pth")
    args = parser.parse_args()

    train_view_detector(args.train_dir, args.val_dir, args.epochs, args.batch_size,
                        args.lr, args.save_path)
