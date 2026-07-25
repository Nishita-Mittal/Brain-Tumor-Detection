# Brain Tumor Classification

**Super-Hybrid Model** (ResNet18 + Custom HybridCNN) → 1024-D features → SVM Classifier

## Overview

This project classifies brain MRI images into **4 categories**:
- **Glioma** — Glioma tumor
- **Meningioma** — Meningioma tumor
- **No Tumor** — Healthy brain
- **Pituitary** — Pituitary tumor

### Architecture

```
MRI Image (224×224)
    ├── Branch 1: ResNet18 (pretrained) → 512-D
    └── Branch 2: HybridCNN + handcrafted features → 512-D
                    ↓
          Concatenated → 1024-D Feature Vector
                    ↓
           SVM Classifier (RBF kernel)
                    ↓
            Tumor Type + Confidence
```

## Quick Setup

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Put dataset in data/ with this structure:
#    data/Training/{glioma, meningioma, notumor, pituitary}/
#    data/Testing/{glioma, meningioma, notumor, pituitary}/

# 3. Train the Super-Hybrid CNN model
python src/train_cnn.py --epochs 15

# 4. Extract 1024-D features
python src/feature_extraction.py

# 5. Train SVM classifier
python src/train_classifier.py

# 6. Evaluate on test set
python src/evaluate_classifier.py

# 7. Predict single image
python src/predict.py data/Testing/glioma/Te-gl_0001.jpg

# 8. Generate occlusion heatmap (XAI)
python src/occlusion_explain.py data/Testing/glioma/Te-gl_0001.jpg
```

## Project Structure

```
├── data/
│   ├── Training/          ← Training images (4 class subfolders)
│   └── Testing/           ← Testing images (4 class subfolders)
├── features/              ← Extracted 1024-D feature arrays (.npy)
├── models/                ← Trained model weights
│   ├── super_hybrid.pth   ← PyTorch model
│   └── classifier.joblib  ← SVM + scaler
├── outputs/               ← Occlusion heatmaps
├── src/
│   ├── super_hybrid_model.py   ← Model architecture
│   ├── train_cnn.py            ← End-to-end CNN training
│   ├── feature_extraction.py   ← Extract 1024-D features
│   ├── train_classifier.py     ← Train SVM on features
│   ├── evaluate_classifier.py  ← Test set evaluation
│   ├── predict.py              ← Single image prediction
│   └── occlusion_explain.py    ← XAI heatmap generation
└── requirements.txt
```

## Notes

- GPU (CUDA) will be used automatically if available.
- Feature arrays and models are saved in `features/` and `models/`.
- Occlusion heatmaps are saved in `outputs/`.
