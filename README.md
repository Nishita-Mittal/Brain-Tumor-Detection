# 🧠 Brain Tumor Detection and Classification using MRI Images

<p align="center">

# 🧠 Brain Tumor Detection & Classification

**Super-Hybrid Deep Learning Framework using ResNet18 + Hybrid CNN + SVM**

Detects and classifies brain tumors from MRI images into **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor** using Deep Learning, Machine Learning, and Explainable AI.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black?style=for-the-badge&logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green?style=for-the-badge&logo=opencv)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=for-the-badge&logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

</p>

---

# 📌 Overview

Brain Tumor Detection and Classification is an AI-powered medical imaging application that automatically detects and classifies brain tumors from MRI scans using a **Super-Hybrid Deep Learning and Machine Learning architecture**.

The proposed framework combines:

- Pretrained **ResNet18**
- Custom **Hybrid CNN**
- Handcrafted Image Features
- **Support Vector Machine (SVM)** Classifier

to achieve accurate four-class brain tumor classification.

The project also integrates **Explainable AI (XAI)** through Occlusion Sensitivity Heatmaps, enabling visualization of image regions that contribute most to model predictions.

---

# ✨ Features

- 🧠 Automatic Brain Tumor Detection
- 🔍 Four-Class Classification
- 🤖 Hybrid Deep Learning Architecture
- 📈 Confidence Score Prediction
- 🔥 Explainable AI (Occlusion Heatmaps)
- 🌐 Flask-based Web Application
- ⚡ GPU (CUDA) Support
- 📊 MRI Feature Extraction
- 🎯 SVM Classification
- 📂 Easy-to-use Training Pipeline

---

# 🛠 Tech Stack

| Category | Technologies |
|-----------|--------------|
| Programming Language | Python |
| Deep Learning | PyTorch, ResNet18 |
| Machine Learning | Scikit-learn (SVM) |
| Image Processing | OpenCV, Pillow |
| Data Processing | NumPy |
| Web Framework | Flask |
| Visualization | Matplotlib |

---

# 🎯 Classification Classes

The model classifies MRI images into the following categories:

| Class | Description |
|-------|-------------|
| Glioma | Glioma Brain Tumor |
| Meningioma | Meningioma Brain Tumor |
| Pituitary | Pituitary Brain Tumor |
| No Tumor | Healthy Brain MRI |

---

# 🏗 Model Architecture

```
                    MRI Image (224 × 224)

                           │

          ┌────────────────┴────────────────┐

          ▼                                 ▼

   ResNet18 (Pretrained)          Hybrid CNN + Handcrafted Features

          │                                 │

      512-D Features                  512-D Features

          └──────────────┬──────────────┘

                         ▼

              1024-D Feature Vector

                         ▼

                SVM Classifier (RBF)

                         ▼

      Tumor Prediction + Confidence Score
```

---

# 📈 Complete Workflow

```
MRI Image

      │

      ▼

Image Preprocessing

      │

      ▼

Feature Extraction

      │

      ▼

ResNet18 + Hybrid CNN

      │

      ▼

1024-D Feature Vector

      │

      ▼

SVM Classifier

      │

      ▼

Brain Tumor Prediction

      │

      ▼

Occlusion Heatmap (Explainable AI)
```

---

# 📂 Project Structure

```text
Brain-Tumor-Detection/

├── data/
│   ├── Training/
│   └── Testing/
│
├── features/
│   └── Extracted 1024-D Feature Arrays
│
├── models/
│   ├── super_hybrid.pth
│   └── classifier.joblib
│
├── outputs/
│   └── Occlusion Heatmaps
│
├── src/
│   ├── super_hybrid_model.py
│   ├── train_cnn.py
│   ├── feature_extraction.py
│   ├── train_classifier.py
│   ├── evaluate_classifier.py
│   ├── predict.py
│   └── occlusion_explain.py
│
├── templates/
├── static/
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/Nishita-Mittal/Brain-Tumor-Detection.git
```

Move inside the project

```bash
cd Brain-Tumor-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Training Pipeline

Organize the dataset as follows:

```text
data/

├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Train the CNN model

```bash
python src/train_cnn.py --epochs 15
```

Extract 1024-D Features

```bash
python src/feature_extraction.py
```

Train SVM Classifier

```bash
python src/train_classifier.py
```

Train the SVM classifier

```bash
python src/train_classifier.py
```

Evaluate the model

```bash
python src/evaluate_classifier.py
```

Predict a single MRI image

```bash
python src/predict.py data/Testing/glioma/Te-gl_0001.jpg
```

Generate Explainable AI Heatmap

```bash
python src/occlusion_explain.py data/Testing/glioma/Te-gl_0001.jpg
```

---

# 🌐 Running the Web Application

Start the Flask application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

# 📸 Application Screenshots

> Add screenshots inside a folder named **screenshots** in your repository.

### 🏠 Home Page

```
screenshots/home.png
```

---

### 📤 Upload MRI Image

```
screenshots/upload.png
```

---

### 📊 Prediction Result

```
screenshots/result.png
```

---

### 🔥 Occlusion Heatmap

```
screenshots/heatmap.png
```

---

# 📊 Model Summary

| Component | Description |
|-----------|-------------|
| CNN Backbone | ResNet18 (Pretrained) |
| Custom Network | Hybrid CNN |
| Feature Vector | 1024-D |
| Classifier | Support Vector Machine (RBF Kernel) |
| Number of Classes | 4 |
| Explainability | Occlusion Sensitivity Heatmaps |
| Framework | Flask |

---

# 📝 Notes

- GPU (CUDA) is automatically used if available.
- Trained models are stored inside the `models/` directory.
- Extracted features are saved inside the `features/` directory.
- Generated heatmaps are stored inside the `outputs/` directory.
- The file `models/super_hybrid.pth` is excluded from this repository because it exceeds GitHub's maximum file size limit.
- The repository contains the complete source code required to reproduce the project.

---

# 🚀 Future Improvements

- 🌐 Cloud Deployment (Render / Hugging Face Spaces)
- 📱 Mobile-Friendly Interface
- 🧠 Grad-CAM Visualization
- 📂 DICOM Image Support
- 🔐 User Authentication
- 🐳 Docker Containerization
- ⚡ REST API Integration
- ☁️ Cloud Storage Support

---

# 📚 Dataset

The project is trained using Brain MRI images categorized into:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

> **Note:** The dataset is not included in this repository due to its size. Place the dataset inside the `data/` folder following the directory structure shown above before training the model.

---

# 💡 Skills Demonstrated

- Deep Learning
- Machine Learning
- Computer Vision
- Medical Image Analysis
- Explainable AI (XAI)
- Feature Engineering
- Flask Web Development
- Python Programming
- Model Deployment Pipeline
- Data Preprocessing

---

# 👩‍💻 Author

## Nishita Mittal

🎓 B.Tech – Computer Science Engineering

🔗 GitHub: https://github.com/Nishita-Mittal

🔗 LinkedIn: https://www.linkedin.com/in/nishitamittal/

📧 Email: nishitamittal0816@gmail.com

---

# ⭐ If you like this project

If you found this repository useful or interesting, consider giving it a ⭐ on GitHub.

It helps others discover the project and motivates future improvements.

---

<p align="center">

Made with ❤️ by **Nishita Mittal**

</p>
