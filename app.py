"""
Brain Tumor Classification — Flask Web Application
===================================================
Premium frontend for the Super-Hybrid V2 brain tumor classifier
with ScoreCAM XAI heatmap generation.
"""

import os
import sys
import time
import uuid

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import joblib
import torchvision.transforms as T
from flask import Flask, render_template, request, jsonify, send_from_directory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from super_hybrid_model import SuperHybridModel
from skull_strip import skull_strip_image_adaptive

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'web_uploads')
HEATMAP_FOLDER = os.path.join(os.path.dirname(__file__), 'web_heatmaps')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'models', 'super_hybrid.pth')
CLF_PATH = os.path.join(os.path.dirname(__file__), 'models', 'classifier.joblib')

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

TUMOR_INFO = {
    'glioma': {
        'name': 'Glioma',
        'full_name': 'Glioma (Glial Cell Tumor)',
        'description': 'Gliomas are tumors that arise from glial cells — the supportive cells of the brain and spinal cord. They are the most common type of primary brain tumor, accounting for about 33% of all brain tumors.',
        'grade': 'WHO Grade I–IV (Low-grade to High-grade)',
        'location': 'Can occur in the cerebral hemispheres, brainstem, cerebellum, or spinal cord. Most commonly found in the frontal and temporal lobes.',
        'prevalence': 'Approximately 6 per 100,000 people annually. More common in males (ratio 1.6:1).',
        'age_group': 'Most common in adults aged 40–65 years, though can occur at any age.',
        'symptoms': [
            'Persistent headaches that worsen over time',
            'Seizures (often the first symptom)',
            'Nausea and vomiting',
            'Cognitive changes and memory problems',
            'Personality or behavior changes',
            'Vision problems or speech difficulties',
            'Weakness or numbness in limbs'
        ],
        'treatment': [
            'Surgical resection (primary treatment)',
            'Radiation therapy (external beam or stereotactic)',
            'Chemotherapy (Temozolomide is standard)',
            'Targeted therapy and immunotherapy (emerging)',
            'Corticosteroids to reduce brain swelling'
        ],
        'survival_rate': '5-year relative survival rate varies by grade: Grade I/II ~80%, Grade III ~30%, Grade IV (GBM) ~5-10%.',
        'risk_factors': 'Ionizing radiation exposure, family history, genetic conditions (NF1, Li-Fraumeni syndrome), age.',
        'severity': 'high',
        'color': '#ff4444'
    },
    'meningioma': {
        'name': 'Meningioma',
        'full_name': 'Meningioma (Meningeal Tumor)',
        'description': 'Meningiomas arise from the meninges — the protective membranes surrounding the brain and spinal cord. They are typically slow-growing and often benign. They account for approximately 37% of all primary brain tumors.',
        'grade': 'WHO Grade I (Benign, ~80%), Grade II (Atypical, ~15-20%), Grade III (Malignant, ~1-3%)',
        'location': 'Most commonly found along the falx cerebri, convexities, sphenoid wing, and skull base. They grow inward, compressing brain tissue.',
        'prevalence': 'Approximately 8.6 per 100,000 people annually. More common in females (ratio 2:1).',
        'age_group': 'Peak incidence between 60–70 years. Rare in children.',
        'symptoms': [
            'Headaches (gradual onset)',
            'Vision changes or loss',
            'Hearing loss or ringing in ears',
            'Memory loss',
            'Seizures',
            'Weakness in arms or legs',
            'Language difficulty'
        ],
        'treatment': [
            'Observation with regular MRI monitoring (for small, asymptomatic tumors)',
            'Surgical removal (primary treatment for symptomatic tumors)',
            'Stereotactic radiosurgery (Gamma Knife)',
            'Fractionated radiation therapy',
            'Medication to control symptoms'
        ],
        'survival_rate': '5-year relative survival rate: Grade I ~95%, Grade II ~80%, Grade III ~55%. Overall excellent prognosis for benign meningiomas.',
        'risk_factors': 'Prior radiation therapy, female hormones, NF2 genetic condition, obesity.',
        'severity': 'moderate',
        'color': '#ff9800'
    },
    'notumor': {
        'name': 'No Tumor',
        'full_name': 'No Tumor Detected',
        'description': 'The MRI scan does not show evidence of a brain tumor. The brain tissue appears within normal limits based on the classification model\'s analysis.',
        'grade': 'N/A — No tumor detected',
        'location': 'N/A',
        'prevalence': 'N/A',
        'age_group': 'N/A',
        'symptoms': [],
        'treatment': [
            'No tumor-specific treatment required',
            'Regular health check-ups recommended',
            'Consult a neurologist if symptoms persist',
            'Follow-up MRI if clinically indicated'
        ],
        'survival_rate': 'N/A — No tumor detected. This is a healthy result.',
        'risk_factors': 'N/A',
        'severity': 'none',
        'color': '#4caf50'
    },
    'pituitary': {
        'name': 'Pituitary',
        'full_name': 'Pituitary Adenoma (Pituitary Tumor)',
        'description': 'Pituitary adenomas are tumors that develop in the pituitary gland — the "master gland" at the base of the brain that controls hormone production. They are almost always benign and account for about 15% of all intracranial neoplasms.',
        'grade': 'Typically benign (WHO Grade I). Rarely malignant (pituitary carcinoma).',
        'location': 'Sella turcica — the bony cavity at the base of the skull housing the pituitary gland. May extend upward to compress the optic chiasm.',
        'prevalence': 'Approximately 7.5-15 per 100,000 people. Found incidentally in ~10-20% of autopsies.',
        'age_group': 'Most common between ages 30–60 years. Equal distribution between males and females.',
        'symptoms': [
            'Vision problems (bitemporal hemianopia — loss of peripheral vision)',
            'Hormonal imbalances (excess or deficiency)',
            'Headaches',
            'Fatigue and weakness',
            'Unexplained weight changes',
            'Changes in menstrual cycle (women)',
            'Erectile dysfunction (men)',
            'Growth abnormalities (acromegaly or gigantism)'
        ],
        'treatment': [
            'Transsphenoidal surgery (through the nose — minimally invasive)',
            'Medication (Dopamine agonists for prolactinomas)',
            'Radiation therapy (for residual or recurrent tumors)',
            'Hormone replacement therapy (if gland function is compromised)',
            'Regular endocrine monitoring'
        ],
        'survival_rate': '10-year survival rate: >95%. Most pituitary adenomas are curable with appropriate treatment.',
        'risk_factors': 'MEN1 syndrome, Carney complex, familial isolated pituitary adenoma (FIPA).',
        'severity': 'low',
        'color': '#2196f3'
    }
}

model = None
clf_data = None


def load_globals():
    """Load model and classifier into global scope (called once at startup)."""
    global model, clf_data

    print(f"[STARTUP] Device: {DEVICE}")
    print(f"[STARTUP] Loading Super-Hybrid model from {WEIGHTS_PATH}...")
    m = SuperHybridModel(num_classes=len(CLASSES))
    state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    m.load_state_dict(state, strict=False)
    m = m.to(DEVICE)
    m.eval()
    model = m
    print("[STARTUP] Model loaded OK")

    print(f"[STARTUP] Loading SVM classifier from {CLF_PATH}...")
    clf_data = joblib.load(CLF_PATH)
    print("[STARTUP] Classifier loaded OK")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_feature(img_bgr):
    """Extract 1024-D feature from a BGR numpy image."""
    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inp = transform(img_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _, fused = model(inp)
    return fused.cpu().numpy().flatten()


def predict_proba(feature):
    """Predict class probabilities using SVM classifier."""
    scaler = clf_data['scaler']
    classifier = clf_data['classifier']
    feat_scaled = scaler.transform([feature])
    return classifier.predict_proba(feat_scaled)[0]


def auto_crop_borders(img):
    """
    Remove black/dark borders from MRI screenshots.
    Internet screenshots often have black padding, text bars, etc.
    """
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
    Preprocessing pipeline that MATCHES the training data pipeline.

    Training data was: raw image → skull_strip → save
    So prediction must: raw image → skull_strip → feed to model

    NOTE: auto_crop_borders was REMOVED because it was never part
    of the training pipeline and was corrupting brain MRIs by
    mis-detecting dark brain regions (CSF, ventricles) as borders,
    producing near-empty images that always predicted "notumor".
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


def extract_feature_tta(img_bgr):
    """
    Extract features using Test-Time Augmentation.
    Creates 8 augmented versions of the image, extracts features
    from each, and averages the SVM predictions.

    This dramatically improves robustness on external/unseen images
    because the model gets multiple "views" of the same image.
    """
    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    all_probs = []
    scaler = clf_data['scaler']
    classifier = clf_data['classifier']

    for t in tta_transforms:
        inp = t(img_rgb).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _, fused = model(inp)
        feat = fused.cpu().numpy().flatten()
        feat_scaled = scaler.transform([feat])
        probs = classifier.predict_proba(feat_scaled)[0]
        all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)
    return avg_probs


def get_confidence_level(probs):
    """
    Assess prediction confidence and return a human-readable level.
    Prevents overconfident wrong answers.
    """
    max_prob = np.max(probs)
    sorted_probs = np.sort(probs)[::-1]
    margin = sorted_probs[0] - sorted_probs[1]

    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(len(probs))
    normalized_entropy = entropy / max_entropy

    if max_prob >= 0.75 and margin >= 0.3:
        return 'high', 'High Confidence — Model is very certain'
    elif max_prob >= 0.50 and margin >= 0.15:
        return 'medium', 'Moderate Confidence — Likely correct, but verify'
    elif max_prob >= 0.35:
        return 'low', '⚠️ Low Confidence — Model is uncertain, consult a specialist'
    else:
        return 'very_low', '⚠️ Very Low Confidence — Prediction unreliable, seek expert opinion'


def generate_heatmap(image_path, top_k=75):
    """
    Generate occlusion-based sensitivity heatmap at TENSOR level.

    Occlusion is done on the 224x224 model input tensor, NOT on the raw image.
    This ensures we test what the model ACTUALLY sees, and skull-adjacent
    tumors (like meningiomas) are not missed due to skull stripping.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]

    # Brain mask from original image (not stripped) for overlay
    gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    _, head_mask = cv2.threshold(cv2.GaussianBlur(gray_orig, (5, 5), 0), 15, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    brain_mask = np.zeros_like(gray_orig)
    if cnts:
        cv2.drawContours(brain_mask, [max(cnts, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    # Erode slightly to avoid skull edge
    ero_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    brain_mask = cv2.erode(brain_mask, ero_k, iterations=1)
    mask_float = (brain_mask / 255.0).astype(np.float32)

    # Skull-strip for model input (same as prediction pipeline)
    stripped_img, _ = skull_strip_image_adaptive(img)
    if len(stripped_img.shape) == 2:
        stripped_img = cv2.cvtColor(stripped_img, cv2.COLOR_GRAY2BGR)
    img_rgb = cv2.cvtColor(stripped_img, cv2.COLOR_BGR2RGB)

    # Create the 224x224 base tensor
    base_tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)

    # Get baseline prediction
    with torch.no_grad():
        _, base_fused = model(base_tensor)
    base_feat = base_fused.cpu().numpy().flatten()
    base_probs = predict_proba(base_feat)
    target_class = int(np.argmax(base_probs))
    base_p = float(base_probs[target_class])

    # --- Tensor-level occlusion (224x224 grid) ---
    ts = 224  # tensor size
    patch = 20  # patch size in tensor space
    stride = 12  # stride in tensor space

    cam_224 = np.zeros((ts, ts), dtype=np.float32)
    counts_224 = np.zeros((ts, ts), dtype=np.float32)

    for y in range(0, ts, stride):
        for x in range(0, ts, stride):
            y2 = min(y + patch, ts)
            x2 = min(x + patch, ts)

            # Occlude this patch in the tensor
            inp_occl = base_tensor.clone()
            inp_occl[0, :, y:y2, x:x2] = 0.0

            with torch.no_grad():
                _, fused = model(inp_occl)
            feat = fused.cpu().numpy().flatten()
            probs = predict_proba(feat)
            drop = base_p - probs[target_class]

            drop = max(drop, 0.0)

            cam_224[y:y2, x:x2] += drop
            counts_224[y:y2, x:x2] += 1

    counts_224[counts_224 == 0] = 1
    cam_224 = cam_224 / counts_224

    # Resize heatmap from 224x224 to original image size
    heatmap = cv2.resize(cam_224, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap = np.clip(heatmap, 0, None)

    # Apply brain mask
    heatmap = heatmap * mask_float

    brain_vals = heatmap[brain_mask > 0]
    if len(brain_vals) > 0 and brain_vals.max() > 0:
        heatmap = heatmap / brain_vals.max()

    # Smooth
    heatmap = cv2.GaussianBlur(heatmap, (21, 21), sigmaX=0)
    heatmap = heatmap * mask_float

    # Threshold to show only significant regions
    brain_vals = heatmap[brain_mask > 0]
    if len(brain_vals) > 0 and np.any(brain_vals > 0):
        pos_vals = brain_vals[brain_vals > 0]
        low = np.percentile(pos_vals, 60)
        high = np.percentile(pos_vals, 99)
    else:
        low, high = 0, 1

    hm = np.clip(heatmap, low, high)
    hm = hm - low
    if (high - low) > 0:
        hm = hm / (high - low)

    hm = np.power(hm, 1.8)
    hm = hm * mask_float

    if hm.max() > 0:
        hm = hm / hm.max()

    # Alpha-based overlay on ORIGINAL image
    hm_uint8 = (hm * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)

    alpha_map = hm[:, :, np.newaxis]
    overlay = img.copy().astype(np.float32)
    brain_3ch = np.stack([brain_mask > 0] * 3, axis=-1)

    blend = overlay * (1 - alpha_map * 0.6) + hm_color.astype(np.float32) * alpha_map * 0.6
    overlay = np.where(brain_3ch, blend, overlay).astype(np.uint8)

    uid = uuid.uuid4().hex[:10]
    out_filename = f"heatmap_{uid}.jpg"
    out_path = os.path.join(HEATMAP_FOLDER, out_filename)
    cv2.imwrite(out_path, overlay)

    return out_filename


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/heatmaps/<filename>')
def serve_heatmap(filename):
    return send_from_directory(HEATMAP_FOLDER, filename)


@app.route('/predict', methods=['POST'])
def predict():
    """STEP 1: Instant classification only (1-2 seconds)."""
    start_time = time.time()

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type. Use JPG, PNG, or BMP.'}), 400

    ext = file.filename.rsplit('.', 1)[1].lower()
    uid = uuid.uuid4().hex[:10]
    safe_filename = f'{uid}.{ext}'
    filepath = os.path.join(UPLOAD_FOLDER, safe_filename)
    file.save(filepath)

    try:
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'success': False, 'error': 'Could not read image file.'}), 400

        h, w = img.shape[:2]
        c = img.shape[2] if len(img.shape) == 3 else 1

        img_clean, view = preprocess_for_prediction(img)

        probs = extract_feature_tta(img_clean)

        pred_idx = int(np.argmax(probs))
        pred_class = CLASSES[pred_idx]
        confidence = float(probs[pred_idx] * 100)

        conf_level, conf_message = get_confidence_level(probs)

        probabilities = {}
        for i, cls in enumerate(CLASSES):
            probabilities[cls] = round(float(probs[i] * 100), 2)

        elapsed = round(time.time() - start_time, 2)

        response = {
            'success': True,
            'prediction': pred_class,
            'confidence': round(confidence, 2),
            'confidence_level': conf_level,
            'confidence_message': conf_message,
            'probabilities': probabilities,
            'original_image': f'/uploads/{safe_filename}',
            'image_filename': safe_filename,
            'image_metadata': {
                'width': w,
                'height': h,
                'channels': c,
                'file_size_kb': round(os.path.getsize(filepath) / 1024, 1)
            },
            'detected_view': view,
            'tumor_info': TUMOR_INFO.get(pred_class, {}),
            'model_info': {
                'architecture': 'Super-Hybrid V2 (ResNet50 + DeepHybridCNN + SE + CBAM)',
                'feature_dim': 3072,
                'classifier': 'SVM (RBF kernel)',
                'inference': '4-way Test-Time Augmentation',
                'xai_method': 'Occlusion Sensitivity',
                'device': DEVICE
            },
            'processing_time': elapsed
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/heatmap', methods=['POST'])
def heatmap():
    """STEP 2: Generate ScoreCAM heatmap (optional, called separately)."""
    start_time = time.time()

    image_filename = request.json.get('image_filename', '')

    if not image_filename:
        return jsonify({'success': False, 'error': 'No image filename provided'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, image_filename)
    if not os.path.isfile(filepath):
        return jsonify({'success': False, 'error': 'Image not found on server'}), 404

    try:
        heatmap_filename = generate_heatmap(filepath, top_k=75)
        elapsed = round(time.time() - start_time, 2)

        return jsonify({
            'success': True,
            'heatmap_image': f'/heatmaps/{heatmap_filename}',
            'processing_time': elapsed
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    load_globals()
    app.run(debug=False, host='0.0.0.0', port=5000)

