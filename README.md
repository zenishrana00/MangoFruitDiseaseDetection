# Mango Fruit Disease Detection Using EfficientNetB0

A machine learning web application for automated detection and diagnosis of mango fruit diseases using fine-tuned **EfficientNetB0**, **TensorFlow**, and **Flask**.

## 📌 Project Overview
Mango is one of the most culturally and economically significant fruit crops worldwide, but fungal and bacterial infections often cause severe pre- and post-harvest yield loss. This project delivers an accessible web application where farmers, agricultural extension workers, and students can upload an image or capture a photo using their device's camera to instantly detect diseases with high confidence.

## 🏆 Model Performance & Architecture
- **Base Architecture**: EfficientNetB0 (ImageNet transfer learning)
- **Input Resolution**: 224 × 224 × 3 (RGB)
- **Optimization**: Two-stage training (Stage 1 feature extraction + Stage 2 top-layers fine-tuning)
- **Test Accuracy**: **~93.33%** on unseen test images
- **Aspect-Ratio Preserved Resizing**: Proportion-preserving resize with letterbox padding to avoid stretching or distorting fruit features.

### Supported Categories (7 Classes)
1. **Anthracnose** (*Colletotrichum gloeosporioides*)
2. **Bacterial Canker** (*Xanthomonas citri pv. mangiferaeindicae*)
3. **Healthy Mango** (*Mangifera indica*)
4. **Powdery Mildew** (*Oidium mangiferae*)
5. **Scab** (*Elsinoë mangiferae*)
6. **Sooty Mould** (*Capnodium mangiferae*)
7. **Stem End Rot** (*Lasiodiplodia theobromae*)

---

## 🚀 Key Features
- **Dual Input Modes**: Upload from local device (JPG, JPEG, PNG, WEBP) or capture directly using the device camera (`getUserMedia`).
- **Mobile & Desktop Responsive**: Rear camera preferred on mobile devices; clean, modern mango-inspired UI.
- **Smart Image Optimization**: Automatic EXIF orientation correction and proportion-preserving letterbox padding.
- **Detailed Agricultural Insights**: Pathogen classification, confidence percentage, symptoms checklist, and practical management recommendations.
- **Fast Inference**: Model loaded into memory once on startup with pre-warmed tensors.

---

## 🛠️ Project Structure
```text
MangoDiseaseDetection/
├── app/
│   ├── app.py                      # Flask backend API
│   ├── templates/
│   │   └── index.html              # Responsive frontend UI
│   ├── static/
│   │   ├── css/style.css           # Mango-inspired styling
│   │   └── js/script.js            # Camera & upload interactions
│   └── uploads/                    # Temporary image upload cache
├── model/
│   ├── stage1_model.keras          # Initial transfer learning checkpoint
│   └── mango_disease_model.keras   # Final fine-tuned production model
├── prediction/
│   └── predict.py                  # Standalone CLI prediction script
├── training/
│   └── train.py                    # Model training & fine-tuning script
├── tools/
│   └── test_app.py                 # Automated test suite
├── wsgi.py                         # WSGI entrypoint for production (Gunicorn)
├── requirements.txt                # Production Python dependencies
├── .gitignore                      # Excludes dataset and virtual environment
└── README.md                       # Project documentation
```

---

## 💻 Local Setup & Execution

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<YOUR_USERNAME>/MangoDiseaseDetection.git
   cd MangoDiseaseDetection
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application**:
   ```bash
   python app/app.py
   ```

5. **Open in browser**:
   Navigate to `http://127.0.0.1:5000`

---

## ☁️ Deployment on Render

1. Create a **New Web Service** connected to this GitHub repository.
2. Configure settings:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn wsgi:app --workers 1 --threads 4 --timeout 120`
3. Select the **Free** instance type and click **Deploy**.
