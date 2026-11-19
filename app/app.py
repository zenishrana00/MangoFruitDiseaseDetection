import os
import traceback
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# ---------------------------------------------------------
# App Initialization & Configuration
# ---------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "model", "mango_disease_model.keras")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size
IMG_SIZE = (224, 224)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------------
# Exact Class Names (Matching Training & predict.py)
# ---------------------------------------------------------
CLASS_NAMES = [
    "Anthracnose",
    "Bacterial Canker",
    "Healthy",
    "Powdery Mildew",
    "Scab",
    "Sooty Mould",
    "Stem End Rot"
]

# Educational disease knowledge base
DISEASE_INFO = {
    "Anthracnose": {
        "status": "Diseased",
        "scientific_name": "Colletotrichum gloeosporioides",
        "description": "Anthracnose is one of the most severe fungal diseases affecting mangoes worldwide. It produces dark, sunken, irregular lesions that expand rapidly on the peel as the fruit ripens, causing fruit rot and quality loss.",
        "symptoms": [
            "Dark brown to black sunken spots on the peel",
            "Lesions coalesce forming large tear-stain patterns",
            "Rapid decay under warm, humid storage conditions"
        ],
        "management": [
            "Hot water treatment (48°C - 52°C for 5-10 minutes) post-harvest",
            "Pre-harvest copper-based or approved antifungal fungicide sprays",
            "Pruning infected twigs and maintaining orchard hygiene"
        ],
        "severity": "High"
    },
    "Bacterial Canker": {
        "status": "Diseased",
        "scientific_name": "Xanthomonas citri pv. mangiferaeindicae",
        "description": "Bacterial Canker is a devastating bacterial infection causing star-shaped, raised, dark cracking lesions on the mango fruit peel often exuding gummy sap.",
        "symptoms": [
            "Water-soaked dark lesions that turn black and crack open",
            "Characteristic star-shaped fissures on the fruit surface",
            "Bacterial gummy ooze visible in lesion cracks"
        ],
        "management": [
            "Orchard sanitization and removal of infected twigs and fallen fruit",
            "Application of copper oxychloride combined with streptomycin",
            "Planting windbreaks to reduce windblown transmission and fruit abrasions"
        ],
        "severity": "High"
    },
    "Healthy": {
        "status": "Healthy",
        "scientific_name": "Mangifera indica (Healthy)",
        "description": "The mango fruit exhibits clean, uniform skin without visible necrotic lesions, fungal spore carpets, bacterial cankers, or stem end decay. It is in good physiological condition.",
        "symptoms": [
            "Clear, smooth, and naturally pigmented peel",
            "No black spots, sunken lesions, or powdery patches",
            "Firm fruit structure with intact stem connection"
        ],
        "management": [
            "Maintain balanced nutrient and water management in the orchard",
            "Continue regular monitoring for early signs of pests or fungal spores",
            "Follow proper post-harvest handling, sorting, and temperature control"
        ],
        "severity": "None"
    },
    "Powdery Mildew": {
        "status": "Diseased",
        "scientific_name": "Oidium mangiferae",
        "description": "Powdery Mildew is an airborne fungal disease characterized by white, powdery superficial fungal patches on the surface of young fruits, floral panicles, and leaves.",
        "symptoms": [
            "White or greyish powdery fungal coating on fruit peel",
            "Premature fruit dropping or severe russeting/corking of skin",
            "Deformed, stunted fruit growth"
        ],
        "management": [
            "Prophylactic wettable sulfur or systemic fungicide sprays at bloom",
            "Proper orchard spacing and canopy pruning to enhance sunlight penetration",
            "Regular inspection during dry, warm days with cool nights"
        ],
        "severity": "Medium"
    },
    "Scab": {
        "status": "Diseased",
        "scientific_name": "Elsinoë mangiferae",
        "description": "Mango Scab is a fungal infection that leads to grayish-brown, corky, raised wart-like encrustations on the fruit epidermis, severely degrading the fruit market grade.",
        "symptoms": [
            "Small circular dark spots that develop into elevated, rough corky scabs",
            "Cracking of the fruit surface under heavy infestation",
            "Skin russeting and irregular fruit distortion"
        ],
        "management": [
            "Fungicidal protective sprays (copper hydroxide or mancozeb) during fruit set",
            "Removal and destruction of infected twigs and old leaves",
            "Promoting air circulation through systematic pruning"
        ],
        "severity": "Medium"
    },
    "Sooty Mould": {
        "status": "Diseased",
        "scientific_name": "Capnodium mangiferae / Meliola mangiferae",
        "description": "Sooty Mould is a non-parasitic fungal superficial growth that develops on sugary honeydew secreted by sap-sucking insects like hoppers, mealybugs, and scales.",
        "symptoms": [
            "Thin black, soot-like velvety sheet peeling off the fruit surface",
            "Peel underneath remains unrotted but cosmetically damaged",
            "Associated presence of honeydew-producing insect pests"
        ],
        "management": [
            "Control sap-feeding insect populations (mango leafhoppers and scales)",
            "Spray starch solution (2%) or dilute soap solution to flake off sooty crust",
            "Spray neem oil or appropriate systemic insecticides"
        ],
        "severity": "Low to Medium"
    },
    "Stem End Rot": {
        "status": "Diseased",
        "scientific_name": "Lasiodiplodia theobromae / Dothiorella spp.",
        "description": "Stem End Rot is a severe post-harvest fungal disease that begins as a dark patch around the fruit stem button and rapidly progresses downward through the peel and pulp as the fruit ripens.",
        "symptoms": [
            "Water-soaked dark brown discoloration starting at the stem attachment",
            "Rapid, soft rotting progressing longitudinally toward the fruit tip",
            "Foul sour odor accompanied by soft pulp liquefaction"
        ],
        "management": [
            "Harvest with a 1-2 cm pedicel (stem) intact to prevent pathogen entry",
            "Hot water dip (48°C for 5-10 minutes) combined with prochloraz or thiabendazole",
            "Cold chain storage and rapid cooling following harvest"
        ],
        "severity": "High"
    }
}

# ---------------------------------------------------------
# Load Model Once at Server Startup
# ---------------------------------------------------------
print(f"Loading trained model from: {MODEL_PATH} ...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Check directory structure!")

model = tf.keras.models.load_model(MODEL_PATH)
print("TensorFlow EfficientNetB0 model loaded successfully!")

# Warm up model with a dummy prediction
try:
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    _ = model.predict(dummy_input, verbose=0)
    print("Model warmup completed!")
except Exception as e:
    print(f"Warmup warning: {e}")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image_smart(image_path):
    """
    Automatic Image Resolution Adjustment & Aspect Ratio Preservation:
    1. Loads image with PIL
    2. Corrects EXIF camera orientation (ImageOps.exif_transpose)
    3. Converts to 3-channel RGB (handles RGBA, grayscale, etc.)
    4. Resizes to 224x224 preserving proportions using letterbox/padding (ImageOps.pad)
       to avoid stretching or distorting the mango fruit.
    5. Converts to NumPy array and adds batch dimension (1, 224, 224, 3)
    """
    with Image.open(image_path) as raw_img:
        # Correct mobile phone camera orientation
        img = ImageOps.exif_transpose(raw_img)
        orig_w, orig_h = img.size
        
        # Ensure 3-channel RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Proportion-preserving resize with padding to 224x224
        # If square, resizes directly. If rectangular, pads margins with black (0,0,0)
        img_resized = ImageOps.pad(img, IMG_SIZE, color=(0, 0, 0))
        
        # Convert to tensor array matching predict.py
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, axis=0)
        
        return img_array, f"{orig_w} × {orig_h}"


# ---------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", diseases=DISEASE_INFO, class_names=CLASS_NAMES)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image file provided in the request."
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "No image file was selected or captured."
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": "Unsupported file format. Please upload or capture a JPG, JPEG, PNG, or WEBP image."
        }), 400

    try:
        # Secure and save uploaded/captured file
        orig_filename = secure_filename(file.filename) or "capture.jpg"
        unique_name = f"{uuid.uuid4().hex[:8]}_{orig_filename}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(save_path)

        # Smart preprocessing with EXIF correction and aspect-ratio preservation
        processed_img, orig_dims = preprocess_image_smart(save_path)

        # Run inference using the trained EfficientNetB0 model
        raw_preds = model.predict(processed_img, verbose=0)[0]

        # Calculate prediction results
        pred_index = int(np.argmax(raw_preds))
        predicted_class = CLASS_NAMES[pred_index]
        confidence = float(raw_preds[pred_index] * 100.0)

        # Build detailed class probability breakdown
        probabilities = []
        for i, class_name in enumerate(CLASS_NAMES):
            probabilities.append({
                "class_name": class_name,
                "confidence": round(float(raw_preds[i] * 100.0), 2)
            })

        probabilities.sort(key=lambda x: x["confidence"], reverse=True)

        # Retrieve disease details
        info = DISEASE_INFO.get(predicted_class, {
            "status": "Diseased" if predicted_class != "Healthy" else "Healthy",
            "scientific_name": "N/A",
            "description": "No description available.",
            "symptoms": [],
            "management": [],
            "severity": "Unknown"
        })

        return jsonify({
            "success": True,
            "predicted_class": predicted_class,
            "display_name": "Healthy Mango" if predicted_class == "Healthy" else predicted_class,
            "is_healthy": (predicted_class == "Healthy"),
            "confidence": round(confidence, 2),
            "image_url": f"/uploads/{unique_name}",
            "filename": orig_filename,
            "original_resolution": orig_dims,
            "optimized_resolution": "224 × 224 × 3",
            "optimization_message": "Image optimized for AI analysis.",
            "info": info,
            "top_probabilities": probabilities,
            "disclaimer": "AI prediction based on trained EfficientNetB0. Intended for educational and screening assistance."
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Prediction failed: {str(e)}"
        }), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "classes": CLASS_NAMES,
        "input_resolution": "224x224x3"
    })


if __name__ == "__main__":
    print("Starting Mango Disease Detection Web Server...")
    print("Access URL: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
