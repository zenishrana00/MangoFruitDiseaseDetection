import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------
# 1. Load final model
# -----------------------------

MODEL_PATH = r"D:\MangoDiseaseDetection\model\mango_disease_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

print("Final model loaded successfully!")


# -----------------------------
# 2. Load test dataset
# -----------------------------

TEST_PATH = r"D:\MangoDiseaseDetection\dataset\final\test"

test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=(224, 224),
    batch_size=32,
    label_mode="categorical",
    shuffle=False
)

print("Test dataset loaded successfully!")


# -----------------------------
# 3. Generate predictions
# -----------------------------

predictions = model.predict(test_data)

predicted_classes = tf.argmax(
    predictions,
    axis=1
).numpy()

print("Predictions generated successfully!")


# -----------------------------
# 4. Get actual classes
# -----------------------------

actual_classes = tf.concat(
    [labels for images, labels in test_data],
    axis=0
).numpy()

actual_classes = tf.argmax(
    actual_classes,
    axis=1
).numpy()

print("Actual classes generated!")


# -----------------------------
# 5. Class names
# -----------------------------

class_names = [
    "Anthracnose",
    "Bacterial Canker",
    "Healthy",
    "Powdery Mildew",
    "Scab",
    "Sooty Mould",
    "Stem End Rot"
]


# -----------------------------
# 6. Confusion Matrix
# -----------------------------

cm = confusion_matrix(
    actual_classes,
    predicted_classes
)

print("\nConfusion Matrix:")
print(cm)


# -----------------------------
# 7. Classification Report
# -----------------------------

print("\nClassification Report:")

print(
    classification_report(
        actual_classes,
        predicted_classes,
        target_names=class_names
    )
)