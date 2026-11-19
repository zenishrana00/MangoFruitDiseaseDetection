import tensorflow as tf

MODEL_PATH = r"D:\MangoDiseaseDetection\model\mango_disease_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# IMAGE_PATH = r"D:\MangoDiseaseDetection\prediction\test.jpg"
IMAGE_PATH = input("Enter image path: ")

image = tf.keras.utils.load_img(
    IMAGE_PATH,
    target_size=(224, 224)
)

print("Image loaded successfully!")

image_array = tf.keras.utils.img_to_array(image)
image_array = tf.expand_dims(image_array, axis=0)

print("Image shape:", image_array.shape)

predictions = model.predict(image_array)

print("Predictions:", predictions)

class_names = [
    "Anthracnose",
    "Bacterial Canker",
    "Healthy",
    "Powdery Mildew",
    "Scab",
    "Sooty Mould",
    "Stem End Rot"
]

predicted_index = tf.argmax(predictions[0]).numpy()
predicted_class = class_names[predicted_index]
confidence = predictions[0][predicted_index] * 100

print("Prediction:", predicted_class)
print("Confidence:", f"{confidence:.2f}%")


