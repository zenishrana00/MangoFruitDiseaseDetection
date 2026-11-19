import tensorflow as tf
from keras import layers, models
from keras.applications import EfficientNetB0


# -----------------------------
# 1. Basic settings
# -----------------------------

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 7


# -----------------------------
# 2. Dataset paths
# -----------------------------

train_path = r"D:\MangoDiseaseDetection\dataset\final\train"
val_path = r"D:\MangoDiseaseDetection\dataset\final\val"


# -----------------------------
# 3. Load training dataset
# -----------------------------

train_data = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

test_path = r"D:\MangoDiseaseDetection\dataset\final\test"

# -----------------------------
# 4. Load validation dataset
# -----------------------------

val_data = tf.keras.utils.image_dataset_from_directory(
    val_path,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

test_data = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)


# -----------------------------
# 5. Data augmentation
# -----------------------------

data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])


# -----------------------------
# 6. Load EfficientNetB0
# -----------------------------

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)


# Freeze EfficientNet initially
base_model.trainable = False

print("Total layers:", len(base_model.layers))


# -----------------------------
# 7. Build our model
# -----------------------------

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = data_augmentation(inputs)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.2)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)


# -----------------------------
# 8. Compile model
# -----------------------------

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# -----------------------------
# 9. Display model
# -----------------------------

model.summary()

# model train starts here  : Stage 1:

# history = model.fit(
#     train_data,
#     validation_data=val_data,
#     epochs=10
# )

# # this saves our model 

# model.save(r"D:\MangoDiseaseDetection\model\stage1_model.keras")

# test_loss, test_accuracy = model.evaluate(test_data)

# print("Test Loss:", test_loss)
# print("Test Accuracy:", test_accuracy)

# Load the trained Stage 1 model
model = tf.keras.models.load_model(
    r"D:\MangoDiseaseDetection\model\stage1_model.keras"
)

# -----------------------------
# Stage 2: Fine-tuning
# -----------------------------

base_model = model.layers[2]

base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

fine_tune_history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

test_loss, test_accuracy = model.evaluate(test_data)

print("Fine-tuned Test Loss:", test_loss)
print("Fine-tuned Test Accuracy:", test_accuracy)

model.save(r"D:\MangoDiseaseDetection\model\mango_disease_model.keras")