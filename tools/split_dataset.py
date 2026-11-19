import os
import shutil
import random

# Original dataset
raw_path = r"D:\MangoDiseaseDetection\dataset\raw\Amrapali_Mango_Fruit_Diseases"

# New dataset
final_path = r"D:\MangoDiseaseDetection\dataset\final"

# Split ratio
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

# Make the result reproducible
random.seed(42)

# Get all classes
classes = os.listdir(raw_path)

for class_name in classes:

    class_path = os.path.join(raw_path, class_name)

    if not os.path.isdir(class_path):
        continue

    # Get image files
    images = os.listdir(class_path)

    # Shuffle images
    random.shuffle(images)

    # Calculate split sizes
    total = len(images)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    # Create class folders
    train_class_path = os.path.join(final_path, "train", class_name)
    val_class_path = os.path.join(final_path, "val", class_name)
    test_class_path = os.path.join(final_path, "test", class_name)

    os.makedirs(train_class_path, exist_ok=True)
    os.makedirs(val_class_path, exist_ok=True)
    os.makedirs(test_class_path, exist_ok=True)

    # Copy training images
    for image in train_images:
        shutil.copy2(
            os.path.join(class_path, image),
            os.path.join(train_class_path, image)
        )

    # Copy validation images
    for image in val_images:
        shutil.copy2(
            os.path.join(class_path, image),
            os.path.join(val_class_path, image)
        )

    # Copy testing images
    for image in test_images:
        shutil.copy2(
            os.path.join(class_path, image),
            os.path.join(test_class_path, image)
        )

    print(
        f"{class_name}: "
        f"Train={len(train_images)}, "
        f"Val={len(val_images)}, "
        f"Test={len(test_images)}"
    )

print("\nDataset splitting completed!")