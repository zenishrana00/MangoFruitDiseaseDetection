import os
from PIL import Image

dataset_path = r"D:\MangoDiseaseDetection\dataset\raw\Amrapali_Mango_Fruit_Diseases"

valid_images = 0
invalid_images = 0

for class_name in os.listdir(dataset_path):

    class_path = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_path):
        continue

    for file_name in os.listdir(class_path):

        file_path = os.path.join(class_path, file_name)

        try:
            img = Image.open(file_path)
            img.verify()
            valid_images += 1

        except:
            invalid_images += 1
            print("Invalid image:", file_path)

print("----------------------------")
print("Image check completed")
print("Valid images:", valid_images)
print("Invalid images:", invalid_images)