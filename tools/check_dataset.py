import os

dataset_path = r"D:\MangoDiseaseDetection\dataset\raw\Amrapali_Mango_Fruit_Diseases"

classes = os.listdir(dataset_path)

print("Classes found:")
print("----------------")

for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)

    if os.path.isdir(class_path):
        images = os.listdir(class_path)
        print(f"{class_name}: {len(images)} images")