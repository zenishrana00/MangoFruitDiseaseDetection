import os

final_path = r"D:\MangoDiseaseDetection\dataset\final"

splits = ["train", "val", "test"]

for split in splits:

    split_path = os.path.join(final_path, split)

    print(f"\n{split.upper()}")
    print("----------------")

    total = 0

    for class_name in os.listdir(split_path):

        class_path = os.path.join(split_path, class_name)

        if os.path.isdir(class_path):

            count = len(os.listdir(class_path))
            total += count

            print(f"{class_name}: {count}")

    print(f"Total: {total}")