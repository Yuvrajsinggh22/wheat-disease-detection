import os
import shutil
from sklearn.model_selection import train_test_split

RAW_DIR = "dataset/raw/maize"
OUTPUT_DIR = "dataset/processed"

classes = ["common rust", "gray leaf spot", "healthy", "northern leaf blight"]

for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

for cls in classes:
    class_path = os.path.join(RAW_DIR, cls)
    images = os.listdir(class_path)

    train_imgs, temp_imgs = train_test_split(
        images, test_size=0.30, random_state=42
    )

    val_imgs, test_imgs = train_test_split(
        temp_imgs, test_size=0.50, random_state=42
    )

    for img in train_imgs:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(OUTPUT_DIR, "train", cls, img)
        )

    for img in val_imgs:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(OUTPUT_DIR, "val", cls, img)
        )

    for img in test_imgs:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(OUTPUT_DIR, "test", cls, img)
        )

print("Dataset preprocessing completed successfully!")
