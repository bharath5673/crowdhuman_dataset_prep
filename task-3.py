import os
import shutil
import random

# === Config ===
base_path = "5/CrowdHuman Cropped/Dataset CrowdHuman/"  # Base path for your dataset
annotations_folder = "yolov5-crowdhuman/annotations/"  # Folder where annotations are located
output_base_path = "./custom_dataset/"  # The path where the new dataset split will be saved

# Paths for the images and annotations for 'crowd' and 'non_crowd'
image_paths = {
    "crowd": os.path.join(base_path, "crowd"),
    "non_crowd": os.path.join(base_path, "non crowd")
}

annotation_paths = {
    "crowd": os.path.join(annotations_folder, "crowd"),
    "non_crowd": os.path.join(annotations_folder, "non crowd")
}

# === Split Configuration ===
train_ratio = 0.8  # 80% for training
val_ratio = 0.1    # 10% for validation
test_ratio = 0.1   # 10% for testing

# === Create output folder structure ===
output_paths = {
    "train": os.path.join(output_base_path, "train"),
    "val": os.path.join(output_base_path, "val"),
    "test": os.path.join(output_base_path, "test")
}

# Create the necessary folders for images and annotations
for split in output_paths.values():
    os.makedirs(os.path.join(split, "images", "crowd"), exist_ok=True)
    os.makedirs(os.path.join(split, "images", "non_crowd"), exist_ok=True)
    os.makedirs(os.path.join(split, "annotations", "crowd"), exist_ok=True)
    os.makedirs(os.path.join(split, "annotations", "non_crowd"), exist_ok=True)

# === Function to copy files to the appropriate folder ===
def copy_files(image_folder, annotation_folder, split_folder):
    images = os.listdir(image_folder)
    annotations = os.listdir(annotation_folder)
    
    # Shuffle and split the data
    random.shuffle(images)
    split_idx_train = int(len(images) * train_ratio)
    split_idx_val = int(len(images) * (train_ratio + val_ratio))

    train_images = images[:split_idx_train]
    val_images = images[split_idx_train:split_idx_val]
    test_images = images[split_idx_val:]

    # Copy train images and annotations
    for img in train_images:
        shutil.copy(os.path.join(image_folder, img), os.path.join(split_folder, "images", img))
        shutil.copy(os.path.join(annotation_folder, img.replace('.jpg', '.txt')), os.path.join(split_folder, "annotations", img.replace('.jpg', '.txt')))

    # Copy val images and annotations
    for img in val_images:
        shutil.copy(os.path.join(image_folder, img), os.path.join(split_folder, "images", img))
        shutil.copy(os.path.join(annotation_folder, img.replace('.jpg', '.txt')), os.path.join(split_folder, "annotations", img.replace('.jpg', '.txt')))

    # Copy test images and annotations
    for img in test_images:
        shutil.copy(os.path.join(image_folder, img), os.path.join(split_folder, "images", img))
        shutil.copy(os.path.join(annotation_folder, img.replace('.jpg', '.txt')), os.path.join(split_folder, "annotations", img.replace('.jpg', '.txt')))

# === Copy data for 'crowd' and 'non_crowd' ===
for category in ["crowd", "non_crowd"]:
    # Copy for training, validation, and testing
    copy_files(image_paths[category], annotation_paths[category], output_paths['train'])
    copy_files(image_paths[category], annotation_paths[category], output_paths['val'])
    copy_files(image_paths[category], annotation_paths[category], output_paths['test'])

print("Dataset splitting completed!")