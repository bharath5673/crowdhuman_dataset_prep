import os
import shutil

# === Config ===
base_path = "5/CrowdHuman Cropped/Dataset CrowdHuman/"  # Base path for your dataset
labels_folder = "yolov5-crowdhuman/annotations/"         # Path to annotation (label) .txt files
output_base_path = "./custom_dataset/"                   # Output directory for the split dataset

# Paths for the images and labels
image_paths = {
    "crowd": os.path.join(base_path, "crowd"),
    "non_crowd": os.path.join(base_path, "non crowd")
}

label_paths = {
    "crowd": os.path.join(labels_folder, "crowd"),
    "non_crowd": os.path.join(labels_folder, "non crowd")
}

# === Split Configuration ===
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# === Create output folder structure ===
splits = ["train", "val", "test"]
for split in splits:
    os.makedirs(os.path.join(output_base_path, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_base_path, split, "labels"), exist_ok=True)

# === Function to copy files without crowd/non_crowd split ===
def copy_files_in_order(image_folder, label_folder, split_type, image_offset):
    images = sorted([f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')])
    total = len(images)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    if split_type == "train":
        selected = images[:train_end]
    elif split_type == "val":
        selected = images[train_end:val_end]
    else:
        selected = images[val_end:]

    for img in selected:
        img_src = os.path.join(image_folder, img)
        lbl_src = os.path.join(label_folder, img.replace('.jpg', '.txt'))

        img_dst = os.path.join(output_base_path, split_type, "images", img)
        lbl_dst = os.path.join(output_base_path, split_type, "labels", img.replace('.jpg', '.txt'))

        if os.path.exists(lbl_src):
            shutil.copy(img_src, img_dst)
            shutil.copy(lbl_src, lbl_dst)

# === Merge and copy files ===
for split_type in splits:
    for category in ["crowd", "non_crowd"]:
        copy_files_in_order(image_paths[category], label_paths[category], split_type, image_offset=0)

print("✅ Dataset splitting completed with merged 'images' and 'labels' folders!")
