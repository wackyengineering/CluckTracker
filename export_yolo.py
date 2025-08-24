import fiftyone as fo
import os
import shutil
import random
from clucktrack import ALL_CLASSES

# --- Configuration ---
DATASET_NAME = "clucktrack-dataset-v1"
EXPORT_DIR = "./yolo_dataset"  # The script will create this
TRAIN_SPLIT_PERCENT = 0.8

print("--- STARTING FULLY MANUAL EXPORT ---")

# --- 1. Clean Slate ---
if os.path.exists(EXPORT_DIR):
    shutil.rmtree(EXPORT_DIR)
    print(f"Removed old directory: {EXPORT_DIR}")

# --- 2. Create Directory Structure ---
print("Creating new directory structure...")
os.makedirs(f"{EXPORT_DIR}/images/train")
os.makedirs(f"{EXPORT_DIR}/images/val")
os.makedirs(f"{EXPORT_DIR}/labels/train")
os.makedirs(f"{EXPORT_DIR}/labels/val")
print("Directory structure created.")

# --- 3. Load Dataset ---
dataset = fo.load_dataset(DATASET_NAME)
class_map = {name: i for i, name in enumerate(ALL_CLASSES)}

# --- 4. Manual Splitting ---
sample_ids = dataset.values("id")
random.shuffle(sample_ids)
split_index = int(TRAIN_SPLIT_PERCENT * len(sample_ids))
train_ids = sample_ids[:split_index]
val_ids = sample_ids[split_index:]

# --- 5. Manual File Writing Loop ---
print(f"Manually processing {len(dataset)} samples...")
for sample in dataset.iter_samples(progress=True):
    # Determine which split this sample belongs to
    if sample.id in train_ids:
        split = "train"
    elif sample.id in val_ids:
        split = "val"
    else:
        continue # Should not happen

    # --- Copy Image File ---
    source_path = sample.filepath
    file_name = os.path.basename(source_path)
    destination_path = f"{EXPORT_DIR}/images/{split}/{file_name}"
    shutil.copy(source_path, destination_path)

    # --- Create and Write Label File ---
    label_name = os.path.splitext(file_name)[0] + ".txt"
    label_path = f"{EXPORT_DIR}/labels/{split}/{label_name}"
    
    with open(label_path, "w") as f:
        # Check if sample has detections
        if sample.ground_truth is None or sample.ground_truth.detections is None:
            continue
            
        for det in sample.ground_truth.detections:
            # Get class index
            class_index = class_map.get(det.label)
            if class_index is None:
                continue

            # Bounding box is [x_min, y_min, width, height]
            box = det.bounding_box
            x_center = box[0] + box[2] / 2
            y_center = box[1] + box[3] / 2
            width = box[2]
            height = box[3]

            f.write(f"{class_index} {x_center} {y_center} {width} {height}\n")

print("\n--- MANUAL EXPORT COMPLETE ---")
print(f"Please check the '{EXPORT_DIR}' folder.")