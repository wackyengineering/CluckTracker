import fiftyone as fo
import random
import os
import shutil

# --- Configuration ---
DATASET_NAME = "clucktrack-dataset-v1"
EXPORT_DIR = "./yolo_dataset"
TRAIN_SPLIT_PERCENT = 0.8 # 80% for training, 20% for validation

# --- 1. Clean Slate ---
# Forcefully delete the old export directory to prevent any caching issues
if os.path.exists(EXPORT_DIR):
    shutil.rmtree(EXPORT_DIR)
    print(f"Removed old directory: {EXPORT_DIR}")

# --- 2. Load Dataset ---
print(f"Loading dataset '{DATASET_NAME}'...")
dataset = fo.load_dataset(DATASET_NAME)
dataset.persistent = True # Ensure it's persistent

# --- 3. Manual, Foolproof Splitting ---
print("Performing a manual, foolproof data split...")

# Get a shuffled list of all sample IDs
sample_ids = dataset.values("id")
random.shuffle(sample_ids)

# Manually calculate the split point
split_index = int(TRAIN_SPLIT_PERCENT * len(sample_ids))

# Divide the IDs into train and validation sets
train_ids = sample_ids[:split_index]
val_ids = sample_ids[split_index:]

# Create two 'views' into the dataset based on our ID lists
train_view = dataset.select(train_ids)
val_view = dataset.select(val_ids)

# Forcefully assign the 'split' field to each view
# This is the most critical step
train_view.set_values("split", "train")
val_view.set_values("split", "val")

# Save the changes to the dataset
dataset.save()

print(f"Manual split complete.")
print(f"  - Samples in train set: {len(train_view)}")
print(f"  - Samples in val set:   {len(val_view)}")


# --- 4. Export the Entire Dataset ---
from clucktrack import ALL_CLASSES # Import your class list

print("\n--- Starting final YOLO format export ---")
dataset.export(
    export_dir=EXPORT_DIR,
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
    classes=ALL_CLASSES,
    # The exporter will use the 'split' field we just created
)
print(f"--- Export finished successfully to {EXPORT_DIR} ---")