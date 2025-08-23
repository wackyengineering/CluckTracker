import fiftyone as fo
# Import the final class list from your main script to ensure they match
from clucktrack import ALL_CLASSES

DATASET_NAME = "clucktrack-dataset-v1"
EXPORT_DIR = "./yolo_dataset"

# Load the dataset you just downloaded
dataset = fo.load_dataset(DATASET_NAME)

# Export the dataset in YOLO format
dataset.export(
    export_dir=EXPORT_DIR,
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
    classes=ALL_CLASSES,
)

print(f"Dataset exported in YOLO format to {EXPORT_DIR}")