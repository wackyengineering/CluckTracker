from ultralytics import YOLO

# --- Model Configuration ---
# Load a pre-trained YOLO11 model.
# Options:
# - 'yolo11s.pt' (Small): Best balance of speed and accuracy for Jetson. (Recommended)
# - 'yolo11m.pt' (Medium): Higher accuracy, but slower inference (FPS). Use if you have a Jetson Orin.
model = YOLO('yolo11s.pt') 
# model = YOLO('yolo11m.pt') # Uncomment this line and comment the above to use the Medium model.

# --- Training ---
# Train the model on your custom data using the 'data.yaml' configuration file.
results = model.train(
    data='dataColab.yaml', 
    epochs=300,      # Increased epochs for better convergence
    patience=50,     # Early stopping if no improvement for 50 epochs
    imgsz=640,       # Resize all input images to 640x640 pixels.
    device=0,
    batch=32,        # Reduced batch size for larger model
    
    # --- Augmentation ---
    degrees=10.0,    # Rotate images +/- 10 degrees
    translate=0.1,   # Translate images +/- 10%
    scale=0.5,       # Scale images +/- 50%
    fliplr=0.5,      # 50% probability of horizontal flip
    mosaic=1.0,      # Mosaic augmentation (combining 4 images)
    mixup=0.1,       # Mixup augmentation
)

# The best model will be saved automatically.
print("Training complete! The best model is saved as best.pt")
print("You can find it in the 'runs/detect/train/weights/' directory.")