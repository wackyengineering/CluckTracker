from ultralytics import YOLO

# --- Model Configuration ---
# Load a pre-trained YOLOv8 Nano model. 
# 'n' is for nano, which is small and fast, ideal for real-time detection.
model = YOLO('yolov8n.pt')

# --- Training ---
# Train the model on your custom data using the 'data.yaml' configuration file.
results = model.train(
    data='data.yaml', 
    epochs=100,      # Number of times to cycle through the entire dataset.
    imgsz=640,       # Resize all input images to 640x640 pixels.
    device='mps'     # Explicitly use the M1/M2/M3 Apple Silicon GPU.
)

# The best model will be saved automatically.
print("Training complete! The best model is saved as best.pt")
print("You can find it in the 'runs/detect/train/weights/' directory.")