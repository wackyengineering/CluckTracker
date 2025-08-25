from ultralytics import YOLO

# --- Model Configuration ---
# Load a pre-trained YOLOv8 Nano model. 
# 'n' is for nano, which is small and fast, ideal for real-time detection.
model = YOLO('/content/drive/MyDrive/Video Ideas/Ultimate AI Chicken Coop/CluckTracker/runs/detect/train14/weights/best.pt')

# --- Training ---
# Train the model on your custom data using the 'data.yaml' configuration file.
results = model.train(
    data='dataColab.yaml', 
    epochs=100,      # Number of times to cycle through the entire dataset.
    imgsz=640,       # Resize all input images to 640x640 pixels.
    device=0,
    batch=96   
)

# The best model will be saved automatically.
print("Training complete! The best model is saved as best.pt")
print("You can find it in the 'runs/detect/train/weights/' directory.")