import cv2
from ultralytics import YOLO

# --- Configuration ---
# IMPORTANT: Update this path to point to your trained model file
MODEL_PATH = 'models/best.pt' 
# This is usually 0 for the built-in webcam. If you have multiple cameras,
# you might need to try 1, 2, etc.
CAMERA_INDEX = 0

# --- Initialization ---
# Load your custom-trained YOLOv8 model
try:
    model = YOLO(MODEL_PATH)
    print("AI model loaded successfully.")
except Exception as e:
    print(f"FATAL: Error loading model at '{MODEL_PATH}'. Ensure the path is correct.")
    print(f"Error details: {e}")
    exit()

# Initialize the camera feed
try:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise IOError(f"Cannot open camera at index {CAMERA_INDEX}.")
    print("Camera stream initialized. Press 'q' in the video window to quit.")
except Exception as e:
    print(f"FATAL: Error initializing camera: {e}")
    exit()


# --- Main Detection Loop ---
while True:
    # Read one frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read frame from camera. Exiting.")
        break

    # Run AI inference on the frame
    # The 'mps' device uses your Mac's GPU for fast performance
    results = model(frame, device='mps', verbose=False)

    # --- Your Custom Logic Goes Here ---
    # In the future, you can loop through the results to count chickens,
    # check for predators, and trigger alerts.
    # For now, we'll just display the visual results.

    # Get the frame with bounding boxes and labels drawn on it
    annotated_frame = results[0].plot()

    # Display the resulting frame in a window
    cv2.imshow("CluckTrack Live Monitor", annotated_frame)

    # Wait for 1 millisecond, and break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
# Release the camera and destroy all windows when the loop is broken
print("Shutting down...")
cap.release()
cv2.destroyAllWindows()