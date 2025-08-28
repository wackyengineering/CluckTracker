import os
import shutil

# --- Configuration ---
# This should match the EXPORT_DIR in your dataset export script
EXPORT_DIR = "./yolo_dataset"
# The name of the zip file to be created. It will now match the directory name.
ZIP_FILE_NAME = os.path.basename(EXPORT_DIR) # This will create yolo_dataset.zip

print(f"--- STARTING ZIP ARCHIVE CREATION FOR '{EXPORT_DIR}' ---")

# --- Check if the export directory exists ---
if not os.path.exists(EXPORT_DIR):
    print(f"Error: The directory '{EXPORT_DIR}' does not exist.")
    print("Please run your dataset export script first to create the YOLO dataset.")
else:
    # --- Create the zip archive ---
    try:
        # shutil.make_archive(base_name, format, root_dir)
        # base_name: the name of the archive file to create, including the path (without extension)
        # format: the archive format (e.g., 'zip', 'tar', 'gztar')
        # root_dir: the directory to start archiving from (the contents of this directory are archived)
        archive_path = shutil.make_archive(
            base_name=ZIP_FILE_NAME, # Now matches the base name of EXPORT_DIR
            format='zip',
            root_dir=EXPORT_DIR
        )
        print(f"\n--- ZIP ARCHIVE COMPLETE ---")
        print(f"Successfully created zip archive: '{archive_path}'")
        print(f"The original directory '{EXPORT_DIR}' remains unchanged.")
    except Exception as e:
        print(f"\n--- ERROR CREATING ZIP ARCHIVE ---")
        print(f"An error occurred: {e}")
        print(f"Please ensure you have write permissions in the current directory.")
