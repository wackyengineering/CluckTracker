import fiftyone as fo
import fiftyone.zoo as foz

def download_dataset():
    """
    Downloads and sets up the initial dataset for the CluckTrack project
    using FiftyOne Zoo.
    """
    # --- Define Animal Classes by Category ---

    # The primary animal we want to protect
    target_animal = ["Chicken"]

    # Animals that pose a potential threat
    predators_and_pests = [
        "Raccoon", "Fox", "Wolf", "Coyote", "Hawk", "Opossum", "Skunk",
    ]

    # Common, non-threatening animals to prevent false alarms
    harmless_visitors = [
        "Cat", "Dog", "Squirrel", "Rabbit", "Deer",
    ]

    # Combine all lists into the final list for the dataset
    all_classes = target_animal + predators_and_pests + harmless_visitors

    print("Downloading dataset with the following classes:")
    print(all_classes)

    # --- Download and Launch the Dataset ---
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        splits=["train", "validation"],
        label_types=["detections"],
        classes=all_classes,
        max_samples=3000,
        dataset_name="clucktrack-dataset-v1"
    )

    print("Dataset download complete.")
    return dataset


if __name__ == "__main__":
    # Download the dataset
    clucktrack_dataset = download_dataset()

    # Launch the FiftyOne App to view your final dataset
    # This will open in your browser
    session = fo.launch_app(clucktrack_dataset)

    print("FiftyOne App is running. Press Ctrl+C in the terminal to exit.")
    session.wait()