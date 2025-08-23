import fiftyone as fo
import fiftyone.zoo as foz

# --- Define Animal Classes by Category ---

# The primary animal we want to protect
TARGET_ANIMAL = ["Chicken"]

# Animals that are a clear and immediate danger
HIGH_RISK_PREDATORS = [
    "Raccoon",
    "Fox",
    "Carnivore", # Proxy for Canidae/Wolf/Coyote
    "Skunk",
]

# Animals that could potentially harm chickens and should be monitored
POTENTIAL_THREATS = [
    "Cat",
    "Dog",
]

# Animals that are generally harmless
HARMLESS_VISITORS = [
    "Squirrel",
    "Rabbit",
    "Deer",
]

# Combine all lists into the final list for the dataset
ALL_CLASSES = TARGET_ANIMAL + HIGH_RISK_PREDATORS + POTENTIAL_THREATS + HARMLESS_VISITORS

def download_dataset(
    classes: list[str],
    dataset_name: str = "clucktrack-dataset-v1",
    max_samples: int = 10000,
) -> fo.Dataset:
    """Downloads a dataset from the FiftyOne Zoo for the CluckTrack project.

    Args:
        classes: A list of class names to include in the dataset.
        dataset_name: The name to assign to the downloaded dataset.
        max_samples: The maximum number of samples to download.

    Returns:
        The downloaded FiftyOne dataset.
    """
    print("Downloading dataset with the following classes:")
    print(classes)

    # --- Download and Launch the Dataset ---
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        splits=["train", "validation"],
        label_types=["detections"],
        classes=classes,
        max_samples=max_samples,
        dataset_name=dataset_name,
    )

    dataset.persistent = True
    dataset.save() # Explicitly save the dataset to the database

    print("Dataset download complete.")
    return dataset


if __name__ == "__main__":
    # Download the dataset using the predefined classes
    clucktrack_dataset = download_dataset(classes=ALL_CLASSES)

    # Launch the FiftyOne App to view your final dataset
    # This will open in your browser
    session = fo.launch_app(clucktrack_dataset)

    print("FiftyOne App is running. Press Ctrl+C in the terminal to exit.")
    session.wait()