import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Define emotion labels based on RAVDESS filename convention
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Paths
SOURCE_DIR = "ravdess_data"  # Root folder containing Actor_XX subfolders
OUTPUT_DIR = "organized_ravdess"  # Directory to store organized files

# Train-test split ratio
TEST_SIZE = 0.2

def organize_files():
    all_files = []
    
    # Traverse Actor subfolders
    for actor_folder in os.listdir(SOURCE_DIR):
        actor_path = os.path.join(SOURCE_DIR, actor_folder)
        if os.path.isdir(actor_path):
            for file in os.listdir(actor_path):
                if file.endswith(".wav"):
                    # Full path to the file
                    file_path = os.path.join(actor_path, file)
                    # Extract emotion from filename (e.g., 03-01-05-01-01-01-01.wav -> "05")
                    emotion_id = file.split("-")[2]
                    emotion_label = EMOTION_MAP.get(emotion_id)
                    if emotion_label:
                        all_files.append((file_path, emotion_label))
    
    # Split files into train and test sets
    train_files, test_files = train_test_split(all_files, test_size=TEST_SIZE, random_state=42)
    
    # Organize train and test files into respective directories
    for split, files in [("train", train_files), ("test", test_files)]:
        for file_path, emotion_label in files:
            # Create destination directory
            dest_dir = os.path.join(OUTPUT_DIR, split, emotion_label)
            os.makedirs(dest_dir, exist_ok=True)
            # Copy file to the destination
            shutil.copy(file_path, dest_dir)

if __name__ == "__main__":
    organize_files()
    print(f"Files organized into '{OUTPUT_DIR}' directory.")

