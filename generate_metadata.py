import os
import csv

# Define paths
BASE_DIR = "organized_ravdess"  # Root directory containing train/test folders
OUTPUT_FILE = "metadata.csv"    # Output CSV file name

# Function to generate metadata
def generate_metadata():
    rows = []
    # Iterate over train and test directories
    for split in ["train", "test"]:
        split_dir = os.path.join(BASE_DIR, split)
        # Iterate over emotion folders
        for emotion in os.listdir(split_dir):
            emotion_dir = os.path.join(split_dir, emotion)
            if os.path.isdir(emotion_dir):
                # Iterate over WAV files
                for wav_file in os.listdir(emotion_dir):
                    if wav_file.endswith(".wav"):
                        file_path = os.path.join(split_dir, emotion, wav_file)
                        rows.append([f"{split}_{emotion}_{wav_file}", file_path, emotion])
    
    # Write rows to CSV
    with open(OUTPUT_FILE, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["ID", "file", "label"])
        # Write data
        writer.writerows(rows)

if __name__ == "__main__":
    generate_metadata()
    print(f"Metadata file '{OUTPUT_FILE}' created successfully.")

