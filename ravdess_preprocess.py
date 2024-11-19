import os
import json
from speechbrain.dataio.dataio import read_audio

# Data directory
DATA_DIR = "./organized_ravdess"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
TRAIN_JSON = os.path.join(DATA_DIR, "train.json")
TEST_JSON = os.path.join(DATA_DIR, "test.json")

# Sampling rate
SAMPLERATE = 16000

def process_folder(folder_path):
    """
    Process a folder to extract .wav file paths and their corresponding labels.

    Args:
        folder_path (str): Path containing subfolders for each label.

    Returns:
        List[Tuple[str, str]]: List of (file_path, label) tuples.
    """
    data = []
    for label_folder in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label_folder)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(label_path, file)
                    data.append((file_path, label_folder))
    return data

def create_json(data, json_path):
    """
    Create a JSON file from the data.

    Args:
        data (List[Tuple[str, str]]): List of (file_path, label) tuples.
        json_path (str): Path to save the JSON file.
    """
    json_dict = {}
    for wav_file, emo in data:
        signal = read_audio(wav_file)
        duration = signal.shape[0] / SAMPLERATE
        uttid = os.path.basename(wav_file).replace(".wav", "")
        json_dict[uttid] = {
            "wav": wav_file,
            "length": duration,
            "emo": emo,
        }
    with open(json_path, "w", encoding="utf-8") as json_f:
        json.dump(json_dict, json_f, indent=2)
    print(f"JSON file created: {json_path}")

def prepare_data():
    """
    Process train and test folders to generate train.json and test.json.
    """
    train_data = process_folder(TRAIN_DIR)
    test_data = process_folder(TEST_DIR)

    create_json(train_data, TRAIN_JSON)
    create_json(test_data, TEST_JSON)

if __name__ == "__main__":
    prepare_data()
