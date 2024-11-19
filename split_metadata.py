import pandas as pd

# Input metadata file
METADATA_FILE = "metadata.csv"

# Output files
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

def split_metadata():
    # Read the metadata file
    metadata = pd.read_csv(METADATA_FILE)

    # Split into train and test datasets based on the "ID" column
    train_data = metadata[metadata["ID"].str.startswith("train_")]
    test_data = metadata[metadata["ID"].str.startswith("test_")]

    # Save the splits to separate CSV files
    train_data.to_csv(TRAIN_FILE, index=False)
    test_data.to_csv(TEST_FILE, index=False)

    print(f"Train and test splits created:")
    print(f"  - Train set: {len(train_data)} samples")
    print(f"  - Test set: {len(test_data)} samples")
    print(f"  - Train file: {TRAIN_FILE}")
    print(f"  - Test file: {TEST_FILE}")

if __name__ == "__main__":
    split_metadata()

