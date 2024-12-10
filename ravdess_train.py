from sklearn.model_selection import KFold
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch
import soundfile as sf
import os
import pandas as pd
from datasets import load_metric

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def parse_filename(file_name):
    parts = file_name.split("-")
    emotion_label = parts[2]
    return int(emotion_label) - 1

def parse_actor(file_name):
    parts = file_name.split("-")
    return int(parts[-1].split(".")[0])

def create_metadata(audio_dir):
    data = []
    for file_name in os.listdir(audio_dir):
        if file_name.endswith(".wav"):
            file_path = os.path.join(audio_dir, file_name)
            label = parse_filename(file_name)
            actor = parse_actor(file_name)
            data.append({"path": file_path, "label": label, "actor": actor})
    return pd.DataFrame(data)

audio_dir = "path_to_audio_files"
metadata = create_metadata(audio_dir)

folds = {
    0: [2, 5, 14, 15, 16],
    1: [3, 6, 7, 13, 18],
    2: [10, 11, 12, 19, 20],
    3: [8, 17, 21, 23, 24],
    4: [1, 4, 9, 22, 25],
}

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

def preprocess_function(batch):
    speech, rate = sf.read(batch["path"])
    inputs = processor(speech, sampling_rate=rate, return_tensors="pt", padding=True)
    inputs["labels"] = torch.tensor(batch["label"])
    return inputs

def train_on_fold(train_df, val_df, fold_index):
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    train_dataset = train_dataset.map(preprocess_function, remove_columns=["path", "actor"])
    val_dataset = val_dataset.map(preprocess_function, remove_columns=["path", "actor"])

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53",
        num_labels=len(emotion_map),
        problem_type="single_label_classification",
        hidden_dropout_prob=0.1,
    )
    model.freeze_feature_encoder()

    training_args = TrainingArguments(
        output_dir=f"./results_fold_{fold_index}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-3,
        per_device_train_batch_size=32,
        num_train_epochs=10,
        logging_dir=f"./logs_fold_{fold_index}",
        logging_steps=50,
        save_total_limit=2,
        warmup_steps=500,
        weight_decay=0.01,
    )

    metric = load_metric("accuracy")

    def compute_metrics(pred):
        predictions = torch.argmax(torch.tensor(pred.predictions), axis=1)
        return metric.compute(predictions=predictions.numpy(), references=pred.label_ids)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    model.save_pretrained(f"./wav2vec2-emotion-fold{fold_index}")
    processor.save_pretrained(f"./wav2vec2-emotion-fold{fold_index}")

for fold_index, actors in folds.items():
    val_df = metadata[metadata["actor"].isin(actors)]
    train_df = metadata[~metadata["actor"].isin(actors)]
    train_on_fold(train_df, val_df, fold_index)
