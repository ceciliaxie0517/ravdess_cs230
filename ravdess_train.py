from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from datasets import Dataset
import torch
import soundfile as sf
from transformers import TrainingArguments, Trainer
from datasets import load_metric
import os
import pandas as pd

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

train_test_split = metadata.sample(frac=0.9, random_state=42), metadata.sample(frac=0.1, random_state=42)
train_metadata, test_metadata = train_test_split

train_dataset = Dataset.from_pandas(train_metadata)
test_dataset = Dataset.from_pandas(test_metadata)

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

def preprocess_function(batch):
    speech, rate = sf.read(batch["path"])
    inputs = processor(speech, sampling_rate=rate, return_tensors="pt", padding=True)
    inputs["labels"] = torch.tensor(batch["label"])
    return inputs

train_dataset = train_dataset.map(preprocess_function, remove_columns=["path", "actor"])
test_dataset = test_dataset.map(preprocess_function, remove_columns=["path", "actor"])

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53",
    num_labels=len(emotion_map),
    problem_type="single_label_classification",
    hidden_dropout_prob=0.1,
)
model.freeze_feature_encoder()

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-3,
    per_device_train_batch_size=32,
    num_train_epochs=10,
    logging_dir="./logs",
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
    eval_dataset=test_dataset,
    tokenizer=processor,
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained("./wav2vec2-emotion")
processor.save_pretrained("./wav2vec2-emotion")
