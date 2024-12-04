import os
import sys
import torch
import numpy as np
import torchaudio
from sklearn.model_selection import train_test_split


root = sys.path[0]

data_path = os.path.join(root, 'RAVDESS_1s_4categories')
classes = os.listdir(data_path)
classes.sort()
print(classes)
cls2id = {j:i for i, j in enumerate(classes)}
dataset = {'audios': [], 'labels': []}
for name in classes:
        label = cls2id[name]
        audio_path = os.path.join(data_path, name)
        for file in os.listdir(audio_path):
            file_path = os.path.join(audio_path, file)
            dataset['audios'].append(file_path)
            dataset['labels'].append(label)


# use train_test_split to split the dataset
X = dataset['audios']
y = dataset['labels']
X_train_ids, X_test_ids, y_train, y_test = train_test_split(np.arange(len(X)).reshape(-1, 1), y, stratify=y, test_size=0.1, random_state=59)
X_train = [X[i] for i in X_train_ids.reshape(-1)]
X_test = [X[i] for i in X_test_ids.reshape(-1)]


print("training set size:", len(X_train))
print("test set size:", len(X_test))


def random_crop_audio(audio, target_length):
    current_length = audio.shape[-1]
    if current_length < target_length:
        pad_size = target_length - current_length
        audio = torch.nn.functional.pad(audio, (0, pad_size))

    if current_length > target_length:
        start = torch.randint(0, current_length - target_length + 1, (1,))
        audio = audio[..., start:start+target_length]
    
    return audio

def random_volume_change(waveform, min_gain=0.7, max_gain=1.2):

    gain = torch.rand(1) * (max_gain - min_gain) + min_gain
    augmented_waveform = waveform * gain
    
    return augmented_waveform


class Dataset:
    def __init__(self, mode='train'):
        super(Dataset, self).__init__()

        self.mode = mode
        self.classes = classes

        if mode == 'train':
            self.audios = X_train
            self.labels = y_train
        elif mode == 'test':
            self.audios = X_test
            self.labels = y_test
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        audio_path = self.audios[index]

        waveform, sample_rate = torchaudio.load(audio_path)
        

        waveform = random_crop_audio(waveform, sample_rate)

        if self.mode == 'train':
            waveform = random_volume_change(waveform)


        mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,       
            hop_length=512,   
            n_mels=40        
        )
        mel_spectrogram = mel_spectrogram_transform(waveform)

        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)

        # to torch.Tensor
        image = torch.as_tensor(mel_spec_db, dtype=torch.float32)
        label = torch.as_tensor(label, dtype=torch.long)

        return image, label


if __name__ == '__main__':
     
    train_dataset = Dataset(mode='train')
    test_dataset = Dataset(mode='test')

    print(len(train_dataset))
    print(len(test_dataset))

    image, label = train_dataset[0]
    print(image.shape, image.max(), image.min(), image.dtype)
    print(label.shape, label, label.dtype)
