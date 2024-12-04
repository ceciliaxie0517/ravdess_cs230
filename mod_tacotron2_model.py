import torch
import torch.nn as nn
import torch.nn.functional as F

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encoder_dim):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, encoder_dim, num_layers=2, batch_first=True, bidirectional=True)

    def forward(self, text):
        embedded = self.embedding(text)  # Shape: (batch, seq_len, embedding_dim)
        outputs, _ = self.lstm(embedded)  # Shape: (batch, seq_len, encoder_dim * 2)
        return outputs

class EmotionEmbeddingModule(nn.Module):
    def __init__(self, num_emotions, embedding_dim):
        super(EmotionEmbeddingModule, self).__init__()
        self.embedding = nn.Embedding(num_emotions, embedding_dim)

    def forward(self, emotion_labels, seq_len):
        emotion_vectors = self.embedding(emotion_labels)  # Shape: (batch, embedding_dim)
        emotion_vectors = emotion_vectors.unsqueeze(1).expand(-1, seq_len, -1)  # Broadcast to (batch, seq_len, embedding_dim)
        return emotion_vectors

class Decoder(nn.Module):
    def __init__(self, input_dim, mel_dim):
        super(Decoder, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, 512, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(512, 512, num_layers=1, batch_first=True)
        self.linear = nn.Linear(512, mel_dim)

    def forward(self, combined_features):
        x, _ = self.lstm1(combined_features)
        x, _ = self.lstm2(x)
        mel_outputs = self.linear(x)  # Shape: (batch, seq_len, mel_dim)
        return mel_outputs

class Tacotron2WithEmotionEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encoder_dim, num_emotions, emotion_dim, mel_dim):
        super(Tacotron2WithEmotionEmbedding, self).__init__()
        self.text_encoder = TextEncoder(vocab_size, embedding_dim, encoder_dim)
        self.emotion_embedding = EmotionEmbeddingModule(num_emotions, emotion_dim)
        self.decoder = Decoder(encoder_dim * 2 + emotion_dim, mel_dim)

    def forward(self, text, emotion_labels):
        text_features = self.text_encoder(text)  # Shape: (batch, seq_len, encoder_dim * 2)
        seq_len = text_features.size(1)
        emotion_features = self.emotion_embedding(emotion_labels, seq_len)  # Shape: (batch, seq_len, emotion_dim)
        combined_features = torch.cat((text_features, emotion_features), dim=-1)  # Shape: (batch, seq_len, encoder_dim * 2 + emotion_dim)
        mel_outputs = self.decoder(combined_features)  # Shape: (batch, seq_len, mel_dim)
        return mel_outputs

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    vocab_size = 50  # Example vocabulary size
    embedding_dim = 256
    encoder_dim = 512
    num_emotions = 4  # e.g., happy, sad, angry, neutral
    emotion_dim = 256
    mel_dim = 80  # Mel-spectrogram dimensions
    batch_size = 16
    seq_len = 100

    # Model initialization
    model = Tacotron2WithEmotionEmbedding(vocab_size, embedding_dim, encoder_dim, num_emotions, emotion_dim, mel_dim)

    # Example inputs
    text = torch.randint(0, vocab_size, (batch_size, seq_len))  # Random text input
    emotion_labels = torch.randint(0, num_emotions, (batch_size,))  # Random emotion labels

    # Forward pass
    mel_outputs = model(text, emotion_labels)
    print("Mel-spectrogram output shape:", mel_outputs.shape)  # Should be (batch_size, seq_len, mel_dim)
