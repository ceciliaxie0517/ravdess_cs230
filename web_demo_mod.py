import gradio as gr
from model import *
from dataset import random_crop_audio
import matplotlib.pyplot as plt
from PIL import Image
import os, sys
import torchaudio
import torch
import numpy as np
import whisper
from openai import OpenAI
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

client = OpenAI(
    #api_key="YOUR_API"
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(os.path.join(sys.path[0], 'classifier.pth'), map_location=device).eval()

labels = ['angry', 'happy', 'neutral', 'sad']

def text_to_speech(text, output_file="output.mp3"):

    tts = gTTS(text, lang='en')
    tts.save(output_file)
    return output_file

def generate_response(user_text, user_emotion):

    prompt = (
        f"User said (emotion: {user_emotion}): \"{user_text}\". "
        f"Please reply empathetically and consider the user's emotion."
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an empathetic conversational assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content

def classify_and_respond(audio_path):
    if audio_path is None:
        return None, None, "Please upload an audio file.", None, None
    
    waveform, sample_rate = torchaudio.load(audio_path.name)
    waveform = waveform / waveform.abs().max()
    
    num_crops = 5
    cropped_waveforms = [random_crop_audio(waveform, sample_rate) for _ in range(num_crops)]

    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=256,
        n_mels=64
    )

    mel_spectrograms = []
    for cropped_waveform in cropped_waveforms:
        mel_spec = mel_spectrogram_transform(cropped_waveform)
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        mel_spec_db = mel_spec_db.unsqueeze(0).to(device)
        mel_spectrograms.append(mel_spec_db)

    probabilities = []
    with torch.no_grad():
        for mel_spec_db in mel_spectrograms:
            logits = model(mel_spec_db).detach().cpu()
            proba = torch.nn.functional.softmax(logits, dim=-1).squeeze(0).numpy()
            probabilities.append(proba)

    avg_proba = np.mean(probabilities, axis=0)
    detected_emotion = labels[np.argmax(avg_proba)]

    try:
        whisper_model = whisper.load_model("base")
        result = whisper_model.transcribe(audio_path.name)
        transcribed_text = result["text"]

        gpt_response = generate_response(transcribed_text, detected_emotion)
        
        audio_response = text_to_speech(gpt_response)
        
        mel_spec_db = mel_spectrograms[-1].squeeze(0).squeeze(0).cpu().numpy()
        mel_spec_db = 255 * (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db))
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spec_db, aspect='auto', origin='lower', cmap='viridis')
        plt.axis('off')
        plt.savefig('./mel_spec_db.png', format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        image = Image.open('./mel_spec_db.png')

        return (
            image, 
            {labels[i]: float(avg_proba[i]) for i in range(len(labels))},
            transcribed_text,
            gpt_response,
            audio_response
        )
    except Exception as e:
        return None, None, f"Error processing audio: {str(e)}", None, None

with gr.Blocks() as iface:
    gr.Markdown("# Emotional Voice Assistant")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.File(label="Upload audio file", file_types=["audio/*"])
            show_mel_spec_db = gr.Image(label="Mel Spectrogram")
        with gr.Column():
            emotion_output = gr.Label(label="Detected Emotions")
            transcription = gr.Textbox(label="Transcribed Text", lines=2)
            gpt_response_text = gr.Textbox(label="Assistant's Response", lines=3)
            response_audio = gr.Audio(label="Assistant's Voice Response")
    
    analyze_btn = gr.Button("Analyze and Respond")
    analyze_btn.click(
        fn=classify_and_respond,
        inputs=audio_input,
        outputs=[
            show_mel_spec_db,
            emotion_output,
            transcription,
            gpt_response_text,
            response_audio
        ]
    )

iface.launch(share=True)
