import os
import time
from datetime import datetime
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import torch
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel
import torch.nn as nn


class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits


class VoiceEmotion:
    def __init__(self, model_name='audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim', device='cpu'):
        self.device = device
        self.model_name = model_name
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = EmotionModel.from_pretrained(model_name).to(device)
        self.sampling_rate = 16000
        os.makedirs("recordings", exist_ok=True)

    def record_and_save(self, duration=5):
        print(f"Recording {duration}s of audio...")
        recording = sd.rec(int(duration * self.sampling_rate), samplerate=self.sampling_rate, channels=1, dtype='float32')
        sd.wait()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recordings/voice_{timestamp}.wav"
        write(filename, self.sampling_rate, recording)
        print(f"Saved: {filename}")
        return filename

    def analyze_voice(self, file_path):
        from scipy.io import wavfile
        sr, data = wavfile.read(file_path)
        if data.ndim > 1:
            data = data[:, 0]  # mono
        data = data.astype(np.float32) / np.max(np.abs(data))
        inputs = self.processor(data, sampling_rate=sr, return_tensors="pt")
        input_values = inputs["input_values"].to(self.device)

        with torch.no_grad():
            _, logits = self.model(input_values)

        emotions = logits.cpu().numpy()[0]
        return dict(arousal=emotions[0], dominance=emotions[1], valence=emotions[2])

    def interpret_emotion(self, vad):
        v, a, d = vad['valence'], vad['arousal'], vad['dominance']
        high = 0.6
        low = 0.4

        if v > high and a > high:
            return "Excited"
        elif v > high and a < low:
            return "Calm"
        elif v < low and a > high:
            return "Angry or Fearful"
        elif v < low and a < low:
            return "Sad"
        elif low <= v <= high and a > high:
            return "Alert or Anxious"
        elif low <= v <= high and a < low:
            return "Tired or Bored"
        else:
            return "Neutral or Mixed"

    # Function to analyze the latest captured face without a loop
    def analyze_latest(self):
        latest_voice = max(os.listdir('recordings'), key=lambda x: os.path.getctime(os.path.join('recordings', x)))
        recording_path = os.path.join('recordings', latest_voice)
        return self.analyze_voice(recording_path)


    def run(self, duration=5, iterations=3, delay=2):
        for i in range(iterations):
            file_path = self.record_and_save(duration=duration)
            emotions = self.analyze_voice(file_path)
            emotion_label = self.interpret_emotion(emotions)
            print(f"Emotion Prediction [{i + 1}]: {emotions} → {emotion_label}")
            time.sleep(delay)

