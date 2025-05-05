import librosa
import soundfile as sf

audio_path = "sample_audio.wav"
output_path = "sample_audio_16k.wav"

# Load and resample to 16kHz
audio, sr = librosa.load(audio_path, sr=16000)
sf.write(output_path, audio, 16000)

print("Converted audio saved as:", output_path)

