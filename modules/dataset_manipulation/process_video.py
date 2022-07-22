import subprocess
import os
from pydub import AudioSegment

from modules.asr import asr
from modules.emotion_model.emotion_model import analyze_emotions
from modules.feature_extractor.gen_feature_extraction import get_features
from modules.visualization.spectrogram import make_spectrogram


def convert_video_to_audio_ffmpeg(video_file, output_ext="wav"):
    """Converts video to audio directly using `ffmpeg` command
    with the help of subprocess module"""
    filename, ext = os.path.splitext(video_file)
    subprocess.call(["ffmpeg", "-y", "-i", video_file, f"{filename}.{output_ext}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)


if __name__ == "__main__":
    vf = "samples/video/1.mp4"
    wav_file_path = "samples/video/1.wav"
    convert_video_to_audio_ffmpeg(vf)
    sound = AudioSegment.from_wav(wav_file_path)
    sound = sound.set_channels(1)
    sound.set_frame_rate(16000)
    sound.export(wav_file_path, format="wav")
    make_spectrogram(wav_file_path)
    feature_vector = get_features(wav_file_path)
    stress_probabilities = analyze_emotions(feature_vector)
    print(stress_probabilities)
    words = asr.recognize_speech_ru(wav_file_path)
    print(words)
    sentiment = asr.sentiment_analysis(words)
    print(sentiment)
