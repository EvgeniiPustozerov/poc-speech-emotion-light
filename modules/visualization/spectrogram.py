import os

import librosa
import matplotlib.pyplot as plt
import numpy as np


def make_spectrogram(file_path):
    signal, sample_rate = librosa.load(file_path, sr=None)
    spectrogram_file_name = display_waveform(signal, file_path, sample_rate)
    return spectrogram_file_name


def display_waveform(signal, file_name, sample_rate=16000, text='Audio recording', overlay_color=None):
    os.makedirs("../../images", exist_ok=True)
    if overlay_color is None:
        overlay_color = []
    fig, ax = plt.subplots(1, 1)
    fig.set_figwidth(10)
    fig.set_figheight(3)
    plt.scatter(np.arange(len(signal)), signal, s=1, marker='o', c='k')
    if len(overlay_color):
        plt.scatter(np.arange(len(signal)), signal, s=1, marker='o', c=overlay_color)
    fig.suptitle(text, fontsize=16)
    plt.ylabel('signal strength', fontsize=14)
    plt.axis([0, len(signal), -0.5, +0.5])
    time_axis, _ = plt.xticks()
    plt.xticks(time_axis[:-1], np.round(time_axis[:-1] / sample_rate, 1))
    spectrogram_file_name = 'images/' + file_name.split('/')[2].split('.')[0] + '.png'
    plt.savefig(spectrogram_file_name)
    return spectrogram_file_name


def add_waveform(signal, sample_rate=16000, ax=None, overlay_color=None):
    if ax is None:
        ax = plt.gca()
    if overlay_color is None:
        overlay_color = []
    plt.scatter(np.arange(len(signal)), signal, s=1, marker='o', c='k')
    if len(overlay_color):
        plt.scatter(np.arange(len(signal)), signal, s=1, marker='o', c=overlay_color)
    plt.ylabel('signal strength', fontsize=14)
    plt.axis([0, len(signal), -0.5, +0.5])
    time_axis, _ = plt.xticks()
    plt.xticks(time_axis[:-1], np.round(time_axis[:-1] / sample_rate, 1))
    return ax


def get_color(signal, speech_labels, sample_rate=16000):
    c = np.array(['k'] * len(signal))
    for time_stamp in speech_labels:
        start, end, label = time_stamp.split()
        start, end = int(float(start) * sample_rate), int(float(end) * sample_rate),
        if label == "speech":
            code = 'red'
        else:
            code = COLORS[int(label.split('_')[-1])]
        c[start:end] = code
    return c


COLORS = "b g c m y".split()
