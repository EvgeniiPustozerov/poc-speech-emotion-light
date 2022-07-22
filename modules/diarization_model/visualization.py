import librosa
import matplotlib.pyplot as plt
import numpy as np
from pyannote.audio.utils.signal import binarize
from pyannote.core import notebook
from pyannote.audio import Pipeline


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


def make_inference_plot(inference, sound_filename):
    output = inference(sound_filename)
    output = binarize(output)
    speakers = np.reshape(output.data, (output.data.shape[0] * output.data.shape[1], output.data.shape[2]))
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    signal, sample_rate = librosa.load(sound_filename)
    add_waveform(signal, sample_rate=sample_rate, ax=ax1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(np.arange(len(speakers)), speakers, linewidth=2.0)
    ax2.axis([0, len(speakers), -0.2, 1.2])
    ax2.set_xlabel('time (secs)', fontsize=18)
    ax2.legend(['Speaker 1', 'Speaker 2', 'Speaker 3', 'Speaker 4'], loc='upper right')
    plt.show()


def make_annotated_inference_plot(inference, sound_file):

    # Signal waveform
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    signal, sample_rate = librosa.load("samples/crema_d_diarization/" + sound_file["uri"] + ".wav")

    # Voice activity
    pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")
    voice_activity = pipeline(sound_file)
    voice_activity = voice_activity.discretize(notebook.crop, resolution=0.010)
    ax12 = ax1.twinx()
    ax12.plot(np.arange(len(voice_activity)), voice_activity*0.1, linewidth=1.0)
    ax12.axis([0, len(voice_activity), -0.2, 0.2])
    add_waveform(signal, sample_rate=sample_rate, ax=ax1)

    # Predicted speakers
    output = binarize(inference(sound_file))
    speakers = np.reshape(output.data, (output.data.shape[0] * output.data.shape[1], output.data.shape[2]))
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(np.arange(len(speakers)), speakers, linewidth=2.0)
    ax2.axis([0, len(speakers), -0.2, 1.2])
    ax2.legend(['Speaker 1', 'Speaker 2', 'Speaker 3', 'Speaker 4'], loc='upper right')

    # Referenced speakers
    reference = sound_file["annotation"].discretize(notebook.crop, resolution=0.010)
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(np.arange(len(reference)), reference, linewidth=2.0)
    ax3.axis([0, len(reference), -0.2, 1.2])
    plt.show()
