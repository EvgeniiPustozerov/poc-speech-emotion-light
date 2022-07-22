from pyannote.audio import Pipeline


def detect_vad(audio_file):
    pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")
    output = pipeline(audio_file)
    speech = output.get_timeline().support()
    if len(speech) > 0:
        return speech[0].start, speech[0].duration
    else:
        return 0, 0
