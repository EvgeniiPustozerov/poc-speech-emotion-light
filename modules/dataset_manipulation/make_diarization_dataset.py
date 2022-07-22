import glob
import random
from pydub import AudioSegment

from modules.diarization_model.vad_model import detect_vad


def add_line_to_rttm(timestamp_1, timestamp_2, chunk_file_name, file_name):
    speaker_label = chunk_file_name.split("\\")[1].split("_")[0]
    rttm_line = "SPEAKER " + file_name + " 1 " + str(round(timestamp_1, 3)) + " " + str(round((
        timestamp_2 - 0.001), 3)) + " <NA> <NA> SPEAKER_" + speaker_label + " <NA> <NA>" + "\n"
    return rttm_line


def add_line_to_uem(timestamp_1, timestamp_2, file_name):
    uem_line = file_name + " NA " + str(round(timestamp_1, 3)) + " " + str(round((timestamp_2 - 0.001), 3)) + "\n"
    return uem_line


def make_diarization_dataset(speakers=2, records=500, phrases=6, persist_emotion=True):
    FOLDER_AUDIOS = "samples/crema_d/"
    FOLDER_EXPORT = "samples/crema_d_diarization/"
    DATASET_LIST = "info/configs/diarization/lists/dataset_diarization_reference.lst"
    RTTM_CONFIG = "info/configs/diarization/rttms/dataset_diarization_reference.rttm"
    OEM_CONFIG = "info/configs/diarization/uems/dataset_diarization_reference.uem"
    list_all_audio = glob.glob("samples/crema_d/*.wav")
    list_speakers = set([w.split("\\")[1].split("_")[0] for w in list_all_audio])
    list_emotions = set([w.split("\\")[1].split("_")[2] for w in list_all_audio])
    phrases_per_speaker = int(phrases / speakers)
    rttms = ""
    uems = ""
    files = ""

    for j in range(0, records):
        print("Making a sample: %s" % j)
        chosen_speakers = []
        chosen_phrases = []
        file_name = str(j + 1001)
        for i in range(0, speakers):
            chosen_speakers.append(random.choice(list(list_speakers)))
            if persist_emotion:
                chosen_emotion = random.choice(list(list_emotions))
                current_speaker_phrases = glob.glob(
                    FOLDER_AUDIOS + chosen_speakers[i] + "*" + chosen_emotion + "*.wav")
            else:
                current_speaker_phrases = glob.glob(FOLDER_AUDIOS + chosen_speakers[i] + "*.wav")
            chosen_phrases.extend(random.sample(current_speaker_phrases, phrases_per_speaker))
        random.shuffle(chosen_phrases)
        combined_phrases = AudioSegment.from_wav(chosen_phrases[0])
        timestamp_1, timestamp_2 = detect_vad(chosen_phrases[0])
        rttms = rttms + add_line_to_rttm(
            timestamp_1, timestamp_2, chosen_phrases[0], file_name)
        for i in range(1, len(chosen_phrases)):
            sound = AudioSegment.from_wav(chosen_phrases[i])
            timestamp_1 = combined_phrases.duration_seconds
            combined_phrases = combined_phrases + sound
            timestamp_1_1, timestamp_2 = detect_vad(chosen_phrases[i])
            if timestamp_2 == 0:
                timestamp_2 = sound.duration_seconds
            rttms = rttms + add_line_to_rttm(timestamp_1 + timestamp_1_1, timestamp_2, chosen_phrases[i], file_name)
        uems = uems + add_line_to_uem(0.000, combined_phrases.duration_seconds, file_name)
        combined_phrases.export(FOLDER_EXPORT + file_name + ".wav", format="wav")
        files = files + file_name + "\n"
    with open(RTTM_CONFIG, "w") as text_file:
        text_file.write(rttms)
    with open(OEM_CONFIG, "w") as text_file:
        text_file.write(uems)
    with open(DATASET_LIST, "w") as text_file:
        text_file.write(files)
    return None


make_diarization_dataset(records=1000)
