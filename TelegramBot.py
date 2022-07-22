import glob
import random
import sys

from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from pydub import AudioSegment

from modules.asr import asr
from modules.emotion_model.emotion_model import analyze_emotions
from modules.feature_extractor.gen_feature_extraction import get_features
from modules.visualization.spectrogram import make_spectrogram

TELEGRAM_API_TOKEN = ''

# Initialize bot and dispatcher
print(sys.version)
bot = Bot(token=TELEGRAM_API_TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())
audio_file = ''


def make_wav_from_ogg(ogg_file_path):
    file_name = ogg_file_path.split("/")[-1].split(".")[0]
    sound = AudioSegment.from_ogg(ogg_file_path)
    wav_file_path = "data/audio/wav/" + file_name + ".wav"
    sound.export(wav_file_path, format="wav")
    return wav_file_path


async def process_audio(msg, wav_file_path):
    spectrogram_file_name = make_spectrogram(wav_file_path)
    await bot.send_photo(msg.chat.id, photo=open(spectrogram_file_name, 'rb'))
    feature_vector = get_features(wav_file_path)
    stress_probabilities = analyze_emotions(feature_vector)
    await bot.send_message(msg.chat.id, stress_probabilities)
    words, transcript, pic_file_name = asr.recognize_speech(wav_file_path)
    await bot.send_message(msg.chat.id, ' '.join(word for word in words))
    sentiment = asr.sentiment_analysis(words)
    await bot.send_message(msg.chat.id, sentiment)
    if transcript is not None:
        await bot.send_message(msg.chat.id, ' '.join(word for word in transcript))
        await bot.send_photo(msg.chat.id, photo=open(pic_file_name, 'rb'))


# Introduction and possibility to choose a sample
@dp.message_handler(commands='start')
async def introduction(message: types.Message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add("Anger")
    markup.add("Disgust")
    markup.add("Fear")
    markup.add("Happiness")
    markup.add("Neutrality")
    markup.add("Sadness")
    markup.add("Speaker Diarization (Experimental)")
    await message.reply("Hello! I am a chatbot by Exposit that recognizes emotions on your speech. Send me your voice "
                        "message.",
                        reply_markup=markup)


# Operations when the user has sent his/her own audio
@dp.message_handler(content_types=types.ContentTypes.VOICE | types.ContentTypes.AUDIO)
async def audio_message_handler(msg: types.Message):
    if msg.content_type == "voice":
        await msg.reply("Thank you for your voice message. It will now be processed.")
    elif msg.content_type == "audio":
        await msg.reply("Thank you for your audio message. It will now be processed.")
    file_id = msg.voice.file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path
    ogg_file_path = "data/audio/ogg/" + str(msg.from_user.id) + "_" + file_id + ".ogg"
    await bot.download_file(file_path, ogg_file_path)
    wav_file_path = make_wav_from_ogg(ogg_file_path)
    await process_audio(msg, wav_file_path)


# Operations when the user has chosen one of the sample audios
@dp.message_handler(content_types=types.ContentTypes.TEXT)
async def select_operation(msg: types.Message):
    samples = ["Anger", "Disgust", "Fear", "Happiness", "Neutrality", "Sadness"]
    emo_dict = \
        {"Anger": "ANG", "Disgust": "DIS", "Fear": "FEA", "Happiness": "HAP", "Neutrality": "NEU", "Sadness": "SAD"}
    if msg.text in samples:
        tag = emo_dict[msg.text]
        sample_list = glob.glob('samples/crema_d/*' + tag + "*.wav")
        wav_file_path = random.choice(sample_list).replace('\\', '/')
        await bot.send_voice(msg.chat.id, voice=open(wav_file_path, 'rb'))
        await process_audio(msg, wav_file_path)
    if msg.text == "Speaker Diarization (Experimental)":
        sample_list = glob.glob('samples/crema_d_diarization/*.wav')
        wav_file_path = random.choice(sample_list).replace('\\', '/')
        await bot.send_voice(msg.chat.id, voice=open(wav_file_path, 'rb'))
        await process_audio(msg, wav_file_path)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=False)
