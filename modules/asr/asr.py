import os

from nemo.collections.asr.models import EncDecCTCModel, EncDecRNNTModel
from nemo.collections.nlp.models import PunctuationCapitalizationModel, TextClassificationModel

FOLDER_MODELS = "data/models/"
print(EncDecRNNTModel.list_available_models())
punctuation_capitalization_model = PunctuationCapitalizationModel.from_pretrained("punctuation_en_distilbert")
sentiment_analysis_model_path = os.path.join(FOLDER_MODELS, "text_classification_model.nemo")
sentiment_analysis_model = TextClassificationModel.restore_from(restore_path=sentiment_analysis_model_path)


def sentiment_analysis(text):
    result = sentiment_analysis_model.classifytext(queries=text)[0]
    if result == 1:
        text_sentiment = "SENTIMENT: positive"
    else:
        text_sentiment = "SENTIMENT: negative"
    return text_sentiment


def punctuation_capitalization(text):
    return punctuation_capitalization_model.add_punctuation_capitalization(text)


def recognize_speech(file_path):
    model = EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
    words = model.transcribe(paths2audio_files=[file_path])
    text = punctuation_capitalization(words)
    return text, None, None


def recognize_speech_ru(file_path):
    model = EncDecCTCModel.from_pretrained(model_name="stt_ru_quartznet15x5")
    words = model.transcribe(paths2audio_files=[file_path])
    return words
