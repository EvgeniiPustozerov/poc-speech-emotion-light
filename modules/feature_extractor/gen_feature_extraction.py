from modules.feature_extractor.librosa_features import get_librosa_features
from modules.feature_extractor.nlm_features import get_nlm_features
from modules.feature_extractor.pause_features import get_pause_features
from modules.feature_extractor.praat_features import get_praat_features


def get_features(file_name):
    # PRAAT features
    dict_formants = {}
    praat_features, dict_formants = get_praat_features(file_name, dict_formants, unit="Hertz")

    # Librosa features
    dict_mfcc = {}
    dict_mfcc = get_librosa_features(file_name, dict_mfcc)

    # PAUSE features
    pause_features = get_pause_features(file_name, min_pause=0.02)

    # Non-linear model features
    nlm_features = get_nlm_features(file_name)

    # all features merged
    # full_features = praat_features | pause_features | librosa_features | dict_formants | dict_mfcc
    full_features = praat_features | pause_features | dict_formants | dict_mfcc | nlm_features

    return full_features
