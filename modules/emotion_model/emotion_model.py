from pickle import dump, load

import pandas as pd
import xgboost as xgb
from sklearn import preprocessing

FOLDER_MODELS = "data/models/"
FOLDER_TABLES = "data/tables/"


def analyze_emotions(X_test):
    dict_answer = make_prediction(X_test)
    dict_answer_sorted = sorted(((v, k) for k, v in dict_answer.items()), reverse=True)

    if max(dict_answer.values()) < 0.5:
        model_answer = "EMOTION: \n No dominant emotion.\n" \
                       + dict_answer_sorted[0][1] + ": " + '{:.1%}'.format(dict_answer_sorted[0][0]) + "\n" \
                       + dict_answer_sorted[1][1] + ": " + '{:.1%}'.format(dict_answer_sorted[1][0])
    else:
        model_answer = "EMOTION: \n" \
                       + dict_answer_sorted[0][1] + ": " + '{:.1%}'.format(dict_answer_sorted[0][0]) + "\n" \
                       + dict_answer_sorted[1][1] + ": " + '{:.1%}'.format(dict_answer_sorted[1][0]) + "\n" \
                       + dict_answer_sorted[2][1] + ": " + '{:.1%}'.format(dict_answer_sorted[2][0])
    return model_answer


def train_scaler():
    supplemental_columns = ["filename", "actor", "word", "emotions", "intensity", "repetition"]
    df = pd.read_excel(FOLDER_TABLES + "df_stress.xlsx", usecols=lambda x: x not in supplemental_columns)
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(df)
    dump(min_max_scaler, open(FOLDER_MODELS + 'emotion_scaler.pkl', 'wb'))


def make_prediction(X_test):
    # train_scaler()
    min_max_scaler = load(open(FOLDER_MODELS + 'emotion_scaler.pkl', 'rb'))
    X_test = {k: [v] for k, v in X_test.items()}
    X_test = pd.DataFrame.from_dict(X_test)
    X_test = min_max_scaler.transform(X_test)
    best_model = xgb.Booster()
    best_model.load_model(FOLDER_MODELS + 'emotion_model.model')
    predicted = best_model.predict(xgb.DMatrix(X_test))[0]
    emo_dict = {'Anger': 0, 'Disgust': 1, 'Fear': 2, 'Happiness': 3, 'Neutrality': 4, 'Sadness': 5}
    predicted_class_probs = emo_dict
    for key, value in emo_dict.items():
        predicted_class_probs[key] = predicted[value]
    # predicted_class_probs = collections.OrderedDict(predicted_class_probs)
    return predicted_class_probs
