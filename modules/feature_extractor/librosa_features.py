import librosa
import numpy as np
from scipy import stats


def get_librosa_features(filename, dict_mfcc):
    sound, sr = librosa.load(filename)

    # Fundamental frequency
    # f0 = librosa.yin(sound, frame_length=512, fmin=117.5, fmax=600, sr=sr)
    # f0_mean_librosa = np.nanmean(f0)
    # f0_sd_librosa = np.nanstd(f0)
    # f0_range_librosa = np.nanmax(f0)-np.nanmin(f0)

    # MFCC
    mfcc = librosa.feature.mfcc(y=sound, sr=sr, n_mfcc=13)
    mfcc_des = stats.describe(mfcc.T)
    mfcc_mean = mfcc_des.mean[1:]  # mean
    mfcc_variance = mfcc_des.variance[1:]  # variance
    mfcc_min, mfcc_max = mfcc_des.minmax
    mfcc_min = mfcc_min[1:]  # min
    mfcc_max = mfcc_max[1:]  # max
    mfcc_kurtosis = mfcc_des.kurtosis[1:]  # kurtosis
    mfcc_skewness = mfcc_des.skewness[1:]  # skewness
    mfcc_data = np.hstack((mfcc_mean, mfcc_variance, mfcc_min, mfcc_max, mfcc_kurtosis, mfcc_skewness))

    if not list(dict_mfcc.keys()):
        # if df_mfcc doesn't exist
        mfcc_names = []
        post = ('mean', 'variance', 'min', 'max', 'kurtosis', 'skewness')
        for p in post:
            for d in range(1, len(mfcc_mean) + 1):
                mfcc_names.append(f"MFCC{d}_{p}")
        dict_mfcc = dict.fromkeys(mfcc_names)

    for idx, key in enumerate(dict_mfcc.keys()):
        dict_mfcc[key] = mfcc_data[idx]

    return dict_mfcc
