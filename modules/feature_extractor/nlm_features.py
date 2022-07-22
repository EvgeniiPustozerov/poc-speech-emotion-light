import numpy as np
import parselmouth
from scipy import signal
from sklearn.preprocessing import MinMaxScaler


def hl_envelopes_idx(s, d_min=1, d_max=1, split=False):
    """
    Input : s: 1d-array, data signal from which to extract high and low envelopes d_min, d_max: int, optional,
    size of chunks, use this if the size of the input signal is too big split: bool, optional, if True,
    split the signal in half along its mean, might help to generate the envelope in some cases Output : l_min,
    l_max : high/low envelope idx of input signal s
    """

    # locals min
    l_min = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    l_max = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s)
        # pre-sorting of locals min based on relative position with respect to s_mid
        l_min = l_min[s[l_min] < s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid
        l_max = l_max[s[l_max] > s_mid]

    # global max of d_max-chunks of locals max
    l_min = l_min[[i + np.argmin(s[l_min[i:i + d_min]]) for i in range(0, len(l_min), d_min)]]
    # global min of d_min-chunks of locals min
    l_max = l_max[[i + np.argmax(s[l_max[i:i + d_max]]) for i in range(0, len(l_max), d_max)]]

    return l_min, l_max


def get_nlm_features(filename):
    sound = parselmouth.Sound(filename)
    snd_values = sound.values.ravel()
    sr = sound.sampling_frequency
    f1 = (100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150)
    f2 = (200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700)
    z = []
    corr, area = [], []
    for f in range(len(f1)):
        order = int(7372 / 8)
        # noinspection PyTypeChecker
        coef = signal.firwin(order, [f1[f], f2[f]], pass_zero=False, fs=sr)
        z.append(signal.filtfilt(coef, [1], snd_values))
        data = np.array(z[-1])
        psi = ((data[1:-1] ** 2) - data[0:-2] * data[1:-1]) * 100000
        corr.append(signal.correlate(psi, psi) / len(z[f]))
        _, high_idx = hl_envelopes_idx(corr[-1])
        d_idx = high_idx - np.array([0, *high_idx[:-1]])
        area_curr = sum(d_idx * corr[-1][high_idx])
        area.append(area_curr)

    area = np.array(area).reshape(len(area), 1)
    scaler = MinMaxScaler()
    area = scaler.fit_transform(area)

    nlm_features = {}

    if not list(nlm_features.keys()):
        # if df_mfcc doesn't exist
        nlm_names = []
        for d in range(1, len(area) + 1):
            nlm_names.append(f"TEO_CB_{d}")
        nlm_features = dict.fromkeys(nlm_names)

    for idx, key in enumerate(nlm_features.keys()):
        nlm_features[key] = float(area[idx])

    return nlm_features
