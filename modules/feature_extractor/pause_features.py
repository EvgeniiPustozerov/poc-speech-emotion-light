from math import factorial

import numpy as np
import parselmouth


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less than `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    rate: int
        the rate to multiply
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over an odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()

    References
    ----------
    . [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    . [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # Precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # Pad the signal at the extremes with values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


# Records pause points below thr, combines them into a list of pauses
def pauses_by_thr(values, thr):
    pauses_points = []
    for j in range(len(values)):
        if abs(values[j]) < thr:
            pauses_points.append(j)
    pauses = []
    pause_start = 0
    for z in range(len(pauses_points) - 1):
        if pauses_points[z + 1] - pauses_points[z] > 1:
            pauses.append(pauses_points[pause_start:z])
            pause_start = z + 1
    # pauses.append(pauses_points[pause_start:z])
    return pauses


def pause_length_check(pauses, min_pause_len):
    pauses_a_duration = []
    pauses_a_new = pauses.copy()
    for j in range(len(pauses) - 1, -1, -1):
        length = len(pauses[j])
        pauses_a_duration.append(len(pauses[j]))
        if length < min_pause_len:
            pauses_a_new.pop(j)
    pauses = pauses_a_new
    return pauses


# noinspection PyUnusedLocal
def get_pause_features(file_name, min_pause):
    sound = parselmouth.Sound(file_name)
    snd_values = sound.values.ravel()
    min_pause_len = np.ceil(min_pause / sound.dt)
    win_len = int(np.ceil(0.1 / sound.dt))
    sound_abs = abs(snd_values)
    mean_peak = []
    for i in range(len(sound_abs) - win_len):
        mean_peak.append(np.mean(sound_abs[i:i + win_len]) ** 2)

    thr_b = 0.05 * max(mean_peak)
    pause_b = np.ones(win_len // 2, np.bool)
    pause_b = np.append(pause_b, mean_peak <= thr_b)

    pauses_b = []
    pause_start = 0
    for z in range(len(pause_b) - 1):
        if (pause_b[z]) and not (pause_b[z + 1]):
            pauses_b.append(list(range(pause_start, z)))
        elif (pause_b[z + 1]) and not (pause_b[z]):
            pause_start = z + 1

    pauses_b = pause_length_check(pauses_b, min_pause_len)  # Pause length check

    # Features
    if pauses_b[0][0] == 0 and pauses_b[-1][-1] == sound.n_frames - 1:
        speech_units_count = len(pauses_b) - 1
    elif pauses_b[0][0] == 0 or pauses_b[-1][-1] == sound.n_frames - 1:
        speech_units_count = len(pauses_b)
    else:
        speech_units_count = len(pauses_b) + 1

    pauses_duration = []
    for j in range(len(pauses_b)):
        pauses_duration.append(len(pauses_b[j]) * sound.dt)

    # Pause features
    pauses_total = sum(pauses_duration)
    time_talking = sound.duration - pauses_total
    pause_mean = pauses_total / len(pauses_duration)
    pause_std = np.std(pauses_duration)
    pause_var = np.var(pauses_duration)
    pause_rate = pauses_total / sound.duration
    speech_rate = speech_units_count / time_talking

    pause_features = {}
    for variable in ['pauses_total', 'time_talking', 'pause_mean', 'pause_std', 'pause_var', 'pause_rate',
                     'speech_rate']:
        pause_features[variable] = eval(variable)

    return pause_features
