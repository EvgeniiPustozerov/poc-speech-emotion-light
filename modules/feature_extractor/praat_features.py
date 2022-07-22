import numpy as np
import parselmouth
# noinspection PyUnresolvedReferences
from parselmouth.praat import call


# noinspection PyUnusedLocal
def get_praat_features(filename, dict_formants, unit):
    sound = parselmouth.Sound(filename)
    pitch_floor = 117.5  # 512 points for the 22050 Hz frequency
    pitch_ceiling = 600  # Default value for PRAAT
    pitch = call(sound, "To Pitch", 0.0, pitch_floor, pitch_ceiling)  # Create a praat pitch object
    point_process = call([sound, pitch], "To PointProcess (cc)")
    # time step, min pitch in Hz, silence threshold, periods per window
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)

    # Pitch
    f0_mean_praat = call(pitch, "Get mean", 0, 0, unit)  # Get the mean pitch (file_name, time range, time range, unit)
    f0_sd_praat = call(pitch, "Get standard deviation", 0, 0, unit)  # Get the standard deviation
    f0_min_praat = call(pitch, "Get minimum", 0, 0, 'Hertz', 'None')  # Last arg - interpolation type: None / Parabolic
    f0_max_praat = call(pitch, "Get maximum", 0, 0, 'Hertz', 'None')
    f0_range_praat = f0_max_praat - f0_min_praat

    # HNR
    mean_hnr = call(harmonicity, "Get mean", 0, 0)  # Harmonic to Noise

    # Jitter
    local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
    local_absolute_jitter = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rap_jitter = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3) * 100
    ppq5_jitter = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3) * 100
    ddp_jitter = call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3) * 100

    # Shimmer
    local_shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
    local_db_shimmer = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3_shimmer = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
    apq5_shimmer = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 2.4) * 100
    dda_shimmer = call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100

    # Intensity
    intensity = sound.to_intensity(minimum_pitch=100)
    avg_intensity = np.mean(intensity.values)

    # Formants
    n_formants = 5
    formant = sound.to_formant_burg(max_number_of_formants=n_formants, maximum_formant=5000, window_length=0.023)
    T = formant.xs()
    F = [tuple(formant.get_value_at_time(f, t) for f in range(1, n_formants + 1)) for t in T]

    def formant_parameters(n1, F1):
        f_n = [data[n1 - 1] for data in F1]  # data[0] - F1 first formant
        f_n[f_n == 0] = np.nan
        f_n_mean = np.nanmean(f_n)
        f_n_sd = np.nanstd(f_n)
        f_n_range = np.nanmax(f_n) - np.nanmin(f_n)
        formant_data = np.array([f_n_mean, f_n_sd, f_n_range])

        return formant_data

    formants_data = np.array([])
    for n in range(1, n_formants + 1):
        formants_data = np.hstack((formants_data, formant_parameters(n, F)))

    if not list(dict_formants.keys()):
        formants_names = []
        post = ('mean', 'sd', 'range')
        for d in range(1, n_formants + 1):
            for p in post:
                formants_names.append(f"f{d}_{p}")
        dict_formants = dict.fromkeys(formants_names)

    for idx, key in enumerate(dict_formants.keys()):
        dict_formants[key] = formants_data[idx]

    praat_features = {}
    for variable in ['f0_mean_praat', 'f0_sd_praat', 'f0_range_praat',
                     'mean_hnr', 'local_jitter', 'local_absolute_jitter', 'rap_jitter', 'ppq5_jitter', 'ddp_jitter',
                     'local_shimmer', 'local_db_shimmer', 'apq3_shimmer', 'apq5_shimmer', 'dda_shimmer',
                     'avg_intensity']:
        praat_features[variable] = eval(variable)

    return praat_features, dict_formants
