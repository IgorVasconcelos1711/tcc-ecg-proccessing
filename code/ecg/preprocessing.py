import numpy as np
from scipy.signal import butter, lfilter

from .config import DEFAULT_FS, FILTER_ORDER


def z_score(signal):
    media = np.mean(signal)
    desvio_padrao = np.std(signal)

    if desvio_padrao == 0:
        return signal - media

    return (signal - media) / desvio_padrao


def minmax(sinal_janela):
    s_min = np.min(sinal_janela)
    s_max = np.max(sinal_janela)

    if s_max == s_min:
        return np.zeros_like(sinal_janela)

    return (sinal_janela - s_min) / (s_max - s_min)


def butterworth_filter(signal, order, fs=DEFAULT_FS):
    nyquist = fs / 2
    low_cut = 5
    high_cut = 15
    low, high = low_cut / nyquist, high_cut / nyquist

    b, a = butter(order, [low, high], btype="band")
    return lfilter(b, a, signal)


def causal_derivative(filtered_signal, fs):
    T = 1 / fs
    y = np.zeros_like(filtered_signal)

    for n in range(4, len(filtered_signal)):
        y[n] = (
            filtered_signal[n]
            + 2 * filtered_signal[n - 1]
            - 2 * filtered_signal[n - 3]
            - filtered_signal[n - 4]
        ) / (8 * T)

    return y


def square_signal(filtered_signal):
    return np.square(filtered_signal)


def moving_window_integration(squared_signal, fs):
    window_width = int(0.150 * fs)
    kernel = np.ones(window_width) / window_width
    return np.convolve(squared_signal, kernel, mode="same")


def preprocess_signal(raw_signal, order=FILTER_ORDER, fs=DEFAULT_FS, normalization_type="z_score"):
    filtered_signal = butterworth_filter(raw_signal, order, fs)
    if normalization_type == "z_score":
        return z_score(filtered_signal)
    if normalization_type == "min_max":
        return minmax(filtered_signal)
    raise ValueError(f"Unknown normalization_type: {normalization_type}")


def pan_tompkins_preprocess(raw_signal, order=FILTER_ORDER, fs=DEFAULT_FS):
    filtered_signal = butterworth_filter(raw_signal, order, fs)
    derivated_signal = causal_derivative(filtered_signal, fs)
    squared = square_signal(derivated_signal)
    integrated_signal = moving_window_integration(squared, fs)
    return filtered_signal, derivated_signal, squared, integrated_signal


def get_pt_signals(raw_signal, order=FILTER_ORDER, fs=DEFAULT_FS):
    ch1 = pan_tompkins_preprocess(raw_signal[:, 0], order=order, fs=fs)
    ch2 = pan_tompkins_preprocess(raw_signal[:, 1], order=order, fs=fs)

    filtered_signal = np.column_stack((ch1[0], ch2[0]))
    derivated_signal = np.column_stack((ch1[1], ch2[1]))
    squared_signal = np.column_stack((ch1[2], ch2[2]))
    integrated_signal = np.column_stack((ch1[3], ch2[3]))
    return filtered_signal, derivated_signal, squared_signal, integrated_signal
