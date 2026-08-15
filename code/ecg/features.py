import numpy as np

from .config import ARRHYTHMIA_SYMBOLS


def initialize_templates(windows_list, channel_idx=0):
    candidates = [item.get("window", item.get("signal")) for item in windows_list[:6]]

    if len(candidates) < 2:
        raise ValueError("Número insuficiente de candidatos (mínimo 2).")

    areas = np.array([np.sum(np.abs(cand[:, channel_idx])) for cand in candidates])
    mean_area = np.mean(areas)
    diff_to_mean = np.abs(areas - mean_area)

    group_lower_idx = np.where(areas < mean_area)[0]
    group_higher_idx = np.where(areas >= mean_area)[0]

    group_lower_idx = group_lower_idx[np.argsort(diff_to_mean[group_lower_idx])]
    group_higher_idx = group_higher_idx[np.argsort(diff_to_mean[group_higher_idx])]

    ranked_indices = np.concatenate((group_lower_idx, group_higher_idx))

    templates = None
    for i in range(len(ranked_indices) - 1):
        idx1, idx2 = ranked_indices[i], ranked_indices[i + 1]
        cand1 = candidates[idx1][:, channel_idx]
        cand2 = candidates[idx2][:, channel_idx]
        corr = np.corrcoef(cand1, cand2)[0, 1]
        if corr > 0.95:
            templates = [candidates[idx1], candidates[idx2]]
            break

    if templates is None:
        templates = [candidates[ranked_indices[0]], candidates[ranked_indices[1]]]

    return templates


def update_templates(templates, new_normal_window, channel_idx=0):
    cand_new = new_normal_window[:, channel_idx]
    corr0 = np.corrcoef(cand_new, templates[0][:, channel_idx])[0, 1]
    corr1 = np.corrcoef(cand_new, templates[1][:, channel_idx])[0, 1]

    if corr0 > corr1:
        templates[0] = new_normal_window
    else:
        templates[1] = new_normal_window

    return templates


def extract_average_energy(window, window_size=0):
    signal = window["signal"] if isinstance(window, dict) else window
    if window_size == 0:
        window_size = len(signal)
    segment = signal[:window_size]
    return float(np.mean(np.square(segment)))


def split_annotations_by_type(annotation, fs, arrhythmia_symbols=None):
    if arrhythmia_symbols is None:
        arrhythmia_symbols = ARRHYTHMIA_SYMBOLS

    normal_times = []
    arrhythmia_times = []
    for sample, symbol in zip(annotation.sample, annotation.symbol):
        time = sample / fs
        member = (time, symbol)
        if symbol in arrhythmia_symbols:
            arrhythmia_times.append(member)
        else:
            normal_times.append(member)
    return normal_times, arrhythmia_times
