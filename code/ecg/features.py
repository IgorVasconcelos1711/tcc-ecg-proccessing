import numpy as np

from .config import ARRHYTHMIA_SYMBOLS


# TODO: single-channel handling is temporary. ML will use both ECG channels;
# keep 2D (samples, channels) windows and drop this 1D fallback when that lands.
def _as_1d(window, channel_idx=0):
    arr = np.asarray(window)
    if arr.ndim >= 2:
        return arr[:, channel_idx]
    return arr


def _window_signal(window, channel_idx=0):
    if isinstance(window, dict):
        window = window.get("window", window.get("signal"))
    return _as_1d(window, channel_idx)


def initialize_templates(windows_list, channel_idx=0):
    """Hearty / Gradl template init from the first 6 candidate beats."""
    candidates = [item.get("window", item.get("signal")) for item in windows_list[:6]]

    if len(candidates) < 2:
        raise ValueError("Número insuficiente de candidatos (mínimo 2).")

    areas = np.array([np.sum(np.abs(_as_1d(cand, channel_idx))) for cand in candidates])
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
        cand1 = _as_1d(candidates[idx1], channel_idx)
        cand2 = _as_1d(candidates[idx2], channel_idx)
        corr = np.corrcoef(cand1, cand2)[0, 1]
        if corr > 0.95:
            templates = [candidates[idx1], candidates[idx2]]
            break

    if templates is None:
        templates = [candidates[ranked_indices[0]], candidates[ranked_indices[1]]]

    return templates


NORMAL_TEMPLATE_SYMBOLS = {"N"}


def update_templates(templates, new_normal_window, channel_idx=0):
    """Replace the more correlated template with a beat labeled normal (Gradl)."""
    new_wave = _window_signal(new_normal_window, channel_idx)
    cand_new = _as_1d(new_wave, channel_idx)
    corr0 = np.corrcoef(cand_new, _as_1d(templates[0], channel_idx))[0, 1]
    corr1 = np.corrcoef(cand_new, _as_1d(templates[1], channel_idx))[0, 1]

    if corr0 > corr1:
        templates[0] = new_wave
    else:
        templates[1] = new_wave

    return templates


def is_normal_beat(label, normal_symbols=None):
    if normal_symbols is None:
        normal_symbols = NORMAL_TEMPLATE_SYMBOLS
    return label in normal_symbols


def adapt_templates(
    windows_list,
    templates,
    start_idx=6,
    channel_idx=0,
    normal_symbols=None,
    is_normal=None,
):
    """Walk beats after template init and update only on normal QRS.

    Pass a callable ``is_normal(window, index) -> bool`` to drive updates from
    the Gradl decision tree; otherwise MIT-BIH symbol ``N`` is used.
    """
    adapted = [np.array(t, copy=True) for t in templates]
    n_updates = 0

    for i, window in enumerate(windows_list[start_idx:], start=start_idx):
        if is_normal is None:
            label = window["label"] if isinstance(window, dict) else None
            should_update = is_normal_beat(label, normal_symbols)
        else:
            should_update = is_normal(window, i)

        if should_update:
            update_templates(adapted, window, channel_idx=channel_idx)
            n_updates += 1

    return adapted, n_updates


def extract_average_energy(window, window_size=0):
    signal = window["signal"]
    if window_size == 0:
        window_size = len(signal)
    segment = signal[:window_size]
    return np.mean(np.square(segment))


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


def _ardiff_vs_template(beat_area, template_area):
    """Hearty/Krasteva ratio vs template area (Gradl decision-tree thresholds)."""
    if template_area == 0:
        return 0.0
    if beat_area > template_area:
        return float((beat_area - template_area) / template_area)
    return float((template_area - beat_area) / template_area)


def extract_ardiff(window, templates, channel_idx=0):
    """Minimum absolute-area difference vs both templates (Krasteva ArDiff)."""
    beat_area = float(np.sum(np.abs(_window_signal(window, channel_idx))))
    diffs = [
        _ardiff_vs_template(beat_area, float(np.sum(np.abs(_as_1d(t, channel_idx)))))
        for t in templates
    ]
    return float(min(diffs))


def extract_maxcorr(window, templates, channel_idx=0):
    """Maximal normalized cross-correlation vs both templates (Krasteva MaxCorr)."""
    # TODO: for real-time / low-power, use lag-0 (R-aligned) correlation instead of
    # np.correlate(..., mode="full"). Gradl notes that cheaper fallback.
    beat = _window_signal(window, channel_idx)
    beat = beat - np.mean(beat)
    beat_norm = np.linalg.norm(beat)

    max_corr = -1.0
    for template in templates:
        tmpl = _as_1d(template, channel_idx)
        tmpl = tmpl - np.mean(tmpl)
        tmpl_norm = np.linalg.norm(tmpl)
        if tmpl_norm == 0:
            continue
        corr = np.correlate(beat, tmpl, mode="full") / (beat_norm * tmpl_norm)
        max_corr = max(max_corr, float(np.max(corr)))
    return max_corr if max_corr > -1.0 else 0.0


def extract_qrs_width(filtered_window, fs, compensation=0.85, channel_idx=0):
    """QRS width in ms from bandpass Q–S span (Hearty PanTompkins fallback)."""
    signal = _window_signal(filtered_window, channel_idx)
    if len(signal) < 3:
        return 0.0

    r_idx = len(signal) // 2
    pre_samples = int(0.08 * fs)
    post_samples = int(0.08 * fs)
    gap = int(0.015 * fs)

    q_start = max(0, r_idx - pre_samples)
    q_end = max(q_start, r_idx - gap)
    q_region = signal[q_start:q_end]
    q_idx = q_start + int(np.argmin(q_region)) if len(q_region) else r_idx

    s_start = min(len(signal), r_idx + gap)
    s_end = min(len(signal), r_idx + post_samples)
    s_region = signal[s_start:s_end]
    s_idx = s_start + int(np.argmin(s_region)) if len(s_region) else r_idx

    return max((s_idx - q_idx) / fs * 1000.0 * compensation, 0.0)


def extract_rr_interval(prev_peak_idx, peak_idx, fs):
    """R-R interval in ms between consecutive detected peaks."""
    return float((peak_idx - prev_peak_idx) / fs * 1000.0)


def extract_beat_features(
    window,
    templates,
    prev_peak_idx,
    peak_idx,
    fs,
    integrated_window=None,
    filtered_window=None,
    channel_idx=0,
):
    features = {
        "ardiff": extract_ardiff(window, templates, channel_idx=channel_idx),
        "maxcorr": extract_maxcorr(window, templates, channel_idx=channel_idx),
        "rr_ms": extract_rr_interval(prev_peak_idx, peak_idx, fs),
    }
    width_window = filtered_window if filtered_window is not None else window
    features["qrs_width_ms"] = extract_qrs_width(width_window, fs, channel_idx=channel_idx)
    return features
