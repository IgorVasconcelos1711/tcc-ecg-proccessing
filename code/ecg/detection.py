import numpy as np

from .config import (
    ALIGN_LOOKBACK_MS,
    ALIGN_LOOKFORWARD_MS,
    DEFAULT_FS,
    PEAK_TOLERANCE_MS,
)


class PanTompkinsQRS:
    """Pan-Tompkins QRS detection via adaptive thresholds on the integrator."""

    REFRACTORY_MS = 200
    SEARCHBACK_RR_FACTOR = 1.66
    LEARNING_SECONDS = 2.0

    def __init__(self, sampling_rate=DEFAULT_FS):
        self.sampling_rate = sampling_rate
        self.reset()

    def reset(self):
        self.signal_peak = 0.0
        self.noise_peak = 0.0
        self.threshold_1 = 0.0
        self.threshold_2 = 0.0
        self.rr_intervals = []
        self.last_qrs_index = -1

    def _initialize_thresholds(self, integrated_signal):
        init_samples = min(int(self.LEARNING_SECONDS * self.sampling_rate), len(integrated_signal))
        if init_samples <= 0:
            return

        init_segment = integrated_signal[:init_samples]
        self.signal_peak = float(np.max(init_segment)) * 0.25
        self.noise_peak = float(np.mean(init_segment)) * 0.5
        if self.signal_peak <= self.noise_peak:
            self.signal_peak = self.noise_peak + 1e-6
        self.threshold_1 = self.noise_peak + 0.25 * (self.signal_peak - self.noise_peak)
        self.threshold_2 = 0.5 * self.threshold_1

    def update_thresholds(self, peak_value, is_qrs, is_searchback=False):
        alpha = 0.25 if is_searchback else 0.125

        if is_qrs:
            self.signal_peak = alpha * peak_value + (1 - alpha) * self.signal_peak
        else:
            self.noise_peak = 0.125 * peak_value + 0.875 * self.noise_peak

        self.threshold_1 = self.noise_peak + 0.25 * (self.signal_peak - self.noise_peak)
        self.threshold_2 = 0.5 * self.threshold_1

    def _track_integrator_peak(self, integrated_signal, start_index):
        """Follow an integrator ridge while it stays above T2; return (peak_idx, peak_val, next_i)."""
        peak_idx = start_index
        peak_val = float(integrated_signal[start_index])
        i = start_index + 1
        n = len(integrated_signal)

        while i < n and integrated_signal[i] > self.threshold_2:
            val = float(integrated_signal[i])
            if val > peak_val:
                peak_val = val
                peak_idx = i
            i += 1

        return peak_idx, peak_val, i

    def _refractory_samples(self):
        return int(self.REFRACTORY_MS * self.sampling_rate / 1000)

    def _try_searchback(self, integrated_signal, current_peak_index, qrs_indices):
        if self.last_qrs_index < 0 or not self.rr_intervals:
            return None

        rr_limit = self.SEARCHBACK_RR_FACTOR * float(np.mean(self.rr_intervals))
        time_since_last = (current_peak_index - self.last_qrs_index) / self.sampling_rate
        if time_since_last <= rr_limit:
            return None

        segment = integrated_signal[self.last_qrs_index + 1 : current_peak_index]
        if len(segment) == 0:
            return None

        local_idx = int(np.argmax(segment))
        peak_idx = self.last_qrs_index + 1 + local_idx
        peak_val = float(integrated_signal[peak_idx])
        if peak_val <= self.threshold_2:
            return None

        return peak_idx, peak_val

    def _accept_qrs(self, peak_idx, peak_val, qrs_indices):
        qrs_indices.append(peak_idx)
        if self.last_qrs_index >= 0:
            rr = (peak_idx - self.last_qrs_index) / self.sampling_rate
            self.rr_intervals.append(rr)
            if len(self.rr_intervals) > 8:
                self.rr_intervals.pop(0)
        self.last_qrs_index = peak_idx
        self.update_thresholds(peak_val, is_qrs=True, is_searchback=False)

    def detect_peaks(self, integrated_signal):
        """
        Detect QRS locations using Pan-Tompkins adaptive thresholds on the integrator.

        Peaks are found when the integrated signal crosses T1; each ridge is tracked
        until it falls below T2. R-peak alignment on the bandpass is done separately
        via ``align_peaks``.
        """
        integrated_signal = np.asarray(integrated_signal, dtype=float)
        self.reset()
        self._initialize_thresholds(integrated_signal)

        qrs_indices = []
        refractory = self._refractory_samples()
        i = 0
        n = len(integrated_signal)

        while i < n:
            if integrated_signal[i] <= self.threshold_1:
                i += 1
                continue

            peak_idx, peak_val, i = self._track_integrator_peak(integrated_signal, i)

            if qrs_indices and (peak_idx - qrs_indices[-1]) < refractory:
                self.update_thresholds(peak_val, is_qrs=False)
                continue

            searchback = self._try_searchback(integrated_signal, peak_idx, qrs_indices)
            if searchback is not None:
                sb_idx, sb_val = searchback
                if not qrs_indices or (sb_idx - qrs_indices[-1]) >= refractory:
                    self._accept_qrs(sb_idx, sb_val, qrs_indices)
                    self.update_thresholds(sb_val, is_qrs=True, is_searchback=True)

            if peak_val > self.threshold_1:
                self._accept_qrs(peak_idx, peak_val, qrs_indices)
            else:
                self.update_thresholds(peak_val, is_qrs=False)

        return np.array(qrs_indices, dtype=int)


def detect_qrs_for_subject(subject, channel=0, detector=None):
    if detector is None:
        detector = PanTompkinsQRS(sampling_rate=subject.fs)
    else:
        detector.reset()

    integrated_signal = subject.integrated_signal[:, channel]
    return detector.detect_peaks(integrated_signal)


def detect_qrs_for_subjects(subjects, channel=0):
    detected_indices = []
    for subject in subjects:
        detector = PanTompkinsQRS(sampling_rate=subject.fs)
        detected_indices.append(detect_qrs_for_subject(subject, channel=channel, detector=detector))
    return detected_indices


def align_peaks(
    peak_indices,
    signal,
    search_window_ms=None,
    fs=DEFAULT_FS,
    lookback_ms=ALIGN_LOOKBACK_MS,
    lookforward_ms=ALIGN_LOOKFORWARD_MS,
):
    """Snap integrator peaks to the R wave on the bandpass (or raw) signal.

    Pan-Tompkins finds QRS on the integrator, which peaks after the R wave.
    Search mostly *backward* (MWI delay) and a little forward. ``search_window_ms``
    is kept as a lookback override for older callers.
    """
    if search_window_ms is not None:
        lookback_ms = search_window_ms

    lookback = int(lookback_ms * fs / 1000)
    lookforward = int(lookforward_ms * fs / 1000)
    n = len(signal)
    corrected_indices = []

    for peak in peak_indices:
        start_index = max(0, int(peak) - lookback)
        end_index = min(n, int(peak) + lookforward)
        segment = signal[start_index:end_index]
        if len(segment) == 0:
            corrected_indices.append(int(peak))
            continue
        local_max_index = int(np.argmax(np.abs(segment)))
        corrected_indices.append(start_index + local_max_index)

    return np.array(corrected_indices, dtype=int)


def synchronize_peaks(
    annotated_peaks,
    labels,
    detected_r_peaks,
    detected_int_peaks,
    fs=DEFAULT_FS,
    tolerance_ms=PEAK_TOLERANCE_MS,
):
    tolerance_samples = int((tolerance_ms / 1000.0) * fs)

    synced_r_peaks = []
    synced_int_peaks = []
    synced_labels = []

    annotated_peaks = np.array(annotated_peaks)

    for i, det_peak in enumerate(detected_r_peaks):
        distances = np.abs(annotated_peaks - det_peak)
        closest_idx = np.argmin(distances)
        min_distance = distances[closest_idx]

        if min_distance <= tolerance_samples:
            synced_r_peaks.append(det_peak)
            synced_int_peaks.append(detected_int_peaks[i])
            synced_labels.append(labels[closest_idx])

    return synced_r_peaks, synced_int_peaks, synced_labels
