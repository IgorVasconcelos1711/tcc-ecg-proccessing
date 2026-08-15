import numpy as np

from .config import DEFAULT_FS, PEAK_TOLERANCE_MS, SEARCH_WINDOW_MS


class PanTompkinsQRS:
    def __init__(self, sampling_rate=DEFAULT_FS):
        self.sampling_rate = sampling_rate
        self.reset()

    def reset(self):
        self.signal_peak_integrated = 0.0
        self.noise_peak_integrated = 0.0
        self.threshold_integrated_1 = 0.0
        self.threshold_integrated_2 = 0.0

        self.signal_peak_filtered = 0.0
        self.noise_peak_filtered = 0.0
        self.threshold_filtered_1 = 0.0
        self.threshold_filtered_2 = 0.0

        self.rr_intervals = []
        self.last_qrs_index = 0

    def find_peaks(self, integrated_signal):
        peaks_indices = []
        tracking_peak = False
        max_val = -np.inf
        max_pos = 0

        noise_floor = np.mean(integrated_signal)

        for i in range(1, len(integrated_signal)):
            if not tracking_peak:
                if integrated_signal[i] > integrated_signal[i - 1] and integrated_signal[i] > noise_floor:
                    tracking_peak = True
                    max_val = integrated_signal[i]
                    max_pos = i
            else:
                if integrated_signal[i] > max_val:
                    if (integrated_signal[i] - integrated_signal[i - 1]) > (max_val * 0.01):
                        max_pos = i
                    max_val = integrated_signal[i]
                elif integrated_signal[i] < max_val * 0.5:
                    peaks_indices.append(max_pos)
                    tracking_peak = False

        return np.array(peaks_indices)

    def update_thresholds(self, peak_value, signal_peak, noise_peak, is_qrs, is_searchback=False):
        alpha = 0.25 if is_searchback else 0.125

        if is_qrs:
            signal_peak = alpha * peak_value + (1 - alpha) * signal_peak
        else:
            noise_peak = 0.125 * peak_value + 0.875 * noise_peak

        threshold_1 = noise_peak + 0.25 * (signal_peak - noise_peak)
        threshold_2 = 0.5 * threshold_1

        return signal_peak, noise_peak, threshold_1, threshold_2

    def apply_pan_tompkins_thresholds(self, peaks_indices, integrated_signal, filtered_signal):
        qrs_indices = []

        for i, current_peak_index in enumerate(peaks_indices):
            peak_val_integrated = integrated_signal[current_peak_index]
            peak_val_filtered = filtered_signal[current_peak_index]

            time_since_last_qrs = (
                (current_peak_index - self.last_qrs_index) / self.sampling_rate
                if self.last_qrs_index != 0
                else 0
            )

            if 0 < time_since_last_qrs < 0.200:
                continue

            rr_missed_limit = (
                1.66 * np.mean(self.rr_intervals) if len(self.rr_intervals) > 0 else float("inf")
            )
            is_qrs = False

            if time_since_last_qrs > rr_missed_limit:
                searchback_candidates = [p for p in peaks_indices[:i] if p > self.last_qrs_index]

                if searchback_candidates:
                    best_searchback_peak = max(
                        searchback_candidates, key=lambda p: integrated_signal[p]
                    )
                    searchback_val_integrated = integrated_signal[best_searchback_peak]
                    searchback_val_filtered = filtered_signal[best_searchback_peak]

                    if (
                        searchback_val_integrated > self.threshold_integrated_2
                        and searchback_val_filtered > self.threshold_filtered_2
                    ):
                        qrs_indices.append(best_searchback_peak)

                        (
                            self.signal_peak_integrated,
                            self.noise_peak_integrated,
                            self.threshold_integrated_1,
                            self.threshold_integrated_2,
                        ) = self.update_thresholds(
                            searchback_val_integrated,
                            self.signal_peak_integrated,
                            self.noise_peak_integrated,
                            is_qrs=True,
                            is_searchback=True,
                        )

                        (
                            self.signal_peak_filtered,
                            self.noise_peak_filtered,
                            self.threshold_filtered_1,
                            self.threshold_filtered_2,
                        ) = self.update_thresholds(
                            searchback_val_filtered,
                            self.signal_peak_filtered,
                            self.noise_peak_filtered,
                            is_qrs=True,
                            is_searchback=True,
                        )

                        self.last_qrs_index = best_searchback_peak

                        if len(qrs_indices) > 1:
                            self.rr_intervals.append(
                                (best_searchback_peak - qrs_indices[-2]) / self.sampling_rate
                            )
                        if len(self.rr_intervals) > 8:
                            self.rr_intervals.pop(0)

                        time_since_last_qrs = (
                            current_peak_index - self.last_qrs_index
                        ) / self.sampling_rate

            if (
                peak_val_integrated > self.threshold_integrated_1
                and peak_val_filtered > self.threshold_filtered_1
            ):
                is_qrs = True

            (
                self.signal_peak_integrated,
                self.noise_peak_integrated,
                self.threshold_integrated_1,
                self.threshold_integrated_2,
            ) = self.update_thresholds(
                peak_val_integrated,
                self.signal_peak_integrated,
                self.noise_peak_integrated,
                is_qrs,
                is_searchback=False,
            )

            (
                self.signal_peak_filtered,
                self.noise_peak_filtered,
                self.threshold_filtered_1,
                self.threshold_filtered_2,
            ) = self.update_thresholds(
                peak_val_filtered,
                self.signal_peak_filtered,
                self.noise_peak_filtered,
                is_qrs,
                is_searchback=False,
            )

            if is_qrs:
                qrs_indices.append(current_peak_index)
                if self.last_qrs_index != 0:
                    self.rr_intervals.append(time_since_last_qrs)
                    if len(self.rr_intervals) > 8:
                        self.rr_intervals.pop(0)
                self.last_qrs_index = current_peak_index

        return np.sort(np.unique(qrs_indices))


def detect_qrs_for_subject(subject, channel=0, detector=None):
    if detector is None:
        detector = PanTompkinsQRS(sampling_rate=subject.fs)
    else:
        detector.reset()

    integrated_signal = subject.integrated_signal[:, channel]
    filtered_signal = subject.filtered_signal[:, channel]
    candidates = detector.find_peaks(integrated_signal)
    return detector.apply_pan_tompkins_thresholds(
        peaks_indices=candidates,
        integrated_signal=integrated_signal,
        filtered_signal=filtered_signal,
    )


def detect_qrs_for_subjects(subjects, channel=0):
    detected_indices = []
    for subject in subjects:
        detector = PanTompkinsQRS(sampling_rate=subject.fs)
        detected_indices.append(detect_qrs_for_subject(subject, channel=channel, detector=detector))
    return detected_indices


def align_peaks(peak_indices, raw_signal, search_window_ms=SEARCH_WINDOW_MS, fs=DEFAULT_FS):
    corrected_indices = []
    look_around_indices = int(search_window_ms * fs / 1000)

    for peak in peak_indices:
        start_index = peak - look_around_indices
        end_index = min(len(raw_signal), peak + look_around_indices)
        segment = raw_signal[start_index:end_index]
        if len(segment) > 0:
            local_max_index = np.argmax(segment)
            corrected_indices.append(local_max_index + start_index)

    return np.sort(corrected_indices)


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
