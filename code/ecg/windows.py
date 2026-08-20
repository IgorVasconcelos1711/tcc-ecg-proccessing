from .config import DEFAULT_FS, WINDOW_SPAN_MS


class WindowManager:
    def __init__(
        self,
        synced_filtered_peaks,
        synced_labels,
        synced_integrated_peaks,
        integrated_signal,
        filtered_signal,
        window_span_ms=WINDOW_SPAN_MS,
        fs=DEFAULT_FS,
    ):
        self.fs = fs
        # TODO: currently called with one channel (1D). Restore both channels for ML.
        self.filtered_r_peaks_windows, self.filtered_peak_indices = self.get_r_peaks_windows(
            synced_labels, synced_filtered_peaks, filtered_signal, window_span_ms
        )
        self.integrated_r_peaks_windows, self.integrated_peak_indices = self.get_r_peaks_windows(
            synced_labels, synced_integrated_peaks, integrated_signal, window_span_ms
        )

    def get_r_peaks_windows(self, synced_labels, peaks, signal, window_span_ms):
        total_window_samples = int(window_span_ms * self.fs / 1000)
        half_window = total_window_samples // 2
        windows_list = []
        kept_peaks = []

        labels = synced_labels if synced_labels is not None else [None] * len(peaks)

        for peak, label in zip(peaks, labels):
            start_idx = int(peak) - half_window
            end_idx = int(peak) + half_window
            if start_idx >= 0 and end_idx <= len(signal):
                window = signal[start_idx:end_idx]
                windows_list.append({"signal": window, "window": window, "label": label})
                kept_peaks.append(int(peak))

        return windows_list, kept_peaks
