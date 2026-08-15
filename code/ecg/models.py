from .config import DEFAULT_FS, FILTER_ORDER
from .preprocessing import get_pt_signals


class Subject:
    def __init__(
        self,
        number,
        age,
        gender,
        meds,
        signal,
        annotations,
        filter_order=FILTER_ORDER,
        fs=DEFAULT_FS,
    ):
        self.number = number
        self.age = age
        self.gender = gender
        self.meds = meds
        self.annotations = annotations
        self.raw_signal = signal
        self.fs = fs

        (
            self.filtered_signal,
            self.derivated_signal,
            self.squared_signal,
            self.integrated_signal,
        ) = get_pt_signals(signal, filter_order, fs)

    def _windows_for_channel(self, channel):
        if channel == 1:
            return self.windows_ch1
        if channel == 2:
            return self.windows_ch2
        raise ValueError("channel must be 1 or 2")

    def get_summary(self, channel):
        windows = self._windows_for_channel(channel)
        total = len(windows)
        positives = sum(w["label"] for w in windows)
        print(f"Paciente {self.number}: {total} janelas geradas.")
        print(
            f"Arritmias detectadas (Classe 1): {positives} | Normais (Classe 0): {total - positives}"
        )

    def get_window_label(self, channel, index):
        window = self._windows_for_channel(channel)[index]
        return "Normal" if window["label"] == 0 else "Arrythimic"

    def get_window_data(self, channel, index):
        return self._windows_for_channel(channel)[index]["signal"]
