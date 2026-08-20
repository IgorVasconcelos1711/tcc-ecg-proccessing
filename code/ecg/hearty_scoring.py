"""Hearty `HeartyActivity.SimResult` scoring (Gradl et al. 2012 Table I)."""

from .classifier import BeatClassification, RhythmClass, WaveformClass
from .config import HEARTY_TOTAL_DELAY


# MIT-BIH annotation symbols that are not beat labels (Hearty decrements ref count).
_NON_BEAT_ANNOTATIONS = set("\"~@[]!()ptu|+DT'`*=")


class HeartySimResult:
    """Faithful port of HeartyActivity.SimResult newLabel / newBeat / finish."""

    def __init__(self, total_delay=HEARTY_TOTAL_DELAY):
        self.total_delay = total_delay
        self.num_total_beats_ref = 0
        self.num_total_beats = 0
        self.num_tp = 0
        self.num_tn = 0
        self.num_fn = 0
        self.num_fp = 0
        self.last_label_count = 0
        self.current_label = "$"
        self.next_label = "\0"
        self.last_label = "\0"
        self.open_label = False
        self._finished = False

    def new_label(self, label, learning=False):
        """Call once per sample. ``label`` is a one-char symbol or ``\\0`` if none."""
        if not label:
            label = "\0"
        elif isinstance(label, str) and len(label) > 1:
            label = label[0]

        self.last_label_count += 1

        if self.last_label_count == 0:
            self.last_label = self.current_label
            self.current_label = self.next_label

        if label == "\0":
            return

        self.num_total_beats_ref += 1
        self.last_label_count = -self.total_delay + 2

        if learning:
            self.next_label = label
            return

        if label == "x":
            self.num_total_beats_ref -= 1
        elif label in _NON_BEAT_ANNOTATIONS:
            self.num_total_beats_ref -= 1
            label = self.next_label if self.next_label not in ("\0", "") else label
        elif label in ("N", "/"):
            self.open_label = True

        self.next_label = label

    def new_beat(self, classification: BeatClassification, learning=False):
        """Call when a QRS has been detected and classified.

        In Hearty this runs when segmentation finishes (~``TOTAL_DELAY`` samples
        after the R peak), so callers should schedule at ``peak + TOTAL_DELAY``.
        """
        self.num_total_beats += 1

        if learning:
            return

        self.open_label = False
        current = self.current_label

        # Hearty: AV_BLOCK against 'x' counts as TP before NORMAL/abnormal branch.
        if classification.rhythm == RhythmClass.AV_BLOCK and current == "x":
            self.num_tp += 1

        if classification.waveform == WaveformClass.NORMAL:
            if current != "$":
                if current in ("N", "/"):
                    self.num_tn += 1
                else:
                    if classification.rhythm != RhythmClass.NONE:
                        self.num_tp += 1
                    elif current not in ("x", "S", "A", "a", "f"):
                        self.num_fn += 1
        else:
            if current != "$":
                # Check previous beat if filter delay shifted labels for fused beats.
                if (current not in ("N", "/")) or self.last_label != "N":
                    self.num_tp += 1
                else:
                    self.num_fp += 1

    def finish(self):
        if self._finished:
            return
        self._finished = True
        if self.num_total_beats_ref > self.num_total_beats:
            self.num_fn += self.num_total_beats_ref - self.num_total_beats

    @property
    def detection_rate(self):
        if self.num_total_beats_ref <= 0:
            return float("nan")
        return min(self.num_total_beats, self.num_total_beats_ref) / self.num_total_beats_ref
