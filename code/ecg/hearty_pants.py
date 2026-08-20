"""Hearty-style streaming Pan–Tompkins detection + QRS segmentation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import DEFAULT_FS, HEARTY_TOTAL_DELAY
from .hearty_filters import (
    DIFF_A,
    DIFF_B,
    HP_A,
    HP_B,
    LP_A,
    LP_B,
    LmeFilter,
    MeanFilter,
    MinDetectionFilter,
    PeakDetectionFilter,
    StepHistory,
    WndIntFilter,
)


@dataclass
class HeartyBeat:
    """One segmented QRS from the Hearty pipeline."""

    r_sample: int
    finish_sample: int
    feat_width_ms: float
    feat_rr_ms: float
    values: np.ndarray
    q_idx: int
    r_idx: int
    s_idx: int
    feat_qrsta: float
    mean: float
    invalid: bool = False


@dataclass
class HeartyRunResult:
    beats: list = field(default_factory=list)
    band: np.ndarray | None = None
    integrated: np.ndarray | None = None


def _get_indirect(values: np.ndarray, r_idx: int) -> float:
    """Port of FloatValueList/DoubleCircularList.getIndirect for a dense prefix buffer."""
    num = len(values)
    if num == 0:
        return 0.0
    if r_idx < -num:
        return float(values[(num * 2 + r_idx) % num])
    if r_idx < 0:
        return float(values[num + r_idx])
    if r_idx >= num * 2:
        return float(values[r_idx % num])
    if r_idx >= num:
        return float(values[r_idx - num])
    return float(values[r_idx])


def hearty_maxcorr(beat_values: np.ndarray, beat_mean: float, other_values: np.ndarray, other_mean: float) -> float:
    """Hearty ``QRS.maxCorr`` (±8 lag, getIndirect on template)."""
    n = len(beat_values)
    if n == 0 or len(other_values) == 0:
        return 0.0
    max_cc = 0.0
    for lag in range(-8, 8):
        cc = 0.0
        sumx = 0.0
        sumy = 0.0
        for i in range(n):
            x = float(beat_values[i]) - beat_mean
            y = _get_indirect(other_values, lag + i) - other_mean
            cc += x * y
            sumx += x * x
            sumy += y * y
        if cc != 0.0 and sumx > 0.0 and sumy > 0.0:
            cc = cc / np.sqrt(sumx * sumy)
            if cc > max_cc:
                max_cc = float(cc)
    return max_cc


def hearty_ardiff(qrsta: float, other_qrsta: float) -> float:
    if other_qrsta == 0.0:
        return 0.0
    if qrsta > other_qrsta:
        return (qrsta - other_qrsta) / other_qrsta
    return (other_qrsta - qrsta) / other_qrsta


def _qrsta(values: np.ndarray) -> float:
    return float(np.sum(np.abs(values)))


class HeartyPanTompkins:
    """Streaming Pan–Tompkins + segmentation as in Gradl Hearty ``PanTompkins``."""

    WND_INT_COMPENSATION = 0.85

    def __init__(self, sampling_rate=DEFAULT_FS):
        self.fs = int(sampling_rate)
        self.sampling_time = 1000.0 / self.fs
        self.wnd_length = int(150 * self.fs / 1000)
        self.pre_segment = int(120 * self.fs / 1000)
        self.post_segment = int(280 * self.fs / 1000)
        self.max_qrs_size = self.pre_segment + self.post_segment
        if self.max_qrs_size < self.wnd_length + HEARTY_TOTAL_DELAY + 2:
            self.max_qrs_size = self.wnd_length + HEARTY_TOTAL_DELAY + 2

        self.lowpass = LmeFilter(LP_B, LP_A)
        self.highpass = LmeFilter(HP_B, HP_A)
        self.diff = LmeFilter(DIFF_B, DIFF_A)
        self.wnd_int = WndIntFilter(self.wnd_length)
        self.wnd_mean = MeanFilter(self.max_qrs_size)

        self.band_out = StepHistory(self.max_qrs_size)
        self.int_out = StepHistory(self.max_qrs_size)

        self.r_peak = PeakDetectionFilter(1, 0)
        self.rising_peak = PeakDetectionFilter(3, 0)
        self.q_peak = MinDetectionFilter(1, 0)
        self.s_peak = MinDetectionFilter(1, 0)

        self.start_processing = self.fs * 2
        self.reset_state()

    def reset_state(self):
        self.lowpass.reset()
        self.highpass.reset()
        self.diff.reset()
        self.wnd_int.reset()
        self.wnd_mean.reset()
        self.band_out.reset()
        self.int_out.reset()
        self.r_peak.reset()
        self.rising_peak.reset()
        self.q_peak.reset()
        self.s_peak.reset()

        self.seg_state = "INVALID"  # INVALID, THRESHOLD_CROSSED, R_FOUND, FINISHED
        self.values: list[float] = []
        self.q_idx = -1
        self.q_amp = 0.0
        self.r_idx = -1
        self.r_amp = 0.0
        self.r_sample = -1
        self.s_idx = -1
        self.s_amp = 0.0
        self.feat_width = 0.0
        self.r_pass_num = 0
        self.last_crossing = 0
        self.last_band_peak = 0
        self.last_beat_samples = 0
        self.prev_r_sample = None
        self.start_processing = self.fs * 2
        self.qrs_threshold = 1.0

    def _reset_current_qrs(self):
        self.seg_state = "INVALID"
        self.values = []
        self.q_idx = -1
        self.q_amp = 0.0
        self.r_idx = -1
        self.r_amp = 0.0
        self.r_sample = -1
        self.s_idx = -1
        self.s_amp = 0.0
        self.feat_width = 0.0
        self.r_pass_num = 0
        self.last_band_peak = 0

    def process(self, signal: np.ndarray) -> HeartyRunResult:
        signal = np.asarray(signal, dtype=float).ravel()
        n = len(signal)

        # Batch Hearty LP→HP→diff→square→MWI (same taps as LmeFilter).
        from scipy.signal import lfilter

        a_lp = np.array(LP_A, dtype=float)
        a_lp[1:] *= -1
        a_hp = np.array(HP_A, dtype=float)
        a_hp[1:] *= -1
        a_diff = np.array(DIFF_A, dtype=float)
        a_diff[1:] *= -1

        y2 = lfilter(LP_B, a_lp, signal)
        band = lfilter(HP_B, a_hp, y2)
        y4 = lfilter(DIFF_B, a_diff, band)
        y5 = y4 * y4

        # MWI = causal moving mean of length wnd_length (Hearty WndIntFilter).
        kernel = np.ones(self.wnd_length, dtype=float)
        cs = np.cumsum(y5, dtype=float)
        integrated = np.empty(n, dtype=float)
        for i in range(n):
            left = i + 1 - self.wnd_length
            if left <= 0:
                integrated[i] = cs[i] / (i + 1)
            else:
                integrated[i] = (cs[i] - cs[left - 1]) / self.wnd_length

        # Threshold: Hearty MeanFilter(maxQrsSize) on integrator.
        thr_series = np.empty(n, dtype=float)
        mean_y = 0.0
        mean_n = 0
        max_n = self.max_qrs_size
        for i, v in enumerate(integrated):
            mean_y = (mean_y * mean_n + v) / (mean_n + 1)
            if mean_n < max_n:
                mean_n += 1
            thr_series[i] = mean_y

        beats: list[HeartyBeat] = []
        self.reset_state()
        # refill band history as we stream the state machine
        self.band_out.reset()
        refractory = int(0.2 * self.fs)
        refractory_until = 0
        start = self.fs * 2

        for sample_idx in range(n):
            y3 = float(band[sample_idx])
            y6 = float(integrated[sample_idx])
            self.band_out.add(y3)
            self.qrs_threshold = float(thr_series[sample_idx])
            self.last_beat_samples += 1

            if sample_idx < start:
                continue

            in_refractory = sample_idx < refractory_until
            above = (y3 > self.qrs_threshold) or (y6 > self.qrs_threshold) or (self.seg_state == "R_FOUND")

            if above and not in_refractory:
                self.last_crossing += 1
                # Hearty: initial trigger; require integrator for a new search to cut noise.
                if self.seg_state == "INVALID" and y6 > self.qrs_threshold:
                    self.r_peak.reset()
                    self.r_peak.next(self.band_out.get_past_value(2))
                    self.r_peak.next(self.band_out.get_past_value(1))
                    self.r_peak.next(y3)
                    self.last_crossing = 0
                    self.seg_state = "THRESHOLD_CROSSED"
            else:
                if self.last_crossing > 0:
                    self.last_crossing -= 1
                if self.seg_state == "FINISHED":
                    self._reset_current_qrs()

            finished = None
            if self.seg_state == "THRESHOLD_CROSSED" and not in_refractory:
                finished = self._handle_threshold_crossed(sample_idx, y3, y6)
            elif self.seg_state == "R_FOUND":
                finished = self._handle_r_found(sample_idx, y3, y6)

            if finished is not None:
                beats.append(finished)
                self.seg_state = "FINISHED"
                refractory_until = sample_idx + refractory
                self.last_beat_samples = 0

        return HeartyRunResult(beats=beats, band=band, integrated=integrated)

    def _handle_threshold_crossed(self, sample_idx, y3, y6):
        self.r_peak.next(y3)
        self.rising_peak.reset()
        self.rising_peak.next(y6)
        self.q_peak.reset()
        self.s_peak.reset()

        if self.r_peak.peak_idx == -1:
            return None

        if y6 < self.qrs_threshold and self.last_crossing > 0:
            self.r_peak.reset()
            self.seg_state = "THRESHOLD_CROSSED"
            self.last_crossing = int(-1000 * self.sampling_time)
            return None

        self.values = []
        self.q_idx = -1
        for i in range(self.pre_segment + 1):
            v = self.band_out.get_past_value(self.pre_segment - i)
            self.values.append(v)
            if self.q_idx == -1:
                self.q_peak.next(self.band_out.get_past_value(i))
                if self.q_peak.peak_idx != -1:
                    self.q_amp = self.q_peak.peak_value
                    self.q_idx = self.pre_segment - i

        if self.q_idx == -1:
            self.q_amp = self.values[0]
            self.q_idx = 0

        self.r_idx = (len(self.values) - 1) - self.r_peak.peak_idx
        self.r_amp = self.r_peak.peak_value
        self.r_sample = sample_idx - self.r_peak.peak_idx
        self.r_pass_num = 1

        if self.r_amp - self.q_amp < self.band_out.range * 0.1:
            self._reset_current_qrs()
            return None

        self.last_band_peak = 0
        self.seg_state = "R_FOUND"
        self.s_idx = -1
        self.s_peak.next(y3)
        self.feat_width = 0.0
        return None

    def _handle_r_found(self, sample_idx, y3, y6):
        self.values.append(float(y3))

        if self.r_pass_num > 0:
            self.r_pass_num += 1
            self.rising_peak.next(y6)
            if self.rising_peak.peak_idx != -1:
                self.feat_width = self.r_pass_num * self.WND_INT_COMPENSATION * self.sampling_time
                self.r_pass_num = 0

        self.last_band_peak += 1

        if self.s_idx == -1:
            self.s_peak.next(y3)
            if self.s_peak.peak_idx != -1:
                self.s_amp = self.s_peak.peak_value
                self.s_idx = (len(self.values) - 1) - self.s_peak.peak_idx

        if self.last_band_peak < self.post_segment:
            return None

        if self.s_idx == -1:
            self.s_amp = y3
            self.s_idx = len(self.values) - 1

        if self.feat_width < 1:
            self.feat_width = (self.s_idx - self.q_idx) * self.WND_INT_COMPENSATION * self.sampling_time

        values = np.asarray(self.values, dtype=float)
        mean = float(np.mean(values)) if len(values) else 0.0
        qrsta = _qrsta(values)

        rr_ms = 0.0
        if self.prev_r_sample is not None and self.r_sample >= 0:
            rr_ms = (self.r_sample - self.prev_r_sample) / self.fs * 1000.0
        self.prev_r_sample = self.r_sample

        return HeartyBeat(
            r_sample=max(int(self.r_sample), 0),
            finish_sample=int(sample_idx),
            feat_width_ms=float(self.feat_width),
            feat_rr_ms=float(rr_ms),
            values=values,
            q_idx=int(self.q_idx),
            r_idx=int(self.r_idx),
            s_idx=int(self.s_idx),
            feat_qrsta=qrsta,
            mean=mean,
            invalid=False,
        )


def detect_hearty_beats(signal: np.ndarray, fs=DEFAULT_FS) -> HeartyRunResult:
    return HeartyPanTompkins(sampling_rate=fs).process(signal)
