"""Hearty/JELY filter primitives (LmeFilter, MWI, peak detectors)."""

from __future__ import annotations

import numpy as np


class LmeFilter:
    """Direct-form filter matching Hearty ``LmeFilter.next`` (a[1:] terms added)."""

    def __init__(self, b_taps, a_taps=None):
        self.b = np.asarray(b_taps, dtype=float)
        if a_taps is None:
            self.a = np.array([1.0], dtype=float)
        else:
            self.a = np.asarray(a_taps, dtype=float)
        self.x = np.zeros(len(self.b), dtype=float)
        self.y = np.zeros(len(self.a), dtype=float)

    def reset(self):
        self.x[:] = 0.0
        self.y[:] = 0.0

    def next(self, xnow: float) -> float:
        if len(self.b) > 1:
            self.x[1:] = self.x[:-1]
        self.x[0] = float(xnow)

        if len(self.a) > 1:
            self.y[1:] = self.y[:-1]

        y0 = float(np.dot(self.b, self.x))
        if len(self.a) > 1:
            y0 += float(np.dot(self.a[1:], self.y[1:]))
        if self.a[0] != 1.0:
            y0 /= self.a[0]
        self.y[0] = y0
        return y0


class MeanFilter:
    """Hearty nested MeanFilter (growing mean, then fixed-weight recursive)."""

    def __init__(self, max_num=0):
        self.max_num = int(max_num)
        self.num = 0
        self.y0 = 0.0
        self.y1 = 0.0

    def reset(self):
        self.num = 0
        self.y0 = 0.0
        self.y1 = 0.0

    def next(self, xnow: float) -> float:
        self.y1 = self.y0
        self.y0 = (self.y1 * self.num + float(xnow)) / (self.num + 1)
        if self.max_num == 0 or self.num < self.max_num:
            self.num += 1
        return self.y0


class WndIntFilter:
    """Moving-window integrator / mean (Pan–Tompkins MWI)."""

    def __init__(self, wnd_length: int):
        self.wnd_length = max(int(wnd_length), 1)
        self.buf = np.zeros(self.wnd_length, dtype=float)
        self.head = -1
        self.num = 0
        self.sum = 0.0

    def reset(self):
        self.buf[:] = 0.0
        self.head = -1
        self.num = 0
        self.sum = 0.0

    def next(self, xnow: float) -> float:
        xnow = float(xnow)
        self.head = (self.head + 1) % self.wnd_length
        if self.num < self.wnd_length:
            self.sum += xnow
            self.num += 1
        else:
            self.sum += xnow - self.buf[self.head]
        self.buf[self.head] = xnow
        return self.sum / self.num if self.num else 0.0


class PeakDetectionFilter:
    """Local-max detector (Hearty PeakDetectionFilter; Java buffer quirk retained)."""

    def __init__(self, min_range=1, min_diff=0.0):
        self.min_range = int(min_range)
        self.min_diff = float(min_diff)
        # Java: new double[minRange << 1 + 1] == 4*minRange due to << precedence
        self.size = max(self.min_range * 4, self.min_range * 2 + 1)
        self.x = np.zeros(self.size, dtype=float)
        self.peak_idx = -1
        self.peak_value = float("nan")
        self.block = self.size

    def reset(self):
        self.x[:] = 0.0
        self.peak_idx = -1
        self.peak_value = float("nan")
        self.block = self.size

    def next(self, xnow: float) -> float:
        self.x[1:] = self.x[:-1]
        self.x[0] = float(xnow)
        self.peak_value = float("nan")
        self.peak_idx = -1
        if self.block > 0:
            self.block -= 1
            return float("nan")

        mid = self.min_range
        for i in range(1, self.min_range + 1):
            if mid + i >= len(self.x) or mid - i < 0:
                return float("nan")
            if self.x[mid] - self.min_diff <= self.x[mid + i]:
                return float("nan")
            if self.x[mid] - self.min_diff < self.x[mid - i]:
                return float("nan")
        self.peak_value = float(self.x[mid])
        self.peak_idx = mid
        return self.peak_value


class MinDetectionFilter:
    """Local-min detector (Hearty MinDetectionFilter)."""

    def __init__(self, min_range=1, min_diff=0.0):
        self.min_range = int(min_range)
        self.min_diff = float(min_diff)
        self.size = max(self.min_range * 4, self.min_range * 2 + 1)
        self.x = np.zeros(self.size, dtype=float)
        self.peak_idx = -1
        self.peak_value = float("nan")
        self.block = self.size

    def reset(self):
        self.x[:] = 0.0
        self.peak_idx = -1
        self.peak_value = float("nan")
        self.block = self.size

    def next(self, xnow: float) -> float:
        self.x[1:] = self.x[:-1]
        self.x[0] = float(xnow)
        self.peak_value = float("nan")
        self.peak_idx = -1
        if self.block > 0:
            self.block -= 1
            return float("nan")

        mid = self.min_range
        for i in range(1, self.min_range + 1):
            if mid + i >= len(self.x) or mid - i < 0:
                return float("nan")
            if self.x[mid] - self.min_diff >= self.x[mid + i]:
                return float("nan")
            if self.x[mid] - self.min_diff > self.x[mid - i]:
                return float("nan")
        self.peak_value = float(self.x[mid])
        self.peak_idx = mid
        return self.peak_value


class StepHistory:
    """Ring buffer of recent samples (bandOut / intOut)."""

    def __init__(self, size_max: int):
        self.size_max = max(int(size_max), 1)
        self.values = np.zeros(self.size_max, dtype=float)
        self.head = -1
        self.num = 0
        self.min_value = 0.0
        self.max_value = 0.0

    def reset(self):
        self.values[:] = 0.0
        self.head = -1
        self.num = 0
        self.min_value = 0.0
        self.max_value = 0.0

    @property
    def range(self) -> float:
        return float(self.max_value - self.min_value)

    def add(self, value: float):
        value = float(value)
        self.head = (self.head + 1) % self.size_max
        self.values[self.head] = value
        if self.num < self.size_max:
            self.num += 1
        # track extremes over current contents
        live = self.values[: self.num] if self.num < self.size_max else self.values
        self.min_value = float(np.min(live))
        self.max_value = float(np.max(live))

    def get_past_value(self, idx_past: int) -> float:
        """0 = newest sample."""
        if self.num == 0:
            return 0.0
        idx_past = int(idx_past)
        if idx_past < 0:
            idx_past = 0
        if idx_past >= self.num:
            idx_past = self.num - 1
        idx = self.head - idx_past
        while idx < 0:
            idx += self.size_max
        return float(self.values[idx % self.size_max])


# Classic Pan–Tompkins taps used by Hearty (fixed; windows scale with fs).
LP_B = [0.03125, 0, 0, 0, 0, 0, -0.0625, 0, 0, 0, 0, 0, 0.03125]
LP_A = [1.0, 2.0, -1.0]
HP_B = (
    [-0.03125]
    + [0] * 15
    + [1.0, -1.0]
    + [0] * 14
    + [0.03125]
)
HP_A = [1.0, 1.0]
DIFF_B = [2.0, 1.0, 0.0, -1.0, -2.0]
DIFF_A = [8.0]
