"""Gradl / Krasteva rule-based beat classifier (Hearty PanTompkins.java)."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .features import (
    _as_1d,
    _window_signal,
    extract_qrs_width,
    extract_rr_interval,
    update_templates,
)


class WaveformClass(str, Enum):
    UNKNOWN = "unknown"
    INVALID = "invalid"
    NORMAL = "normal"
    PVC = "pvc"
    PVC_ABERRANT = "pvc_aberrant"
    BB_BLOCK = "bb_block"
    ESCAPE = "escape"
    APC = "apc"
    APC_ABERRANT = "apc_aberrant"
    PREMATURE = "premature"
    ABERRANT = "aberrant"
    VIRTUAL = "virtual"


class RhythmClass(str, Enum):
    NONE = "none"
    ARTIFACT = "artifact"
    FUSION = "fusion"
    AV_BLOCK = "av_block"
    TACHYCARDIA = "tachycardia"
    BRADYCARDIA = "bradycardia"
    FIBRILLATION = "fibrillation"
    CARDIAC_ARREST = "cardiac_arrest"


ABNORMAL_WAVEFORM = {
    WaveformClass.PVC,
    WaveformClass.PVC_ABERRANT,
    WaveformClass.BB_BLOCK,
    WaveformClass.ESCAPE,
    WaveformClass.APC,
    WaveformClass.APC_ABERRANT,
    WaveformClass.PREMATURE,
    WaveformClass.ABERRANT,
}

ABNORMAL_RHYTHM = {
    RhythmClass.ARTIFACT,
    RhythmClass.FUSION,
    RhythmClass.AV_BLOCK,
    RhythmClass.TACHYCARDIA,
    RhythmClass.BRADYCARDIA,
    RhythmClass.FIBRILLATION,
    RhythmClass.CARDIAC_ARREST,
}


@dataclass
class BeatClassification:
    waveform: WaveformClass
    rhythm: RhythmClass
    maxcorr_t1: float
    maxcorr_t2: float
    ardiff_t1: float
    ardiff_t2: float
    qrs_width_ms: float
    rr_ms: float

    @property
    def is_normal(self) -> bool:
        return self.waveform == WaveformClass.NORMAL


def _waveform_area(window, channel_idx=0):
    signal = _window_signal(window, channel_idx)
    return float(np.sum(np.abs(signal)))


def ardiff_vs_template(beat_area, template_area):
    """Hearty/Krasteva area ratio (not sum-normalized)."""
    if template_area == 0:
        return 0.0
    if beat_area > template_area:
        return (beat_area - template_area) / template_area
    return (template_area - beat_area) / template_area


def maxcorr_vs_template(window, template, channel_idx=0, max_lag=8):
    beat = _window_signal(window, channel_idx)
    tmpl = _as_1d(template, channel_idx)
    beat_mean = beat - np.mean(beat)
    tmpl_mean = tmpl - np.mean(tmpl)
    beat_norm = np.linalg.norm(beat_mean)
    tmpl_norm = np.linalg.norm(tmpl_mean)
    if beat_norm == 0 or tmpl_norm == 0:
        return 0.0

    max_corr = -1.0
    for lag in range(-max_lag, max_lag):
        if lag >= 0:
            x = beat_mean[lag:]
            y = tmpl_mean[: len(x)]
        else:
            x = beat_mean[: lag]
            y = tmpl_mean[-lag : -lag + len(x)]
        if len(x) == 0:
            continue
        num = float(np.dot(x, y))
        den = float(np.linalg.norm(x) * np.linalg.norm(y))
        if den > 0:
            max_corr = max(max_corr, num / den)
    return max_corr if max_corr > -1.0 else 0.0


def extract_template_features(window, templates, channel_idx=0):
    beat_area = _waveform_area(window, channel_idx)
    tmpl_areas = [_waveform_area(t, channel_idx) for t in templates]
    return {
        "maxcorr_t1": maxcorr_vs_template(window, templates[0], channel_idx),
        "maxcorr_t2": maxcorr_vs_template(window, templates[1], channel_idx),
        "ardiff_t1": ardiff_vs_template(beat_area, tmpl_areas[0]),
        "ardiff_t2": ardiff_vs_template(beat_area, tmpl_areas[1]),
    }


def classify_beat(
    maxcorr_t1,
    maxcorr_t2,
    ardiff_t1,
    ardiff_t2,
    qrs_width_ms,
    rr_ms,
    prev_rr_ms=None,
    prev_qrs_width_ms=None,
    templates_ready=True,
):
    """Transcribed from Gradl Hearty `PanTompkins.QRS.classify()`."""
    if not templates_ready:
        return BeatClassification(
            WaveformClass.UNKNOWN,
            RhythmClass.NONE,
            maxcorr_t1,
            maxcorr_t2,
            ardiff_t1,
            ardiff_t2,
            qrs_width_ms,
            rr_ms,
        )

    waveform = WaveformClass.NORMAL
    rhythm = RhythmClass.NONE

    if qrs_width_ms > 130:
        waveform = WaveformClass.BB_BLOCK
    elif qrs_width_ms < 45:
        waveform = WaveformClass.PVC

    if maxcorr_t1 < 0.2 or maxcorr_t2 < 0.2:
        rhythm = RhythmClass.ARTIFACT

    if maxcorr_t1 < 0.3 or maxcorr_t2 < 0.3:
        waveform = WaveformClass.ABERRANT
    elif maxcorr_t1 < 0.6 or maxcorr_t2 < 0.6:
        waveform = WaveformClass.PVC_ABERRANT
    elif maxcorr_t1 < 0.9 and maxcorr_t2 < 0.9:
        waveform = WaveformClass.PVC
    elif maxcorr_t1 < 0.98 and maxcorr_t2 < 0.98:
        if ardiff_t1 > 0.7 or ardiff_t2 > 0.7:
            waveform = WaveformClass.ABERRANT
        elif ardiff_t1 > 0.5 or ardiff_t2 > 0.5:
            waveform = WaveformClass.PVC_ABERRANT
        elif ardiff_t1 > 0.2 and ardiff_t2 > 0.2:
            waveform = WaveformClass.PVC

    if prev_rr_ms is not None and prev_rr_ms > 0:
        if (rr_ms >= prev_rr_ms * 1.5 and rr_ms > 800) or rr_ms > 1700:
            rhythm = RhythmClass.AV_BLOCK
            if waveform == WaveformClass.NORMAL:
                waveform = WaveformClass.APC
        elif 1 < rr_ms < 460:
            if rr_ms > prev_rr_ms * 0.92:
                if waveform == WaveformClass.NORMAL and (maxcorr_t1 < 0.96 or maxcorr_t2 < 0.96):
                    waveform = WaveformClass.APC
            else:
                rhythm = RhythmClass.FUSION

            if rr_ms < 400:
                if maxcorr_t1 < 0.6 or maxcorr_t2 < 0.6:
                    waveform = WaveformClass.APC_ABERRANT
                else:
                    waveform = WaveformClass.APC
        elif prev_rr_ms > 800 and rr_ms < prev_rr_ms * 0.6:
            waveform = WaveformClass.ESCAPE
        elif (
            waveform == WaveformClass.NORMAL
            and prev_qrs_width_ms is not None
            and qrs_width_ms > 10
            and qrs_width_ms < prev_qrs_width_ms * 0.6
            and (ardiff_t1 > 0.1 or ardiff_t2 > 0.1)
        ):
            waveform = WaveformClass.PREMATURE

    return BeatClassification(
        waveform=waveform,
        rhythm=rhythm,
        maxcorr_t1=maxcorr_t1,
        maxcorr_t2=maxcorr_t2,
        ardiff_t1=ardiff_t1,
        ardiff_t2=ardiff_t2,
        qrs_width_ms=qrs_width_ms,
        rr_ms=rr_ms,
    )


def classify_and_adapt(
    windows_list,
    templates,
    peak_indices,
    fs,
    integrated_windows=None,
    start_idx=6,
    channel_idx=0,
):
    """Classify beats after template init; update templates on NORMAL (Hearty)."""
    adapted = [np.array(t, copy=True) for t in templates]
    classifications = []
    prev_rr_ms = None
    prev_qrs_width_ms = None
    n_updates = 0

    for i in range(start_idx, len(windows_list)):
        window = windows_list[i]
        tmpl_feats = extract_template_features(window, adapted, channel_idx)
        qrs_width_ms = extract_qrs_width(window, fs, channel_idx=channel_idx)

        rr_ms = extract_rr_interval(peak_indices[i - 1], peak_indices[i], fs)
        result = classify_beat(
            qrs_width_ms=qrs_width_ms,
            rr_ms=rr_ms,
            prev_rr_ms=prev_rr_ms,
            prev_qrs_width_ms=prev_qrs_width_ms,
            **tmpl_feats,
        )
        classifications.append(result)

        if result.is_normal:
            update_templates(adapted, window, channel_idx=channel_idx)
            n_updates += 1

        prev_rr_ms = rr_ms
        prev_qrs_width_ms = qrs_width_ms

    return adapted, classifications, n_updates
