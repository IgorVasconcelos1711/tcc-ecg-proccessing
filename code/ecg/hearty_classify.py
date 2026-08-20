"""Classify Hearty-segmented beats with Gradl decision tree + template adapt."""

from __future__ import annotations

import numpy as np

from .classifier import BeatClassification, RhythmClass, WaveformClass, classify_beat
from .config import TEMPLATE_INIT_BEATS
from .hearty_pants import HeartyBeat, hearty_ardiff, hearty_maxcorr


def _init_templates_hearty(beats: list[HeartyBeat], n=TEMPLATE_INIT_BEATS):
    """Hearty template selection from first ``n`` valid beats (qrsta sort, corr>0.9)."""
    pool = [b for b in beats[:n] if not b.invalid and len(b.values) > 0]
    if len(pool) < 2:
        raise ValueError("insufficient beats for Hearty templates")

    areas = np.array([b.feat_qrsta for b in pool], dtype=float)
    avg = float(np.mean(areas))

    # Sort indices: prefer below-average qrsta ascending (Hearty sortList logic, simplified)
    below = [i for i, a in enumerate(areas) if a < avg]
    above = [i for i, a in enumerate(areas) if a >= avg]
    below_sorted = sorted(below, key=lambda i: areas[i])
    above_sorted = sorted(above, key=lambda i: areas[i])
    ranked = below_sorted + above_sorted
    if len(ranked) < 2:
        ranked = list(range(len(pool)))

    t1 = t2 = None
    for i in range(min(3, len(ranked) - 1)):
        a = pool[ranked[i]]
        b = pool[ranked[i + 1]]
        if hearty_maxcorr(a.values, a.mean, b.values, b.mean) > 0.9:
            t1, t2 = a, b
            break
    if t1 is None:
        t1 = pool[ranked[0]]
        t2 = pool[ranked[1]]

    return [
        {"values": np.array(t1.values, copy=True), "mean": t1.mean, "qrsta": t1.feat_qrsta},
        {"values": np.array(t2.values, copy=True), "mean": t2.mean, "qrsta": t2.feat_qrsta},
    ]


def classify_hearty_beats(beats: list[HeartyBeat], start_idx=TEMPLATE_INIT_BEATS):
    """Return (templates, classifications for beats[start_idx:])."""
    if len(beats) < start_idx + 1:
        raise ValueError("insufficient beats")

    templates = _init_templates_hearty(beats, n=start_idx)
    classifications = []
    prev_rr = None
    prev_width = None

    for beat in beats[start_idx:]:
        mc1 = hearty_maxcorr(beat.values, beat.mean, templates[0]["values"], templates[0]["mean"])
        mc2 = hearty_maxcorr(beat.values, beat.mean, templates[1]["values"], templates[1]["mean"])
        ar1 = hearty_ardiff(beat.feat_qrsta, templates[0]["qrsta"])
        ar2 = hearty_ardiff(beat.feat_qrsta, templates[1]["qrsta"])

        result = classify_beat(
            maxcorr_t1=mc1,
            maxcorr_t2=mc2,
            ardiff_t1=ar1,
            ardiff_t2=ar2,
            qrs_width_ms=beat.feat_width_ms,
            rr_ms=beat.feat_rr_ms,
            prev_rr_ms=prev_rr,
            prev_qrs_width_ms=prev_width,
        )
        classifications.append(result)

        if result.is_normal:
            # Hearty: replace more correlated template
            if mc1 > mc2:
                templates[0] = {
                    "values": np.array(beat.values, copy=True),
                    "mean": beat.mean,
                    "qrsta": beat.feat_qrsta,
                }
            else:
                templates[1] = {
                    "values": np.array(beat.values, copy=True),
                    "mean": beat.mean,
                    "qrsta": beat.feat_qrsta,
                }

        prev_rr = beat.feat_rr_ms
        prev_width = beat.feat_width_ms

    return templates, classifications
