"""Evaluation metrics and confusion matrices vs MIT-BIH annotations.

Binary TP/TN/FP/FN follow Hearty ``SimResult`` (Gradl Table I), not a simple
per-synced-beat comparison.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .classifier import BeatClassification, RhythmClass, WaveformClass, classify_and_adapt
from .config import HEARTY_TOTAL_DELAY, PEAK_TOLERANCE_MS, TEMPLATE_INIT_BEATS
from .detection import synchronize_peaks
from .features import initialize_templates
from .hearty_classify import classify_hearty_beats
from .hearty_pants import detect_hearty_beats
from .hearty_scoring import HeartySimResult
from .windows import WindowManager

GRADL_OMIT_RECORDS = {"104", "109", "111", "118", "124", "203", "214", "231", "232"}

EVAL_BEAT_CLASSES = [
    "normal",
    "pvc",
    "apc",
    "bb_block",
    "escape",
    "aberrant",
    "premature",
    "fusion",
    "av_block",
    "other",
    "unknown",
]

MITDB_SYMBOL_TO_CLASS = {
    "N": "normal",
    "/": "normal",
    "V": "pvc",
    "F": "pvc",
    "A": "apc",
    "a": "apc",
    "J": "apc",
    "L": "bb_block",
    "R": "bb_block",
    "B": "bb_block",
    "E": "escape",
    "e": "escape",
    "j": "escape",
    "n": "escape",
    "Q": "aberrant",
    "S": "aberrant",
    "r": "aberrant",
    "f": "fusion",
    "x": "av_block",
    "?": "other",
}

WAVEFORM_TO_EVAL = {
    WaveformClass.NORMAL: "normal",
    WaveformClass.PVC: "pvc",
    WaveformClass.PVC_ABERRANT: "pvc",
    WaveformClass.BB_BLOCK: "bb_block",
    WaveformClass.ESCAPE: "escape",
    WaveformClass.APC: "apc",
    WaveformClass.APC_ABERRANT: "apc",
    WaveformClass.PREMATURE: "premature",
    WaveformClass.ABERRANT: "aberrant",
    WaveformClass.UNKNOWN: "unknown",
    WaveformClass.INVALID: "unknown",
    WaveformClass.VIRTUAL: "unknown",
}


def reference_class_from_symbol(symbol):
    return MITDB_SYMBOL_TO_CLASS.get(symbol, "other")


def predicted_class_from_classification(classification):
    """Map classifier output to evaluation beat class (Hearty-style grouping)."""
    if classification.rhythm == RhythmClass.FUSION:
        return "fusion"
    if classification.rhythm == RhythmClass.AV_BLOCK:
        return "av_block"

    eval_class = WAVEFORM_TO_EVAL.get(classification.waveform, "unknown")
    if eval_class == "normal" and classification.rhythm != RhythmClass.NONE:
        rhythm_map = {
            RhythmClass.TACHYCARDIA: "normal",
            RhythmClass.BRADYCARDIA: "normal",
            RhythmClass.ARTIFACT: "aberrant",
        }
        return rhythm_map.get(classification.rhythm, eval_class)
    return eval_class


def is_reference_normal(symbol):
    return symbol in {"N", "/"}


def is_predicted_normal(classification):
    return classification.is_normal


def check_symbol_mapping(symbols):
    """Report MIT-BIH symbols and whether each has an evaluation class."""
    unique = sorted(set(symbols))
    rows = []
    for symbol in unique:
        rows.append(
            {
                "symbol": symbol,
                "eval_class": reference_class_from_symbol(symbol),
                "binary": "normal" if is_reference_normal(symbol) else "abnormal",
                "mapped": symbol in MITDB_SYMBOL_TO_CLASS,
            }
        )
    return pd.DataFrame(rows)


def check_eval_class_coverage(reference_classes, predicted_classes):
    """Verify every eval class appears in reference or prediction sets."""
    ref_set = set(reference_classes)
    pred_set = set(predicted_classes)
    rows = []
    for cls in EVAL_BEAT_CLASSES:
        rows.append(
            {
                "eval_class": cls,
                "in_reference": cls in ref_set,
                "in_predicted": cls in pred_set,
                "reference_count": reference_classes.count(cls),
                "predicted_count": predicted_classes.count(cls),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class SubjectEvaluation:
    record_id: str
    n_annotated: int = 0
    n_detected_synced: int = 0
    n_classified: int = 0
    n_undetected: int = 0
    reference_symbols: list = field(default_factory=list)
    reference_classes: list = field(default_factory=list)
    predicted_classes: list = field(default_factory=list)
    classifications: list = field(default_factory=list)
    binary_tp: int = 0
    binary_tn: int = 0
    binary_fp: int = 0
    binary_fn: int = 0
    template_init_failed: bool = False
    skipped: bool = False
    skip_reason: str = ""
    detection_rate: float = float("nan")


def _annotation_map(subject):
    mapping = {}
    for sample, symbol in zip(subject.annotations.sample, subject.annotations.symbol):
        mapping[int(sample)] = symbol
    return mapping


def _score_hearty(subject, beat_times, classifications, learning_times, channel=0):
    """Drive HeartySimResult sample-by-sample (HeartyActivity bookkeeping).

    ``beat_times`` / ``learning_times`` are absolute sample indices when
    ``newBeat`` fires (segmentation FINISHED), matching Hearty.
    """
    n_samples = len(subject.raw_signal[:, channel])
    ann_at = _annotation_map(subject)
    delay = HEARTY_TOTAL_DELAY

    learning_at = {min(int(t), n_samples - 1) for t in learning_times}
    classified_at = {
        min(int(t), n_samples - 1): c for t, c in zip(beat_times, classifications)
    }

    sim = HeartySimResult(total_delay=delay)
    learning = True
    learning_left = len(learning_at)

    for sample in range(n_samples):
        label = ann_at.get(sample, "\0")
        sim.new_label(label, learning=learning)

        if sample in learning_at:
            placeholder = BeatClassification(
                WaveformClass.UNKNOWN,
                RhythmClass.NONE,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            sim.new_beat(placeholder, learning=True)
            learning_left -= 1
            if learning_left <= 0:
                learning = False
        elif sample in classified_at:
            sim.new_beat(classified_at[sample], learning=False)

    sim.finish()
    return sim


def _process_subject(
    subject,
    channel=None,
    template_beats=TEMPLATE_INIT_BEATS,
    align_search_ms=None,
    peak_tolerance_ms=PEAK_TOLERANCE_MS,
):
    # Same lead index for every subject (channel 0).
    if channel is None:
        channel = 0

    run = detect_hearty_beats(subject.raw_signal[:, channel], fs=subject.fs)
    beats = [b for b in run.beats if not b.invalid and len(b.values) > 0]

    result = SubjectEvaluation(
        record_id=str(subject.number),
        n_annotated=len(subject.annotations.sample),
        n_detected_synced=len(beats),
        n_undetected=max(len(subject.annotations.sample) - len(beats), 0),
    )

    if len(beats) < template_beats + 1:
        result.template_init_failed = True
        result.skip_reason = "insufficient windows for templates"
        return result

    try:
        _, classifications = classify_hearty_beats(beats, start_idx=template_beats)
    except ValueError:
        result.template_init_failed = True
        result.skip_reason = "template initialization failed"
        return result

    learning_beats = beats[:template_beats]
    scored_beats = beats[template_beats:]

    sim = _score_hearty(
        subject,
        beat_times=[b.finish_sample for b in scored_beats],
        classifications=classifications,
        learning_times=[b.finish_sample for b in learning_beats],
        channel=channel,
    )

    result.n_classified = len(classifications)
    result.classifications = classifications
    result.binary_tp = sim.num_tp
    result.binary_tn = sim.num_tn
    result.binary_fp = sim.num_fp
    result.binary_fn = sim.num_fn
    result.n_annotated = sim.num_total_beats_ref
    result.n_detected_synced = sim.num_total_beats
    result.detection_rate = sim.detection_rate
    result.n_undetected = max(sim.num_total_beats_ref - sim.num_total_beats, 0)

    scored_r = [b.r_sample for b in scored_beats]
    synced_r, _, synced_labels = synchronize_peaks(
        annotated_peaks=subject.annotations.sample,
        labels=subject.annotations.symbol,
        detected_r_peaks=scored_r,
        detected_int_peaks=scored_r,
        fs=subject.fs,
        tolerance_ms=peak_tolerance_ms,
    )
    clf_by_peak = {int(p): c for p, c in zip(scored_r, classifications)}
    kept_clf = []
    kept_sym = []
    for p, sym in zip(synced_r, synced_labels):
        if int(p) in clf_by_peak:
            kept_clf.append(clf_by_peak[int(p)])
            kept_sym.append(sym)
    result.reference_symbols = kept_sym
    result.reference_classes = [reference_class_from_symbol(s) for s in kept_sym]
    result.predicted_classes = [predicted_class_from_classification(c) for c in kept_clf]

    return result


def evaluate_subject(subject, channel=None, **kwargs):
    record_id = str(subject.number)
    if record_id in GRADL_OMIT_RECORDS:
        out = SubjectEvaluation(record_id=record_id, skipped=True, skip_reason="Gradl omitted record")
        return out
    return _process_subject(subject, channel=channel, **kwargs)


def evaluate_subjects(subjects, **kwargs):
    return [evaluate_subject(s, **kwargs) for s in subjects]


def aggregate_evaluations(evaluations):
    usable = [e for e in evaluations if not e.skipped and not e.template_init_failed]
    if not usable:
        return {
            "n_records": 0,
            "binary_tp": 0,
            "binary_tn": 0,
            "binary_fp": 0,
            "binary_fn": 0,
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "detection_rate": float("nan"),
        }

    ref_all = []
    pred_all = []
    sym_all = []
    tp = tn = fp = fn = 0
    n_ann = n_det = 0

    for e in usable:
        ref_all.extend(e.reference_classes)
        pred_all.extend(e.predicted_classes)
        sym_all.extend(e.reference_symbols)
        tp += e.binary_tp
        tn += e.binary_tn
        fp += e.binary_fp
        fn += e.binary_fn
        n_ann += e.n_annotated
        n_det += e.n_detected_synced

    se = tp / (tp + fn) if (tp + fn) else float("nan")
    sp = tn / (tn + fp) if (tn + fp) else float("nan")
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = se

    return {
        "n_records": len(usable),
        "n_skipped": sum(1 for e in evaluations if e.skipped),
        "n_template_failed": sum(1 for e in evaluations if e.template_init_failed),
        "binary_tp": tp,
        "binary_tn": tn,
        "binary_fp": fp,
        "binary_fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "sensitivity": se,
        "specificity": sp,
        "detection_rate": n_det / n_ann if n_ann else float("nan"),
        "reference_classes": ref_all,
        "predicted_classes": pred_all,
        "reference_symbols": sym_all,
    }


def confusion_matrix_df(reference, predicted, labels=None, margins=True):
    ref = pd.Series(reference, name="reference")
    pred = pd.Series(predicted, name="predicted")
    cm = pd.crosstab(ref, pred, margins=margins, dropna=False)
    if labels is not None:
        index = list(labels)
        columns = list(labels)
        if margins:
            index.append("All")
            columns.append("All")
        cm = cm.reindex(index=index, columns=columns, fill_value=0)
    return cm


def symbol_confusion_matrix(reference_symbols, predicted_classes):
    """Rows = MIT-BIH symbol, columns = predicted eval class."""
    ref = pd.Series(reference_symbols, name="symbol")
    pred = pd.Series(predicted_classes, name="predicted")
    return pd.crosstab(ref, pred, margins=True)


def per_class_metrics(confusion_df):
    """Precision, recall, F1 per class from a square confusion matrix (no margins)."""
    if "All" in confusion_df.index:
        cm = confusion_df.iloc[:-1, :-1]
    else:
        cm = confusion_df.copy()

    rows = []
    for cls in cm.index:
        tp = cm.loc[cls, cls] if cls in cm.columns else 0
        fp = cm[cls].sum() - tp if cls in cm.columns else 0
        fn = cm.loc[cls].sum() - tp
        support = cm.loc[cls].sum()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "class": cls,
                "support_ref": int(support),
                "tp": int(tp),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def general_confusion_matrix(summary):
    """2×2 normal/abnormal matrix (Gradl Table I). Rows=reference, cols=predicted."""
    return pd.DataFrame(
        [
            [summary["binary_tp"], summary["binary_fn"]],
            [summary["binary_fp"], summary["binary_tn"]],
        ],
        index=pd.Index(["abnormal", "normal"], name="reference"),
        columns=pd.Index(["abnormal", "normal"], name="predicted"),
    )


def binary_confusion_matrix(summary):
    return general_confusion_matrix(summary)


def _pct(value):
    if value != value:
        return "n/a"
    return f"{value:.2%}"


def format_classification_metrics(summary):
    """Accuracy, precision, recall, sensitivity (all %) plus Gradl counts."""
    return pd.DataFrame(
        {
            "metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "Sensitivity",
                "Specificity",
                "Detected beats",
                "True Positive",
                "True Negative",
                "False Positive",
                "False Negative",
            ],
            "value": [
                _pct(summary.get("accuracy", float("nan"))),
                _pct(summary.get("precision", float("nan"))),
                _pct(summary.get("recall", float("nan"))),
                _pct(summary.get("sensitivity", float("nan"))),
                _pct(summary.get("specificity", float("nan"))),
                _pct(summary.get("detection_rate", float("nan"))),
                summary.get("binary_tp", 0),
                summary.get("binary_tn", 0),
                summary.get("binary_fp", 0),
                summary.get("binary_fn", 0),
            ],
        }
    )


def format_gradl_table(summary):
    return format_classification_metrics(summary)


def median_fn_per_record(evaluations):
    usable = [e for e in evaluations if not e.skipped and not e.template_init_failed]
    if not usable:
        return float("nan")
    return float(np.median([e.binary_fn for e in usable]))
