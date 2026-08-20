from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECORDS_FOLDER = PROJECT_ROOT / "data" / "physionet.org" / "files" / "mitdb" / "1.0.0"

DEFAULT_FS = 360
FILTER_ORDER = 2
WINDOW_SPAN_MS = 400
PEAK_TOLERANCE_MS = 150
SEARCH_WINDOW_MS = 20
# Integrator peak lags the R wave by ~half the 150 ms MWI; search back that far.
ALIGN_LOOKBACK_MS = 150
ALIGN_LOOKFORWARD_MS = 40
# Hearty PanTompkins.TOTAL_DELAY (samples) for annotation alignment in scoring.
HEARTY_TOTAL_DELAY = 24
TEMPLATE_INIT_BEATS = 6

ARRHYTHMIA_SYMBOLS = [
    "L", "R", "A", "a", "J", "S", "V", "F", "e", "j", "E", "Q", "?",
]
