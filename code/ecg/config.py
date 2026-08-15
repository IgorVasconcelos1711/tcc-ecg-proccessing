from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECORDS_FOLDER = PROJECT_ROOT / "data" / "physionet.org" / "files" / "mitdb" / "1.0.0"

DEFAULT_FS = 360
FILTER_ORDER = 2
WINDOW_SPAN_MS = 400
PEAK_TOLERANCE_MS = 150
SEARCH_WINDOW_MS = 20

ARRHYTHMIA_SYMBOLS = [
    "L", "R", "A", "a", "J", "S", "V", "F", "e", "j", "E", "Q", "?",
]
