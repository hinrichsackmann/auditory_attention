"""Run the filter/ICA pipeline for every subject in data/raw and save checkpoints."""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from loading import load_subject
from preprocessing import filter_raw, fit_ica, apply_ica, save_cleaned_raw

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    subject_files = sorted(RAW_DIR.glob("S*.mat"))
    print(f"Found {len(subject_files)} subject file(s).")

    for f in subject_files:
        subject_num = int(f.stem.lstrip("S"))
        t0 = time.time()
        print(f"\n=== Subject {subject_num} ===")

        raw = load_subject(subject_num, RAW_DIR)
        raw_hp = filter_raw(raw, "ica_prep")
        raw_analysis = filter_raw(raw, "analysis")
        ica = fit_ica(raw_hp, n_components=20)
        cleaned = apply_ica(raw_analysis, ica)
        save_cleaned_raw(cleaned, subject_num, PROCESSED_DIR)

        print(f"Subject {subject_num} done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
