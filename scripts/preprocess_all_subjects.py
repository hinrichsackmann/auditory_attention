"""Run the filter/ICA pipeline for every subject in data/raw and save checkpoints."""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocessing import run_subject

RAW_DIR = PROJECT_ROOT / "data" / "raw"
STIMULI_DIR = RAW_DIR / "stimuli"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ICA_DIR = PROCESSED_DIR / "ica"


def main():
    subject_files = sorted(RAW_DIR.glob("S*.mat"))
    print(f"Found {len(subject_files)} subject file(s).")

    for f in subject_files:
        subject_num = int(f.stem.lstrip("S"))
        t0 = time.time()
        print(f"\n=== Subject {subject_num} ===")

        run_subject(subject_num, f, STIMULI_DIR, PROCESSED_DIR, ICA_DIR)

        print(f"Subject {subject_num} done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
