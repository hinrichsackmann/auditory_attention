"""Filtering utilities for KU Leuven AAD preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import mne
import numpy as np

_FILTER_PARAMS = {
    "ica_prep": {"l_freq": 1.0, "h_freq": None},
    "analysis": {"l_freq": 1.0, "h_freq": 8.0},
}
_ANALYSIS_RESAMPLE_FREQ = 64.0


def filter_raw(raw: mne.io.BaseRaw, kind: str) -> mne.io.BaseRaw:
    """
    Filter a copy of `raw` and return it; the input is left unmodified.

    Parameters
    ----------
    raw : mne.io.BaseRaw
    kind : "ica_prep" for a 1 Hz high-pass only (kept at the original
        sample rate, for ICA decomposition), or "analysis" for a 1-8 Hz
        band-pass, downsampled to 64 Hz afterwards.
    """
    if kind not in _FILTER_PARAMS:
        raise ValueError(f"kind must be one of {list(_FILTER_PARAMS)}, got {kind!r}")

    filtered = raw.copy()
    params = _FILTER_PARAMS[kind]
    filtered.filter(l_freq=params["l_freq"], h_freq=params["h_freq"], fir_design="firwin")

    if kind == "analysis":
        filtered.resample(_ANALYSIS_RESAMPLE_FREQ)

    filtered, _ = mne.set_eeg_reference(filtered, "average")

    return filtered


def fit_ica(raw_hp: mne.io.BaseRaw, n_components: int = 20,
            save_path: Optional[Union[str, Path]] = None) -> mne.preprocessing.ICA:
    """
    Fit ICA on the high-pass filtered Raw and auto-flag EOG components.

    This dataset has no dedicated EOG channel, so Fp1 (the most frontal
    electrode, closest to the eyes) is used as an EOG proxy for detection,
    via `ICA.find_bads_eog`.

    Parameters
    ----------
    raw_hp : high-pass filtered Raw (see filter_raw(..., "ica_prep")).
    n_components : number of ICA components to fit.
    save_path : if given, the fitted ICA is saved here (must end in
        "-ica.fif" or "-ica.fif.gz", per MNE's naming convention).

    Returns
    -------
    mne.preprocessing.ICA, fitted, with `.exclude` set to the detected
    EOG-related components.
    """
    ica = mne.preprocessing.ICA(n_components=n_components, max_iter="auto", rng=97)
    ica.fit(raw_hp)

    eog_indices, _ = ica.find_bads_eog(raw_hp, ch_name="Fp1")
    ica.exclude = eog_indices

    print(f"fit_ica: flagged {len(ica.exclude)} EOG-related component(s) for exclusion: {ica.exclude}")

    if save_path is not None:
        ica.save(save_path, overwrite=True)

    return ica


def apply_ica(raw_analysis: mne.io.BaseRaw, ica: mne.preprocessing.ICA,
              exclude: Optional[Sequence[int]] = None) -> mne.io.BaseRaw:
    """
    Apply a fitted ICA solution to the analysis-band Raw and return the
    cleaned Raw; the input is left unmodified.

    Parameters
    ----------
    raw_analysis : band-pass filtered Raw (see filter_raw(..., "analysis")).
    ica : fitted ICA (see fit_ica).
    exclude : component indices to remove; defaults to `ica.exclude`.

    Returns
    -------
    mne.io.BaseRaw, with the excluded components removed.
    """
    if exclude is None:
        exclude = ica.exclude

    print(f"apply_ica: excluding {len(exclude)} component(s): {list(exclude)}")

    cleaned = raw_analysis.copy()
    ica.apply(cleaned, exclude=exclude)
    return cleaned


def get_trial_segments(raw_clean: mne.io.BaseRaw,
                        trial_markers: Union[mne.Annotations, Sequence[float]]) -> list:
    """
    Split the cleaned continuous Raw into one Raw per trial and print each
    trial's duration.

    Parameters
    ----------
    raw_clean : the cleaned, continuous Raw (output of apply_ica, or
        reloaded via load_cleaned_raw -- trial boundaries survive both).
    trial_markers : the trial boundaries, either an mne.Annotations object
        (typically `raw_clean.annotations`, as set by `load_subject`) giving
        each trial's onset and duration directly, or a plain 1D sequence of
        trial onset times in seconds, in which case each trial is assumed to
        run up to the next onset (the last one runs to the end of the data).

    Returns
    -------
    list of mne.io.BaseRaw, one per trial, in marker order.
    """
    if isinstance(trial_markers, mne.Annotations):
        onsets = trial_markers.onset
        durations = trial_markers.duration
        labels = list(trial_markers.description)
    else:
        onsets = np.asarray(trial_markers, dtype=float)
        end_time = raw_clean.times[-1] + 1 / raw_clean.info["sfreq"]
        durations = np.diff(np.append(onsets, end_time))
        labels = [f"trial_{i + 1:02d}" for i in range(len(onsets))]

    segments = []
    for label, onset, duration in zip(labels, onsets, durations):
        segment = raw_clean.copy().crop(tmin=onset, tmax=onset + duration, include_tmax=False)
        actual_duration = segment.times[-1] + 1 / segment.info["sfreq"]
        print(f"get_trial_segments: {label}: {actual_duration:.2f} s ({segment.n_times} samples)")
        segments.append(segment)

    return segments


def checkpoint_path(subject_id: Union[int, str], processed_dir: Union[str, Path]) -> Path:
    """Path to a subject's cleaned, analysis-band Raw checkpoint, e.g. ``data/processed/sub01_raw.fif``."""
    subject_num = int(str(subject_id).lstrip("Ss"))
    return Path(processed_dir) / f"sub{subject_num:02d}_raw.fif"


def save_cleaned_raw(raw_cleaned: mne.io.BaseRaw, subject_id: Union[int, str],
                      processed_dir: Union[str, Path]) -> Path:
    """Save the cleaned, analysis-band Raw (see apply_ica) as the subject's checkpoint .fif."""
    path = checkpoint_path(subject_id, processed_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_cleaned.save(path, overwrite=True)
    print(f"save_cleaned_raw: wrote {path}")
    return path


def load_cleaned_raw(subject_id: Union[int, str], processed_dir: Union[str, Path]) -> mne.io.Raw:
    """
    Load a subject's cleaned, analysis-band Raw checkpoint saved by
    `save_cleaned_raw`. Everything downstream (epoching, TRF fitting,
    decoding, ...) should load from this checkpoint rather than re-running
    load_subject / filter_raw / fit_ica / apply_ica.
    """
    path = checkpoint_path(subject_id, processed_dir)
    return mne.io.read_raw_fif(path, preload=True)
