"""Filtering utilities for KU Leuven AAD preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import mne
import numpy as np

from loading import load_subject_trials

_FILTER_PARAMS = {
    "ica_prep": {"l_freq": 1.0, "h_freq": None},
    "analysis": {"l_freq": 1.0, "h_freq": 8.0},
}
_ANALYSIS_RESAMPLE_FREQ = 64.0

# The source .mat files are already hardware/amplifier high-pass filtered at
# 0.5 Hz (RawData.HighPass; README: "Each trial was high-pass filtered (0.5 Hz
# cut off)..."), so content below 0.5 Hz is already gone before we see it.
# MNE's default transition band for a 1 Hz 'auto' high-pass extends all the
# way down to 0 Hz (-6 dB point at 0.5 Hz), i.e. it's designed as if the
# input were unfiltered full-spectrum data. We instead set an explicit,
# narrower transition band so our filter's stopband starts at (not below)
# the hardware cutoff, rather than re-filtering a region that's already gone.
_HARDWARE_HIGHPASS_HZ = 0.5


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
    l_trans_bandwidth = params["l_freq"] - _HARDWARE_HIGHPASS_HZ
    filtered.filter(l_freq=params["l_freq"], h_freq=params["h_freq"], fir_design="firwin",
                     l_trans_bandwidth=l_trans_bandwidth)

    if kind == "analysis":
        filtered.resample(_ANALYSIS_RESAMPLE_FREQ)

    filtered, _ = mne.set_eeg_reference(filtered, "average")

    return filtered


def make_raw(trial: dict) -> mne.io.RawArray:
    """
    Convert one `loading.load_trial` dict into an MNE Raw, so it can be fed
    into `filter_raw` / `fit_ica` / `apply_ica` exactly like `load_subject`'s
    continuous Raw.

    Channel names come from `mne.channels.make_standard_montage("biosemi64")`
    directly rather than from the trial dict (this dataset's own channel
    order matches that montage order exactly, as used elsewhere in this
    project), so this only needs ``eeg`` and ``fs`` from `trial`. An average
    EEG reference projection is added but not applied -- call
    `raw.apply_proj()` (or let a downstream MNE operation apply it) when ready.

    Parameters
    ----------
    trial : a dict as returned by `loading.load_trial` (needs ``eeg`` and ``fs``).

    Returns
    -------
    mne.io.RawArray with the "biosemi64" montage set and an unapplied
    average EEG reference projection.
    """
    montage = mne.channels.make_standard_montage("biosemi64")
    info = mne.create_info(ch_names=montage.ch_names, sfreq=trial["fs"], ch_types="eeg")
    raw = mne.io.RawArray(trial["eeg"] * 1e-6, info)  # stored in microvolts; MNE expects volts
    raw.set_montage(montage)
    raw.set_eeg_reference("average", projection=True)
    return raw


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
    ica = mne.preprocessing.ICA(n_components=n_components, max_iter="auto", random_state=97)
    ica.fit(raw_hp)

    eog_indices, _ = ica.find_bads_eog(raw_hp, ch_name="Fp1")
    ica.exclude = eog_indices

    print(f"fit_ica: flagged {len(ica.exclude)} EOG-related component(s) for exclusion: {ica.exclude}")

    if save_path is not None:
        ica.save(save_path, overwrite=True)

    return ica


def load_ica(path: Union[str, Path]) -> mne.preprocessing.ICA:
    """Load an ICA solution saved by `fit_ica(..., save_path=...)`, without refitting."""
    return mne.preprocessing.read_ica(path)


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


def run_subject(subject_id: Union[int, str], mat_path: Union[str, Path],
                 stimuli_dir: Union[str, Path], processed_dir: Union[str, Path],
                 ica_dir: Union[str, Path]) -> Path:
    """
    Run the full per-subject pipeline from raw trials to a saved, cleaned
    checkpoint: load every trial, build and concatenate a continuous Raw,
    fit ICA (saved to disk), filter for analysis, apply ICA, and save the
    cleaned checkpoint.

    Parameters
    ----------
    subject_id : subject number, e.g. ``1`` or ``"1"`` or ``"S1"``.
    mat_path : path to the subject's .mat file.
    stimuli_dir : directory holding the stimulus .wav files.
    processed_dir : directory to save the cleaned Raw checkpoint into
        (see `save_cleaned_raw`).
    ica_dir : directory to save the fitted ICA into, as ``sub{id:02d}-ica.fif``.

    Returns
    -------
    Path to the saved, cleaned Raw checkpoint.
    """
    subject_num = int(str(subject_id).lstrip("Ss"))
    tag = f"run_subject {subject_num}"

    print(f"{tag}: loading trials from {mat_path}...")
    trials = load_subject_trials(mat_path, stimuli_dir)

    print(f"{tag}: building and concatenating {len(trials)} trial Raw(s)...")
    raw = mne.concatenate_raws([make_raw(t) for t in trials])

    print(f"{tag}: high-pass filtering for ICA...")
    raw_hp = filter_raw(raw, "ica_prep")

    print(f"{tag}: fitting ICA...")
    ica_path = Path(ica_dir) / f"sub{subject_num:02d}-ica.fif"
    ica_path.parent.mkdir(parents=True, exist_ok=True)
    ica = fit_ica(raw_hp, save_path=ica_path)

    print(f"{tag}: band-pass filtering for analysis...")
    raw_analysis = filter_raw(raw, "analysis")

    print(f"{tag}: applying ICA...")
    cleaned = apply_ica(raw_analysis, ica)

    print(f"{tag}: saving cleaned checkpoint...")
    path = save_cleaned_raw(cleaned, subject_num, processed_dir)

    print(f"{tag}: done -> {path}")
    return path
