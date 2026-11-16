"""Filtering utilities for KU Leuven AAD preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import mne
import numpy as np

from loading import _ANALYSIS_RESAMPLE_FREQ, load_subject_trials

_FILTER_PARAMS = {
    "ica_prep": {"l_freq": 1.0, "h_freq": None},
    "analysis": {"l_freq": 1.0, "h_freq": 8.0},
}

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
    project), so this only needs ``eeg``, ``fs``, and ``trial_id`` from
    `trial`. A single annotation spanning the whole Raw records the trial's
    identity (``trial_NN``, from ``trial_id``) so it survives
    `mne.concatenate_raws` -- see `run_subject` and `get_trial_segments`. An
    average EEG reference projection is added but not applied -- call
    `raw.apply_proj()` (or let a downstream MNE operation apply it) when ready.

    Parameters
    ----------
    trial : a dict as returned by `loading.load_trial` (needs ``eeg``, ``fs``,
        and ``trial_id``).

    Returns
    -------
    mne.io.RawArray with the "biosemi64" montage set, a ``trial_NN``
    annotation spanning its full duration, and an unapplied average EEG
    reference projection.
    """
    montage = mne.channels.make_standard_montage("biosemi64")
    info = mne.create_info(ch_names=montage.ch_names, sfreq=trial["fs"], ch_types="eeg")
    raw = mne.io.RawArray(trial["eeg"] * 1e-6, info)  # stored in microvolts; MNE expects volts
    raw.set_montage(montage)
    raw.set_annotations(mne.Annotations(
        onset=[0.0],
        duration=[raw.times[-1] + 1 / raw.info["sfreq"]],
        description=[f"trial_{trial['trial_id']:02d}"],
    ))
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
    ica = mne.preprocessing.ICA(n_components=n_components, max_iter="auto", rng=97)
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
        (typically `raw_clean.annotations`, as set per-trial by `make_raw`
        and carried through `run_subject`'s concatenation) giving each
        trial's onset and duration directly -- entries other than the
        ``trial_NN`` ones (e.g. the "BAD boundary"/"EDGE boundary"
        annotations `mne.concatenate_raws` adds at every join) are ignored --
        or a plain 1D sequence of trial onset times in seconds, in which
        case each trial is assumed to run up to the next onset (the last one
        runs to the end of the data).

    Returns
    -------
    list of mne.io.BaseRaw, one per trial, in marker order.
    """
    if isinstance(trial_markers, mne.Annotations):
        is_trial = np.array([d.startswith("trial_") for d in trial_markers.description])
        onsets = trial_markers.onset[is_trial]
        durations = trial_markers.duration[is_trial]
        labels = [d for d, keep in zip(trial_markers.description, is_trial) if keep]
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


def align_trial_lengths(eeg: np.ndarray, envelope_attended: np.ndarray,
                         envelope_unattended: np.ndarray) -> tuple:
    """
    Trim a trial's EEG and envelopes to their shared minimum length.

    Independent resampling (MNE for EEG via `filter_raw`, scipy for envelopes
    via `loading._extract_envelope`) can leave EEG and envelopes a few samples
    apart even at the same rate -- this makes them exactly equal length.

    Parameters
    ----------
    eeg : (n_channels, n_samples)
    envelope_attended, envelope_unattended : (n_samples,)

    Returns
    -------
    (eeg, envelope_attended, envelope_unattended), all trimmed to the same
    number of samples.
    """
    n = min(eeg.shape[1], len(envelope_attended), len(envelope_unattended))
    return eeg[:, :n], envelope_attended[:n], envelope_unattended[:n]


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
    checkpoint: load every trial, filter each trial separately (never across
    trial boundaries), concatenate the filtered trials, fit ICA (saved to
    disk), apply ICA, and save the cleaned checkpoint.

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
    trials = load_subject_trials(mat_path, stimuli_dir, extract_envelope=False)

    print(f"{tag}: filtering {len(trials)} trial(s) separately (ica_prep + analysis)...")
    raw_hp_trials = []
    raw_analysis_trials = []
    for trial in trials:
        raw = make_raw(trial)
        raw_hp_trials.append(filter_raw(raw, "ica_prep"))
        raw_analysis_trials.append(filter_raw(raw, "analysis"))

    print(f"{tag}: concatenating high-passed trials and fitting ICA...")
    raw_hp = mne.concatenate_raws(raw_hp_trials)
    ica_path = Path(ica_dir) / f"sub{subject_num:02d}-ica.fif"
    ica_path.parent.mkdir(parents=True, exist_ok=True)
    ica = fit_ica(raw_hp, save_path=ica_path)

    print(f"{tag}: concatenating analysis-band trials and applying ICA...")
    raw_analysis = mne.concatenate_raws(raw_analysis_trials)
    cleaned = apply_ica(raw_analysis, ica)

    print(f"{tag}: saving cleaned checkpoint...")
    path = save_cleaned_raw(cleaned, subject_num, processed_dir)

    print(f"{tag}: done -> {path}")
    return path
