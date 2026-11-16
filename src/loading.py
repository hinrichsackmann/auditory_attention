"""Inspect KU Leuven AAD .mat files (MATLAB v7.3 / HDF5 format).

scipy.io.loadmat cannot read these files -- they're saved as MATLAB 7.3,
which stores data as HDF5. h5py is used instead, and MATLAB cell arrays /
struct-array fields are represented as HDF5 object references into the
file's shared '#refs#' group, which this module dereferences recursively.
"""

from __future__ import annotations

from math import gcd
from pathlib import Path
from typing import Union

import h5py
import librosa
import mne
import numpy as np
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt

_ENVELOPE_BAND_HZ = (80, 3000)
_ENVELOPE_FILTER_ORDER = 4

# EEG is resampled to this rate for analysis (see preprocessing.filter_raw's
# "analysis" kind); envelopes default to the same rate so they line up with
# the cleaned EEG. Kept here (not in preprocessing.py) since preprocessing.py
# already imports from loading.py, and this avoids a circular import.
_ANALYSIS_RESAMPLE_FREQ = 64.0

_INTERNAL_GROUPS = {"#refs#"}
_REF_DTYPE = h5py.special_dtype(ref=h5py.Reference)
_VALUE_PREVIEW_MAX_ELEMENTS = 20
_STRING_PREVIEW_MAX_CHARS = 60


def _matlab_class(obj: Union[h5py.Dataset, h5py.Group]) -> str:
    cls = obj.attrs.get("MATLAB_class")
    if cls is None:
        return "group" if isinstance(obj, h5py.Group) else "dataset"
    return cls.decode() if isinstance(cls, bytes) else str(cls)


def _decode_matlab_string(dataset: h5py.Dataset) -> Union[str, None]:
    try:
        data = np.asarray(dataset[()]).ravel()
        return "".join(chr(c) for c in data)
    except Exception:
        return None

def scalar(dataset):
    """Extract a scalar value from an HDF5 dataset of any shape."""
    return dataset[()].flat[0]


def _value_preview(dataset: h5py.Dataset, matlab_class: str) -> Union[str, None]:
    if matlab_class == "char":
        text = _decode_matlab_string(dataset)
        if text is None:
            return None
        if len(text) > _STRING_PREVIEW_MAX_CHARS:
            text = text[: _STRING_PREVIEW_MAX_CHARS - 3] + "..."
        return repr(text)

    if matlab_class in ("double", "single", "logical", "int8", "int16", "int32", "int64",
                        "uint8", "uint16", "uint32", "uint64") and dataset.size <= _VALUE_PREVIEW_MAX_ELEMENTS:
        return str(np.asarray(dataset[()]).squeeze().tolist())

    return None


def _describe_dataset(name: str, dataset: h5py.Dataset, indent: str, file: h5py.File,
                       max_cell_items: int, max_depth: int, depth: int) -> None:
    mclass = _matlab_class(dataset)
    header = f"{indent}{name}: shape={dataset.shape} dtype={dataset.dtype} matlab_class={mclass!r}"

    if dataset.dtype == _REF_DTYPE:
        print(header + f"  [cell array, {dataset.size} element(s)]")
        if depth >= max_depth:
            print(f"{indent}  ... max depth reached, not expanding")
            return
        refs = np.asarray(dataset[()]).ravel()
        for i, ref in enumerate(refs[:max_cell_items]):
            if not ref:
                continue
            target = file[ref]
            print(f"{indent}  element[{i}]:")
            _describe(target, f"element[{i}]", indent + "    ", file, max_cell_items, max_depth, depth + 1)
        if dataset.size > max_cell_items:
            print(f"{indent}  ... {dataset.size - max_cell_items} more element(s) not shown")
        return

    preview = _value_preview(dataset, mclass)
    print(header + (f"  value={preview}" if preview is not None else ""))


def _describe(obj: Union[h5py.Dataset, h5py.Group], name: str, indent: str, file: h5py.File,
              max_cell_items: int, max_depth: int, depth: int) -> None:
    if isinstance(obj, h5py.Dataset):
        _describe_dataset(name, obj, indent, file, max_cell_items, max_depth, depth)
        return

    fields = list(obj.keys())
    print(f"{indent}{name}: struct, fields={fields}")
    if depth >= max_depth:
        print(f"{indent}  ... max depth reached, not expanding")
        return
    for field in fields:
        _describe(obj[field], field, indent + "  ", file, max_cell_items, max_depth, depth + 1)


def inspect_mat(filepath: Union[str, Path], max_cell_items: int = 1, max_depth: int = 8) -> None:
    """
    Load a KU Leuven AAD .mat file (MATLAB v7.3 / HDF5 format) and pretty-print
    every top-level key, recursively, with shapes, dtypes, MATLAB classes, and
    (for small/char entries) a value preview.

    Parameters
    ----------
    filepath : path to the .mat file.
    max_cell_items : how many elements of each cell array to expand. Cell
        arrays in this dataset (e.g. ``trials``) hold many structurally
        identical elements, so only the first `max_cell_items` are shown.
    max_depth : recursion limit for nested structs/cells, as a safety net
        against unexpectedly deep or self-referential structures.
    """
    filepath = Path(filepath)
    print(f"Inspecting: {filepath}")
    print("=" * 80)

    with h5py.File(filepath, "r") as f:
        top_keys = list(f.keys())
        internal = [k for k in top_keys if k in _INTERNAL_GROUPS]
        visible = [k for k in top_keys if k not in _INTERNAL_GROUPS]

        for key in visible:
            print()
            _describe(f[key], key, "", f, max_cell_items, max_depth, depth=0)

        if internal:
            print(f"\n(skipped internal HDF5 reference pool: {internal})")


def load_subject(subject_id: Union[int, str], data_dir: Union[str, Path]) -> mne.io.RawArray:
    """
    Load one subject's KU Leuven AAD recording into a continuous MNE Raw object.

    Each subject .mat file stores 20 trials (``trials`` cell array; see
    ``inspect_mat``), each with its own ``RawData.EegData`` (64 channels x
    samples, in microvolts) and shared ``FileHeader.SampleRate`` /
    ``FileHeader.Channels.Label`` (standard 10-20 names matching MNE's
    "biosemi64" montage exactly). Trials are sorted by TrialID and
    concatenated along time into a single continuous recording -- the same
    approach the dataset's own `preprocess_data.m` takes. Note this means
    there is a discontinuity at each trial boundary.

    Trial boundaries are stored as annotations (description ``"trial_NN"``,
    onset/duration in raw-relative seconds), so they survive filtering,
    resampling, and a .fif save/reload -- see `preprocessing.get_trial_segments`.

    Parameters
    ----------
    subject_id : subject number, e.g. ``1`` or ``"1"`` or ``"S1"``.
    data_dir : directory containing the subject .mat files (``data/raw``).

    Returns
    -------
    mne.io.RawArray with a "biosemi64" montage and trial-boundary
    annotations set.
    """
    subject_num = str(subject_id).lstrip("Ss")
    filepath = Path(data_dir) / f"S{subject_num}.mat"

    with h5py.File(filepath, "r") as f:
        trial_refs = np.asarray(f["trials"][()]).ravel()
        trials = [f[ref] for ref in trial_refs]

        label_refs = np.asarray(trials[0]["FileHeader"]["Channels"]["Label"][()]).ravel()
        ch_names = [_decode_matlab_string(f[ref]) for ref in label_refs]
        sfreq = float(np.asarray(trials[0]["FileHeader"]["SampleRate"][()]).item())

        trial_ids = [float(np.asarray(t["TrialID"][()]).item()) for t in trials]
        eeg_segments = [np.asarray(t["RawData"]["EegData"][()]) for t in trials]

    order = np.argsort(trial_ids)
    eeg_uv = np.concatenate([eeg_segments[i] for i in order], axis=1)
    eeg_v = eeg_uv * 1e-6  # stored in microvolts; MNE expects volts

    durations_samples = [eeg_segments[i].shape[1] for i in order]
    onsets_samples = np.concatenate([[0], np.cumsum(durations_samples)[:-1]])
    onset_times = onsets_samples / sfreq
    duration_times = np.array(durations_samples) / sfreq
    descriptions = [f"trial_{int(trial_ids[i]):02d}" for i in order]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(eeg_v, info)
    raw.set_montage(mne.channels.make_standard_montage("biosemi64"))
    raw.set_annotations(mne.Annotations(onset=onset_times, duration=duration_times, description=descriptions))

    return raw


def decode_str(arr) -> str:
    """Decode a MATLAB char array (stored as uint16 codepoints) to a string."""
    return "".join(chr(int(c)) for c in np.asarray(arr).ravel())


def _extract_envelope(audio: np.ndarray, audio_fs: float, target_fs: float) -> np.ndarray:
    """Bandpass (80-3000 Hz, 4th-order Butterworth) -> Hilbert envelope -> resample."""
    sos = butter(_ENVELOPE_FILTER_ORDER, _ENVELOPE_BAND_HZ, btype="bandpass",
                 fs=audio_fs, output="sos")
    filtered = sosfiltfilt(sos, audio)
    envelope = np.abs(hilbert(filtered))

    g = gcd(round(target_fs), round(audio_fs))
    up, down = round(target_fs) // g, round(audio_fs) // g
    return resample_poly(envelope, up, down)


def load_trial(filepath: Union[str, Path], trial_idx: int,
               stimuli_dir: Union[str, Path], extract_envelope: bool = True,
               target_fs: float = _ANALYSIS_RESAMPLE_FREQ,
               apply_trigger_shift: bool = False) -> dict:
    """
    Load one trial's EEG together with its attended/unattended audio.

    Parameters
    ----------
    filepath : path to a subject .mat file (MATLAB v7.3 / HDF5).
    trial_idx : index into the ``trials`` cell array (0-19).
    stimuli_dir : directory holding the stimulus .wav files
        (``data/raw/stimuli``).
    extract_envelope : if True (default), the attended/unattended audio is
        bandpass filtered (80-3000 Hz, 4th-order Butterworth), Hilbert-
        enveloped, and resampled to `target_fs`, truncated to match the
        trial's EEG duration per the README (audio runs longer than the EEG
        per trial). If False, the raw loaded audio is returned instead, for
        debugging.
    target_fs : sample rate the envelopes are resampled to. Defaults to
        `_ANALYSIS_RESAMPLE_FREQ` (64 Hz), matching the rate EEG ends up at
        after `preprocessing.filter_raw(..., "analysis")`, so envelope and
        (separately-processed) cleaned EEG line up. ``eeg`` in the returned
        dict stays at its native rate (``fs``) regardless -- this only
        controls the envelope's rate.
    apply_trigger_shift : if True, trims `eeg` and the envelopes by
        `RawData.TriggerAlignmentShift` samples, as an EEG/audio sync
        correction. UNVERIFIED: the value is a constant 10 across every
        trial checked (S1, S8, S16; 60 trials total), and neither the
        dataset README nor preprocess_data.m mentions this field -- a
        constant across every trial/subject is inconsistent with it being a
        measured per-trial lag, and more consistent with a fixed
        filter/decimation artifact from the dataset's own stated 0.5 Hz
        high-pass + 8192->128 Hz downsampling step. Defaults to False (no
        trim) until this is confirmed against the KUL dataset paper or its
        authors -- see PLAN.md's "Decisions to hold fixed" table, row
        "Trigger shift".

    Returns
    -------
    dict with keys ``eeg`` (64, n_samples, in microvolts as stored, in the
    standard BioSemi 10-20 channel order matching MNE's "biosemi64" montage
    -- see `preprocessing.make_raw`), ``fs``, ``attended_track``,
    ``attended_ear``, ``trial_id``, ``subject``, the two stimulus filenames,
    and either ``envelope_attended``/``envelope_unattended`` (n_samples, at
    ``envelope_fs``) when `extract_envelope` is True, or
    ``audio_attended``/``audio_unattended``/``audio_fs`` when False.
    """
    stimuli_dir = Path(stimuli_dir)

    with h5py.File(filepath, "r") as f:
        trial = f[f["trials"][trial_idx, 0]]

        eeg = np.asarray(trial["RawData"]["EegData"][()])
        fs = float(scalar(trial["FileHeader"]["SampleRate"]))
        attended_track = int(scalar(trial["attended_track"]))
        attended_ear = decode_str(trial["attended_ear"][()])
        trial_id = int(scalar(trial["TrialID"]))
        subject = decode_str(trial["subject"][()])
        trigger_alignment_shift = int(scalar(trial["RawData"]["TriggerAlignmentShift"]))

        stim_names = [decode_str(f[trial["stimuli"][0, i]][()]) for i in (0, 1)]

    # stimuli[0]/stimuli[1] are the files played in the LEFT/RIGHT ear (README),
    # so attention is resolved by ear. attended_track is the story track, which
    # matches the ear-selected file only on half the trials.
    attended_idx = 0 if attended_ear == "L" else 1
    attended_name = stim_names[attended_idx]
    unattended_name = stim_names[1 - attended_idx]

    audio_attended, audio_fs = librosa.load(stimuli_dir / attended_name, sr=None)
    audio_unattended, _ = librosa.load(stimuli_dir / unattended_name, sr=None)

    result = {
        "eeg": eeg,
        "fs": fs,
        "attended_track": attended_track,
        "attended_ear": attended_ear,
        "trial_id": trial_id,
        "subject": subject,
        "audio_attended_file": attended_name,
        "audio_unattended_file": unattended_name,
    }

    if extract_envelope:
        max_env_samples = round(eeg.shape[1] * target_fs / fs)
        envelope_attended = _extract_envelope(audio_attended, audio_fs, target_fs)[:max_env_samples]
        envelope_unattended = _extract_envelope(audio_unattended, audio_fs, target_fs)[:max_env_samples]

        if apply_trigger_shift:
            # trigger_alignment_shift is in native-EEG-rate (fs) sample units;
            # rescale for the envelopes, which may be at a different rate.
            shift_env = round(trigger_alignment_shift * target_fs / fs)
            envelope_attended = envelope_attended[shift_env:]
            envelope_unattended = envelope_unattended[shift_env:]
            eeg = eeg[:, trigger_alignment_shift:]

        result["eeg"] = eeg
        result["envelope_attended"] = envelope_attended
        result["envelope_unattended"] = envelope_unattended
        result["envelope_fs"] = target_fs
    else:
        result["audio_attended"] = audio_attended
        result["audio_unattended"] = audio_unattended
        result["audio_fs"] = int(audio_fs)

    return result


def load_subject_trials(filepath: Union[str, Path], stimuli_dir: Union[str, Path],
                        extract_envelope: bool = True) -> list:
    """
    Load every trial of one subject via `load_trial`, returning a list of dicts.

    Named to avoid colliding with `load_subject`, which returns a continuous
    MNE Raw for the filtering/ICA pipeline instead.

    Parameters
    ----------
    extract_envelope : passed through to `load_trial`. Set False when only
        EEG is needed (e.g. ICA fitting) -- envelope extraction (bandpass +
        Hilbert + resample on both audio files, per trial) is one of the
        slower parts of loading a subject and is otherwise wasted work.
    """
    with h5py.File(filepath, "r") as f:
        n_trials = f["trials"].shape[0]

    return [load_trial(filepath, i, stimuli_dir, extract_envelope=extract_envelope)
            for i in range(n_trials)]


if __name__ == "__main__":
    RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
    trial = load_trial(RAW_DIR / "S1.mat", 0, RAW_DIR / "stimuli")

    print(f"subject:          {trial['subject']}")
    print(f"trial_id:         {trial['trial_id']}")
    print(f"attended_ear:     {trial['attended_ear']}")
    print(f"attended_track:   {trial['attended_track']}")
    print(f"eeg:                 shape={trial['eeg'].shape}, fs={trial['fs']} Hz")
    print(f"envelope_attended:   shape={trial['envelope_attended'].shape}, "
          f"fs={trial['envelope_fs']} Hz  <- {trial['audio_attended_file']}")
    print(f"envelope_unattended: shape={trial['envelope_unattended'].shape}, "
          f"fs={trial['envelope_fs']} Hz  <- {trial['audio_unattended_file']}")
