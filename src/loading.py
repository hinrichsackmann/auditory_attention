"""Inspect KU Leuven AAD .mat files (MATLAB v7.3 / HDF5 format).

scipy.io.loadmat cannot read these files -- they're saved as MATLAB 7.3,
which stores data as HDF5. h5py is used instead, and MATLAB cell arrays /
struct-array fields are represented as HDF5 object references into the
file's shared '#refs#' group, which this module dereferences recursively.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import h5py
import mne
import numpy as np

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


if __name__ == "__main__":
    DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
    raw = load_subject(1, DATA_DIR)
    print(raw.info)
