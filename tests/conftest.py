"""Shared fixtures: synthetic data only -- real .mat files are never committed
and are too large (~300 MB each) for fast tests."""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.io import wavfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _encode_str(s: str) -> np.ndarray:
    """Encode a string as the uint16-codepoint column vector MATLAB char arrays use."""
    return np.array([[ord(c)] for c in s], dtype=np.uint16)


def _write_trial_group(parent: h5py.Group, name: str, *, trial_id: int, subject: str,
                        attended_ear: str, attended_track: int, fs: float,
                        n_eeg_samples: int, trigger_shift: int,
                        stim_names: tuple) -> h5py.Group:
    """Build one trial's HDF5 struct, matching what `loading.load_trial` reads."""
    trial = parent.create_group(name)

    raw_data = trial.create_group("RawData")
    rng = np.random.default_rng(trial_id)
    raw_data.create_dataset("EegData", data=rng.standard_normal((64, n_eeg_samples)))
    raw_data.create_dataset("TriggerAlignmentShift", data=np.array([[float(trigger_shift)]]))

    file_header = trial.create_group("FileHeader")
    file_header.create_dataset("SampleRate", data=np.array([[fs]]))

    trial.create_dataset("attended_track", data=np.array([[float(attended_track)]]))
    trial.create_dataset("attended_ear", data=_encode_str(attended_ear))
    trial.create_dataset("TrialID", data=np.array([[float(trial_id)]]))
    trial.create_dataset("subject", data=_encode_str(subject))

    stim0 = trial.create_dataset(f"{name}_stim0", data=_encode_str(stim_names[0]))
    stim1 = trial.create_dataset(f"{name}_stim1", data=_encode_str(stim_names[1]))
    ref_dtype = h5py.special_dtype(ref=h5py.Reference)
    stimuli = trial.create_dataset("stimuli", shape=(1, 2), dtype=ref_dtype)
    stimuli[0, 0] = stim0.ref
    stimuli[0, 1] = stim1.ref

    return trial


@pytest.fixture
def synthetic_mat_file(tmp_path):
    """
    Write a tiny, fully-synthetic 2-trial .mat file (MATLAB v7.3/HDF5 layout)
    plus matching .wav stimulus files, mirroring exactly what `load_trial`
    reads: trials[i,0] -> struct with RawData.{EegData,TriggerAlignmentShift},
    FileHeader.SampleRate, attended_track, attended_ear, TrialID, subject,
    stimuli[0,0]/[0,1].

    Returns (mat_path, stimuli_dir, params) where params has the known ground
    truth (fs, n_eeg_samples, trigger_shift, audio_fs) tests assert against.
    """
    fs = 128.0
    n_eeg_samples = 256  # 2.0 s
    trigger_shift = 10
    audio_fs = 8000
    audio_duration_s = 3.0  # longer than the EEG trial, per the dataset's README

    stimuli_dir = tmp_path / "stimuli"
    stimuli_dir.mkdir()

    stim_names = ("stim_a.wav", "stim_b.wav")
    rng = np.random.default_rng(0)
    n_audio_samples = int(audio_duration_s * audio_fs)
    for name in stim_names:
        audio = rng.standard_normal(n_audio_samples).astype(np.float32) * 0.1
        wavfile.write(stimuli_dir / name, audio_fs, audio)

    mat_path = tmp_path / "S_test.mat"
    with h5py.File(mat_path, "w") as f:
        trial_refs = []
        trial_specs = [
            dict(trial_id=1, attended_ear="R", attended_track=1),
            dict(trial_id=2, attended_ear="L", attended_track=2),
        ]
        for i, spec in enumerate(trial_specs):
            trial_grp = _write_trial_group(
                f, f"trial{i}", subject="S_test", fs=fs,
                n_eeg_samples=n_eeg_samples, trigger_shift=trigger_shift,
                stim_names=stim_names, **spec,
            )
            trial_refs.append(trial_grp.ref)

        ref_dtype = h5py.special_dtype(ref=h5py.Reference)
        trials_ds = f.create_dataset("trials", shape=(len(trial_refs), 1), dtype=ref_dtype)
        for i, ref in enumerate(trial_refs):
            trials_ds[i, 0] = ref

    params = dict(fs=fs, n_eeg_samples=n_eeg_samples, trigger_shift=trigger_shift,
                  audio_fs=audio_fs)
    return mat_path, stimuli_dir, params


@pytest.fixture
def synthetic_trial_dict():
    """A single trial dict as `loading.load_trial` would return it (minimal keys
    `make_raw` needs: eeg, fs, trial_id), for tests that don't need real HDF5/audio."""
    def _make(trial_id=1, n_samples=256, fs=128.0, seed=0):
        rng = np.random.default_rng(seed)
        return {
            "eeg": rng.standard_normal((64, n_samples)),
            "fs": fs,
            "trial_id": trial_id,
            "subject": "S_test",
        }
    return _make
