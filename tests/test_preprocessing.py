import mne
import numpy as np

from preprocessing import (align_trial_lengths, checkpoint_path, filter_raw,
                            get_trial_segments, load_cleaned_raw, make_raw, run_subject)


def _tiny_raw(n_channels=4, n_samples=2000, sfreq=128.0, seed=0):
    rng = np.random.default_rng(seed)
    info = mne.create_info([f"ch{i}" for i in range(n_channels)], sfreq, "eeg")
    return mne.io.RawArray(rng.standard_normal((n_channels, n_samples)) * 1e-6, info, verbose=False)


def test_filter_raw_per_trial_matches_isolated_filtering():
    """Filtering two trials separately then concatenating must be identical to
    filtering each trial alone -- this is the regression test for the
    trial-boundary leakage bug (concatenate-then-filter bled filter transients
    across the join; filter-then-concatenate must not)."""
    trial_a = _tiny_raw(seed=1)
    trial_b = _tiny_raw(seed=2)

    filtered_a = filter_raw(trial_a, "ica_prep")
    filtered_b = filter_raw(trial_b, "ica_prep")
    concatenated = mne.concatenate_raws([filtered_a.copy(), filtered_b.copy()])

    isolated_b = filter_raw(trial_b, "ica_prep")

    b_segment = concatenated.get_data()[:, filtered_a.n_times:]
    assert np.allclose(b_segment, isolated_b.get_data())


def test_filter_raw_analysis_resamples_to_64hz():
    raw = _tiny_raw(sfreq=128.0)

    assert filter_raw(raw, "analysis").info["sfreq"] == 64.0
    assert filter_raw(raw, "ica_prep").info["sfreq"] == 128.0


def test_make_raw_sets_trial_annotation(synthetic_trial_dict):
    trial = synthetic_trial_dict(trial_id=5)

    raw = make_raw(trial)

    assert list(raw.annotations.description) == ["trial_05"]
    assert raw.annotations.onset[0] == 0.0
    assert np.isclose(raw.annotations.duration[0], raw.times[-1] + 1 / raw.info["sfreq"])


def test_concatenate_and_get_trial_segments_roundtrip(synthetic_trial_dict):
    trial1 = synthetic_trial_dict(trial_id=1, n_samples=300)
    trial2 = synthetic_trial_dict(trial_id=2, n_samples=500, seed=1)

    raw = mne.concatenate_raws([make_raw(trial1), make_raw(trial2)])

    # mne.concatenate_raws also inserts "BAD boundary"/"EDGE boundary"
    # annotations at the join -- get_trial_segments must ignore those.
    trial_labels = [d for d in raw.annotations.description if d.startswith("trial_")]
    assert trial_labels == ["trial_01", "trial_02"]
    trial_onsets = raw.annotations.onset[[d.startswith("trial_") for d in raw.annotations.description]]
    assert trial_onsets[1] > 0.0  # second trial's onset shifted by concatenation

    segments = get_trial_segments(raw, raw.annotations)
    assert len(segments) == 2
    assert segments[0].n_times == 300
    assert segments[1].n_times == 500


def test_align_trial_lengths_trims_to_common_minimum():
    eeg = np.zeros((64, 100))
    env_attended = np.zeros(97)
    env_unattended = np.zeros(99)

    eeg_t, att_t, unatt_t = align_trial_lengths(eeg, env_attended, env_unattended)

    assert eeg_t.shape == (64, 97)
    assert att_t.shape == (97,)
    assert unatt_t.shape == (97,)


def test_run_subject_smoke(monkeypatch, synthetic_trial_dict, tmp_path):
    trials = [synthetic_trial_dict(trial_id=i, n_samples=1000, seed=i) for i in range(1, 4)]
    monkeypatch.setattr("preprocessing.load_subject_trials", lambda *a, **k: trials)

    processed_dir = tmp_path / "processed"
    ica_dir = tmp_path / "ica"

    path = run_subject(1, "unused.mat", "unused_stimuli", processed_dir, ica_dir)

    assert path == checkpoint_path(1, processed_dir)
    assert path.exists()
    assert (ica_dir / "sub01-ica.fif").exists()

    cleaned = load_cleaned_raw(1, processed_dir)
    assert cleaned.info["sfreq"] == 64.0
    assert len(get_trial_segments(cleaned, cleaned.annotations)) == 3
