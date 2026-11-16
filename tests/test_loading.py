from loading import load_trial


def test_load_trial_envelope_defaults_to_64hz(synthetic_mat_file):
    mat_path, stimuli_dir, params = synthetic_mat_file

    trial = load_trial(mat_path, 0, stimuli_dir)

    expected_len = round(params["n_eeg_samples"] * 64.0 / params["fs"])
    assert trial["envelope_fs"] == 64.0
    assert trial["envelope_attended"].shape[0] == expected_len
    assert trial["envelope_unattended"].shape[0] == expected_len


def test_load_trial_custom_target_fs(synthetic_mat_file):
    mat_path, stimuli_dir, params = synthetic_mat_file

    trial = load_trial(mat_path, 0, stimuli_dir, target_fs=128.0)

    assert trial["envelope_fs"] == 128.0
    assert trial["envelope_attended"].shape[0] == params["n_eeg_samples"]


def test_load_trial_no_shift_by_default(synthetic_mat_file):
    mat_path, stimuli_dir, params = synthetic_mat_file

    trial = load_trial(mat_path, 0, stimuli_dir)

    # Regression test: load_trial must not silently trim eeg by default.
    assert trial["eeg"].shape[1] == params["n_eeg_samples"]


def test_load_trial_apply_trigger_shift_scales_correctly(synthetic_mat_file):
    mat_path, stimuli_dir, params = synthetic_mat_file
    target_fs = 64.0

    without_shift = load_trial(mat_path, 0, stimuli_dir, target_fs=target_fs,
                                apply_trigger_shift=False)
    with_shift = load_trial(mat_path, 0, stimuli_dir, target_fs=target_fs,
                             apply_trigger_shift=True)

    assert with_shift["eeg"].shape[1] == without_shift["eeg"].shape[1] - params["trigger_shift"]

    expected_env_shift = round(params["trigger_shift"] * target_fs / params["fs"])
    assert (with_shift["envelope_attended"].shape[0]
            == without_shift["envelope_attended"].shape[0] - expected_env_shift)


def test_load_trial_attended_resolved_by_ear(synthetic_mat_file):
    mat_path, stimuli_dir, _ = synthetic_mat_file

    trial0 = load_trial(mat_path, 0, stimuli_dir)  # attended_ear="R", attended_track=1
    trial1 = load_trial(mat_path, 1, stimuli_dir)  # attended_ear="L", attended_track=2

    assert trial0["attended_ear"] == "R"
    assert trial0["audio_attended_file"] == "stim_b.wav"  # stimuli[1] = right ear
    assert trial1["attended_ear"] == "L"
    assert trial1["audio_attended_file"] == "stim_a.wav"  # stimuli[0] = left ear


def test_load_trial_extract_envelope_false(synthetic_mat_file):
    mat_path, stimuli_dir, params = synthetic_mat_file

    trial = load_trial(mat_path, 0, stimuli_dir, extract_envelope=False)

    assert "envelope_attended" not in trial
    assert trial["audio_fs"] == params["audio_fs"]
    assert trial["eeg"].shape[1] == params["n_eeg_samples"]
