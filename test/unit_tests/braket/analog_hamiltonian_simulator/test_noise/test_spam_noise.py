import numpy as np
import pytest
from braket.task_result.analog_hamiltonian_simulation_task_result_v1 import (
    AnalogHamiltonianSimulationShotMeasurement,
    AnalogHamiltonianSimulationShotMetadata,
    AnalogHamiltonianSimulationShotResult,
    AnalogHamiltonianSimulationTaskResult,
)
from braket.task_result.task_metadata_v1 import TaskMetadata

from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_helpers import (
    _find_configuration_index,
    apply_SPAM_noises,
    noise_type,
    sample_result,
)
from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_result_converter import (
    convert_result,
)

configurations_0 = ["g", "r"]
configurations_1 = ["gg", "gr", "rg", "rr"]
configurations_2 = ["ggg", "ggr", "grg", "grr", "rgg", "rgr", "rrg", "rrr"]
configurations_2_blockade = ["ggg", "ggr", "grg", "rgg", "rgr"]


shots = int(1e7)

mock_dist_0 = [100, 200]
mock_dist_1 = [100, 200, 300, 400]
mock_dist_2 = [100, 200, 300, 400, 500, 600, 700, 800]
mock_dist_2_blockade = [100, 200, 300, 400, 500]

noise_ad = {noise_type.ATOM_DETECTION: 0.1}
noise_vd = {noise_type.VACANCY_DETECTION: 0.2}
noise_ad_vd = {noise_type.ATOM_DETECTION: 0.15, noise_type.VACANCY_DETECTION: 0.25}
noise_all = {
    noise_type.ATOM_DETECTION: 0.21,
    noise_type.VACANCY_DETECTION: 0.23,
    noise_type.GROUND_STATE_DETECTION: 0.25,
    noise_type.RYDBERG_STATE_DETECTION: 0.27,
}

mock_taskMetadata = TaskMetadata(
    id="12345",
    shots=shots,
    deviceId="rydbergLocalSimulator",
)


@pytest.mark.parametrize(
    "pre_sequence, dist, configurations, noises, true_pre, full_dist",
    [
        ([1, 1], mock_dist_1, configurations_1, noise_ad, [0.9, 0.9], mock_dist_1),
        ([0, 1], mock_dist_0, configurations_0, noise_ad_vd, [0.25, 0.85], mock_dist_0),
        (
            [1, 1, 1],
            mock_dist_2_blockade,
            configurations_2_blockade,
            noise_all,
            [0.79, 0.79, 0.79],
            [100, 200, 300, 0, 400, 500, 0, 0],
        ),
    ],
)
def test_post_processing_noisy(pre_sequence, dist, configurations, noises, true_pre, full_dist):
    dist /= sum(np.array(dist))
    full_dist /= sum(np.array(full_dist))
    (pre_seq_dist, post_seq_dist, non_empty_sites, empty_sites, eps_vd) = apply_SPAM_noises(
        pre_sequence, dist, configurations, noises
    )

    assert np.allclose(pre_seq_dist, true_pre, atol=1e-3)
    assert np.isclose(eps_vd, noises.get(noise_type.VACANCY_DETECTION, 0))

    eps_g = noises.get(noise_type.GROUND_STATE_DETECTION, 0)
    eps_r = noises.get(noise_type.RYDBERG_STATE_DETECTION, 0)

    if full_dist.shape[0] == 2:
        probs = np.zeros(2)
        probs[0] = full_dist[0] * (1 - eps_g) + full_dist[1] * eps_r
        probs[1] = full_dist[0] * eps_g + full_dist[1] * (1 - eps_r)
    else:
        confusion_mtx_qubit = np.array([[1 - eps_g, eps_r], [eps_g, 1 - eps_r]])
        confusion_mtx_full = confusion_mtx_qubit
        while confusion_mtx_full.shape[1] < full_dist.shape[0]:
            confusion_mtx_full = np.kron(confusion_mtx_full, confusion_mtx_qubit)
        probs = confusion_mtx_full @ full_dist

    assert np.allclose(post_seq_dist, probs, atol=1e-3)


@pytest.mark.parametrize(
    "pre_seq_dist, post_seq_dist, non_empty_sites, empty_sites, eps_vd, shots, noises",
    [
        (np.array([0.79, 0.23, 0.23, 0.79]), mock_dist_1, [0, 3], [1, 2], 0, shots, noise_all),
        (np.array([0.79, 0.23, 0.79, 0.79]), mock_dist_2, [0, 2, 3], [1], 0, shots, noise_all),
        (np.array([0.79, 0.79, 0.79]), mock_dist_2, [0, 1, 2], [], 0, shots, noise_all),
    ],
)
def test_sample_result_noiseless(
    pre_seq_dist,
    post_seq_dist,
    non_empty_sites,
    empty_sites,
    eps_vd,
    shots,
    noises,
):
    post_seq_dist /= sum(np.array(post_seq_dist))
    (
        pre_sequences,
        post_sequences_empty,
        state_freq,
        non_empty_sites,
        empty_sites,
    ) = sample_result(
        (pre_seq_dist, post_seq_dist, non_empty_sites, empty_sites, eps_vd), shots, noises
    )
    assert pre_sequences.shape[0] == shots
    assert post_sequences_empty.shape[0] == shots

    pre_sequences_density = np.sum(pre_sequences, axis=0) / shots
    assert np.allclose(pre_sequences_density, pre_seq_dist, atol=1e-3)
    # Note on the 1e-3 probability difference threshold
    # Because shots is very large (10_000_000), result[0] * shots is
    # approximately Poisson distributed with lambda = mock_dist[0] * shots = 1e6.
    # The standard deviation is std(result[0]) = sqrt(lambda) / 1e7 = 1e-4
    # a deviation of 1e-3 from the expected result[0] would constitute a 10-sigma event,
    # which happen with probability 7e-24.
    # We are not expecting this test to fail because of statistical fluctuation.


@pytest.mark.parametrize(
    "state_freq, non_empty_sites, empty_sites, taskMetadata",
    [
        (mock_dist_1, [0, 2], [1], mock_taskMetadata),
    ],
)
def test_convert_result_noisy(state_freq, non_empty_sites, empty_sites, taskMetadata):
    noisy_pre_sequences = []
    noisy_pre_sequences += [[1, 0, 1] for _ in range(400)]
    noisy_pre_sequences += [[1, 0, 0] for _ in range(300)]
    noisy_pre_sequences += [[0, 0, 1] for _ in range(200)]
    noisy_pre_sequences += [[0, 0, 0] for _ in range(100)]
    # random.shuffle(noisy_pre_sequences)

    noisy_empty_post_sequences = []
    noisy_empty_post_sequences += [[1] for _ in range(300)]
    noisy_empty_post_sequences += [[0] for _ in range(700)]
    # random.shuffle(noisy_empty_post_sequences)

    state_post_sequences = []
    state_post_sequences += [[1, 1] for _ in range(100)]
    state_post_sequences += [[1, 0] for _ in range(200)]
    state_post_sequences += [[0, 1] for _ in range(300)]
    state_post_sequences += [[0, 0] for _ in range(400)]

    result = convert_result(
        (
            np.array(noisy_pre_sequences),
            np.array(noisy_empty_post_sequences),
            state_freq,
            non_empty_sites,
            empty_sites,
        ),
        taskMetadata,
    )

    true_post_sequences = [
        [state[0], empty[0], state[1]]
        for state, empty in zip(state_post_sequences, noisy_empty_post_sequences)
    ]

    measurements = [
        AnalogHamiltonianSimulationShotMeasurement(
            shotMetadata=AnalogHamiltonianSimulationShotMetadata(shotStatus="Success"),
            shotResult=AnalogHamiltonianSimulationShotResult(preSequence=pre, postSequence=post),
        )
        for pre, post in zip(noisy_pre_sequences, true_post_sequences)
    ]

    trueresult = AnalogHamiltonianSimulationTaskResult(
        taskMetadata=taskMetadata, measurements=measurements
    )

    assert pytest.approx(result) == pytest.approx(trueresult)


@pytest.mark.parametrize(
    "configuration, true_idx",
    [
        ("rrgg", 12),
        ("rrrrrggggg", 992),
    ],
)
def test_find_configuration_index(configuration, true_idx):
    assert _find_configuration_index(configuration) == true_idx
