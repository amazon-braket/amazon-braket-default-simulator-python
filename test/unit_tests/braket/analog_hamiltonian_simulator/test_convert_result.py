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
    apply_SPAM_noises,
    sample_result,
)
from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_result_converter import (
    convert_result,
)

configurations_0 = ["g", "r"]
configurations_1 = ["gg", "gr", "rg", "rr"]
configurations_2 = ["ggg", "ggr", "grg", "grr", "rgg", "rgr", "rrg", "rrr"]
configurations_2_blockade = ["ggg", "ggr", "grg", "rgg", "rgr"]


shots = 10000000

mock_taskMetadata = TaskMetadata(
    id="12345",
    shots=shots,
    deviceId="rydbergLocalSimulator",
)

mock_dist_0 = [100, 200]
mock_dist_1 = [100, 200, 300, 400]
mock_dist_2 = [100, 200, 300, 400, 500, 600, 700, 800]
mock_dist_2_blockade = [100, 200, 300, 400, 500]


@pytest.mark.xfail
def test_post_processing_fail():
    apply_SPAM_noises([1, 1], mock_dist_1, configurations_1)


@pytest.mark.parametrize(
    "pre_sequence, dist, configurations, full_dist",
    [
        ([1, 1], mock_dist_1, configurations_1, mock_dist_1),
        ([0, 1], mock_dist_0, configurations_0, mock_dist_0),
        ([1, 0], mock_dist_0, configurations_0, mock_dist_0),
        ([1, 1, 1], mock_dist_2, configurations_2, mock_dist_2),
        (
            [1, 1, 1],
            mock_dist_2_blockade,
            configurations_2_blockade,
            [100, 200, 300, 0, 400, 500, 0, 0],
        ),
    ],
)
def test_post_processing_noiseless(pre_sequence, dist, configurations, full_dist):
    dist /= sum(np.array(dist))
    full_dist /= sum(np.array(full_dist))
    (pre_seq_dist, post_seq_dist, non_empty_sites, empty_sites, eps_vd) = apply_SPAM_noises(
        pre_sequence, dist, configurations
    )
    assert all([prob == occupancy for prob, occupancy in zip(pre_seq_dist, pre_sequence)])
    assert np.allclose(post_seq_dist, full_dist)
    assert all([pre_sequence[index] == 1 for index in non_empty_sites])
    assert all([pre_sequence[index] == 0 for index in empty_sites])
    assert eps_vd == 0


@pytest.mark.parametrize(
    "pre_seq_dist, post_seq_dist, non_empty_sites, empty_sites, eps_vd, shots",
    [
        (np.array([1, 0, 0, 1]), mock_dist_1, [0, 3], [1, 2], 0, shots),
        (np.array([1, 0, 1, 1]), mock_dist_2, [0, 2, 3], [1], 0, shots),
        (np.array([1, 1, 1]), mock_dist_2, [0, 1, 2], [], 0, shots),
    ],
)
def test_sample_result_noiseless(
    pre_seq_dist,
    post_seq_dist,
    non_empty_sites,
    empty_sites,
    eps_vd,
    shots,
):
    post_seq_dist /= sum(np.array(post_seq_dist))
    (
        pre_sequences,
        post_sequences_empty,
        state_freq,
        non_empty_sites,
        empty_sites,
    ) = sample_result(
        (pre_seq_dist, post_seq_dist, non_empty_sites, empty_sites, eps_vd), shots, {}
    )
    assert pre_sequences.shape[0] == 1
    assert post_sequences_empty.shape[0] == 1
    assert sum(state_freq) == shots

    state_density = state_freq / shots
    assert np.isclose(state_density, post_seq_dist, atol=1e-3).all()
    # Note on the 1e-3 probability difference threshold
    # Because shots is very large (10_000_000), result[0] * shots is
    # approximately Poisson distributed with lambda = mock_dist[0] * shots = 1e6.
    # The standard deviation is std(result[0]) = sqrt(lambda) / 1e7 = 1e-4
    # a deviation of 1e-3 from the expected result[0] would constitute a 10-sigma event,
    # which happen with probability 7e-24.
    # We are not expecting this test to fail because of statistical fluctuation.


@pytest.mark.parametrize(
    "pre_sequences, post_sequences_empty, state_freq, non_empty_sites, empty_sites, taskMetadata",
    [
        (
            np.array([1, 1]).reshape((1, -1)),
            np.array([]).reshape((1, -1)),
            mock_dist_1,
            [0, 1],
            [],
            mock_taskMetadata,
        ),
    ],
)
def test_convert_result(
    pre_sequences, post_sequences_empty, state_freq, non_empty_sites, empty_sites, taskMetadata
):
    result = convert_result(
        (
            pre_sequences,
            post_sequences_empty,
            state_freq,
            non_empty_sites,
            empty_sites,
        ),
        taskMetadata,
    )

    def get_meas(postSequence, num):
        return [
            AnalogHamiltonianSimulationShotMeasurement(
                shotMetadata=AnalogHamiltonianSimulationShotMetadata(shotStatus="Success"),
                shotResult=AnalogHamiltonianSimulationShotResult(
                    preSequence=pre_sequences[0].tolist(), postSequence=postSequence
                ),
            )
            for _ in range(num)
        ]

    measurements = (
        get_meas([1, 1], 100)
        + get_meas([1, 0], 200)
        + get_meas([0, 1], 300)
        + get_meas([0, 0], 400)
    )

    trueresult = AnalogHamiltonianSimulationTaskResult(
        taskMetadata=taskMetadata, measurements=measurements
    )

    assert pytest.approx(result) == pytest.approx(trueresult)
