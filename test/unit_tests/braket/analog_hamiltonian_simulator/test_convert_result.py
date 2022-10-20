import numpy as np
import pytest
from braket.task_result.analog_hamiltonian_simulation_task_result_v1 import (
    AnalogHamiltonianSimulationShotMeasurement,
    AnalogHamiltonianSimulationShotMetadata,
    AnalogHamiltonianSimulationShotResult,
    AnalogHamiltonianSimulationTaskResult,
)
from braket.task_result.task_metadata_v1 import TaskMetadata

from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_helpers import sample_state
from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_result_converter import (
    convert_result,
)

configurations_1 = ["gg", "gr", "rg", "rr"]

shots = 10000000

mock_taskMetadata = TaskMetadata(
    id="12345",
    shots=shots,
    deviceId="rydbergLocalSimulator",
)

mock_dist = [100, 200, 300, 400]
mock_preSequence = [1, 1]


@pytest.mark.parametrize("para", [[mock_dist, shots]])
def test_sample_state(para):
    dist, shots = para[0], para[1]
    state = [np.sqrt(item / sum(dist)) for item in dist]
    result = sample_state(state, shots) / shots

    probs = (np.abs(state) ** 2).flatten()

    assert np.isclose(result, probs, atol=1e-3).all()
    # Note on the 1e-3 probability difference threshold
    # Because shots is very large (10_000_000), result[0] * shots is
    # approximately Poisson distributed with lambda = mock_dist[0] * shots = 1e6.
    # The standard deviation is std(result[0]) = sqrt(lambda) / 1e7 = 1e-4
    # a deviation of 1e-3 from the expected result[0] would constitute a 10-sigma event,
    # which happen with probability 7e-24.
    # We are not expecting this test to fail because of statistical fluctuation.


@pytest.mark.parametrize(
    "dist, preSequence, configurations, taskMetadata", [
        (mock_dist, mock_preSequence, configurations_1, mock_taskMetadata),
    ]
)
def test_convert_result(dist, preSequence, configurations, taskMetadata):
    result = convert_result(dist, preSequence, configurations, taskMetadata)

    def get_meas(postSequence, num):
        return [
            AnalogHamiltonianSimulationShotMeasurement(
                shotMetadata=AnalogHamiltonianSimulationShotMetadata(shotStatus="Success"),
                shotResult=AnalogHamiltonianSimulationShotResult(
                    preSequence=preSequence, postSequence=postSequence
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

@pytest.mark.parametrize(
    "dist, preSequence, configurations, taskMetadata, expected_postSequence", [
        ([1, 0, 0, 0], [1, 1, 0], ["gg", "gr", "rg", "rr"], mock_taskMetadata, [1, 1, 0]),
        ([0, 1, 0, 0], [1, 1, 0], ["gg", "gr", "rg", "rr"], mock_taskMetadata, [1, 0, 0]),
        ([0, 0, 1, 0], [1, 1, 0], ["gg", "gr", "rg", "rr"], mock_taskMetadata, [0, 1, 0]),
        ([0, 0, 0, 1], [1, 1, 0], ["gg", "gr", "rg", "rr"], mock_taskMetadata, [0, 0, 0]),
        
        ([1, 0, 0, 0], [1, 0, 0, 1], ["gg", "gr", "rg", "rr"], mock_taskMetadata, [1, 0, 0, 0]),
        ([0, 1, 0, 0], [1, 0, 0, 1], ["gg", "gr", "rg", "rr"], mock_taskMetadata, [1, 0, 0, 0]),
        ([0, 0, 1, 0], [1, 0, 0, 1], ["gg", "gr", "rg", "rr"], mock_taskMetadata, [0, 0, 0, 1]),
        ([0, 0, 0, 1], [1, 0, 0, 1], ["gg", "gr", "rg", "rr"], mock_taskMetadata, [0, 0, 0, 0]),

        ([1, 0], [0, 0, 0, 1], ["g", "r"], mock_taskMetadata, [0, 0, 0, 1]),
        ([0, 1], [0, 0, 0, 1], ["g", "r"], mock_taskMetadata, [0, 0, 0, 0]),
    ]
)
def test_convert_result_with_empty_sites(dist, preSequence, configurations, taskMetadata, expected_postSequence):
    result = convert_result(dist, preSequence, configurations, taskMetadata)
    postSequence = result.measurements[0].shotResult.postSequence
    assert result.measurements[0].shotResult.postSequence == expected_postSequence
