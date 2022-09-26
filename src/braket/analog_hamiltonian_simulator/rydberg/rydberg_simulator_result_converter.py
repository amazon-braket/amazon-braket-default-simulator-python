from typing import List

import numpy as np
from braket.task_result.analog_hamiltonian_simulation_task_result_v1 import (
    AnalogHamiltonianSimulationShotMeasurement,
    AnalogHamiltonianSimulationShotMetadata,
    AnalogHamiltonianSimulationShotResult,
    AnalogHamiltonianSimulationTaskResult,
)
from braket.task_result.task_metadata_v1 import TaskMetadata


def convert_result(
    dist: np.ndarray,
    pre_sequence: List[int],
    configurations: List[str],
    task_Metadata: TaskMetadata,
) -> AnalogHamiltonianSimulationTaskResult:
    """Convert a given sampled distribution to the analog simulation result schema

    Args:
        dist (np.ndarray): The sample results to convert
        pre_sequence (List[int]): the same pre-sequence measurement results used for all shots
        configuraiton (List[str]): The list of configurations that comply with the blockade
            approximation.
        task_Metadata (TaskMetadata): The metadata for the task

    Returns:
        AnalogHamiltonianSimulationTaskResult: Results from sampling a distribution
    """
    measurements = []

    for key, val in zip(configurations, dist):
        post_sequence = list(key)
        post_sequence = [0 if item == "r" else 1 for item in post_sequence]

        shot_measurement = AnalogHamiltonianSimulationShotMeasurement(
            shotMetadata=AnalogHamiltonianSimulationShotMetadata(shotStatus="Success"),
            shotResult=AnalogHamiltonianSimulationShotResult(
                preSequence=pre_sequence, postSequence=post_sequence
            ),
        )
        measurements += [shot_measurement for _ in range(val)]

    return AnalogHamiltonianSimulationTaskResult(
        taskMetadata=task_Metadata, measurements=measurements
    )
