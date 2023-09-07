import itertools
from typing import List, Tuple

import numpy as np
from braket.task_result.analog_hamiltonian_simulation_task_result_v1 import (
    AnalogHamiltonianSimulationShotMeasurement,
    AnalogHamiltonianSimulationShotMetadata,
    AnalogHamiltonianSimulationShotResult,
    AnalogHamiltonianSimulationTaskResult,
)
from braket.task_result.task_metadata_v1 import TaskMetadata


# TODO: consider creating a class for result post-processing and sampling
def convert_result(
    shot_results: Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[int],
        List[int],
    ],
    task_Metadata: TaskMetadata,
) -> AnalogHamiltonianSimulationTaskResult:
    """Convert sampled noisy pre- and post-sequence distributions
    to the analog simulation result schema

    Args:
        shot_results (Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            List[int],
            List[int],
            ]): A tuple containing the sampled pre-sequences and vacant pose-sequences,
            state frequencies and index of filled sites and empty sites.
        task_Metadata (TaskMetadata): The metadata for the task

    Returns:
        AnalogHamiltonianSimulationTaskResult: Results from sampling distributions
    """
    measurements = []

    (
        pre_sequences,
        post_sequences_empty,
        state_freq,
        non_empty_sites,
        empty_sites,
    ) = shot_results

    num_qubits = len(non_empty_sites)
    full_configurations = [
        "".join(item) for item in itertools.product(["g", "r"], repeat=num_qubits)
    ]

    num_sites = len(non_empty_sites) + len(empty_sites)
    # the 0th axis of pre_sequences and post_sequences_empty are of shape
    # either 1 (noiseless) or number_of_shots (noisy)
    pre_iter = itertools.cycle(pre_sequences)
    post_iter = itertools.cycle(post_sequences_empty)
    for configuration, count in zip(full_configurations, state_freq):
        post_sequence = np.zeros(num_sites)
        post_sequence[non_empty_sites] = np.array(list(configuration)) == "g"

        for _ in range(count):
            # pre-sequence for this shot, which is the requested filling (noiseless)
            # or the next item in the sampled list (noisy)
            shot_pre_sequence = next(pre_iter).tolist()

            # post-sequence for the sites that are requested to be empty
            shot_empty_sites_sequence = next(post_iter)
            post_sequence[empty_sites] = shot_empty_sites_sequence

            shot_measurement = AnalogHamiltonianSimulationShotMeasurement(
                shotMetadata=AnalogHamiltonianSimulationShotMetadata(shotStatus="Success"),
                shotResult=AnalogHamiltonianSimulationShotResult(
                    preSequence=shot_pre_sequence, postSequence=post_sequence.tolist()
                ),
            )
            measurements.append(shot_measurement)

    return AnalogHamiltonianSimulationTaskResult(
        taskMetadata=task_Metadata, measurements=measurements
    )
