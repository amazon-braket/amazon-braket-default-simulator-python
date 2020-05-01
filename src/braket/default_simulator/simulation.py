# Copyright 2019-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

from typing import List

import numpy as np
from braket.default_simulator.operation import GateOperation, Observable, Operation
from braket.default_simulator.simulation_strategies import operation_batch_strategy, single_operation_strategy


class StateVectorSimulation:
    """
    This class encapsulates a simulation of a quantum system of `qubit_count` qubits.
    The evolution of the state of the quantum system is achieved through `GateOperation`s
    using the `evolve()` method.

    The simulation defaults to applying gates one at a time, but can also be configured
    to apply multiple gates at once by supplying the `batch_size` keyword argument.

    When `batch_size` is supplied, the operation list is split into contiguous partitions
    of length `batch_size` (with remainder) to contract. Within each partition, contraction
    order is optimized among the gates, and the partitions themselves are contracted in the order
    they appear. Larger partitions can be significantly faster, although this is not guaranteed,
    but will use more memory.
    """

    def __init__(self, qubit_count: int, shots: int, batch_size: int):
        r"""
        Args:
            qubit_count (int): The number of qubits being simulated.
                All the qubits start in the :math:`\ket{\mathbf{0}}` computational basis state.
            shots (int): The number of samples to take from the simulation.
                If set to 0, only results that do not require sampling, such as state vector
                or expectation, will be generated.
            batch_size (int): The size of the partitions to contract; if set to 1,
                the gates are applied one at a time, without any optimization of
                contraction order. Must be a positive integer.
        """
        if not isinstance(batch_size, int):
            raise TypeError(f"batch_size must be of type `int`, but {type(batch_size)} provided")
        if batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, but {batch_size} provided")

        initial_state = np.zeros(2 ** qubit_count, dtype=complex)
        initial_state[0] = 1
        self._state_vector = initial_state
        self._qubit_count = qubit_count
        self._shots = shots
        self._batch_size = batch_size
        self._post_observables = None

    def evolve(self, operations: List[GateOperation]) -> None:
        """ Evolves the state of the simulation under the action of
        the specified gate operations.

        Args:
            operations (List[GateOperation]): Gate operations to apply for
                evolving the state of the simulation.

        Note:
            This method mutates the state of the simulation.
        """
        self._state_vector = StateVectorSimulation._apply_operations(
            self._state_vector, self._qubit_count, operations, self._batch_size
        )

    def apply_observables(self, observables: List[Observable]) -> None:
        """ Applies the diagonalizing matrices of the given observables
        to the state of the simulation.

        This method can only be called once.

        Args:
            observables (List[Observable]): The observables to apply

        Raises:
            RuntimeError: If this method is called more than once
        """
        if self._post_observables is not None:
            raise RuntimeError("Observables have already been applied.")
        self._post_observables = StateVectorSimulation._apply_operations(
            self._state_vector, self._qubit_count, observables, self._batch_size
        )

    @staticmethod
    def _apply_operations(
        state: np.ndarray, qubit_count: int, operations: List[Operation], batch_size: int
    ) -> np.ndarray:
        state_tensor = np.reshape(state, [2] * qubit_count)
        final = (
            single_operation_strategy.apply_operations(state_tensor, qubit_count, operations)
            if batch_size == 1
            else operation_batch_strategy.apply_operations(
                state_tensor, qubit_count, operations, batch_size
            )
        )
        return np.reshape(final, 2 ** qubit_count)

    def retrieve_samples(self) -> List[int]:
        """ Retrieves samples of states from the state vector of the simulation,
        based on the probabilities.

        Returns:
            List[int]: List of states sampled according to their probabilities
            in the state vector. Each integer represents the decimal encoding of the
            corresponding computational basis state.
        """
        return np.random.choice(len(self._state_vector), p=self.probabilities, size=self._shots)

    @property
    def state_vector(self) -> np.ndarray:
        """
        np.ndarray: The state vector specifying the current state of the simulation.
        """
        return self._state_vector

    @property
    def state_with_observables(self) -> np.ndarray:
        """
        np.ndarray: The final state vector of the simulation after application of observables.

        Raises:
            RuntimeError: If observables have not been applied
        """
        if self._post_observables is None:
            raise RuntimeError("No observables applied")
        return self._post_observables

    @property
    def qubit_count(self) -> int:
        """int: The number of qubits being simulated by the simulation."""
        return self._qubit_count

    @property
    def shots(self) -> int:
        """
        int: The number of samples to take from the simulation.

        0 means no samples are taken, and results that require sampling
        to calculate cannot be returned.
        """
        return self._shots

    @property
    def probabilities(self) -> np.ndarray:
        """np.ndarray: The probabilities of each computational basis state."""
        return np.abs(self._state_vector) ** 2
