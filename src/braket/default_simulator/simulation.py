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
from braket.default_simulator.operation import GateOperation, Observable


class StateVectorSimulation:
    """
    The class `StateVectorSimulation` encapsulates a simulation of a quantum system of
    `qubit_count` qubits. The evolution of the state of the quantum system is achieved
    through `GateOperation`s using the `evolve()` method.
    """

    def __init__(self, qubit_count: int, shots: int = 0):
        r"""
        Args:
            qubit_count (int): The number of qubits being simulated.
                All the qubits start in the :math:`\ket{\mathbf{0}}` computational basis state.
            shots (int): The number of samples to take from the simulation.
                Defaults to 0, which means only results that do not require sampling,
                such as state vector or expectation, will be generated.
        """
        initial_state = np.zeros(2 ** qubit_count, dtype=complex)
        initial_state[0] = 1
        self._state_vector = initial_state
        self._qubit_count = qubit_count
        self._shots = shots
        self._post_observables = None

    def evolve(self, operations: List[GateOperation]) -> None:
        """ Evolves the state of the simulation under the action of
        the specified gate operations.
        Note: This method mutates the state of the simulation.

        Args:
            operations (List[GateOperation]): Gate operations to apply for
                evolving the state of the simulation.
        """
        self._state_vector = np.reshape(self._state_vector, [2] * self._qubit_count)
        for operation in operations:
            self._state_vector = StateVectorSimulation._apply_operation(
                self._state_vector, self._qubit_count, operation.matrix, operation.targets
            )
        self._state_vector = np.reshape(self._state_vector, 2 ** self._qubit_count)

    def apply_observables(self, observables: List[Observable]) -> None:
        """ Returns the diagonalizing matrices of the given observables
        to the state of the simulation.

        This method can only be called once.

        Args:
            observables (List[Observable]): The observables to apply

        Raises:
            RuntimeError: If this method is called more than once
        """
        if self._post_observables is not None:
            raise RuntimeError("Observables have already been applied.")

        state = np.reshape(self._state_vector, [2] * self._qubit_count)
        for observable in observables:
            # Only add to contraction parameters if the observable
            # has a nontrivial diagonalizing matrix
            if observable.diagonalizing_matrix is not None:
                if observable.targets:
                    state = StateVectorSimulation._apply_operation(
                        state,
                        self._qubit_count,
                        observable.diagonalizing_matrix,
                        observable.targets,
                    )
                else:
                    # There is only one element in `observables`
                    for qubit in range(self._qubit_count):
                        state = StateVectorSimulation._apply_operation(
                            state, self._qubit_count, observable.diagonalizing_matrix, [qubit]
                        )
        self._post_observables = state.reshape(2 ** self._qubit_count)

    @staticmethod
    def _apply_operation(state, qubit_count, matrix, targets) -> np.ndarray:
        """ Updates the given state vector by multiplying the it with
        the given unitary matrix acting on the given targets.

        Args:
            state: The state vector to update
            qubit_count: The number of qubits in the state
            matrix: The matrix to apply
            targets: The targets the matrix will act on

        Returns:
            np.ndarray: The updated state vector
        """
        gate_matrix = np.reshape(matrix, [2] * len(targets) * 2)
        axes = (
            np.arange(len(targets), 2 * len(targets)),
            targets,
        )
        dot_product = np.tensordot(gate_matrix, state, axes=axes)

        # Axes given in `operation.targets` are in the first positions.
        unused_idxs = [idx for idx in range(qubit_count) if idx not in targets]
        permutation = targets + unused_idxs
        # Invert the permutation to put the indices in the correct place
        inverse_permutation = np.argsort(permutation)
        return np.transpose(dot_product, inverse_permutation)

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
