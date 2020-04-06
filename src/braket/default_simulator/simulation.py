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
from braket.default_simulator import GateOperation, Observable
from opt_einsum import contract


class StateVectorSimulation:
    """
    The class `StateVectorSimulation` encapsulates a simulation of a quantum system of
    `qubit_count` qubits. The evolution of the state of the quantum system is achieved
    through `GateOperation`s using the `evolve()` method.
    """

    def __init__(self, qubit_count: int):
        """
        Args:
            qubit_count (int): The number of qubits being simulated.
                All the qubits start in the `|0>` computational basis state.
        """
        initial_state = np.zeros(2 ** qubit_count, dtype=complex)
        initial_state[0] = 1
        self._state_vector = np.reshape(initial_state, [2] * qubit_count)
        self._qubit_count = qubit_count
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
            self._apply_operation(operation)
        self._state_vector = np.reshape(self._state_vector, 2 ** self._qubit_count)

    def _apply_operation(self, operation: GateOperation) -> None:
        """ Updates the current state of the simulation by multiplying the state with
        the unitary matrix corresponding to the operation.

        Args:
            operation (GateOperation): Unitary gate to apply to evolve the current state
                of the simulation.
        """
        gate_matrix = np.reshape(operation.matrix, [2] * len(operation.targets) * 2)
        axes = (
            np.arange(len(operation.targets), 2 * len(operation.targets)),
            operation.targets,
        )
        dot_product = np.tensordot(gate_matrix, self._state_vector, axes=axes)

        # Axes given in `operation.targets` are in the first positions.
        unused_idxs = [idx for idx in range(self._qubit_count) if idx not in operation.targets]
        permutation = operation.targets + unused_idxs
        # Invert the permutation to put the indices in the correct place
        inverse_permutation = np.argsort(permutation)
        self._state_vector = np.transpose(dot_product, inverse_permutation)

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
        contracted = contract(*self._build_contraction_parameters(observables))
        self._post_observables = contracted.reshape(2 ** self._qubit_count)

    def _build_contraction_parameters(self, observables: List[Observable]) -> list:
        state = np.reshape(self._state_vector, [2] * self._qubit_count)
        contraction_parameters = [state, list(range(self.qubit_count))]
        index_substitutions = {}
        next_index = self.qubit_count
        for observable in observables:
            diagonalizing_matrix = observable.diagonalizing_matrix

            # Only add to contraction parameters if the observable
            # has a nontrivial diagonalizing matrix
            if diagonalizing_matrix is not None:
                # lower indices, which will be traced out
                covariant = observable.targets

                # upper indices, which will replace the contracted indices in the state vector
                contravariant = list(range(next_index, next_index + len(covariant)))
                indices = contravariant + covariant
                # matrix as type-(len(contravariant), len(covariant)) tensor
                matrix_as_tensor = observable.diagonalizing_matrix.reshape([2] * len(indices))

                contraction_parameters += [matrix_as_tensor, indices]
                next_index += len(covariant)

                index_substitutions.update(
                    {covariant[i]: contravariant[i] for i in range(len(covariant))}
                )
        # Ensure state is in correct order
        new_indices = [
            index_substitutions[i] if i in index_substitutions else i
            for i in range(self.qubit_count)
        ]
        contraction_parameters.append(new_indices)
        return contraction_parameters

    def retrieve_samples(self, num_samples: int) -> List[int]:
        """ Retrieves `num_samples` samples of states from the state vector of the simulation,
        based on the probability amplitudes.

        Args:
            num_samples (int): Number of samples to retrieve from the state vector.

        Returns:
            List[int]: List of states sampled according to their probability amplitudes
            in the state vector. Each integer represents the decimal encoding of the
            corresponding computational basis state.
        """
        return np.random.choice(
            len(self._state_vector), p=self.probability_amplitudes, size=num_samples
        )

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
    def probability_amplitudes(self) -> np.ndarray:
        """np.ndarray: The probability amplitudes corresponding to each basis state."""
        amplitudes = np.abs(self._state_vector)
        amplitudes **= 2
        return amplitudes
