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

from functools import singledispatch
from typing import List

import numpy as np
import opt_einsum
from braket.default_simulator.operation import GateOperation, Observable, Operation


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

    def evolve(self, operations: List[GateOperation], partition_size: int) -> None:
        """ Evolves the state of the simulation under the action of
        the specified gate operations.

        The gates and state vector are treated as tensors, and the evolution contracts
        them in the order of the gates. The gate operation list can be split into contiguous
        partitions of a given length (with remainder), with each partition being contracted
        sequentially. Smaller partitions may run longer, but can save memory.

        Args:
            operations (List[GateOperation]): Gate operations to apply for
                evolving the state of the simulation.
            partition_size (int): The size of the partitions to contract.
                If set to 0, the entire circuit is contracted.

        Note:
            This method mutates the state of the simulation.
        """
        # TODO: Write algorithm to determine partition size based on qubit count
        self._state_vector = np.reshape(self._state_vector, [2] * self._qubit_count)

        partitions = (
            [operations[i : i + partition_size] for i in range(0, len(operations), partition_size)]
            if partition_size
            else [operations]
        )

        for partition in partitions:
            self._state_vector = opt_einsum.contract(
                *StateVectorSimulation._build_contraction_parameters(
                    self._state_vector, self._qubit_count, partition
                )
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
        contracted = opt_einsum.contract(
            *StateVectorSimulation._build_contraction_parameters(
                self._state_vector, self._qubit_count, observables
            )
        )
        self._post_observables = contracted.reshape(2 ** self._qubit_count)

    @staticmethod
    def _build_contraction_parameters(state, qubit_count, operations: List[Operation]) -> list:
        state = np.reshape(state, [2] * qubit_count)
        contraction_parameters = [state, list(range(qubit_count))]
        index_substitutions = {i: i for i in range(qubit_count)}
        next_index = qubit_count
        for operation in operations:
            matrix = _get_matrix(operation)
            targets = operation.targets

            # Only add to contraction parameters if the observable
            # has a nontrivial diagonalizing matrix
            if matrix is not None and targets:
                # lower indices, which will be traced out
                targets = operation.targets
                covariant = [index_substitutions[i] for i in targets]

                # upper indices, which will replace the contracted indices in the state vector
                contravariant = list(range(next_index, next_index + len(covariant)))
                indices = contravariant + covariant
                # matrix as type-(len(contravariant), len(covariant)) tensor
                matrix_as_tensor = matrix.reshape([2] * len(indices))

                contraction_parameters += [matrix_as_tensor, indices]
                next_index += len(covariant)

                index_substitutions.update(
                    {targets[i]: contravariant[i] for i in range(len(targets))}
                )
            elif not targets:
                # `operation` is an observable, and the only element in `operations`
                for qubit in range(qubit_count):
                    # Since observables don't overlap, there's no need to track index replacements
                    contraction_parameters += [matrix, [next_index, qubit]]
                    index_substitutions[qubit] = next_index
                    next_index += 1

        # Ensure state is in correct order
        new_indices = [
            index_substitutions[i] if i in index_substitutions else i for i in range(qubit_count)
        ]
        contraction_parameters.append(new_indices)
        return contraction_parameters

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


@singledispatch
def _get_matrix(operation):
    raise ValueError(f"Unrecognized operation: {operation}")


@_get_matrix.register
def _(gate: GateOperation):
    return gate.matrix


@_get_matrix.register
def _(observable: Observable):
    return observable.diagonalizing_matrix
