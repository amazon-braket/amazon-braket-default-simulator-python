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
from typing import List, Optional, Tuple

import numpy as np
import opt_einsum
from braket.default_simulator.operation import GateOperation, Observable, Operation


class StateVectorSimulation:
    """
    This class encapsulates a simulation of a quantum system of `qubit_count` qubits.
    The evolution of the state of the quantum system is achieved through `GateOperation`s
    using the `evolve()` method.

    The simulation defaults to applying gates one at a time, but can also be configured
    to apply multiple gates at once by supplying the `partition_size` keyword argument.

    When `partition_size` is supplied, the operation list is split into contiguous partitions
    of length `partition_size` (with remainder) to contract. Within each partition, contraction
    order is optimized among the gates, and the partitions themselves are contracted in the order
    they appear. Larger partitions can be significantly faster, but will use more memory.
    """

    def __init__(self, qubit_count: int, shots: int, partition_size: Optional[int]):
        r"""
        Args:
            qubit_count (int): The number of qubits being simulated.
                All the qubits start in the :math:`\ket{\mathbf{0}}` computational basis state.
            shots (int): The number of samples to take from the simulation.
                If set to 0, only results that do not require sampling, such as state vector
                or expectation, will be generated.
            partition_size (Optional[int]): If specified, the size of the partitions to contract
                with If `None`, the gates are applied one at a time, without any optimization of
                contraction order. If set to 0, the entire circuit is contracted.
        """
        initial_state = np.zeros(2 ** qubit_count, dtype=complex)
        initial_state[0] = 1
        self._state_vector = initial_state
        self._qubit_count = qubit_count
        self._shots = shots
        self._partition_size = partition_size
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
            self._state_vector, self._qubit_count, operations, self._partition_size
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
            self._state_vector, self._qubit_count, observables, self._partition_size
        )

    @staticmethod
    def _apply_operations(
        state: np.ndarray, qubit_count: int, operations: List[Operation], partition_size: int
    ) -> np.ndarray:
        return (
            StateVectorSimulation._apply_individual_operations(state, qubit_count, operations)
            if partition_size is None
            else StateVectorSimulation._partition_and_contract(
                state, qubit_count, operations, partition_size
            )
        )

    @staticmethod
    def _apply_individual_operations(
        state: np.ndarray, qubit_count: int, operations: List[Operation]
    ) -> np.ndarray:
        state_tensor = np.reshape(state, [2] * qubit_count)
        for operation in operations:
            matrix = _get_matrix(operation)
            targets = operation.targets
            # `operation` is ignored if it is an observable with a trivial diagonalizing matrix
            if matrix is not None:
                if operation.targets:
                    state_tensor = StateVectorSimulation._apply_operation(
                        state_tensor, qubit_count, matrix, targets
                    )
                else:
                    # `operation` is an observable, and the only element in `operations`
                    for qubit in range(qubit_count):
                        state_tensor = StateVectorSimulation._apply_operation(
                            state_tensor, qubit_count, matrix, (qubit,)
                        )
        return np.reshape(state_tensor, 2 ** qubit_count)

    @staticmethod
    def _apply_operation(
        state: np.ndarray, qubit_count: int, matrix: np.ndarray, targets: Tuple[int]
    ) -> np.ndarray:
        gate_matrix = np.reshape(matrix, [2] * len(targets) * 2)
        axes = (
            np.arange(len(targets), 2 * len(targets)),
            targets,
        )
        dot_product = np.tensordot(gate_matrix, state, axes=axes)

        # Axes given in `operation.targets` are in the first positions.
        unused_idxs = [idx for idx in range(qubit_count) if idx not in targets]
        permutation = list(targets) + unused_idxs
        # Invert the permutation to put the indices in the correct place
        inverse_permutation = np.argsort(permutation)
        return np.transpose(dot_product, inverse_permutation)

    @staticmethod
    def _partition_and_contract(
        state: np.ndarray, qubit_count: int, operations: List[Operation], partition_size: int
    ) -> np.ndarray:
        # TODO: Write algorithm to determine partition size based on operations and qubit count
        state_tensor = np.reshape(state, [2] * qubit_count)

        partitions = (
            [operations[i : i + partition_size] for i in range(0, len(operations), partition_size)]
            if partition_size
            else [operations]
        )

        for partition in partitions:
            state_tensor = StateVectorSimulation._contract_operations(
                state_tensor, qubit_count, partition
            )

        return np.reshape(state_tensor, 2 ** qubit_count)

    @staticmethod
    def _contract_operations(
        state: np.ndarray, qubit_count: int, operations: List[Operation]
    ) -> np.ndarray:
        contraction_parameters = [state, list(range(qubit_count))]
        index_substitutions = {i: i for i in range(qubit_count)}
        next_index = qubit_count
        for operation in operations:
            matrix = _get_matrix(operation)
            targets = operation.targets

            # `operation` is not added tp the contraction parameters if
            # it is an observable with a trivial diagonalizing matrix
            if matrix is not None:
                if targets:
                    # lower indices, which will be traced out
                    covariant = [index_substitutions[i] for i in targets]

                    # upper indices, which will replace the contracted indices in the state vector
                    contravariant = list(range(next_index, next_index + len(covariant)))

                    indices = contravariant + covariant
                    # matrix as type-(len(contravariant), len(covariant)) tensor
                    matrix_as_tensor = np.reshape(matrix, [2] * len(indices))

                    contraction_parameters += [matrix_as_tensor, indices]
                    next_index += len(covariant)
                    index_substitutions.update(
                        {targets[i]: contravariant[i] for i in range(len(targets))}
                    )
                else:
                    # `operation` is an observable, and the only element in `operations`
                    for qubit in range(qubit_count):
                        # Since observables don't overlap,
                        # there's no need to track index replacements
                        contraction_parameters += [matrix, [next_index, qubit]]
                        index_substitutions[qubit] = next_index
                        next_index += 1

        # Ensure state is in correct order
        new_indices = [index_substitutions[i] for i in range(qubit_count)]
        contraction_parameters.append(new_indices)
        return opt_einsum.contract(*contraction_parameters)

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
