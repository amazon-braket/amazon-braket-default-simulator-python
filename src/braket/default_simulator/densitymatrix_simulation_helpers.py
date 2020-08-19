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

from typing import List, Tuple

import numpy as np

from braket.default_simulator.operation import GateOperation, KrausOperation, Observable, Operation
from braket.default_simulator.operation_helpers import get_matrix


def _apply_operations(
    state: np.ndarray, qubit_count: int, operations: List[Operation]
) -> np.ndarray:
    """ Apply the operations to the density matrix.

    Args:
        state (np.array): initial density matrix
        qubit_count (int): number of qubit in the circuit
        operations (List[Operation]): list of GateOperation and
            KrausOperation to be applied to the density matrix

    Returns:
        state (nd.array): output density matrix
    """
    dm_tensor = np.reshape(state, [2] * 2 * qubit_count)
    for operation in operations:
        matrix = get_matrix(operation)
        targets = operation.targets

        if targets:
            if isinstance(operation, (GateOperation, Observable)):
                dm_tensor = _apply_gate(dm_tensor, qubit_count, matrix, targets)
            if isinstance(operation, KrausOperation):
                dm_tensor = _apply_kraus(dm_tensor, qubit_count, matrix, targets)
        elif targets is None:
            if isinstance(operation, (GateOperation, Observable)):
                for qubit in range(qubit_count):
                    dm_tensor = _apply_gate(dm_tensor, qubit_count, matrix, (qubit,))
            if isinstance(operation, KrausOperation):
                for qubit in range(qubit_count):
                    dm_tensor = _apply_kraus(dm_tensor, qubit_count, matrix, (qubit,))

    return np.reshape(dm_tensor, (2 ** qubit_count, 2 ** qubit_count))


def _apply_gate(
    state: np.ndarray, qubit_count: int, matrix: np.ndarray, targets: Tuple[int, ...]
) -> np.ndarray:
    """ Apply a matrix M to a density matrix D according to:

        .. math::
            D \rightarrow M D M^{\\dagger}

    Args:
        state (np.ndarray): initial density matrix
        qubit_count (int): number of qubit in the circuit
        matrix (nd.array): matrix to be applied to the density matrix
        targets (Tuple[int,...]): qubits of the density matrix the matrix applied to.

    Returns:
        state (nd.array): output density matrix
    """
    # left product
    gate_matrix = np.reshape(matrix, [2] * len(targets) * 2)
    dm_targets = targets
    axes = (
        np.arange(len(targets), 2 * len(targets)),
        dm_targets,
    )
    state = np.tensordot(gate_matrix, state, axes=axes)

    # Arrange the index to the correct place.
    unused_idxs = [idx for idx in range(2 * qubit_count) if idx not in dm_targets]
    permutation = list(dm_targets) + unused_idxs
    inverse_permutation = np.argsort(permutation)
    state = np.transpose(state, inverse_permutation)

    # right product
    gate_matrix = np.reshape(matrix.conjugate(), [2] * len(targets) * 2)
    dm_targets = tuple(i + qubit_count for i in targets)
    axes = (
        np.arange(len(targets), 2 * len(targets)),
        dm_targets,
    )
    state = np.tensordot(gate_matrix, state, axes=axes)

    # Arrange the index to the correct place.
    unused_idxs = [idx for idx in range(2 * qubit_count) if idx not in dm_targets]
    permutation = list(dm_targets) + unused_idxs
    inverse_permutation = np.argsort(permutation)
    state = np.transpose(state, inverse_permutation)

    return state


def _apply_kraus(
    state: np.ndarray, qubit_count: int, matrices: List[np.ndarray], targets: Tuple[int, ...]
) -> np.ndarray:
    """ Apply a list of matrices {E_i} to a density matrix D according to:

        .. math::
            D \rightarrow \\sum_i E_i D E_i^{\\dagger}

    Args:
        state (np.ndarray): initial density matrix
        qubit_count (int): number of qubit in the circuit
        matrices (List[nd.array]): matrices to be applied to the density matrix
        targets (Tuple[int,...]): qubits of the density matrix the matrices applied to.

    Returns:
        state (nd.array): output density matrix
    """
    new_state = np.zeros_like(state)
    for matrix in matrices:
        new_state = new_state + _apply_gate(state, qubit_count, matrix, targets)
    return new_state
