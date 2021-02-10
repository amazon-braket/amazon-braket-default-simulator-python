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

from braket.default_simulator.operation import Operation
from braket.default_simulator.operation_helpers import get_matrix


def apply_operations(
    state: np.ndarray, qubit_count: int, operations: List[Operation]
) -> np.ndarray:
    """Applies operations to a state vector one at a time.

    Args:
        state (np.ndarray): The state vector to apply the given operations to, as a type
            (num_qubits, 0) tensor
        qubit_count (int): The number of qubits in the state
        operations (List[Operation]): The operations to apply to the state vector

    Returns:
        np.ndarray: The state vector after applying the given operations, as a type
        (qubit_count, 0) tensor
    """
    for operation in operations:
        matrix = get_matrix(operation)
        targets = operation.targets
        # `operation` is ignored if it acts trivially on its targets
        if targets:
            state = _apply_operation(state, qubit_count, matrix, targets)
        elif targets is None:
            # `operation` is an observable, and the only element in `operations`
            for qubit in range(qubit_count):
                state = _apply_operation(state, qubit_count, matrix, (qubit,))
    return state


def _apply_operation(
    state: np.ndarray, qubit_count: int, matrix: np.ndarray, targets: Tuple[int, ...]
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
