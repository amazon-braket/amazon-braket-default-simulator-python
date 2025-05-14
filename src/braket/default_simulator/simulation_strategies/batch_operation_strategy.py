# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
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


import numpy as np
import opt_einsum

from braket.default_simulator.linalg_utils import multiply_matrix
from braket.default_simulator.operation import GateOperation


def apply_operations(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation], batch_size: int = 1
) -> np.ndarray:
    r"""Applies operations to a state vector in batches of size :math:`batch\_size`.

    :math:`operations` is partitioned into contiguous batches of size :math:`batch\_size` (with
    remainder). The state vector is treated as a type :math:`(qubit\_count, 0)` tensor, and each
    operation is treated as a type :math:`(target\_length, target\_length)` tensor (where
    :math:`target\_length` is the number of targets the operation acts on), and each batch is
    contracted in an order optimized among the operations in the batch. Larger batches can be
    significantly faster (although this is not guaranteed), but will use more memory.

    For example, if we have a 4-qubit state :math:`S` and a batch with two gates :math:`G1` and
    :math:`G2` that act on qubits 0 and 1 and 1 and 3, respectively, then the state vector after
    applying the batch is :math:`S^{mokp} = S^{ijkl} G1^{mn}_{ij} G2^{op}_{nl}`.

    Depending on the batch size, number of qubits, and the number and types of gates, the speed can
    be more than twice that of applying operations one at a time. Empirically, noticeable
    performance improvements were observed starting with a batch size of 10, with increasing
    performance gains up to a batch size of 50. We tested this with 16 GB of memory. For batch sizes
    greater than 50, consider using an environment with more than 16 GB of memory.

    Args:
        state (np.ndarray): The state vector to apply :math:`operations` to, as a type
            :math:`(qubit\_count, 0)` tensor
        qubit_count (int): The number of qubits in the state
        operations (list[GateOperation]): The operations to apply to the state vector
        batch_size (int): The number of operations to contract in each batch. Defaults to 1.

    Returns:
        np.ndarray: The state vector after applying the given operations, as a type
        (num_qubits, 0) tensor
    """
    if not operations:
        return state

    if batch_size == 1:
        return _apply_operations_sequential(state, operations)

    return _apply_operations_batched(state, qubit_count, operations, batch_size)


def _apply_operations_sequential(state: np.ndarray, operations: list[GateOperation]) -> np.ndarray:
    """Apply operations sequentially without batching."""
    for op in operations:
        state = _apply_operation(state, op)
    return state


def _apply_operations_batched(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation], batch_size: int
) -> np.ndarray:
    """Apply operations in optimized batches."""
    current_batch = []

    def process_batch():
        nonlocal state
        if current_batch:
            state = _process_optimized_batch(state, qubit_count, current_batch)
            current_batch.clear()

    for op in operations:
        current_batch.append(op)
        if len(current_batch) >= batch_size:
            process_batch()

    process_batch()
    return state


def _apply_operation(state: np.ndarray, op: GateOperation):
    """Apply an operation to the state."""
    matrix = op.matrix
    all_targets = op.targets
    num_ctrl = len(op._ctrl_modifiers)
    control_state = op._ctrl_modifiers
    controls = all_targets[:num_ctrl]
    targets = all_targets[num_ctrl:]
    return multiply_matrix(state, matrix, targets, controls, control_state)


def _process_optimized_batch(state, qubit_count, operations):
    """Process a batch of operations with optimized tensor contraction."""
    if len(operations) <= 2:
        for op in operations:
            state = multiply_matrix(state, op.matrix, op.targets, [], [])
        return state

    contraction_parameters = [state, [*range(qubit_count)]]
    index_substitutions = {i: i for i in range(qubit_count)}
    next_index = qubit_count

    for operation in operations:
        matrix = operation.matrix
        targets = operation.targets

        covariant = [index_substitutions[i] for i in targets]
        contravariant = [*range(next_index, next_index + len(targets))]
        indices = contravariant + covariant

        matrix_dim = int(np.log2(matrix.shape[0]))
        shape = [2] * (2 * matrix_dim)

        contraction_parameters.extend([np.reshape(matrix, shape), indices])

        for i, target in enumerate(targets):
            index_substitutions[target] = contravariant[i]

        next_index += len(targets)

    new_indices = [index_substitutions[i] for i in range(qubit_count)]
    contraction_parameters.append(new_indices)

    return opt_einsum.contract(*contraction_parameters, optimize="auto")
