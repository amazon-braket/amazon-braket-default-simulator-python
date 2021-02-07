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

from functools import reduce
from typing import List

import numpy as np
import opt_einsum

from braket.default_simulator.gate_operations import Unitary
from braket.default_simulator.operation import GateOperation


def apply_operations(
    state: np.ndarray, qubit_count: int, operations: List[GateOperation], batch_size: int
) -> np.ndarray:
    r"""Combines consecutive operations on the same targets and applies them in batches.

    First, consecutive operations on the same target qubits are combined into single unitary gates.

    For example::

        |0> -[H]-[X]------[YY]-[ZZ]-
                           |    |
        |1> -[X]--@--[XX]--|----|---
                  |   |    |    |
        |2> -----[X]-[XX]-[YY]-[ZZ]-

    becomes::

        |0> -[X * H]-----------[ZZ * YY]-
                                   |
        |1> ---[X]---[XX * CX]-----|-----
                         |         |
        |2> ---------[XX * CX]-[ZZ * YY]-

    (the gates are reversed in each group because matrices are multiplied right to left)

    The combined operations are then partitioned into contiguous batches of size ``batch_size``
    (with remainder). The state vector is treated as a type :math:`(qubit\_count, 0)` tensor,
    and each operation is treated as a type :math:`(target\_length, target\_length)` tensor
    (where :math:`target\_length` is the number of targets the operation acts on), and each batch
    is contracted in an order optimized among the operations in the batch.

    For example, if we have a 4-qubit state :math:`S` and a batch with two gates :math:`G1` and
    :math:`G2` that act on qubits 0 and 1 and 1 and 3, respectively, then the state vector after
    applying the batch is :math:`S^{mokp} = S^{ijkl} G1^{mn}_{ij} G2^{op}_{nl}`.

    Depending on the batch size, number of qubits, and the number and types of gates, the speed can
    be more than twice that of applying operations one at a time, but memory use will increase.
    Empirically, noticeable performance improvements were observed starting with a batch size of 10,
    with increasing performance gains up to a batch size of 50. We tested this with 16 GB of memory.
    For batch sizes greater than 50, consider using an environment with more than 16 GB of memory.

    Args:
        state (np.ndarray): The state vector to apply :math:`operations` to, as a type
            :math:`(qubit\_count, 0)` tensor
        qubit_count (int): The number of qubits in the state
        operations (List[GateOperation]): The operations to apply to the state vector
        batch_size: The number of operations to contract in each batch

    Returns:
        np.ndarray: The state vector after applying the given operations, as a type
        (num_qubits, 0) tensor
    """
    combined_operations = _combine_operations(operations, qubit_count)

    # TODO: Write algorithm to determine partition size based on operations and qubit count
    partitions = [
        combined_operations[i : i + batch_size]
        for i in range(0, len(combined_operations), batch_size)
    ]

    for partition in partitions:
        state = _contract_operations(state, qubit_count, partition)

    return state


def _combine_operations(operations: List[GateOperation], qubit_count: int) -> List[GateOperation]:
    groups = [[] for _ in range(qubit_count)]
    last_targets_for_qubits = [None for _ in range(qubit_count)]
    group_order = []

    for operation in operations:
        targets = operation.targets
        first_qubit = sorted(targets)[0]
        if all(last_targets_for_qubits[qubit] == targets for qubit in targets):
            groups[first_qubit][-1].append(operation)
        else:
            group_order.append((first_qubit, len(groups[first_qubit])))
            groups[first_qubit].append([operation])
            for qubit in targets:
                last_targets_for_qubits[qubit] = targets

    combined_operations = []
    for qubit, index in group_order:
        group = groups[qubit][index]
        combined_operations.append(
            Unitary(
                group[0].targets,
                reduce(np.dot, [operation.matrix for operation in reversed(group)]),
            )
        )
    return combined_operations


def _contract_operations(
    state: np.ndarray, qubit_count: int, operations: List[GateOperation]
) -> np.ndarray:
    contraction_parameters = [state, list(range(qubit_count))]
    index_substitutions = {i: i for i in range(qubit_count)}
    next_index = qubit_count
    for operation in operations:
        matrix = operation.matrix
        targets = operation.targets

        # Lower indices, which will be traced out
        covariant = [index_substitutions[i] for i in targets]

        # Upper indices, which will replace the contracted indices in the state vector
        contravariant = list(range(next_index, next_index + len(covariant)))

        indices = contravariant + covariant
        # `matrix` as type-(len(contravariant), len(covariant)) tensor
        matrix_as_tensor = np.reshape(matrix, [2] * len(indices))

        contraction_parameters += [matrix_as_tensor, indices]
        next_index += len(covariant)
        index_substitutions.update({targets[i]: contravariant[i] for i in range(len(targets))})

    # Ensure state is in correct order
    new_indices = [index_substitutions[i] for i in range(qubit_count)]
    contraction_parameters.append(new_indices)
    return opt_einsum.contract(*contraction_parameters)
